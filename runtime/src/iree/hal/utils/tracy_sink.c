// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/tracy_sink.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/base/tracing.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// Upper bound on tracing GPU contexts this sink will allocate. The tracing
// provider has a hard limit of 255 contexts shared with stream tracing and
// other users so we leave headroom instead of wrapping into garbage ids.
#define IREE_HAL_PROFILE_TRACY_MAX_CONTEXT_COUNT 128

// Upper bound on device-timestamped events buffered per physical device while
// waiting for a clock mapping. Events beyond this are counted as unmapped.
#define IREE_HAL_PROFILE_TRACY_MAX_PENDING_EVENT_COUNT (1u << 20)

// Pseudo source file reported for all zones emitted by the sink.
static const char kIreeHalProfileTracyFile[] = "iree/hal/profile";

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Which timeline a zone belongs to. Each (device, queue, lane) gets its own
// tracing GPU context so that overlapping event families do not interleave.
typedef enum iree_hal_profile_tracy_lane_e {
  // Device-timestamped dispatch execution.
  IREE_HAL_PROFILE_TRACY_LANE_DISPATCH = 0,
  // Device-timestamped queue operations (execute/copy/fill/...).
  IREE_HAL_PROFILE_TRACY_LANE_QUEUE_DEVICE,
  // Host-timestamped execution spans (CPU backends, command buffer replay).
  IREE_HAL_PROFILE_TRACY_LANE_HOST_EXECUTION,
  // Host-timestamped queue submissions (submit -> ready).
  IREE_HAL_PROFILE_TRACY_LANE_QUEUE_HOST,
} iree_hal_profile_tracy_lane_t;

typedef struct iree_hal_profile_tracy_context_t {
  uint32_t physical_device_ordinal;
  uint32_t queue_ordinal;
  iree_hal_profile_tracy_lane_t lane;
  // Tracing provider GPU context id.
  uint8_t tracy_id;
  // Rotating query id; begin and end each consume one.
  uint16_t next_query_id;
} iree_hal_profile_tracy_context_t;

typedef struct iree_hal_profile_tracy_function_t {
  bool occupied;
  uint64_t executable_id;
  uint32_t function_ordinal;
  uint32_t name_length;
  // Name storage owned by the sink.
  char* name;
} iree_hal_profile_tracy_function_t;

// Device-timestamped event awaiting a clock mapping.
typedef struct iree_hal_profile_tracy_pending_event_t {
  iree_hal_profile_tracy_lane_t lane;
  uint32_t queue_ordinal;
  uint32_t function_ordinal;
  uint32_t type;
  uint64_t executable_id;
  uint64_t start_tick;
  uint64_t end_tick;
} iree_hal_profile_tracy_pending_event_t;

// Linear mapping from device ticks onto IREE host time.
typedef struct iree_hal_profile_tracy_clock_fit_t {
  bool valid;
  // Device tick at the reference point.
  uint64_t tick_ref;
  // Host time at the reference point.
  int64_t host_ref_ns;
  // Slope as a ratio of host nanoseconds per |den_ticks| device ticks.
  uint64_t num_ns;
  uint64_t den_ticks;
} iree_hal_profile_tracy_clock_fit_t;

typedef struct iree_hal_profile_tracy_device_t {
  uint32_t physical_device_ordinal;
  uint32_t clock_sample_count;
  uint32_t invalid_clock_sample_count;
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
  iree_hal_profile_tracy_clock_fit_t fit;
  // Events buffered until |fit| becomes valid.
  iree_hal_profile_tracy_pending_event_t* pending;
  iree_host_size_t pending_count;
  iree_host_size_t pending_capacity;
} iree_hal_profile_tracy_device_t;

typedef struct iree_hal_profile_tracy_sink_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Serializes producer calls; producers may flush from several threads.
  iree_slim_mutex_t mutex;

  // Reference pair used to place host time on the tracing provider timeline.
  // Captured once when the first context is allocated and shared by all
  // contexts so that zones from different sessions line up.
  bool has_reference;
  int64_t reference_tracing_time;
  int64_t reference_host_time_ns;

  // Tracing GPU contexts. These persist for the sink lifetime as the tracing
  // provider cannot release them.
  iree_hal_profile_tracy_context_t* contexts;
  iree_host_size_t context_count;
  iree_host_size_t context_capacity;

  // Session-local executable function names, open-addressed by
  // (executable_id, function_ordinal).
  iree_hal_profile_tracy_function_t* functions;
  iree_host_size_t function_count;
  iree_host_size_t function_capacity;

  // Session-local per-physical-device clock state.
  iree_hal_profile_tracy_device_t* devices;
  iree_host_size_t device_count;
  iree_host_size_t device_capacity;

  iree_hal_profile_tracy_sink_statistics_t statistics;
} iree_hal_profile_tracy_sink_t;

static const iree_hal_profile_sink_vtable_t iree_hal_profile_tracy_sink_vtable;

static iree_hal_profile_tracy_sink_t* iree_hal_profile_tracy_sink_cast(
    iree_hal_profile_sink_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_profile_tracy_sink_vtable);
  return (iree_hal_profile_tracy_sink_t*)base_value;
}

//===----------------------------------------------------------------------===//
// Arithmetic helpers
//===----------------------------------------------------------------------===//

static uint64_t iree_hal_profile_tracy_mix_u64(uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

// Computes round(value * numerator / denominator) without intermediate
// overflow. Returns false if the result does not fit in 64 bits.
static bool iree_hal_profile_tracy_round_mul_div_u64(uint64_t value,
                                                     uint64_t numerator,
                                                     uint64_t denominator,
                                                     uint64_t* out_result) {
  *out_result = 0;
  if (denominator == 0) return false;
  if (value == 0 || numerator == 0) return true;
#if defined(__SIZEOF_INT128__)
  __uint128_t product = (__uint128_t)value * (__uint128_t)numerator;
  product += denominator / 2;
  __uint128_t quotient = product / denominator;
  if (quotient > UINT64_MAX) return false;
  *out_result = (uint64_t)quotient;
  return true;
#else
  const uint64_t whole = value / denominator;
  const uint64_t remainder = value % denominator;
  if (whole > UINT64_MAX / numerator) return false;
  uint64_t scaled = whole * numerator;
  if (remainder != 0) {
    if (remainder > UINT64_MAX / numerator) return false;
    uint64_t fractional_product = remainder * numerator;
    if (fractional_product > UINT64_MAX - denominator / 2) return false;
    uint64_t fractional = (fractional_product + denominator / 2) / denominator;
    if (scaled > UINT64_MAX - fractional) return false;
    scaled += fractional;
  }
  *out_result = scaled;
  return true;
#endif  // defined(__SIZEOF_INT128__)
}

//===----------------------------------------------------------------------===//
// Clock mapping
//===----------------------------------------------------------------------===//

static bool iree_hal_profile_tracy_clock_sample_host_midpoint(
    const iree_hal_profile_clock_correlation_record_t* sample,
    int64_t* out_time_ns) {
  if (!iree_all_bits_set(
          sample->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET) ||
      sample->host_time_begin_ns < 0 ||
      sample->host_time_end_ns < sample->host_time_begin_ns) {
    return false;
  }
  *out_time_ns = sample->host_time_begin_ns +
                 (sample->host_time_end_ns - sample->host_time_begin_ns) / 2;
  return true;
}

static bool iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(
    const iree_hal_profile_clock_correlation_record_t* sample,
    int64_t* out_time_ns) {
  if (!iree_all_bits_set(
          sample->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      sample->host_cpu_timestamp_ns > INT64_MAX) {
    return false;
  }
  *out_time_ns = (int64_t)sample->host_cpu_timestamp_ns;
  return true;
}

// Recomputes the tick->host mapping from the first and last clock samples.
//
// The slope prefers the driver-sampled CPU timestamps (tight bracket, but a
// driver-chosen clock domain) and the offset prefers the IREE host-time bracket
// midpoint (the domain host-timestamped records use). When no bracket is
// available the driver timestamp is used for the offset as well, which places
// device zones on a clock that may differ from the host timeline by a small
// constant.
static void iree_hal_profile_tracy_device_refit(
    iree_hal_profile_tracy_device_t* device) {
  iree_hal_profile_tracy_clock_fit_t fit;
  memset(&fit, 0, sizeof(fit));
  device->fit = fit;
  if (device->clock_sample_count < 2) return;
  if (device->invalid_clock_sample_count != 0) return;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(first->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      !iree_all_bits_set(last->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      last->device_tick <= first->device_tick) {
    return;
  }

  int64_t first_time_ns = 0;
  int64_t last_time_ns = 0;
  if (!iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(first,
                                                              &first_time_ns) ||
      !iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(last,
                                                              &last_time_ns)) {
    if (!iree_hal_profile_tracy_clock_sample_host_midpoint(first,
                                                           &first_time_ns) ||
        !iree_hal_profile_tracy_clock_sample_host_midpoint(last,
                                                           &last_time_ns)) {
      return;
    }
  }
  if (last_time_ns <= first_time_ns) return;

  int64_t host_ref_ns = 0;
  if (!iree_hal_profile_tracy_clock_sample_host_midpoint(last, &host_ref_ns) &&
      !iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(last,
                                                              &host_ref_ns)) {
    return;
  }

  fit.valid = true;
  fit.tick_ref = last->device_tick;
  fit.host_ref_ns = host_ref_ns;
  fit.num_ns = (uint64_t)(last_time_ns - first_time_ns);
  fit.den_ticks = last->device_tick - first->device_tick;
  device->fit = fit;
}

static bool iree_hal_profile_tracy_clock_fit_map(
    const iree_hal_profile_tracy_clock_fit_t* fit, uint64_t tick,
    int64_t* out_host_time_ns) {
  uint64_t delta_ns = 0;
  if (tick >= fit->tick_ref) {
    if (!iree_hal_profile_tracy_round_mul_div_u64(
            tick - fit->tick_ref, fit->num_ns, fit->den_ticks, &delta_ns) ||
        delta_ns > (uint64_t)(INT64_MAX - fit->host_ref_ns)) {
      return false;
    }
    *out_host_time_ns = fit->host_ref_ns + (int64_t)delta_ns;
  } else {
    if (!iree_hal_profile_tracy_round_mul_div_u64(
            fit->tick_ref - tick, fit->num_ns, fit->den_ticks, &delta_ns) ||
        delta_ns > (uint64_t)INT64_MAX) {
      return false;
    }
    *out_host_time_ns = fit->host_ref_ns - (int64_t)delta_ns;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Tables
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_profile_tracy_sink_ensure_device(
    iree_hal_profile_tracy_sink_t* sink, uint32_t physical_device_ordinal,
    iree_hal_profile_tracy_device_t** out_device) {
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    if (sink->devices[i].physical_device_ordinal == physical_device_ordinal) {
      *out_device = &sink->devices[i];
      return iree_ok_status();
    }
  }
  if (sink->device_count == sink->device_capacity) {
    iree_host_size_t new_capacity =
        sink->device_capacity ? sink->device_capacity * 2 : 4;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        sink->host_allocator, new_capacity * sizeof(*sink->devices),
        (void**)&sink->devices));
    sink->device_capacity = new_capacity;
  }
  iree_hal_profile_tracy_device_t* device =
      &sink->devices[sink->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static iree_hal_profile_tracy_device_t* iree_hal_profile_tracy_sink_find_device(
    iree_hal_profile_tracy_sink_t* sink, uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    if (sink->devices[i].physical_device_ordinal == physical_device_ordinal) {
      return &sink->devices[i];
    }
  }
  return NULL;
}

static uint64_t iree_hal_profile_tracy_function_hash(
    uint64_t executable_id, uint32_t function_ordinal) {
  return iree_hal_profile_tracy_mix_u64(executable_id) ^
         iree_hal_profile_tracy_mix_u64((uint64_t)function_ordinal |
                                        (UINT64_C(1) << 40));
}

static iree_hal_profile_tracy_function_t*
iree_hal_profile_tracy_sink_find_function_slot(
    iree_hal_profile_tracy_sink_t* sink, uint64_t executable_id,
    uint32_t function_ordinal) {
  if (sink->function_capacity == 0) return NULL;
  const uint64_t hash =
      iree_hal_profile_tracy_function_hash(executable_id, function_ordinal);
  const iree_host_size_t mask = sink->function_capacity - 1;
  for (iree_host_size_t probe = 0; probe < sink->function_capacity; ++probe) {
    iree_hal_profile_tracy_function_t* slot =
        &sink->functions[(hash + probe) & mask];
    if (!slot->occupied) return slot;
    if (slot->executable_id == executable_id &&
        slot->function_ordinal == function_ordinal) {
      return slot;
    }
  }
  return NULL;
}

static iree_status_t iree_hal_profile_tracy_sink_grow_functions(
    iree_hal_profile_tracy_sink_t* sink) {
  const iree_host_size_t new_capacity =
      sink->function_capacity ? sink->function_capacity * 2 : 64;
  iree_hal_profile_tracy_function_t* new_functions = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      sink->host_allocator, new_capacity * sizeof(*new_functions),
      (void**)&new_functions));
  memset(new_functions, 0, new_capacity * sizeof(*new_functions));
  iree_hal_profile_tracy_function_t* old_functions = sink->functions;
  const iree_host_size_t old_capacity = sink->function_capacity;
  sink->functions = new_functions;
  sink->function_capacity = new_capacity;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (!old_functions[i].occupied) continue;
    iree_hal_profile_tracy_function_t* slot =
        iree_hal_profile_tracy_sink_find_function_slot(
            sink, old_functions[i].executable_id,
            old_functions[i].function_ordinal);
    *slot = old_functions[i];
  }
  iree_allocator_free(sink->host_allocator, old_functions);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_insert_function(
    iree_hal_profile_tracy_sink_t* sink, uint64_t executable_id,
    uint32_t function_ordinal, iree_string_view_t name) {
  if ((sink->function_count + 1) * 2 > sink->function_capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_profile_tracy_sink_grow_functions(sink));
  }
  iree_hal_profile_tracy_function_t* slot =
      iree_hal_profile_tracy_sink_find_function_slot(sink, executable_id,
                                                     function_ordinal);
  if (slot->occupied) {
    // Producers may re-emit metadata on flush; keep the first name.
    return iree_ok_status();
  }
  char* name_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      sink->host_allocator, name.size ? name.size : 1, (void**)&name_storage));
  memcpy(name_storage, name.data, name.size);
  slot->occupied = true;
  slot->executable_id = executable_id;
  slot->function_ordinal = function_ordinal;
  slot->name = name_storage;
  slot->name_length = (uint32_t)name.size;
  ++sink->function_count;
  return iree_ok_status();
}

static bool iree_hal_profile_tracy_sink_lookup_function(
    iree_hal_profile_tracy_sink_t* sink, uint64_t executable_id,
    uint32_t function_ordinal, iree_string_view_t* out_name) {
  iree_hal_profile_tracy_function_t* slot =
      iree_hal_profile_tracy_sink_find_function_slot(sink, executable_id,
                                                     function_ordinal);
  if (!slot || !slot->occupied) return false;
  *out_name = iree_make_string_view(slot->name, slot->name_length);
  return true;
}

static void iree_hal_profile_tracy_sink_reset_session_tables(
    iree_hal_profile_tracy_sink_t* sink) {
  for (iree_host_size_t i = 0; i < sink->function_capacity; ++i) {
    if (sink->functions[i].occupied) {
      iree_allocator_free(sink->host_allocator, sink->functions[i].name);
    }
  }
  iree_allocator_free(sink->host_allocator, sink->functions);
  sink->functions = NULL;
  sink->function_count = 0;
  sink->function_capacity = 0;
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    iree_allocator_free(sink->host_allocator, sink->devices[i].pending);
  }
  iree_allocator_free(sink->host_allocator, sink->devices);
  sink->devices = NULL;
  sink->device_count = 0;
  sink->device_capacity = 0;
  sink->statistics.pending_event_count = 0;
}

//===----------------------------------------------------------------------===//
// Zone emission
//===----------------------------------------------------------------------===//

static const char* iree_hal_profile_tracy_lane_name(
    iree_hal_profile_tracy_lane_t lane) {
  switch (lane) {
    case IREE_HAL_PROFILE_TRACY_LANE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_TRACY_LANE_QUEUE_DEVICE:
      return "queue";
    case IREE_HAL_PROFILE_TRACY_LANE_HOST_EXECUTION:
      return "host";
    case IREE_HAL_PROFILE_TRACY_LANE_QUEUE_HOST:
      return "submit";
    default:
      return "unknown";
  }
}

static const char* iree_hal_profile_tracy_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE:
      return "execute";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ:
      return "read";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE:
      return "write";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA:
      return "alloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA:
      return "dealloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    default:
      return NULL;
  }
}

// Finds or allocates the tracing context for a (device, queue, lane) tuple.
// Returns NULL if the context budget is exhausted.
static iree_hal_profile_tracy_context_t*
iree_hal_profile_tracy_sink_acquire_context(
    iree_hal_profile_tracy_sink_t* sink, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, iree_hal_profile_tracy_lane_t lane) {
  for (iree_host_size_t i = 0; i < sink->context_count; ++i) {
    iree_hal_profile_tracy_context_t* context = &sink->contexts[i];
    if (context->physical_device_ordinal == physical_device_ordinal &&
        context->queue_ordinal == queue_ordinal && context->lane == lane) {
      return context;
    }
  }
  if (sink->context_count >= IREE_HAL_PROFILE_TRACY_MAX_CONTEXT_COUNT) {
    return NULL;
  }
  if (sink->context_count == sink->context_capacity) {
    iree_host_size_t new_capacity =
        sink->context_capacity ? sink->context_capacity * 2 : 8;
    if (!iree_status_is_ok(iree_allocator_realloc(
            sink->host_allocator, new_capacity * sizeof(*sink->contexts),
            (void**)&sink->contexts))) {
      return NULL;
    }
    sink->context_capacity = new_capacity;
  }

  if (!sink->has_reference) {
    // Bracket the host clock sample with the tracing clock so the reference
    // pair describes the same instant as closely as we can manage.
    const int64_t tracing_time_begin = iree_tracing_time();
    const int64_t host_time_ns = (int64_t)iree_time_now();
    const int64_t tracing_time_end = iree_tracing_time();
    sink->reference_tracing_time =
        tracing_time_begin + (tracing_time_end - tracing_time_begin) / 2;
    sink->reference_host_time_ns = host_time_ns;
    sink->has_reference = true;
  }

  char name[64];
  int name_length = 0;
  if (queue_ordinal == UINT32_MAX) {
    name_length = snprintf(name, sizeof(name), "dev%" PRIu32 " %s",
                           physical_device_ordinal,
                           iree_hal_profile_tracy_lane_name(lane));
  } else {
    name_length = snprintf(name, sizeof(name), "dev%" PRIu32 "/q%" PRIu32 " %s",
                           physical_device_ordinal, queue_ordinal,
                           iree_hal_profile_tracy_lane_name(lane));
  }
  if (name_length < 0) name_length = 0;
  if ((size_t)name_length >= sizeof(name)) name_length = sizeof(name) - 1;

  // Timestamps handed to the provider are already host nanoseconds so the
  // context maps host time 1:1 onto the tracing clock via the reference pair.
  iree_hal_profile_tracy_context_t* context =
      &sink->contexts[sink->context_count++];
  memset(context, 0, sizeof(*context));
  context->physical_device_ordinal = physical_device_ordinal;
  context->queue_ordinal = queue_ordinal;
  context->lane = lane;
  context->tracy_id = iree_tracing_gpu_context_allocate(
      IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN, name, (size_t)name_length,
      /*is_calibrated=*/false, (uint64_t)sink->reference_tracing_time,
      (uint64_t)sink->reference_host_time_ns, /*timestamp_period=*/1.0f);
  ++sink->statistics.context_count;
  return context;
}

// Emits one complete zone on |context| spanning host times [start, end].
static void iree_hal_profile_tracy_sink_emit_zone(
    iree_hal_profile_tracy_sink_t* sink,
    iree_hal_profile_tracy_context_t* context, iree_string_view_t name,
    int64_t start_host_time_ns, int64_t end_host_time_ns) {
  const uint16_t begin_query_id = context->next_query_id++;
  const uint16_t end_query_id = context->next_query_id++;
  iree_tracing_gpu_zone_begin_external(
      context->tracy_id, begin_query_id, kIreeHalProfileTracyFile,
      sizeof(kIreeHalProfileTracyFile) - 1, /*line=*/0, name.data, name.size,
      name.data, name.size);
  iree_tracing_gpu_zone_end(context->tracy_id, end_query_id);
  iree_tracing_gpu_zone_notify(context->tracy_id, begin_query_id,
                               start_host_time_ns);
  iree_tracing_gpu_zone_notify(context->tracy_id, end_query_id,
                               end_host_time_ns);
  ++sink->statistics.emitted_zone_count;
}

// Resolves the display name for an event: the executable function name when
// known, otherwise the queue operation type or a synthesized fallback.
static iree_string_view_t iree_hal_profile_tracy_sink_event_name(
    iree_hal_profile_tracy_sink_t* sink, uint64_t executable_id,
    uint32_t function_ordinal, iree_hal_profile_queue_event_type_t type,
    char* scratch, size_t scratch_length) {
  iree_string_view_t name = iree_string_view_empty();
  if (executable_id != 0 && function_ordinal != UINT32_MAX) {
    if (iree_hal_profile_tracy_sink_lookup_function(sink, executable_id,
                                                    function_ordinal, &name)) {
      return name;
    }
    int length = snprintf(scratch, scratch_length,
                          "dispatch exec=%" PRIu64 " fn=%" PRIu32,
                          executable_id, function_ordinal);
    if (length < 0) length = 0;
    if ((size_t)length >= scratch_length) length = (int)scratch_length - 1;
    return iree_make_string_view(scratch, (iree_host_size_t)length);
  }
  const char* type_name = iree_hal_profile_tracy_queue_event_type_name(type);
  if (type_name) return iree_make_cstring_view(type_name);
  int length = snprintf(scratch, scratch_length, "queue_op_%" PRIu32, type);
  if (length < 0) length = 0;
  if ((size_t)length >= scratch_length) length = (int)scratch_length - 1;
  return iree_make_string_view(scratch, (iree_host_size_t)length);
}

// Emits a host-timestamped span.
static void iree_hal_profile_tracy_sink_emit_host_span(
    iree_hal_profile_tracy_sink_t* sink, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, iree_hal_profile_tracy_lane_t lane,
    uint64_t executable_id, uint32_t function_ordinal,
    iree_hal_profile_queue_event_type_t type, int64_t start_host_time_ns,
    int64_t end_host_time_ns) {
  if (start_host_time_ns <= 0 || end_host_time_ns < start_host_time_ns) {
    ++sink->statistics.invalid_event_count;
    return;
  }
  iree_hal_profile_tracy_context_t* context =
      iree_hal_profile_tracy_sink_acquire_context(sink, physical_device_ordinal,
                                                  queue_ordinal, lane);
  if (!context) {
    ++sink->statistics.context_exhausted_event_count;
    return;
  }
  char scratch[64];
  iree_string_view_t name = iree_hal_profile_tracy_sink_event_name(
      sink, executable_id, function_ordinal, type, scratch, sizeof(scratch));
  iree_hal_profile_tracy_sink_emit_zone(sink, context, name, start_host_time_ns,
                                        end_host_time_ns);
}

// Emits a device-timestamped event using the device's current clock mapping.
static void iree_hal_profile_tracy_sink_emit_device_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_tracy_device_t* device,
    const iree_hal_profile_tracy_pending_event_t* event) {
  int64_t start_host_time_ns = 0;
  int64_t end_host_time_ns = 0;
  if (!iree_hal_profile_tracy_clock_fit_map(&device->fit, event->start_tick,
                                            &start_host_time_ns) ||
      !iree_hal_profile_tracy_clock_fit_map(&device->fit, event->end_tick,
                                            &end_host_time_ns)) {
    ++sink->statistics.invalid_event_count;
    return;
  }
  iree_hal_profile_tracy_sink_emit_host_span(
      sink, device->physical_device_ordinal, event->queue_ordinal, event->lane,
      event->executable_id, event->function_ordinal, event->type,
      start_host_time_ns, end_host_time_ns);
}

// Emits or buffers a device-timestamped event depending on whether a clock
// mapping exists for its device yet.
static iree_status_t iree_hal_profile_tracy_sink_handle_device_event(
    iree_hal_profile_tracy_sink_t* sink, uint32_t physical_device_ordinal,
    const iree_hal_profile_tracy_pending_event_t* event) {
  if (event->start_tick == 0 || event->end_tick < event->start_tick) {
    ++sink->statistics.invalid_event_count;
    return iree_ok_status();
  }
  iree_hal_profile_tracy_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_tracy_sink_ensure_device(
      sink, physical_device_ordinal, &device));
  if (device->fit.valid) {
    iree_hal_profile_tracy_sink_emit_device_event(sink, device, event);
    return iree_ok_status();
  }
  if (device->pending_count >= IREE_HAL_PROFILE_TRACY_MAX_PENDING_EVENT_COUNT) {
    ++sink->statistics.unmapped_event_count;
    return iree_ok_status();
  }
  if (device->pending_count == device->pending_capacity) {
    iree_host_size_t new_capacity =
        device->pending_capacity ? device->pending_capacity * 2 : 256;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        sink->host_allocator, new_capacity * sizeof(*device->pending),
        (void**)&device->pending));
    device->pending_capacity = new_capacity;
  }
  device->pending[device->pending_count++] = *event;
  ++sink->statistics.pending_event_count;
  return iree_ok_status();
}

// Emits all buffered events for |device| if a mapping exists (or drops them as
// unmapped when |force| is set at session end).
static void iree_hal_profile_tracy_sink_drain_device(
    iree_hal_profile_tracy_sink_t* sink,
    iree_hal_profile_tracy_device_t* device, bool force) {
  if (device->pending_count == 0) return;
  if (!device->fit.valid && !force) return;
  for (iree_host_size_t i = 0; i < device->pending_count; ++i) {
    if (device->fit.valid) {
      iree_hal_profile_tracy_sink_emit_device_event(sink, device,
                                                    &device->pending[i]);
    } else {
      ++sink->statistics.unmapped_event_count;
    }
  }
  sink->statistics.pending_event_count -= device->pending_count;
  device->pending_count = 0;
}

//===----------------------------------------------------------------------===//
// Chunk parsing
//===----------------------------------------------------------------------===//

typedef iree_status_t (*iree_hal_profile_tracy_record_fn_t)(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length);

// Walks records that begin with a uint32_t record_length field, tolerating
// records longer than the version of the struct this sink was built against.
static iree_status_t iree_hal_profile_tracy_sink_for_each_record(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_host_size_t min_record_length,
    iree_hal_profile_tracy_record_fn_t record_fn) {
  if (iovec_count > 0 && !iovecs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile chunk iovec list is required");
  }
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    iree_host_size_t offset = 0;
    while (offset < iovecs[i].data_length) {
      const iree_host_size_t remaining_length = iovecs[i].data_length - offset;
      uint32_t record_length = 0;
      if (remaining_length < sizeof(record_length)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS, "profile chunk '%.*s' has a partial record",
            (int)metadata->content_type.size, metadata->content_type.data);
      }
      memcpy(&record_length, iovecs[i].data + offset, sizeof(record_length));
      if (record_length < min_record_length ||
          record_length > remaining_length) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "profile chunk '%.*s' record length %" PRIu32
                                " is invalid",
                                (int)metadata->content_type.size,
                                metadata->content_type.data, record_length);
      }
      IREE_RETURN_IF_ERROR(
          record_fn(sink, metadata, iovecs[i].data + offset, record_length));
      offset += record_length;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_process_function(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)metadata;
  iree_hal_profile_executable_function_record_t function_record;
  memcpy(&function_record, record, sizeof(function_record));
  if ((iree_host_size_t)function_record.name_length >
      record_length - sizeof(function_record)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile executable-function record name length is invalid");
  }
  const iree_string_view_t name =
      iree_make_string_view((const char*)record + sizeof(function_record),
                            function_record.name_length);
  return iree_hal_profile_tracy_sink_insert_function(
      sink, function_record.executable_id, function_record.function_ordinal,
      name);
}

static iree_status_t iree_hal_profile_tracy_sink_process_clock_correlation(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)metadata;
  (void)record_length;
  iree_hal_profile_clock_correlation_record_t sample;
  memcpy(&sample, record, sizeof(sample));
  iree_hal_profile_tracy_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_tracy_sink_ensure_device(
      sink, sample.physical_device_ordinal, &device));
  if (iree_any_bit_set(
          sample.flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED)) {
    ++device->invalid_clock_sample_count;
  }
  if (device->clock_sample_count == 0) device->first_clock_sample = sample;
  device->last_clock_sample = sample;
  ++device->clock_sample_count;
  iree_hal_profile_tracy_device_refit(device);
  iree_hal_profile_tracy_sink_drain_device(sink, device, /*force=*/false);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_process_dispatch_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)record_length;
  iree_hal_profile_dispatch_event_t dispatch_event;
  memcpy(&dispatch_event, record, sizeof(dispatch_event));
  iree_hal_profile_tracy_pending_event_t event;
  memset(&event, 0, sizeof(event));
  event.lane = IREE_HAL_PROFILE_TRACY_LANE_DISPATCH;
  event.queue_ordinal = metadata->queue_ordinal;
  event.executable_id = dispatch_event.executable_id;
  event.function_ordinal = dispatch_event.function_ordinal;
  event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  event.start_tick = dispatch_event.start_tick;
  event.end_tick = dispatch_event.end_tick;
  return iree_hal_profile_tracy_sink_handle_device_event(
      sink, metadata->physical_device_ordinal, &event);
}

static iree_status_t iree_hal_profile_tracy_sink_process_queue_device_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)metadata;
  (void)record_length;
  iree_hal_profile_queue_device_event_t queue_event;
  memcpy(&queue_event, record, sizeof(queue_event));
  iree_hal_profile_tracy_pending_event_t event;
  memset(&event, 0, sizeof(event));
  event.lane = IREE_HAL_PROFILE_TRACY_LANE_QUEUE_DEVICE;
  event.queue_ordinal = queue_event.queue_ordinal;
  event.function_ordinal = UINT32_MAX;
  event.type = queue_event.type;
  event.start_tick = queue_event.start_tick;
  event.end_tick = queue_event.end_tick;
  return iree_hal_profile_tracy_sink_handle_device_event(
      sink, queue_event.physical_device_ordinal, &event);
}

static iree_status_t iree_hal_profile_tracy_sink_process_host_execution_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)metadata;
  (void)record_length;
  iree_hal_profile_host_execution_event_t event;
  memcpy(&event, record, sizeof(event));
  iree_hal_profile_tracy_sink_emit_host_span(
      sink, event.physical_device_ordinal, event.queue_ordinal,
      IREE_HAL_PROFILE_TRACY_LANE_HOST_EXECUTION, event.executable_id,
      event.function_ordinal, event.type, event.start_host_time_ns,
      event.end_host_time_ns);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_process_queue_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)metadata;
  (void)record_length;
  iree_hal_profile_queue_event_t event;
  memcpy(&event, record, sizeof(event));
  // Readiness is not observable for every producer; a submission without it
  // has no span to draw.
  if (event.ready_host_time_ns == 0) return iree_ok_status();
  iree_hal_profile_tracy_sink_emit_host_span(
      sink, event.physical_device_ordinal, event.queue_ordinal,
      IREE_HAL_PROFILE_TRACY_LANE_QUEUE_HOST, /*executable_id=*/0, UINT32_MAX,
      event.type, event.host_time_ns, event.ready_host_time_ns);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_profile_sink_t implementation
//===----------------------------------------------------------------------===//

static void iree_hal_profile_tracy_sink_destroy(
    iree_hal_profile_sink_t* base_sink) {
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_hal_profile_tracy_sink_reset_session_tables(sink);
  iree_allocator_free(host_allocator, sink->contexts);
  iree_slim_mutex_deinitialize(&sink->mutex);
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t iree_hal_profile_tracy_sink_begin_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)metadata;
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  iree_slim_mutex_lock(&sink->mutex);
  // Executable ids and clock samples are session-local.
  iree_hal_profile_tracy_sink_reset_session_tables(sink);
  iree_slim_mutex_unlock(&sink->mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  iree_slim_mutex_lock(&sink->mutex);

  if (iree_any_bit_set(metadata->flags,
                       IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
    sink->statistics.dropped_record_count += metadata->dropped_record_count;
  }

  iree_status_t status = iree_ok_status();
  const iree_string_view_t content_type = metadata->content_type;
  if (iree_string_view_equal(content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    status = iree_hal_profile_tracy_sink_for_each_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_dispatch_event_t),
        iree_hal_profile_tracy_sink_process_dispatch_event);
  } else if (iree_string_view_equal(
                 content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    status = iree_hal_profile_tracy_sink_for_each_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_queue_device_event_t),
        iree_hal_profile_tracy_sink_process_queue_device_event);
  } else if (iree_string_view_equal(
                 content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
    status = iree_hal_profile_tracy_sink_for_each_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_host_execution_event_t),
        iree_hal_profile_tracy_sink_process_host_execution_event);
  } else if (iree_string_view_equal(
                 content_type, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    status = iree_hal_profile_tracy_sink_for_each_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_queue_event_t),
        iree_hal_profile_tracy_sink_process_queue_event);
  } else if (iree_string_view_equal(
                 content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    status = iree_hal_profile_tracy_sink_for_each_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_clock_correlation_record_t),
        iree_hal_profile_tracy_sink_process_clock_correlation);
  } else if (iree_string_view_equal(
                 content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS)) {
    status = iree_hal_profile_tracy_sink_for_each_record(
        sink, metadata, iovec_count, iovecs,
        sizeof(iree_hal_profile_executable_function_record_t),
        iree_hal_profile_tracy_sink_process_function);
  }
  // All other content types (memory events, counters, metrics, traces, and
  // session metadata) have no tracing-provider representation.

  iree_slim_mutex_unlock(&sink->mutex);
  return status;
}

static iree_status_t iree_hal_profile_tracy_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)metadata;
  (void)session_status_code;
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  iree_slim_mutex_lock(&sink->mutex);

  // Whatever is still buffered gets the best mapping we have or is dropped.
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    iree_hal_profile_tracy_sink_drain_device(sink, &sink->devices[i],
                                             /*force=*/true);
  }

  if (sink->statistics.unmapped_event_count != 0 ||
      sink->statistics.dropped_record_count != 0 ||
      sink->statistics.context_exhausted_event_count != 0) {
    char message[160];
    int length = snprintf(message, sizeof(message),
                          "HAL profile sink: %" PRIu64
                          " events without a clock mapping, %" PRIu64
                          " records dropped by producers, %" PRIu64
                          " events over the context limit",
                          sink->statistics.unmapped_event_count,
                          sink->statistics.dropped_record_count,
                          sink->statistics.context_exhausted_event_count);
    if (length > 0) {
      if ((size_t)length >= sizeof(message)) length = sizeof(message) - 1;
      IREE_TRACE_MESSAGE_DYNAMIC(WARNING, message, (size_t)length);
    }
  }

  iree_slim_mutex_unlock(&sink->mutex);
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t iree_hal_profile_tracy_sink_vtable =
    {
        .destroy = iree_hal_profile_tracy_sink_destroy,
        .begin_session = iree_hal_profile_tracy_sink_begin_session,
        .write = iree_hal_profile_tracy_sink_write,
        .end_session = iree_hal_profile_tracy_sink_end_session,
};

IREE_API_EXPORT iree_status_t iree_hal_profile_tracy_sink_create(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;
  iree_hal_profile_tracy_sink_t* sink = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink));
  iree_hal_resource_initialize(&iree_hal_profile_tracy_sink_vtable,
                               &sink->resource);
  sink->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&sink->mutex);
  *out_sink = (iree_hal_profile_sink_t*)sink;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_profile_tracy_sink_statistics_t
iree_hal_profile_tracy_sink_statistics(iree_hal_profile_sink_t* base_sink) {
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  iree_slim_mutex_lock(&sink->mutex);
  iree_hal_profile_tracy_sink_statistics_t statistics = sink->statistics;
  iree_slim_mutex_unlock(&sink->mutex);
  return statistics;
}

#else

IREE_API_EXPORT iree_status_t iree_hal_profile_tracy_sink_create(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "the tracy profile sink requires a runtime built with device "
      "instrumentation (IREE_ENABLE_RUNTIME_TRACING=ON with the tracy provider "
      "and an IREE_TRACING_MODE that includes "
      "IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE)");
}

IREE_API_EXPORT iree_hal_profile_tracy_sink_statistics_t
iree_hal_profile_tracy_sink_statistics(iree_hal_profile_sink_t* sink) {
  (void)sink;
  iree_hal_profile_tracy_sink_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  return statistics;
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
