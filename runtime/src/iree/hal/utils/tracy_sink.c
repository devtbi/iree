// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/tracy_sink.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
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

// Upper bound on clock-correlation samples retained per physical device. The
// mapping interpolates between neighboring samples so more samples bound the
// error; when the table is full the oldest interior sample is dropped, which
// preserves both the longest baseline and the most recent history.
#define IREE_HAL_PROFILE_TRACY_MAX_CLOCK_SAMPLE_COUNT 256

// Floor on the uncertainty attributed to a clock sample, in nanoseconds. Even
// a driver timestamp taken in the same clock domain as IREE host time is only
// as good as the scheduling noise around the sampling call.
#define IREE_HAL_PROFILE_TRACY_CLOCK_UNCERTAINTY_FLOOR_NS 500

// Maximum relative error the tick rate may carry before the mapping is used,
// in parts per million. Device events are buffered until a sample pair with a
// long enough baseline brings the estimate under this bound; without it a
// flush taken microseconds after the session began would fix a rate that is
// wrong by percent and cannot be corrected once zones have been emitted.
#define IREE_HAL_PROFILE_TRACY_CLOCK_MAX_RATE_ERROR_PPM 5000

// Rate error expected from host clock domains that differ only by NTP slew
// (CLOCK_MONOTONIC vs CLOCK_MONOTONIC_RAW), in parts per million. Once the
// bracket-derived rate estimate is better than this it is preferred over the
// driver-derived one because it is expressed in the domain host-timestamped
// records use.
#define IREE_HAL_PROFILE_TRACY_CLOCK_DOMAIN_SLEW_PPM 50

// Widest host bracket that still lets a driver timestamp inside it be taken as
// evidence that both sides read the same clock. A sampling call that was
// descheduled produces a bracket wide enough to contain a timestamp from an
// entirely different clock, which would otherwise be promoted to the most
// trusted anchor in the table.
#define IREE_HAL_PROFILE_TRACY_CLOCK_MAX_AGREEMENT_BRACKET_NS 20000

// How many samples at each end of the table are considered when picking the
// pair the tick rate is fitted over. Using only the outermost pair lets one
// descheduled sampling call decide both the rate and whether it is trusted.
#define IREE_HAL_PROFILE_TRACY_CLOCK_RATE_CANDIDATE_COUNT 4

// How far a zone may start before the newest one already emitted on the same
// tracing context. The provider's server reads a GPU timestamp more than 2^31ns
// (~2.147s) below the previous one on a context as a hardware counter wrapping
// and permanently shifts every later timestamp on that context by centuries;
// a second of headroom keeps that unreachable.
#define IREE_HAL_PROFILE_TRACY_MAX_BACKWARD_STEP_NS INT64_C(1000000000)

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

// Identity of the producer stream a record arrived on.
//
// Sessions are independent (device ticks and executable ids restart with each
// one) and two drivers in the same process can both report physical device
// ordinal 0, so every table is keyed by all of it rather than by the ordinal
// alone.
typedef struct iree_hal_profile_tracy_scope_t {
  uint64_t session_id;
  uint64_t producer_key;
  iree_string_view_t producer_name;
} iree_hal_profile_tracy_scope_t;

typedef struct iree_hal_profile_tracy_context_t {
  uint64_t producer_key;
  uint32_t physical_device_ordinal;
  uint32_t queue_ordinal;
  iree_hal_profile_tracy_lane_t lane;
  // Tracing provider GPU context id.
  uint8_t tracy_id;
  // Rotating query id; begin and end each consume one.
  uint16_t next_query_id;
  // Start of the newest zone already emitted here. The provider appends to its
  // timelines in arrival order but searches them as if sorted, and its counter
  // wrap heuristic watches the same sequence, so emission must not step back.
  int64_t last_start_ns;
  bool has_emitted;
} iree_hal_profile_tracy_context_t;

// A zone waiting to be emitted. Producers report completions rather than
// starts, so zones are staged and ordered before they are handed over; the name
// is resolved at emission time because a scratch-formatted fallback name would
// not outlive the record that produced it.
typedef struct iree_hal_profile_tracy_zone_t {
  iree_host_size_t context_index;
  uint32_t sequence;
  int64_t start_host_time_ns;
  int64_t end_host_time_ns;
  uint64_t session_id;
  uint64_t executable_id;
  uint32_t function_ordinal;
  uint32_t type;
} iree_hal_profile_tracy_zone_t;

typedef struct iree_hal_profile_tracy_function_t {
  bool occupied;
  uint64_t session_id;
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

// Which host clock a device's anchors are expressed in. Chosen once per device
// from its first usable sample and required of every later one: a mapping whose
// anchors straddle two clocks bakes the offset between them into every segment.
typedef enum iree_hal_profile_tracy_anchor_domain_e {
  IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_UNSET = 0,
  // The driver's own CPU timestamp, sampled simultaneously with the tick.
  IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_DRIVER,
  // Midpoint of the host time bracket IREE took around the sampling call.
  IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_BRACKET,
  // The driver's system timestamp converted to nanoseconds.
  IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_SYSTEM,
} iree_hal_profile_tracy_anchor_domain_t;

// One clock-correlation observation reduced to what the mapping needs: a device
// tick, the host time it corresponds to, and how far that may be off.
typedef struct iree_hal_profile_tracy_clock_sample_t {
  uint64_t tick;
  // Host time at |tick| in the device's anchor domain, and how far it may be
  // off. Used to place events.
  int64_t anchor_ns;
  int64_t uncertainty_ns;
  // The driver's own timestamp, read simultaneously with the tick. It may be a
  // different clock than the anchors, but it carries no bracket jitter, so
  // differences between two of them measure the tick rate far better than the
  // anchors do. Used only for the rate, where any constant offset cancels.
  int64_t driver_ns;
  bool has_driver_ns;
} iree_hal_profile_tracy_clock_sample_t;

// Tick -> host time mapping derived from the retained samples.
//
// Rate and placement are fitted independently because they want different
// inputs. The rate is a single number every duration is computed from, so it
// wants the most precise time differences available - the driver's own
// timestamps, where a constant offset between clocks cancels. Placement has to
// land on the same timeline as host-timestamped zones, so it wants the host
// bracket even though that is noisier. Keeping them separate is what makes a
// reported duration the device tick delta scaled by one number, the same
// quantity the file sink and iree-profile report.
typedef struct iree_hal_profile_tracy_clock_map_t {
  bool valid;
  int64_t rate_num_ns;
  uint64_t rate_den_ticks;
  // Estimated relative error of the rate in parts per million.
  uint32_t rate_error_ppm;
} iree_hal_profile_tracy_clock_map_t;

typedef struct iree_hal_profile_tracy_device_t {
  // Session the clock state belongs to. Device ticks restart with the producer
  // so state is never shared across sessions.
  uint64_t session_id;
  // Producer identity hash; two drivers may both report ordinal 0.
  uint64_t producer_key;
  // Copy of the producer name, used when a drain has to create a context.
  char producer_name[64];
  uint8_t producer_name_length;
  uint32_t physical_device_ordinal;
  // Samples sorted by tick.
  iree_hal_profile_tracy_clock_sample_t samples
      [IREE_HAL_PROFILE_TRACY_MAX_CLOCK_SAMPLE_COUNT];
  iree_host_size_t sample_count;
  // Totals across the session, including samples that were dropped.
  uint32_t clock_sample_count;
  uint32_t invalid_clock_sample_count;
  uint32_t rejected_clock_sample_count;
  // Samples whose tick went backwards relative to the previous one to arrive,
  // which means the device counter was reset (suspend, power gating) and no
  // single mapping describes the session.
  uint32_t regressed_clock_sample_count;
  uint64_t last_arrival_tick;
  bool has_last_arrival_tick;
  // Host clock all of this device's anchors are expressed in.
  iree_hal_profile_tracy_anchor_domain_t anchor_domain;
  // Whether zones have already been handed over using this device's mapping.
  // Producers can invalidate a clock after the fact and those cannot be taken
  // back, so it is reported rather than quietly ignored.
  bool has_emitted;
  iree_hal_profile_tracy_clock_map_t map;
  // Events buffered until |map| becomes valid.
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

  // Number of producer sessions that have begun and not yet ended. Several
  // devices may share one sink so session-local tables are only reset when the
  // last session ends.
  uint32_t active_session_count;

  // Session-local executable function names, open-addressed by
  // (session_id, executable_id, function_ordinal).
  iree_hal_profile_tracy_function_t* functions;
  iree_host_size_t function_count;
  iree_host_size_t function_capacity;

  // Session-local per-physical-device clock state.
  iree_hal_profile_tracy_device_t* devices;
  iree_host_size_t device_count;
  iree_host_size_t device_capacity;

  // Zones staged by the chunk currently being written, ordered before emission.
  iree_hal_profile_tracy_zone_t* zones;
  iree_host_size_t zone_count;
  iree_host_size_t zone_capacity;

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

static uint64_t iree_hal_profile_tracy_hash_string(iree_string_view_t value) {
  // FNV-1a; only used to key tables by producer identity.
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    hash ^= (uint64_t)(uint8_t)value.data[i];
    hash *= UINT64_C(0x100000001b3);
  }
  return hash;
}

static iree_hal_profile_tracy_scope_t
iree_hal_profile_tracy_scope_from_metadata(
    const iree_hal_profile_chunk_metadata_t* metadata) {
  iree_hal_profile_tracy_scope_t scope;
  scope.session_id = metadata->session_id;
  scope.producer_name = metadata->name;
  scope.producer_key = iree_hal_profile_tracy_hash_string(metadata->name);
  return scope;
}

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

// Converts a driver system timestamp to nanoseconds.
static bool iree_hal_profile_tracy_clock_sample_host_system_timestamp(
    const iree_hal_profile_clock_correlation_record_t* sample,
    int64_t* out_time_ns) {
  if (!iree_all_bits_set(
          sample->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP) ||
      sample->host_system_frequency_hz == 0) {
    return false;
  }
  // Split into whole seconds and remainder so a high-frequency counter does not
  // overflow on the way to nanoseconds.
  const uint64_t seconds =
      sample->host_system_timestamp / sample->host_system_frequency_hz;
  const uint64_t remainder =
      sample->host_system_timestamp % sample->host_system_frequency_hz;
  uint64_t remainder_ns = 0;
  if (!iree_hal_profile_tracy_round_mul_div_u64(remainder, 1000000000ull,
                                                sample->host_system_frequency_hz,
                                                &remainder_ns)) {
    return false;
  }
  if (seconds > (uint64_t)INT64_MAX / 1000000000ull) return false;
  const uint64_t total_ns = seconds * 1000000000ull + remainder_ns;
  if (total_ns > (uint64_t)INT64_MAX) return false;
  *out_time_ns = (int64_t)total_ns;
  return true;
}

// Picks the host clock a device's anchors will be expressed in, from its first
// usable sample. Preference order: a driver timestamp that demonstrably reads
// the same clock as IREE (exact and in the right domain), then the host bracket
// (right domain but jittery), then a driver or system timestamp on its own
// (exact but possibly offset from the host timeline by a constant).
static iree_hal_profile_tracy_anchor_domain_t
iree_hal_profile_tracy_choose_anchor_domain(
    const iree_hal_profile_clock_correlation_record_t* record) {
  int64_t bracket_midpoint_ns = 0;
  const bool has_bracket = iree_hal_profile_tracy_clock_sample_host_midpoint(
      record, &bracket_midpoint_ns);
  int64_t driver_ns = 0;
  const bool has_driver =
      iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(record, &driver_ns);
  if (has_driver && has_bracket) {
    const int64_t bracket_width_ns =
        record->host_time_end_ns - record->host_time_begin_ns;
    // A zero-width bracket would demand bit-exact equality and a very wide one
    // proves nothing, so only a tight bracket can establish agreement.
    if (bracket_width_ns > 0 &&
        bracket_width_ns <=
            IREE_HAL_PROFILE_TRACY_CLOCK_MAX_AGREEMENT_BRACKET_NS &&
        driver_ns >= record->host_time_begin_ns &&
        driver_ns <= record->host_time_end_ns) {
      return IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_DRIVER;
    }
  }
  if (has_bracket) return IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_BRACKET;
  if (has_driver) return IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_DRIVER;
  int64_t system_ns = 0;
  if (iree_hal_profile_tracy_clock_sample_host_system_timestamp(record,
                                                                &system_ns)) {
    return IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_SYSTEM;
  }
  return IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_UNSET;
}

// Reduces a clock-correlation record to a sample in the device's chosen anchor
// domain. Returns false when the record cannot supply that domain.
static bool iree_hal_profile_tracy_clock_sample_import(
    iree_hal_profile_tracy_anchor_domain_t domain,
    const iree_hal_profile_clock_correlation_record_t* record,
    iree_hal_profile_tracy_clock_sample_t* out_sample) {
  memset(out_sample, 0, sizeof(*out_sample));
  if (!iree_all_bits_set(record->flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
    return false;
  }
  out_sample->tick = record->device_tick;
  out_sample->uncertainty_ns =
      IREE_HAL_PROFILE_TRACY_CLOCK_UNCERTAINTY_FLOOR_NS;
  out_sample->has_driver_ns =
      iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(
          record, &out_sample->driver_ns);
  switch (domain) {
    case IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_DRIVER:
      return iree_hal_profile_tracy_clock_sample_host_cpu_timestamp(
          record, &out_sample->anchor_ns);
    case IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_BRACKET: {
      if (!iree_hal_profile_tracy_clock_sample_host_midpoint(
              record, &out_sample->anchor_ns)) {
        return false;
      }
      const int64_t half_width_ns =
          (record->host_time_end_ns - record->host_time_begin_ns) / 2;
      if (half_width_ns > out_sample->uncertainty_ns) {
        out_sample->uncertainty_ns = half_width_ns;
      }
      return true;
    }
    case IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_SYSTEM:
      return iree_hal_profile_tracy_clock_sample_host_system_timestamp(
          record, &out_sample->anchor_ns);
    default:
      return false;
  }
}

// Inserts |sample| into the device's tick-sorted table. Returns false if the
// tick is already present, as a zero-length segment carries no information and
// would divide by zero.
static bool iree_hal_profile_tracy_device_insert_sample(
    iree_hal_profile_tracy_device_t* device,
    const iree_hal_profile_tracy_clock_sample_t* sample) {
  iree_host_size_t index = device->sample_count;
  while (index > 0 && device->samples[index - 1].tick > sample->tick) --index;
  if (index > 0 && device->samples[index - 1].tick == sample->tick) return false;

  if (device->sample_count == IREE_HAL_PROFILE_TRACY_MAX_CLOCK_SAMPLE_COUNT) {
    // Drop the oldest interior sample. Samples are rejected if their tick
    // regresses, so table order is arrival order and index 1 is the oldest one
    // that is not holding up an end of the baseline.
    memmove(&device->samples[1], &device->samples[2],
            (device->sample_count - 2) * sizeof(device->samples[0]));
    --device->sample_count;
    if (index > 1) --index;
  }
  memmove(&device->samples[index + 1], &device->samples[index],
          (device->sample_count - index) * sizeof(device->samples[0]));
  device->samples[index] = *sample;
  ++device->sample_count;
  return true;
}

// Recomputes the tick -> host time mapping from the retained samples.
//
// The rate is fitted over the sample pair with the smallest relative error, not
// simply the outermost pair: one descheduled sampling call at an end would
// otherwise set the rate for the whole session and decide whether it is trusted
// along with it. When |relaxed| the quality gate is skipped, which happens once
// at session end so that events held behind a short baseline are still placed.
static void iree_hal_profile_tracy_device_refit(
    iree_hal_profile_tracy_device_t* device, bool relaxed) {
  iree_hal_profile_tracy_clock_map_t map;
  memset(&map, 0, sizeof(map));
  if (device->sample_count < 2 || device->invalid_clock_sample_count != 0 ||
      device->regressed_clock_sample_count != 0) {
    // Either there is nothing to fit, the producer reported that its samples do
    // not cover the ticks it emitted, or the device counter was reset partway
    // through - in each case no single mapping describes the session, which is
    // the same conclusion the file sink and iree-profile reach.
    device->map = map;
    return;
  }

  iree_host_size_t candidates = IREE_HAL_PROFILE_TRACY_CLOCK_RATE_CANDIDATE_COUNT;
  if (candidates > device->sample_count) candidates = device->sample_count;
  uint64_t best_error_ppm = UINT64_MAX;
  for (iree_host_size_t i = 0; i < candidates; ++i) {
    for (iree_host_size_t k = 0; k < candidates; ++k) {
      const iree_host_size_t j = device->sample_count - 1 - k;
      if (j <= i) continue;
      const iree_hal_profile_tracy_clock_sample_t* first = &device->samples[i];
      const iree_hal_profile_tracy_clock_sample_t* last = &device->samples[j];
      const uint64_t tick_span = last->tick - first->tick;
      if (tick_span == 0) continue;

      // Two ways to measure how much host time those ticks took. The driver's
      // timestamps are exact but may run at a slightly different rate than the
      // host clock the anchors use; the anchors are in the right clock but
      // carry the bracket jitter of both endpoints. Take whichever is expected
      // to be closer over this particular baseline: jitter shrinks with the
      // baseline while the rate difference between two clocks does not, so
      // short runs want the driver and long ones eventually want the anchors.
      const int64_t anchor_span_ns = last->anchor_ns - first->anchor_ns;
      if (anchor_span_ns > 0) {
        const int64_t uncertainty_ns =
            first->uncertainty_ns + last->uncertainty_ns;
        const uint64_t error_ppm =
            (uint64_t)((uncertainty_ns * INT64_C(1000000)) / anchor_span_ns);
        if (error_ppm < best_error_ppm) {
          best_error_ppm = error_ppm;
          map.rate_num_ns = anchor_span_ns;
          map.rate_den_ticks = tick_span;
        }
      }
      if (first->has_driver_ns && last->has_driver_ns) {
        const int64_t driver_span_ns = last->driver_ns - first->driver_ns;
        if (driver_span_ns > 0) {
          const int64_t uncertainty_ns =
              2 * IREE_HAL_PROFILE_TRACY_CLOCK_UNCERTAINTY_FLOOR_NS;
          uint64_t error_ppm =
              (uint64_t)((uncertainty_ns * INT64_C(1000000)) / driver_span_ns);
          if (device->anchor_domain !=
              IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_DRIVER) {
            // Anchors and driver timestamps are not known to be the same clock,
            // so charge this estimate the rate difference two host clocks can
            // drift apart by.
            error_ppm += IREE_HAL_PROFILE_TRACY_CLOCK_DOMAIN_SLEW_PPM;
          }
          if (error_ppm < best_error_ppm) {
            best_error_ppm = error_ppm;
            map.rate_num_ns = driver_span_ns;
            map.rate_den_ticks = tick_span;
          }
        }
      }
    }
  }
  if (best_error_ppm == UINT64_MAX) {
    device->map = map;
    return;
  }
  map.rate_error_ppm =
      (uint32_t)(best_error_ppm > UINT32_MAX ? UINT32_MAX : best_error_ppm);
  map.valid =
      relaxed ||
      best_error_ppm < IREE_HAL_PROFILE_TRACY_CLOCK_MAX_RATE_ERROR_PPM;
  device->map = map;
}

// Places a device tick on the host timeline.
//
// Inside the sampled range the placement interpolates between the neighboring
// anchors, so it is continuous wherever consecutive samples disagree with the
// fitted rate; outside it extrapolates from the nearest end at that rate. Only
// use this for a span's start: deriving its end from the rate is what keeps
// durations independent of where the anchors fell.
static bool iree_hal_profile_tracy_device_map_tick(
    const iree_hal_profile_tracy_device_t* device, uint64_t tick,
    int64_t* out_host_time_ns) {
  *out_host_time_ns = 0;
  if (!device->map.valid || device->sample_count < 2) return false;
  const iree_hal_profile_tracy_clock_sample_t* samples = device->samples;
  const iree_host_size_t count = device->sample_count;

  if (tick >= samples[0].tick && tick <= samples[count - 1].tick) {
    iree_host_size_t lo = 0;
    iree_host_size_t hi = count - 1;
    while (hi - lo > 1) {
      const iree_host_size_t mid = lo + (hi - lo) / 2;
      if (samples[mid].tick <= tick) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const iree_hal_profile_tracy_clock_sample_t* first = &samples[lo];
    const iree_hal_profile_tracy_clock_sample_t* last = &samples[hi];
    const uint64_t segment_ticks = last->tick - first->tick;
    const int64_t segment_ns = last->anchor_ns - first->anchor_ns;
    if (segment_ticks != 0 && segment_ns >= 0) {
      uint64_t delta_ns = 0;
      if (!iree_hal_profile_tracy_round_mul_div_u64(
              tick - first->tick, (uint64_t)segment_ns, segment_ticks,
              &delta_ns) ||
          delta_ns > (uint64_t)(INT64_MAX - first->anchor_ns)) {
        return false;
      }
      *out_host_time_ns = first->anchor_ns + (int64_t)delta_ns;
      return true;
    }
  }

  // Outside the sampled range: extrapolate from the nearer end.
  const iree_hal_profile_tracy_clock_sample_t* anchor =
      tick < samples[0].tick ? &samples[0] : &samples[count - 1];
  const uint64_t distance =
      tick >= anchor->tick ? tick - anchor->tick : anchor->tick - tick;
  uint64_t delta_ns = 0;
  if (!iree_hal_profile_tracy_round_mul_div_u64(
          distance, (uint64_t)device->map.rate_num_ns,
          device->map.rate_den_ticks, &delta_ns)) {
    return false;
  }
  if (tick >= anchor->tick) {
    if (delta_ns > (uint64_t)(INT64_MAX - anchor->anchor_ns)) return false;
    *out_host_time_ns = anchor->anchor_ns + (int64_t)delta_ns;
  } else {
    if (delta_ns > (uint64_t)INT64_MAX) return false;
    *out_host_time_ns = anchor->anchor_ns - (int64_t)delta_ns;
  }
  return true;
}

// Places a device-timestamped span on the host timeline.
//
// The start is placed through the anchors and the end is derived from the start
// plus the fitted rate. Mapping both ends independently would fold the
// disagreement between two neighboring anchors into the duration of any span
// that straddles a clock sample - in practice the long ones, which are the ones
// being measured - and would make the same run report different kernel times
// here than through the file sink.
static bool iree_hal_profile_tracy_device_map_span(
    const iree_hal_profile_tracy_device_t* device, uint64_t start_tick,
    uint64_t end_tick, int64_t* out_start_host_time_ns,
    int64_t* out_end_host_time_ns);

// Converts a tick count to a duration at the fitted rate.
static bool iree_hal_profile_tracy_device_map_duration(
    const iree_hal_profile_tracy_device_t* device, uint64_t tick_count,
    int64_t* out_duration_ns) {
  *out_duration_ns = 0;
  if (!device->map.valid) return false;
  uint64_t duration_ns = 0;
  if (!iree_hal_profile_tracy_round_mul_div_u64(
          tick_count, (uint64_t)device->map.rate_num_ns,
          device->map.rate_den_ticks, &duration_ns) ||
      duration_ns > (uint64_t)INT64_MAX) {
    return false;
  }
  *out_duration_ns = (int64_t)duration_ns;
  return true;
}

static bool iree_hal_profile_tracy_device_map_span(
    const iree_hal_profile_tracy_device_t* device, uint64_t start_tick,
    uint64_t end_tick, int64_t* out_start_host_time_ns,
    int64_t* out_end_host_time_ns) {
  *out_start_host_time_ns = 0;
  *out_end_host_time_ns = 0;
  if (end_tick < start_tick) return false;
  int64_t start_host_time_ns = 0;
  int64_t duration_ns = 0;
  if (!iree_hal_profile_tracy_device_map_tick(device, start_tick,
                                              &start_host_time_ns) ||
      !iree_hal_profile_tracy_device_map_duration(device, end_tick - start_tick,
                                                  &duration_ns) ||
      duration_ns > INT64_MAX - start_host_time_ns) {
    return false;
  }
  *out_start_host_time_ns = start_host_time_ns;
  *out_end_host_time_ns = start_host_time_ns + duration_ns;
  return true;
}

//===----------------------------------------------------------------------===//
// Tables
//===----------------------------------------------------------------------===//

// Whether the producer has told us, one way or another, that no single mapping
// describes this device's session.
static bool iree_hal_profile_tracy_device_clock_is_invalid(
    const iree_hal_profile_tracy_device_t* device) {
  return device->invalid_clock_sample_count != 0 ||
         device->regressed_clock_sample_count != 0;
}

static iree_hal_profile_tracy_scope_t iree_hal_profile_tracy_device_scope(
    const iree_hal_profile_tracy_device_t* device) {
  iree_hal_profile_tracy_scope_t scope;
  scope.session_id = device->session_id;
  scope.producer_key = device->producer_key;
  scope.producer_name = iree_make_string_view(device->producer_name,
                                              device->producer_name_length);
  return scope;
}

static iree_status_t iree_hal_profile_tracy_sink_ensure_device(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_tracy_scope_t* scope,
    uint32_t physical_device_ordinal,
    iree_hal_profile_tracy_device_t** out_device) {
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    if (sink->devices[i].session_id == scope->session_id &&
        sink->devices[i].producer_key == scope->producer_key &&
        sink->devices[i].physical_device_ordinal == physical_device_ordinal) {
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
  device->session_id = scope->session_id;
  device->producer_key = scope->producer_key;
  iree_host_size_t producer_name_length = scope->producer_name.size;
  if (producer_name_length > sizeof(device->producer_name)) {
    producer_name_length = sizeof(device->producer_name);
  }
  memcpy(device->producer_name, scope->producer_name.data,
         producer_name_length);
  device->producer_name_length = (uint8_t)producer_name_length;
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static uint64_t iree_hal_profile_tracy_function_hash(
    uint64_t session_id, uint64_t executable_id, uint32_t function_ordinal) {
  return iree_hal_profile_tracy_mix_u64(session_id) ^
         iree_hal_profile_tracy_mix_u64(executable_id) ^
         iree_hal_profile_tracy_mix_u64((uint64_t)function_ordinal |
                                        (UINT64_C(1) << 40));
}

static iree_hal_profile_tracy_function_t*
iree_hal_profile_tracy_sink_find_function_slot(
    iree_hal_profile_tracy_sink_t* sink, uint64_t session_id,
    uint64_t executable_id, uint32_t function_ordinal) {
  if (sink->function_capacity == 0) return NULL;
  const uint64_t hash = iree_hal_profile_tracy_function_hash(
      session_id, executable_id, function_ordinal);
  const iree_host_size_t mask = sink->function_capacity - 1;
  for (iree_host_size_t probe = 0; probe < sink->function_capacity; ++probe) {
    iree_hal_profile_tracy_function_t* slot =
        &sink->functions[(hash + probe) & mask];
    if (!slot->occupied) return slot;
    if (slot->session_id == session_id &&
        slot->executable_id == executable_id &&
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
            sink, old_functions[i].session_id, old_functions[i].executable_id,
            old_functions[i].function_ordinal);
    *slot = old_functions[i];
  }
  iree_allocator_free(sink->host_allocator, old_functions);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_insert_function(
    iree_hal_profile_tracy_sink_t* sink, uint64_t session_id,
    uint64_t executable_id, uint32_t function_ordinal,
    iree_string_view_t name) {
  if ((sink->function_count + 1) * 2 > sink->function_capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_profile_tracy_sink_grow_functions(sink));
  }
  iree_hal_profile_tracy_function_t* slot =
      iree_hal_profile_tracy_sink_find_function_slot(
          sink, session_id, executable_id, function_ordinal);
  if (slot->occupied) {
    // Producers may re-emit metadata on flush; keep the first name.
    return iree_ok_status();
  }
  char* name_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      sink->host_allocator, name.size ? name.size : 1, (void**)&name_storage));
  memcpy(name_storage, name.data, name.size);
  slot->occupied = true;
  slot->session_id = session_id;
  slot->executable_id = executable_id;
  slot->function_ordinal = function_ordinal;
  slot->name = name_storage;
  slot->name_length = (uint32_t)name.size;
  ++sink->function_count;
  return iree_ok_status();
}

static bool iree_hal_profile_tracy_sink_lookup_function(
    iree_hal_profile_tracy_sink_t* sink, uint64_t session_id,
    uint64_t executable_id, uint32_t function_ordinal,
    iree_string_view_t* out_name) {
  iree_hal_profile_tracy_function_t* slot =
      iree_hal_profile_tracy_sink_find_function_slot(
          sink, session_id, executable_id, function_ordinal);
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
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_tracy_scope_t* scope,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    iree_hal_profile_tracy_lane_t lane) {
  // Contexts outlive sessions (the tracing provider cannot release them) so a
  // device profiled twice reuses its timeline, but two producers are never
  // merged just because they picked the same ordinal.
  for (iree_host_size_t i = 0; i < sink->context_count; ++i) {
    iree_hal_profile_tracy_context_t* context = &sink->contexts[i];
    if (context->producer_key == scope->producer_key &&
        context->physical_device_ordinal == physical_device_ordinal &&
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

  char name[96];
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
  // Producers that identify themselves get their name appended so two drivers
  // reporting the same ordinal stay distinguishable in the timeline.
  if (scope->producer_name.size > 0) {
    const int suffix_length =
        snprintf(name + name_length, sizeof(name) - (size_t)name_length,
                 " [%.*s]", (int)scope->producer_name.size,
                 scope->producer_name.data);
    if (suffix_length > 0) {
      name_length += suffix_length;
      if ((size_t)name_length >= sizeof(name)) name_length = sizeof(name) - 1;
    }
  }

  // Timestamps handed to the provider are already host nanoseconds so the
  // context maps host time 1:1 onto the tracing clock via the reference pair.
  iree_hal_profile_tracy_context_t* context =
      &sink->contexts[sink->context_count++];
  memset(context, 0, sizeof(*context));
  context->producer_key = scope->producer_key;
  context->physical_device_ordinal = physical_device_ordinal;
  context->queue_ordinal = queue_ordinal;
  context->lane = lane;
  // Uncalibrated: the timestamps handed over are already host nanoseconds, so
  // the provider only needs the fixed offset it derives from the reference
  // pair. Declaring calibration would additionally cost the viewer's manual
  // drift control and buys nothing without periodic calibration events.
  context->tracy_id = iree_tracing_gpu_context_allocate(
      IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN, name, (size_t)name_length,
      /*is_calibrated=*/false, (uint64_t)sink->reference_tracing_time,
      (uint64_t)sink->reference_host_time_ns, /*timestamp_period=*/1.0f);
  if (context->tracy_id == IREE_TRACING_GPU_CONTEXT_ID_INVALID) {
    // The process has used up the provider's 255 context ids; drop this lane
    // rather than emitting against an id another context already owns.
    --sink->context_count;
    return NULL;
  }
  ++sink->statistics.context_count;
  return context;
}

static int iree_hal_profile_tracy_zone_compare(const void* lhs,
                                               const void* rhs) {
  const iree_hal_profile_tracy_zone_t* a =
      (const iree_hal_profile_tracy_zone_t*)lhs;
  const iree_hal_profile_tracy_zone_t* b =
      (const iree_hal_profile_tracy_zone_t*)rhs;
  if (a->context_index != b->context_index) {
    return a->context_index < b->context_index ? -1 : 1;
  }
  if (a->start_host_time_ns != b->start_host_time_ns) {
    return a->start_host_time_ns < b->start_host_time_ns ? -1 : 1;
  }
  if (a->end_host_time_ns != b->end_host_time_ns) {
    return a->end_host_time_ns < b->end_host_time_ns ? -1 : 1;
  }
  // Keeps the sort deterministic where the spans are identical.
  return a->sequence < b->sequence ? -1 : (a->sequence > b->sequence ? 1 : 0);
}

// Resolves the display name for an event: the executable function name when
// known, otherwise the queue operation type or a synthesized fallback.
static iree_string_view_t iree_hal_profile_tracy_sink_event_name(
    iree_hal_profile_tracy_sink_t* sink, uint64_t session_id,
    uint64_t executable_id, uint32_t function_ordinal,
    iree_hal_profile_queue_event_type_t type, char* scratch,
    size_t scratch_length) {
  iree_string_view_t name = iree_string_view_empty();
  if (executable_id != 0 && function_ordinal != UINT32_MAX) {
    if (iree_hal_profile_tracy_sink_lookup_function(
            sink, session_id, executable_id, function_ordinal, &name)) {
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

// Stages a host-timestamped span for emission.
static void iree_hal_profile_tracy_sink_queue_host_span(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_tracy_scope_t* scope,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    iree_hal_profile_tracy_lane_t lane, uint64_t executable_id,
    uint32_t function_ordinal, iree_hal_profile_queue_event_type_t type,
    int64_t start_host_time_ns, int64_t end_host_time_ns) {
  if (start_host_time_ns <= 0 || end_host_time_ns < start_host_time_ns) {
    ++sink->statistics.invalid_event_count;
    return;
  }
  iree_hal_profile_tracy_context_t* context =
      iree_hal_profile_tracy_sink_acquire_context(
          sink, scope, physical_device_ordinal, queue_ordinal, lane);
  if (!context) {
    ++sink->statistics.context_exhausted_event_count;
    return;
  }
  // The context table can be reallocated by a later acquire, so zones remember
  // where their context is rather than a pointer to it.
  const iree_host_size_t context_index =
      (iree_host_size_t)(context - sink->contexts);
  if (sink->zone_count == sink->zone_capacity) {
    const iree_host_size_t new_capacity =
        sink->zone_capacity ? sink->zone_capacity * 2 : 256;
    if (!iree_status_is_ok(iree_allocator_realloc(
            sink->host_allocator, new_capacity * sizeof(*sink->zones),
            (void**)&sink->zones))) {
      // With nowhere to stage it, emitting it now would put this context's
      // timeline out of order for the rest of the capture.
      ++sink->statistics.invalid_event_count;
      return;
    }
    sink->zone_capacity = new_capacity;
  }
  iree_hal_profile_tracy_zone_t* zone = &sink->zones[sink->zone_count];
  memset(zone, 0, sizeof(*zone));
  zone->context_index = context_index;
  zone->sequence = (uint32_t)sink->zone_count;
  zone->start_host_time_ns = start_host_time_ns;
  zone->end_host_time_ns = end_host_time_ns;
  zone->session_id = scope->session_id;
  zone->executable_id = executable_id;
  zone->function_ordinal = function_ordinal;
  zone->type = (uint32_t)type;
  ++sink->zone_count;
}

// Emits one staged zone as a complete begin/end pair with its two timestamps.
static void iree_hal_profile_tracy_sink_emit_zone(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_tracy_zone_t* zone) {
  iree_hal_profile_tracy_context_t* context =
      &sink->contexts[zone->context_index];
  if (context->has_emitted &&
      zone->start_host_time_ns <
          context->last_start_ns -
              IREE_HAL_PROFILE_TRACY_MAX_BACKWARD_STEP_NS) {
    // Handing this over would look like a counter wrap to the server and would
    // displace every later zone on this context; losing one zone is cheaper.
    ++sink->statistics.reordered_event_count;
    return;
  }
  char scratch[64];
  const iree_string_view_t name = iree_hal_profile_tracy_sink_event_name(
      sink, zone->session_id, zone->executable_id, zone->function_ordinal,
      (iree_hal_profile_queue_event_type_t)zone->type, scratch,
      sizeof(scratch));
  const uint16_t begin_query_id = context->next_query_id++;
  const uint16_t end_query_id = context->next_query_id++;
  iree_tracing_gpu_zone_begin_external(
      context->tracy_id, begin_query_id, kIreeHalProfileTracyFile,
      sizeof(kIreeHalProfileTracyFile) - 1, /*line=*/0, name.data, name.size,
      name.data, name.size);
  iree_tracing_gpu_zone_end(context->tracy_id, end_query_id);
  iree_tracing_gpu_zone_notify(context->tracy_id, begin_query_id,
                               zone->start_host_time_ns);
  iree_tracing_gpu_zone_notify(context->tracy_id, end_query_id,
                               zone->end_host_time_ns);
  if (!context->has_emitted ||
      zone->start_host_time_ns > context->last_start_ns) {
    context->last_start_ns = zone->start_host_time_ns;
  }
  context->has_emitted = true;
  ++sink->statistics.emitted_zone_count;
}

// Orders everything staged so far and hands it to the tracing provider.
static void iree_hal_profile_tracy_sink_flush_zones(
    iree_hal_profile_tracy_sink_t* sink) {
  if (sink->zone_count == 0) return;
  if (sink->zone_count > 1) {
    qsort(sink->zones, sink->zone_count, sizeof(sink->zones[0]),
          iree_hal_profile_tracy_zone_compare);
  }
  for (iree_host_size_t i = 0; i < sink->zone_count; ++i) {
    iree_hal_profile_tracy_sink_emit_zone(sink, &sink->zones[i]);
  }
  sink->zone_count = 0;
}

// Emits a device-timestamped event using the device's current clock mapping.
static void iree_hal_profile_tracy_sink_emit_device_event(
    iree_hal_profile_tracy_sink_t* sink,
    iree_hal_profile_tracy_device_t* device,
    const iree_hal_profile_tracy_scope_t* scope,
    const iree_hal_profile_tracy_pending_event_t* event) {
  int64_t start_host_time_ns = 0;
  int64_t end_host_time_ns = 0;
  if (!iree_hal_profile_tracy_device_map_span(device, event->start_tick,
                                              event->end_tick,
                                              &start_host_time_ns,
                                              &end_host_time_ns)) {
    ++sink->statistics.invalid_event_count;
    return;
  }
  device->has_emitted = true;
  iree_hal_profile_tracy_sink_queue_host_span(
      sink, scope, device->physical_device_ordinal, event->queue_ordinal,
      event->lane, event->executable_id, event->function_ordinal, event->type,
      start_host_time_ns, end_host_time_ns);
}

// Emits or buffers a device-timestamped event depending on whether a clock
// mapping exists for its device yet.
static iree_status_t iree_hal_profile_tracy_sink_handle_device_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_tracy_scope_t* scope,
    uint32_t physical_device_ordinal,
    const iree_hal_profile_tracy_pending_event_t* event) {
  if (event->start_tick == 0 || event->end_tick < event->start_tick) {
    ++sink->statistics.invalid_event_count;
    return iree_ok_status();
  }
  iree_hal_profile_tracy_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_tracy_sink_ensure_device(
      sink, scope, physical_device_ordinal, &device));
  if (device->map.valid) {
    iree_hal_profile_tracy_sink_emit_device_event(sink, device, scope, event);
    return iree_ok_status();
  }
  if (iree_hal_profile_tracy_device_clock_is_invalid(device)) {
    // The producer has already reported that no mapping describes this device's
    // session, so none will arrive; buffering these only to discard them at the
    // end wastes memory.
    ++sink->statistics.unaligned_event_count;
    return iree_ok_status();
  }
  if (device->pending_count >= IREE_HAL_PROFILE_TRACY_MAX_PENDING_EVENT_COUNT) {
    ++sink->statistics.pending_overflow_count;
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
  if (!device->map.valid) {
    if (!force) return;
    // Last chance before the session closes: place these with whatever rate the
    // samples support, which beats discarding a whole device timeline because
    // the producer only flushed twice in quick succession.
    iree_hal_profile_tracy_device_refit(device, /*relaxed=*/true);
    if (device->map.valid) {
      sink->statistics.degraded_event_count += device->pending_count;
    }
  }
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_device_scope(device);
  for (iree_host_size_t i = 0; i < device->pending_count; ++i) {
    if (device->map.valid) {
      iree_hal_profile_tracy_sink_emit_device_event(sink, device, &scope,
                                                    &device->pending[i]);
    } else if (iree_hal_profile_tracy_device_clock_is_invalid(device)) {
      ++sink->statistics.unaligned_event_count;
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
      sink, metadata->session_id, function_record.executable_id,
      function_record.function_ordinal, name);
}

static iree_status_t iree_hal_profile_tracy_sink_process_clock_correlation(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)record_length;
  iree_hal_profile_clock_correlation_record_t correlation;
  memcpy(&correlation, record, sizeof(correlation));
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_scope_from_metadata(metadata);
  iree_hal_profile_tracy_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_profile_tracy_sink_ensure_device(
      sink, &scope, correlation.physical_device_ordinal, &device));
  ++device->clock_sample_count;
  const bool was_invalid =
      iree_hal_profile_tracy_device_clock_is_invalid(device);
  const bool has_device_tick = iree_all_bits_set(
      correlation.flags, IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK);

  bool usable = false;
  if (!has_device_tick) {
    // Nothing to anchor: the record may still carry host times but they pair
    // with no point on the device clock.
    ++device->rejected_clock_sample_count;
    ++sink->statistics.rejected_clock_sample_count;
  } else if (iree_any_bit_set(
                 correlation.flags,
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED)) {
    // Only meaningful alongside a tick: on its own the flag says nothing about
    // a mapping and must not take the device down.
    ++device->invalid_clock_sample_count;
  } else if (device->has_last_arrival_tick &&
             correlation.device_tick < device->last_arrival_tick) {
    // The counter went backwards between two samples, so it was reset (suspend,
    // power gating) and no single rate spans the session. Holding the table in
    // tick order would otherwise hide this behind a well-formed but meaningless
    // series.
    ++device->regressed_clock_sample_count;
    ++sink->statistics.clock_regression_count;
  } else if (device->has_last_arrival_tick &&
             correlation.device_tick == device->last_arrival_tick) {
    // The same tick observed twice - an idle device between two flushes. It
    // adds nothing and a zero-length segment would divide by zero.
    ++device->rejected_clock_sample_count;
    ++sink->statistics.rejected_clock_sample_count;
  } else {
    device->last_arrival_tick = correlation.device_tick;
    device->has_last_arrival_tick = true;
    usable = true;
  }

  if (usable) {
    if (device->anchor_domain == IREE_HAL_PROFILE_TRACY_ANCHOR_DOMAIN_UNSET) {
      device->anchor_domain =
          iree_hal_profile_tracy_choose_anchor_domain(&correlation);
    }
    iree_hal_profile_tracy_clock_sample_t sample;
    if (!iree_hal_profile_tracy_clock_sample_import(device->anchor_domain,
                                                    &correlation, &sample) ||
        !iree_hal_profile_tracy_device_insert_sample(device, &sample)) {
      ++device->rejected_clock_sample_count;
      ++sink->statistics.rejected_clock_sample_count;
    }
  }

  if (!was_invalid && device->has_emitted &&
      iree_hal_profile_tracy_device_clock_is_invalid(device)) {
    // Zones for this device are already in the capture and cannot be taken
    // back. Unlike a consumer reading a finished file, a streaming sink commits
    // before it has seen every sample, so this is reported rather than implied.
    ++sink->statistics.late_clock_invalidation_count;
  }

  iree_hal_profile_tracy_device_refit(device, /*relaxed=*/false);
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
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_scope_from_metadata(metadata);
  return iree_hal_profile_tracy_sink_handle_device_event(
      sink, &scope, metadata->physical_device_ordinal, &event);
}

static iree_status_t iree_hal_profile_tracy_sink_process_queue_device_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
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
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_scope_from_metadata(metadata);
  return iree_hal_profile_tracy_sink_handle_device_event(
      sink, &scope, queue_event.physical_device_ordinal, &event);
}

static iree_status_t iree_hal_profile_tracy_sink_process_host_execution_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)record_length;
  iree_hal_profile_host_execution_event_t event;
  memcpy(&event, record, sizeof(event));
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_scope_from_metadata(metadata);
  iree_hal_profile_tracy_sink_queue_host_span(
      sink, &scope, event.physical_device_ordinal, event.queue_ordinal,
      IREE_HAL_PROFILE_TRACY_LANE_HOST_EXECUTION, event.executable_id,
      event.function_ordinal, event.type, event.start_host_time_ns,
      event.end_host_time_ns);
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_tracy_sink_process_queue_event(
    iree_hal_profile_tracy_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata, const uint8_t* record,
    iree_host_size_t record_length) {
  (void)record_length;
  iree_hal_profile_queue_event_t event;
  memcpy(&event, record, sizeof(event));
  // Readiness is not observable for every producer; a submission without it
  // has no span to draw.
  if (event.ready_host_time_ns == 0) return iree_ok_status();
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_scope_from_metadata(metadata);
  iree_hal_profile_tracy_sink_queue_host_span(
      sink, &scope, event.physical_device_ordinal, event.queue_ordinal,
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
  iree_allocator_free(host_allocator, sink->zones);
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
  // Executable ids and clock samples are session-local, but one sink can be
  // shared by several devices that each open their own session. Clearing on
  // every begin would discard the other session's executable names and its
  // buffered events, so the tables only reset when none is live.
  if (sink->active_session_count == 0) {
    iree_hal_profile_tracy_sink_reset_session_tables(sink);
  }
  ++sink->active_session_count;
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

  iree_hal_profile_tracy_sink_flush_zones(sink);
  iree_slim_mutex_unlock(&sink->mutex);
  return status;
}

static iree_status_t iree_hal_profile_tracy_sink_end_session(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)session_status_code;
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  iree_slim_mutex_lock(&sink->mutex);

  // Whatever this session buffered gets the best mapping we have or is dropped.
  // Sessions that are still open keep theirs: another device may yet send the
  // clock sample that maps them.
  const iree_hal_profile_tracy_scope_t scope =
      iree_hal_profile_tracy_scope_from_metadata(metadata);
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    if (sink->devices[i].session_id != scope.session_id ||
        sink->devices[i].producer_key != scope.producer_key) {
      continue;
    }
    iree_hal_profile_tracy_sink_drain_device(sink, &sink->devices[i],
                                             /*force=*/true);
  }
  iree_hal_profile_tracy_sink_flush_zones(sink);
  if (sink->active_session_count > 0) --sink->active_session_count;
  if (sink->active_session_count != 0) {
    iree_slim_mutex_unlock(&sink->mutex);
    return iree_ok_status();
  }

  // Nothing further can arrive: flush anything a producer whose end-of-session
  // metadata did not match its chunks left behind rather than losing it.
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    iree_hal_profile_tracy_sink_drain_device(sink, &sink->devices[i],
                                             /*force=*/true);
  }
  iree_hal_profile_tracy_sink_flush_zones(sink);

  const iree_hal_profile_tracy_sink_statistics_t* statistics =
      &sink->statistics;
  if (statistics->unmapped_event_count != 0 ||
      statistics->unaligned_event_count != 0 ||
      statistics->degraded_event_count != 0 ||
      statistics->pending_overflow_count != 0 ||
      statistics->reordered_event_count != 0 ||
      statistics->clock_regression_count != 0 ||
      statistics->late_clock_invalidation_count != 0 ||
      statistics->dropped_record_count != 0 ||
      statistics->rejected_clock_sample_count != 0 ||
      statistics->context_exhausted_event_count != 0) {
    char message[384];
    int length =
        snprintf(message, sizeof(message),
                 "HAL profile sink: %" PRIu64 " events without a clock mapping"
                 ", %" PRIu64 " on a device clock the producer flagged as"
                 " unaligned, %" PRIu64 " placed with a low-confidence tick"
                 " rate, %" PRIu64 " over the buffering limit, %" PRIu64
                 " over the context limit, %" PRIu64
                 " reordered past the safe window, %" PRIu64
                 " records dropped by producers, %" PRIu64
                 " clock samples rejected, %" PRIu64
                 " device clock resets, %" PRIu64
                 " device clocks invalidated after zones were emitted",
                 statistics->unmapped_event_count,
                 statistics->unaligned_event_count,
                 statistics->degraded_event_count,
                 statistics->pending_overflow_count,
                 statistics->context_exhausted_event_count,
                 statistics->reordered_event_count,
                 statistics->dropped_record_count,
                 statistics->rejected_clock_sample_count,
                 statistics->clock_regression_count,
                 statistics->late_clock_invalidation_count);
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

IREE_API_EXPORT bool iree_hal_profile_tracy_sink_map_device_span_for_testing(
    iree_hal_profile_sink_t* base_sink, uint64_t session_id,
    iree_string_view_t producer_name, uint32_t physical_device_ordinal,
    uint64_t start_tick, uint64_t end_tick, int64_t* out_start_host_time_ns,
    int64_t* out_end_host_time_ns) {
  IREE_ASSERT_ARGUMENT(out_start_host_time_ns);
  IREE_ASSERT_ARGUMENT(out_end_host_time_ns);
  *out_start_host_time_ns = 0;
  *out_end_host_time_ns = 0;
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  const uint64_t producer_key =
      iree_hal_profile_tracy_hash_string(producer_name);
  bool mapped = false;
  iree_slim_mutex_lock(&sink->mutex);
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    const iree_hal_profile_tracy_device_t* device = &sink->devices[i];
    if (device->session_id != session_id ||
        device->producer_key != producer_key ||
        device->physical_device_ordinal != physical_device_ordinal) {
      continue;
    }
    mapped = iree_hal_profile_tracy_device_map_span(
        device, start_tick, end_tick, out_start_host_time_ns,
        out_end_host_time_ns);
    break;
  }
  iree_slim_mutex_unlock(&sink->mutex);
  return mapped;
}

IREE_API_EXPORT bool iree_hal_profile_tracy_sink_map_device_tick_for_testing(
    iree_hal_profile_sink_t* base_sink, uint64_t session_id,
    iree_string_view_t producer_name, uint32_t physical_device_ordinal,
    uint64_t device_tick, int64_t* out_host_time_ns) {
  IREE_ASSERT_ARGUMENT(out_host_time_ns);
  *out_host_time_ns = 0;
  iree_hal_profile_tracy_sink_t* sink =
      iree_hal_profile_tracy_sink_cast(base_sink);
  const uint64_t producer_key =
      iree_hal_profile_tracy_hash_string(producer_name);
  bool mapped = false;
  iree_slim_mutex_lock(&sink->mutex);
  for (iree_host_size_t i = 0; i < sink->device_count; ++i) {
    const iree_hal_profile_tracy_device_t* device = &sink->devices[i];
    if (device->session_id != session_id ||
        device->producer_key != producer_key ||
        device->physical_device_ordinal != physical_device_ordinal) {
      continue;
    }
    mapped = iree_hal_profile_tracy_device_map_tick(device, device_tick,
                                                    out_host_time_ns);
    break;
  }
  iree_slim_mutex_unlock(&sink->mutex);
  return mapped;
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

IREE_API_EXPORT bool iree_hal_profile_tracy_sink_map_device_tick_for_testing(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t producer_name, uint32_t physical_device_ordinal,
    uint64_t device_tick, int64_t* out_host_time_ns) {
  (void)sink;
  (void)session_id;
  (void)producer_name;
  (void)physical_device_ordinal;
  (void)device_tick;
  if (out_host_time_ns) *out_host_time_ns = 0;
  return false;
}

IREE_API_EXPORT bool iree_hal_profile_tracy_sink_map_device_span_for_testing(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t producer_name, uint32_t physical_device_ordinal,
    uint64_t start_tick, uint64_t end_tick, int64_t* out_start_host_time_ns,
    int64_t* out_end_host_time_ns) {
  (void)sink;
  (void)session_id;
  (void)producer_name;
  (void)physical_device_ordinal;
  (void)start_tick;
  (void)end_tick;
  if (out_start_host_time_ns) *out_start_host_time_ns = 0;
  if (out_end_host_time_ns) *out_end_host_time_ns = 0;
  return false;
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
