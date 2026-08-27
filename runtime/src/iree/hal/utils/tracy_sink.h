// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_TRACY_SINK_H_
#define IREE_HAL_UTILS_TRACY_SINK_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/profile_schema.h"
#include "iree/hal/profile_sink.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_profile_tracy_sink_t
//===----------------------------------------------------------------------===//

// Profile sink forwarding HAL-native profiling records to the runtime tracing
// provider as GPU zones.
//
// Device-timestamped dispatch and queue events become zones on per-queue
// tracing GPU contexts ("dev0/q1 dispatch", "dev0/q1 queue"). Host-timestamped
// execution spans and queue submissions get their own contexts ("dev0/q1 host",
// "dev0/q1 submit"). Device ticks are placed on the host timeline using the
// clock-correlation samples emitted by the producer: events that arrive before
// at least two samples exist for their physical device are buffered until a
// mapping is available or the session ends. Memory lifecycle events, counter
// samples, and executable traces have no tracing-provider equivalent and are
// ignored.
//
// The sink never touches the device. The producer already captured the
// timestamps for its own profile records, so tracing through this sink costs
// exactly what the HAL profiling session costs plus a handful of queue writes
// per zone at flush time - no per-dispatch event records, no host
// synchronization, and no additional device packets.
//
// Zones within one context are emitted in producer order. Tracing providers
// may assume timestamps within a context are roughly monotonic; producers
// emitting device events more than ~2s out of order will render incorrectly.
//
// Requires a runtime built with the tracy provider and device instrumentation
// (IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE); creation fails with
// IREE_STATUS_UNAVAILABLE otherwise.

// Counters describing what a tracy sink has forwarded.
typedef struct iree_hal_profile_tracy_sink_statistics_t {
  // Zones emitted to the tracing provider.
  uint64_t emitted_zone_count;
  // Device-timestamped events dropped because no clock mapping was available
  // for their physical device when the session ended.
  uint64_t unmapped_event_count;
  // Device-timestamped events dropped because no mapping describes their
  // device's session: either the producer reported that its clock samples do
  // not cover the ticks it emitted
  // (IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED) or the
  // device counter was reset. The file sink and iree-profile reject such a
  // device too, though they see every sample before deciding; see
  // late_clock_invalidation_count.
  uint64_t unaligned_event_count;
  // Device-timestamped events placed at session end using a tick rate whose
  // estimated error exceeded the confidence bound. Their durations and
  // placement are approximate; a producer flush interval long enough to give
  // two well-separated clock samples avoids this.
  uint64_t degraded_event_count;
  // Device-timestamped events dropped because the per-device buffer of events
  // awaiting a clock mapping was full.
  uint64_t pending_overflow_count;
  // Clock-correlation records that carried no usable tick/host pairing, could
  // not supply the host clock the device's other samples use, or duplicated a
  // tick already recorded.
  uint64_t rejected_clock_sample_count;
  // Clock samples whose device tick went backwards, meaning the counter was
  // reset partway through the session and no single rate describes it.
  uint64_t clock_regression_count;
  // Devices whose clock was invalidated after zones had already been handed to
  // the tracing provider. A streaming sink commits before it has seen every
  // clock sample, so unlike a consumer reading a finished profile it cannot
  // retract those zones.
  uint64_t late_clock_invalidation_count;
  // Events dropped because the tracing provider's GPU context limit was
  // reached.
  uint64_t context_exhausted_event_count;
  // Events dropped because they began more than a second before the newest
  // event already emitted on their timeline. The tracing provider reads a
  // larger backwards step as a hardware counter wrapping and would displace
  // every later event on that timeline.
  uint64_t reordered_event_count;
  // Events dropped because their span was invalid (zero start or end before
  // start).
  uint64_t invalid_event_count;
  // Source records reported as dropped by truncated producer chunks.
  uint64_t dropped_record_count;
  // Device-timestamped events currently buffered awaiting a clock mapping.
  uint64_t pending_event_count;
  // Tracing GPU contexts allocated so far.
  uint32_t context_count;
} iree_hal_profile_tracy_sink_statistics_t;

// Creates a profile sink forwarding records to the tracing provider.
IREE_API_EXPORT iree_status_t iree_hal_profile_tracy_sink_create(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink);

// Returns counters describing what |sink| has forwarded so far.
// |sink| must have been created with iree_hal_profile_tracy_sink_create.
IREE_API_EXPORT iree_hal_profile_tracy_sink_statistics_t
iree_hal_profile_tracy_sink_statistics(iree_hal_profile_sink_t* sink);

// Resolves |device_tick| through the clock mapping |sink| currently holds for
// the producer identified by (|session_id|, |producer_name|) and its physical
// device |physical_device_ordinal|. Returns false when that device has no
// usable mapping yet.
//
// Exposed for tests: zone timestamps are otherwise only observable by capturing
// the process, which makes the clock arithmetic - the part most likely to be
// silently wrong - the part hardest to check.
IREE_API_EXPORT bool iree_hal_profile_tracy_sink_map_device_tick_for_testing(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t producer_name, uint32_t physical_device_ordinal,
    uint64_t device_tick, int64_t* out_host_time_ns);

// Resolves a device-timestamped span the way the sink emits one: the start is
// placed through the clock samples and the end is derived from the fitted tick
// rate. Returns false when the device has no usable mapping yet.
//
// Exposed for tests, for the same reason as above.
IREE_API_EXPORT bool iree_hal_profile_tracy_sink_map_device_span_for_testing(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t producer_name, uint32_t physical_device_ordinal,
    uint64_t start_tick, uint64_t end_tick, int64_t* out_start_host_time_ns,
    int64_t* out_end_host_time_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_TRACY_SINK_H_
