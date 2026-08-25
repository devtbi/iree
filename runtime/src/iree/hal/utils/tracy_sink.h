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
  // Events dropped because the tracing provider's GPU context limit was
  // reached.
  uint64_t context_exhausted_event_count;
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_TRACY_SINK_H_
