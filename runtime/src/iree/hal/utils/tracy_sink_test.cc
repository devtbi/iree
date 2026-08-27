// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/tracy_sink.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "iree/base/tracing.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// The sink only exists when the runtime is built with device instrumentation;
// otherwise creation reports UNAVAILABLE and there is nothing to test.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
constexpr bool kSinkAvailable = true;
#else
constexpr bool kSinkAvailable = false;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

class TracySinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status =
        iree_hal_profile_tracy_sink_create(iree_allocator_system(), &sink_);
    if (!kSinkAvailable) {
      EXPECT_TRUE(iree_status_is_unavailable(status));
      iree_status_ignore(status);
      GTEST_SKIP() << "runtime built without device instrumentation";
    }
    IREE_ASSERT_OK(status);
    iree_hal_profile_chunk_metadata_t session_metadata =
        iree_hal_profile_chunk_metadata_default();
    session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
    IREE_ASSERT_OK(
        iree_hal_profile_sink_begin_session(sink_, &session_metadata));
  }

  void TearDown() override {
    if (sink_) iree_hal_profile_sink_release(sink_);
  }

  void EndSession() {
    iree_hal_profile_chunk_metadata_t session_metadata =
        iree_hal_profile_chunk_metadata_default();
    session_metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
    IREE_ASSERT_OK(iree_hal_profile_sink_end_session(sink_, &session_metadata,
                                                     IREE_STATUS_OK));
  }

  void WriteChunk(iree_string_view_t content_type,
                  uint32_t physical_device_ordinal, uint32_t queue_ordinal,
                  const void* data, iree_host_size_t data_length) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = content_type;
    metadata.physical_device_ordinal = physical_device_ordinal;
    metadata.queue_ordinal = queue_ordinal;
    iree_const_byte_span_t iovec = iree_make_const_byte_span(data, data_length);
    IREE_ASSERT_OK(iree_hal_profile_sink_write(sink_, &metadata, 1, &iovec));
  }

  void WriteClockSample(uint32_t physical_device_ordinal, uint64_t sample_id,
                        uint64_t device_tick, int64_t host_time_ns) {
    iree_hal_profile_clock_correlation_record_t sample =
        iree_hal_profile_clock_correlation_record_default();
    sample.flags = IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
                   IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
    sample.physical_device_ordinal = physical_device_ordinal;
    sample.sample_id = sample_id;
    sample.device_tick = device_tick;
    sample.host_time_begin_ns = host_time_ns - 50;
    sample.host_time_end_ns = host_time_ns + 50;
    WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
               physical_device_ordinal, /*queue_ordinal=*/0, &sample,
               sizeof(sample));
  }

  void WriteFunction(uint64_t executable_id, uint32_t function_ordinal,
                     const char* name) {
    const iree_host_size_t name_length = strlen(name);
    std::vector<uint8_t> storage(
        sizeof(iree_hal_profile_executable_function_record_t) + name_length);
    iree_hal_profile_executable_function_record_t record =
        iree_hal_profile_executable_function_record_default();
    record.record_length = static_cast<uint32_t>(storage.size());
    record.executable_id = executable_id;
    record.function_ordinal = function_ordinal;
    record.name_length = static_cast<uint32_t>(name_length);
    memcpy(storage.data(), &record, sizeof(record));
    memcpy(storage.data() + sizeof(record), name, name_length);
    WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS,
               /*physical_device_ordinal=*/0, /*queue_ordinal=*/0,
               storage.data(), storage.size());
  }

  void WriteDispatchEvents(uint32_t physical_device_ordinal,
                           uint32_t queue_ordinal,
                           std::vector<std::pair<uint64_t, uint64_t>> spans) {
    std::vector<iree_hal_profile_dispatch_event_t> events;
    for (const auto& span : spans) {
      iree_hal_profile_dispatch_event_t event =
          iree_hal_profile_dispatch_event_default();
      event.executable_id = 7;
      event.function_ordinal = 3;
      event.start_tick = span.first;
      event.end_tick = span.second;
      events.push_back(event);
    }
    WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS,
               physical_device_ordinal, queue_ordinal, events.data(),
               events.size() * sizeof(events[0]));
  }

  // Identity of a producer stream. The sink keys its tables by both because
  // sessions restart device ticks and executable ids, and two drivers in one
  // process can each call their first device "0".
  struct Producer {
    uint64_t session_id = 0;
    const char* name = "";
    iree_string_view_t name_view() const {
      return iree_make_cstring_view(name);
    }
  };

  // Everything a clock-correlation record can carry, so tests can drive the
  // mapping's inputs rather than only observe its counters.
  struct ClockSample {
    uint32_t physical_device_ordinal = 0;
    uint64_t sample_id = 1;
    uint64_t device_tick = 0;
    int64_t host_time_ns = 0;
    int64_t bracket_half_width_ns = 50;
    bool has_device_tick = true;
    bool has_bracket = true;
    bool has_driver_timestamp = false;
    int64_t driver_timestamp_ns = 0;
    bool unaligned = false;
  };

  using MaybeTime = std::optional<int64_t>;

  static ClockSample MakeSample(uint64_t sample_id, uint64_t device_tick,
                                int64_t host_time_ns) {
    ClockSample sample;
    sample.sample_id = sample_id;
    sample.device_tick = device_tick;
    sample.host_time_ns = host_time_ns;
    return sample;
  }

  void BeginSessionFrom(const Producer& producer) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
    metadata.name = producer.name_view();
    metadata.session_id = producer.session_id;
    IREE_ASSERT_OK(iree_hal_profile_sink_begin_session(sink_, &metadata));
  }

  void EndSessionFrom(const Producer& producer) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
    metadata.name = producer.name_view();
    metadata.session_id = producer.session_id;
    IREE_ASSERT_OK(
        iree_hal_profile_sink_end_session(sink_, &metadata, IREE_STATUS_OK));
  }

  void WriteChunkFrom(const Producer& producer, iree_string_view_t content_type,
                      uint32_t physical_device_ordinal, uint32_t queue_ordinal,
                      const void* data, iree_host_size_t data_length) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = content_type;
    metadata.name = producer.name_view();
    metadata.session_id = producer.session_id;
    metadata.physical_device_ordinal = physical_device_ordinal;
    metadata.queue_ordinal = queue_ordinal;
    iree_const_byte_span_t iovec = iree_make_const_byte_span(data, data_length);
    IREE_ASSERT_OK(iree_hal_profile_sink_write(sink_, &metadata, 1, &iovec));
  }

  void WriteClockSampleFrom(const Producer& producer,
                            const ClockSample& source) {
    iree_hal_profile_clock_correlation_record_t sample =
        iree_hal_profile_clock_correlation_record_default();
    uint32_t flags = 0;
    if (source.has_device_tick) {
      flags |= IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK;
    }
    if (source.has_bracket) {
      flags |= IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
    }
    if (source.has_driver_timestamp) {
      flags |= IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP;
    }
    if (source.unaligned) {
      flags |= IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED;
    }
    sample.flags =
        static_cast<iree_hal_profile_clock_correlation_flags_t>(flags);
    sample.physical_device_ordinal = source.physical_device_ordinal;
    sample.sample_id = source.sample_id;
    sample.device_tick = source.device_tick;
    sample.host_time_begin_ns =
        source.host_time_ns - source.bracket_half_width_ns;
    sample.host_time_end_ns =
        source.host_time_ns + source.bracket_half_width_ns;
    sample.host_cpu_timestamp_ns =
        static_cast<uint64_t>(source.driver_timestamp_ns);
    WriteChunkFrom(producer, IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS,
                   source.physical_device_ordinal, /*queue_ordinal=*/0, &sample,
                   sizeof(sample));
  }

  void WriteDispatchEventsFrom(
      const Producer& producer, uint32_t physical_device_ordinal,
      uint32_t queue_ordinal,
      std::vector<std::pair<uint64_t, uint64_t>> spans) {
    std::vector<iree_hal_profile_dispatch_event_t> events;
    for (const auto& span : spans) {
      iree_hal_profile_dispatch_event_t event =
          iree_hal_profile_dispatch_event_default();
      event.executable_id = 7;
      event.function_ordinal = 3;
      event.start_tick = span.first;
      event.end_tick = span.second;
      events.push_back(event);
    }
    WriteChunkFrom(producer, IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS,
                   physical_device_ordinal, queue_ordinal, events.data(),
                   events.size() * sizeof(events[0]));
  }

  // Host-timestamped spans, appended in the order given so that tests can
  // reproduce the completion order producers actually emit.
  void WriteHostExecutionEventsFrom(
      const Producer& producer, uint32_t physical_device_ordinal,
      uint32_t queue_ordinal, std::vector<std::pair<int64_t, int64_t>> spans) {
    std::vector<iree_hal_profile_host_execution_event_t> events;
    for (const auto& span : spans) {
      iree_hal_profile_host_execution_event_t event =
          iree_hal_profile_host_execution_event_default();
      event.physical_device_ordinal = physical_device_ordinal;
      event.queue_ordinal = queue_ordinal;
      event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
      event.function_ordinal = UINT32_MAX;
      event.start_host_time_ns = span.first;
      event.end_host_time_ns = span.second;
      events.push_back(event);
    }
    WriteChunkFrom(producer,
                   IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
                   physical_device_ordinal, queue_ordinal, events.data(),
                   events.size() * sizeof(events[0]));
  }

  // Resolves a device-timestamped span exactly as the sink emits one.
  std::optional<std::pair<int64_t, int64_t>> MapSpan(
      const Producer& producer, uint32_t physical_device_ordinal,
      uint64_t start_tick, uint64_t end_tick) {
    int64_t start_host_time_ns = 0;
    int64_t end_host_time_ns = 0;
    if (!iree_hal_profile_tracy_sink_map_device_span_for_testing(
            sink_, producer.session_id, producer.name_view(),
            physical_device_ordinal, start_tick, end_tick, &start_host_time_ns,
            &end_host_time_ns)) {
      return std::nullopt;
    }
    return std::make_pair(start_host_time_ns, end_host_time_ns);
  }

  // Resolves a device tick through the mapping the sink currently holds.
  MaybeTime MapTick(const Producer& producer, uint32_t physical_device_ordinal,
                    uint64_t device_tick) {
    int64_t host_time_ns = 0;
    if (!iree_hal_profile_tracy_sink_map_device_tick_for_testing(
            sink_, producer.session_id, producer.name_view(),
            physical_device_ordinal, device_tick, &host_time_ns)) {
      return std::nullopt;
    }
    return host_time_ns;
  }

  iree_hal_profile_tracy_sink_statistics_t Statistics() {
    return iree_hal_profile_tracy_sink_statistics(sink_);
  }

  iree_hal_profile_sink_t* sink_ = nullptr;
};

TEST_F(TracySinkTest, BuffersDispatchEventsUntilClockMapped) {
  WriteFunction(/*executable_id=*/7, /*function_ordinal=*/3, "kernel_main");
  WriteClockSample(/*physical_device_ordinal=*/2, /*sample_id=*/1,
                   /*device_tick=*/1000, /*host_time_ns=*/1000000);

  // One sample is not enough to establish a tick frequency: buffer.
  WriteDispatchEvents(/*physical_device_ordinal=*/2, /*queue_ordinal=*/1,
                      {{1010, 1030}, {1035, 1045}});
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 0u);
  EXPECT_EQ(statistics.pending_event_count, 2u);
  EXPECT_EQ(statistics.context_count, 0u);

  // The second sample yields a mapping and drains the buffer.
  WriteClockSample(/*physical_device_ordinal=*/2, /*sample_id=*/2,
                   /*device_tick=*/2000, /*host_time_ns=*/2000000);
  statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 2u);
  EXPECT_EQ(statistics.pending_event_count, 0u);
  EXPECT_EQ(statistics.context_count, 1u);

  // Subsequent events stream directly.
  WriteDispatchEvents(/*physical_device_ordinal=*/2, /*queue_ordinal=*/1,
                      {{2100, 2200}});
  statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 3u);
  EXPECT_EQ(statistics.pending_event_count, 0u);
  EXPECT_EQ(statistics.invalid_event_count, 0u);

  EndSession();
  statistics = Statistics();
  EXPECT_EQ(statistics.unmapped_event_count, 0u);
}

TEST_F(TracySinkTest, DropsUnmappedEventsAtSessionEnd) {
  WriteClockSample(/*physical_device_ordinal=*/0, /*sample_id=*/1,
                   /*device_tick=*/1000, /*host_time_ns=*/1000000);
  WriteDispatchEvents(/*physical_device_ordinal=*/0, /*queue_ordinal=*/0,
                      {{1010, 1030}, {1035, 1045}, {1050, 1060}});
  EXPECT_EQ(Statistics().pending_event_count, 3u);
  EndSession();
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 0u);
  EXPECT_EQ(statistics.pending_event_count, 0u);
  EXPECT_EQ(statistics.unmapped_event_count, 3u);
}

TEST_F(TracySinkTest, RejectsInvalidSpans) {
  WriteClockSample(/*physical_device_ordinal=*/0, /*sample_id=*/1,
                   /*device_tick=*/1000, /*host_time_ns=*/1000000);
  WriteClockSample(/*physical_device_ordinal=*/0, /*sample_id=*/2,
                   /*device_tick=*/2000, /*host_time_ns=*/2000000);
  // Zero start and end-before-start are both invalid; the third is fine.
  WriteDispatchEvents(/*physical_device_ordinal=*/0, /*queue_ordinal=*/0,
                      {{0, 1030}, {1500, 1400}, {1600, 1700}});
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 1u);
  EXPECT_EQ(statistics.invalid_event_count, 2u);
}

TEST_F(TracySinkTest, EmitsHostTimestampedEventsImmediately) {
  iree_hal_profile_host_execution_event_t host_event =
      iree_hal_profile_host_execution_event_default();
  host_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  host_event.executable_id = 5;
  host_event.function_ordinal = 2;
  host_event.physical_device_ordinal = 0;
  host_event.queue_ordinal = 1;
  host_event.start_host_time_ns = 100000;
  host_event.end_host_time_ns = 160000;
  WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/1, &host_event,
             sizeof(host_event));

  iree_hal_profile_queue_device_event_t queue_device_event =
      iree_hal_profile_queue_device_event_default();
  queue_device_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
  queue_device_event.physical_device_ordinal = 0;
  queue_device_event.queue_ordinal = 1;
  queue_device_event.start_tick = 1000;
  queue_device_event.end_tick = 1100;
  WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/1,
             &queue_device_event, sizeof(queue_device_event));

  iree_hal_profile_queue_event_t queue_event =
      iree_hal_profile_queue_event_default();
  queue_event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
  queue_event.physical_device_ordinal = 0;
  queue_event.queue_ordinal = 1;
  queue_event.host_time_ns = 100000;
  queue_event.ready_host_time_ns = 120000;
  WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/1, &queue_event,
             sizeof(queue_event));

  auto statistics = Statistics();
  // Host execution and submission spans are emitted directly; the device queue
  // event waits for a clock mapping.
  EXPECT_EQ(statistics.emitted_zone_count, 2u);
  EXPECT_EQ(statistics.pending_event_count, 1u);
  EXPECT_EQ(statistics.context_count, 2u);

  WriteClockSample(/*physical_device_ordinal=*/0, /*sample_id=*/1,
                   /*device_tick=*/1000, /*host_time_ns=*/1000000);
  WriteClockSample(/*physical_device_ordinal=*/0, /*sample_id=*/2,
                   /*device_tick=*/2000, /*host_time_ns=*/2000000);
  statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 3u);
  EXPECT_EQ(statistics.pending_event_count, 0u);
  EXPECT_EQ(statistics.context_count, 3u);
}

TEST_F(TracySinkTest, CountsDroppedRecordsAndIgnoresUnknownChunks) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  metadata.flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  metadata.dropped_record_count = 7;
  IREE_ASSERT_OK(iree_hal_profile_sink_write(sink_, &metadata,
                                             /*iovec_count=*/0,
                                             /*iovecs=*/nullptr));

  iree_hal_profile_memory_event_t memory_event;
  memset(&memory_event, 0, sizeof(memory_event));
  memory_event.record_length = sizeof(memory_event);
  WriteChunk(IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS,
             /*physical_device_ordinal=*/0, /*queue_ordinal=*/0, &memory_event,
             sizeof(memory_event));

  auto statistics = Statistics();
  EXPECT_EQ(statistics.dropped_record_count, 7u);
  EXPECT_EQ(statistics.emitted_zone_count, 0u);
}

TEST_F(TracySinkTest, RejectsPartialRecords) {
  iree_hal_profile_dispatch_event_t event =
      iree_hal_profile_dispatch_event_default();
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS;
  metadata.physical_device_ordinal = 0;
  metadata.queue_ordinal = 0;
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(&event, sizeof(event) - 4);
  iree_status_t status =
      iree_hal_profile_sink_write(sink_, &metadata, 1, &iovec);
  EXPECT_TRUE(iree_status_is_data_loss(status));
  iree_status_ignore(status);
}

// The mapping keeps every clock sample rather than a first/last pair, and
// places ticks by interpolating between the neighboring ones. Anchoring on a
// single sample instead would step the timeline at every refit, underneath
// zones that have already been handed over and cannot be moved.
TEST_F(TracySinkTest, InterpolatesBetweenNeighboringClockSamples) {
  const Producer producer;
  WriteClockSampleFrom(producer, MakeSample(1, 1000, 1000000));
  // Deliberately 10us off the line through the outer samples.
  WriteClockSampleFrom(producer, MakeSample(2, 2000, 2010000));
  WriteClockSampleFrom(producer, MakeSample(3, 3000, 3000000));

  // Every sample is honored exactly...
  EXPECT_EQ(MapTick(producer, 0, 1000), MaybeTime(1000000));
  EXPECT_EQ(MapTick(producer, 0, 2000), MaybeTime(2010000));
  EXPECT_EQ(MapTick(producer, 0, 3000), MaybeTime(3000000));
  // ...and ticks between them are placed proportionally, so the timeline has no
  // step at a sample boundary.
  EXPECT_EQ(MapTick(producer, 0, 1500), MaybeTime(1505000));
  EXPECT_EQ(MapTick(producer, 0, 2500), MaybeTime(2505000));
  // Outside the sampled range the fitted rate extrapolates: 1000ns per tick.
  EXPECT_EQ(MapTick(producer, 0, 500), MaybeTime(500000));
}

// Durations must come from the tick rate alone. Placing both ends of a span
// through the anchors would fold the gap between two neighboring samples into
// the measurement of any span that straddles one - which is exactly the long
// dispatches people profile - and would make this sink disagree with the file
// sink about the same run.
TEST_F(TracySinkTest, KeepsDurationsIndependentOfWhereTheAnchorsFell) {
  const Producer producer;
  WriteClockSampleFrom(producer, MakeSample(1, 1000, 1000000));
  WriteClockSampleFrom(producer, MakeSample(2, 2000, 2010000));
  WriteClockSampleFrom(producer, MakeSample(3, 3000, 3000000));

  // 700 ticks at the fitted 1000ns per tick, straddling the middle sample.
  const auto span = MapSpan(producer, 0, 1500, 2200);
  ASSERT_TRUE(span.has_value());
  EXPECT_EQ(span->first, 1505000);
  EXPECT_EQ(span->second - span->first, 700000);
}

// A counter that goes backwards means the device was reset partway through, so
// no single rate describes the session. Holding the samples in tick order would
// otherwise fold the two halves into one well-formed but meaningless series.
TEST_F(TracySinkTest, RejectsADeviceWhoseClockWentBackwards) {
  const Producer producer;
  WriteClockSampleFrom(producer, MakeSample(1, 3000, 3000000));
  WriteClockSampleFrom(producer, MakeSample(2, 4000, 4000000));
  WriteDispatchEventsFrom(producer, 0, 0, {{3100, 3200}});
  EXPECT_EQ(Statistics().emitted_zone_count, 1u);

  WriteClockSampleFrom(producer, MakeSample(3, 500, 5000000));
  auto statistics = Statistics();
  EXPECT_EQ(statistics.clock_regression_count, 1u);
  // Zones were already in the capture when the reset showed up; a streaming
  // sink cannot take them back, so it says so.
  EXPECT_EQ(statistics.late_clock_invalidation_count, 1u);
  EXPECT_FALSE(MapTick(producer, 0, 3500).has_value());
}

// A driver timestamp taken inside the window IREE observed around the same call
// is the same clock read simultaneously with the tick, which is strictly better
// than the midpoint of that window.
TEST_F(TracySinkTest, PrefersTheDriverTimestampWhenItAgreesWithTheBracket) {
  const Producer producer;
  ClockSample first = MakeSample(1, 1000, 1000000);
  first.bracket_half_width_ns = 2000;
  first.has_driver_timestamp = true;
  first.driver_timestamp_ns = 1000300;
  WriteClockSampleFrom(producer, first);
  ClockSample second = MakeSample(2, 2000, 2000000);
  second.bracket_half_width_ns = 2000;
  second.has_driver_timestamp = true;
  second.driver_timestamp_ns = 2000300;
  WriteClockSampleFrom(producer, second);

  EXPECT_EQ(MapTick(producer, 0, 1000), MaybeTime(1000300));
  EXPECT_EQ(MapTick(producer, 0, 2000), MaybeTime(2000300));
}

// When the driver samples a different clock than IREE (KFD reports
// CLOCK_MONOTONIC_RAW while host records use CLOCK_MONOTONIC) its timestamp
// cannot place events on the host timeline, so the bracket has to.
TEST_F(TracySinkTest, AnchorsOnTheHostBracketWhenTheDriverClockIsAnotherDomain) {
  const Producer producer;
  ClockSample first = MakeSample(1, 1000, 1000000);
  first.has_driver_timestamp = true;
  first.driver_timestamp_ns = 9000000;
  WriteClockSampleFrom(producer, first);
  ClockSample second = MakeSample(2, 2000, 2000000);
  second.has_driver_timestamp = true;
  second.driver_timestamp_ns = 10000000;
  WriteClockSampleFrom(producer, second);

  // Placed on the host timeline, not eight milliseconds into the driver's.
  EXPECT_EQ(MapTick(producer, 0, 1000), MaybeTime(1000000));
  EXPECT_EQ(MapTick(producer, 0, 1500), MaybeTime(1500000));
}

// The driver's own deltas give an exact tick rate but in its own domain, so
// they are only worth the domain error while the host brackets are noisier than
// that error. Over a long baseline the brackets win and the rate switches.
TEST_F(TracySinkTest, PrefersHostBracketsForTheRateOnceTheBaselineIsLong) {
  const Producer producer;
  ClockSample first = MakeSample(1, 1000, 1000000);
  first.has_driver_timestamp = true;
  first.driver_timestamp_ns = 9000000;
  WriteClockSampleFrom(producer, first);
  // One second of baseline; the driver clock claims 9ns per tick over it while
  // the host brackets say 1ns per tick.
  ClockSample second = MakeSample(2, 1000000, 1000000000);
  second.has_driver_timestamp = true;
  second.driver_timestamp_ns = 9000000000;
  WriteClockSampleFrom(producer, second);

  EXPECT_EQ(MapTick(producer, 0, 500000), MaybeTime(500000000));
}

// Zones handed to the tracing provider cannot be moved afterwards, so a rate
// fitted between two samples taken microseconds apart must not be committed to.
TEST_F(TracySinkTest, BuffersUntilTheTickRateIsTrustworthy) {
  const Producer producer;
  ClockSample first = MakeSample(1, 1000, 1000000);
  first.bracket_half_width_ns = 30000;
  WriteClockSampleFrom(producer, first);
  ClockSample second = MakeSample(2, 2000, 2000000);
  second.bracket_half_width_ns = 30000;
  WriteClockSampleFrom(producer, second);

  // 60us of uncertainty over a 1ms baseline is 6% of rate error.
  EXPECT_FALSE(MapTick(producer, 0, 1500).has_value());
  WriteDispatchEventsFrom(producer, 0, 0, {{1100, 1200}, {1300, 1400}});
  EXPECT_EQ(Statistics().pending_event_count, 2u);
  EXPECT_EQ(Statistics().emitted_zone_count, 0u);

  // A sample a second later brings that to 60ppm and releases the backlog.
  ClockSample third = MakeSample(3, 1001000, 1001000000);
  third.bracket_half_width_ns = 30000;
  WriteClockSampleFrom(producer, third);
  EXPECT_EQ(Statistics().pending_event_count, 0u);
  EXPECT_EQ(Statistics().emitted_zone_count, 2u);
  EXPECT_EQ(Statistics().degraded_event_count, 0u);
}

// If that better sample never arrives, an approximate timeline still beats no
// timeline - but it is reported as approximate.
TEST_F(TracySinkTest, PlacesHeldEventsWithADegradedRateAtSessionEnd) {
  const Producer producer;
  ClockSample first = MakeSample(1, 1000, 1000000);
  first.bracket_half_width_ns = 30000;
  WriteClockSampleFrom(producer, first);
  ClockSample second = MakeSample(2, 2000, 2000000);
  second.bracket_half_width_ns = 30000;
  WriteClockSampleFrom(producer, second);
  WriteDispatchEventsFrom(producer, 0, 0, {{1100, 1200}, {1300, 1400}});
  EXPECT_EQ(Statistics().pending_event_count, 2u);

  EndSession();
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 2u);
  EXPECT_EQ(statistics.degraded_event_count, 2u);
  EXPECT_EQ(statistics.unmapped_event_count, 0u);
  EXPECT_EQ(statistics.pending_event_count, 0u);
}

// A producer that flags its clock samples as not covering the ticks it reports
// gets its device rejected, the same conclusion the file sink and iree-profile
// reach - but counted separately so it is not confused with a missing sample.
TEST_F(TracySinkTest, DropsEventsOnAClockTheProducerFlaggedUnaligned) {
  const Producer producer;
  WriteClockSampleFrom(producer, MakeSample(1, 1000, 1000000));
  WriteClockSampleFrom(producer, MakeSample(2, 2000, 2000000));
  WriteDispatchEventsFrom(producer, 0, 0, {{1100, 1200}});
  EXPECT_EQ(Statistics().emitted_zone_count, 1u);

  ClockSample unaligned = MakeSample(3, 3000, 3000000);
  unaligned.unaligned = true;
  WriteClockSampleFrom(producer, unaligned);
  EXPECT_FALSE(MapTick(producer, 0, 2500).has_value());

  WriteDispatchEventsFrom(producer, 0, 0, {{3100, 3200}, {3300, 3400}});
  auto statistics = Statistics();
  EXPECT_EQ(statistics.unaligned_event_count, 2u);
  EXPECT_EQ(statistics.pending_event_count, 0u);
  EXPECT_EQ(statistics.emitted_zone_count, 1u);

  EndSession();
  EXPECT_EQ(Statistics().unmapped_event_count, 0u);
}

// One unusable sample must not cost a mapping that already worked.
TEST_F(TracySinkTest, KeepsTheMappingWhenALaterSampleIsUnusable) {
  const Producer producer;
  WriteClockSampleFrom(producer, MakeSample(1, 1000, 1000000));
  WriteClockSampleFrom(producer, MakeSample(2, 2000, 2000000));
  EXPECT_EQ(MapTick(producer, 0, 1500), MaybeTime(1500000));

  // Neither a record without a device tick nor a repeat of a tick already
  // recorded (an idle device between two flushes) is a counter reset.
  ClockSample without_tick = MakeSample(3, 3000, 3000000);
  without_tick.has_device_tick = false;
  WriteClockSampleFrom(producer, without_tick);
  WriteClockSampleFrom(producer, MakeSample(4, 2000, 2222222));

  EXPECT_EQ(Statistics().rejected_clock_sample_count, 2u);
  EXPECT_EQ(Statistics().clock_regression_count, 0u);
  EXPECT_EQ(MapTick(producer, 0, 1500), MaybeTime(1500000));
  WriteDispatchEventsFrom(producer, 0, 0, {{1100, 1200}});
  EXPECT_EQ(Statistics().emitted_zone_count, 1u);
}

// Two drivers in one process both call their first device "0". Merging their
// tick spaces would fit one rate across two unrelated clocks.
TEST_F(TracySinkTest, KeepsProducersWithTheSameDeviceOrdinalApart) {
  Producer alpha;
  alpha.session_id = 11;
  alpha.name = "alpha";
  Producer beta;
  beta.session_id = 12;
  beta.name = "beta";
  BeginSessionFrom(alpha);
  BeginSessionFrom(beta);

  WriteClockSampleFrom(alpha, MakeSample(1, 1000, 1000000));
  WriteClockSampleFrom(alpha, MakeSample(2, 2000, 2000000));
  // Same ordinal, unrelated ticks, three times the rate.
  WriteClockSampleFrom(beta, MakeSample(1, 5000, 5000000));
  WriteClockSampleFrom(beta, MakeSample(2, 6000, 8000000));

  EXPECT_EQ(MapTick(alpha, 0, 1500), MaybeTime(1500000));
  EXPECT_EQ(MapTick(beta, 0, 5500), MaybeTime(6500000));

  WriteDispatchEventsFrom(alpha, 0, 0, {{1100, 1200}});
  WriteDispatchEventsFrom(beta, 0, 0, {{5100, 5200}});
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 2u);
  // Separate timelines rather than one interleaved row.
  EXPECT_EQ(statistics.context_count, 2u);

  EndSessionFrom(beta);
  EndSessionFrom(alpha);
}

// Devices share one sink, so each opens its own session on it. Ending one must
// not flush or discard what another is still waiting on.
TEST_F(TracySinkTest, KeepsConcurrentSessionsApart) {
  Producer alpha;
  alpha.session_id = 21;
  alpha.name = "alpha";
  Producer beta;
  beta.session_id = 22;
  beta.name = "beta";
  BeginSessionFrom(alpha);
  BeginSessionFrom(beta);

  // alpha has one sample so far, so its events wait for a rate.
  WriteClockSampleFrom(alpha, MakeSample(1, 1000, 1000000));
  WriteDispatchEventsFrom(alpha, 0, 0, {{1100, 1200}, {1300, 1400}});
  EXPECT_EQ(Statistics().pending_event_count, 2u);

  WriteClockSampleFrom(beta, MakeSample(1, 1000, 1000000));
  WriteClockSampleFrom(beta, MakeSample(2, 2000, 2000000));
  WriteDispatchEventsFrom(beta, 0, 0, {{1100, 1200}});
  EXPECT_EQ(Statistics().emitted_zone_count, 1u);

  EndSessionFrom(beta);
  auto statistics = Statistics();
  EXPECT_EQ(statistics.pending_event_count, 2u);
  EXPECT_EQ(statistics.unmapped_event_count, 0u);

  EndSessionFrom(alpha);
  statistics = Statistics();
  EXPECT_EQ(statistics.pending_event_count, 0u);
  EXPECT_EQ(statistics.unmapped_event_count, 2u);
}

// Producers append events when they complete, so a long operation lands after
// short ones that began later. Handed over in that order, the tracing provider
// reads a backwards step of more than 2^31ns as a hardware counter wrapping and
// displaces the rest of the timeline, so the sink orders each chunk first.
TEST_F(TracySinkTest, OrdersEventsThatArriveInCompletionOrder) {
  const Producer producer;
  WriteHostExecutionEventsFrom(producer, 0, 0,
                               {{3000000000, 3100000000},
                                {500000000, 3200000000}});
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 2u);
  EXPECT_EQ(statistics.reordered_event_count, 0u);
}

// The ordering only reaches as far as one chunk. An event that turns up a whole
// chunk later and more than a second behind is dropped rather than allowed to
// displace everything already on that timeline.
TEST_F(TracySinkTest, DropsEventsReorderedPastTheSafeWindow) {
  const Producer producer;
  WriteHostExecutionEventsFrom(producer, 0, 0, {{3000000000, 3100000000}});
  EXPECT_EQ(Statistics().emitted_zone_count, 1u);

  WriteHostExecutionEventsFrom(producer, 0, 0, {{500000000, 600000000}});
  auto statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 1u);
  EXPECT_EQ(statistics.reordered_event_count, 1u);

  // A late event inside the window is still kept.
  WriteHostExecutionEventsFrom(producer, 0, 0, {{2500000000, 2600000000}});
  statistics = Statistics();
  EXPECT_EQ(statistics.emitted_zone_count, 2u);
  EXPECT_EQ(statistics.reordered_event_count, 1u);
}

}  // namespace
