// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/tracy_sink.h"

#include <cstdint>
#include <cstring>
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

}  // namespace
