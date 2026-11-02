// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// State-machine tests for the local transient buffer wrapper.
//
// Host access is gated on the committed backing only, not on whether a
// queue_dealloca has been accepted. queue_dealloca merely enqueues the release:
// work ordered before it (including work on other devices that maps the
// wrapper when it executes) must still reach the committed backing. Memory
// safety comes from decommit (which happens when the dealloca executes) and
// from mappings retaining the backing they map, which these tests pin down.

#include "iree/hal/local/transient_buffer.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_device_size_t kBufferSize = 256;
constexpr uint8_t kPattern = 0xA5;
// Not IREE_HAL_MEMORY_ACCESS_ALL: that includes DISCARD, which debug heap
// buffers honor by scribbling over the mapped range.
constexpr iree_hal_memory_access_t kReadWrite =
    IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;

class TransientBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("transient_buffer_test"),
        iree_allocator_system(), iree_allocator_system(), &heap_allocator_));
  }

  void TearDown() override {
    iree_hal_buffer_release(transient_);
    iree_hal_buffer_release(backing_);
    iree_hal_allocator_release(heap_allocator_);
  }

  static iree_hal_buffer_params_t MakeParams() {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                   IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
    return params;
  }

  // Creates an uncommitted wrapper and a heap backing filled with kPattern.
  void CreateUncommitted() {
    iree_hal_buffer_placement_t placement = {0};
    placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    placement.flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS;
    IREE_ASSERT_OK(iree_hal_local_transient_buffer_create(
        placement, MakeParams(), kBufferSize, kBufferSize,
        iree_allocator_system(), &transient_));
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        heap_allocator_, MakeParams(), kBufferSize, &backing_));
    std::vector<uint8_t> fill(kBufferSize, kPattern);
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(backing_, 0, fill.data(), fill.size()));
  }

  // Creates a wrapper that has been staged and committed, as after the alloca
  // signal has been published.
  void CreateCommitted() {
    CreateUncommitted();
    iree_hal_local_transient_buffer_stage_backing(transient_, backing_);
    iree_hal_local_transient_buffer_commit(transient_);
  }

  // Reads the full wrapper contents through a scoped host mapping.
  iree_status_t ReadAll(iree_hal_buffer_t* buffer, std::vector<uint8_t>* out) {
    out->assign(iree_hal_buffer_byte_length(buffer), 0);
    return iree_hal_buffer_map_read(buffer, 0, out->data(), out->size());
  }

  iree_hal_allocator_t* heap_allocator_ = nullptr;
  iree_hal_buffer_t* transient_ = nullptr;
  iree_hal_buffer_t* backing_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Lifecycle gates that do not depend on dealloca state
//===----------------------------------------------------------------------===//

TEST_F(TransientBufferTest, MapBeforeStageFails) {
  CreateUncommitted();
  std::vector<uint8_t> data;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        ReadAll(transient_, &data));
}

// A staged backing is only visible to queue packet emission; host access still
// requires the commit that happens when the alloca signal is published.
TEST_F(TransientBufferTest, MapStagedButUncommittedFails) {
  CreateUncommitted();
  iree_hal_local_transient_buffer_stage_backing(transient_, backing_);
  EXPECT_FALSE(iree_hal_local_transient_buffer_is_committed(transient_));
  EXPECT_EQ(iree_hal_local_transient_buffer_backing_buffer(transient_),
            backing_);
  std::vector<uint8_t> data;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        ReadAll(transient_, &data));
}

TEST_F(TransientBufferTest, MapCommittedSucceeds) {
  CreateCommitted();
  std::vector<uint8_t> data;
  IREE_ASSERT_OK(ReadAll(transient_, &data));
  EXPECT_EQ(data, std::vector<uint8_t>(kBufferSize, kPattern));
}

//===----------------------------------------------------------------------===//
// Window between queue_dealloca acceptance and dealloca execution
//===----------------------------------------------------------------------===//

// queue_dealloca has been accepted but has not executed: the backing is still
// committed and must remain reachable for work ordered before the dealloca.
TEST_F(TransientBufferTest, ScopedMapAfterDeallocaQueuedSucceeds) {
  CreateCommitted();
  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  ASSERT_TRUE(iree_hal_local_transient_buffer_is_dealloca_queued(transient_));

  std::vector<uint8_t> data;
  IREE_ASSERT_OK(ReadAll(transient_, &data));
  EXPECT_EQ(data, std::vector<uint8_t>(kBufferSize, kPattern));

  const uint8_t update[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(transient_, 16, update, sizeof(update)));
  uint8_t readback[4] = {0};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(backing_, 16, readback, sizeof(readback)));
  EXPECT_EQ(0, memcmp(update, readback, sizeof(update)));
}

// Persistent mappings keep the wrapper as the mapped buffer, so cache
// maintenance re-enters the wrapper's flush/invalidate on every call. They do
// not own the backing: once decommitted, maintenance and unmap fail cleanly.
TEST_F(TransientBufferTest, PersistentMappingMaintenanceAfterDeallocaQueued) {
  CreateCommitted();
  iree_hal_buffer_mapping_t mapping = {{0}};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      transient_, IREE_HAL_MAPPING_MODE_PERSISTENT, kReadWrite, 0,
      IREE_HAL_WHOLE_BUFFER, &mapping));
  ASSERT_EQ(mapping.buffer, transient_);

  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  IREE_EXPECT_OK(iree_hal_buffer_mapping_invalidate_range(
      &mapping, 0, IREE_HAL_WHOLE_BUFFER));
  IREE_EXPECT_OK(
      iree_hal_buffer_mapping_flush_range(&mapping, 0, IREE_HAL_WHOLE_BUFFER));

  iree_hal_local_transient_buffer_decommit(transient_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_buffer_mapping_invalidate_range(
                            &mapping, 0, IREE_HAL_WHOLE_BUFFER));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_mapping_flush_range(&mapping, 0, IREE_HAL_WHOLE_BUFFER));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_buffer_unmap_range(&mapping));
}

// Bindings frequently reference a subspan of the transient allocation. A
// subspan forwards to the wrapper and must see the same window semantics.
TEST_F(TransientBufferTest, SubspanFollowsWrapperState) {
  CreateCommitted();
  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(transient_, 64, 64,
                                         iree_allocator_system(), &subspan));
  ASSERT_EQ(iree_hal_buffer_allocated_buffer(subspan), transient_);

  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  std::vector<uint8_t> data;
  IREE_EXPECT_OK(ReadAll(subspan, &data));
  EXPECT_EQ(data, std::vector<uint8_t>(64, kPattern));

  iree_hal_local_transient_buffer_decommit(transient_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        ReadAll(subspan, &data));
  iree_hal_buffer_release(subspan);
}

//===----------------------------------------------------------------------===//
// After the dealloca executes
//===----------------------------------------------------------------------===//

// Decommit is what makes stale host references fail cleanly; it does not
// depend on the dealloca_queued flag.
TEST_F(TransientBufferTest, MapAfterDecommitFails) {
  CreateCommitted();
  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  iree_hal_local_transient_buffer_decommit(transient_);
  EXPECT_FALSE(iree_hal_local_transient_buffer_is_committed(transient_));
  EXPECT_EQ(iree_hal_local_transient_buffer_backing_buffer(transient_),
            nullptr);
  std::vector<uint8_t> data;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        ReadAll(transient_, &data));
}

// Same as above with dealloca_queued never set, e.g. a dealloca path that only
// decommits.
TEST_F(TransientBufferTest, MapAfterDecommitWithoutDeallocaQueuedFails) {
  CreateCommitted();
  iree_hal_local_transient_buffer_decommit(transient_);
  std::vector<uint8_t> data;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        ReadAll(transient_, &data));
}

// A scoped mapping taken before decommit owns a reference to the backing it
// mapped, not to the wrapper. Decommitting while it is live must not
// invalidate the mapping or its unmap path (no use-after-free under ASan).
TEST_F(TransientBufferTest, ScopedMappingOutlivesDecommit) {
  CreateCommitted();
  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));

  iree_hal_buffer_mapping_t mapping = {{0}};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      transient_, IREE_HAL_MAPPING_MODE_SCOPED, kReadWrite, 0,
      IREE_HAL_WHOLE_BUFFER, &mapping));
  ASSERT_EQ(mapping.buffer, backing_);

  // Drop every reference except the one held by the mapping.
  iree_hal_local_transient_buffer_decommit(transient_);
  iree_hal_buffer_release(backing_);
  backing_ = nullptr;

  ASSERT_EQ(mapping.contents.data_length, kBufferSize);
  EXPECT_EQ(mapping.contents.data[0], kPattern);
  mapping.contents.data[0] = 0x5A;
  IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
}

//===----------------------------------------------------------------------===//
// The dealloca_queued flag's remaining job: one dealloca per buffer
//===----------------------------------------------------------------------===//

TEST_F(TransientBufferTest, BeginDeallocaIsExclusive) {
  CreateCommitted();
  EXPECT_FALSE(iree_hal_local_transient_buffer_is_dealloca_queued(transient_));
  EXPECT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  EXPECT_FALSE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  EXPECT_TRUE(iree_hal_local_transient_buffer_is_dealloca_queued(transient_));
}

TEST_F(TransientBufferTest, AbortDeallocaAllowsRetry) {
  CreateCommitted();
  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  iree_hal_local_transient_buffer_abort_dealloca(transient_);
  EXPECT_FALSE(iree_hal_local_transient_buffer_is_dealloca_queued(transient_));
  EXPECT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
}

}  // namespace
