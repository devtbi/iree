// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/transient_buffer.h"

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_device_size_t kBufferSize = 256;
constexpr uint8_t kPattern = 0xA5;

// Transient buffer committed to a heap backing filled with kPattern, as after
// its alloca signal has been published.
class TransientBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("transient_buffer_test"),
        iree_allocator_system(), iree_allocator_system(), &heap_allocator_));
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(heap_allocator_, params,
                                                      kBufferSize, &backing_));
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(backing_, 0, kBufferSize, &kPattern,
                                            sizeof(kPattern)));

    iree_hal_buffer_placement_t placement = {0};
    placement.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    placement.flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS;
    IREE_ASSERT_OK(iree_hal_local_transient_buffer_create(
        placement, params, kBufferSize, kBufferSize, iree_allocator_system(),
        &transient_));
    iree_hal_local_transient_buffer_stage_backing(transient_, backing_);
    iree_hal_local_transient_buffer_commit(transient_);
  }

  void TearDown() override {
    iree_hal_buffer_release(transient_);
    iree_hal_buffer_release(backing_);
    iree_hal_allocator_release(heap_allocator_);
  }

  iree_hal_allocator_t* heap_allocator_ = nullptr;
  iree_hal_buffer_t* backing_ = nullptr;
  iree_hal_buffer_t* transient_ = nullptr;
};

// queue_dealloca only enqueues the release: work ordered before it may still
// map the buffer until the dealloca executes and decommits the backing.
TEST_F(TransientBufferTest, MapAllowedUntilDecommit) {
  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  uint8_t value = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(transient_, 0, &value, sizeof(value)));
  EXPECT_EQ(value, kPattern);

  iree_hal_local_transient_buffer_decommit(transient_);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_map_read(transient_, 0, &value, sizeof(value)));
}

// A scoped mapping retains the backing it mapped, so decommitting while the
// mapping is live does not free memory under it.
TEST_F(TransientBufferTest, ScopedMappingOutlivesDecommit) {
  ASSERT_TRUE(iree_hal_local_transient_buffer_begin_dealloca(transient_));
  iree_hal_buffer_mapping_t mapping = {{0}};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      transient_, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_HAL_WHOLE_BUFFER, &mapping));

  // Drop every reference to the backing except the one held by the mapping.
  iree_hal_local_transient_buffer_decommit(transient_);
  iree_hal_buffer_release(backing_);
  backing_ = nullptr;

  EXPECT_EQ(mapping.contents.data[0], kPattern);
  IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
}

}  // namespace
