// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Read-only view over a Tracy capture for the iree-tracy-profile tool.
//
// Wraps tracy::Worker with the handful of traversals every command needs:
// flattened CPU zones (with thread, depth, and self time), flattened GPU zones
// (with the sink's dev/queue/lane naming decoded), and string helpers. All
// times are nanoseconds in Tracy's capture timeline; TimeSinceStart() rebases
// onto the first recorded event.

#ifndef IREE_TRACY_PROFILE_TRACE_H_
#define IREE_TRACY_PROFILE_TRACE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "TracyFileRead.hpp"
#include "TracyWorker.hpp"

namespace iree_tracy_profile {

// Which timeline lane a GPU context belongs to when it was produced by the
// IREE HAL profile sink, decoded from the context name "dev<N>/q<M> <lane>".
enum class Lane {
  kUnknown,
  kDispatch,  // Device-timestamped dispatch execution.
  kQueue,     // Device-timestamped queue operations.
  kHost,      // Host-timestamped execution spans.
  kSubmit,    // Host-timestamped submission spans.
  kStream,    // A tracy stream tracing context (HIP/CUDA/Vulkan drivers).
};

const char* LaneName(Lane lane);

struct GpuContextInfo {
  uint8_t id = 0;
  std::string name;
  // Producer that announced the context, when it identified itself. Two drivers
  // in one process can both call their first device "0".
  std::string producer;
  Lane lane = Lane::kUnknown;
  std::optional<uint32_t> physical_device_ordinal;
  std::optional<uint32_t> queue_ordinal;
  const tracy::GpuCtxData* data = nullptr;
};

struct CpuZoneRef {
  const tracy::ZoneEvent* zone = nullptr;
  uint64_t thread_id = 0;
  uint32_t depth = 0;
  // Ordinal in traversal order; stable within one load of a file.
  uint64_t event_id = 0;
};

struct GpuZoneRef {
  const tracy::GpuEvent* zone = nullptr;
  uint8_t context = 0;
  uint64_t thread_id = 0;
  uint32_t depth = 0;
  uint64_t event_id = 0;
};

class Trace {
 public:
  // Loads |path|, blocking until Tracy's background statistics are ready.
  // Returns nullptr and fills |out_error| on failure.
  static std::unique_ptr<Trace> Load(const std::string& path,
                                     std::string* out_error);

  const tracy::Worker& worker() const { return *worker_; }
  tracy::Worker& worker() { return *worker_; }

  int64_t first_time() const { return first_time_; }
  int64_t last_time() const { return worker_->GetLastTime(); }
  int64_t TimeSinceStart(int64_t t) const { return t - first_time_; }

  // All CPU zones in per-thread pre-order (parents before children).
  const std::vector<CpuZoneRef>& cpu_zones() const { return cpu_zones_; }
  // All GPU zones grouped by context in pre-order.
  const std::vector<GpuZoneRef>& gpu_zones() const { return gpu_zones_; }
  const std::vector<GpuContextInfo>& gpu_contexts() const {
    return gpu_contexts_;
  }
  const GpuContextInfo* FindGpuContext(uint8_t id) const;

  // Naming helpers.
  const char* ZoneName(const tracy::ZoneEvent& zone) const;
  const char* ZoneName(const tracy::GpuEvent& zone) const;
  const tracy::SourceLocation& SourceLocation(int16_t srcloc) const;
  std::string ThreadName(uint64_t thread_id) const;
  const char* String(const tracy::StringRef& ref) const;
  const char* String(const tracy::StringIdx& idx) const;
  const char* String(uint64_t ptr) const;
  std::string ZoneText(const tracy::ZoneEvent& zone) const;

  // Timing helpers.
  int64_t ZoneEnd(const tracy::ZoneEvent& zone) const;
  int64_t ZoneEnd(const tracy::GpuEvent& zone) const;
  // Time spent in direct children (for self-time computations).
  int64_t ChildTime(const tracy::ZoneEvent& zone) const;

  const char* FailureString() const;

 private:
  Trace() = default;
  void Index();
  void IndexCpuZones(
      const tracy::Vector<tracy::short_ptr<tracy::ZoneEvent>>& timeline,
      uint64_t thread_id, uint32_t depth);
  void IndexGpuZones(
      const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& timeline,
      uint8_t context, uint64_t thread_id, uint32_t depth);

  std::unique_ptr<tracy::FileRead> file_;
  std::unique_ptr<tracy::Worker> worker_;
  int64_t first_time_ = 0;
  std::vector<CpuZoneRef> cpu_zones_;
  std::vector<GpuZoneRef> gpu_zones_;
  std::vector<GpuContextInfo> gpu_contexts_;
};

// Matches |value| against |pattern| with '*' (any run) and '?' (any char),
// the same semantics as iree_string_view_match_pattern. Empty pattern matches
// everything.
bool MatchPattern(std::string_view value, std::string_view pattern);

// Merges [start, end) intervals and returns the covered duration.
int64_t UnionDuration(std::vector<std::pair<int64_t, int64_t>>& intervals);

// Simple running statistics over durations.
struct DurationStats {
  uint64_t count = 0;
  int64_t total = 0;
  int64_t min = INT64_MAX;
  int64_t max = INT64_MIN;
  double sum_sq = 0.0;
  std::vector<int64_t> samples;  // Kept for percentiles.

  void Add(int64_t value) {
    ++count;
    total += value;
    if (value < min) min = value;
    if (value > max) max = value;
    sum_sq += static_cast<double>(value) * static_cast<double>(value);
    samples.push_back(value);
  }
  double Mean() const { return count ? static_cast<double>(total) / count : 0; }
  double StdDev() const;
  // |p| in [0, 1]; sorts |samples| on first use.
  int64_t Percentile(double p);
};

}  // namespace iree_tracy_profile

#endif  // IREE_TRACY_PROFILE_TRACE_H_
