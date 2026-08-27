// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "trace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace iree_tracy_profile {

const char* LaneName(Lane lane) {
  switch (lane) {
    case Lane::kDispatch:
      return "dispatch";
    case Lane::kQueue:
      return "queue";
    case Lane::kHost:
      return "host";
    case Lane::kSubmit:
      return "submit";
    case Lane::kStream:
      return "stream";
    default:
      return "unknown";
  }
}

// Decodes the IREE HAL profile sink context naming convention:
//   "dev<N>/q<M> <lane>" or "dev<N> <lane>", either optionally followed by
//   " [<producer>]".
static void DecodeContextName(GpuContextInfo* info) {
  std::string name = info->name;
  // Split off the producer suffix before parsing the lane.
  if (!name.empty() && name.back() == ']') {
    const size_t open = name.rfind(" [");
    if (open != std::string::npos) {
      info->producer = name.substr(open + 2, name.size() - open - 3);
      name.resize(open);
    }
  }
  if (name.rfind("dev", 0) != 0) {
    // Stream tracing contexts are named after the device identifier.
    info->lane = Lane::kStream;
    return;
  }
  size_t pos = 3;
  auto parse_uint = [&](uint32_t* out) -> bool {
    size_t start = pos;
    uint64_t value = 0;
    while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
      value = value * 10 + (name[pos] - '0');
      ++pos;
    }
    if (pos == start || value > UINT32_MAX) return false;
    *out = static_cast<uint32_t>(value);
    return true;
  };
  uint32_t device = 0;
  if (!parse_uint(&device)) return;
  info->physical_device_ordinal = device;
  if (pos < name.size() && name[pos] == '/' && pos + 1 < name.size() &&
      name[pos + 1] == 'q') {
    pos += 2;
    uint32_t queue = 0;
    if (!parse_uint(&queue)) return;
    info->queue_ordinal = queue;
  }
  if (pos < name.size() && name[pos] == ' ') ++pos;
  std::string lane = name.substr(pos);
  if (lane == "dispatch") {
    info->lane = Lane::kDispatch;
  } else if (lane == "queue") {
    info->lane = Lane::kQueue;
  } else if (lane == "host") {
    info->lane = Lane::kHost;
  } else if (lane == "submit") {
    info->lane = Lane::kSubmit;
  }
}

std::unique_ptr<Trace> Trace::Load(const std::string& path,
                                   std::string* out_error) {
  std::unique_ptr<Trace> trace(new Trace());
  trace->file_.reset(tracy::FileRead::Open(path.c_str()));
  if (!trace->file_) {
    *out_error = "could not open trace file '" + path + "'";
    return nullptr;
  }
  try {
    trace->worker_.reset(new tracy::Worker(*trace->file_));
  } catch (const tracy::UnsupportedVersion& e) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer),
             "trace file was written by a newer tracy (file version %d.%d.%d)",
             e.version >> 16, (e.version >> 8) & 0xFF, e.version & 0xFF);
    *out_error = buffer;
    return nullptr;
  } catch (const tracy::LegacyVersion& e) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer),
             "trace file version %d.%d.%d is too old to read", e.version >> 16,
             (e.version >> 8) & 0xFF, e.version & 0xFF);
    *out_error = buffer;
    return nullptr;
  } catch (const std::exception& e) {
    *out_error = std::string("failed to load trace: ") + e.what();
    return nullptr;
  }
  // All aggregates are computed by the tool itself from the zone trees, which
  // are complete once the file has loaded; tracy's optional background
  // statistics are not needed (the server library may be built without them).
  trace->Index();
  return trace;
}

void Trace::Index() {
  first_time_ = worker_->GetFirstTime();

  for (const tracy::ThreadData* td : worker_->GetThreadData()) {
    IndexCpuZones(td->timeline, td->id, 0);
  }

  const auto& gpu_data = worker_->GetGpuData();
  for (size_t i = 0; i < gpu_data.size(); ++i) {
    const tracy::GpuCtxData* ctx = gpu_data[i];
    // tracy does not expose the wire context id; contexts are listed in
    // creation order so the index is a stable per-file identifier.
    const uint8_t id = static_cast<uint8_t>(i);
    GpuContextInfo info;
    info.id = id;
    info.name = ctx->name.Active() ? worker_->GetString(ctx->name) : "";
    info.data = ctx;
    DecodeContextName(&info);
    gpu_contexts_.push_back(std::move(info));
    for (const auto& td : ctx->threadData) {
      IndexGpuZones(td.second.timeline, id, td.first, 0);
    }
  }
}

void Trace::IndexCpuZones(
    const tracy::Vector<tracy::short_ptr<tracy::ZoneEvent>>& timeline,
    uint64_t thread_id, uint32_t depth) {
  auto visit = [&](const tracy::ZoneEvent& zone) {
    CpuZoneRef ref;
    ref.zone = &zone;
    ref.thread_id = thread_id;
    ref.depth = depth;
    ref.event_id = cpu_zones_.size();
    cpu_zones_.push_back(ref);
    if (zone.HasChildren()) {
      IndexCpuZones(worker_->GetZoneChildren(zone.Child()), thread_id,
                    depth + 1);
    }
  };
  if (timeline.is_magic()) {
    auto& vec =
        *reinterpret_cast<const tracy::Vector<tracy::ZoneEvent>*>(&timeline);
    for (const auto& zone : vec) visit(zone);
  } else {
    for (const auto& ptr : timeline) visit(*ptr);
  }
}

void Trace::IndexGpuZones(
    const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& timeline,
    uint8_t context, uint64_t thread_id, uint32_t depth) {
  auto visit = [&](const tracy::GpuEvent& zone) {
    GpuZoneRef ref;
    ref.zone = &zone;
    ref.context = context;
    ref.thread_id = thread_id;
    ref.depth = depth;
    ref.event_id = gpu_zones_.size();
    gpu_zones_.push_back(ref);
    if (zone.Child() >= 0) {
      IndexGpuZones(worker_->GetGpuChildren(zone.Child()), context, thread_id,
                    depth + 1);
    }
  };
  if (timeline.is_magic()) {
    auto& vec =
        *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&timeline);
    for (const auto& zone : vec) visit(zone);
  } else {
    for (const auto& ptr : timeline) visit(*ptr);
  }
}

const GpuContextInfo* Trace::FindGpuContext(uint8_t id) const {
  for (const auto& info : gpu_contexts_) {
    if (info.id == id) return &info;
  }
  return nullptr;
}

const char* Trace::ZoneName(const tracy::ZoneEvent& zone) const {
  return worker_->GetZoneName(zone);
}

const char* Trace::ZoneName(const tracy::GpuEvent& zone) const {
  return worker_->GetZoneName(zone);
}

const tracy::SourceLocation& Trace::SourceLocation(int16_t srcloc) const {
  return worker_->GetSourceLocation(srcloc);
}

std::string Trace::ThreadName(uint64_t thread_id) const {
  const char* name = worker_->GetThreadName(thread_id);
  if (name && *name) return name;
  return std::to_string(thread_id);
}

const char* Trace::String(const tracy::StringRef& ref) const {
  return worker_->GetString(ref);
}

const char* Trace::String(const tracy::StringIdx& idx) const {
  return worker_->GetString(idx);
}

const char* Trace::String(uint64_t ptr) const {
  return worker_->GetString(ptr);
}

std::string Trace::ZoneText(const tracy::ZoneEvent& zone) const {
  if (!worker_->HasZoneExtra(zone)) return "";
  const auto& extra = worker_->GetZoneExtra(zone);
  if (!extra.text.Active()) return "";
  return worker_->GetString(extra.text);
}

int64_t Trace::ZoneEnd(const tracy::ZoneEvent& zone) const {
  return worker_->GetZoneEnd(zone);
}

int64_t Trace::ZoneEnd(const tracy::GpuEvent& zone) const {
  return worker_->GetZoneEnd(zone);
}

int64_t Trace::ChildTime(const tracy::ZoneEvent& zone) const {
  if (!zone.HasChildren()) return 0;
  int64_t time = 0;
  const auto& children = worker_->GetZoneChildren(zone.Child());
  if (children.is_magic()) {
    auto& vec =
        *reinterpret_cast<const tracy::Vector<tracy::ZoneEvent>*>(&children);
    for (const auto& child : vec) time += ZoneEnd(child) - child.Start();
  } else {
    for (const auto& child : children) time += ZoneEnd(*child) - child->Start();
  }
  return time;
}

const char* Trace::FailureString() const {
  if (worker_->GetFailureType() == tracy::Worker::Failure::None) return nullptr;
  return tracy::Worker::GetFailureString(worker_->GetFailureType());
}

bool MatchPattern(std::string_view value, std::string_view pattern) {
  if (pattern.empty()) return true;
  size_t v = 0, p = 0;
  size_t star_p = std::string_view::npos, star_v = 0;
  while (v < value.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == value[v])) {
      ++v;
      ++p;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star_p = p++;
      star_v = v;
    } else if (star_p != std::string_view::npos) {
      p = star_p + 1;
      v = ++star_v;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') ++p;
  return p == pattern.size();
}

int64_t UnionDuration(std::vector<std::pair<int64_t, int64_t>>& intervals) {
  if (intervals.empty()) return 0;
  std::sort(intervals.begin(), intervals.end());
  int64_t covered = 0;
  int64_t cur_start = intervals[0].first;
  int64_t cur_end = intervals[0].second;
  for (size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].first <= cur_end) {
      cur_end = std::max(cur_end, intervals[i].second);
    } else {
      covered += cur_end - cur_start;
      cur_start = intervals[i].first;
      cur_end = intervals[i].second;
    }
  }
  covered += cur_end - cur_start;
  return covered;
}

double DurationStats::StdDev() const {
  if (count < 2) return 0.0;
  const double mean = Mean();
  const double variance = sum_sq / count - mean * mean;
  return variance > 0 ? std::sqrt(variance) : 0.0;
}

int64_t DurationStats::Percentile(double p) {
  if (samples.empty()) return 0;
  if (!std::is_sorted(samples.begin(), samples.end())) {
    std::sort(samples.begin(), samples.end());
  }
  const double idx = p * (static_cast<double>(samples.size()) - 1.0);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, samples.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return static_cast<int64_t>(
      std::llround(samples[lo] + (samples[hi] - samples[lo]) * frac));
}

}  // namespace iree_tracy_profile
