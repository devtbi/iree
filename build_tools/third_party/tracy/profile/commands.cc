// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "commands.h"

#include <algorithm>
#include <cinttypes>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "json.h"

namespace iree_tracy_profile {
namespace {

//===----------------------------------------------------------------------===//
// Formatting helpers
//===----------------------------------------------------------------------===//

std::string FormatNs(int64_t ns) {
  char buffer[64];
  const double abs_ns = static_cast<double>(ns < 0 ? -ns : ns);
  if (abs_ns >= 1e9) {
    snprintf(buffer, sizeof(buffer), "%.3f s", ns / 1e9);
  } else if (abs_ns >= 1e6) {
    snprintf(buffer, sizeof(buffer), "%.3f ms", ns / 1e6);
  } else if (abs_ns >= 1e3) {
    snprintf(buffer, sizeof(buffer), "%.3f us", ns / 1e3);
  } else {
    snprintf(buffer, sizeof(buffer), "%" PRId64 " ns", ns);
  }
  return buffer;
}

std::string FormatBytes(uint64_t bytes) {
  char buffer[64];
  if (bytes >= (1ull << 30)) {
    snprintf(buffer, sizeof(buffer), "%.2f GiB", bytes / double(1ull << 30));
  } else if (bytes >= (1ull << 20)) {
    snprintf(buffer, sizeof(buffer), "%.2f MiB", bytes / double(1ull << 20));
  } else if (bytes >= (1ull << 10)) {
    snprintf(buffer, sizeof(buffer), "%.2f KiB", bytes / double(1ull << 10));
  } else {
    snprintf(buffer, sizeof(buffer), "%" PRIu64 " B", bytes);
  }
  return buffer;
}

const char* SeverityName(tracy::MessageSeverity severity) {
  switch (severity) {
    case tracy::MessageSeverity::Trace:
      return "trace";
    case tracy::MessageSeverity::Debug:
      return "debug";
    case tracy::MessageSeverity::Info:
      return "info";
    case tracy::MessageSeverity::Warning:
      return "warning";
    case tracy::MessageSeverity::Error:
      return "error";
    case tracy::MessageSeverity::Fatal:
      return "fatal";
    default:
      return "unknown";
  }
}

const char* MessageSourceName(tracy::MessageSourceType source) {
  switch (source) {
    case tracy::MessageSourceType::User:
      return "user";
    case tracy::MessageSourceType::Tracy:
      return "tracy";
    default:
      return "unknown";
  }
}

const char* PlotTypeName(tracy::PlotType type) {
  switch (type) {
    case tracy::PlotType::User:
      return "user";
    case tracy::PlotType::Memory:
      return "memory";
    case tracy::PlotType::SysTime:
      return "systime";
    case tracy::PlotType::Power:
      return "power";
    default:
      return "unknown";
  }
}

const char* PlotFormatName(tracy::PlotValueFormatting format) {
  switch (format) {
    case tracy::PlotValueFormatting::Number:
      return "number";
    case tracy::PlotValueFormatting::Memory:
      return "memory";
    case tracy::PlotValueFormatting::Percentage:
      return "percentage";
    case tracy::PlotValueFormatting::Watt:
      return "watt";
    default:
      return "unknown";
  }
}

// Adds the standard aggregate fields to |row|.
void AddStats(JsonRow& row, DurationStats& stats) {
  row.UInt("count", stats.count);
  row.Int("total_ns", stats.total);
  row.Double("avg_ns", stats.Mean());
  row.Int("min_ns", stats.count ? stats.min : 0);
  row.Int("max_ns", stats.count ? stats.max : 0);
  row.Double("stddev_ns", stats.StdDev());
  row.Int("p50_ns", stats.Percentile(0.50));
  row.Int("p90_ns", stats.Percentile(0.90));
  row.Int("p99_ns", stats.Percentile(0.99));
}

void AddContext(JsonRow& row, const GpuContextInfo* context) {
  if (!context) {
    row.Null("context").Null("context_name").Null("lane");
    row.Null("physical_device_ordinal").Null("queue_ordinal");
    return;
  }
  row.UInt("context", context->id);
  row.Str("context_name", context->name);
  row.Str("lane", LaneName(context->lane));
  row.OptUInt("physical_device_ordinal", context->physical_device_ordinal);
  row.OptUInt("queue_ordinal", context->queue_ordinal);
}

//===----------------------------------------------------------------------===//
// GPU zone grouping (shared by dispatch/queue/statistics/explain)
//===----------------------------------------------------------------------===//

struct GpuGroup {
  std::string key;
  uint8_t context = 0;
  DurationStats stats;
  uint64_t unresolved = 0;
  int64_t first = INT64_MAX;
  int64_t last = INT64_MIN;
};

bool LaneMatches(Lane lane, bool dispatch_like) {
  if (dispatch_like) {
    return lane == Lane::kDispatch || lane == Lane::kHost ||
           lane == Lane::kStream;
  }
  return lane == Lane::kQueue || lane == Lane::kSubmit || lane == Lane::kHost;
}

// Host-execution lanes carry both dispatch bodies (named after the executable
// function) and queue operations (named after the operation type). The HAL
// profile sink names the latter with these fixed strings.
bool IsQueueOperationName(std::string_view name) {
  static const char* const kNames[] = {
      "barrier", "dispatch", "execute", "copy",     "fill",     "update",
      "read",    "write",    "alloca",  "dealloca", "host_call"};
  for (const char* candidate : kNames) {
    if (name == candidate) return true;
  }
  return name.rfind("queue_op_", 0) == 0;
}

// True if a zone on |lane| belongs to the dispatch-like or queue-like view.
bool ZoneMatchesView(Lane lane, std::string_view name, bool dispatch_like) {
  if (!LaneMatches(lane, dispatch_like)) return false;
  if (lane != Lane::kHost) return true;
  return IsQueueOperationName(name) != dispatch_like;
}

// Selects the GPU zones a command operates on. Dispatch-like commands take
// dispatch/host/stream lanes, queue-like commands take queue/submit lanes; a
// capture with no decodable lane names falls back to every context.
std::vector<const GpuZoneRef*> SelectGpuZones(const Trace& trace,
                                              const Options& options,
                                              bool dispatch_like) {
  bool any_decoded = false;
  for (const auto& context : trace.gpu_contexts()) {
    if (context.lane != Lane::kUnknown && context.lane != Lane::kStream) {
      any_decoded = true;
    }
  }
  std::vector<const GpuZoneRef*> selected;
  for (const auto& ref : trace.gpu_zones()) {
    if (options.context && ref.context != *options.context) continue;
    const GpuContextInfo* context = trace.FindGpuContext(ref.context);
    if (!options.context && any_decoded && context &&
        !ZoneMatchesView(context->lane, trace.ZoneName(*ref.zone),
                         dispatch_like)) {
      continue;
    }
    if (!options.context && !any_decoded && !dispatch_like) continue;
    if (!options.filter.empty() &&
        !MatchPattern(trace.ZoneName(*ref.zone), options.filter)) {
      continue;
    }
    selected.push_back(&ref);
  }
  return selected;
}

std::vector<GpuGroup> GroupGpuZones(
    const Trace& trace, const std::vector<const GpuZoneRef*>& zones) {
  std::map<std::pair<uint8_t, std::string>, GpuGroup> groups;
  for (const GpuZoneRef* ref : zones) {
    const tracy::GpuEvent& zone = *ref->zone;
    auto key = std::make_pair(ref->context, std::string(trace.ZoneName(zone)));
    GpuGroup& group = groups[key];
    group.key = key.second;
    group.context = ref->context;
    if (zone.GpuStart() < 0 || zone.GpuEnd() < 0) {
      ++group.unresolved;
      continue;
    }
    group.stats.Add(zone.GpuEnd() - zone.GpuStart());
    group.first = std::min(group.first, zone.GpuStart());
    group.last = std::max(group.last, zone.GpuEnd());
  }
  std::vector<GpuGroup> result;
  result.reserve(groups.size());
  for (auto& it : groups) result.push_back(std::move(it.second));
  std::sort(result.begin(), result.end(),
            [](const GpuGroup& a, const GpuGroup& b) {
              return a.stats.total > b.stats.total;
            });
  return result;
}

void WriteGpuEventRow(const Trace& trace, const GpuZoneRef& ref,
                      const char* type, FILE* out) {
  const tracy::GpuEvent& zone = *ref.zone;
  JsonRow row(type);
  row.UInt("event_id", ref.event_id);
  row.Str("key", trace.ZoneName(zone));
  AddContext(row, trace.FindGpuContext(ref.context));
  row.UInt("thread", ref.thread_id);
  row.UInt("depth", ref.depth);
  if (zone.GpuStart() >= 0 && zone.GpuEnd() >= 0) {
    row.Int("gpu_start_ns", zone.GpuStart());
    row.Int("gpu_end_ns", zone.GpuEnd());
    row.Int("duration_ns", zone.GpuEnd() - zone.GpuStart());
    row.Int("since_start_ns", trace.TimeSinceStart(zone.GpuStart()));
  } else {
    row.Null("gpu_start_ns").Null("gpu_end_ns").Null("duration_ns");
    row.Null("since_start_ns");
  }
  row.Int("cpu_start_ns", zone.CpuStart());
  row.Int("cpu_end_ns", zone.CpuEnd());
  row.Write(out);
}

void WriteGpuGroupRow(const Trace& trace, GpuGroup& group, int64_t lane_total,
                      const char* type, FILE* out) {
  JsonRow row(type);
  row.Str("key", group.key);
  AddContext(row, trace.FindGpuContext(group.context));
  AddStats(row, group.stats);
  row.UInt("unresolved_count", group.unresolved);
  row.Double("share", lane_total ? double(group.stats.total) / lane_total : 0);
  if (group.stats.count) {
    row.Int("first_ns", group.first).Int("last_ns", group.last);
  } else {
    row.Null("first_ns").Null("last_ns");
  }
  row.Write(out);
}

int RunGpuCommand(Trace& trace, const Options& options, FILE* out,
                  bool dispatch_like, const char* title, const char* group_type,
                  const char* event_type) {
  std::vector<const GpuZoneRef*> zones =
      SelectGpuZones(trace, options, dispatch_like);

  if (options.id) {
    const GpuZoneRef* found = nullptr;
    for (const GpuZoneRef* ref : zones) {
      if (ref->event_id == *options.id) found = ref;
    }
    if (!found) {
      fprintf(stderr, "no %s event with id %" PRIu64 "\n", title, *options.id);
      return 1;
    }
    if (options.format == Format::kJsonl) {
      WriteGpuEventRow(trace, *found, event_type, out);
    } else {
      const tracy::GpuEvent& zone = *found->zone;
      const GpuContextInfo* context = trace.FindGpuContext(found->context);
      fprintf(out, "%s event %" PRIu64 "\n", title, found->event_id);
      fprintf(out, "  key: %s\n", trace.ZoneName(zone));
      fprintf(out, "  context: [%u] %s\n", found->context,
              context ? context->name.c_str() : "?");
      fprintf(out, "  thread: %s\n",
              trace.ThreadName(found->thread_id).c_str());
      if (zone.GpuStart() >= 0 && zone.GpuEnd() >= 0) {
        fprintf(out, "  gpu: start=%" PRId64 " end=%" PRId64 " duration=%s\n",
                zone.GpuStart(), zone.GpuEnd(),
                FormatNs(zone.GpuEnd() - zone.GpuStart()).c_str());
      } else {
        fprintf(out, "  gpu: (unresolved timestamps)\n");
      }
      fprintf(out, "  cpu: start=%" PRId64 " end=%" PRId64 "\n",
              zone.CpuStart(), zone.CpuEnd());
    }
    return 0;
  }

  if (options.events) {
    for (const GpuZoneRef* ref : zones) {
      WriteGpuEventRow(trace, *ref, event_type, out);
    }
    return 0;
  }

  std::vector<GpuGroup> groups = GroupGpuZones(trace, zones);
  std::map<uint8_t, int64_t> lane_totals;
  for (const GpuGroup& group : groups) {
    lane_totals[group.context] += group.stats.total;
  }

  if (options.format == Format::kJsonl) {
    for (GpuGroup& group : groups) {
      WriteGpuGroupRow(trace, group, lane_totals[group.context], group_type,
                       out);
    }
    return 0;
  }

  fprintf(out, "%s groups (%zu, %zu events)\n", title, groups.size(),
          zones.size());
  if (groups.empty()) {
    fprintf(out,
            "  (no matching GPU zones; capture with --device_profiling_tracy "
            "or --hip_tracing / --cuda_tracing)\n");
    return 0;
  }
  fprintf(out, "  %-8s %8s %12s %12s %12s %12s %6s  %s\n", "ctx", "count",
          "total", "avg", "p50", "max", "share", "key");
  for (GpuGroup& group : groups) {
    const int64_t lane_total = lane_totals[group.context];
    fprintf(out, "  %-8u %8" PRIu64 " %12s %12s %12s %12s %5.1f%%  %s\n",
            group.context, group.stats.count,
            FormatNs(group.stats.total).c_str(),
            FormatNs(static_cast<int64_t>(group.stats.Mean())).c_str(),
            FormatNs(group.stats.Percentile(0.5)).c_str(),
            FormatNs(group.stats.count ? group.stats.max : 0).c_str(),
            lane_total ? 100.0 * group.stats.total / lane_total : 0.0,
            group.key.c_str());
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// CPU zone grouping
//===----------------------------------------------------------------------===//

struct CpuGroup {
  std::string key;
  std::string function;
  std::string file;
  uint32_t line = 0;
  DurationStats stats;
  int64_t self_total = 0;
  std::map<uint64_t, uint64_t> thread_counts;
};

std::vector<const CpuZoneRef*> SelectCpuZones(const Trace& trace,
                                              const Options& options) {
  std::vector<const CpuZoneRef*> selected;
  for (const auto& ref : trace.cpu_zones()) {
    if (options.thread && ref.thread_id != *options.thread) continue;
    if (!options.filter.empty() &&
        !MatchPattern(trace.ZoneName(*ref.zone), options.filter)) {
      continue;
    }
    selected.push_back(&ref);
  }
  return selected;
}

std::vector<CpuGroup> GroupCpuZones(
    const Trace& trace, const std::vector<const CpuZoneRef*>& zones) {
  std::map<std::string, CpuGroup> groups;
  for (const CpuZoneRef* ref : zones) {
    const tracy::ZoneEvent& zone = *ref->zone;
    const auto& srcloc = trace.SourceLocation(zone.SrcLoc());
    const char* name = trace.ZoneName(zone);
    const char* file = trace.String(srcloc.file);
    std::string key = std::string(name) + "\x1f" + file + "\x1f" +
                      std::to_string(srcloc.line);
    CpuGroup& group = groups[key];
    if (group.key.empty()) {
      group.key = name;
      group.function = trace.String(srcloc.function);
      group.file = file;
      group.line = srcloc.line;
    }
    const int64_t duration = trace.ZoneEnd(zone) - zone.Start();
    group.stats.Add(duration);
    group.self_total += duration - trace.ChildTime(zone);
    group.thread_counts[ref->thread_id]++;
  }
  std::vector<CpuGroup> result;
  result.reserve(groups.size());
  for (auto& it : groups) result.push_back(std::move(it.second));
  std::sort(result.begin(), result.end(),
            [](const CpuGroup& a, const CpuGroup& b) {
              return a.self_total > b.self_total;
            });
  return result;
}

void WriteCpuEventRow(const Trace& trace, const CpuZoneRef& ref, FILE* out) {
  const tracy::ZoneEvent& zone = *ref.zone;
  const auto& srcloc = trace.SourceLocation(zone.SrcLoc());
  const int64_t end = trace.ZoneEnd(zone);
  JsonRow row("zone_event");
  row.UInt("event_id", ref.event_id);
  row.Str("key", trace.ZoneName(zone));
  row.Str("function", trace.String(srcloc.function));
  row.Str("file", trace.String(srcloc.file));
  row.UInt("line", srcloc.line);
  row.UInt("thread", ref.thread_id);
  row.Str("thread_name", trace.ThreadName(ref.thread_id));
  row.UInt("depth", ref.depth);
  row.Int("start_ns", zone.Start());
  row.Int("end_ns", end);
  row.Int("duration_ns", end - zone.Start());
  row.Int("self_ns", end - zone.Start() - trace.ChildTime(zone));
  row.Int("since_start_ns", trace.TimeSinceStart(zone.Start()));
  std::string text = trace.ZoneText(zone);
  if (text.empty()) {
    row.Null("text");
  } else {
    row.Str("text", text);
  }
  row.Write(out);
}

//===----------------------------------------------------------------------===//
// Memory pool summaries
//===----------------------------------------------------------------------===//

// tracy's MemData::high/low are the address range of the pool and
// active/usage are not rebuilt when loading from a file, so peak and live
// usage are derived from the alloc/free events here.
struct MemoryPoolSummary {
  uint64_t allocation_count = 0;
  uint64_t free_count = 0;
  uint64_t live_count = 0;
  uint64_t live_bytes = 0;
  uint64_t total_allocated_bytes = 0;
  uint64_t high_water_bytes = 0;
};

MemoryPoolSummary SummarizeMemoryPool(const tracy::MemData& mem) {
  MemoryPoolSummary summary;
  std::vector<std::pair<int64_t, int64_t>> deltas;
  deltas.reserve(mem.data.size() * 2);
  for (const auto& event : mem.data) {
    summary.allocation_count++;
    summary.total_allocated_bytes += event.Size();
    deltas.emplace_back(event.TimeAlloc(), static_cast<int64_t>(event.Size()));
    if (event.TimeFree() >= 0) {
      summary.free_count++;
      deltas.emplace_back(event.TimeFree(),
                          -static_cast<int64_t>(event.Size()));
    } else {
      summary.live_count++;
      summary.live_bytes += event.Size();
    }
  }
  std::sort(deltas.begin(), deltas.end());
  int64_t usage = 0;
  for (const auto& delta : deltas) {
    usage += delta.second;
    if (usage > static_cast<int64_t>(summary.high_water_bytes)) {
      summary.high_water_bytes = static_cast<uint64_t>(usage);
    }
  }
  return summary;
}

//===----------------------------------------------------------------------===//
// Per-thread and per-context summaries
//===----------------------------------------------------------------------===//

struct ThreadSummary {
  uint64_t id = 0;
  std::string name;
  uint64_t zone_count = 0;
  uint64_t message_count = 0;
  uint64_t sample_count = 0;
  int64_t busy_ns = 0;
  int64_t first = INT64_MAX;
  int64_t last = INT64_MIN;
};

std::vector<ThreadSummary> SummarizeThreads(const Trace& trace) {
  std::map<uint64_t, ThreadSummary> threads;
  std::map<uint64_t, std::vector<std::pair<int64_t, int64_t>>> intervals;
  for (const tracy::ThreadData* td : trace.worker().GetThreadData()) {
    ThreadSummary& summary = threads[td->id];
    summary.id = td->id;
    summary.name = trace.ThreadName(td->id);
    summary.message_count = td->messages.size();
    summary.sample_count = td->samples.size();
  }
  for (const auto& ref : trace.cpu_zones()) {
    ThreadSummary& summary = threads[ref.thread_id];
    summary.zone_count++;
    const int64_t start = ref.zone->Start();
    const int64_t end = trace.ZoneEnd(*ref.zone);
    summary.first = std::min(summary.first, start);
    summary.last = std::max(summary.last, end);
    if (ref.depth == 0) intervals[ref.thread_id].emplace_back(start, end);
  }
  std::vector<ThreadSummary> result;
  for (auto& it : threads) {
    it.second.busy_ns = UnionDuration(intervals[it.first]);
    result.push_back(std::move(it.second));
  }
  std::sort(result.begin(), result.end(),
            [](const ThreadSummary& a, const ThreadSummary& b) {
              return a.busy_ns > b.busy_ns;
            });
  return result;
}

struct ContextSummary {
  const GpuContextInfo* info = nullptr;
  uint64_t zone_count = 0;
  uint64_t unresolved = 0;
  int64_t total_ns = 0;
  int64_t busy_ns = 0;
  int64_t first = INT64_MAX;
  int64_t last = INT64_MIN;
};

std::vector<ContextSummary> SummarizeContexts(const Trace& trace) {
  std::map<uint8_t, ContextSummary> contexts;
  std::map<uint8_t, std::vector<std::pair<int64_t, int64_t>>> intervals;
  for (const auto& info : trace.gpu_contexts()) {
    contexts[info.id].info = &info;
  }
  for (const auto& ref : trace.gpu_zones()) {
    ContextSummary& summary = contexts[ref.context];
    summary.zone_count++;
    const tracy::GpuEvent& zone = *ref.zone;
    if (zone.GpuStart() < 0 || zone.GpuEnd() < 0) {
      summary.unresolved++;
      continue;
    }
    summary.total_ns += zone.GpuEnd() - zone.GpuStart();
    summary.first = std::min(summary.first, zone.GpuStart());
    summary.last = std::max(summary.last, zone.GpuEnd());
    if (ref.depth == 0) {
      intervals[ref.context].emplace_back(zone.GpuStart(), zone.GpuEnd());
    }
  }
  std::vector<ContextSummary> result;
  for (auto& it : contexts) {
    it.second.busy_ns = UnionDuration(intervals[it.first]);
    result.push_back(std::move(it.second));
  }
  return result;
}

void WriteContextSummaryRow(const Trace& trace, const ContextSummary& summary,
                            const char* type, FILE* out) {
  JsonRow row(type);
  AddContext(row, summary.info);
  row.UInt("zone_count", summary.zone_count);
  row.UInt("unresolved_count", summary.unresolved);
  row.Int("total_ns", summary.total_ns);
  row.Int("busy_ns", summary.busy_ns);
  if (summary.zone_count > summary.unresolved) {
    row.Int("first_ns", summary.first).Int("last_ns", summary.last);
    row.Int("span_ns", summary.last - summary.first);
    row.Double("busy_fraction",
               summary.last > summary.first
                   ? double(summary.busy_ns) / (summary.last - summary.first)
                   : 0.0);
  } else {
    row.Null("first_ns").Null("last_ns").Null("span_ns").Null("busy_fraction");
  }
  row.Write(out);
}

}  // namespace

//===----------------------------------------------------------------------===//
// summary
//===----------------------------------------------------------------------===//

int RunSummary(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  std::vector<ThreadSummary> threads = SummarizeThreads(trace);
  std::vector<ContextSummary> contexts = SummarizeContexts(trace);

  uint64_t memory_events = 0;
  for (const auto& it : worker.GetMemNameMap()) {
    memory_events += it.second->data.size();
  }
  const int64_t span = trace.last_time() - trace.first_time();
  const char* failure = trace.FailureString();

  if (options.format == Format::kJsonl) {
    JsonRow row("summary");
    row.Str("capture_name", worker.GetCaptureName());
    row.Str("program", worker.GetCaptureProgram());
    row.Str("host_info", worker.GetHostInfo());
    row.UInt("pid", worker.GetPid());
    row.UInt("capture_time_unix", worker.GetCaptureTime());
    row.Int("resolution_ns", worker.GetResolution());
    row.Int("first_ns", trace.first_time());
    row.Int("last_ns", trace.last_time());
    row.Int("span_ns", span);
    row.UInt("cpu_zone_count", trace.cpu_zones().size());
    row.UInt("gpu_zone_count", trace.gpu_zones().size());
    row.UInt("gpu_context_count", trace.gpu_contexts().size());
    row.UInt("thread_count", threads.size());
    row.UInt("message_count", worker.GetMessages().size());
    row.UInt("plot_count", worker.GetPlots().size());
    row.UInt("memory_pool_count", worker.GetMemNameMap().size());
    row.UInt("memory_event_count", memory_events);
    row.UInt("frame_set_count", worker.GetFrames().size());
    row.UInt("callstack_sample_count", worker.GetCallstackSampleCount());
    row.UInt("lock_count", worker.GetLockMap().size());
    row.StrOrNull("failure", failure);
    row.Write(out);
    for (const ThreadSummary& thread : threads) {
      JsonRow trow("summary_thread");
      trow.UInt("thread", thread.id).Str("thread_name", thread.name);
      trow.UInt("zone_count", thread.zone_count);
      trow.UInt("message_count", thread.message_count);
      trow.UInt("sample_count", thread.sample_count);
      trow.Int("busy_ns", thread.busy_ns);
      trow.Double("busy_fraction", span ? double(thread.busy_ns) / span : 0.0);
      trow.Write(out);
    }
    for (const ContextSummary& context : contexts) {
      WriteContextSummaryRow(trace, context, "summary_gpu_context", out);
    }
    return 0;
  }

  fprintf(out, "Tracy capture summary\n");
  fprintf(out, "capture: name='%s' program='%s' pid=%" PRIu64 "\n",
          worker.GetCaptureName().c_str(), worker.GetCaptureProgram().c_str(),
          worker.GetPid());
  fprintf(out, "host: %s\n", worker.GetHostInfo().c_str());
  fprintf(out,
          "timeline: first_ns=%" PRId64 " last_ns=%" PRId64
          " span=%s resolution=%s\n",
          trace.first_time(), trace.last_time(), FormatNs(span).c_str(),
          FormatNs(worker.GetResolution()).c_str());
  fprintf(out,
          "events: cpu_zones=%zu gpu_zones=%zu gpu_contexts=%zu threads=%zu "
          "messages=%zu plots=%zu memory_pools=%zu memory_events=%" PRIu64
          " frame_sets=%zu callstack_samples=%" PRIu64 " locks=%zu\n",
          trace.cpu_zones().size(), trace.gpu_zones().size(),
          trace.gpu_contexts().size(), threads.size(),
          worker.GetMessages().size(), worker.GetPlots().size(),
          worker.GetMemNameMap().size(), memory_events,
          worker.GetFrames().size(), worker.GetCallstackSampleCount(),
          worker.GetLockMap().size());
  fprintf(out, "health: %s\n", failure ? failure : "ok");
  fprintf(out, "threads:\n");
  for (const ThreadSummary& thread : threads) {
    fprintf(out,
            "  %-24s zones=%-8" PRIu64 " messages=%-6" PRIu64
            " samples=%-8" PRIu64 " busy=%s (%.1f%%)\n",
            thread.name.c_str(), thread.zone_count, thread.message_count,
            thread.sample_count, FormatNs(thread.busy_ns).c_str(),
            span ? 100.0 * thread.busy_ns / span : 0.0);
  }
  fprintf(out, "gpu contexts:\n");
  if (contexts.empty()) fprintf(out, "  (none)\n");
  for (const ContextSummary& context : contexts) {
    const int64_t ctx_span = context.zone_count > context.unresolved
                                 ? context.last - context.first
                                 : 0;
    fprintf(out,
            "  [%u] %-28s lane=%-8s zones=%-8" PRIu64
            " total=%-12s busy=%s (%.1f%% of %s)\n",
            context.info ? context.info->id : 0,
            context.info ? context.info->name.c_str() : "?",
            context.info ? LaneName(context.info->lane) : "?",
            context.zone_count, FormatNs(context.total_ns).c_str(),
            FormatNs(context.busy_ns).c_str(),
            ctx_span ? 100.0 * context.busy_ns / ctx_span : 0.0,
            FormatNs(ctx_span).c_str());
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// statistics
//===----------------------------------------------------------------------===//

int RunStatistics(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  const bool jsonl = options.format == Format::kJsonl;

  // CPU zones by source location key.
  std::vector<CpuGroup> cpu_groups =
      GroupCpuZones(trace, SelectCpuZones(trace, options));
  // GPU zones by (context, name) across all lanes.
  Options all_contexts = options;
  std::vector<const GpuZoneRef*> gpu_zones;
  for (const auto& ref : trace.gpu_zones()) {
    if (!options.filter.empty() &&
        !MatchPattern(trace.ZoneName(*ref.zone), options.filter)) {
      continue;
    }
    gpu_zones.push_back(&ref);
  }
  std::vector<GpuGroup> gpu_groups = GroupGpuZones(trace, gpu_zones);

  std::map<int, uint64_t> severity_counts;
  for (const auto& message : worker.GetMessages()) {
    severity_counts[static_cast<int>(message->severity)]++;
  }

  if (jsonl) {
    for (CpuGroup& group : cpu_groups) {
      JsonRow row("statistics_row");
      row.Str("row_type", "cpu_zone").Str("event_name", group.key);
      row.Str("function", group.function).Str("file", group.file);
      row.UInt("line", group.line);
      AddStats(row, group.stats);
      row.Int("self_total_ns", group.self_total);
      row.UInt("thread_count", group.thread_counts.size());
      row.Write(out);
    }
    for (GpuGroup& group : gpu_groups) {
      JsonRow row("statistics_row");
      row.Str("row_type", "gpu_zone").Str("event_name", group.key);
      AddContext(row, trace.FindGpuContext(group.context));
      AddStats(row, group.stats);
      row.UInt("unresolved_count", group.unresolved);
      row.Write(out);
    }
    for (const auto& it : severity_counts) {
      JsonRow row("statistics_row");
      row.Str("row_type", "message_severity");
      row.Str("event_name",
              SeverityName(static_cast<tracy::MessageSeverity>(it.first)));
      row.UInt("count", it.second);
      row.Write(out);
    }
    for (const tracy::PlotData* plot : worker.GetPlots()) {
      const char* name = trace.String(plot->name);
      if (!MatchPattern(name, options.filter)) continue;
      JsonRow row("statistics_row");
      row.Str("row_type", "plot").Str("event_name", name);
      row.UInt("count", plot->data.size());
      row.Double("min", plot->min).Double("max", plot->max);
      row.Double("avg",
                 plot->data.empty() ? 0.0 : plot->sum / plot->data.size());
      row.Write(out);
    }
    for (const auto& it : worker.GetMemNameMap()) {
      const tracy::MemData& mem = *it.second;
      const char* name = it.first ? trace.String(it.first) : "default";
      if (!MatchPattern(name, options.filter)) continue;
      JsonRow row("statistics_row");
      MemoryPoolSummary pool = SummarizeMemoryPool(mem);
      row.Str("row_type", "memory_pool").Str("event_name", name);
      row.UInt("allocation_count", pool.allocation_count);
      row.UInt("free_count", pool.free_count);
      row.UInt("live_count", pool.live_count);
      row.UInt("live_bytes", pool.live_bytes);
      row.UInt("high_water_bytes", pool.high_water_bytes);
      row.UInt("total_allocated_bytes", pool.total_allocated_bytes);
      row.Write(out);
    }
    for (const tracy::FrameData* frames : worker.GetFrames()) {
      const char* name = trace.String(frames->name);
      if (!MatchPattern(name, options.filter)) continue;
      JsonRow row("statistics_row");
      row.Str("row_type", "frame_set").Str("event_name", name);
      row.UInt("count", frames->frames.size());
      row.Int("total_ns", frames->total);
      row.Int("min_ns", frames->frames.empty() ? 0 : frames->min);
      row.Int("max_ns", frames->frames.empty() ? 0 : frames->max);
      row.Write(out);
    }
    return 0;
  }

  fprintf(out, "Tracy capture statistics:\n");
  fprintf(out, "  cpu_zone_groups=%zu gpu_zone_groups=%zu\n", cpu_groups.size(),
          gpu_groups.size());
  fprintf(out, "cpu zones (by self time):\n");
  size_t shown = 0;
  for (CpuGroup& group : cpu_groups) {
    if (options.top && shown++ >= options.top * 3) break;
    fprintf(out, "  %-48s count=%-8" PRIu64 " total=%-12s self=%-12s avg=%s\n",
            group.key.c_str(), group.stats.count,
            FormatNs(group.stats.total).c_str(),
            FormatNs(group.self_total).c_str(),
            FormatNs(static_cast<int64_t>(group.stats.Mean())).c_str());
  }
  fprintf(out, "gpu zones (by total):\n");
  for (GpuGroup& group : gpu_groups) {
    fprintf(out, "  [%u] %-44s count=%-8" PRIu64 " total=%-12s avg=%s\n",
            group.context, group.key.c_str(), group.stats.count,
            FormatNs(group.stats.total).c_str(),
            FormatNs(static_cast<int64_t>(group.stats.Mean())).c_str());
  }
  fprintf(out, "messages:\n");
  for (const auto& it : severity_counts) {
    fprintf(out, "  %-8s count=%" PRIu64 "\n",
            SeverityName(static_cast<tracy::MessageSeverity>(it.first)),
            it.second);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// dispatch / queue
//===----------------------------------------------------------------------===//

int RunDispatch(Trace& trace, const Options& options, FILE* out) {
  return RunGpuCommand(trace, options, out, /*dispatch_like=*/true, "dispatch",
                       "dispatch_group", "dispatch_event");
}

int RunQueue(Trace& trace, const Options& options, FILE* out) {
  return RunGpuCommand(trace, options, out, /*dispatch_like=*/false, "queue",
                       "queue_group", "queue_event");
}

//===----------------------------------------------------------------------===//
// zone
//===----------------------------------------------------------------------===//

int RunZone(Trace& trace, const Options& options, FILE* out) {
  std::vector<const CpuZoneRef*> zones = SelectCpuZones(trace, options);

  if (options.id) {
    const CpuZoneRef* found = nullptr;
    for (const CpuZoneRef* ref : zones) {
      if (ref->event_id == *options.id) found = ref;
    }
    if (!found) {
      fprintf(stderr, "no zone event with id %" PRIu64 "\n", *options.id);
      return 1;
    }
    if (options.format == Format::kJsonl) {
      WriteCpuEventRow(trace, *found, out);
    } else {
      const tracy::ZoneEvent& zone = *found->zone;
      const auto& srcloc = trace.SourceLocation(zone.SrcLoc());
      const int64_t end = trace.ZoneEnd(zone);
      fprintf(out, "zone event %" PRIu64 "\n", found->event_id);
      fprintf(out, "  key: %s\n", trace.ZoneName(zone));
      fprintf(out, "  source: %s (%s:%u)\n", trace.String(srcloc.function),
              trace.String(srcloc.file), srcloc.line);
      fprintf(out, "  thread: %s depth=%u\n",
              trace.ThreadName(found->thread_id).c_str(), found->depth);
      fprintf(out,
              "  time: start=%" PRId64 " end=%" PRId64 " duration=%s self=%s\n",
              zone.Start(), end, FormatNs(end - zone.Start()).c_str(),
              FormatNs(end - zone.Start() - trace.ChildTime(zone)).c_str());
      std::string text = trace.ZoneText(zone);
      if (!text.empty()) fprintf(out, "  text: %s\n", text.c_str());
    }
    return 0;
  }

  if (options.events) {
    for (const CpuZoneRef* ref : zones) WriteCpuEventRow(trace, *ref, out);
    return 0;
  }

  std::vector<CpuGroup> groups = GroupCpuZones(trace, zones);
  if (options.format == Format::kJsonl) {
    for (CpuGroup& group : groups) {
      JsonRow row("zone_group");
      row.Str("key", group.key).Str("function", group.function);
      row.Str("file", group.file).UInt("line", group.line);
      AddStats(row, group.stats);
      row.Int("self_total_ns", group.self_total);
      row.Double("self_avg_ns", group.stats.count ? double(group.self_total) /
                                                        group.stats.count
                                                  : 0.0);
      row.UInt("thread_count", group.thread_counts.size());
      row.Write(out);
    }
    return 0;
  }

  fprintf(out, "zone groups (%zu, %zu events)\n", groups.size(), zones.size());
  fprintf(out, "  %8s %12s %12s %12s  %s\n", "count", "total", "self", "avg",
          "key (file:line)");
  for (CpuGroup& group : groups) {
    fprintf(out, "  %8" PRIu64 " %12s %12s %12s  %s (%s:%u)\n",
            group.stats.count, FormatNs(group.stats.total).c_str(),
            FormatNs(group.self_total).c_str(),
            FormatNs(static_cast<int64_t>(group.stats.Mean())).c_str(),
            group.key.c_str(), group.file.c_str(), group.line);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// thread
//===----------------------------------------------------------------------===//

int RunThread(Trace& trace, const Options& options, FILE* out) {
  std::vector<ThreadSummary> threads = SummarizeThreads(trace);
  const int64_t span = trace.last_time() - trace.first_time();
  if (options.format == Format::kJsonl) {
    for (const ThreadSummary& thread : threads) {
      if (options.id && thread.id != *options.id) continue;
      JsonRow row("thread");
      row.UInt("thread", thread.id).Str("thread_name", thread.name);
      row.UInt("zone_count", thread.zone_count);
      row.UInt("message_count", thread.message_count);
      row.UInt("sample_count", thread.sample_count);
      row.Int("busy_ns", thread.busy_ns);
      row.Double("busy_fraction", span ? double(thread.busy_ns) / span : 0.0);
      if (thread.zone_count) {
        row.Int("first_ns", thread.first).Int("last_ns", thread.last);
      } else {
        row.Null("first_ns").Null("last_ns");
      }
      row.Write(out);
    }
    return 0;
  }
  fprintf(out, "threads (%zu)\n", threads.size());
  fprintf(out, "  %-10s %-24s %8s %8s %8s %12s %6s\n", "id", "name", "zones",
          "msgs", "samples", "busy", "share");
  for (const ThreadSummary& thread : threads) {
    if (options.id && thread.id != *options.id) continue;
    fprintf(out,
            "  %-10" PRIu64 " %-24s %8" PRIu64 " %8" PRIu64 " %8" PRIu64
            " %12s %5.1f%%\n",
            thread.id, thread.name.c_str(), thread.zone_count,
            thread.message_count, thread.sample_count,
            FormatNs(thread.busy_ns).c_str(),
            span ? 100.0 * thread.busy_ns / span : 0.0);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// message
//===----------------------------------------------------------------------===//

int RunMessage(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  for (const auto& message : worker.GetMessages()) {
    const uint64_t thread = worker.DecompressThread(message->thread);
    if (options.thread && thread != *options.thread) continue;
    const char* text = trace.String(message->ref);
    if (!MatchPattern(text, options.filter)) continue;
    if (options.format == Format::kJsonl) {
      JsonRow row("message");
      row.Int("time_ns", message->time);
      row.Int("since_start_ns", trace.TimeSinceStart(message->time));
      row.UInt("thread", thread).Str("thread_name", trace.ThreadName(thread));
      row.Str("severity", SeverityName(message->severity));
      row.Str("source", MessageSourceName(message->source));
      row.UInt("color", message->color);
      row.Str("text", text);
      row.Write(out);
    } else {
      fprintf(out, "%14s %-8s %-20s %s\n",
              FormatNs(trace.TimeSinceStart(message->time)).c_str(),
              SeverityName(message->severity), trace.ThreadName(thread).c_str(),
              text);
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// plot
//===----------------------------------------------------------------------===//

int RunPlot(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  for (const tracy::PlotData* plot : worker.GetPlots()) {
    const char* name = trace.String(plot->name);
    if (!MatchPattern(name, options.filter)) continue;
    if (options.events) {
      for (const auto& item : plot->data) {
        JsonRow row("plot_sample");
        row.Str("name", name);
        row.Int("time_ns", item.time.Val());
        row.Int("since_start_ns", trace.TimeSinceStart(item.time.Val()));
        row.Double("value", item.val);
        row.Write(out);
      }
      continue;
    }
    const double avg = plot->data.empty() ? 0.0 : plot->sum / plot->data.size();
    if (options.format == Format::kJsonl) {
      JsonRow row("plot");
      row.Str("name", name);
      row.Str("plot_type", PlotTypeName(plot->type));
      row.Str("format", PlotFormatName(plot->format));
      row.UInt("count", plot->data.size());
      row.Double("min", plot->min).Double("max", plot->max).Double("avg", avg);
      if (!plot->data.empty()) {
        row.Int("first_ns", plot->data.front().time.Val());
        row.Int("last_ns", plot->data.back().time.Val());
      } else {
        row.Null("first_ns").Null("last_ns");
      }
      row.Write(out);
    } else {
      fprintf(out, "  %-40s type=%-7s samples=%-8zu min=%g max=%g avg=%g\n",
              name, PlotTypeName(plot->type), plot->data.size(), plot->min,
              plot->max, avg);
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// memory
//===----------------------------------------------------------------------===//

int RunMemory(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  for (const auto& it : worker.GetMemNameMap()) {
    const tracy::MemData& mem = *it.second;
    const char* name = it.first ? trace.String(it.first) : "default";
    if (!MatchPattern(name, options.filter)) continue;
    if (options.events) {
      for (const auto& event : mem.data) {
        JsonRow row("memory_event");
        row.Str("pool", name);
        char ptr[32];
        snprintf(ptr, sizeof(ptr), "0x%" PRIx64, event.Ptr());
        row.Str("ptr", ptr);
        row.UInt("size", event.Size());
        row.Int("alloc_time_ns", event.TimeAlloc());
        row.UInt("alloc_thread", worker.DecompressThread(event.ThreadAlloc()));
        if (event.TimeFree() >= 0) {
          row.Int("free_time_ns", event.TimeFree());
          row.UInt("free_thread", worker.DecompressThread(event.ThreadFree()));
          row.Int("lifetime_ns", event.TimeFree() - event.TimeAlloc());
        } else {
          row.Null("free_time_ns").Null("free_thread").Null("lifetime_ns");
        }
        row.Write(out);
      }
      continue;
    }
    MemoryPoolSummary pool = SummarizeMemoryPool(mem);
    if (options.format == Format::kJsonl) {
      JsonRow row("memory_pool");
      row.Str("pool", name);
      row.UInt("allocation_count", pool.allocation_count);
      row.UInt("free_count", pool.free_count);
      row.UInt("live_count", pool.live_count);
      row.UInt("live_bytes", pool.live_bytes);
      row.UInt("high_water_bytes", pool.high_water_bytes);
      row.UInt("total_allocated_bytes", pool.total_allocated_bytes);
      row.Write(out);
    } else {
      fprintf(out,
              "  %-24s allocations=%-8" PRIu64 " frees=%-8" PRIu64
              " live=%" PRIu64 " (%s) high_water=%s total_allocated=%s\n",
              name, pool.allocation_count, pool.free_count, pool.live_count,
              FormatBytes(pool.live_bytes).c_str(),
              FormatBytes(pool.high_water_bytes).c_str(),
              FormatBytes(pool.total_allocated_bytes).c_str());
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// frame
//===----------------------------------------------------------------------===//

int RunFrame(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  for (const tracy::FrameData* frames : worker.GetFrames()) {
    const char* name = trace.String(frames->name);
    if (!MatchPattern(name, options.filter)) continue;
    const size_t count = worker.GetFrameCount(*frames);
    if (options.events) {
      for (size_t i = 0; i < count; ++i) {
        JsonRow row("frame_event");
        row.Str("frame_set", name).UInt("index", i);
        row.Int("start_ns", worker.GetFrameBegin(*frames, i));
        row.Int("end_ns", worker.GetFrameEnd(*frames, i));
        row.Int("duration_ns", worker.GetFrameTime(*frames, i));
        row.Write(out);
      }
      continue;
    }
    if (options.format == Format::kJsonl) {
      JsonRow row("frame_set");
      row.Str("frame_set", name).Bool("continuous", frames->continuous != 0);
      row.UInt("count", count);
      row.Int("total_ns", frames->total);
      row.Int("min_ns", count ? frames->min : 0);
      row.Int("max_ns", count ? frames->max : 0);
      row.Double("avg_ns", count ? double(frames->total) / count : 0.0);
      row.Write(out);
    } else {
      fprintf(out,
              "  %-24s frames=%-8zu total=%-12s min=%-12s max=%-12s avg=%s\n",
              name, count, FormatNs(frames->total).c_str(),
              FormatNs(count ? frames->min : 0).c_str(),
              FormatNs(count ? frames->max : 0).c_str(),
              FormatNs(count ? frames->total / int64_t(count) : 0).c_str());
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// explain
//===----------------------------------------------------------------------===//

int RunExplain(Trace& trace, const Options& options, FILE* out) {
  const tracy::Worker& worker = trace.worker();
  const bool jsonl = options.format == Format::kJsonl;
  const size_t top = options.top ? options.top : SIZE_MAX;
  const int64_t span = trace.last_time() - trace.first_time();

  std::vector<ContextSummary> contexts = SummarizeContexts(trace);
  std::vector<ThreadSummary> threads = SummarizeThreads(trace);
  Options unfiltered;
  std::vector<GpuGroup> dispatches =
      GroupGpuZones(trace, SelectGpuZones(trace, unfiltered, true));
  std::vector<GpuGroup> queue_ops =
      GroupGpuZones(trace, SelectGpuZones(trace, unfiltered, false));
  std::vector<CpuGroup> cpu_groups =
      GroupCpuZones(trace, SelectCpuZones(trace, unfiltered));

  int64_t dispatch_total = 0;
  for (const GpuGroup& group : dispatches) dispatch_total += group.stats.total;
  int64_t device_busy = 0, device_span = 0;
  for (const ContextSummary& context : contexts) {
    if (!context.info || !LaneMatches(context.info->lane, true)) continue;
    device_busy += context.busy_ns;
    if (context.zone_count > context.unresolved) {
      device_span = std::max(device_span, context.last - context.first);
    }
  }

  // Hints, evidence-backed.
  struct Hint {
    const char* severity;
    std::string message;
  };
  std::vector<Hint> hints;
  if (const char* failure = trace.FailureString()) {
    hints.push_back({"error", std::string("capture failed: ") + failure});
  }
  if (trace.gpu_contexts().empty()) {
    hints.push_back({"info",
                     "no device timelines in this capture; run with "
                     "--device_profiling_mode=dispatch-events "
                     "--device_profiling_tracy (HAL profiling backends) or "
                     "--hip_tracing=2 / --cuda_tracing=2"});
  }
  if (device_span > 0 && device_busy < device_span * 7 / 10) {
    char buffer[160];
    snprintf(
        buffer, sizeof(buffer),
        "device is idle %.1f%% of its active span (%s idle of %s); look at "
        "submission gaps and host-side work between dispatches",
        100.0 * (device_span - device_busy) / device_span,
        FormatNs(device_span - device_busy).c_str(),
        FormatNs(device_span).c_str());
    hints.push_back({"warning", buffer});
  }
  if (!dispatches.empty() && dispatch_total > 0 &&
      dispatches[0].stats.total * 2 > dispatch_total) {
    char buffer[200];
    snprintf(buffer, sizeof(buffer),
             "'%s' alone is %.1f%% of dispatch time (%s over %" PRIu64
             " events); optimizing it dominates everything else",
             dispatches[0].key.c_str(),
             100.0 * dispatches[0].stats.total / dispatch_total,
             FormatNs(dispatches[0].stats.total).c_str(),
             dispatches[0].stats.count);
    hints.push_back({"info", buffer});
  }
  uint64_t unresolved = 0;
  for (const ContextSummary& context : contexts)
    unresolved += context.unresolved;
  if (unresolved) {
    hints.push_back(
        {"warning", std::to_string(unresolved) +
                        " GPU zones have no device timestamps; the capture "
                        "ended before their queries were collected"});
  }
  for (const auto& message : worker.GetMessages()) {
    if (message->severity < tracy::MessageSeverity::Warning) continue;
    hints.push_back({SeverityName(message->severity),
                     std::string("message: ") + trace.String(message->ref)});
    if (hints.size() > 32) break;
  }

  if (jsonl) {
    {
      JsonRow row("explain_span");
      row.Int("first_ns", trace.first_time()).Int("last_ns", trace.last_time());
      row.Int("span_ns", span);
      row.Int("device_busy_ns", device_busy);
      row.Int("device_span_ns", device_span);
      row.Double("device_busy_fraction",
                 device_span ? double(device_busy) / device_span : 0.0);
      row.Int("dispatch_total_ns", dispatch_total);
      row.UInt("dispatch_group_count", dispatches.size());
      row.UInt("cpu_zone_group_count", cpu_groups.size());
      row.Write(out);
    }
    for (const ContextSummary& context : contexts) {
      WriteContextSummaryRow(trace, context, "explain_context", out);
    }
    size_t rank = 0;
    for (GpuGroup& group : dispatches) {
      if (rank >= top) break;
      JsonRow row("explain_top_dispatch");
      row.UInt("rank", ++rank).Str("key", group.key);
      AddContext(row, trace.FindGpuContext(group.context));
      AddStats(row, group.stats);
      row.Double("share", dispatch_total
                              ? double(group.stats.total) / dispatch_total
                              : 0);
      row.Write(out);
    }
    rank = 0;
    for (GpuGroup& group : queue_ops) {
      if (rank >= top) break;
      JsonRow row("explain_top_queue");
      row.UInt("rank", ++rank).Str("key", group.key);
      AddContext(row, trace.FindGpuContext(group.context));
      AddStats(row, group.stats);
      row.Write(out);
    }
    rank = 0;
    for (CpuGroup& group : cpu_groups) {
      if (rank >= top) break;
      JsonRow row("explain_top_cpu_self");
      row.UInt("rank", ++rank).Str("key", group.key);
      row.Str("file", group.file).UInt("line", group.line);
      row.UInt("count", group.stats.count);
      row.Int("total_ns", group.stats.total);
      row.Int("self_total_ns", group.self_total);
      row.Double("self_share", span ? double(group.self_total) / span : 0.0);
      row.Write(out);
    }
    rank = 0;
    for (const ThreadSummary& thread : threads) {
      if (rank >= top) break;
      JsonRow row("explain_thread");
      row.UInt("rank", ++rank).UInt("thread", thread.id);
      row.Str("thread_name", thread.name);
      row.Int("busy_ns", thread.busy_ns);
      row.Double("busy_fraction", span ? double(thread.busy_ns) / span : 0.0);
      row.UInt("zone_count", thread.zone_count);
      row.Write(out);
    }
    for (const Hint& hint : hints) {
      JsonRow row("explain_hint");
      row.Str("severity", hint.severity).Str("message", hint.message);
      row.Write(out);
    }
    return 0;
  }

  fprintf(out, "Tracy capture explain\n");
  fprintf(out, "span: %s (first_ns=%" PRId64 " last_ns=%" PRId64 ")\n",
          FormatNs(span).c_str(), trace.first_time(), trace.last_time());
  if (device_span) {
    fprintf(
        out, "device: busy %s of %s active span (%.1f%%), dispatch total %s\n",
        FormatNs(device_busy).c_str(), FormatNs(device_span).c_str(),
        100.0 * device_busy / device_span, FormatNs(dispatch_total).c_str());
  }
  fprintf(out, "device contexts:\n");
  if (contexts.empty()) fprintf(out, "  (none)\n");
  for (const ContextSummary& context : contexts) {
    const int64_t ctx_span = context.zone_count > context.unresolved
                                 ? context.last - context.first
                                 : 0;
    fprintf(out, "  [%u] %-28s zones=%-8" PRIu64 " busy=%-12s (%.1f%% of %s)\n",
            context.info ? context.info->id : 0,
            context.info ? context.info->name.c_str() : "?", context.zone_count,
            FormatNs(context.busy_ns).c_str(),
            ctx_span ? 100.0 * context.busy_ns / ctx_span : 0.0,
            FormatNs(ctx_span).c_str());
  }
  fprintf(out, "top dispatches (by total):\n");
  size_t rank = 0;
  for (GpuGroup& group : dispatches) {
    if (rank++ >= top) break;
    fprintf(out, "  %5.1f%%  %-12s x%-7" PRIu64 " avg=%-11s p99=%-11s %s\n",
            dispatch_total ? 100.0 * group.stats.total / dispatch_total : 0.0,
            FormatNs(group.stats.total).c_str(), group.stats.count,
            FormatNs(static_cast<int64_t>(group.stats.Mean())).c_str(),
            FormatNs(group.stats.Percentile(0.99)).c_str(), group.key.c_str());
  }
  if (!queue_ops.empty()) {
    fprintf(out, "top queue operations (by total):\n");
    rank = 0;
    for (GpuGroup& group : queue_ops) {
      if (rank++ >= top) break;
      fprintf(out, "  %-12s x%-7" PRIu64 " avg=%-11s [%u] %s\n",
              FormatNs(group.stats.total).c_str(), group.stats.count,
              FormatNs(static_cast<int64_t>(group.stats.Mean())).c_str(),
              group.context, group.key.c_str());
    }
  }
  fprintf(out, "top CPU zones (by self time):\n");
  rank = 0;
  for (CpuGroup& group : cpu_groups) {
    if (rank++ >= top) break;
    fprintf(out, "  %-12s self x%-7" PRIu64 " total=%-11s %s (%s:%u)\n",
            FormatNs(group.self_total).c_str(), group.stats.count,
            FormatNs(group.stats.total).c_str(), group.key.c_str(),
            group.file.c_str(), group.line);
  }
  fprintf(out, "busiest threads:\n");
  rank = 0;
  for (const ThreadSummary& thread : threads) {
    if (rank++ >= top) break;
    fprintf(out, "  %5.1f%%  %-12s %s\n",
            span ? 100.0 * thread.busy_ns / span : 0.0,
            FormatNs(thread.busy_ns).c_str(), thread.name.c_str());
  }
  fprintf(out, "hints:\n");
  if (hints.empty()) fprintf(out, "  (none)\n");
  for (const Hint& hint : hints) {
    fprintf(out, "  [%s] %s\n", hint.severity, hint.message.c_str());
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// cat
//===----------------------------------------------------------------------===//

int RunCat(Trace& trace, const Options& options, FILE* out) {
  if (options.format != Format::kJsonl) {
    fprintf(stderr, "cat requires --format=jsonl\n");
    return 2;
  }
  const tracy::Worker& worker = trace.worker();
  for (const tracy::ThreadData* td : worker.GetThreadData()) {
    JsonRow row("thread");
    row.UInt("thread", td->id).Str("thread_name", trace.ThreadName(td->id));
    row.Write(out);
  }
  for (const auto& info : trace.gpu_contexts()) {
    JsonRow row("gpu_context");
    AddContext(row, &info);
    row.Str("api", [&]() -> const char* {
      switch (info.data->type) {
        case tracy::GpuContextType::Vulkan:
          return "vulkan";
        case tracy::GpuContextType::OpenGl:
          return "opengl";
        case tracy::GpuContextType::OpenCL:
          return "opencl";
        case tracy::GpuContextType::Direct3D12:
          return "d3d12";
        case tracy::GpuContextType::Direct3D11:
          return "d3d11";
        case tracy::GpuContextType::Metal:
          return "metal";
        default:
          return "other";
      }
    }());
    row.Write(out);
  }
  for (const auto& ref : trace.cpu_zones()) WriteCpuEventRow(trace, ref, out);
  for (const auto& ref : trace.gpu_zones()) {
    WriteGpuEventRow(trace, ref, "gpu_zone_event", out);
  }
  Options everything;
  everything.format = Format::kJsonl;
  everything.events = true;
  RunMessage(trace, everything, out);
  RunPlot(trace, everything, out);
  RunMemory(trace, everything, out);
  RunFrame(trace, everything, out);
  return 0;
}

}  // namespace iree_tracy_profile
