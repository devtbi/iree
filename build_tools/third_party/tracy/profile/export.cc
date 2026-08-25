// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ireeperf-jsonl (schema version 15) export of a Tracy capture.
//
// The stream mirrors what `iree-profile export --format=ireeperf-jsonl`
// produces from a .ireeprof bundle so that iree-profile-render and other
// downstream adapters consume Tracy captures unchanged. Only device/queue
// lanes are exported: dispatch-lane GPU zones become dispatch_event records,
// queue-lane zones queue_device_event, host-lane zones host_execution_event,
// and submit-lane zones queue_event. Tracy has no separate device clock (GPU
// zones are already on the host timeline) so device ticks are host
// nanoseconds and two identity clock_correlation samples are emitted per
// device; every device-timed record additionally carries derived host times so
// consumers may use either mapping. CPU zones, messages, plots, and memory
// events have no counterpart in the HAL profile schema and are not exported.

#include <cinttypes>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "TracyVersion.hpp"
#include "commands.h"

namespace iree_tracy_profile {
namespace {

constexpr int kSchemaVersion = 15;

// Queue operation names used by the HAL profile sink, in
// iree_hal_profile_queue_event_type_t order (1-based).
const char* const kQueueOpNames[] = {"barrier", "dispatch", "execute",  "copy",
                                     "fill",    "update",   "read",     "write",
                                     "alloca",  "dealloca", "host_call"};

uint32_t QueueOpValue(std::string_view name) {
  for (size_t i = 0; i < sizeof(kQueueOpNames) / sizeof(kQueueOpNames[0]);
       ++i) {
    if (name == kQueueOpNames[i]) return static_cast<uint32_t>(i + 1);
  }
  return 0;
}

class Exporter {
 public:
  Exporter(Trace& trace, FILE* out) : trace_(trace), out_(out) {}

  void Run() {
    CollectTopology();
    WriteSchema();
    WriteSession("begin");
    WriteDevicesAndQueues();
    WriteClockCorrelations();
    WriteExecutableFunctions();
    WriteEvents();
    WriteSession("end");
  }

 private:
  struct QueueKey {
    uint32_t device;
    uint32_t queue;
    bool operator<(const QueueKey& other) const {
      return device != other.device ? device < other.device
                                    : queue < other.queue;
    }
  };

  // Resolves the (device, queue) a context maps to. Undecodable contexts
  // (stream tracing) are assigned device 0 and a queue per context.
  QueueKey KeyForContext(const GpuContextInfo* context) const {
    QueueKey key{0, 0};
    if (!context) return key;
    if (context->physical_device_ordinal) {
      key.device = *context->physical_device_ordinal;
    }
    if (context->queue_ordinal) {
      key.queue = *context->queue_ordinal;
    } else if (context->lane == Lane::kStream ||
               context->lane == Lane::kUnknown) {
      key.queue = context->id;
    }
    return key;
  }

  uint64_t StreamId(const QueueKey& key) const {
    return static_cast<uint64_t>(key.device) * 1000 + key.queue + 1;
  }

  void Begin(std::string& buffer, const char* record_type) {
    buffer.clear();
    buffer += "{\"schema_version\":";
    buffer += std::to_string(kSchemaVersion);
    buffer += ",\"record_type\":\"";
    buffer += record_type;
    buffer += "\",\"source_record_index\":";
    buffer += std::to_string(record_index_++);
  }

  static void AppendString(std::string& buffer, std::string_view value) {
    buffer += '"';
    for (unsigned char c : value) {
      switch (c) {
        case '"':
          buffer += "\\\"";
          break;
        case '\\':
          buffer += "\\\\";
          break;
        case '\n':
          buffer += "\\n";
          break;
        case '\r':
          buffer += "\\r";
          break;
        case '\t':
          buffer += "\\t";
          break;
        default:
          if (c < 0x20) {
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "\\u%04x", c);
            buffer += tmp;
          } else {
            buffer += static_cast<char>(c);
          }
      }
    }
    buffer += '"';
  }
  static void Key(std::string& buffer, const char* key) {
    buffer += ",\"";
    buffer += key;
    buffer += "\":";
  }
  static void Str(std::string& buffer, const char* key, std::string_view v) {
    Key(buffer, key);
    AppendString(buffer, v);
  }
  static void Null(std::string& buffer, const char* key) {
    Key(buffer, key);
    buffer += "null";
  }
  static void Int(std::string& buffer, const char* key, int64_t v) {
    Key(buffer, key);
    buffer += std::to_string(v);
  }
  static void UInt(std::string& buffer, const char* key, uint64_t v) {
    Key(buffer, key);
    buffer += std::to_string(v);
  }
  static void Bool(std::string& buffer, const char* key, bool v) {
    Key(buffer, key);
    buffer += v ? "true" : "false";
  }
  static void Triple(std::string& buffer, const char* key) {
    Key(buffer, key);
    buffer += "[0,0,0]";
  }
  void End(std::string& buffer) {
    buffer += "}\n";
    fputs(buffer.c_str(), out_);
  }

  void CollectTopology() {
    for (const auto& ref : trace_.gpu_zones()) {
      const GpuContextInfo* context = trace_.FindGpuContext(ref.context);
      QueueKey key = KeyForContext(context);
      queues_.insert(key);
      devices_.insert(key.device);
      if (context &&
          (context->lane == Lane::kDispatch || context->lane == Lane::kStream ||
           context->lane == Lane::kUnknown || context->lane == Lane::kHost)) {
        const char* name = trace_.ZoneName(*ref.zone);
        if (context->lane != Lane::kHost || QueueOpValue(name) == 0) {
          if (!functions_.count(name)) {
            functions_[name] = static_cast<uint32_t>(functions_.size());
          }
        }
      }
    }
  }

  void WriteSchema() {
    std::string buffer;
    buffer += "{\"schema_version\":";
    buffer += std::to_string(kSchemaVersion);
    buffer +=
        ",\"record_type\":\"schema\",\"format\":\"ireeperf-jsonl\","
        "\"contract\":\"schema_versioned_interchange\",\"row_key\":"
        "\"record_type\",\"source_format\":\"tracy\",";
    buffer +=
        "\"source_version_major\":" + std::to_string(tracy::Version::Major);
    buffer +=
        ",\"source_version_minor\":" + std::to_string(tracy::Version::Minor);
    buffer +=
        ",\"time_domains\":{\"iree_host_time_ns\":\"iree_time_now monotonic "
        "nanoseconds\",\"device_tick\":\"tracy capture timeline nanoseconds "
        "(identity to host time)\",\"driver_host_cpu_timestamp_ns\":\"tracy "
        "capture timeline nanoseconds\",\"driver_host_system_timestamp\":"
        "\"unused\",\"device_tick_duration_ns\":\"duration nanoseconds\"},"
        "\"clock_fit\":{\"basis\":\"first_last_clock_correlation\","
        "\"rounding\":"
        "\"nearest_integer_nanosecond\"}}\n";
    fputs(buffer.c_str(), out_);
  }

  void WriteSession(const char* event) {
    std::string buffer;
    Begin(buffer, "session");
    Str(buffer, "event", event);
    UInt(buffer, "session_id", 1);
    UInt(buffer, "stream_id", 0);
    UInt(buffer, "event_id", 0);
    UInt(buffer, "session_status_code", 0);
    Str(buffer, "session_status", "OK");
    End(buffer);
  }

  void WriteDevicesAndQueues() {
    std::string buffer;
    for (uint32_t device : devices_) {
      uint32_t queue_count = 0;
      for (const QueueKey& key : queues_) {
        if (key.device == device) ++queue_count;
      }
      Begin(buffer, "device");
      UInt(buffer, "physical_device_ordinal", device);
      UInt(buffer, "flags", 0);
      UInt(buffer, "queue_count", queue_count);
      Bool(buffer, "physical_device_uuid_present", false);
      Null(buffer, "physical_device_uuid");
      End(buffer);
    }
    for (const QueueKey& key : queues_) {
      Begin(buffer, "queue");
      UInt(buffer, "physical_device_ordinal", key.device);
      UInt(buffer, "queue_ordinal", key.queue);
      UInt(buffer, "stream_id", StreamId(key));
      End(buffer);
    }
  }

  void WriteClockCorrelations() {
    std::string buffer;
    const int64_t samples[2] = {trace_.first_time(), trace_.last_time()};
    uint64_t sample_id = 1;
    for (uint32_t device : devices_) {
      for (int64_t t : samples) {
        Begin(buffer, "clock_correlation");
        UInt(buffer, "sample_id", sample_id++);
        // DEVICE_TICK | HOST_CPU_TIMESTAMP | HOST_TIME_BRACKET
        UInt(buffer, "flags", 1u | 2u | 8u);
        UInt(buffer, "physical_device_ordinal", device);
        Str(buffer, "device_tick_domain", "device_tick");
        UInt(buffer, "device_tick", static_cast<uint64_t>(t));
        Str(buffer, "host_cpu_timestamp_domain",
            "driver_host_cpu_timestamp_ns");
        UInt(buffer, "host_cpu_timestamp_ns", static_cast<uint64_t>(t));
        Str(buffer, "host_system_timestamp_domain",
            "driver_host_system_timestamp");
        UInt(buffer, "host_system_timestamp", 0);
        UInt(buffer, "host_system_frequency_hz", 0);
        Str(buffer, "host_time_domain", "iree_host_time_ns");
        Int(buffer, "host_time_begin_ns", t);
        Int(buffer, "host_time_end_ns", t);
        Int(buffer, "host_time_uncertainty_ns", 0);
        End(buffer);
      }
    }
  }

  void WriteExecutableFunctions() {
    if (functions_.empty()) return;
    std::string buffer;
    Begin(buffer, "executable");
    UInt(buffer, "executable_id", 1);
    UInt(buffer, "flags", 0);
    UInt(buffer, "function_count", functions_.size());
    Bool(buffer, "code_object_hash_present", false);
    Null(buffer, "code_object_hash");
    End(buffer);
    // Emit in ordinal order.
    std::vector<const std::string*> ordered(functions_.size());
    for (const auto& it : functions_) ordered[it.second] = &it.first;
    for (size_t i = 0; i < ordered.size(); ++i) {
      Begin(buffer, "executable_function");
      UInt(buffer, "executable_id", 1);
      UInt(buffer, "function_ordinal", i);
      UInt(buffer, "flags", 0);
      Str(buffer, "name", *ordered[i]);
      UInt(buffer, "constant_count", 0);
      UInt(buffer, "binding_count", 0);
      UInt(buffer, "parameter_count", 0);
      Triple(buffer, "workgroup_size");
      Bool(buffer, "function_hash_present", false);
      Null(buffer, "function_hash");
      End(buffer);
    }
  }

  void DeviceTimeBlock(std::string& buffer, int64_t start, int64_t end,
                       bool valid) {
    UInt(buffer, "duration_ticks", valid ? uint64_t(end - start) : 0);
    Bool(buffer, "valid", valid);
    Bool(buffer, "derived_time_available", valid);
    Str(buffer, "derived_time_domain", "driver_host_cpu_timestamp_ns");
    Str(buffer, "derived_time_basis", "first_last_clock_correlation");
    UInt(buffer, "clock_fit_first_sample_id", valid ? 1 : 0);
    UInt(buffer, "clock_fit_last_sample_id", valid ? 2 : 0);
    Int(buffer, "start_driver_host_cpu_time_ns", valid ? start : 0);
    Int(buffer, "end_driver_host_cpu_time_ns", valid ? end : 0);
    Str(buffer, "duration_time_domain", "device_tick_duration_ns");
    Int(buffer, "duration_ns", valid ? end - start : 0);
  }

  void WriteDispatchEvent(std::string& buffer, const GpuZoneRef& ref,
                          const QueueKey& key) {
    const tracy::GpuEvent& zone = *ref.zone;
    const char* name = trace_.ZoneName(zone);
    const bool valid = zone.GpuStart() >= 0 && zone.GpuEnd() >= zone.GpuStart();
    Begin(buffer, "dispatch_event");
    UInt(buffer, "physical_device_ordinal", key.device);
    UInt(buffer, "queue_ordinal", key.queue);
    UInt(buffer, "stream_id", StreamId(key));
    UInt(buffer, "event_id", ref.event_id + 1);
    UInt(buffer, "submission_id", 0);
    UInt(buffer, "command_buffer_id", 0);
    UInt(buffer, "command_index", UINT32_MAX);
    UInt(buffer, "executable_id", 1);
    auto fn = functions_.find(name);
    UInt(buffer, "function_ordinal",
         fn != functions_.end() ? fn->second : UINT32_MAX);
    Str(buffer, "key", name);
    UInt(buffer, "flags", 0);
    Triple(buffer, "workgroup_count");
    Triple(buffer, "workgroup_size");
    Str(buffer, "device_tick_domain", "device_tick");
    UInt(buffer, "start_tick", valid ? uint64_t(zone.GpuStart()) : 0);
    UInt(buffer, "end_tick", valid ? uint64_t(zone.GpuEnd()) : 0);
    DeviceTimeBlock(buffer, zone.GpuStart(), zone.GpuEnd(), valid);
    End(buffer);
  }

  void WriteQueueDeviceEvent(std::string& buffer, const GpuZoneRef& ref,
                             const QueueKey& key) {
    const tracy::GpuEvent& zone = *ref.zone;
    const char* name = trace_.ZoneName(zone);
    const uint32_t op = QueueOpValue(name);
    const bool valid = zone.GpuStart() >= 0 && zone.GpuEnd() >= zone.GpuStart();
    Begin(buffer, "queue_device_event");
    UInt(buffer, "physical_device_ordinal", key.device);
    UInt(buffer, "queue_ordinal", key.queue);
    UInt(buffer, "stream_id", StreamId(key));
    UInt(buffer, "event_id", ref.event_id + 1);
    UInt(buffer, "submission_id", 0);
    UInt(buffer, "command_buffer_id", 0);
    UInt(buffer, "allocation_id", 0);
    Str(buffer, "op", op ? name : "unknown");
    UInt(buffer, "type_value", op);
    UInt(buffer, "flags", 0);
    UInt(buffer, "payload_length", 0);
    UInt(buffer, "operation_count", 0);
    Str(buffer, "device_tick_domain", "device_tick");
    UInt(buffer, "start_tick", valid ? uint64_t(zone.GpuStart()) : 0);
    UInt(buffer, "end_tick", valid ? uint64_t(zone.GpuEnd()) : 0);
    DeviceTimeBlock(buffer, zone.GpuStart(), zone.GpuEnd(), valid);
    End(buffer);
  }

  void WriteHostExecutionEvent(std::string& buffer, const GpuZoneRef& ref,
                               const QueueKey& key) {
    const tracy::GpuEvent& zone = *ref.zone;
    const char* name = trace_.ZoneName(zone);
    const uint32_t op = QueueOpValue(name);
    const bool valid = zone.GpuStart() >= 0 && zone.GpuEnd() >= zone.GpuStart();
    Begin(buffer, "host_execution_event");
    UInt(buffer, "physical_device_ordinal", key.device);
    UInt(buffer, "queue_ordinal", key.queue);
    UInt(buffer, "stream_id", StreamId(key));
    UInt(buffer, "event_id", ref.event_id + 1);
    UInt(buffer, "submission_id", 0);
    UInt(buffer, "command_buffer_id", 0);
    UInt(buffer, "command_index", UINT32_MAX);
    if (op == 0) {
      auto fn = functions_.find(name);
      UInt(buffer, "executable_id", 1);
      UInt(buffer, "function_ordinal",
           fn != functions_.end() ? fn->second : UINT32_MAX);
    } else {
      UInt(buffer, "executable_id", 0);
      UInt(buffer, "function_ordinal", UINT32_MAX);
    }
    UInt(buffer, "allocation_id", 0);
    Str(buffer, "op", op ? name : "dispatch");
    if (op == 0) {
      Str(buffer, "key", name);
    } else {
      Null(buffer, "key");
    }
    UInt(buffer, "type_value", op ? op : 2);
    UInt(buffer, "flags", 0);
    UInt(buffer, "status_code", 0);
    Triple(buffer, "workgroup_count");
    Triple(buffer, "workgroup_size");
    Str(buffer, "host_time_domain", "iree_host_time_ns");
    Int(buffer, "start_host_time_ns", valid ? zone.GpuStart() : -1);
    Int(buffer, "end_host_time_ns", valid ? zone.GpuEnd() : -1);
    Int(buffer, "duration_ns", valid ? zone.GpuEnd() - zone.GpuStart() : 0);
    Bool(buffer, "valid", valid);
    UInt(buffer, "payload_length", 0);
    UInt(buffer, "tile_count", 0);
    Int(buffer, "tile_duration_sum_ns", 0);
    UInt(buffer, "operation_count", 0);
    End(buffer);
  }

  void WriteQueueEvent(std::string& buffer, const GpuZoneRef& ref,
                       const QueueKey& key) {
    const tracy::GpuEvent& zone = *ref.zone;
    const char* name = trace_.ZoneName(zone);
    const uint32_t op = QueueOpValue(name);
    const bool valid = zone.GpuStart() >= 0 && zone.GpuEnd() >= zone.GpuStart();
    Begin(buffer, "queue_event");
    UInt(buffer, "event_id", ref.event_id + 1);
    Str(buffer, "op", op ? name : "unknown");
    UInt(buffer, "type_value", op);
    UInt(buffer, "flags", 0);
    Str(buffer, "dependency_strategy", "none");
    UInt(buffer, "submission_id", 0);
    UInt(buffer, "command_buffer_id", 0);
    UInt(buffer, "allocation_id", 0);
    UInt(buffer, "physical_device_ordinal", key.device);
    UInt(buffer, "queue_ordinal", key.queue);
    UInt(buffer, "stream_id", StreamId(key));
    Int(buffer, "host_time_ns", valid ? zone.GpuStart() : 0);
    Int(buffer, "ready_host_time_ns", valid ? zone.GpuEnd() : 0);
    Str(buffer, "host_time_domain", "iree_host_time_ns");
    UInt(buffer, "wait_count", 0);
    UInt(buffer, "signal_count", 0);
    UInt(buffer, "barrier_count", 0);
    UInt(buffer, "operation_count", 0);
    UInt(buffer, "payload_length", 0);
    End(buffer);
  }

  void WriteEvents() {
    std::string buffer;
    buffer.reserve(1024);
    for (const auto& ref : trace_.gpu_zones()) {
      const GpuContextInfo* context = trace_.FindGpuContext(ref.context);
      const QueueKey key = KeyForContext(context);
      const Lane lane = context ? context->lane : Lane::kUnknown;
      const char* name = trace_.ZoneName(*ref.zone);
      switch (lane) {
        case Lane::kDispatch:
          WriteDispatchEvent(buffer, ref, key);
          break;
        case Lane::kQueue:
          WriteQueueDeviceEvent(buffer, ref, key);
          break;
        case Lane::kHost:
          WriteHostExecutionEvent(buffer, ref, key);
          break;
        case Lane::kSubmit:
          WriteQueueEvent(buffer, ref, key);
          break;
        case Lane::kStream:
        case Lane::kUnknown:
        default:
          // Stream tracing: command-buffer level zones enclose the dispatch
          // zones recorded inside them.
          if (ref.depth == 0 && ref.zone->Child() >= 0 &&
              QueueOpValue(name) == 0) {
            WriteQueueDeviceEventAs(buffer, ref, key, "execute", 3);
          } else {
            WriteDispatchEvent(buffer, ref, key);
          }
          break;
      }
    }
  }

  void WriteQueueDeviceEventAs(std::string& buffer, const GpuZoneRef& ref,
                               const QueueKey& key, const char* op,
                               uint32_t type_value) {
    const tracy::GpuEvent& zone = *ref.zone;
    const bool valid = zone.GpuStart() >= 0 && zone.GpuEnd() >= zone.GpuStart();
    Begin(buffer, "queue_device_event");
    UInt(buffer, "physical_device_ordinal", key.device);
    UInt(buffer, "queue_ordinal", key.queue);
    UInt(buffer, "stream_id", StreamId(key));
    UInt(buffer, "event_id", ref.event_id + 1);
    UInt(buffer, "submission_id", 0);
    UInt(buffer, "command_buffer_id", 0);
    UInt(buffer, "allocation_id", 0);
    Str(buffer, "op", op);
    UInt(buffer, "type_value", type_value);
    UInt(buffer, "flags", 0);
    UInt(buffer, "payload_length", 0);
    UInt(buffer, "operation_count", 0);
    Str(buffer, "device_tick_domain", "device_tick");
    UInt(buffer, "start_tick", valid ? uint64_t(zone.GpuStart()) : 0);
    UInt(buffer, "end_tick", valid ? uint64_t(zone.GpuEnd()) : 0);
    DeviceTimeBlock(buffer, zone.GpuStart(), zone.GpuEnd(), valid);
    End(buffer);
  }

  Trace& trace_;
  FILE* out_;
  uint64_t record_index_ = 0;
  std::set<uint32_t> devices_;
  std::set<QueueKey> queues_;
  std::map<std::string, uint32_t> functions_;
};

}  // namespace

int RunExport(Trace& trace, const Options& options, FILE* out) {
  (void)options;
  if (trace.gpu_zones().empty()) {
    fprintf(stderr,
            "warning: the capture has no GPU/device zones; the export will "
            "contain metadata only (CPU zones are not part of the HAL profile "
            "interchange schema)\n");
  }
  Exporter exporter(trace, out);
  exporter.Run();
  return 0;
}

}  // namespace iree_tracy_profile
