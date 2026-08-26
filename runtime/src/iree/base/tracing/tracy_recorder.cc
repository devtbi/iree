// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pulls in the same configuration tracy.cc compiles the client with, so this
// translation unit agrees with it on TRACY_ENABLE and the protocol structs.
#include "iree/base/tracing.h"
#include "iree/base/tracing/tracy_recorder.h"

#if defined(TRACY_ENABLE)

#include <atomic>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// Only the wire protocol and the transport, never the profiler API: the
// profiler's symbols are mangled by its build configuration and belong to the
// translation unit that compiles the client.
#include "common/TracyProtocol.hpp"
#include "common/TracyQueue.hpp"
#include "common/TracySocket.hpp"
#include "common/tracy_lz4.hpp"

namespace {

//===----------------------------------------------------------------------===//
// Recording file format
//===----------------------------------------------------------------------===//
//
//   magic "IREETRC" + u8 version
//   u32 welcome_length, welcome bytes
//   u8 on_demand, u64 frames, u64 current_time
//   blocks until kBlockEnd:
//     kBlockEvents : u32 length, item bytes
//     kBlockAnswer : u8 query_type, u64 key, u32 length, item bytes
//
// The event section is the client's item stream with every answer removed, so
// a replayer can send it straight through without delivering an answer the
// server never asked for. Answers are stored separately, keyed the way the
// server asks for them.

constexpr char kMagic[7] = {'I', 'R', 'E', 'E', 'T', 'R', 'C'};
constexpr uint8_t kVersion = 1;
constexpr uint8_t kBlockEnd = 0;
constexpr uint8_t kBlockEvents = 1;
constexpr uint8_t kBlockAnswer = 2;

// Largest event block written out. Blocks end on an item boundary and fit in
// one wire frame, so a replayer can send each one as-is without having to know
// how to find item boundaries itself.
constexpr size_t kMaxEventBlockSize = 192 * 1024;

//===----------------------------------------------------------------------===//
// Item classification
//===----------------------------------------------------------------------===//

using tracy::QueueType;

// Items the client only ever sends in reply to a query.
bool IsAnswerItem(QueueType type) {
  switch (type) {
    case QueueType::StringData:
    case QueueType::ThreadName:
    case QueueType::PlotName:
    case QueueType::FrameName:
    case QueueType::FiberName:
    case QueueType::ExternalName:
    case QueueType::ExternalThreadName:
    case QueueType::SourceLocation:
    case QueueType::SymbolCode:
    case QueueType::SourceCode:
    case QueueType::AckServerQueryNoop:
    case QueueType::AckSourceCodeNotAvailable:
    case QueueType::AckSymbolCodeNotAvailable:
      return true;
    default:
      return false;
  }
}

// Variable-length items that describe the item that follows them.
bool IsPayloadItem(QueueType type) {
  switch (type) {
    case QueueType::SingleStringData:
    case QueueType::SingleStringData8:
    case QueueType::SecondStringData:
    case QueueType::SecondStringData8:
    case QueueType::SourceLocationPayload:
    case QueueType::CallstackPayload:
    case QueueType::CallstackAllocPayload:
    case QueueType::FrameImageData:
    case QueueType::MemNamePayload:
      return true;
    default:
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Recorder
//===----------------------------------------------------------------------===//

class Recorder {
 public:
  bool Start(const char* path, uint16_t port);
  void Stop();

 private:
  struct QueryKey {
    uint8_t type;
    uint64_t ptr;
    bool operator==(const QueryKey& other) const {
      return type == other.type && ptr == other.ptr;
    }
  };
  struct QueryKeyHash {
    size_t operator()(const QueryKey& key) const {
      return std::hash<uint64_t>()(key.ptr) ^ (size_t)key.type * 0x9e3779b9u;
    }
  };

  void ThreadMain();
  bool Handshake();
  // Reads one frame off the socket into |decompressed_|. Returns false when the
  // connection ends.
  bool ReadFrame(const char** out_data, size_t* out_size);
  void WalkFrame(const char* data, size_t size);
  // Returns the length of the item at |data|, or 0 if it is truncated.
  size_t ItemLength(const char* data, size_t available) const;
  void Ask(uint8_t query, uint64_t ptr, uint32_t extra = 0);
  void FlushQueries();
  void CollectQueries(QueueType type, const char* data, size_t length);
  void CollectFromSourceLocation(const char* data, size_t length);
  void AppendEvents(const char* data, size_t size);
  void FlushEvents();
  bool WriteFile();

  std::string path_;
  uint16_t port_ = 0;
  tracy::Socket socket_;
  std::thread thread_;
  // Set only when the client has stopped talking to us for too long; the reads
  // are otherwise blocking so that a slow drain is never cut short.
  std::atomic<bool> hard_stop_{false};
  std::atomic<bool> finished_{false};
  std::atomic<bool> running_{false};

  // Handshake results.
  std::vector<char> welcome_;
  bool on_demand_ = false;
  uint64_t on_demand_frames_ = 0;
  uint64_t on_demand_current_time_ = 0;

  // LZ4 streaming decode state, mirroring the server's ring discipline.
  tracy::LZ4_streamDecode_t* lz4_stream_ = nullptr;
  std::vector<char> ring_;
  size_t ring_offset_ = 0;
  std::vector<char> frame_;

  // Output.
  FILE* file_ = nullptr;
  std::vector<char> events_;
  std::vector<char> pending_payload_;
  std::unordered_set<QueryKey, QueryKeyHash> asked_;
  std::vector<tracy::ServerQueryPacket> query_backlog_;
  // Source-location answers carry no key, so they are matched to the queries in
  // the order they were asked for.
  std::vector<uint64_t> source_location_fifo_;
  size_t source_location_next_ = 0;
  int64_t queries_in_flight_ = 0;
  int64_t query_budget_ = 0;
  bool terminated_ = false;
  bool failed_ = false;
  uint64_t answer_count_ = 0;
  uint64_t event_bytes_ = 0;
};

Recorder* g_recorder = nullptr;

bool Recorder::Start(const char* path, uint16_t port) {
  path_ = path;
  port_ = port;
  file_ = fopen(path, "wb");
  if (!file_) return false;
  lz4_stream_ = tracy::LZ4_createStreamDecode();
  if (!lz4_stream_) {
    fclose(file_);
    file_ = nullptr;
    return false;
  }
  ring_.resize(tracy::TargetFrameSize * 3 + 1);
  frame_.resize(tracy::LZ4Size);
  running_.store(true, std::memory_order_release);
  thread_ = std::thread([this]() { ThreadMain(); });
  return true;
}

void Recorder::Stop() {
  if (!running_.load(std::memory_order_acquire)) return;
  // The client has been asked to shut down by now, so the recorder should see
  // its Terminate and finish on its own. Wait for that, but not forever: a peer
  // that stops talking must not hold the process open.
  for (int i = 0; i < 1000 && !finished_.load(std::memory_order_acquire); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  hard_stop_.store(true, std::memory_order_release);
  if (thread_.joinable()) thread_.join();
  running_.store(false, std::memory_order_release);
  if (lz4_stream_) {
    tracy::LZ4_freeStreamDecode(lz4_stream_);
    lz4_stream_ = nullptr;
  }
}

bool Recorder::Handshake() {
  // The profiled process is the listener, so the recorder connects to it. The
  // client may not be listening yet at process start; retry briefly.
  for (int attempt = 0; attempt < 500; ++attempt) {
    if (socket_.Connect("127.0.0.1", port_)) break;
    if (hard_stop_.load(std::memory_order_acquire)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!socket_.IsValid()) return false;

  if (socket_.Send(tracy::HandshakeShibboleth,
                   tracy::HandshakeShibbolethSize) == -1) {
    return false;
  }
  uint32_t protocol_version = tracy::ProtocolVersion;
  if (socket_.Send(&protocol_version, sizeof(protocol_version)) == -1) {
    return false;
  }
  const auto should_exit = [this]() {
    return hard_stop_.load(std::memory_order_acquire);
  };
  uint8_t status = 0;
  if (!socket_.Read(&status, sizeof(status), 10, should_exit)) return false;
  if (status != tracy::HandshakeWelcome) return false;

  welcome_.resize(sizeof(tracy::WelcomeMessage));
  if (!socket_.Read(welcome_.data(), (int)welcome_.size(), 10, should_exit)) {
    return false;
  }
  tracy::WelcomeMessage welcome;
  memcpy(&welcome, welcome_.data(), sizeof(welcome));
  on_demand_ = (welcome.flags & tracy::WelcomeFlag::OnDemand) != 0;
  if (on_demand_) {
    tracy::OnDemandPayloadMessage payload;
    if (!socket_.Read(&payload, sizeof(payload), 10, should_exit)) return false;
    on_demand_frames_ = payload.frames;
    on_demand_current_time_ = payload.currentTime;
  }

  tracy::LZ4_setStreamDecode(lz4_stream_, nullptr, 0);
  // Match the server's in-flight query accounting so a burst of queries can
  // never fill the socket buffers and block the client mid-dequeue.
  query_budget_ = socket_.GetSendBufSize() / (int)tracy::ServerQueryPacketSize;
  if (query_budget_ > 8192) query_budget_ = 8192;
  query_budget_ -= 4;
  if (query_budget_ < 16) query_budget_ = 16;
  return true;
}

bool Recorder::ReadFrame(const char** out_data, size_t* out_size) {
  const auto should_exit = [this]() {
    return hard_stop_.load(std::memory_order_acquire);
  };
  tracy::lz4sz_t compressed_size = 0;
  if (!socket_.Read(&compressed_size, sizeof(compressed_size), 10,
                    should_exit)) {
    return false;
  }
  if (compressed_size > tracy::LZ4Size) return false;
  if (!socket_.Read(frame_.data(), (int)compressed_size, 10, should_exit)) {
    return false;
  }
  char* target = ring_.data() + ring_offset_;
  const int size = tracy::LZ4_decompress_safe_continue(
      lz4_stream_, frame_.data(), target, (int)compressed_size,
      tracy::TargetFrameSize);
  if (size < 0) return false;
  *out_data = target;
  *out_size = (size_t)size;
  ring_offset_ += (size_t)size;
  if (ring_offset_ > tracy::TargetFrameSize * 2) ring_offset_ = 0;
  return true;
}

size_t Recorder::ItemLength(const char* data, size_t available) const {
  const uint8_t raw_type = (uint8_t)data[0];
  if (raw_type >= (uint8_t)QueueType::NUM_TYPES) return 0;
  const QueueType type = (QueueType)raw_type;
  if (raw_type >= (uint8_t)QueueType::StringData) {
    // {u8 type, u64 ptr} then a length-prefixed blob; three types use a 32-bit
    // length because their payloads can exceed 64KB.
    if (type == QueueType::FrameImageData || type == QueueType::SymbolCode ||
        type == QueueType::SourceCode) {
      if (available < 9 + sizeof(uint32_t)) return 0;
      uint32_t length = 0;
      memcpy(&length, data + 9, sizeof(length));
      const size_t total = 9 + sizeof(uint32_t) + length;
      return total <= available ? total : 0;
    }
    if (available < 9 + sizeof(uint16_t)) return 0;
    uint16_t length = 0;
    memcpy(&length, data + 9, sizeof(length));
    const size_t total = 9 + sizeof(uint16_t) + length;
    return total <= available ? total : 0;
  }
  if (type == QueueType::SingleStringData ||
      type == QueueType::SecondStringData) {
    // Lengths below 256 use the 8-bit forms, so the 16-bit ones are biased.
    if (available < 1 + sizeof(uint16_t)) return 0;
    uint16_t length = 0;
    memcpy(&length, data + 1, sizeof(length));
    const size_t total = 1 + sizeof(uint16_t) + length + 256;
    return total <= available ? total : 0;
  }
  if (type == QueueType::SingleStringData8 ||
      type == QueueType::SecondStringData8) {
    if (available < 2) return 0;
    const size_t total = 2 + (uint8_t)data[1];
    return total <= available ? total : 0;
  }
  const size_t total = tracy::QueueDataSize[raw_type];
  return total <= available ? total : 0;
}

void Recorder::Ask(uint8_t query, uint64_t ptr, uint32_t extra) {
  const QueryKey key{query, ptr};
  if (!asked_.insert(key).second) return;
  if (query == (uint8_t)tracy::ServerQuerySourceLocation) {
    source_location_fifo_.push_back(ptr);
  }
  tracy::ServerQueryPacket packet;
  packet.type = (tracy::ServerQuery)query;
  packet.ptr = ptr;
  packet.extra = extra;
  query_backlog_.push_back(packet);
}

void Recorder::FlushQueries() {
  while (!query_backlog_.empty() && queries_in_flight_ < query_budget_) {
    const tracy::ServerQueryPacket& packet = query_backlog_.front();
    if (socket_.Send(&packet, (int)tracy::ServerQueryPacketSize) == -1) {
      failed_ = true;
      return;
    }
    ++queries_in_flight_;
    query_backlog_.erase(query_backlog_.begin());
  }
}

// Offset of a wire field within its item, including the leading type byte.
#define IREE_TRACY_FIELD_OFFSET(struct_name, field) \
  (sizeof(tracy::QueueHeader) + offsetof(tracy::struct_name, field))

// Issues the queries an event item implies. Only fields the client is willing
// to dereference are asked for; payload pointers name buffers it has freed.
void Recorder::CollectQueries(QueueType type, const char* data, size_t length) {
  const auto read_u64 = [&](size_t offset) -> uint64_t {
    uint64_t value = 0;
    if (offset + sizeof(value) <= length) {
      memcpy(&value, data + offset, sizeof(value));
    }
    return value;
  };
  const auto read_u32 = [&](size_t offset) -> uint32_t {
    uint32_t value = 0;
    if (offset + sizeof(value) <= length) {
      memcpy(&value, data + offset, sizeof(value));
    }
    return value;
  };
  const uint8_t kSourceLocation = (uint8_t)tracy::ServerQuerySourceLocation;
  const uint8_t kString = (uint8_t)tracy::ServerQueryString;
  const uint8_t kThreadString = (uint8_t)tracy::ServerQueryThreadString;
  const uint8_t kPlotName = (uint8_t)tracy::ServerQueryPlotName;
  const uint8_t kFrameName = (uint8_t)tracy::ServerQueryFrameName;

  switch (type) {
    // Zone begins name a static source location. The time field in front of it
    // is packed to 8, 4 or 2 bytes depending on the variant.
    case QueueType::ZoneBegin:
    case QueueType::ZoneBeginCallstack:
      Ask(kSourceLocation,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueZoneBegin, srcloc)));
      break;
    case QueueType::ZoneBegin32:
    case QueueType::ZoneBeginCallstack32:
      Ask(kSourceLocation,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueZoneBegin32, srcloc)));
      break;
    case QueueType::ZoneBegin16:
    case QueueType::ZoneBeginCallstack16:
      Ask(kSourceLocation,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueZoneBegin16, srcloc)));
      break;
    case QueueType::GpuZoneBegin:
    case QueueType::GpuZoneBeginCallstack:
    case QueueType::GpuZoneBeginSerial:
    case QueueType::GpuZoneBeginCallstackSerial:
      Ask(kSourceLocation,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueGpuZoneBegin, srcloc)));
      break;
    case QueueType::LockAnnounce:
      Ask(kSourceLocation,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueLockAnnounce, lckloc)));
      break;
    case QueueType::LockMark:
      Ask(kSourceLocation,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueLockMark, srcloc)));
      break;
    case QueueType::MessageLiteral:
    case QueueType::MessageLiteralCallstack: {
      // The pointer shares its field with the message metadata byte.
      const uint64_t text =
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueMessageLiteral,
                                           textAndMetadata)) >>
          8;
      if (text != 0) Ask(kString, text);
      break;
    }
    case QueueType::MessageLiteralColor:
    case QueueType::MessageLiteralColorCallstack: {
      const uint64_t text =
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueMessageColorLiteral,
                                           textAndMetadata)) >>
          8;
      if (text != 0) Ask(kString, text);
      break;
    }
    case QueueType::ThreadContext:
      Ask(kThreadString,
          read_u32(IREE_TRACY_FIELD_OFFSET(QueueThreadContext, thread)));
      break;
    case QueueType::PlotDataInt:
    case QueueType::PlotDataFloat:
    case QueueType::PlotDataDouble:
      Ask(kPlotName,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueuePlotDataBase, name)));
      break;
    case QueueType::PlotConfig:
      Ask(kPlotName, read_u64(IREE_TRACY_FIELD_OFFSET(QueuePlotConfig, name)));
      break;
    case QueueType::FrameMarkMsg:
    case QueueType::FrameMarkMsgStart:
    case QueueType::FrameMarkMsgEnd: {
      const uint64_t name =
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueFrameMark, name));
      if (name != 0) Ask(kFrameName, name);
      break;
    }
    case QueueType::CrashReport:
      Ask(kString, read_u64(IREE_TRACY_FIELD_OFFSET(QueueCrashReport, text)));
      break;
    case QueueType::ParamSetup:
      Ask(kString, read_u64(IREE_TRACY_FIELD_OFFSET(QueueParamSetup, name)));
      break;
    case QueueType::MemNamePayload:
      Ask(kString,
          read_u64(IREE_TRACY_FIELD_OFFSET(QueueMemNamePayload, name)));
      break;
    default:
      break;
  }
}

// A source-location answer names three more strings by pointer.
void Recorder::CollectFromSourceLocation(const char* data, size_t length) {
  if (length < sizeof(tracy::QueueHeader) + sizeof(tracy::QueueSourceLocation)) {
    return;
  }
  const uint8_t kString = (uint8_t)tracy::ServerQueryString;
  uint64_t name = 0, function = 0, file = 0;
  memcpy(&name, data + IREE_TRACY_FIELD_OFFSET(QueueSourceLocation, name),
         sizeof(name));
  memcpy(&function,
         data + IREE_TRACY_FIELD_OFFSET(QueueSourceLocation, function),
         sizeof(function));
  memcpy(&file, data + IREE_TRACY_FIELD_OFFSET(QueueSourceLocation, file),
         sizeof(file));
  if (name != 0) Ask(kString, name);
  if (function != 0) Ask(kString, function);
  if (file != 0) Ask(kString, file);
}

void Recorder::AppendEvents(const char* data, size_t size) {
  // Close the block before the item that would overflow it, never inside one.
  if (!events_.empty() && events_.size() + size > kMaxEventBlockSize) {
    FlushEvents();
  }
  events_.insert(events_.end(), data, data + size);
}

void Recorder::FlushEvents() {
  if (events_.empty() || !file_) return;
  const uint8_t kind = kBlockEvents;
  const uint32_t length = (uint32_t)events_.size();
  fwrite(&kind, 1, 1, file_);
  fwrite(&length, sizeof(length), 1, file_);
  fwrite(events_.data(), 1, events_.size(), file_);
  event_bytes_ += events_.size();
  events_.clear();
}

void Recorder::WalkFrame(const char* data, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const size_t length = ItemLength(data + offset, size - offset);
    if (length == 0) {
      // A frame never splits an item, so this means the stream is not what we
      // think it is; stop rather than write nonsense.
      failed_ = true;
      return;
    }
    const char* item = data + offset;
    const QueueType type = (QueueType)(uint8_t)item[0];
    offset += length;

    if (IsPayloadItem(type)) {
      // Belongs to whatever comes next; hold it until that is classified. It
      // can still name something by pointer of its own - a named memory pool
      // does - so it is inspected like any other item.
      CollectQueries(type, item, length);
      pending_payload_.insert(pending_payload_.end(), item, item + length);
      continue;
    }

    if (type == QueueType::Terminate) {
      terminated_ = true;
      continue;
    }
    if (type == QueueType::KeepAlive) continue;

    if (IsAnswerItem(type)) {
      // Answers are stored out of band so the replayer can serve them when the
      // real server asks, rather than delivering them unbidden.
      uint64_t key = 0;
      uint8_t query = 0;
      switch (type) {
        case QueueType::SourceLocation:
          query = (uint8_t)tracy::ServerQuerySourceLocation;
          if (source_location_next_ < source_location_fifo_.size()) {
            key = source_location_fifo_[source_location_next_++];
          }
          CollectFromSourceLocation(item, length);
          break;
        case QueueType::StringData:
          query = (uint8_t)tracy::ServerQueryString;
          memcpy(&key, item + 1, sizeof(key));
          break;
        case QueueType::ThreadName:
          query = (uint8_t)tracy::ServerQueryThreadString;
          memcpy(&key, item + 1, sizeof(key));
          break;
        case QueueType::PlotName:
          query = (uint8_t)tracy::ServerQueryPlotName;
          memcpy(&key, item + 1, sizeof(key));
          break;
        case QueueType::FrameName:
          query = (uint8_t)tracy::ServerQueryFrameName;
          memcpy(&key, item + 1, sizeof(key));
          break;
        case QueueType::FiberName:
          query = (uint8_t)tracy::ServerQueryFiberName;
          memcpy(&key, item + 1, sizeof(key));
          break;
        default:
          // Acks and transfers we never ask for; nothing to store.
          break;
      }
      --queries_in_flight_;
      if (queries_in_flight_ < 0) queries_in_flight_ = 0;
      if (query != 0 && file_) {
        FlushEvents();
        const uint8_t kind = kBlockAnswer;
        const uint32_t answer_length =
            (uint32_t)(pending_payload_.size() + length);
        fwrite(&kind, 1, 1, file_);
        fwrite(&query, 1, 1, file_);
        fwrite(&key, sizeof(key), 1, file_);
        fwrite(&answer_length, sizeof(answer_length), 1, file_);
        if (!pending_payload_.empty()) {
          fwrite(pending_payload_.data(), 1, pending_payload_.size(), file_);
        }
        fwrite(item, 1, length, file_);
        ++answer_count_;
      }
      pending_payload_.clear();
      continue;
    }

    // A regular event: keep it, and ask for anything it names by pointer.
    if (!pending_payload_.empty()) {
      AppendEvents(pending_payload_.data(), pending_payload_.size());
      pending_payload_.clear();
    }
    CollectQueries(type, item, length);
    AppendEvents(item, length);
  }
}

void Recorder::ThreadMain() {
  if (!Handshake()) {
    if (file_) {
      fclose(file_);
      file_ = nullptr;
    }
    finished_.store(true, std::memory_order_release);
    return;
  }
  if (!WriteFile()) {
    finished_.store(true, std::memory_order_release);
    return;
  }

  bool sent_terminate_query = false;
  for (;;) {
    const char* data = nullptr;
    size_t size = 0;
    if (!ReadFrame(&data, &size)) break;
    WalkFrame(data, size);
    if (failed_) break;
    FlushQueries();
    if (terminated_ && !sent_terminate_query && query_backlog_.empty() &&
        queries_in_flight_ <= 0) {
      // The client waits forever after its Terminate for permission to exit.
      tracy::ServerQueryPacket packet;
      packet.type = tracy::ServerQueryTerminate;
      packet.ptr = 0;
      packet.extra = 0;
      socket_.Send(&packet, (int)tracy::ServerQueryPacketSize);
      sent_terminate_query = true;
    }
  }
  if (!sent_terminate_query) {
    // Either something went wrong or the client stopped early; let it exit.
    tracy::ServerQueryPacket packet;
    packet.type = tracy::ServerQueryTerminate;
    packet.ptr = 0;
    packet.extra = 0;
    socket_.Send(&packet, (int)tracy::ServerQueryPacketSize);
  }

  FlushEvents();
  if (file_) {
    const uint8_t kind = kBlockEnd;
    fwrite(&kind, 1, 1, file_);
    fclose(file_);
    file_ = nullptr;
  }
  socket_.Close();
  finished_.store(true, std::memory_order_release);
  fprintf(stderr,
          "iree: tracy recording written to '%s' (%llu bytes of events, %llu "
          "resolved names)\n",
          path_.c_str(), (unsigned long long)event_bytes_,
          (unsigned long long)answer_count_);
}

bool Recorder::WriteFile() {
  if (!file_) return false;
  fwrite(kMagic, 1, sizeof(kMagic), file_);
  fwrite(&kVersion, 1, 1, file_);
  const uint32_t welcome_length = (uint32_t)welcome_.size();
  fwrite(&welcome_length, sizeof(welcome_length), 1, file_);
  fwrite(welcome_.data(), 1, welcome_.size(), file_);
  const uint8_t on_demand = on_demand_ ? 1 : 0;
  fwrite(&on_demand, 1, 1, file_);
  fwrite(&on_demand_frames_, sizeof(on_demand_frames_), 1, file_);
  fwrite(&on_demand_current_time_, sizeof(on_demand_current_time_), 1, file_);
  return true;
}

}  // namespace

extern "C" bool iree_tracing_recorder_start(const char* path, uint16_t port) {
  if (g_recorder) return false;
  g_recorder = new Recorder();
  if (!g_recorder->Start(path, port)) {
    delete g_recorder;
    g_recorder = nullptr;
    return false;
  }
  return true;
}

extern "C" void iree_tracing_recorder_stop(void) {
  if (!g_recorder) return;
  g_recorder->Stop();
  delete g_recorder;
  g_recorder = nullptr;
}

#else

extern "C" bool iree_tracing_recorder_start(const char* path, uint16_t port) {
  (void)path;
  (void)port;
  return false;
}

extern "C" void iree_tracing_recorder_stop(void) {}

#endif  // TRACY_ENABLE
