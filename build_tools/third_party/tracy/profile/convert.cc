// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "convert.h"

#include <cinttypes>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "TracyFileWrite.hpp"
#include "TracyWorker.hpp"
#include "TracyProtocol.hpp"
#include "TracyQueue.hpp"
#include "TracySocket.hpp"
#include "tracy_lz4.hpp"

namespace iree_tracy_profile {
namespace {

constexpr char kMagic[7] = {'I', 'R', 'E', 'E', 'T', 'R', 'C'};
constexpr uint8_t kVersion = 1;
constexpr uint8_t kBlockEnd = 0;
constexpr uint8_t kBlockEvents = 1;
constexpr uint8_t kBlockAnswer = 2;

// First loopback port tried for the replay connection. Deliberately not 8086:
// a profiler listening for live processes should not pick this up.
constexpr uint16_t kFirstPort = 8710;
constexpr int kPortAttempts = 64;

struct Recording {
  std::vector<char> welcome;
  bool on_demand = false;
  uint64_t on_demand_frames = 0;
  uint64_t on_demand_current_time = 0;
  std::vector<std::vector<char>> event_blocks;
  // (query type, key) -> the exact item bytes the client replied with.
  std::map<std::pair<uint8_t, uint64_t>, std::vector<char>> answers;
};

class Reader {
 public:
  explicit Reader(FILE* file) : file_(file) {}
  bool Bytes(void* out, size_t size) {
    return fread(out, 1, size, file_) == size;
  }
  template <typename T>
  bool Value(T* out) {
    return Bytes(out, sizeof(T));
  }

 private:
  FILE* file_;
};

bool LoadRecording(const std::string& path, Recording* out,
                   std::string* out_error) {
  FILE* file = fopen(path.c_str(), "rb");
  if (!file) {
    *out_error = "could not open recording '" + path + "'";
    return false;
  }
  Reader reader(file);
  char magic[sizeof(kMagic)];
  uint8_t version = 0;
  if (!reader.Bytes(magic, sizeof(magic)) || !reader.Value(&version) ||
      memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    *out_error = "'" + path + "' is not a tracy recording";
    fclose(file);
    return false;
  }
  if (version != kVersion) {
    *out_error = "recording version " + std::to_string(version) +
                 " is not supported by this build";
    fclose(file);
    return false;
  }
  uint32_t welcome_length = 0;
  if (!reader.Value(&welcome_length) ||
      welcome_length != sizeof(tracy::WelcomeMessage)) {
    *out_error = "recording has a malformed session header";
    fclose(file);
    return false;
  }
  out->welcome.resize(welcome_length);
  uint8_t on_demand = 0;
  if (!reader.Bytes(out->welcome.data(), welcome_length) ||
      !reader.Value(&on_demand) || !reader.Value(&out->on_demand_frames) ||
      !reader.Value(&out->on_demand_current_time)) {
    *out_error = "recording has a truncated session header";
    fclose(file);
    return false;
  }
  out->on_demand = on_demand != 0;

  for (;;) {
    uint8_t kind = 0;
    if (!reader.Value(&kind)) {
      // A recording from a process that died mid-run still replays as far as
      // it got, which is usually the interesting part.
      break;
    }
    if (kind == kBlockEnd) break;
    if (kind == kBlockEvents) {
      uint32_t length = 0;
      if (!reader.Value(&length)) break;
      std::vector<char> block(length);
      if (!reader.Bytes(block.data(), length)) break;
      out->event_blocks.push_back(std::move(block));
    } else if (kind == kBlockAnswer) {
      uint8_t query = 0;
      uint64_t key = 0;
      uint32_t length = 0;
      if (!reader.Value(&query) || !reader.Value(&key) ||
          !reader.Value(&length)) {
        break;
      }
      std::vector<char> block(length);
      if (!reader.Bytes(block.data(), length)) break;
      out->answers[{query, key}] = std::move(block);
    } else {
      *out_error = "recording has an unrecognized block";
      fclose(file);
      return false;
    }
  }
  fclose(file);
  return true;
}

// Plays the client side of the protocol from a recording.
class Replayer {
 public:
  Replayer(const Recording& recording, tracy::Socket* socket)
      : recording_(recording), socket_(socket) {
    stream_ = tracy::LZ4_createStream();
    ring_.resize(tracy::TargetFrameSize * 3);
    compressed_.resize(tracy::LZ4Size);
  }
  ~Replayer() {
    if (stream_) tracy::LZ4_freeStream(stream_);
  }

  bool Handshake(std::string* out_error);
  bool SendEvents(std::string* out_error);
  bool ServeQueries(std::string* out_error);

 private:
  bool SendFrame(const char* data, size_t size);
  bool SendItems(const std::vector<char>& items) {
    return SendFrame(items.data(), items.size());
  }
  void Answer(uint8_t query, uint64_t ptr, uint32_t extra);
  void Synthesize(uint8_t query, uint64_t ptr, uint32_t extra,
                  std::vector<char>* out);

  const Recording& recording_;
  tracy::Socket* socket_;
  tracy::LZ4_stream_t* stream_ = nullptr;
  std::vector<char> ring_;
  size_t ring_offset_ = 0;
  std::vector<char> compressed_;
  std::vector<char> outbox_;
  // Source-location answers are matched by the order they were asked for.
  std::vector<uint64_t> source_location_order_;
  size_t source_location_next_ = 0;
  uint64_t synthesized_ = 0;

 public:
  uint64_t synthesized() const { return synthesized_; }
};

bool Replayer::Handshake(std::string* out_error) {
  char shibboleth[tracy::HandshakeShibbolethSize];
  if (!socket_->ReadRaw(shibboleth, sizeof(shibboleth), 10000) ||
      memcmp(shibboleth, tracy::HandshakeShibboleth, sizeof(shibboleth)) != 0) {
    *out_error = "the tracy server did not identify itself";
    return false;
  }
  uint32_t protocol_version = 0;
  if (!socket_->ReadRaw(&protocol_version, sizeof(protocol_version), 10000)) {
    *out_error = "the tracy server did not send a protocol version";
    return false;
  }
  if (protocol_version != tracy::ProtocolVersion) {
    const uint8_t mismatch = tracy::HandshakeProtocolMismatch;
    socket_->Send(&mismatch, 1);
    *out_error = "tracy protocol version mismatch";
    return false;
  }
  const uint8_t welcome = tracy::HandshakeWelcome;
  if (socket_->Send(&welcome, 1) == -1) return false;

  std::vector<char> message = recording_.welcome;
  tracy::WelcomeMessage header;
  memcpy(&header, message.data(), sizeof(header));
  // The process is gone, so nothing can answer a request for machine code or
  // source text; clearing the flag stops the server from asking.
  header.flags &= (uint8_t)~tracy::WelcomeFlag::CodeTransfer;
  header.exectime = 0;
  memcpy(message.data(), &header, sizeof(header));
  if (socket_->Send(message.data(), (int)message.size()) == -1) return false;

  if (recording_.on_demand) {
    tracy::OnDemandPayloadMessage payload;
    payload.frames = recording_.on_demand_frames;
    payload.currentTime = recording_.on_demand_current_time;
    if (socket_->Send(&payload, sizeof(payload)) == -1) return false;
  }
  tracy::LZ4_resetStream(stream_);
  return true;
}

bool Replayer::SendFrame(const char* data, size_t size) {
  if (size == 0) return true;
  if (size > tracy::TargetFrameSize) return false;
  // The compressor references its previous input, so frames must be staged in
  // a ring the same way the client stages them.
  char* staged = ring_.data() + ring_offset_;
  memcpy(staged, data, size);
  const int compressed_size = tracy::LZ4_compress_fast_continue(
      stream_, staged, compressed_.data() + sizeof(tracy::lz4sz_t), (int)size,
      (int)(compressed_.size() - sizeof(tracy::lz4sz_t)), 1);
  if (compressed_size <= 0) return false;
  ring_offset_ += size;
  if (ring_offset_ > tracy::TargetFrameSize * 2) ring_offset_ = 0;
  const tracy::lz4sz_t wire_size = (tracy::lz4sz_t)compressed_size;
  memcpy(compressed_.data(), &wire_size, sizeof(wire_size));
  return socket_->Send(compressed_.data(),
                       (int)(sizeof(wire_size) + compressed_size)) != -1;
}

bool Replayer::SendEvents(std::string* out_error) {
  for (const std::vector<char>& block : recording_.event_blocks) {
    if (!SendItems(block)) {
      *out_error = "failed to replay the event stream";
      return false;
    }
  }
  const char terminate = (char)tracy::QueueType::Terminate;
  if (!SendFrame(&terminate, 1)) {
    *out_error = "failed to end the replayed session";
    return false;
  }
  return true;
}

// Builds a reply for a query the recording has no answer to. The server treats
// a missing answer as a reason to wait forever, so every query gets something.
void Replayer::Synthesize(uint8_t query, uint64_t ptr, uint32_t extra,
                          std::vector<char>* out) {
  ++synthesized_;
  const auto put = [&](const void* data, size_t size) {
    const char* bytes = (const char*)data;
    out->insert(out->end(), bytes, bytes + size);
  };
  const auto put_string_item = [&](tracy::QueueType type, uint64_t key,
                                   const char* text) {
    const uint8_t type_byte = (uint8_t)type;
    const uint16_t length = (uint16_t)strlen(text);
    put(&type_byte, 1);
    put(&key, sizeof(key));
    put(&length, sizeof(length));
    put(text, length);
  };
  switch ((tracy::ServerQuery)query) {
    case tracy::ServerQueryString:
      put_string_item(tracy::QueueType::StringData, ptr, "???");
      break;
    case tracy::ServerQueryThreadString:
      put_string_item(tracy::QueueType::ThreadName, ptr, "???");
      break;
    case tracy::ServerQueryPlotName:
      put_string_item(tracy::QueueType::PlotName, ptr, "???");
      break;
    case tracy::ServerQueryFrameName:
      put_string_item(tracy::QueueType::FrameName, ptr, "???");
      break;
    case tracy::ServerQueryFiberName:
      put_string_item(tracy::QueueType::FiberName, ptr, "???");
      break;
    case tracy::ServerQuerySourceLocation: {
      // Zero name/function/file pointers stop the server asking for three more
      // strings it will never get an answer for.
      const uint8_t type_byte = (uint8_t)tracy::QueueType::SourceLocation;
      tracy::QueueSourceLocation body;
      memset(&body, 0, sizeof(body));
      put(&type_byte, 1);
      put(&body, sizeof(body));
      break;
    }
    case tracy::ServerQueryCallstackFrame: {
      // An empty image name, one frame, then "???" for function and file. A
      // zero symbol address stops the server asking about the symbol.
      const uint8_t single = (uint8_t)tracy::QueueType::SingleStringData8;
      const uint8_t second = (uint8_t)tracy::QueueType::SecondStringData8;
      const uint8_t zero_length = 0;
      put(&single, 1);
      put(&zero_length, 1);
      const uint8_t size_type = (uint8_t)tracy::QueueType::CallstackFrameSize;
      const uint8_t frame_count = 1;
      put(&size_type, 1);
      put(&ptr, sizeof(ptr));
      put(&frame_count, 1);
      const uint8_t three = 3;
      put(&single, 1);
      put(&three, 1);
      put("???", 3);
      put(&second, 1);
      put(&three, 1);
      put("???", 3);
      const uint8_t frame_type = (uint8_t)tracy::QueueType::CallstackFrame;
      const uint32_t line = 0;
      const uint64_t symbol = 0;
      const uint32_t symbol_length = 0;
      put(&frame_type, 1);
      put(&line, sizeof(line));
      put(&symbol, sizeof(symbol));
      put(&symbol_length, sizeof(symbol_length));
      break;
    }
    case tracy::ServerQuerySymbol: {
      const uint8_t single = (uint8_t)tracy::QueueType::SingleStringData8;
      const uint8_t three = 3;
      put(&single, 1);
      put(&three, 1);
      put("???", 3);
      const uint8_t type_byte = (uint8_t)tracy::QueueType::SymbolInformation;
      const uint32_t line = 0;
      put(&type_byte, 1);
      put(&line, sizeof(line));
      put(&ptr, sizeof(ptr));
      break;
    }
    case tracy::ServerQuerySymbolCode: {
      const uint8_t ack = (uint8_t)tracy::QueueType::AckSymbolCodeNotAvailable;
      put(&ack, 1);
      break;
    }
    case tracy::ServerQuerySourceCode: {
      const uint8_t ack = (uint8_t)tracy::QueueType::AckSourceCodeNotAvailable;
      const uint32_t id = (uint32_t)ptr;
      put(&ack, 1);
      put(&id, sizeof(id));
      break;
    }
    case tracy::ServerQueryExternalName: {
      put_string_item(tracy::QueueType::ExternalName, ptr, "???");
      put_string_item(tracy::QueueType::ExternalThreadName, ptr, "???");
      break;
    }
    default: {
      const uint8_t ack = (uint8_t)tracy::QueueType::AckServerQueryNoop;
      put(&ack, 1);
      break;
    }
  }
  (void)extra;
}

void Replayer::Answer(uint8_t query, uint64_t ptr, uint32_t extra) {
  auto it = recording_.answers.find({query, ptr});
  if (it != recording_.answers.end()) {
    outbox_.insert(outbox_.end(), it->second.begin(), it->second.end());
    return;
  }
  Synthesize(query, ptr, extra, &outbox_);
}

bool Replayer::ServeQueries(std::string* out_error) {
  for (;;) {
    tracy::ServerQueryPacket packet;
    if (!socket_->Read(&packet, (int)sizeof(packet), 100)) {
      if (socket_->IsValid()) continue;
      // The server closed: it has everything it is going to get.
      return true;
    }
    if (packet.type == tracy::ServerQueryTerminate ||
        packet.type == tracy::ServerQueryDisconnect) {
      return true;
    }
    outbox_.clear();
    Answer((uint8_t)packet.type, packet.ptr, packet.extra);
    // Drain any queries that arrived while we were building this reply so a
    // burst becomes one frame instead of one frame each.
    while (socket_->HasData() && outbox_.size() < tracy::TargetFrameSize / 2) {
      tracy::ServerQueryPacket next;
      if (!socket_->Read(&next, (int)sizeof(next), 10)) break;
      if (next.type == tracy::ServerQueryTerminate ||
          next.type == tracy::ServerQueryDisconnect) {
        if (!outbox_.empty()) SendFrame(outbox_.data(), outbox_.size());
        return true;
      }
      Answer((uint8_t)next.type, next.ptr, next.extra);
    }
    if (!outbox_.empty() && !SendFrame(outbox_.data(), outbox_.size())) {
      *out_error = "failed to answer a tracy server query";
      return false;
    }
  }
}

}  // namespace

bool ConvertRecording(const std::string& input_path,
                      const std::string& output_path, bool verbose,
                      std::string* out_error) {
  Recording recording;
  if (!LoadRecording(input_path, &recording, out_error)) return false;
  if (recording.event_blocks.empty()) {
    *out_error = "recording contains no events";
    return false;
  }

  tracy::ListenSocket listen;
  uint16_t port = 0;
  for (int i = 0; i < kPortAttempts; ++i) {
    if (listen.Listen((uint16_t)(kFirstPort + i), 4)) {
      port = (uint16_t)(kFirstPort + i);
      break;
    }
  }
  if (port == 0) {
    *out_error = "could not open a loopback port to replay the recording";
    return false;
  }

  if (verbose) {
    printf("replaying %zu event blocks and %zu resolved names\n",
           recording.event_blocks.size(), recording.answers.size());
    fflush(stdout);
  }

  // The server connects to us, so it has to exist before we can accept.
  tracy::Worker worker("127.0.0.1", port, /*memoryLimit=*/-1);
  std::unique_ptr<tracy::Socket> socket(listen.Accept());
  if (!socket) {
    *out_error = "the tracy server never connected";
    return false;
  }

  Replayer replayer(recording, socket.get());
  if (!replayer.Handshake(out_error)) return false;
  if (!replayer.SendEvents(out_error)) return false;
  if (!replayer.ServeQueries(out_error)) return false;
  socket->Close();

  for (int i = 0; i < 600 && worker.IsConnected(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (worker.IsConnected()) {
    worker.Disconnect();
    while (worker.IsConnected()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  if (worker.GetFailureType() != tracy::Worker::Failure::None) {
    *out_error = std::string("the replayed stream was rejected: ") +
                 tracy::Worker::GetFailureString(worker.GetFailureType());
    return false;
  }

  auto file = std::unique_ptr<tracy::FileWrite>(tracy::FileWrite::Open(
      output_path.c_str(), tracy::FileCompression::Zstd, 3, 4));
  if (!file) {
    *out_error = "could not open '" + output_path + "' for writing";
    return false;
  }
  worker.Write(*file, false);
  file->Finish();
  if (verbose) {
    printf("wrote %s: %llu zones, %llu names synthesized\n",
           output_path.c_str(), (unsigned long long)worker.GetZoneCount(),
           (unsigned long long)replayer.synthesized());
  }
  return true;
}

}  // namespace iree_tracy_profile
