// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal JSON Lines row writer. Rows are flat objects; every row written by
// the tool carries a "type" key first so consumers can filter with jq.

#ifndef IREE_TRACY_PROFILE_JSON_H_
#define IREE_TRACY_PROFILE_JSON_H_

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

namespace iree_tracy_profile {

class JsonRow {
 public:
  explicit JsonRow(std::string_view type) {
    buffer_.reserve(256);
    buffer_ += "{\"type\":";
    AppendString(type);
  }

  JsonRow& Str(std::string_view key, std::string_view value) {
    Key(key);
    AppendString(value);
    return *this;
  }
  JsonRow& StrOrNull(std::string_view key, const char* value) {
    Key(key);
    if (value) {
      AppendString(value);
    } else {
      buffer_ += "null";
    }
    return *this;
  }
  JsonRow& Int(std::string_view key, int64_t value) {
    Key(key);
    buffer_ += std::to_string(value);
    return *this;
  }
  JsonRow& UInt(std::string_view key, uint64_t value) {
    Key(key);
    buffer_ += std::to_string(value);
    return *this;
  }
  JsonRow& OptUInt(std::string_view key, std::optional<uint32_t> value) {
    Key(key);
    if (value) {
      buffer_ += std::to_string(*value);
    } else {
      buffer_ += "null";
    }
    return *this;
  }
  JsonRow& OptInt(std::string_view key, std::optional<int64_t> value) {
    Key(key);
    if (value) {
      buffer_ += std::to_string(*value);
    } else {
      buffer_ += "null";
    }
    return *this;
  }
  JsonRow& Double(std::string_view key, double value) {
    Key(key);
    char tmp[64];
    if (value != value) {  // NaN is not valid JSON.
      snprintf(tmp, sizeof(tmp), "null");
    } else {
      snprintf(tmp, sizeof(tmp), "%.6g", value);
    }
    buffer_ += tmp;
    return *this;
  }
  JsonRow& Bool(std::string_view key, bool value) {
    Key(key);
    buffer_ += value ? "true" : "false";
    return *this;
  }
  JsonRow& Null(std::string_view key) {
    Key(key);
    buffer_ += "null";
    return *this;
  }

  void Write(FILE* file) {
    buffer_ += "}\n";
    fputs(buffer_.c_str(), file);
  }

 private:
  void Key(std::string_view key) {
    buffer_ += ',';
    AppendString(key);
    buffer_ += ':';
  }
  void AppendString(std::string_view value) {
    buffer_ += '"';
    for (unsigned char c : value) {
      switch (c) {
        case '"':
          buffer_ += "\\\"";
          break;
        case '\\':
          buffer_ += "\\\\";
          break;
        case '\n':
          buffer_ += "\\n";
          break;
        case '\r':
          buffer_ += "\\r";
          break;
        case '\t':
          buffer_ += "\\t";
          break;
        default:
          if (c < 0x20) {
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "\\u%04x", c);
            buffer_ += tmp;
          } else {
            buffer_ += static_cast<char>(c);
          }
      }
    }
    buffer_ += '"';
  }

  std::string buffer_;
};

}  // namespace iree_tracy_profile

#endif  // IREE_TRACY_PROFILE_JSON_H_
