// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Turns a recording produced in-process by IREE_TRACY_CAPTURE_FILE into a
// normal .tracy file.
//
// A recording is the client's event stream plus the strings and source
// locations it referenced by pointer, which the recorder asked for while the
// process was still alive. Building a trace out of that still needs a real
// tracy server, so this replays the recording to one: it listens on loopback,
// lets a tracy::Worker connect, plays the client side of the protocol, and
// writes what the Worker assembled.

#ifndef IREE_TRACY_PROFILE_CONVERT_H_
#define IREE_TRACY_PROFILE_CONVERT_H_

#include <string>

namespace iree_tracy_profile {

// Converts |input_path| into a .tracy file at |output_path|. Returns false and
// fills |out_error| on failure.
bool ConvertRecording(const std::string& input_path,
                      const std::string& output_path, bool verbose,
                      std::string* out_error);

}  // namespace iree_tracy_profile

#endif  // IREE_TRACY_PROFILE_CONVERT_H_
