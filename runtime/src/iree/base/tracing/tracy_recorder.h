// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// In-process recording of the tracy event stream to a file.
//
// The tracy client only ever talks to a server, but the server it talks to does
// not have to be the profiler: this recorder is a thread inside the profiled
// process that connects to the client over loopback, speaks enough of the
// server side of the protocol to make the client hand over everything it has,
// and writes the result to a file. Nothing is spawned and no profiler needs to
// be attached or even installed.
//
// The client references strings and source locations by pointer and sends the
// bytes behind them only when asked, so the recording is not simply the bytes
// off the socket: the recorder asks for each one as it sees it referenced and
// stores the answers alongside the event stream. What comes out is therefore
// self-contained but is not a .tracy file - it still has to be replayed into a
// real tracy server, which `iree-tracy-profile convert` does offline.

#ifndef IREE_BASE_TRACING_TRACY_RECORDER_H_
#define IREE_BASE_TRACING_TRACY_RECORDER_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Starts recording this process's tracy stream to |path|.
//
// Must be called before the events of interest are produced; on a client built
// without TRACY_ON_DEMAND everything since process start is queued in memory
// and delivered when the recorder connects, and with it events produced while
// nothing is connected are dropped. Returns false if the recorder could not
// start, in which case the process runs untouched.
bool iree_tracing_recorder_start(const char* path, uint16_t port);

// Asks the client to shut down, waits for it to hand over the tail of the
// stream, answers whatever it still needs to exit, and writes the recording.
// Safe to call when the recorder was never started.
void iree_tracing_recorder_stop(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_TRACING_TRACY_RECORDER_H_
