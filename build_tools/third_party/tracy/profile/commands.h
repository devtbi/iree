// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TRACY_PROFILE_COMMANDS_H_
#define IREE_TRACY_PROFILE_COMMANDS_H_

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "trace.h"

namespace iree_tracy_profile {

enum class Format { kText, kJsonl, kIreeperfJsonl };

struct Options {
  Format format = Format::kText;
  // Glob applied to names (zone/dispatch/message/plot/pool) per command.
  std::string filter;
  // Per-command id drilldown (event id, thread id, context id, ...).
  std::optional<uint64_t> id;
  // Stream individual events instead of only aggregate rows.
  bool events = false;
  // Restrict to one thread (zone/message commands) when set.
  std::optional<uint64_t> thread;
  // Restrict to one GPU context id when set.
  std::optional<uint64_t> context;
  // Maximum rows for ranked output (explain); 0 = unlimited.
  size_t top = 10;
  // Output path for export; "-" or empty = stdout.
  std::string output;
};

int RunSummary(Trace& trace, const Options& options, FILE* out);
int RunStatistics(Trace& trace, const Options& options, FILE* out);
int RunDispatch(Trace& trace, const Options& options, FILE* out);
int RunQueue(Trace& trace, const Options& options, FILE* out);
int RunZone(Trace& trace, const Options& options, FILE* out);
int RunThread(Trace& trace, const Options& options, FILE* out);
int RunMessage(Trace& trace, const Options& options, FILE* out);

// Sections: named spans on category tracks carrying free-form text. Used for
// structure a zone cannot express - which iteration this was, which submission
// a span belongs to - and readable even when host zones are switched off.
int RunSection(Trace& trace, const Options& options, FILE* out);
int RunPlot(Trace& trace, const Options& options, FILE* out);
int RunMemory(Trace& trace, const Options& options, FILE* out);
int RunFrame(Trace& trace, const Options& options, FILE* out);
int RunExplain(Trace& trace, const Options& options, FILE* out);
int RunCat(Trace& trace, const Options& options, FILE* out);
int RunExport(Trace& trace, const Options& options, FILE* out);

// Prints the agent-oriented markdown guide.
void PrintAgentsMarkdown(FILE* out);

}  // namespace iree_tracy_profile

#endif  // IREE_TRACY_PROFILE_COMMANDS_H_
