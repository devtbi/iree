// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-tracy-profile: inspects Tracy captures (.tracy) from the command line
// with the same command shape as iree-profile, emitting text or JSONL.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "GitRef.hpp"
#include "TracyVersion.hpp"
#include "commands.h"

namespace iree_tracy_profile {
namespace {

const char kUsage[] =
    "Inspects Tracy captures produced by IREE tracing.\n"
    "\n"
    "A .tracy capture holds everything the runtime emitted while a Tracy\n"
    "server was connected: CPU zones per thread, GPU zones per context\n"
    "(including HAL profiling records forwarded by --device_profiling_tracy),\n"
    "messages, plots, memory events, frames, and sampled callstacks. The\n"
    "commands below let you enter from the object you care about and drill\n"
    "down with JSONL.\n"
    "\n"
    "Usage:\n"
    "  iree-tracy-profile summary [--format=text|jsonl] <file.tracy>\n"
    "  iree-tracy-profile statistics [--format=text|jsonl] [--filter=glob]\n"
    "      <file.tracy>\n"
    "  iree-tracy-profile dispatch [--format=text|jsonl] [--filter=glob]\n"
    "      [--id=event_id] [--context=N] [--dispatch_events] <file.tracy>\n"
    "  iree-tracy-profile queue [--format=text|jsonl] [--filter=glob]\n"
    "      [--id=event_id] [--context=N] [--queue_events] <file.tracy>\n"
    "  iree-tracy-profile zone [--format=text|jsonl] [--filter=glob]\n"
    "      [--id=event_id] [--thread=tid] [--zone_events] <file.tracy>\n"
    "  iree-tracy-profile thread [--format=text|jsonl] [--id=tid] "
    "<file.tracy>\n"
    "  iree-tracy-profile message [--format=text|jsonl] [--filter=glob]\n"
    "      [--thread=tid] <file.tracy>\n"
    "  iree-tracy-profile plot [--format=text|jsonl] [--filter=glob]\n"
    "      [--plot_samples] <file.tracy>\n"
    "  iree-tracy-profile memory [--format=text|jsonl] [--filter=glob]\n"
    "      [--memory_events] <file.tracy>\n"
    "  iree-tracy-profile frame [--format=text|jsonl] [--filter=glob]\n"
    "      [--frame_events] <file.tracy>\n"
    "  iree-tracy-profile explain [--format=text|jsonl] [--top=N] "
    "<file.tracy>\n"
    "  iree-tracy-profile export --format=ireeperf-jsonl [--output=path|-]\n"
    "      <file.tracy>\n"
    "  iree-tracy-profile cat [--format=jsonl] <file.tracy>\n"
    "  iree-tracy-profile --agents_md\n"
    "\n"
    "Commands:\n"
    "  summary      Capture metadata, per-thread and per-GPU-context counts,\n"
    "               time span, and capture health.\n"
    "  statistics   Aggregate rows for CPU zones (by source location), GPU\n"
    "               zones (by name and context), messages, plots, memory\n"
    "               pools, and frame sets. One compact row stream.\n"
    "  dispatch     GPU zones on dispatch contexts grouped by name with\n"
    "               count/total/avg/min/max/p50/p90/p99, or individual\n"
    "               events with --dispatch_events.\n"
    "  queue        GPU zones on queue/submit contexts (execute, copy,\n"
    "               fill, alloca, ...) grouped by name or as events.\n"
    "  zone         CPU zones grouped by source location with total and\n"
    "               self time, or individual events with --zone_events.\n"
    "  thread       Threads with zone/message/sample counts and busy time.\n"
    "  message      Messages with severity, thread, and time.\n"
    "  plot         Plot series summaries, or samples with --plot_samples.\n"
    "  memory       Allocation pools with high-water and live-at-end\n"
    "               summaries, or events with --memory_events.\n"
    "  frame        Frame set timing statistics.\n"
    "  explain      Opinionated bottleneck summary: device spans, busy\n"
    "               fractions, top dispatches and CPU self time, hints.\n"
    "  export       Decoded interchange export (ireeperf-jsonl) consumable\n"
    "               by iree-profile-render and other iree-profile tooling.\n"
    "  cat          Every event as a JSONL row, for archaeology/debugging.\n"
    "\n"
    "Important flags:\n"
    "  --format=FORMAT         text/jsonl for reports; ireeperf-jsonl for\n"
    "                          export.\n"
    "  --filter=glob           Wildcard ('*', '?') over "
    "zone/dispatch/message/\n"
    "                          plot/pool names, such as '*matmul*'.\n"
    "  --id=N                  dispatch/queue/zone: event_id from a previous\n"
    "                          event row; thread: thread id.\n"
    "  --context=N             dispatch/queue: restrict to one GPU context.\n"
    "  --thread=TID            zone/message: restrict to one thread.\n"
    "  --top=N                 explain: rows per ranking (default 10).\n"
    "  --*_events              Stream individual event rows instead of\n"
    "                          aggregates (dispatch_events, queue_events,\n"
    "                          zone_events, plot_samples, memory_events,\n"
    "                          frame_events). Requires JSONL.\n"
    "  --output=path|-         Export destination, or '-' for stdout.\n"
    "  --agents_md             Print a Markdown guide optimized for "
    "AGENTS.md.\n"
    "\n"
    "JSONL contract:\n"
    "  Rows are keyed by `type`. Times are nanoseconds on Tracy's capture\n"
    "  timeline (`*_ns`), plus `since_start_ns` for the first event in the\n"
    "  capture. Aggregate rows carry `count`, `total_ns`, `avg_ns`, `min_ns`,\n"
    "  `max_ns`, `stddev_ns`, `p50_ns`, `p90_ns`, `p99_ns`.\n"
    "\n"
    "Capture examples:\n"
    "  iree-tracy-capture -o run.tracy -f &\n"
    "  TRACY_NO_EXIT=1 iree-benchmark-module --device=amdgpu \\\n"
    "      --module=model.vmfb --function=main --benchmark_min_time=20x \\\n"
    "      --device_profiling_mode=dispatch-events,device-queue-events \\\n"
    "      --device_profiling_tracy\n"
    "\n"
    "Analysis examples:\n"
    "  iree-tracy-profile summary run.tracy\n"
    "  iree-tracy-profile explain run.tracy\n"
    "  iree-tracy-profile dispatch --format=jsonl run.tracy | \\\n"
    "      jq 'select(.type==\"dispatch_group\") | {key,avg_ns,count}'\n"
    "  iree-tracy-profile dispatch --format=jsonl --dispatch_events \\\n"
    "      --filter='*matmul*' run.tracy | jq -c "
    "'select(.duration_ns>100000)'\n"
    "  iree-tracy-profile zone --format=jsonl run.tracy | \\\n"
    "      jq 'select(.type==\"zone_group\") | {key,self_total_ns}'\n"
    "  iree-tracy-profile export --format=ireeperf-jsonl --output=- run.tracy "
    "\\\n"
    "      | iree-profile-render --format=perfetto - -o run.pftrace\n"
    "\n"
    "Use `iree-tracy-profile --agents_md` for the JSONL record types and\n"
    "cross-reference recipes.\n";

void PrintUsage(FILE* file) {
  fputs("iree-tracy-profile\n", file);
  fputs(kUsage, file);
}

bool ParseUInt(const char* text, uint64_t* out) {
  if (!text || !*text) return false;
  char* end = nullptr;
  unsigned long long value = strtoull(text, &end, 10);
  if (!end || *end) return false;
  *out = value;
  return true;
}

}  // namespace
}  // namespace iree_tracy_profile

int main(int argc, char** argv) {
  using namespace iree_tracy_profile;
  if (argc < 2) {
    PrintUsage(stderr);
    return 2;
  }

  std::string command;
  std::string path;
  Options options;
  bool agents_md = false;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      PrintUsage(stdout);
      return 0;
    } else if (strcmp(arg, "--version") == 0) {
      printf("iree-tracy-profile (tracy %i.%i.%i / %s)\n",
             tracy::Version::Major, tracy::Version::Minor,
             tracy::Version::Patch, tracy::GitRef);
      return 0;
    } else if (strcmp(arg, "--agents_md") == 0) {
      agents_md = true;
    } else if (strncmp(arg, "--format=", 9) == 0) {
      const char* value = arg + 9;
      if (strcmp(value, "text") == 0) {
        options.format = Format::kText;
      } else if (strcmp(value, "jsonl") == 0) {
        options.format = Format::kJsonl;
      } else if (strcmp(value, "ireeperf-jsonl") == 0) {
        options.format = Format::kIreeperfJsonl;
      } else {
        fprintf(stderr, "unsupported --format value '%s'\n", value);
        return 2;
      }
    } else if (strncmp(arg, "--filter=", 9) == 0) {
      options.filter = arg + 9;
    } else if (strncmp(arg, "--output=", 9) == 0) {
      options.output = arg + 9;
    } else if (strncmp(arg, "--id=", 5) == 0) {
      uint64_t value = 0;
      if (!ParseUInt(arg + 5, &value)) {
        fprintf(stderr, "invalid --id value '%s'\n", arg + 5);
        return 2;
      }
      options.id = value;
    } else if (strncmp(arg, "--thread=", 9) == 0) {
      uint64_t value = 0;
      if (!ParseUInt(arg + 9, &value)) {
        fprintf(stderr, "invalid --thread value '%s'\n", arg + 9);
        return 2;
      }
      options.thread = value;
    } else if (strncmp(arg, "--context=", 10) == 0) {
      uint64_t value = 0;
      if (!ParseUInt(arg + 10, &value)) {
        fprintf(stderr, "invalid --context value '%s'\n", arg + 10);
        return 2;
      }
      options.context = value;
    } else if (strncmp(arg, "--top=", 6) == 0) {
      uint64_t value = 0;
      if (!ParseUInt(arg + 6, &value)) {
        fprintf(stderr, "invalid --top value '%s'\n", arg + 6);
        return 2;
      }
      options.top = static_cast<size_t>(value);
    } else if (strcmp(arg, "--dispatch_events") == 0 ||
               strcmp(arg, "--queue_events") == 0 ||
               strcmp(arg, "--zone_events") == 0 ||
               strcmp(arg, "--plot_samples") == 0 ||
               strcmp(arg, "--memory_events") == 0 ||
               strcmp(arg, "--frame_events") == 0 ||
               strcmp(arg, "--events") == 0) {
      options.events = true;
    } else if (arg[0] == '-' && arg[1] == '-') {
      fprintf(stderr, "unknown flag '%s'\n", arg);
      PrintUsage(stderr);
      return 2;
    } else if (command.empty()) {
      command = arg;
    } else if (path.empty()) {
      path = arg;
    } else {
      fprintf(stderr, "unexpected argument '%s'\n", arg);
      return 2;
    }
  }

  if (agents_md) {
    PrintAgentsMarkdown(stdout);
    return 0;
  }
  if (command.empty() || path.empty()) {
    PrintUsage(stderr);
    return 2;
  }
  if (options.events && options.format == Format::kText) {
    fprintf(stderr, "--*_events requires --format=jsonl\n");
    return 2;
  }
  if (command == "export") {
    if (options.format != Format::kIreeperfJsonl) {
      fprintf(stderr, "export requires --format=ireeperf-jsonl\n");
      return 2;
    }
  } else if (options.format == Format::kIreeperfJsonl) {
    fprintf(stderr, "--format=ireeperf-jsonl is only valid for export\n");
    return 2;
  }

  std::string error;
  std::unique_ptr<Trace> trace = Trace::Load(path, &error);
  if (!trace) {
    fprintf(stderr, "iree-tracy-profile: %s\n", error.c_str());
    return 1;
  }

  FILE* out = stdout;
  FILE* owned = nullptr;
  if (command == "export" && !options.output.empty() && options.output != "-") {
    owned = fopen(options.output.c_str(), "wb");
    if (!owned) {
      fprintf(stderr, "could not open output '%s'\n", options.output.c_str());
      return 1;
    }
    out = owned;
  }

  int rc = 0;
  if (command == "summary") {
    rc = RunSummary(*trace, options, out);
  } else if (command == "statistics") {
    rc = RunStatistics(*trace, options, out);
  } else if (command == "dispatch") {
    rc = RunDispatch(*trace, options, out);
  } else if (command == "queue") {
    rc = RunQueue(*trace, options, out);
  } else if (command == "zone") {
    rc = RunZone(*trace, options, out);
  } else if (command == "thread") {
    rc = RunThread(*trace, options, out);
  } else if (command == "message") {
    rc = RunMessage(*trace, options, out);
  } else if (command == "plot") {
    rc = RunPlot(*trace, options, out);
  } else if (command == "memory") {
    rc = RunMemory(*trace, options, out);
  } else if (command == "frame") {
    rc = RunFrame(*trace, options, out);
  } else if (command == "explain") {
    rc = RunExplain(*trace, options, out);
  } else if (command == "cat") {
    rc = RunCat(*trace, options, out);
  } else if (command == "export") {
    rc = RunExport(*trace, options, out);
  } else {
    fprintf(stderr, "unknown command '%s'\n", command.c_str());
    PrintUsage(stderr);
    rc = 2;
  }
  if (owned) fclose(owned);
  return rc;
}
