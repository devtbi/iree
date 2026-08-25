// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>

#include "commands.h"

namespace iree_tracy_profile {

int RunExport(Trace& trace, const Options& options, FILE* out) {
  (void)trace;
  (void)options;
  (void)out;
  fprintf(stderr,
          "export --format=ireeperf-jsonl is not implemented yet in this "
          "build\n");
  return 2;
}

}  // namespace iree_tracy_profile
