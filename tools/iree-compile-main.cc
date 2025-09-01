// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/tool_entry_points_api.h"

#include <cstring>
#include <cstdlib>

int main(int argc, char **argv) {
  const char *trace_file = nullptr;
  int dst = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--hal_trace_file=", 17) == 0) {
      trace_file = argv[i] + 17;
      continue;
    }
    if (std::strcmp(argv[i], "--hal_trace_file") == 0 && i + 1 < argc) {
      trace_file = argv[i + 1];
      ++i;
      continue;
    }
    argv[dst++] = argv[i];
  }
  if (trace_file) {
#if defined(_WIN32)
    _putenv_s("IREE_HAL_TRACE_FILE", trace_file);
#else
    setenv("IREE_HAL_TRACE_FILE", trace_file, 1);
#endif
  }
  return ireeCompilerRunMain(dst, argv);
}
