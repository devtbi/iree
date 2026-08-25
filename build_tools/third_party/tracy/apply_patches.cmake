# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Idempotently applies patch files to a source tree.
#
# FetchContent may re-run its patch step (for example after its stamps are
# invalidated), so a plain `patch` invocation would fail the second time with
# "Reversed (or previously applied) patch detected". A reverse dry-run tells an
# already-applied patch apart from one that genuinely does not apply, so a stale
# patch after a submodule bump still fails loudly.
#
# Every invocation passes -f: without it GNU patch asks questions on the
# controlling tty ("Unreversed patch detected! Ignore -R? [n]") and a configure
# run from a terminal hangs on a prompt hidden by OUTPUT_QUIET.
#
# Usage:
#   cmake -DPATCH_EXECUTABLE=... -DSOURCE_DIR=... -P apply_patches.cmake
#         -- <patch> [<patch>...]
#
# NOTE: the patches are passed after `--` rather than as a `;`-separated cache
# variable because ExternalProject splits list arguments in PATCH_COMMAND.

set(_patches "")
set(_collecting OFF)
math(EXPR _last_arg "${CMAKE_ARGC} - 1")
foreach(_i RANGE 0 ${_last_arg})
  if(_collecting)
    list(APPEND _patches "${CMAKE_ARGV${_i}}")
  elseif("${CMAKE_ARGV${_i}}" STREQUAL "--")
    set(_collecting ON)
  endif()
endforeach()

if(NOT _patches)
  message(FATAL_ERROR "No patches passed after `--`")
endif()

set(_patch_flags -p1 -f --no-backup-if-mismatch)

foreach(_patch IN LISTS _patches)
  if(NOT EXISTS "${_patch}")
    message(FATAL_ERROR "Patch file not found: ${_patch}")
  endif()

  # Already applied? Then there is nothing to do.
  execute_process(
    COMMAND "${PATCH_EXECUTABLE}" ${_patch_flags} -R --dry-run -i "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _reverse_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(_reverse_result EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${_patch}")
    continue()
  endif()

  # Check the whole patch applies before touching the tree so a stale hunk
  # cannot leave the FetchContent source half-patched (and wedged, since
  # neither the reverse probe nor the forward apply would then succeed).
  execute_process(
    COMMAND "${PATCH_EXECUTABLE}" ${_patch_flags} --dry-run -i "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _dry_run_result
    OUTPUT_VARIABLE _dry_run_output
    ERROR_VARIABLE _dry_run_output
  )
  if(NOT _dry_run_result EQUAL 0)
    message(FATAL_ERROR
      "Patch ${_patch} does not apply to ${SOURCE_DIR}; it likely needs "
      "updating after a tracy submodule bump:\n${_dry_run_output}")
  endif()

  execute_process(
    COMMAND "${PATCH_EXECUTABLE}" ${_patch_flags} -i "${_patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _result
  )
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Failed to apply patch ${_patch} in ${SOURCE_DIR}")
  endif()
endforeach()
