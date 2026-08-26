# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)

project(IREETracyServer C CXX)

set(TRACY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/tracy")

find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
# Detect package manager
#-------------------------------------------------------------------------------

message(STATUS "Checking for Tracy dependencies...")
find_program(PKGCONFIG pkg-config)
if(NOT PKGCONFIG)
  message(STATUS "Could not find pkg-config to get dependencies; you need to specify them manually or use pkg-config")
  message(STATUS "  Ubuntu/Debian: `apt install pkg-config`")
  message(STATUS "  MacOS: `brew install pkg-config`")
else()
  include(FindPkgConfig)

  # Deps slightly differ by platform but some are common.
  pkg_check_modules(TRACY_DEPS
    libzstd
  )

  if(NOT TRACY_DEPS_FOUND)
    message(STATUS "Could not find Tracy dependencies (Tracy server will not be built).")
    message(STATUS "To build Tracy, install libzstd:")
    message(STATUS "  Ubuntu/Debian: `apt install libzstd-dev`")
    message(STATUS "  MacOS: `brew install zstd`")
    return()
  endif()
endif()

#-------------------------------------------------------------------------------
# Source dependencies
# See https://github.com/wolfpld/tracy/blob/master/cmake/vendor.cmake
#-------------------------------------------------------------------------------

# Used to apply tracy's own vendored dependency patches (see below).
find_program(PATCH_EXECUTABLE patch)
if(NOT PATCH_EXECUTABLE)
  message(STATUS "Could not find `patch` (Tracy server will not be built).")
  message(STATUS "  Ubuntu/Debian: `apt install patch`")
  message(STATUS "  MacOS: ships with the Xcode command line tools")
  return()
endif()

include(FetchContent)

# NOTE: pin and option list mirror the CPMAddPackage(capstone) block in
# third_party/tracy/cmake/vendor.cmake; re-diff against it when bumping tracy.
FetchContent_Declare(
  capstone
  GIT_REPOSITORY https://github.com/capstone-engine/capstone.git
  GIT_TAG        55c20f4260100b59c9d74ad4f3004256f619e60b # 6.0.0-Alpha10
  EXCLUDE_FROM_ALL
)
set(CAPSTONE_X86_ATT_DISABLE ON CACHE BOOL "" FORCE)
set(CAPSTONE_ALPHA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_ARC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_HPPA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_LOONGARCH_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_M680X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_M68K_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_MIPS_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_MOS65XX_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_PPC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SPARC_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SYSTEMZ_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_XCORE_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_TRICORE_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_TMS320C64X_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_EVM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_WASM_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BPF_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_RISCV_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_SH_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_XTENSA_SUPPORT OFF CACHE BOOL "" FORCE)
set(CAPSTONE_BUILD_MACOS_THIN ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(capstone)

# NOTE: the patches come from tracy itself; keep them in sync with
# third_party/tracy/cmake/vendor.cmake when updating the submodule.
FetchContent_Declare(
  PPQSort
  GIT_REPOSITORY https://github.com/GabTux/PPQSort.git
  GIT_TAG        21c5a20ca3daf572caaf0a8522db32e27e28fa1d # v1.0.6
  PATCH_COMMAND
    ${CMAKE_COMMAND}
      -DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}
      -DSOURCE_DIR=<SOURCE_DIR>
      -P "${CMAKE_CURRENT_SOURCE_DIR}/apply_patches.cmake"
      --
      "${TRACY_SOURCE_DIR}/cmake/ppqsort-nodebug.patch"
      "${TRACY_SOURCE_DIR}/cmake/ppqsort-semaphore.patch"
  UPDATE_DISCONNECTED TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(PPQSort)

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

function(setup_cxx_options name)
  set_target_properties(${name}
    PROPERTIES
      CXX_STANDARD 20
  )
  target_compile_options(${name}
    PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-unused-result>
  )
  # The capture tool never displays per-zone statistics; upstream's
  # capture/CMakeLists.txt builds the server with them compiled out too.
  target_compile_definitions(${name}
    PUBLIC
      TRACY_NO_STATISTICS
  )
  target_include_directories(${name}
    PUBLIC
      ${TRACY_SOURCE_DIR}/public/common
      ${TRACY_SOURCE_DIR}/server
      ${TRACY_DEPS_INCLUDE_DIRS}
      ${capstone_SOURCE_DIR}/include/capstone
  )
  target_link_libraries(${name}
    PRIVATE
      ${TRACY_DEPS_LIBRARIES}
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
      capstone_static
      PPQSort::PPQSort
  )
  target_link_directories(${name}
    PRIVATE
      ${TRACY_DEPS_LIBRARY_DIRS}
  )
endfunction()

#-------------------------------------------------------------------------------
# Common library
#-------------------------------------------------------------------------------

file(GLOB COMMON_SRCS ${TRACY_SOURCE_DIR}/public/common/*.cpp)
add_library(IREETracyCommon
  ${COMMON_SRCS}
)
setup_cxx_options(IREETracyCommon)

#-------------------------------------------------------------------------------
# Server library
#-------------------------------------------------------------------------------

file(GLOB SERVER_SRCS ${TRACY_SOURCE_DIR}/server/*.cpp)
add_library(IREETracyServer
  ${SERVER_SRCS}
)
setup_cxx_options(IREETracyServer)
target_link_libraries(IREETracyServer
  PRIVATE
    IREETracyCommon
)

#-------------------------------------------------------------------------------
# Standalone capture server
#-------------------------------------------------------------------------------

# capture.cpp includes a generated GitRef.hpp, which upstream fills from git via
# cmake/GitRef.cmake so a development checkout can report the commit it was built
# from. The submodule is pinned to a release tag, so that answer is a constant -
# and capture.cpp prints it next to tracy::Version, which already says the same
# thing. Write the tag directly rather than depending on git at configure time;
# update it when bumping tracy.
set(_TRACY_GIT_REF "v0.14.1")
# Write via a temp + copy-if-different so a reconfigure does not bump the mtime
# and force capture.cpp to rebuild.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/GitRef.hpp.tmp"
  "#pragma once\n"
  "namespace tracy { static inline const char* GitRef = \"${_TRACY_GIT_REF}\"; }\n"
)
configure_file(
  "${CMAKE_CURRENT_BINARY_DIR}/GitRef.hpp.tmp"
  "${CMAKE_CURRENT_BINARY_DIR}/GitRef.hpp"
  COPYONLY
)

add_executable(IREETracyCaptureServer
  ${TRACY_SOURCE_DIR}/capture/src/capture.cpp
  ${TRACY_SOURCE_DIR}/capture/src/CaptureOutput.cpp
)
set_target_properties(IREETracyCaptureServer
  PROPERTIES
    OUTPUT_NAME "iree-tracy-capture"
)
setup_cxx_options(IREETracyCaptureServer)
target_include_directories(IREETracyCaptureServer
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(IREETracyCaptureServer
  PRIVATE
    IREETracyCommon
    IREETracyServer
)

#-------------------------------------------------------------------------------
# Offline capture inspection tool
#-------------------------------------------------------------------------------

add_executable(IREETracyProfile
  profile/agents_md.cc
  profile/commands.cc
  profile/convert.cc
  profile/export.cc
  profile/main.cc
  profile/trace.cc
)
set_target_properties(IREETracyProfile
  PROPERTIES
    OUTPUT_NAME "iree-tracy-profile"
)
setup_cxx_options(IREETracyProfile)
target_include_directories(IREETracyProfile
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(IREETracyProfile
  PRIVATE
    IREETracyCommon
    IREETracyServer
)

if(IREE_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME iree_tracy_profile_smoke_test
      COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/profile/test/smoke_test.py"
        "$<TARGET_FILE:IREETracyProfile>"
        "${CMAKE_CURRENT_SOURCE_DIR}/profile/test/hal_profile_sink.tracy"
    )
    # IREE_TRACY_CAPTURE_FILE needs a tracing-enabled tool to drive; tools are
    # configured before this directory so the target check is reliable.
    if(IREE_ENABLE_RUNTIME_TRACING AND TARGET iree-run-module)
      add_test(
        NAME iree_tracy_capture_file_test
        COMMAND ${Python3_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/profile/test/capture_file_test.py"
          "$<TARGET_FILE:iree-run-module>"
          "$<TARGET_FILE:IREETracyProfile>"
      )
    endif()
  endif()
endif()
