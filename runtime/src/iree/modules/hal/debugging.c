// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/debugging.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Debug Sink
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_null(void) {
  iree_hal_module_debug_sink_t sink = {0};
  return sink;
}

#if IREE_FILE_IO_ENABLE

typedef struct iree_hal_module_stdio_state_t {
  FILE* file;
  iree_hal_module_debug_trace_options_t options;
  iree_host_size_t dispatch_count;
  bool pytorch_header_emitted;
  bool close_file;
} iree_hal_module_stdio_state_t;

static void iree_hal_module_stdio_release(void* user_data) {
  iree_hal_module_stdio_state_t* state =
      (iree_hal_module_stdio_state_t*)user_data;
  if (state->close_file && state->file) {
    fclose(state->file);
  }
  iree_allocator_free(iree_allocator_system(), state);
}

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
static const char* iree_hal_pytorch_dtype_string(
    iree_hal_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return "float32";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return "float64";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return "float16";
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      return "bfloat16";
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      return "int8";
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      return "int16";
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return "int32";
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return "int64";
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return "uint8";
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return "uint16";
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return "uint32";
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return "uint64";
    default:
      return "uint8";
  }
}

static uint64_t iree_hal_fnv1a_u64(const uint8_t* data, iree_host_size_t len) {
  uint64_t hash = 1469598103934665603ULL;
  for (iree_host_size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

static void iree_hal_sanitize_identifier(iree_string_view_t src, char* dst,
                                         size_t dst_size) {
  size_t j = 0;
  for (size_t i = 0; i < src.size && j < dst_size - 1; ++i) {
    unsigned char c = (unsigned char)src.data[i];
    if (isalnum(c)) {
      dst[j++] = (char)c;
    } else {
      dst[j++] = '_';
    }
  }
  if (j == 0) {
    dst[j++] = '_';
  }
  if (isdigit(dst[0])) {
    // Prefix with underscore if the first character is a digit.
    memmove(dst + 1, dst, j);
    dst[0] = '_';
    ++j;
  }
  dst[j] = '\0';
}

static bool iree_hal_dispatch_matches(iree_string_view_t key,
                                      iree_string_view_t filter) {
  if (iree_string_view_is_empty(filter)) return true;
  iree_string_view_t remaining = filter;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t item;
    iree_string_view_split(remaining, ',', &item, &remaining);
    item = iree_string_view_trim(item);
    if (iree_string_view_equal(key, item)) return true;
  }
  return false;
}

static iree_status_t iree_hal_module_buffer_view_trace_stdio(
    void* user_data, iree_string_view_t key, iree_host_size_t buffer_view_count,
    iree_hal_buffer_view_t** buffer_views, iree_allocator_t host_allocator) {
  iree_hal_module_stdio_state_t* state =
      (iree_hal_module_stdio_state_t*)user_data;
  FILE* file = state->file;

  uint64_t dispatch_index = state->dispatch_count++;
  if (!iree_hal_dispatch_matches(key, state->options.dispatch_filter)) {
    return iree_ok_status();
  }
  int32_t percent = state->options.dispatch_sample_percent;
  if (percent >= 0 && percent < 100) {
    if ((dispatch_index % 100) >= (uint64_t)percent) {
      return iree_ok_status();
    }
  }

  fprintf(file, "# === %.*s ===\n", (int)key.size, key.data);
  char name_buffer[64];
  iree_hal_sanitize_identifier(key, name_buffer, sizeof(name_buffer));
  for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
    iree_hal_buffer_view_t* buffer_view = buffer_views[i];

    if (state->options.format == IREE_HAL_BUFFER_ELEMENTS_FORMAT_PYTORCH) {
      iree_hal_buffer_mapping_t buffer_mapping = {{0}};
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          iree_hal_buffer_view_buffer(buffer_view),
          IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
          IREE_HAL_WHOLE_BUFFER, &buffer_mapping));
      iree_host_size_t value_length = 0;
      iree_status_t status = iree_hal_format_buffer_elements(
          iree_make_const_byte_span(buffer_mapping.contents.data,
                                    buffer_mapping.contents.data_length),
          iree_hal_buffer_view_shape_rank(buffer_view),
          iree_hal_buffer_view_shape_dims(buffer_view),
          iree_hal_buffer_view_element_type(buffer_view),
          state->options.max_element_count, state->options.max_depth,
          IREE_HAL_BUFFER_ELEMENTS_FORMAT_PYTORCH, 0, NULL, &value_length);
      if (!iree_status_is_out_of_range(status)) {
        return status;
      }
      ++value_length;
      char* value_buffer = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, value_length,
                                                 (void**)&value_buffer));
      status = iree_hal_format_buffer_elements(
          iree_make_const_byte_span(buffer_mapping.contents.data,
                                    buffer_mapping.contents.data_length),
          iree_hal_buffer_view_shape_rank(buffer_view),
          iree_hal_buffer_view_shape_dims(buffer_view),
          iree_hal_buffer_view_element_type(buffer_view),
          state->options.max_element_count, state->options.max_depth,
          IREE_HAL_BUFFER_ELEMENTS_FORMAT_PYTORCH, value_length, value_buffer,
          &value_length);
      status = iree_status_join(status,
                                iree_hal_buffer_unmap_range(&buffer_mapping));
      if (iree_status_is_ok(status)) {
        const char* dtype = iree_hal_pytorch_dtype_string(
            iree_hal_buffer_view_element_type(buffer_view));
        if (!state->pytorch_header_emitted) {
          fprintf(file, "import torch\n");
          state->pytorch_header_emitted = true;
        }
        uint64_t guid = (dispatch_index << 16) | i;
        uint64_t hash = iree_hal_fnv1a_u64(buffer_mapping.contents.data,
                                           buffer_mapping.contents.data_length);
        fprintf(file, "%s_%08" PRIx64 " = torch.tensor(%s, dtype=torch.%s)\n",
                name_buffer, guid, value_buffer, dtype);
        fprintf(file, "# hash=0x%016" PRIx64 "\n", hash);
      }
      iree_allocator_free(host_allocator, value_buffer);
      IREE_RETURN_IF_ERROR(status);
    } else {
      // Query total length (excluding NUL terminator).
      iree_host_size_t result_length = 0;
      iree_status_t status = iree_hal_buffer_view_format_options(
          buffer_view, state->options.max_element_count,
          state->options.max_depth, IREE_HAL_BUFFER_ELEMENTS_FORMAT_IREE, 0,
          NULL, &result_length);
      if (!iree_status_is_out_of_range(status)) {
        return status;
      }
      ++result_length;
      char* result_str = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, result_length,
                                                 (void**)&result_str));
      status = iree_hal_buffer_view_format_options(
          buffer_view, state->options.max_element_count,
          state->options.max_depth, IREE_HAL_BUFFER_ELEMENTS_FORMAT_IREE,
          result_length, result_str, &result_length);
      if (iree_status_is_ok(status)) {
        fprintf(file, "%.*s\n", (int)result_length, result_str);
      }
      iree_allocator_free(host_allocator, result_str);
      IREE_RETURN_IF_ERROR(status);
    }
  }
  fprintf(file, "\n");
  return iree_ok_status();
}
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

static iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_stdio_options_internal(
    FILE* file, iree_hal_module_debug_trace_options_t options,
    bool close_file) {
  iree_hal_module_debug_sink_t sink = {0};
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
  iree_hal_module_stdio_state_t* state = NULL;
  if (iree_allocator_malloc(iree_allocator_system(), sizeof(*state),
                            (void**)&state) == iree_ok_status()) {
    state->file = file;
    state->options = options;
    state->dispatch_count = 0;
    state->pytorch_header_emitted = false;
    state->close_file = close_file;
    sink.buffer_view_trace.fn = iree_hal_module_buffer_view_trace_stdio;
    sink.buffer_view_trace.user_data = state;
    sink.release.fn = iree_hal_module_stdio_release;
    sink.release.user_data = state;
  }
#else
  (void)file;
  (void)options;
  (void)close_file;
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  return sink;
}

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_stdio_options(
    FILE* file, iree_hal_module_debug_trace_options_t options) {
  return iree_hal_module_debug_sink_stdio_options_internal(
      file, options,
      /*close_file=*/false);
}

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_owned_file_options(
    FILE* file, iree_hal_module_debug_trace_options_t options) {
  return iree_hal_module_debug_sink_stdio_options_internal(file, options,
                                                           /*close_file=*/true);
}

IREE_API_EXPORT iree_hal_module_debug_sink_t
iree_hal_module_debug_sink_stdio(FILE* file) {
  iree_hal_module_debug_trace_options_t options = {
      .format = IREE_HAL_BUFFER_ELEMENTS_FORMAT_IREE,
      .max_element_count = IREE_HOST_SIZE_MAX,
      .max_depth = IREE_HOST_SIZE_MAX,
      .dispatch_filter = iree_string_view_empty(),
      .dispatch_sample_percent = 100,
  };
  return iree_hal_module_debug_sink_stdio_options(file, options);
}

#endif  // IREE_FILE_IO_ENABLE
