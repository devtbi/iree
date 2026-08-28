// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures what a profiler timestamp costs on this machine.
//
// Tracy reads a timestamp twice per zone and IREE emits on the order of a
// hundred host zones around each dispatch, so the cost of a single read is
// multiplied by a few hundred per dispatch. On x86 that read is one rdtsc. On
// aarch64 Linux tracy has no hardware timer path and falls back to
// clock_gettime(CLOCK_MONOTONIC_RAW) - which is itself either a vDSO call or a
// full syscall depending on the kernel - even though the hardware counter the
// vDSO reads is available to userspace directly.
//
// Build native: cc -O2 -o clock_probe clock_probe.c
// Build cross:  aarch64-linux-gnu-gcc -O2 -o clock_probe clock_probe.c

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ITERATIONS 2000000

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static uint64_t read_monotonic_raw(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static uint64_t read_monotonic(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

#if defined(__aarch64__)
static uint64_t read_hardware_counter(void) {
  uint64_t value;
  __asm__ __volatile__("mrs %0, CNTVCT_EL0" : "=r"(value) : : "memory");
  return value;
}
static uint64_t read_counter_frequency(void) {
  uint64_t value;
  __asm__ __volatile__("mrs %0, CNTFRQ_EL0" : "=r"(value));
  return value;
}
#elif defined(__x86_64__)
#include <x86intrin.h>
static uint64_t read_hardware_counter(void) { return __rdtsc(); }
#endif

static double measure(const char* label, uint64_t (*read)(void)) {
  // Warm up so the vDSO mapping and any lazy resolution are already paid for.
  volatile uint64_t sink = 0;
  for (int i = 0; i < 10000; ++i) sink += read();
  const int64_t start = now_ns();
  for (int i = 0; i < ITERATIONS; ++i) sink += read();
  const int64_t end = now_ns();
  (void)sink;
  const double per_read = (double)(end - start) / ITERATIONS;
  printf("  %-44s %7.2f ns/read\n", label, per_read);
  return per_read;
}

int main(void) {
  printf("timestamp cost (%d iterations each)\n", ITERATIONS);
  const double raw = measure(
      "clock_gettime(CLOCK_MONOTONIC_RAW)   [tracy today]", read_monotonic_raw);
  measure("clock_gettime(CLOCK_MONOTONIC)       [iree]", read_monotonic);

#if defined(__aarch64__) || defined(__x86_64__)
#if defined(__aarch64__)
  const double hardware =
      measure("mrs CNTVCT_EL0                       [proposed]",
              read_hardware_counter);
  const uint64_t frequency = read_counter_frequency();
  printf("\n  CNTFRQ_EL0 = %llu Hz -> %.1f ns per tick\n",
         (unsigned long long)frequency,
         frequency ? 1e9 / (double)frequency : 0.0);
#else
  const double hardware = measure(
      "rdtsc                                [tracy today]", read_hardware_counter);
#endif
  printf("  timestamp speedup: %.1fx\n", hardware > 0 ? raw / hardware : 0.0);
  printf(
      "\n  At ~100 host zones per dispatch (two timestamps each) that is\n"
      "  %.1f us per dispatch on the fallback versus %.1f us on the counter.\n",
      raw * 200 / 1000.0, hardware * 200 / 1000.0);
  if (raw > 100.0) {
    printf(
        "\n  NOTE: over 100ns per read suggests CLOCK_MONOTONIC_RAW is a real\n"
        "  syscall here rather than a vDSO call. That makes the gap far larger\n"
        "  than the resolution tradeoff and is worth checking first.\n");
  }
#endif
  return 0;
}
