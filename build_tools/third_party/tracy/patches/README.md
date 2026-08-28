# Local patches to the vendored tracy

`third_party/tracy` is a pinned submodule, so changes to it cannot be committed
from this repository. Patches that have not landed upstream yet live here.

Apply one with:

```shell
git -C third_party/tracy apply \
    ../../build_tools/third_party/tracy/patches/<name>.patch
```

and drop it again with `git -C third_party/tracy checkout .`.

## 0001-enable-the-arm64-generic-timer-on-linux

Tracy reads a timestamp twice per zone. On x86 that is one `rdtsc`; on aarch64
Linux it is `clock_gettime(CLOCK_MONOTONIC_RAW)`, because `TRACY_HAS_CNTVCT` is
only defined for Windows-on-ARM and Apple ("For now only supported on Apple
devices"), so `TRACY_HW_TIMER` is never set and every timestamp falls through to
the software path. The one-instruction `mrs CNTVCT_EL0` read that Apple uses is
in the same file, and the arm64 vDSO already reads that very counter to service
`clock_gettime` - so the patch is reading the same clock without the call around
it.

This matters because IREE emits on the order of a hundred host zones around each
dispatch, i.e. a few hundred timestamps per dispatch, which is why profiling a
small model on a small core can cost more than the model.

The tradeoff is resolution: `CNTFRQ_EL0` is commonly 19.2-100MHz, so ticks are
10-52ns rather than 1ns. Zones shorter than a microsecond quantize visibly.
`TRACY_DISALLOW_HW_TIMER` restores the old behavior.

Measure both on the target first with
`build_tools/third_party/tracy/tools/clock_probe.c` - if `CLOCK_MONOTONIC_RAW`
is not served by the vDSO on that kernel it is a full syscall and the gap is
much larger than the resolution cost.

Not yet submitted upstream.
