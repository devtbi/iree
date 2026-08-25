#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke test for iree-tracy-profile against a checked-in capture.

The fixture was produced by replaying a synthetic HAL profiling session through
the Tracy sink (iree_hal_profile_tracy_sink_t): one device with two queues, 200
iterations of an `execute` containing three dispatches plus a `copy`, and
host-side submission spans. Numbers asserted below follow from that shape.
"""

import json
import subprocess
import sys


def run(tool, trace, *args):
    result = subprocess.run(
        [tool, *args, trace], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} failed ({result.returncode}): {result.stderr}"
        )
    return result.stdout


def rows(tool, trace, *args):
    out = run(tool, trace, "--format=jsonl", *args)
    parsed = []
    for line in out.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "type" in row, f"row without type: {line}"
        parsed.append(row)
    return parsed


def by_type(parsed, type_name):
    return [row for row in parsed if row["type"] == type_name]


def main():
    tool, trace = sys.argv[1], sys.argv[2]

    summary = rows(tool, trace, "summary")
    [head] = by_type(summary, "summary")
    assert head["gpu_zone_count"] == 1200, head
    assert head["gpu_context_count"] == 4, head
    assert head["failure"] is None, head
    contexts = by_type(summary, "summary_gpu_context")
    lanes = sorted(row["lane"] for row in contexts)
    assert lanes == ["dispatch", "queue", "queue", "submit"], lanes
    for row in contexts:
        assert row["physical_device_ordinal"] == 0, row
        assert row["unresolved_count"] == 0, row

    dispatch = rows(tool, trace, "dispatch")
    groups = by_type(dispatch, "dispatch_group")
    assert len(groups) == 3, groups
    assert all(group["count"] == 200 for group in groups), groups
    keys = sorted(group["key"] for group in groups)
    assert keys == [
        "attention_dispatch_0_flash_attn_fwd",
        "matmul_dispatch_0_matmul_4096x4096xf16",
        "softmax_dispatch_1_generic_4096xf32",
    ], keys
    matmul = next(g for g in groups if g["key"].startswith("matmul"))
    assert matmul["avg_ns"] == 50000, matmul
    assert matmul["p99_ns"] == 50000, matmul
    # Shares are printed with six significant digits.
    assert abs(sum(group["share"] for group in groups) - 1.0) < 1e-4, groups

    events = rows(tool, trace, "dispatch", "--dispatch_events", "--filter=*softmax*")
    assert len(events) == 200, len(events)
    assert all(row["duration_ns"] == 80000 for row in events), events[0]
    event_id = events[0]["event_id"]
    [single] = rows(tool, trace, "dispatch", f"--id={event_id}")
    assert single["key"] == events[0]["key"], single

    queue = rows(tool, trace, "queue")
    queue_groups = {row["key"]: row for row in by_type(queue, "queue_group")}
    # execute appears on the device queue lane and the host submit lane.
    assert sorted(queue_groups) == ["copy", "execute"], sorted(queue_groups)
    assert queue_groups["copy"]["count"] == 200, queue_groups["copy"]
    assert queue_groups["copy"]["avg_ns"] == 40000, queue_groups["copy"]

    statistics = rows(tool, trace, "statistics")
    row_types = {row["row_type"] for row in by_type(statistics, "statistics_row")}
    assert {"gpu_zone", "message_severity"} <= row_types, row_types

    messages = rows(tool, trace, "message", "--filter=HAL profile sink*")
    assert len(messages) == 1 and messages[0]["severity"] == "warning", messages

    explain = rows(tool, trace, "explain", "--top=2")
    assert by_type(explain, "explain_span"), explain
    assert len(by_type(explain, "explain_top_dispatch")) == 2, explain
    assert any(
        "HAL profile sink" in row["message"] for row in by_type(explain, "explain_hint")
    ), by_type(explain, "explain_hint")

    cat = rows(tool, trace, "cat")
    assert len(by_type(cat, "gpu_zone_event")) == 1200, len(cat)
    assert len(by_type(cat, "gpu_context")) == 4, len(cat)

    memory = rows(tool, trace, "memory")
    [pool] = by_type(memory, "memory_pool")
    assert pool["allocation_count"] == pool["free_count"] == 9, pool
    assert pool["live_count"] == 0 and pool["live_bytes"] == 0, pool
    assert 0 < pool["high_water_bytes"] <= pool["total_allocated_bytes"], pool

    # Text mode renders for every command without failing; commands whose
    # category is populated in the fixture must print something.
    for command in ["summary", "statistics", "dispatch", "queue", "zone", "thread",
                    "message", "memory", "frame", "explain"]:
        assert run(tool, trace, command), command
    run(tool, trace, "plot")  # No plots in this capture: empty but successful.
    assert "Record types" in run(tool, trace, "--agents_md")

    print("iree-tracy-profile smoke test passed")


if __name__ == "__main__":
    main()
