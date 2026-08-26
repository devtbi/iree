#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks IREE_TRACY_CAPTURE_FILE end to end.

A tracing-enabled tool records its own tracy stream to a file with no profiler
attached and nothing spawned, and `iree-tracy-profile convert` turns that
recording into a .tracy that reads back with its zone names intact. The names
are the point: the client sends them only when a server asks, so a recording
that did not ask would still convert, just with every zone called "???".

Runs iree-run-module against a module path that does not exist: the tool still
enters and exits tracing (emitting startup zones) and fails cleanly, so no
compiled module is needed.
"""

import json
import os
import subprocess
import sys
import tempfile


def main():
    run_module, profile_tool = sys.argv[1:3]
    with tempfile.TemporaryDirectory() as tmp:
        recording = os.path.join(tmp, "run.tracyrec")
        trace = os.path.join(tmp, "run.tracy")
        env = dict(os.environ)
        env["IREE_TRACY_CAPTURE_FILE"] = recording
        env.pop("TRACY_NO_EXIT", None)
        # A port unlikely to collide with a developer's live profiler.
        env.setdefault("TRACY_PORT", "8195")
        result = subprocess.run(
            [run_module, "--device=local-sync", "--module=/nonexistent.vmfb",
             "--function=main"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode != 0, "expected the missing module to fail"
        assert "NOT_FOUND" in result.stderr, result.stderr
        assert os.path.exists(recording), f"no recording written: {result.stderr}"
        assert os.path.getsize(recording) > 0, "recording is empty"

        convert = subprocess.run(
            [profile_tool, "convert", recording, f"--output={trace}"],
            capture_output=True, text=True, timeout=300,
        )
        assert convert.returncode == 0, convert.stderr + convert.stdout
        assert os.path.exists(trace), convert.stderr + convert.stdout
        # Every name the stream referenced was resolved while the process was
        # alive; anything the converter had to invent would show up here.
        assert "0 names synthesized" in convert.stdout, convert.stdout

        summary = subprocess.run(
            [profile_tool, "summary", "--format=jsonl", trace],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout
        rows = [json.loads(line) for line in summary.splitlines() if line.strip()]
        [head] = [row for row in rows if row["type"] == "summary"]
        assert head["cpu_zone_count"] > 0, head
        assert head["failure"] is None, head

        # Zone names only survive if the recorder asked the client for them.
        zones = subprocess.run(
            [profile_tool, "zone", "--format=jsonl", trace],
            capture_output=True, text=True, check=True, timeout=120,
        ).stdout
        groups = [
            row
            for row in (json.loads(line) for line in zones.splitlines() if line.strip())
            if row["type"] == "zone_group"
        ]
        assert groups, "no zone groups in the converted trace"
        unresolved = [
            row
            for row in groups
            if row.get("key", "???") == "???" or row.get("file", "???") == "???"
        ]
        assert not unresolved, f"{len(unresolved)}/{len(groups)} zone names lost"
        names = groups
        print(
            f"capture file test passed ({head['cpu_zone_count']} zones, "
            f"{len(names)} named zone groups)"
        )


if __name__ == "__main__":
    main()
