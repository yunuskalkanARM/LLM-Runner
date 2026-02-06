#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


def _read_failed_tests_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "<index>:<testname>"
        if ":" in line:
            _, name = line.split(":", 1)
            name = name.strip()
            if name:
                names.append(name)
        else:
            names.append(line)
    return names


def _unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rerun_failing_ctest.py",
        description="Runs ctest once, then reruns only failing tests verbosely.",
    )
    parser.add_argument("build_dir", nargs="?", default="build")
    parser.add_argument(
        "--filter",
        default="",
        help="Only rerun failing tests whose names contain this substring.",
    )
    parser.add_argument(
        "--cpp-transcript",
        default="",
        help="If set, enables C++ test debug output and appends a transcript to this path.",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.is_dir():
        print(f"Missing build dir: {build_dir}", file=sys.stderr)
        return 2

    env = dict(os.environ)

    if args.cpp_transcript:
        env["LLM_TEST_DEBUG_RESPONSES"] = "1"
        env["LLM_TEST_TRANSCRIPT_PATH"] = args.cpp_transcript

    initial = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        env=env,
    )
    if initial.returncode == 0:
        return 0

    failed_file = build_dir / "Testing" / "Temporary" / "LastTestsFailed.log"
    failed = _unique_in_order(_read_failed_tests_file(failed_file))
    if args.filter:
        failed = [t for t in failed if args.filter in t]

    if not failed:
        print(
            f"ctest failed (exit {initial.returncode}) but no {failed_file} found/parsed; re-run with: "
            f"ctest --test-dir {build_dir} -V --output-on-failure",
            file=sys.stderr,
        )
        return initial.returncode

    overall = initial.returncode
    for test_name in failed:
        print()
        print(f"=== Rerunning failing test: {test_name}")
        rerun = subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "-R", f"^{test_name}$", "-V", "--output-on-failure"],
            env=env,
        )
        if rerun.returncode != 0:
            overall = rerun.returncode

    return overall


if __name__ == "__main__":
    raise SystemExit(main())
