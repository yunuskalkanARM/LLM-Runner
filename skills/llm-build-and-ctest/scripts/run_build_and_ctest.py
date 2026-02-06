#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_build_and_ctest.py",
        description="Configure (CMake preset), build, and run CTest for this repository.",
    )
    parser.add_argument("preset")
    parser.add_argument("build_dir")
    parser.add_argument(
        "cmake_configure_args",
        nargs=argparse.REMAINDER,
        help="Optional extra args after `--`, e.g. -- -DBUILD_JNI_LIB=OFF",
    )
    args = parser.parse_args()

    extra = args.cmake_configure_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    elif extra:
        print("Extra configure args must come after `--`.", file=sys.stderr)
        return 2

    subprocess.run(
        ["cmake", f"--preset={args.preset}", "-B", args.build_dir, *extra],
        check=True,
    )
    subprocess.run(["cmake", "--build", args.build_dir], check=True)
    subprocess.run(["ctest", "--test-dir", args.build_dir, "--output-on-failure"], check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        returncode = e.returncode if isinstance(e.returncode, int) else 1
        raise SystemExit(returncode)
