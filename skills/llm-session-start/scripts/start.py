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
        prog="start.py",
        description="Runs doctor checks and prints pinned framework versions; optionally checks upstream for updates.",
        epilog="For bumping pinned framework versions, see skills/llm-session-start/SKILL.md.",
    )
    parser.add_argument("build_dir", nargs="?", default="build")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Also run scripts/dev/check_framework_updates.py (requires network).",
    )
    args = parser.parse_args()

    subprocess.run(
        [sys.executable, "scripts/dev/llm_doctor.py", "--build-dir", args.build_dir],
        check=True,
    )
    print()
    subprocess.run(
        [sys.executable, "scripts/dev/framework_versions.py", "--build-dir", args.build_dir],
        check=True,
    )

    if args.network:
        print()
        result = subprocess.run(
            [sys.executable, "scripts/dev/check_framework_updates.py", "--build-dir", args.build_dir],
            check=False,
        )
        return result.returncode

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)
