#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="assert_config_listed.py",
        description="Checks that the given config filename appears in test/CMakeLists.txt.",
    )
    parser.add_argument("config_filename", help="e.g. llamaTextConfig-phi-2.json")
    args = parser.parse_args()

    cmake_lists = Path("test") / "CMakeLists.txt"
    if not cmake_lists.is_file():
        print(f"Missing {cmake_lists}", file=sys.stderr)
        return 1

    needle = f"\"{args.config_filename}\""
    content = cmake_lists.read_text(encoding="utf-8", errors="replace")
    if needle in content:
        print(f"OK: {args.config_filename} is referenced in {cmake_lists}")
        return 0

    print(f"Missing: {args.config_filename} is not referenced in {cmake_lists}", file=sys.stderr)
    print("Add it under the correct LLM_FRAMEWORK branch (CONFIG_FILE_NAME list).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
