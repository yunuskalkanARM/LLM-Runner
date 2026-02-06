#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sha256_file.py",
        description="Prints the SHA256 hash of a local file (for scripts/py/requirements.json entries).",
    )
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"Missing file: {path}", file=sys.stderr)
        return 1

    print(sha256_file(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
