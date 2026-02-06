#!/usr/bin/env python3

#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile


def _run_capture(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    out = (result.stdout or "") + (result.stderr or "")
    return out.strip()


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text.strip() else ""


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.is_file():
        dst = dst_dir / src.name
        shutil.copyfile(src, dst)


@dataclass(frozen=True)
class BundlePaths:
    build_dir: Path
    out_dir: Path
    bundle_dir: Path


def _write_env_txt(paths: BundlePaths) -> None:
    env_txt = paths.bundle_dir / "env.txt"
    now = datetime.now(timezone.utc).astimezone()

    cmake_version = _first_line(_run_capture(["cmake", "--version"]))
    py_version = _first_line(_run_capture([sys.executable, "--version"])) or _first_line(
        _run_capture(["python3", "--version"])
    )
    java_version = _first_line(_run_capture(["java", "-version"]))

    java_home = os.environ.get("JAVA_HOME", "<unset>")

    env_txt.write_text(
        "\n".join(
            [
                f"date: {now.isoformat(timespec='seconds')}",
                f"platform: {platform.platform()}",
                f"cmake: {cmake_version}",
                f"python: {py_version}",
                f"java: {java_version}",
                f"JAVA_HOME: {java_home}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _git_in_worktree() -> bool:
    try:
        result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _write_git_info(paths: BundlePaths) -> None:
    if not _git_in_worktree():
        return

    (paths.bundle_dir / "git_head.txt").write_text(_run_capture(["git", "rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (paths.bundle_dir / "git_status.txt").write_text(
        _run_capture(["git", "status", "--porcelain=v1"]) + "\n", encoding="utf-8"
    )
    (paths.bundle_dir / "git_diff.patch").write_text(_run_capture(["git", "diff"]) + "\n", encoding="utf-8")


def _copy_build_files(paths: BundlePaths) -> None:
    _copy_if_exists(paths.build_dir / "CMakeCache.txt", paths.bundle_dir)
    _copy_if_exists(paths.build_dir / "CMakeFiles" / "CMakeOutput.log", paths.bundle_dir)
    _copy_if_exists(paths.build_dir / "CMakeFiles" / "CMakeError.log", paths.bundle_dir)

    last_test_log = paths.build_dir / "Testing" / "Temporary" / "LastTest.log"
    if last_test_log.is_file():
        dest_dir = paths.bundle_dir / "Testing" / "Temporary"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(last_test_log, dest_dir / "LastTest.log")


def _copy_repo_logs(paths: BundlePaths) -> None:
    download_log = Path("download.log")
    if download_log.is_file():
        _copy_if_exists(download_log, paths.bundle_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="collect_debug_bundle.py",
        description="Creates a tar.gz bundle with high-signal build/test logs and environment info for bug reports.",
    )
    parser.add_argument("build_dir")
    parser.add_argument("out_dir", nargs="?", default=".")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    out_dir = Path(args.out_dir)

    if not build_dir.is_dir():
        print(f"Missing build dir: {build_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle_path = out_dir / f"llm-debug-bundle-{ts}.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp)
        paths = BundlePaths(build_dir=build_dir, out_dir=out_dir, bundle_dir=bundle_dir)

        _write_env_txt(paths)
        _write_git_info(paths)
        _copy_build_files(paths)
        _copy_repo_logs(paths)

        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=".")

    print(f"Wrote {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
