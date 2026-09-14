"""Reject Office documents in the Git index without reading document contents."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

OFFICE_SUFFIXES = {".docm", ".docx", ".pptm", ".pptx", ".xlsm", ".xlsx"}


def main() -> int:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write("Could not inspect staged files; refusing the commit.\n")
        return 2
    blocked = [
        os.fsdecode(name)
        for name in result.stdout.split(b"\0")
        if name and Path(os.fsdecode(name)).suffix.lower() in OFFICE_SUFFIXES
    ]
    if not blocked:
        return 0
    sys.stderr.write("Office documents must remain private and must not be committed:\n")
    for name in blocked:
        sys.stderr.write(f"  {name!r}\n")
    sys.stderr.write("Unstage these files and retain them in private document storage.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
