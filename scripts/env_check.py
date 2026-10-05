"""Bounded, dependency-free environment report for local or cluster setup."""

from __future__ import annotations

import platform
import shutil
import sys


def main() -> int:
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    print(f"python={sys.version.split()[0]}")
    print(f"python_executable={sys.executable}")
    for command in ("git", "gh", "ffmpeg", "nvidia-smi"):
        print(f"{command}={'available' if shutil.which(command) else 'missing'}")
    print("status=environment inventory only; no training started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
