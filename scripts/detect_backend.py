"""Work out which llama-cpp-python wheel index to use on this machine.

Batch and shell are terrible at parsing `nvidia-smi`, so the detection lives
here and the installers just call it. Prints an ordered, whitespace-separated
list of candidate index URLs, best first -- the installer tries each in turn.

    python scripts/detect_backend.py              # -> URLs, best first
    python scripts/detect_backend.py --explain    # -> human-readable
"""
from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys

BASE = "https://abetlen.github.io/llama-cpp-python/whl"

# Only these CUDA lines have published wheels. A newer driver reports a newer
# CUDA version than any wheel targets, so we clamp downward rather than
# constructing a URL that 404s.
CUDA_WHEELS = ("cu126", "cu125", "cu124", "cu123", "cu122", "cu121")


def cuda_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return f"cu{m.group(1)}{m.group(2)}" if m else None


def candidates() -> tuple[list[str], list[str]]:
    """(index urls best-first, human notes)"""
    notes: list[str] = []
    urls: list[str] = []
    sysname = platform.system()

    if sysname == "Darwin" and platform.machine() == "arm64":
        urls.append(f"{BASE}/metal")
        notes.append("Apple Silicon detected -> Metal wheels")
    else:
        cu = cuda_version()
        if cu:
            notes.append(f"nvidia-smi reports {cu}")
            if cu in CUDA_WHEELS:
                urls.append(f"{BASE}/{cu}")
            else:
                # Driver newer than any published wheel: step down to the
                # highest wheel that exists. CUDA minor versions are
                # backward compatible, so this is safe.
                lower = [w for w in CUDA_WHEELS if w <= cu]
                pick = lower[0] if lower else CUDA_WHEELS[0]
                urls.append(f"{BASE}/{pick}")
                notes.append(f"no wheel for {cu}; stepping down to {pick}")
        else:
            notes.append("no NVIDIA GPU detected")

    urls.append(f"{BASE}/cpu")
    notes.append("CPU wheels as fallback")
    urls.append("")                      # plain PyPI / source build, last
    notes.append("plain PyPI last (may build from source)")

    v = sys.version_info
    if v >= (3, 13):
        notes.append(
            f"WARNING: Python {v.major}.{v.minor} often has no prebuilt "
            "llama-cpp-python wheel yet. Python 3.10-3.12 is the safe range.")
    return urls, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()
    urls, notes = candidates()
    if args.explain:
        for n in notes:
            print(f"  {n}")
        print("\n  trying, in order:")
        for u in urls:
            print(f"    {u or '(plain PyPI)'}")
    else:
        print(" ".join(u or "PYPI" for u in urls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
