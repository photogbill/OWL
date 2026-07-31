"""Install a PREBUILT llama-cpp-python wheel. Never builds from source.

This lives in Python rather than in install.bat because cmd's `for /f` with
backquotes mangles quoted paths that contain spaces -- a directory called
"owl-engine - Test" produced:

    'D:\\Analyst_Toolkit\\OWL\\owl-engine' is not recognized as an internal
    or external command

...and, worse, silently left LLAMA_OK unset even though the wheel HAD
installed, so the script reported nothing at all. Path quoting is not
something to hand-roll in batch.

    python scripts/install_backend.py           try each candidate in turn
    python scripts/install_backend.py --check   report only, install nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_backend import candidates          # noqa: E402


def installed_version() -> str | None:
    if importlib.util.find_spec("llama_cpp") is None:
        return None
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import llama_cpp;print(llama_cpp.__version__)"],
            capture_output=True, text=True, timeout=120)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _pip(*args: str) -> tuple[bool, str]:
    """Run pip QUIETLY. `--quiet` does not suppress pip's ERROR lines, so a
    working fallback chain printed two alarming errors before succeeding and
    looked like a failure. Output is captured and only surfaced if every
    candidate fails."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--disable-pip-version-check", *args]
    try:
        r = subprocess.run(cmd, timeout=1800, capture_output=True, text=True)
        return r.returncode == 0, (r.stderr or r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    have = installed_version()
    if have:
        print(f"        [OK]   llama-cpp-python {have}  - Tier 1/2 available")
        return 0
    if args.check:
        print("        [SKIP] llama-cpp-python not installed")
        return 1

    urls, _ = candidates()
    errors: list[str] = []
    for url in urls:
        label = url.rsplit("/", 1)[-1] if url else "PyPI"
        print(f"        trying {label} ...", end=" ", flush=True)
        if url:
            ok, err = _pip("llama-cpp-python", "--extra-index-url", url,
                           "--only-binary=:all:")
        else:
            ok, err = _pip("llama-cpp-python", "--only-binary=:all:")
        if ok and (v := installed_version()):
            print(f"\n        [OK]   llama-cpp-python {v}  - Tier 1/2 "
                  f"available  (via {label})")
            if label != urls[0].rsplit("/", 1)[-1]:
                print(f"               note: no wheel at "
                      f"{urls[0].rsplit('/', 1)[-1]}, so this build is CPU-"
                      "only unless")
                print("               that index gains one. Embedding will "
                      "be slower than GPU.")
            return 0
        print("no wheel")
        errors.append(f"{label}: {err.splitlines()[-1] if err else 'failed'}")

    print("\n        [SKIP] No prebuilt llama-cpp-python wheel matched this "
          "setup.")
    for e in errors:
        print(f"               {e}")
    print("               Tier 0 is COMPLETE without it - provenance, decay,")
    print("               FOK triage, decisions, blast radius and poisoning")
    print("               defence all work. Only semantic recall is off.")
    v = sys.version_info
    if v >= (3, 13):
        print(f"               Python {v.major}.{v.minor} is the likely cause;"
              " 3.10-3.12 is the safe range.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
