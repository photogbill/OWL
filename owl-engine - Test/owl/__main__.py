"""`python -m owl <command>` — preflight and inspection without writing code.

    python -m owl doctor            environment + optional deps
    python -m owl doctor mind.owl   ...and health of a specific store
    python -m owl check             what tier can this machine run?
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path


def _have(mod: str) -> str | None:
    if importlib.util.find_spec(mod) is None:
        return None
    try:
        import importlib.metadata as md
        for dist in ("llama-cpp-python" if mod == "llama_cpp" else mod,
                     mod.replace("_", "-"), mod):
            try:
                return md.version(dist)
            except md.PackageNotFoundError:
                continue
    except Exception:                                     # noqa: BLE001
        pass
    return "present"


def cmd_check() -> int:
    from owl import __version__

    print("=" * 62)
    print("  O.W.L. preflight")
    print("=" * 62)
    v = sys.version_info
    print(f"  owl-engine      {__version__}")
    print(f"  python          {v.major}.{v.minor}.{v.micro}  "
          f"({platform.system()} {platform.machine()})")
    print(f"  sqlite          {__import__('sqlite3').sqlite_version}")
    print(f"  running from    {Path(sys.prefix).name}"
          f"{'  [venv]' if sys.prefix != sys.base_prefix else '  [SYSTEM PYTHON]'}")
    print()

    llama = _have("llama_cpp")
    numpy = _have("numpy")
    onnx = _have("onnxruntime")
    print(f"  llama_cpp       {llama or '-- not installed'}")
    print(f"  numpy           {numpy or '-- not installed (optional; speeds search)'}")
    print(f"  onnxruntime     {onnx or '-- not installed (optional)'}")
    print()

    tier = 2 if llama else 0
    print(f"  MAX TIER        {tier}")
    if tier == 0:
        print("    Tier 0 is complete and needs nothing: provenance, decay,")
        print("    FOK triage, interference, partitions, decisions, blast")
        print("    radius, poisoning defence. Only semantic recall is off.")
        print()
        print("    For Tier 1/2, install a llama-cpp-python wheel:")
        print("      pip install llama-cpp-python --extra-index-url \\")
        print("        https://abetlen.github.io/llama-cpp-python/whl/cpu")
        print("    or just run install.bat / ./install.sh")
    else:
        print("    Semantic recall available. Validate a model before")
        print("    trusting any Tier 1 number:")
        print("      python bench/validate_embedder.py <path.gguf>")
    if sys.prefix == sys.base_prefix:
        print()
        print("  NOTE: running under system Python, not a venv.")
        print("        install.bat / ./install.sh sets one up.")
    print("=" * 62)
    return 0


def cmd_doctor(store: str | None) -> int:
    rc = cmd_check()
    if not store:
        return rc
    from owl import Owl

    p = Path(store)
    if not p.exists():
        print(f"\n  [FAIL] no such store: {p}")
        return 1
    print(f"\n  store: {p}")
    with Owl.open(p) as mind:
        d = mind.doctor()
        for k in ("observations", "derived", "episodes", "vectors",
                  "quarantined", "open_impacts", "tier", "semantic",
                  "self_audit"):
            if k in d:
                print(f"    {k:16s} {d[k]}")
        if d["problems"]:
            print("    problems:")
            for pr in d["problems"]:
                print(f"      - {pr}")
        print(f"    healthy          {d['healthy']}")
        return 0 if d["healthy"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m owl")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("doctor", help="preflight, plus a store's health")
    d.add_argument("store", nargs="?")
    sub.add_parser("check", help="what tier can this machine run?")
    args = ap.parse_args(argv)
    if args.cmd == "doctor":
        return cmd_doctor(args.store)
    return cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
