"""Fetch an ONNX embedding model for later offline use. Run this ONLINE.

    python -m owl.adapters.fetch_model --out models/all-MiniLM-L6-v2

Then copy that directory onto the air-gapped machine. OWL itself never makes
a network call; this is a separate, explicit, opt-in step, which is why it
lives in its own module rather than inside the embedder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "sentence-transformers/all-MiniLM-L6-v2"
FILES = [("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="models/all-MiniLM-L6-v2")
    args = ap.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("needs `pip install huggingface_hub`", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES:
        p = hf_hub_download(repo_id=args.repo, filename=remote)
        (out / local).write_bytes(Path(p).read_bytes())
        print(f"  {out / local}")
    print(f"\nCopy {out} to the offline machine, then:\n"
          f"  OnnxEmbedder('{out}/model.onnx', '{out}/tokenizer.json')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
