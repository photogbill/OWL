"""Read the handful of GGUF metadata keys that change how a model must be used.

Embedding models do NOT share conventions. Getting them wrong is silent:
the vectors come back the right shape and the retrieval quality is merely
bad, which is the hardest kind of bug to notice.

    bge-m3              bert  + pooling CLS   no query instruction
    Qwen3-Embedding     qwen3 + pooling LAST  REQUIRES a query instruction

Stdlib only, and it reads the header rather than the tensors, so it is
milliseconds on a 5 GB file.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# GGUF value types -> (struct code, byte width)
_T = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
      5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
      12: ("d", 8)}

POOLING = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}

WANTED = ("general.architecture", "general.name", ".pooling_type",
          ".embedding_length", ".context_length", ".block_count")


@dataclass(frozen=True)
class GgufMeta:
    architecture: str = "unknown"
    name: str = ""
    pooling: str = "unknown"
    dim: int = 0
    context: int = 0
    blocks: int = 0

    @property
    def is_causal(self) -> bool:
        """A decoder LLM used as an embedder, rather than a BERT encoder.

        Matters because position 0 of a causal model has attended to nothing,
        so CLS pooling is meaningless there.
        """
        return self.architecture not in ("bert", "nomic-bert", "jina-bert-v2",
                                         "roberta", "unknown")


def read(path: str | Path) -> GgufMeta:
    p = Path(path)
    with p.open("rb") as f:
        if f.read(4) != b"GGUF":
            return GgufMeta()
        struct.unpack("<I", f.read(4))          # version
        struct.unpack("<Q", f.read(8))          # tensor count
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def skip(t: int):
            if t == 8:
                return rstr()
            if t == 9:                           # array
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if et == 8:
                    for _ in range(n):
                        rstr()
                    return None
                if et in _T:
                    f.seek(_T[et][1] * n, 1)
                return None
            if t in _T:
                return struct.unpack("<" + _T[t][0], f.read(_T[t][1]))[0]
            raise ValueError(f"unknown gguf type {t}")

        found: dict[str, object] = {}
        for _ in range(n_kv):
            try:
                k = rstr()
                t = struct.unpack("<I", f.read(4))[0]
                v = skip(t)
            except (ValueError, struct.error):
                break
            if v is not None and any(w in k for w in WANTED):
                found[k] = v

    def get(suffix: str, default=0):
        for k, v in found.items():
            if k.endswith(suffix):
                return v
        return default

    pool = get(".pooling_type", -1)
    return GgufMeta(
        architecture=str(found.get("general.architecture", "unknown")),
        name=str(found.get("general.name", p.stem)),
        pooling=POOLING.get(pool, "unknown") if pool != -1 else "unknown",
        dim=int(get(".embedding_length")),
        context=int(get(".context_length")),
        blocks=int(get(".block_count")),
    )


# ── per-family conventions ───────────────────────────────────────────────
# Qwen3-Embedding is trained with an instruction on the QUERY side only.
# Their own guidance puts the cost of omitting it at roughly 1-5%.
QWEN3_INSTRUCT = ("Instruct: Given a search query, retrieve relevant "
                  "passages that answer it\nQuery: ")

# BGE-large-en (not m3) wants this; bge-m3 explicitly does not.
BGE_EN_INSTRUCT = "Represent this sentence for searching relevant passages: "


def conventions(meta: GgufMeta) -> dict:
    """Query prefix and pooling fallback for this model family."""
    arch = meta.architecture.lower()
    name = meta.name.lower()

    if "qwen3" in arch or "qwen3-embedding" in name:
        prefix, why = QWEN3_INSTRUCT, "Qwen3-Embedding expects a query instruction"
    elif "bge-m3" in name or "bge_m3" in name:
        prefix, why = "", "bge-m3 is trained without a query instruction"
    elif "bge" in name and "m3" not in name:
        prefix, why = BGE_EN_INSTRUCT, "bge-*-en expects a query instruction"
    elif "e5" in name:
        prefix, why = "query: ", "e5 expects 'query:' / 'passage:' prefixes"
    else:
        prefix, why = "", "unknown family; no query instruction applied"

    # Pooling fallback, used only if llama.cpp hands back per-token output.
    # Never CLS for a causal model: position 0 has attended to nothing.
    if meta.pooling in ("cls", "mean", "last"):
        pooling = meta.pooling
    else:
        pooling = "last" if meta.is_causal else "cls"

    return {"query_prefix": prefix, "pooling": pooling, "reason": why,
            "doc_prefix": "passage: " if "e5" in name else ""}
