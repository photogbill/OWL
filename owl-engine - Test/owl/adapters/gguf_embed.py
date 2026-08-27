"""GGUF embedder via llama.cpp. Local model path only — nothing is downloaded.

Air-gap is the default, not a mode. A library that quietly reaches for the
network the first time it is asked a question is unusable in the environments
OWL is built for.

    from owl.adapters.gguf_embed import GgufEmbedder
    emb = GgufEmbedder("embedding model/bge-m3-Q6_K.gguf")

Conventions are read from the GGUF and applied automatically, because
embedding models do NOT share them and getting it wrong is SILENT -- the
vectors come back the right shape and retrieval is merely bad:

    bge-m3            bert  + CLS pooling   no query instruction
    Qwen3-Embedding   qwen3 + LAST pooling  REQUIRES a query instruction
    bge-*-en          bert  + CLS pooling   "Represent this sentence..."
    e5                bert  + mean pooling  "query:" / "passage:"

The asymmetry matters: an instruction belongs on the QUERY side only. That
is also the first reason beyond pattern separation for OWL's READ/WRITE
space split to exist -- the two spaces genuinely encode different text.

Why llama.cpp rather than ONNX: the host (ATK) already ships
llama-cpp-python with prebuilt CUDA/CPU wheels, so this adds no new native
dependency and reuses a model loader the user already trusts.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Sequence

from ..protocols import Space


def available() -> bool:
    """Is llama-cpp-python importable? Probe WITHOUT importing it.

    Callers need to distinguish "the backend is missing" from "the model
    failed to load" -- they have completely different fixes, and collapsing
    them into one exception at construction time produces the unhelpful
    traceback this function exists to prevent.
    """
    return importlib.util.find_spec("llama_cpp") is not None

from . import calibration
from .gguf_meta import conventions, read as read_meta


class GgufEmbedder:
    """Wraps a llama.cpp embedding model. Deterministic and CPU-friendly."""

    is_semantic = True
    # A NOISE CUTOFF, not a decision threshold. Measured on BGE-M3 against a
    # small field corpus the bands genuinely overlap -- related 0.426-0.691,
    # unrelated 0.213-0.514 -- so no threshold separates them. Set low enough
    # to keep true matches; the margin test in `semantic_density` does the
    # actual discriminating.
    noise_floor = 0.40
    search_floor = 0.15
    # The top of the encoder's real range. 1.0 is a lie for every embedder
    # and scaling scores by it buries genuine matches near zero.
    ceiling = 1.0

    def __init__(self, model_path: str | Path, *, n_ctx: int | None = None,
                 n_threads: int | None = None, n_gpu_layers: int = 0,
                 query_prefix: str | None = None, verbose: bool = False,
                 max_chars: int | None = None, noise_floor: float | None = None,
                 pooling: str | None = None):
        try:
            from llama_cpp import Llama
        except ImportError as e:                          # pragma: no cover
            raise ImportError(
                "GgufEmbedder needs llama-cpp-python. Install the prebuilt "
                "wheel: pip install owl-engine[llama]"
            ) from e

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. OWL never downloads models -- place the .gguf "
                "yourself and pass its path."
            )
        self.path = p
        self.meta = read_meta(p)
        conv = conventions(self.meta)
        self.name = p.stem
        # Explicit arguments win; otherwise follow the model's own convention.
        self.query_prefix = (conv["query_prefix"] if query_prefix is None
                             else query_prefix)
        self.doc_prefix = conv["doc_prefix"]
        self.convention_reason = conv["reason"]
        # A context window this large is not free: llama.cpp allocates a KV
        # cache for it. Qwen3-Embedding declares 40960, which would reserve
        # gigabytes for texts a few hundred tokens long.
        declared = self.meta.context or 8192
        n_ctx = n_ctx or min(declared, 8192)
        # Leave headroom for the instruction prefix and special tokens.
        self.max_chars = max_chars or max(512, (n_ctx - 64) * 3)
        # Only consulted if llama.cpp hands back per-token output. "cls"
        # matches how BGE-M3 was trained; "mean" is the usual fallback for
        # encoders without a pooling declaration.
        self.pooling = conv["pooling"] if pooling is None else pooling
        self.pooled_by_llama: bool | None = None
        self._llm = Llama(
            model_path=str(p),
            embedding=True,
            n_ctx=n_ctx,
            n_threads=n_threads or max(1, (os.cpu_count() or 4) - 1),
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        # Measured parameters travel with the model file. Which signal is
        # informative -- absolute level or margin above background -- differs
        # between encoders, and so does the absolute scale, so these cannot
        # be constants in the source.
        self.calibration = calibration.load(p)
        if self.calibration is not None:
            self.noise_floor = self.calibration.noise_floor
            self.search_floor = self.calibration.search_floor
            self.level_weight = self.calibration.level_weight
            self.ceiling = self.calibration.ceiling
        if noise_floor is not None:
            self.noise_floor = noise_floor
        probe = self._embed_raw(["dimension probe"])
        self.dim = len(probe[0])

    # ── internals ────────────────────────────────────────────────────
    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        out = self._llm.create_embedding(texts)
        vecs: list[list[float]] = []
        for item in out["data"]:
            v = item["embedding"]
            # llama.cpp returns [dim] when it applied pooling itself, and
            # [n_tokens][dim] when it did not. Which one matters a great
            # deal, and the right answer is per-model: CLS for BERT-style
            # encoders, LAST for causal ones. Taking token 0 from a causal
            # model is meaningless -- position 0 has attended to nothing.
            if v and isinstance(v[0], (list, tuple)):
                self.pooled_by_llama = False
                if self.pooling == "cls":
                    v = list(v[0])
                elif self.pooling == "last":
                    v = list(v[-1])
                else:
                    cols = list(zip(*v))
                    v = [sum(c) / len(c) for c in cols]
            else:
                self.pooled_by_llama = True
            vecs.append([float(x) for x in v])
        return vecs

    # ── protocol ─────────────────────────────────────────────────────
    def embed(self, texts: Sequence[str], space: Space) -> list[list[float]]:
        if not texts:
            return []
        prepared = []
        for t in texts:
            t = t[: self.max_chars]
            # Asymmetric by design: the instruction goes on the query side
            # only. Applying it to documents too would defeat its purpose.
            if space is Space.READ and self.query_prefix:
                t = self.query_prefix + t
            elif space is Space.WRITE and self.doc_prefix:
                t = self.doc_prefix + t
            prepared.append(t or " ")
        return self._embed_raw(prepared)

    def close(self) -> None:
        self._llm = None

    def describe(self) -> str:
        cal = self.calibration
        cal_line = (f"\n    calibrated: floor={cal.noise_floor}"
                    f"..{cal.ceiling} "
                    f"level_weight={cal.level_weight} "
                    f"(separator: {cal.separator})"
                    if cal else
                    "\n    NOT CALIBRATED - run --calibrate; defaults were "
                    "measured on a different model")
        # Headroom is the encoder's verdict on itself: how far a true match
        # outscores an unrelated one. Worth saying every load, not once.
        if cal and cal.anisotropy_p95:
            cal_line += (f"\n    query->doc background={cal.anisotropy:.3f} "
                         f"headroom={cal.headroom:+.3f}")
            if cal.headroom <= 0:
                cal_line += "  <- POOR: check pooling/prefix, then the quant"
            elif cal.headroom < 0.05:
                cal_line += "  <- thin"
        if cal and cal.doc_anisotropy >= 0.25:
            d, c2 = cal.fusion_thresholds(0.85, 0.75)
            cal_line += (f"\n    doc->doc background={cal.doc_anisotropy:.3f}"
                         f" -> fusion rescaled to {d}/{c2}")
        return (f"{self.meta.name}  arch={self.meta.architecture} "
                f"dim={getattr(self, 'dim', '?')} pooling={self.pooling}"
                + cal_line
                + (f"\n    query prefix: {self.query_prefix[:52]!r}"
                   if self.query_prefix else "\n    no query prefix")
                + f"\n    ({self.convention_reason})")

    def __repr__(self) -> str:                            # pragma: no cover
        return f"<GgufEmbedder {self.name} dim={getattr(self, 'dim', '?')}>"
