"""Deterministic hashing embedder. No dependencies, no model, no download.

This is NOT a good semantic embedder -- it is a bag-of-words projection, so it
captures overlap and not meaning. It exists for two reasons:

  * tests need a deterministic Embedder that runs in microseconds
  * the two-space machinery should be exercisable by anyone who just ran
    `pip install owl-engine`, without fetching anything

Use `OnnxEmbedder` for real work. The honest framing matters: a fallback that
silently produces poor retrieval while looking like it works is worse than no
fallback, so `is_semantic` is False and OWL reports the tier accordingly.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from ..lexical import tokenize
from ..protocols import Space


class HashingEmbedder:
    is_semantic = False
    name = "hashing-256"
    noise_floor = 0.25
    search_floor = 0.05

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str], space: Space) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]
