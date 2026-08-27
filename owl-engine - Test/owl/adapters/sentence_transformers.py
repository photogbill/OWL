"""Wrap an existing sentence-transformers model. ~5 lines of real work.

For hosts that already have one loaded -- ATK's `atk/core/vector_store.py`,
for instance -- reuse it rather than paying for a second model in RAM:

    from sentence_transformers import SentenceTransformer
    from owl.adapters.sentence_transformers import STEmbedder
    emb = STEmbedder(SentenceTransformer("all-MiniLM-L6-v2"))
"""
from __future__ import annotations

from typing import Any, Sequence

from ..protocols import Space


class STEmbedder:
    is_semantic = True

    def __init__(self, model: Any, name: str = "sentence-transformers"):
        self._m = model
        self.name = name
        self.dim = int(model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str], space: Space) -> list[list[float]]:
        if not texts:
            return []
        return [list(map(float, v)) for v in self._m.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False)]
