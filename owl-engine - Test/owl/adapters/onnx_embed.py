"""ONNX sentence embedder. Local model path only -- nothing is downloaded.

Air-gap is the default, not a mode. The adapter takes a path to a model you
placed yourself; there is no fetch-on-first-use, because a library that
silently reaches for the network the first time it is asked a question is
unusable in the environments OWL is built for.

    from owl.adapters.onnx_embed import OnnxEmbedder
    emb = OnnxEmbedder("models/all-MiniLM-L6-v2/model.onnx",
                       "models/all-MiniLM-L6-v2/tokenizer.json")

`python -m owl.adapters.fetch_model` will download one for you when you ARE
online, into a directory you can then copy onto the air-gapped machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..protocols import Space


class OnnxEmbedder:
    is_semantic = True

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path,
                 *, dim: int | None = None, max_length: int = 256,
                 providers: list[str] | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:      # pragma: no cover
            raise ImportError(
                "OnnxEmbedder needs `pip install owl-engine[embed]`"
            ) from e

        mp, tp = Path(model_path), Path(tokenizer_path)
        for p in (mp, tp):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. OWL never downloads models. Run "
                    "`python -m owl.adapters.fetch_model` on a networked "
                    "machine and copy the directory across."
                )
        self.name = mp.parent.name
        self.max_length = max_length
        self._tok = Tokenizer.from_file(str(tp))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(length=None)
        self._sess = ort.InferenceSession(
            str(mp), providers=providers or ["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.dim = dim or int(self._sess.get_outputs()[0].shape[-1])

    def embed(self, texts: Sequence[str], space: Space) -> list[list[float]]:
        if not texts:
            return []
        import numpy as np

        enc = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in self._inputs}

        out = self._sess.run(None, feed)[0]
        if out.ndim == 3:                       # mean-pool over tokens
            m = mask[..., None].astype(np.float32)
            out = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return (out / np.clip(norm, 1e-9, None)).astype(np.float32).tolist()
