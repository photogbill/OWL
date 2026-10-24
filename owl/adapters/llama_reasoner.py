"""A Reasoner over a local GGUF model via llama-cpp-python (the `llama` extra).

In-process, no server, no network. GBNF grammars are applied natively, so
the engine's JSON shapes are enforced at sampling time -- the cheapest way
to get parseable output from a 7B. The engine still re-parses and
re-verifies everything; the grammar improves the hit rate, it is not what
makes the output safe.

`name` defaults to the model file's stem, which is what A6 records as the
producing model. Give it something more specific if you run two quants of
the same model.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("owl.adapters.llama_reasoner")


class LlamaCppReasoner:
    def __init__(self, model_path: str | Path, *, n_ctx: int = 4096,
                 n_gpu_layers: int = 0, seed: int | None = None,
                 name: str | None = None, verbose: bool = False, **kw):
        try:
            from llama_cpp import Llama
        except ImportError as e:                          # pragma: no cover
            raise ImportError(
                "LlamaCppReasoner needs llama-cpp-python: from the owl-engine "
                "clone, pip install -e \".[llama]\"") from e
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"no GGUF model at {p}")
        self.model_path = p
        self.name = name or p.stem
        t0 = time.perf_counter()
        args = dict(model_path=str(p), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    verbose=verbose, **kw)
        if seed is not None:
            args["seed"] = seed
        self._llm = Llama(**args)
        self._grammars: dict[str, object] = {}
        self.stats = {"requests": 0, "errors": 0, "ms_total": 0.0}
        log.info("loaded %s (n_ctx=%d, gpu layers=%d) in %.1f s", p.name,
                 n_ctx, n_gpu_layers, time.perf_counter() - t0)

    def _grammar(self, text: str):
        g = self._grammars.get(text)
        if g is None:
            from llama_cpp import LlamaGrammar
            g = LlamaGrammar.from_string(text, verbose=False)
            self._grammars[text] = g
        return g

    def complete(self, prompt: str, *, grammar: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        t0 = time.perf_counter()
        self.stats["requests"] += 1
        try:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
                grammar=self._grammar(grammar) if grammar else None)
            text = out["choices"][0]["message"]["content"] or ""
        except Exception:
            self.stats["errors"] += 1
            log.exception("llama-cpp completion failed (%s)", self.name)
            raise
        ms = (time.perf_counter() - t0) * 1000
        self.stats["ms_total"] += ms
        log.info("reasoner %s: %d prompt chars -> %d chars in %.0f ms "
                 "(temp=%.2f, grammar=%s)", self.name, len(prompt), len(text),
                 ms, temperature, "yes" if grammar else "no")
        return text

    def describe(self) -> dict:
        return {"adapter": "llama_cpp", "name": self.name,
                "model_path": str(self.model_path), **self.stats}
