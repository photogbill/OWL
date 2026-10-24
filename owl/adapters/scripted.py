"""A scripted Reasoner for tests, demos and the Tier 2 ablation bench.

Tier 2 has to be testable without a GPU, a network or a model -- the
correctness suite must stay green on a laptop with nothing installed. This
adapter answers from a script instead of a model, and records every call so
a test can assert on exactly what the engine asked.

Three script shapes, because different tests need different control:

  * a list    -- answers in order; `cycle=True` repeats it, otherwise running
                 off the end raises (a test that makes more calls than it
                 planned for should fail loudly, not get a default);
  * a dict    -- {substring: answer-or-list}; the first key found in the
                 prompt wins, in insertion order. A list value is consumed in
                 order, then its last item repeats. `default` answers prompts
                 no key matches (None means raise);
  * a callable -- fn(prompt, temperature) -> str, for anything else.

It is deliberately NOT clever. A fake that guesses what the engine wanted
makes tests pass that should fail.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

log = logging.getLogger("owl.adapters.scripted")

Script = (Sequence[str] | Mapping[str, "str | Sequence[str]"]
          | Callable[[str, float], str])


class ScriptExhausted(RuntimeError):
    """The engine made a call the script did not plan for."""


class ScriptedReasoner:
    """Deterministic stand-in for a model. `name` is recorded as A6 provenance."""

    def __init__(self, script: Script, *, name: str = "scripted",
                 cycle: bool = False, default: str | None = None):
        self.script = script
        self.name = name
        self.cycle = cycle
        self.default = default
        self.calls: list[dict] = []
        self._i = 0
        self._per_key: dict[str, int] = {}

    def complete(self, prompt: str, *, grammar: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        self.calls.append({"prompt": prompt, "grammar": grammar,
                           "max_tokens": max_tokens,
                           "temperature": temperature})
        out = self._answer(prompt, temperature)
        log.debug("scripted call #%d (temp=%.2f) -> %r", len(self.calls),
                  temperature, out[:120])
        return out

    def _answer(self, prompt: str, temperature: float) -> str:
        s = self.script
        if callable(s):
            return s(prompt, temperature)
        if isinstance(s, Mapping):
            for key, val in s.items():
                if key in prompt:
                    if isinstance(val, str):
                        return val
                    n = self._per_key.get(key, 0)
                    self._per_key[key] = n + 1
                    return val[min(n, len(val) - 1)]
            if self.default is not None:
                return self.default
            raise ScriptExhausted(
                f"no script key matches this prompt: {prompt[:160]!r}")
        if not s:
            raise ScriptExhausted("empty script")
        if self._i >= len(s):
            if not self.cycle:
                raise ScriptExhausted(
                    f"script of {len(s)} answers exhausted on call "
                    f"{self._i + 1}")
            self._i = 0
        out = s[self._i]
        self._i += 1
        return out

    def describe(self) -> dict:
        return {"adapter": "scripted", "name": self.name,
                "calls": len(self.calls)}
