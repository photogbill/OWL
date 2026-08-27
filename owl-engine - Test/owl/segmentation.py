"""Event segmentation at surprise boundaries.

Almost every memory system chunks by size -- 512 tokens, a paragraph, a turn.
That is an artifact of the tokenizer, not a property of the material. Human
memory segments experience into *events* at moments of prediction error
(Event Segmentation Theory; Zacks & Tversky), and EM-LLM showed that
LLM-derived surprise reproduces human-perceived event boundaries closely
enough to improve long-context retrieval.

OWL segments the observation stream the same way. An episode boundary is
declared when surprise exceeds a running threshold, which means episodes are
as long as the material stays coherent and as short as it needs to be when
the subject changes. Retrieval can then return whole episodes rather than
arbitrary windows.

With no Reasoner configured this degrades to lexical novelty, which is crude
but still far better than fixed-size chunking -- and, importantly, it keeps
Tier 0 working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .lexical import tokenize


@dataclass
class Segmenter:
    """Online boundary detector. Feed it observations in order."""

    window: int = 8
    z_threshold: float = 1.0
    min_len: int = 2
    _recent: list[set[str]] = field(default_factory=list)
    _surprises: list[float] = field(default_factory=list)
    _since_boundary: int = 0

    def surprise(self, text: str) -> float:
        """Lexical novelty vs the running window. 0 = fully predicted, 1 = new."""
        toks = set(tokenize(text))
        if not toks:
            return 0.0
        if not self._recent:
            return 1.0
        seen: set[str] = set()
        for s in self._recent:
            seen |= s
        return len(toks - seen) / len(toks)

    def push(self, text: str) -> tuple[float, bool]:
        """Return (surprise, is_boundary) and advance the window."""
        s = self.surprise(text)
        boundary = self._is_boundary(s)
        self._surprises.append(s)
        if len(self._surprises) > 64:
            self._surprises.pop(0)
        # NOTE: the window is NOT cleared at a boundary. Clearing it makes the
        # next observation look maximally surprising against an empty context,
        # which fires a spurious second boundary immediately after every real
        # one. The sliding window ages the old episode out on its own.
        self._recent.append(set(tokenize(text)))
        if len(self._recent) > self.window:
            self._recent.pop(0)
        self._since_boundary = 0 if boundary else self._since_boundary + 1
        return s, boundary

    def _is_boundary(self, s: float) -> bool:
        if self._since_boundary < self.min_len:
            return False
        if len(self._surprises) < 3:        # warmup: never guess early
            return False
        mean = sum(self._surprises) / len(self._surprises)
        var = sum((x - mean) ** 2 for x in self._surprises) / len(self._surprises)
        sd = var ** 0.5
        if sd < 1e-6:
            return False
        return (s - mean) / sd > self.z_threshold
