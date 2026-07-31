"""Tier-0 lexical index. No embedder, no numpy, no model.

This exists so `pip install owl-engine` does something useful with zero
dependencies. It is deliberately simple: normalized term frequency with an
IDF weight computed from the real document frequency (not the self-referential
formula that ships in a lot of hand-rolled 'TF-IDF' implementations).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'_-]*")
_STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its it's he she they them their
i you we us our your my me do does did not no so such can will would should
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def term_frequencies(text: str) -> dict[str, float]:
    toks = tokenize(text)
    if not toks:
        return {}
    counts = Counter(toks)
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return {t: c / norm for t, c in counts.items()}


def idf(doc_freq: int, n_docs: int) -> float:
    if n_docs <= 0 or doc_freq <= 0:
        return 0.0
    return math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
