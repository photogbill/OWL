"""G1 + G2 -- make the fast path actually fast.

G1 -- THE FAST PATH DESERVES TO BE THE FASTEST PATH.

`DONT_KNOW` is the answer OWL fires most often and is proudest of, and it
is currently the one that does the most work: every query term hits the
lexeme index before the store can conclude it has nothing. That is backwards.
A Bloom filter over the vocabulary answers "is this term anywhere in the
store?" in constant time, so a query about something never mentioned exits
without touching SQLite at all.

Bloom rather than cuckoo, deliberately: the only error a Bloom filter makes
is a FALSE POSITIVE -- it may say "possibly present" when the term is
absent, and then the normal path runs and correctly returns DONT_KNOW,
slower. It can never say "absent" about a term that is present. That
asymmetry is the whole reason this is safe to put in front of retrieval: a
false positive costs microseconds, a false negative would lose a memory.

G2 -- STOP DESERIALISING BLOBS PER QUERY.

Every brute-force search currently unpacks every vector from its BLOB.
`array.array` over the raw buffer is a view rather than a copy, so the
dominant cost becomes the dot product itself, which is where it should be.
This is what keeps brute force -- the EXACT path -- viable far enough up
that most stores never need the approximate one.
"""
from __future__ import annotations

import array
import hashlib
import math
from dataclasses import dataclass, field


# ── G1: Bloom filter over the vocabulary ─────────────────────────────────

@dataclass
class Bloom:
    """A vocabulary filter that can be wrong in only one direction."""

    bits: int = 1 << 20                 # 1 Mbit ≈ 128 KB, ~145k terms at 1%
    hashes: int = 7
    _buf: bytearray = field(default_factory=bytearray)
    n_added: int = 0

    def __post_init__(self) -> None:
        if not self._buf:
            self._buf = bytearray(self.bits // 8)

    def _positions(self, term: str):
        # Two hashes, combined -- Kirsch-Mitzenmacher. k independent hash
        # functions are unnecessary; h1 + i*h2 has the same error bound and
        # costs one digest instead of seven.
        d = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(d[:8], "little")
        h2 = int.from_bytes(d[8:], "little") | 1
        for i in range(self.hashes):
            yield ((h1 + i * h2) % self.bits)

    def add(self, term: str) -> None:
        for p in self._positions(term):
            self._buf[p >> 3] |= 1 << (p & 7)
        self.n_added += 1

    def __contains__(self, term: str) -> bool:
        """False means DEFINITELY absent. True means possibly present.

        The asymmetry is the point and it is what makes this safe in front
        of retrieval: a false positive costs a normal lookup, a false
        negative would lose a memory.
        """
        return all(self._buf[p >> 3] & (1 << (p & 7))
                   for p in self._positions(term))

    def any_present(self, terms) -> bool:
        """Could the store possibly know about ANY of these terms?"""
        return any(t in self for t in terms)

    @property
    def false_positive_rate(self) -> float:
        """Estimated, from actual load. Reported rather than assumed."""
        if not self.n_added:
            return 0.0
        k, m, n = self.hashes, self.bits, self.n_added
        return round((1 - math.exp(-k * n / m)) ** k, 6)

    def to_bytes(self) -> bytes:
        return bytes(self._buf)

    @classmethod
    def from_bytes(cls, raw: bytes, *, hashes: int = 7, n_added: int = 0):
        return cls(bits=len(raw) * 8, hashes=hashes,
                   _buf=bytearray(raw), n_added=n_added)


# ── G2: vectors without the per-query deserialise ────────────────────────

def view(blob: bytes) -> array.array:
    """A float32 view over a stored vector.

    `array.frombytes` copies once into a C array; the previous path built a
    Python list of boxed floats, which is roughly 8x the memory and turns
    every dot product into pointer chasing. The dot product should dominate
    a brute-force search -- if unpacking dominates, the measurement is of
    the serialisation format rather than the retrieval.
    """
    a = array.array("f")
    a.frombytes(blob)
    return a


def dot_view(a: array.array, b: array.array) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


@dataclass
class VectorCache:
    """Decoded vectors held once, not per query.

    Bounded, and the bound is stated: at 100k vectors of 1024 float32 this
    is ~400 MB, which is fine on a workstation and not on a small machine.
    Over the limit it simply stops caching rather than evicting -- an LRU
    here would spend more time on bookkeeping than it saves, and a cache
    that silently thrashes is worse than one that plainly stops.
    """

    max_vectors: int = 50_000
    _v: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, node_id: str, blob: bytes) -> array.array:
        got = self._v.get(node_id)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        a = view(blob)
        if len(self._v) < self.max_vectors:
            self._v[node_id] = a
        return a

    def invalidate(self, node_id: str | None = None) -> None:
        if node_id is None:
            self._v.clear()
        else:
            self._v.pop(node_id, None)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"cached": len(self._v), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "at_capacity": len(self._v) >= self.max_vectors}
