"""Vector storage and search. numpy if present, pure stdlib if not.

Deliberately brute-force. At a few tens of thousands of nodes on a laptop that
is a handful of milliseconds, and it has no index to corrupt, no rebuild step,
and no extra dependency. When a store outgrows it, swap in an ANN index behind
`VectorIndex` -- which is the only place in the codebase that knows how
similarity is computed.
"""
from __future__ import annotations

import array
import hashlib
import math
from typing import Iterable, Sequence

try:                                   # optional, and genuinely optional
    import numpy as _np
except ImportError:                    # pragma: no cover
    _np = None

from .protocols import Space

# How much of the WRITE vector is context rather than meaning.
# Two identical texts from different episodes end up at cosine ~= 1 - BETA**2,
# i.e. ~0.88 at BETA=0.35: still recognisably the same subject, but no longer
# collapsed onto each other.
SEPARATION_BETA = 0.35

# Context signatures are random projections, and random vectors are only
# reliably near-orthogonal in enough dimensions. This was originally derived
# from the semantic dim (`max(8, dim // 8)`), which is wrong: with a small
# encoder it produced 8-dim signatures whose worst-case |cos| was 0.93, so
# two identical texts from different episodes scored 0.99 in the write space
# and separation silently vanished. Measured worst case over 400 pairs:
#
#     dim   8 -> |cos| up to 0.93   write-sim 0.99   (separation gone)
#     dim  64 -> |cos| up to 0.34   write-sim 0.92
#     dim 128 -> |cos| up to 0.28   write-sim 0.91   <- chosen
#
# Fixed and independent of the encoder. 128 floats is 512 bytes per node.
CONTEXT_DIM = 128

# How hard to discount a document for being a hub. A hub is the nearest
# neighbour of many unrelated queries, so its similarity carries less
# information than the same number from a discriminating document.
HUBNESS_WEIGHT = 0.6
HUBNESS_SAMPLE = 48


def context_signature(dim: int, *parts: object) -> list[float]:
    """A deterministic pseudo-random unit vector for a context tuple.

    Random projections of distinct tuples are near-orthogonal in any
    reasonable dimension, which is exactly what pattern separation needs.
    """
    seed = "\x1f".join("" if p is None else str(p) for p in parts).encode()
    vec = [0.0] * dim
    block, i = b"", 0
    while i < dim:
        block = hashlib.blake2b(seed + i.to_bytes(4, "little"),
                                digest_size=64).digest()
        for j in range(0, len(block) - 1, 2):
            if i >= dim:
                break
            vec[i] = (int.from_bytes(block[j:j + 2], "little") / 32767.5) - 1.0
            i += 1
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def unit(vec: Sequence[float]) -> list[float]:
    """L2-normalise. Applied at the embedder boundary, not left to callers."""
    n = math.sqrt(sum(float(x) * float(x) for x in vec))
    return [float(x) / n for x in vec] if n > 0 else [0.0] * len(vec)


def separate(semantic: Sequence[float], context: Sequence[float],
             beta: float = SEPARATION_BETA) -> list[float]:
    """Build the WRITE vector: mostly meaning, deliberately some context.

    Concatenation rather than a prefix in the text. Prepending "[week2]" to
    the string and hoping a mean-pooled encoder notices is not separation --
    it is a wish. This makes the effect structural, controllable by one
    constant, and identical across every embedding model.

    BOTH inputs are normalised first. Embedders do not agree on output scale:
    BGE-M3 through llama.cpp returns vectors of norm ~25, and against that
    the 0.35-weighted context component is 1.5% of the vector -- so
    separation silently vanished and identical text from different episodes
    scored 0.9998. The toy embedder returned unit vectors and hid it
    completely. Normalise; never assume.
    """
    alpha = math.sqrt(max(0.0, 1.0 - beta * beta))
    sem, ctx = unit(semantic), unit(context)
    return [alpha * x for x in sem] + [beta * y for y in ctx]


def pack(vec: Sequence[float]) -> bytes:
    """L2-normalise and serialise. Normalising at write time means search is
    a dot product rather than a cosine, which is measurably faster in Python."""
    n = math.sqrt(sum(float(x) * float(x) for x in vec))
    if n <= 0:
        n = 1.0
    return array.array("f", [float(x) / n for x in vec]).tobytes()


def unpack(blob: bytes) -> array.array:
    a = array.array("f")
    a.frombytes(blob)
    return a


def dot(a: array.array, b: array.array) -> float:
    if _np is not None:
        return float(_np.dot(_np.frombuffer(a.tobytes(), dtype=_np.float32),
                             _np.frombuffer(b.tobytes(), dtype=_np.float32)))
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """The only place that knows how similarity is computed."""

    def __init__(self, store):
        self._s = store

    def put(self, node_id: str, space: Space, vec: Sequence[float],
            model: str) -> None:
        blob = pack(vec)
        self._s.write(lambda c: c.execute(
            "INSERT OR REPLACE INTO vector(node_id,space,dim,model,data) "
            "VALUES(?,?,?,?,?)", (node_id, space.value, len(vec), model, blob)))

    def put_many(self, rows: Iterable[tuple[str, Space, Sequence[float], str]]
                 ) -> None:
        payload = [(n, sp.value, len(v), m, pack(v)) for n, sp, v, m in rows]
        if not payload:
            return
        self._s.write(lambda c: c.executemany(
            "INSERT OR REPLACE INTO vector(node_id,space,dim,model,data) "
            "VALUES(?,?,?,?,?)", payload))

    def search(self, query_vec: Sequence[float], *, space: Space,
               allowed: Iterable[str] | None = None, top_k: int = 40,
               floor: float = 0.15, hubness: bool = True,
               model: str | None = None) -> list[tuple[str, float]]:
        """Similarity search, discounted for hubness.

        A document that is the nearest neighbour of many unrelated queries
        carries less information in the same cosine than a discriminating
        one does. Without this, a bland filler note ("Solar panels on block C
        were wiped down") outranked the real answer to "what part is the
        borehole missing" -- and the same hub pulled an unrelated query up
        into KNOW_WHERE.

        `model` restricts the comparison to vectors the CURRENT encoder
        produced. Two encoders put vectors in unrelated spaces, and swapping
        models on an existing store otherwise compares a Qwen3 query against
        BGE-M3 documents -- a cosine between coordinate systems, which is a
        number rather than an error. Dimensions usually differ too (4096 vs
        1024), so the result is not even meaningful garbage. Silent, and
        exactly the failure class this library exists to prevent.
        """
        q = unpack(pack(query_vec))
        sql = ("SELECT v.node_id, v.data, v.hubness FROM vector v "
               "JOIN mem_index m ON m.node_id = v.node_id "
               "WHERE v.space=? AND m.tier<>'pruned'")
        params: tuple = (space.value,)
        if model is not None:
            sql += " AND v.model=?"
            params += (model,)
        rows = self._s.query(sql, params)
        allow = set(allowed) if allowed is not None else None
        out: list[tuple[str, float]] = []
        for r in rows:
            nid = r["node_id"]
            if allow is not None and nid not in allow:
                continue
            sim = dot(q, unpack(r["data"]))
            if hubness and r["hubness"]:
                sim -= HUBNESS_WEIGHT * r["hubness"]
            if sim >= floor:
                out.append((nid, sim))
        out.sort(key=lambda x: -x[1])
        return out[:top_k]

    def recompute_hubness(self, space: Space = Space.READ,
                          sample: int = HUBNESS_SAMPLE) -> int:
        """Measure how close each document is to everything else.

        Deliberately relative to the corpus MEAN, so it is zero for a typical
        document and positive only for genuine hubs -- a global offset would
        change nothing.
        """
        rows = self._s.query(
            "SELECT node_id, data FROM vector WHERE space=?", (space.value,))
        if len(rows) < 8:
            return 0
        vecs = [(r["node_id"], unpack(r["data"])) for r in rows]
        step = max(1, len(vecs) // sample)
        probe = vecs[::step][:sample]
        raw: list[tuple[str, float]] = []
        for nid, v in vecs:
            sims = [dot(v, p) for pid, p in probe if pid != nid]
            raw.append((nid, sum(sims) / len(sims) if sims else 0.0))
        mean = sum(h for _, h in raw) / len(raw)
        payload = [(max(0.0, h - mean), nid, space.value) for nid, h in raw]
        self._s.write(lambda c: c.executemany(
            "UPDATE vector SET hubness=? WHERE node_id=? AND space=?", payload))
        return len(payload)

    def neighbours(self, node_id: str, *, space: Space = Space.WRITE,
                   threshold: float = 0.86, limit: int = 12
                   ) -> list[tuple[str, float]]:
        """Semantic confusability -- the input to the de-interference sweep.

        Note this runs in the WRITE space by default. Two memories that are
        confusable are ones that landed close together *despite* separation
        having been applied, which is a much stronger signal than closeness in
        a single blended space.
        """
        me = self._s.one(
            "SELECT data FROM vector WHERE node_id=? AND space=?",
            (node_id, space.value))
        if me is None:
            return []
        return [(n, s) for n, s in
                self.search(unpack(me["data"]), space=space, top_k=limit + 1,
                            floor=threshold) if n != node_id][:limit]

    def count(self) -> int:
        row = self._s.one("SELECT COUNT(*) FROM vector")
        return int(row[0]) if row else 0
