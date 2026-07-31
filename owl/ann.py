"""B9 -- an approximate index behind `VectorIndex`, with no new dependency.

Brute force is O(n) per query and stays the default because at ten thousand
memories it is fast enough and *exact*. Past roughly 100k it stops being
fast enough, and that is what this is for.

WHY NOT zvec / faiss / hnswlib: OWL's install story is `pip install
owl-engine` with no compiler, no server, no wheels that might not exist for
your Python. Trading that for a speedup most stores will never need is a
bad deal. `VectorIndex` already abstracts search, so a caller who wants
faiss can implement four methods and pass it in -- what ships is a pure
stdlib IVF that needs nothing.

INVERTED FILE, not a graph. Cluster the vectors, keep a list per centroid,
and at query time scan only the `nprobe` nearest lists. HNSW is faster at
the same recall but needs a real implementation to be worth anything; IVF
in plain Python is ~200 lines and its failure mode is legible -- if recall
drops you raise nprobe, and the relationship is monotone.

THE HONEST PART. The plan's acceptance criterion says "identical results to
brute force at >=10x speed", and that is not achievable: an approximate
index is approximate by construction. Pretending otherwise would be the
same species of claim this project keeps catching. So `recall_at_k()`
MEASURES the loss against brute force, `search()` reports whether it was
exact, and the benchmark prints both numbers. You choose the trade with
the figure in front of you rather than discovering it in production.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class IvfIndex:
    """Inverted-file index over unit vectors. Pure stdlib."""

    n_lists: int = 0                # 0 = choose from corpus size
    nprobe: int = 8
    seed: int = 7
    centroids: list[list[float]] = field(default_factory=list)
    lists: list[list[str]] = field(default_factory=list)
    vectors: dict[str, list[float]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.vectors)

    def build(self, items: list[tuple[str, list[float]]], *,
              iters: int = 6) -> "IvfIndex":
        """k-means over the corpus. Deterministic given `seed`.

        Determinism is not incidental. A9 asks the same store to produce the
        same maintenance output, and an index that reshuffles on every
        rebuild makes retrieval differences impossible to attribute.
        """
        self.vectors = {nid: v for nid, v in items}
        n = len(items)
        if n == 0:
            self.centroids, self.lists = [], []
            return self
        k = self.n_lists or max(1, min(int(math.sqrt(n)), n))
        rng = random.Random(self.seed)
        dim = len(items[0][1])
        # k-means++ style seeding, cheap version: spread the initial picks.
        picks = rng.sample(range(n), k) if k <= n else list(range(n))
        cents = [list(items[i][1]) for i in picks]

        assign = [0] * n
        for _ in range(iters):
            moved = 0
            for i, (_, v) in enumerate(items):
                best, bi = -2.0, 0
                for ci, c in enumerate(cents):
                    s = _dot(v, c)
                    if s > best:
                        best, bi = s, ci
                if assign[i] != bi:
                    assign[i] = bi
                    moved += 1
            sums = [[0.0] * dim for _ in range(k)]
            counts = [0] * k
            for i, (_, v) in enumerate(items):
                c = assign[i]
                counts[c] += 1
                acc = sums[c]
                for d, x in enumerate(v):
                    acc[d] += x
            for ci in range(k):
                if not counts[ci]:
                    continue
                acc = sums[ci]
                norm = math.sqrt(sum(x * x for x in acc)) or 1.0
                cents[ci] = [x / norm for x in acc]
            if not moved:
                break

        self.centroids = cents
        self.lists = [[] for _ in range(k)]
        for i, (nid, _) in enumerate(items):
            self.lists[assign[i]].append(nid)
        return self

    def search(self, q: list[float], *, top_k: int = 40,
               floor: float = 0.0, allowed: set | None = None,
               nprobe: int | None = None) -> tuple[list, bool]:
        """Returns (hits, exact). `exact` is True only if every list was
        scanned, which is the one case the result is guaranteed complete."""
        if not self.centroids:
            return [], True
        probes = nprobe or self.nprobe
        order = sorted(range(len(self.centroids)),
                       key=lambda ci: -_dot(q, self.centroids[ci]))
        chosen = order[:max(1, probes)]
        exact = len(chosen) >= len(self.centroids)

        out = []
        for ci in chosen:
            for nid in self.lists[ci]:
                if allowed is not None and nid not in allowed:
                    continue
                s = _dot(q, self.vectors[nid])
                if s >= floor:
                    out.append((nid, s))
        out.sort(key=lambda x: -x[1])
        return out[:top_k], exact


def brute(q: list[float], items: list[tuple[str, list[float]]], *,
          top_k: int = 40, floor: float = 0.0) -> list:
    out = [(nid, _dot(q, v)) for nid, v in items]
    out = [x for x in out if x[1] >= floor]
    out.sort(key=lambda x: -x[1])
    return out[:top_k]


def recall_at_k(index: IvfIndex, items, queries, *, k: int = 10,
                nprobe: int | None = None) -> dict:
    """How much is actually lost. The number the plan's criterion needs.

    An approximate index cannot return identical results, so the useful
    question is what fraction of the true top-k survives -- and whether the
    top-1 in particular does, since that is what most callers act on.
    """
    hit = total = top1 = top3 = 0
    for q in queries:
        truth = [nid for nid, _ in brute(q, items, top_k=k)]
        got = [nid for nid, _ in index.search(q, top_k=k, nprobe=nprobe)[0]]
        if truth:
            hit += len(set(truth) & set(got))
            total += len(truth)
            top1 += bool(got) and got[0] == truth[0]
            # OWL returns 4-7 chunks by design, so how much of the HEAD
            # survives matters more than the tail of a top-10 nobody reads.
            # Reporting only recall@10 would understate a setting that is
            # perfectly good for the way this engine actually retrieves.
            top3 += len(set(truth[:3]) & set(got[:3])) / 3.0
    return {
        "recall_at_k": round(hit / total, 4) if total else 1.0,
        "top1_agreement": round(top1 / len(queries), 4) if queries else 1.0,
        "top3_agreement": round(top3 / len(queries), 4) if queries else 1.0,
        "k": k, "nprobe": nprobe or index.nprobe,
        "n_lists": len(index.centroids), "n_vectors": index.size,
    }
