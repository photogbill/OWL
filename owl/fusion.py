"""Record fusion — clustering and composites with zero model calls.

LycheeMem's Record Fusion Engine, reimplemented on OWL's substrate:

    1. dedupe    -- near-identical records (cosine > 0.85) are soft-expired
    2. cluster   -- union-find over the similarity graph (cosine > 0.75)
    3. composite -- each component becomes one denser node
    4. hierarchy -- the same pass runs over composites, growing a tree upward

All four steps are arithmetic. No LLM, no prompt, no latency, no non-
determinism -- which matters more than it sounds given MiniRAG's finding that
model-dependent pipelines go NEGATIVE below about 7B (see MINIRAG_NOTES.md).

Two things OWL adds:

  * A composite is a `derived` node, so the monotonicity invariant applies to
    it: it can never be more certain than its least certain member, and
    `why()` walks into it like any other derivation.

  * `verbatim` content is excluded from fusion entirely. A grid reference and
    a serial number are not "near-duplicates" to be merged, no matter what
    the cosine says.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEDUPE_THRESHOLD = 0.85
CLUSTER_THRESHOLD = 0.75
MIN_CLUSTER = 2
MAX_LEVELS = 3


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def components(self) -> dict:
        out: dict = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return out


@dataclass
class FusionPlan:
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)
    skipped_verbatim: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.duplicates and not self.clusters


def plan(pairs: list[tuple[str, str, float]], *,
         verbatim: set[str] | None = None,
         dedupe_at: float = DEDUPE_THRESHOLD,
         cluster_at: float = CLUSTER_THRESHOLD,
         calibration=None) -> FusionPlan:
    """Turn a list of (a, b, similarity) into a dedupe + cluster plan.

    Deliberately takes pairs rather than reaching for a store, so the whole
    algorithm is testable with a handful of tuples and no database.

    `calibration` is an optional `adapters.calibration.Calibration`. 0.85 and
    0.75 encode "near-identical" and "clearly related" on the assumption
    that unrelated text scores near zero. Measured on Qwen3-Embedding-8B,
    unrelated documents sit at 0.406 mean / 0.529 p95 -- so 0.75, meant to
    be three quarters of the way from chance to identical, is about a fifth
    of the way.

    That is a shrunken margin rather than an active bug: the tail of a cone
    is tight, so at 0.40 background nothing unrelated actually reaches 0.75.
    It becomes a real false-merge source around 0.65 background, which a
    heavier quant or a smaller model will reach. Given a calibration the
    thresholds move into the encoder's real range and the intent survives
    the change of model. See `bench/scoreboard.py::fusion_false_merge`.
    """
    if calibration is not None:
        dedupe_at, cluster_at = calibration.fusion_thresholds(dedupe_at,
                                                              cluster_at)
    protect = verbatim or set()
    out = FusionPlan()
    survivors: set[str] = set()
    usable: list[tuple[str, str, float]] = []

    for a, b, sim in pairs:
        if a in protect or b in protect:
            out.skipped_verbatim += 1
            continue
        usable.append((a, b, sim))
        survivors.add(a)
        survivors.add(b)

    merged: set[str] = set()
    for a, b, sim in usable:
        if sim >= dedupe_at and a not in merged and b not in merged:
            keep, drop = sorted((a, b))     # deterministic; caller may re-rank
            out.duplicates.append((keep, drop))
            merged.add(drop)

    live = [s for s in survivors if s not in merged]
    if len(live) < MIN_CLUSTER:
        return out
    uf = UnionFind(live)
    for a, b, sim in usable:
        if sim >= cluster_at and a in uf.parent and b in uf.parent:
            uf.union(a, b)
    for members in uf.components().values():
        if len(members) >= MIN_CLUSTER:
            out.clusters.append(sorted(members))
    return out
