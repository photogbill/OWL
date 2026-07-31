"""G3 recall caching, G4 incremental maintenance.

Both are the same shape of risk: a speedup that is only correct if you get
invalidation exactly right, and whose failure mode is SILENTLY SERVING
STALE ANSWERS. For an engine whose entire pitch is knowing what it knows,
a cache that returns yesterday's answer with today's confidence is worse
than no cache.

So both are built around one rule:

    **A cache must be wrong in the safe direction.**

Over-invalidation costs a recomputation. Under-invalidation costs
correctness. Every ambiguous case here resolves to invalidate.

G3 -- KEYED ON (query, partition, clock bucket, write generation).

Repeat queries within a session are extremely common -- an agent asks "what
do I know about X" at the top of every turn. But recall is time-dependent
(retrievability decays) AND write-dependent, so the key needs both. The
write generation is a single counter bumped by any write: it makes
invalidation O(1) and total, rather than trying to work out which cached
queries a given write could have affected. Working that out is where cache
bugs live.

G4 -- `tend()` CURRENTLY RESCANS THE WHOLE STORE.

Every idle pass costs O(store) even when three memories changed. Dirty
tracking makes it O(changes). The trap is that consolidation passes are not
independent -- fusion changes what interference sees -- so a naive dirty
set misses second-order work. The answer is to let passes ENLARGE the
dirty set as they run, and to keep a full-sweep escape hatch that runs
periodically regardless, because a dirty-set bug that loses work would be
invisible exactly the way this project keeps catching.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# Recall depends on elapsed time through retrievability decay. A bucket
# coarse enough to hit, fine enough that decay inside one bucket cannot
# change an answer: retrievability moves by well under a percent in a
# minute at any realistic stability.
CLOCK_BUCKET = 60.0

FULL_SWEEP_EVERY = 20


@dataclass
class RecallCache:
    max_entries: int = 256
    bucket: float = CLOCK_BUCKET
    _entries: dict = field(default_factory=dict)
    generation: int = 0
    hits: int = 0
    misses: int = 0
    invalidations: int = 0

    def key(self, query: str, partition: str, now: float, **kw) -> tuple:
        extra = tuple(sorted((k, v) for k, v in kw.items()
                             if v is not None and not callable(v)))
        return (query, partition, int(now // self.bucket), self.generation,
                extra)

    def get(self, k):
        got = self._entries.get(k)
        if got is None:
            self.misses += 1
            return None
        self.hits += 1
        return got

    def put(self, k, value) -> None:
        if self.max_entries <= 0:
            return                      # caching disabled; not an error
        if len(self._entries) >= self.max_entries:
            # Drop the oldest insertion. Python dicts preserve insertion
            # order, so this is FIFO -- deliberately not LRU, because LRU
            # bookkeeping on a 256-entry cache costs more than the misses
            # it prevents.
            self._entries.pop(next(iter(self._entries)))
        self._entries[k] = value

    def invalidate(self) -> None:
        """Any write invalidates everything. Total, and O(1).

        The alternative -- working out which cached queries a given write
        could have affected -- is exactly where cache bugs live, and the
        bug is invisible: a stale answer looks like a fresh one. Bumping a
        generation counter makes every existing key unreachable at once.
        """
        self.generation += 1
        self._entries.clear()
        self.invalidations += 1

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses, "invalidations": self.invalidations,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "generation": self.generation}


@dataclass
class DirtySet:
    """What has changed since the last maintenance pass.

    Passes may ADD to it while running -- fusing two memories changes what
    the interference sweep should look at, and a dirty set fixed at entry
    would miss that second-order work.
    """

    nodes: set = field(default_factory=set)
    passes_since_full: int = 0
    full_sweep_every: int = FULL_SWEEP_EVERY
    _last_full: float = 0.0

    def mark(self, node_id: str) -> None:
        if node_id:
            self.nodes.add(node_id)

    def mark_many(self, ids) -> None:
        self.nodes.update(i for i in ids if i)

    @property
    def needs_full_sweep(self) -> bool:
        """A periodic full pass, regardless.

        Not belt-and-braces nervousness. A dirty-set bug loses consolidation
        work SILENTLY -- the store simply never notices two memories
        interfering -- which is precisely the failure mode this project has
        caught four times in other guises. The full sweep is the thing that
        would eventually surface it.
        """
        return self.passes_since_full >= self.full_sweep_every

    def take(self, *, all_nodes) -> tuple[set, str]:
        """The working set for this pass, and why it is that.

        Returns everything on a full sweep, the dirty set otherwise.
        """
        if self.needs_full_sweep or not self.nodes:
            self.passes_since_full = 0
            self._last_full = time.time()
            scope = set(all_nodes)
            why = ("full sweep: periodic re-check, because a dirty-set bug "
                   "would lose consolidation work silently"
                   if self.nodes else
                   "full sweep: nothing marked dirty, so this is the cheap "
                   "case anyway")
            self.nodes.clear()
            return scope, why
        self.passes_since_full += 1
        scope = set(self.nodes)
        self.nodes.clear()
        return scope, (f"incremental: {len(scope)} changed node(s) of "
                       f"{len(all_nodes)}")

    @property
    def stats(self) -> dict:
        return {"pending": len(self.nodes),
                "passes_since_full": self.passes_since_full,
                "full_sweep_every": self.full_sweep_every}
