"""G5 -- partition-sharded storage.

    *A large work partition never slows a small private one.*

Partitions are already an ENFORCED boundary: `readable_from()` decides what
a query may see and the SQL filters on it. What partitions were not was a
STORAGE boundary. Every hot read walked the whole table and discarded the
rows it was not allowed to see, so the cost of a question asked inside a
200-memory private partition was set by the 200,000-memory work partition
next to it. The answer was always right. The bill was someone else's.

That is worse than a performance defect, because of what the private
partition IS. It is the confidential one -- the companion context, the
sealed clinical notes, the thing whose entire justification is that it is
separate. Making its latency a function of the work partition's size leaks
the work partition's size into it. A boundary that holds for content and
leaks for timing is a boundary with a side channel, and this engine does
not get to call that acceptable.

WHY LOGICAL SHARDS AND NOT A FILE PER PARTITION

A file per partition is the obvious reading of "sharded" and it is the
wrong one here. It would cost: the single-file store (`.owl` is one file
you can copy, mail, or mount read-only), one-writer atomicity across a
write that touches two partitions, the append-only trigger's coverage, the
joins that make graded permeability work at all -- summary-level flow
requires reading derived rows from a partition you cannot read raw from --
and `.owlpack` handover. That is most of the substrate, traded for an
isolation property that an index gives for free.

So: one file, and the partition becomes a leading index column. The unit
of work stops being the table and starts being the shard.

    lexeme(partition, term, node_id, tf)   <- covering; the posting list
                                              for a term IS per-partition
    vector(space, partition)               <- blobs outside the visible set
                                              are never read off disk
    mem_index(partition, tier)             <- counts and scope by seek

THE DENORMALISATION IS DELIBERATE

`partition` already lives on `mem_index`, and every one of these tables
joins to it. Copying the column onto `lexeme` and `vector` is duplicated
state, which is normally a bug waiting to happen. It is here because the
join is exactly the thing being paid for: with the partition only on
`mem_index`, SQLite must visit every posting for a term -- all partitions
-- and join each one before it can discard it. The filter cannot run
before the scan it is meant to prevent. Denormalising moves the predicate
in front of the scan, which is the entire optimisation.

The duplicate is safe because a node's partition is IMMUTABLE. Nothing in
OWL moves a memory between partitions -- there is no API for it, and there
should not be, because a memory that could change partition would let
confidential material walk out of a sealed one. So the copy cannot drift:
it is written once with the row and never updated. `verify()` below checks
that claim against the store rather than trusting this paragraph.

WHAT THIS FILE DOES NOT DO

It does not change a single answer. Every predicate here is one OWL was
already applying after the fact; the only difference is when. The tests in
`tests/test_shards.py` pin that by running the same corpus and queries
against a sharded and an unsharded store and demanding identical output.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .fastpath import Bloom

# Tables carrying a denormalised partition, and where to backfill it from.
# `mem_index` is the authority: it is written in the same transaction as
# the substrate row, so it cannot disagree with it.
SHARDED_TABLES = ("lexeme", "vector")

SHARD_DDL = (
    # Covering: term lookup inside a partition never touches the table.
    "CREATE INDEX IF NOT EXISTS idx_lexeme_shard "
    "ON lexeme(partition, term, node_id, tf)",
    # Brute-force search reads BLOBs. This is the index that stops it
    # reading the ones it is not allowed to look at.
    "CREATE INDEX IF NOT EXISTS idx_vector_shard "
    "ON vector(space, partition)",
    # Partition-first. `idx_index_tier` is (tier, partition), which serves
    # "everything cold" and cannot serve "this partition's nodes".
    "CREATE INDEX IF NOT EXISTS idx_index_shard "
    "ON mem_index(partition, tier)",
)

# A private partition's filter should cost kilobytes, not the 128 KB the
# single global filter took. Start small, grow by doubling when load would
# push the false-positive rate past the target.
MIN_BITS = 1 << 16              # 8 KB, comfortable to ~6.5k terms
MAX_BITS = 1 << 22              # 512 KB, ~420k terms; a very large shard
BITS_PER_TERM = 10              # m/n = 10 at k=7 gives FP ~= 0.008


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def is_sharded(conn: sqlite3.Connection) -> bool:
    """Does this store carry the denormalised partition columns?

    Asked rather than assumed, because a store opened READ-ONLY cannot be
    migrated and must still answer questions -- F2 is not negotiable for a
    speedup. The read paths keep a legacy predicate for exactly this case.
    """
    try:
        return all("partition" in _columns(conn, t) for t in SHARDED_TABLES)
    except sqlite3.Error:                       # not an OWL store yet
        return False


def migrate(conn: sqlite3.Connection) -> dict:
    """Bring an existing store up to the sharded layout. Idempotent.

    Runs on open, inside the connection that just created the schema, so a
    store written by an older version starts answering faster with no user
    action and no separate command to forget. Reported rather than silent:
    the doctor surfaces it, because a migration that ran is a fact about
    the store and this engine does not do invisible state changes.
    """
    report = {"migrated": [], "rows": 0, "indexes": 0}
    for table in SHARDED_TABLES:
        if "partition" in _columns(conn, table):
            continue
        # A NOT NULL default is required by ALTER TABLE ADD COLUMN, and
        # 'default' is the right one: it is the partition every store has,
        # so a row whose node somehow has no index entry degrades to the
        # ordinary partition rather than to an unreadable one.
        conn.execute(f"ALTER TABLE {table} ADD COLUMN partition TEXT "
                     f"NOT NULL DEFAULT 'default'")
        n = conn.execute(
            f"UPDATE {table} SET partition = COALESCE("
            f"  (SELECT m.partition FROM mem_index m "
            f"   WHERE m.node_id = {table}.node_id), 'default')").rowcount
        report["migrated"].append(table)
        report["rows"] += max(0, n)
    for ddl in SHARD_DDL:
        conn.execute(ddl)
        report["indexes"] += 1
    return report


def verify(conn: sqlite3.Connection) -> list[str]:
    """Check the denormalised copy against the authority. Diagnostic.

    The whole safety argument for duplicating `partition` is that a node
    never changes partition, so the copy cannot drift. That is an argument,
    and arguments are what this project checks with code. Any row here is a
    genuine confidentiality defect -- a lexeme filed under the wrong
    partition is a term visible from a partition that must not see it --
    so this is wired into `doctor`, not left as a comment.
    """
    if not is_sharded(conn):
        return []
    problems = []
    for table in SHARDED_TABLES:
        rows = conn.execute(
            f"SELECT COUNT(*) FROM {table} t JOIN mem_index m "
            f"ON m.node_id = t.node_id WHERE t.partition <> m.partition"
        ).fetchone()[0]
        if rows:
            problems.append(
                f"{rows} {table} row(s) carry a partition that disagrees "
                f"with mem_index -- content is filed under a partition it "
                f"does not belong to")
    return problems


def partition_col(alias: str, sharded: bool) -> str:
    """Which column carries the partition for a scoped read.

    One function rather than two query bodies. The sharded and legacy paths
    differ by this token and nothing else, so there is no second
    implementation to drift -- the predicate is identical, only its
    position in the plan changes.
    """
    return f"{alias}.partition" if sharded else "m.partition"


@dataclass
class Shard:
    """One partition's share of the fast path."""

    partition: str
    bloom: Bloom = field(default_factory=lambda: Bloom(bits=MIN_BITS))
    n_nodes: int | None = None
    built: bool = False
    rebuilds: int = 0

    @property
    def saturated(self) -> bool:
        """Would another term push the false-positive rate off target?

        A filter that silently fills up does not fail -- it just stops
        rejecting, degrading to the scan it existed to prevent while still
        reporting itself as working. Growth is measured, not hoped for.
        """
        return (self.bloom.n_added + 1) * BITS_PER_TERM > self.bloom.bits


class ShardSet:
    """Per-partition fast-path state: vocabulary filter and node count.

    The single global Bloom filter this replaces had two faults, and only
    one of them was speed.

    THE SLOW ONE: its size, and therefore its false-positive rate, was set
    by the whole store's vocabulary. A private partition holding forty
    distinct words inherited the work partition's collision rate.

    THE WRONG ONE: it answered "is this term anywhere in the STORE", and
    the caller used the answer to decide whether to scan A PARTITION.
    Those are different questions. A term present only in `work` made the
    filter say "possibly present" to a query inside `private`, which then
    scanned and found nothing -- the fast path declining to fire in
    precisely the case it was built for.

    Per-partition filters answer the question that was actually being
    asked.
    """

    def __init__(self, *, load_terms: Callable[[str], Iterable[str]],
                 count_nodes: Callable[[str], int], sharded: bool = True):
        self._load_terms = load_terms
        self._count_nodes = count_nodes
        # An unmigrated store cannot enumerate terms per partition, so it
        # gets one shard standing for the whole store -- the old behaviour,
        # kept honest by being named.
        self.sharded = sharded
        self._shards: dict[str, Shard] = {}

    # ── shard access ─────────────────────────────────────────────────
    def _key(self, partition: str) -> str:
        return partition if self.sharded else "*"

    def shard(self, partition: str) -> Shard:
        k = self._key(partition)
        got = self._shards.get(k)
        if got is None:
            got = self._shards[k] = Shard(partition=k)
        return got

    # ── G1, now per partition ────────────────────────────────────────
    def _build(self, sh: Shard) -> None:
        terms = list(self._load_terms(sh.partition))
        # Size for what is actually there, once, rather than growing into
        # it one doubling at a time on the first query.
        want = max(MIN_BITS, min(MAX_BITS,
                                 1 << max(len(terms) * BITS_PER_TERM,
                                          MIN_BITS).bit_length()))
        sh.bloom = Bloom(bits=want)
        for t in terms:
            sh.bloom.add(t)
        sh.built = True

    def note_term(self, partition: str, term: str) -> None:
        """Record a term written into this partition.

        THE BUG THIS EXISTS TO CLOSE, recorded because it was live and it
        was invisible. The filter used to be built once, on the first
        recall, and never updated -- on the reasoning that adding a term
        can only move the filter towards "possibly present", which is the
        safe direction. True of adding to the FILTER. The code added to the
        STORE and not the filter, which is the other direction: a term
        written after the filter was built was DEFINITELY ABSENT according
        to it, so the posting scan was skipped and a memory the store
        genuinely held came back DONT_KNOW.

        It needed no embedder to reproduce, it never raised, and it looked
        exactly like ordinary forgetting. Every write now feeds the filter
        in the same breath as the index.
        """
        sh = self.shard(partition)
        if not sh.built:
            return          # will be built from SQL on first use; nothing lost
        if sh.saturated:
            # Rebuild at the next size up rather than degrading quietly.
            sh.rebuilds += 1
            sh.built = False
            return
        sh.bloom.add(term)

    def note_terms(self, partition: str, terms: Iterable[str]) -> None:
        for t in terms:
            self.note_term(partition, t)

    def definitely_unknown(self, terms: list[str],
                           partitions: Iterable[str]) -> bool:
        """Is the posting-list scan CERTAIN to return nothing?

        True only when every visible partition's filter rules out every
        term. A Bloom filter's one error is the false positive, so this can
        never wrongly claim ignorance -- and the caller must keep it
        narrow: this is evidence about the lexical index of these
        partitions and nothing else. The semantic tier, the gap
        explanation, recorded absences and receipts all still run. That
        distinction is the subject of a long comment in
        `Owl._definitely_unknown` and it was learned the hard way.
        """
        if not terms:
            return False
        for p in partitions:
            sh = self.shard(p)
            if not sh.built:
                self._build(sh)
            if sh.bloom.any_present(terms):
                return False
        return True

    # ── node counts, per partition ───────────────────────────────────
    def count(self, partition: str) -> int:
        sh = self.shard(partition)
        if sh.n_nodes is None:
            sh.n_nodes = self._count_nodes(sh.partition)
        return sh.n_nodes

    def total(self, partitions: Iterable[str]) -> int:
        return sum(self.count(p) for p in partitions)

    # ── invalidation ─────────────────────────────────────────────────
    def touch(self, partition: str) -> None:
        """A write landed in this partition.

        Counts go; the filter does NOT. A stale count changes an IDF
        weight, which changes a ranking. A dropped filter costs a rebuild
        of the whole partition's vocabulary on the next query, which is the
        cost this module exists to avoid paying -- so the filter is kept
        current incrementally through `note_term` instead.
        """
        self.shard(partition).n_nodes = None

    def invalidate(self, partition: str | None = None) -> None:
        if partition is None:
            self._shards.clear()
            return
        self._shards.pop(self._key(partition), None)

    @property
    def stats(self) -> dict:
        return {
            "sharded": self.sharded,
            "shards": len(self._shards),
            "partitions": {
                k: {"terms": s.bloom.n_added,
                    "bits": s.bloom.bits,
                    "false_positive_rate": s.bloom.false_positive_rate,
                    "nodes": s.n_nodes,
                    "rebuilds": s.rebuilds}
                for k, s in sorted(self._shards.items())
            },
        }
