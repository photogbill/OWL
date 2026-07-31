"""G5 -- a large work partition never slows a small private one.

The acceptance criterion is a LATENCY claim, and latency claims are the
easiest kind to fake. So it is tested twice, in two different currencies:

  * WORK, counted -- how many rows a private-partition recall pulls into
    Python. Deterministic, and it is the quantity that actually scaled: the
    old vector path read every BLOB in the store and dropped the ones it
    was not allowed to see, in Python, after paying for them.
  * TIME, measured -- because the criterion is stated in time and a proxy
    that has quietly stopped tracking the real thing is worse than no
    proxy at all.

The second is the flaky one and it is deliberately loose. The first is the
one that would catch a regression.

There is a confidentiality argument here as well as a performance one, and
it is the reason this is not simply an optimisation. A private partition's
whole justification is being separate. Making its response time a function
of the work partition's size publishes the work partition's size into it --
a boundary that holds for content and leaks through timing is a boundary
with a side channel.
"""
import hashlib
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from owl import Owl, State, shards
from owl.protocols import Space
from owl.shards import ShardSet


class Toy:
    """Bag-of-tokens, and the SAME encoder every run.

    Deliberately not `hash(w)`: str hashing is salted per process, so that
    version was a different model on every run and turned partition tests
    into coin flips -- see the note on the same class in
    `test_performance.py`, where it cost two thirds of all seeds.
    """
    is_semantic = True
    name = "toy"
    DIM = 64

    @staticmethod
    def _bucket(word: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest(),
            "little") % Toy.DIM

    def embed(self, texts, space):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in t.lower().split():
                v[self._bucket(w)] += 1.0
            out.append(v or [1.0] * self.DIM)
        return out


def _mind(**kw):
    return Owl.open(os.path.join(tempfile.mkdtemp(), "p.owl"), **kw)


def _two_partitions(work_notes: int, *, embedder=None):
    """A small private partition beside a work partition of a given size."""
    m = _mind(embedder=embedder)
    m.partition("work")
    m.partition("private", sealed=True)
    for i in range(work_notes):
        m.observe(f"Work note {i}: convoy routing, fuel, depot logistics.",
                  partition="work", origin="document", source_ref="sitrep")
    m.observe("I keep replaying the fourth one.", partition="private",
              origin="user_utterance", source_ref="session")
    m.observe("The ward smell still wakes me up.", partition="private",
              origin="user_utterance", source_ref="session")
    return m


# ── the acceptance criterion, in counted work ────────────────────────────

class _Counter:
    """Rows a recall pulls out of SQLite and into Python.

    Not a timer. This is the quantity the old code let grow: the brute
    force search selected every vector row in the store, BLOB included, and
    applied the partition check afterwards in a Python loop.
    """

    def __init__(self, mind):
        self.mind = mind
        self.rows = 0

    def __enter__(self):
        store = self.mind._s
        self._real = store.query

        def counting(sql, params=()):
            out = self._real(sql, params)
            self.rows += len(out)
            return out

        store.query = counting
        return self

    def __exit__(self, *exc):
        self.mind._s.query = self._real
        return False


def _private_recall_cost(work_notes: int) -> int:
    m = _two_partitions(work_notes, embedder=Toy())
    m.recall("ward smell", partition="private")       # warm the caches
    m._rcache.invalidate()
    with _Counter(m) as c:
        r = m.recall("ward smell", partition="private")
    assert r.state is not State.DONT_KNOW, "the private memory must be found"
    m.close()
    return c.rows


def test_private_partition_work_is_independent_of_work_partition_size():
    """G5's acceptance criterion, in the currency that is not flaky.

    An eightfold growth in the work partition must not move what a
    private-partition query costs. Some slack is allowed for counts and
    flow-graph reads that are genuinely O(partitions); none is allowed for
    anything that scales with the other partition's CONTENT.
    """
    small = _private_recall_cost(25)
    large = _private_recall_cost(200)
    assert large <= small + 10, (
        f"a private recall cost {small} rows beside 25 work notes and "
        f"{large} beside 200 -- the work partition is being paid for")


def test_the_measurement_can_fail():
    """The negative control. A benchmark that always passes measures
    nothing, and this project has the scars to prove it.

    Restores exactly the pre-G5 behaviour -- visibility applied in Python,
    AFTER every BLOB in the store has been read off disk -- and demands
    that the assertion above then breaks. If this test ever passes
    silently, the one above has stopped measuring anything.
    """
    from owl.vectors import VectorIndex
    real = VectorIndex.search

    def unscoped(self, qv, **kw):
        kw.pop("partitions", None)
        return real(self, qv, **kw)

    VectorIndex.search = unscoped
    try:
        small, large = _private_recall_cost(25), _private_recall_cost(200)
    finally:
        VectorIndex.search = real
    assert large > small + 10, (
        "the pre-G5 path did NOT scale with the work partition, so the "
        "acceptance test above is not testing what it claims")


def test_the_same_holds_with_no_embedder():
    """The lexical path alone. A store that never had a model attached is a
    supported configuration, not a degraded one, and it gets the same
    guarantee."""
    def cost(n):
        m = _two_partitions(n)
        m.recall("ward smell", partition="private")
        m._rcache.invalidate()
        with _Counter(m) as c:
            m.recall("ward smell", partition="private")
        m.close()
        return c.rows

    assert cost(200) <= cost(25) + 10


@pytest.mark.parametrize("_run", range(1))
def test_private_partition_latency_is_independent_of_work_size(_run):
    """The criterion as written, in time. Loose on purpose.

    Wall-clock on a shared CI box is noise with a signal in it, so this
    takes the MINIMUM of several runs -- the least contaminated statistic
    available -- and asserts a ratio a genuine O(store) regression would
    blow through by an order of magnitude, not a percentage.
    """
    def latency(n):
        m = _two_partitions(n, embedder=Toy())
        best = float("inf")
        for _ in range(5):
            m._rcache.invalidate()
            t0 = time.perf_counter()
            m.recall("ward smell", partition="private")
            best = min(best, time.perf_counter() - t0)
        m.close()
        return best

    small, large = latency(25), latency(200)
    assert large < small * 4 + 0.005, (
        f"private recall took {small * 1000:.2f} ms beside 25 work notes "
        f"and {large * 1000:.2f} ms beside 200")


# ── the boundary is still the boundary ───────────────────────────────────

def test_sharding_does_not_open_the_sealed_partition():
    """The optimisation touches the same predicate confidentiality rests
    on. If it were ever going to break something, it would be this."""
    m = _two_partitions(30, embedder=Toy())
    assert m.recall("replaying the fourth", partition="work").state \
        is State.DONT_KNOW
    assert m.recall("convoy routing fuel", partition="private").state \
        is State.DONT_KNOW
    assert m.recall("replaying the fourth", partition="private").state \
        is not State.DONT_KNOW
    m.close()


def test_the_shard_predicate_and_the_visibility_set_agree():
    """Belt and braces, checked. `partitions` scopes the SQL and `allowed`
    remains the authority; a divergence would mean the fast path can see
    something the slow one cannot."""
    m = _two_partitions(20, embedder=Toy())
    visible = m._s.readable_from("private")
    allowed = {r["node_id"] for r in m._s.query(
        "SELECT node_id FROM mem_index WHERE partition IN ({})".format(
            ",".join("?" * len(visible))), tuple(visible))}
    qv = m._embed(["ward smell"], Space.READ)
    scoped = dict(m._vec.search(qv[0], space=Space.READ, floor=0.0,
                                partitions=visible.keys()))
    unscoped = {n: s for n, s in m._vec.search(qv[0], space=Space.READ,
                                               floor=0.0, allowed=allowed)}
    assert set(scoped) == set(unscoped)
    m.close()


def test_interference_does_not_pair_across_partitions():
    """Confusability between memories no single query can retrieve together
    is not a fact about either of them -- and recording it writes a
    cross-partition observation into a table both sides read."""
    m = _mind(embedder=Toy())
    m.partition("work")
    m.partition("private", sealed=True)
    text = "The generator failed during the night shift."
    m.observe(text, partition="work", origin="document", source_ref="a")
    m.observe(text, partition="private", origin="document", source_ref="b")
    pairs = m._interference_sweep("private")
    assert pairs["confusable_pairs"] == 0, \
        "a private memory was paired with an identical work one"
    assert not m._s.query(
        "SELECT 1 FROM assoc_edge e JOIN mem_index a ON a.node_id=e.src "
        "JOIN mem_index b ON b.node_id=e.dst "
        "WHERE e.kind='confusable' AND a.partition<>b.partition"), \
        "a confusability edge crosses a partition boundary"
    m.close()


# ── the vocabulary filter, now per partition ─────────────────────────────

def test_a_term_only_in_work_is_definitely_unknown_from_private():
    """The question the global filter got wrong. It answered 'is this term
    anywhere in the STORE' and the caller used it to decide whether to scan
    a PARTITION -- so a work-only term made the fast path decline to fire
    in exactly the case it exists for."""
    m = _two_partitions(20)
    private = m._s.readable_from("private")
    assert m._definitely_unknown(["convoy"], private) is True
    assert m._definitely_unknown(["ward"], private) is False
    work = m._s.readable_from("work")
    assert m._definitely_unknown(["convoy"], work) is False
    m.close()


def test_a_term_written_after_the_filter_was_built_is_still_findable():
    """THE REGRESSION THIS MODULE WAS WRITTEN AFTER FINDING.

    The filter was built once, on first recall, and never updated -- on the
    reasoning that adding a term only moves a Bloom filter towards
    'possibly present'. True of adding to the FILTER; the code added to the
    STORE and not the filter, which is the other direction. Every term
    written after the first recall read as DEFINITELY ABSENT, the posting
    scan was skipped, and a memory the store held came back DONT_KNOW.

    No embedder needed, nothing raised, indistinguishable from forgetting.
    """
    m = _mind()                                   # lexical only: no rescue
    m.observe("The clinic generator runs on depot fuel.",
              origin="document", source_ref="s1")
    assert m.recall("depot fuel").state is State.KNOW   # builds the filter
    m.observe("The borehole pump serial is QQ-9981.",
              origin="document", source_ref="s2")
    r = m.recall("borehole pump")
    assert r.state is not State.DONT_KNOW, \
        "a memory written after the filter was built became unfindable"
    assert any("borehole" in c.content for c in r.chunks)
    m.close()


def test_a_filter_that_fills_up_rebuilds_rather_than_degrading():
    """A saturated Bloom filter does not fail -- it stops rejecting, and
    goes on reporting itself as working."""
    built = {"n": 0}

    def load(_p):
        built["n"] += 1
        return [f"t{i}" for i in range(20)]

    ss = ShardSet(load_terms=load, count_nodes=lambda _p: 0)
    ss.definitely_unknown(["t1"], ["default"])            # builds
    sh = ss.shard("default")
    sh.bloom.n_added = sh.bloom.bits                      # force saturation
    ss.note_term("default", "brand_new")
    assert sh.built is False and sh.rebuilds == 1
    ss.definitely_unknown(["t1"], ["default"])
    assert built["n"] == 2, "it should have rebuilt from the store"


def test_counts_are_per_partition_and_invalidated_on_write():
    calls = {"n": 0}

    def count(_p):
        calls["n"] += 1
        return 7

    ss = ShardSet(load_terms=lambda _p: [], count_nodes=count)
    assert ss.count("work") == 7 and ss.count("work") == 7
    assert calls["n"] == 1, "the count should be cached"
    ss.touch("work")
    assert ss.count("work") == 7 and calls["n"] == 2


# ── maintenance stays inside its shard ───────────────────────────────────

def test_tending_one_partition_does_not_consume_anothers_pending_work():
    """One shared dirty set meant take() -- which CLEARS what it returns --
    handed a maintenance pass on `private` the nodes queued by writes to
    `work`, and then dropped them. Silent loss of consolidation, which is
    the failure the periodic full sweep exists to catch and the partitions
    were turning from rare into routine."""
    m = _mind()
    m.partition("work")
    m.partition("private", sealed=True)
    for i in range(30):
        m.observe(f"Work note {i} about the depot.", partition="work",
                  origin="document", source_ref="s")
    m.tend(partition="work")                    # first pass: full sweep
    m.observe("One new work note.", partition="work", origin="document",
              source_ref="s")
    m.observe("One new private note.", partition="private",
              origin="user_utterance", source_ref="s")

    m.tend(partition="private")
    assert m._dirty_for("work").stats["pending"] == 1, \
        "tending private consumed work's pending consolidation"
    rep = m.tend(partition="work")
    assert "incremental" in rep["scope"]["why"]
    m.close()


# ── migration ────────────────────────────────────────────────────────────

def _legacy_store(path: Path) -> None:
    """A store as written before G5: no denormalised partition columns."""
    raw = (Path(__file__).resolve().parents[1] / "owl" / "schema.sql"
           ).read_text(encoding="utf-8")
    legacy = raw.replace(
        "    partition TEXT NOT NULL DEFAULT 'default',\n    PRIMARY KEY "
        "(term, node_id)", "    PRIMARY KEY (term, node_id)").replace(
        "    partition TEXT NOT NULL DEFAULT 'default',\n    PRIMARY KEY "
        "(node_id, space)", "    PRIMARY KEY (node_id, space)")
    conn = sqlite3.connect(str(path))
    conn.executescript(legacy)
    conn.commit()
    conn.close()


def _unshard(path: Path) -> None:
    """Take a live store back to the pre-G5 layout.

    The indexes go first: SQLite refuses DROP COLUMN while an index still
    names the column, which is a small demonstration that the shard
    indexes are real rather than advisory.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("DROP INDEX IF EXISTS idx_lexeme_shard")
    conn.execute("DROP INDEX IF EXISTS idx_vector_shard")
    conn.execute("ALTER TABLE lexeme DROP COLUMN partition")
    conn.execute("ALTER TABLE vector DROP COLUMN partition")
    conn.commit()
    conn.close()


def test_a_legacy_store_is_missing_the_columns_in_the_first_place():
    """Guards the fixture. If schema.sql is reworded and this stops
    producing a genuinely legacy store, every migration test below silently
    becomes a test of nothing."""
    p = Path(tempfile.mkdtemp()) / "legacy.owl"
    _legacy_store(p)
    conn = sqlite3.connect(str(p))
    assert not shards.is_sharded(conn)
    conn.close()


def test_an_old_store_migrates_on_open_and_backfills_correctly():
    p = Path(tempfile.mkdtemp()) / "legacy.owl"
    _legacy_store(p)
    conn = sqlite3.connect(str(p))
    conn.executescript(
        "INSERT INTO partition(name,sealed,created_at) VALUES('work',0,0);"
        "INSERT INTO mem_index(node_id,partition,stability,difficulty,"
        "last_review) VALUES('obs_x','work',1.0,5.0,0.0);"
        "INSERT INTO lexeme(term,node_id,tf) VALUES('depot','obs_x',1.0);"
        "INSERT INTO vector(node_id,space,dim,model,data) "
        "VALUES('obs_x','read',2,'toy',X'0000803F00000000');")
    conn.commit()
    conn.close()

    m = Owl.open(p)
    assert m._s.sharded is True
    assert set(m._s.migration["migrated"]) == {"lexeme", "vector"}
    assert m._s.one("SELECT partition FROM lexeme WHERE term='depot'"
                    )["partition"] == "work"
    assert m._s.one("SELECT partition FROM vector WHERE node_id='obs_x'"
                    )["partition"] == "work"
    m.close()


def test_migration_is_idempotent():
    p = Path(tempfile.mkdtemp()) / "legacy.owl"
    _legacy_store(p)
    Owl.open(p).close()
    m = Owl.open(p)
    assert m._s.migration["migrated"] == [], "second open should migrate nothing"
    assert m._s.migration["indexes"] == len(shards.SHARD_DDL)
    m.close()


def test_a_migrated_store_still_answers_what_it_held():
    """The migration rewrites the index layer of a store full of memories.
    Nothing may become unfindable."""
    p = Path(tempfile.mkdtemp()) / "s.owl"
    m = Owl.open(p)
    m.observe("Route Alpha floods above 40mm rainfall.", origin="document",
              source_ref="sitrep")
    before = [c.content for c in m.recall("route alpha flooding").chunks]
    m.close()
    # Strip the columns back off and reopen, forcing a real migration over
    # existing content rather than over an empty store.
    _unshard(p)
    m = Owl.open(p)
    assert [c.content for c in m.recall("route alpha flooding").chunks] == before
    assert shards.verify(sqlite3.connect(str(p))) == []
    m.close()


def test_an_unmigrated_store_opened_readonly_still_answers():
    """F2 is not suspended for a speedup. Read-only media cannot be
    migrated, and a store that cannot be migrated must not become a store
    that cannot be read."""
    p = Path(tempfile.mkdtemp()) / "s.owl"
    m = Owl.open(p)
    m.observe("The generator serial is GX-4419.", origin="document",
              source_ref="sitrep")
    m.close()
    _unshard(p)

    ro = Owl.open(p, readonly=True)
    assert ro._s.sharded is False, "nothing may have been written"
    r = ro.recall("GX-4419")
    assert r.state is not State.DONT_KNOW
    ro.close()


def _check(mind, cid):
    from owl import diagnostics
    return next(c for c in diagnostics.run(mind).checks if c.id == cid)


def test_doctor_reports_a_healthy_shard_layout():
    m = _two_partitions(5)
    assert _check(m, "shards.layout").status == "PASS"
    assert _check(m, "shards.partition_agrees").status == "PASS"
    m.close()


def test_doctor_flags_an_unmigrated_store():
    """Drives `shards.layout` red. WARN and not FAIL on purpose: the store
    WORKS, it is merely paying the pre-G5 price, and a read-only archive
    cannot do anything about it. Warning on a condition the user cannot
    act on is how you train people to ignore warnings."""
    p = Path(tempfile.mkdtemp()) / "s.owl"
    m = Owl.open(p)
    m.observe("A note.", origin="document", source_ref="s")
    m.close()
    _unshard(p)
    ro = Owl.open(p, readonly=True)
    c = _check(ro, "shards.layout")
    assert c.status == "WARN" and "not been migrated" in c.detail
    assert c.remedy, "a check without a remedy is a complaint"
    ro.close()


def test_doctor_flags_a_partition_that_disagrees():
    """Drives `shards.partition_agrees` red. This one is a FAIL, because
    an index row filed under the wrong partition is content reachable from
    a boundary that was supposed to exclude it."""
    p = Path(tempfile.mkdtemp()) / "s.owl"
    m = Owl.open(p)
    m.partition("work")
    m.observe("A note about the depot.", partition="work", origin="document",
              source_ref="s")
    m.close()
    conn = sqlite3.connect(str(p))
    conn.execute("UPDATE lexeme SET partition='default'")
    conn.commit()
    conn.close()
    m = Owl.open(p)
    c = _check(m, "shards.partition_agrees")
    assert c.status == "FAIL" and "lexeme" in c.detail
    m.close()


def test_drift_between_the_copy_and_the_authority_is_detectable():
    """The whole safety case for denormalising `partition` is that a node
    never changes partition. That is an argument; this is the check."""
    p = Path(tempfile.mkdtemp()) / "s.owl"
    m = Owl.open(p)
    m.partition("work")
    m.observe("A note about the depot.", partition="work", origin="document",
              source_ref="s")
    m.close()
    conn = sqlite3.connect(str(p))
    assert shards.verify(conn) == []
    conn.execute("UPDATE lexeme SET partition='default'")
    conn.commit()
    problems = shards.verify(conn)
    assert problems and "lexeme" in problems[0]
    conn.close()


# ── answers, unchanged ───────────────────────────────────────────────────

NOTES = [f"Field note {i} about depot fuel, routes and the clinic."
         for i in range(20)] + [
    "Dr Warsame runs the Bardera clinic.",
    "Route Alpha floods above 40mm rainfall.",
    "The generator serial is GX-4419.",
]
QUERIES = ["who runs the clinic", "GX-4419", "route alpha",
           "what is the helicopter tail number", "depot fuel"]


def test_the_shard_predicate_changes_no_answer():
    """The property that outranks every number in this file.

    Two identical stores, one asked with the shard predicate and one with
    the legacy join. Recall is not idempotent -- it reinforces what it
    returns -- so each store is asked exactly once.
    """
    def answers(sharded: bool):
        m = _mind(embedder=Toy())
        for n in NOTES:
            m.observe(n, origin="document", source_ref="sitrep")
        if not sharded:
            m._s.sharded = False          # take the pre-G5 query path
            m._shards.sharded = False
            m._shards.invalidate()
        out = [(q, r.state, [c.content for c in r.chunks])
               for q in QUERIES for r in [m.recall(q)]]
        m.close()
        return out

    assert answers(True) == answers(False)


def test_an_ann_shard_is_built_and_used_per_partition():
    """A store-wide IVF index lets the work partition's vectors decide
    where the private partition's centroids sit, and a probe then spends
    its budget on lists it may not read from."""
    m = _two_partitions(40, embedder=Toy())
    rep = m._vec.build_ann(Space.READ, partition="work")
    assert rep["partition"] == "work"
    assert ("read", "work") in m._vec._ann
    assert ("read", None) not in m._vec._ann, \
        "a shard index must not masquerade as the store-wide one"
    # The private partition has no shard and no global index, so it stays
    # on the exact path rather than borrowing work's approximation.
    assert m.recall("ward smell", partition="private").state \
        is not State.DONT_KNOW
    m.close()


def test_the_shard_layer_reports_itself():
    m = _two_partitions(15)
    m.recall("convoy routing", partition="work")
    m.recall("ward smell", partition="private")
    st = m._shards.stats
    assert st["sharded"] is True
    assert {"work", "private"} <= set(st["partitions"])
    assert st["partitions"]["private"]["terms"] < \
        st["partitions"]["work"]["terms"]
    assert st["partitions"]["private"]["false_positive_rate"] < 0.01
    m.close()
