"""G1-G4 -- speed that is not allowed to change answers.

Every test here exists because an optimisation's failure mode is a WRONG
ANSWER DELIVERED FASTER, and that is invisible. So each one pins the
behaviour first and the speed second.

The G1 story is worth reading. The plan says "make DONT_KNOW O(1)", which
reads like an invitation to short-circuit recall entirely. Doing that broke
paraphrase recall outright -- a paraphrase shares no lexical terms with its
target, so a vocabulary filter says "absent" for exactly the queries the
semantic tier exists to answer -- plus the gap explanation, recorded
absences, receipts, and the deferred-capture notice. Five features, one
optimisation. The filter now skips only the posting-list scan.
"""
import hashlib
import os
import tempfile
import time

import pytest

from owl import Owl
from owl.fastpath import Bloom, VectorCache, view
from owl.freshness import DirtySet, RecallCache


class Toy:
    """A bag-of-tokens encoder that is the SAME encoder every run.

    It used to bucket on `hash(w) % 16`, and `hash()` on a str is salted
    per process -- so this was a different embedding model on every run,
    and `test_paraphrase_recall_survives_the_fast_path` failed on roughly
    two thirds of PYTHONHASHSEED values. The test is right: the toy must
    not be able to match a paraphrase, because it asserts the full path
    RAN rather than that an answer was found. At 16 buckets it sometimes
    could, by collision -- "how is the health facility powered" landed on
    the clinic note at cosine 0.75 and the recall came back KNOW.

    blake2b for a stable bucket, and 64 of them: same seed, same vectors,
    same verdict, every process. A suite that passes two runs in three is
    not evidence about anything, and the flakiness reads as an intermittent
    engine bug, which is the expensive kind to chase.
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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "p.owl"),
                    embedder=Toy(), **kw)


# ── G1: the vocabulary filter ────────────────────────────────────────────

def test_a_bloom_filter_is_never_wrong_about_absence():
    """The asymmetry the whole design rests on: false positives are free,
    false negatives would lose a memory."""
    b = Bloom(bits=1 << 16)
    present = [f"term{i}" for i in range(500)]
    for t in present:
        b.add(t)
    for t in present:
        assert t in b, "a false NEGATIVE would silently lose a memory"


def test_it_reports_its_own_error_rate():
    b = Bloom(bits=1 << 16)
    for i in range(2000):
        b.add(f"t{i}")
    assert 0.0 < b.false_positive_rate < 0.2
    absent = sum(1 for i in range(2000, 4000) if f"t{i}" in b)
    assert absent / 2000 < 0.25, "measured rate should track the estimate"


def test_the_filter_survives_a_round_trip():
    b = Bloom(bits=1 << 14)
    b.add("depot")
    b2 = Bloom.from_bytes(b.to_bytes(), hashes=b.hashes, n_added=b.n_added)
    assert "depot" in b2


def test_paraphrase_recall_survives_the_fast_path():
    """The regression that made this test file necessary. A paraphrase has
    ZERO lexical overlap with its target -- that is what makes it a
    paraphrase -- so a filter over the lexical vocabulary must never be
    allowed to decide the whole query."""
    with _mind() as m:
        m.observe("The clinic generator runs on depot fuel.",
                  origin="document", source_ref="sitrep")
        r = m.recall("how is the health facility powered")
        # The Toy embedder is a bag-of-token hash and cannot match a
        # paraphrase either -- so this asserts the FULL PATH RAN, not that
        # an answer was found. Whether the encoder can do it is the
        # encoder's business; whether the optimisation let it try is ours.
        assert "vocabulary filter" not in r.reason
        assert len(r.reason) > 30, "the gap explanation must still be built"


def test_a_genuinely_absent_query_still_explains_the_gap():
    """B7 lives in the full path. Skipping to a bare DONT_KNOW would drop
    the useful half of the answer."""
    with _mind() as m:
        m.observe("The clinic generator runs on depot fuel.",
                  origin="document", source_ref="sitrep")
        r = m.recall("what is the helicopter tail number")
        assert r.state.value == "dont_know"
        assert len(r.reason) > 30, r.reason


def test_the_filter_skips_the_scan_without_changing_the_answer():
    with _mind() as m:
        for i in range(20):
            m.observe(f"Field note {i} about depot fuel.", origin="document",
                      source_ref="sitrep")
        assert m._definitely_unknown(["zzzznonexistent"]) is True
        assert m._definitely_unknown(["depot"]) is False
        assert m._definitely_unknown([]) is False


# ── G2: vectors without the per-query deserialise ────────────────────────

def test_a_view_round_trips_exactly():
    import struct
    vals = [0.5, -0.25, 1.0, 0.0]
    blob = struct.pack("<4f", *vals)
    assert list(view(blob)) == pytest.approx(vals)


def test_the_cache_returns_the_same_object_not_a_copy():
    import struct
    c = VectorCache()
    blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    a = c.get("n1", blob)
    assert c.get("n1", blob) is a
    assert c.stats["hits"] == 1 and c.stats["misses"] == 1


def test_the_cache_stops_rather_than_thrashing():
    """An LRU on a hot path costs more bookkeeping than it saves, and a
    cache that silently thrashes is worse than one that plainly stops."""
    import struct
    c = VectorCache(max_vectors=3)
    blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    for i in range(10):
        c.get(f"n{i}", blob)
    assert c.stats["cached"] == 3
    assert c.stats["at_capacity"] is True


def test_invalidation_is_targeted_and_total():
    import struct
    c = VectorCache()
    blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    c.get("a", blob)
    c.get("b", blob)
    c.invalidate("a")
    assert c.stats["cached"] == 1
    c.invalidate()
    assert c.stats["cached"] == 0


# ── G3: recall caching ───────────────────────────────────────────────────

def test_a_cache_hit_returns_identical_results():
    """G3's acceptance criterion."""
    with _mind() as m:
        for i in range(10):
            m.observe(f"Note {i} about depot fuel and vehicles.",
                      origin="document", source_ref="sitrep")
        first = m.recall("depot fuel")
        second = m.recall("depot fuel")
        assert second is first, "a repeat query should not recompute"
        assert m._rcache.stats["hits"] >= 1


def test_any_write_invalidates_everything():
    """Total, and O(1). Working out WHICH cached queries a write could
    have affected is where cache bugs live, and a stale answer looks
    exactly like a fresh one."""
    with _mind() as m:
        m.observe("Route Alpha is open.", origin="document",
                  source_ref="sitrep")
        a = m.recall("route alpha")
        gen = m._rcache.generation
        m.observe("Route Alpha is closed by flooding.", origin="document",
                  source_ref="sitrep-2")
        assert m._rcache.generation > gen
        b = m.recall("route alpha")
        assert b is not a, "the write must have invalidated the entry"


def test_the_clock_bucket_is_part_of_the_key():
    """Recall is time-dependent through retrievability decay, so a cache
    keyed only on the query would serve a stale ranking forever."""
    c = RecallCache()
    k1 = c.key("q", "default", 1000.0)
    k2 = c.key("q", "default", 1000.0 + c.bucket * 2)
    assert k1 != k2


def test_different_arguments_are_different_keys():
    c = RecallCache()
    assert c.key("q", "default", 0.0, budget=5) != \
        c.key("q", "default", 0.0, budget=9)
    assert c.key("q", "work", 0.0) != c.key("q", "private", 0.0)


def test_a_degraded_answer_is_not_cached():
    """It records a transient condition -- a dead model, a half-drained
    queue -- and serving it again after the condition clears would turn a
    temporary problem into a sticky one."""
    class Broken(Toy):
        def embed(self, texts, space):
            raise RuntimeError("model down")

    path = os.path.join(tempfile.mkdtemp(), "d.owl")
    with Owl.open(path, embedder=Broken()) as m:
        m.observe("A note about the depot.", origin="document",
                  source_ref="s")
        r = m.recall("depot")
        assert r.degraded
        assert m._rcache.stats["entries"] == 0


def test_the_cache_is_bounded():
    c = RecallCache(max_entries=4)
    for i in range(20):
        c.put(c.key(f"q{i}", "default", 0.0), i)
    assert c.stats["entries"] == 4


# ── G4: incremental maintenance ──────────────────────────────────────────

def test_tend_scopes_to_what_changed():
    """G4's acceptance criterion: cost scales with changes, not store."""
    with _mind() as m:
        for i in range(30):
            m.observe(f"Note {i} about the compound.", origin="document",
                      source_ref="sitrep")
        m.tend()                              # first pass: full
        m.observe("One new note.", origin="document", source_ref="sitrep")
        rep = m.tend()
        assert rep["scope"]["nodes"] < rep["scope"]["of"]
        assert "incremental" in rep["scope"]["why"]


def test_a_full_sweep_still_happens_periodically():
    """A dirty-set bug loses consolidation work silently. The periodic full
    pass is the thing that would eventually surface it."""
    d = DirtySet(full_sweep_every=3)
    everything = {f"n{i}" for i in range(100)}
    for _ in range(3):
        d.mark("n1")
        scope, why = d.take(all_nodes=everything)
        assert "incremental" in why
    d.mark("n1")
    scope, why = d.take(all_nodes=everything)
    assert scope == everything and "full sweep" in why


def test_an_empty_dirty_set_takes_the_cheap_full_path():
    d = DirtySet()
    scope, why = d.take(all_nodes={"a", "b"})
    assert scope == {"a", "b"} and "cheap case" in why


def test_passes_may_enlarge_the_dirty_set_while_running():
    """Consolidation passes are not independent -- fusing two memories
    changes what interference should look at."""
    d = DirtySet()
    d.mark("a")
    d.mark_many(["b", "c"])
    assert d.stats["pending"] == 3


# ── the property that matters more than any of the above ─────────────────

def test_none_of_this_changes_a_single_answer():
    """Every optimisation here has the same failure mode: a wrong answer
    delivered faster. This runs the same queries with the fast paths warm
    and cold and demands identical results."""
    notes = [f"Field note {i} about depot fuel, routes and the clinic."
             for i in range(25)] + [
        "Dr Warsame runs the Bardera clinic.",
        "Route Alpha floods above 40mm rainfall.",
        "The generator serial is GX-4419.",
    ]
    queries = ["who runs the clinic", "GX-4419", "route alpha",
               "what is the helicopter tail number", "depot fuel"]

    with _mind() as m:
        for n in notes:
            m.observe(n, origin="document", source_ref="sitrep")
        pass

    # Recall is deliberately NOT idempotent -- it reinforces what it
    # returns and demotes the near-misses -- so comparing two consecutive
    # computations is comparing retrieval-induced forgetting, not the
    # cache. This test made that mistake twice before landing on the only
    # sound comparison: two IDENTICAL stores, one with the fast paths
    # disabled, each asked once.
    def _answers(disable_fastpath):
        mm = _mind()
        for n in notes:
            mm.observe(n, origin="document", source_ref="sitrep")
        if disable_fastpath:
            # G5 gave this a `visible` argument: the filter answers a
            # question about the visible PARTITIONS, not the store.
            mm._definitely_unknown = lambda terms, visible=None: False
            mm._rcache.max_entries = 0
        # Compare CONTENT, not node ids. Two independently built stores
        # mint different uuids for the same material, so an id comparison
        # tests uuid generation rather than retrieval -- a mistake this
        # project has now made twice, once in the A10 determinism test.
        out = [(q, m2.state, [c.content for c in m2.chunks])
               for q in queries for m2 in [mm.recall(q)]]
        mm.close()
        return out

    assert _answers(True) == _answers(False)
