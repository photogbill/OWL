"""F1 -- capture must never block, and must never lie about it.

Measured on Qwen3-Embedding-8B, an inline embed costs ~330 ms. A thousand
notes is six minutes with the session frozen, which makes the better encoder
unusable for exactly the workload it was chosen for.

The interesting half is not the queue. It is that a memory which has been
captured but not embedded is neither present nor absent, and reporting it as
absent would be the same dishonesty as returning five bad matches instead of
admitting to none.
"""
import os
import tempfile
import time

import pytest

from owl import Owl, State


class SlowEmbedder:
    is_semantic = True
    name = "slow-probe"

    def __init__(self, ms=20.0, fail_on=()):
        self.ms, self.fail_on, self.calls = ms, set(fail_on), 0

    def embed(self, texts, space):
        self.calls += 1
        out = []
        for t in texts:
            if any(f in t for f in self.fail_on):
                raise RuntimeError("model refused this text")
            time.sleep(self.ms / 1000.0)
            v = [0.0] * 16
            for w in t.lower().split():
                v[hash(w) % 16] += 1.0
            out.append(v or [1.0] * 16)
        return out


def _mind(**kw):
    path = os.path.join(tempfile.mkdtemp(), "f1.owl")
    return Owl.open(path, **kw)


def test_capture_latency_drops_below_the_target():
    """F1's acceptance criterion: p99 capture under 10 ms WITH an embedder."""
    slow = SlowEmbedder(ms=20.0)
    inline, deferred = [], []
    for target, defer in ((inline, False), (deferred, True)):
        mind = _mind(embedder=slow, defer_embedding=defer)
        with mind:
            for i in range(40):
                t = time.perf_counter()
                mind.observe(f"Field note {i} about depot fuel and vehicles.")
                target.append((time.perf_counter() - t) * 1000)

    def p99(xs):
        return sorted(xs)[int(len(xs) * 0.99) - 1]

    assert p99(inline) > 15.0, "the inline path should be dominated by the model"
    assert p99(deferred) < 10.0, f"p99 was {p99(deferred):.1f} ms"


def test_an_unembedded_memory_is_not_reported_as_absent():
    """The honesty requirement. DONT_KNOW with a full queue is 'not yet';
    DONT_KNOW with an empty one is 'not there'. Different claims."""
    mind = _mind(embedder=SlowEmbedder(), defer_embedding=True)
    with mind:
        mind.observe("The clinic generator runs on depot fuel.")
        r = mind.recall("something entirely unrelated to anything stored")
        assert r.pending == 1
        assert r.provisional is True
        assert "not yet embedded" in r.reason
        assert "provisional" in r.reason

        mind.absorb()
        r2 = mind.recall("something entirely unrelated to anything stored")
        assert r2.pending == 0 and r2.provisional is False
        assert "provisional" not in r2.reason


def test_pending_is_orthogonal_to_state_not_a_seventh_one():
    """You can KNOW and still have unread material. Making this a state
    would have forced a false choice between the two facts."""
    mind = _mind(embedder=SlowEmbedder(), defer_embedding=True)
    with mind:
        mind.observe("Route Alpha floods above 40mm rainfall.")
        mind.absorb()                       # findable
        mind.observe("The depot holds 4000 litres.")   # queued
        r = mind.recall("Route Alpha floods")
        assert r.state in (State.KNOW, State.KNOW_WHERE)
        assert r.provisional is True        # both true at once


def test_lexical_still_works_while_the_queue_is_full():
    """Deferring embedding degrades retrieval; it must not break it."""
    mind = _mind(embedder=SlowEmbedder(), defer_embedding=True)
    with mind:
        mind.observe("The generator serial is GX-4419.")
        assert mind.pending() == 1
        r = mind.recall("GX-4419")
        assert r.chunks and "GX-4419" in r.chunks[0].content


def test_a_memory_that_cannot_be_embedded_is_not_lost():
    """An embedder failure must degrade to lexical, not drop the record."""
    mind = _mind(embedder=SlowEmbedder(fail_on=("poison",)),
                 defer_embedding=True)
    with mind:
        good = mind.observe("An ordinary note about the depot.")
        bad = mind.observe("A poison note the model refuses.")
        rep = mind.absorb()
        assert rep["embedded"] == 1 and rep["failed"] == 1

        row = mind._s.one("SELECT last_error, attempts FROM embed_queue "
                          "WHERE node_id=?", (bad,))
        assert row["attempts"] == 1 and "RuntimeError" in row["last_error"]
        # still retrievable, still auditable
        assert mind.why(bad) or mind.recall("poison note").chunks


def test_absorb_gives_up_rather_than_spinning():
    mind = _mind(embedder=SlowEmbedder(fail_on=("poison",)),
                 defer_embedding=True)
    with mind:
        mind.observe("A poison note the model refuses.")
        for _ in range(5):
            mind.absorb()
        row = mind._s.one("SELECT attempts FROM embed_queue")
        assert row["attempts"] == 3, "must stop retrying, not loop forever"


def test_absorb_respects_a_budget():
    mind = _mind(embedder=SlowEmbedder(ms=0.0), defer_embedding=True)
    with mind:
        for i in range(10):
            mind.observe(f"Note {i} about supplies.")
        assert mind.absorb(budget=4)["embedded"] == 4
        assert mind.pending() == 6
        mind.absorb()
        assert mind.pending() == 0


def test_tend_drains_the_queue_first():
    """tend() is the idle hook, and everything in it reads vectors -- running
    the maintenance passes against a half-embedded store would measure the
    queue rather than the memory."""
    mind = _mind(embedder=SlowEmbedder(ms=0.0), defer_embedding=True)
    with mind:
        for i in range(6):
            mind.observe(f"Note {i} about the northern compound.")
        rep = mind.tend()
        # 6 captured, plus anything tend() itself created (fusion composites)
        assert rep["absorbed"]["embedded"] >= 6
        assert mind.pending() == 0, "tend() must not leave idle work behind"


def test_context_is_captured_at_write_time_not_drain_time():
    """The WRITE vector encodes where a memory sat when it arrived. Looking
    that up at drain time would embed the wrong context for anything captured
    before a period closed."""
    mind = _mind(embedder=SlowEmbedder(ms=0.0), defer_embedding=True)
    with mind:
        with mind.period("deployment-one"):
            nid = mind.observe("A note filed during the first deployment.")
            row = mind._s.one("SELECT period_id FROM embed_queue "
                              "WHERE node_id=?", (nid,))
            assert row["period_id"] is not None
        with mind.period("deployment-two"):    # context moves on
            mind.absorb()
        # the vector was built from deployment-one, which is where it belongs
        assert mind._s.one("SELECT COUNT(*) n FROM vector WHERE node_id=?",
                           (nid,))["n"] == 2


def test_default_is_still_inline():
    """Deferral changes when a memory becomes findable. Callers opt in."""
    mind = _mind(embedder=SlowEmbedder(ms=0.0))
    with mind:
        mind.observe("Immediately findable.")
        assert mind.pending() == 0
        assert mind.recall("immediately findable").pending == 0
