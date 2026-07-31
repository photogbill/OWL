"""B6 counter-evidence + E3 the action-outcome loop.

Both exist because agreeing with yourself is the default failure of a memory
system: similarity search returns support, and unmeasured confidence drifts
upward. One retrieves the disagreement, the other scores whether the
confidence was ever honest.
"""
import os
import tempfile

import pytest

from owl import Owl
from owl import calibration_loop as cal
from owl import counter


class Toy:
    is_semantic = True
    name = "toy"

    def embed(self, texts, space):
        out = []
        for t in texts:
            v = [0.0] * 16
            for w in t.lower().split():
                v[hash(w) % 16] += 1.0
            out.append(v or [1.0] * 16)
        return out


def _mind(**kw):
    return Owl.open(os.path.join(tempfile.mkdtemp(), "c.owl"),
                    embedder=Toy(), **kw)


# ── B6 ───────────────────────────────────────────────────────────────────

def test_presupposition_is_extracted_from_the_frame():
    assert "pump" in counter.presupposition("why is the pump failing?")
    assert "failing" in counter.presupposition("why is the pump failing?")
    assert counter.presupposition("what caused the bridge to collapse?")


def test_polarity_flip_is_narrow_on_purpose():
    """A big antonym list produces confident nonsense, and this feature dies
    the first time it does that."""
    assert "closed" in counter.polarity_flip("route alpha is open")
    assert "open" in counter.polarity_flip("route alpha is closed")
    assert counter.polarity_flip("the meeting was productive") == set()


def test_it_finds_the_note_saying_the_opposite():
    """The planted contradiction. This is B6's acceptance criterion."""
    cands = [
        {"node_id": "a", "content": "The north pump is failing intermittently."},
        {"node_id": "b", "content": "The north pump is working normally."},
        {"node_id": "c", "content": "Coffee was restocked from the market."},
    ]
    cs = counter.find("why is the north pump failing?", cands)
    assert cs.found
    ids = [c.node_id for c in cs.counters]
    assert "b" in ids and "c" not in ids


def test_it_finds_an_explicit_negation():
    cands = [{"node_id": "n",
              "content": "Route Alpha is not closed; the survey was wrong."}]
    cs = counter.find("why is route alpha closed?", cands)
    assert cs.found and cs.counters[0].kind == "negation"


def test_what_the_premise_replaced_is_counter_evidence_by_construction():
    """Free, exact, no model -- the supersession graph already holds it."""
    cs = counter.find("why is the bridge down?", [],
                      superseded=[{"node_id": "old",
                                   "content": "The bridge at Km 42 is intact."}])
    assert cs.counters[0].kind == "superseded_premise"


def test_the_semantic_shape_is_skipped_loudly_at_tier_0():
    """Half-running it would surface noise and discredit the rest."""
    cs = counter.find("why is the pump failing?", [], semantic_available=False)
    assert any("semantic opposition" in s for s in cs.skipped)
    assert not counter.find("why is the pump failing?", [],
                            semantic_available=True).skipped


def test_finding_nothing_is_not_evidence_the_premise_is_right():
    with _mind() as m:
        m.observe("The north pump is failing intermittently.")
        out = m.challenge("why is the north pump failing?")
        assert out["found"] is False
        assert "not the same as the premise being right" in out["note"]


def test_challenge_end_to_end_surfaces_the_contradiction():
    with _mind() as m:
        m.observe("The north pump is failing intermittently.",
                  origin="document", source_ref="sitrep-2")
        m.observe("The north pump is working normally after the service.",
                  origin="document", source_ref="sitrep-1")
        out = m.challenge("why is the north pump failing?")
        assert out["found"]
        assert any("working normally" in c["content"] for c in out["counters"])


def test_challenge_does_not_contaminate_recall():
    """A counter-set merged into the answer is just a confusing answer."""
    with _mind() as m:
        m.observe("The north pump is failing intermittently.")
        m.observe("The north pump is working normally.")
        r = m.recall("why is the north pump failing?")
        assert "counters" not in vars(r)


# ── E3 ───────────────────────────────────────────────────────────────────

def test_the_table_finally_has_a_writer():
    """It sat in the schema with nothing writing to it, which is worse than
    absent: it looks like the question is being tracked."""
    with _mind() as m:
        n = m.observe("Fuel arrives Thursday.", origin="document",
                      source_ref="depot-clerk")
        pid = m.predicted(n)
        assert m.calibration()["predictions"] == 1
        assert m.calibration()["resolved"] == 0
        m.outcome(pid, correct=True)
        assert m.calibration()["resolved"] == 1


def test_overconfidence_is_named_as_the_dangerous_direction():
    rows = [{"producer": "bold", "claim_kind": "status",
             "confidence": 0.9, "outcome": i < 5} for i in range(10)]
    c = cal.score(rows)[0]
    assert c.overconfident and "OVERCONFIDENT" in c.verdict
    assert c.brier > 0.2


def test_a_calibrated_producer_is_recognised():
    rows = [{"producer": "steady", "claim_kind": "status",
             "confidence": 0.8, "outcome": i < 8} for i in range(10)]
    c = cal.score(rows)[0]
    assert not c.overconfident and "calibrated" in c.verdict
    assert c.brier < 0.2


def test_underconfidence_is_reported_but_not_alarming():
    rows = [{"producer": "hedger", "claim_kind": "status",
             "confidence": 0.5, "outcome": True} for _ in range(12)]
    c = cal.score(rows)[0]
    assert "underconfident" in c.verdict and not c.overconfident


def test_small_samples_get_no_verdict():
    """A Brier score from three events is theatre."""
    rows = [{"producer": "new", "claim_kind": "status",
             "confidence": 0.9, "outcome": True} for _ in range(3)]
    assert "insufficient" in cal.score(rows)[0].verdict


def test_open_predictions_are_not_evidence():
    rows = [{"producer": "p", "claim_kind": "status", "confidence": 0.9,
             "outcome": None} for _ in range(20)]
    assert cal.score(rows) == []


def test_the_curve_is_plottable_and_shows_where_it_sags():
    """The shape says more than the score: sagging only at the top means
    'trustworthy except when certain', which is specific and fixable."""
    rows = ([{"producer": "p", "claim_kind": "k", "confidence": 0.3,
              "outcome": i < 3} for i in range(10)]
            + [{"producer": "p", "claim_kind": "k", "confidence": 0.95,
                "outcome": i < 5} for i in range(10)])
    curve = cal.reliability_curve(rows)
    assert len(curve) == 5
    low = next(b for b in curve if b["n"] and b["stated"] < 0.5)
    high = next(b for b in curve if b["n"] and b["stated"] > 0.9)
    assert abs(low["stated"] - low["observed"]) < 0.1      # honest down low
    assert high["stated"] - high["observed"] > 0.4         # sags up top


def test_calibration_never_reweights_retrieval():
    """Replacing an auditable problem with an unauditable one is not a fix."""
    with _mind() as m:
        n = m.observe("Fuel arrives Thursday.", origin="document",
                      source_ref="depot-clerk")
        before = [c.node_id for c in m.recall("when does fuel arrive").chunks]
        for _ in range(12):
            m.outcome(m.predicted(n), correct=False)
        after = [c.node_id for c in m.recall("when does fuel arrive").chunks]
        assert before == after
        assert "reported only" in m.calibration()["note"]
        assert "depot-clerk" in m.calibration()["overconfident"]
