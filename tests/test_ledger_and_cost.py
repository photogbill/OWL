"""E4 the jointly-edited ledger, and C7's missing half -- inferring cost.

E4's claim: making memory legible is a cheaper path to accuracy than making
extraction smarter. The engineering consequence is that a correction has to
be first-class evidence rather than an edit, or the legibility buys nothing
you can audit afterwards.
"""
import os
import tempfile

import pytest

from owl import Owl, salience


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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "l.owl"),
                    embedder=Toy(), **kw)


# ── E4: the ledger ───────────────────────────────────────────────────────

def test_the_ledger_separates_what_it_was_told_from_what_it_decided():
    """The first question anyone asks of a memory about them: did I say
    that, or did you decide it? A flat list makes that unanswerable."""
    with _mind() as m:
        a = m.observe("Dr Warsame runs the Bardera clinic.",
                      origin="document", source_ref="sitrep")
        m.derive("The clinic has stable leadership.", parents=[a],
                 kind="abstraction", producer="analyst",
                 falsifier="check staff turnover")
        md = m.ledger()["markdown"]
        assert "Things I was told or shown" in md
        assert "Things I worked out myself" in md
        assert "These are not facts" in md


def test_an_empty_ledger_says_so():
    with _mind() as m:
        assert "Nothing on record" in m.ledger()["markdown"]


def test_a_correction_supersedes_and_does_not_erase():
    """The original is evidence. 'The system thought X until Bill said
    otherwise' is a better record than X quietly becoming Y."""
    with _mind() as m:
        a = m.observe("Dr Osman runs the Bardera clinic.",
                      origin="document", source_ref="sitrep-1")
        new = m.correct(a, "Dr Warsame runs the Bardera clinic.", by="bill",
                        reason="Osman transferred in March")
        assert new != a
        # the original still exists, verbatim
        row = m._node_row(a)
        assert row["content"] == "Dr Osman runs the Bardera clinic."
        assert m._s.one("SELECT 1 FROM supersession WHERE old_node=? "
                        "AND new_node=?", (a, new))


def test_the_correction_carries_what_it_replaced():
    """A correction stating only the new value loses the fact that a
    specific wrong thing was believed -- the part an audit needs."""
    with _mind() as m:
        a = m.observe("The depot holds 900 litres.", origin="document",
                      source_ref="sitrep")
        new = m.correct(a, "The depot holds 4000 litres.", by="bill",
                        reason="miscounted drums")
        text = m._node_row(new)["content"]
        assert "4000 litres" in text
        assert "bill" in text and "miscounted drums" in text
        assert "900 litres" in text, "what it replaced must survive"


def test_a_correction_is_a_maximum_depth_exposure():
    """E4's acceptance criterion. They retrieved it, judged it, and
    generated a replacement -- the generation effect."""
    from owl.ledger import CORRECTION_DEPTH
    with _mind() as m:
        a = m.observe("Route Alpha is open.", origin="document",
                      source_ref="sitrep")
        told = m.observe("The fence needs wire.", origin="document",
                         source_ref="sitrep")
        m.tell("bill", told, channel="briefing")
        new = m.correct(a, "Route Alpha is closed.", by="bill")

        depths = {r["node_id"]: r["depth"] for r in m._s.query(
            "SELECT node_id, depth FROM exposure WHERE who='bill'")}
        assert depths[new] == CORRECTION_DEPTH
        assert depths[new] > depths[told], \
            "correcting must encode deeper than being told"


def test_a_correction_appears_in_the_why_chain():
    with _mind() as m:
        a = m.observe("Two clinics reported fever cases.", origin="document",
                      source_ref="sitrep")
        h = m.derive("An outbreak is underway.", parents=[a],
                     kind="hypothesis", producer="analyst",
                     falsifier="check intake curves")
        new = m.correct(h, "A seasonal spike is underway.", by="bill",
                        reason="matches last year")
        chain = m.why(new)
        assert any(n["id"] == h for n in chain), \
            "the corrected claim must stay in its own provenance"


def test_correcting_an_inference_cannot_promote_it_to_fact():
    """Monotonicity applies to corrections like everything else. A user
    cannot launder a guess by fixing its wording."""
    with _mind() as m:
        a = m.observe("Two clinics reported fever cases.", origin="document",
                      source_ref="sitrep")
        h = m.derive("An outbreak is underway.", parents=[a],
                     kind="hypothesis", producer="analyst",
                     falsifier="check intake curves")
        new = m.correct(h, "A seasonal spike is underway.", by="bill")
        row = m._node_row(new)
        assert row["epistemic"] != "observed"
        assert row["confidence"] <= m._node_row(h)["confidence"] + 1e-9


def test_correcting_an_observation_is_attributed_to_the_person():
    """Not laundered into the original document's credibility."""
    with _mind() as m:
        a = m.observe("The bridge is intact.", origin="document",
                      source_ref="https://reliefweb.int/x", reliability="A",
                      credibility=1)
        new = m.correct(a, "The bridge collapsed.", by="bill")
        row = m._node_row(new)
        assert row["source_ref"] == "correction:bill"
        assert row["source_ref"] != "https://reliefweb.int/x"


def test_the_ledger_marks_what_has_been_corrected():
    with _mind() as m:
        a = m.observe("Dr Osman runs the clinic.", origin="document",
                      source_ref="sitrep")
        m.correct(a, "Dr Warsame runs the clinic.", by="bill")
        entry = next(e for e in m.ledger()["entries"] if e["node_id"] == a)
        assert entry["corrected"] is True


def test_correcting_an_unknown_node_raises():
    with _mind() as m:
        with pytest.raises(Exception):
            m.correct("obs_nope", "anything", by="bill")


# ── C7: inferring what it cost ───────────────────────────────────────────

def test_a_glance_costs_almost_nothing():
    assert salience.infer_acquisition_cost(elapsed_seconds=2.0) < 0.05


def test_a_canvass_is_expensive():
    canvass = salience.infer_acquisition_cost(
        elapsed_seconds=3600, tool_calls=8, sources_consulted=6,
        human_minutes=45)
    assert canvass > 0.6


def test_travel_is_a_step_change_not_a_bigger_number():
    """Anything requiring physical presence is expensive by definition and
    cannot be re-acquired from a desk."""
    desk = salience.infer_acquisition_cost(elapsed_seconds=60, tool_calls=1)
    trip = salience.infer_acquisition_cost(elapsed_seconds=60, tool_calls=1,
                                           travel=True)
    assert desk < 0.2 and trip >= 0.75


def test_cost_saturates_rather_than_scaling_linearly():
    """A linear scale would let one expensive outlier flatten everything
    else into indistinguishable cheapness."""
    few = salience.infer_acquisition_cost(sources_consulted=3)
    many = salience.infer_acquisition_cost(sources_consulted=40)
    lots = salience.infer_acquisition_cost(sources_consulted=400)
    assert few < many
    assert (lots - many) < (many - few), "returns must diminish"
    assert lots <= 1.0


def test_a_persons_time_dominates_machine_time():
    machine = salience.infer_acquisition_cost(elapsed_seconds=3600,
                                              tool_calls=20)
    human = salience.infer_acquisition_cost(human_minutes=60)
    assert human > machine


def test_expensive_memories_survive_pressure_that_removes_cheap_ones():
    """C7's acceptance criterion, at equal access frequency."""
    cheap = salience.salience(stability=2.0, difficulty=0.3,
                              elapsed=30 * 86400.0, surprise=0.0,
                              open_loop=False, acquisition_cost=0.0,
                              criticality=0.0)
    dear = salience.salience(stability=2.0, difficulty=0.3,
                             elapsed=30 * 86400.0, surprise=0.0,
                             open_loop=False, acquisition_cost=1.0,
                             criticality=0.0)
    assert dear > cheap
    assert dear / max(cheap, 1e-9) > 1.5, \
        "the difference has to be big enough to change what gets pruned"
