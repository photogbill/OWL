"""Actionable DONT_KNOW, and memory as an investment."""
from owl import State


def test_dont_know_says_what_would_be_needed(mind):
    """'I have nothing' is honest but inert. This turns it into a task."""
    mind.observe("The clinic has twelve beds.", source_ref="survey")
    r = mind.recall("who is the depot fuel supplier")
    assert r.state is State.DONT_KNOW
    assert "would need" in r.reason
    assert "supplier" in r.reason or "depot" in r.reason


def test_the_gap_statement_is_typed_by_the_question(mind):
    mind.observe("Unrelated content about tents.", source_ref="d1")
    person = mind.recall("who runs the Bardera warehouse")
    when = mind.recall("when does the Kismayo convoy depart")
    assert "person" in person.reason or "naming" in person.reason
    assert "date" in when.reason or "schedule" in when.reason


def test_expensive_memories_resist_decay(mind, clock):
    """The right question for forgetting is not 'how often was this used'
    but 'what would it cost me to get it back'."""
    cheap = mind.observe("The gate was repainted last spring.",
                         source_ref="note", acquisition_cost=0.0)
    dear = mind.observe("Only vendor stocking diesel is in Kismayo, "
                        "three days away.", source_ref="canvass",
                        acquisition_cost=1.0)
    clock.advance(days=120)

    got = {c.node_id: c.score for c in
           mind.recall("diesel vendor gate spring Kismayo", budget=5).chunks}
    assert dear in got, "expensive knowledge should survive"
    if cheap in got:
        assert got[dear] > got[cheap], (
            "an expensive memory must outrank a cheap one of equal age")


def test_acquisition_cost_is_clamped(mind):
    n = mind.observe("Something.", source_ref="d1", acquisition_cost=99.0)
    assert mind._node_row(n)["acquisition_cost"] == 1.0
