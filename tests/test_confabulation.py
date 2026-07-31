"""A confident answer to an absent fact is the one unforgivable failure."""
from owl import State


def test_absent_fact_returns_dont_know(mind):
    mind.observe("The clinic in Bardera has 12 beds.")
    mind.observe("Fuel convoy departs at 0600.")
    r = mind.recall("what is the helicopter tail number")
    assert r.state is State.DONT_KNOW
    assert r.chunks == []


def test_dont_know_is_fast(mind):
    for i in range(200):
        mind.observe(f"Log entry {i}: routine supply check completed.")
    r = mind.recall("quantum chromodynamics lattice spacing")
    assert r.state is State.DONT_KNOW
    assert r.latency_ms < 100


def test_partial_overlap_does_not_fabricate_certainty(mind):
    mind.observe("The water tanker arrives Tuesday.")
    r = mind.recall("when does the fuel tanker arrive")
    # It may legitimately surface the water tanker, but must not claim KNOW.
    # FAMILIAR is the most honest of these: partial match, nothing anchoring
    # it, so "I have seen something like this and cannot place it".
    if r.chunks:
        assert r.state in (State.KNOW_WHERE, State.FAMILIAR,
                           State.TIP_OF_TONGUE)
