"""Retrievability and credibility are different questions."""
from owl import State
from owl.epistemics import (DAY, admiralty_weight, classify, corroborate,
                            credibility, fit_halflife)


def test_claim_classes_have_different_half_lives():
    assert classify("Route Alpha is open.") == "status"
    assert classify("The clinic has 12 beds.") == "capacity"
    assert classify("Dr Warsame speaks Somali.") == "identity"
    assert credibility(30 * DAY, "identity") == 1.0
    assert credibility(30 * DAY, "status") < 0.01
    assert credibility(30 * DAY, "capacity") > 0.8


def test_stale_status_stays_retrievable(mind, clock):
    """The dangerous case: perfectly findable, and no longer true."""
    mind.observe("Route Alpha is open.", source_ref="sitrep-1")
    clock.advance(days=21)
    r = mind.recall("route alpha")
    c = r.chunks[0]
    assert c.retrievability > 0.4, "should still be easy to find"
    assert c.staleness > 0.9, "should be flagged as almost certainly stale"
    assert not c.trustworthy


def test_halflife_is_learned_from_supersessions(mind, clock):
    prev = None
    for i in range(14):
        nid = mind.observe(f"Generator fuel level is {90 - i * 5} percent.",
                           source_ref=f"gauge-{i}", claim_class="status",
                           supersedes=prev)
        prev = nid
        clock.advance(days=2)
    fitted = mind._halflife("status")
    assert fitted is not None, "should fit once enough supersessions accumulate"
    assert 0.5 * DAY < fitted < 4 * DAY


def test_admiralty_corroboration_raises_credibility():
    assert admiralty_weight("A", 1) > admiralty_weight("D", 4)
    # 'cannot be judged' is mid-scale, not low: treating unknown provenance as
    # unreliable is as wrong as trusting it.
    assert admiralty_weight("F", 6) > admiralty_weight("E", 5)
    rel, cred = corroborate(("B", 3), ("C", 3))
    assert rel == "B" and cred == 2
