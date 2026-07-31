"""Theory of Mind — the epistemic plane. Not affect, not persuasion."""
from owl import State
from owl.theory_of_mind import CHANNEL_DEPTH, model_retention, resolve_direction

DAY = 86400.0


def test_models_the_persons_forgetting_not_its_own(mind, clock):
    """Every system models what the MACHINE knows. This models the person."""
    nid = mind.observe("Checkpoint protocol: radio ahead, headlights off at 200m.",
                       origin="document", source_ref="security-brief-v3")
    mind.tell("bill", nid, channel="briefing")

    assert mind.knows("bill", nid).retrievability > 0.9
    clock.advance(days=21)
    held = mind.knows("bill", nid)
    assert held.at_risk, f"retention {held.retrievability:.2f} should be at risk"
    # The system itself still holds it perfectly well -- that's the whole point.
    assert mind.recall("checkpoint protocol radio").state is State.KNOW


def test_depth_of_processing_beats_repetition(clock):
    """Craik & Lockhart: depth predicts retention better than exposure count."""
    now = 0.0
    skimmed = [(-30 * DAY, "briefing")]
    generated = [(-30 * DAY, "generated")]
    assert model_retention(generated, now) > model_retention(skimmed, now)


def test_at_risk_ranks_what_they_are_about_to_lose(mind, clock):
    a = mind.observe("Dr Warsame runs the clinic and speaks Somali.",
                     source_ref="brief")
    b = mind.observe("Fuel depot access code is 4471.", source_ref="brief")
    mind.tell("bill", a, channel="briefing")
    mind.tell("bill", b, channel="briefing")
    clock.advance(days=40)
    risk = mind.at_risk("bill", threshold=0.6)
    assert {h.node_id for h in risk} == {a, b}
    assert all(h.retrievability < 0.6 for h in risk)


def test_false_belief_detection(mind, clock):
    """Sally-Anne, made operational: the person holds a superseded belief."""
    old = mind.observe("Route Alpha is open.", source_ref="sitrep-1",
                       claim_class="status")
    mind.tell("bill", old, channel="conversation")
    clock.advance(days=2)
    mind.observe("Route Alpha is closed by flooding.", source_ref="sitrep-2",
                 claim_class="status", supersedes=old,
                 reliability="B", credibility=2)

    div = mind.divergence("bill")
    assert div, "failed to detect that the user is acting on stale information"
    d = div[0]
    assert d.held_node == old and d.direction == "user_stale"
    assert d.severity > 0.0

    # Once they have been told, the divergence clears.
    new = mind._s.one("SELECT new_node FROM supersession")["new_node"]
    mind.tell("bill", new, channel="conversation")
    assert not mind.divergence("bill")


def test_divergence_is_symmetric_the_machine_can_be_wrong():
    """A system that assumes the record wins will confidently correct someone
    who was standing at the checkpoint an hour ago."""
    direction, _ = resolve_direction(
        user_source_recency=3600.0, user_was_present=True,
        ledger_recency=3 * DAY, ledger_admiralty=0.85)
    assert direction == "ledger_stale"
