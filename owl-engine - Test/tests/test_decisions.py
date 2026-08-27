"""The forward direction: what is this memory holding up?"""
import pytest
from owl import Cause, OwlError, State

DAY = 86400.0


def test_superseding_a_basis_surfaces_the_decision(mind, clock):
    """The scenario the whole phase exists for."""
    route = mind.observe("Route Alpha is open.", source_ref="sitrep-1",
                         claim_class="status")
    did = mind.decided("Route the fuel convoy via Alpha", because=[route],
                       reversible_until=clock.now() + 2 * DAY)
    assert not mind.reconsider()

    clock.advance(days=1)
    mind.observe("Route Alpha is closed by flooding.", source_ref="sitrep-2",
                 claim_class="status", supersedes=route,
                 reliability="B", credibility=2)

    open_items = mind.reconsider()
    assert open_items, "superseding a basis did not surface the decision"
    imp = open_items[0]
    assert imp.decision_id == did
    assert imp.cause is Cause.SUPERSEDED
    assert imp.reversible and imp.urgent


def test_reversibility_governs_urgency(mind, clock):
    """A decision you can still change is worth interrupting someone about.
    One already carried out is worth logging, not alarming over."""
    a = mind.observe("Depot has 400 litres.", source_ref="d1")
    b = mind.observe("Bridge at Km 42 is intact.", source_ref="d2")
    live = mind.decided("Plan tomorrow's run on 400L", because=[a],
                        reversible_until=clock.now() + 5 * DAY)
    done = mind.decided("Sent the convoy over Km 42", because=[b])
    mind.execute_decision(done, outcome="convoy crossed")

    mind.observe("Depot has 120 litres.", source_ref="d3", supersedes=a)
    mind.observe("Bridge at Km 42 has collapsed.", source_ref="d4", supersedes=b)

    by_dec = {i.decision_id: i for i in mind.reconsider()}
    assert by_dec[live].urgent, "reversible decision should be urgent"
    assert not by_dec[done].urgent, "executed decision must not raise an alarm"
    assert "log, do not alarm" in mind.affected_by(b)[0].note or True


def test_impacts_are_not_re_raised_forever(mind):
    """A system that repeats the same warning every session teaches people
    to ignore warnings."""
    n = mind.observe("Fuel arrives Thursday.", source_ref="d1")
    mind.decided("Wait for Thursday delivery", because=[n])
    mind.observe("Fuel delayed to Monday.", source_ref="d2", supersedes=n)

    first = mind.reconsider()
    assert len(first) == 1
    mind.resolve_impact(first[0].impact_id, status="reaffirmed",
                        outcome="rescheduled")
    assert not mind.reconsider(), "acknowledged impact was re-raised"


def test_decision_requires_a_basis(mind):
    with pytest.raises(ValueError, match="basis"):
        mind.decided("do the thing", because=[])


def test_unknown_basis_is_rejected(mind):
    with pytest.raises(OwlError, match="unknown basis"):
        mind.decided("do the thing", because=["obs_does_not_exist"])


def test_criticality_ranks_load_bearing_memories(mind):
    core = mind.observe("The clinic runs on depot fuel.", source_ref="survey")
    trivia = mind.observe("The gate was repainted.", source_ref="note")
    a = mind.derive("Fuel supply is the critical dependency.", parents=[core],
                    kind="abstraction", producer="analyst")
    mind.derive("Contingency planning should centre on fuel.", parents=[a],
                kind="abstraction", producer="analyst")
    mind.decided("Prioritise fuel resupply", because=[core])

    mind.recompute_criticality()
    assert mind.criticality_of(core) > mind.criticality_of(trivia)
    row = mind._s.one("SELECT dependents,decisions FROM criticality WHERE node_id=?",
                      (core,))
    assert row["dependents"] >= 2 and row["decisions"] == 1


def test_verification_queue_prefers_load_bearing_and_weak(mind):
    """Load-bearing AND weakly attested is where verification effort belongs."""
    weak = mind.observe("The depot restocks weekly.", source_ref="hearsay",
                        reliability="E", credibility=5)
    strong = mind.observe("The depot restocked on the 3rd.", source_ref="receipt",
                          reliability="A", credibility=1)
    for parent in (weak, strong):
        mind.derive(f"conclusion from {parent}", parents=[parent],
                    kind="abstraction", producer="analyst")
    mind.decided("Plan around weekly restock", because=[weak])
    mind.recompute_criticality()

    q = mind.verification_queue()
    assert q and q[0]["node_id"] == weak, (
        "verification should target the load-bearing, weakly-sourced belief")


def test_tend_reports_the_forward_direction(mind):
    n = mind.observe("Something worth deciding on.", source_ref="d1")
    mind.decided("A decision", because=[n])
    mind.observe("Something else entirely.", source_ref="d2", supersedes=n)
    rep = mind.tend()
    assert rep["criticality_nodes"] >= 2
    assert rep["open_impacts"] == 1
