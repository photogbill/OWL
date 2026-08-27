"""Retroactive trust propagation — the inverse of why()."""
from owl import Epistemic, State


def _contaminated(mind):
    src = mind.observe("Depot holds 4000 litres of diesel.", origin="document",
                       source_ref="file://survey.pdf", reliability="B",
                       credibility=2)
    a = mind.derive("Fuel is not a constraint this month.", parents=[src],
                    kind="abstraction", producer="analyst", confidence=0.9)
    b = mind.derive("Generator runtime can be extended.", parents=[a],
                    kind="abstraction", producer="analyst", confidence=0.8)
    mind.tell("bill", a, channel="conversation")
    did = mind.decided("Extend clinic generator hours", because=[a])
    return src, a, b, did


def test_blast_radius_finds_everything_downstream(mind):
    src, a, b, did = _contaminated(mind)
    r = mind.blast_radius(src)
    assert a in r["derived"] and b in r["derived"]
    assert r["depth"] >= 2
    assert any(d["id"] == did for d in r["decisions"])
    assert any(t["who"] == "bill" for t in r["told"]), (
        "must surface who was told, not just what was concluded")


def test_discredit_cascades_and_never_deletes(mind):
    src, a, b, _ = _contaminated(mind)
    before = {n: mind._node_row(n)["confidence"] for n in (a, b)}

    plan = mind.discredit(src, reason="survey was three years out of date",
                          reliability="E")

    assert plan["people_to_notify"] == ["bill"]
    for n in (a, b):
        row = mind._node_row(n)
        assert row is not None, "discrediting must never delete"
        assert row["confidence"] < before[n]
        assert Epistemic(row["epistemic"]).rank >= Epistemic.INFERRED.rank
    # the original evidence is untouched -- having believed it is itself a fact
    assert mind._node_row(src)["content"].startswith("Depot holds 4000")


def test_discredit_dry_run_changes_nothing(mind):
    src, a, _, _ = _contaminated(mind)
    before = mind._node_row(a)["confidence"]
    plan = mind.discredit(src, reason="checking", dry_run=True)
    assert plan["demoted"] or plan["quarantined"]
    assert mind._node_row(a)["confidence"] == before


def test_discredit_flags_dependent_decisions(mind):
    src, _, _, did = _contaminated(mind)
    mind.discredit(src, reason="forged")
    open_items = mind.reconsider()
    assert any(i.decision_id == did for i in open_items), (
        "a decision resting on discredited evidence must be surfaced")


def test_the_killer_query(mind):
    """'I just learned that PDF was out of date. What did I conclude from it,
    what did I tell the team, and which decisions rested on it?'"""
    src, a, b, did = _contaminated(mind)
    r = mind.blast_radius(src)
    assert r["count"] == 2
    assert len(r["decisions"]) == 1
    assert len(r["told"]) == 1


def test_reliability_is_a_revisable_judgement_not_a_record(mind):
    """The append-only trigger caught this design error: what we believed at
    ingest is history; what we believe now is a separate, mutable layer."""
    nid = mind.observe("Depot holds 4000 litres.", origin="document",
                       source_ref="file://survey.pdf", reliability="B",
                       credibility=2)
    assert mind.effective_grade(nid) == ("B", 2)

    mind.assess_source("file://survey.pdf", reliability="E", credibility=5,
                       reason="three years out of date")

    assert mind.effective_grade(nid) == ("E", 5), "current grade must update"
    row = mind._node_row(nid)
    assert row["reliability"] == "B", (
        "the ingest-time grade is history and must not be rewritten")


def test_assessment_is_keyed_by_source_not_node(mind):
    """One write revalues every observation drawn from a bad source."""
    a = mind.observe("Claim one.", origin="document", source_ref="file://bad.pdf",
                     reliability="B", credibility=2)
    b = mind.observe("Claim two.", origin="document", source_ref="file://bad.pdf",
                     reliability="B", credibility=2)
    mind.assess_source("file://bad.pdf", reliability="E", reason="forged")
    assert mind.effective_grade(a)[0] == "E"
    assert mind.effective_grade(b)[0] == "E"
