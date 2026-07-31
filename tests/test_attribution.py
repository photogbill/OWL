"""Source independence, attributed belief, and the commitment loop."""
import pytest
from owl import State
from owl.attribution import (Record, canonical_name, corroboration_weight,
                             independence, origin_key, proposition_hash)

DAY = 86400.0


# ── source independence ──────────────────────────────────────────────────

def test_origin_key_separates_people_not_just_hosts():
    """conv:ahmed and conv:warsame are two sources, not one."""
    assert origin_key("conv:ahmed:14") != origin_key("conv:warsame:2")
    assert origin_key("conv:ahmed:14") == origin_key("conv:ahmed:88")
    assert origin_key("file://survey/2023/a.pdf") == \
           origin_key("file://survey/2023/b.pdf")


def test_flooding_earns_no_corroboration_credit(mind):
    """200 documents from one origin is ONE source. This is the attack."""
    flood = [mind.observe("The depot is empty.", origin="document",
                          source_ref=f"file://attacker/doc{i}.pdf")
             for i in range(20)]
    r = mind.independent_sources(flood)
    assert r["documents"] == 20
    assert r["independent"] == 1
    assert r["weight"] == 0.0, "one origin must earn no corroboration credit"
    assert "single origin" in r["note"]


def test_genuinely_independent_sources_do_earn_credit(mind):
    nodes = [
        mind.observe("The bridge at Km 42 is out.", origin="document",
                     source_ref="file://survey/report.pdf"),
        mind.observe("The bridge at Km 42 is out.", origin="user_utterance",
                     source_ref="conv:ahmed:3"),
        mind.observe("The bridge at Km 42 is out.", origin="document",
                     source_ref="https://reliefweb.int/sitrep/9"),
    ]
    r = mind.independent_sources(nodes)
    assert r["independent"] == 3
    assert r["weight"] > 0.5


def test_corroboration_weight_curve():
    assert corroboration_weight(1) == 0.0
    assert corroboration_weight(2) == 0.5
    assert corroboration_weight(3) > corroboration_weight(2)


def test_corroborated_finds_the_same_proposition_across_sources(mind):
    for ref in ("file://a/x.pdf", "conv:ahmed:1", "https://relief.int/y"):
        mind.observe("The north well pump is broken.", origin="document",
                     source_ref=ref)
    r = mind.corroborated("the north well pump is broken")
    assert r["independent"] == 3


# ── attributed belief ────────────────────────────────────────────────────

def test_claim_is_separate_from_claimant(mind):
    n = mind.observe("Ahmed said the parts arrive Thursday.",
                     origin="user_utterance", source_ref="conv:ahmed:1")
    mind.claimed("Ahmed", "the parts arrive Thursday", node_id=n)

    who = mind.who_claims("the parts arrive Thursday")
    assert who and who[0]["who"] == "Ahmed"
    assert mind.record_of("Ahmed").claims_made == 1


def test_claimants_dedupe_conservatively(mind):
    a = mind.claimant("Dr. Warsame")
    b = mind.claimant("Warsame")
    assert a == b
    assert mind.claimant("Ahmed Hassan") != mind.claimant("Ahmed Hussein")


def test_unknown_claimant_is_unknown_not_bad(mind):
    """Treating an unrated source as unreliable is as wrong as trusting it."""
    rec = mind.record_of("Somebody New")
    assert rec.grade == "F", "F means 'cannot be judged', not 'bad'"
    assert rec.accuracy == 0.5


def test_reliability_is_learned_from_outcomes(mind):
    n = mind.observe("Claims go here.", source_ref="conv:ahmed:1")
    ids = [mind.claimed("Ahmed", f"proposition {i}", node_id=n)
           for i in range(5)]
    for i, cid in enumerate(ids):
        mind.resolve_claim(cid, confirmed=(i < 4))
    rec = mind.record_of("Ahmed")
    assert rec.confirmed == 4 and rec.refuted == 1
    assert rec.grade in ("B", "C")
    assert "accurate over" in rec.describe()


# ── commitments ──────────────────────────────────────────────────────────

def test_a_promise_is_not_a_fact(mind, clock):
    n = mind.observe("Ahmed: I'll bring fuel Thursday.",
                     origin="user_utterance", source_ref="conv:ahmed:1")
    mind.committed("Ahmed", "bring fuel", due=clock.now() + 3 * DAY, node_id=n)

    assert not mind.due_commitments(), "not due yet"
    clock.advance(days=4)
    due = mind.due_commitments()
    assert due and due[0]["who"] == "Ahmed"
    assert due[0]["overdue_days"] >= 1


def test_broken_promises_revalue_everything_that_source_said(mind, clock):
    """The loop nobody closes: broken promise -> lower reliability ->
    automatic revaluation of every claim from that source."""
    n = mind.observe("Ahmed reports the depot is full.",
                     origin="user_utterance", source_ref="conv:ahmed:1",
                     reliability="B", credibility=2)
    mind.claimed("Ahmed", "the depot is full", node_id=n)
    assert mind.effective_grade(n)[0] == "B"

    for i in range(4):
        m = mind.committed("Ahmed", f"promise {i}",
                           due=clock.now() + DAY, node_id=n)
        clock.advance(days=2)
        rec = mind.resolve_commitment(m, kept=False)

    assert rec.broken == 4
    assert rec.grade == "E", f"four broken promises should tank the grade"
    assert mind.effective_grade(n)[0] == "E", (
        "the source Ahmed spoke through must be revalued automatically")


def test_a_good_record_raises_the_grade(mind, clock):
    """A perfect but SHORT record does not earn top grade. Laplace smoothing
    means grade A has to be earned over time -- five kept promises is a good
    sign, not a certification."""
    n = mind.observe("Warsame reports clinic stock levels.",
                     origin="user_utterance", source_ref="conv:warsame:1")

    def keep(count):
        rec = None
        for i in range(count):
            m = mind.committed("Warsame", f"delivery {i}",
                               due=clock.now() + DAY, node_id=n)
            clock.advance(days=2)
            rec = mind.resolve_commitment(m, kept=True)
        return rec

    rec = keep(5)
    assert rec.grade == "B", "5/5 is good, not yet grade A"
    assert mind.effective_grade(n)[0] == "B"

    rec = keep(8)                       # 13/13
    assert rec.grade == "A", "a long clean record does earn it"
    assert mind.effective_grade(n)[0] == "A"


def test_too_little_history_does_not_move_the_grade(mind, clock):
    """A confident grade from two data points is worse than no grade."""
    n = mind.observe("Someone reports something.", source_ref="conv:new:1",
                     reliability="C", credibility=3)
    m = mind.committed("New Person", "one promise", due=clock.now() + DAY,
                       node_id=n)
    clock.advance(days=2)
    rec = mind.resolve_commitment(m, kept=False)
    assert rec.resolved == 1
    assert rec.grade == "F"
    assert mind.effective_grade(n)[0] == "C", (
        "one data point must not overwrite the assigned grade")


def test_a_due_commitment_raises_a_prospective_intention(mind, clock):
    n = mind.observe("Ahmed: fuel Thursday.", source_ref="conv:ahmed:1")
    mind.committed("Ahmed", "bring fuel", due=clock.now() + DAY, node_id=n)
    clock.advance(days=2)
    assert any("bring fuel" in i["action"] for i in mind.due())


def test_credibility_tracks_the_grade_not_string_order(mind, clock):
    """`3 if grade <= "C" else 4` is a string compare: "E" <= "C" is False,
    so the worst grades were silently given the BETTER credibility."""
    n = mind.observe("Claims from a bad source.", source_ref="conv:bad:1",
                     reliability="B", credibility=2)
    for i in range(5):
        m = mind.committed("Bad Source", f"promise {i}",
                           due=clock.now() + DAY, node_id=n)
        clock.advance(days=2)
        mind.resolve_commitment(m, kept=False)
    rel, cred = mind.effective_grade(n)
    assert rel == "E"
    assert cred == 5, f"grade E must carry credibility 5, got {cred}"


def test_a_good_source_gets_good_credibility(mind, clock):
    n = mind.observe("Claims from a good source.", source_ref="conv:good:1")
    for i in range(13):
        m = mind.committed("Good Source", f"promise {i}",
                           due=clock.now() + DAY, node_id=n)
        clock.advance(days=2)
        mind.resolve_commitment(m, kept=True)
    assert mind.effective_grade(n) == ("A", 1)
