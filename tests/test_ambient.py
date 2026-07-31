"""F3, F4, F6, F7 -- ambient operation.

The common thread is restraint. Each of these features is a way for memory
to act without being asked, and the failure mode of all of them is the same:
being right often enough to be trusted and frequent enough to be ignored.
"""
import os
import tempfile

import pytest

from owl import Owl
from owl.anticipation import Watcher


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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "a.owl"),
                    embedder=Toy(), **kw)


# ── F3: session prefix ───────────────────────────────────────────────────

def test_prefix_is_empty_when_nothing_is_outstanding():
    """Silence is the right answer for a clean slate. A prefix that always
    has something in it trains people to skip it."""
    with _mind() as m:
        m.observe("An ordinary note about the compound.")
        assert m.prefix()["empty"] is True


def test_prefix_leads_with_consequence_not_recency():
    with _mind() as m:
        basis = m.observe("Route Alpha is open.", origin="document",
                          source_ref="sitrep-1")
        m.decided("Send the convoy via Route Alpha", because=[basis])
        m.intend("call the depot", on_event="next contact")
        m.observe("The newest note, which matters least.")
        m.observe("Route Alpha is closed by flooding.", origin="document",
                  source_ref="sitrep-2", supersedes=basis)

        p = m.prefix()
        assert not p["empty"]
        titles = [s["title"] for s in p["sections"]]
        assert titles[0] == "Standing on shifted ground"
        assert "newest note" not in p["text"]


def test_prefix_drops_whole_tiers_rather_than_truncating():
    """Half an open loop reads as a complete one, which is worse than
    omitting it."""
    with _mind() as m:
        for i in range(8):
            m.intend(f"a fairly long open loop number {i} about logistics",
                     on_event=f"depot contact number {i}")
        tight = m.prefix(token_budget=12)
        assert tight["tokens"] <= 12
        for s in tight["sections"]:
            assert all(not i.endswith("…") for i in s["items"])


def test_prefix_respects_its_budget():
    with _mind() as m:
        for i in range(20):
            m.intend(f"open loop {i}", on_event=f"depot contact number {i}")
        assert m.prefix(token_budget=400)["tokens"] <= 400


# ── F4: anticipation, which is mostly silence ────────────────────────────

CANDS = [
    {"node_id": "d1", "kind": "shifted_basis",
     "text": "Send the convoy via Route Alpha",
     "message": "This rests on something that changed"},
    {"node_id": "i1", "kind": "open_loop",
     "text": "call the depot about fuel",
     "message": "Still open: call the depot about fuel"},
]


def test_it_stays_quiet_on_an_unrelated_turn():
    w = Watcher()
    assert w.consider("what is the weather like today", CANDS) is None
    assert w.spent == 0


def test_it_speaks_when_the_turn_is_actually_about_it():
    w = Watcher()
    n = w.consider("should we send the convoy via route alpha", CANDS)
    assert n is not None and n.node_id == "d1"


def test_it_never_raises_the_same_thing_twice():
    """The single most important restraint. Being right twice about the
    same thing is being wrong the second time."""
    w = Watcher(cooldown=0)
    assert w.consider("send the convoy via route alpha", CANDS) is not None
    assert w.consider("send the convoy via route alpha", CANDS) is None


def test_interruptions_cannot_cluster():
    w = Watcher(cooldown=5)
    assert w.consider("send the convoy via route alpha", CANDS) is not None
    assert w.consider("call the depot about fuel", CANDS) is None, \
        "a cooldown must apply across DIFFERENT candidates too"


def test_the_session_budget_is_hard():
    w = Watcher(max_per_session=1, cooldown=0)
    assert w.consider("send the convoy via route alpha", CANDS) is not None
    assert w.consider("call the depot about fuel", CANDS) is None
    assert w.exhausted


def test_consequence_outranks_a_mere_open_loop():
    """One is something you meant to do; the other is something you already
    did for a reason that stopped being true."""
    w = Watcher(cooldown=0)
    both = [{"node_id": "x", "kind": "open_loop", "text": "the depot fuel run",
             "message": "loop"},
            {"node_id": "y", "kind": "shifted_basis",
             "text": "the depot fuel run", "message": "shifted"}]
    assert w.consider("about the depot fuel run", both).node_id == "y"


def test_the_verdict_can_say_turn_it_off():
    """The acceptance criterion has to be able to fail, or shipping it off
    by default was theatre."""
    good, bad, thin = Watcher(), Watcher(), Watcher()
    for i in range(10):
        good.record_outcome(f"n{i}", acted_on=i < 8)
        bad.record_outcome(f"n{i}", acted_on=i < 2)
    thin.record_outcome("n0", acted_on=True)

    assert good.verdict()["keep_enabled"] is True
    assert bad.verdict()["keep_enabled"] is False
    assert "turn it off" in bad.verdict()["verdict"]
    assert thin.verdict()["ratio"] is None
    assert thin.verdict()["keep_enabled"] is False, \
        "a single success is not evidence"


def test_watcher_is_off_unless_asked_for():
    with _mind() as m:
        assert not hasattr(m, "_watcher")
        assert isinstance(m.watch(), Watcher)


def test_candidates_are_consequences_not_mere_relevance():
    with _mind() as m:
        m.observe("A topically relevant note about convoys and routes.")
        m.intend("call the depot", on_event="next contact")
        cands = m.nudge_candidates()
        kinds = {c["kind"] for c in cands}
        assert kinds <= {"shifted_basis", "commitment", "open_loop"}
        assert not any("topically relevant" in c["text"] for c in cands)


# ── F6: a pack you can read ──────────────────────────────────────────────

def _pack(tmp):
    with _mind() as m:
        a = m.observe("Route Alpha floods above 40mm rainfall.",
                      origin="document", source_ref="file://survey.pdf",
                      reliability="B", credibility=2)
        b = m.observe("The clinic generator runs on depot fuel.",
                      origin="document", source_ref="conv:ahmed:2")
        m.derive("Fuel resupply is currently viable.", parents=[a, b],
                 kind="abstraction", producer="analyst",
                 falsifier="check depot stock")
        m.record_absence("helicopter tail number", scope="all sitreps")
        m.intend("confirm the delivery window", on_event="next contact")
        p = os.path.join(tmp, "h.owlpack")
        man = m.export_pack(p, exporter="Warsame", label="Bardera handover")
        return p, man


def test_export_writes_a_review_copy_beside_the_pack():
    tmp = tempfile.mkdtemp()
    path, man = _pack(tmp)
    assert os.path.exists(man["review_copy"])
    assert man["review_copy"].endswith(".md")


def test_the_review_copy_shows_what_the_RECIPIENT_will_see():
    """Not what the exporter sees. The demotion is the whole point of the
    format, so a review that hides it reviews the wrong thing."""
    tmp = tempfile.mkdtemp()
    path, _ = _pack(tmp)
    with _mind() as m:
        md = m.review_pack(path)
    assert "drops one rank on import" in md
    assert "inferred → hypothesized" in md
    assert "These are not facts" in md


def test_the_review_copy_carries_the_expensive_parts():
    tmp = tempfile.mkdtemp()
    path, _ = _pack(tmp)
    with _mind() as m:
        md = m.review_pack(path)
    assert "helicopter tail number" in md, "absences are the costly part"
    assert "confirm the delivery window" in md
    assert "file://survey.pdf" in md, "provenance must survive the rendering"
    assert "Admiralty B2" in md


# ── F7: convergence ──────────────────────────────────────────────────────

def _operator(tmp, name, claims):
    with _mind() as m:
        for text, ref in claims:
            m.observe(text, origin="document", source_ref=ref)
        p = os.path.join(tmp, f"{name}.owlpack")
        m.export_pack(p, exporter=name)
        return p


def test_two_operators_with_separate_origins_corroborate():
    tmp = tempfile.mkdtemp()
    claim = "The bridge at Km 42 has collapsed."
    a = _operator(tmp, "Warsame", [(claim, "https://reliefweb.int/a")])
    b = _operator(tmp, "Osman", [(claim, "conv:driver:7")])
    with _mind() as m:
        out = m.converge([a, b])
    assert len(out["corroborated"]) == 1
    row = out["corroborated"][0]
    assert row["n_operators"] == 2 and row["independent_origins"] == 2
    assert row["weight"] > 0


def test_operators_echoing_one_document_earn_nothing():
    """The attack, and the honest mistake. Three people who read the same
    sitrep are one source, and counting packs would call that triple
    corroboration."""
    tmp = tempfile.mkdtemp()
    claim = "The depot is empty."
    same = "https://reliefweb.int/the-same-sitrep"
    a = _operator(tmp, "Warsame", [(claim, same)])
    b = _operator(tmp, "Osman", [(claim, same)])
    c = _operator(tmp, "Ahmed", [(claim, same)])
    with _mind() as m:
        out = m.converge([a, b, c])
    assert out["corroborated"] == []
    row = next(r for r in out["uncorroborated"] if claim in r["claim"])
    assert row["n_operators"] == 3 and row["independent_origins"] == 1
    assert "ONE origin" in row["note"]


def test_three_operator_scenario():
    """F7's acceptance criterion, with a mix of both cases at once."""
    tmp = tempfile.mkdtemp()
    shared = "Route Alpha is closed."
    solo = "The clinic has a new generator."
    corr = "Fuel arrives on Thursday."
    a = _operator(tmp, "Warsame", [(shared, "sitrep://common"),
                                   (corr, "conv:ahmed:1"), (solo, "obs://a")])
    b = _operator(tmp, "Osman", [(shared, "sitrep://common"),
                                 (corr, "https://logs.example/x")])
    c = _operator(tmp, "Ahmed", [(shared, "sitrep://common")])
    with _mind() as m:
        out = m.converge([a, b, c])
    corroborated = {r["claim"] for r in out["corroborated"]}
    assert corr in corroborated, "two operators, two origins"
    assert shared not in corroborated, "three operators, one document"
    assert solo not in corroborated, "one operator"
    assert len(out["operators"]) == 3


def test_converge_imports_nothing():
    """Convergence is evidence for a graft decision, not the decision."""
    tmp = tempfile.mkdtemp()
    a = _operator(tmp, "Warsame", [("A claim.", "ref://a")])
    with _mind() as m:
        before = m._scalar("SELECT COUNT(*) FROM observation")
        m.converge([a])
        assert m._scalar("SELECT COUNT(*) FROM observation") == before
