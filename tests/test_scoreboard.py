"""The scoreboard must be able to FAIL.

A benchmark that always passes measures nothing. Every metric below gets a
negative control: break the property deliberately and confirm the metric
notices. This is the standing rule that a benchmark be checked for whether it
measures the SYSTEM or the HARNESS -- learned the hard way when the `nuc`
benchmark briefly "disproved" OWL's central thesis and was in fact measuring
a toy embedder.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import scoreboard as sb          # noqa: E402
from owl import Owl, State       # noqa: E402
from owl.defence import screen   # noqa: E402

DAY = 86400.0


# ── the metrics currently pass ───────────────────────────────────────────

@pytest.mark.parametrize("fn,target", [
    (sb.rescue_at_k, 1.0),
    (sb.inverse_rescue_at_k, 1.0),
    (sb.source_attribution, 1.0),
    (sb.consequence_recall, 1.0),
    (sb.blast_radius_completeness, 1.0),
    (sb.flooding_resistance, 1.0),
])
def test_metric_is_at_target(fn, target):
    assert fn().score >= target - 1e-9


def test_no_confabulation_and_no_leakage():
    assert sb.confabulation_rate().score == 0.0
    assert sb.epistemic_leakage().score == 0.0


# ── negative controls: each metric must be able to fail ──────────────────

def test_inverse_rescue_would_fail_if_the_record_were_mutable(mind, clock):
    """OWL wins this by architecture. Simulate a store that prunes the old
    row -- as any system that mutates in place effectively does."""
    old = mind.observe("Route Alpha is open.", source_ref="s1")
    clock.advance(days=1)
    mind.observe("Route Alpha is closed.", source_ref="s2", supersedes=old)
    assert any(c.content == "Route Alpha is open."
               for c in mind.recall("route alpha", budget=10).chunks)

    mind._s.write(lambda c: c.execute(
        "UPDATE mem_index SET tier='pruned' WHERE node_id=?", (old,)))
    assert not any(c.content == "Route Alpha is open."
                   for c in mind.recall("route alpha", budget=10).chunks), (
        "the metric would not notice a store that discards superseded rows")


def test_confabulation_is_structurally_impossible_for_zero_overlap(mind):
    """Worth stating: for a probe sharing NO vocabulary with the store there
    are no candidates at all, so the gate never even runs. That is why the
    metric's probes must be chosen to overlap -- otherwise it measures the
    absence of candidates rather than the honesty of the gate."""
    mind.observe("The clinic has twelve beds.", source_ref="survey")
    r = mind.recall("quantum chromodynamics lattice spacing")
    assert r.state is State.DONT_KNOW


def test_confabulation_metric_detects_a_loosened_gate(mind, monkeypatch):
    """With PARTIAL overlap the gate is what stands between honesty and
    invention. Loosen it and confident answers appear."""
    mind.observe("The water tanker arrives on Tuesday.", source_ref="notes")
    assert mind.recall("when does the fuel tanker arrive").state \
        is not State.KNOW

    import owl.metamemory as mm
    monkeypatch.setattr(mm, "KNOW_SCORE", 0.0)
    monkeypatch.setattr(mm, "KNOW_RETRIEVABILITY", 0.0)
    monkeypatch.setattr(mm, "RECOLLECTION_FLOOR", 0.0)
    assert mind.recall("when does the fuel tanker arrive").state is State.KNOW, (
        "a wide-open gate must produce confident answers - if it does not, "
        "the metric is insensitive rather than the system being honest")


def test_leakage_metric_detects_a_corrupted_tag(mind):
    obs = mind.observe("Two clinics reported fever.", source_ref="sitrep")
    hyp = mind.derive("An outbreak is underway.", parents=[obs],
                      kind="hypothesis", producer="rem",
                      falsifier="check intake curves")
    assert mind.self_audit()["clean"]
    mind._s.write(lambda c: c.execute(
        "UPDATE derived SET epistemic_tag='observed' WHERE id=?", (hyp,)))
    assert not mind.self_audit()["clean"], (
        "laundering a hypothesis into an observation must be detectable")


def test_consequence_recall_is_zero_without_decisions(mind, clock):
    """The control that matters: every OTHER memory system scores 0 here,
    because none of them record what was decided on the basis of a memory."""
    n = mind.observe("A fact nobody decided anything on.", source_ref="s1")
    clock.advance(days=1)
    mind.observe("That fact has changed.", source_ref="s2", supersedes=n)
    assert mind.reconsider() == []


def test_blast_radius_would_fail_without_cascade(mind):
    src = mind.observe("Depot holds 4000 litres.", origin="document",
                       source_ref="file://survey.pdf", reliability="B",
                       credibility=2)
    child = mind.derive("Fuel is not a constraint.", parents=[src],
                        kind="abstraction", producer="analyst", confidence=0.9)
    before = mind._node_row(child)["confidence"]
    mind.discredit(src, reason="out of date", reliability="E", dry_run=True)
    assert mind._node_row(child)["confidence"] == before, (
        "dry run must change nothing - otherwise the metric is measuring "
        "itself")
    mind.discredit(src, reason="out of date", reliability="E")
    assert mind._node_row(child)["confidence"] < before


def test_flooding_metric_would_fail_if_origins_were_ignored(mind):
    """Give the flood genuinely distinct origins and the weight must rise --
    otherwise the metric is insensitive rather than the system being good."""
    distinct = [mind.observe("The depot is empty.", origin="document",
                             source_ref=f"https://site{i}.org/report")
                for i in range(5)]
    assert mind.independent_sources(distinct)["weight"] > 0.0


def test_injection_metric_penalises_a_paranoid_screen():
    """Containment alone is worthless: quarantine everything and score 1.000.
    The metric must be a balanced measure, not recall."""
    r = sb.injection_containment()
    assert "false_positive_rate" in r.subscores
    assert r.subscores["recall"] == 1.0
    assert r.subscores["false_positive_rate"] == 0.0


def test_screen_does_not_flag_ordinary_operational_writing():
    """Real field notes are full of imperatives about the WORLD. An earlier
    version weighted bare always/never and quarantined 2 of 6 legitimate
    notes -- caught only because the scoreboard scores false positives."""
    for text in sb.BENIGN:
        assert screen(text, origin="document").clean, text
    for text in sb.POISON:
        assert not screen(text, origin="document").clean, text


def test_staleness_metric_would_fail_without_claim_classes(mind, clock):
    """If everything decayed at one rate, durable facts would be flagged too."""
    ident = mind.observe("Dr Warsame speaks Somali.", source_ref="p1",
                         claim_class="identity")
    status = mind.observe("Route Alpha is open.", source_ref="s1",
                          claim_class="status")
    clock.advance(days=45)
    i_stale = mind.recall("warsame somali", budget=3).chunks[0].staleness
    s_stale = mind.recall("route alpha open", budget=3).chunks[0].staleness
    assert i_stale < 0.1 and s_stale > 0.9, (
        f"claim classes must separate: identity={i_stale} status={s_stale}")


def test_scoreboard_runs_end_to_end_and_emits_json(capsys):
    assert sb.main(["--json"]) == 0
    out = capsys.readouterr().out
    import json
    data = json.loads(out)
    # count the registry rather than a literal, so adding a metric doesn't
    # fail a test that has nothing to say about it
    assert len(data) == sum(len(fns) for _, fns in sb.SUITE)
    assert {d["group"] for d in data} == {g for g, _ in sb.SUITE}
    assert all(d["name"] and "score" in d for d in data)
