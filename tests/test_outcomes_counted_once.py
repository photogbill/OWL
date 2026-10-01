"""An outcome is counted once (2026-10-01).

Found from ATK, which puts Confirmed / Refuted and Kept / Broken buttons on
claims and promises. A second click added a second outcome: one claim
confirmed and then refuted counted as both, and three clicks on ONE claim
crossed the three-outcome threshold after which every source the person
spoke through is revalued. A track record that a double-click can write is
not a track record.

Also here: a correction of an observation supersedes it, and why() on the
correction now reaches what was believed before -- it already did for a
corrected inference, which is derived from what it corrects.
"""
import os
import tempfile
import time

import pytest

from owl import Owl, OwlError


def _mind():
    return Owl.open(os.path.join(tempfile.mkdtemp(), "x.owl"))


def _claim(m, who="Farid Noori", text="the depot holds 4000 L"):
    n = m.observe(f"{who}: {text}", origin="user_utterance",
                  source_ref=f"conv:{who}")
    return m.claimed(who, text, node_id=n)


def test_the_same_outcome_twice_is_counted_once():
    with _mind() as m:
        c = _claim(m)
        m.resolve_claim(c, confirmed=True)
        m.resolve_claim(c, confirmed=True)
        rec = m.record_of("Farid Noori")
        assert (rec.confirmed, rec.refuted) == (1, 0)


def test_the_other_outcome_is_refused_without_revise():
    with _mind() as m:
        c = _claim(m)
        m.resolve_claim(c, confirmed=True)
        with pytest.raises(OwlError) as e:
            m.resolve_claim(c, confirmed=False)
        assert "already resolved as confirmed" in str(e.value)
        rec = m.record_of("Farid Noori")
        assert (rec.confirmed, rec.refuted) == (1, 0), "nothing moved"


def test_revise_moves_the_count_it_does_not_add_one():
    with _mind() as m:
        c = _claim(m)
        m.resolve_claim(c, confirmed=True)
        m.resolve_claim(c, confirmed=False, revise=True)
        rec = m.record_of("Farid Noori")
        assert (rec.confirmed, rec.refuted) == (0, 1)
        assert rec.resolved == 1
        row = m._s.one("SELECT outcome FROM claim WHERE id=?", (c,))
        assert row["outcome"] == "refuted"


def test_one_claim_clicked_three_times_does_not_judge_a_source():
    with _mind() as m:
        c = _claim(m)
        for _ in range(3):
            m.resolve_claim(c, confirmed=False)
        assert m.record_of("Farid Noori").resolved == 1


def test_a_promise_is_kept_once():
    with _mind() as m:
        n = m.observe("Logistics: 2 generators by Friday",
                      origin="user_utterance", source_ref="conv:log")
        p = m.committed("Logistics cell", "2 generators",
                        due=time.time() + 86400, node_id=n)
        m.resolve_commitment(p, kept=True)
        m.resolve_commitment(p, kept=True, note="both delivered")
        rec = m.record_of("Logistics cell")
        assert (rec.kept, rec.broken) == (1, 0)
        row = m._s.one("SELECT note FROM commitment WHERE id=?", (p,))
        assert row["note"] == "both delivered", "a later note is kept"
        with pytest.raises(OwlError):
            m.resolve_commitment(p, kept=False)
        m.resolve_commitment(p, kept=False, revise=True, note="one failed")
        rec = m.record_of("Logistics cell")
        assert (rec.kept, rec.broken) == (0, 1)


def test_an_unknown_claim_or_promise_is_still_none():
    with _mind() as m:
        assert m.resolve_claim("clm_nope", confirmed=True) is None
        assert m.resolve_commitment("cmt_nope", kept=True) is None


def test_why_on_a_corrected_observation_reaches_the_original():
    with _mind() as m:
        a = m.observe("The depot at Varna holds 4000 litres.",
                      origin="document", source_ref="sitrep-14")
        new = m.correct(a, "It holds 1500 litres.", by="bill",
                        reason="stock count")
        chain = m.why(new)
        assert chain[0]["id"] == new
        assert {"id": a, "role": "superseded"} in chain[0]["parents"], \
            "the role is the parent's, as on a derivation edge"
        assert any(n["id"] == a and n["source_ref"] == "sitrep-14"
                   for n in chain), "what was believed, and where it came from"


def test_why_on_an_uncorrected_observation_is_unchanged():
    with _mind() as m:
        a = m.observe("Route Alpha is open.", origin="document",
                      source_ref="sitrep")
        chain = m.why(a)
        assert len(chain) == 1 and chain[0]["parents"] == []


def test_the_same_claim_recorded_twice_while_open_is_one_claim():
    """A double-click on "Claimed by…" was two assertions, and then two
    outcomes."""
    with _mind() as m:
        a = _claim(m)
        b = _claim(m)
        assert a == b
        assert m.record_of("Farid Noori").claims_made == 1
        m.resolve_claim(a, confirmed=True)
        c = _claim(m)
        assert c != a, "said again after it was settled: a new claim"
        assert m.record_of("Farid Noori").claims_made == 2


def test_two_people_asserting_one_thing_are_two_claims():
    """Negative control: corroboration is not deduplicated away."""
    with _mind() as m:
        a = _claim(m, who="Farid Noori")
        b = _claim(m, who="Ahmad Karimi")
        assert a != b
