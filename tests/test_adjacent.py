"""B8 -- adjacent material, and the guard that keeps it from being a lie.

"Nothing on tanker arrival, but I have the depot dispatch schedule" is a
genuinely useful answer and it is ONE STEP from "I have nothing, but here
are five loosely related things" -- the exact behaviour the six-state
design exists to prevent. Adding an apology to the front of a bad result
set does not make it a good one.

So most of these tests are about staying quiet.
"""
import os
import tempfile

import pytest

from owl import Owl
from owl import adjacent as adj


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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "b8.owl"),
                    embedder=Toy(), **kw)


def _c(nid, content, score):
    return {"node_id": nid, "content": content, "score": score}


# ── the guard ────────────────────────────────────────────────────────────

def test_a_weak_match_is_not_offered_as_adjacent():
    """Taking the best of a bad set and calling it adjacent is
    max-normalisation wearing a hat -- a defect already fixed twice in this
    codebase, in the lexical scorer and again in the semantic blend."""
    weak = [_c("n1", "The tanker depot roster was revised.", 0.10),
            _c("n2", "The tanker route was repainted.", 0.12)]
    assert adj.find("when does the tanker arrive", weak).offered is False


def test_strength_is_absolute_not_relative_to_the_pack():
    """The best of three bad candidates is still bad."""
    pack = [_c("n1", "tanker depot dispatch schedule", 0.20),
            _c("n2", "tanker note two", 0.05),
            _c("n3", "tanker note three", 0.04)]
    assert adj.find("when does the tanker arrive", pack).offered is False


def test_unrelated_material_is_not_adjacent():
    unrelated = [_c("n1", "Solar panels on block C were wiped down.", 0.9)]
    assert adj.find("when does the tanker arrive", unrelated).offered is False


def test_it_offers_a_genuinely_neighbouring_note():
    """The plan's own example."""
    cands = [_c("n1", "The depot dispatch schedule lists tanker runs on "
                      "request.", 0.8)]
    s = adj.find("when does the tanker arrive", cands)
    assert s.offered
    assert "dispatch schedule" in s.sentence()


def test_at_most_two_are_offered():
    """A list is a search result. Two is a pointer."""
    many = [_c(f"n{i}", f"tanker dispatch note number {i}", 0.9)
            for i in range(6)]
    assert len(adj.find("when does the tanker arrive", many).items) == 2


def test_the_sentence_leads_with_the_absence():
    """A sentence opening with the suggestion invites it to be read as the
    answer."""
    cands = [_c("n1", "The depot dispatch schedule lists tanker runs.", 0.8)]
    text = adj.find("when does the tanker arrive", cands).sentence()
    assert text.startswith("Nothing directly on that")


def test_an_empty_query_offers_nothing():
    assert adj.find("", [_c("n1", "anything", 0.9)]).offered is False


def test_a_candidate_of_the_asked_for_type_is_not_re_offered():
    """If it carries the type the question wanted, the gate already
    considered and rejected it. Re-offering here would overrule that
    decision through a side door."""
    person = [_c("n1", "Dr Warsame runs the clinic in Bardera.", 0.9)]
    assert adj.find("who runs the clinic", person).offered is False


# ── integration ──────────────────────────────────────────────────────────

def test_adjacent_never_attaches_to_a_real_answer():
    """It never appends to a KNOW. A confident answer trailing suggestions
    is a confident answer you have made harder to read."""
    with _mind() as m:
        m.observe("The depot dispatch schedule lists tanker runs on request.",
                  origin="document", source_ref="sitrep")
        r = m.recall("depot dispatch schedule")
        if r.chunks:
            assert r.adjacent == ()


def test_adjacent_is_never_mixed_into_chunks():
    """Its own field, always. Mixing it in would make 'I have nothing' and
    'here are five bad matches' indistinguishable again -- which is the
    thing the six states exist to separate."""
    with _mind() as m:
        for i in range(5):
            m.observe(f"The depot dispatch schedule note {i}.",
                      origin="document", source_ref="sitrep")
        # The invariant, stated so it does not depend on whether the toy
        # embedder happens to match: adjacent material may exist ONLY when
        # there is no answer. Asserting a particular state here would be
        # testing the fixture's encoder rather than the guard.
        for q in ("what is the helicopter tail number", "depot dispatch",
                  "when does the school term start", "solar panels"):
            r = m.recall(q)
            assert isinstance(r.adjacent, tuple)
            if r.adjacent:
                assert r.chunks == [], q
                assert r.state.value == "dont_know", q


def test_a_store_with_nothing_nearby_stays_silent():
    with _mind() as m:
        m.observe("Solar panels on block C were wiped down.",
                  origin="document", source_ref="sitrep")
        r = m.recall("what is the helicopter tail number")
        assert r.adjacent == ()
        assert "Nearby" not in r.reason


# ── does it earn its keep? ───────────────────────────────────────────────

def test_the_keep_verdict_can_say_cut_it():
    """The plan says cut this if it does not earn its keep, so the verdict
    has to be able to say so."""
    good, bad = adj.Keep(), adj.Keep()
    for i in range(20):
        good.record(was_offered=True, was_used=i < 12)
        bad.record(was_offered=True, was_used=i < 2)
    assert good.verdict()["keep"] is True
    assert bad.verdict()["keep"] is False
    assert "clever and ignored" in bad.verdict()["verdict"]


def test_a_thin_sample_keeps_observing_rather_than_deciding():
    k = adj.Keep()
    for _ in range(3):
        k.record(was_offered=True, was_used=False)
    v = k.verdict()
    assert v["rate"] is None and v["keep"] is True
    assert "insufficient" in v["verdict"]
