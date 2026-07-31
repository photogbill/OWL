"""A7 second-order uncertainty + B9 approximate search.

Both are about refusing to hide a trade. A scalar confidence hides the
difference between "no evidence" and "evidence both ways"; an ANN index
hides the recall it costs you. Each here reports what it gave up.
"""
import math
import os
import random
import tempfile

import pytest

from owl import Owl, ann
from owl import opinion as op
from owl.protocols import Space


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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "o.owl"),
                    embedder=Toy(), **kw)


# ── A7 ───────────────────────────────────────────────────────────────────

def test_no_evidence_and_balanced_evidence_are_distinguishable():
    """A7's acceptance criterion, and the reason it exists. Both project to
    ~0.5 and they call for opposite actions."""
    nothing = op.from_evidence(0, 0)
    contested = op.from_evidence(6, 6)

    assert abs(nothing.expectation - contested.expectation) < 0.01
    assert nothing.vacuous and not nothing.contested
    assert contested.contested and not contested.vacuous
    assert "go and look" in nothing.verdict
    assert "not resolve it" in contested.verdict


def test_masses_must_sum_to_one():
    """Mass that does not sum to 1 came from nowhere."""
    with pytest.raises(ValueError):
        op.Opinion(belief=0.9, disbelief=0.9, uncertainty=0.9)
    with pytest.raises(ValueError):
        op.Opinion(belief=-0.1, disbelief=0.5, uncertainty=0.6)


def test_evidence_moves_mass_out_of_ignorance():
    seq = [op.from_evidence(n, 0).uncertainty for n in (0, 1, 5, 50)]
    assert seq == sorted(seq, reverse=True)
    assert seq[0] == 1.0
    assert seq[-1] < 0.05


def test_expectation_reproduces_the_scalar():
    """The retrofit contract: nothing downstream had to change."""
    assert op.Opinion(0.0, 0.0, 1.0).expectation == 0.5
    assert op.Opinion(1.0, 0.0, 0.0).expectation == 1.0
    assert op.Opinion(0.0, 1.0, 0.0).expectation == 0.0


def test_distrusting_a_source_creates_IGNORANCE_not_disbelief():
    """Getting this backwards turns a low-reliability report into evidence
    against its own content."""
    confident = op.from_evidence(20, 0)
    halved = op.discount(0.5, confident)
    assert halved.belief < confident.belief
    assert halved.uncertainty > confident.uncertainty
    assert halved.disbelief <= confident.disbelief + 1e-9


def test_fusing_independent_opinions_reduces_ignorance():
    a = op.from_evidence(3, 0)
    fused = op.fuse(a, op.from_evidence(3, 0))
    assert fused.uncertainty < a.uncertainty
    assert fused.belief > a.belief


def test_fusing_opposed_opinions_produces_contest_not_confidence():
    fused = op.fuse(op.from_evidence(8, 0), op.from_evidence(0, 8))
    assert fused.contested
    assert 0.4 < fused.expectation < 0.6


def test_monotonicity_generalises_to_ignorance():
    """A conclusion cannot be less ignorant than its evidence. Inference
    redistributes mass; it does not create it."""
    parents = [op.from_evidence(2, 0), op.from_evidence(1, 0)]
    child = op.derive_opinion(parents, own=op.from_evidence(50, 0))
    assert child.belief <= min(p.belief for p in parents) + 1e-9
    assert child.uncertainty >= max(p.uncertainty for p in parents) - 1e-9


def test_opinion_works_on_an_existing_store_with_no_migration():
    with _mind() as m:
        n = m.observe("The north pump is failing intermittently.",
                      origin="document", source_ref="sitrep-2")
        o = m.opinion(n)
        for k in ("belief", "disbelief", "uncertainty", "expectation",
                  "contested", "vacuous", "verdict", "scalar_confidence"):
            assert k in o
        assert abs(o["belief"] + o["disbelief"] + o["uncertainty"] - 1) < 1e-3


def test_a_contradicted_claim_carries_disbelief_mass():
    with _mind() as m:
        a = m.observe("The north pump is working normally.",
                      origin="document", source_ref="sitrep-1")
        b = m.observe("The north pump is failing intermittently.",
                      origin="document", source_ref="sitrep-2", supersedes=a)
        assert m.opinion(a)["disbelief"] > 0, \
            "being superseded is the store's own record of having stopped " \
            "believing this"
        assert m.opinion(a)["disbelief"] > m.opinion(b)["disbelief"]


# ── B9 ───────────────────────────────────────────────────────────────────

def _corpus(n, dim=32, seed=3):
    rng = random.Random(seed)
    items = []
    for i in range(n):
        v = [rng.gauss(0, 1) for _ in range(dim)]
        nrm = math.sqrt(sum(x * x for x in v)) or 1.0
        items.append((f"n{i}", [x / nrm for x in v]))
    return items


def test_scanning_every_list_is_exact_by_construction():
    items = _corpus(300)
    idx = ann.IvfIndex(n_lists=10, nprobe=99).build(items)
    for q, _ in [(v, k) for k, v in items[:5]]:
        hits, exact = idx.search(q, top_k=10)
        assert exact
        assert [n for n, _ in hits] == [n for n, _ in
                                        ann.brute(q, items, top_k=10)]


def test_it_reports_when_it_did_NOT_scan_everything():
    """An approximate index that does not say it approximated is a wrong
    index."""
    items = _corpus(300)
    idx = ann.IvfIndex(n_lists=17, nprobe=2).build(items)
    _, exact = idx.search(items[0][1], top_k=10)
    assert exact is False


def test_recall_is_measured_rather_than_asserted():
    """The plan asks for 'identical results'; an approximate index cannot
    give that. Measuring the loss is the honest substitute."""
    items = _corpus(800)
    queries = [v for _, v in items[:25]]
    idx = ann.IvfIndex(nprobe=1).build(items)
    tight = ann.recall_at_k(idx, items, queries, k=10)
    loose = ann.recall_at_k(idx, items, queries, k=10, nprobe=99)
    assert 0.0 <= tight["recall_at_k"] <= 1.0
    assert loose["recall_at_k"] > tight["recall_at_k"]
    assert loose["recall_at_k"] > 0.99, "full probe must recover everything"


def test_recall_rises_monotonically_with_nprobe():
    """The failure mode has to be legible: if recall drops you raise nprobe,
    and the relationship must actually hold."""
    items = _corpus(600)
    queries = [v for _, v in items[:20]]
    idx = ann.IvfIndex().build(items)
    r = [ann.recall_at_k(idx, items, queries, k=10, nprobe=p)["recall_at_k"]
         for p in (1, 4, 16, 99)]
    assert r == sorted(r), r


def test_the_index_is_deterministic():
    """A store that reshuffles on rebuild makes retrieval differences
    impossible to attribute."""
    items = _corpus(200)
    a = ann.IvfIndex(seed=7).build(items)
    b = ann.IvfIndex(seed=7).build(items)
    assert a.lists == b.lists and a.centroids == b.centroids


def test_brute_force_stays_the_default():
    with _mind() as m:
        for i in range(20):
            m.observe(f"Field note {i} about depot fuel and vehicles.")
        assert not getattr(m._vec, "_ann", {})
        m.recall("depot fuel")
        assert m._vec.last_search_exact is True


def test_an_empty_index_degrades_to_brute_force():
    with _mind() as m:
        m.observe("A single note.")
        m._vec.build_ann(Space.READ)
        assert m.recall("single note").chunks


def test_building_the_index_reports_the_caveat():
    with _mind() as m:
        for i in range(40):
            m.observe(f"Note {i} about supplies and logistics.")
        rep = m._vec.build_ann(Space.READ)
        assert rep["vectors"] == 40
        assert "brute force remains exact" in rep["note"]
        assert m.recall("supplies logistics").chunks
        m._vec.drop_ann(Space.READ)
        assert not m._vec._ann
