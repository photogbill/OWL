"""H1-H4, the speculative phase.

Explicitly research. The discipline that makes research honest rather than
decorative is that each of these can come out negative: the VSA reports its
own crosstalk, the maturity contract can say "too new to trust", the nightly
budget can refuse, and the unification report can conclude that the
unification does not hold.
"""
import os
import tempfile

import pytest

from owl import Owl
from owl import hyperdimensional as hd
from owl import maturity as mat


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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "h.owl"),
                    embedder=Toy(), **kw)


# ── H1: structural recall ────────────────────────────────────────────────

def test_binding_is_dissimilar_to_its_inputs_and_self_inverse():
    """The two properties the whole scheme rests on."""
    a, b = hd._symbol("ahmed"), hd._symbol("ROLE::AGENT")
    bound = hd.bind(a, b)
    assert abs(hd.cosine(bound, a)) < 0.05, "a binding is not a blend"
    assert abs(hd.cosine(bound, b)) < 0.05
    assert hd.cosine(hd.bind(bound, b), a) > 0.99, "unbinding must recover a"


def test_bundling_is_similar_to_everything_in_it():
    parts = [hd._symbol(f"item{i}") for i in range(4)]
    s = hd.bundle(*parts)
    for p in parts:
        assert hd.cosine(s, p) > 0.2
    assert abs(hd.cosine(s, hd._symbol("unrelated"))) < 0.1


def test_symbols_are_deterministic_across_processes():
    """Derived from a hash, not stored -- so an index rebuilds from content
    alone, and A10 still holds."""
    assert hd._symbol("ahmed") == hd._symbol("ahmed")
    assert hd._symbol("ahmed") != hd._symbol("warsame")


def test_it_answers_who_did_what_to_whom():
    """The thing embeddings cannot do. This is H1's whole point."""
    idx = hd.StructuralIndex()
    idx.encode("n1", AGENT="ahmed", ACTION="delivered", OBJECT="gasket",
               RECIPIENT="warsame")
    idx.encode("n2", AGENT="warsame", ACTION="delivered", OBJECT="fuel",
               RECIPIENT="ahmed")

    who = idx.query("AGENT", ACTION="delivered", OBJECT="gasket")
    assert who and who[0]["answer"] == "ahmed"

    to_whom = idx.query("RECIPIENT", ACTION="delivered", OBJECT="gasket")
    assert to_whom and to_whom[0]["answer"] == "warsame"


def test_direction_is_preserved_where_an_embedding_loses_it():
    """'Ahmed delivered to Warsame' and the reverse are near-identical as
    bags of words. Structurally they are opposites."""
    idx = hd.StructuralIndex()
    idx.encode("forward", AGENT="ahmed", ACTION="paid", RECIPIENT="warsame")
    idx.encode("reverse", AGENT="warsame", ACTION="paid", RECIPIENT="ahmed")

    a = idx.query("AGENT", ACTION="paid", RECIPIENT="warsame")
    assert a and a[0]["answer"] == "ahmed" and a[0]["node_id"] == "forward"
    b = idx.query("AGENT", ACTION="paid", RECIPIENT="ahmed")
    assert b and b[0]["answer"] == "warsame" and b[0]["node_id"] == "reverse"


def test_a_low_dimension_is_refused_rather_than_returning_nonsense():
    """Below ~1000 dims the crosstalk swamps the signal. Not a tuning knob."""
    with pytest.raises(ValueError) as e:
        hd.StructuralIndex(dim=128)
    assert "not a tuning knob" in str(e.value)


def test_an_empty_trace_is_refused():
    with pytest.raises(ValueError):
        hd.StructuralIndex().encode("n1")


def test_crosstalk_is_measured_not_assumed():
    """A VSA silently over-capacity returns plausible WRONG fillers, so the
    limit is reported -- same reason the ANN index reports its recall."""
    idx = hd.StructuralIndex()
    for i in range(12):
        idx.encode(f"n{i}", AGENT=f"person{i}", ACTION=f"act{i}",
                   OBJECT=f"thing{i}")
    ct = idx.crosstalk()
    assert ct["pairs"] == 66
    assert ct["healthy"] is True
    assert ct["max"] < 8 * ct["expected"]


def test_shared_structure_is_not_counted_as_noise():
    """Twenty reports all saying ACTION='delivered' SHOULD be similar --
    that is the representation working, not the index degrading."""
    idx = hd.StructuralIndex()
    for i in range(12):
        idx.encode(f"n{i}", AGENT=f"person{i}", ACTION="delivered",
                   OBJECT=f"thing{i}")
    ct = idx.crosstalk()
    assert ct["shared_structure_pairs"] == 66
    assert ct["pairs"] == 0
    assert ct["healthy"] is True
    assert "no independent pairs" in ct["note"]


def test_a_query_with_no_match_returns_nothing():
    idx = hd.StructuralIndex()
    idx.encode("n1", AGENT="ahmed", ACTION="delivered", OBJECT="gasket")
    assert idx.query("AGENT", ACTION="delivered", OBJECT="helicopter") == []


def test_the_facade_exposes_it():
    with _mind() as m:
        idx = m.structural()
        idx.encode("n1", AGENT="ahmed", ACTION="delivered", OBJECT="gasket")
        assert m.structural() is idx, "one index per store"
        assert idx.query("AGENT", OBJECT="gasket")[0]["answer"] == "ahmed"


# ── H2: cold-start honesty ───────────────────────────────────────────────

def test_a_new_store_says_it_is_new():
    m = mat.assess_maturity(days=3, memories=12, sources=2)
    assert m.young
    assert "treat every gap as ignorance" in m.contract()


def test_a_mature_store_says_a_gap_is_real_evidence():
    m = mat.assess_maturity(days=400, memories=5000, sources=60)
    assert not m.young
    assert "real evidence of absence" in m.contract()


def test_volume_cannot_paper_over_a_one_day_history():
    """A store with 10,000 memories from one source over one day is not
    mature, and a mean would let volume hide that."""
    lopsided = mat.assess_maturity(days=1, memories=10_000, sources=1)
    balanced = mat.assess_maturity(days=30, memories=300, sources=10)
    assert lopsided.coverage < balanced.coverage
    assert lopsided.young


def test_an_empty_store_has_zero_coverage():
    assert mat.assess_maturity(days=0, memories=0, sources=0).coverage == 0.0


def test_maturity_reaches_the_facade():
    with _mind() as m:
        for i in range(5):
            m.observe(f"Note {i}.", origin="document", source_ref=f"s{i}")
        rep = m.maturity()
        assert rep["memories"] == 5 and rep["sources"] == 5
        assert rep["young"] is True
        assert "ignorance" in rep["contract"] or "thin" in rep["contract"]


# ── H3: the nightly budget ───────────────────────────────────────────────

def test_a_quota_that_bends_is_not_a_quota():
    b = mat.NightlyBudget(calls_per_night=1)
    ripe = mat.assess_maturity(days=90, memories=800, sources=20)
    ok, _ = b.should_run(sleep_pressure=99.0, maturity=ripe)
    assert ok
    b.spend()
    ok, why = b.should_run(sleep_pressure=99.0, maturity=ripe)
    assert not ok and "not a quota" in why


def test_it_refuses_when_nothing_is_owed():
    b = mat.NightlyBudget()
    ripe = mat.assess_maturity(days=90, memories=800, sources=20)
    ok, why = b.should_run(sleep_pressure=0.5, maturity=ripe)
    assert not ok and "scheduling by clock" in why


def test_it_refuses_to_generalise_from_a_new_store():
    """A model asked to find patterns in eleven memories will find some,
    and they will be noise wearing the shape of insight."""
    b = mat.NightlyBudget()
    new = mat.assess_maturity(days=2, memories=11, sources=1)
    ok, why = b.should_run(sleep_pressure=99.0, maturity=new)
    assert not ok and "noise" in why


def test_the_budget_resets():
    b = mat.NightlyBudget()
    b.spend()
    assert b.exhausted
    b.reset()
    assert not b.exhausted


# ── H4: does the unification actually hold? ──────────────────────────────

def _samples(coupled, decoupled):
    """Subsystems that track prediction error, and one that does not."""
    out = []
    for i in range(20):
        pe = i / 19.0
        row = {"prediction_error": pe}
        for k in coupled:
            row[k] = pe * 0.9 + 0.05
        for k in decoupled:
            row[k] = (i % 3) / 2.0          # unrelated to pe
        out.append(row)
    return out


def test_it_can_conclude_that_the_unification_HOLDS():
    rep = mat.unification_report(_samples(
        ["encoding_priority", "attention", "consolidation"], []))
    assert rep["not_unified"] == []
    assert "holds across all of them" in rep["verdict"]


def test_it_can_conclude_that_the_unification_DOES_NOT_hold():
    """The important direction. 'One objective behind everything' is an
    elegant claim that is very easy to assert and never check."""
    rep = mat.unification_report(_samples(
        ["encoding_priority", "attention"], ["forgetting"]))
    assert "forgetting" in rep["not_unified"]
    assert "does not hold" in rep["verdict"]
    assert "honest outcome, not a failure to simplify" in rep["verdict"]
    assert "NOT unified" in rep["subsystems"]["forgetting"]["verdict"]


def test_a_partly_coupled_subsystem_is_named_as_such():
    samples = []
    for i in range(20):
        pe = i / 19.0
        samples.append({"prediction_error": pe,
                        "consolidation": pe * 0.5 + (i % 4) * 0.12})
    r = mat.unification_report(samples)["subsystems"]["consolidation"]
    assert 0.6 <= r["correlation"] < 0.85
    assert "residual is doing real work" in r["verdict"]


def test_small_samples_get_no_verdict():
    assert "insufficient" in mat.unification_report(
        [{"prediction_error": 0.5, "attention": 0.5}])["verdict"]


def test_prediction_error_is_bounded():
    assert mat.prediction_error(observed_novelty=0.9, expected_novelty=0.1) == 0.8
    assert mat.prediction_error(observed_novelty=0.1, expected_novelty=0.9) == 0.8
    assert mat.prediction_error(observed_novelty=5.0, expected_novelty=0.0) == 1.0
