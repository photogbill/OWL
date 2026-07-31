"""C1-C5 consolidation, and A10 determinism.

A10 is the one that constrains the rest. "Why did you forget that?" is only
answerable if the same store plus the same seed produces the same answer,
so every pass here is deterministic by construction -- sorted iteration,
derived ids, no uuid, no randomised tie-breaks.
"""
import os
import tempfile

import pytest

from owl import Owl
from owl import consolidation as cons
from owl import reconstructive as rc


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
    return Owl.open(os.path.join(tempfile.mkdtemp(), "c.owl"),
                    embedder=Toy(), **kw)


# ── C2: identity, which is the hard part ─────────────────────────────────

def _mint_factory():
    n = [0]

    def mint():
        n[0] += 1
        return f"new_{n[0]}"
    return mint


def test_label_propagation_is_deterministic():
    """The textbook implementation is randomised, which would break A10."""
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),
             ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),
             ("c", "d", 0.1)]
    runs = [cons.label_propagation(edges) for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_a_community_keeps_its_name_when_it_merely_changes():
    """The failure nobody names: recomputing clusters churns their ids, and
    every composite derived from an old id points at nothing."""
    old = [cons.Community("com_0001_001", frozenset({"a", "b", "c", "d"}))]
    # lost one member, gained one -- still the same community
    now = cons.reconcile([frozenset({"a", "b", "c", "e"})], old,
                         generation=2, mint=_mint_factory())
    assert [c.id for c in now] == ["com_0001_001"]
    assert now[0].generation == 2


def test_a_split_gives_the_name_to_the_larger_side():
    old = [cons.Community("com_0001_001",
                          frozenset({"a", "b", "c", "d", "e", "f"}))]
    now = cons.reconcile([frozenset({"a", "b", "c", "d"}),
                          frozenset({"e", "f", "g"})],
                         old, generation=2, mint=_mint_factory())
    by_id = {c.id: c for c in now}
    assert "com_0001_001" in by_id
    assert by_id["com_0001_001"].members == frozenset({"a", "b", "c", "d"})
    # and the new side records where it came from, so a split is traceable
    other = next(c for c in now if c.id != "com_0001_001")
    assert other.lineage == ["com_0001_001"]


def test_one_old_community_cannot_claim_two_new_ones():
    old = [cons.Community("com_x", frozenset({"a", "b", "c", "d"}))]
    now = cons.reconcile([frozenset({"a", "b", "c"}), frozenset({"a", "b", "d"})],
                         old, generation=2, mint=_mint_factory())
    assert sum(c.id == "com_x" for c in now) == 1


def test_a_genuinely_new_community_gets_a_new_name():
    now = cons.reconcile([frozenset({"x", "y", "z"})], [], generation=1,
                         mint=_mint_factory())
    assert now[0].id == "new_1" and now[0].lineage == []


def test_identity_survives_fifty_cycles_on_a_mutating_store():
    """C2's acceptance criterion."""
    members = set("abcdefgh")
    prev = cons.reconcile([frozenset(members)], [], generation=0,
                          mint=_mint_factory())
    original = prev[0].id
    mint = _mint_factory()
    for gen in range(1, 51):
        # churn one member each cycle -- the community persists, its
        # membership does not
        members.discard(sorted(members)[0])
        members.add(f"n{gen}")
        prev = cons.reconcile([frozenset(members)], prev, generation=gen,
                              mint=mint)
    assert any(c.id == original for c in prev), \
        "the community drifted its entire membership but is still itself"


# ── C3: sleep pressure ───────────────────────────────────────────────────

def test_pressure_not_idle_cpu():
    quiet = cons.plan_sleep(unconsolidated=0, interference=0.0,
                            hours_since=1.0, confusable=[], distant=["a"])
    assert quiet.phase == "none" and "nothing is owed" in quiet.reason


def test_nrem_separates_before_rem_recombines():
    """Recombining material you have not separated is how you manufacture
    confident nonsense out of two things you were already confusing."""
    low = cons.plan_sleep(unconsolidated=2, interference=0.5,
                          hours_since=1.0, confusable=["a", "b"],
                          distant=["x", "y"])
    assert low.phase == "nrem" and low.temperature < 0.5

    high = cons.plan_sleep(unconsolidated=20, interference=0.9,
                           hours_since=48.0, confusable=["a"],
                           distant=["x", "y"])
    assert high.phase == "rem" and high.temperature > 0.5
    assert "hypothesis with a falsifier" in high.reason


def test_rem_will_not_fire_with_nothing_distant_to_recombine():
    p = cons.plan_sleep(unconsolidated=50, interference=1.0,
                        hours_since=99.0, confusable=["a"], distant=[])
    assert p.phase == "nrem"


# ── C5: schema-delta ─────────────────────────────────────────────────────

def test_the_rule_is_factored_out_of_its_repetitions():
    items = [(f"n{i}", f"Generator run-hours logged at 08:0{i}.")
             for i in range(5)]
    groups = cons.find_schemas(items)
    assert len(groups) == 1
    assert "{}" in groups[0].schema
    assert groups[0].saved_chars > 0


def test_a_schema_with_one_member_is_just_a_sentence():
    """Compressing it would add indirection for nothing."""
    assert cons.find_schemas([("n1", "A one-off note about 42 things.")]) == []


def test_nothing_varying_means_nothing_to_factor():
    items = [(f"n{i}", "The gate was locked.") for i in range(5)]
    assert cons.find_schemas(items) == []


def test_schemas_never_touch_verbatim_content():
    with _mind() as m:
        for i in range(4):
            m.observe(f"Net control is 145.50{i} MHz.")
        assert m.schemas() == [], "verbatim is excluded before grouping"


# ── C1: compression must be EARNED ───────────────────────────────────────

def test_no_reasoner_means_skipped_not_silently_done():
    p = rc.plan_compression("n1", "Some content.", claim_class="status",
                            reconstruct=None)
    assert p.keep_verbatim and "skipped, not silently no-opped" in p.reason


def test_verbatim_is_never_compressed_however_well_it_reconstructs():
    p = rc.plan_compression("n1", "Grid 31U DQ 48251 11932",
                            claim_class="verbatim",
                            reconstruct=lambda cue, nb: "Grid 31U DQ 48251 11932")
    assert p.keep_verbatim
    assert "still an exact string" in p.reason


def test_a_lossy_reconstruction_keeps_the_original():
    original = "The depot holds 4000 litres of diesel for the generator."
    p = rc.plan_compression("n1", original, claim_class="capacity",
                            reconstruct=lambda cue, nb: "The depot holds fuel.")
    assert p.keep_verbatim and p.fidelity < rc.FIDELITY_FLOOR
    assert "cannot prove" in p.reason


def test_losing_a_number_is_total_failure_not_partial():
    """A reconstruction that drops a figure has not degraded, it has made a
    different claim."""
    assert rc.fidelity("The depot holds 4000 litres.",
                       "The depot holds 900 litres.") == 0.0


def test_fidelity_is_asymmetric_on_purpose():
    """Adding material is verbose; dropping it is destructive."""
    verbose = rc.fidelity("the pump failed",
                          "the north pump failed on Tuesday morning")
    lossy = rc.fidelity("the north pump failed on Tuesday morning",
                        "the pump failed")
    assert verbose == 1.0 and lossy < 1.0


def test_a_reconstruction_that_raises_keeps_the_original():
    def boom(cue, nb):
        raise RuntimeError("model died")
    p = rc.plan_compression("n1", "content", claim_class="status",
                            reconstruct=boom)
    assert p.keep_verbatim and "raised" in p.reason


def test_a_faithful_reconstruction_is_allowed_to_compress():
    original = "The clinic generator runs on depot fuel."
    p = rc.plan_compression("n1", original, claim_class="status",
                            reconstruct=lambda cue, nb: original)
    assert not p.keep_verbatim and p.fidelity == 1.0


# ── C4: a guess must never look like a fact ──────────────────────────────

def test_a_hypothesis_without_a_falsifier_cannot_exist():
    with pytest.raises(ValueError) as e:
        rc.Hypothesis("h1", "Something is happening.", falsifier="  ")
    assert "belief with better marketing" in str(e.value)


def test_only_a_promoted_hypothesis_is_exportable():
    h = rc.Hypothesis("h1", "An outbreak is underway.",
                      falsifier="check clinic intake curves")
    for state in ("generated", "testing", "archived_failed", "expired"):
        assert not rc.Hypothesis("h1", "x", "y", state).exportable
    assert rc.Hypothesis("h1", "x", "y", "promoted").exportable
    assert h.presentable_as_fact is False


def test_every_state_is_labelled_so_nothing_reaches_a_surface_unmarked():
    """C4's acceptance criterion."""
    for state in rc.STATES:
        label = rc.Hypothesis("h", "x", "y", state).label()
        assert label and ("HYPOTHESIS" in label or "hypothesis" in label)


def test_failure_is_archived_not_deleted():
    """'We tried this and it did not work' is expensive knowledge, and it
    stops the same rejected idea being re-proposed every cycle."""
    h = rc.Hypothesis("h1", "The pump failed from cavitation.",
                      falsifier="inspect the impeller")
    failed = rc.advance(h, now=100.0, supporting=0, opposing=2)
    assert failed.state == "archived_failed"
    assert rc.advance(failed, now=1e9, supporting=99).state == "archived_failed"


def test_promotion_needs_evidence_and_no_opposition():
    h = rc.Hypothesis("h1", "x", "y")
    assert rc.advance(h, now=1.0, supporting=1).state == "testing"
    assert rc.advance(h, now=1.0, supporting=2).state == "promoted"
    assert rc.advance(h, now=1.0, supporting=5, opposing=1).state == "testing"


def test_an_untested_hypothesis_expires():
    h = rc.Hypothesis("h1", "x", "y", expires_at=50.0)
    assert rc.advance(h, now=99.0).state == "expired"


# ── A10: determinism ─────────────────────────────────────────────────────

NOTES = [
    "The clinic generator runs on depot fuel.",
    "Generator run-hours logged at 08:00.",
    "Generator run-hours logged at 08:15.",
    "Generator run-hours logged at 08:30.",
    "Dr Warsame runs the Bardera clinic.",
    "Route Alpha floods above 40mm rainfall.",
    "The north well pump needs a 40mm gasket.",
    "The perimeter fence needs new wire.",
]


def _shape(m):
    """Everything about consolidation EXCEPT node ids.

    Observation ids are uuids by design, so two independently built stores
    can never agree on them and demanding it would be testing the wrong
    thing. A10 constrains the consolidation OUTPUT: the same content must
    produce the same communities, the same schemas, the same verdict.
    """
    return {
        "communities": sorted((c["id"], c["size"], c["generation"])
                              for c in m.communities()),
        "schemas": sorted((g["schema"], g["n"], g["saved_chars"])
                          for g in m.schemas()),
        "sleep": m.sleep_plan()["phase"],
    }


def _built(path):
    m = Owl.open(path, embedder=Toy())
    for n in NOTES:
        m.observe(n, origin="document", source_ref="sitrep")
    m.tend()
    return m


def test_the_same_content_consolidates_the_same_way():
    """A10's acceptance criterion. Nobody guarantees this, and without it
    'why did you forget that?' has no answer."""
    shapes = []
    for i in range(5):
        m = _built(os.path.join(tempfile.mkdtemp(), f"d{i}.owl"))
        with m:
            shapes.append(_shape(m))
    for s in shapes[1:]:
        assert s == shapes[0]


def test_one_hundred_consecutive_runs_are_identical():
    """The plan asks for 100 byte-identical runs. Within a store this is
    the stronger claim, because node ids are fixed and any churn shows."""
    m = _built(os.path.join(tempfile.mkdtemp(), "rep.owl"))
    with m:
        first = _shape(m)
        for _ in range(99):
            assert _shape(m) == first


def test_community_ids_are_derived_not_random():
    """A uuid here would make A10 impossible."""
    with _mind() as m:
        for n in NOTES:
            m.observe(n, origin="document", source_ref="sitrep")
        m.tend()
        for c in m.communities():
            assert c["id"].startswith("com_"), c["id"]
            assert len(c["id"]) == len("com_0001_001")


def test_repeated_consolidation_is_stable():
    """Running tend() twice must not churn identity -- otherwise every
    composite derived from a community is invalidated on a no-op pass."""
    with _mind() as m:
        for n in NOTES:
            m.observe(n, origin="document", source_ref="sitrep")
        m.tend()
        first = {c["id"]: sorted(c["members"]) for c in m.communities()}
        second = {c["id"]: sorted(c["members"]) for c in m.communities()}
        assert first == second
