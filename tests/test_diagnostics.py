"""F5 -- named checks, and proof each one can actually fail.

The negative control matters more than the positive one here. A check that
cannot go red is documentation pretending to be diagnosis, and this project
has already shipped one of those: an Identifier-precision metric that scored
1.000 with the guard removed.

So every check below is driven to FAIL or WARN deliberately.
"""
import os
import sqlite3
import tempfile

import pytest

from owl import Owl
from owl import diagnostics as dx


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


NOTES = [
    "The clinic generator runs on depot fuel.",
    "Dr Warsame runs the Bardera clinic and speaks Somali.",
    "Route Alpha floods above 40mm rainfall.",
]


def _mind(**kw):
    path = os.path.join(tempfile.mkdtemp(), "dx.owl")
    mind = Owl.open(path, **kw)
    for n in NOTES:
        mind.observe(n, origin="document", source_ref="sitrep")
    return mind


def _by_id(rep, cid):
    return next(c for c in rep.checks if c.id == cid)


def test_a_healthy_store_passes_everything():
    with _mind(embedder=Toy()) as mind:
        rep = dx.run(mind)
        assert not rep.failed, [c.id for c in rep.failed]
        assert len(rep.checks) >= 12, "F5 wants real coverage, not three checks"


def test_every_check_has_a_stable_id_and_failures_carry_a_remedy():
    """A diagnostic without a fix has moved the burden, not lifted it."""
    with _mind(embedder=Toy()) as mind:
        rep = dx.run(mind)
        ids = [c.id for c in rep.checks]
        assert len(ids) == len(set(ids)), "ids must be unique to be tracked"
        assert all("." in i for i in ids), "ids are namespaced"
        for c in rep.checks:
            if c.status in (dx.FAIL, dx.WARN):
                assert c.remedy, f"{c.id} reports a problem with no remedy"


def test_missing_append_only_trigger_is_caught():
    """The store still WORKS without it, which is exactly why it is checked."""
    with _mind(embedder=Toy()) as mind:
        mind._s.write(lambda c: [
            c.execute(f"DROP TRIGGER {r['name']}") for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND tbl_name='observation'")])
        rep = dx.run(mind)
        c = _by_id(rep, "substrate.append_only")
        assert c.status == dx.FAIL and "MISSING" in c.detail


def test_dangling_provenance_is_caught():
    with _mind(embedder=Toy()) as mind:
        obs = mind.observe("Two clinics reported fever cases.")
        mind.derive("An outbreak may be underway.", parents=[obs],
                    kind="hypothesis", producer="test",
                    falsifier="check intake curves")
        mind._s.write(lambda c: c.execute(
            "UPDATE derivation_edge SET parent_id='obs_ghost'"))
        rep = dx.run(mind)
        assert _by_id(rep, "substrate.provenance_intact").status == dx.FAIL


def test_an_unindexed_observation_is_caught():
    with _mind(embedder=Toy()) as mind:
        mind._s.write(lambda c: c.execute(
            "DELETE FROM mem_index WHERE node_id = "
            "(SELECT id FROM observation LIMIT 1)"))
        rep = dx.run(mind)
        c = _by_id(rep, "substrate.indexed")
        assert c.status == dx.FAIL and "reindex" in c.remedy


def test_model_drift_is_caught():
    """Vectors from two encoders are not comparable. Silent before F2."""
    with _mind(embedder=Toy()) as mind:
        mind._s.write(lambda c: c.execute(
            "UPDATE vector SET model='some-other-model' WHERE space='read'"))
        rep = dx.run(mind)
        c = _by_id(rep, "vectors.single_model")
        assert c.status == dx.FAIL and "not comparable" in c.detail


def test_missing_vectors_are_caught_but_queued_ones_are_not():
    """A queued memory is on its way; a missing one never arrives. Reporting
    them identically would send you chasing a problem that resolves itself."""
    with _mind(embedder=Toy()) as mind:
        mind._s.write(lambda c: c.execute("DELETE FROM vector"))
        assert _by_id(dx.run(mind), "vectors.coverage").status == dx.FAIL

    path = os.path.join(tempfile.mkdtemp(), "q.owl")
    with Owl.open(path, embedder=Toy(), defer_embedding=True) as m2:
        for n in NOTES:
            m2.observe(n)
        rep = dx.run(m2)
        assert _by_id(rep, "vectors.coverage").status == dx.PASS
        assert _by_id(rep, "queue.pending").status == dx.WARN


def test_an_abandoned_embedding_is_caught():
    class Broken(Toy):
        def embed(self, texts, space):
            raise RuntimeError("model unavailable")

    path = os.path.join(tempfile.mkdtemp(), "b.owl")
    with Owl.open(path, embedder=Broken(), defer_embedding=True) as mind:
        mind.observe("A note the model cannot handle.")
        for _ in range(3):
            mind.absorb()
        c = _by_id(dx.run(mind), "queue.abandoned")
        assert c.status == dx.FAIL and "lexically" in c.detail


def test_an_uncalibrated_embedder_warns_without_failing():
    """Uncalibrated is worse retrieval, not a broken store."""
    with _mind(embedder=Toy()) as mind:
        c = _by_id(dx.run(mind), "embedder.calibrated")
        assert c.status == dx.WARN and "--calibrate" in c.remedy


def test_a_stale_ceiling_is_caught():
    """The exact regression that came back twice: a sidecar written before
    ceiling existed silently supplies 1.0, and good matches read DONT_KNOW."""
    from owl.adapters.calibration import Calibration

    class Calibrated(Toy):
        def __init__(self, ceiling):
            self.calibration = Calibration(noise_floor=0.3, ceiling=ceiling,
                                           separability=0.99)

    with _mind(embedder=Calibrated(1.0)) as mind:
        c = _by_id(dx.run(mind), "embedder.ceiling_measured")
        assert c.status == dx.WARN and "DONT_KNOW" in c.detail
    with _mind(embedder=Calibrated(0.53)) as mind:
        assert _by_id(dx.run(mind),
                      "embedder.ceiling_measured").status == dx.PASS


def test_weak_separability_warns():
    from owl.adapters.calibration import Calibration

    class Weak(Toy):
        calibration = Calibration(noise_floor=0.3, ceiling=0.6,
                                  separability=0.88)

    with _mind(embedder=Weak()) as mind:
        c = _by_id(dx.run(mind), "embedder.separability")
        assert c.status == dx.WARN and "0.88" in c.detail


def test_no_embedder_warns_rather_than_failing():
    with _mind() as mind:
        c = _by_id(dx.run(mind), "embedder.present")
        assert c.status == dx.WARN and "lexical only" in c.detail


def test_readonly_and_liveness_are_reported():
    path = os.path.join(tempfile.mkdtemp(), "ro.owl")
    with Owl.open(path, embedder=Toy()) as m:
        for n in NOTES:
            m.observe(n)
    with Owl.open(path, embedder=Toy(), readonly=True) as ro:
        rep = dx.run(ro)
        # PASS, not WARN: read-only is a mode. Warning on an intentional
        # choice trains people to ignore warnings.
        assert _by_id(rep, "store.readonly").status == dx.PASS
        assert not rep.failed, "read-only is a mode, not a fault"


def test_a_crashing_check_does_not_take_the_diagnosis_down():
    """doctor() is for when things are broken. It has to survive that."""
    def boom(mind, rep):
        raise ValueError("check itself is buggy")

    original = dx.ALL
    dx.ALL = (dx.check_store, boom, dx.check_substrate)
    try:
        with _mind(embedder=Toy()) as mind:
            rep = dx.run(mind)
            assert any(c.id.startswith("boom") for c in rep.checks)
            assert any(c.id == "substrate.indexed" for c in rep.checks), \
                "checks after the crash must still run"
    finally:
        dx.ALL = original


def test_doctor_keeps_the_legacy_shape():
    with _mind(embedder=Toy()) as mind:
        rep = mind.doctor()
        for key in ("version", "tier", "path", "problems", "observations",
                    "derived", "vectors", "healthy", "self_audit"):
            assert key in rep, key
        assert isinstance(rep["report"], str)
        assert isinstance(rep["checks"], list)
