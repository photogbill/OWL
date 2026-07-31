"""F2 -- the store is readable when the engine is down.

Background machinery is never a gatekeeper on recall. The failure this
prevents is a memory system that cannot be remembered from because a
component it does not strictly need is unavailable.

The non-obvious part: `recall()` WRITES. Retrieval reinforces what it
returns and demotes the competitors -- retrieval-induced forgetting is a
real effect and OWL models it. That makes recall a memory event rather than
a passive read, and it meant recall could not run against a store it could
not write to. The fix is not to drop the reinforcement silently; it is to
skip it and SAY so.
"""
import os
import shutil
import sqlite3
import tempfile

import pytest

from owl import Owl, State
from owl.protocols import ReadOnlyError


class Toy:
    is_semantic = True
    name = "toy"

    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, texts, space):
        if self.fail:
            raise RuntimeError("model unavailable")
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
    "The generator serial is GX-4419.",
]


def _populated(**kw):
    path = os.path.join(tempfile.mkdtemp(), "f2.owl")
    mind = Owl.open(path, embedder=Toy(), **kw)
    with mind:
        for n in NOTES:
            mind.observe(n, origin="document", source_ref="sitrep")
    return path


def test_recall_works_on_a_readonly_store():
    path = _populated()
    mind = Owl.open(path, embedder=Toy(), readonly=True)
    with mind:
        r = mind.recall("who runs the clinic")
        assert r.chunks, "a read-only store must still answer"
        assert r.state is not State.DONT_KNOW


def test_recall_works_on_readonly_MEDIA():
    """Not just the flag -- the actual filesystem permission, which is the
    real scenario: an archived store, a mounted image, a shared copy."""
    path = _populated()
    ro_dir = tempfile.mkdtemp()
    ro = os.path.join(ro_dir, "archive.owl")
    shutil.copy(path, ro)
    os.chmod(ro, 0o444)
    os.chmod(ro_dir, 0o555)
    try:
        with pytest.raises(sqlite3.OperationalError):
            Owl.open(ro)                    # normal open needs to write
        mind = Owl.open(ro, embedder=Toy(), readonly=True)
        with mind:
            assert mind.recall("depot fuel").chunks
    finally:
        os.chmod(ro_dir, 0o755)
        os.chmod(ro, 0o644)


def test_writes_fail_loudly_rather_than_silently():
    """A store that silently drops writes is a store that forgets."""
    mind = Owl.open(_populated(), embedder=Toy(), readonly=True)
    with mind:
        with pytest.raises(ReadOnlyError):
            mind.observe("This must not vanish quietly.")


def test_readonly_recall_says_it_did_not_reinforce():
    mind = Owl.open(_populated(), embedder=Toy(), readonly=True)
    with mind:
        r = mind.recall("who runs the clinic")
        assert any("reinforcement" in d for d in r.degraded)
        assert r.full_strength is False
        assert "read-only" in r.reason


def test_reinforcement_really_is_skipped_not_just_reported():
    path = _populated()
    before = Owl.open(path, embedder=Toy(), readonly=True)
    with before:
        for _ in range(3):
            before.recall("who runs the clinic")
    check = Owl.open(path, embedder=Toy(), readonly=True)
    with check:
        rows = check._s.query("SELECT SUM(review_count) n FROM mem_index")
        assert (rows[0]["n"] or 0) == 0, "read-only must not mutate the index"


def test_a_missing_embedder_is_reported_not_hidden():
    """Lexical-only is a correct answer produced by weaker machinery. A
    system that returns the same shaped result either way is
    indistinguishable from one that is working."""
    mind = Owl.open(_populated(), embedder=None)
    with mind:
        r = mind.recall("generator")
        assert r.chunks
        assert any("no embedder" in d for d in r.degraded)
        assert r.full_strength is False


def test_a_broken_embedder_degrades_rather_than_failing():
    mind = Owl.open(_populated(), embedder=Toy(fail=True))
    with mind:
        r = mind.recall("generator")
        assert r.chunks, "lexical must survive a dead model"
        assert any("embedder raised" in d for d in r.degraded)


def test_full_strength_is_true_when_nothing_is_missing():
    mind = Owl.open(_populated(), embedder=Toy())
    with mind:
        r = mind.recall("who runs the clinic")
        assert r.degraded == () and r.pending == 0
        assert r.full_strength is True


def test_readonly_store_is_readable_while_another_process_writes():
    """WAL gives a consistent snapshot to readers. A busy store is still a
    readable one -- this is the 'engine is up but occupied' case."""
    path = _populated()
    writer = Owl.open(path, embedder=Toy())
    reader = Owl.open(path, embedder=Toy(), readonly=True)
    try:
        assert reader.recall("depot fuel").chunks
        writer.observe("A new note written while the reader is open.")
        assert reader.recall("depot fuel").chunks
    finally:
        writer.close()
        reader.close()


def test_readonly_refuses_a_file_that_is_not_a_store():
    """Read-only cannot create the schema, so it must say what is wrong
    rather than producing an empty mind that answers DONT_KNOW to
    everything."""
    junk = os.path.join(tempfile.mkdtemp(), "notes.txt")
    with open(junk, "w") as f:
        f.write("this is not a database")
    with pytest.raises(Exception) as e:
        Owl.open(junk, readonly=True)
    assert "not an OWL store" in str(e.value) or "file is not a database" in str(
        e.value)


def test_doctor_runs_readonly():
    """Diagnosis is exactly what you want when something is wrong, so it
    must not be the thing that needs the store to be healthy."""
    mind = Owl.open(_populated(), embedder=Toy(), readonly=True)
    with mind:
        rep = mind.doctor()
        assert isinstance(rep, dict)
