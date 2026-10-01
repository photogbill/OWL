"""A closed store is the whole store, and a sealed one leaves nothing behind.

Found 2026-10-01 while running the suite: `test_the_store_is_unreadable_
without_the_key` passed on its own and failed inside the full run. The cause
was not the test. `SqliteStore.close()` stopped the writer but never closed
the thread-local READER connections, and in WAL mode SQLite folds the
write-ahead log back into the main file only when the last connection
closes. A reader lives as long as the Owl object is referenced -- which,
after `with Owl.open(...) as m:`, is the rest of the function -- so every
write of the session stayed in `<store>-wal` and out of the main file.

Two consequences, both silent:

  * `Owl.sealed()` re-sealed the main file WITHOUT the session's writes, so
    the next session opened a store that had forgotten them;
  * it shredded the main file and left the plaintext `-wal` beside the
    ciphertext -- the exact leak A11 exists to prevent.

Whether a test saw it depended on when the garbage collector ran, which is
why it hid. These tests hold a reference on purpose, so the collector cannot
rescue them.
"""
import os
import tempfile

import pytest

from owl import Owl
from owl import crypto

needs_crypto = pytest.mark.skipif(
    not crypto.available(),
    reason="cryptography not installed (owl-engine[crypto])")

NOTE = "Dr Warsame runs the Bardera clinic."


def _used(mind):
    """Write, then read on this thread -- the read opens the reader that
    used to outlive close()."""
    mind.observe(NOTE, origin="document", source_ref="sitrep")
    mind.recall("Warsame clinic")
    return mind


def test_close_leaves_the_whole_store_in_the_main_file():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.owl")
    mind = _used(Owl.open(p))
    mind.close()                      # `mind` stays referenced on purpose
    assert b"Warsame" in open(p, "rb").read(), \
        "the session's writes are not in the main file after close()"
    wal = p + "-wal"
    assert not os.path.exists(wal) or os.path.getsize(wal) == 0, \
        "a non-empty -wal outlived close()"
    del mind


def test_reads_after_close_are_refused_not_silently_reopened():
    """A read after close used to open a fresh, untracked connection -- the
    same leak again, one call later."""
    d = tempfile.mkdtemp()
    mind = _used(Owl.open(os.path.join(d, "m.owl")))
    mind.close()
    with pytest.raises(RuntimeError):
        mind._s.query("SELECT 1")


def test_close_is_idempotent_and_a_store_reopens_whole():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.owl")
    mind = _used(Owl.open(p))
    mind.close()
    mind.close()
    with Owl.open(p) as again:
        assert again.recall("Warsame clinic").chunks, "reopened store lost it"


def test_shred_takes_the_sidecars_with_it():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.owl")
    for suffix in ("", "-wal", "-shm", "-journal"):
        with open(p + suffix, "wb") as f:
            f.write(b"plaintext Warsame")
    crypto.shred(p)
    assert sorted(os.listdir(d)) == [], f"left behind: {os.listdir(d)}"


@needs_crypto
def test_a_sealed_session_keeps_its_writes_and_leaves_no_plaintext():
    d = tempfile.mkdtemp()
    key = crypto.generate_key(os.path.join(d, "m.key"))
    sealed = os.path.join(d, "m.owl.sealed")
    holder = []
    with Owl.sealed(sealed, key) as mind:
        _used(mind)
        holder.append(mind)          # outlives the block, like a caller's var
    leftovers = sorted(f for f in os.listdir(d)
                       if f not in ("m.key", "m.owl.sealed"))
    assert leftovers == [], f"plaintext left beside the sealed store: {leftovers}"
    assert b"Warsame" not in open(sealed, "rb").read()
    with Owl.sealed(sealed, key) as mind:
        assert mind.recall("Warsame clinic").chunks, \
            "the sealed store forgot the previous session's writes"


@needs_crypto
def test_sealing_a_store_with_a_live_log_seals_everything():
    """crypto.seal() on a store that is still open folds the log in first,
    so the ciphertext is never missing the newest writes."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.owl")
    key = crypto.load_key(crypto.generate_key(os.path.join(d, "m.key")))
    mind = _used(Owl.open(p))
    try:
        out = crypto.seal(p, os.path.join(d, "m.owl.sealed"), key)
        back = os.path.join(d, "back.owl")
        crypto.unseal(out, back, key)
        assert b"Warsame" in open(back, "rb").read(), \
            "the sealed copy is missing writes that were still in the -wal"
    finally:
        mind.close()
