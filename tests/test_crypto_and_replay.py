"""A11 encryption at rest + D5 time-travel replay.

Both refuse a comfortable falsehood. A11 will not call field-level content
encryption "encrypted at rest" while the inverted index sits beside it in
plaintext; D5 will not call agreement "verified" without checking the replay
against what was actually recorded at the time.
"""
import os
import stat
import tempfile

import pytest

from owl import Owl
from owl import crypto

needs_crypto = pytest.mark.skipif(
    not crypto.available(),
    reason="cryptography not installed (owl-engine[crypto])")


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
    "Route Alpha is open as of this morning.",
    "The clinic generator runs on depot fuel.",
    "Dr Warsame runs the Bardera clinic.",
]


# ── A11: keys ────────────────────────────────────────────────────────────

def test_a_generated_key_is_owner_only():
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    assert len(k.read_bytes()) == crypto.KEY_BYTES
    if os.name != "nt":
        assert stat.S_IMODE(k.stat().st_mode) == 0o600
    assert crypto.key_permissions(k) == []


def test_it_refuses_to_overwrite_a_key():
    """Overwriting makes every store it protects permanently unreadable."""
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    with pytest.raises(crypto.SealError) as e:
        crypto.generate_key(k)
    assert "permanently unreadable" in str(e.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_a_readable_key_is_refused():
    """An encrypted store beside a world-readable key is an unencrypted
    store with extra steps."""
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    os.chmod(k, 0o644)
    assert crypto.key_permissions(k)
    with pytest.raises(crypto.SealError):
        crypto.load_key(k)
    assert crypto.load_key(k, strict=False)          # explicit opt-out only


def test_a_wrong_sized_key_is_rejected():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "short.key")
    open(p, "wb").write(b"too short")
    with pytest.raises(crypto.SealError):
        crypto.load_key(p, strict=False)


# ── A11: sealing ─────────────────────────────────────────────────────────

@needs_crypto
def test_the_store_is_unreadable_without_the_key():
    """A11's acceptance criterion, checked against the CONTENT rather than
    the file header -- the failure that matters is the vocabulary leaking,
    not the magic bytes."""
    d = tempfile.mkdtemp()
    plain = os.path.join(d, "m.owl")
    with Owl.open(plain, embedder=Toy()) as m:
        for n in NOTES:
            m.observe(n, origin="document", source_ref="sitrep")

    assert b"Warsame" in open(plain, "rb").read(), "plaintext store leaks"

    k = crypto.generate_key(os.path.join(d, "m.key"))
    sealed = os.path.join(d, "m.owl.sealed")
    Owl.seal_store(plain, sealed, k)
    raw = open(sealed, "rb").read()
    assert b"Warsame" not in raw
    assert b"Bardera" not in raw
    assert b"generator" not in raw
    assert raw.startswith(crypto.MAGIC)


@needs_crypto
def test_a_round_trip_preserves_everything():
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    sealed = os.path.join(d, "m.owl.sealed")

    with Owl.sealed(sealed, k, embedder=Toy()) as m:
        for n in NOTES:
            m.observe(n, origin="document", source_ref="sitrep")
    assert os.path.exists(sealed)
    assert not os.path.exists(sealed + ".open"), "working copy must not linger"

    with Owl.sealed(sealed, k, embedder=Toy()) as m:
        r = m.recall("who runs the clinic")
        assert any("Warsame" in c.content for c in r.chunks)


@needs_crypto
def test_the_wrong_key_does_not_open_it():
    d = tempfile.mkdtemp()
    good = crypto.generate_key(os.path.join(d, "a.key"))
    bad = crypto.generate_key(os.path.join(d, "b.key"))
    sealed = os.path.join(d, "m.owl.sealed")
    with Owl.sealed(sealed, good, embedder=Toy()) as m:
        m.observe("A secret note.")
    with pytest.raises(crypto.SealError):
        with Owl.sealed(sealed, bad, embedder=Toy()):
            pass


@needs_crypto
def test_tampering_is_detected_rather_than_silently_decrypted():
    """GCM cannot tell you whether it was the wrong key or a modified file,
    and that is the point -- a tampered store must not open at all."""
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    sealed = os.path.join(d, "m.owl.sealed")
    with Owl.sealed(sealed, k, embedder=Toy()) as m:
        m.observe("A note whose bytes will be flipped.")

    raw = bytearray(open(sealed, "rb").read())
    raw[-1] ^= 0x01
    open(sealed, "wb").write(bytes(raw))

    with pytest.raises(crypto.SealError) as e:
        with Owl.sealed(sealed, k, embedder=Toy()):
            pass
    assert "modified" in str(e.value)


@needs_crypto
def test_a_plain_file_is_not_mistaken_for_a_sealed_one():
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    plain = os.path.join(d, "m.owl")
    with Owl.open(plain, embedder=Toy()) as m:
        m.observe("Just a normal store.")
    with pytest.raises(crypto.SealError) as e:
        crypto.unseal(plain, os.path.join(d, "x"), crypto.load_key(k))
    assert "open it normally" in str(e.value)


def test_missing_crypto_never_silently_downgrades():
    """The worst failure available in a crypto path."""
    if crypto.available():
        pytest.skip("cryptography is installed")
    with pytest.raises(crypto.CryptoUnavailable) as e:
        crypto._aesgcm(b"\x00" * 32)
    assert "will not fall back" in str(e.value)


def test_doctor_checks_key_permissions():
    d = tempfile.mkdtemp()
    k = crypto.generate_key(os.path.join(d, "m.key"))
    from owl import diagnostics as dx
    with Owl.open(os.path.join(d, "m.owl"), embedder=Toy()) as m:
        m.observe("A note.")
        assert not any(c.id == "crypto.key_permissions"
                       for c in dx.run(m).checks), "no keyfile, no check"
        m.keyfile = str(k)
        c = next(c for c in dx.run(m).checks
                 if c.id == "crypto.key_permissions")
        assert c.status == dx.PASS
        if os.name != "nt":
            os.chmod(k, 0o644)
            c = next(c for c in dx.run(m).checks
                     if c.id == "crypto.key_permissions")
            assert c.status == dx.FAIL and "chmod 600" in c.remedy


# ── D5: replay ───────────────────────────────────────────────────────────

def _recorded(tmp):
    from conftest import FakeClock
    clock = FakeClock()
    m = Owl.open(os.path.join(tmp, "r.owl"), embedder=Toy(), clock=clock)
    m.receipts = True
    a = m.observe("Route Alpha is open as of this morning.",
                  origin="document", source_ref="sitrep-1")
    m.observe("The clinic generator runs on depot fuel.",
              origin="document", source_ref="sitrep-1")
    m.recall("is route alpha open")
    return m, clock, a


def test_replay_needs_a_receipt():
    with Owl.open(os.path.join(tempfile.mkdtemp(), "n.owl"),
                  embedder=Toy()) as m:
        m.observe("A note.")
        assert "error" in m.replay()


def test_an_unchanged_store_replays_faithfully():
    m, clock, _ = _recorded(tempfile.mkdtemp())
    with m:
        out = m.replay()
        assert out["faithful"], out
        assert not out["drifted"]
        assert "same question, same evidence" in out["verdict"]


def test_replay_shows_WHY_the_old_answer_looked_right():
    """Not 'what did you record' -- bitemporal answers that. This is 'what
    would you have answered, and why was it wrong?'"""
    tmp = tempfile.mkdtemp()
    m, clock, a = _recorded(tmp)
    with m:
        clock.advance(days=3)
        m.observe("Route Alpha is closed by flooding.", origin="document",
                  source_ref="sitrep-2", supersedes=a)
        out = m.replay()
        assert out["faithful"], "the past must still reconstruct"
        assert out["drifted"], "the present must differ"
        assert "the evidence moved" in out["verdict"]
        assert out["no_longer_returned"] or out["newly_returned"]


def test_a_regression_is_named_as_one():
    """If replay does not match the receipt, the ENGINE changed -- and a
    memory whose past answers are not reproducible cannot be audited."""
    import json
    tmp = tempfile.mkdtemp()
    m, clock, _ = _recorded(tmp)
    with m:
        rid = m.receipts_log()[0]["id"]
        m._s.write(lambda c: c.execute(
            "UPDATE receipt SET returned=? WHERE id=?",
            (json.dumps([{"node_id": "obs_never_existed"}]), rid)))
        out = m.replay(rid)
        assert not out["faithful"]
        assert "REGRESSION" in out["verdict"]


def test_the_receipt_log_is_readable():
    m, _, _ = _recorded(tempfile.mkdtemp())
    with m:
        log = m.receipts_log()
        assert log and log[0]["query"] == "is route alpha open"
        assert "returned" in log[0]
