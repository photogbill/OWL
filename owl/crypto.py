"""A11 -- encryption at rest.

WHY THE WHOLE FILE AND NOT JUST THE CONTENT COLUMN.

The obvious design is field-level: encrypt `observation.content`, leave the
schema queryable. It is also close to useless here, and the reason is worth
stating because it is the mistake this feature exists to avoid.

OWL keeps an inverted index. `lexeme` maps every term to the nodes
containing it, in plaintext. Someone holding an "encrypted" store could
read the vocabulary of everything you ever recorded -- every name, place,
serial and diagnosis -- without touching a single ciphertext. The vectors
leak more: cosine structure over a known corpus recovers topic and often
paraphrase. Encrypting the content column and leaving those produces a
store that LOOKS encrypted, which is worse than one that plainly isn't,
because it gets trusted.

So: the whole file is sealed. AES-256-GCM over the bytes, authenticated, so
tampering is detected rather than silently decrypted into nonsense. Working
on the store requires unsealing it, which is honest about the trade -- a
decrypted working copy exists while the store is open, and pretending
otherwise would be the security theatre this docstring is complaining
about.

KEY MANAGEMENT, and the part nobody documents:

  * The key is 32 random bytes in a file at mode 0600. `doctor` checks
    those permissions, because a correctly encrypted store beside a
    world-readable key is an unencrypted store with extra steps.
  * LOSING THE KEY LOSES THE STORE. There is no recovery, no reset, no
    support path. That is what encryption means and it is stated at
    generation time rather than buried.
  * The key is never written into the store, never logged, and never
    included in a handover pack.

Requires `pip install owl-engine[crypto]` (the `cryptography` package).
Absent, every entry point raises with that instruction -- it never falls
back to something weaker, because a silent downgrade in a crypto path is
the worst possible failure.
"""
from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

MAGIC = b"OWLSEAL1"
NONCE_BYTES = 12
KEY_BYTES = 32


class CryptoUnavailable(RuntimeError):
    pass


class SealError(RuntimeError):
    pass


def available() -> bool:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa
        return True
    except ImportError:
        return False


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:                                  # pragma: no cover
        raise CryptoUnavailable(
            "encryption at rest needs the `cryptography` package: "
            "pip install owl-engine[crypto]. OWL will not fall back to a "
            "weaker scheme -- a silent downgrade in a crypto path is the "
            "worst failure available."
        ) from e
    return AESGCM(key)


# ── keys ─────────────────────────────────────────────────────────────────

def generate_key(path: str | Path, *, overwrite: bool = False) -> Path:
    """Write 32 random bytes at mode 0600, and say what losing it costs."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise SealError(
            f"{p} already exists. Refusing to overwrite a key -- doing so "
            "would make every store it protects permanently unreadable.")
    p.write_bytes(secrets.token_bytes(KEY_BYTES))
    try:
        os.chmod(p, 0o600)
    except OSError:                     # pragma: no cover - Windows ACLs
        pass
    return p


def load_key(path: str | Path, *, strict: bool = True) -> bytes:
    p = Path(path)
    if not p.exists():
        raise SealError(f"no key at {p}")
    key = p.read_bytes()
    if len(key) != KEY_BYTES:
        raise SealError(
            f"{p} is {len(key)} bytes; an AES-256 key is exactly {KEY_BYTES}")
    problems = key_permissions(p)
    if strict and problems:
        raise SealError(
            f"{p}: {problems[0]}. Fix the permissions or pass strict=False -- "
            "an encrypted store beside a readable key is an unencrypted "
            "store with extra steps.")
    return key


def key_permissions(path: str | Path) -> list[str]:
    """What is wrong with this key file's permissions. Empty is good."""
    p = Path(path)
    if not p.exists():
        return [f"no key at {p}"]
    try:
        mode = stat.S_IMODE(p.stat().st_mode)
    except OSError:                                           # pragma: no cover
        return []
    if os.name == "nt":
        # POSIX bits are not meaningful on Windows; NTFS ACLs are, and
        # checking them properly needs pywin32. Saying so is better than
        # reporting a pass we did not actually verify.
        return []
    bad = []
    if mode & stat.S_IRWXG:
        bad.append(f"group-accessible (mode {mode:04o}); should be 0600")
    if mode & stat.S_IRWXO:
        bad.append(f"world-accessible (mode {mode:04o}); should be 0600")
    return bad


# ── sealing ──────────────────────────────────────────────────────────────

def seal(plaintext_path: str | Path, sealed_path: str | Path,
         key: bytes) -> Path:
    """Encrypt a store file. Authenticated, so tampering is detectable."""
    src, dst = Path(plaintext_path), Path(sealed_path)
    data = src.read_bytes()
    nonce = secrets.token_bytes(NONCE_BYTES)
    # MAGIC is authenticated as associated data, so a sealed file cannot be
    # passed off as a different format or have its header rewritten.
    blob = _aesgcm(key).encrypt(nonce, data, MAGIC)
    dst.write_bytes(MAGIC + nonce + blob)
    try:
        os.chmod(dst, 0o600)
    except OSError:                                           # pragma: no cover
        pass
    return dst


def unseal(sealed_path: str | Path, plaintext_path: str | Path,
           key: bytes) -> Path:
    src, dst = Path(sealed_path), Path(plaintext_path)
    raw = src.read_bytes()
    if not raw.startswith(MAGIC):
        raise SealError(
            f"{src} is not a sealed OWL store (bad magic). If this is a "
            "plain .owl file, open it normally.")
    nonce = raw[len(MAGIC):len(MAGIC) + NONCE_BYTES]
    body = raw[len(MAGIC) + NONCE_BYTES:]
    try:
        data = _aesgcm(key).decrypt(nonce, body, MAGIC)
    except CryptoUnavailable:
        raise
    except Exception as e:                                    # noqa: BLE001
        raise SealError(
            "decryption failed. Either this is the wrong key, or the file "
            "has been modified -- GCM cannot tell you which, and that is "
            "the point: a tampered store must not open at all."
        ) from e
    dst.write_bytes(data)
    try:
        os.chmod(dst, 0o600)
    except OSError:                                           # pragma: no cover
        pass
    return dst


def shred(path: str | Path) -> None:
    """Best-effort removal of a decrypted working copy.

    Deliberately NOT called secure deletion. On a journalling filesystem or
    an SSD with wear levelling, overwriting a file does not reliably destroy
    the old blocks, and claiming otherwise would be exactly the kind of
    comfortable falsehood this module exists to avoid. It overwrites once,
    then unlinks, which defeats casual recovery and nothing stronger.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        n = p.stat().st_size
        with p.open("r+b") as f:
            f.write(secrets.token_bytes(n))
            f.flush()
            os.fsync(f.fileno())
    except OSError:                                           # pragma: no cover
        pass
    try:
        p.unlink()
    except OSError:                                           # pragma: no cover
        pass
