"""H1 -- retrieval by the SHAPE of a memory, not its similarity.

Embeddings answer "what is this about". They cannot answer "who did what to
whom", because a vector is a bag: *Ahmed delivered the gasket to Warsame*
and *Warsame delivered the gasket to Ahmed* embed almost identically. Every
memory system in the field inherits that blindness, and for an analyst
toolkit it is a serious one -- the direction of an action is frequently the
entire content of the report.

Vector Symbolic Architectures (Plate, Kanerva) fix it with two operations
on high-dimensional bipolar vectors:

    BIND    (x)  elementwise multiply. Produces something DISSIMILAR to
                 both inputs -- it is a key-value pairing, not a blend.
                 Self-inverse: (a x b) x b == a, which is what makes
                 querying possible at all.
    BUNDLE  (+)  elementwise sum, then sign. Produces something SIMILAR to
                 every input -- a set.

A memory becomes a structured trace:

    trace = (AGENT x ahmed) + (ACTION x delivered) + (OBJECT x gasket)

and "who delivered the gasket?" is not a similarity search. It is algebra:
unbind the trace by AGENT, get a noisy vector, clean it up against the item
memory. The answer falls out of the structure.

WHY THIS WORKS IN 8192 DIMENSIONS AND NOT IN 300. Random bipolar vectors in
very high dimensions are quasi-orthogonal -- two random 8192-d vectors have
expected cosine 0 with standard deviation ~1/sqrt(8192) ≈ 0.011. That
concentration is what lets a bundle of five bindings still be recognisably
similar to each of them while the crosstalk stays in the noise. Below about
1000 dimensions the noise swamps the signal and the whole scheme quietly
returns nonsense, which is why the dimension is not a tuning knob.

DELIBERATELY SEPARATE FROM THE EMBEDDING SPACES. Episodic detail, semantic
gist and structural role are different kinds of information, and collapsing
them into one vector is how a system loses the ability to distinguish them.
This is a third index, consulted for structural questions, not a
replacement for either existing space.

Stdlib only, deterministic, no model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Below ~1000 the quasi-orthogonality this depends on breaks down and the
# scheme returns confident nonsense. Not a tuning knob.
DIM = 8192
MIN_SAFE_DIM = 1024

ROLES = ("AGENT", "ACTION", "OBJECT", "RECIPIENT", "LOCATION", "TIME")


def _symbol(name: str, dim: int = DIM) -> tuple[int, ...]:
    """A deterministic pseudo-random bipolar vector for a symbol.

    Derived from a hash of the name rather than stored, so the same symbol
    is the same vector in every process and across restarts -- A10 again,
    and it means an index can be rebuilt from content alone.
    """
    out = []
    counter = 0
    while len(out) < dim:
        h = hashlib.blake2b(f"{name}#{counter}".encode(), digest_size=64)
        for byte in h.digest():
            for bit in range(8):
                out.append(1 if (byte >> bit) & 1 else -1)
                if len(out) == dim:
                    break
            if len(out) == dim:
                break
        counter += 1
    return tuple(out)


def bind(a, b):
    """Key-value pairing. Dissimilar to both inputs, and self-inverse."""
    return tuple(x * y for x, y in zip(a, b))


def bundle(*vs):
    """Set membership. Similar to every input.

    Ties break to +1 rather than randomly -- with an even number of
    components the sum can be exactly zero, and a random tie-break there
    would make encoding non-reproducible.
    """
    if not vs:
        return ()
    return tuple(1 if s >= 0 else -1
                 for s in (sum(col) for col in zip(*vs)))


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / len(a)


@dataclass
class Trace:
    node_id: str
    vector: tuple
    roles: dict = field(default_factory=dict)   # role -> filler name

    def render(self) -> str:
        return " ".join(f"{r}={self.roles[r]}" for r in ROLES
                        if r in self.roles)


@dataclass
class StructuralIndex:
    """Role-filler traces plus an item memory to clean up against."""

    dim: int = DIM
    traces: list = field(default_factory=list)
    items: dict = field(default_factory=dict)   # name -> vector

    def __post_init__(self) -> None:
        if self.dim < MIN_SAFE_DIM:
            raise ValueError(
                f"dim={self.dim} is below {MIN_SAFE_DIM}. Quasi-orthogonality "
                "is what makes VSA work; at this dimension the crosstalk "
                "swamps the signal and every query returns confident "
                "nonsense. This is not a tuning knob.")

    def _sym(self, name: str):
        v = self.items.get(name)
        if v is None:
            v = self.items[name] = _symbol(name, self.dim)
        return v

    def encode(self, node_id: str, **roles) -> Trace:
        """Store one structured memory: encode(n, AGENT='ahmed', ...)."""
        parts, kept = [], {}
        for role, filler in roles.items():
            r = role.upper()
            if r not in ROLES or not filler:
                continue
            kept[r] = filler
            parts.append(bind(self._sym(f"ROLE::{r}"), self._sym(str(filler))))
        if not parts:
            raise ValueError("a trace with no roles encodes nothing")
        t = Trace(node_id, bundle(*parts), kept)
        self.traces.append(t)
        return t

    def query(self, role: str, *, top_k: int = 3, **constraints) -> list[dict]:
        """Answer "who did X to Y" structurally.

        Traces are filtered by the constrained roles first -- BIND is exact,
        so a trace either contains (OBJECT x gasket) or it does not, and
        checking that is a similarity test that either clears the noise
        floor or does not. Then the unknown role is unbound and cleaned up
        against item memory.
        """
        want = role.upper()
        rv = self._sym(f"ROLE::{want}")
        # Noise floor for "this trace really does contain that binding".
        # 3 sigma of the quasi-orthogonal distribution: anything below is
        # indistinguishable from an unrelated trace.
        floor = 4.0 / (self.dim ** 0.5)

        out = []
        for t in self.traces:
            ok = True
            for crole, cfill in constraints.items():
                probe = bind(self._sym(f"ROLE::{crole.upper()}"),
                             self._sym(str(cfill)))
                if cosine(t.vector, probe) < floor:
                    ok = False
                    break
            if not ok:
                continue
            noisy = bind(t.vector, rv)          # unbind: self-inverse
            best, score = None, -1.0
            for name, v in self.items.items():
                if name.startswith("ROLE::"):
                    continue
                s = cosine(noisy, v)
                if s > score:
                    best, score = name, s
            if best is not None and score >= floor:
                out.append({"node_id": t.node_id, "answer": best,
                            "score": round(score, 4),
                            "trace": t.render()})
        out.sort(key=lambda r: -r["score"])
        return out[:top_k]

    def crosstalk(self) -> dict:
        """How much NOISE the bundling is producing. Report, don't assume.

        A VSA that is silently over-capacity returns plausible wrong
        fillers, so the capacity limit is measured rather than trusted --
        the same reason the ANN index reports its recall.

        Measured only between traces sharing NO fillers. The first version
        compared every pair, which meant a store where twenty reports all
        had ACTION='delivered' showed high similarity and got called
        unhealthy -- but that similarity is the shared structure being
        represented correctly, and it is the entire point of the scheme.
        Only similarity between traces with nothing in common is noise.
        """
        n = len(self.traces)
        if n < 2:
            return {"pairs": 0, "mean": 0.0, "max": 0.0, "expected": 0.0,
                    "shared_structure_pairs": 0, "healthy": True}
        expected = 1.0 / (self.dim ** 0.5)
        noise, shared = [], 0
        for i, a in enumerate(self.traces):
            for b in self.traces[i + 1:]:
                if set(a.roles.values()) & set(b.roles.values()):
                    shared += 1          # legitimately similar, not noise
                    continue
                noise.append(abs(cosine(a.vector, b.vector)))
        if not noise:
            return {"pairs": 0, "mean": 0.0, "max": 0.0,
                    "expected": round(expected, 4),
                    "shared_structure_pairs": shared, "healthy": True,
                    "note": "every pair shares a filler; no independent "
                            "pairs to measure noise between"}
        return {
            "pairs": len(noise),
            "mean": round(sum(noise) / len(noise), 4),
            "max": round(max(noise), 4),
            "expected": round(expected, 4),
            "shared_structure_pairs": shared,
            "healthy": max(noise) < 8 * expected,
        }
