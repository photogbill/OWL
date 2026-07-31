"""C1 reconstructive compression + C4 quarantined hypotheses.

Both are Tier 2 -- they need a Reasoner. What is built here is the part
that must be right *regardless* of which model is attached: the proof
obligation and the lifecycle. The model is allowed to be wrong; it is not
allowed to lose something silently or to have a guess presented as a fact.

C1 -- COMPRESS ONLY WHAT YOU CAN PROVE YOU CAN REBUILD.

Human memory is reconstructive: you do not store the scene, you store
enough to rebuild it, and the rebuild is lossy in ways you cannot detect
from the inside. That last clause is why every "summarise old memories"
feature is dangerous -- the system cannot tell what it dropped.

So compression here is gated on a test the system runs against itself:
take the cue plus the surviving neighbours, reconstruct, and compare to the
original. If reconstruction fails, the content STAYS VERBATIM. The
compression is earned, not assumed, and the failure mode is "no space
saved" rather than "a memory quietly became wrong".

Never applies to `verbatim` content, whatever the reconstruction says. A
grid reference that round-trips by luck is still a grid reference.

C4 -- A GUESS MUST NEVER LOOK LIKE A FACT.

REM recombination produces hypotheses. The lifecycle is explicit --
generated, tested, then promoted or archived-failed or expired -- and every
state except `promoted` is barred from export. A hypothesis with no
falsifier cannot be created at all, which the schema already enforces:
something that cannot be wrong is not a hypothesis, it is a belief with
better marketing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# How close a reconstruction must be to count. High on purpose: this is a
# gate on destroying the original, so the burden of proof sits with the
# compressor.
FIDELITY_FLOOR = 0.92


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def fidelity(original: str, reconstructed: str) -> float:
    """How much of the original survived the round trip.

    RECALL-weighted, deliberately asymmetric: a reconstruction that adds
    material is merely verbose, one that drops material has destroyed
    something. Symmetric similarity would let a fluent paraphrase that lost
    a number score well.
    """
    a, b = _tokens(original), set(_tokens(reconstructed))
    if not a:
        return 1.0
    kept = sum(1 for t in a if t in b)
    recall = kept / len(a)
    # Numbers and identifiers are all-or-nothing. Losing one is not a
    # fractional loss of meaning, it is a different claim.
    hard = [t for t in a if any(ch.isdigit() for ch in t)]
    if hard and not all(t in b for t in hard):
        return 0.0
    return round(recall, 4)


@dataclass
class CompressionPlan:
    node_id: str
    keep_verbatim: bool
    fidelity: float
    reason: str
    compressed: str = ""


def plan_compression(node_id: str, original: str, *, claim_class: str,
                     reconstruct, cue: str = "",
                     neighbours: list[str] | None = None,
                     floor: float = FIDELITY_FLOOR) -> CompressionPlan:
    """Decide whether this memory may be compressed. Default: no.

    `reconstruct(cue, neighbours) -> str` is the Reasoner's job. Everything
    around it is the safety property, and it holds whether the model is
    good, bad, or absent.
    """
    if claim_class == "verbatim":
        return CompressionPlan(node_id, True, 1.0,
                               "verbatim content is never compressed -- an "
                               "exact string that round-trips by luck is "
                               "still an exact string")
    if reconstruct is None:
        return CompressionPlan(node_id, True, 0.0,
                               "no Reasoner attached; Tier 2 pass skipped, "
                               "not silently no-opped")
    try:
        rebuilt = reconstruct(cue or original[:60], neighbours or [])
    except Exception as exc:                                  # noqa: BLE001
        return CompressionPlan(node_id, True, 0.0,
                               f"reconstruction raised ({exc.__class__.__name__});"
                               " keeping the original")
    f = fidelity(original, rebuilt or "")
    if f < floor:
        return CompressionPlan(
            node_id, True, f,
            f"reconstruction scored {f:.2f} against a floor of {floor:.2f} "
            "-- the system cannot prove it could rebuild this, so it keeps it")
    return CompressionPlan(node_id, False, f,
                           f"reconstructible at {f:.2f}; safe to compress",
                           compressed=rebuilt)


# ── C4: hypothesis lifecycle ─────────────────────────────────────────────

STATES = ("generated", "testing", "promoted", "archived_failed", "expired")
EXPORTABLE = ("promoted",)


@dataclass
class Hypothesis:
    node_id: str
    statement: str
    falsifier: str
    state: str = "generated"
    created_at: float = 0.0
    expires_at: float = 0.0
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.falsifier.strip():
            raise ValueError(
                "a hypothesis without a falsifier is not a hypothesis. If "
                "nothing could show it wrong, it is a belief with better "
                "marketing, and it must not enter the store as one.")

    @property
    def exportable(self) -> bool:
        return self.state in EXPORTABLE

    @property
    def presentable_as_fact(self) -> bool:
        """Never. Included so the answer is explicit rather than assumed."""
        return False

    def label(self) -> str:
        return {
            "generated": "*HYPOTHESIS, untested*",
            "testing": "*HYPOTHESIS, under test*",
            "promoted": "*was a hypothesis, now supported by evidence*",
            "archived_failed": "*FAILED hypothesis, kept as a dead end*",
            "expired": "*hypothesis, never tested, expired*",
        }[self.state]


def advance(h: Hypothesis, *, now: float,
            supporting: int = 0, opposing: int = 0) -> Hypothesis:
    """Move a hypothesis through its lifecycle. Deterministic.

    Failure is ARCHIVED, not deleted. "We tried this and it did not work"
    is expensive knowledge and stops the same rejected idea being
    re-proposed every cycle -- the same reason recorded absence is worth
    keeping.
    """
    if h.state in ("promoted", "archived_failed"):
        return h
    if h.expires_at and now >= h.expires_at and h.state != "testing":
        return Hypothesis(h.node_id, h.statement, h.falsifier, "expired",
                          h.created_at, h.expires_at, h.evidence_for,
                          h.evidence_against)
    if opposing > supporting and opposing >= 1:
        return Hypothesis(h.node_id, h.statement, h.falsifier,
                          "archived_failed", h.created_at, h.expires_at,
                          h.evidence_for, h.evidence_against)
    if supporting >= 2 and opposing == 0:
        return Hypothesis(h.node_id, h.statement, h.falsifier, "promoted",
                          h.created_at, h.expires_at, h.evidence_for,
                          h.evidence_against)
    if supporting or opposing:
        return Hypothesis(h.node_id, h.statement, h.falsifier, "testing",
                          h.created_at, h.expires_at, h.evidence_for,
                          h.evidence_against)
    return h
