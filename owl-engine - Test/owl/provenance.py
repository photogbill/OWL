"""The monotonicity invariant — OWL's single most important property.

Source-monitoring failure (Johnson, Hashtroudi & Lindsay 1993) is how false
memories actually form: content is remembered correctly and its SOURCE is
remembered wrongly. A memory system that mixes user facts, document facts,
model inferences and self-generated hypotheses in one store, with no
provenance, will eventually assert a dream as a fact -- and be unable to tell,
including to itself.

Two rules, enforced here and fuzz-tested in CI:

    confidence(node)  <= min(confidence(parents))
    epistemic(node)   >= max(epistemic(parents))

Abstraction therefore cannot launder speculation into fact. Any node with a
hypothesized ancestor is hypothesized, forever, until an explicit corroboration
event promotes it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .protocols import Epistemic, MonotonicityError


@dataclass(frozen=True)
class ParentFacts:
    node_id: str
    confidence: float
    epistemic: Epistemic


def resolve(parents: list[ParentFacts], *, proposed_confidence: float,
            proposed_epistemic: Epistemic) -> tuple[float, Epistemic]:
    """Clamp a proposed derived node into the legal region. Never raises.

    Use this on the write path: it is always better to silently weaken a claim
    than to reject the write and lose the derivation.
    """
    if not parents:
        return proposed_confidence, proposed_epistemic
    ceiling = min(p.confidence for p in parents)
    floor = max(parents, key=lambda p: p.epistemic.rank).epistemic
    conf = min(proposed_confidence, ceiling)
    epi = proposed_epistemic if proposed_epistemic.rank >= floor.rank else floor
    return conf, epi


def assert_monotonic(parents: list[ParentFacts], *, confidence: float,
                     epistemic: Epistemic, node_id: str = "?") -> None:
    """Verify the invariant. Use in tests and in `doctor()`. Raises."""
    if not parents:
        return
    ceiling = min(p.confidence for p in parents)
    if confidence > ceiling + 1e-9:
        raise MonotonicityError(
            f"{node_id}: confidence {confidence:.3f} exceeds parent ceiling "
            f"{ceiling:.3f}"
        )
    floor = max(parents, key=lambda p: p.epistemic.rank).epistemic
    if epistemic.rank < floor.rank:
        raise MonotonicityError(
            f"{node_id}: epistemic '{epistemic.value}' is more certain than "
            f"parent floor '{floor.value}'"
        )


def is_presentable_as_fact(epistemic: Epistemic) -> bool:
    """Only observed/reported content may be stated flatly to a user.

    inferred  -> must be marked as the system's own conclusion
    hypothesized -> must never leave the machine without explicit promotion
    """
    return epistemic.rank <= Epistemic.REPORTED.rank
