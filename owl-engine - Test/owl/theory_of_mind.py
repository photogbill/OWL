"""Theory of Mind — the EPISTEMIC plane only.

A deliberate and load-bearing restriction. ToM decomposes into capacities that
are usually lumped together:

    1. Epistemic     -- what does the other party know, not know, believe falsely?
    2. Intentional   -- what are they trying to do?
    3. Attentional   -- what are they attending to right now?
    4. Affective     -- what are they feeling?
    5. Recursive     -- what do they think I think?

OWL implements (1) and, through the successor representation, a weak (2).
It implements NONE of 3-5, and that is not an omission.

Why: (4) is where sycophancy lives. A memory that models "will this upset
them?" and adjusts has been given an objective function of user comfort, and
in an analyst tool the most valuable output is frequently the one that
contradicts a deeply held premise. The v1 ATHENA brief proposed exactly this
check -- "does this contradict a deeply held premise?" -- in order to SOFTEN.

The correct use of the same mechanism is the exact inverse: detect the
contradiction in order to SURFACE it. Same computation, opposite sign.

So: OWL computes what the person knows and where their belief has diverged
from the record. It never decides how to say anything. Presentation belongs to
the host, behind a firewall, with the claim set diffed before and after.

--------------------------------------------------------------------------
Transactive memory (Wegner) is the operational core. Long-married couples and
effective teams maintain a model of what the OTHER party knows and a directory
of who is responsible for remembering what -- so the pair remembers more than
the sum of its members. Every memory system in the field models what the
machine knows. None models what the person knows. That is the asymmetry this
module closes.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import salience

DAY = 86400.0

# Levels of processing (Craik & Lockhart 1972): depth of encoding predicts
# retention far better than exposure count. A person who generated a claim
# themselves retains it far better than one who skimmed it in a briefing
# (the generation effect, Slamecka & Graf 1978).
CHANNEL_DEPTH = {
    "briefing": 0.5,        # read once, passively
    "document": 0.6,
    "conversation": 1.0,    # participated
    "recall": 1.3,          # they retrieved it themselves -- testing effect
    "generated": 1.5,       # they said it -- generation effect
    "correction": 1.6,      # they corrected the system: maximum depth
}


@dataclass(frozen=True)
class Held:
    node_id: str
    retrievability: float
    exposures: int
    last_exposed: float
    channel: str

    @property
    def at_risk(self) -> bool:
        return self.retrievability < 0.4


@dataclass(frozen=True)
class Divergence:
    who: str
    held_node: str
    truth_node: str
    direction: str          # 'user_stale' | 'ledger_stale'
    severity: float
    note: str


def model_retention(exposures: list[tuple[float, str]], now: float) -> float:
    """Run the forgetting model on the HUMAN.

    `exposures` is [(timestamp, channel)] in order. Each exposure is a review;
    depth maps to grade, so a skimmed briefing produces far less stability than
    a claim the person generated themselves. Spacing falls out for free,
    because FSRS rewards a successful review at low retrievability.
    """
    if not exposures:
        return 0.0
    t0, ch0 = exposures[0]
    s, d = salience.initial_state(_grade(ch0))
    last = t0
    for ts, ch in exposures[1:]:
        s, d = salience.review(s, d, ts - last, _grade(ch))
        last = ts
    return salience.retrievability(now - last, s)


def _grade(channel: str) -> int:
    depth = CHANNEL_DEPTH.get(channel, 0.8)
    if depth >= 1.4:
        return 4
    if depth >= 1.0:
        return 3
    if depth >= 0.6:
        return 2
    return 1


def divergence_severity(*, user_retention: float, claim_staleness: float,
                        consequence: float = 1.0) -> float:
    """How much it matters that the person is out of date.

    High when they still CONFIDENTLY hold the old version (retention high) and
    the ledger has moved on (staleness high). A belief they have already
    forgotten is not dangerous -- they will ask. A belief they are sure of and
    that is wrong is the one that gets someone hurt.
    """
    return max(0.0, min(1.0, user_retention * claim_staleness * consequence))


def resolve_direction(*, user_source_recency: float, user_was_present: bool,
                      ledger_recency: float,
                      ledger_admiralty: float) -> tuple[str, str]:
    """Symmetric false-belief resolution. The MACHINE MIGHT BE THE WRONG ONE.

    Sally-Anne is usually framed as 'the other party holds the false belief',
    and a system built on that assumption will confidently correct a person
    who was standing at the checkpoint an hour ago. In the field the human
    frequently holds better information than the ledger: they were there, and
    first-hand observation outranks a three-day-old document.

    So divergence resolves on provenance quality and recency, never on the
    assumption that the record wins.
    """
    if user_was_present and user_source_recency < ledger_recency:
        return "ledger_stale", (
            "the person has more recent first-hand observation than the "
            "record; treat the record as the stale side")
    if ledger_admiralty >= 0.65 and ledger_recency < user_source_recency:
        return "user_stale", (
            "the record is more recent and better sourced; the person is "
            "likely acting on superseded information")
    return "user_stale", "record is more recent; flag for confirmation, not correction"
