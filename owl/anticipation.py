"""F4 -- memory that raises its hand, and mostly does not.

Anticipatory retrieval is easy to build and almost always built wrong. The
retrieval half is a cheap match against open loops and past decisions. The
half that decides whether the feature is useful or hated is RESTRAINT.

An assistant that interrupts with something relevant is helpful once. An
assistant that interrupts with something relevant every fourth turn is
noise, and people do not disable it selectively -- they disable it, and the
one time it mattered it was off. So the budget is not a tuning knob, it is
the feature:

  * a hard cap per session, small
  * never twice for the same memory, ever
  * a cooling period after any interruption, so they cannot cluster
  * silence is the default and the correct answer nearly always

And it is measurable. `record_outcome()` logs whether an interruption was
acted on or dismissed. If that ratio is not strongly positive the honest
move is to leave the feature off, which is why it ships off.

Deliberately Tier 0: lexical overlap plus decision structure, no model. A
watcher that needs an inference call per turn cannot run per turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import lexical

# Restraint parameters. These are the feature, not its configuration.
MAX_PER_SESSION = 3
COOLDOWN_TURNS = 8
MIN_OVERLAP = 0.34          # fraction of the loop's terms present in the turn


@dataclass
class Nudge:
    node_id: str
    kind: str               # "open_loop" | "shifted_basis" | "commitment"
    message: str
    score: float
    why: str


@dataclass
class Watcher:
    """Per-session interruption budget. Cheap, stateful, and stingy."""

    max_per_session: int = MAX_PER_SESSION
    cooldown: int = COOLDOWN_TURNS
    min_overlap: float = MIN_OVERLAP
    fired: list[str] = field(default_factory=list)
    _turn: int = 0
    _last_fired_turn: int = -999
    outcomes: list[dict] = field(default_factory=list)

    @property
    def spent(self) -> int:
        return len(self.fired)

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.max_per_session

    def _silent_because(self) -> str | None:
        if self.exhausted:
            return (f"session budget spent ({self.max_per_session}); staying "
                    "quiet is the correct behaviour, not a failure")
        if self._turn - self._last_fired_turn < self.cooldown:
            return (f"within cooldown ({self.cooldown} turns); interruptions "
                    "must not cluster")
        return None

    def consider(self, turn_text: str, candidates: list[dict]) -> Nudge | None:
        """Look at the live turn. Usually return None.

        `candidates` are dicts with node_id, kind, text, message. Passed in
        rather than fetched so the whole restraint policy is testable with a
        list literal and no store.
        """
        self._turn += 1
        if self._silent_because() is not None:
            return None

        terms = set(lexical.tokenize(turn_text))
        if not terms:
            return None

        best: Nudge | None = None
        for c in candidates:
            if c["node_id"] in self.fired:
                continue                    # never twice for the same thing
            ct = set(lexical.tokenize(c.get("text", "")))
            if not ct:
                continue
            overlap = len(terms & ct) / len(ct)
            if overlap < self.min_overlap:
                continue
            # A decision resting on moved ground outranks an open loop at
            # the same overlap: one is a thing you meant to do, the other is
            # a thing you already did for a reason that stopped being true.
            weight = {"shifted_basis": 1.0, "commitment": 0.8,
                      "open_loop": 0.6}.get(c["kind"], 0.5)
            score = overlap * weight
            if best is None or score > best.score:
                best = Nudge(c["node_id"], c["kind"], c["message"],
                             round(score, 3),
                             f"{len(terms & ct)}/{len(ct)} terms overlap")
        if best is None:
            return None
        self.fired.append(best.node_id)
        self._last_fired_turn = self._turn
        return best

    def record_outcome(self, node_id: str, acted_on: bool) -> None:
        """The number that decides whether this feature should exist."""
        self.outcomes.append({"node_id": node_id, "acted_on": bool(acted_on)})

    def verdict(self) -> dict:
        """Acted-on vs dismissed. Honest about small samples.

        The acceptance criterion is that this ratio is strongly positive or
        the feature stays off -- so it has to be reported in a form that can
        actually fail, including 'not enough evidence yet'.
        """
        n = len(self.outcomes)
        acted = sum(o["acted_on"] for o in self.outcomes)
        if n < 5:
            return {"n": n, "acted_on": acted, "ratio": None,
                    "verdict": "insufficient evidence -- keep it off",
                    "keep_enabled": False}
        ratio = acted / n
        return {
            "n": n, "acted_on": acted, "ratio": round(ratio, 3),
            "verdict": ("earning its interruptions" if ratio >= 0.6 else
                        "borderline; tighten the budget" if ratio >= 0.4 else
                        "not earning them -- turn it off"),
            "keep_enabled": ratio >= 0.6,
        }
