"""B8 -- "nothing on that, but I have this next to it."

    "Nothing on tanker arrival, but I have the depot dispatch schedule and
     a note that tankers dispatch on request."

That is a genuinely useful answer, and it is ONE STEP from the failure this
whole engine exists to prevent. "I have nothing, but here are five loosely
related things" is exactly the behaviour the six-state design was built to
stop; adding an apology to the front of it does not make it better. So the
guard is strict, and the feature is designed to stay silent:

  1. **DONT_KNOW only.** It never appends to a real answer. A KNOW that
     trails suggestions is a KNOW you have made harder to read.

  2. **Absolute strength, not relative.** The adjacent match must be strong
     on its own terms. Taking the best of a bad set and calling it adjacent
     is max-normalisation wearing a hat -- the same defect already fixed
     twice in this codebase, in the lexical scorer and again in the
     semantic blend.

  3. **It must share a SUBJECT, not merely words.** "The clinic has twelve
     beds" is not adjacent to "who runs the clinic" in any way a reader
     benefits from; it just shares a noun. Adjacency requires overlap on
     the query's content terms AND a different answer-type, which is what
     makes it a neighbouring question rather than a worse answer to this
     one.

  4. **Two, at most.** A list is a search result. Two is a pointer.

  5. **Labelled, always.** The caller must never be able to mistake an
     adjacent note for an answer, so it is returned in its own field and
     the prose says what it is.

The plan flags this as low priority and says to cut it if it does not earn
its keep. `earned_its_keep()` is therefore part of the feature: it counts
how often the suggestion is offered versus acted on, in the same shape as
F4's verdict, so the decision to keep it can be made from data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import entities, lexical

# Absolute floor. A suggestion below this is a bad answer with an apology
# attached, and offering it is worse than silence.
MIN_STRENGTH = 0.45
MAX_SUGGESTIONS = 2
MIN_SHARED_TERMS = 1


@dataclass
class Adjacent:
    node_id: str
    content: str
    strength: float
    why: str


@dataclass
class Suggestion:
    items: list = field(default_factory=list)

    @property
    def offered(self) -> bool:
        return bool(self.items)

    def sentence(self) -> str:
        """Prose the caller can hand straight to a user.

        Deliberately leads with the absence. The adjacent material is
        offered as a lead, never as an answer -- a sentence that opens with
        the suggestion invites it to be read as one.
        """
        if not self.items:
            return ""
        if len(self.items) == 1:
            return (f"Nothing directly on that. Nearby, I do have: "
                    f"{self.items[0].content}")
        joined = "; ".join(i.content for i in self.items)
        return f"Nothing directly on that. Nearby, I do have: {joined}"


def find(query: str, candidates: list[dict], *,
         min_strength: float = MIN_STRENGTH,
         limit: int = MAX_SUGGESTIONS) -> Suggestion:
    """Adjacent material for a query that came back empty.

    `candidates` are dicts with node_id, content and score -- passed in
    rather than fetched, so the restraint policy is testable with a list
    literal and no store.
    """
    q_terms = set(lexical.tokenize(query))
    if not q_terms:
        return Suggestion()
    want = entities.predict_answer_type(query)

    out: list[Adjacent] = []
    for c in candidates:
        score = float(c.get("score") or 0.0)
        if score < min_strength:
            continue                    # absolute, not best-of-a-bad-set
        terms = set(lexical.tokenize(c.get("content", "")))
        shared = q_terms & terms
        if len(shared) < MIN_SHARED_TERMS:
            continue                    # unrelated, not adjacent

        # A NEIGHBOURING QUESTION, not a worse answer to this one. If the
        # candidate carries the type the query asked for, it was already
        # considered and rejected by the gate -- re-offering it here would
        # be overruling that decision through a side door.
        if want and entities.content_affinity(c.get("content", ""),
                                              want) > 1.0:
            continue

        out.append(Adjacent(
            c["node_id"], c.get("content", ""), round(score, 4),
            f"shares {sorted(shared)!r} with the question but answers a "
            "different one"))

    out.sort(key=lambda a: -a.strength)
    return Suggestion(out[:limit])


@dataclass
class Keep:
    """Does this feature earn its keep? The plan says cut it if not.

    Same shape as F4's verdict, and for the same reason: a feature that
    fires rarely and is ignored when it does is costing attention for
    nothing, and that can only be known by counting.
    """

    offered: int = 0
    acted_on: int = 0

    def record(self, *, was_offered: bool, was_used: bool = False) -> None:
        self.offered += bool(was_offered)
        self.acted_on += bool(was_used)

    def verdict(self) -> dict:
        if self.offered < 10:
            return {"offered": self.offered, "acted_on": self.acted_on,
                    "rate": None,
                    "verdict": "insufficient evidence -- keep observing",
                    "keep": True}
        rate = self.acted_on / self.offered
        return {
            "offered": self.offered, "acted_on": self.acted_on,
            "rate": round(rate, 3),
            "verdict": ("earning its keep" if rate >= 0.3 else
                        "clever and ignored -- the plan said cut it if this "
                        "happened, and this is that"),
            "keep": rate >= 0.3,
        }
