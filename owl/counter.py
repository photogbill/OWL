"""B6 -- deliberately retrieve what would change your mind.

Similarity search is a confirmation-bias engine, and not by accident: its
objective function is "find text like this query". Ask *"why is the pump
failing?"* and every candidate that survives ranking is pump-failure
material. Last week's note saying the pump was fine is semantically distant
from the query -- it shares no failure vocabulary -- so it ranks nowhere, and
the system agrees with you fluently.

That is worse than a gap. A memory that only ever returns support makes an
analyst *more* confident with each query, using their own premise as the
retrieval key.

Three disconfirming shapes, cheapest first:

  1. **Presupposition denial.** The query assumes something. Find memories
     that assert its negation directly -- "the pump is fine", "route alpha
     is open" against a query that assumes it is closed.

  2. **Superseded-by-the-premise.** If the belief the query rests on
     replaced an earlier one, that earlier one IS the counter-evidence, and
     the supersession graph already holds it. Free, exact, no model.

  3. **Adjacent but oppositely valenced.** Semantically near the topic while
     carrying the opposite polarity -- which needs real embeddings, hence
     the B1 gate. At Tier 0 this surfaces noise and discredits the feature,
     so it is skipped and SAID to be skipped rather than half-run.

The output is never merged into the main result. A counter-set presented as
if it were part of the answer is just a confusing answer; presented
separately it is a challenge, which is the point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Polarity pairs that actually invert a claim, rather than merely differing.
# Kept small and specific on purpose: a big list of loose antonyms produces
# confident nonsense, and this feature dies the first time it does that.
    # Curated OPPOSED GROUPS rather than word pairs. A pair table said
# failing/functioning and missed "working normally" -- the same claim in
# different words -- so the contradiction it exists to find slipped past.
# Groups let synonyms share an opposite without inviting a thesaurus, which
# is what would make this produce confident nonsense.
OPPOSED: tuple[tuple[frozenset, frozenset], ...] = (
    (frozenset({"working", "functioning", "operational", "running",
                "serviceable", "fine", "normal", "healthy"}),
     frozenset({"failing", "failed", "broken", "down", "unserviceable",
                "faulty", "dead"})),
    (frozenset({"open", "passable", "clear", "accessible"}),
     frozenset({"closed", "impassable", "blocked", "inaccessible"})),
    (frozenset({"intact", "standing", "sound"}),
     frozenset({"collapsed", "destroyed", "damaged"})),
    (frozenset({"available", "present", "stocked", "full"}),
     frozenset({"unavailable", "absent", "out", "empty", "depleted"})),
    (frozenset({"safe", "secure", "permissive"}),
     frozenset({"unsafe", "insecure", "hostile"})),
    (frozenset({"arrived", "delivered", "ontime"}),
     frozenset({"delayed", "missing", "late", "overdue"})),
    (frozenset({"confirmed", "verified"}),
     frozenset({"unconfirmed", "unverified", "disputed"})),
    (frozenset({"rising", "increase", "increasing", "up"}),
     frozenset({"falling", "decrease", "decreasing", "down"})),
)

NEGATORS = ("not", "no", "never", "isn't", "aren't", "wasn't", "weren't",
            "cannot", "can't", "won't", "without", "failed to", "denies",
            "contradicts", "disputes", "refutes")

# WORD boundaries, not substrings. A plain `"no" in text` matches inside
# "north", "nothing" and "another" -- which made "The north pump is failing"
# register as a negation of "the north pump is failing" and returned the
# supporting memory as its own contradiction. Precisely the confident
# nonsense this module's docstring warns about, produced by its own code.
_NEGATION = re.compile(
    r"(?<![\w'])(" + "|".join(re.escape(n) for n in NEGATORS) + r")(?![\w'])",
    re.I)


def negated(text: str) -> bool:
    return bool(_NEGATION.search(text))

_FAILURE_FRAME = re.compile(
    r"\b(why (is|are|was|were|did|does)|what caused|how did .* fail|"
    r"reason for|cause of)\b", re.I)


@dataclass
class Counter:
    node_id: str
    content: str
    kind: str            # "negation" | "antonym" | "superseded_premise"
    why: str
    strength: float = 0.0


@dataclass
class CounterSet:
    presupposition: str = ""
    counters: list[Counter] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.counters)


def presupposition(query: str) -> str:
    """What the question takes for granted.

    "Why is the pump failing?" presupposes the pump is failing. Stripping
    the interrogative frame leaves the assumed proposition, which is the
    thing to look for the negation of.
    """
    q = query.strip().rstrip("?")
    q = _FAILURE_FRAME.sub("", q).strip()
    q = re.sub(r"^(the|a|an)\s+", "", q, flags=re.I)
    return q.strip()


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if len(w) > 2}


def polarity_flip(text: str) -> set[str]:
    """Every word whose presence would INVERT this text's claim."""
    have = _terms(text)
    flips: set[str] = set()
    for pos, neg in OPPOSED:
        if have & pos:
            flips |= neg
        if have & neg:
            flips |= pos
    return flips


def find(query: str, candidates: list[dict], *,
         superseded: list[dict] | None = None,
         semantic_available: bool = False,
         limit: int = 5) -> CounterSet:
    """Assemble the disconfirming set.

    `candidates` are dicts with node_id and content -- passed in rather than
    fetched so the whole policy is testable with a list literal.
    """
    pre = presupposition(query)
    q_terms = _terms(pre)
    wanted_flips = polarity_flip(pre)
    out = CounterSet(presupposition=pre)

    for c in candidates:
        content = c.get("content", "")
        terms = _terms(content)
        shared = q_terms & terms
        if len(shared) < 2:
            continue                        # not about the same thing at all

        # 1 -- explicit negation of the assumed proposition. Only counts if
        # the PREMISE is not itself negated: "the pump is not working" does
        # not contradict a question asking why it is failing, it agrees.
        if negated(content) and not negated(pre):
            out.counters.append(Counter(
                c["node_id"], content, "negation",
                f"negates a claim sharing {len(shared)} term(s) with the "
                "question's premise",
                round(len(shared) / max(1, len(q_terms)), 3)))
            continue

        # 2 -- opposite polarity on the same subject
        hit = wanted_flips & terms
        if hit:
            out.counters.append(Counter(
                c["node_id"], content, "antonym",
                f"asserts {sorted(hit)!r} where the question assumes the "
                "opposite",
                round(len(shared) / max(1, len(q_terms)) + 0.2, 3)))

    # 3 -- the premise replaced something, and that something disagrees
    for s in (superseded or []):
        out.counters.append(Counter(
            s["node_id"], s.get("content", ""), "superseded_premise",
            "this is what the current belief replaced -- the record of "
            "having thought otherwise", 0.5))

    if not semantic_available:
        out.skipped.append(
            "semantic opposition (needs a real embedder; at Tier 0 it "
            "surfaces noise and discredits the rest)")

    # One memory, one entry. A note can be both the opposite polarity AND
    # the thing the premise replaced -- that is two reasons, not two pieces
    # of evidence, and listing it twice inflates how much disagreement
    # there appears to be. Keep the strongest reason, name the others.
    out.counters.sort(key=lambda c: -c.strength)
    seen: dict[str, Counter] = {}
    for c in out.counters:
        if c.node_id in seen:
            seen[c.node_id].why += f"; also {c.kind}"
        else:
            seen[c.node_id] = c
    out.counters = list(seen.values())[:limit]
    return out
