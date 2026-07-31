"""Write-time defence. Deterministic, Tier 0, no model.

A memory system is a persistence layer for beliefs. Anything that writes to
it is writing to the agent's mind, permanently: prompt injection is transient,
memory poisoning is not. A poisoned memory survives every restart, propagates
into every derived summary, and is retrieved as context forever.

OWL is unusually well placed -- provenance, immutability, Admiralty grading
and flow partitions are already defensive primitives -- but they were built
for honesty, not adversaries. This module is the difference.

Everything here is regex and arithmetic. Screening that needs a model cannot
run at ingest, and screening that does not run at ingest is not screening.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .lexical import tokenize

# Injection detection has to distinguish imperatives about the WORLD from
# imperatives about the AGENT. Real operational writing is full of the first:
# "always check the fuel gauge", "the generator must never be run without
# coolant". An earlier version weighted bare always/never and quarantined
# 2 of 6 legitimate field notes -- which the scoreboard caught, because it
# scores false positives alongside recall. Containment alone is worthless:
# a screen that quarantines everything scores 1.000.

# STRONG: an explicit attempt to override prior state. Nothing legitimate
# phrases itself this way.
_OVERRIDE = re.compile(
    r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?"
    r"(previous|prior|earlier|other|above|preceding)\b"
    r"|\b(this|the following)\s+(supersedes|overrides|replaces)\s+all\b"
    r"|\bdisregard\s+(all\s+)?(other|previous|prior)\b", re.I)

# STRONG: talks about the agent's own machinery.
_META = re.compile(
    r"\b(system prompt|previous instructions?|your (instructions?|rules?|"
    r"guidelines?|memory)|as an ai|prior directives?|memory (entry|record)|"
    r"remember that you|you are (required|instructed) to)\b", re.I)

# MODERATE: directed at the agent rather than describing the world.
_AGENT_DIRECTED = re.compile(
    r"\b(you (must|should|will|are to) (always|never)?\s*"
    r"(report|treat|trust|say|tell|assume|consider)"
    r"|from now on(?:,)? you"
    r"|do not (tell|mention|reveal|report)"
    r"|no matter what (?:the |any )?(other|source|record|evidence))\b", re.I)

# WEAK: only counts in combination. On its own this is ordinary prose.
_BARE_IMPERATIVE = re.compile(
    r"\b(always|never|from now on|regardless of|no matter what)\b", re.I)

_META = re.compile(
    r"\b(system prompt|previous instructions?|your (instructions?|rules?|"
    r"guidelines?)|as an ai|prior directives?|memory (entry|record)|"
    r"remember that you)\b", re.I)

_AUTHORITY = re.compile(
    r"\b(this (supersedes|overrides|replaces) all|highest priority|"
    r"authoritative|verified by|confirmed by (all|every)|"
    r"disregard (all )?(other|previous))\b", re.I)

_URGENCY = re.compile(
    r"\b(immediately|urgent(ly)?|critical(ly)?|at once|without delay|"
    r"do not (verify|check|question))\b", re.I)


@dataclass
class Screen:
    verdict: str = "clean"          # clean | suspect | blocked
    score: float = 0.0
    signals: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.verdict == "clean"


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def screen(content: str, *, origin: str = "document",
           contradiction_rate: float = 0.0) -> Screen:
    """Score a write for signs of poisoning.

    `contradiction_rate` is the fraction of established beliefs this content
    contradicts. One contradiction is an update. Mass contradiction in a
    single write is either a major event or an attack, and either way a human
    should look at it.
    """
    s = Screen()
    hits: list[tuple[str, float]] = []

    if _OVERRIDE.search(content):
        hits.append(("overrides_prior_state", 0.65))
    if _META.search(content):
        hits.append(("references_agent_internals", 0.55))
    if _AGENT_DIRECTED.search(content):
        hits.append(("directs_the_agent", 0.40))
    if _AUTHORITY.search(content):
        hits.append(("self_asserted_authority", 0.30))
    if _URGENCY.search(content):
        hits.append(("urgency_pressure", 0.15))
    # A bare imperative is ordinary operational prose ("always check the
    # gauge"). It only counts once something else has already fired.
    if hits and _BARE_IMPERATIVE.search(content):
        hits.append(("bare_imperative", 0.15))

    toks = tokenize(content)
    if len(content) > 200 and toks:
        # Injected payloads are often unusually dense or unusually repetitive
        # relative to prose.
        ent = shannon_entropy(content)
        if ent < 2.8:
            hits.append(("low_entropy_payload", 0.25))
        uniq = len(set(toks)) / len(toks)
        if uniq < 0.25:
            hits.append(("high_repetition", 0.20))

    if contradiction_rate > 0.3:
        hits.append(("mass_contradiction", 0.50))

    # A user speaking directly is a different trust proposition from a file of
    # unknown provenance. Imperatives from a person are normal speech.
    weight = 0.45 if origin == "user_utterance" else 1.0
    s.score = round(min(1.0, sum(w for _, w in hits) * weight), 3)
    s.signals = [n for n, _ in hits]
    if s.score >= 0.6:
        s.verdict = "blocked"
    elif s.score >= 0.3:
        s.verdict = "suspect"
    return s


# ── supersession authority ───────────────────────────────────────────────

_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 2}
COUP_WINDOW = 3600.0
COUP_LIMIT = 5


def may_supersede(new_grade: tuple[str, int], old_grade: tuple[str, int],
                  *, trust: str = "trusted") -> tuple[bool, str]:
    """A grade-F source may not overwrite a grade-B one.

    It can still register a CONFLICT -- disagreement is information and must
    not be silently discarded. What it cannot do is quietly replace a
    better-attested belief, which is the supersession-hijack attack.
    """
    if trust != "trusted":
        return False, (
            f"{trust} content may not supersede; recorded as a conflict")
    if _RANK.get(new_grade[0], 2) < _RANK.get(old_grade[0], 2):
        return False, (
            f"source grade {new_grade[0]} is weaker than the existing "
            f"{old_grade[0]}; recorded as a conflict, not a replacement")
    return True, "authority sufficient"


def is_coup(recent_attempts: int, window_limit: int = COUP_LIMIT
            ) -> tuple[bool, str]:
    """One source rapidly superseding many established claims is either a
    major real-world event or an attack. Either way, a human decides."""
    if recent_attempts > window_limit:
        return True, (
            f"{recent_attempts} supersessions from one source in the window "
            "-- belief coup; held for review")
    return False, ""
