"""Epistemic half-life — confidence decays on its own curve.

Every memory system that models forgetting decays RETRIEVABILITY: can I still
find this? None decays CREDIBILITY: should I still believe it?

Those are different questions and conflating them is dangerous in the field.
"Route Alpha is open" stays perfectly retrievable for six months and becomes
worthless. "Dr Warsame speaks Somali" is true forever. A memory that cannot
tell these apart will hand a person stale operational intelligence with the
same confidence as a permanent fact.

Half-lives are LEARNED, not configured. Every supersession is a labelled
survival observation: this class of claim held for N days before it changed.
"""
from __future__ import annotations

import math
import re

DAY = 86400.0

# Priors, used until enough supersession events accumulate to fit.
# Deliberately conservative: it is safer to over-flag staleness than under.
PRIOR_HALFLIFE = {
    "verbatim": float("inf"),   # exact strings do not become less exact
    "identity": float("inf"),   # who someone is, what a building is
    "capacity": 180 * DAY,      # bed count, generator wattage, stock ceiling
    "status": 3 * DAY,          # route open, who is on duty, stock level
    "position": 6 * 3600.0,     # where a convoy is right now
    "unknown": 30 * DAY,
}

MIN_EVENTS_TO_FIT = 12

# Content that is worthless unless exact. Checked FIRST, and it wins:
# a grid reference is not a "position claim" to be aged out, it is a string
# that must survive verbatim or not at all.
_VERBATIM = (
    r"\b\d{1,3}\s*[.,]\s*\d{3,}\s*[,/]\s*-?\d{1,3}\s*[.,]\s*\d{3,}",  # lat/long
    r"\b\d{1,2}[A-Z]{1,3}\s?[A-Z]{2}\s?\d{4,10}\b",                    # MGRS
    r"\b[A-Z]{1,4}-?\d{3,}\b",                                        # serials
    r"\b\d{1,4}(\.\d+)?\s?(mg|ml|mcg|g|iu|units?)\b",                  # dosage
    r"\b\d{2,3}\.\d{1,4}\s?(mhz|khz|ghz)\b",                          # frequency
    r"\b(\+?\d[\d\s().-]{7,}\d)\b",                                   # phone
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",                      # email
    r"\b(passcode|password|pin|combination|bypass|access code|call ?sign)\b",
)

# Patterns that name a SPECIFIC thing, where a near-miss is a wrong answer
# rather than a worse one. Deliberately narrower than _VERBATIM: "password"
# is verbatim content but is not itself a token to match on.
_EXACT = _VERBATIM[:7]


def verbatim_tokens(text: str) -> dict[int, set[str]]:
    """Exact strings in `text` that must match exactly or not at all.

    Keyed by which pattern matched, because "same shape, different value"
    is only meaningful WITHIN a kind. A radio frequency is not a wrong
    answer to a serial-number query, it is an unrelated one, and collapsing
    those two cases loses the distinction that makes the penalty safe.
    """
    found: dict[int, set[str]] = {}
    for i, pat in enumerate(_EXACT):
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.setdefault(i, set()).add(m.group(0).strip().upper())
    return found


# Enough to beat any similarity score, because this is not a similarity
# question. Applied as a multiplier, never a filter -- the patterns are
# heuristics and must not be able to veto a genuine hit outright.
IMPOSTOR_PENALTY = 0.10
EXACT_BONUS = 1.5
NO_IDENTIFIER = 0.6


def exact_match_factor(content: str, wanted: dict[int, set[str]]) -> float:
    """Rank an identifier query by EXACTNESS, not similarity.

    'GX-4419' and 'GX-4491' are maximally similar as strings and as vectors
    -- same subword pieces, same shape, near-identical embeddings -- and
    completely different as facts. Retrieval that treats this as a
    similarity problem returns a plausible wrong serial, which is worse
    than returning nothing: the answer looks right.

    Measured on Qwen3-Embedding-8B, "GX-4419" retrieved "Generator serial
    is GX-4491." An impostor is therefore demoted HARDER than an unrelated
    document, because an unrelated document is obviously unhelpful and a
    transposed serial is not.
    """
    if not wanted:
        return 1.0
    have = verbatim_tokens(content)
    impostor = False
    for kind, values in wanted.items():
        mine = have.get(kind)
        if mine and values & mine:
            return EXACT_BONUS           # exact hit wins outright
        if mine:
            impostor = True              # same kind, different value
    return IMPOSTOR_PENALTY if impostor else NO_IDENTIFIER


_CUES = (
    ("position", r"\b(currently|right now|en ?route|at the moment|heading|"
                 r"位置|located at|last seen|grid|moving)\b"),
    ("status", r"\b(is |are |now |today|tonight|open|closed|down|up|"
               r"available|unavailable|on duty|in stock|out of stock|"
               r"working|broken|delayed)\b"),
    ("capacity", r"\b(\d+\s*(beds?|litres?|liters?|kw|kg|tons?|units?|cases?|"
                 r"vials?|seats?)|capacity|holds|rated)\b"),
    ("identity", r"\b(is called|named|speaks|born|founded|located in|"
                 r"runs the|is the (director|head|owner)|serial)\b"),
)


def classify(content: str) -> str:
    """Cheap Tier-0 classifier. A Reasoner overrides this when available.

    Order matters: 'position' beats 'status' beats 'capacity' beats
    'identity', because the shorter-lived reading is the safer default when a
    sentence supports more than one.
    """
    for pat in _VERBATIM:
        if re.search(pat, content, re.IGNORECASE):
            return "verbatim"
    low = content.lower()
    for cls, pattern in _CUES:
        if re.search(pattern, low):
            return cls
    return "unknown"


def credibility(elapsed: float, claim_class: str,
                halflife: float | None = None) -> float:
    """Probability the claim is STILL true. Exponential in the hazard sense.

    Note this is genuinely exponential, unlike retrievability, which is
    power-law. That asymmetry is not an inconsistency: forgetting is a
    retrieval process with a heavy tail, whereas a fact ceasing to be true is
    a memoryless hazard. Different phenomena, different curves.
    """
    h = halflife if halflife is not None else PRIOR_HALFLIFE.get(
        claim_class, PRIOR_HALFLIFE["unknown"])
    if h == float("inf"):
        return 1.0
    if h <= 0:
        return 0.0
    return float(0.5 ** (max(0.0, elapsed) / h))


def staleness(elapsed: float, claim_class: str,
              halflife: float | None = None) -> float:
    return 1.0 - credibility(elapsed, claim_class, halflife)


def fit_halflife(survivals: list[float]) -> float | None:
    """Fit a half-life from observed supersession intervals.

    Uses the MLE for an exponential (mean survival), converted to half-life.
    Right-censoring is ignored deliberately: claims that were never superseded
    contribute no row, which biases the estimate SHORT. In a safety context a
    short estimate over-flags staleness, which is the error worth making.
    """
    if len(survivals) < MIN_EVENTS_TO_FIT:
        return None
    mean = sum(survivals) / len(survivals)
    if mean <= 0:
        return None
    return mean * math.log(2.0)


# ── Admiralty scale (STANAG 2511) ────────────────────────────────────────
# Intelligence practice solved source grading decades ago and no LLM memory
# system carries it. Two axes, because reliability and credibility are
# independent: a reliable source can report something implausible.
RELIABILITY = {"A": 1.00, "B": 0.85, "C": 0.65, "D": 0.45, "E": 0.20, "F": 0.50}
CREDIBILITY = {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.45, 5: 0.20, 6: 0.50}
#                                                         ^ 'cannot be judged'
# F and 6 both mean "cannot be judged" -- deliberately mid-scale, NOT low.
# Treating unknown provenance as unreliable is as wrong as trusting it.


def admiralty_weight(reliability: str, cred: int) -> float:
    return RELIABILITY.get(reliability, 0.5) * CREDIBILITY.get(cred, 0.5)


def corroborate(a: tuple[str, int], b: tuple[str, int]) -> tuple[str, int]:
    """Two INDEPENDENT sources concurring raises credibility, not reliability.

    Every other system either dedupes identical claims (losing the
    corroboration signal entirely) or keeps both (losing the quality
    distinction). With two axes you can do the actual epistemics: keep the
    better source, and promote the credibility because someone else saw it too.
    """
    best_rel = min(a[0], b[0])                    # 'A' < 'B' lexically
    best_cred = min(a[1], b[1])
    if a[0] != b[0] or a[1] != b[1]:
        best_cred = max(1, best_cred - 1)         # independent concurrence
    return best_rel, best_cred
