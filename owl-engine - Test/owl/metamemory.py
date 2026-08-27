"""The Feeling-of-Knowing gate.

Humans answer "capital of a country I've never heard of?" with "no idea" in
under a second, without searching. That triage is metamemory (Koriat), it is
fast, and it is separate from retrieval itself. Almost no agent memory system
has it: they retrieve first and evaluate after, on every query.

OWL triages BEFORE any retrieval or model load. Two payoffs beyond speed:

  DONT_KNOW is a first-class answer. A system that reliably says "I have
  nothing on this" is worth more to an analyst than one that always produces
  something.

  TIP_OF_TONGUE is a diagnostic, not a failure. High familiarity with low
  retrievability and high neighbour conflict is the signature of INTERFERENCE
  -- competing memories blocking each other -- which is the thing most worth
  detecting, because interference (not decay) is what actually kills recall.
"""
from __future__ import annotations

from dataclasses import dataclass

from .protocols import State


@dataclass(frozen=True)
class Signals:
    coverage: float          # fraction of query terms seen anywhere in the store
    best_score: float        # top candidate match strength, 0..1
    best_retrievability: float
    conflict: float          # fraction of candidates bunched near the top
    n_candidates: int
    semantic_density: float = 0.0   # nearest neighbour in embedding space, 0..1
    answer_type: str | None = None  # predicted kind of thing being asked for
    has_answer_type: bool | None = None   # None = unknowable, not negative
    # Dual-process recognition (Yonelinas): familiarity is a fast,
    # context-free "I have seen this"; recollection retrieves the details
    # AND the context they sat in. They dissociate, and conflating them
    # loses a real answer.
    recollection: float = 0.0

    @property
    def familiarity(self) -> float:
        """Density estimate over BOTH channels.

        This must not be lexical-only. A paraphrase query -- "how is the health
        facility powered" against "the clinic generator runs on depot fuel" --
        has zero term overlap and coverage 0.00, so a lexical-only gate returns
        DONT_KNOW no matter how good the semantic match is, and the whole
        embedding tier is silently switched off by a check upstream of it.
        (This was a real bug, caught by the first paraphrase test written.)
        """
        return max(self.coverage, self.semantic_density)


# Thresholds are deliberately explicit and tunable rather than learned:
# a field build must behave the same on every machine, every time.
COVERAGE_FLOOR = 0.25
KNOW_SCORE = 0.55
KNOW_RETRIEVABILITY = 0.50
KNOW_WHERE_SCORE = 0.18
CONFLICT_CEILING = 0.55
RECOLLECTION_FLOOR = 0.20


def triage(s: Signals) -> tuple[State, str]:
    if s.n_candidates == 0 or s.familiarity < COVERAGE_FLOOR:
        return State.DONT_KNOW, (
            f"familiarity {s.familiarity:.2f} below floor {COVERAGE_FLOOR} "
            f"(lexical {s.coverage:.2f}, semantic {s.semantic_density:.2f})"
        )

    familiar = s.best_score >= KNOW_WHERE_SCORE
    retrievable = s.best_retrievability >= KNOW_RETRIEVABILITY
    contested = s.conflict > CONFLICT_CEILING
    recollects = s.recollection >= RECOLLECTION_FLOOR

    if familiar and not retrievable:
        return State.TIP_OF_TONGUE, (
            f"familiar (score {s.best_score:.2f}) but retrievability "
            f"{s.best_retrievability:.2f} is low -- likely decayed or contested"
        )
    if familiar and contested:
        return State.TIP_OF_TONGUE, (
            f"{s.n_candidates} candidates bunched (conflict {s.conflict:.2f}) "
            "-- interference, widen search and schedule a de-interference pass"
        )
    if s.best_score >= KNOW_SCORE and retrievable:
        # Answer-type check, in MiniRAG's spirit: if the question wants a
        # person and the store holds no person at all, a high lexical score is
        # probably topical overlap rather than an answer. Demote rather than
        # reject -- the type predictor is a heuristic and must not be able to
        # veto a genuine hit.
        if s.has_answer_type is False:
            return State.KNOW_WHERE, (
                f"strong match, but no '{s.answer_type}' entity exists to "
                "answer with")
        return State.KNOW, f"direct match, score {s.best_score:.2f}"
    if familiar and not recollects:
        # Dual-process recognition: this is familiarity WITHOUT recollection.
        # KNOW_WHERE claims to know roughly where something lives, which needs
        # context -- an episode with siblings, a period, a real source, links.
        # With none of that, "I have seen this and cannot place it" is the
        # honest answer, and it is neither KNOW_WHERE nor DONT_KNOW.
        return State.FAMILIAR, (
            f"familiar (score {s.best_score:.2f}) but recollection "
            f"{s.recollection:.2f} is thin -- I have seen this and cannot "
            "place it")
    if familiar:
        return State.KNOW_WHERE, f"neighbourhood match, score {s.best_score:.2f}"
    return State.DONT_KNOW, f"best score {s.best_score:.2f} below useful floor"


# A pure noise cutoff, deliberately LOW. Measured on BGE-M3 against a small
# field corpus, the two bands overlap:
#
#     related    0.426 .. 0.691
#     unrelated  0.213 .. 0.514
#
# No threshold separates them. Setting it high (0.55) discarded almost every
# true match; setting it low admits unrelated text. So the floor is not the
# decision -- it only removes obvious noise before the real test.
SEM_FLOOR = 0.40

# How far above the noise a match must rise to count fully.
#
# MEASURED on bge-m3-Q6_K over a 22-document field corpus
# (`validate_embedder.py --calibrate`):
#
#     related margins    0.067  0.166  0.304  0.306
#     unrelated margins         0.106  0.142  0.198
#
# These OVERLAP. Two genuinely related probes sit below the worst unrelated
# one, so no value of this constant separates them -- and the honest
# consequence is that borderline queries land in KNOW_WHERE, or in
# DONT_KNOW when the margin is as thin as 0.067.
#
# That is the correct behaviour, not a limitation to tune away. Forcing a
# confident answer out of evidence that is indistinguishable from noise is
# confabulation, which is the precise failure this whole engine exists to
# prevent. 0.30 sits where it separates what CAN be separated.
MARGIN_SCALE = 0.30
# How much of the decision rests on absolute level rather than margin.
# WHICH SIGNAL CARRIES IS A PROPERTY OF THE ENCODER, measured on the same
# 22-document corpus:
#
#     bge-m3    absolute bands OVERLAP by 0.088   -> margin is the signal
#     Qwen3-8B  absolute bands separate           -> level is the signal
#
# So this is only a default. A calibrated embedder overrides it via its
# `.owlcal.json` sidecar, and the engine reads it from the embedder.
LEVEL_WEIGHT = 0.25


def level_of(similarity: float, floor: float, ceiling: float = 1.0) -> float:
    """Where a cosine sits in the encoder's OWN usable range, 0..1.

    `(sim - floor) / (1.0 - floor)` is wrong whenever an encoder cannot
    reach 1.0, and none of them can. Qwen3-Embedding-8B's best true match
    over a 24-document corpus is 0.531; dividing by `1 - 0.33` maps a good
    match at 0.430 to 0.149, under KNOW_WHERE_SCORE, and a correct answer
    the encoder ranked FIRST comes back DONT_KNOW.

    It stayed hidden while LEVEL_WEIGHT was 0.25 -- margin carried three
    quarters of the decision and margin has no such defect. Calibration
    raised the weight to 0.75 for an encoder where level separates better,
    and immediately exposed it.

    Divide by the range the encoder actually occupies instead.
    """
    span = max(0.05, ceiling - floor)
    return max(0.0, min(1.0, (similarity - floor) / span))


def semantic_density(best_similarity: float, floor: float = SEM_FLOOR,
                     background: float | None = None,
                     level_weight: float = LEVEL_WEIGHT,
                     ceiling: float = 1.0) -> float:
    """Turn a cosine into a 0..1 density using LEVEL and MARGIN.

    Absolute similarity alone cannot decide this, because encoders put
    unrelated short texts at a high baseline and a genuine oblique paraphrase
    can score lower than an accidental collision. What distinguishes them is
    whether the best match RISES ABOVE its background:

        "how is the health facility powered"  best 0.69, rest ~0.35  -> real
        "what is the helicopter tail number"  best 0.51, rest ~0.45  -> noise

    A real match stands out from the pack. An accidental one sits in it. This
    is also how a person decides, and unlike a fixed threshold it transfers
    across encoders without retuning.
    """
    if best_similarity <= floor:
        return 0.0
    level = level_of(best_similarity, floor, ceiling)
    if background is None:
        return level
    margin = min(1.0, max(0.0, best_similarity - background) / MARGIN_SCALE)
    # Blended, not multiplied. Multiplying let a near-floor absolute score
    # veto a large margin, so a genuine oblique paraphrase (0.426, standing
    # 0.18 above its background) scored BELOW an accidental collision (0.514,
    # standing 0.05 above its background) -- exactly backwards.
    return round(min(1.0, level_weight * level
                     + (1.0 - level_weight) * margin), 4)


def recollection_score(*, has_episode: bool, has_period: bool,
                       n_neighbours: int, has_provenance: bool,
                       decontextualised: bool) -> float:
    """How much CONTEXT surrounds the best match.

    Familiarity is cheap -- an embedding is near the query. Recollection is
    what lets you PLACE a memory: which episode it sat in, what else was
    around it, where it came from, whether it reads standalone.

    `has_episode` must mean "the episode has other members". Every
    observation is trivially assigned one, so counting bare membership made
    this 0.30 for free and FAMILIAR could never fire -- the state existed
    and was unreachable.
    """
    score = 0.0
    score += 0.30 if has_episode else 0.0
    score += 0.15 if has_period else 0.0
    score += 0.25 if has_provenance else 0.0
    score += 0.15 if decontextualised else 0.0
    score += min(0.15, 0.05 * n_neighbours)
    return round(min(1.0, score), 3)


def conflict_ratio(scores: list[float], band: float = 0.15) -> float:
    """Fraction of candidates within `band` of the best. High = confusable."""
    if len(scores) < 2:
        return 0.0
    best = max(scores)
    if best <= 0:
        return 0.0
    near = sum(1 for s in scores if best - s <= band * best)
    return (near - 1) / (len(scores) - 1)
