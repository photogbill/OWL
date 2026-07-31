"""FSRS (DSR) memory model — Difficulty, Stability, Retrievability.

Replaces the naive `S0*exp(-lambda*t) + a*c` from the v1 brief, which
(a) used an exponential where human forgetting is power-law, (b) had an
additive access term that made frequently-touched junk immortal, and
(c) could not represent the spacing effect at all.

Stability is Bjork & Bjork's *storage strength*: it does not decrease.
Retrievability is *retrieval strength*: it decays. Forgetting happens to
retrievability. Nothing is deleted because retrievability fell.
"""
from __future__ import annotations

import math

DAY = 86400.0

# FSRS power forgetting curve: R(t) = (1 + F * t/S) ** C
_F = 19.0 / 81.0
_C = -0.5

# Initial stability by grade (days). Grades: 1 fail, 2 hard, 3 good, 4 easy.
_S0 = (0.40, 1.18, 3.17, 15.69)
_D0_BASE, _D0_SLOPE = 7.19, 1.30
_W_MEAN_REVERT = 0.07
_W_DIFF_DELTA = 1.05
_S_FACTOR = 2.0
_S_DECAY = -0.4
_S_GAIN = 1.6


def retrievability(elapsed_seconds: float, stability_days: float) -> float:
    """Probability of successful recall now. 1.0 at t=0, decaying power-law."""
    if stability_days <= 0:
        return 0.0
    t = max(0.0, elapsed_seconds) / DAY
    return float((1.0 + _F * t / stability_days) ** _C)


def initial_state(grade: int = 3) -> tuple[float, float]:
    """(stability_days, difficulty) for a brand-new memory."""
    g = _clamp_grade(grade)
    stability = _S0[g - 1]
    difficulty = _clamp(_D0_BASE - _D0_SLOPE * (g - 3), 1.0, 10.0)
    return stability, difficulty


def review(
    stability: float, difficulty: float, elapsed_seconds: float, grade: int = 3
) -> tuple[float, float]:
    """Update (S, D) after a retrieval attempt.

    A successful recall at LOW retrievability produces a much larger stability
    gain than one at high retrievability — that is the spacing effect, and it
    falls out of the `(1 - r)` term rather than being bolted on.
    """
    g = _clamp_grade(grade)
    r = retrievability(elapsed_seconds, stability)

    # Difficulty: mean-reverting toward the easy anchor.
    d_target = difficulty - _W_DIFF_DELTA * (g - 3)
    d_easy, _ = initial_state(4)
    difficulty = _clamp(
        _W_MEAN_REVERT * (_D0_BASE - _D0_SLOPE) + (1 - _W_MEAN_REVERT) * d_target,
        1.0, 10.0,
    )
    del d_easy

    if g == 1:                                   # lapse: storage strength persists,
        new_s = max(0.1, stability * 0.3)        # retrieval strength collapses
    else:
        hard_penalty = 0.85 if g == 2 else 1.0
        easy_bonus = 1.25 if g == 4 else 1.0
        gain = (
            1.0
            + _S_GAIN
            * math.exp(_S_FACTOR * (1.0 - difficulty / 10.0))
            * (stability ** _S_DECAY)
            * (math.e ** (1.0 - r) - 1.0)
            * hard_penalty
            * easy_bonus
        )
        new_s = stability * max(1.0, gain)
    return min(new_s, 36500.0), difficulty


def actr_activation(access_log: list[float], now: float, d: float = 0.5) -> float:
    """ACT-R base-level activation: B = ln(sum_j t_j^-d).

    Offered as an alternative to FSRS for callers who want Anderson &
    Schooler's rational-analysis framing (activation as the log-odds that an
    item is needed right now). Sums a power-law decay over EVERY access, so
    the spacing effect is native here too.
    """
    total = 0.0
    for ts in access_log:
        dt = max(1.0, now - ts) / DAY
        total += dt ** (-d)
    return math.log(total) if total > 0 else float("-inf")


def infer_acquisition_cost(*, elapsed_seconds: float = 0.0,
                           tool_calls: int = 0, sources_consulted: int = 0,
                           human_minutes: float = 0.0,
                           travel: bool = False) -> float:
    """C7 -- what did it COST to learn this, when nobody said.

    `acquisition_cost` can be supplied by the host, and usually isn't --
    which left the most interesting retention signal in the engine sitting
    at its default. This estimates it from what the host already knows it
    did.

    The framing is the point. Everyone models memory as storage, so the
    question they ask about forgetting is "how often was this used?". But
    memory is an INVESTMENT, and the right question is **"what would it
    cost me to get this back?"** A phone number found in a filename and a
    phone number obtained by driving to a district office and waiting two
    hours are not interchangeable, however equally they are accessed.

    Deliberately saturating rather than linear: the difference between one
    source and four matters a great deal, between forty and fifty almost
    nothing, and a linear scale would let one expensive outlier flatten
    everything else into indistinguishable cheapness.
    """
    def sat(x: float, half: float) -> float:
        """0 at zero, 0.5 at `half`, asymptotic to 1."""
        return x / (x + half) if x > 0 else 0.0

    # Weights, and the reason a person's time carries nearly half of it:
    # machine time is cheap to spend again. An hour of wall clock and twenty
    # tool calls can be re-run tonight; an hour of somebody's attention
    # cannot, and if the memory is lost that hour is spent a second time.
    # The first version of this scored 60 human-minutes BELOW an hour of
    # machine time, which contradicted the sentence above it -- a docstring
    # and its constants disagreeing, caught by the test that asserted the
    # docstring.
    cost = (
        0.15 * sat(elapsed_seconds, 900.0)      # 15 min of wall clock
        + 0.15 * sat(tool_calls, 4.0)
        + 0.25 * sat(sources_consulted, 3.0)    # canvassing is expensive
        + 0.45 * sat(human_minutes, 20.0)       # a person's time dominates
    )
    if travel:
        # Going somewhere is a step change, not a bigger number. Anything
        # that required physical presence is expensive by definition and
        # cannot be re-acquired from a desk.
        cost = max(cost, 0.75)
    return round(max(0.0, min(1.0, cost)), 4)


def salience(
    *,
    stability: float,
    difficulty: float,
    elapsed: float,
    surprise: float = 0.5,
    open_loop: bool = False,
    acquisition_cost: float = 0.0,
    criticality: float = 0.0,
) -> float:
    """Ranking score. Surprise raises PRIORITY. It must never raise confidence.

    (Flashbulb memories are held with very high confidence and decay in
    accuracy at ordinary rates -- Talarico & Rubin 2003. Priority and
    confidence are two variables; conflating them builds a system most
    certain about exactly what it should doubt.)
    """
    r = retrievability(elapsed, stability)
    score = r * (1.0 + 0.5 * surprise)
    if open_loop:
        score *= 1.35                     # Zeigarnik: unfinished stays accessible
    # Everyone models memory as storage. It is an INVESTMENT. A fact that
    # cost a three-day trip is not interchangeable with one that cost a
    # glance at a filename, and the right question for forgetting is not
    # "how often was this used" but "what would it cost me to get it back".
    score *= 1.0 + 0.6 * max(0.0, min(1.0, acquisition_cost))
    # And a memory nothing depends on is safe to let go; a load-bearing one
    # is not, however rarely it is touched.
    score *= 1.0 + 0.5 * max(0.0, min(1.0, criticality))
    return score


def tier_for(retrievability_now: float) -> str:
    if retrievability_now >= 0.60:
        return "hot"
    if retrievability_now >= 0.25:
        return "warm"
    if retrievability_now >= 0.05:
        return "cold"
    return "pruned"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clamp_grade(g: int) -> int:
    return max(1, min(4, int(g)))
