"""H2 cold-start honesty, H3 the nightly budget, H4 predictive unification.

H2 -- EVERY MEMORY SYSTEM'S FIRST WEEK IS ITS WORST, AND ALL OF THEM PRETEND
OTHERWISE.

A store with three days of history answers "what do we know about the depot?"
in exactly the same tone as one with three years. The user cannot tell the
difference, so they calibrate their trust on the confident-sounding early
answers and get burned, or they never trust it and the good answers are
wasted too.

The fix is not better retrieval. It is saying so:

    "I have 3 days on this project, 12 memories, 2 sources. Treat gaps as
     ignorance rather than absence."

Trust is won by being right about your own limits. This pairs exactly with
A7: a young store has high IGNORANCE mass, and reporting maturity is the
same claim at the level of the whole store rather than one node.

H3 -- ONE BOUNDED CALL A NIGHT.

Tier 2 needs a model. Requiring a separate API key excludes exactly the
users OWL is for. The elegant route is one bounded call per night through
whatever subscription the host already has -- so what OWL owns is not the
credential but the CONTRACT: a hard quota, a deterministic decision about
whether tonight's call is worth spending, and a refusal to silently exceed
either. The host supplies the model; this decides whether to ask.

H4 -- ONE OBJECTIVE, OR AN HONEST ACCOUNT OF WHY NOT.

Predictive processing says encoding priority, attention, consolidation
scheduling and dreaming are all the same thing: minimise prediction error.
That would replace five separately-tuned subsystems with one quantity, and
it is a genuinely attractive claim.

It is also exactly the kind of claim that gets asserted rather than checked.
So `unification_report()` computes each subsystem's signal BOTH ways --
from the existing hand-tuned rule and from prediction error alone -- and
reports the correlation. Where they agree, the unification is real and the
tuned constant can go. Where they diverge, that is reported as a divergence
rather than resolved in favour of the prettier theory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DAY = 86400.0


# ── H2: cold-start honesty ───────────────────────────────────────────────

@dataclass
class Maturity:
    days: float
    memories: int
    sources: int
    coverage: float          # 0..1 confidence that gaps mean absence

    @property
    def young(self) -> bool:
        return self.coverage < 0.5

    def contract(self) -> str:
        """What to tell the user before they trust an answer."""
        age = (f"{self.days:.0f} day{'s' if self.days != 1 else ''}"
               if self.days < 60 else f"{self.days / 30:.0f} months")
        base = (f"{age} of history, {self.memories} memories, "
                f"{self.sources} source{'s' if self.sources != 1 else ''}")
        if self.coverage < 0.25:
            return (f"{base}. This store is NEW -- treat every gap as "
                    "ignorance, not absence. 'I don't know' here mostly "
                    "means 'nobody has told me yet'.")
        if self.coverage < 0.5:
            return (f"{base}. Still thin. A 'not found' is weak evidence of "
                    "absence; check elsewhere before concluding.")
        if self.coverage < 0.8:
            return (f"{base}. Reasonable coverage. Gaps in well-covered "
                    "areas are meaningful; gaps at the edges are not.")
        return (f"{base}. Mature. A recorded absence here is real evidence "
                "of absence.")


def assess_maturity(*, days: float, memories: int, sources: int,
                    partitions: int = 1) -> Maturity:
    """How much a gap in this store is worth believing.

    Saturating in all three terms, because maturity is not a count. A store
    with 10,000 memories from one source over one day is not mature, and
    the geometric mean punishes exactly that -- a single weak dimension
    caps the whole score, where an average would let volume paper over a
    one-day history.
    """
    def sat(x, half):
        return x / (x + half) if x > 0 else 0.0

    t = sat(days, 21.0)                    # three weeks to half
    v = sat(memories, 200.0)
    s = sat(sources, 8.0)
    coverage = (t * v * s) ** (1 / 3) if min(t, v, s) > 0 else 0.0
    return Maturity(days, memories, sources, round(coverage, 4))


# ── H3: the nightly budget ───────────────────────────────────────────────

@dataclass
class NightlyBudget:
    calls_per_night: int = 1
    max_input_chars: int = 24_000
    spent_tonight: int = 0

    @property
    def exhausted(self) -> bool:
        return self.spent_tonight >= self.calls_per_night

    def should_run(self, *, sleep_pressure: float, maturity: Maturity,
                   min_pressure: float = 4.0) -> tuple[bool, str]:
        """Is tonight's call worth spending? Deterministic.

        Three refusals, in order of how often they fire:

        1. Budget spent. A quota that can be exceeded under interesting
           circumstances is not a quota.
        2. Nothing owed. Running a model because it is midnight is
           scheduling by clock rather than by need.
        3. Too new to generalise. A model asked to find patterns in eleven
           memories will find some, and they will be noise wearing the
           shape of insight -- the most expensive failure available here,
           because the output is a hypothesis that then propagates.
        """
        if self.exhausted:
            return False, (f"budget spent ({self.spent_tonight}/"
                           f"{self.calls_per_night}); a quota that bends is "
                           "not a quota")
        if sleep_pressure < min_pressure:
            return False, (f"pressure {sleep_pressure:.2f} < {min_pressure}; "
                           "nothing is owed, and running because it is "
                           "midnight is scheduling by clock rather than need")
        if maturity.coverage < 0.25:
            return False, (f"store too new (coverage {maturity.coverage:.2f}) "
                           "-- a model asked to generalise from this little "
                           "will find patterns that are noise")
        return True, (f"pressure {sleep_pressure:.2f}, coverage "
                      f"{maturity.coverage:.2f}: worth one call")

    def spend(self) -> None:
        self.spent_tonight += 1

    def reset(self) -> None:
        self.spent_tonight = 0


# ── H4: predictive-processing unification ────────────────────────────────

def prediction_error(*, observed_novelty: float, expected_novelty: float
                     ) -> float:
    """How much this violated expectation. The single proposed quantity."""
    return max(0.0, min(1.0, abs(observed_novelty - expected_novelty)))


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else 0.0


def unification_report(samples: list[dict]) -> dict:
    """Does ONE objective actually reproduce the five tuned subsystems?

    Each sample carries the prediction error and what each subsystem's
    existing hand-tuned rule produced. If prediction error correlates
    strongly with a subsystem, that subsystem is derivable and its constants
    are redundant. If it does not, the unification does not hold there --
    and that gets reported, not explained away.

    Written this way on purpose. "One objective behind everything" is an
    elegant claim that would be very easy to assert and never check, and
    this project has already caught itself doing a version of that twice.
    """
    if len(samples) < 3:
        return {"n": len(samples),
                "verdict": "insufficient samples to say anything"}
    pe = [s["prediction_error"] for s in samples]
    subsystems = ("encoding_priority", "attention", "consolidation",
                  "dreaming", "forgetting")
    found = {}
    for key in subsystems:
        if key not in samples[0]:
            continue
        r = _corr(pe, [s[key] for s in samples])
        found[key] = {
            "correlation": r,
            "verdict": ("derivable from prediction error alone"
                        if r >= 0.85 else
                        "mostly derivable; the residual is doing real work"
                        if r >= 0.6 else
                        "NOT unified -- this subsystem is measuring "
                        "something prediction error does not capture"),
        }
    strong = [k for k, v in found.items() if v["correlation"] >= 0.85]
    weak = [k for k, v in found.items() if v["correlation"] < 0.6]
    return {
        "n": len(samples), "subsystems": found,
        "unified": strong, "not_unified": weak,
        "verdict": (
            f"{len(strong)}/{len(found)} subsystems reduce to prediction "
            "error. " + (
                "The unification does not hold for " + ", ".join(weak) +
                " -- keeping those tuned separately is the honest outcome, "
                "not a failure to simplify." if weak else
                "The unification holds across all of them.")),
    }
