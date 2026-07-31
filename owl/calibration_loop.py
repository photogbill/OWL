"""E3 -- did the memory actually help, and was the confidence honest?

The `calibration` table sat in the schema with nothing writing to it. That is
worse than not having it: it looks like the question is being tracked when
nothing is. This wires it, and the point is narrower than "learning".

**Brier score, per producer and per claim kind.** A producer that says 0.9
and is right 90% of the time is calibrated. One that says 0.9 and is right
55% of the time is confident, which is a different and much more dangerous
thing -- and it is exactly the failure OWL's whole pitch is about. Nothing
in the field scores its own confidence.

Three things this deliberately does NOT do:

  * **It does not change retrieval.** A feedback loop that quietly reweights
    what you see, based on outcomes you may have recorded carelessly, is a
    system that drifts somewhere you cannot audit. Calibration is REPORTED.
    Acting on it is a decision a human makes with the numbers in front of
    them.

  * **It does not reward being useful.** Retrieval count is logged, but a
    memory retrieved often is not thereby true. Conflating "used" with
    "correct" is how a popularity score gets mistaken for an accuracy one.

  * **It does not pretend small samples mean anything.** Under ten outcomes
    a producer gets "insufficient", not a number. A Brier score computed
    from three events is theatre.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Calibration:
    producer: str
    claim_kind: str
    n: int
    mean_confidence: float
    accuracy: float
    brier: float
    verdict: str

    @property
    def overconfident(self) -> bool:
        return self.mean_confidence - self.accuracy > 0.10


MIN_SAMPLE = 10


def score(rows: list[dict]) -> list[Calibration]:
    """Brier score per (producer, claim_kind).

    Brier is mean squared error between stated confidence and outcome, so
    lower is better and 0.25 is what you get by always saying 0.5. Chosen
    over accuracy because it punishes confident wrongness far harder than
    hedged wrongness, which is the asymmetry that matters when the output
    feeds a decision.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("outcome") is None:
            continue                        # still open; not evidence yet
        groups.setdefault((r["producer"], r["claim_kind"]), []).append(r)

    out: list[Calibration] = []
    for (producer, kind), rs in sorted(groups.items()):
        n = len(rs)
        conf = sum(float(r["confidence"]) for r in rs) / n
        acc = sum(int(bool(r["outcome"])) for r in rs) / n
        brier = sum((float(r["confidence"]) - int(bool(r["outcome"]))) ** 2
                    for r in rs) / n
        if n < MIN_SAMPLE:
            verdict = f"insufficient ({n}/{MIN_SAMPLE}) -- no claim made"
        elif conf - acc > 0.10:
            verdict = (f"OVERCONFIDENT: says {conf:.2f}, right {acc:.2f} "
                       "-- the dangerous direction")
        elif acc - conf > 0.15:
            verdict = (f"underconfident: says {conf:.2f}, right {acc:.2f} "
                       "-- hedging more than the record justifies")
        else:
            verdict = f"calibrated: says {conf:.2f}, right {acc:.2f}"
        out.append(Calibration(producer, kind, n, round(conf, 3),
                               round(acc, 3), round(brier, 4), verdict))
    out.sort(key=lambda c: -c.brier)
    return out


def reliability_curve(rows: list[dict], bins: int = 5) -> list[dict]:
    """The plottable form: stated confidence vs observed frequency.

    A perfectly calibrated producer traces the diagonal. The shape of the
    departure says more than the score does -- a curve that sags only at the
    top means "trustworthy except when certain", which is a specific and
    fixable failure.
    """
    buckets: list[list[dict]] = [[] for _ in range(bins)]
    for r in rows:
        if r.get("outcome") is None:
            continue
        c = min(0.999, max(0.0, float(r["confidence"])))
        buckets[int(c * bins)].append(r)
    curve = []
    for i, b in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if not b:
            curve.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0,
                          "stated": None, "observed": None})
            continue
        curve.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": len(b),
            "stated": round(sum(float(r["confidence"]) for r in b) / len(b), 3),
            "observed": round(
                sum(int(bool(r["outcome"])) for r in b) / len(b), 3),
        })
    return curve
