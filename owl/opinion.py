"""A7 -- second-order uncertainty. Belief, disbelief, and IGNORANCE.

A scalar confidence of 0.5 conflates two states that call for opposite
actions:

    no evidence at all      b=0.00  d=0.00  u=1.00   -> go and look
    strong evidence, both   b=0.45  d=0.45  u=0.10   -> stop looking, decide

Every memory system in the field reports 0.5 for both. The difference is
frequently the entire answer, and it is invisible in a point estimate.

This is subjective logic (Jøsang): an opinion is a mass distribution over
{believe, disbelieve, don't know} plus a base rate `a` for how to bet when
you don't know. It is equivalent to a Beta posterior, which is the useful
part -- evidence counts map onto it directly, so OWL can derive opinions
from what it ALREADY records rather than needing a schema migration:

    belief mass      <- independent corroborating ORIGINS (A5's math)
    disbelief mass   <- counter-evidence and supersession (B6's math)
    ignorance        <- whatever neither has claimed

RETROFIT STRATEGY, chosen deliberately. Scalar `confidence` stays exactly
where it is and keeps its meaning; `expectation` reproduces it. Nothing
downstream had to change, nothing needed migrating, and an opinion can be
computed for any node in any existing store. The plan warned that
retrofitting A7 mid-build would be painful; it is only painful if you try
to *replace* the scalar rather than derive it.

MONOTONICITY GENERALISES, which is the real reason this is safe to add:

    belief(child)     <=  min(belief(parents))
    ignorance(child)  >=  max(ignorance(parents))

The second is the interesting one. A conclusion cannot be less ignorant
than the evidence it rests on -- inference does not create knowledge, it
only moves existing mass around. That is exactly the scalar invariant OWL
already enforces, one level up.
"""
from __future__ import annotations

from dataclasses import dataclass

# How much prior mass sits in ignorance before any evidence arrives. Two is
# the standard uninformative Beta(1,1) prior expressed as evidence weight:
# one imaginary success and one imaginary failure.
PRIOR_WEIGHT = 2.0


@dataclass(frozen=True)
class Opinion:
    belief: float = 0.0
    disbelief: float = 0.0
    uncertainty: float = 1.0
    base_rate: float = 0.5          # how to bet when you don't know

    def __post_init__(self) -> None:
        total = self.belief + self.disbelief + self.uncertainty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"opinion masses must sum to 1, got {total:.6f}. This is not "
                "a rounding nicety -- an opinion that does not sum to 1 is "
                "claiming mass that came from nowhere.")
        if min(self.belief, self.disbelief, self.uncertainty) < -1e-9:
            raise ValueError("mass cannot be negative")

    # ── the scalar everything downstream still uses ──────────────────
    @property
    def expectation(self) -> float:
        """Projected probability. THIS is `confidence` as previously meant.

        Ignorance is resolved by the base rate rather than ignored, which is
        why a node with no evidence projects to 0.5 and a contested one also
        projects near 0.5 -- and why the point estimate alone cannot tell
        you which you are looking at.
        """
        return round(self.belief + self.base_rate * self.uncertainty, 4)

    @property
    def confidence_in_the_estimate(self) -> float:
        """How much the expectation should be trusted. 1 - u."""
        return round(1.0 - self.uncertainty, 4)

    @property
    def contested(self) -> bool:
        """Real evidence on both sides, rather than an absence of it."""
        return (self.belief > 0.2 and self.disbelief > 0.2
                and self.uncertainty < 0.5)

    @property
    def vacuous(self) -> bool:
        """Nothing is actually known. The 'go and look' state."""
        return self.uncertainty > 0.8

    @property
    def verdict(self) -> str:
        if self.vacuous:
            return "no evidence either way -- go and look"
        if self.contested:
            return "evidence on both sides -- decide under uncertainty, "\
                   "more searching will not resolve it"
        if self.belief > self.disbelief:
            return f"supported (b={self.belief:.2f}, u={self.uncertainty:.2f})"
        return f"disputed (d={self.disbelief:.2f}, u={self.uncertainty:.2f})"

    def as_dict(self) -> dict:
        return {"belief": round(self.belief, 4),
                "disbelief": round(self.disbelief, 4),
                "uncertainty": round(self.uncertainty, 4),
                "expectation": self.expectation,
                "contested": self.contested, "vacuous": self.vacuous,
                "verdict": self.verdict}


def from_evidence(supporting: float, opposing: float, *,
                  base_rate: float = 0.5,
                  prior: float = PRIOR_WEIGHT) -> Opinion:
    """Beta posterior -> opinion. The bridge from counts to mass.

    `supporting` and `opposing` are WEIGHTS, not raw document counts --
    forty files from one upstream source are one piece of evidence, and
    passing forty here would be the source-flooding attack wearing a
    different hat.
    """
    s, o = max(0.0, supporting), max(0.0, opposing)
    total = s + o + prior
    return Opinion(belief=s / total, disbelief=o / total,
                   uncertainty=prior / total, base_rate=base_rate)


def fuse(a: Opinion, b: Opinion) -> Opinion:
    """Cumulative fusion of two INDEPENDENT opinions (Jøsang's ⊕).

    Independence is the precondition and it is not decorative: fusing two
    opinions derived from the same origin manufactures certainty out of one
    observation. Callers must establish independence first -- which is what
    `independent_sources()` is for.
    """
    ua, ub = a.uncertainty, b.uncertainty
    if ua < 1e-9 and ub < 1e-9:
        # Both dogmatic. Average, because cumulative fusion is undefined
        # here and silently returning one of them would be arbitrary.
        return Opinion((a.belief + b.belief) / 2,
                       (a.disbelief + b.disbelief) / 2, 0.0,
                       (a.base_rate + b.base_rate) / 2)
    denom = ua + ub - ua * ub
    return Opinion(
        belief=(a.belief * ub + b.belief * ua) / denom,
        disbelief=(a.disbelief * ub + b.disbelief * ua) / denom,
        uncertainty=(ua * ub) / denom,
        base_rate=(a.base_rate + b.base_rate) / 2,
    )


def discount(source_trust: float, op: Opinion) -> Opinion:
    """Weigh an opinion by how much the source is trusted (Jøsang's ⊗).

    A perfectly confident claim from a source you half-trust is a half-
    confident claim -- and the mass that leaves belief goes to IGNORANCE,
    not to disbelief. Distrusting a source does not mean the opposite is
    true; it means you have learned less than you thought. Getting this
    backwards is how a low-reliability report becomes evidence against its
    own content.
    """
    t = max(0.0, min(1.0, source_trust))
    return Opinion(belief=t * op.belief, disbelief=t * op.disbelief,
                   uncertainty=1.0 - t * (op.belief + op.disbelief),
                   base_rate=op.base_rate)


def derive_opinion(parents: list[Opinion], *,
                   own: Opinion | None = None) -> Opinion:
    """Clamp a conclusion to what its evidence supports.

    The scalar rule (`confidence <= min(parents)`) generalised: belief
    cannot exceed the least-believing parent, and ignorance cannot fall
    below the most-ignorant one. Inference redistributes mass; it does not
    create it.
    """
    if not parents:
        return own or Opinion()
    cap_b = min(p.belief for p in parents)
    floor_u = max(p.uncertainty for p in parents)
    start = own or Opinion(belief=cap_b, disbelief=0.0,
                           uncertainty=1.0 - cap_b)
    b = min(start.belief, cap_b)
    u = max(start.uncertainty, floor_u)
    if b + u > 1.0:                 # ignorance wins; it is the safe direction
        b = max(0.0, 1.0 - u)
    return Opinion(belief=b, disbelief=max(0.0, 1.0 - b - u), uncertainty=u,
                   base_rate=start.base_rate)
