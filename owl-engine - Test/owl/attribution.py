"""Who said it, and does it hold up?

Three pieces that form one self-maintaining loop:

  independence -- corroboration counts independent ORIGINS, not documents
  claimants    -- the claim is separate from the person making it
  commitments  -- promises resolve, and the outcome revalues the claimant

The loop matters more than any single piece. A broken promise lowers a
claimant's reliability; lowered reliability triggers revaluation of every
proposition resting on their word. Nobody closes this.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

DAY = 86400.0

# ── source independence ──────────────────────────────────────────────────

_SCHEME = re.compile(r"^([a-z][a-z0-9+.-]*)://", re.I)
_HOSTISH = re.compile(r"^(?:[a-z]+://)?([^/\\:]+)", re.I)
BATCH_WINDOW = 300.0        # ingests within 5 minutes are one batch


def origin_key(source_ref: str) -> str:
    """Coarse origin for a source reference.

    Deliberately blunt: everything under one host, one archive, or one
    directory is treated as ONE origin until proven otherwise. Over-merging
    costs a little corroboration credit; under-merging lets an attacker
    manufacture consensus by publishing the same claim forty times. The
    asymmetry says which error to make.
    """
    s = (source_ref or "").strip().lower()
    if not s:
        return "unknown"

    # Colon-delimited refs like conv:ahmed:14 identify a PERSON, and two
    # different people are two different origins. Collapsing them all to
    # "conv" would make every conversation one source and destroy exactly
    # the corroboration signal this function exists to measure.
    if ":" in s and not _SCHEME.match(s):
        parts = [p for p in s.split(":") if p]
        return ":".join(parts[:2]) if len(parts) > 1 else parts[0]

    m = _HOSTISH.match(s)
    head = m.group(1) if m else s
    if _SCHEME.match(s) and s.startswith("file:"):
        # file://a/b/c.pdf -> collapse to the directory
        path = s.split("://", 1)[1]
        parts = [p for p in re.split(r"[/\\]", path) if p]
        return "file:" + "/".join(parts[:-1]) if len(parts) > 1 else "file:"
    if "/" in s or "\\" in s:
        parts = [p for p in re.split(r"[/\\]", s) if p]
        if len(parts) > 1:
            return "/".join(parts[:-1])
    return head


def independence(clusters: list[str]) -> int:
    """How many genuinely distinct origins are represented here."""
    return len({c for c in clusters if c})


def corroboration_weight(n_independent: int) -> float:
    """Diminishing returns, and no credit at all for a single origin.

    One source is one source however many files it produced. Two independent
    sources is the big jump. Beyond about four, extra agreement adds little --
    if they were going to disagree, they would have by then.
    """
    if n_independent <= 1:
        return 0.0
    return round(min(1.0, 0.5 + 0.2 * (n_independent - 2)), 3)


# ── claimants ────────────────────────────────────────────────────────────

def canonical_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an|dr|mr|mrs|ms|prof|sr|jr|capt|lt)\.?\s+", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def proposition_hash(text: str) -> str:
    """Stable identity for a proposition, so two people asserting the same
    thing are recognisable as corroboration rather than two facts."""
    from .lexical import tokenize
    toks = sorted(set(tokenize(text)))
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Record:
    name: str
    claims_made: int
    confirmed: int
    refuted: int
    kept: int
    broken: int

    @property
    def resolved(self) -> int:
        return self.confirmed + self.refuted + self.kept + self.broken

    @property
    def accuracy(self) -> float:
        """Laplace-smoothed. With no track record the answer is 0.5, which is
        'unknown' -- not 'bad'. Treating an unknown source as unreliable is as
        wrong as trusting it."""
        good = self.confirmed + self.kept
        bad = self.refuted + self.broken
        return round((good + 1) / (good + bad + 2), 3)

    @property
    def grade(self) -> str:
        """Admiralty reliability derived from outcomes.

        Stays at F ('cannot be judged') until there is enough history to
        judge. A confident grade from two data points is worse than no grade.
        """
        if self.resolved < 3:
            return "F"
        a = self.accuracy
        if a >= 0.90:
            return "A"
        if a >= 0.75:
            return "B"
        if a >= 0.60:
            return "C"
        if a >= 0.40:
            return "D"
        return "E"

    def describe(self) -> str:
        if self.resolved < 3:
            return f"{self.name}: too little history to judge " \
                   f"({self.resolved} resolved)"
        return (f"{self.name}: {self.accuracy:.0%} accurate over "
                f"{self.resolved} resolved ({self.broken} promises broken) "
                f"-> grade {self.grade}")
