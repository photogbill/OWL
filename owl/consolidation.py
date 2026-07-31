"""C2, C3, C5 -- structure, sleep, and storing the exception rather than
the rule.

All three are deterministic and model-free. That is not a limitation
accepted reluctantly; MiniRAG's finding was that model-dependent graph
pipelines invert below ~7B, and a consolidation pass that behaves
differently depending on which model was loaded makes "why did you forget
that?" unanswerable.

C2 -- COMMUNITY IDENTITY IS THE HARD PART, not clustering.
Anyone can partition a graph. The problem nobody names is that communities
are recomputed every cycle, so their IDs churn: a community splits, both
halves get fresh IDs, and every composite node derived from the old one now
points at something that no longer exists. Provenance chains break silently
and the store slowly fills with orphans.

The fix is to give a community identity by its persistent CORE rather than
its membership. If the largest overlap with a previous community exceeds a
threshold, it IS that community, grown or shrunk. Splits inherit the name on
the larger side; merges keep the older name. Identity survives change, which
is what identity means.

C3 -- SLEEP PRESSURE, not idle CPU.
Consolidating when the machine happens to be free is scheduling by
accident. Pressure accumulates from unconsolidated writes weighted by how
much interference they create, and the two phases do different jobs:

    NREM   low temperature, high-interference nodes -- separate what is
           being confused. Conservative, safe, runs often.
    REM    high temperature, deliberately DISTANT nodes -- recombine.
           Speculative by construction, so everything it produces is a
           hypothesis with a falsifier and can never present as fact.

C5 -- SCHEMA-DELTA.
Twenty notes saying "generator run-hours logged at 0800" carry one rule and
twenty timestamps. Store the rule once; store what varies. Storage then
scales with novelty rather than volume, which is the right shape for a
memory that runs for years.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# C2
IDENTITY_OVERLAP = 0.5      # Jaccard above which a community IS the old one
MIN_COMMUNITY = 3

# C3
NREM_PRESSURE = 1.0
REM_PRESSURE = 4.0


# ── C2: communities with stable identity ─────────────────────────────────

@dataclass
class Community:
    id: str
    members: frozenset
    generation: int = 0
    lineage: list[str] = field(default_factory=list)   # ids it descends from

    @property
    def size(self) -> int:
        return len(self.members)


def label_propagation(edges: list[tuple[str, str, float]], *,
                      iters: int = 12) -> list[frozenset]:
    """Deterministic community detection over a weighted graph.

    Label propagation, with ties broken by sorted node id rather than
    randomly -- the usual implementation is randomised, which would make
    consolidation non-reproducible and violate A10.
    """
    adj: dict[str, list[tuple[str, float]]] = {}
    for a, b, w in edges:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    label = {n: n for n in sorted(adj)}

    for _ in range(iters):
        changed = False
        for n in sorted(adj):                 # sorted: determinism
            weights: dict[str, float] = {}
            for nb, w in adj[n]:
                weights[label[nb]] = weights.get(label[nb], 0.0) + w
            if not weights:
                continue
            # max weight, then lexicographically smallest label
            best = min(sorted(weights), key=lambda L: (-weights[L], L))
            if best != label[n]:
                label[n] = best
                changed = True
        if not changed:
            break

    groups: dict[str, set] = {}
    for n, L in label.items():
        groups.setdefault(L, set()).add(n)
    out = [frozenset(g) for g in groups.values() if len(g) >= MIN_COMMUNITY]
    return sorted(out, key=lambda g: (-len(g), sorted(g)[0]))


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def reconcile(new_groups: list[frozenset], previous: list[Community], *,
              generation: int, mint) -> list[Community]:
    """Give this cycle's groups the RIGHT names, not fresh ones.

    `mint` produces an id for a genuinely new community. Everything else
    inherits, because a community that lost two members and gained one is
    the same community and the composites derived from it must stay valid.
    """
    out: list[Community] = []
    claimed: set[str] = set()

    # Greedy by best overlap, strongest first, so a split gives the name to
    # the larger surviving side rather than to whichever was scanned first.
    pairs = []
    for i, g in enumerate(new_groups):
        for old in previous:
            j = _jaccard(g, old.members)
            if j >= IDENTITY_OVERLAP:
                pairs.append((j, len(g), i, old))
    pairs.sort(key=lambda p: (-p[0], -p[1], p[3].id))

    taken_new: set[int] = set()
    for j, _, i, old in pairs:
        if i in taken_new or old.id in claimed:
            continue
        taken_new.add(i)
        claimed.add(old.id)
        out.append(Community(old.id, new_groups[i], generation,
                             lineage=old.lineage + [old.id]))

    for i, g in enumerate(new_groups):
        if i not in taken_new:
            # Genuinely new -- but record which old communities it came out
            # of, so a split is traceable rather than looking like creation.
            parents = sorted({o.id for o in previous
                              if _jaccard(g, o.members) > 0.0})
            out.append(Community(mint(), g, generation, lineage=parents))
    return sorted(out, key=lambda c: c.id)


# ── C3: two-phase sleep ──────────────────────────────────────────────────

@dataclass
class SleepPlan:
    phase: str                  # "none" | "nrem" | "rem"
    pressure: float
    targets: list[str] = field(default_factory=list)
    reason: str = ""
    temperature: float = 0.0


def sleep_pressure(unconsolidated: int, interference: float,
                   hours_since: float) -> float:
    """How much consolidation is OWED.

    Not idle CPU. A machine being free says nothing about whether there is
    anything worth doing, and a machine being busy does not mean there
    isn't. Pressure is driven by unconsolidated writes weighted by the
    interference they create, with a mild time term so a quiet store still
    gets tended eventually.
    """
    return round(unconsolidated * (0.5 + interference) + hours_since / 24.0, 4)


def plan_sleep(*, unconsolidated: int, interference: float,
               hours_since: float, confusable: list[str],
               distant: list[str]) -> SleepPlan:
    """Which phase to run, if any.

    REM only fires at high pressure and only after NREM territory is
    covered, because recombining material you have not yet separated is how
    you manufacture confident nonsense out of two things you were already
    confusing.
    """
    p = sleep_pressure(unconsolidated, interference, hours_since)
    if p >= REM_PRESSURE and distant:
        return SleepPlan("rem", p, distant[:8], temperature=0.9,
                         reason="high pressure; recombining distant material "
                                "-- everything produced here is a hypothesis "
                                "with a falsifier, never a fact")
    if p >= NREM_PRESSURE and confusable:
        return SleepPlan("nrem", p, confusable[:16], temperature=0.1,
                         reason="separating memories that are interfering "
                                "with each other")
    return SleepPlan("none", p, [],
                     reason=f"pressure {p:.2f} below the {NREM_PRESSURE} "
                            "threshold; nothing is owed")


# ── C5: schema-delta ─────────────────────────────────────────────────────

_VARIABLE = re.compile(
    r"\b(\d{1,4}[:.]\d{2}|\d+(?:\.\d+)?|"
    r"mon|tue|wed|thu|fri|sat|sun"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)


def schema_of(text: str) -> str:
    """The invariant part of a sentence, with the varying parts blanked."""
    return _VARIABLE.sub("{}", text.strip()).lower()


def deltas_of(text: str) -> list[str]:
    return [m.group(0) for m in _VARIABLE.finditer(text)]


@dataclass
class SchemaGroup:
    schema: str
    members: list[str]              # node ids
    deltas: list[list[str]]
    texts: list[str] = field(default_factory=list)

    @property
    def saved_chars(self) -> int:
        """Bytes the rule-plus-exception form avoids storing.

        Measured against the TEXTS, not the node ids -- the first version
        of this summed the length of identifiers and reported a saving of
        zero for a group that compresses perfectly well. A compression
        scheme that cannot say what it saved is asking to be trusted on a
        claim it never checked, and the check has to measure the right
        thing.
        """
        raw = sum(len(t) for t in self.texts)
        kept = len(self.schema) + sum(len(",".join(d)) for d in self.deltas)
        return max(0, raw - kept)


def find_schemas(items: list[tuple[str, str]], *,
                 min_members: int = 3) -> list[SchemaGroup]:
    """Group by invariant structure. Storage then scales with NOVELTY.

    Only groups where the schema genuinely repeats are returned -- a
    "schema" with one member is just the sentence, and compressing it would
    add indirection for nothing.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for nid, text in items:
        s = schema_of(text)
        if s.count("{}") == 0:
            continue                    # nothing varies; nothing to factor
        groups.setdefault(s, []).append((nid, text))

    out = []
    for s, rows in sorted(groups.items()):
        if len(rows) < min_members:
            continue
        out.append(SchemaGroup(s, [nid for nid, _ in rows],
                               [deltas_of(t) for _, t in rows],
                               [t for _, t in rows]))
    return sorted(out, key=lambda g: -g.saved_chars)
