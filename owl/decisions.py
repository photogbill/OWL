"""The forward direction — what is this memory holding up?

`why()` walks backward: how do I know this? Nothing in the field walks
forward: what do I believe because of this, and what did I DO about it?

That asymmetry is the gap this module fills. Bitemporality says what changed.
Provenance says why you believed it. Divergence says the operator is out of
date. This says *what to go and fix.*

Three pieces, all pure graph work with no model in the loop:

    decisions   -- first-class nodes with edges to the memories they rest on
    criticality -- which memories, if wrong, invalidate the most
    impact      -- when a basis moves, which decisions are now standing on air
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Cause(str, Enum):
    SUPERSEDED = "superseded"      # a newer claim replaced the basis
    DISCREDITED = "discredited"    # the source was revalued downward
    STALE = "stale"                # credibility decayed past threshold
    RETRACTED = "retracted"        # explicitly withdrawn


class Status(str, Enum):
    STANDING = "standing"
    REVISIT = "revisit"
    REAFFIRMED = "reaffirmed"
    REVERSED = "reversed"
    EXECUTED = "executed"          # carried out; no longer reversible


@dataclass(frozen=True)
class Impact:
    decision_id: str
    statement: str
    basis_node: str
    cause: Cause
    severity: float
    reversible: bool
    detected_at: float
    note: str = ""
    impact_id: str = ""     # pass to resolve_impact(); empty until persisted

    @property
    def urgent(self) -> bool:
        """Reversible AND high severity. The only combination worth an alarm.

        A decision you can still change is worth interrupting someone about.
        One already executed is worth logging, not alarming over -- and a
        system that alarms about things you cannot change teaches people to
        ignore alarms.
        """
        return self.reversible and self.severity >= 0.5


def severity(*, criticality: float, weight: float, cause: Cause,
             reversible: bool) -> float:
    """How much it matters that this basis moved.

    Criticality dominates: a basis carrying many conclusions moving is far
    worse than a peripheral one moving. Reversibility raises severity rather
    than lowering it -- an actionable problem outranks an unfixable one for
    the purpose of getting someone's attention.
    """
    base = {
        Cause.RETRACTED: 1.00,
        Cause.DISCREDITED: 0.90,
        Cause.SUPERSEDED: 0.75,
        Cause.STALE: 0.45,
    }[cause]
    s = base * (0.4 + 0.6 * min(1.0, criticality)) * max(0.2, min(1.0, weight))
    if reversible:
        s = min(1.0, s * 1.25)
    return round(s, 4)


# ── criticality ──────────────────────────────────────────────────────────

DAMPING = 0.85
ITERATIONS = 20
DECISION_BOOST = 2.0


def criticality(edges: list[tuple[str, str]], decision_edges: list[tuple[str, str]],
                nodes: set[str] | None = None) -> dict[str, float]:
    """Reverse PageRank over the derivation graph.

    `edges` are (child, parent) -- a child DEPENDS ON its parent, so rank
    flows from dependents back to what they rest on. `decision_edges` are
    (decision_id, basis_node); decisions inject extra mass because a memory
    holding up an action matters more than one holding up a summary.

    Deliberately not normalised to sum to 1: the absolute magnitude is
    meaningless, the ordering is what gets used, and a store-size-dependent
    normaliser makes scores incomparable across time.
    """
    parents: dict[str, list[str]] = {}
    all_nodes: set[str] = set(nodes or ())
    for child, parent in edges:
        parents.setdefault(child, []).append(parent)
        all_nodes.add(child)
        all_nodes.add(parent)
    for did, basis in decision_edges:
        parents.setdefault(did, []).append(basis)
        all_nodes.add(did)
        all_nodes.add(basis)
    if not all_nodes:
        return {}

    seed = {n: 1.0 for n in all_nodes}
    for did, _ in decision_edges:
        seed[did] = DECISION_BOOST

    rank = dict(seed)
    for _ in range(ITERATIONS):
        nxt = {n: (1.0 - DAMPING) * seed[n] for n in all_nodes}
        for child, ps in parents.items():
            if not ps:
                continue
            share = DAMPING * rank.get(child, 0.0) / len(ps)
            for p in ps:
                nxt[p] = nxt.get(p, 0.0) + share
        rank = nxt

    top = max(rank.values()) or 1.0
    return {n: round(v / top, 6) for n, v in rank.items()}


def dependent_counts(edges: list[tuple[str, str]],
                     decision_edges: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    """(transitive dependents, decisions) per node. Exact, for explanation.

    PageRank gives ranking; this gives the number a person can act on --
    "four conclusions and one decision rest on this" is checkable in a way
    that "criticality 0.83" is not.
    """
    children: dict[str, list[str]] = {}
    for child, parent in edges:
        children.setdefault(parent, []).append(child)
    dec_of: dict[str, set[str]] = {}
    for did, basis in decision_edges:
        dec_of.setdefault(basis, set()).add(did)
        children.setdefault(basis, []).append(did)

    out: dict[str, tuple[int, int]] = {}
    for node in set(children) | {p for _, p in edges} | {b for _, b in decision_edges}:
        seen: set[str] = set()
        stack = list(children.get(node, []))
        decs: set[str] = set(dec_of.get(node, ()))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n.startswith("dec_"):
                decs.add(n)
            stack.extend(children.get(n, []))
        out[node] = (len(seen), len(decs))
    return out
