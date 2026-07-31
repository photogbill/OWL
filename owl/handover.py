"""Handover — portable memory packs with automatic epistemic demotion.

The scenario this exists for: someone lands where they know nobody and needs
to help fast. The highest-value thing you can give them is not the files. It
is the *ledger* of the person who was there before -- with provenance intact,
with their open questions, with the searches they already ran and the dead
ends they already hit.

Nobody else can do this safely. Import a Mem0 or A-MEM store and you inherit
the previous operator's inferences as your facts, because nothing in those
formats distinguishes what they SAW from what they CONCLUDED. OWL's
monotonicity lattice makes the distinction structural, so the transplant rule
is one line:

    on import, every epistemic tag shifts DOWN one rank.

        observed      -> reported        (you did not see it; they did)
        inferred      -> hypothesized    (their conclusion is your guess)
        hypothesized  -> dropped         (their guess is nothing to you)

Their certainties become your reports. Their conclusions become your
hypotheses. That is exactly how a careful analyst treats a predecessor's
handover notes, and here it happens automatically and cannot be bypassed --
the invariant that clamps derived nodes clamps grafted ones identically.

SAFETY RULES, enforced in code, not documented as guidance:
  * A sealed partition NEVER exports. Raises.
  * Suppressed material never exports.
  * Affect-marked material never exports by default. Someone's distress is
    not a deliverable.
  * Every imported node is attributed to the source operator, permanently.
  * The pack is plain JSON. You must be able to READ what you are handing
    someone before you hand it to them.
"""
from __future__ import annotations

import datetime as _dt

import gzip
import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .protocols import Epistemic, OwlError, PartitionError

PACK_VERSION = 2

_DEMOTE = {
    Epistemic.OBSERVED: Epistemic.REPORTED,
    Epistemic.REPORTED: Epistemic.INFERRED,
    Epistemic.INFERRED: Epistemic.HYPOTHESIZED,
    Epistemic.HYPOTHESIZED: None,          # dropped
}


class HandoverError(OwlError):
    pass


@dataclass
class Manifest:
    version: int
    exported_at: float
    exporter: str
    source_partition: str
    label: str
    counts: dict[str, int] = field(default_factory=dict)
    checksum: str = ""
    notes: str = ""


def demote(tag: Epistemic, steps: int = 1) -> Epistemic | None:
    cur: Epistemic | None = tag
    for _ in range(max(0, steps)):
        if cur is None:
            return None
        cur = _DEMOTE[cur]
    return cur


def _checksum(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ── export ───────────────────────────────────────────────────────────────
def build_pack(store, *, partition: str, exporter: str, now: float,
               label: str = "", include_affect: bool = False,
               affect_ceiling: float = 0.2, notes: str = "") -> dict:
    sealed = store.one("SELECT sealed FROM partition WHERE name=?", (partition,))
    if sealed is None:
        raise HandoverError(f"unknown partition {partition!r}")
    if sealed["sealed"]:
        raise PartitionError(
            f"partition '{partition}' is sealed and can never be exported. "
            "That is the entire meaning of sealed; there is no override flag."
        )

    obs, exposures, derived, edges, absences, intentions = [], [], [], [], [], []

    for r in store.query(
            "SELECT o.* FROM observation o JOIN mem_index m ON m.node_id=o.id "
            "WHERE o.partition=? AND m.suppressed_at IS NULL", (partition,)):
        if not include_affect and (r["affect"] or 0.0) > affect_ceiling:
            continue
        obs.append({k: r[k] for k in r.keys()})

    ids = {o["id"] for o in obs}

    for r in store.query(
            "SELECT d.* FROM derived d JOIN mem_index m ON m.node_id=d.id "
            "WHERE d.partition=? AND m.suppressed_at IS NULL", (partition,)):
        derived.append({k: r[k] for k in r.keys()})
    ids |= {d["id"] for d in derived}

    for r in store.query("SELECT * FROM derivation_edge"):
        if r["child_id"] in ids and r["parent_id"] in ids:
            edges.append({k: r[k] for k in r.keys()})

    # The exposure log is most of a real handover briefing: not just what the
    # previous operator knew, but what they had been TOLD and when.
    for r in store.query("SELECT * FROM exposure"):
        if r["node_id"] in ids:
            exposures.append({k: r[k] for k in r.keys() if k != "id"})

    # Failed searches are expensive to establish and free to carry.
    for r in store.query("SELECT * FROM absence WHERE partition=?", (partition,)):
        absences.append({k: r[k] for k in r.keys()})

    # Open loops: "here is what I was in the middle of."
    for r in store.query(
            "SELECT * FROM intention WHERE partition=? AND status='pending'",
            (partition,)):
        intentions.append({k: r[k] for k in r.keys()})

    payload = {
        "observations": obs, "derived": derived, "edges": edges,
        "exposures": exposures, "absences": absences, "intentions": intentions,
    }
    man = Manifest(
        version=PACK_VERSION, exported_at=now, exporter=exporter,
        source_partition=partition, label=label or partition,
        counts={k: len(v) for k, v in payload.items()},
        checksum=_checksum(payload), notes=notes,
    )
    return {"manifest": asdict(man), **payload}


def render_markdown(pack: dict, *, max_items: int = 0) -> str:
    """F6 -- a pack you can actually read before you hand it to someone.

    The justification for the whole format is that a transfer should be
    reviewable. JSON is inspectable, which is not the same thing: nobody
    proof-reads six thousand lines of it, so in practice the review does not
    happen and the guarantee is theatre.

    Organised by EPISTEMIC STATUS rather than by table, because the question
    a reviewer is actually asking is "what am I about to be told, and how
    solid is it?" -- not "what rows are in here". Facts first, then things
    the previous operator concluded, then what they were still chasing.

    Every derived claim carries the demotion it will undergo on import, so
    the reviewer sees what the recipient will see, not what the exporter
    sees. That difference is the entire point of the transplant rule.
    """
    man = pack.get("manifest", {})
    obs = pack.get("observations", [])
    derived = pack.get("derived", [])
    absences = pack.get("absences", [])
    intentions = pack.get("intentions", [])
    exposures = pack.get("exposures", [])

    def cut(rows):
        return rows[:max_items] if max_items else rows

    def when(ts):
        if not ts:
            return "unknown"
        return _dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")

    L: list[str] = []
    L.append(f"# Handover pack — {man.get('label', 'unlabelled')}")
    L.append("")
    L.append(f"**From:** {man.get('exporter', 'unknown')}  ")
    L.append(f"**Partition:** `{man.get('source_partition', '?')}`  ")
    L.append(f"**Exported:** {when(man.get('exported_at'))}  ")
    L.append(f"**Checksum:** `{man.get('checksum', '')[:16]}`")
    if man.get("notes"):
        L.append("")
        L.append(f"> {man['notes']}")
    L.append("")
    L.append("## What you are receiving")
    L.append("")
    L.append("| | count |")
    L.append("|---|---|")
    for k, v in (man.get("counts") or {}).items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("> **Every epistemic tag drops one rank on import.** What the "
             "previous")
    L.append("> operator *observed*, you receive as *reported* — you did not "
             "see it, they did.")
    L.append("> Nothing in a pack can arrive as a first-hand fact.")
    L.append("")

    if obs:
        L.append("---")
        L.append("")
        L.append("## Evidence — what was actually seen")
        L.append("")
        L.append("Arrives as `reported`. Trace anything load-bearing to its "
                 "source before relying on it.")
        L.append("")
        for o in cut(obs):
            src = o.get("source_ref") or o.get("origin") or "unattributed"
            grade = f"{o.get('reliability', '?')}{o.get('credibility', '?')}"
            flag = "" if o.get("trust") == "trusted" else \
                f"  **[{o.get('trust')}]**"
            L.append(f"- {o.get('content', '').strip()}{flag}")
            L.append(f"  <sub>{when(o.get('observed_at'))} · `{src}` · "
                     f"Admiralty {grade} · {o.get('claim_class', '?')}</sub>")
        if max_items and len(obs) > max_items:
            L.append(f"- *… and {len(obs) - max_items} more*")
        L.append("")

    if derived:
        L.append("---")
        L.append("")
        L.append("## Conclusions — what the previous operator worked out")
        L.append("")
        L.append("**These are not facts.** They were inferences, and they "
                 "arrive demoted again.")
        L.append("Read the falsifier: it says what would prove each one "
                 "wrong.")
        L.append("")
        for d in cut(derived):
            tag = d.get("epistemic_tag", "?")
            after = demote(Epistemic(tag)) if tag in {
                e.value for e in Epistemic} else None
            arrow = f"{tag} → {after.value}" if after else f"{tag} → *dropped*"
            L.append(f"- {d.get('content', '').strip()}")
            L.append(f"  <sub>{arrow} · confidence "
                     f"{float(d.get('confidence') or 0):.2f} · via "
                     f"{d.get('producer', '?')}"
                     + (f" · falsifier: {d['falsifier']}"
                        if d.get("falsifier") else "") + "</sub>")
        if max_items and len(derived) > max_items:
            L.append(f"- *… and {len(derived) - max_items} more*")
        L.append("")

    if absences:
        L.append("---")
        L.append("")
        L.append("## Established absences — where it is NOT")
        L.append("")
        L.append("Expensive to establish, free to carry. This is the section "
                 "that saves you")
        L.append("repeating someone else's dead ends.")
        L.append("")
        for a in cut(absences):
            L.append(f"- Looked for **{a.get('query', '?')}** on "
                     f"{when(a.get('searched_at'))} — not found"
                     + (f" (scope: {a['scope']})" if a.get("scope") else ""))
        L.append("")

    if intentions:
        L.append("---")
        L.append("")
        L.append("## Open loops — what was still in flight")
        L.append("")
        for i in cut(intentions):
            due = f" · due {when(i.get('due_at'))}" if i.get("due_at") else ""
            L.append(f"- [ ] {i.get('action', '?')}{due}")
        L.append("")

    if exposures:
        people = sorted({e.get("who") for e in exposures if e.get("who")})
        if people:
            L.append("---")
            L.append("")
            L.append("## Who has been told what")
            L.append("")
            L.append("Not a courtesy note. These people believe things on "
                     "the strength of having")
            L.append("been told them, and some of those things have since "
                     "changed.")
            L.append("")
            for p in people:
                n = sum(1 for e in exposures if e.get("who") == p)
                L.append(f"- **{p}** — {n} item(s)")
            L.append("")

    L.append("---")
    L.append("")
    L.append(f"<sub>OWL pack v{man.get('version', '?')} · verify with "
             "`owl.verify_pack(path)` before importing · this rendering is "
             "for review only, import from the `.owlpack`</sub>")
    return "\n".join(L)


def write_pack(pack: dict, path: str | Path, *, compress: bool = False) -> Path:
    p = Path(path)
    blob = json.dumps(pack, indent=2, sort_keys=True).encode()
    if compress or p.suffix == ".gz":
        p.write_bytes(gzip.compress(blob))
    else:
        p.write_bytes(blob)
    return p


def read_pack(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    pack = json.loads(raw)
    man = pack.get("manifest", {})
    if man.get("version") != PACK_VERSION:
        raise HandoverError(
            f"pack version {man.get('version')} != {PACK_VERSION}")
    body = {k: pack[k] for k in
            ("observations", "derived", "edges", "exposures", "absences",
             "intentions")}
    if _checksum(body) != man.get("checksum"):
        raise HandoverError(
            "checksum mismatch: this pack was modified after export. Refusing "
            "to graft -- a handover you cannot verify is worse than none.")
    return pack


# ── import ───────────────────────────────────────────────────────────────
def plan_graft(pack: dict, *, steps: int = 1) -> dict:
    """Dry run. What would be admitted, demoted, and dropped.

    Always available, and worth calling first: a handover is a trust decision,
    and it should be inspectable before it is irreversible.
    """
    kept_obs = len(pack["observations"])          # observed -> reported
    kept_der, dropped_der = 0, 0
    for d in pack["derived"]:
        if demote(Epistemic(d["epistemic_tag"]), steps) is None:
            dropped_der += 1
        else:
            kept_der += 1
    return {
        "observations_admitted_as": demote(Epistemic.OBSERVED, steps).value,
        "observations": kept_obs,
        "derived_admitted": kept_der,
        "derived_dropped": dropped_der,
        "exposures": len(pack["exposures"]),
        "absences": len(pack["absences"]),
        "open_intentions": len(pack["intentions"]),
        "exporter": pack["manifest"]["exporter"],
    }
