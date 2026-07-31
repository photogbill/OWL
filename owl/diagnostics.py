"""F5 -- named checks that turn "it doesn't work" into something fixable.

Every check has a stable id, a status, and a REMEDY. The remedy is the part
that matters: a diagnostic that reports a problem without saying what to do
about it has moved the burden rather than lifted it.

Three principles, each learned the hard way:

1. **A check must be able to fail.** Several of these exist because a real
   model disagreed with an assumption that fifteen rounds of green tests had
   confirmed. A check that cannot go red is documentation, not diagnosis.

2. **Silence is not health.** The worst failures in this engine have all
   been silent: a floor above every true match, scores divided by a
   similarity no encoder reaches, vectors from two different models compared
   against each other. None raised. All produced plausible output. Anything
   that degrades quality without erroring gets a check here.

3. **Diagnosis must work when the thing is broken.** These run read-only and
   never write, because the moment you most need `doctor()` is the moment
   the store is damaged, busy, or on media you cannot write to.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    id: str
    title: str
    status: str
    detail: str = ""
    remedy: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, id, title, status, detail="", remedy=""):
        self.checks.append(Check(id, title, status, detail, remedy))

    def verdict(self, id, ok, title, *, ok_detail="", bad_detail="",
                remedy="", warn_only=False):
        """Most checks are a boolean plus two sentences. Keep them cheap to
        write, or they do not get written."""
        self.add(id, title, PASS if ok else (WARN if warn_only else FAIL),
                 ok_detail if ok else bad_detail, "" if ok else remedy)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def as_dict(self) -> dict:
        return {
            "checks": [asdict(c) for c in self.checks],
            "passed": sum(c.status == PASS for c in self.checks),
            "warned": len(self.warnings),
            "failed": len(self.failed),
            "healthy": not self.failed,
        }

    def render(self) -> str:
        width = max((len(c.id) for c in self.checks), default=10)
        lines = []
        for c in self.checks:
            lines.append(f"  {c.status:4s}  {c.id:{width}s}  {c.detail}")
            if c.remedy:
                lines.append(f"        {'':{width}s}  -> {c.remedy}")
        n = len(self.checks)
        lines.append(f"\n  {sum(c.status == PASS for c in self.checks)}/{n} "
                     f"passed, {len(self.warnings)} warning(s), "
                     f"{len(self.failed)} failure(s)")
        return "\n".join(lines)


# ── the checks ───────────────────────────────────────────────────────────
# Grouped by what they protect. Each takes the Owl and the Report.

def check_store(mind, rep: Report) -> None:
    rep.add("store.path", "Store location", PASS, mind._s.path)
    if mind.readonly:
        # PASS, not WARN. Read-only is a MODE, not a defect -- doctor reports
        # the health of the store, and `python -m owl doctor` always opens
        # this way. The consequence that actually matters (this recall was
        # not reinforced) is reported per-answer in Recall.degraded, which is
        # where a caller can act on it. Warning here would train people to
        # ignore warnings.
        rep.add("store.readonly", "Opened read-only", PASS,
                "reads work; nothing will be recorded, including retrieval "
                "reinforcement")
        if getattr(mind._s, "immutable", False):
            rep.add("store.liveness", "Immutable snapshot", WARN,
                    "WAL could not be used, so this is frozen at open time "
                    "-- writes by other processes are invisible",
                    "copy the store to writable media to see live updates")


def check_substrate(mind, rep: Report) -> None:
    orphan = mind._scalar(
        "SELECT COUNT(*) FROM derivation_edge e WHERE NOT EXISTS "
        "(SELECT 1 FROM observation o WHERE o.id=e.parent_id) AND NOT EXISTS "
        "(SELECT 1 FROM derived d WHERE d.id=e.parent_id)")
    rep.verdict("substrate.provenance_intact", orphan == 0,
                "Every derivation traces to a real parent",
                ok_detail="no dangling edges",
                bad_detail=f"{orphan} derivation edges point nowhere",
                remedy="run doctor(repair=True), or why() will raise on "
                       "the affected nodes")

    unindexed = mind._scalar(
        "SELECT COUNT(*) FROM observation o WHERE NOT EXISTS "
        "(SELECT 1 FROM mem_index m WHERE m.node_id=o.id)")
    rep.verdict("substrate.indexed", unindexed == 0,
                "Every observation has an index row",
                ok_detail="all observations indexed",
                bad_detail=f"{unindexed} observations are invisible to recall",
                remedy="reindex()")

    # The append-only guarantee is enforced by a trigger. If the trigger is
    # gone the store still WORKS -- which is why this is worth checking.
    trig = mind._scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name='observation'")
    rep.verdict("substrate.append_only", trig > 0,
                "Evidence is immutable at the database level",
                ok_detail=f"{trig} guard trigger(s) present",
                bad_detail="the append-only trigger is MISSING -- evidence "
                           "can be silently rewritten",
                remedy="this store was altered outside OWL; treat its "
                       "history as untrusted and re-import from source")


def check_monotonicity(mind, rep: Report) -> None:
    from .provenance import assert_monotonic
    from .protocols import Epistemic
    bad = 0
    for d in mind._s.query("SELECT id,confidence,epistemic_tag FROM derived"):
        pf = [mind._parent_facts(e["parent_id"]) for e in mind._s.query(
            "SELECT parent_id FROM derivation_edge WHERE child_id=?",
            (d["id"],))]
        try:
            assert_monotonic(pf, confidence=d["confidence"],
                             epistemic=Epistemic(d["epistemic_tag"]),
                             node_id=d["id"])
        except Exception:                                     # noqa: BLE001
            bad += 1
    rep.verdict("epistemics.monotonic", bad == 0,
                "No conclusion outranks its evidence",
                ok_detail="confidence and epistemic tags are consistent",
                bad_detail=f"{bad} nodes claim more certainty than their "
                           "parents support",
                remedy="the single invariant OWL exists to hold. Identify "
                       "with why(node_id); these nodes must be superseded, "
                       "not edited")


def check_embedder(mind, rep: Report) -> None:
    emb = mind.embedder
    if emb is None:
        rep.add("embedder.present", "Semantic retrieval available", WARN,
                "no embedder attached; recall is lexical only, so "
                "paraphrases will miss",
                "pass embedder= to Owl.open()")
        return
    rep.add("embedder.present", "Semantic retrieval available", PASS,
            getattr(emb, "name", "unknown"))
    rep.verdict("embedder.semantic", mind._semantic,
                "Embedder produces real semantic vectors",
                ok_detail="semantic",
                bad_detail="this is a hashing fallback, not a model -- it "
                           "cannot match a paraphrase, only shared tokens",
                remedy="attach a real model; see validate_embedder.py")

    cal = getattr(emb, "calibration", None)
    rep.verdict("embedder.calibrated", cal is not None,
                "Gate parameters measured for THIS encoder",
                ok_detail=f"floor {getattr(cal, 'noise_floor', '?')} .. "
                          f"ceiling {getattr(cal, 'ceiling', '?')}",
                bad_detail="running on defaults measured from a different "
                           "model; thresholds may sit above real matches",
                remedy='validate.bat "<model>.gguf" --calibrate',
                warn_only=True)

    if cal is not None:
        # Both of these were live bugs, and neither raised anything.
        rep.verdict("embedder.ceiling_measured", cal.ceiling < 1.0,
                    "Scores scaled to the encoder's real range",
                    ok_detail=f"ceiling {cal.ceiling}",
                    bad_detail="ceiling is 1.0, a similarity no encoder "
                               "reaches -- genuine matches are compressed "
                               "toward zero and read as DONT_KNOW",
                    remedy="re-run --calibrate (the sidecar predates this "
                           "measurement)", warn_only=True)
        if cal.separability:
            weak = cal.separability < 0.95
            rep.verdict("embedder.separability", not weak,
                        "Encoder distinguishes a match from noise",
                        ok_detail=f"AUC {cal.separability:.4f}",
                        bad_detail=f"AUC {cal.separability:.4f} -- a true "
                                   "match outscores a random one less often "
                                   "than it should",
                        remedy="try a larger model or a lighter quant",
                        warn_only=True)


def check_vectors(mind, rep: Report) -> None:
    if mind.embedder is None:
        return
    name = getattr(mind.embedder, "name", None)
    rows = mind._s.query(
        "SELECT model, COUNT(*) n FROM vector WHERE space='read' GROUP BY model")
    stale = {r["model"]: r["n"] for r in rows if r["model"] != name}
    rep.verdict("vectors.single_model", not stale,
                "All vectors come from the current encoder",
                ok_detail=f"all from {name!r}",
                bad_detail="; ".join(f"{v} from {k!r}" for k, v in
                                     stale.items()) +
                           " -- vectors from different encoders are not "
                           "comparable and are excluded from search",
                remedy="reindex() to re-embed them, or reopen with the "
                       "original encoder")

    n_nodes = (mind._scalar("SELECT COUNT(*) FROM observation")
               + mind._scalar("SELECT COUNT(*) FROM derived"))
    have = mind._vec.count()
    queued = mind.pending()
    missing = max(0, n_nodes * 2 - have - queued * 2)
    rep.verdict("vectors.coverage", missing == 0,
                "Every memory is semantically findable",
                ok_detail=(f"{have} vectors for {n_nodes} nodes"
                           + (f", {queued} queued" if queued else "")),
                bad_detail=f"{missing} vectors missing and NOT queued -- "
                           "these will never be embedded on their own",
                remedy="reindex()")


def check_queue(mind, rep: Report) -> None:
    if not mind.defer_embedding:
        return
    waiting = mind.pending()
    rep.verdict("queue.pending", waiting == 0,
                "Nothing waiting to be embedded",
                ok_detail="queue empty",
                bad_detail=f"{waiting} captured but not yet findable by "
                           "meaning",
                remedy="absorb() or tend()", warn_only=True)
    dead = mind._scalar("SELECT COUNT(*) FROM embed_queue WHERE attempts >= 3")
    rep.verdict("queue.abandoned", dead == 0,
                "No memory has been given up on",
                ok_detail="none abandoned",
                bad_detail=f"{dead} memories failed embedding three times "
                           "and will not be retried; they remain findable "
                           "lexically only",
                remedy="inspect last_error in embed_queue; fix the model or "
                       "the content, then reset attempts to 0")


def check_defence(mind, rep: Report) -> None:
    quarantined = mind._scalar(
        "SELECT COUNT(*) FROM observation WHERE trust<>'trusted'")
    rep.verdict("defence.quarantine_reviewed", quarantined == 0,
                "No untrusted content awaiting review",
                ok_detail="none quarantined",
                bad_detail=f"{quarantined} observations are quarantined and "
                           "excluded from recall",
                remedy="review them; they are held, not deleted",
                warn_only=True)
    audit = mind.self_audit()
    rep.verdict("defence.self_audit", audit["clean"],
                "Internal consistency audit",
                ok_detail="clean",
                bad_detail="; ".join(
                    f"{f['kind']}: {f['detail']}"
                    for f in audit["findings"][:5]),
                remedy="see self_audit() for the full list")


def check_crypto(mind, rep: Report) -> None:
    """A11 -- an encrypted store beside a readable key is not encrypted."""
    from . import crypto
    keyfile = getattr(mind, "keyfile", None)
    if not keyfile:
        return
    bad = crypto.key_permissions(keyfile)
    rep.verdict("crypto.key_permissions", not bad,
                "Key file is readable only by its owner",
                ok_detail=f"{keyfile} is 0600",
                bad_detail="; ".join(bad),
                remedy=f"chmod 600 {keyfile}")


def check_decisions(mind, rep: Report) -> None:
    open_impacts = mind._scalar(
        "SELECT COUNT(*) FROM decision_impact WHERE acknowledged_at IS NULL")
    rep.verdict("decisions.impacts_acknowledged", open_impacts == 0,
                "No decision is resting on retracted evidence",
                ok_detail="nothing outstanding",
                bad_detail=f"{open_impacts} decision(s) rest on evidence "
                           "that has since changed and nobody has looked",
                remedy="impacts() lists them; this is the question OWL "
                       "exists to answer",
                warn_only=True)


def check_shards(mind, rep: Report) -> None:
    """G5 -- is storage actually partitioned, and does the copy still agree?

    Two different questions, and only the second can go wrong quietly.
    """
    from . import shards
    if not getattr(mind._s, "sharded", False):
        # A real state, not a failure: read-only media cannot be migrated
        # and must still be readable. WARN rather than FAIL because
        # everything WORKS -- it is merely paying the old price.
        rep.add("shards.layout", "Partition-sharded storage", WARN,
                "this store predates G5 and has not been migrated, so every "
                "query is scoped after the scan rather than before it -- "
                "correct, and O(store) instead of O(partition)",
                remedy="open it read-write once; migration is automatic")
        return

    parts = mind._scalar("SELECT COUNT(*) FROM partition")
    rep.add("shards.layout", "Partition-sharded storage", PASS,
            f"{parts} partition(s); lexeme, vector and mem_index are all "
            f"indexed partition-first")

    # The load-bearing one. The safety case for denormalising `partition`
    # onto lexeme and vector is that a node never changes partition -- so
    # the copy cannot drift from mem_index. Drift here is not a slow query,
    # it is content filed under a partition it does not belong to, which is
    # a confidentiality defect. Checked rather than asserted in a comment.
    problems = shards.verify(mind._s._reader)
    rep.verdict("shards.partition_agrees", not problems,
                "Every index row agrees with its node's partition",
                ok_detail="the denormalised copy matches mem_index exactly",
                bad_detail="; ".join(problems),
                remedy="this store was altered outside OWL -- content is "
                       "filed under the wrong partition and may be visible "
                       "across a boundary; re-import from source")


ALL = (check_store, check_substrate, check_monotonicity, check_embedder,
       check_vectors, check_queue, check_defence, check_crypto,
       check_decisions, check_shards)


def run(mind) -> Report:
    rep = Report()
    for fn in ALL:
        try:
            fn(mind, rep)
        except Exception as exc:                              # noqa: BLE001
            # A check that crashes must not take the diagnosis with it --
            # that would make doctor() useless in exactly the conditions it
            # is for.
            rep.add(f"{fn.__name__}.crashed", "Check could not run", FAIL,
                    f"{exc.__class__.__name__}: {exc}",
                    "this is a bug in the check itself; the store may still "
                    "be fine")
    return rep
