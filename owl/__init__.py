"""O.W.L. — Observation & Wisdom Ledger.

A provenance-first memory engine. The one thing it always does:

    it can tell you how it knows.

Design commitments, in priority order:
  1. Evidence is immutable. Forgetting happens in the index, never the record.
  2. Provenance is transitive and monotone. Speculation cannot become fact.
  3. Metamemory precedes memory. Knowing whether you know is cheap and comes first.

Tier 0 (no models at all) is fully functional: provenance, decay, event
segmentation, interference detection, partitions, prospective memory.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path as FsPath

# `Path` is ALSO an entity-graph type in owl.entities, and that import
# shadows this one further down the file. Aliasing the filesystem one keeps
# the two apart -- the collision already produced an AttributeError on
# with_suffix() that looked like a stdlib fault.
Path = FsPath
from typing import Any, Iterable, Sequence

from . import (attribution, decisions, decontext, defence, entities,
               epistemics, fusion, handover, lexical, metamemory,
               quantities, salience, shards, theory_of_mind, vectors)
from .protocols import (
    Chunk, Clock, Embedder, Epistemic, Origin, OwlError, PartitionError,
    Provenance, Reasoner, Recall, Space, State, SystemClock,
)
from .provenance import ParentFacts, assert_monotonic, is_presentable_as_fact, resolve
from .segmentation import Segmenter
from .attribution import Record
from .decisions import Cause, Impact, Status
from .entities import Entity, Path, PathStep
from .handover import HandoverError
from .theory_of_mind import Divergence, Held
from .store.sqlite import SqliteStore
from .vectors import VectorIndex

__all__ = [
    "Owl", "Recall", "State", "Origin", "Epistemic", "Chunk", "Provenance",
    "PartitionError", "OwlError", "Held", "Divergence", "HandoverError",
    "Entity", "Path", "PathStep", "Impact", "Cause", "Status", "Record",
]
__version__ = "0.1.0"

DAY = 86400.0
# Cowan's ~4 chunks, and the lost-in-the-middle degradation in long contexts,
# point the same way: cap retrieval and spend the budget on density instead.
MAX_BUDGET = 7
# Below this cosine, a match is noise. Calibrated for sentence encoders such
# as BGE-M3, where unrelated short texts sit around 0.4-0.55.
# Where 'unrelated' sits for a typical sentence encoder -- the zero-point
# for judging absolute match quality.
SEMANTIC_FLOOR = 0.40
# Where to stop LOOKING. Much lower on purpose: these are two different jobs.
# Using the noise floor as the search cutoff starved the candidate set, which
# tripped the "fewer than three -> no distribution" path, which falls back to
# absolute level alone. A genuine match was rejected twice over for one
# mistake.
SEARCH_FLOOR = 0.15


class Owl:
    """The whole public API is: observe, recall, tend."""

    # ── lifecycle ────────────────────────────────────────────────────
    def __init__(self, store: SqliteStore, *, embedder: Embedder | None = None,
                 reasoner: Reasoner | None = None, clock: Clock | None = None,
                 defer_embedding: bool = False):
        self._s = store
        self.defer_embedding = defer_embedding
        self.embedder = embedder
        self.reasoner = reasoner
        self.clock: Clock = clock or SystemClock()
        self._vec = VectorIndex(store)
        self._segmenters: dict[str, Segmenter] = {}
        self._episode: dict[str, str] = {}
        self._period: dict[str, str] = {}
        self._last_recalled: list[str] = []
        self._warnings: list[str] = []
        self._last_background: float | None = None
        self._embed_failed_this_call = False
        # G1-G5. All opt-in-by-effect: they change speed, never answers.
        from .fastpath import VectorCache
        from .freshness import DirtySet, RecallCache
        from .shards import ShardSet
        self._vcache = VectorCache()
        self._rcache = RecallCache()
        # G5: the vocabulary filter and the node count are per PARTITION,
        # because both were previously answers to a question about the
        # whole store being used to decide work on one shard of it.
        self._shards = ShardSet(load_terms=self._shard_terms,
                                count_nodes=self._shard_count,
                                sharded=getattr(store, "sharded", False))
        # G4+G5: one dirty set per partition. A single shared set meant
        # tend("private") CONSUMED the work partition's pending nodes --
        # take() clears what it returns -- so consolidation work queued by
        # a write to one partition was silently discarded by a maintenance
        # pass on another. Silent loss of consolidation is the exact
        # failure mode the periodic full sweep exists to catch, and the
        # partitions made it routine rather than rare.
        self._dirty: dict[str, DirtySet] = {}
        self._s._on_write.append(self._on_write)
        # Receipts are on by default: a retrieval you cannot reconstruct is a
        # decision you cannot audit, and the whole point is auditability.
        self.receipts: bool = True
        # Model provenance: cheap now, impossible to backfill. When the host
        # upgrades from a 7B to a 24B, every conclusion resting on the smaller
        # model's judgement becomes identifiable.
        self.model_id: str | None = getattr(reasoner, "name", None)
        # Opening must not be a write. A read-only store already has whatever
        # partitions it has, and requiring one to be created is exactly the
        # kind of incidental dependency that makes an archive unreadable.
        if not self._s.readonly:
            self.partition("default")

    @classmethod
    def open(cls, path: str | Path, *, embedder: Embedder | None = None,
             reasoner: Reasoner | None = None, clock: Clock | None = None,
             defer_embedding: bool = False, readonly: bool = False) -> "Owl":
        """`readonly=True` opens without taking a lock or creating anything.

        F2: the store is always directly readable, and background machinery
        is never a gatekeeper on recall. A file on read-only media, a
        forensic copy, an archived pack, or a store another process is
        actively writing must still answer questions. Writes raise
        `ReadOnlyError` rather than failing quietly.

        `defer_embedding=True` takes embedding off the capture path.

        `observe()` then writes the substrate row and queues the vector work,
        returning in about a millisecond instead of however long the model
        takes -- 330 ms for Qwen3-Embedding-8B on CPU. Call `absorb()` when
        idle to drain the queue.

        Off by default because it changes when a memory becomes findable,
        and a caller should opt into that knowingly rather than discover it.
        """
        return cls(SqliteStore(path, readonly=readonly), embedder=embedder,
                   reasoner=reasoner, clock=clock,
                   defer_embedding=defer_embedding)

    def close(self) -> None:
        self._s.close()

    def __enter__(self) -> "Owl":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def tier(self) -> int:
        """0 = deterministic only, 1 = +embeddings, 2 = +reasoning.

        A non-semantic embedder (the hashing fallback) does NOT count as
        Tier 1. A fallback that silently produces poor retrieval while looking
        like it works is worse than no fallback at all.
        """
        if self.reasoner is not None and self._semantic:
            return 2
        return 1 if self._semantic else 0

    @property
    def _semantic(self) -> bool:
        return (self.embedder is not None
                and getattr(self.embedder, "is_semantic", True))

    def _embed(self, texts: list[str], space: Space) -> list[list[float]] | None:
        if self.embedder is None or not texts:
            return None
        try:
            self._embed_failed_this_call = False
            # Normalise at the boundary. Embedders disagree on output scale
            # and everything downstream -- separation weighting, cosine
            # thresholds, density estimates -- assumes unit vectors.
            return [vectors.unit(v)
                    for v in self.embedder.embed(texts, space)]
        except Exception as exc:                      # noqa: BLE001
            # Never let an embedder failure lose a write. Lexical still
            # works -- but SAY so, because a silent fallback returns the same
            # shaped answer from weaker machinery.
            self._embed_failed_this_call = True
            self._warn(f"embedder failed ({exc.__class__.__name__}); "
                       "falling back to lexical for this call")
            return None

    def _warn(self, msg: str) -> None:
        self._warnings.append(msg)

    # ── partitions ───────────────────────────────────────────────────
    def partition(self, name: str, *, flows_to: Iterable[str] = (),
                  reads_from: Iterable[str] = (), sealed: bool = False,
                  summary_reads_from: Iterable[str] = ()) -> str:
        """Declare a partition. Flow is denied by default and is not symmetric.

        `sealed=True` blocks OUTflow only -- nothing here can ever surface
        anywhere else. It does NOT block inflow, which is a separate decision:

            mind.partition("athena", sealed=True,
                           summary_reads_from=["work"])

        gives a one-way membrane. Athena sees what the day held, at the level
        of episode and period summaries. She never sees the raw material, and
        nothing she thinks can ever reach the work arena.

        `summary_reads_from` is graded permeability: only derived, abstracted
        content crosses -- so a companion can know *that the clinic day was
        bad* without holding the individual images. That distinction matters
        more than it looks: unbidden recall of raw traumatic detail is the
        specific harm to avoid.
        """
        self._s.ensure_partition(
            name, sealed=sealed, now=self.clock.now(),
            flows_to=[(t, "full") for t in flows_to],
            reads_from=([(s_, "full") for s_ in reads_from]
                        + [(s_, "summary") for s_ in summary_reads_from]))
        return name

    # ── periods (Self-Memory System hierarchy) ───────────────────────
    @contextmanager
    def period(self, label: str, *, partition: str = "default"):
        pid = f"per_{uuid.uuid4().hex[:12]}"
        now = self.clock.now()
        self._s.write(lambda c: c.execute(
            "INSERT INTO period(id,partition,label,opened_at) VALUES(?,?,?,?)",
            (pid, partition, label, now)))
        prev = self._period.get(partition)
        self._period[partition] = pid
        try:
            yield pid
        finally:
            if prev is None:
                self._period.pop(partition, None)
            else:
                self._period[partition] = prev
            self._s.write(lambda c: c.execute(
                "UPDATE period SET closed_at=? WHERE id=?", (self.clock.now(), pid)))

    # ── WRITE ────────────────────────────────────────────────────────
    def observe(self, content: str, *, origin: str | Origin = Origin.USER,
                source_ref: str = "inline", partition: str = "default",
                valid_from: float | None = None, valid_to: float | None = None,
                affect: float = 0.0, context: dict | None = None,
                claim_class: str | None = None, reliability: str = "F",
                credibility: int = 6, supersedes: str | None = None,
                trust: str = "trusted", producer_model: str | None = None,
                acquisition_cost: float = 0.0) -> str:
        """Record a primary observation. Immutable once written."""
        if not content or not content.strip():
            raise ValueError("refusing to store empty content")
        o = Origin(origin) if not isinstance(origin, Origin) else origin
        now = self.clock.now()
        nid = f"obs_{uuid.uuid4().hex[:16]}"
        chash = hashlib.sha256(content.encode()).hexdigest()

        seg = self._segmenters.setdefault(partition, Segmenter())
        surprise, boundary = seg.push(content)
        episode_id = self._episode.get(partition)
        if boundary or episode_id is None:
            episode_id = f"epi_{uuid.uuid4().hex[:12]}"
            self._episode[partition] = episode_id

        period_id = self._period.get(partition)
        cls = claim_class or epistemics.classify(content)

        # ── write screening ──────────────────────────────────────────
        verdict = defence.screen(content, origin=o.value)
        if verdict.verdict == "blocked" and trust == "trusted":
            # Not refused -- refusing loses the evidence that an attack was
            # attempted, which is itself worth keeping. Quarantined instead:
            # retrievable, never authoritative.
            trust = "quarantined"
        elif verdict.verdict == "suspect" and trust == "trusted":
            trust = "untrusted"

        # ── supersession authority ───────────────────────────────────
        blocked_supersede = None
        if supersedes:
            old_grade = self.effective_grade(supersedes)
            ok, why_not = defence.may_supersede(
                (reliability, credibility), old_grade, trust=trust)
            if ok:
                recent = self._scalar(
                    "SELECT COUNT(*) FROM supersession_attempt WHERE "
                    "source_ref=? AND at > ? AND allowed=1",
                    (source_ref, now - defence.COUP_WINDOW))
                coup, coup_why = defence.is_coup(recent)
                if coup:
                    ok, why_not = False, coup_why
            if not ok:
                blocked_supersede, supersedes = (supersedes, why_not), None

        stability, difficulty = salience.initial_state(3)
        tfs = lexical.term_frequencies(content)
        ctx = json.dumps(context or {}, separators=(",", ":"))

        def _w(c: sqlite3.Connection) -> None:
            c.execute("INSERT OR IGNORE INTO partition(name,sealed,created_at) "
                      "VALUES(?,0,?)", (partition, now))
            if boundary or not c.execute(
                    "SELECT 1 FROM episode WHERE id=?", (episode_id,)).fetchone():
                c.execute(
                    "INSERT OR IGNORE INTO episode"
                    "(id,partition,period_id,started_at,boundary_surprise) "
                    "VALUES(?,?,?,?,?)",
                    (episode_id, partition, period_id, now, surprise))
            c.execute(
                "INSERT INTO observation(id,partition,observed_at,valid_from,"
                "valid_to,origin,source_ref,content,content_hash,context_env,"
                "episode_id,period_id,affect,claim_class,reliability,trust,"
                "producer_model,acquisition_cost,credibility) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (nid, partition, now, valid_from, valid_to, o.value, source_ref,
                 content, chash, ctx, episode_id, period_id, affect, cls,
                 reliability, trust, producer_model,
                 max(0.0, min(1.0, acquisition_cost)), credibility))
            if supersedes:
                old = c.execute("SELECT observed_at,claim_class FROM observation "
                                "WHERE id=?", (supersedes,)).fetchone()
                if old:
                    # Every supersession is a labelled survival observation:
                    # this class of claim held for N seconds before it changed.
                    # That is the training signal for per-class half-life.
                    c.execute(
                        "INSERT OR REPLACE INTO supersession"
                        "(old_node,new_node,claim_class,survived,at) "
                        "VALUES(?,?,?,?,?)",
                        (supersedes, nid, old["claim_class"],
                         now - old["observed_at"], now))
                    c.execute("UPDATE mem_index SET tier='cold' WHERE node_id=?",
                              (supersedes,))
            c.execute(
                "INSERT INTO mem_index(node_id,partition,stability,difficulty,"
                "last_review,review_count,access_log,surprise) "
                "VALUES(?,?,?,?,?,0,'[]',?)",
                (nid, partition, stability, difficulty, now, surprise))
            self._index_terms(c, nid, partition, tfs)
            c.execute(
                "INSERT OR IGNORE INTO source_origin(source_ref,origin_cluster,"
                "domain,ingest_batch,first_seen) VALUES(?,?,?,?,?)",
                (source_ref, attribution.origin_key(source_ref),
                 attribution.origin_key(source_ref),
                 str(int(now // attribution.BATCH_WINDOW)), now))
            c.execute(
                "INSERT OR REPLACE INTO write_screen(node_id,at,verdict,"
                "signals,score) VALUES(?,?,?,?,?)",
                (nid, now, verdict.verdict,
                 json.dumps(verdict.signals), verdict.score))
            if blocked_supersede is not None:
                c.execute(
                    "INSERT INTO supersession_attempt(at,source_ref,old_node,"
                    "allowed,reason) VALUES(?,?,?,0,?)",
                    (now, source_ref, blocked_supersede[0],
                     blocked_supersede[1]))
            elif supersedes:
                c.execute(
                    "INSERT INTO supersession_attempt(at,source_ref,old_node,"
                    "allowed,reason) VALUES(?,?,?,1,'authority sufficient')",
                    (now, source_ref, supersedes))
            for prev in self._last_recalled[-3:]:
                c.execute(
                    "INSERT INTO succession(src,dst,count) VALUES(?,?,1) "
                    "ON CONFLICT(src,dst) DO UPDATE SET count=count+1",
                    (prev, nid))

        self._s.write(_w)
        if blocked_supersede is not None:
            # Disagreement is information. A rejected supersession is recorded
            # as a conflict rather than discarded, so the contest is visible.
            self.derive(
                f"CONTESTED: {content}", parents=[blocked_supersede[0], nid],
                kind="conflict", producer="defence", partition=partition,
                confidence=0.3, epistemic=Epistemic.REPORTED)
        if supersedes:
            self.raise_impacts(supersedes, cause="superseded")
        self._dirty_for(partition).mark(nid)
        self._index_vectors(nid, content, partition=partition,
                            episode_id=episode_id, period_id=period_id,
                            source_ref=source_ref, when=now)
        return nid

    def derive(self, content: str, *, parents: Sequence[str], kind: str,
               producer: str, producer_model: str | None = None,
               confidence: float = 0.7,
               epistemic: Epistemic = Epistemic.INFERRED,
               partition: str = "default", falsifier: str | None = None,
               supersedes: str | None = None) -> str:
        """Write a derived node. Confidence and epistemic tag are CLAMPED to
        what the parents allow -- abstraction cannot launder speculation."""
        if kind == "hypothesis":
            if not falsifier:
                raise OwlError(
                    "a hypothesis must carry a falsifier: a concrete check that "
                    "would show it false. Untestable hypotheses are not stored."
                )
            # kind and epistemic tag are not allowed to disagree. A caller
            # writing a hypothesis while labelling it 'inferred' would produce
            # a node that quietly reads as a conclusion; the tag is forced.
            epistemic = Epistemic.HYPOTHESIZED
        if kind == "reflection" and epistemic.rank < Epistemic.INFERRED.rank:
            epistemic = Epistemic.INFERRED
        # Dimensional integrity: a derivation that drops a unit or changes a
        # magnitude has produced a DANGEROUS sentence, not a shorter one.
        for pid in parents:
            prow = self._node_row(pid)
            if prow is None:
                continue
            problems = quantities.conflicts(prow["content"], content)
            if problems and kind in ("summary", "abstraction", "graft"):
                raise OwlError(
                    f"dimensional integrity: {problems[0]} "
                    f"(deriving from {pid})")
        pf = [self._parent_facts(p) for p in parents]
        conf, epi = resolve(pf, proposed_confidence=confidence,
                            proposed_epistemic=epistemic)
        assert_monotonic(pf, confidence=conf, epistemic=epi)
        did = f"der_{uuid.uuid4().hex[:16]}"
        now = self.clock.now()

        def _w(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO derived(id,partition,created_at,kind,epistemic_tag,"
                "producer,producer_model,content,confidence,supersedes,"
                "falsifier) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (did, partition, now, kind, epi.value, producer,
                 producer_model or self.model_id, content, conf,
                 supersedes, falsifier))
            c.executemany(
                "INSERT OR IGNORE INTO derivation_edge(child_id,parent_id,role) "
                "VALUES(?,?,'evidence')", [(did, p) for p in parents])
            c.execute(
                "INSERT INTO mem_index(node_id,partition,stability,difficulty,"
                "last_review,review_count,access_log,surprise) "
                "VALUES(?,?,?,?,?,0,'[]',0.5)",
                (did, partition, *salience.initial_state(3), now))
            self._index_terms(c, did, partition,
                              lexical.term_frequencies(content))

        self._s.write(_w)
        self._index_vectors(did, content, partition=partition,
                            source_ref=producer, when=now)
        return did

    @property
    def readonly(self) -> bool:
        return self._s.readonly

    def pending(self) -> int:
        """How many memories are captured but not yet findable."""
        row = self._s.one("SELECT COUNT(*) n FROM embed_queue")
        return int(row["n"]) if row else 0

    def absorb(self, budget: int | None = None) -> dict:
        """Drain the embedding queue. Call when idle.

        Deliberately explicit rather than a background thread. A thread would
        make write-then-read nondeterministic, and every hard bug in this
        engine so far has been something invisible -- adding a race to a
        library whose entire pitch is auditability would be a poor trade.
        A caller who wants a thread can run this in one; the semantics stay
        the same either way.
        """
        done = failed = 0
        # One attempt per item per call. Without this the loop re-reads the
        # queue, finds the row it just failed on still sitting there, and
        # burns all three attempts in a single pass -- turning a transient
        # error into a permanent one and reporting three failures for one
        # memory.
        seen: set[str] = set()
        while budget is None or done + failed < budget:
            rows = [r for r in self._s.query(
                "SELECT * FROM embed_queue WHERE attempts < 3 "
                "ORDER BY attempts, queued_at LIMIT ?",
                (min(64, (budget - done - failed) + len(seen))
                 if budget else 64,))
                if r["node_id"] not in seen]
            if not rows:
                break
            for r in rows:
                if budget is not None and done + failed >= budget:
                    break
                seen.add(r["node_id"])
                try:
                    self._index_vectors(
                        r["node_id"], r["content"], partition=r["partition"],
                        episode_id=r["episode_id"], period_id=r["period_id"],
                        source_ref=r["source_ref"], when=r["when_ts"],
                        _queue=False)
                    # POSTCONDITION, not trust. `_embed` swallows embedder
                    # exceptions by design -- a model failure must never lose
                    # a write -- so a failure here returns quietly and the
                    # row would be dequeued with no vector, silently
                    # unembeddable forever. Check the work was actually done.
                    n = self._s.one(
                        "SELECT COUNT(*) n FROM vector WHERE node_id=?",
                        (r["node_id"],))["n"]
                    if not n:
                        raise RuntimeError(
                            "embedder produced no vector (see warnings)")
                except Exception as exc:                      # noqa: BLE001
                    failed += 1
                    # Leave it queued with the reason. A memory that cannot
                    # be embedded is still a memory -- the lexical path finds
                    # it -- so this degrades rather than losing anything.
                    self._s.write(lambda c, r=r, exc=exc: c.execute(
                        "UPDATE embed_queue SET attempts=attempts+1, "
                        "last_error=? WHERE node_id=?",
                        (f"{exc.__class__.__name__}: {exc}"[:200],
                         r["node_id"])))
                    continue
                done += 1
                self._s.write(lambda c, r=r: c.execute(
                    "DELETE FROM embed_queue WHERE node_id=?", (r["node_id"],)))
        return {"embedded": done, "failed": failed, "pending": self.pending()}

    def _index_vectors(self, node_id: str, content: str, *, partition: str,
                       episode_id: str | None = None,
                       period_id: str | None = None, source_ref: str = "",
                       when: float = 0.0, _queue: bool = True) -> None:
        """Two vectors per node, in two DIFFERENT spaces.

        WRITE gets content PLUS its distinguishing context -- partition,
        period, episode, source, day bucket. That is pattern separation: it
        deliberately pushes near-duplicates apart, so two structurally similar
        weekly meetings do not collapse onto each other. Standard RAG embeds
        one blended vector and then fights the interference this prevents.

        READ gets the bare semantic content, because a query arrives without
        any of that context and must complete against meaning alone.
        """
        if self.embedder is None:
            return
        if _queue and self.defer_embedding:
            # Capture is pure SQLite from here. The context is stored WITH
            # the row rather than looked up at drain time, because the write
            # vector encodes where the memory sat when it arrived.
            self._s.write(lambda c: c.execute(
                "INSERT OR REPLACE INTO embed_queue(node_id, content, "
                "partition, episode_id, period_id, source_ref, when_ts, "
                "queued_at) VALUES(?,?,?,?,?,?,?,?)",
                (node_id, content, partition, episode_id, period_id,
                 source_ref, when, self.clock.now())))
            return
        vecs = self._embed([content], Space.READ)
        if vecs is None:
            return
        semantic = vecs[0]
        ctx = vectors.context_signature(
            vectors.CONTEXT_DIM,
            partition, period_id, episode_id, source_ref,
            int(when // DAY) if when else 0)
        model = getattr(self.embedder, "name", "embedder")
        self._vec.put_many([
            (node_id, Space.WRITE, vectors.separate(semantic, ctx), model),
            (node_id, Space.READ, semantic, model),
        ], partition=partition)

    # ── READ ─────────────────────────────────────────────────────────
    def recall(self, query: str, *, budget: int = 5, partition: str = "default",
               as_of: float | None = None, suppress_affect_above: float = 1.01,
               allow_inferred: bool = True, group_by: str | None = "source",
               per_group: int = 2, token_budget: int | None = None) -> Recall:
        """Retrieve, then stamp what is still unread onto the answer.

        A wrapper rather than a line at the end of the body, because the body
        has several early returns -- including the fast "nothing in the store
        mentions it" path, which is precisely where "not yet read" is most
        likely to be mistaken for "not there". Stamping at each return site
        would work until someone adds a return. This cannot be missed.
        """
        # G1: the fast path deserves to be the fastest path. If no query
        # term exists anywhere in the store, there is nothing to retrieve
        # and no reason to ask the database.
        # Health warnings are NOT part of the retrieval path and must not
        # be skippable by an optimisation. The first version of the fast
        # path below returned before this ran, so a store whose vectors
        # were unusable said nothing about it -- a speedup silencing a
        # correctness message, which is precisely the failure class this
        # engine exists to catch.
        if self.embedder is not None:
            self._check_model_drift(getattr(self.embedder, "name", None))

        # G3: repeat queries within a session are extremely common. Keyed on
        # a write generation, so any write makes every entry unreachable.
        ck = self._rcache.key(query, partition, self.clock.now(),
                              budget=budget, as_of=as_of,
                              group_by=group_by, per_group=per_group,
                              allow_inferred=allow_inferred,
                              token_budget=token_budget)
        cached = self._rcache.get(ck)
        if cached is not None:
            return cached

        rec = self._recall(
            query, budget=budget, partition=partition, as_of=as_of,
            suppress_affect_above=suppress_affect_above,
            allow_inferred=allow_inferred, group_by=group_by,
            per_group=per_group, token_budget=token_budget)

        missing: list[str] = list(rec.degraded)
        if self.embedder is None:
            missing.append("semantic (no embedder; lexical only)")
        elif self._embed_failed_this_call:
            missing.append("semantic (embedder raised; lexical only)")
        if self.readonly:
            # Recall normally REINFORCES what it returns -- retrieval is a
            # memory event, not a passive read. Read-only cannot do that, so
            # the answer is correct but this recall will not have happened
            # as far as the store is concerned.
            missing.append("reinforcement (read-only; this recall is not "
                           "recorded)")
            if getattr(self._s, "immutable", False):
                # Fell back past WAL because the media would not accept even
                # the -shm file. Correct, but frozen at open time -- and a
                # stale answer that looks live is the failure mode this
                # whole engine argues against.
                missing.append("liveness (immutable snapshot; anything "
                               "written since this store was opened is "
                               "invisible)")
        if missing:
            rec.degraded = tuple(dict.fromkeys(missing))
            rec.reason += f" [degraded: {'; '.join(rec.degraded)}]"

        if self.defer_embedding and not rec.pending:
            waiting = self.pending()
            if waiting:
                rec.pending = waiting
                rec.reason += (f" [{waiting} captured but not yet embedded -- "
                               "this answer is provisional; call absorb()]")
        # Only cache full-strength answers. A degraded one records a
        # transient condition -- a dead embedder, a half-drained queue --
        # and serving it again after the condition clears would turn a
        # temporary problem into a sticky one.
        # B8: only ever on DONT_KNOW, and only from material strong on its
        # own terms. Attached in its own field so it cannot be mistaken for
        # an answer.
        if rec.state is State.DONT_KNOW and not rec.chunks:
            rec.adjacent = self._adjacent(query, partition)
            if rec.adjacent:
                from .adjacent import Suggestion
                rec.reason += (" | " + Suggestion(list(rec.adjacent)
                                                  ).sentence())

        if rec.full_strength:
            self._rcache.put(ck, rec)
        return rec

    def _adjacent(self, query: str, partition: str) -> tuple:
        """Neighbouring material for a query that came back empty."""
        from . import adjacent as adj
        visible = self._s.readable_from(partition)
        terms = lexical.tokenize(query)
        if not terms:
            return ()
        cands = self._lexical_candidates(terms, visible,
                                         self._count_nodes(visible))
        rows = []
        for nid, score in list(cands.items())[:30]:
            r = self._node_row(nid)
            if r is not None:
                rows.append({"node_id": nid, "content": r["content"],
                             "score": score})
        return tuple(adj.find(query, rows).items)

    def _recall(self, query: str, *, budget: int = 5, partition: str = "default",
                as_of: float | None = None, suppress_affect_above: float = 1.01,
                allow_inferred: bool = True, group_by: str | None = "source",
                per_group: int = 2, token_budget: int | None = None) -> Recall:
        """Retrieve. ALWAYS check `.state` before consuming `.chunks`.

        `as_of` queries the bitemporal record: "what did we hold to be true at
        time T", using world-time validity, not ingestion order.

        `suppress_affect_above` lets a companion context decline to surface
        distressing material unbidden. It filters presentation; it never
        deletes, and the memory remains addressable by an explicit request.
        """
        t0 = time.perf_counter()
        budget = max(1, min(budget, MAX_BUDGET))
        visible = self._s.readable_from(partition)   # {partition: level}
        terms = lexical.tokenize(query)
        if not terms:
            return Recall(State.DONT_KNOW, [], query, "empty query")

        n_docs = self._count_nodes(visible)
        # G1: skip the posting-list scan when the vocabulary filter proves
        # it would return nothing. Everything else below still runs -- the
        # semantic tier especially, since a paraphrase has no lexical
        # overlap with its target by construction.
        cands = ({} if self._definitely_unknown(terms, visible)
                 else self._lexical_candidates(terms, visible, n_docs))
        best_sim = 0.0
        if self.embedder is not None:
            cands, best_sim = self._blend_semantic(query, cands, visible)
        cands = self._entity_bridge(cands, query, visible)
        cands = self._expand_assoc(cands, visible)

        now = self.clock.now()
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for node_id, base in cands.items():
            row = self._node_row(node_id)
            if row is None or row["partition"] not in visible:
                continue
            if (visible[row["partition"]] == "summary"
                    and row["kind"] not in ("summary", "abstraction", "community")):
                continue          # graded permeability: raw detail does not cross
            if as_of is not None and not self._valid_at(row, as_of):
                continue
            if (row["affect"] or 0.0) > suppress_affect_above:
                continue
            if not allow_inferred and row["epistemic"] not in (
                    "observed", "reported"):
                continue
            idx = self._s.one(
                "SELECT * FROM mem_index WHERE node_id=?", (node_id,))
            if idx is None or idx["tier"] == "pruned":
                # Pruned means gone from the hot index. The lexical path
                # filters these; the entity bridge and associative spread did
                # not, so pruned content leaked back in through association.
                continue
            if idx["suppressed_at"]:
                base *= 0.02          # demotion, never an exclusion filter
            r = salience.retrievability(now - idx["last_review"], idx["stability"])
            sc = base * salience.salience(
                stability=idx["stability"], difficulty=idx["difficulty"],
                elapsed=now - idx["last_review"], surprise=idx["surprise"],
                open_loop=bool(idx["open_loop"]),
                acquisition_cost=(row["acquisition_cost"]
                                  if "acquisition_cost" in row.keys() else 0.0),
                criticality=self.criticality_of(row["id"]))
            scored.append((sc, r, row))

        # Answer-type affinity: a question asking WHO should prefer text that
        # actually contains a person. A mild multiplier, never a filter --
        # the predictor is a regex heuristic.
        want = entities.predict_answer_type(query)
        if want:
            scored = [(sc * entities.content_affinity(row["content"], want),
                       r, row) for sc, r, row in scored]
        # Exactness, where similarity is the wrong question entirely. A query
        # naming GX-4419 must not be answered with GX-4491 -- the two are
        # near-identical as vectors and different as facts, and the wrong
        # one is worse than nothing because it looks right.
        exact = epistemics.verbatim_tokens(query)
        if exact:
            scored = [(sc * epistemics.exact_match_factor(row["content"],
                                                          exact), r, row)
                      for sc, r, row in scored]
        # A TOTAL order, and it has to be. Score alone leaves ties, and a
        # stable sort then preserves whatever order the candidates happened
        # to arrive in -- which is the order SQLite returned rows in, which
        # is a property of WHICH INDEX THE PLANNER CHOSE. G5 changed that
        # index and the returned chunks reordered, with every score
        # identical: the ranking had a dependency on the physical plan and
        # nobody knew, because nothing had ever changed the plan before.
        #
        # Whatever breaks the tie must be derived from the MEMORY ITSELF,
        # or two stores built from the same material rank differently and
        # the fix is no fix. That rules out more than it looks like:
        #
        #   node_id     - a fresh uuid per store
        #   observed_at - WALL CLOCK, and the first attempt here. It reads
        #                 as content-derived and is not: two memories
        #                 written in the same loop land on the same
        #                 time.time() tick in one run and separate ticks in
        #                 the next, so the tie exists in one store and not
        #                 the other. Caught by the same test, one round
        #                 later, which is the argument for that test.
        #
        # Content it is. Alphabetical is arbitrary -- but it is consulted
        # ONLY between candidates whose scores are exactly equal, where
        # every order is arbitrary and the sole virtue available is being
        # the same one twice.
        scored.sort(key=lambda x: (-x[0], x[2]["content"]))
        covered = self._coverage(terms, visible)
        raw = [s for s, _, _ in scored]
        sig = metamemory.Signals(
            coverage=covered,
            best_score=min(1.0, raw[0]) if raw else 0.0,
            best_retrievability=scored[0][1] if scored else 0.0,
            conflict=metamemory.conflict_ratio(raw[:12]),
            n_candidates=len(scored),
            semantic_density=metamemory.semantic_density(
                best_sim, getattr(self.embedder, "noise_floor", SEMANTIC_FLOOR),
                getattr(self, "_last_background", None),
                getattr(self.embedder, "level_weight",
                        metamemory.LEVEL_WEIGHT),
                getattr(self.embedder, "ceiling", 1.0)),
            recollection=(self._recollection(scored[0][2]) if scored else 0.0),
            answer_type=entities.predict_answer_type(query),
            has_answer_type=self._has_entity_kind(
                entities.predict_answer_type(query), visible),
        )
        state, reason = metamemory.triage(sig)

        # A recorded absence OUTRANKS a weak positive match. If I canvassed all
        # six vendors for diesel and found none, that answer must survive the
        # existence of a tangentially-related note that happens to share the
        # word "Bardera" -- otherwise the expensive knowledge is buried by
        # lexical noise and gets re-derived forever.
        absent = self._absence_answer(query, partition, t0)
        if absent is not None and state is not State.KNOW:
            absent.chunks = [
                Chunk(node_id=row["id"], content=row["content"], score=float(sc),
                      retrievability=float(r),
                      claim_class=row["claim_class"],
                      provenance=self._provenance(row))
                for sc, r, row in scored[:2]
            ]
            return absent

        if state is State.DONT_KNOW:
            # Before answering "nothing", check the thing every other memory
            # system is structurally unable to check, because they delete rows.
            ans = self._knew_once(terms, visible, query, t0)
            if ans is not None:
                return ans
            # "I have nothing" is honest but inert. Say what would have to
            # exist, so the dead end becomes a task.
            gap = self._explain_gap(query, terms, visible)
            rec = Recall(State.DONT_KNOW, [], query,
                         f"{reason}. {gap}" if gap else reason,
                         (time.perf_counter() - t0) * 1000)
            if self.receipts:
                self._write_receipt(rec, partition, scored[:5])
            return rec

        take = budget if state is not State.TIP_OF_TONGUE else min(budget + 2,
                                                                   MAX_BUDGET)
        # Diversity by construction, not by reranking. Without this, five
        # chunks routinely come from one document -- which looks like five
        # pieces of evidence and is one.
        picked = self._group_cap(scored, group_by, per_group, take)
        chunks: list[Chunk] = []
        used_tokens = 0
        for sc, r, row in picked:
            cls = row["claim_class"] if "claim_class" in row.keys() else "unknown"
            grade = self.effective_grade(row["id"])
            chunks.append(Chunk(
                node_id=row["id"], content=row["content"], score=float(sc),
                retrievability=float(r), affect=float(row["affect"] or 0.0),
                staleness=epistemics.staleness(
                    now - row["observed_at"], cls,
                    self._halflife(cls)),
                claim_class=cls,
                reliability=grade[0], credibility=grade[1],
                trust=(row["trust"] if "trust" in row.keys() else "trusted"),
                provenance=self._provenance(row),
            ))
            used_tokens += max(1, len(row["content"]) // 4)
            if token_budget is not None and used_tokens >= token_budget:
                break
        # Retrieval is a memory EVENT -- it reinforces what it returns and
        # demotes the competitors. But that is an enhancement to remembering,
        # not a precondition for it, so a store that cannot be written to
        # still answers.
        if not self.readonly:
            self._mark_used([c.node_id for c in chunks])
            # Reinforcement is recall's own side effect. It nudges
            # retrievability; it does not change which memories exist, so
            # it must not invalidate the answer cache -- see
            # SqliteStore.write(affects_answers=...).
        self._last_recalled = [c.node_id for c in chunks]
        rec = Recall(state, chunks, query, reason,
                     (time.perf_counter() - t0) * 1000, tokens=used_tokens)
        if self.receipts and not self.readonly:
            self._write_receipt(rec, partition, scored[len(chunks):len(chunks) + 5])
        return rec

    # ── THEORY OF MIND (epistemic plane only) ────────────────────────
    def tell(self, who: str, node_id: str, *, channel: str = "conversation",
             at: float | None = None) -> None:
        """Record that a PERSON was exposed to this. The core ToM primitive.

        Everything else in this section is derived from the exposure log plus
        the forgetting model you already have, pointed at a different subject.
        """
        ts = at if at is not None else self.clock.now()
        depth = theory_of_mind.CHANNEL_DEPTH.get(channel, 0.8)
        self._s.write(lambda c: c.execute(
            "INSERT INTO exposure(who,node_id,at,channel,depth) "
            "VALUES(?,?,?,?,?)", (who, node_id, ts, channel, depth)))

    def knows(self, who: str, node_id: str) -> Held | None:
        """Model this person's retention of one item. Not the system's."""
        rows = self._s.query(
            "SELECT at,channel FROM exposure WHERE who=? AND node_id=? "
            "ORDER BY at", (who, node_id))
        if not rows:
            return None
        exposures = [(r["at"], r["channel"]) for r in rows]
        r = theory_of_mind.model_retention(exposures, self.clock.now())
        return Held(node_id=node_id, retrievability=r, exposures=len(rows),
                    last_exposed=exposures[-1][0], channel=exposures[-1][1])

    def at_risk(self, who: str, *, threshold: float = 0.4,
                partition: str = "default", limit: int = 10) -> list[Held]:
        """What this person is about to forget, ranked by consequence.

        This inverts spaced repetition. SRS asks the human to review on a
        schedule. Transactive memory just carries the load and surfaces the
        item at the moment of predicted failure -- the way a colleague does.
        """
        rows = self._s.query(
            "SELECT DISTINCT e.node_id FROM exposure e "
            "JOIN mem_index m ON m.node_id=e.node_id "
            "WHERE e.who=? AND m.partition=?", (who, partition))
        out: list[tuple[float, Held]] = []
        now = self.clock.now()
        for r in rows:
            held = self.knows(who, r["node_id"])
            if held is None or held.retrievability >= threshold:
                continue
            obs = self._s.one(
                "SELECT observed_at,claim_class FROM observation WHERE id=?",
                (r["node_id"],))
            # Consequence: a still-true claim they are losing matters more than
            # a stale one they are losing -- the stale one should go.
            still_true = (epistemics.credibility(
                now - obs["observed_at"], obs["claim_class"]) if obs else 0.5)
            out.append((still_true * (1.0 - held.retrievability), held))
        out.sort(key=lambda x: -x[0])
        return [h for _, h in out[:limit]]

    def divergence(self, who: str, *, partition: str = "default"
                   ) -> list[Divergence]:
        """Detect false-belief states. Sally-Anne, made operational.

        The system knows Route Alpha closed on the 14th. It knows the person
        was told it was open on the 12th. It knows they have not been exposed
        to the update. Therefore it can COMPUTE that they are about to act on
        a false belief -- from the exposure log and the bitemporal record,
        with no model call.

        Resolution is symmetric: the person is frequently the one holding
        better information, because they were there. See
        theory_of_mind.resolve_direction.
        """
        now = self.clock.now()
        out: list[Divergence] = []
        rows = self._s.query(
            "SELECT s.old_node,s.new_node,s.claim_class,s.at "
            "FROM supersession s JOIN observation o ON o.id=s.new_node "
            "WHERE o.partition=?", (partition,))
        for r in rows:
            held = self.knows(who, r["old_node"])
            if held is None:
                continue
            if self.knows(who, r["new_node"]) is not None:
                continue                        # they have seen the update
            new = self._s.one(
                "SELECT observed_at,origin,reliability,credibility,content "
                "FROM observation WHERE id=?", (r["new_node"],))
            old = self._s.one(
                "SELECT observed_at,origin FROM observation WHERE id=?",
                (r["old_node"],))
            direction, note = theory_of_mind.resolve_direction(
                user_source_recency=now - held.last_exposed,
                user_was_present=(old["origin"] == "user_utterance"
                                  and held.channel in ("generated", "recall")),
                ledger_recency=now - new["observed_at"],
                ledger_admiralty=epistemics.admiralty_weight(
                    new["reliability"], new["credibility"]),
            )
            sev = theory_of_mind.divergence_severity(
                user_retention=held.retrievability,
                claim_staleness=epistemics.staleness(
                    now - old["observed_at"], r["claim_class"]))
            out.append(Divergence(who=who, held_node=r["old_node"],
                                  truth_node=r["new_node"], direction=direction,
                                  severity=sev, note=note))
        out.sort(key=lambda d: -d.severity)
        return out

    # ── SOURCE INDEPENDENCE ──────────────────────────────────────────
    def independent_sources(self, node_ids: Sequence[str]) -> dict:
        """How many genuinely distinct origins are behind these nodes?

        Corroboration must count ORIGINS, not documents. Forty files from one
        upstream source are one source -- and treating them as forty is
        exactly the source-flooding attack.
        """
        clusters: list[str] = []
        refs: list[str] = []
        for nid in node_ids:
            row = self._node_row(nid)
            if row is None:
                continue
            refs.append(row["source_ref"])
            o = self._s.one("SELECT origin_cluster FROM source_origin WHERE "
                            "source_ref=?", (row["source_ref"],))
            clusters.append(o["origin_cluster"] if o
                            else attribution.origin_key(row["source_ref"]))
        n = attribution.independence(clusters)
        return {
            "documents": len(refs), "independent": n,
            "clusters": sorted(set(clusters)),
            "weight": attribution.corroboration_weight(n),
            "note": ("single origin -- no corroboration credit"
                     if n <= 1 else f"{n} independent origins"),
        }

    def corroborated(self, text: str, *, partition: str = "default") -> dict:
        """Who else says this, and are they actually independent?"""
        h = attribution.proposition_hash(text)
        nodes = [r["node_id"] for r in self._s.query(
            "SELECT DISTINCT node_id FROM claim WHERE prop_hash=?", (h,))]
        for r in self._s.query(
                "SELECT id,content FROM observation WHERE partition=?",
                (partition,)):
            if attribution.proposition_hash(r["content"]) == h:
                nodes.append(r["id"])
        out = self.independent_sources(sorted(set(nodes)))
        out["proposition"] = text
        return out

    # ── ATTRIBUTED BELIEF ────────────────────────────────────────────
    def claimant(self, name: str, *, partition: str = "default",
                 kind: str = "person") -> str:
        canon = attribution.canonical_name(name)
        if not canon:
            raise ValueError("claimant name is empty after canonicalisation")
        row = self._s.one(
            "SELECT id FROM claimant WHERE partition=? AND canonical=?",
            (partition, canon))
        if row:
            return row["id"]
        cid = f"who_{uuid.uuid4().hex[:12]}"
        self._s.write(lambda c: c.execute(
            "INSERT INTO claimant(id,partition,name,canonical,kind,first_seen)"
            " VALUES(?,?,?,?,?,?)",
            (cid, partition, name, canon, kind, self.clock.now())))
        return cid

    def claimed(self, who: str, proposition: str, *, node_id: str,
                partition: str = "default") -> str:
        """Record that a PERSON asserted something.

        Distinct from the proposition itself. Multiple claimants asserting one
        proposition is corroboration; a claimant accumulates a record from
        outcomes; and when that record moves, everything they said revalues.
        """
        cid = self.claimant(who, partition=partition)
        clid = f"clm_{uuid.uuid4().hex[:12]}"

        def _w(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO claim(id,claimant_id,node_id,proposition,"
                "prop_hash,asserted_at) VALUES(?,?,?,?,?,?)",
                (clid, cid, node_id, proposition,
                 attribution.proposition_hash(proposition), self.clock.now()))
            c.execute("UPDATE claimant SET claims_made=claims_made+1 WHERE id=?",
                      (cid,))

        self._s.write(_w)
        return clid

    def record_of(self, who: str, *, partition: str = "default") -> Record:
        """A claimant's track record, learned from outcomes."""
        canon = attribution.canonical_name(who)
        r = self._s.one(
            "SELECT * FROM claimant WHERE partition=? AND canonical=?",
            (partition, canon))
        if r is None:
            return Record(who, 0, 0, 0, 0, 0)
        return Record(r["name"], r["claims_made"], r["claims_confirmed"],
                      r["claims_refuted"], r["kept_count"], r["broken_count"])

    def who_claims(self, proposition: str, *, partition: str = "default"
                   ) -> list[dict]:
        h = attribution.proposition_hash(proposition)
        out = []
        for r in self._s.query(
                "SELECT c.name,cl.node_id,cl.asserted_at,cl.outcome "
                "FROM claim cl JOIN claimant c ON c.id=cl.claimant_id "
                "WHERE cl.prop_hash=? AND c.partition=?", (h, partition)):
            rec = self.record_of(r["name"], partition=partition)
            out.append({"who": r["name"], "node": r["node_id"],
                        "at": r["asserted_at"], "outcome": r["outcome"],
                        "grade": rec.grade, "accuracy": rec.accuracy})
        return out

    def resolve_claim(self, claim_id: str, *, confirmed: bool) -> Record | None:
        """An outcome lands. The claimant's record moves, and if it moves far
        enough, everything they said is revalued."""
        row = self._s.one("SELECT claimant_id FROM claim WHERE id=?", (claim_id,))
        if row is None:
            return None
        col = "claims_confirmed" if confirmed else "claims_refuted"

        def _w(c: sqlite3.Connection) -> None:
            c.execute("UPDATE claim SET outcome=? WHERE id=?",
                      ("confirmed" if confirmed else "refuted", claim_id))
            c.execute(f"UPDATE claimant SET {col}={col}+1 WHERE id=?",
                      (row["claimant_id"],))

        self._s.write(_w)
        return self._revalue_claimant(row["claimant_id"])

    # ── COMMITMENTS ──────────────────────────────────────────────────
    def committed(self, who: str, statement: str, *, due: float,
                  node_id: str, partition: str = "default") -> str:
        """A promise is not a fact. It has a lifecycle and a deadline.

        Surfaces automatically when due (prospective memory), and its outcome
        feeds the claimant's record -- which is the loop nobody closes.
        """
        cid = self.claimant(who, partition=partition)
        mid = f"cmt_{uuid.uuid4().hex[:12]}"
        self._s.write(lambda c: c.execute(
            "INSERT INTO commitment(id,partition,claimant_id,node_id,statement,"
            "made_at,due_at) VALUES(?,?,?,?,?,?,?)",
            (mid, partition, cid, node_id, statement, self.clock.now(), due)))
        self.intend(f"check: {statement}", when=due, partition=partition,
                    origin_ref=f"commitment:{mid}")
        return mid

    def due_commitments(self, *, partition: str = "default") -> list[dict]:
        now = self.clock.now()
        out = []
        for r in self._s.query(
                "SELECT m.id,m.statement,m.due_at,m.status,c.name "
                "FROM commitment m JOIN claimant c ON c.id=m.claimant_id "
                "WHERE m.partition=? AND m.status IN ('open','due') "
                "AND m.due_at<=? ORDER BY m.due_at", (partition, now)):
            out.append({"id": r["id"], "who": r["name"],
                        "statement": r["statement"], "due_at": r["due_at"],
                        "overdue_days": round((now - r["due_at"]) / DAY, 1)})
        return out

    def resolve_commitment(self, commitment_id: str, *, kept: bool,
                           note: str = "") -> Record | None:
        """Kept or broken. A broken promise degrades the claimant's
        reliability, and that automatically revalues everything they ever
        told you."""
        row = self._s.one("SELECT claimant_id FROM commitment WHERE id=?",
                          (commitment_id,))
        if row is None:
            return None
        col = "kept_count" if kept else "broken_count"
        now = self.clock.now()

        def _w(c: sqlite3.Connection) -> None:
            c.execute("UPDATE commitment SET status=?,resolved_at=?,note=? "
                      "WHERE id=?",
                      ("kept" if kept else "broken", now, note, commitment_id))
            c.execute(f"UPDATE claimant SET {col}={col}+1 WHERE id=?",
                      (row["claimant_id"],))
            c.execute("UPDATE intention SET status='completed' WHERE "
                      "origin_ref=?", (f"commitment:{commitment_id}",))

        self._s.write(_w)
        return self._revalue_claimant(row["claimant_id"])

    def _revalue_claimant(self, claimant_id: str) -> Record:
        """Push a claimant's learned grade onto every source they spoke through.

        This is where the loop closes. The grade is DERIVED from outcomes, so
        nothing is configured; and because assessments are keyed by source,
        one write revalues everything that claimant ever said.
        """
        r = self._s.one("SELECT * FROM claimant WHERE id=?", (claimant_id,))
        rec = Record(r["name"], r["claims_made"], r["claims_confirmed"],
                     r["claims_refuted"], r["kept_count"], r["broken_count"])
        if rec.resolved < 3:
            return rec                     # not enough history to judge
        refs = {row["source_ref"] for row in self._s.query(
            "SELECT DISTINCT o.source_ref FROM observation o JOIN claim cl "
            "ON cl.node_id=o.id WHERE cl.claimant_id=?", (claimant_id,))}
        refs |= {row["source_ref"] for row in self._s.query(
            "SELECT DISTINCT o.source_ref FROM observation o JOIN commitment m "
            "ON m.node_id=o.id WHERE m.claimant_id=?", (claimant_id,))}
        for ref in refs:
            # Credibility tracks the grade explicitly. An earlier version
            # used `3 if grade <= "C" else 4`, which is a *string* compare:
            # "E" <= "C" is False, so the worst grades silently received the
            # better credibility. Lexical ordering is not epistemic ordering.
            cred = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}[rec.grade]
            self.assess_source(ref, reliability=rec.grade, credibility=cred,
                               reason=rec.describe())
        return rec

    # ── THE FORWARD DIRECTION ────────────────────────────────────────
    def decided(self, statement: str, *, because: Sequence[str],
                partition: str = "default", by: str = "user",
                reversible_until: float | None = None) -> str:
        """Record a decision and what it rests on.

        This is the piece no memory system has. Without it, superseding a
        fact updates the fact and stops -- and nobody learns that a choice
        made three days ago is now standing on a false premise.
        """
        if not because:
            raise ValueError(
                "a decision with no basis cannot be revisited when its basis "
                "changes; record what it rested on, even if only one thing")
        did = f"dec_{uuid.uuid4().hex[:14]}"
        now = self.clock.now()
        missing = [n for n in because if self._node_row(n) is None]
        if missing:
            raise OwlError(f"unknown basis node(s): {missing}")

        def _w(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO decision(id,partition,statement,decided_at,"
                "decided_by,reversible_until) VALUES(?,?,?,?,?,?)",
                (did, partition, statement, now, by, reversible_until))
            c.executemany(
                "INSERT OR IGNORE INTO decision_basis(decision_id,node_id,weight)"
                " VALUES(?,?,1.0)", [(did, n) for n in because])

        self._s.write(_w)
        return did

    def execute_decision(self, decision_id: str, *, outcome: str = "") -> None:
        """Mark a decision carried out. It stops being reversible."""
        self._s.write(lambda c: c.execute(
            "UPDATE decision SET status='executed',outcome=?,resolved_at=?,"
            "reversible_until=NULL WHERE id=?",
            (outcome, self.clock.now(), decision_id)))

    def resolve_impact(self, impact_id: str, *, status: str = "reaffirmed",
                       outcome: str = "") -> None:
        """Acknowledge an impact. Persisted so it is not re-raised forever --
        a system that repeats the same warning every session teaches people
        to ignore warnings."""
        now = self.clock.now()

        def _w(c: sqlite3.Connection) -> None:
            row = c.execute("SELECT decision_id FROM decision_impact WHERE id=?",
                            (impact_id,)).fetchone()
            c.execute("UPDATE decision_impact SET acknowledged_at=? WHERE id=?",
                      (now, impact_id))
            if row:
                c.execute("UPDATE decision SET status=?,outcome=?,resolved_at=? "
                          "WHERE id=?", (status, outcome, now, row["decision_id"]))

        self._s.write(_w)

    def affected_by(self, node_id: str, *, cause: str = "superseded"
                    ) -> list[Impact]:
        """Which decisions rest on this memory? The forward query.

        Does not write. Use `raise_impacts` to persist, which happens
        automatically on supersession.
        """
        c = Cause(cause)
        crit = self.criticality_of(node_id)
        now = self.clock.now()
        out: list[Impact] = []
        for r in self._s.query(
                "SELECT d.id,d.statement,d.reversible_until,d.status,b.weight "
                "FROM decision d JOIN decision_basis b ON b.decision_id=d.id "
                "WHERE b.node_id=? AND d.status IN "
                "('standing','revisit','executed')", (node_id,)):
            reversible = (r["reversible_until"] is None
                          or r["reversible_until"] > now) and \
                         r["status"] != "executed"
            out.append(Impact(
                decision_id=r["id"], statement=r["statement"],
                basis_node=node_id, cause=c, reversible=bool(reversible),
                severity=decisions.severity(
                    criticality=crit, weight=r["weight"], cause=c,
                    reversible=bool(reversible)),
                detected_at=now,
                note=("still reversible -- act on this"
                      if reversible else "already executed -- log, do not alarm"),
            ))
        # Executed decisions still surface -- "the convoy already crossed the
        # bridge that has since collapsed" is exactly what after-action review
        # needs. They are simply never `urgent`, because urgency implies
        # something can still be done.
        out.sort(key=lambda i: (not i.reversible, -i.severity))
        return out

    def raise_impacts(self, node_id: str, *, cause: str = "superseded"
                      ) -> list[Impact]:
        """Detect and persist impacts. Called automatically on supersession."""
        found = self.affected_by(node_id, cause=cause)
        if not found:
            return []
        now = self.clock.now()

        def _w(c: sqlite3.Connection) -> None:
            for imp in found:
                dupe = c.execute(
                    "SELECT 1 FROM decision_impact WHERE decision_id=? AND "
                    "basis_node=? AND cause=? AND acknowledged_at IS NULL",
                    (imp.decision_id, node_id, imp.cause.value)).fetchone()
                if dupe:
                    continue
                c.execute(
                    "INSERT INTO decision_impact(id,decision_id,basis_node,"
                    "cause,detected_at,severity) VALUES(?,?,?,?,?,?)",
                    (f"imp_{uuid.uuid4().hex[:12]}", imp.decision_id, node_id,
                     imp.cause.value, now, imp.severity))
                c.execute("UPDATE decision SET status='revisit' WHERE id=? AND "
                          "status='standing'", (imp.decision_id,))
                # 'executed' is deliberately NOT downgraded to 'revisit':
                # the decision is history, the impact is the record of it.

        self._s.write(_w)
        return found

    def reconsider(self, *, partition: str = "default",
                   urgent_only: bool = False) -> list[Impact]:
        """Every decision whose basis has moved and that has not been resolved.

        The morning question: what am I standing on that has shifted?
        """
        out: list[Impact] = []
        for r in self._s.query(
                "SELECT i.id,i.decision_id,i.basis_node,i.cause,i.detected_at,"
                "i.severity,d.statement,d.reversible_until,d.status "
                "FROM decision_impact i JOIN decision d ON d.id=i.decision_id "
                "WHERE i.acknowledged_at IS NULL AND d.partition=? "
                "ORDER BY i.severity DESC", (partition,)):
            now = self.clock.now()
            reversible = (r["reversible_until"] is None
                          or r["reversible_until"] > now) and \
                         r["status"] != "executed"
            imp = Impact(
                decision_id=r["decision_id"], statement=r["statement"],
                basis_node=r["basis_node"], cause=Cause(r["cause"]),
                severity=r["severity"], reversible=bool(reversible),
                detected_at=r["detected_at"], impact_id=r["id"],
                note=("still reversible -- act on this" if reversible
                      else "already executed -- log, do not alarm"))
            if urgent_only and not imp.urgent:
                continue
            out.append(imp)
        return out

    # ── blast radius & revaluation ───────────────────────────────────
    def blast_radius(self, node_id: str, *, include_decisions: bool = True
                     ) -> dict:
        """What do I believe BECAUSE of this? The inverse of `why()`.

        `why()` walks backward to sources. Nothing in the field walks forward.
        So when a document turns out to be forged, or a source was lying, or
        an ingest batch mis-parsed, every other system leaves the contamination
        sitting in the store: the summaries stay, the composites stay, the
        conclusions stay -- now unsourced but still confident.

        Transitive forward closure over derivation edges, composite membership
        and decision bases.
        """
        seen: set[str] = set()
        layers: list[list[str]] = []
        frontier = [node_id]
        while frontier and len(layers) < 12:
            nxt: list[str] = []
            for n in frontier:
                for r in self._s.query(
                        "SELECT child_id FROM derivation_edge WHERE parent_id=?",
                        (n,)):
                    if r["child_id"] not in seen and r["child_id"] != node_id:
                        seen.add(r["child_id"])
                        nxt.append(r["child_id"])
                for r in self._s.query(
                        "SELECT composite_id FROM composite_member WHERE "
                        "member_id=?", (n,)):
                    if r["composite_id"] not in seen:
                        seen.add(r["composite_id"])
                        nxt.append(r["composite_id"])
            if nxt:
                layers.append(nxt)
            frontier = nxt

        decs: list[dict] = []
        if include_decisions:
            for nid in [node_id, *seen]:
                for r in self._s.query(
                        "SELECT d.id,d.statement,d.status FROM decision d "
                        "JOIN decision_basis b ON b.decision_id=d.id "
                        "WHERE b.node_id=?", (nid,)):
                    decs.append({"id": r["id"], "statement": r["statement"],
                                 "status": r["status"], "via": nid})

        told: list[dict] = []
        for nid in [node_id, *seen]:
            for r in self._s.query(
                    "SELECT DISTINCT who FROM exposure WHERE node_id=?", (nid,)):
                told.append({"who": r["who"], "node": nid})

        return {
            "root": node_id,
            "derived": sorted(seen),
            "depth": len(layers),
            "decisions": decs,
            "told": told,
            "count": len(seen),
        }

    def discredit(self, node_id: str, *, reason: str,
                  reliability: str = "E", dry_run: bool = False) -> dict:
        """Revalue a source downward and cascade through everything it touched.

        This is what makes provenance PAY. Everyone else's provenance is
        decorative because nothing acts on it. Here the same monotonicity
        clamp that runs on write re-runs on revaluation and cascades.

        Nothing is deleted. Having once believed something is itself evidence,
        and a system that erases its own errors cannot be audited.
        """
        radius = self.blast_radius(node_id)
        now = self.clock.now()
        penalty = epistemics.RELIABILITY.get(reliability, 0.2)
        plan = {"root": node_id, "reason": reason, "demoted": [],
                "quarantined": [], "decisions_flagged": len(radius["decisions"]),
                "people_to_notify": sorted({t["who"] for t in radius["told"]})}

        for nid in radius["derived"]:
            row = self._node_row(nid)
            if row is None or not nid.startswith("der_"):
                continue
            new_conf = round(float(row["confidence"]) * penalty, 4)
            cur = Epistemic(row["epistemic"])
            new_epi = cur if cur.rank >= Epistemic.HYPOTHESIZED.rank else \
                Epistemic(list(Epistemic)[min(cur.rank + 1, 3)])
            if new_conf < 0.15:
                plan["quarantined"].append(nid)
            else:
                plan["demoted"].append(
                    {"node": nid, "confidence": new_conf,
                     "epistemic": new_epi.value})

        if dry_run:
            return plan

        row0 = self._node_row(node_id)
        src_ref = row0["source_ref"] if row0 else node_id

        def _w(c: sqlite3.Connection) -> None:
            # NOT an update to the observation: what we believed at ingest is
            # history and stays. The current assessment is a separate layer,
            # keyed by source so one write revalues every observation from it.
            if node_id.startswith("obs_"):
                c.execute(
                    "INSERT INTO source_assessment(source_ref,reliability,"
                    "credibility,reason,assessed_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(source_ref) DO UPDATE SET reliability=?,"
                    "reason=?,assessed_at=?",
                    (src_ref, reliability, 5, reason, now,
                     reliability, reason, now))
            else:
                c.execute("UPDATE derived SET confidence=confidence*? WHERE id=?",
                          (penalty, node_id))
            for d in plan["demoted"]:
                c.execute("UPDATE derived SET confidence=?,epistemic_tag=? "
                          "WHERE id=?", (d["confidence"], d["epistemic"],
                                         d["node"]))
            for nid in plan["quarantined"]:
                c.execute("UPDATE derived SET confidence=0.05,"
                          "epistemic_tag='hypothesized' WHERE id=?", (nid,))
                c.execute("UPDATE mem_index SET tier='cold' WHERE node_id=?",
                          (nid,))
            c.execute(
                "INSERT INTO absence(id,partition,query,scope,searched_at,"
                "reason) VALUES(?,?,?,?,?,?)",
                (f"abs_{uuid.uuid4().hex[:12]}",
                 (self._node_row(node_id) or {"partition": "default"})["partition"],
                 f"discredited:{node_id}", "revaluation", now,
                 f"source discredited: {reason}"))

        self._s.write(_w)
        for nid in [node_id, *radius["derived"]]:
            self.raise_impacts(nid, cause="discredited")
        return plan

    # ── criticality ──────────────────────────────────────────────────
    def recompute_criticality(self) -> int:
        """Reverse PageRank over derivation + decision edges.

        Answers "which beliefs carry the most weight" -- verification triage,
        fragility detection, and a far better forgetting signal than access
        count.
        """
        edges = [(r["child_id"], r["parent_id"]) for r in self._s.query(
            "SELECT child_id,parent_id FROM derivation_edge")]
        dec = [(r["decision_id"], r["node_id"]) for r in self._s.query(
            "SELECT decision_id,node_id FROM decision_basis")]
        nodes = {r["node_id"] for r in self._s.query(
            "SELECT node_id FROM mem_index")}
        scores = decisions.criticality(edges, dec, nodes)
        counts = decisions.dependent_counts(edges, dec)
        now = self.clock.now()
        rows = [(n, sc, counts.get(n, (0, 0))[0], counts.get(n, (0, 0))[1], now)
                for n, sc in scores.items() if not n.startswith("dec_")]
        if rows:
            self._s.write(lambda c: c.executemany(
                "INSERT OR REPLACE INTO criticality(node_id,score,dependents,"
                "decisions,computed_at) VALUES(?,?,?,?,?)", rows))
        return len(rows)

    def assess_source(self, source_ref: str, *, reliability: str,
                      credibility: int = 3, reason: str = "") -> None:
        """Set the CURRENT assessment of a source. Revisable; the ingest-time
        grade on each observation is history and never changes."""
        self._s.write(lambda c: c.execute(
            "INSERT INTO source_assessment(source_ref,reliability,credibility,"
            "reason,assessed_at) VALUES(?,?,?,?,?) ON CONFLICT(source_ref) "
            "DO UPDATE SET reliability=?,credibility=?,reason=?,assessed_at=?",
            (source_ref, reliability, credibility, reason, self.clock.now(),
             reliability, credibility, reason, self.clock.now())))

    def effective_grade(self, node_id: str) -> tuple[str, int]:
        """Current trust in a node's source, not the grade assigned at ingest."""
        row = self._node_row(node_id)
        if row is None:
            return ("F", 6)
        a = self._s.one(
            "SELECT reliability,credibility FROM source_assessment WHERE "
            "source_ref=?", (row["source_ref"],))
        if a is not None:
            return (a["reliability"], a["credibility"])
        return (row["reliability"], row["credibility"])

    def criticality_of(self, node_id: str) -> float:
        """Effective criticality, never dependent on whether `tend()` has run.

        The stored PageRank score is the refined answer, but severity must not
        be lower merely because a background job has not fired yet -- that
        made a superseded convoy-routing basis score 0.375 and fall below the
        alarm threshold, which is exactly backwards.

        So: a direct floor. A memory that a DECISION rests on is load-bearing
        by definition; you do not need PageRank to tell you that.
        """
        stored = self._s.one("SELECT score FROM criticality WHERE node_id=?",
                             (node_id,))
        score = float(stored["score"]) if stored else 0.0
        n_dec = self._scalar(
            "SELECT COUNT(*) FROM decision_basis b JOIN decision d "
            "ON d.id=b.decision_id WHERE b.node_id=? AND d.status<>'reversed'",
            (node_id,))
        n_dep = self._scalar(
            "SELECT COUNT(*) FROM derivation_edge WHERE parent_id=?", (node_id,))
        floor = 0.0
        if n_dec:
            floor = min(1.0, 0.6 + 0.15 * (n_dec - 1))
        elif n_dep:
            floor = min(0.5, 0.15 * n_dep)
        return max(score, floor)

    def verification_queue(self, *, partition: str = "default",
                           limit: int = 10) -> list[dict]:
        """Where to spend verification effort.

        Load-bearing AND weakly attested. "These four beliefs carry most of
        your conclusions; two are single-sourced grade-C; verify those first."
        No other memory system offers this.
        """
        out: list[dict] = []
        for r in self._s.query(
                "SELECT c.node_id,c.score,c.dependents,c.decisions,"
                "o.content,o.source_ref FROM criticality c "
                "JOIN observation o ON o.id=c.node_id "
                "WHERE o.partition=? ORDER BY c.score DESC LIMIT 200",
                (partition,)):
            # Current assessment, not the grade assigned at ingest -- a source
            # discredited yesterday must show as weak today.
            rel, cred = self.effective_grade(r["node_id"])
            weakness = 1.0 - epistemics.admiralty_weight(rel, cred)
            single = 1.0 if r["dependents"] else 0.5
            out.append({
                "node_id": r["node_id"], "content": r["content"][:90],
                "criticality": r["score"], "dependents": r["dependents"],
                "decisions": r["decisions"], "source": r["source_ref"],
                "grade": f"{rel}/{cred}",
                "priority": round(r["score"] * weakness * single, 4),
            })
        out.sort(key=lambda x: -x["priority"])
        return out[:limit]

    # ── HETEROGENEOUS GRAPH ──────────────────────────────────────────
    def link(self, node_id: str, *, mentions: Sequence[tuple[str, str]] = (),
             relations: Sequence[tuple[str, str, str]] = (),
             partition: str = "default") -> dict[str, str]:
        """Attach entities and relations to an observation.

        OWL does NOT extract entities -- that needs a model, belongs to the
        host, and most hosts already do it. ATK builds a link chart from every
        message; this is where that work gets reused instead of duplicated.

            mind.link(nid,
                      mentions=[("Dr Warsame", "person"),
                                ("Bardera clinic", "org")],
                      relations=[("Dr Warsame", "runs", "Bardera clinic")])

        Every relation records `node_id` as its evidence, so a retrieved path
        can be traced back to the observations that justify each hop.
        """
        now = self.clock.now()
        ids: dict[str, str] = {}
        named = list(mentions) + [(a, "other") for a, _, b in relations
                                  for a in (a, b)]

        def _w(c: sqlite3.Connection) -> None:
            for name, kind in named:
                canon = entities.canonicalise(name)
                if not canon:
                    continue
                row = c.execute(
                    "SELECT id,kind FROM entity WHERE partition=? AND "
                    "canonical=?", (partition, canon)).fetchone()
                if row is None:
                    eid = f"ent_{uuid.uuid4().hex[:12]}"
                    c.execute(
                        "INSERT INTO entity(id,partition,name,kind,canonical,"
                        "first_seen) VALUES(?,?,?,?,?,?)",
                        (eid, partition, name, kind, canon, now))
                else:
                    eid = row["id"]
                    # Never downgrade a known kind to 'other'.
                    if row["kind"] == "other" and kind != "other":
                        c.execute("UPDATE entity SET kind=? WHERE id=?",
                                  (kind, eid))
                ids[canon] = eid
                c.execute(
                    "INSERT OR IGNORE INTO mention(entity_id,node_id,role) "
                    "VALUES(?,?,'mentions')", (eid, node_id))
            for src, kind, dst in relations:
                a = ids.get(entities.canonicalise(src))
                b = ids.get(entities.canonicalise(dst))
                if a and b and a != b:
                    c.execute(
                        "INSERT OR IGNORE INTO relation(src,dst,kind,"
                        "evidence_node,weight) VALUES(?,?,?,?,1.0)",
                        (a, b, kind, node_id))

        self._s.write(_w)
        return ids

    def entities_in(self, node_id: str) -> list[Entity]:
        return [Entity(r["id"], r["name"], r["kind"]) for r in self._s.query(
            "SELECT e.id,e.name,e.kind FROM entity e JOIN mention m "
            "ON m.entity_id=e.id WHERE m.node_id=?", (node_id,))]

    def paths(self, query: str, *, partition: str = "default",
              max_hops: int = 3, limit: int = 5) -> list[Path]:
        """Relationship paths seeded from entities named in the query.

        Topology does the reasoning; the model only reads the result. That is
        MiniRAG's central claim, and it is the right shape for a slow local
        model -- a path is far denser than the notes it was derived from.
        """
        visible = self._s.readable_from(partition)
        seeds = self._query_entities(query, visible)
        if not seeds:
            return []
        found: list[Path] = []
        for start in seeds[:4]:
            found.extend(self._walk(start, max_hops, visible))
        found.sort(key=lambda p: (len(p), -len(p.evidence)))
        seen: set[tuple] = set()
        out: list[Path] = []
        for p in found:
            key = tuple((s.src, s.kind, s.dst) for s in p.steps)
            if key in seen or not p.steps:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= limit:
                break
        return out

    def _query_entities(self, query: str, visible: dict[str, str]) -> list[str]:
        """Match query text against known entity names. No model needed."""
        toks = set(lexical.tokenize(query))
        if not toks:
            return []
        pq = ",".join("?" * len(visible))
        hits: list[tuple[float, str]] = []
        for r in self._s.query(
                f"SELECT id,name,canonical FROM entity WHERE partition IN ({pq})",
                tuple(visible)):
            et = set(lexical.tokenize(r["canonical"]))
            if not et:
                continue
            overlap = len(et & toks) / len(et)
            if overlap >= 0.5:
                hits.append((overlap, r["id"]))
        hits.sort(key=lambda x: -x[0])
        return [i for _, i in hits]

    def _walk(self, start: str, max_hops: int,
              visible: dict[str, str]) -> list[Path]:
        names = {r["id"]: r["name"] for r in self._s.query("SELECT id,name FROM entity")}
        out: list[Path] = []
        frontier: list[tuple[str, tuple[PathStep, ...]]] = [(start, ())]
        seen = {start}
        for _ in range(max_hops):
            nxt: list[tuple[str, tuple[PathStep, ...]]] = []
            for node, steps in frontier:
                for r in self._s.query(
                        "SELECT src,dst,kind,evidence_node FROM relation "
                        "WHERE src=? OR dst=?", (node, node)):
                    other = r["dst"] if r["src"] == node else r["src"]
                    if other in seen:
                        continue
                    step = PathStep(names.get(node, node), r["kind"],
                                    names.get(other, other), r["evidence_node"])
                    path = steps + (step,)
                    out.append(Path(path))
                    seen.add(other)
                    nxt.append((other, path))
            frontier = nxt
            if not frontier:
                break
        return out

    def _entity_bridge(self, cands: dict[str, float], query: str,
                       visible: dict[str, str], weight: float = 0.55
                       ) -> dict[str, float]:
        """Bridge observations that share an entity but share no words.

        Two field notes five weeks apart about the same person have no lexical
        overlap and may sit far apart in embedding space. One hop through the
        entity connects them. This is the multi-hop case flat retrieval misses.
        """
        seeds = self._query_entities(query, visible)
        if not seeds:
            return cands
        out = dict(cands)
        # Walk the entity graph, not just the mention list. Direct mentions
        # alone would only recover notes that already name the query entity --
        # which lexical search had anyway. The value is in the SECOND hop:
        # Warsame --signed--> cold-chain-log, and the log is named in a note
        # written five weeks later that never mentions Warsame at all.
        level, w = list(seeds), weight
        seen_ent = set(seeds)
        for hop in range(2):
            if not level:
                break
            qmark = ",".join("?" * len(level))
            for r in self._s.query(
                    f"SELECT DISTINCT node_id FROM mention WHERE entity_id "
                    f"IN ({qmark})", tuple(level)):
                nid = r["node_id"]
                out[nid] = max(out.get(nid, 0.0), w)
            nxt: list[str] = []
            for r in self._s.query(
                    f"SELECT src,dst FROM relation WHERE src IN ({qmark}) "
                    f"OR dst IN ({qmark})", tuple(level) * 2):
                for e in (r["src"], r["dst"]):
                    if e not in seen_ent:
                        seen_ent.add(e)
                        nxt.append(e)
            level, w = nxt, w * 0.7
        return out

    # ── HANDOVER ─────────────────────────────────────────────────────
    def prefix(self, *, partition: str = "default", token_budget: int = 400,
               who: str | None = None) -> dict:
        """F3 -- what to put in front of a session before anyone asks.

        This is what makes memory AMBIENT rather than a tool you have to
        remember to reach for. The failure mode of every "inject context"
        feature is that it injects recency, which is almost never what
        matters: the most recent thing is usually the thing you still
        remember.

        Ordered by CONSEQUENCE instead:

          1. decisions whose basis has moved and nobody has looked
          2. commitments that are due or overdue
          3. open loops -- what was in flight when you stopped
          4. load-bearing memories at risk of being forgotten

        Hard token budget, because an unbounded prefix is a context leak
        that quietly degrades every session it is supposed to help. When
        the budget binds, the lower tiers are dropped whole rather than
        truncated -- half an open loop is worse than none, since it reads
        as complete.
        """
        used = 0
        sections: list[dict] = []

        def take(title: str, items: list[str], why: str) -> None:
            nonlocal used
            if not items:
                return
            cost = sum(len(i) // 4 + 2 for i in items) + len(title) // 4
            if used + cost > token_budget:
                return                      # drop the tier, never half of it
            used += cost
            sections.append({"title": title, "why": why, "items": items})

        impacts = self.reconsider(partition=partition)
        take("Standing on shifted ground",
             [f"{i.statement} — {i.cause}"
              + ("  [still reversible]" if i.reversible else "  [EXECUTED]")
              for i in impacts[:5]],
             "a decision's basis changed and nobody has acknowledged it")

        try:
            due = self.due(partition=partition)
        except Exception:                                     # noqa: BLE001
            due = []
        take("Due now", [d.get("action", str(d)) for d in due[:5]],
             "commitments at or past their trigger")

        loops = [r["action"] for r in self._s.query(
            "SELECT action FROM intention WHERE partition=? AND "
            "status='pending' ORDER BY created_at DESC LIMIT 8", (partition,))]
        take("Still in flight", loops[:5], "open loops from last session")

        if who:
            at_risk = self.at_risk(who)
            take(f"{who} is losing",
                 [a["content"][:90] for a in at_risk[:3]],
                 "they were told, and the retention curve says it is fading")

        text = []
        for s in sections:
            text.append(f"## {s['title']}")
            text.append(f"*{s['why']}*")
            text.append("")
            for it in s["items"]:
                text.append(f"- {it}")
            text.append("")
        return {"sections": sections, "text": "\n".join(text).strip(),
                "tokens": used, "budget": token_budget,
                "empty": not sections}

    # ── H1/H2: structural recall, cold-start honesty ─────────────────
    def structural(self, *, partition: str = "default"):
        """H1 -- an index that answers "who did what to whom".

        Embeddings cannot: *Ahmed delivered the gasket to Warsame* and the
        reverse embed almost identically, because a vector is a bag. This
        is a third index, consulted for structural questions, deliberately
        separate from both embedding spaces -- episodic detail, semantic
        gist and structural role are different kinds of information and
        collapsing them loses the ability to tell them apart.
        """
        from . import hyperdimensional as hd
        idx = getattr(self, "_structural", None)
        if idx is None:
            idx = self._structural = hd.StructuralIndex()
        return idx

    def maturity(self, *, partition: str = "default") -> dict:
        """H2 -- how much a gap in this store is worth believing.

        Every memory system's first week is its worst and all of them
        sound identical to their third year. Trust is won by being right
        about your own limits.
        """
        from . import maturity as mat
        now = self.clock.now()
        first = self._scalar(
            "SELECT MIN(observed_at) FROM observation WHERE partition=?",
            (partition,)) or now
        n = self._scalar(
            "SELECT COUNT(*) FROM observation WHERE partition=?", (partition,))
        srcs = self._scalar(
            "SELECT COUNT(DISTINCT source_ref) FROM observation WHERE "
            "partition=?", (partition,))
        m = mat.assess_maturity(days=max(0.0, (now - first) / DAY),
                                memories=n, sources=srcs)
        return {"days": round(m.days, 2), "memories": m.memories,
                "sources": m.sources, "coverage": m.coverage,
                "young": m.young, "contract": m.contract()}

    # ── E4: the jointly-edited ledger ────────────────────────────────
    def ledger(self, *, partition: str = "default", limit: int = 100,
               include_derived: bool = True) -> dict:
        """E4 -- what the system holds, in a form a person will actually read.

        Making memory legible is a cheaper path to accuracy than making
        extraction smarter: once people can see what a system believes about
        them, they correct it proactively, and they correct what matters to
        them rather than what a benchmark measures.
        """
        from . import ledger as lg
        entries: list = []
        for r in self._s.query(
                "SELECT o.id,o.content,o.epistemic_tag,o.source_ref,"
                "o.observed_at FROM observation o JOIN mem_index m "
                "ON m.node_id=o.id WHERE o.partition=? AND "
                "m.suppressed_at IS NULL ORDER BY o.observed_at DESC LIMIT ?"
                if False else
                "SELECT o.id,o.content,'observed' AS epistemic_tag,"
                "o.source_ref,o.observed_at FROM observation o "
                "JOIN mem_index m ON m.node_id=o.id WHERE o.partition=? "
                "AND m.suppressed_at IS NULL ORDER BY o.observed_at DESC "
                "LIMIT ?", (partition, limit)):
            sup = self._s.one(
                "SELECT new_node FROM supersession WHERE old_node=?", (r["id"],))
            entries.append(lg.LedgerEntry(
                r["id"], r["content"], "observation", r["epistemic_tag"], 1.0,
                r["source_ref"], r["observed_at"], corrected=sup is not None))
        if include_derived:
            for r in self._s.query(
                    "SELECT d.id,d.content,d.epistemic_tag,d.producer,"
                    "d.created_at,d.confidence FROM derived d JOIN mem_index m "
                    "ON m.node_id=d.id WHERE d.partition=? AND "
                    "m.suppressed_at IS NULL ORDER BY d.created_at DESC "
                    "LIMIT ?", (partition, limit)):
                entries.append(lg.LedgerEntry(
                    r["id"], r["content"], "derived", r["epistemic_tag"],
                    r["confidence"], f"produced by {r['producer']}",
                    r["created_at"]))
        led = lg.Ledger(entries, partition)
        return {"entries": [vars(e) for e in entries],
                "markdown": led.render(), "n": len(entries)}

    def correct(self, node_id: str, corrected: str, *, by: str,
                reason: str = "", partition: str = "default") -> str:
        """Record a person's correction. First-class provenance, not an edit.

        The original is never rewritten -- the append-only trigger would
        reject it. A correction SUPERSEDES, so `why()` shows both what was
        believed and who changed it: "the system thought X until Bill said
        otherwise on the 14th" is a better record than X quietly becoming Y.

        Also logged as a maximum-depth exposure for the corrector. They
        retrieved it, judged it, and generated a replacement -- the
        generation effect -- so they will hold it far longer than someone
        who was merely told, and the transactive model should know that.
        """
        from . import ledger as lg
        row = self._node_row(node_id)
        if row is None:
            raise OwlError(f"unknown node {node_id!r}")
        text = lg.correction_note(row["content"], corrected, by, reason)

        if node_id.startswith("obs_"):
            # The person is asserting this on their own authority, so it
            # enters as evidence -- but attributed to them, not laundered
            # into the original document's credibility.
            new_id = self.observe(
                text, origin="user_utterance", source_ref=f"correction:{by}",
                partition=partition, supersedes=node_id,
                reliability="B", credibility=2)
        else:
            # Correcting an inference produces a corrected INFERENCE.
            # Monotonicity applies to corrections like everything else; a
            # user cannot promote a guess to a fact by fixing its wording.
            new_id = self.derive(
                text, parents=[node_id], kind="correction",
                producer=f"user:{by}", partition=partition)

        self.tell(by, new_id, channel="correction")
        self._s.write(lambda c: c.execute(
            "UPDATE exposure SET depth=? WHERE who=? AND node_id=?",
            (lg.CORRECTION_DEPTH, by, new_id)))
        return new_id

    # ── A11: encryption at rest ──────────────────────────────────────
    @staticmethod
    @contextmanager
    def sealed(sealed_path: str | Path, keyfile: str | Path, **kw):
        """Open an encrypted store. Re-seals and shreds on exit.

            with Owl.sealed("mind.owl.sealed", "mind.key") as mind:
                mind.observe(...)

        A decrypted working copy exists on disk while this is open. That is
        stated rather than hidden -- SQLite cannot query ciphertext, and a
        design that claimed otherwise would be the security theatre A11
        exists to avoid.
        """
        from . import crypto
        key = crypto.load_key(keyfile)
        sealed = FsPath(sealed_path)
        work = sealed.with_suffix(sealed.suffix + ".open")
        existed = sealed.exists()
        if existed:
            crypto.unseal(sealed, work, key)
        mind = None
        try:
            mind = Owl.open(work, **kw)
            yield mind
        finally:
            if mind is not None:
                mind.close()
            if work.exists():
                # Re-seal BEFORE shredding, so a failure here loses the
                # session's writes rather than the whole store.
                crypto.seal(work, sealed, key)
                crypto.shred(work)

    @staticmethod
    def seal_store(plain_path: str | Path, sealed_path: str | Path,
                   keyfile: str | Path, *, shred_original: bool = False) -> dict:
        """Encrypt an existing plaintext store."""
        from . import crypto
        key = crypto.load_key(keyfile)
        out = crypto.seal(plain_path, sealed_path, key)
        if shred_original:
            crypto.shred(plain_path)
        return {"sealed": str(out), "original_removed": shred_original,
                "warning": "losing the key loses the store. There is no "
                           "recovery path."}

    # ── D5: time-travel replay ───────────────────────────────────────
    def replay(self, receipt_id: str | None = None, *,
               partition: str = "default") -> dict:
        """D5 -- what WOULD you have answered, and why was it wrong?

        Bitemporal recall answers "what did you record". This answers the
        harder question: re-run a past query against the index as it stood
        at that moment and compare to the receipt written at the time.

        Only possible because the substrate is append-only. A store that
        mutates in place cannot reconstruct its own past, which is why
        nothing else does this -- it is not a hard feature, it is a feature
        that most architectures have already made impossible.

        Divergence is the interesting output, not agreement:

          drift      the same query now returns something else, because the
                     evidence moved. This is the audit trail for a decision
                     that looked right at the time.
          regression the replay does NOT match the receipt. The engine's
                     behaviour changed, and a memory whose past answers are
                     not reproducible cannot be audited at all.
        """
        row = (self._s.one("SELECT * FROM receipt WHERE id=?", (receipt_id,))
               if receipt_id else
               self._s.one("SELECT * FROM receipt ORDER BY at DESC LIMIT 1"))
        if row is None:
            return {"error": "no receipt found; recall with receipts=True "
                             "to record them"}

        # Receipts store node ids under "n" (they are written on every
        # recall, so the schema is terse on purpose). Reading them as
        # "node_id" silently produced an empty set, which made every replay
        # look like a regression -- a bug that would have discredited the
        # feature the first time anyone used it.
        then = [c.get("n", c.get("node_id"))
                for c in json.loads(row["returned"])]
        # As-of the receipt: the index as it stood, not as it stands.
        past = self.recall(row["query"], partition=row["partition"],
                           as_of=row["at"])
        now = self.recall(row["query"], partition=row["partition"])
        replayed = [c.node_id for c in past.chunks]
        current = [c.node_id for c in now.chunks]

        faithful = replayed == then
        drifted = current != then
        lost, gained = set(then) - set(current), set(current) - set(then)
        return {
            "receipt_id": row["id"], "query": row["query"], "at": row["at"],
            "then": {"state": row["state"], "returned": then,
                     "reason": row["reason"]},
            "replayed": {"state": past.state.value, "returned": replayed},
            "now": {"state": now.state.value, "returned": current},
            "faithful": faithful,
            "drifted": drifted,
            "no_longer_returned": sorted(lost),
            "newly_returned": sorted(gained),
            "verdict": (
                "REGRESSION: replay does not match the receipt -- the engine "
                "changed, and past answers are no longer reproducible"
                if not faithful else
                "the evidence moved; this is why the old answer looked right"
                if drifted else
                "unchanged: same question, same evidence, same answer"),
        }

    def receipts_log(self, *, limit: int = 20) -> list[dict]:
        return [{"id": r["id"], "at": r["at"], "query": r["query"],
                 "state": r["state"], "returned": len(json.loads(r["returned"]))}
                for r in self._s.query(
                    "SELECT * FROM receipt ORDER BY at DESC LIMIT ?", (limit,))]

    # ── C2/C3/C5: consolidation ──────────────────────────────────────
    def communities(self, *, partition: str = "default") -> list[dict]:
        """C2 -- clusters whose IDENTITY survives splits and merges.

        Recomputing communities every cycle churns their IDs, and every
        composite derived from an old one is left pointing at something
        that no longer exists. Provenance chains break silently. So a
        community keeps its name if its core persists, and a split gives
        the name to the larger side rather than minting two strangers.
        """
        from . import consolidation as cons
        edges = [(r["src"], r["dst"], r["weight"]) for r in self._s.query(
            "SELECT a.src, a.dst, a.weight FROM assoc_edge a "
            "JOIN mem_index m ON m.node_id = a.src WHERE m.partition=?",
            (partition,))]
        groups = cons.label_propagation(edges)
        prev = [cons.Community(r["id"], frozenset(json.loads(r["members"])),
                               r["generation"], json.loads(r["lineage"]))
                for r in self._s.query(
                    "SELECT * FROM community WHERE partition=?", (partition,))]
        # A no-op pass must BE a no-op. Recomputing identical groups used to
        # bump the generation and rewrite every row, so calling this twice
        # changed the store without changing anything in it -- which is the
        # churn A10 exists to forbid, produced by the very code meant to
        # prevent it.
        if prev and {frozenset(g) for g in groups} == {c.members for c in prev}:
            return [{"id": c.id, "size": c.size, "generation": c.generation,
                     "lineage": c.lineage, "members": sorted(c.members)}
                    for c in sorted(prev, key=lambda c: c.id)]

        gen = (max((c.generation for c in prev), default=0)) + 1
        counter = [0]

        def mint():
            counter[0] += 1
            # Deterministic: derived from generation and ordinal, never a
            # uuid, or A10 could not hold.
            return f"com_{gen:04d}_{counter[0]:03d}"

        now = cons.reconcile(groups, prev, generation=gen, mint=mint)

        def _w(c):
            c.execute("DELETE FROM community WHERE partition=?", (partition,))
            for com in now:
                c.execute(
                    "INSERT INTO community(id,partition,members,generation,"
                    "lineage) VALUES(?,?,?,?,?)",
                    (com.id, partition, json.dumps(sorted(com.members)),
                     com.generation, json.dumps(com.lineage)))
        self._s.write(_w)
        return [{"id": c.id, "size": c.size, "generation": c.generation,
                 "lineage": c.lineage, "members": sorted(c.members)}
                for c in now]

    def sleep_plan(self, *, partition: str = "default") -> dict:
        """C3 -- which consolidation phase is OWED, if any.

        Scheduled by accumulated pressure, not by idle CPU. A free machine
        says nothing about whether there is anything worth doing.
        """
        from . import consolidation as cons
        unconsolidated = self._scalar(
            "SELECT COUNT(*) FROM mem_index WHERE partition=? AND "
            "review_count=0", (partition,))
        # The interference sweep records confusable pairs as assoc edges.
        # NREM's whole job is separating exactly these, so they ARE the
        # phase's targets -- no separate bookkeeping needed.
        pairs = self._s.query(
            "SELECT a.src, a.dst FROM assoc_edge a JOIN mem_index m "
            "ON m.node_id=a.src WHERE a.kind='confusable' AND m.partition=?",
            (partition,))
        confusable = sorted({r["src"] for r in pairs}
                            | {r["dst"] for r in pairs})
        rows = self._s.query(
            "SELECT node_id FROM mem_index WHERE partition=? "
            "ORDER BY last_review LIMIT 40", (partition,))
        distant = [r["node_id"] for r in rows]
        last = self._scalar(
            "SELECT MAX(last_review) FROM mem_index WHERE partition=?",
            (partition,)) or self.clock.now()
        plan = cons.plan_sleep(
            unconsolidated=unconsolidated,
            interference=min(1.0, len(pairs) / max(1, unconsolidated or 1)),
            hours_since=max(0.0, (self.clock.now() - last) / 3600.0),
            confusable=confusable, distant=distant)
        return {"phase": plan.phase, "pressure": plan.pressure,
                "targets": plan.targets, "temperature": plan.temperature,
                "reason": plan.reason}

    def schemas(self, *, partition: str = "default",
                min_members: int = 3) -> list[dict]:
        """C5 -- the rule, factored out of its repetitions.

        Twenty notes saying the same thing about different days carry one
        rule and twenty dates. Reported rather than applied: this shows
        what compression is available and what it would save, so the
        decision to spend it is made with the number in hand.
        """
        from . import consolidation as cons
        rows = self._s.query(
            "SELECT o.id, o.content FROM observation o JOIN mem_index m "
            "ON m.node_id=o.id WHERE o.partition=? AND o.claim_class<>'verbatim'",
            (partition,))
        found = cons.find_schemas([(r["id"], r["content"]) for r in rows],
                                  min_members=min_members)
        return [{"schema": g.schema, "members": g.members,
                 "n": len(g.members), "saved_chars": g.saved_chars,
                 "deltas": g.deltas} for g in found]

    def compression_plan(self, *, partition: str = "default",
                         limit: int = 20) -> list[dict]:
        """C1 -- what may be compressed, and the proof for each.

        Nothing is compressed without the system first demonstrating it can
        rebuild the content from cue plus neighbours. Without a Reasoner
        every answer is "keep verbatim", reported as SKIPPED rather than
        silently no-opped.
        """
        from . import reconstructive as rc
        recon = getattr(self.reasoner, "reconstruct", None)
        out = []
        for r in self._s.query(
                "SELECT o.id, o.content, o.claim_class FROM observation o "
                "JOIN mem_index m ON m.node_id=o.id WHERE o.partition=? "
                "AND m.tier IN ('cold','warm') LIMIT ?", (partition, limit)):
            nb = [x["content"] for x in self._s.query(
                "SELECT o2.content FROM assoc_edge a JOIN observation o2 "
                "ON o2.id=a.dst WHERE a.src=? LIMIT 5", (r["id"],))]
            p = rc.plan_compression(r["id"], r["content"],
                                    claim_class=r["claim_class"],
                                    reconstruct=recon, neighbours=nb)
            out.append({"node_id": p.node_id, "keep_verbatim": p.keep_verbatim,
                        "fidelity": p.fidelity, "reason": p.reason})
        return out

    # ── A7: second-order uncertainty ─────────────────────────────────
    def opinion(self, node_id: str, *, partition: str = "default") -> dict:
        """Belief / disbelief / IGNORANCE for one claim.

        Derived from what the store already records, so this works on any
        existing store with no migration: corroborating independent origins
        supply belief mass, counter-evidence and supersession supply
        disbelief, and whatever neither has claimed stays ignorance.

        The distinction scalar confidence cannot make:

            nothing on file      b=0.00 d=0.00 u=1.00  -> go and look
            two credible, opposed b=0.40 d=0.40 u=0.20 -> stop looking

        Both project to ~0.5. Only one of them is worth more searching.
        """
        from . import opinion as op
        row = self._node_row(node_id)
        if row is None:
            raise OwlError(f"unknown node {node_id!r}")

        # Belief: independent ORIGINS agreeing, not documents. Forty files
        # from one upstream source are one piece of evidence.
        same = self.corroborated(row["content"], partition=partition)
        support = float(same.get("weight") or 0.0) or (
            1.0 if row["origin"] != "derived" else 0.0)

        # Disbelief: anything that argues against it, plus the fact of
        # having been superseded -- which is the store's own record of
        # having stopped believing this.
        against = 0.0
        ch = self.challenge(row["content"], partition=partition)
        against += sum(float(c["strength"]) for c in ch["counters"])
        if self._s.one("SELECT 1 FROM supersession WHERE old_node=?",
                       (node_id,)):
            against += 1.0

        base = op.from_evidence(support, against)
        # A source you half-trust makes a claim half as informative -- the
        # mass leaves BELIEF for IGNORANCE, never for disbelief. Admiralty
        # runs A(1) best to F(6) worst on both axes; map the pair onto a
        # single 0..1 trust so a Grade-A source is taken at face value and
        # an F6 barely moves anything out of ignorance.
        rel, cred = self.effective_grade(node_id)
        # F and 6 are NOT the bottom of the scale -- they mean "cannot be
        # judged", which is ignorance about the source rather than distrust
        # of it. Mapping them to zero (the obvious reading of the letter
        # order) discounted every ordinary unrated observation to nothing
        # and left every opinion vacuous. E/5 is the actually-distrusted end.
        rel_w = {"A": 1.0, "B": 0.9, "C": 0.75, "D": 0.55, "E": 0.3,
                 "F": 0.6}.get(str(rel).upper()[:1], 0.6)
        cred_w = {1: 1.0, 2: 0.9, 3: 0.75, 4: 0.55, 5: 0.3,
                  6: 0.6}.get(int(cred or 6), 0.6)
        o = op.discount(max(0.0, min(1.0, (rel_w + cred_w) / 2)), base)

        out = o.as_dict()
        out.update({"node_id": node_id, "supporting_weight": round(support, 3),
                    "opposing_weight": round(against, 3),
                    "scalar_confidence": row["confidence"],
                    "note": "expectation reproduces the scalar; uncertainty "
                            "is what the scalar could never say"})
        return out

    def challenge(self, query: str, *, partition: str = "default",
                  limit: int = 5) -> dict:
        """B6 -- what in the store argues AGAINST the question's premise.

        Returned separately from `recall()`, never merged into it. Similarity
        search optimises for agreement, so a system that only ever returns
        support makes an analyst more confident with every query, using their
        own premise as the retrieval key. Mixing the counter-set into the
        answer would just make the answer confusing; keeping it separate
        makes it a challenge, which is the useful thing.
        """
        from . import counter
        pre = counter.presupposition(query)
        visible = self._s.readable_from(partition)
        terms = lexical.tokenize(pre)
        cands = self._lexical_candidates(terms, visible,
                                         self._count_nodes(visible))
        rows = []
        for nid in list(cands)[:60]:
            r = self._node_row(nid)
            if r is not None:
                rows.append({"node_id": nid, "content": r["content"]})

        # Anything the current belief REPLACED is counter-evidence by
        # construction, and the supersession graph already holds it.
        superseded = []
        for nid in list(cands)[:20]:
            # Whatever this node REPLACED. `superseded_by` points forward,
            # so the counter-evidence is everything pointing AT the current
            # belief -- the record of having thought otherwise.
            for s in self._s.query(
                    "SELECT old_node FROM supersession WHERE new_node=?",
                    (nid,)):
                old = self._node_row(s["old_node"])
                if old is not None:
                    superseded.append({"node_id": s["old_node"],
                                       "content": old["content"]})

        cs = counter.find(query, rows, superseded=superseded,
                          semantic_available=self._semantic, limit=limit)
        return {
            "presupposition": cs.presupposition,
            "counters": [vars(c) for c in cs.counters],
            "found": cs.found,
            "skipped": cs.skipped,
            "note": ("nothing in the store disputes the premise -- which is "
                     "not the same as the premise being right"
                     if not cs.found else
                     f"{len(cs.counters)} memories argue against the premise"),
        }

    # ── E3: the action-outcome loop ──────────────────────────────────
    def predicted(self, node_id: str, *, confidence: float | None = None,
                  claim_kind: str | None = None) -> int:
        """Log that a claim was ACTED ON, so its outcome can be checked later.

        Separate from `derive()` deliberately. Producing a claim and betting
        on one are different events, and only the second is a prediction --
        scoring everything a model ever emitted would drown the signal in
        material nobody relied on.
        """
        row = self._node_row(node_id)
        if row is None:
            raise OwlError(f"unknown node {node_id!r}")
        producer = row["source_ref"] or row["origin"] or "unknown"
        conf = row["confidence"] if confidence is None else confidence
        kind = claim_kind or row["claim_class"] or "unknown"

        def _w(c):
            cur = c.execute(
                "INSERT INTO calibration(producer,claim_kind,confidence,"
                "outcome,recorded_at) VALUES(?,?,?,NULL,?)",
                (producer, kind, float(conf), self.clock.now()))
            return cur.lastrowid
        return int(self._s.write(_w))

    def outcome(self, prediction_id: int, *, correct: bool) -> None:
        """Close the loop. Nothing is scored until this is called."""
        self._s.write(lambda c: c.execute(
            "UPDATE calibration SET outcome=? WHERE id=?",
            (int(bool(correct)), int(prediction_id))))

    def calibration(self, *, curve: bool = False) -> dict:
        """Is this system's confidence honest? Reported, never acted on.

        A producer saying 0.9 and being right 55% of the time is the exact
        failure OWL argues the field ignores -- so measuring it and then
        silently reweighting retrieval would be replacing an auditable
        problem with an unauditable one.
        """
        from . import calibration_loop
        rows = [dict(r) for r in self._s.query(
            "SELECT producer,claim_kind,confidence,outcome FROM calibration")]
        scored = calibration_loop.score(rows)
        out = {
            "predictions": len(rows),
            "resolved": sum(1 for r in rows if r["outcome"] is not None),
            "by_producer": [vars(c) for c in scored],
            "overconfident": [c.producer for c in scored if c.overconfident],
            "note": "reported only; nothing here reweights retrieval",
        }
        if curve:
            out["curve"] = calibration_loop.reliability_curve(rows)
        return out

    def converge(self, paths: Sequence[str | Path], *,
                 partition: str = "default") -> dict:
        """F7 -- what several operators independently agree on.

        Three people hand you their ledgers. The valuable question is not
        "what is in them" but *what did more than one of them see
        separately* -- because that is the only thing in a handover that
        gains credibility rather than merely accumulating.

        The trap is that agreement is easy to fake and easy to fake by
        accident. If three operators all read the same sitrep, three packs
        assert the same claim from ONE origin. Counting packs would call
        that triple-corroborated; it is single-sourced. So this counts
        independent ORIGINS across packs, reusing the same math that defends
        against source flooding, and an operator's own restatement of a
        shared document earns nothing.

        Reports only. Nothing is imported -- convergence is evidence for a
        graft decision, not the decision.
        """
        from . import attribution as attr
        packs = []
        for p in paths:
            pk = handover.read_pack(p)
            packs.append(pk)

        # proposition -> {operator: {origin_cluster, ...}}
        claims: dict[str, dict[str, set]] = {}
        text_of: dict[str, str] = {}
        for pk in packs:
            who = attr.canonical_name(pk["manifest"]["exporter"])
            for o in pk.get("observations", []):
                h = attr.proposition_hash(o.get("content", ""))
                text_of.setdefault(h, o.get("content", ""))
                origin = attr.origin_key(o.get("source_ref") or "")
                claims.setdefault(h, {}).setdefault(who, set()).add(origin)

        agreed, single = [], []
        for h, by_op in claims.items():
            origins = set().union(*by_op.values())
            n_ind = attr.independence(sorted(origins))
            row = {
                "claim": text_of[h],
                "operators": sorted(by_op),
                "n_operators": len(by_op),
                "independent_origins": n_ind,
                "weight": attr.corroboration_weight(n_ind),
            }
            # Two operators, one origin, is one source wearing two coats.
            if len(by_op) > 1 and n_ind > 1:
                row["note"] = (f"{len(by_op)} operators, {n_ind} independent "
                               "origins -- genuine corroboration")
                agreed.append(row)
            elif len(by_op) > 1:
                row["note"] = (f"{len(by_op)} operators but ONE origin -- "
                               "they read the same document; no credit")
                single.append(row)
            else:
                row["note"] = "single operator"
                single.append(row)

        agreed.sort(key=lambda r: (-r["independent_origins"],
                                   -r["n_operators"]))
        return {
            "operators": [p["manifest"]["exporter"] for p in packs],
            "corroborated": agreed,
            "uncorroborated": single,
            "note": ("claims are promoted only when independent ORIGINS "
                     "agree; operators echoing one document are one source"),
        }

    def watch(self, *, partition: str = "default", **kw):
        """F4 -- a watcher for anticipatory retrieval. OFF unless you ask.

            w = mind.watch()
            n = w.consider(user_turn, mind.nudge_candidates())
            if n: ...                       # usually None, by design
            w.record_outcome(n.node_id, acted_on=True)
            w.verdict()                     # keep it, or turn it off

        Not wired into recall(). A memory system that interrupts on its own
        schedule has to earn that, and the only honest way to find out is to
        run it beside a session and measure acted-on vs dismissed.
        """
        from .anticipation import Watcher
        return Watcher(**kw)

    def nudge_candidates(self, *, partition: str = "default") -> list[dict]:
        """What the watcher is allowed to raise its hand about.

        Only things with a CONSEQUENCE: a decision resting on evidence that
        moved, a commitment that is due, a loop left open. Deliberately not
        "relevant memories" -- relevance is what recall() is for, and a
        watcher that surfaces merely relevant things is a distraction with
        good intentions.
        """
        out: list[dict] = []
        for i in self.reconsider(partition=partition):
            out.append({"node_id": i.decision_id, "kind": "shifted_basis",
                        "text": i.statement,
                        "message": f"This rests on something that changed: "
                                   f"{i.statement} ({i.cause})"})
        try:
            for d in self.due(partition=partition):
                act = d.get("action", "")
                out.append({"node_id": d.get("id", act), "kind": "commitment",
                            "text": act, "message": f"Due: {act}"})
        except Exception:                                     # noqa: BLE001
            pass
        for r in self._s.query(
                "SELECT id, action FROM intention WHERE partition=? AND "
                "status='pending'", (partition,)):
            out.append({"node_id": r["id"], "kind": "open_loop",
                        "text": r["action"],
                        "message": f"Still open: {r['action']}"})
        return out

    def export_pack(self, path: str | Path, *, partition: str = "default",
                    exporter: str = "unknown", label: str = "",
                    notes: str = "", include_affect: bool = False,
                    compress: bool = False) -> dict:
        """Write a portable `.owlpack`. Plain JSON -- readable before you send it.

        Refuses on sealed partitions, with no override. Skips suppressed and
        affect-marked material: someone's distress is not a deliverable.
        """
        pack = handover.build_pack(
            self._s, partition=partition, exporter=exporter,
            now=self.clock.now(), label=label, notes=notes,
            include_affect=include_affect)
        handover.write_pack(pack, path, compress=compress)
        # F6: also write the reviewable rendering. The pack format's whole
        # justification is that a transfer can be checked before it happens,
        # and nobody proof-reads six thousand lines of JSON -- so in practice
        # the review does not happen and the guarantee is theatre.
        md = FsPath(str(path)).with_suffix(".md")
        md.write_text(handover.render_markdown(pack), encoding="utf-8")
        man = dict(pack["manifest"])
        man["review_copy"] = str(md)
        return man

    def review_pack(self, path: str | Path, *, max_items: int = 0) -> str:
        """The human-readable rendering of a pack, without importing it."""
        return handover.render_markdown(handover.read_pack(path),
                                        max_items=max_items)

    def inspect_pack(self, path: str | Path, *, steps: int = 1) -> dict:
        """Dry run before grafting. A handover is a trust decision."""
        return handover.plan_graft(handover.read_pack(path), steps=steps)

    def graft(self, path: str | Path, *, as_source: str,
              partition: str = "default", steps: int = 1,
              carry_exposures: bool = True) -> dict:
        """Import another operator's ledger, with epistemic demotion.

        Every tag shifts down one rank: their certainties become your reports,
        their conclusions become your hypotheses, their hypotheses are dropped.
        This is not a policy applied on top -- it is the same monotonicity
        invariant that governs every other derived node, so it cannot be
        bypassed by a caller who passes the wrong argument.
        """
        pack = handover.read_pack(path)
        man = pack["manifest"]
        now = self.clock.now()
        stats = {"observations": 0, "derived": 0, "dropped": 0,
                 "absences": 0, "intentions": 0, "exposures": 0,
                 "corroborated": 0, "from": man["exporter"]}
        idmap: dict[str, str] = {}
        admitted = handover.demote(Epistemic.OBSERVED, steps)
        if admitted is None:
            raise HandoverError("demotion depth admits nothing")

        existing = {r["content_hash"]: r["id"] for r in self._s.query(
            "SELECT id,content_hash FROM observation WHERE partition=?",
            (partition,))}

        def _w(c: sqlite3.Connection) -> None:
            for o in pack["observations"]:
                prior = existing.get(o["content_hash"])
                if prior is not None:
                    # Two INDEPENDENT operators recording the same thing is
                    # corroboration, not a duplicate. Every other system
                    # either dedupes (losing the signal) or keeps both (losing
                    # the quality distinction). Two axes lets you do the real
                    # epistemics: keep the better source, raise credibility.
                    cur = c.execute("SELECT reliability,credibility FROM "
                                    "observation WHERE id=?", (prior,)).fetchone()
                    rel, cred = epistemics.corroborate(
                        (cur["reliability"], cur["credibility"]),
                        (o["reliability"], o["credibility"]))
                    c.execute("INSERT OR IGNORE INTO derivation_edge"
                              "(child_id,parent_id,role) VALUES(?,?,'evidence')",
                              (prior, prior))
                    idmap[o["id"]] = prior
                    stats["corroborated"] += 1
                    continue
                nid = f"obs_{uuid.uuid4().hex[:16]}"
                idmap[o["id"]] = nid
                c.execute(
                    "INSERT INTO observation(id,partition,observed_at,"
                    "valid_from,valid_to,origin,source_ref,content,"
                    "content_hash,context_env,episode_id,period_id,affect,"
                    "claim_class,reliability,credibility) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (nid, partition, o["observed_at"], o["valid_from"],
                     o["valid_to"], "document",          # not first-hand to us
                     f"{as_source}::{o['source_ref']}", o["content"],
                     o["content_hash"], o["context_env"], None, None, 0.0,
                     o["claim_class"],
                     # An import can never be grade A: we did not see it.
                     max(o["reliability"], "C"), max(o["credibility"], 3)))
                c.execute(
                    "INSERT INTO mem_index(node_id,partition,stability,"
                    "difficulty,last_review,review_count,access_log,surprise,"
                    "tier) VALUES(?,?,?,?,?,0,'[]',0.5,'warm')",
                    (nid, partition, *salience.initial_state(2), now))
                self._index_terms(c, nid, partition,
                                  lexical.term_frequencies(o["content"]))
                stats["observations"] += 1

            for d in pack["derived"]:
                tag = handover.demote(Epistemic(d["epistemic_tag"]), steps)
                if tag is None:
                    stats["dropped"] += 1
                    continue
                did = f"der_{uuid.uuid4().hex[:16]}"
                idmap[d["id"]] = did
                fals = d["falsifier"]
                if d["kind"] == "hypothesis" and not fals:
                    fals = f"inherited from {as_source}; verify independently"
                c.execute(
                    "INSERT INTO derived(id,partition,created_at,kind,"
                    "epistemic_tag,producer,content,confidence,falsifier) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (did, partition, d["created_at"], d["kind"], tag.value,
                     f"{as_source}::{d['producer']}", d["content"],
                     min(d["confidence"], 0.6), fals))
                c.execute(
                    "INSERT INTO mem_index(node_id,partition,stability,"
                    "difficulty,last_review,review_count,access_log,surprise,"
                    "tier) VALUES(?,?,?,?,?,0,'[]',0.5,'warm')",
                    (did, partition, *salience.initial_state(2), now))
                self._index_terms(c, did, partition,
                                  lexical.term_frequencies(d["content"]))
                stats["derived"] += 1

            for e in pack["edges"]:
                a, b = idmap.get(e["child_id"]), idmap.get(e["parent_id"])
                if a and b:
                    c.execute("INSERT OR IGNORE INTO derivation_edge"
                              "(child_id,parent_id,role) VALUES(?,?,?)",
                              (a, b, e["role"]))

            if carry_exposures:
                # You inherit not just what they knew, but what they had been
                # TOLD and when -- which is most of a real handover briefing.
                for x in pack["exposures"]:
                    nid = idmap.get(x["node_id"])
                    if nid:
                        c.execute(
                            "INSERT INTO exposure(who,node_id,at,channel,depth) "
                            "VALUES(?,?,?,?,?)",
                            (f"{as_source}:{x['who']}", nid, x["at"],
                             x["channel"], x["depth"]))
                        stats["exposures"] += 1

            for a in pack["absences"]:
                c.execute(
                    "INSERT OR REPLACE INTO absence(id,partition,query,scope,"
                    "searched_at,reason,expires_at) VALUES(?,?,?,?,?,?,?)",
                    (f"abs_{uuid.uuid4().hex[:12]}", partition, a["query"],
                     a["scope"], a["searched_at"],
                     f"{a['reason']} (per {as_source})", a["expires_at"]))
                stats["absences"] += 1

            for i in pack["intentions"]:
                c.execute(
                    "INSERT INTO intention(id,partition,created_at,"
                    "trigger_kind,trigger_spec,action,status,origin_ref) "
                    "VALUES(?,?,?,?,?,?,'pending',?)",
                    (f"int_{uuid.uuid4().hex[:12]}", partition,
                     i["created_at"], i["trigger_kind"], i["trigger_spec"],
                     i["action"], as_source))
                stats["intentions"] += 1

        self._s.write(_w)
        return stats

    # ── DECONTEXTUALISATION ──────────────────────────────────────────
    def decontextualise(self, node_id: str, *, speaker: str | None = None
                        ) -> dict:
        """Make one memory readable cold.

        "He said it'd arrive Thursday" is useless six weeks later, and a large
        fraction of conversational memory looks like that. The expansion is a
        DERIVED node -- the raw utterance stays untouched, because it is the
        evidence (and because the append-only trigger would reject anything
        else).

        Ambiguous references are REFUSED, not guessed. A wrong substitution is
        invisible to the reader; an unresolved pronoun is not.
        """
        row = self._node_row(node_id)
        if row is None:
            raise OwlError(f"unknown node {node_id!r}")
        text = row["content"]
        if not decontext.needs_context(text):
            return {"node": node_id, "changed": False,
                    "reason": "already standalone"}

        cands = self._recency_entities(node_id, row)
        exp = decontext.expand(text, at=row["observed_at"],
                               candidates=cands, speaker=speaker)
        if not exp.changed:
            return {"node": node_id, "changed": False,
                    "unresolved": exp.unresolved,
                    "reason": "nothing could be resolved without guessing"}

        did = self.derive(
            exp.text, parents=[node_id], kind="decontext",
            producer="decontext", partition=row["partition"],
            confidence=float(row["confidence"]) * (1.0 if exp.standalone else 0.9),
            epistemic=Epistemic(row["epistemic"]))
        return {"node": node_id, "derived": did, "changed": True,
                "text": exp.text, "substitutions": exp.substitutions,
                "unresolved": exp.unresolved, "standalone": exp.standalone}

    def _recency_entities(self, node_id: str, row) -> list[tuple[str, str]]:
        """Entities mentioned earlier in the same episode, most recent first."""
        epi = row["episode_id"] if "episode_id" in row.keys() else None
        if not epi:
            epi = (self._s.one("SELECT episode_id FROM observation WHERE id=?",
                               (node_id,)) or {"episode_id": None})["episode_id"]
        if not epi:
            return []
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for r in self._s.query(
                "SELECT e.name,e.kind,o.observed_at FROM entity e "
                "JOIN mention m ON m.entity_id=e.id "
                "JOIN observation o ON o.id=m.node_id "
                "WHERE o.episode_id=? AND o.observed_at<=? "
                "ORDER BY o.observed_at DESC",
                (epi, row["observed_at"])):
            if r["name"] not in seen:
                seen.add(r["name"])
                out.append((r["name"], r["kind"]))
        return out

    def decontextualise_all(self, *, partition: str = "default",
                            limit: int = 200) -> dict:
        """Batch pass. Runs in `tend()`; safe to call directly."""
        stats = {"scanned": 0, "expanded": 0, "refused": 0}
        for r in self._s.query(
                "SELECT o.id,o.content FROM observation o "
                "WHERE o.partition=? AND NOT EXISTS ("
                "  SELECT 1 FROM derivation_edge e JOIN derived d "
                "  ON d.id=e.child_id WHERE e.parent_id=o.id "
                "  AND d.kind='decontext') "
                "ORDER BY o.observed_at DESC LIMIT ?", (partition, limit)):
            if not decontext.needs_context(r["content"]):
                continue
            stats["scanned"] += 1
            res = self.decontextualise(r["id"])
            if res.get("changed"):
                stats["expanded"] += 1
            else:
                stats["refused"] += 1
        return stats

    # ── FAILURE PATTERNS ─────────────────────────────────────────────
    def failed(self, approach: str, *, reason: str, context: str = "",
               partition: str = "default", node_id: str | None = None) -> str:
        """'We tried this, and here is why it did not work.'

        Distinct from ABSENCE ("I looked, it is not there"), and for an
        analyst toolkit arguably more valuable: it is what stops the same
        rejected option being re-proposed every week, which is the specific
        behaviour that reads as not listening.
        """
        canon = " ".join(sorted(set(lexical.tokenize(approach))))
        existing = self._s.one(
            "SELECT id,recurrence FROM failure WHERE partition=? AND "
            "approach=?", (partition, approach))
        if existing:
            self._s.write(lambda c: c.execute(
                "UPDATE failure SET recurrence=recurrence+1,failed_at=? "
                "WHERE id=?", (self.clock.now(), existing["id"])))
            return existing["id"]
        fid = f"fai_{uuid.uuid4().hex[:12]}"
        self._s.write(lambda c: c.execute(
            "INSERT INTO failure(id,partition,approach,reason,context,"
            "failed_at,node_id) VALUES(?,?,?,?,?,?,?)",
            (fid, partition, approach, reason, context or canon,
             self.clock.now(), node_id)))
        return fid

    def prior_failures(self, proposal: str, *, partition: str = "default",
                       threshold: float = 0.45) -> list[dict]:
        """Has this been tried before? Checked BEFORE proposing, not after."""
        out = []
        for r in self._s.query(
                "SELECT * FROM failure WHERE partition=? AND "
                "superseded_by IS NULL", (partition,)):
            sim = lexical.jaccard(proposal, r["approach"])
            if sim >= threshold:
                out.append({
                    "id": r["id"], "approach": r["approach"],
                    "reason": r["reason"], "similarity": round(sim, 3),
                    "recurrence": r["recurrence"],
                    "days_ago": round((self.clock.now() - r["failed_at"]) / DAY, 1),
                })
        out.sort(key=lambda x: -x["similarity"])
        return out

    def supersede_failure(self, failure_id: str, *, because: str) -> None:
        """Conditions change. A failure that no longer applies stops firing."""
        self._s.write(lambda c: c.execute(
            "UPDATE failure SET superseded_by=? WHERE id=?",
            (because, failure_id)))

    # ── NEGATIVE MEMORY ──────────────────────────────────────────────
    def record_absence(self, query: str, *, scope: str = "all",
                       partition: str = "default", reason: str = "",
                       expires_in: float | None = None) -> str:
        """Absence is expensive to establish and free to store.

        'I looked for a fuel supplier in Bardera and there wasn't one' costs a
        full search every single time it is asked, forever, in every other
        system. It also stops the same rejected option being re-proposed
        weekly, which is the specific behaviour that reads as not listening.
        """
        aid = f"abs_{uuid.uuid4().hex[:12]}"
        now = self.clock.now()
        self._s.write(lambda c: c.execute(
            "INSERT INTO absence(id,partition,query,scope,searched_at,reason,"
            "expires_at) VALUES(?,?,?,?,?,?,?)",
            (aid, partition, query, scope, now,
             reason or "searched, not found",
             now + expires_in if expires_in else None)))
        return aid

    def suppress(self, node_id: str, *, reason: str) -> None:
        """'Stop bringing this up.' NOT deletion -- those are different asks.

        Implemented as ranking demotion, never as an exclusion filter checked
        at retrieval time: a system that must ask 'is this the suppressed
        item?' has already retrieved it. Reversible, logged, and the memory
        stays addressable on an explicit request.
        """
        now = self.clock.now()
        self._s.write(lambda c: c.execute(
            "UPDATE mem_index SET suppressed_at=?,suppress_reason=?,"
            "tier='cold',open_loop=0 WHERE node_id=?",
            (now, reason, node_id)))

    def _write_receipt(self, rec: Recall, partition: str,
                       rejected: Sequence[tuple] = ()) -> str:
        """Immutable record of a retrieval: what came back, why, and what
        was considered and rejected.

        Downstream errors become traceable to a retrieval decision -- which
        currently evaporates the moment the call returns. Also the substrate
        for calibration: you cannot score confidence against outcome if you
        did not record what was said.
        """
        rid = f"rcp_{uuid.uuid4().hex[:14]}"
        ret = json.dumps([
            {"n": c.node_id, "s": round(c.score, 4),
             "r": round(c.retrievability, 3), "stale": round(c.staleness, 3),
             "epi": c.provenance.epistemic.value, "src": c.provenance.source_ref}
            for c in rec.chunks], separators=(",", ":"))
        rej = json.dumps([{"n": r[2]["id"], "s": round(r[0], 4)}
                          for r in rejected], separators=(",", ":"))
        self._s.write(affects_answers=False, fn=lambda c: c.execute(
            "INSERT INTO receipt(id,at,partition,query,state,reason,returned,"
            "rejected,tier,latency_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (rid, self.clock.now(), partition, rec.query, rec.state.value,
             rec.reason, ret, rej, self.tier, rec.latency_ms)))
        return rid

    def receipts_for(self, *, query: str | None = None, limit: int = 20
                     ) -> list[dict]:
        """Audit trail. 'What did you tell me on the 14th, and why?'"""
        if query:
            rows = self._s.query(
                "SELECT * FROM receipt WHERE query LIKE ? ORDER BY at DESC "
                "LIMIT ?", (f"%{query}%", limit))
        else:
            rows = self._s.query(
                "SELECT * FROM receipt ORDER BY at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["returned"] = SqliteStore.jload(r["returned"], [])
            d["rejected"] = SqliteStore.jload(r["rejected"], [])
            out.append(d)
        return out

    def why(self, node_id: str) -> list[dict]:
        """Full derivation chain back to primary sources. The headline feature."""
        out: list[dict] = []
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            row = self._node_row(nid)
            if row is None:
                continue
            rec = {
                "id": nid, "kind": row["kind"], "origin": row["origin"],
                "source_ref": row["source_ref"], "epistemic": row["epistemic"],
                "confidence": row["confidence"], "content": row["content"][:200],
                "presentable_as_fact": is_presentable_as_fact(
                    Epistemic(row["epistemic"])),
                "parents": [],
            }
            for e in self._s.query(
                    "SELECT parent_id,role FROM derivation_edge WHERE child_id=?",
                    (nid,)):
                rec["parents"].append({"id": e["parent_id"], "role": e["role"]})
                stack.append(e["parent_id"])
            out.append(rec)
        return out

    # ── MAINTAIN ─────────────────────────────────────────────────────
    def tend(self, *, budget_seconds: float | None = None,
             partition: str = "default") -> dict:
        """One deterministic maintenance pass. Tier 0 -- no model required.

        Runs: tier transitions (decay), duplicate merge, interference detection,
        retrieval-induced-forgetting demotion, expired intentions.
        Tier 2 additionally runs reconstructive compression; without a Reasoner
        that pass is SKIPPED and reported, never silently no-opped.
        """
        t0 = time.perf_counter()
        now = self.clock.now()
        report = {"partition": partition, "tier": self.tier, "skipped": []}

        # G4: tend() rescanned the whole store on every idle pass, which
        # made consolidation O(store) even when three memories changed.
        # The working set is the dirty set -- with a periodic full sweep,
        # because a dirty-set bug loses consolidation work SILENTLY and the
        # full pass is the thing that would eventually surface it.
        #
        # G5: the dirty set is PER PARTITION. One shared set meant
        # tend("private") consumed the work partition's pending nodes --
        # take() clears what it returns -- so consolidation queued by a
        # write to one partition was discarded by a maintenance pass on
        # another, silently, which is the failure the periodic full sweep
        # exists to catch and the partitions were making routine.
        all_nodes = [r["node_id"] for r in self._s.query(
            "SELECT node_id FROM mem_index WHERE partition=?", (partition,))]
        scope, why = self._dirty_for(partition).take(all_nodes=all_nodes)
        report["scope"] = {"nodes": len(scope), "of": len(all_nodes),
                           "why": why}

        # Drain the capture queue FIRST. tend() is the idle hook, so this is
        # where deferred embedding lands -- and everything below (duplicate
        # merge, interference, hubness) reads vectors, so running it against
        # a half-embedded store would measure the queue rather than the memory.
        if self.defer_embedding:
            report["absorbed"] = self.absorb()

        moved = {"hot": 0, "warm": 0, "cold": 0, "pruned": 0}
        rows = self._s.query(
            "SELECT node_id,stability,last_review,tier FROM mem_index "
            "WHERE partition=?", (partition,))
        updates = []
        for row in rows:
            r = salience.retrievability(now - row["last_review"], row["stability"])
            t = salience.tier_for(r)
            moved[t] += 1
            if t != row["tier"]:
                updates.append((t, row["node_id"]))
        if updates:
            self._s.write(lambda c: c.executemany(
                "UPDATE mem_index SET tier=? WHERE node_id=?", updates))
        report["tiers"] = moved
        report["retiered"] = len(updates)

        report["interference"] = self._interference_sweep(partition)
        report["fusion"] = self._fuse(partition)
        report["decontext"] = self.decontextualise_all(partition=partition)
        if self._semantic:
            # G5: scored within the shard. Hubness is a discount for being
            # close to the corpus, and the corpus a query can see is this
            # partition's -- so measuring against the store penalised a
            # private note for resembling documents it could never be
            # retrieved alongside.
            report["hubness_scored"] = self._vec.recompute_hubness(
                partition=partition)
        report["criticality_nodes"] = self.recompute_criticality()
        report["open_impacts"] = len(self.reconsider(partition=partition))

        expired = self._s.write(lambda c: c.execute(
            "UPDATE intention SET status='expired' WHERE status='pending' "
            "AND trigger_kind='time' AND CAST(trigger_spec AS REAL) < ?",
            (now,)).rowcount)
        report["intentions_expired"] = expired

        if self.reasoner is None:
            report["skipped"].append(
                "reconstructive_compression (no Reasoner configured)")
        if not self._semantic:
            report["skipped"].append(
                "semantic_interference (no semantic Embedder; using lexical "
                "jaccard, which will miss paraphrase)")
        report["vectors"] = self._vec.count()
        if self._warnings:
            report["warnings"] = list(self._warnings)
            self._warnings.clear()
        # Drain AGAIN. The passes above create nodes -- fusion composites,
        # abstractions -- and those follow the same deferral policy as any
        # other write, so a single drain at the top leaves tend() reporting
        # an empty queue that isn't. Idle work should finish idle.
        if self.defer_embedding and self.pending():
            after = self.absorb()
            first = report.get("absorbed", {})
            report["absorbed"] = {
                "embedded": first.get("embedded", 0) + after["embedded"],
                "failed": first.get("failed", 0) + after["failed"],
                "pending": after["pending"],
                "created_during_tend": after["embedded"] + after["failed"]}

        report["ms"] = (time.perf_counter() - t0) * 1000
        return report

    def intend(self, action: str, *, when: float | None = None,
               on_event: str | None = None, partition: str = "default",
               origin_ref: str = "user") -> str:
        """Prospective memory: memory for intentions, not for the past.

        Cue specificity is enforced -- prospective memory fails on vague cues,
        so a fuzzy trigger is rejected at insert rather than silently never
        firing.
        """
        if (when is None) == (on_event is None):
            raise ValueError("give exactly one of when= or on_event=")
        if on_event is not None and len(lexical.tokenize(on_event)) < 2:
            raise ValueError(
                f"trigger {on_event!r} is too vague to match reliably; "
                "give a concrete cue")
        iid = f"int_{uuid.uuid4().hex[:12]}"
        kind = "time" if when is not None else "event"
        spec = str(when) if when is not None else on_event
        self._s.write(lambda c: c.execute(
            "INSERT INTO intention(id,partition,created_at,trigger_kind,"
            "trigger_spec,action,origin_ref) VALUES(?,?,?,?,?,?,?)",
            (iid, partition, self.clock.now(), kind, spec, action, origin_ref)))
        return iid

    def due(self, *, partition: str = "default") -> list[dict]:
        now = self.clock.now()
        rows = self._s.query(
            "SELECT * FROM intention WHERE partition=? AND status='pending' "
            "AND trigger_kind='time' AND CAST(trigger_spec AS REAL) <= ?",
            (partition, now))
        return [dict(r) for r in rows]

    def reindex(self, *, batch: int = 128) -> int:
        """Embed everything written before an embedder was attached.

        A store that ran at Tier 0 for a month is still fully valid -- the
        substrate is complete and the lexical index works. Attaching an
        embedder later backfills the vectors; nothing has to be re-ingested.
        """
        if self.embedder is None:
            return 0
        done = 0
        rows = self._s.query(
            "SELECT o.id,o.content,o.partition,o.episode_id,o.period_id,"
            "o.source_ref,o.observed_at FROM observation o "
            "WHERE NOT EXISTS (SELECT 1 FROM vector v WHERE v.node_id=o.id)")
        for r in rows:
            self._index_vectors(r["id"], r["content"], _queue=False,
                                partition=r["partition"],
                                episode_id=r["episode_id"],
                                period_id=r["period_id"],
                                source_ref=r["source_ref"],
                                when=r["observed_at"])
            done += 1
        for r in self._s.query(
                "SELECT id,content,partition,producer,created_at FROM derived d "
                "WHERE NOT EXISTS (SELECT 1 FROM vector v WHERE v.node_id=d.id)"):
            self._index_vectors(r["id"], r["content"], _queue=False,
                                partition=r["partition"],
                                source_ref=r["producer"], when=r["created_at"])
            done += 1
        return done

    def self_audit(self) -> dict:
        """Attack OWL's own invariants, in the live store, over time.

        CI proves the invariants hold at commit. This proves they still hold
        after months of writes, consolidation, grafts and revaluation -- which
        is a different claim, and the one that matters in the field.
        """
        findings: list[dict] = []

        for d in self._s.query(
                "SELECT id,confidence,epistemic_tag,kind,falsifier FROM derived"):
            pf = [self._parent_facts(e["parent_id"]) for e in self._s.query(
                "SELECT parent_id FROM derivation_edge WHERE child_id=?",
                (d["id"],))]
            try:
                assert_monotonic(pf, confidence=d["confidence"],
                                 epistemic=Epistemic(d["epistemic_tag"]),
                                 node_id=d["id"])
            except Exception as exc:                       # noqa: BLE001
                findings.append({"kind": "monotonicity", "node": d["id"],
                                 "detail": str(exc)})
            if d["kind"] == "hypothesis" and not d["falsifier"]:
                findings.append({"kind": "untestable_hypothesis",
                                 "node": d["id"], "detail": "no falsifier"})
            # Enforced on write, but nothing audited it afterwards. A
            # hypothesis relabelled 'observed' passed monotonicity (its
            # parent WAS observed) and read downstream as a fact.
            if d["kind"] == "hypothesis" and d["epistemic_tag"] != "hypothesized":
                findings.append({
                    "kind": "kind_tag_mismatch", "node": d["id"],
                    "detail": f"kind='hypothesis' but epistemic_tag="
                              f"'{d['epistemic_tag']}' - reads as a conclusion"})
            if d["kind"] == "decontext" and d["epistemic_tag"] == "hypothesized":
                findings.append({
                    "kind": "kind_tag_mismatch", "node": d["id"],
                    "detail": "a decontextualisation cannot be more "
                              "speculative than what it expands"})

        # Anything presentable as fact must trace to a primary source.
        for d in self._s.query(
                "SELECT id,epistemic_tag FROM derived WHERE epistemic_tag IN "
                "('observed','reported')"):
            chain = self.why(d["id"])
            if not any(n["origin"] in ("user_utterance", "document",
                                       "tool_output") for n in chain):
                findings.append({"kind": "untraceable_fact", "node": d["id"],
                                 "detail": "presentable as fact, no primary "
                                           "source in its derivation"})

        orphans = self._scalar(
            "SELECT COUNT(*) FROM derivation_edge e WHERE NOT EXISTS "
            "(SELECT 1 FROM observation o WHERE o.id=e.parent_id) AND NOT "
            "EXISTS (SELECT 1 FROM derived d WHERE d.id=e.parent_id)")
        if orphans:
            findings.append({"kind": "orphan_edges", "node": None,
                             "detail": f"{orphans} edges point nowhere"})

        # Flow control: no node may be visible from a partition that has no
        # declared inbound path to its owner.
        for p in self._s.query("SELECT name FROM partition"):
            visible = self._s.readable_from(p["name"])
            for owner in {r["partition"] for r in self._s.query(
                    "SELECT DISTINCT partition FROM observation")}:
                if owner not in visible:
                    leaked = self._scalar(
                        "SELECT COUNT(*) FROM observation o JOIN mem_index m "
                        "ON m.node_id=o.id WHERE o.partition=? AND m.partition=?",
                        (owner, p["name"]))
                    if leaked:
                        findings.append(
                            {"kind": "flow_violation", "node": None,
                             "detail": f"{leaked} nodes of '{owner}' indexed "
                                       f"under '{p['name']}'"})
        return {"clean": not findings, "findings": findings,
                "checked_at": self.clock.now()}

    def quarantine_report(self, *, partition: str = "default") -> dict:
        """What has been held back, and why."""
        rows = self._s.query(
            "SELECT o.id,o.content,o.source_ref,o.trust,w.verdict,w.signals,"
            "w.score FROM observation o LEFT JOIN write_screen w "
            "ON w.node_id=o.id WHERE o.partition=? AND o.trust<>'trusted'",
            (partition,))
        blocked = [dict(r) for r in rows]
        for b in blocked:
            b["signals"] = SqliteStore.jload(b["signals"], [])
            b["content"] = b["content"][:110]
        rejected = [dict(r) for r in self._s.query(
            "SELECT at,source_ref,old_node,reason FROM supersession_attempt "
            "WHERE allowed=0 ORDER BY at DESC LIMIT 25")]
        return {"quarantined": blocked, "rejected_supersessions": rejected}

    def doctor(self) -> dict:
        """Named checks with remedies. Read-only, safe in production.

        F5. The point is not the list of problems -- it is that every entry
        says what to DO. A diagnostic that reports a fault without a fix has
        moved the burden rather than lifted it.

        Returns the structured report plus the legacy flat keys, so existing
        callers keep working:

            rep = mind.doctor()
            print(rep["report"])            # human
            rep["checks"]                   # machine
            rep["healthy"]                  # bool, FAIL only
        """
        from . import diagnostics
        report = diagnostics.run(self)
        out: dict[str, Any] = {"version": __version__, "tier": self.tier,
                               "path": self._s.path, "problems": []}
        out.update(report.as_dict())
        out["report"] = report.render()
        out["problems"] = [f"{c.id}: {c.detail}" for c in report.failed]
        out["warnings"] = [f"{c.id}: {c.detail}" for c in report.warnings]
        # Counts, kept flat for backwards compatibility and because they
        # are the first thing anyone wants to see.
        out["observations"] = self._scalar("SELECT COUNT(*) FROM observation")
        out["derived"] = self._scalar("SELECT COUNT(*) FROM derived")
        out["episodes"] = self._scalar("SELECT COUNT(*) FROM episode")
        out["partitions"] = [dict(r) for r in
                             self._s.query("SELECT * FROM partition")]
        out["embedder"] = (getattr(self.embedder, "name", "unknown")
                           if self.embedder else None)
        out["semantic"] = self._semantic
        out["vectors"] = self._vec.count()
        out["quarantined"] = self._scalar(
            "SELECT COUNT(*) FROM observation WHERE trust<>'trusted'")
        out["open_impacts"] = self._scalar(
            "SELECT COUNT(*) FROM decision_impact WHERE acknowledged_at IS NULL")
        out["self_audit"] = self.self_audit()["clean"]
        out["healthy"] = not out["problems"]
        return out

    # ── internals ────────────────────────────────────────────────────
    def _parent_facts(self, node_id: str) -> ParentFacts:
        row = self._node_row(node_id)
        if row is None:
            raise OwlError(f"unknown parent {node_id!r}")
        return ParentFacts(node_id, float(row["confidence"]),
                           Epistemic(row["epistemic"]))

    def _node_row(self, node_id: str) -> sqlite3.Row | None:
        if node_id.startswith("obs_"):
            return self._s.one(
                "SELECT id,partition,content,origin,source_ref,observed_at,"
                "valid_from,valid_to,affect,claim_class,reliability,trust,"
                "producer_model,acquisition_cost,credibility,1.0 AS confidence,"
                "'observed' AS epistemic,'observation' AS kind "
                "FROM observation WHERE id=?", (node_id,))
        return self._s.one(
            "SELECT id,partition,content,'derived' AS origin,producer AS "
            "source_ref,created_at AS observed_at,NULL AS valid_from,"
            "NULL AS valid_to,0.0 AS affect,'unknown' AS claim_class,"
            "'F' AS reliability,'trusted' AS trust,producer_model,"
            "0.0 AS acquisition_cost,6 AS credibility,"
            "confidence,epistemic_tag AS epistemic,kind "
            "FROM derived WHERE id=?", (node_id,))

    def _provenance(self, row: sqlite3.Row) -> Provenance:
        parents = tuple(e["parent_id"] for e in self._s.query(
            "SELECT parent_id FROM derivation_edge WHERE child_id=?", (row["id"],)))
        return Provenance(
            origin=row["origin"], source_ref=row["source_ref"],
            epistemic=Epistemic(row["epistemic"]), observed_at=row["observed_at"],
            valid_from=row["valid_from"], valid_to=row["valid_to"],
            derivation=parents)

    @staticmethod
    def _valid_at(row: sqlite3.Row, t: float) -> bool:
        if row["valid_from"] is not None and t < row["valid_from"]:
            return False
        if row["valid_to"] is not None and t > row["valid_to"]:
            return False
        return row["observed_at"] <= t or row["valid_from"] is not None

    # ── G5: the shard layer ──────────────────────────────────────────
    def _shard_terms(self, partition: str):
        """This partition's vocabulary. The input to its own filter."""
        if not self._s.sharded:
            # Unmigrated store: one filter over everything, as before.
            return [r["term"] for r in
                    self._s.query("SELECT DISTINCT term FROM lexeme")]
        return [r["term"] for r in self._s.query(
            "SELECT DISTINCT term FROM lexeme WHERE partition=?",
            (partition,))]

    def _shard_count(self, partition: str) -> int:
        if not self._s.sharded:
            return self._scalar("SELECT COUNT(*) FROM mem_index")
        return self._scalar(
            "SELECT COUNT(*) FROM mem_index WHERE partition=?", (partition,))

    def _index_terms(self, c: sqlite3.Connection, node_id: str,
                     partition: str, tfs: dict) -> None:
        """Write a node's terms into the inverted index, and the filter.

        THE ONE PLACE lexemes are written, and it is one place on purpose.
        The vocabulary filter used to be built once on first recall and
        never updated again, on the reasoning that adding a term only ever
        moves a Bloom filter towards "possibly present" -- the safe
        direction. That is true of adding to the FILTER. The code added to
        the STORE and not the filter, which is the opposite direction: any
        term written after the filter was built read as DEFINITELY ABSENT,
        the posting scan was skipped, and a memory the store genuinely held
        came back DONT_KNOW.

        No embedder required to reproduce, nothing raised, and it was
        indistinguishable from ordinary forgetting. Four call sites wrote
        lexemes and none of them touched the filter; now none of them can
        forget to, because there is one site.
        """
        c.executemany(
            "INSERT OR REPLACE INTO lexeme(term,node_id,tf,partition) "
            "VALUES(?,?,?,?)",
            [(t, node_id, v, partition) for t, v in tfs.items()])
        self._shards.note_terms(partition, tfs)
        self._shards.touch(partition)

    def _count_nodes(self, visible: dict[str, str]) -> int:
        return self._shards.total(visible)

    def _dirty_for(self, partition: str):
        from .freshness import DirtySet
        got = self._dirty.get(partition)
        if got is None:
            got = self._dirty[partition] = DirtySet()
        return got

    def _on_write(self) -> None:
        """Every write: drop the recall cache, keep the vocabulary filter.

        Dropping the filter per write would cost a full vocabulary rebuild
        on the next query, which is the exact cost G1 and G5 exist to
        avoid. It stays current incrementally instead -- see
        `_index_terms`, and the bug that made that necessary.
        """
        self._rcache.invalidate()

    def _definitely_unknown(self, terms: list[str],
                            visible: dict[str, str] | None = None) -> bool:
        """G1 -- is the LEXICAL scan certain to return nothing?

        True only when the filter says every query term is definitely
        absent from the vocabulary. A Bloom filter's one error is the false
        positive, so this can never wrongly claim ignorance.

        G5 made this a question about the VISIBLE PARTITIONS rather than
        the store. The two are different questions and the old answer was
        to the wrong one: a term appearing only in `work` made the global
        filter say "possibly present" to a query inside `private`, which
        then scanned `private` and found nothing. The fast path declined to
        fire in precisely the case it was built for, and the more the work
        partition grew the more often that happened.

        NARROWER THAN IT LOOKS, and the first version got this wrong badly
        enough to be worth recording. The plan asks for "DONT_KNOW in O(1)",
        which reads like an invitation to short-circuit `recall()` entirely.
        Doing that broke five things at once:

          * PARAPHRASE RECALL, fatally. A paraphrase shares no terms with
            its target -- that is what makes it a paraphrase -- so the
            lexical filter says "absent" for precisely the queries the
            semantic tier exists to answer.
          * the B7 gap explanation, which is the useful half of DONT_KNOW
          * recorded absences, which OUTRANK a plain DONT_KNOW
          * receipts, which are written for DONT_KNOW too
          * the deferred-capture "provisional" notice

        Lexical absence is evidence about the lexical index of these
        partitions and nothing else. So this now skips only the
        posting-list scan -- the O(terms x postings) work the plan was
        actually pointing at -- and every other path runs untouched.
        """
        if not terms:
            return False
        return self._shards.definitely_unknown(
            terms, visible if visible is not None else {"default": "full"})

    def _lexical_candidates(self, terms: list[str], visible: dict[str, str],
                            n_docs: int) -> dict[str, float]:
        """Score = query coverage x match quality.

        The coverage factor is load-bearing. Normalising scores to the best
        candidate -- the obvious implementation -- makes the top hit score 1.0
        no matter how bad it is, so a store containing only "the water tanker
        arrives Tuesday" answers "when does the FUEL tanker arrive" with
        KNOW. That is confabulation by ranking artefact, and it is exactly the
        failure a memory system must not have. A node can only reach KNOW if it
        covers most of what was actually asked.
        """
        raw: dict[str, float] = {}
        hits: dict[str, set[str]] = {}
        uniq = set(terms)
        pq = ",".join("?" * len(visible))
        # G5: the partition predicate reads off `lexeme`, so SQLite drives
        # from idx_lexeme_shard(partition, term, ...) and a term's posting
        # list is walked only inside the visible shards. With the predicate
        # on `mem_index` it had to visit every posting for the term across
        # every partition and join each one before it could discard it --
        # the filter running after the scan it was there to prevent.
        col = shards.partition_col("l", self._s.sharded)
        for term in uniq:
            rows = self._s.query(
                f"SELECT l.node_id,l.tf FROM lexeme l JOIN mem_index m "
                f"ON m.node_id=l.node_id WHERE l.term=? AND {col} IN ({pq}) "
                f"AND m.tier<>'pruned'", (term, *visible))
            if not rows:
                continue
            w = max(lexical.idf(len(rows), max(n_docs, 1)), 0.05)
            for r in rows:
                raw[r["node_id"]] = raw.get(r["node_id"], 0.0) + r["tf"] * w
                hits.setdefault(r["node_id"], set()).add(term)
        if not raw:
            return {}
        top = max(raw.values()) or 1.0
        return {nid: (len(hits[nid]) / len(uniq)) * (v / top)
                for nid, v in raw.items()}

    def _blend_semantic(self, query: str, cands: dict[str, float],
                        visible: dict[str, str]
                        ) -> tuple[dict[str, float], float]:
        """Pattern completion: retrieve a NEIGHBOURHOOD, then discriminate.

        Fusion is max-of-normalised rather than a weighted sum, on purpose.
        A weighted sum lets a strong lexical hit and a strong semantic hit
        average each other down to a mediocre score, which is exactly wrong --
        either signal firing hard is good evidence. Taking the max keeps
        paraphrase recall (semantic) without sacrificing exact-identifier
        recall (lexical), and identifiers are where embeddings are weakest:
        no embedding model reliably distinguishes serial GX-4419 from GX-4491.
        """
        qv = self._embed([query], Space.READ)
        if qv is None:
            return cands, 0.0
        allowed = {r["node_id"] for r in self._s.query(
            "SELECT node_id FROM mem_index WHERE partition IN ({})".format(
                ",".join("?" * len(visible))), tuple(visible))}
        floor = getattr(self.embedder, "noise_floor", SEMANTIC_FLOOR)
        search_floor = getattr(self.embedder, "search_floor",
                               min(SEARCH_FLOOR, floor))
        model = getattr(self.embedder, "name", None)
        # G5: `partitions` scopes the SQL; `allowed` stays as the authority
        # on visibility. Belt and braces on purpose -- the shard predicate
        # is an optimisation and optimisations are allowed to be wrong,
        # but the confidentiality boundary is not, so the set that decides
        # what may be seen is the one that was already load-bearing.
        hits = self._vec.search(qv[0], space=Space.READ, allowed=allowed,
                                top_k=60, floor=search_floor, model=model,
                                partitions=visible.keys())
        self._check_model_drift(model)
        if not hits:
            return cands, 0.0
        # ABSOLUTE, not relative. Dividing by the best similarity makes the
        # top semantic hit score 1.0 no matter how bad it is -- the exact
        # defect already fixed once in `_lexical_candidates`, left in place
        # here, and invisible until a real model produced tightly-clustered
        # cosines. With BGE-M3, unrelated field notes sit around 0.5 and
        # paraphrases around 0.8; max-normalising flattened that to ~1.0 for
        # everything and swamped the lexical signal, so a query for the exact
        # string "GX-4419" returned a note about generator fuel.
        # RANKING is monotone in similarity. The margin test belongs to the
        # GATE (know vs dont_know), not to the ordering -- feeding it into
        # ranking crushes the runner-up whenever several candidates are
        # plausible, because margin is measured against the pack. Measured
        # on BGE-M3, that buried the only candidate containing a person
        # under "The clinic has twelve beds."
        sims = [s for _, s in hits]
        best = max(sims)
        # Background is a ROBUST estimate of the NOISE level: the mean of the
        # bottom 60%. The mean of everything below the winner is dragged up
        # by a genuine runner-up, and the median of a short list IS the
        # winner -- with a single candidate that made margin exactly zero and
        # a perfect 0.73 match returned DONT_KNOW.
        #
        # With fewer than three candidates there is NO distribution to
        # measure against, and inventing one (background = floor) is worse
        # than admitting it: two unrelated notes both sitting just above the
        # cutoff then looked like a strong margin. In that case judge on
        # absolute level alone, which is stricter and honest about what a
        # near-empty store can support.
        ordered = sorted(sims)
        if len(ordered) < 3:
            background = None
        else:
            tail = ordered[:max(1, int(len(ordered) * 0.6))]
            background = sum(tail) / len(tail)
        out = dict(cands)
        ceiling = getattr(self.embedder, "ceiling", 1.0)
        for nid, sim in hits:
            # monotone in `sim`, so ordering is exactly the encoder's.
            # Scored against the NOISE floor, searched against the lower one,
            # and scaled by the CEILING the encoder can actually reach --
            # dividing by (1.0 - floor) assumed a cosine of 1.0 was possible
            # and turned a correct answer at 0.430 into 0.149, below
            # KNOW_WHERE_SCORE, so the encoder's own top-ranked hit came
            # back DONT_KNOW.
            out[nid] = max(out.get(nid, 0.0),
                           metamemory.level_of(sim, floor, ceiling))
        self._last_background = background
        return out, best

    def _check_model_drift(self, model: str | None) -> None:
        """Say so, once, when the store holds vectors this encoder can't use.

        Filtering by model makes retrieval CORRECT but quieter -- memories
        embedded by the previous model simply stop being findable, which
        looks identical to forgetting. Silence is the wrong answer to
        "half your store just became invisible".
        """
        if model is None or getattr(self, "_drift_checked", False):
            return
        self._drift_checked = True
        rows = self._s.query(
            "SELECT model, COUNT(*) n FROM vector WHERE space=? GROUP BY model",
            (Space.READ.value,))
        stale = {r["model"]: r["n"] for r in rows if r["model"] != model}
        if stale:
            listed = ", ".join(f"{k} ({v})" for k, v in stale.items())
            self._warn(
                f"{sum(stale.values())} memories were embedded by a different "
                f"model [{listed}] and are excluded from semantic search -- "
                f"vectors from different encoders are not comparable. Re-embed "
                f"them for '{model}', or reopen with the original encoder.")

    def hubness_of(self, content: str, space: Space = Space.READ) -> float:
        """How much this memory is discounted for being close to everything.

        Diagnostic. When a retrieval misses, the cause is either the gate
        (score below threshold) or hubness (a genuine hub demoted), and the
        two have opposite fixes -- raise the ceiling, or stop discounting.
        Distinguishing them by argument wasted a round; this measures it.
        """
        row = self._s.one(
            "SELECT v.hubness FROM vector v JOIN observation o "
            "ON o.id = v.node_id WHERE o.content = ? AND v.space = ?",
            (content, space.value))
        return float(row["hubness"] or 0.0) if row else 0.0

    def _expand_assoc(self, cands: dict[str, float], visible: dict[str, str],
                      damping: float = 0.35, hops: int = 1) -> dict[str, float]:
        """One-step Personalized-PageRank-style spread over the assoc graph.

        Cheap multi-hop association without a second model call. HippoRAG
        showed graph propagation from seed nodes recovers multi-hop links that
        flat vector search misses; this is the same trick at Tier 0 scale.
        """
        out = dict(cands)
        frontier = dict(cands)
        for _ in range(hops):
            nxt: dict[str, float] = {}
            for src, w in frontier.items():
                for e in self._s.query(
                        "SELECT dst,weight FROM assoc_edge WHERE src=?", (src,)):
                    nxt[e["dst"]] = nxt.get(e["dst"], 0.0) + w * damping * e["weight"]
                for e in self._s.query(
                        "SELECT dst,count FROM succession WHERE src=?", (src,)):
                    nxt[e["dst"]] = nxt.get(e["dst"], 0.0) + w * damping * 0.5 * min(
                        1.0, e["count"] / 5.0)
            for k, v in nxt.items():
                out[k] = out.get(k, 0.0) + v
            frontier = nxt
        return out

    def _coverage(self, terms: list[str], visible: dict[str, str]) -> float:
        if not terms:
            return 0.0
        pq = ",".join("?" * len(visible))
        col = shards.partition_col("l", self._s.sharded)
        seen = 0
        for t in set(terms):
            if self._s.one(
                    f"SELECT 1 FROM lexeme l JOIN mem_index m ON m.node_id=l.node_id "
                    f"WHERE l.term=? AND {col} IN ({pq}) LIMIT 1",
                    (t, *visible)):
                seen += 1
        return seen / len(set(terms))

    def _mark_used(self, node_ids: list[str]) -> None:
        """A successful retrieval is a review (FSRS), and it suppresses the
        near-neighbours that were NOT used -- retrieval-induced forgetting.
        Suppression is ranking demotion, never exclusion: the store sharpens
        around actual use without anything being deleted."""
        if not node_ids:
            return
        now = self.clock.now()

        def _w(c: sqlite3.Connection) -> None:
            for nid in node_ids:
                row = c.execute("SELECT * FROM mem_index WHERE node_id=?",
                                (nid,)).fetchone()
                if row is None:
                    continue
                s, d = salience.review(row["stability"], row["difficulty"],
                                       now - row["last_review"], grade=3)
                log = SqliteStore.jload(row["access_log"], [])[-63:] + [now]
                c.execute(
                    "UPDATE mem_index SET stability=?,difficulty=?,last_review=?,"
                    "review_count=review_count+1,access_log=?,tier='hot' "
                    "WHERE node_id=?",
                    (s, d, now, json.dumps(log), nid))
            for a in node_ids:
                for b in node_ids:
                    if a != b:
                        c.execute(
                            "INSERT INTO assoc_edge(src,dst,weight,kind) "
                            "VALUES(?,?,1.0,'cooccur') ON CONFLICT(src,dst,kind) "
                            "DO UPDATE SET weight=MIN(weight+0.25,4.0)", (a, b))

        # Recall's own side effect: it nudges retrievability and does not
        # change which memories exist, so it must not invalidate the answer
        # cache. Without this the cache invalidates itself on every read --
        # a feature that ships and then never fires once.
        self._s.write(_w, affects_answers=False)

    def _interference_sweep(self, partition: str) -> dict:
        """Interference, not decay, is what actually kills retrieval.

        Age-based pruning does nothing about forty near-identical memories on a
        recurring topic -- and may make it worse by deleting the one
        distinctive old record. So this runs AHEAD of decay and partitions
        confusable pairs into redundant / contradictory / merely-confusable.
        """
        rows = self._s.query(
            "SELECT o.id,o.content,o.content_hash FROM observation o "
            "JOIN mem_index m ON m.node_id=o.id "
            "WHERE o.partition=? AND m.tier<>'pruned' "
            "ORDER BY o.observed_at DESC LIMIT 400", (partition,))
        exact: dict[str, list[str]] = {}
        for r in rows:
            exact.setdefault(r["content_hash"], []).append(r["id"])
        dupes = sum(len(v) - 1 for v in exact.values() if len(v) > 1)

        confusable = 0
        pairs: list[tuple[str, str]] = []
        if self._semantic and self._vec.count():
            # Confusability in the WRITE space is the strong signal: these are
            # memories that landed close together DESPITE separation having
            # been applied to push them apart.
            seen: set[tuple[str, str]] = set()
            for r in rows:
                # G5: neighbours WITHIN the shard. Unscoped, the sweep
                # could pair a private memory with a work one and record
                # the interference -- a cross-partition fact, derived from
                # content one side may not read, written into a table both
                # can. Confusability only means something between memories
                # that can be retrieved by the same query anyway.
                for other, sim in self._vec.neighbours(
                        r["id"], space=Space.WRITE, threshold=0.86,
                        partitions=[partition]):
                    key = tuple(sorted((r["id"], other)))
                    if key in seen:
                        continue
                    seen.add(key)
                    confusable += 1
                    pairs.append((key[0], key[1]))
        else:
            for i in range(len(rows)):
                for j in range(i + 1, min(i + 25, len(rows))):
                    if rows[i]["content_hash"] == rows[j]["content_hash"]:
                        continue
                    if lexical.jaccard(rows[i]["content"],
                                       rows[j]["content"]) > 0.6:
                        confusable += 1
                        pairs.append((rows[i]["id"], rows[j]["id"]))
        if pairs:
            self._s.write(lambda c: c.executemany(
                "INSERT INTO assoc_edge(src,dst,weight,kind) VALUES(?,?,1.0,"
                "'confusable') ON CONFLICT(src,dst,kind) DO NOTHING", pairs))
        return {"exact_duplicates": dupes, "confusable_pairs": confusable,
                "scanned": len(rows)}

    def _halflife(self, claim_class: str) -> float | None:
        """Learned half-life for a claim class, or None to use the prior.

        Fitted from real supersession intervals: every time a claim is replaced
        we learn how long that CLASS of claim tends to survive. Nobody
        configures this.
        """
        rows = self._s.query(
            "SELECT survived FROM supersession WHERE claim_class=?",
            (claim_class,))
        return epistemics.fit_halflife([r["survived"] for r in rows])

    def _explain_gap(self, query: str, terms: list[str],
                     visible: dict[str, str]) -> str:
        """Say what would have to EXIST for the answer to be knowable.

        Turns a dead end into a task. "I have nothing" is honest but inert;
        "I would need a document naming the depot's fuel supplier" is
        actionable, and composes with prospective memory -- the gap becomes
        a standing intention.
        """
        want = entities.predict_answer_type(query)
        pq = ",".join("?" * len(visible))
        # Interrogatives survive tokenisation (they are content words in
        # general text) but they are noise in a gap statement: "a source
        # naming the person 'who, depot, fuel'" reads badly.
        skip = {"who", "what", "when", "where", "which", "why", "how",
                "whose", "whom", "many", "much", "there", "does", "did"}
        col = shards.partition_col("l", self._s.sharded)
        unknown = [t for t in dict.fromkeys(terms) if t not in skip
                   and not self._s.one(
            f"SELECT 1 FROM lexeme l JOIN mem_index m ON m.node_id=l.node_id "
            f"WHERE l.term=? AND {col} IN ({pq}) LIMIT 1",
            (t, *visible))]
        subject = " ".join(unknown[:4]) if unknown else query
        kind = {
            "person": "naming the person",
            "place": "giving the location",
            "time": "with a date or schedule",
            "quantity": "with the figure",
            "identifier": "carrying the exact identifier",
            "org": "naming the organisation",
        }.get(want or "", "covering")
        return (f"To answer this I would need a source {kind} "
                f"'{subject}' - nothing in the store mentions it.")

    def _absence_answer(self, query: str, partition: str,
                        t0: float) -> Recall | None:
        """'I looked on the 14th and it was not there' beats 'I don't know'."""
        now = self.clock.now()
        qt = set(lexical.tokenize(query))
        best = None
        for r in self._s.query(
                "SELECT * FROM absence WHERE partition=? AND "
                "(expires_at IS NULL OR expires_at > ?)", (partition, now)):
            if lexical.jaccard(query, r["query"]) > 0.5 or qt <= set(
                    lexical.tokenize(r["query"])):
                if best is None or r["searched_at"] > best["searched_at"]:
                    best = r
        if best is None:
            return None
        age_days = (now - best["searched_at"]) / DAY
        return Recall(
            State.SEARCHED_AND_ABSENT, [], query,
            f"{best['reason']} (scope={best['scope']}, checked "
            f"{age_days:.0f}d ago)", (time.perf_counter() - t0) * 1000)

    def _knew_once(self, terms: list[str], visible: dict[str, str],
                   query: str, t0: float) -> Recall | None:
        """'You told me, I no longer hold the detail, here is the source.'

        NOT the same as never having been told, and a completely different
        instruction to the user: go look at the source. Every system that
        deletes rows is incapable of distinguishing these two, and answering
        'I don't know' to something you were in fact told is the answer that
        destroys trust fastest.
        """
        pq = ",".join("?" * len(visible))
        col = shards.partition_col("l", self._s.sharded)
        found: dict[str, int] = {}
        for term in set(terms):
            for r in self._s.query(
                    f"SELECT l.node_id FROM lexeme l JOIN mem_index m "
                    f"ON m.node_id=l.node_id WHERE l.term=? "
                    f"AND {col} IN ({pq}) AND m.tier='pruned'",
                    (term, *visible)):
                found[r["node_id"]] = found.get(r["node_id"], 0) + 1
        if not found:
            return None
        best = max(found, key=lambda k: found[k])
        if found[best] / max(1, len(set(terms))) < 0.5:
            return None
        row = self._node_row(best)
        if row is None:
            return None
        chunk = Chunk(
            node_id=best, content="", score=0.0, retrievability=0.0,
            claim_class=row["claim_class"], provenance=self._provenance(row))
        return Recall(
            State.KNEW_ONCE, [chunk], query,
            f"recorded {(self.clock.now() - row['observed_at']) / DAY:.0f}d ago "
            f"from {row['source_ref']}; detail no longer held -- pull the source",
            (time.perf_counter() - t0) * 1000)

    def _group_cap(self, scored: list, group_by: str | None, per_group: int,
                   take: int) -> list:
        """Top-k per group rather than globally.

        `group_by` is 'source', 'episode', 'period' or None. Anything the
        cap excludes is appended afterwards, so a small store still fills the
        budget -- the cap shapes the ORDER, it never starves the answer.
        """
        if not group_by or per_group <= 0:
            return scored[:take]
        key = {"source": "source_ref", "episode": "episode_id",
               "period": "period_id"}.get(group_by, "source_ref")
        counts: dict[str, int] = {}
        primary, overflow = [], []
        for item in scored:
            row = item[2]
            g = row[key] if key in row.keys() and row[key] else "?"
            if counts.get(g, 0) < per_group:
                counts[g] = counts.get(g, 0) + 1
                primary.append(item)
            else:
                overflow.append(item)
            if len(primary) >= take:
                break
        return (primary + overflow)[:take]

    def _recollection(self, row) -> float:
        """How much context anchors this memory? Familiarity is not enough."""
        nid = row["id"]
        epi = per = None
        if nid.startswith("obs_"):
            r = self._s.one("SELECT episode_id,period_id FROM observation "
                            "WHERE id=?", (nid,))
            if r:
                epi, per = r["episode_id"], r["period_id"]
        # An episode of ONE is not context. Every observation is assigned an
        # episode, so counting bare membership gave every memory a free 0.30
        # and made the FAMILIAR state unreachable.
        has_epi = bool(epi) and self._scalar(
            "SELECT COUNT(*) FROM observation WHERE episode_id=?", (epi,)) > 1
        has_per = bool(per)
        neighbours = self._scalar(
            "SELECT COUNT(*) FROM assoc_edge WHERE src=?", (nid,))
        neighbours += self._scalar(
            "SELECT COUNT(*) FROM mention WHERE node_id=?", (nid,))
        decon = bool(self._s.one(
            "SELECT 1 FROM derivation_edge e JOIN derived d ON d.id=e.child_id "
            "WHERE e.parent_id=? AND d.kind='decontext' LIMIT 1", (nid,)))
        has_prov = bool(row["source_ref"] and row["source_ref"] != "inline")
        return metamemory.recollection_score(
            has_episode=has_epi, has_period=has_per, n_neighbours=neighbours,
            has_provenance=has_prov, decontextualised=decon)

    def _has_entity_kind(self, kind: str | None,
                         visible: dict[str, str]) -> bool | None:
        """Does the store hold ANY entity of the predicted answer type?

        Returns None when unknowable (no prediction, or no entity graph at
        all) -- which the gate treats as 'no evidence either way', never as a
        negative. A signal that is absent must not look like a signal that
        fired.
        """
        if kind is None:
            return None
        if not self._scalar("SELECT COUNT(*) FROM entity"):
            return None
        pq = ",".join("?" * len(visible))
        return bool(self._s.one(
            f"SELECT 1 FROM entity WHERE kind=? AND partition IN ({pq}) LIMIT 1",
            (kind, *visible)))

    def _fuse(self, partition: str, *, levels: int = 2) -> dict:
        """Cluster confusable memories into composites. Zero model calls.

        The interference sweep finds confusable PAIRS. Fusion resolves them:
        near-duplicates merge, and genuine clusters are promoted to one denser
        node that the retriever can return instead of five overlapping ones.
        Composites are ordinary `derived` nodes, so monotonicity applies and
        a composite is never more certain than its least certain member.
        """
        stats = {"merged": 0, "composites": 0, "protected": 0, "levels": 0}
        # Quarantined content never fuses: crafting near-duplicates so a
        # false representative wins the cluster is a real attack, and the
        # cheapest defence is simply not letting untrusted material into the
        # clustering pass at all.
        verbatim = {r["id"] for r in self._s.query(
            "SELECT id FROM observation WHERE partition=? AND "
            "(claim_class='verbatim' OR trust<>'trusted')", (partition,))}
        for level in range(levels):
            pairs = self._similar_pairs(partition, level)
            if not pairs:
                break
            plan = fusion.plan(pairs, verbatim=verbatim)
            stats["protected"] += plan.skipped_verbatim
            if plan.is_empty:
                break
            for keep, drop in plan.duplicates:
                self._s.write(lambda c, k=keep, d=drop: c.execute(
                    "UPDATE mem_index SET tier='cold' WHERE node_id=?", (d,)))
                stats["merged"] += 1
            for members in plan.clusters:
                if self._make_composite(members, partition, level):
                    stats["composites"] += 1
            stats["levels"] = level + 1
        return stats

    def _similar_pairs(self, partition: str, level: int
                       ) -> list[tuple[str, str, float]]:
        if not self._semantic or not self._vec.count():
            return []
        if level == 0:
            rows = self._s.query(
                "SELECT node_id FROM mem_index WHERE partition=? AND "
                "tier IN ('hot','warm') LIMIT 400", (partition,))
        else:
            rows = self._s.query(
                "SELECT id AS node_id FROM derived WHERE partition=? AND "
                "kind='summary' AND producer='fusion' LIMIT 200", (partition,))
        out: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for r in rows:
            for other, sim in self._vec.neighbours(
                    r["node_id"], space=Space.READ,
                    threshold=fusion.CLUSTER_THRESHOLD,
                    partitions=[partition]):     # G5: never fuse across a shard
                key = tuple(sorted((r["node_id"], other)))
                if key not in seen:
                    seen.add(key)
                    out.append((key[0], key[1], sim))
        return out

    def _make_composite(self, members: Sequence[str], partition: str,
                        level: int) -> str | None:
        rows = [self._node_row(m) for m in members]
        rows = [r for r in rows if r is not None]
        if len(rows) < 2:
            return None
        if any(r["claim_class"] == "verbatim" for r in rows):
            return None
        # Representative = highest confidence, then most recent. Its text
        # stands in for the cluster; nothing is paraphrased, because
        # paraphrasing needs a model and this pass deliberately has none.
        rep = max(rows, key=lambda r: (r["confidence"], r["observed_at"]))
        cid = self.derive(
            rep["content"], parents=list(members), kind="summary",
            producer="fusion", partition=partition,
            confidence=min(r["confidence"] for r in rows),
            epistemic=Epistemic.INFERRED)
        self._s.write(lambda c: c.executemany(
            "INSERT OR IGNORE INTO composite_member(composite_id,member_id,"
            "level) VALUES(?,?,?)", [(cid, m, level) for m in members]))
        return cid

    def _scalar(self, sql: str, params: Sequence[Any] = ()) -> int:
        row = self._s.one(sql, params)
        return int(row[0]) if row else 0
