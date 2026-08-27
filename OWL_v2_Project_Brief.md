# Project Brief: The O.W.L. Engine v2
### Observation & Wisdom Ledger — a provenance-first cognitive memory engine for offline analysis

**Version:** 2.0 (supersedes the v1 "Ontological Workspace Lattice" brief)
**Target:** single desktop machine, fully offline, ~16 GB VRAM, llama.cpp / GGUF, Python 3.11+, SQLite
**Companion document:** `OWL_Critique_and_Improvements.md` (rationale for every change from v1)

---

## 0. What changed from v1, in one paragraph

v1 argued from a metaphor ("be more like a brain") to mechanisms that, in a system whose value is analytical trustworthiness, reproduce the brain's *failure* modes — reconstructive overwriting, unattributed synthesis, self-generated content re-entering the evidence base. v2 keeps the good instinct (memory is metabolized, not accumulated) and makes it safe by splitting the biological metaphor along the seam biology can't: **an immutable evidence substrate, a decaying index, and a freely-rewritten derived layer.** Forgetting happens in the index. Reconstruction happens in derivation. Nothing that was ever observed is ever altered. On top of that, v2 adds the four things v1 lacked entirely — **provenance, metamemory, interference management, and an evaluation harness** — and replaces v1's hand-specified decay math with a fitted, benchmarked memory model.

---

## 1. Core philosophy

O.W.L. is an offline memory engine for analytical work. Its differentiating claim is not that it remembers more. It is:

> **O.W.L. can always tell you how it knows.**

Three commitments follow, and they order every other decision:

1. **Evidence is immutable.** Anything observed — a user utterance, a document, a tool result — is written once, never edited, never deleted except on explicit user command. Compression, abstraction, and forgetting operate on *indices and derivations*, never on the record.
2. **Provenance is transitive and monotone.** Every node knows its origin and its parents. Confidence can never exceed the minimum confidence of its ancestors. Abstraction cannot launder speculation into fact.
3. **Metamemory precedes memory.** The system knows whether it knows before it spends compute finding out, and "I have nothing on this" is a fast, reliable, first-class answer.

Everything else — decay, consolidation, the specialist swarm, dreaming — is subordinate to these and can be cut without breaking them.

---

## 2. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Inference | `llama-cpp-python`, async | **One resident base model + LoRA hot-swap** as the primary specialization mechanism |
| Base model | 7–8B instruct, Q4_K_M | Stays resident. ~5 GB. Never unloaded. |
| Specialists | LoRA adapters on the base | Tens–hundreds of MB, millisecond swaps |
| Composer | 8B (same base + composer LoRA) | v1's 24B is unnecessary once aggregation is mechanical (§6) |
| Structured output | **GBNF grammars** | Schema-valid JSON *structurally guaranteed*, not hoped for |
| Store | SQLite, WAL, **single writer task** | All writes through one queue. Non-negotiable. |
| Vectors | Abstracted behind `VectorIndex` | `sqlite-vec` (still 0.1.x) for < ~100k nodes; swap to HNSW beyond. Never call it directly from business logic. |
| Embeddings | Two spaces — see §5.2 | Separate write/read encoders |
| Validation | Pydantic v2, strict | |
| Scheduling | Sleep-pressure model (§8) | `psutil` is one input, not the trigger |

**VRAM budget (steady state):** base 5 GB + active LoRAs <1 GB + embedding model 0.5 GB + KV cache 2–4 GB ≈ **9–10 GB**, leaving headroom. No model swapping on the critical path.

---

## 3. Data model

### 3.1 The three layers

```sql
-- LAYER 1: SUBSTRATE — append-only, immutable, never rewritten
CREATE TABLE observation (
  id            TEXT PRIMARY KEY,
  created_at    INTEGER NOT NULL,
  origin        TEXT NOT NULL CHECK (origin IN
                  ('user_utterance','document','tool_output')),
  source_ref    TEXT NOT NULL,          -- URI, doc hash + byte offset, or session:turn
  content       TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  context_env   TEXT NOT NULL,          -- JSON: encoding-specificity cue environment (§5.3)
  period_id     TEXT REFERENCES period(id)
);
-- No UPDATE trigger permitted on this table. Enforce at the DB level.

-- LAYER 2: INDEX — mutable, decays, this is where forgetting lives
CREATE TABLE mem_index (
  node_id            TEXT PRIMARY KEY,
  -- FSRS / DSR state
  stability          REAL NOT NULL,     -- days until retrievability -> 0.9. Storage strength.
  difficulty         REAL NOT NULL,
  last_review        INTEGER NOT NULL,
  review_count       INTEGER NOT NULL DEFAULT 0,
  access_log         TEXT NOT NULL,     -- JSON array of timestamps -> enables ACT-R & spacing
  -- salience inputs
  surprise           REAL NOT NULL,     -- prediction error at encode (§5.4)
  open_loop          INTEGER DEFAULT 0, -- Zeigarnik bonus, clears on resolution
  -- tier
  tier               TEXT NOT NULL CHECK (tier IN ('hot','warm','cold','pruned')),
  write_vec_id       INTEGER,           -- pattern-separated
  read_vec_id        INTEGER            -- pattern-completion
);

-- LAYER 3: DERIVATION — freely rewritten, always traceable
CREATE TABLE derived (
  id            TEXT PRIMARY KEY,
  created_at    INTEGER NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN
                  ('summary','abstraction','graft','hypothesis','conflict','intention')),
  epistemic_tag TEXT NOT NULL CHECK (epistemic_tag IN
                  ('observed','reported','inferred','hypothesized')),
  producer      TEXT NOT NULL,          -- which node/LoRA/phase made this
  content       TEXT NOT NULL,
  confidence    REAL NOT NULL,
  supersedes    TEXT REFERENCES derived(id),   -- never overwrite; supersede
  locked_at     INTEGER,                        -- reconsolidation window (§5.6)
  falsifier     TEXT                            -- REQUIRED when kind='hypothesis'
);

CREATE TABLE derivation_edge (
  child_id  TEXT NOT NULL,
  parent_id TEXT NOT NULL,              -- observation.id or derived.id
  role      TEXT NOT NULL,              -- 'evidence' | 'contradicts' | 'graft' | 'context'
  PRIMARY KEY (child_id, parent_id, role)
);
```

### 3.2 The monotonicity invariant

Enforced in code, tested in CI:

```
confidence(node)      <= MIN(confidence(parents))
epistemic_tag(node)   >= MAX(epistemic_tag(parents))    -- observed < reported < inferred < hypothesized
```

A node with any `hypothesized` ancestor is `hypothesized`, forever. There is no path by which dream output becomes fact without an explicit, logged, human-or-corroboration promotion event.

### 3.3 The hierarchy (Self-Memory System)

Replaces v1's flat table + type column. Three levels, accessed top-down:

```sql
CREATE TABLE period (      -- engagements, projects, cases. Long-lived.
  id TEXT PRIMARY KEY, label TEXT, opened_at INTEGER, closed_at INTEGER,
  summary_id TEXT REFERENCES derived(id)
);
CREATE TABLE general_event (  -- recurring activities within a period
  id TEXT PRIMARY KEY, period_id TEXT REFERENCES period(id), label TEXT,
  summary_id TEXT REFERENCES derived(id)
);
-- event-specific knowledge = observation rows, linked to a general_event
```

Retrieval descends: period → general event → specifics. Closing a period demotes it coherently as a unit. Summaries live at real structural boundaries rather than wherever a chunker happened to cut.

---

## 4. Runtime path (the critical path — everything else is async)

```
user input
   │
   ├─► [1] ENCODE      (async, off critical path)
   │
   ├─► [2] FOK GATE    ~50 ms — triage before any expensive work
   │        │
   │        ├─ DON'T-KNOW ────────────────────────► "I have nothing on this."   [DONE]
   │        ├─ KNOW ─────────────► direct answer from memory                    [DONE]
   │        ├─ KNOW-WHERE ──────► [3] targeted retrieve → [5] compose
   │        └─ TIP-OF-TONGUE ───► [3] wide retrieve → [4] full swarm → [5]
   │
   ├─► [3] RETRIEVE    hierarchy descent + read-space completion + SR prefetch
   ├─► [4] SWARM       LoRA specialists, structured rounds, capped at 4–7 chunks each
   ├─► [5] COMPOSE     mechanical pooling → prose generation
   └─► [6] ToM         presentation-only, claim-diff enforced
```

**Async, never in the request path:** reconsolidation, grafting, de-interference sweeps, compression tests, calibration updates, prefetch-matrix updates, all of §8.

---

## 5. Pillar I — Memory that works

### 5.1 Salience: FSRS, not a hand-rolled exponential

Adopt the **DSR model** (Difficulty / Stability / Retrievability) from FSRS. Vendor an open implementation (~100 lines).

- **Retrievability** `R(t)` — probability of successful recall now → **drives retrieval ranking**
- **Stability** `S` — days until `R` falls to 0.9; this is Bjork's *storage strength* → **drives tier transitions**
- **Difficulty** `D` — how hard this item is to stabilize → **drives compute allocation**

Every retrieval is a "review": success raises stability, failure lowers it. Because `access_log` retains *every* timestamp, the spacing effect is represented natively — ten accesses in an hour and ten spread over a year produce different stability, which v1's `S₀·e^(−λt) + a·c` could not express.

**Rejected from v1:** the additive `(a·c)` term (creates immortal junk), the exponential form (wrong — human forgetting is power-law; Wixted & Ebbesen 1991), and the fixed 30/60-day thresholds (arbitrary; the fitted model gives a per-item answer).

*Alternative if you prefer first-principles over fitted:* ACT-R base-level activation, `B_i = ln(Σ_j t_j^(−d))`, `d ≈ 0.5` (Anderson & Schooler 1991) — derived as the optimal Bayesian estimate of *"will this be needed right now?"* from environmental statistics. Equally acceptable. Pick one; don't blend them.

### 5.2 Two embedding spaces — pattern separation and pattern completion

Standard RAG collapses similar items together in one space, which *causes* the interference problem it then struggles with. Complementary Learning Systems says the write path should do the opposite of the read path.

**Write encoder (pattern separation).** Embed `content + distinguishing context` — period id, timestamp bucket, source, surrounding turn. Deliberately pushes near-duplicates apart. Optionally sparsify (top-k) the vector. Two similar weekly meetings must land in different places.

**Read encoder (pattern completion).** Embed the query semantically. Retrieve a *neighbourhood*, not a top-1. Discriminate afterward, in the completion step.

> Retrieve broadly, then discriminate — rather than retrieve narrowly and hope.

### 5.3 Encoding: deep, not shallow

v1's Apperception Node did entity extraction — shallow processing in Craik & Lockhart's sense. Depth of encoding predicts retention far better than downstream cleverness can recover. Three additions, all paid once at write time:

- **Elaborative encoding** — extract not just *what* was claimed but *why it matters here* and *what it connects to*.
- **Generation effect** — have the encoder generate *the questions this memory answers*, and index those. Retrieval then matches question↔question rather than question↔statement, a far better-posed problem.
- **Encoding specificity** — store `context_env`: the active goal, the prior turns, the open period, what else was in play. Reinstate it as part of the retrieval query.

### 5.4 Surprise-gated priority

Before writing, the resident model predicts the content from existing context. **Encode priority ∝ prediction error.** Predictable content is nearly free (store a pointer, it's derivable). Surprising content is expensive and valuable — surprise *is* information content.

> **Hard rule: surprise raises retention priority; surprise must never raise confidence.**

(Flashbulb memories are held with very high confidence and decay in accuracy at ordinary rates — Talarico & Rubin 2003. Priority and confidence are two variables. Conflating them builds a system most certain about exactly what it should doubt.)

### 5.5 Apperception Node — contextualized, not blind

v1 made this node blind to prevent anchoring. Blindness breaks encoding specificity and makes reference resolution impossible ("same as last time," "that's still broken"), producing confidently-wrong payloads. v2 gives it the context it needs and prevents anchoring structurally instead:

- **Two separate calls, different prompts:** fact extraction, then stance/framing extraction. Never one call doing both.
- **Extract claims *and* their negations.** Symmetry defeats one-sided framing.
- **`user_framing` is a labelled, quarantined field** — not discarded. It is data about the user, which §7 consumes.
- **GBNF-constrained decoding** guarantees schema validity structurally.

```python
class ApperceptionPayload(BaseModel):
    factual_claims:   list[Claim]        # each with polarity + negation
    entities:         list[Entity]       # with resolved referents
    core_intent:      str
    constraints:      list[str]
    temporal_context: str | None
    user_framing:     UserFraming        # QUARANTINED — routed only to the ToM worker
    unresolved_refs:  list[str]          # explicit failures, not silent guesses
```

`unresolved_refs` matters: a blind extractor fails silently. This one fails loudly.

### 5.6 The reconsolidation window

On retrieval, a derived node enters `labile` state for a bounded window (the session, or N minutes). While labile it can be updated in place. When the window closes, `locked_at` is set and any further change must create a **superseding** node. Clean concurrency semantics, no version thrash, and "what did we used to think, and what changed our minds?" stays a one-line query.

**Never overwrite. Always supersede.**

---

## 6. Pillar II — Memory that improves

### 6.1 Reconstructive compression (replaces v1's timer-based summarization)

**This is the most important mechanism in v2.** v1 compressed memories on a 30-day timer and pruned at 60 — a mechanism functionally identical to Bartlett's levelling/sharpening/rationalization, which produces confident summaries that no longer correspond to what happened.

v2 makes compression an **empirical test** rather than a schedule. During the NREM phase:

1. Take a candidate node. Hide the body. Show the system only the cue and neighbouring context.
2. Ask it to reconstruct the content.
3. Diff the reconstruction against the original.

| Result | Meaning | Action |
|---|---|---|
| Reconstructs accurately | Content is derivable from the model + neighbours | **Compress to cue + delta.** Safe by demonstration. |
| Reconstructs partially | Some residual information | Store the residual verbatim; compress the rest |
| Fails to reconstruct | Carries real non-derivable information | **Keep verbatim. Raise priority.** |

This is self-verifying lossy compression: you only discard what you have *proven* you can rebuild. It also applies the retrieval-practice effect (Roediger & Karpicke 2006 — testing beats restudying) to the machine, so the same pass that decides compression also strengthens the memory.

The substrate is untouched throughout. Compression only ever rewrites the derived layer and the index.

### 6.2 Schema-delta storage

Compress the rule; keep the exception verbatim. Schema-consistent material is safe to compress because the schema regenerates it. Schema-*violating* material is the information content, and is exactly what a naive summarizer deletes first because it reads as an outlier. Storage cost then scales with **novelty**, not with volume — which is the correct scaling law for an analyst tool.

### 6.3 De-interference sweep (the primary maintenance job)

v1 pruned by age. But retrieval failures are caused by **interference**, not decay (Underwood 1957 onward) — forty near-identical memories about a recurring topic, none clearly right. Age-based pruning does nothing about this and can make it worse.

Run periodically, ahead of decay. Find high-similarity node pairs and partition:

- **Redundant** (same content, same source) → merge, keep highest fidelity, sum access statistics.
- **Contradictory** (same topic, incompatible claims) → this is "belief tension," now with a principled trigger. Resolve by provenance quality; if unresolvable, **write the conflict as an explicit `kind='conflict'` node** rather than silently picking a winner.
- **Distinct but confusable** → **differentiate**: rewrite the *retrieval cues* (never the content) to emphasize what distinguishes them. This is what a human expert does when learning to tell two similar things apart.

**Retrieval-induced forgetting.** When a memory is successfully used, mildly demote the *retrieval strength* of near-neighbours that weren't used (Anderson, Bjork & Bjork 1994). Free, self-organizing disambiguation — the index sharpens around actual use without deleting anything. Implement as **ranking demotion, never as an exclusion filter.**

### 6.4 Mycelial grafting (retained from v1, constrained)

If two concept nodes are repeatedly co-retrieved, an idle worker prompts for the logical connection and writes a `kind='graft'` node. **Constraints v1 lacked:** the graft inherits `epistemic_tag` from its parents monotonically, carries full `derivation_edge` links, and is `inferred` — not fact — unless corroborated by an independent observation.

### 6.5 Successor-representation prefetch

Maintain a sparse transition matrix over nodes: given that A was retrieved, what tends to be retrieved within the next few turns? On retrieving A, **prefetch A's successors** into working memory before they're requested (Dayan 1993; Stachenfeld et al. 2017 on the hippocampus as a predictive map).

A sparse matrix and a counter. Highest performance-per-line-of-code item in the design, because your dominant latency is context assembly, and this assembles it before the question arrives.

---

## 7. Pillar III — Metamemory (new in v2)

### 7.1 The Feeling-of-Knowing gate

Fires before any retrieval or model load. Inputs: embedding-space density around the query, best-match retrievability, neighbourhood conflict rate, period-hierarchy match. Output in ~50 ms:

| State | Signature | Action |
|---|---|---|
| **KNOW** | high-`R` direct match, low conflict | answer directly; skip the swarm |
| **KNOW-WHERE** | no direct match, dense neighbourhood | targeted retrieval + one specialist |
| **TIP-OF-TONGUE** | high familiarity, **low** retrievability, high neighbour conflict | **escalate** — this is the interference signature; widen search and flag for a de-interference pass |
| **DON'T-KNOW** | sparse neighbourhood | say so immediately |

Two payoffs beyond speed. **DON'T-KNOW as a fast, reliable first-class answer** is worth more to an analyst than always producing something. And **TIP-OF-TONGUE is a diagnostic**, not a failure — it's precisely where competing memories are blocking each other, so it feeds §6.3.

### 7.2 Calibration tracking

Every stated confidence is logged against outcome. Maintain a running **Brier score** per specialist, per claim type. These measured numbers replace v1's approach of having a large model *guess* at its subordinates' biases (§9.2). Calibration is a plot you can look at; a homunculus's opinion is not.

---

## 8. Pillar IV — Sleep (async, and the whole of it is optional)

### 8.1 Sleep pressure, not idle CPU

v1 triggered on `psutil` idle. That's a resource check, not a need check. Use the **two-process model** (Borbély 1982):

- **Process S** — homeostatic pressure, accumulating with use: unconsolidated writes, unresolved conflicts, orphan nodes, pending reconstruction tests.
- **Process C** — the learned rhythm of when this machine is reliably free.

Consolidate when **pressure is high AND the window is open**, additionally gated on: explicit user consent, AC power, thermal headroom, and a visible "consolidating" indicator with a kill switch.

### 8.2 Two phases, opposite parameters

SWS and REM do different jobs, and they want opposite sampling and opposite node selection. v1 tried to do both in one pass, which gets you neither.

| | **NREM phase** (first, longer) | **REM phase** (second, shorter, capped) |
|---|---|---|
| Job | Consolidate | Recombine |
| Node selection | Recent, high-surprise, **high-interference** | **Distant** pairs — deliberately *low* similarity |
| Sampling | Low temperature, grammar-constrained | High temperature, unconstrained |
| Work | Reconstruction tests → compression; de-interference; conflict resolution; index rebuild; SR matrix update | Remote-association search; analogy; hypothesis generation |
| Output tag | `inferred`, tied to substrate | **`hypothesized`, quarantined** |

(Cai et al. 2009 found REM specifically improved Remote Associates Test performance where quiet rest did not — the phase split isn't decoration, the two jobs genuinely want different regimes.)

### 8.3 Quarantine and mandatory falsifiers

Every REM output is `kind='hypothesis'`, `epistemic_tag='hypothesized'`, `producer='rem_phase'`, and **must** carry a `falsifier` — a concrete, runnable check ("if true, document X should contain Y"). Rows without a falsifier are rejected at insert.

Lifecycle: **generated → tested → promoted / archived-failed / expired.** Untested hypotheses expire. Hypotheses never surface as fact and never leave the machine in a user-exported report without an explicit promotion step.

### 8.4 On weight updates: don't, in v1

v1 proposed fine-tuning the base model on self-play output. Training on freshly-generated adversarial debate without interleaving representative old data is the canonical recipe for **catastrophic interference** (McClelland, McNaughton & O'Reilly 1995 — this is precisely why the brain needs offline interleaved replay) *and* for **model collapse** from recursive training on generated data.

**v2 position:** the base model's weights are frozen. Specialization is LoRA-only, per-domain, reversible, and trained on curated real data. If you later fine-tune, interleaving is mandatory and synthetic data is capped as a fraction of the corpus.

---

## 9. Pillar V — The swarm

### 9.1 Specialists: epistemic roles, not Gardner intelligences

v1 used the eight Gardner multiple intelligences. That theory has essentially no empirical support as an account of human cognition (Visser, Ashton & Vernon 2006; Waterhouse 2006), and more practically it carves the space wrong for this application — "Bodily-Kinesthetic" has no job here, while "verify this number against its source" has no home.

v2 uses **epistemic roles** — the stages an analytical claim actually passes through. Each is a LoRA on the resident base, and each is separately evaluable:

| Role | Job |
|---|---|
| **Extractor** | What does the source actually say? |
| **Corroborator** | What independently supports or contradicts it? |
| **Adversary** | How is this wrong? *(protected budget — see 9.3)* |
| **Quantifier** | What do the numbers say; do they check out? |
| **Historian** | What did we previously believe, and what changed? |
| **Synthesist** | What follows from all of the above? |

*Longer term:* derive the specialist set empirically by clustering the actual query log after a month of use. Nothing beats fitting the real workload.

### 9.2 Decorrelation, not isolation

v1 enforced total blindness between nodes. The goal was decorrelated error; blindness is an expensive and lossy way to get it, and it creates a worse problem — **the hidden profile effect** (Stasser & Titus 1985), where groups fail to surface uniquely-held information. With total isolation, only the Composer can pool unshared information, making it a single point of bias for the exact failure the architecture was meant to prevent.

v2 gets decorrelation from four cheaper sources:

- **Bagging** — different specialists get different *subsets* of retrieved evidence.
- **Varied priors** — different LoRAs, prompts, temperatures, framings.
- **Dialectical bootstrapping** (Herzog & Hertwig 2009) — the "crowd within": ask a specialist to assume its first answer was wrong and argue the other side; average. Nearly free, and ideal for a VRAM-limited box.
- **Delphi rounds** — structured, anonymized, iterated. Information flows; attribution and social pressure don't.

**And the fix v1 needed most — the unshared-information premium:** when exactly one specialist raises a claim, *and that claim falls inside its competence domain*, weight it **up**. Uniqueness is evidence, not noise. One line of aggregation logic, more cognitive diversity than the entire isolation apparatus delivered.

**Isolation is preserved where it matters:** no specialist sees another's output before submitting its own, and none sees `user_framing`. KV-cache sharing of the common evidence prefix does **not** violate this — isolation is a property of content visibility, not of tensor memory pages. Share the prefix; it's a large latency win.

### 9.3 The compute economy, fixed

v1's "good performance earns bigger budgets" is a pure-exploitation bandit with a single stable outcome: one early winner takes everything and you're back to a monolith. It also Goodharts directly — rewarding high-confidence output trains overconfidence, the worst possible trait here.

| v1 | v2 |
|---|---|
| Reward historical effectiveness | **Thompson sampling / UCB** — an explicit exploration term |
| Reward usage | **Marginal contribution** — leave-one-out / Shapley-style impact on final quality |
| Reward confidence | **Reward calibration** — Brier score. "60% and right 60% of the time" beats "95% and right 80%." |
| Budgets can collapse to zero | **Hard floors.** Capacity you starved is capacity you can't discover a use for. |
| — | **Protected Adversary.** Unconditional budget, cannot be earned or lost. Authentic minority dissent improves group reasoning even when the dissenter is wrong (Nemeth) — but only if it can't be starved. |

### 9.4 The Composer, decomposed

v1's Composer allocated compute, modelled every specialist's biases, reweighted outputs, ran ToM, and wrote the answer. That's a homunculus — the monolith rebuilt in the middle of the architecture, and a single point through which all bias flows. It was also the largest model, so the dominant latency and VRAM cost.

v2 makes aggregation **mechanical** and shrinks the LLM's job to prose:

1. **Pool evidence arithmetically** — log-odds pooling over claims, weighted by measured per-specialist reliability (§7.2). Deterministic, auditable, testable, free.
2. **Detect conflict structurally** — contradiction between specialists is found by rule, not by asking a model to notice.
3. **Apply the unshared-information premium** (§9.2).
4. **Then, and only then, generate prose** over a resolved, weighted, conflict-annotated claim set. An 8B model does this well — which is what frees the 24B slot entirely.

### 9.5 Working-memory chunk budget

**Hard cap: 4–7 chunks per specialist context.** Human working memory holds about four chunks (Cowan), and LLMs degrade measurably on long contexts (lost-in-the-middle). Both literatures point the same way. Spend saved compute on making each chunk *denser and better-selected*, not on adding more.

Chunk density over time is also your best proxy for whether the system is genuinely learning (§11) — expertise is better chunks, not more memory (Chase & Simon 1973).

---

## 10. Pillar VI — Theory of Mind, firewalled

### 10.1 The firewall

v1 asked, before delivery: *"will this cause confusion? does it contradict a deeply held premise?"* — and then adjusted. Read as an objective function, that optimizes for user comfort and non-contradiction of existing belief. That is sycophancy, and in an analyst tool the most valuable output is frequently the one that contradicts a deeply held premise.

**v2 rules:**

- ToM receives the draft **after** all claims and confidences are final.
- It may emit **only** presentation directives: ordering, length, vocabulary, how much scaffolding before the hard part, conclusion-first vs. evidence-first.
- It **may not** alter, soften, hedge, drop, or reorder-to-bury any claim, and may not touch any confidence value.
- **Mechanically enforced:** extract the claim set from the pre- and post-ToM drafts, diff them, **reject the revision if the sets differ.** A test, not a guideline.

### 10.2 What ToM is actually for: the curse of knowledge

Camerer, Loewenstein & Weber (1989): people who know something cannot successfully model someone who doesn't. That is the real gap between a system that has read everything and a user who hasn't.

The right ToM question is not *"will this upset them?"* but:

> **"What does this explanation assume that they have never been told?"**

That is a presentation question, answerable from the user model, and it makes the output *better* rather than more agreeable.

### 10.3 Zone of proximal development, and a recursion cap

- **Target the ZPD** (Vygotsky): aim explanations just above demonstrated competence. Estimable from what the user has asked, corrected, and used unprompted.
- **Cap recursion at 2.** System models user; system models user's model of the system. Stop. Humans manage roughly four to five orders of intentionality and degrade sharply past that; deeper recursion here is cost without benefit.

### 10.4 Internal ToM → replaced by measurement

v1 had the Composer maintain a predictive model of its specialists' biases. v2 **measures** them instead (§7.2). A per-specialist Brier score by claim type is strictly better than a large model's introspective guess: cheaper, auditable, and it can be wrong in ways you can see.

---

## 11. Pillar VII — Proactivity as prospective memory

The "Watch Officer" was reinventing **prospective memory** — memory for intentions rather than for the past (Einstein & McDaniel). Adopting the frame converts a vague background monitor into a bounded, auditable queue.

```sql
CREATE TABLE intention (
  id           TEXT PRIMARY KEY,
  created_at   INTEGER,
  trigger_kind TEXT CHECK (trigger_kind IN ('event','time')),
  trigger_spec TEXT NOT NULL,      -- CONCRETE and matchable. Vague cues are rejected.
  action       TEXT NOT NULL,
  status       TEXT CHECK (status IN ('pending','fired','completed','expired')),
  origin_ref   TEXT NOT NULL       -- what created this intention
);
```

- **Event-based triggers** — cue-matching on ingest. Cheap, no polling.
- **Time-based triggers** — expensive for humans (needs self-initiated monitoring), trivial here. Lean on them.
- **Intention superiority** — pending intentions carry a persistent activation bonus that clears on completion.
- **Cue specificity is enforced** — prospective memory fails on vague cues. `trigger_spec` must be concrete and matchable; reject fuzzy topics at insert.

**Zeigarnik open loops.** Unanswered questions are first-class objects with the `open_loop` flag set. They keep pulling on attention until resolved, and they're prime candidates for the REM phase. This is what makes the system feel like a colleague rather than a search box: it remembers what's still open.

---

## 12. Evaluation harness — build this first

v1 specified five elaborate pillars and zero ways to know whether any helped. **The harness ships before the clever parts**, or you will ship all of it and never learn which two were hurting.

| Test | Method | Fail condition |
|---|---|---|
| **Retention curve** | Inject known facts, simulate N days of clock, query | Recall departs from intended curve |
| **Confabulation** | Probe for facts never provided, and for facts explicitly deleted | *Any* confident answer to an absent fact |
| **Source attribution** | "How do you know that?" on every claim | Any claim traced to the wrong origin class |
| **Monotonicity** | Fuzz the derivation graph | Any node's confidence exceeds its parents' minimum |
| **Calibration** | Brier / ECE on stated confidences | Systematic overconfidence — **the metric that matters most** |
| **Interference** | Insert N near-duplicate confusable memories, then query | Accuracy degrades as N rises |
| **Hidden profile** | Give a decisive clue to exactly one specialist | Composer's answer doesn't change |
| **Chunking growth** | Compression ratio of new input over time | Flat → it isn't building expertise |
| **Sleep ablation** | A/B dream engine on/off, next-day task performance | No measurable gain → **cut §8** |
| **Latency / VRAM** | p50, p95, peak | Exceeds budget |

**The sleep ablation is the decisive one.** §8 is the most expensive, most complex, most speculative part of the system. It ships behind a flag, defaulted off, and you should be genuinely willing to delete it.

---

## 13. Privacy and consent

- **Per-directory opt-in.** Never "monitor the drive."
- **Visible ingestion log** — "what have you read?" is a one-click view.
- **Deletion propagates.** Deleting a source tombstones it, then walks `derivation_edge` to invalidate every derived, grafted, and hypothesized descendant. Append-only storage makes this real work — design it in now; retrofitting is brutal.
- **Dream content never exports.** `hypothesized` nodes stay internal and marked until an explicit promotion step.

---

## 14. Build order

Ranked by load-bearing, which is roughly the inverse of v1's ordering by interestingness.

**Phase 0 — Foundations.** Provenance schema. Substrate/index/derivation split. Monotonicity invariant + CI test. Single-writer queue. **Evaluation harness.**

**Phase 1 — Memory that works.** FSRS salience. FOK gate. De-interference sweep. Two embedding spaces. Self-Memory System hierarchy. Deep encoding.
→ *The system is genuinely valuable at the end of Phase 1.*

**Phase 2 — Memory that improves.** Reconstructive compression. Schema-delta storage. Surprise-gated encoding. SR prefetch. Reconsolidation windows.

**Phase 3 — The swarm.** LoRA specialists on one resident base. Mechanical pooling. Calibration-weighted reliability. Protected Adversary. Unshared-information premium. Chunk budget.
→ *The system is excellent at the end of Phase 3.*

**Phase 4 — Proactivity.** Prospective memory queue. Zeigarnik loops. Consent-scoped ingestion + visible log.

**Phase 5 — Sleep, behind a flag, default off.** Sleep pressure. NREM/REM split. Quarantined hypotheses with mandatory falsifiers. **Run the ablation. Be willing to delete it.**

---

## 15. First implementation task

Build **Phase 0** in this order:

1. `owl/schema.sql` — the three-layer schema above, with an `UPDATE` trigger on `observation` that raises. Immutability enforced by the database, not by convention.
2. `owl/provenance.py` — `Node`, `DerivationGraph`, and `assert_monotonic()`. Plus a property-based test that fuzzes the graph and asserts no confidence or epistemic-tag violation is reachable.
3. `owl/writer.py` — the single writer task. Every mutation goes through one `asyncio.Queue`. No exceptions.
4. `owl/eval/harness.py` — the confabulation, source-attribution, and monotonicity tests from §12, runnable against a stub store.

Only once §12's confabulation test passes against a trivial store does any inference code get written.

---

### Design principles, one line each

1. **Decay the index, never the evidence.**
2. **Never overwrite. Supersede.**
3. **Provenance is transitive and monotone.**
4. **Only compress what you can prove you can rebuild.**
5. **Surprise raises priority, never confidence.**
6. **Store the exception; compress the rule.**
7. **Decorrelate errors; don't blindfold nodes.**
8. **Reward calibration, not confidence.**
9. **ToM shapes presentation, never content.**
10. **Measure it, or cut it.**
