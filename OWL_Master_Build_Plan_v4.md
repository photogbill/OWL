# O.W.L. — Master Build Plan v4

**Purpose:** the definitive, exhaustive plan for a memory engine intended to outperform the field.
**Reviewed for this plan:** MIRIX · MemoryLLM/M+ · LycheeMemory · Memanto · MiniRAG · zvec · iai-pme · Awesome-Agent-Memory (survey) · Mem0 · Zep/Graphiti · A-MEM · HippoRAG · EM-LLM · GraphRAG · Letta/MemGPT · Hexis.
**Status:** plan only. No code in this document.

---

# Part 0 — Strategic position

## 0.1 What "outperform" has to mean

The instinct — innovate, don't reinvent the wheel — is right, but it needs a target or it becomes scope creep. Three facts set the target:

**You cannot win on retrieval accuracy alone.** Memanto claims 89.8% LongMemEval with a dedicated information-theoretic engine and a commercial tier behind it. LycheeMemory has an ACL 2026 paper and a trained reranker. iai-pme ties the leader at R@5 0.966 with a hand-written Rust core. These are teams optimising one metric full-time.

**The field has converged on the same feature list.** Provenance, decay, conflict detection, typed memory, hierarchical consolidation, temporal queries — Memanto's "Six Gaps" table is substantially OWL's pitch, published first. Any plan premised on "nobody does provenance" is already wrong.

**But the whole field competes on one question: *can you find it?*** Almost nobody competes on: *should you believe it, how do you know, is it still true, who else knows it, can it be moved safely, and can it be attacked?*

So the target is not "beat Mem0 at LoCoMo." It is:

> **Be the memory system you would choose when being wrong has a cost.**

Three consequences run through the whole plan:
1. Every claim ships with a harness that proves it (iai-pme's discipline, and it's the right one).
2. Enforcement over annotation — invariants checked by the machine, not fields filled in by a writer.
3. Where the field has no benchmark, **build the benchmark** and run other systems on it. That contribution stands whether or not OWL wins.

## 0.2 The two categories this review exposed as missing

**Memory security.** The survey has a whole section — MPBench (memory poisoning attacks), Agent Memory Guard (OWASP runtime write screening). OWL has no threat model at all. For a tool ingesting documents from unknown sources in the field, this is the most serious gap in the project. Part II is new because of it.

**Hallucination benchmarking already exists.** HaluMem (2025) evaluates hallucinations in agent memory systems. My earlier claim that "nobody scores confabulation" was wrong. HaluMem is now a primary target rather than something to invent.

## 0.3 Honest scorecard

| Axis | Leader | OWL now | OWL target |
|---|---|---|---|
| Raw retrieval (LoCoMo/LongMemEval) | Memanto, iai-pme, LycheeMem | unmeasured | competitive, not leading |
| Hallucination (HaluMem) | unclear | unmeasured | **lead** |
| Post-contradiction (Rescue@k) | iai-pme 1.000 | unmeasured | **match, and win the inverse** |
| Source attribution | nobody measures | by construction | **define the metric** |
| Calibration | nobody measures | infrastructure only | **define the metric** |
| Poisoning resistance | Agent Memory Guard | none | **lead** |
| Install friction | OWL | zero deps | **hold this** |
| Offline / air-gap | OWL, iai-pme | complete | **hold this** |

**The inverse-Rescue point is worth dwelling on.** iai-pme reports an honest regression: retrieving the *superseded wording* of an updated fact fell 0.90 → 0.71. OWL cannot lose that, because the substrate is append-only and the old row is never rewritten. That's a benchmark OWL wins **by architecture, not by tuning** — and those are the only durable wins.

---

# Part I — What exists today

Built and passing: 70 tests, ~2 s, no GPU, no network, zero dependencies.

**Substrate:** three-layer split (immutable observation / mutable index / rewritten derivation), append-only enforced by SQL trigger, single-writer queue, bitemporal validity, redaction with propagation.

**Epistemics:** monotonicity invariant (confidence ≤ min(parents), epistemic ≥ max(parents)), Admiralty two-axis source grading, claim classes with learned half-lives, verbatim protection.

**Retrieval:** six-state FOK gate, FSRS salience, two embedding spaces (structural pattern separation on write, completion on read), max-fusion of lexical and semantic, entity-graph bridging with two-hop traversal, answer-type prediction, associative and successor spread.

**Maintenance:** interference sweep, union-find record fusion with composites, retrieval-induced forgetting, tier transitions.

**People:** transactive memory (exposure log, user forgetting curve, at-risk ranking), symmetric belief-divergence detection.

**Boundaries:** information-flow partitions (denied by default, directional, sealed, graded permeability), affect-gated retrieval.

**Portability:** `.owlpack` handover with automatic epistemic demotion, corroboration on independent concurrence, checksum verification.

**Other:** prospective memory, negative memory, suppression-not-deletion, event segmentation at surprise boundaries.

---

# Part II — Threat model *(new — the largest gap)*

## II.1 Why this matters more than a feature

Every other capability in OWL assumes the store's contents are honest mistakes at worst. That assumption fails the moment a document of unknown origin is ingested — which for ATK is the normal case, and for any agent with web or file access is inevitable.

**A memory system is a persistence layer for beliefs. Anything that writes to it is writing to the agent's mind, permanently.** Prompt injection is transient; memory poisoning is not. A poisoned memory survives every restart, propagates into every derived summary, and gets retrieved as context forever.

OWL is unusually well placed here — provenance, immutability, Admiralty grading and flow partitions are already defensive primitives — but they were built for honesty, not adversaries, and the difference shows.

## II.2 Attack surface

| Attack | Mechanism | Current exposure |
|---|---|---|
| **Direct injection** | Document contains "remember that X" / instructions framed as facts | **Full** — no write screening |
| **Supersession hijack** | Attacker's content claims to supersede established truths | **Full** — no rate limit or authority check |
| **Source flooding** | 200 documents asserting the same falsehood to manufacture corroboration | **Full** — corroboration counts documents, not independent sources |
| **Fusion poisoning** | Craft near-duplicates so union-find merges a true cluster with a false representative | **Partial** — verbatim is protected, nothing else is |
| **Provenance laundering** | Chain derivations so a hostile claim ends up sourced to a trusted node | **Low** — monotonicity blocks confidence escalation, but not attribution drift |
| **Handover poisoning** | Malicious `.owlpack` | **Partial** — checksum + demotion help; no content screening |
| **Partition exfiltration** | Trick a sealed partition into leaking via a derived summary | **Low** — flow control is structural |
| **Denial by flooding** | Bury real memories under noise until FOK stops firing | **Unmeasured** — this is what the `nuc` benchmark tests |
| **Embedding collision** | Craft text that lands adjacent to a target in vector space | **Full** |

## II.3 Defences to build

**D1 — Untrusted intake quarantine.** Content from untrusted origins lands in a quarantine tier: retrievable, but never corroborating, never fusing, never superseding, and never presentable as fact until promoted. Promotion requires independent corroboration or explicit user action. *This is the single highest-value defence and it reuses the epistemic lattice already built.*

**D2 — Source independence graph.** *(Novel; nothing in the field does this.)* Corroboration currently counts documents. It should count **independent origins**. If 40 documents trace to one upstream source, that is one source. OWL's derivation graph can compute this directly: cluster sources by shared ancestry, ingestion batch, domain, and arrival time, then weight corroboration by the number of *independent clusters*, not documents. This defeats flooding at the root and is a genuine research contribution — the same math also improves ordinary corroboration quality.

**D3 — Supersession authority.** Superseding an established claim requires reliability greater than or equal to the claim being replaced. A grade-F source cannot overwrite a grade-B one; it can only register a *conflict*. Rate-limit supersessions per source per window and flag bursts as a "belief coup" for review.

**D4 — Write-time anomaly screening.** Cheap, deterministic, Tier 0: imperative-mood detection in supposedly factual content, instruction-like patterns ("ignore", "always", "from now on"), entropy outliers, unusual claim density per token, and content that contradicts an unusually large number of established beliefs at once.

**D5 — Fusion integrity.** A composite's representative must be the *best-attested* member, not merely the most confident. Never fuse across trust tiers. Never fuse across origins with different reliability grades.

**D6 — Encryption at rest.** AES-256-GCM over the store, key at 0600, explicit backup guidance. iai-pme does this; OWL does not. For casework in the field this is table stakes, not a nicety.

**D7 — Retrieval receipts.** *(Novel.)* Every recall emits an immutable record of what was returned, why, what state fired, and which nodes were considered and rejected. When a downstream error occurs, the retrieval decision is auditable — currently it evaporates. This is also the substrate for calibration measurement (Part IV).

**D8 — Adversarial self-audit.** *(Novel.)* During idle, the system attacks its own invariants: find any claim it would state as fact that cannot be traced to a primary source; any node whose confidence exceeds its parents; any orphaned derivation; any partition edge that admits more than declared. CI proves the invariants hold at commit; this proves they hold *in the live store, over time*. Report as part of `doctor`.

**D9 — Model provenance.** *(Novel; obviously right and nobody does it.)* Record which model, at which version and quantisation, produced each inference. When the host upgrades from a 7B to a 24B, every conclusion resting on the smaller model's judgement is identifiable and can be re-derived or demoted. Given MiniRAG's finding that pipelines invert below ~7B, *knowing which of your beliefs were formed by a weak model is a safety property.*

---

# Part III — The build plan

Phases are ordered by dependency and by load-bearing weight. Anything marked **[NEW]** does not exist in any reviewed system.

## Phase A — Foundational hardening
*Goal: nothing later can rest on sand.*

- **A1. Threat model documented.** Written attack surface, defence mapping, explicitly stated non-goals. Ship before defences so the defences have a spec.
- **A2. D1 untrusted quarantine.** Trust tiers on origin; quarantine excluded from corroboration, fusion, supersession, and fact presentation.
- **A3. D3 supersession authority + rate limits.** Belief-coup detection.
- **A4. D4 write-time anomaly screening.** Deterministic, Tier 0, no model.
- **A5. D9 model provenance.** Model identity/version/quantisation on every inference node. Cheap now, impossible to backfill later.
- **A6. D8 adversarial self-audit** wired into `doctor`.
- **A7. Consolidation determinism.** **[NEW]** Same store + same seed ⇒ identical `tend()` output. Nobody guarantees this. It makes "why did you forget that?" answerable, and makes consolidation bugs reproducible instead of folkloric.
- **A8. D6 encryption at rest**, optional extra, key management documented.

## Phase B — Retrieval quality
*Goal: stop losing on the axis everyone measures.*

- **B1. Real ONNX embedder validated.** The toy embedder invalidates every Tier 1 number currently produced. Nothing in Phase B means anything until this lands.
- **B2. Decontextualisation.** *(LycheeMem — the largest remaining quality gap.)* "He said it'd arrive Thursday" is useless standalone. Two stages: a deterministic Tier 0 pass resolving pronouns against the episode's entity mentions and relative dates against the observation timestamp; a Tier 2 pass for the rest. **Store the decontextualised form as a derived node, never by rewriting the observation** — the raw utterance is evidence.
- **B3. Group-by retrieval.** *(zvec.)* Top-K *per episode / period / source* rather than globally, so five chunks can't all come from one document. Diversity by construction rather than by reranking.
- **B4. Recollection vs familiarity split.** **[NEW for this field]** *(RF-Mem; Yonelinas dual-process recognition.)* OWL's FOK conflates two distinct signals: *familiarity* (fast, context-free, "I've seen this") and *recollection* (slower, "I can retrieve the details and their context"). Splitting them sharpens triage and adds an honest state — "familiar but I can't place it" — that is different from both KNOW_WHERE and TIP_OF_TONGUE.
- **B5. Retrieval-as-shape.** **[NEW]** Return a structure, not a list: consensus / dissent / stale / gap / provenance mix. All the data already exists; nothing surfaces it. A chunk list invites the model to blend everything into confident prose; a shape forces disagreement and gaps to survive into the answer.
- **B6. "Explain the gap."** **[NEW]** When the answer is DONT_KNOW, state *what would have to exist* for it to know: "I'd need a document naming the depot's fuel supplier." This converts a dead end into a task, and it composes with prospective memory — the gap becomes a standing intention.
- **B7. ANN backend behind `VectorIndex`.** *(zvec — in-process, no server, Windows支持, WAL, hybrid dense+sparse+FTS+filter in one query.)* Optional extra; brute force stays the default. zvec is the obvious candidate and the abstraction for it already exists.
- **B8. Token-budget measurement.** *(iai-pme.)* Measure and cap *tokens* delivered per recall, not just chunks. Publish the number.

## Phase C — Consolidation and structure

- **C1. Reconstructive compression (Tier 2).** The distinctive mechanism: compress only what the system can *prove* it can reconstruct from cue plus neighbours. Flagged, default off, ablated before recommendation. **Never applies to verbatim.**
- **C2. Community detection with stable identity.** *(iai-pme's MOSAIC.)* Cluster the memory graph for coherent replay and neighbourhood spread — but with **community identity stable across splits and merges**, so composite IDs don't churn every cycle and drag provenance chains with them. This is a subtle problem nobody else names.
- **C3. Two-phase sleep on sleep pressure.** NREM (consolidate, low temperature, high-interference nodes) then REM (recombine, high temperature, deliberately distant nodes). Scheduled by accumulated unconsolidated load × learned usage rhythm, not idle CPU.
- **C4. Quarantined hypotheses with mandatory falsifiers.** Generated → tested → promoted / archived-failed / expired. Never exported without explicit promotion.
- **C5. Schema-delta storage.** Compress the rule, store the exception verbatim. Storage cost scales with novelty rather than volume.
- **C6. `failure_pattern` as a first-class type.** *(LycheeMem.)* OWL records *absence* ("I looked, it isn't there") but not *failure* ("we tried this, it didn't work, here's why"). For an analyst toolkit the second is arguably more valuable, and it stops the same rejected option being re-proposed weekly.

## Phase D — People and continuity

- **D1. Action-outcome loop.** *(LycheeMem.)* The `calibration` table exists and nothing writes to it — a dangling thread. Wire retrieval → use → outcome → Brier score per producer and per claim type, or remove the table.
- **D2. Jointly-edited ledger.** **[NEW]** A surface the person can open, read, and correct — with their edit recorded as a first-class provenance event (`origin: user_correction`). Second-order effect matters more than the first: once people can see what the system holds, they correct it proactively. **Making memory legible is a cheaper path to accuracy than making extraction smarter.**
- **D3. Time-travel replay.** **[NEW]** Reconstruct the entire index as of a past moment and ask it questions — not "what did you record" (bitemporal already answers that) but *"what would you have answered, and why was it wrong?"* A black-box recorder for a cognitive system. Only possible because the substrate is append-only.
- **D4. Anticipatory retrieval.** **[NEW]** Memory that raises its hand: a cheap watcher matching the live turn against open loops and past decisions. *The hard part is restraint, not retrieval* — hard interrupt budget, never twice for the same thing, and measured as acted-on vs dismissed. Ship off by default.
- **D5. Portable markdown export.** *(Memanto's OKF.)* `.owlpack` is JSON — inspectable but not readable. A markdown rendering makes a handover reviewable by a human before transfer, which is the format's entire justification.
- **D6. Multi-operator convergence.** Several operators exchanging packs; claims corroborated across *independent* operators promote (using D2's independence math), single-source claims stay marked as such.

## Phase E — Ambient operation

- **E1. Non-blocking capture.** *(iai-pme.)* Capture must never block a session: append to a buffer as pure file IO (~5 ms), embed and index on idle. Currently `observe()` embeds inline.
- **E2. Store readable when the engine is down.** *(iai-pme.)* The store is always directly readable; background machinery is never a gatekeeper on recall.
- **E3. Session-start memory prefix.** A bounded, budgeted slice injected at session start — the thing that makes memory feel ambient rather than a tool you must remember to call.
- **E4. Doctor expansion.** *(iai-pme runs 23 named checks, PASS/WARN/FAIL.)* OWL's is thinner. Named checks, machine-readable output, and self-diagnosis that turns "it doesn't work" issues into user-resolvable ones.

## Phase F — Speculative, high-ceiling
*Explicitly research. Time-boxed. Cut without sentiment if they don't pay.*

- **F1. Structural / hyperdimensional recall.** *(iai-pme's Lilli HD; VSA/HDC.)* Encode role-filler bindings (agent–action–object) so retrieval can match by the *shape* of a memory rather than only its embedding — "who did X to Y" as a structural query, not a similarity one. The most genuinely novel axis found in this entire review. Distinct representations per memory type also prevent episodic detail, semantic gist, and procedural pattern collapsing into one undifferentiated space.
- **F2. Cold-start honesty contract.** **[NEW]** Every memory system's first week is its worst and all of them pretend otherwise. Model store maturity explicitly and report it: "I have 3 days of history on this project; treat gaps as ignorance, not absence." Trust is won by being right about your own limits.
- **F3. Nightly LLM step via existing subscription.** *(iai-pme.)* One bounded model call per night through the user's existing subscription rather than a separate API key, quota-capped. Elegant solution to Tier 2 without credentials.
- **F4. Predictive-processing unification.** One objective — minimise surprise — behind encoding priority, attention allocation, consolidation scheduling, and dreaming, replacing five separately-tuned subsystems. Supported by the 2025 locus coeruleus / prediction-error-driven memory updating work.

---

# Part IV — Evaluation

**This is not the last section because it is least important. It is the section that decides whether any of the above ships.**

## IV.1 External benchmarks (run these; be honest about losing)

| Benchmark | What it measures | Expectation |
|---|---|---|
| **LongMemEval-S** | Long-term retrieval, temporal/multi-hop/update | Competitive; OWL retrieves 4–7 chunks by design |
| **LoCoMo** (+ LoCoMo-Refined, Locomo-Plus) | Conversational long-term memory | Competitive |
| **HaluMem** | **Hallucination in memory systems** | **Target: lead.** This is OWL's thesis, externally scored |
| **MPBench** | Memory poisoning resistance | **Target: lead** after Phase A |
| **BEAM** | Beyond a million tokens | Informational |
| **MemoryAgentBench** | Incremental multi-turn | Informational |
| **NoLiMa** | Beyond literal matching | Tests the paraphrase path specifically |

## IV.2 Benchmarks to adopt from iai-pme

- **Rescue@k** — after a fact is superseded, does the *current* one still rank top-k? They score 1.000; match it.
- **Inverse-Rescue@k** — can you still retrieve the *superseded wording*? They regressed to 0.71. **OWL should score ~1.0 by architecture.** Publish both numbers together; the pair is the argument for append-only substrate.
- **Personal-fact drift** — retention across N facts / M sessions / K intervening sessions.
- **Sleep ablation** — does consolidation preserve recall? If not, cut consolidation.

## IV.3 Benchmarks OWL should define
*This is the contribution that stands regardless of whether OWL wins.*

- **Source attribution accuracy** — "how do you know that?" on every claim; score the origin class and the specific source. Nothing in the field scores this.
- **Calibration (Brier / ECE)** — stated confidence vs outcome. Nothing scores this.
- **Interference resistance under `nuc`** — MemoryLLM's methodology, extended with a *confusable* condition to separate interference from mere volume.
- **Independence-weighted corroboration** — resistance to source flooding.
- **Staleness precision/recall** — of claims flagged stale, how many were? Of unflagged, how many were wrong?
- **Transactive accuracy** — does the modelled user-forgetting curve predict actual human recall failures? Requires human study; the most scientifically interesting.

**Publish the suite. Run other systems on it.** A memory scoreboard measuring epistemics rather than recall does not exist, and building it is a real contribution.

## IV.4 Standing rules

1. Correctness suite: **under one second, no GPU, no network.** Non-negotiable.
2. Every claim in the README maps to a harness in `bench/`.
3. Regressions are published, not hidden (iai-pme's practice; it is the reason to trust their numbers).
4. **Every benchmark must be checked for whether it measures the system or the harness.** The `nuc` benchmark initially "disproved" OWL's central thesis; it was measuring the toy embedder. Assume this failure mode is present until ruled out.

---

# Part V — Risks

**Scope creep is the primary risk.** This plan is large. The test for any addition: *does it help answer "how do you know that?"* Anything that fails belongs in the host.

**The zero-dependency story is load-bearing and fragile.** Memanto needs Docker, LycheeMem needs a server, MIRIX needs Docker + Postgres, iai-pme needs Rust and is macOS-only. `pip install owl-engine` with no daemon is the most defensible thing OWL has. Every optional extra must stay optional, and Tier 0 must remain genuinely complete.

**Tier 2 may be negative value.** MiniRAG's benchmark shows graph-RAG pipelines *inverting* below ~7B — worse than naive RAG, catastrophically so at 1.5B. Every model-dependent feature ships flagged, default off, with an ablation.

**Benchmarks measuring the harness.** See IV.4 rule 4. This already happened once.

**Being second to a claim.** Memanto published the provenance thesis first. The response is not to argue precedence but to hold the distinction that matters — enforced invariant vs annotated field — and to prove it on benchmarks they don't run.

---

# Part VI — Recommended sequence

If only three things get built:

1. **Phase A (threat model + quarantine + supersession authority + model provenance).** The category the whole field is weak on, where OWL's existing architecture gives an unfair head start, and where being early is worth more than being polished.

2. **B1 + B2 (real embedder + decontextualisation).** Without these, retrieval quality caps out below the field and every Tier 1 number is untrustworthy.

3. **IV.3 (the scoreboard).** Define and publish the epistemic benchmark suite, with Rescue@k and its inverse as the flagship pair. It is the cheapest credibility available and it reframes the comparison onto ground OWL chose.

Everything else is upside.

---

## Appendix — Attribution

| Source | Contribution to this plan |
|---|---|
| iai-pme | Rescue@k, inverse-Rescue, sleep ablation, non-blocking capture, store-readable-when-down, encryption at rest, stable community identity, doctor discipline, published-regression culture, HD structural recall |
| LycheeMemory | Record fusion (built), decontextualisation, `failure_pattern`, action-outcome loop |
| Memanto | The provenance thesis as product framing; OKF portable export; the competitive reality check |
| MIRIX | Knowledge Vault → verbatim protection (built) |
| MiniRAG | Heterogeneous graph (built); the small-model inversion finding |
| MemoryLLM/M+ | `nuc` interference methodology (built) |
| zvec | In-process ANN backend; group-by retrieval |
| Zep/Graphiti | Bi-temporality (built) |
| HippoRAG | PPR graph spread (built) |
| EM-LLM | Surprise-boundary segmentation (built) |
| Awesome-Agent-Memory | Memory security category; HaluMem; RF-Mem recollection/familiarity |
| **Original to OWL** | Monotonicity as enforced invariant · six FOK states · transactive memory & symmetric divergence · information-flow partitions · epistemic-demotion handover · learned claim half-life · source independence graph · retrieval receipts · adversarial self-audit · model provenance · consolidation determinism · explain-the-gap · cold-start honesty · time-travel replay · jointly-edited ledger |
