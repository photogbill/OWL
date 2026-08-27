# O.W.L. — Master Build Plan

**The single authoritative plan.** Supersedes `OWL_Master_Build_Plan_v4.md` and `OWL_Master_Build_Plan_v5_Addendum.md` (retained for history).

**Reviewed for this plan:** MIRIX · MemoryLLM/M+ · LycheeMemory · Memanto · MiniRAG · zvec · iai-pme · Awesome-Agent-Memory (survey) · Mem0 · Zep/Graphiti · A-MEM · HippoRAG · EM-LLM · GraphRAG · Letta/MemGPT · Hexis

**Conventions.** Every item has an ID, dependencies, an effort band (**S** ≈ days, **M** ≈ 1–2 weeks, **L** ≈ 3+ weeks), and an acceptance criterion. **[NEW]** marks work absent from every reviewed system. Plan only — no code in this document.

---

# Part 0 — Strategic position

## 0.1 What "outperform" has to mean

Three facts set the target.

**You cannot win on retrieval accuracy alone.** Memanto claims 89.8% LongMemEval behind a dedicated information-theoretic engine and a commercial tier. LycheeMemory has an ACL 2026 paper and a trained reranker. iai-pme ties the leader at R@5 0.966 with a hand-written Rust core. These teams optimise one metric full-time.

**The field has converged on the same feature list.** Provenance, decay, conflict detection, typed memory, hierarchical consolidation, temporal queries. Memanto's "Six Gaps" table is substantially OWL's pitch, published first. Any plan premised on "nobody does provenance" is already wrong.

**But the entire field competes on one question: *can you find it?*** Almost nobody competes on: should you believe it, how do you know, is it still true, who else knows it, can it be moved safely, can it be attacked — and, the gap this plan is built around, **what is it holding up?**

> **Target: be the memory system you would choose when being wrong has a cost.**

Three principles run throughout:

1. **Every claim ships with a harness that proves it.** (iai-pme's discipline; it is why their numbers are credible.)
2. **Enforcement over annotation.** Invariants checked by the machine, not fields filled in by a writer.
3. **Where the field has no benchmark, build the benchmark** and run other systems on it. That contribution stands whether or not OWL wins.

## 0.2 The central insight of this plan

Everything OWL has built so far points **backwards**: `why()` answers *how do I know this?*

Nothing points **forwards**: *what do I believe because of this, and what did I do about it?*

> **The field has built memories that answer questions. Nobody has built a memory that knows what it is holding up.**

That direction is empty, it is cheap to build on an append-only provenance graph, and it is precisely the direction that matters when a decision rests on the output. **Phase D is the largest single addition in this plan and it exists to fill it.**

This is a deliberate widening of scope. Phases D and E together make OWL a **belief-maintenance system**, not only a memory. That is a bigger claim and a bigger surface. It is stated here explicitly so it is a decision rather than a drift.

## 0.3 Honest scorecard

| Axis | Current leader | OWL today | Target |
|---|---|---|---|
| Raw retrieval (LoCoMo / LongMemEval) | Memanto, iai-pme, LycheeMem | unmeasured | competitive, not leading |
| Hallucination (HaluMem) | unclear | unmeasured | **lead** |
| Post-contradiction (Rescue@k) | iai-pme 1.000 | unmeasured | match |
| **Inverse-Rescue** (superseded wording) | iai-pme 0.71 *(regressed)* | ~1.0 by architecture | **lead** |
| Source attribution | nobody measures | by construction | **define the metric** |
| Calibration | nobody measures | infrastructure only | **define the metric** |
| Poisoning resistance | Agent Memory Guard | none | **lead** |
| Consequence tracking | **nobody** | none | **own the category** |
| Install friction | OWL | zero deps | **hold** |
| Offline / air-gap | OWL, iai-pme | complete | **hold** |

**The inverse-Rescue row is the model for how OWL should win.** iai-pme honestly reports that retrieving the *superseded wording* of an updated fact regressed 0.90 → 0.71. OWL cannot lose that, because the substrate is append-only and the old row is never rewritten. Wins by architecture are the only durable ones.

---

# Part I — What exists today

70 tests, ~2 s, no GPU, no network, zero dependencies.

**Substrate** — three-layer split (immutable observation / mutable index / rewritten derivation); append-only enforced by SQL trigger; single-writer queue; bitemporal validity; redaction with propagation.

**Epistemics** — monotonicity invariant; Admiralty two-axis source grading; claim classes with learned half-lives; verbatim protection.

**Retrieval** — six-state FOK gate; FSRS salience; two embedding spaces (structural separation on write, completion on read); max-fusion of lexical and semantic; entity-graph bridging with two-hop traversal; answer-type prediction; associative and successor spread.

**Maintenance** — interference sweep; union-find record fusion with composites; retrieval-induced forgetting; tier transitions.

**People** — transactive memory (exposure log, user forgetting curve, at-risk ranking); symmetric belief-divergence detection.

**Boundaries** — information-flow partitions (denied by default, directional, sealed, graded permeability); affect-gated retrieval.

**Portability** — `.owlpack` handover with epistemic demotion; corroboration on independent concurrence; checksum verification.

**Other** — prospective memory; negative memory; suppression-not-deletion; event segmentation at surprise boundaries.

---

# Part II — Threat model

## II.1 Why this ranks above features

Every capability OWL has assumes the store's contents are honest mistakes at worst. That fails the moment a document of unknown origin is ingested — the normal case for ATK, and inevitable for any agent with file or web access.

> **A memory system is a persistence layer for beliefs. Anything that writes to it is writing to the agent's mind, permanently.**

Prompt injection is transient. Memory poisoning is not: it survives every restart, propagates into every derived summary, and is retrieved as context forever. OWL is unusually well placed — provenance, immutability, Admiralty grading and flow partitions are already defensive primitives — but they were built for honesty, not adversaries.

## II.2 Attack surface

| Attack | Mechanism | Exposure today |
|---|---|---|
| Direct injection | Document contains instructions framed as facts | **Full** |
| Supersession hijack | Hostile content claims to supersede established truth | **Full** |
| Source flooding | 200 documents assert one falsehood to manufacture corroboration | **Full** |
| Fusion poisoning | Craft near-duplicates so union-find merges a true cluster under a false representative | **Partial** (verbatim protected) |
| Provenance laundering | Chain derivations until a hostile claim appears trusted | **Low** (monotonicity blocks escalation) |
| Handover poisoning | Malicious `.owlpack` | **Partial** (checksum + demotion) |
| Partition exfiltration | Trick a sealed partition into leaking via a summary | **Low** (structural) |
| Denial by flooding | Bury real memories until FOK stops firing | **Unmeasured** |
| Embedding collision | Craft text landing adjacent to a target in vector space | **Full** |

---

# Part III — Phase A: Foundations & defence

*Nothing later can rest on sand. Ship this first.*

**A1 — Threat model documented.** `S` · no deps
Written attack surface, defence mapping, explicit non-goals. Ships *before* the defences so they have a spec.
*Accept:* every row in II.2 maps to a defence ID or a documented accepted risk.

**A2 — Untrusted intake quarantine.** `M` · A1
Trust tier on origin. Quarantined content is retrievable but never corroborates, fuses, supersedes, or presents as fact until promoted by independent corroboration or explicit user action. **Highest-value single defence; reuses the epistemic lattice already built.**
*Accept:* a poisoned document cannot alter any pre-existing belief, and its content never returns `presentable_as_fact`.

**A3 — Supersession authority + rate limits.** `S` · A2
Superseding requires reliability ≥ the claim being replaced; a grade-F source can only register a *conflict*. Rate-limit supersessions per source per window; flag bursts as a "belief coup".
*Accept:* scripted coup attempt produces conflicts and a flag, zero silent overwrites.

**A4 — Write-time anomaly screening.** `M` · A1
Deterministic, Tier 0: imperative mood in supposedly factual content, instruction patterns ("ignore", "always", "from now on"), entropy outliers, unusual claim density per token, mass contradiction of established beliefs.
*Accept:* ≥80% detection on an MPBench-style injection corpus at <5% false positive.

**A5 — Source independence graph. [NEW]** `M` · A2
Corroboration counts *independent origins*, not documents. Cluster sources by shared ancestry, ingest batch, domain and arrival time; weight corroboration by independent clusters. **Defeats flooding at the root and improves ordinary corroboration quality.**
*Accept:* 200 documents from one upstream origin raise credibility no more than one document does.

**A6 — Model provenance. [NEW]** `S` · no deps
Record model identity, version and quantisation on every inference node. **Cheap now, impossible to backfill.** Given MiniRAG's finding that pipelines invert below ~7B, knowing which beliefs a weak model formed is a safety property.
*Accept:* `blast_radius(model="qwen-7b-q4")` returns every conclusion resting on it.

**A7 — Second-order uncertainty. [NEW]** `L` · touches everything
Replace scalar confidence with **belief / disbelief / ignorance** (subjective logic; equivalently a Beta posterior or Dempster–Shafer mass). Evidence moves mass *out of* ignorance.
```
no evidence          b=0.00  d=0.00  u=1.00
strongly contested   b=0.45  d=0.45  u=0.10
well established     b=0.92  d=0.03  u=0.05
```
Today "no evidence" and "strong evidence on both sides" are both 0.5 — completely different states, and the difference is often the whole answer: the first says *go and look*, the second says *stop looking and decide under uncertainty*. Monotonicity generalises cleanly (a child's belief mass cannot exceed its parents'; its ignorance mass cannot be lower). Expose scalar confidence as a derived property so nothing downstream breaks.
**Land this early or accept the approximation** — the cheap 80% is carrying `evidence_count` alongside confidence. The full version is more correct and makes the calibration metric (Part VII) measurable in a way point estimates cannot.
*Accept:* a store with zero evidence and one with balanced evidence are distinguishable in every API surface.

**A8 — Dimensional integrity.** `S` · no deps
Quantities as structured values with units, not substrings. "400L" ≠ "400 gallons"; a summariser that drops a unit has produced a *dangerous* sentence, not a shorter one. Pairs with `verbatim`. For fuel, dosage and distance in the field this is a safety feature.
*Accept:* unit-mismatched quantities never fuse; unit loss during any derivation is rejected.

**A9 — Adversarial self-audit. [NEW]** `S` · A6
Idle-time attack on OWL's own invariants: any claim presentable as fact but untraceable to a primary source; any node whose confidence exceeds its parents; orphaned derivations; partition edges admitting more than declared. CI proves invariants at commit; this proves they hold **in the live store, over time**. Surfaced in `doctor`.
*Accept:* deliberately corrupted store is detected within one `tend()`.

**A10 — Consolidation determinism. [NEW]** `M` · no deps
Same store + same seed ⇒ identical `tend()` output. Nobody guarantees this. It makes "why did you forget that?" answerable and consolidation bugs reproducible rather than folkloric.
*Accept:* 100 consecutive runs byte-identical.

**A11 — Encryption at rest.** `M` · optional extra
AES-256-GCM, key at 0600, documented backup and loss consequences. Table stakes for field casework.
*Accept:* store unreadable without the key; `doctor` verifies key permissions.

**A12 — Fusion integrity.** `S` · A2, A5
Composite representative must be the *best-attested* member, not merely the most confident. Never fuse across trust tiers or across differing reliability grades.
*Accept:* crafted near-duplicate cannot become a cluster representative.

---

# Part IV — Phase B: Retrieval quality

*Stop losing on the axis everyone measures. **B1 gates the entire phase** — every current Tier 1 number is untrustworthy until it lands.*

**B1 — Real ONNX embedder validated.** `M` · no deps
The toy embedder invalidates all Tier 1 measurements. Nothing else in Phase B means anything first.
*Accept:* paraphrase recall measured on a held-out set; interference benchmark re-run and interpreted.

**B2 — Decontextualisation.** `L` · B1
*(LycheeMem — the largest remaining quality gap.)* "He said it'd arrive Thursday" is useless standalone. Two stages: deterministic Tier 0 pass resolving pronouns against episode entity mentions and relative dates against the observation timestamp; Tier 2 pass for the remainder. **Store the decontextualised form as a derived node — never rewrite the observation.** The raw utterance is evidence.
*Accept:* ≥90% of conversational memories independently interpretable; zero observation mutations.

**B3 — Group-by retrieval.** `S` · B1
*(zvec.)* Top-K *per episode / period / source* rather than globally, so five chunks cannot all come from one document. Diversity by construction rather than by reranking.
*Accept:* no recall returns >2 chunks from one source unless the budget demands it.

**B4 — Recollection vs familiarity split.** `M` · B1
*(RF-Mem; Yonelinas dual-process recognition.)* The FOK gate conflates *familiarity* (fast, context-free, "I've seen this") with *recollection* ("I can retrieve the details and their context"). Splitting them sharpens triage and adds an honest state — "familiar but I can't place it" — distinct from both `KNOW_WHERE` and `TIP_OF_TONGUE`.
*Accept:* triage accuracy improves on a labelled query set; the new state fires only where both signals genuinely diverge.

**B5 — Retrieval-as-shape. [NEW]** `M` · B1, D-phase
Return a structure, not a list: consensus / dissent / stale / gap / provenance mix / criticality. All the data exists; nothing surfaces it. A chunk list invites the model to blend everything into confident prose; **a shape forces disagreement and gaps to survive into the answer.**
*Accept:* every recall renders as a shape; downstream prose preserves flagged dissent.

**B6 — Counter-evidence retrieval. [NEW]** `M` · B1, B5
Confirmation bias is not merely a human failing here — **it is baked into the objective function of similarity search.** Ask "why is the pump failing?" and you get pump-failure evidence, never last week's note saying the pump was fine. Deliberately retrieve the strongest *disconfirming* set alongside the supporting one: contradictions of the query's presupposition, semantically adjacent but oppositely valenced material, memories superseded by the belief the query assumes.
**Gated on B1** — at Tier 0 this surfaces noise and discredits itself.
*Accept:* on a planted-contradiction set, the contradiction appears in the shape ≥90% of the time.

**B7 — Explain the gap. [NEW]** `S` · no deps
On `DONT_KNOW`, state *what would have to exist*: "I'd need a document naming the depot's fuel supplier." Converts a dead end into a task and composes with prospective memory — the gap becomes a standing intention.
*Accept:* every `DONT_KNOW` carries an actionable gap statement.

**B8 — Query critique.** `S` · entity graph
When the store holds a strong answer to an *adjacent* question, say so: "Nothing on tanker arrival, but I have the depot dispatch schedule and a note that tankers dispatch on request."
*Low priority — will feel clever and be used rarely. Build after B7, cut if it doesn't earn its keep.*

**B9 — ANN backend behind `VectorIndex`.** `M` · B1
*(zvec — in-process, no server, Windows support, WAL, hybrid dense + sparse + FTS + filter in one query.)* Optional extra; brute force stays the default. The abstraction already exists.
*Accept:* identical results to brute force at ≥10× speed on 100k nodes.

**B10 — Token-budget measurement.** `S` · no deps
*(iai-pme.)* Measure and cap *tokens* delivered per recall, not just chunks. Publish the number.
*Accept:* p50/p95 token cost reported by `doctor`.

**B11 — Cross-lingual claim identity.** `M` · B1 · *host-specific*
ATK translates. A memory recorded in Somali must be findable by an English query, and the *same claim* in two languages must corroborate rather than duplicate. Language tag on write; identity resolved in a shared multilingual space; original text always preserved as evidence.
**Defer until ATK integration is real** — high value, but only for that host.
*Accept:* cross-language corroboration recognised; original never replaced by a translation.

---

# Part V — Phase C: Consolidation & structure

**C1 — Reconstructive compression (Tier 2).** `L` · B1, Reasoner
Compress only what the system can *prove* it can reconstruct from cue plus neighbours. The distinctive mechanism. Flagged, default off, ablated before recommendation. **Never applies to verbatim.**
*Accept:* sleep-ablation shows no recall loss; reconstruction failures keep content verbatim.

**C2 — Community detection with stable identity.** `L` · B1
*(iai-pme's MOSAIC.)* Cluster for coherent replay and neighbourhood spread, with **community identity stable across splits and merges** so composite IDs don't churn every cycle and drag provenance chains with them. A subtle problem nobody else names.
*Accept:* identity stable across 50 consolidation cycles on a mutating store.

**C3 — Two-phase sleep on sleep pressure.** `M` · C2
NREM (consolidate; low temperature; high-interference nodes) then REM (recombine; high temperature; deliberately distant nodes). Scheduled by accumulated unconsolidated load × learned usage rhythm — not idle CPU.
*Accept:* ablation shows measurable next-day benefit, or the phase is cut.

**C4 — Quarantined hypotheses with mandatory falsifiers.** `S` · C3
Generated → tested → promoted / archived-failed / expired. Never exported without explicit promotion.
*Accept:* no hypothesis reaches a user-facing surface unmarked.

**C5 — Schema-delta storage.** `M` · C1
Compress the rule, store the exception verbatim. Storage cost scales with novelty rather than volume.
*Accept:* storage growth sublinear in observation count on repetitive corpora.

**C6 — `failure_pattern` as a first-class type.** `S` · no deps
*(LycheeMem.)* OWL records *absence* ("I looked, it isn't there") but not *failure* ("we tried this, it didn't work, here's why"). For an analyst toolkit the second is arguably more valuable, and it stops the same rejected option being re-proposed weekly.
*Accept:* a previously-failed approach surfaces when a similar approach is proposed.

**C7 — Cost-of-acquisition as a retention signal. [NEW]** `S` · no deps
Everyone models memory as storage. It is an **investment**. Some facts cost a three-day trip or a canvass of six vendors; others cost a glance at a filename. The right question for forgetting is not "how often was this used?" but **"what would it cost me to get this back?"** `acquisition_cost` supplied by the host or inferred from elapsed time, tool calls and sources consulted; feeds salience as a protection factor. Negative memory already encodes this instinct — generalise it.
*Accept:* expensive memories survive decay pressure that removes cheap ones of equal access frequency.

---

# Part VI — Phase D: The forward direction ★

*The largest addition in this plan, and the emptiest ground in the field.*

**D1 — Decision–consequence graph. [NEW]** `M` · no deps · **highest value/effort ratio in the plan**

An analyst reads "Route Alpha is open", routes a convoy, moves on. Three days later a sitrep supersedes it. **Every memory system updates the fact and stops. Nothing tells anyone the convoy decision is standing on a false premise.**

Decisions become first-class nodes with inbound edges to the memories they rested on. Supersession becomes an event with a blast radius: walk forward through `decided_on` edges and surface every downstream decision, ranked by how load-bearing the changed memory was and whether the decision is still reversible.

This is the difference between a memory that answers questions and one that **prevents harm**. It also completes the arc of everything else: bitemporality says what changed, provenance says why you believed it, divergence says the operator is out of date — and this says *what to go and fix.*

**Second-order effect:** a decision journal for free, which is the most effective known intervention against hindsight bias. Combined with D5, it makes honest post-incident review possible.
*Accept:* superseding a memory surfaces every affected decision, correctly ranked by reversibility.

**D2 — Retroactive trust propagation. [NEW]** `M` · D1, A5

`why()` walks backward. Nothing walks forward. Discover a document was forged, a source was lying, or an ingest batch mis-parsed — and every system in the field leaves the contamination in place: summaries stay, composites stay, conclusions stay, now unsourced but still confident.

`blast_radius(node)` = transitive forward closure over derivation edges, mention links, composite membership and decision edges. Then a **revaluation pass**: re-run the monotonicity clamp with the source's new reliability and cascade. Nodes falling below threshold demote to `hypothesized` or quarantine — **never silently deleted**, because having once believed it is itself evidence.

**This is what makes provenance pay.** Everyone else's provenance is decorative because nothing acts on it. The killer query:

> *"I just learned that survey PDF was out of date. What did I conclude from it, what did I tell the team, and which decisions rested on it?"*

*Accept:* discrediting a source correctly demotes its entire forward closure in one pass, with a reversible audit record.

**D3 — Load-bearing criticality. [NEW]** `S` · D1

PageRank (or reverse-reachability count) over the derivation graph, weighted by dependents' epistemic rank and decision edges. Unlocks:
- **Verification triage** — "these four beliefs carry 60% of your conclusions; two are single-sourced grade-C; verify those first."
- **Fragility detection** — a conclusion supported by many nodes that all trace to one origin *looks* robust and is not. Criticality × A5 exposes exactly that.
- **Better forgetting** — criticality beats access count. A memory nothing depends on is safe to compress; a load-bearing one never is, however rarely touched.
*Accept:* criticality ranking correlates with measured blast radius; verification triage is reproducible.

**D4 — Retrieval receipts. [NEW]** `S` · no deps
Every recall emits an immutable record of what was returned, why, which state fired, and what was considered and rejected. Downstream errors become traceable to a retrieval decision — currently that evaporates. Also the substrate for calibration measurement.
*Accept:* any past answer reconstructible from its receipt.

**D5 — Time-travel replay. [NEW]** `M` · A10, D4
Reconstruct the entire index as of a past moment and query it. Not "what did you record" (bitemporal answers that) but **"what would you have answered, and why was it wrong?"** A black-box recorder for a cognitive system — possible only because the substrate is append-only.
*Accept:* replayed answers match historical receipts exactly.

---

# Part VII — Phase E: Trust as a living system

*D2, E1 and E2 form one self-maintaining loop. Each piece is individually cheap.*

**E1 — Attributed belief. [NEW]** `M` · A5

OWL stores *"Ahmed said the parts arrive Thursday"* as a string. The Admiralty grade attaches to the *document*, not to **Ahmed**. So you cannot ask "what does Ahmed believe?", "who else claims this?", or "Ahmed has been wrong four times — what else did he tell me?"

This is the *de dicto / de re* distinction. Intelligence tradecraft has treated it as basic for a century; no LLM memory system implements it.

Model a claim as `(claimant, proposition, assertion_time)`, distinct from the proposition. Multiple claimants asserting one proposition *is* corroboration, and composes with A5. Claimants accumulate a reliability record from outcomes. Correct handling finally exists for the hard case: a *reliable* source reporting something *implausible* — the two Admiralty axes get somewhere proper to live.
*Accept:* per-claimant reliability learned from outcomes; "who told me this and what else did they say" is one hop.

**E2 — Commitment lifecycle. [NEW]** `S` · E1

"I'll bring fuel Thursday" is not a fact. It is a **commitment**, with a lifecycle: made → due → kept or broken.

Speech-act type on write (assertion / commitment / request / hypothesis). Commitments carry a due date and resolution state; on the due date they surface automatically (prospective memory, already built); on resolution the outcome feeds E1's reliability record.

**This closes a loop nobody closes: broken promises automatically degrade source reliability, which via D2 automatically revalues everything that source ever claimed.** Memanto has `commitment` as a *type*; nobody tracks the *lifecycle*, and the lifecycle is where the value is. For field work, directly useful: who actually delivers.
*Accept:* a broken commitment measurably lowers the claimant's reliability and triggers revaluation.

**E3 — Action-outcome loop.** `M` · D4
*(LycheeMem.)* The `calibration` table exists and nothing writes to it — a dangling thread. Wire retrieval → use → outcome → Brier score per producer and per claim type, **or remove the table.**
*Accept:* calibration curve plottable from real usage.

**E4 — Jointly-edited ledger. [NEW]** `M` · D4
A surface the person can open, read and correct, with their edit recorded as first-class provenance (`origin: user_correction`). The second-order effect matters more than the first: once people can see what the system holds, they correct it proactively. **Making memory legible is a cheaper path to accuracy than making extraction smarter.** ATK already has the graph canvas and profile cards to host it.
*Accept:* user corrections appear in `why()` chains as maximum-depth exposure events.

---

# Part VIII — Phase F: Ambient operation

**F1 — Non-blocking capture.** `M` · no deps
*(iai-pme.)* Capture must never block a session: append to a buffer as pure file IO (~5 ms), embed and index on idle. `observe()` currently embeds inline.
*Accept:* p99 capture latency <10 ms with an embedder attached.

**F2 — Store readable when the engine is down.** `S` · F1
The store is always directly readable; background machinery is never a gatekeeper on recall.
*Accept:* recall works with all background workers killed.

**F3 — Session-start memory prefix.** `M` · B10, D3
A bounded, budgeted slice injected at session start — what makes memory ambient rather than a tool you must remember to call. Prioritised by criticality and open loops, not recency alone.
*Accept:* under budget, and ablation shows measurable session-quality benefit.

**F4 — Anticipatory retrieval. [NEW]** `M` · D1
Memory that raises its hand: a cheap watcher matching the live turn against open loops and past decisions. **The hard part is restraint, not retrieval** — hard interrupt budget, never twice for the same thing, measured as acted-on vs dismissed. Ship off by default.
*Accept:* acted-on:dismissed ratio strongly positive, or it stays off.

**F5 — Doctor expansion.** `S` · A9
*(iai-pme runs 23 named checks, PASS/WARN/FAIL.)* Named checks, machine-readable output, self-diagnosis that turns "it doesn't work" into a user-resolvable issue.
*Accept:* every failure mode in Part II and Phase A has a named check.

**F6 — Portable markdown export.** `S` · no deps
*(Memanto's OKF.)* `.owlpack` is JSON — inspectable but not readable. A markdown rendering makes a handover reviewable by a human before transfer, which is the format's entire justification.
*Accept:* a pack is reviewable without tooling.

**F7 — Multi-operator convergence.** `M` · A5, E1
Several operators exchanging packs; claims corroborated across *independent* operators promote; single-source claims stay marked. Uses A5's independence math.
*Accept:* three-operator scenario produces correct independence-weighted credibility.

---

# Part IX — Phase G: Performance

**G1 — Bloom/cuckoo filter over the lexical index.** `S`
Make `DONT_KNOW` genuinely O(1) rather than O(query terms). **The fast path deserves to be the fastest path, and it is the one that fires most often.**
*Accept:* `DONT_KNOW` p99 <1 ms at 1M nodes.

**G2 — Memory-mapped vectors.** `S`
Stop deserialising blobs per query. Brute force stays viable far longer than expected.
*Accept:* 5× search throughput at 100k vectors.

**G3 — Recall caching with write-driven invalidation.** `M`
Keyed on (query, partition, clock bucket). Repeat queries within a session are extremely common.
*Accept:* cache hit returns identical results; any write invalidates correctly.

**G4 — Incremental index maintenance.** `M`
`tend()` currently rescans. Dirty-flag what changed.
*Accept:* `tend()` cost scales with changes, not store size.

**G5 — Partition-sharded storage.** `M`
A large work partition never slows a small private one.
*Accept:* private-partition latency independent of work-partition size.

---

# Part X — Phase H: Speculative

*Explicitly research. Time-boxed. Cut without sentiment.*

**H1 — Structural / hyperdimensional recall.** `L`
*(iai-pme's Lilli HD; VSA/HDC.)* Encode role-filler bindings (agent–action–object) so retrieval can match by the **shape** of a memory rather than only its embedding — "who did X to Y" as a structural query, not a similarity one. **The most genuinely novel axis found in the entire review.** Distinct representations per memory type also prevent episodic detail, semantic gist and procedural pattern collapsing into one space.

**H2 — Cold-start honesty contract. [NEW]** `S`
Every memory system's first week is its worst and all of them pretend otherwise. Model store maturity explicitly and report it: *"I have 3 days of history on this project; treat gaps as ignorance, not absence."* **Trust is won by being right about your own limits.** Pairs naturally with A7's ignorance mass.

**H3 — Nightly LLM step via existing subscription.** `M`
*(iai-pme.)* One bounded model call per night through the user's existing subscription rather than a separate API key, quota-capped. Elegant route to Tier 2 without credentials.

**H4 — Predictive-processing unification.** `L`
One objective — minimise surprise — behind encoding priority, attention allocation, consolidation scheduling and dreaming, replacing five separately-tuned subsystems. Supported by the 2025 locus-coeruleus / prediction-error memory-updating work.

---

# Part XI — Evaluation

**Not last because least important — the section that decides whether anything above ships.**

## XI.1 External benchmarks (run these; be honest about losing)

| Benchmark | Measures | Expectation |
|---|---|---|
| **HaluMem** | Hallucination in memory systems | **Target: lead** — OWL's thesis, externally scored |
| **MPBench** | Memory poisoning resistance | **Target: lead** after Phase A |
| LongMemEval-S | Long-term retrieval; temporal / multi-hop / update | Competitive |
| LoCoMo (+ Refined, Plus) | Conversational long-term memory | Competitive |
| NoLiMa | Beyond literal matching | Tests the paraphrase path |
| BEAM | Beyond a million tokens | Informational |
| MemoryAgentBench | Incremental multi-turn | Informational |

## XI.2 Adopted from iai-pme

- **Rescue@k** — after supersession, does the *current* fact still rank? They score 1.000; match it.
- **Inverse-Rescue@k** — can you still retrieve the *superseded wording*? They regressed to 0.71. **OWL should score ~1.0 by architecture.** Publish both together; the pair is the argument for append-only substrate.
- **Personal-fact drift** — retention across N facts / M sessions / K intervening sessions.
- **Sleep ablation** — does consolidation preserve recall? If not, cut consolidation.

## XI.3 Benchmarks OWL defines
*The contribution that stands regardless of whether OWL wins.*

- **Source attribution accuracy** — score origin class and specific source on every claim.
- **Calibration (Brier / ECE)** — stated confidence vs outcome. With A7, score belief and *ignorance* separately.
- **Interference resistance under `nuc`** — MemoryLLM's methodology plus a *confusable* condition separating interference from volume.
- **Independence-weighted corroboration** — resistance to source flooding.
- **Staleness precision/recall** — of claims flagged stale, how many were? Of unflagged, how many were wrong?
- **Consequence recall [NEW]** — after a memory is superseded, what fraction of affected decisions were surfaced? **A category with no incumbent.**
- **Blast-radius completeness [NEW]** — on discrediting a source, what fraction of contaminated conclusions were correctly demoted?
- **Transactive accuracy** — does the modelled user-forgetting curve predict real human recall failure? Requires a human study; the most scientifically interesting.

**Publish the suite. Run other systems on it.** A memory scoreboard measuring epistemics rather than recall does not exist.

## XI.4 Standing rules

1. Correctness suite: **under one second, no GPU, no network.** Non-negotiable.
2. Every README claim maps to a harness in `bench/`.
3. Regressions are published, not hidden.
4. **Every benchmark is checked for whether it measures the system or the harness.** The `nuc` benchmark initially "disproved" OWL's central thesis; it was measuring the toy embedder. Assume this failure mode until ruled out.

---

# Part XII — Risks

**Scope creep is the primary risk.** This plan is large and the scope has deliberately widened. Original test: *does this help answer "how do you know that?"* **Revised test, given Phase D:** *does this help answer "how do you know that?" or "what is resting on it?"* Anything failing both belongs in the host.

**The zero-dependency story is load-bearing and fragile.** Memanto needs Docker; LycheeMem needs a server; MIRIX needs Docker + Postgres; iai-pme needs Rust and is macOS-only. `pip install owl-engine` with no daemon is the most defensible thing OWL has. Every optional extra stays optional; Tier 0 stays genuinely complete.

**Tier 2 may be negative value.** MiniRAG shows graph-RAG pipelines *inverting* below ~7B — worse than naive RAG, catastrophically at 1.5B. Every model-dependent feature ships flagged, default off, with an ablation.

**A7 is the schedule risk.** Second-order uncertainty touches every confidence computation. Land it early or take the `evidence_count` approximation; retrofitting mid-plan will be painful.

**Belief maintenance is a bigger claim than memory.** Phases D and E change what OWL *is*. Stated in 0.2 as a decision. Revisit if it stops paying.

**Benchmarks measuring the harness.** See XI.4 rule 4. Already happened once.

---

# Part XIII — Recommended sequence

**Wave 1 — the unfair advantage.** A1, A2, A3, A6, A9 · **D1, D3, D4**
Phase A's core defences plus the decision graph. Rationale: the security category is where the field is weakest and OWL's architecture gives an unearned head start, and **D1 is the highest value-per-effort item in the entire plan** — it changes what the product is for the cost of one table and one traversal.

**Wave 2 — make provenance pay.** A5 · **D2** · E1, E2 · A8, B7, C6, C7
The trust loop closes: independence graph → attributed belief → commitments → retroactive revaluation. Each piece cheap; together they make trust self-maintaining. Plus the cheap deterministic wins.

**Wave 3 — retrieval credibility.** **B1** first, then B2, B3, B4, B5, B10 · G1, G2
Nothing here is measurable before B1. B2 is the largest quality gap remaining.

**Wave 4 — the scoreboard.** XI.2 + XI.3, published, with other systems run on it.
Flagship pair: **Rescue@k and inverse-Rescue@k.** Cheapest credibility available, and it reframes the comparison onto ground OWL chose.

**Wave 5 — depth.** A7, A10, A11, A12 · B6, B9 · C1–C5 · D5 · E3, E4 · F1–F7 · G3–G5

**Wave 6 — research, time-boxed.** H1–H4 · B8, B11

---

## If only three things get built

1. **D1 — decision–consequence graph.** Cheapest transformative item; changes what the product is.
2. **A2 + A5 — quarantine and source independence.** The category the field is weakest on, where OWL starts ahead.
3. **B1 + XI.3 — a real embedder and the epistemic scoreboard.** One makes the numbers trustworthy; the other reframes what the numbers should be.

## The through-line

> **Provenance points backwards. Decisions, criticality, blast radius and trust propagation point forwards.** That direction is empty, cheap to build on an append-only graph, and the one that matters when being wrong has a cost.

If OWL is remembered for one thing, let it be that it could answer:

> *"That turned out to be wrong. What did I do about it, and what do I need to undo?"*

---

## Appendix — Attribution

| Source | Contribution |
|---|---|
| iai-pme | Rescue@k and its inverse · sleep ablation · non-blocking capture · store-readable-when-down · encryption at rest · stable community identity · doctor discipline · published-regression culture · HD structural recall |
| LycheeMemory | Record fusion *(built)* · decontextualisation · `failure_pattern` · action-outcome loop |
| Memanto | Provenance thesis as product framing · OKF portable export · the competitive reality check |
| MIRIX | Knowledge Vault → verbatim protection *(built)* |
| MiniRAG | Heterogeneous graph *(built)* · the small-model inversion finding |
| MemoryLLM / M+ | `nuc` interference methodology *(built)* |
| zvec | In-process ANN backend · group-by retrieval |
| Zep / Graphiti | Bi-temporality *(built)* |
| HippoRAG | PPR graph spread *(built)* |
| EM-LLM | Surprise-boundary segmentation *(built)* |
| Awesome-Agent-Memory | Memory security category · HaluMem · RF-Mem recollection/familiarity |
| **Original to OWL** | Monotonicity as enforced invariant · six FOK states · transactive memory & symmetric divergence · information-flow partitions · epistemic-demotion handover · learned claim half-life · **decision–consequence graph** · **retroactive trust propagation** · **second-order uncertainty** · **load-bearing criticality** · **attributed belief** · **commitment lifecycle** · **counter-evidence retrieval** · **cost-of-acquisition retention** · source independence graph · retrieval receipts · adversarial self-audit · model provenance · consolidation determinism · explain-the-gap · cold-start honesty · time-travel replay · jointly-edited ledger · dimensional integrity |
