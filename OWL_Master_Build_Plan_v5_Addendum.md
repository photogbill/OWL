# O.W.L. — Master Build Plan, v5 Addendum

**Twelve additions, thought through with fresh eyes.**
Companion to `OWL_Master_Build_Plan_v4.md`. Plan only; no code.

---

## The question I asked myself

Not "what does the field do that OWL doesn't" — v4 already answered that. Instead:

> **What does a memory system for consequential work still not do?**

Every system in this space, OWL included, is built around one loop: *something happened → store it → find it later.* That loop is missing three things that matter enormously the moment a decision rests on the output:

1. **Nothing tracks what was *done* because of a memory.** When a fact changes, no system can tell you which decisions are now standing on air.
2. **Nothing propagates trust backwards.** Learn a source was wrong and every system leaves the contamination in place.
3. **Confidence is a single number, so "I don't know" and "it's genuinely 50/50" are indistinguishable.** Those are completely different epistemic states and the difference is often the whole answer.

The first four items below address those. They are, I think, the most valuable things left on the table — and all four are cheap, deterministic, Tier 0, and compose with machinery that already exists.

---

# Tier 1 — Transformative

## 1. The decision–consequence graph  ★ my strongest recommendation

**The gap.** OWL can tell you *how it knows* something. It cannot tell you *what you did about it.*

An analyst reads "Route Alpha is open," routes a convoy, and moves on. Three days later a sitrep supersedes that claim. Every memory system in existence updates the fact and stops. **Nothing tells anyone the convoy decision is now standing on a false premise.**

**The mechanism.** Let a decision be a first-class node with inbound edges to the memories it rested on:

```
mind.decided("routed convoy via Alpha", because=[node_ids], at=..., reversible_until=...)
```

Then supersession becomes an *event with a blast radius*. When a memory changes, walk forward through `decided_on` edges and surface every decision downstream of it, ranked by how load-bearing the changed memory was and whether the decision is still reversible.

**Why it's the strongest item here.** It's the difference between a memory that answers questions and one that **prevents harm**. It also completes the arc of everything else in OWL: bitemporality tells you what changed, provenance tells you why you believed it, divergence tells you the operator is out of date — and this tells you *what to actually go and fix.* For ATK's mission, one aid worker, exhausted, who cannot hold it all in their head, that's not a feature. It's the point.

**Cost.** One table, one edge type, one forward traversal. No model. Genuinely a weekend.

**Second-order effect.** It creates a decision journal for free, which is the single most effective known intervention against hindsight bias — you can see what you actually believed at the time rather than what you now think you believed. Combined with time-travel replay (v4 D3), you get honest post-incident review.

---

## 2. Retroactive trust propagation — the inverse of `why()`

**The gap.** `why()` walks *backward*: how do I know this? Nothing walks *forward*: **what do I believe because of this?**

You discover a document was forged, a source was lying, or an entire ingest batch was mis-parsed. Every system in the field leaves the contamination sitting in the store. The derived summaries stay. The composites stay. The conclusions stay, now unsourced but still confident.

**The mechanism.** `blast_radius(node_id)` — the transitive forward closure over derivation edges, mention links, composite membership, and (with item 1) decisions. Then a *revaluation pass*: re-run the monotonicity clamp with the source's new reliability, and cascade. Nodes whose support falls below threshold get demoted to `hypothesized` or quarantined — never silently deleted, because the fact that you once believed it is itself evidence.

**Why it matters.** This is the natural consequence of taking provenance seriously, and it's the thing that makes provenance *pay*. Everyone else's provenance is decorative because nothing acts on it. Enforced monotonicity means OWL can do this correctly and mechanically: the invariant that clamps on write is the same one that re-clamps on revaluation.

**The killer query it enables:**

> *"I just learned that survey PDF was out of date. What did I conclude from it, what did I tell the team, and which decisions rested on it?"*

Nothing in the field can answer that. It should be one call.

**Pairs with:** the source-independence graph from v4 (D2) — flooding attacks become *retroactively* detectable and reversible, not just blockable at the door.

---

## 3. Second-order uncertainty — separate ignorance from uncertainty

**The gap.** Confidence is a point estimate. So these two states are identical in the store:

- "I have no evidence either way." → 0.5
- "I have strong, balanced evidence on both sides." → 0.5

They are not remotely the same, and for a decision-maker the difference is frequently the entire answer. The first says *go and look*. The second says *this is genuinely contested; stop looking and decide under uncertainty.*

**The mechanism.** Replace the scalar with a three-part opinion — **belief / disbelief / ignorance**, summing to 1 (subjective logic; equivalently a Beta posterior, or a Dempster–Shafer mass with explicit uncertainty mass). Evidence moves mass out of ignorance into belief or disbelief. Fusion operators for combining independent sources already exist and are well-specified.

```
"no evidence"        b=0.00  d=0.00  u=1.00
"strongly contested" b=0.45  d=0.45  u=0.10
"well established"   b=0.92  d=0.03  u=0.05
```

**Why it's a genuine differentiator.** It makes OWL's central thesis *quantitative*. "I don't know" stops being a heuristic state from the FOK gate and becomes a measured property of the belief itself. It gives the epistemic lattice a real algebra. And it makes calibration measurable in a way point-confidence cannot: you can score whether the ignorance mass was justified, separately from whether the belief was right.

**Cost and risk.** Moderate — it touches every confidence computation, so it should land early or not at all. Monotonicity generalises cleanly (a child's belief mass can't exceed its parents'; its ignorance mass can't be lower). I'd expose the scalar as a derived property so nothing downstream breaks.

**Honest caveat:** this is the one item here with real complexity cost. If it feels heavy, the cheap 80% is to carry `evidence_count` alongside confidence and let callers distinguish 0.5-from-nothing from 0.5-from-forty. But the full version is more correct and I'd argue for it.

---

## 4. Load-bearing criticality — where to spend verification effort

**The gap.** All memories are treated as equally worth verifying. They aren't. Some, if wrong, invalidate one summary. Others invalidate forty conclusions and three decisions.

**The mechanism.** PageRank (or simple reverse-reachability count) over the derivation graph, weighted by the epistemic rank of dependents and by decision edges from item 1. Output: a criticality score per node.

**What it unlocks:**
- **Verification triage.** "These four beliefs carry 60% of your current conclusions. Two are single-sourced grade-C. Verify those first." That is genuinely actionable and no memory system offers it.
- **Fragility detection.** A conclusion supported by many nodes that all trace to one origin *looks* robust and is not. Criticality plus the independence graph exposes exactly that.
- **Better forgetting.** Criticality is a far better retention signal than access count. A memory nothing depends on is safe to compress; a load-bearing one never is, however rarely it's touched.

**Cost.** Cheap. Sparse graph, one iterative computation during `tend()`.

---

# Tier 2 — Strong and novel

## 5. Attributed belief — separate the claim from the claimant

**The gap.** OWL stores *"Ahmed said the parts arrive Thursday"* as a string. The Admiralty grade attaches to the *source document*, not to **Ahmed**. So you cannot ask "what does Ahmed believe?", "who else claims this?", or "Ahmed has been wrong four times — what else did he tell me?"

This is the *de dicto / de re* distinction and it is fundamental to analytical tradecraft. Intelligence work has treated it as basic for a century; no LLM memory system implements it.

**The mechanism.** Model a claim as `(claimant, proposition, assertion_time)` distinct from the proposition itself. Multiple claimants can assert the same proposition (that's corroboration, and it composes with the independence graph). A claimant accumulates a reliability record from outcomes.

**What it unlocks:**
- Per-claimant reliability learned from outcomes rather than assigned by hand
- "Who told me this?" and "what else did they tell me?" as one-hop queries
- Combined with item 2: when a claimant's reliability drops, every proposition resting on their word revalues automatically
- Correct handling of the case where a *reliable* source reports something *implausible* — the two Admiralty axes finally have somewhere proper to live

## 6. Commitment lifecycle — promises are not facts

**The gap.** "I'll bring fuel Thursday" is stored as a fact. It is not a fact. It is a **commitment**, and commitments have a lifecycle: made → due → kept or broken.

**The mechanism.** A speech-act type on write (assertion / commitment / request / hypothesis), with commitments carrying a due date and a resolution state. On the due date the commitment surfaces automatically — that's prospective memory, already built. On resolution, the outcome feeds back into the claimant's reliability record from item 5.

**Why I like this one.** It closes a loop nobody closes: **broken promises automatically degrade source reliability, which automatically revalues everything that source ever claimed.** Items 2, 5 and 6 form a single self-maintaining trust system, and each is individually cheap. Memanto has `commitment` as a memory *type*; nobody tracks the *lifecycle*, and the lifecycle is where the value is.

For field work this is directly useful: who actually delivers, and who says they will.

## 7. Counter-evidence retrieval — confirmation resistance by construction

**The gap.** Every retriever optimises for *similarity to the query*, which means it systematically returns what agrees with the framing of the question. If you ask "why is the pump failing?", you get pump-failure evidence. You do not get the note saying the pump was fine last week.

Confirmation bias is not just a human failing here — **it's baked into the objective function of similarity search.**

**The mechanism.** Alongside the supporting set, deliberately retrieve the strongest *disconfirming* set: memories that contradict the query's presupposition, that are semantically adjacent but oppositely valenced, or that were superseded by the belief the query assumes. Surface both, labelled, via the retrieval-as-shape structure (v4 B5).

**Why it's novel and right.** It's the retrieval-layer expression of OWL's whole stance — the most valuable output is often the one that contradicts what you assumed. And it composes with the sycophancy firewall in the ToM layer: OWL already refuses to *soften* contradictions; this makes it *seek* them.

## 8. Cost-of-acquisition as a retention signal

**The gap.** Retention is driven by access frequency, recency, and surprise. None of these capture what a memory **cost to obtain**.

Some facts cost a three-day trip, a difficult conversation, or a canvass of six vendors. Others cost a glance at a filename. Losing the first kind is expensive; losing the second is free, because it's re-derivable on demand.

**The mechanism.** An `acquisition_cost` on write — supplied by the host, or inferred from signals it already has (elapsed time, tool calls, number of sources consulted, whether a search preceded it). Feed it into salience as a protection factor.

**Why nobody has it.** Because everyone models memory as storage rather than as an investment. But it's obviously correct: the right question for forgetting is not "how often was this used?" but **"what would it cost me to get this back?"** Negative memory already encodes this instinct — "I canvassed six vendors, none stock diesel" is precious precisely because it was expensive. Generalise it.

---

# Tier 3 — Targeted, cheap, high certainty

## 9. Dimensional integrity

Quantities as structured values with units, not substrings. "400L" and "400 gallons" are not the same, and a summariser that drops the unit has produced a dangerous sentence rather than a shorter one. Deterministic, Tier 0, prevents a real class of error — and for medical or fuel or dosage content in the field, it's a safety feature rather than a nicety. Pairs naturally with `verbatim` protection.

## 10. Cross-lingual claim identity — specifically for ATK

ATK translates. A memory recorded in Somali should be findable by an English query and vice versa, and the *same claim* in two languages should corroborate rather than duplicate.

Mechanism: language tag on write, claim identity resolved in a shared multilingual embedding space, original text always preserved as the evidence. Nobody in the reviewed set handles this, and for the actual mission — an aid worker somewhere they don't speak the language — it may be worth more than several items above it.

## 11. Query critique — "you're asking the wrong question"

Every system treats the query as ground truth. But when the store contains a strong answer to an *adjacent* question, saying so is more useful than answering the one asked. "You asked when the tanker arrives; I have nothing on that, but I have the depot's dispatch schedule and a note that tankers are dispatched on request." Cheap given the entity graph, and it converts a `DONT_KNOW` into a lead — pairs directly with "explain the gap" (v4 B6).

## 12. Speed work

Worth naming explicitly since "performance" reads two ways:

- **Bloom/cuckoo filter over the lexical index** so `DONT_KNOW` is genuinely O(1) rather than O(query terms) — the fast path deserves to be the fastest path, and it's the one that fires most often.
- **Memory-mapped vectors** instead of loading blobs per query. Brute force stays viable far longer than expected once you stop deserialising.
- **Recall caching with write-driven invalidation**, keyed on (query, partition, clock bucket). Repeat queries within a session are extremely common.
- **Incremental index maintenance** — currently `tend()` rescans. Dirty-flag what changed.
- **Partition-sharded storage** so a large work partition never slows a small private one.

---

# What I'd actually do

If it were my call, in this order:

1. **Decision–consequence graph (1)** — cheapest transformative item, and it changes what the product *is*. A memory that answers questions becomes a memory that stops you acting on stale beliefs.
2. **Retroactive trust propagation (2)** — makes provenance pay. Everyone else's provenance is decorative because nothing acts on it.
3. **Criticality (4)** — cheap, and it makes verification effort rational instead of arbitrary.
4. **Attributed belief + commitments (5, 6)** — together they make trust self-maintaining.
5. **Second-order uncertainty (3)** — highest complexity cost, so land it early or accept the `evidence_count` approximation. But it's the most intellectually correct thing here.

**What I'd cut or defer:** query critique (11) is a nice-to-have that will feel clever and get used rarely. Counter-evidence retrieval (7) is excellent but should wait for a real embedder, because at Tier 0 it will surface noise and discredit itself. And I'd hold cross-lingual (10) until ATK integration is real — it's high value but only for that host.

**One caution.** Items 1, 2, 5 and 6 together make OWL a *belief-maintenance system*, not just a memory. That's a bigger claim and a bigger surface. I think it's the right direction — it's where the mission points and where nobody else is standing — but it deserves saying out loud rather than arriving at by accretion.

---

## The through-line

Every item here comes from the same observation, which I think is the real insight of this pass:

> **The field has built memories that answer questions. Nobody has built a memory that knows what it is holding up.**

Provenance points backwards. Decisions, criticality, blast radius and trust propagation all point *forwards* — from a belief to what rests on it. That direction is empty, it's cheap to build on an append-only provenance graph, and it's the direction that matters when being wrong has a cost.

If OWL is remembered for one thing, I'd want it to be that it could answer:

> *"That turned out to be wrong. What did I do about it, and what do I need to undo?"*
