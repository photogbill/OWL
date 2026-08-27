# O.W.L. Engine — Architectural Review & Improvement Proposal

**Reviewing:** *Brain's Memory Storage Explained* (Gemini transcript) → "Project Brief: The O.W.L. Engine (Ontological Workspace Lattice)"
**Date:** 30 July 2026
**Constraint assumed:** offline desktop, single machine, ~16 GB VRAM, llama.cpp + GGUF, Python, SQLite.

---

## Executive summary

The brief has a genuinely good instinct — *memory should be metabolized, not accumulated* — and one genuinely good architectural commitment: separating perception from memory from reasoning. Those are worth keeping.

But the brief reasons **from metaphor to mechanism**, and the metaphor is chosen for vividness rather than for what it predicts. "The brain is biological, databases are rigid, therefore be more biological" is not an argument. Human memory's lossiness is a constraint evolution tolerated, not an optimization target. Several proposed features — most seriously **Associative Reconsolidation** and **Active Dreaming** — replicate the exact mechanisms by which human memory produces *false* memories, in a system whose entire value proposition is analytical trustworthiness.

There are also four things missing that are more important than anything currently in the brief:

1. **Source monitoring** — provenance tracking. Its absence is the single highest-severity defect.
2. **Metamemory** — knowing whether you know, before you spend compute finding out.
3. **Interference management** — the actual cause of retrieval failure, which the brief doesn't model at all.
4. **An evaluation plan** — there is currently no way to tell whether any of this works.

This document is organised as: what's right → what's wrong → what's missing → what to add → what will physically not run on 16 GB. A rewritten, build-ready brief is in `OWL_v2_Project_Brief.md`.

---

# Part I — What the brief gets right

Keep these. They're the load-bearing good ideas.

**1. Memory as an active process with a lifecycle.** Most agent memory systems are append-only vector dumps that rot. Treating retention as a decision, made repeatedly over time, is correct and under-explored.

**2. Separating the encode path from the retrieve path from the reason path.** This maps to a real distinction in cognition and it has real engineering benefits (independent scaling, independent evaluation, cacheable prompts).

**3. Using idle time.** A desktop app is idle 95% of the time. Almost nobody exploits this. There is strong empirical support for offline consolidation improving later performance — Wagner et al. (2004) found sleep more than doubled the rate of insight on the Number Reduction Task, and the sleep-replay literature is robust. The instinct is right; the implementation needs work (see §3, §D).

**4. Prediction that the system should be proactive.** Correct, and the psychology gives you better machinery for it than the brief uses (see §D, *prospective memory*).

**5. Explicit modelling of the user rather than just their words.** Correct instinct. Dangerous as specified (see §13).

---

# Part II — Five load-bearing errors

## 1. The biological metaphor is arguing for the wrong things

The brief's core move is: *human memory is reconstructive and lossy; databases are exact and permanent; therefore be more like human memory.* Run that argument backwards and it collapses. Human memory is reconstructive **because** neurons are expensive, noisy, and can't be indexed — not because reconstruction is desirable. Its known consequences are confabulation, hindsight bias, source amnesia, and the Deese–Roediger–McDermott effect (reliably generating vivid memories of words that were never presented).

You are building an **analyst toolkit**. Its differentiating claim is that it doesn't make things up. The brief proposes, as a headline feature, a mechanism that makes things up.

**The fix is not to abandon the metaphor — it's to split it.** Biology conflates *the trace* with *the retrieval of the trace* because it has no choice. You do have a choice:

| Layer | Behaviour | Rationale |
|---|---|---|
| **Substrate** (what was said/read/observed) | Append-only, immutable, never rewritten, never deleted except on explicit user command | This is your evidence. Corrupting it is unrecoverable. |
| **Index** (what's findable, and how fast) | Decays, re-weights, re-ranks, forgets aggressively | This is where "forgetting" belongs. Cheap, reversible, safe. |
| **Derivation** (summaries, abstractions, grafts, insights) | Rewritten freely, versioned, always traceable to substrate | This is where "reconsolidation" belongs. |

Once you make this split, everything the brief wants becomes safe. "Forgetting" = dropping something out of the hot index. "Reconsolidation" = writing a *new* derived node that supersedes an old derived node, with both pointing back at unchanged substrate. Nothing is lost, and disk is not the bottleneck — a decade of dense text is a few GB; the embeddings and the index are what cost.

> **Rule to adopt:** *Decay the index, never the evidence.*

## 2. The decay formula is wrong, and this is a solved problem

The brief specifies:

```
S(t) = S₀·e^(−λt) + (a·c)
```

Three problems:

- **The functional form is wrong.** Human forgetting follows a *power* function of time, not an exponential. Wixted & Ebbesen (1991) tested this directly across multiple paradigms; power (and hyperbolic) functions beat exponentials consistently. Exponential decay forgets recent items too fast and old items too slowly.
- **The additive access term is degenerate.** `+ (a·c)` is a floor that never decays. Any memory accessed enough times becomes literally immortal regardless of age or value. Your most-accessed junk — greetings, filler, boilerplate — becomes permanent. Meanwhile it's dimensionally incoherent: you're adding a raw count to a decayed salience.
- **It only knows `last_accessed`.** So it cannot represent the **spacing effect** — the single most robust finding in the memory literature. Ten accesses in one hour and ten accesses spread over a year produce identical scores under this formula, and radically different retention in reality.

**Replacement (option A — the principled one).** ACT-R's base-level activation, from Anderson & Schooler's (1991) rational analysis:

```
B_i = ln( Σ_j  t_j^(−d) )
```

where `t_j` is the time since the *j*-th access and `d ≈ 0.5`. This sums a power-law decay over *every individual access*, not one decay from the most recent. It produces the spacing effect for free. And critically, Anderson & Schooler derived it by asking *what is the optimal prediction of whether this item will be needed right now, given environmental statistics?* — which means it isn't a metaphor, it's a Bayesian estimate. Forgetting becomes rational inference, not simulated decrepitude.

**Replacement (option B — the pragmatic one).** Use **FSRS**, the DSR model (Difficulty, Stability, Retrievability) now default in Anki. It's a fitted, benchmarked memory model — 20–30% fewer reviews than SM-2 for equal retention across 500M+ real reviews — with mature open-source implementations you can vendor in ~100 lines. Its three state variables map directly onto what you need:

- **Retrievability** = "can I get this right now?" → drives retrieval ranking
- **Stability** = "how long until I can't?" → drives compression/prune scheduling
- **Difficulty** = "how hard is this to make stable?" → drives how much compute to spend on it

Notably, FSRS's *stability* is explicitly the storage-strength construct from **Bjork & Bjork's New Theory of Disuse** — which gives you the right conceptual model:

> **Storage strength never decreases. Retrieval strength does.**

That is exactly the substrate/index split from §1, arrived at independently from the psychology. Two variables, not one. Never delete anything with high storage strength; just let it fall out of the fast index.

**Drop the 30-day / 60-day thresholds entirely.** They are arbitrary, they will be wrong for your data, and a fitted model gives you the right number per-item.

## 3. Reconsolidation as specified is a confabulation engine

From the brief:

> *"...saving a newly synthesized, slightly more abstracted memory back to the database, deprecating the old row... the persona organically builds generalized wisdom."*

This is Bartlett's *War of the Ghosts* (1932) implemented as a feature. Bartlett's subjects, retelling an unfamiliar story repeatedly, showed systematic **levelling** (loss of detail), **sharpening** (over-emphasis of a few elements), and **rationalization** (assimilation to prior schema) — while remaining fully confident. Iterate that loop and you don't get wisdom, you get a confidently-held summary that no longer corresponds to anything that happened.

Worse: run it on a system that also feeds its own outputs back in as training data (Pillar V), and you have textbook **model collapse** — recursive training on generated data drives distributional narrowing until tails vanish.

There is also a subtler bug. Abstraction *feels* like progress because the abstracted version is shorter and more general. But information-theoretically you've thrown away the residual, and the residual is where analysis lives. An analyst toolkit that gradually converts specifics into generalities is gradually becoming useless at its job.

**Three fixes, all of which you should take:**

**(a) Reconstructive compression — only compress what you can provably rebuild.** Instead of a timer deciding when to summarize, make it an *empirical test*. During idle time, take a memory, hide the body, show the system only the cue, and ask it to reconstruct. Then diff against the original.

- Reconstruction succeeds → the content is redundant with what the model already knows or with neighbouring memories. **Safe to compress to a cue + delta.**
- Reconstruction fails → this memory carries real, non-derivable information. **Keep verbatim. Increase priority.**

This is self-verifying lossy compression. It replaces a hand-tuned heuristic with a measurement, and it directly converts the biggest liability of Pillar II into its biggest strength. It also happens to be the *retrieval practice* effect (Roediger & Karpicke, 2006 — testing beats restudying, substantially) applied to the machine rather than the human. This is, in my view, the single best idea in this document; if you take one thing, take this.

**(b) Never overwrite. Supersede.** New derived node, `supersedes` edge to the old one, old one demoted to cold storage but retrievable. Then "what did you used to think, and what changed your mind?" is a query, not an archaeology project.

**(c) Store the exception, compress the rule.** From schema theory (and the SLIMM/Tse et al. 2007 work showing schema-*consistent* information consolidates unusually fast): information that fits an existing schema is cheap to compress, because the schema regenerates it. Information that *violates* schema is exactly what must be preserved verbatim — it's the surprising part, it's the part with the information content, and it's the part a naive summarizer will delete first because it looks like an outlier.

Concretely: compression should be schema-aware. Compress toward "the usual pattern, plus these three specific deviations." That's minimum description length, and it's also how expertise actually works.

## 4. "Cognitive isolation" is the wrong construct

The brief's Pillar I is total blindness: the Apperception Node sees no memory, no history, no goal. The stated motivation — anchoring (Tversky & Kahneman, 1974) — is a real and serious effect. But total blindness is the wrong remedy, for three reasons.

**It breaks encoding specificity.** Tulving & Thomson's (1973) encoding specificity principle says retrieval succeeds to the extent the retrieval cue overlaps the *encoding context*. A node that extracts facts with no context strips exactly the contextual information that later retrieval will need as a cue. You'd be optimizing a bias problem by creating a much larger retrieval problem.

**It cannot resolve reference.** "Do it the way we did last time." "Same as the Henderson file but for Q3." "That's still broken." A blind extractor produces garbage JSON on all of these, and it produces it *confidently*, which is worse — the downstream swarm receives a clean-looking payload built on a failed dereference. Real analyst input is dense with deixis, anaphora, and user-specific jargon. Blindness makes this unfixable by design.

**What you actually want is decorrelated error, not zero information.** This is the wisdom-of-crowds condition. Lorenz et al. (2011, PNAS) showed that even mild social influence collapses crowd accuracy by shrinking diversity without improving the mean — so the thing worth protecting is *independence of errors*, and you can get that without blindness:

- **Bagging** — give different specialists different *subsets* of the retrieved evidence.
- **Varied priors** — different system prompts, different temperatures, deliberately different framings of the same question.
- **Dialectical bootstrapping** (Herzog & Hertwig, 2009) — the "crowd within." Ask the *same* model to assume its first answer was wrong and argue the other side. Averaging the two beats either. This is nearly free and it's a strong fit for a VRAM-constrained box where you can't afford many distinct models.
- **Delphi structure** — anonymized, structured, iterated rounds. Information flows, but attribution and social pressure don't. Strictly better than both "everyone talks" and "nobody talks."

**And there's a failure the current design makes actively worse: the hidden profile problem.** Stasser & Titus (1985) showed groups systematically fail to surface information held by only one member, over-weighting what everyone already shares. With total isolation, *no* node can pool unshared information — only the Composer can, which makes the Composer a single point of bias for the exact thing the architecture was supposed to fix.

**Fix:** an explicit **unshared-information premium** in aggregation. When exactly one specialist raises a claim, and that claim falls inside that specialist's competence domain, weight it *up*, not down. Uniqueness is evidence, not noise. This one line of aggregation logic does more for cognitive diversity than the entire isolation apparatus.

**Revised Pillar I:** replace *blind* with *contextualized but decorrelated*. Give the extraction node the context it needs to resolve reference and preserve encoding cues. Prevent anchoring by (i) separating fact-extraction from stance-taking as two distinct calls with different prompts, (ii) having it extract claims *and* their negations, and (iii) requiring it to emit the user's framing as a labelled, quarantined field (`user_framing`) rather than silently discarding it — because the framing is itself data about the user, which Pillar IV needs.

## 5. Gardner's multiple intelligences is a discredited taxonomy

The persona set is "the eight Gardner intelligences." Multiple Intelligences theory has very little empirical support as a theory of human cognition. When Visser, Ashton & Vernon (2006) built two tests for each of Gardner's eight proposed intelligences and factor-analysed the results across 200 adults, the purely cognitive domains — Linguistic, Logical/Mathematical, Spatial, Naturalistic, Interpersonal — loaded heavily onto a single general factor rather than separating into independent intelligences. The theory has also been criticized at length for being difficult to falsify and for relabelling *talents* as *intelligences* (Waterhouse, 2006). It survives in education and pop-psych, not in cognitive science.

You're not obliged to care whether your persona taxonomy is neurologically real. But you *are* importing an ontology, and this one carves the space badly for your purpose: "Bodily-Kinesthetic" and "Musical" have no meaningful role in a document-analysis toolkit, while "verify a numerical claim against a source" and "detect a rhetorical move" — things you actually need constantly — have no home.

**Better bases for the persona set, in order of my preference:**

1. **Empirically derived from your own query log.** Run the toolkit with a generic persona for a month, cluster the query types, define specialists around the clusters. This is the only approach guaranteed to fit the actual workload.
2. **Epistemic role decomposition** — the roles an analytical claim actually passes through: *Extractor* (what does the source say), *Corroborator* (what else supports/contradicts it), *Adversary* (how would this be wrong), *Quantifier* (what do the numbers say), *Historian* (what did we previously believe), *Synthesist*. These are functionally distinct, non-overlapping, and each is separately evaluable.
3. **Stanovich's tripartite model** (autonomous / algorithmic / reflective) if you want a defensible cognitive-science grounding, since it maps cleanly onto a fast path, a working path, and an override/metacognitive path.

Option 2 is what I'd build. It also gives you a natural, honest place for the adversary — see §12.

---

# Part III — Structural gaps

## 6. No source monitoring — the highest-severity defect

**Nowhere in the brief does a memory record where it came from.**

This is the fix that matters most. The **source monitoring framework** (Johnson, Hashtroudi & Lindsay, 1993) is the dominant account of how false memories actually form: not by fabricating content from nothing, but by *correctly* remembering content and *incorrectly* attributing its source. You remember the fact; you forget you imagined it.

Your architecture is a source-monitoring catastrophe waiting to happen, because it deliberately mixes four kinds of content in one store:

1. Things the user told you
2. Things a document said
3. Things a specialist model inferred
4. **Things the system invented while dreaming**

Category 4 is written back into the same memory store as categories 1–3 and later retrieved as context. Within a few cycles the system will assert a dream-generated claim with the confidence of a user-provided fact, and there will be no way to tell — including for the system itself.

**Mandatory fix.** Every memory node carries an immutable provenance chain:

```
origin:        user_utterance | document | tool_output | model_inference | dream_synthesis
source_ref:    URI / doc hash + offset / conversation id + turn
derivation:    [list of parent node ids]  # empty for primary sources
epistemic_tag: observed | reported | inferred | hypothesized
corroboration: [ids of independent supporting nodes]
```

And three hard rules:

- **`dream_synthesis` and `model_inference` nodes may never be presented as fact.** They surface with an explicit marker, always.
- **Provenance is transitive and monotone.** Any node derived from a hypothesis is a hypothesis. Confidence cannot exceed the minimum of its parents. Abstraction cannot launder speculation into fact.
- **Every user-facing claim must be traceable to at least one `origin: user_utterance | document | tool_output` node.** If it can't, it's flagged as system-generated.

This one addition does more for trustworthiness than all five original pillars combined, and it costs a few columns.

## 7. No metamemory layer — the biggest missed opportunity

Humans do something machines almost never do: we know *whether we know* something before we try to retrieve it. Ask someone the capital of a country they've never heard of and they answer "no idea" in under a second — without searching. Ask for a word on the tip of their tongue and they'll tell you they know it, that it starts with a hard consonant, and that they'll have it in a minute. This is **metamemory** — Feeling-of-Knowing and Judgment-of-Learning (Koriat's work is the standard reference), and it is *fast, cheap, and separate from retrieval itself*.

O.W.L. currently retrieves first and evaluates after, on every query. That's expensive on a 16 GB box where every retrieval is followed by a model load.

**Add a Feeling-of-Knowing gate.** A tiny, cheap classifier (embedding-space density around the query + a small model, ~50ms) that triages *before* any expensive work:

| FOK state | Meaning | Action |
|---|---|---|
| **KNOW** | high-salience direct match | answer from memory, skip the swarm entirely |
| **KNOW-WHERE** | no direct match, strong neighbourhood | targeted retrieval, one specialist |
| **TIP-OF-TONGUE** | high familiarity, low retrievability — partial match, conflicting neighbours | *escalate*: widen search, this is where the interesting stuff is |
| **DON'T-KNOW** | sparse neighbourhood | say so immediately; don't hallucinate, don't spend 60 seconds |

Two things make this valuable beyond speed. First, **DON'T-KNOW is a first-class output** — a system that reliably says "I have nothing on this" is worth more to an analyst than one that always produces something. Second, **TIP-OF-TONGUE is a signal, not a failure**: high familiarity plus low retrievability is precisely the signature of *interference* between competing memories, which is the thing you most need to detect (see §8).

## 8. Interference, not decay, is what kills retrieval

The brief models forgetting entirely as **decay** — things fade with time. The psychological literature has largely favoured **interference** since Underwood (1957): we forget mainly because *similar competing memories* block retrieval, not because traces evaporate. Proactive interference (old blocks new) and retroactive interference (new blocks old) explain far more forgetting than time alone.

This matters enormously for you, because it changes what the maintenance job should do. As specified, O.W.L.'s background worker prunes by **age**. But your retrieval failures will not be caused by age — they'll be caused by having forty near-identical memories about the same recurring topic, none of which is clearly the right one. Age-based pruning does nothing about that. It may make it worse, by deleting the one distinctive old memory and keeping the forty redundant recent ones.

**Add a de-interference sweep** as the primary maintenance job, ahead of decay:

- **Confusability detection.** Find node pairs with high embedding similarity. Partition them:
  - *Redundant* (same content, same source) → merge, keep highest-fidelity, sum the access statistics.
  - *Contradictory* (similar topic, incompatible claims) → this is the brief's "belief tension," but now with a principled trigger instead of a vague one. Escalate to resolution: check provenance, prefer better-sourced, and if unresolvable, **store the conflict explicitly as a node** rather than picking a winner.
  - *Genuinely distinct but confusable* → **differentiate**. Rewrite the retrieval cues (not the content) to emphasize what distinguishes them. This is the machine version of what a human expert does when they learn to tell two similar things apart.
- **Retrieval-induced forgetting** (Anderson, Bjork & Bjork, 1994): retrieving one item actively suppresses its competitors. Implement as: when a memory is successfully used, mildly *demote the retrieval strength of its near-neighbours that weren't used*. This is free, self-organizing disambiguation — the store sharpens itself around what actually gets used, without deleting anything.

One caution: implement suppression as **demotion in ranking**, never as an exclusion filter. Wegner's ironic process theory is a nice reminder of the general shape of the bug — a system that has to check "is this the suppressed item?" has already retrieved it, and you've paid the cost twice.

## 9. Consolidation ≠ compression: Complementary Learning Systems is missing

The brief names a "Hippocampal Controller" but doesn't use the theory the name comes from. **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995) is the most directly applicable theory in all of memory research to what you're building, and it's absent.

CLS says the brain needs *two* memory systems with incompatible requirements:

| | Hippocampus (fast) | Neocortex (slow) |
|---|---|---|
| Learning rate | High — one-shot | Low — many exposures |
| Representation | Sparse, **pattern-separated** | Dense, overlapping |
| Content | Specific episodes | Statistical structure |
| Failure if used alone | No generalization | **Catastrophic interference** |

And the reason for the split is the punchline: if you write new information directly into the slow, overlapping system, it **catastrophically overwrites** what's there. The only known fix is **interleaved replay** — replaying old material mixed with new, which is what offline consolidation is for.

**Two concrete consequences for O.W.L.:**

**(a) Pillar V walks directly into catastrophic forgetting.** The brief proposes generating synthetic self-play data during idle time and using it to "update its own GGUF weights the next time a training loop runs." Fine-tuning on a corpus of freshly generated adversarial debate, without interleaving representative old data, is the canonical recipe for catastrophic interference *plus* model collapse. If you do any weight updating, interleaving is not optional. Honestly: for v1, don't touch weights at all. LoRA-per-domain with a frozen base gets you most of the benefit with none of the risk, and it's reversible.

**(b) Pattern separation is a real, cheap engineering win that current RAG gets backwards.** The hippocampus *orthogonalizes* similar inputs on the way in — it deliberately makes similar experiences less similar, so they don't blur. Standard vector RAG does the opposite: it embeds everything in one space where similar things collapse together, which is *exactly* the interference problem from §8.

Fix: **use two different embedding spaces.**

- **Write path (pattern separation):** embed with the *distinguishing* context included — timestamp bucket, session, source, the surrounding turn. Deliberately push near-duplicates apart. Optionally sparsify (top-k) the write embedding. Two similar meetings should land in different places.
- **Read path (pattern completion):** embed the query semantically, retrieve a *neighbourhood*, then let the completion step reconstruct. Retrieve broadly, then discriminate — rather than retrieving narrowly and hoping the top-1 is right.

This is one of the few places where the biological metaphor pays real engineering dividends rather than just supplying vocabulary.

## 10. Encoding is where memory is won, and the brief encodes shallowly

The Apperception Node extracts entities, claims, intent, constraints. That is **shallow processing** in Craik & Lockhart's (1972) sense — structural/surface features. Their levels-of-processing finding is that *depth of encoding* predicts later retention far better than amount of rehearsal does. You cannot fix a shallow encode with a clever retriever downstream; the information isn't there.

Three cheap upgrades at encode time, each with strong empirical backing:

- **Elaborative encoding.** Don't just extract *what* was claimed — extract *why it matters here*, and *what it connects to*. One extra generated sentence at write time is worth more than a much larger retrieval budget at read time, and it's paid once instead of every query.
- **Generation effect** (Slamecka & Graf, 1978). Material you generate is remembered better than material you read. Have the encoder generate the *questions this memory answers*, and index those alongside the content. Now retrieval matches question-to-question instead of question-to-statement, which is a much better-posed matching problem. (This is roughly why HyDE-style techniques work, arrived at from the other direction.)
- **Encoding specificity → store the cue environment.** Alongside every memory, store the retrieval context: what the user was doing, what the active goal was, what else was on screen, what the previous three turns were. At retrieval, reinstate this context as part of the query. Context-dependent memory effects are large and well-replicated, and this is the mechanism behind them.

## 11. The Composer is a homunculus

Read Pillar IV closely: the Executive Composer allocates all compute, maintains a predictive model of every specialist's biases, mathematically reweights their outputs, runs the ToM check, and produces the synthesis. It knows everything and decides everything.

You have rejected the monolithic model and rebuilt it in the middle of your architecture. Baddeley made exactly this criticism of his own **central executive** — that it risked being a homunculus, a little person inside doing all the un-explained work — and spent decades decomposing it into specific functions.

Also, practically: the Composer is now the single point through which all bias flows, which nullifies the diversity you paid for upstream. And on a 16 GB box it's your largest model, so it's your slowest component and your dominant VRAM cost.

**Decompose it. Make aggregation mechanical wherever mechanical is possible:**

- **Evidence pooling → arithmetic, not an LLM.** Claims arrive with confidence and provenance. Combine them with an explicit rule (log-odds pooling with per-specialist reliability weights). Deterministic, auditable, testable, free.
- **Reliability weights → measured, not modelled.** Don't have a big model *guess* that "the Intrapersonal node over-indexes on ethics." **Score it.** Keep a per-specialist calibration record — Brier score by claim type — and derive the weights from outcomes. This replaces a hand-wavy "internal Theory of Mind" with a number you can plot.
- **Conflict detection → mechanical.** Contradiction between specialists is detectable structurally.
- **The LLM's job shrinks to prose.** Given a resolved, weighted, conflict-annotated claim set, write the answer. That's a job a much smaller model can do well, which is a significant VRAM win.

## 12. The compute economy will collapse into a monoculture

Pillar III: specialists that perform well get bigger budgets. This is a **pure-exploitation bandit** with no exploration term. It has one stable outcome — an early lucky winner takes everything, the others starve, and you're back to a monolith with extra steps. Classic Matthew effect.

It also invites **Goodhart's law**: any metric the Composer rewards becomes a target the specialists optimize instead of the actual goal. "High-confidence outputs earn larger budgets" trains specialists to be overconfident. That is precisely the wrong incentive for an analyst tool.

**Fixes:**

- **Explore, don't just exploit.** Thompson sampling or UCB over specialist selection. Occasionally route a query to a specialist you don't expect to help — that's how you find out you were wrong about it.
- **Reward marginal contribution, not usage.** Score each specialist by leave-one-out impact on final answer quality (a Shapley-style estimate). Otherwise you reward whoever produces the most text.
- **Reward calibration, not confidence.** Budget should track Brier score. A specialist that says "60%" and is right 60% of the time outranks one that says "95%" and is right 80% of the time. This inverts the Goodhart pressure into the right direction.
- **Hard budget floors.** No specialist can be starved below a minimum. Capacity you've thrown away is capacity you can't discover a use for.
- **A protected Adversary with an unconditional budget.** Nemeth's work on minority dissent is clear that *authentic* dissent improves the quality of group reasoning even when the dissenter is wrong — it forces divergent search. A red-team node whose budget can never be cut, whose only job is to construct the strongest case that the emerging answer is wrong. Don't let it earn or lose resources; that's the point.

## 13. ToM as specified is a sycophancy engine

Pillar IV has the Composer ask, before delivering: *"will this explanation cause confusion? Does it contradict a deeply held premise?"* and then adjust.

Read that as an objective function. You are optimizing outputs for *user comfort and non-contradiction of existing beliefs*. That is the definition of sycophancy, and it is a known, strong, and hard-to-detect failure mode of LLM systems. In an analyst toolkit, the most valuable output is frequently the one that contradicts a deeply held premise.

**Fix: a strict content/presentation firewall.**

- The ToM worker receives the draft **after** claims and confidences are finalized.
- It may emit **only** presentation directives: ordering, length, vocabulary level, how much scaffolding to build before the hard part, whether to lead with the conclusion or the evidence.
- It **may not** alter, soften, hedge, or drop any claim, and may not touch any confidence value.
- **Enforce mechanically:** extract the claim set from pre- and post-ToM drafts, diff them, and reject the revision if the sets differ. Not a guideline — a test.

**And re-aim ToM at the problem it's actually good for: the curse of knowledge.** Camerer, Loewenstein & Weber (1989) showed that people who know something cannot successfully model someone who doesn't. That is the real gap between a system that has read everything and a user who hasn't. The right ToM question is not *"will this upset them?"* but **"what does this explanation assume that they haven't been told?"** — which is a presentation question, answerable from the user model, and which makes the output better rather than more agreeable.

Two refinements:

- **Target the zone of proximal development.** Vygotsky's construct is the right one: aim explanations just above current demonstrated competence. You have the data to estimate it — what they've asked, what they've corrected you on, what vocabulary they use unprompted.
- **Cap recursion.** "I think that you think that I think..." — humans reliably manage about four to five orders of intentionality and degrade sharply beyond. Deeper recursion is expensive and unstable. Cap at 2 (system models user; system models user's model of the system). Everything beyond is cost without benefit.

## 14. There is no evaluation plan

The brief specifies five elaborate pillars and zero ways to know whether any of them help. This is the difference between a research architecture and a build. If you can't measure it, you'll ship all five and never learn that two of them were hurting.

**Minimum viable evaluation, all of it automatable offline:**

| Test | Method | Fails if |
|---|---|---|
| **Retention** | Inject known facts, simulate N days of clock, query | Recall collapses faster than your intended curve |
| **Confabulation** | Probe for facts never provided, and for facts that were *deleted* | Any confident answer to an absent fact |
| **Source attribution** | Ask "how do you know that?" on every claim | Any claim traced to the wrong origin class |
| **Calibration** | Brier score / ECE on stated confidences | Systematic overconfidence — this is the one that matters most |
| **Interference** | Insert near-duplicate confusable memories, then query | Retrieval accuracy degrades as duplicates increase |
| **Hidden profile** | Distribute a decisive clue to exactly one specialist | Composer's answer doesn't change |
| **Chunking / learning** | Compression ratio of new input over time | Flat — means it isn't actually building expertise |
| **Sleep ablation** | A/B: dream engine on vs. off, next-day task performance | No measurable improvement → **cut Pillar V** |
| **Latency & VRAM** | p50/p95 per query, peak VRAM | Exceeds budget (see Part V — it will) |

The **sleep ablation test is the important one.** Pillar V is the most expensive, most complex, and most speculative part of the design. Build it behind a flag, measure it, and be willing to delete it. A system with excellent memory hygiene and no dreaming beats a system with elaborate dreaming and unmeasured drift.

---

# Part IV — Twelve additions

These are the creative extensions. Each is grounded in a specific finding and each is implementable on your hardware.

### A. Reconstructive compression *(see §3a — repeated because it's the most important)*
Compress only what you can prove you can rebuild. Idle-time retrieval practice as the compression decision procedure. Self-verifying, replaces arbitrary thresholds with measurement.

### B. Metamemory / Feeling-of-Knowing triage *(see §7)*
Cheap pre-retrieval gate. KNOW / KNOW-WHERE / TIP-OF-TONGUE / DON'T-KNOW. Saves most of your compute and makes "I don't know" a reliable, fast, first-class answer.

### C. Two-phase sleep, driven by sleep pressure

The brief triggers consolidation on idle CPU. That's a resource check, not a need check. Borbély's **two-process model** of sleep regulation is a better scheduler: **Process S** (homeostatic sleep pressure, accumulating with time awake) plus **Process C** (circadian timing). Map directly:

- **Process S** = accumulated unconsolidated load — new memories written, unresolved contradictions, orphan nodes, pending reconstruction tests. Sleep pressure *builds* with use.
- **Process C** = the user's observed usage rhythm. Learn when the machine is reliably free.

Consolidate when pressure is high *and* the window is open — not merely whenever the CPU dips.

Then **split the night into two phases**, following the differential roles of slow-wave and REM sleep. SWS is associated with declarative consolidation and replay; REM is associated with remote/loose associations and creative recombination — Cai et al. (2009, PNAS, *"REM, not incubation, improves creativity by priming associative networks"*) found REM specifically improved Remote Associates Test performance relative to both quiet rest and non-REM sleep, and showed the effect was not explained by better memory for the primed items.

| Phase | Job | Node selection | Sampling | Output |
|---|---|---|---|---|
| **NREM** (first, longer) | Consolidate | Recent, high-salience, high-interference | Low temp, constrained | Merged nodes, resolved conflicts, compressed summaries, updated indices |
| **REM** (second, shorter, capped) | Recombine | **Distant** pairs — deliberately low similarity | High temp, unconstrained | `hypothesis` nodes, **quarantined**, each with a falsification test |

The phase split is a two-line config change and it makes both jobs better, because "compress reliably" and "make weird connections" want opposite sampling parameters and opposite node-selection policies. Trying to do both in one pass gets you neither.

**REM output goes to quarantine, always.** A dream node is a hypothesis, tagged `dream_synthesis`, and it must carry a **falsification test** — a concrete check the system can run later ("if true, document X should contain Y"). Hypotheses that survive testing get promoted. Hypotheses that fail get archived as failed. Hypotheses that can't be tested expire. Without this, the dream engine is a random-idea generator that pollutes your evidence base.

### D. Prospective memory — what the Watch Officer actually is

The "Watch Officer" is reinventing **prospective memory**: memory for intentions rather than for the past (Einstein & McDaniel's is the standard framework). Adopting the name gets you the whole design for free:

- **Event-based triggers** — "when a document mentioning X appears, do Y." Cheap: cue-matching, no polling.
- **Time-based triggers** — "in three weeks, check whether this held." Expensive in humans (requires self-initiated monitoring), trivial for you. Lean on these.
- **Intention superiority effect** — uncompleted intentions stay more accessible than completed ones. Give pending intentions a persistent activation bonus that drops on completion. Falls out naturally from the salience model.
- **Explicit cue specificity** — prospective memory succeeds when the cue is distinctive and fails when it's vague. So intentions must be stored with a *concrete, matchable* trigger, not a fuzzy topic.

This reframes the Watch Officer from "a background process that monitors folders" (vague, unbounded, a surveillance surface) to "a queue of explicit intentions with explicit triggers" (bounded, auditable, user-inspectable). Much better product, much better security posture.

### E. Replace the flat store with a Self-Memory System hierarchy

The brief has one flat memory table with a type column. Conway & Pleydell-Pearce's **Self-Memory System** describes autobiographical memory as a three-level hierarchy — *lifetime periods* → *general events* → *event-specific knowledge* — accessed top-down and governed by a "working self" that maintains coherence.

This is a far better structure for an analyst toolkit than a flat vector store, because it matches how work is actually organized:

- **Lifetime periods** → engagements, projects, cases. Long-lived, coarse, always in context.
- **General events** → recurring activities within them. "The weekly review." "Source vetting for the Henderson matter."
- **Event-specific knowledge** → the individual turns, documents, findings.

Retrieval descends the hierarchy instead of doing one flat similarity sweep. It gives you natural summarization boundaries (a period's summary is a real object, not an artifact of a chunker), natural decay boundaries (close a project → the whole period demotes together, coherently), and natural scoping (you can *ask about a period*, which flat vector search handles badly).

Two supporting effects worth exploiting: the **reminiscence bump** — disproportionate retention of formative periods — maps to project kickoffs, where the framing decisions get made and are later hard to recover. And **primacy/recency**: bias retention toward the start and the current edge of each period, which is exactly right for how projects are actually recalled.

### F. Successor representation — predictive prefetch

Index memories not only by *what they're about* but by *what tends to be needed next*. The **successor representation** (Dayan, 1993) encodes states by their expected future occupancy, and there's a substantial line of work arguing the hippocampus implements something like a predictive map (Stachenfeld, Botvinick & Gershman, 2017).

Concretely, and cheaply: maintain a transition matrix over memory nodes — when node A is retrieved, which nodes get retrieved within the next few turns? Then, on retrieving A, **prefetch A's successors** into working memory before they're asked for.

Why this is worth real money on your hardware: your dominant latency cost is model loading and retrieval round-trips. Prefetching along learned transitions means the relevant context is already assembled when the follow-up question arrives. It's a sparse matrix and a counter. It is possibly the highest performance-per-line-of-code item in this document.

It also gives you something conceptually nice: a memory system that has *expectations*. Which sets up —

### G. Surprise-gated encoding

The brief lost its "Amygdala Worker" between drafts. Bring the function back, but built on **prediction error** rather than "emotion," which is both more implementable and closer to the actual mechanism. Emotional arousal enhances consolidation (Cahill & McGaugh's work on this is extensive), but the computational substrate that generalizes is *surprise*: the von Restorff isolation effect, Bayesian surprise, reward-prediction-error learning.

Implementation: you already have a predictive model — the system itself. Before writing a memory, ask the cheap model to predict it from existing context. **Encode with priority proportional to prediction error.** Content the system could have predicted is nearly free to store (it's derivable — store a pointer). Content that surprised it is expensive and valuable.

This is a strictly better retention policy than access-frequency, because access-frequency is backward-looking and biased toward the mundane, while surprise is forward-looking and information-theoretically principled — surprise *is* information content.

**Critical caveat, and it's a nice one.** Flashbulb memories — vivid, highly emotional memories of dramatic events — are held with extremely high confidence and are *not* more accurate than ordinary memories; Talarico & Rubin (2003) tracked 9/11 memories and found consistency declined at the same rate as everyday memories while confidence stayed high. So:

> **Surprise raises retention priority. Surprise must never raise confidence.**

Two separate variables. Getting this backwards is how you build a system that is most certain about exactly the things it should doubt.

### H. Schema-delta storage: compress the rule, keep the exception
*(See §3c.)* Store the regularity once; store deviations verbatim. Schema-consistent material compresses safely because the schema regenerates it; schema-violating material is the information. MDL as the compression objective, and it makes your storage cost scale with novelty rather than with volume.

### I. Zeigarnik open loops
Unfinished tasks stay more accessible than finished ones — Zeigarnik's original finding, and one that fits your "curiosity" concept exactly. Make **open questions first-class memory objects** with a persistent activation bonus that only clears on resolution. An unanswered question the user asked three weeks ago should still be pulling on the system's attention, and should be a candidate for the REM phase and the prospective-memory queue. This is what makes the system feel like a colleague rather than a search box: it remembers what's still open.

### J. Working-memory chunk budget instead of context stuffing

The brief assumes bigger context = better. Both literatures disagree. Human working memory holds roughly **four chunks**, not seven (Cowan's revision of Miller). And LLMs degrade with long contexts in a well-documented way — the lost-in-the-middle effect, where information in the middle of a long context is used far less reliably than information at the edges.

So: **hard-cap retrieved items at 4–7 chunks**, and spend the saved compute making each chunk denser and better-selected rather than adding more. Chunking is also the actual mechanism of expertise — Chase & Simon's chess work showed masters don't have bigger memories, they have better chunks; Ericsson's long-term working memory extends this. If O.W.L. is genuinely learning, its chunks should get denser over time, and **that's measurable** (see the compression-ratio metric in §14). It's the best single proxy you have for "is this thing actually getting smarter."

### K. The reconsolidation window — a principled concurrency model

Real reconsolidation has a specific and useful shape: retrieving a memory makes it *temporarily labile*, and it re-stabilizes over a bounded window. Implement literally:

- Memory is retrieved → enters `labile` state for a bounded window (say, the session, or N minutes).
- While labile: new evidence can update it in place, cheaply.
- Window closes → node **locks**, and further changes must create a superseding node.

This gives you a clean answer to "when can this be edited?" — which is otherwise a genuinely annoying concurrency question with background writers — and it prevents thrash where every retrieval spawns a new version.

### L. Structured disagreement: Delphi + dialectical bootstrapping + unshared-information premium
*(See §4.)* Replace total isolation with structured, anonymized, iterated rounds; get extra decorrelation for free by having single models argue against their own first answers; and explicitly up-weight claims raised by exactly one competent specialist.

---

# Part V — Engineering reality check on 16 GB

The brief's own table implies this query path:

```
Apperception (7B) → 3 Specialists (7-8B, sequential) → Composer (24B) → ToM (7B)
```

With `max_active_models=1` on 16 GB, that is **six model load/unload cycles per query**. A 24B Q4 GGUF is roughly 14 GB — it alone nearly fills your VRAM, and loading it from disk takes seconds even on NVMe. Realistic p50 latency for this path is **30–90 seconds per query**, and the overwhelming majority of that is weight-shuffling, not thinking.

That is not an analyst toolkit. That's a batch job.

**Fixes, roughly in order of impact:**

1. **One base model + hot-swapped LoRA adapters as the *primary* mechanism, not the exotic one.** The brief treats LoRA as an exotic "ephemeral sub-committee" feature. Invert it. Keep one 7–8B base resident permanently; make every specialist a LoRA adapter (tens to low hundreds of MB, swappable in milliseconds). You get genuine specialist diversity at near-zero swap cost. This alone is likely a 10× latency improvement.
2. **Shared-prefix KV cache reuse.** All specialists share a common prefix (the evidence payload). Cache it once, reuse across specialists. Note this does *not* violate isolation — isolation is a property of what content each node sees, not of which memory pages the KV tensor lives in. Big win, commonly missed.
3. **Shrink the Composer.** Once aggregation is mechanical (§11), the Composer only writes prose over a resolved claim set. An 8B model does that well. Dropping 24B → 8B may be the difference between a usable tool and an unusable one, and per §11 it likely *improves* auditability.
4. **The FOK gate short-circuits most queries entirely** (§7). Many queries never need the swarm at all.
5. **Move everything possible off the critical path.** Reconsolidation, grafting, interference sweeps, ToM calibration updates — all asynchronous. The only things allowed in the request path are: FOK → retrieve → (specialists) → compose.

**Other specific corrections:**

- **`Qwen2.5-Math-7B` for the Apperception Node is a category error.** It's a math-specialized model being asked to do entity extraction. Use a small general instruct model with **GBNF grammar-constrained decoding** in llama.cpp — that gives you *guaranteed* schema-valid JSON structurally, rather than hoping for it and retrying. This is a real llama.cpp capability and you should use it; it removes an entire class of parse-failure bug.
- **`sqlite-vec` is still pre-1.0.** As of early 2026 it's at 0.1.9-alpha, with ANN indexing (IVF, DiskANN) partly experimental. It's fine for tens of thousands of nodes with brute-force scan. Plan the abstraction layer so you can swap in an HNSW index or an alternative (SQLite's own vector extension work, `sqlite-vector`, or a standalone index) once you cross ~100k nodes. Don't let vector search calls leak throughout your codebase.
- **SQLite WAL with long-running background writers will contend.** WAL gives you concurrent readers with one writer — and you're proposing several background workers that all write. Route every write through a **single writer task** with a queue. Otherwise you'll spend a week debugging intermittent `database is locked`.
- **"ACID" is misused throughout the source document.** The transcript treats ACID as meaning "immutable and permanent." It doesn't — it's about transaction guarantees, and it's entirely compatible with decay, deletion, and rewriting. The criticism of Hexis on this point doesn't land, and more importantly, ACID is something you *want*. Don't design away from it by accident.
- **Destructive rewrite with no backup is a data-loss bug.** Append-only plus tombstones, always. This also makes `undo`, `what changed?`, and `restore` trivial instead of impossible.
- **`psutil` idle detection is a weak trigger.** Idle CPU doesn't mean the user is away; and heavy background inference on a laptop means fans, heat, and battery drain. Gate on explicit user consent, plug/battery state, thermal headroom, and a user-visible "the system is currently consolidating" indicator with a kill switch.

---

# Part VI — Privacy and consent

Not in the original brief at all, and it needs to be, because the Watch Officer as described monitors local folders and network feeds:

- **Explicit, per-directory consent scoping.** Opt-in per folder, never "monitor the drive."
- **A visible ingestion log.** "What have you read?" must be a first-class, one-click view.
- **Deletion must propagate.** A user deleting a source has to delete every derived, grafted, and dream node downstream of it. With append-only storage this is real work — tombstone the source, then walk the `derivation` edges and invalidate. Design for it now; retrofitting it is brutal.
- **Dream content is never exported.** Synthesized hypotheses stay internal and clearly marked until corroborated. They should never appear in a report the user hands to someone else without an explicit, deliberate promotion step.

---

# Part VII — On the name

`O.W.L.` works — short, memorable, ties to Athena, and owls do the two things the system does (watch in the dark, then act). Keep it.

"Ontological Workspace Lattice" is the weak part; it's three abstract nouns that don't tell a developer anything. If you want the expansion to earn its place, tie it to the actual architecture:

- **Observation & Wisdom Ledger** — accurate (append-only substrate + derived layer), and "Ledger" honestly signals the provenance-first design that is now the system's main differentiator.
- **Offline Working Lattice** — plain, precise, communicates the three things a developer needs to know in three words.
- **Orchestrated Worker Lattice** — from the original list, still the clearest description of the runtime.

I'd take **Observation & Wisdom Ledger**. Once provenance becomes the headline feature — and it should — "Ledger" is the most honest word in the whole naming set. It also gives you the one-line pitch the project currently lacks: *an offline analyst's memory that can always tell you how it knows.*

---

# What I'd actually build, in order

If you take nothing else, take the ordering. The brief's five pillars are roughly ranked by how *interesting* they are, which is nearly the inverse of how *load-bearing* they are.

**Phase 0 — Foundations (non-negotiable)**
Provenance schema. Append-only substrate + mutable index split. Single-writer queue. Evaluation harness *first*, before any of the clever parts.

**Phase 1 — Memory that works**
FSRS or ACT-R salience. FOK triage gate. De-interference sweep. Two embedding spaces (separation on write, completion on read). Self-Memory System hierarchy.

**Phase 2 — Memory that improves**
Reconstructive compression. Schema-delta storage. Surprise-gated encoding. Successor-representation prefetch. Reconsolidation windows.

**Phase 3 — The swarm**
LoRA-based specialists on one resident base. Mechanical evidence pooling. Calibration-scored reliability weights. Protected Adversary. Unshared-information premium.

**Phase 4 — Proactivity**
Prospective memory queue. Zeigarnik open loops. Consent-scoped ingestion with a visible log.

**Phase 5 — Dreaming, behind a flag**
Two-phase sleep on sleep pressure. Quarantined hypotheses with mandatory falsification tests. **Ship it off by default, run the ablation, and be genuinely willing to delete it.**

The system is valuable at the end of Phase 1 and excellent at the end of Phase 3. Phase 5 is the part the original brief is most excited about and the part most likely to be cut — which is exactly why it should be last and behind a flag.

---

## Sources

- [FSRS: the DSR (Difficulty/Stability/Retrievability) scheduler](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler)
- [FSRS algorithm overview](https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm)
- [Implementing FSRS in 100 lines](https://borretti.me/article/implementing-fsrs-in-100-lines)
- [Anki's scheduler documentation](https://faqs.ankiweb.net/what-spaced-repetition-algorithm)
- [sqlite-vec releases](https://github.com/asg017/sqlite-vec/releases)
- [The State of Vector Search in SQLite](https://marcobambini.substack.com/p/the-state-of-vector-search-in-sqlite)
- [Cai et al. 2009 — REM, not incubation, improves creativity by priming associative networks (PNAS)](https://www.pnas.org/doi/10.1073/pnas.0900271106)
- [Visser, Ashton & Vernon 2006 — Beyond g: Putting multiple intelligences theory to the test](https://www.sciencedirect.com/science/article/abs/pii/S0160289606000201)
- [Visser, Ashton & Vernon 2006 — g and the measurement of Multiple Intelligences: A response to Gardner](https://www.sciencedirect.com/science/article/abs/pii/S016028960600050X)

Key literature referenced (standard citations, verify before publishing): Anderson & Schooler 1991; Anderson, Bjork & Bjork 1994; Baddeley 1996; Bartlett 1932; Bjork & Bjork 1992; Borbély 1982; Cahill & McGaugh 1998; Cai et al. 2009; Camerer, Loewenstein & Weber 1989; Chase & Simon 1973; Conway & Pleydell-Pearce 2000; Cowan 2001; Craik & Lockhart 1972; Dayan 1993; Einstein & McDaniel 1990; Herzog & Hertwig 2009; Johnson, Hashtroudi & Lindsay 1993; Koriat 1993; Lorenz et al. 2011; McClelland, McNaughton & O'Reilly 1995; Nemeth 1986; Roediger & Karpicke 2006; Roediger & McDermott 1995; Slamecka & Graf 1978; Stachenfeld, Botvinick & Gershman 2017; Stasser & Titus 1985; Talarico & Rubin 2003; Tse et al. 2007; Tulving & Thomson 1973; Tversky & Kahneman 1974; Underwood 1957; Visser, Ashton & Vernon 2006; Wagner et al. 2004; Waterhouse 2006; Wixted & Ebbesen 1991; Zeigarnik 1927.
