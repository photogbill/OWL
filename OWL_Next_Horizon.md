# The Next Horizon — ideas nobody in the field is building

**Context:** written after surveying MemGPT/Letta, Mem0, Zep/Graphiti, A-MEM, HippoRAG, EM-LLM, GraphRAG, MemoryBank, RAPTOR, Titans, Generative Agents, and Hexis. Everything below is absent from all of them as far as I can tell. Ranked by how much I think it matters, not by how easy it is.

First, the observation that generates most of the list:

> **Every memory system in the field models what the *machine* knows. None models what the *person* knows.**

That's the asymmetry. These systems are built as if the human were a stable oracle issuing queries, when the human is the component with the *interesting* memory dynamics — they forget on a curve, they misremember with confidence, they hold stale beliefs, and they don't know which of those is happening. A memory system that models only itself is solving the easier half of the problem.

Six of the nine ideas below fall out of taking that seriously.

---

# Part I — The memory horizon

## 1. Transactive memory — model the user's forgetting curve, not just your own

**The idea.** Wegner's transactive memory system describes what happens in long-married couples and effective teams: each party maintains a model of *what the other knows*, and — crucially — a directory of *who is responsible for remembering what*. The pair remembers more than the sum of its members, not because either has a better memory, but because each can offload.

No AI memory system implements this. They all implement *storage*. Transactive memory is *division of labour*.

**What it unlocks.** You already have a validated forgetting model (FSRS). Run it on the human.

The aid worker read the security brief on day one. FSRS says a single unrehearsed exposure to moderately difficult material has retrievability around 0.4 at three weeks. So on day 21, before he drives to the checkpoint, the system says — unprompted, in one line — *"checkpoint protocol: you were briefed on the 3rd, you haven't referenced it since."* Not a quiz. Not a notification. A colleague's nudge, timed by a model of his decay rather than by a calendar.

This inverts spaced repetition. SRS asks the human to review on a schedule. Transactive memory just *carries the load* and surfaces it at the moment of predicted failure.

**Implementation in OWL.** You already have every piece:

```python
mind.told(user="bill", node_id=obs, channel="briefing")   # exposure event
mind.user_retrievability("bill", obs)                     # FSRS on the human
mind.at_risk("bill", threshold=0.4)                       # what's about to go
```

Exposure events are just review events on a per-user index row. The `told()` call is the only new primitive. Everything else is the salience module you already have, pointed at a different subject.

**Why it's the top of the list.** It reframes the product. OWL stops being *a memory the system has* and becomes *a memory the pair has* — which is what an actual colleague is. And for ATK's stated mission (a partner for someone exhausted who can't think straight), the thing that fails first under exhaustion is exactly retrieval of briefed-once material.

---

## 2. Epistemic half-life — confidence should decay on its own curve

**The idea.** Every system that models forgetting decays *retrievability*: can I still find this? Nobody decays *confidence*: should I still believe this?

These are different, and conflating them is dangerous in a field context. "Route Alpha is open" is perfectly retrievable six months later and almost worthless. "The clinic has 12 beds" is retrievable and probably still true. "Dr Warsame speaks Somali" is true forever.

**The mechanism.** Claims have a **half-life by class**, and the class is learnable:

| Class | Half-life | Example |
|---|---|---|
| Identity / structural | ∞ | who someone is, what a building is |
| Capacity | months | bed count, generator wattage |
| Status | days | route open, stock level, who's on duty |
| Position | hours | where a convoy is |

You don't need to hand-label these. **Corrections are the training signal.** Every time a memory is superseded, you get a labelled observation: this class of claim survived N days before it changed. Fit a survival curve per class. Now the system can say *"I hold this, but it's a status claim from eleven days ago — treat as stale"* without anyone configuring anything.

**Why nobody does it.** Because everyone treats memory as retrieval, and staleness as the caller's problem. In an analyst toolkit it is emphatically not the caller's problem.

**Implementation.** A `claim_class` column, a `superseded_at` timestamp you already have from the bitemporal model, and a per-class Kaplan-Meier fit over supersession events. Then `Chunk.staleness` sits next to `Chunk.retrievability`. Maybe forty lines.

**This one I'd build next.** It's small, it's novel, and for ATK it's the difference between a memory and a liability.

---

## 3. Handover — memory transplant with automatic epistemic demotion

**The idea.** ATK's mission statement is that someone lands where they know nobody and needs to help fast. The single highest-value thing you could hand them is *the memory of the person who was there before*.

Not the files. The ledger — with provenance intact, with the previous operator's open questions, with their confidence, and with their unresolved conflicts.

**Why it's only possible here.** Everyone else's memory is unsafe to transplant, because there's no way to distinguish what the previous operator *observed* from what they *concluded*. Import a Mem0 or A-MEM store and you inherit their inferences as facts.

OWL's monotonicity lattice handles this for free, and elegantly:

> On import, everything the source operator marked `observed` becomes `reported`.
> Everything they marked `inferred` becomes `hypothesized`.
> Everything `hypothesized` is dropped or quarantined.

The epistemic tag shifts by one rank, transitively, and the machinery you already built does the rest. Their certainties become your reports; their conclusions become your hypotheses. That is *exactly right* — it's how a careful analyst treats a predecessor's handover notes, and it happens automatically.

```python
mind.graft(Path("bardera.owlpack"), demote=1, as_source="prev-operator:Ferrand")
```

**What it enables beyond the obvious.** Multi-operator convergence: three people working the same area, each with their own ledger, periodically exchanging packs. Claims corroborated by independent operators get promoted; claims only one person holds stay single-source and are marked as such. That's a distributed intelligence picture built out of nothing but the provenance rules already in the code.

---

## 4. Negative memory — record what *isn't* there

**The idea.** Every memory system stores what was said. None stores:

- **Failed searches.** "I looked for a fuel supplier in Bardera and there wasn't one." Currently this costs a full search every time it's asked, forever.
- **Considered and rejected.** "We evaluated routing via Km 42 and ruled it out because the bridge is out." Without this, the same option gets re-proposed and re-rejected weekly.
- **Explicit absence.** "The manifest has no cold-chain entry" is different from "I don't have the manifest."

**Why it matters more than it sounds.** Absence is expensive knowledge. It costs a full search to establish and nothing to store, and it's the knowledge most likely to be re-derived over and over. It's also the thing that distinguishes a colleague from a search box: a colleague says *"already looked, it's not there."*

And it's the correct fix for a specific failure mode — a system with no negative memory will cheerfully re-suggest the thing you rejected last Tuesday, which reads as not listening.

**Implementation.** A `kind='absence'` derived node whose content is the *query* rather than an answer, with a scope and a timestamp. Then the FOK gate gains a fifth outcome: `SEARCHED_AND_ABSENT` — "I don't have this, and I know I don't, because I looked on the 14th." That's a far better answer than `DONT_KNOW` and it costs almost nothing.

---

## 5. `KNEW_ONCE` — the honest fifth state

**The idea.** OWL currently distinguishes KNOW / KNOW_WHERE / TIP_OF_TONGUE / DONT_KNOW. There's a fifth state that no system reports and that humans experience constantly:

> **"You told me something about this. I no longer hold the detail. Here's where it came from."**

When a memory has been compressed or pruned from the index but the cue and provenance survive, that is not `DONT_KNOW`. `DONT_KNOW` means *never told*. This means *told, and lost* — which is a completely different instruction to the user: go look at the source.

**Why it's worth a whole state.** It is the single most honest thing a memory system can say, and every existing system is structurally incapable of saying it, because they delete rows. OWL keeps the substrate and decays the index, so the cue is always still there. The capability falls out of the architecture; it just needs to be surfaced.

This is also the correct answer to the "did you forget?" question users ask when they're losing trust. "No — you told me on the 3rd, from `fieldnotes/day3`, and I compressed the detail. Want me to pull it back?" rebuilds trust in a way that "I don't know" destroys it.

---

## 6. Time-travel — reconstruct the past mind and audit it

**The idea.** Because the substrate is append-only and the index is derived, you can rebuild the *entire state of the mind* as of any past moment and ask it questions.

Not "what did I record on March 3rd" — bitemporal query already does that. **"What would I have *answered* on March 3rd, and why was it wrong?"**

**What it enables.** Post-incident review of the system itself. A convoy got routed badly; you replay the mind as of the decision point and see exactly which memories were hot, which were stale, what the FOK state was, and where the reasoning went wrong. That's a black-box recorder for a cognitive system, and it's the only honest way to improve one.

It also generates training signal: every case where the past mind was confidently wrong is a labelled calibration example. Feed those to the per-class half-life fit in idea 2 and the two mechanisms compound.

**Why nobody has it.** Everyone mutates in place. Once you overwrite a memory, the past mind is unrecoverable. OWL's supersede-never-overwrite discipline makes it a query.

---

# Part II — The interaction horizon

## 7. Retrieval should return a *shape*, not a list

**The idea.** Every system returns top-k chunks. But the useful answer to *"what do we know about the water situation?"* is not five paragraphs. It's a structure:

```
CONSENSUS   3 independent sources: north well contaminated       [confident]
DISSENT     1 source (Ahmed, 14th) says the pump was repaired    [unresolved]
STALE       chlorine stock figure is 23 days old, status-class   [check]
GAP         nothing on the south borehole since intake           [absent]
PROVENANCE  2 observed, 1 reported, 1 inferred
```

That is what an analyst actually needs, and it's a *rendering* of the claim lattice OWL already builds — agreement, conflict, staleness, absence, and provenance mix are all already in the data model. Nobody surfaces them because nobody tracks them.

**Why this is an interaction change, not a formatting change.** A chunk list invites the model to blend everything into confident prose. A shape forces the disagreement and the gaps to survive into the answer. It changes what the downstream LLM *can* say.

---

## 8. Anticipatory retrieval — memory that raises its hand

**The idea.** Retrieval is currently pull-only: the user asks, memory answers. The next step is memory that **interrupts** — a cheap model watching the live conversation, matching against open loops and past outcomes, with a high bar for speaking.

> *"You're about to route via Km 42. On the 9th you ruled that out — bridge."*

This is the Watch Officer applied to the live turn rather than to folders, and it's the difference between a reference work and a colleague. A colleague's most valuable contribution is usually unrequested.

**The hard part is not retrieval, it's restraint.** A memory that interrupts often is unusable. The gate has to be brutal: interrupt only when there's a *specific* past decision contradicting a *specific* current one, and never twice for the same thing. Budget it explicitly — Hexis's energy model is the right primitive here. Give interruption a hard daily allowance and let it spend down.

I'd build this last and behind a flag, and I'd measure "interruptions the user acted on" against "interruptions dismissed" as a live quality metric. If the ratio isn't strongly positive, ship it off.

---

## 9. Memory as a jointly-edited artifact

**The idea.** In every current system, memory is something done *to* the user. It's opaque; they discover what it holds only through its behaviour, and they can't correct it except by arguing with a chatbot.

Make the ledger a surface the person can open, read, and edit — and make **their edit a first-class provenance event**:

```
origin: user_correction
source_ref: ui://ledger/2026-07-30
supersedes: der_9f2a...
```

Now "the system was wrong and I fixed it" is part of the record rather than a fight with a model. This is Hutchins' distributed cognition taken literally: the memory belongs to the pair, and both parties can write.

**The second-order effect is the interesting one.** Once the user can see what the system holds, they start correcting it proactively — which massively improves the store, for free, without any clever automation. Making memory legible is a cheaper path to accuracy than making extraction smarter.

ATK is uniquely positioned for this: it already has a graph canvas, a project browser, and profile cards. The ledger view is mostly an interface you've already built.

---

## 10. Forgetting on request — which is not deletion

**The idea.** *"Stop bringing that up."*

There is no system in the field where a person can ask to have something let go without deleting it. But those are different requests, and for a companion the distinction is the whole thing. The person doesn't want the record destroyed; they want it to stop surfacing unbidden.

**Implementation.** A suppression with a reason and a review date — retrieval-strength floor removed, ranking demoted hard, excluded from proactive surfacing entirely, but still addressable on explicit request. Reversible. Logged.

One caution from the psychology: implement suppression as **ranking demotion, never as an exclusion filter checked at retrieval time**. A system that has to ask "is this the suppressed item?" has already retrieved it — and Wegner's ironic process work is a good reminder that active monitoring for a thing keeps it active.

---

## 11. Source reliability grading — borrowed from tradecraft, absent from AI

**The idea.** ATK is an analyst toolkit, and intelligence practice already solved this: the Admiralty scale grades **source reliability** (A–F: reliable → cannot be judged) separately from **information credibility** (1–6: confirmed → cannot be judged). Two axes, because a reliable source can report something implausible and an unreliable source can be right.

No LLM memory system carries this, and OWL's provenance lattice is exactly the right place for it. It propagates the same way epistemic tags do — a derivation is no more reliable than its weakest source — and it gives the analyst a vocabulary they already know.

It also gives you a principled way to handle the hardest real case: the same claim from two sources of different quality. Currently every system either dedupes (losing the corroboration signal) or keeps both (losing the quality distinction). With two axes you merge the claim and *raise its credibility* because two independent sources concur — which is the actual epistemics.

---

# What I don't think will work

Worth saying, since enthusiasm is cheap:

- **Fine-tuning on self-generated memory.** Model collapse is real, the interleaving requirement is expensive, and LoRA-per-domain on frozen weights gets ~most of the benefit reversibly. I'd keep Athena's sleep-cycle PEFT as a research track and off by default in field builds, which is what your notes already say.
- **Letting the LLM decide what to forget** (the MemGPT approach). Non-deterministic, expensive, and untestable. Forgetting should be a fitted model you can plot.
- **Bigger context as a memory strategy.** Lost-in-the-middle degradation is well documented, and a 4–7 chunk budget with denser chunks beats stuffing. This is one of the few places human and LLM constraints genuinely converge.
- **Emotion labels as salience.** Prediction error is the better substrate — it's measurable, it's forward-looking, and it doesn't require a model to introspect about feelings it doesn't have.

---

# If I had to pick three

**Epistemic half-life** (#2) — smallest, most novel, and for ATK the difference between a memory and a liability. Build it next.

**Transactive memory** (#1) — the biggest conceptual leap, and it reframes what the product *is*. Also the one most likely to make someone say "oh, that's different."

**Handover** (#3) — because it's the one that serves the actual mission, and because OWL is the only architecture I've seen where it's *safe*. Everyone else's memory transplant imports someone else's guesses as your facts.

The common thread across all three, and most of the list: **the interesting frontier isn't storing more, it's modelling the epistemics** — whose knowledge it is, how sure anyone should be, how long that stays true, and what happens to all of it when it moves between minds.
