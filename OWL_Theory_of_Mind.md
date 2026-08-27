# Should O.W.L. include Theory of Mind?

**Short answer: yes — but only one of its five planes, and the restriction is the whole design.**

Built and running as of this pass. 40 tests green, ~2 s, no GPU.

---

## 1. The decomposition that makes this answerable

"Theory of Mind" gets used as one thing. It isn't. It decomposes into capacities that are separately implementable and separately risky:

| Plane | Question | In OWL? |
|---|---|---|
| **Epistemic** | What does the other party know, not know, believe falsely? | **Yes — fully** |
| **Intentional** | What are they trying to do? | Weakly, via successor representation |
| **Attentional** | What are they attending to right now? | No — host's job |
| **Affective** | What are they feeling? | **Deliberately no** |
| **Recursive** | What do they think I think? | Capped at depth 2, host's job |

The epistemic plane belongs in a memory engine because **it is a memory question**. What has this person been told? How many times? How long ago? Through what channel? Do they still hold it? That's the same machinery as the rest of OWL, pointed at a different subject.

The affective plane does not belong in a memory engine, and I'd argue it doesn't belong in Athena's memory layer either. More on that in §5.

---

## 2. The insight: transactive memory *is* Theory of Mind

Idea #1 on the horizon list turned out not to be adjacent to ToM. It **is** ToM — the rigorous, operationalised, testable version.

Wegner's **transactive memory system** describes what long-married couples and effective teams do: each party maintains a model of what the *other* knows, plus a directory of who is responsible for remembering what. The pair remembers more than the sum of its members. That is a theory of mind with the hand-waving removed — no introspection, no affect modelling, just a tracked belief state and a division of labour.

So the answer to "should we include ToM?" is: **you already decided to, when you accepted transactive memory.** What remained was to build it properly and be explicit about where the boundary is.

---

## 3. What's built

### 3.1 The exposure log — the one new primitive

```python
mind.tell("bill", node_id, channel="briefing")
```

Everything else derives from this plus the forgetting model you already had.

Channels carry a **depth weight**, because Craik & Lockhart's levels-of-processing finding is that depth of encoding predicts retention far better than exposure count does:

| Channel | Depth | Rationale |
|---|---|---|
| `briefing` | 0.5 | read once, passively |
| `conversation` | 1.0 | participated |
| `recall` | 1.3 | they retrieved it themselves — testing effect |
| `generated` | 1.5 | they said it — generation effect (Slamecka & Graf 1978) |
| `correction` | 1.6 | they corrected the system — maximum depth |

Measured output, same three facts, 21 days later:

```
checkpoint protocol (skimmed)    retention=0.27   <- AT RISK
route status (discussed)         retention=0.63
Dr Warsame (he said it)          retention=0.87
```

**The system still holds all three perfectly.** That gap is the entire point: the memory dynamics worth modelling are the human's, not the machine's.

### 3.2 `at_risk()` — spaced repetition, inverted

SRS asks the human to review on a schedule. Transactive memory carries the load and surfaces the item at the moment of predicted failure — the way a colleague does.

Ranked by **consequence**, not just by decay: a still-true claim being lost matters more than a stale one being lost. The stale one *should* go.

### 3.3 `divergence()` — Sally-Anne, made operational

The classic false-belief test, as a computable state:

- The ledger knows Route Alpha closed on the 14th.
- The exposure log knows Bill was told it was open on the 12th.
- The exposure log knows he has **not** seen the update.
- Therefore the system can *compute* that he is about to act on a false belief.

```
[user_stale] severity=0.62
  he holds : Route Alpha is open.
  record   : Route Alpha is closed by flooding.
  the record is more recent and better sourced
```

No model call. It falls out of the exposure log crossed with the bitemporal record — two things that already existed.

Severity is `user_retention × claim_staleness`, which is right: **a belief he has already forgotten isn't dangerous, because he'll ask.** The dangerous one is the belief he's sure of and that's wrong.

### 3.4 The correction that matters most: divergence is symmetric

This is the part I'd flag hardest.

Sally-Anne is conventionally framed as *"the other party holds the false belief."* A system built on that framing will confidently correct someone who was standing at the checkpoint an hour ago. In the field, the human frequently holds better information than the ledger — **they were there**, and first-hand observation outranks a three-day-old document.

So `resolve_direction()` resolves on provenance quality and recency, and can return `ledger_stale`:

```python
resolve_direction(user_source_recency=1*HOUR, user_was_present=True,
                  ledger_recency=3*DAY, ledger_admiralty=0.85)
# -> ('ledger_stale', 'the person has more recent first-hand observation
#     than the record; treat the record as the stale side')
```

A memory system that always assumes it's right is worse than one with no ToM at all, because it converts a disagreement into a confident correction.

### 3.5 The sycophancy inversion

Worth stating plainly, because it's the same computation used two opposite ways.

The v1 ATHENA brief proposed, before delivering a response:

> *"Based on the user's current cognitive state, will this explanation cause confusion? Does it contradict a deeply held premise?"* — and then **adjust**.

Read as an objective function, that optimises for user comfort and non-contradiction of existing belief. That's sycophancy, and in an analyst tool the most valuable output is frequently the one that contradicts a deeply held premise.

OWL runs the same detection and inverts the sign: **find the contradiction in order to surface it.** `divergence()` exists to tell you your operator is out of date, not to phrase around it.

And the architectural rule that keeps it that way:

> **OWL computes what the person knows. It never decides how to say anything.**

Presentation belongs to the host, behind a firewall, with the claim set diffed before and after the presentation layer touches it. If the sets differ, reject the revision. That's a test, not a guideline.

---

## 4. What ToM unlocked that wasn't on the list

Two things emerged from building it that I hadn't anticipated.

**"Did you tell me?" becomes answerable.** No current system can answer this. With an exposure log it's a query, and it's a surprisingly load-bearing interaction — it's what people ask when they're losing trust in a shared memory.

**Corrections become the highest-depth exposure.** When someone corrects the system, that's simultaneously (a) a new observation, (b) a maximum-depth exposure event, and (c) a labelled calibration datapoint. One user action feeding three subsystems is a good sign the abstraction is carved right.

---

## 5. Where I'd stop — and this one is a judgement call

**Don't give Athena a persistent affective model of the user.**

The temptation is obvious: she's a companion for someone in distress, so surely she should model their emotional state? I'd argue no, and specifically not in the memory layer:

- **A persistent psychological profile of a person in crisis is a serious artifact to hold**, on a laptop, in the field, with no clinical oversight. The confidentiality partition protects it from the work arena; it doesn't make it wise to build.
- **It's the shortest path to sycophancy** in exactly the context where warmth must not become agreement.
- **It isn't needed.** The `affect` marker on observations already does the useful work — it marks *the content* as distressing, which is what drives `suppress_affect_above` and stops raw traumatic detail surfacing unbidden. That's a property of the material, not a psychological model of the person.

Your own notes get this right — *attunement without amplification*, reflective listening that validates without spiralling. That's a **conversational** discipline, and it belongs in the persona prompt and the constitutional layer, not in a database row that accumulates a theory of someone's psyche over months.

The line I'd hold: **OWL models what the person KNOWS. It does not model who the person IS.**

---

## 6. Bugs this pass caught (both worth recording)

**The append-only trigger caught me.** I put `suppressed_at` / `suppress_reason` on the `observation` table. The trigger rejected the write — correctly. Suppression is a *forgetting* operation, and forgetting lives in the index, never the record. Exactly the layering mistake the invariant exists to prevent, and it caught its author within an hour of the columns being added. That's the argument for enforcing invariants in the schema rather than in code review.

**Recorded absence must outrank a weak lexical match.** A test asserted `DONT_KNOW` for "diesel supplier in Bardera" and got `KNOW_WHERE`, because an unrelated note shared the word "Bardera." The test premise was wrong, but the finding was real: if absence only fires on `DONT_KNOW`, then expensive knowledge ("I canvassed all six vendors") gets buried by lexical noise and re-derived forever. Absence now outranks any non-`KNOW` state, and attaches the weak matches as context — *"I looked and there's none; here's the related material I do have."*

---

## 7. Current state

```
owl-engine/                      40 tests, ~2 s, no GPU, zero dependencies
├── owl/
│   ├── theory_of_mind.py    NEW  exposure log, transactive retention,
│   │                             symmetric belief-divergence resolution
│   ├── epistemics.py        NEW  claim classes, learned half-life,
│   │                             Admiralty grading, corroboration
│   ├── metamemory.py             six FOK states (was four)
│   ├── salience.py               FSRS — now also runs on the human
│   ├── provenance.py             monotonicity invariant
│   ├── segmentation.py           surprise-boundary episodes
│   └── store/sqlite.py           single writer + flow-control lattice
└── examples/01_theory_of_mind.py    runnable demo of all of the above
```

**Horizon items now built:** transactive memory (#1), epistemic half-life (#2), negative memory (#4), `KNEW_ONCE` (#5), forgetting-on-request (#10), Admiralty grading (#11), plus false-belief detection as the ToM addition.

**Still design-only:** handover/`.owlpack` (#3), time-travel replay (#6), retrieval-as-shape (#7), anticipatory interruption (#8), jointly-edited ledger (#9).

Of those, **handover** is the one I'd build next — it serves the actual mission, and the epistemic demotion rule (`observed → reported`, `inferred → hypothesized`) now has the whole ToM layer behind it, so an imported memory can carry *the previous operator's* exposure history too. You'd inherit not just what they knew, but what they'd been told and when — which is most of a proper handover briefing, generated.
