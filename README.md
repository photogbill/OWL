<div align="center">

# 🦉 O.W.L.

### Observation &amp; Wisdom Ledger

**A memory engine for LLM agents that can always tell you how it knows.**

[![tests](https://github.com/photogbill/OWL/actions/workflows/test.yml/badge.svg)](https://github.com/photogbill/OWL/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml)
[![tests count](https://img.shields.io/badge/tests-465-brightgreen)](tests/)
[![license](https://img.shields.io/badge/license-MIT-informational)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#status)

**[Why](#why-another-memory-library)** ·
**[Quick start](#quick-start)** ·
**[Install](#install)** ·
**[Ideas](#the-ideas-and-where-they-come-from)** ·
**[Comparison](#how-it-compares)** ·
**[Status](#status)**

</div>

---

```bash
git clone https://github.com/photogbill/OWL.git
cd OWL
install.bat          # Windows.  Linux/macOS: see Install
```

**No Docker. No Postgres. No GPU. No model. No API key. No dependencies.**
A store is one file.

---

### The part nobody else does

```python
route  = mind.observe("Route Alpha is open.", source_ref="sitrep-0800")
convoy = mind.decided("Route the fuel convoy via Alpha", because=[route])

# ...four hours later, it isn't.
mind.observe("Route Alpha is closed by flooding.", source_ref="sitrep-1200",
             supersedes=route)

mind.reconsider()
# [URGENT] sev=0.71  Route the fuel convoy via Alpha
#          cause=superseded   still reversible -- act on this
```

Every memory system can answer *"what do I know about Route Alpha?"*
OWL answers the question that costs money when nobody asks it:

> **"That turned out to be wrong. What did I do about it, and what do I need to undo?"**

Nothing is deleted to make that work. The old row is still there, still
retrievable, still marked — which is also why OWL cannot lose the superseded
wording the way a store that mutates in place does.

---

Most agent memory systems can tell you *what* they remember. Very few can tell you **where it came from, how sure to be, whether it's still true, or whether they made it up.**

That gap matters more than it sounds. Every one of these systems writes model-generated content — summaries, inferences, reflections, "insights" — into the same store as source material, and retrieves it identically later. Given enough cycles, a system will assert something it invented with the same confidence as something you told it, and neither of you will be able to tell.

OWL is built around one commitment:

> **Evidence is immutable. Forgetting happens in the index, never the record. And speculation can never become fact.**

---

## Install

**OWL is not on PyPI.** There is no `pip install owl-engine` — it is installed
from a clone of this repository. Python 3.10 or newer; nothing else is
required for Tier 0.

### Windows

```bat
git clone https://github.com/photogbill/OWL.git
cd OWL
install.bat
```

`install.bat` creates an isolated `.venv`, installs OWL editable, detects your
hardware and fetches the matching prebuilt `llama-cpp-python` wheel if there is
one, then runs the test suite and refuses to report success if it fails.

### Linux / macOS

```bash
git clone https://github.com/photogbill/OWL.git
cd OWL
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests -q
```

### Updating

```bash
git pull
```

Then re-run `install.bat` (Windows) or `pip install -e ".[dev]"` (Linux/macOS).
The install is editable, so a pull is usually enough on its own — re-running
the installer matters when dependencies or the schema have moved.

**Your stores are safe across an update.** A `.owl` file written by an older
version is migrated in place the first time it is opened, and the migration is
reported by `python -m owl doctor mind.owl` rather than done silently. A store
on read-only media that cannot be migrated stays readable.

### The scripts

| Script | What it does |
| --- | --- |
| `install.bat` | venv + editable install + backend detection + verify |
| `run_tests.bat` | the correctness suite (extra pytest args pass through) |
| `demo.bat` | every example, in order |
| `bench.bat` | the benchmark harnesses |
| `validate.bat "path\model.gguf"` | validate a real embedder |
| `shell.bat` | a prompt with the venv active |
| `clean.bat` | remove `.venv` and caches |

All batch files are safe to double-click — they detect interactive execution and hold the window open.

> The installer **never builds `llama-cpp-python` from source** (that needs CMake + VS Build Tools and usually fails). It tries prebuilt wheels for your CUDA version, then CPU, then plain PyPI. If none match it says so and continues — **Tier 0 is complete without it.** Python 3.10–3.12 is the safe range; 3.13+ often has no wheel yet.

**Embedding models are not in the repository.** `*.gguf` is gitignored — they
are gigabytes. Tier 0 needs none of them. For Tier 1, put a `.gguf` in an
`embedding model/` folder and run `validate.bat "embedding model\your-model.gguf"`.

---

## Quick start

```python
from owl import Owl, State

mind = Owl.open("./mind.owl")          # a file. that's the whole setup.

mind.observe("The clinic generator runs on depot fuel.",
             origin="document", source_ref="file://survey.pdf#p3")

r = mind.recall("how is the clinic powered?")

match r.state:
    case State.KNOW:                 ...   # direct match
    case State.KNOW_WHERE:           ...   # neighbourhood match, and I can place it
    case State.FAMILIAR:             ...   # seen it, can't place it — nothing anchors it
    case State.TIP_OF_TONGUE:        ...   # familiar but contested — widen search
    case State.KNEW_ONCE:            ...   # you told me; I lost the detail; here's the source
    case State.SEARCHED_AND_ABSENT:  ...   # I looked on the 14th. It isn't there.
    case State.DONT_KNOW:            ...   # nothing. returned in ~0.1 ms, no model call.

for c in r.chunks:
    print(c.content)
    print(c.provenance.source_ref, c.retrievability, c.staleness)

```

**Check `.state` before you touch `.chunks`.** Six states, not two. It's the whole design in one line — most memory libraries make "I have nothing" indistinguishable from "here are five bad matches."

The entire API is three verbs — `observe`, `recall`, `tend` — plus `derive`, `why`, `partition`, `period`, `tell`, `intend`, `doctor`, and the handover pair.

---

## Why another memory library

### 1. Provenance is an enforced invariant, not a metadata field

Two rules, checked on every write and fuzz-tested in CI:

```
confidence(node)  <=  min(confidence(parents))
epistemic(node)   >=  max(epistemic(parents))

```

A node derived from a hypothesis is **permanently** a hypothesis. Abstraction cannot launder speculation into fact. There is no argument that gets you around it, including from the library's own author — the append-only trigger caught me putting suppression state on the evidence table during development.

```python
for node in mind.why(some_conclusion):
    print(node["epistemic"], node["presentable_as_fact"], node["source_ref"])

# *NOT FACT* [hypothesized] conf=0.70  The clinic can sustain two weeks...
# *NOT FACT* [inferred    ] conf=0.90  Fuel resupply is currently viable.
# FACT       [observed    ] conf=1.00  Route Alpha is open as of this morning.

```

### 2. Findable and still-true are different questions

Everyone decays *retrievability* — can I still find this? Nobody decays *credibility* — should I still believe it?

"Route Alpha is open" stays perfectly findable for six months and becomes worthless. "Dr Warsame speaks Somali" is true forever. OWL classifies claims (`identity` / `capacity` / `status` / `position`) and **learns the half-life of each class from real supersession events** — nobody configures this.

```
TRUST  identity   findable=0.63  stale=0.00   Dr Warsame runs the clinic...
STALE  status     findable=0.63  stale=0.99   Route Alpha is open.

```

### 3. It models what *you* know, not just what it knows

Every memory system in the field models the machine's knowledge. The human is the component with the interesting memory dynamics — they forget on a curve, they hold stale beliefs, and they don't know which is happening.

OWL runs the same forgetting model on the person, weighted by depth of encoding:

```python
mind.tell("bill", node_id, channel="briefing")   # they were exposed
mind.knows("bill", node_id).retrievability       # do they still hold it?
mind.at_risk("bill")                             # what they're about to lose
mind.divergence("bill")                          # what they believe that's now false

```

```
day 21  checkpoint protocol (skimmed)    retention=0.27   <- AT RISK
day 21  route status (discussed)         retention=0.63
day 21  Dr Warsame (they said it)        retention=0.87

```

The system still holds all three perfectly. That gap is the point.

**Divergence resolves symmetrically.** A system that assumes the record always wins will confidently correct someone who was standing at the checkpoint an hour ago. First-hand observation outranks a three-day-old document, and OWL can flag *itself* as the stale side.

### 4. Confidentiality is a property of the store

Information flow is **denied by default, directional, and non-transitive**:

```python
mind.partition("work")
mind.partition("private", sealed=True, summary_reads_from=["work"])

```

A one-way membrane. `private` sees what the day held — at the level of summaries, not raw detail — and nothing in it can ever surface in `work`, or be exported, ever. Enforced in SQL, so a future refactor can't quietly break it.

**And the boundary holds for timing, not just content.** A partition is a storage shard: `lexeme`, `vector` and `mem_index` are all indexed partition-first, so a query inside `private` reads `private`'s rows and stops. Measured — a recall in a two-memory private partition, beside a work partition that grows:

```
work partition     25      100      400     1600     4000  memories
rows read           4        4        4        4        4   partition-sharded
                   29      104      404     1604     4004   scoped after the scan
latency          2.87     1.50     1.69     2.33     2.15   ms, sharded
                 2.98     1.48     1.98     4.61     7.94   ms, unsharded

```

This is a confidentiality property before it is a performance one. A private partition's entire justification is being separate; making its response time a function of the work partition's size publishes the work partition's size into it. A boundary that holds for content and leaks through timing is a boundary with a side channel.

### 5. Handover: inherit someone else's ledger, safely

Import a Mem0 or A-MEM store and you inherit the previous operator's inferences as your facts, because nothing in those formats distinguishes what they *saw* from what they *concluded*.

OWL's monotonicity lattice makes the transplant rule one line — **every epistemic tag shifts down one rank**:

| Theirs | Becomes yours |
| --- | --- |
| `observed` | `reported` — you didn't see it; they did |
| `inferred` | `hypothesized` — their conclusion is your guess |
| `hypothesized` | *dropped* — their guess is nothing to you |

You also inherit their **failed searches** (no re-canvassing six vendors), their **open loops**, and their **exposure history** — not just what they knew, but what they'd been told and when.

```python
mind.inspect_pack("bardera.owlpack")               # dry run first
mind.graft("bardera.owlpack", as_source="prev:ferrand")

```

Packs are plain JSON, checksummed, and refuse to load if modified. Sealed partitions never export. Suppressed and affect-marked material never travels.

### 6. It knows what it is holding up

Every memory system in the field points *backward*: `why()` answers "how do I know this?" **Nothing points forward.**

An analyst reads "Route Alpha is open," routes a convoy, moves on. Three days later a sitrep supersedes it. Every other system updates the fact and stops. Nobody learns the convoy decision is standing on a false premise.

```python
route  = mind.observe("Route Alpha is open.", source_ref="sitrep-1")
convoy = mind.decided("Route the fuel convoy via Alpha", because=[route],
                      reversible_until=tomorrow)

mind.observe("Route Alpha is closed by flooding.", source_ref="sitrep-2",
             supersedes=route)

mind.reconsider()
# [URGENT] sev=0.71  Route the fuel convoy via Alpha
#          cause=superseded   still reversible -- act on this

```

Executed decisions surface too, but never as urgent — *"the convoy already crossed the bridge that has since collapsed"* is what after-action review needs. Log it; don't alarm about something nobody can change.

And the inverse of `why()`:

```python
mind.discredit(survey_pdf, reason="three years out of date", reliability="E")
# demotes every conclusion downstream, flags every decision that rested on it,
# and tells you who you told — without deleting anything

```

`blast_radius()` answers the question no other memory system can:

> *"I just learned that PDF was out of date. What did I conclude from it, what did I tell the team, and which decisions rested on it?"*

**Verification triage** falls out of the same graph — which beliefs carry the most weight, ranked by how weakly attested they are:

```
priority=0.960 crit=1.00 deps=2 decisions=1 grade=E/5   Depot holds 4000 litres...
priority=0.473 crit=0.82 deps=1 decisions=1 grade=C/3   Route Alpha is open.

```

### 7. Trust is learned, and it flows backwards

Corroboration everywhere else counts **documents**. It should count **independent origins** — forty files from one upstream source are one source, and treating them as forty *is* the source-flooding attack.

```
20 documents asserting the same thing
   independent origins : 1
   corroboration weight: 0.0    single origin -- no corroboration credit

```

And the claim is separate from the person making it — the *de dicto / de re* distinction that intelligence tradecraft has treated as basic for a century, and that no LLM memory system implements:

```python
mind.claimed("Ahmed", "the depot restocks every Tuesday", node_id=n)
mind.who_claims("the depot restocks every Tuesday")   # who else says this?
mind.record_of("Ahmed")                               # learned from outcomes

```

**Promises are not facts.** They have a lifecycle — made → due → kept or broken — and the outcome closes a loop nobody closes:

```python
cm = mind.committed("Ahmed", "deliver fuel", due=thursday, node_id=n)
mind.resolve_commitment(cm, kept=False)
# Ahmed: 17% accurate over 4 resolved (4 promises broken) -> grade E
# ...and every source he spoke through is revalued automatically: B/2 -> E/5

```

An unrated source is graded `F` — *"cannot be judged"*, which is mid-scale, not low. Treating an unknown source as unreliable is as wrong as trusting it. And a perfect record has to be *long* before it earns grade A: five kept promises is a good sign, not a certification.

### 8. Poisoning is a different threat from a mistake

A memory system is a persistence layer for beliefs. Prompt injection is transient; **memory poisoning survives every restart**, propagates into every derived summary, and is retrieved as context forever.

* Injected content is **quarantined, not refused** — the attempt itself is evidence. It stays retrievable and is never authoritative, never corroborates, never fuses.
* A grade-F source **cannot overwrite** a grade-B one. It registers a *conflict* instead, because disagreement is information.
* One source rapidly superseding many established claims is a **belief coup** — held for review.
* **Model provenance** on every inference: when you upgrade from a 7B to a 24B, every conclusion resting on the smaller model's judgement is identifiable.
* **`self_audit()`** attacks OWL's own invariants in the live store. CI proves they hold at commit; this proves they still hold after months of writes.

### 9. Quantities are values, not substrings

"400L" and "400 gallons" are not the same number. A summariser that drops a
unit has produced a **dangerous** sentence, not a shorter one — and for fuel,
dosage and distance in the field that is a safety property.

```python
mind.derive("Depot holds 4000.", parents=[n], kind="summary", ...)
# OwlError: dimensional integrity: 4000 litres lost its unit -
#           the number 4000 survived without it

mind.derive("Give 250 g every six hours.", parents=[dosage], ...)
# OwlError: 250 mg became 250 g - value changed

```

Abstraction is still allowed — *"depot holds 4000 litres"* → *"fuel is not a
constraint"* is fine. Omitting detail is what abstraction **is**; the danger is
keeping the number and losing the unit. And `4000 litres → 4 m³` passes,
because it converts.

### 10. Answers that aren't dead ends

```
Q: who is the depot fuel supplier
   DONT_KNOW -- To answer this I would need a source naming the person
   'depot fuel supplier' - nothing in the store mentions it.

```

`DONT_KNOW` states *what would have to exist*, typed by the question, so the
dead end becomes a task. And **tried-and-failed** is recorded separately from
*looked-and-absent* — it is what stops the same rejected option being
re-proposed every week:

```python
mind.failed("route the convoy via Km-58",
            reason="impassable after rain; lost a truck")

mind.prior_failures("should we route the convoy via Km-58?")
# [{'reason': 'impassable after rain; lost a truck', 'days_ago': 20.0, ...}]

```

### 11. Memory is an investment, not storage

Everyone weights retention by access frequency. The right question is
**"what would it cost me to get this back?"** A fact that took a three-day trip
or a canvass of six vendors is not interchangeable with one that cost a glance
at a filename.

```python
mind.observe("Only vendor stocking diesel is in Kismayo, three days away.",
             source_ref="canvass", acquisition_cost=1.0)

```

Load-bearing criticality feeds the same score: a memory nothing depends on is
safe to let go; one holding up a decision is not, however rarely it's touched.

### 12. Memories that stand on their own

*"He said it'd arrive Thursday"* is useless six weeks later, and a large
fraction of conversational memory looks exactly like that.

```
He said it would arrive Thursday.
  -> Ahmed said the gasket would arrive Thursday (2023-11-16).

```

Two rules shape it. The expansion is a **derived node** — the raw utterance is
evidence and is never rewritten. And ambiguity is **refused, not guessed**:

```
He told her about it.        [Ahmed, Fatima both present]
  -> unchanged, unresolved: ['He', 'her', 'it']

```

A wrong substitution is invisible to the reader; an unresolved pronoun is not.

### 13. Diversity by construction

Top-k *per source* rather than globally, so five chunks can't all come from one
document and look like five pieces of evidence when they're one. When the
budget can't be filled from distinct groups, overflow backfills — a diversity
rule that silently shrinks answers is worse than no rule.

```python
mind.recall("depot diesel", budget=4, group_by="source", per_group=2)
mind.recall("depot diesel", token_budget=400)   # measured, not guessed

```

### 14. Discrimination by margin, not by threshold

Measured on BGE-M3 against a small field corpus, the two bands **overlap**:

```
related cosines      0.426 .. 0.691
unrelated cosines    0.213 .. 0.514

```

No single threshold separates them. Setting it high discards true matches;
setting it low admits noise. What actually distinguishes the two cases is
whether the best match **rises above its background**:

```
"how is the health facility powered"   best 0.69, rest ~0.35   -> real
"what is the helicopter tail number"   best 0.51, rest ~0.45   -> noise

```

A real match stands out from the pack; an accidental one sits in it. That is
how a person decides, and unlike a fixed cutoff it transfers between encoders
without retuning. The floor that remains is a pure noise cutoff, and it is a
property of the **embedder**, not a global constant.

**And sometimes even that is not enough — which is the honest outcome.**
Measured on BGE-M3 over a 22-document corpus, the *margins* overlap too:

```
related margins    0.067  0.166  0.304  0.306
unrelated margins         0.106  0.142  0.198

```

Two genuinely related probes sit below the worst unrelated one. No parameter
separates that, so those queries land in `KNOW_WHERE` — or `DONT_KNOW` when the
margin is as thin as 0.067. **That is correct behaviour, not a limitation to
tune away.** Forcing a confident answer out of evidence indistinguishable from
noise is confabulation, which is the failure this engine exists to prevent.

Run `validate.bat <model> --calibrate` to measure these numbers for your own
encoder rather than trusting mine.

**Hubness correction.** In high-dimensional embedding spaces a few vectors
become the nearest neighbour of very many queries regardless of meaning — a
well-studied pathology of dense retrieval. Measured on BGE-M3, a bland filler
note ("Solar panels on block C were wiped down") outranked the real answer to
*"what part is the borehole missing"*, and the same hub pulled an unrelated
query up into `KNOW_WHERE`. OWL measures each document's mean similarity to
the corpus **relative to the corpus mean** and discounts the hubs. The
validator reports raw-encoder ranking alongside OWL's, so the correction has
to earn its place rather than be asserted.

### 15. Two embedding spaces, not one

Standard RAG embeds everything into a single space, where similar items collapse together — which *causes* the interference problem it then works around. OWL follows Complementary Learning Systems and makes the write path do the opposite of the read path:

* **WRITE — pattern separation.** The vector is mostly meaning plus a deliberate context component (partition, period, episode, source, day), concatenated structurally. Two structurally identical weekly meetings land *apart*.
* **READ — pattern completion.** Bare semantic content, because a query arrives with none of that context. Retrieve a neighbourhood, then discriminate.

```
identical text, different weeks:
  READ  space similarity = 1.000   (meaning: same)
  WRITE space similarity = 0.817   (episodes: distinct)

```

Note this is done by concatenation, not by prepending `[week2]` to the text and hoping a mean-pooled encoder notices. That's a wish, not a mechanism.

Fusion of lexical and semantic scores is **max-of-normalised, not a weighted sum** — either signal firing hard is good evidence, and averaging them down to mediocre is exactly wrong. It also keeps exact-identifier recall, which is where embeddings are weakest: no model reliably separates serial `GX-4419` from `GX-4491`.

### 16. Air-gapped by default, and it grows with you

```bash
pip install -e .              # Tier 0 — everything above. stdlib only.
pip install -e ".[embed]"     # Tier 1 — + semantic recall (ONNX)
pip install -e ".[llama]"     # Tier 2 — + reconstructive compression

```

**OWL never downloads anything.** The ONNX adapter takes a path to a model you placed yourself; fetching one is a separate, explicit, opt-in command you run on a networked machine:

```bash
python -m owl.adapters.fetch_model --out models/all-MiniLM-L6-v2

```

A library that quietly reaches for the network the first time it's asked a question is unusable in the environments this is built for. If you already have a `sentence-transformers` model loaded, `STEmbedder` wraps it in about five lines rather than paying for a second model in RAM.

A store that ran at Tier 0 for a month is **still fully valid** — the substrate is complete and the lexical index works. Attach an embedder later and `reindex()` backfills the vectors. Nothing gets re-ingested.

The hashing fallback is deliberately **not** counted as Tier 1: a fallback that silently produces poor retrieval while looking like it works is worse than no fallback, so `doctor()` says so.

---

## The ideas, and where they come from

OWL is mostly not novel research. It's an attempt to take findings that are decades old and well-replicated, and actually implement them.

| Mechanism | Grounding |
| --- | --- |
| Substrate / index / derivation split | Bjork & Bjork's New Theory of Disuse — storage strength never decreases; retrieval strength does |
| Power-law forgetting, spacing effect | FSRS (DSR model); Wixted & Ebbesen 1991 |
| Feeling-of-Knowing gate | Koriat's metamemory work — triage is fast and separate from retrieval |
| Interference over decay | Underwood 1957 onward — confusable neighbours, not age, kill recall |
| Retrieval-induced forgetting | Anderson, Bjork & Bjork 1994 |
| Event segmentation at surprise | Zacks & Tversky; [EM-LLM](https://arxiv.org/abs/2407.09450) |
| Source monitoring | Johnson, Hashtroudi & Lindsay 1993 — the actual mechanism of false memory |
| Transactive memory | Wegner — theory of mind with the hand-waving removed |
| Levels of processing | Craik & Lockhart 1972; generation effect, Slamecka & Graf 1978 |
| Two embedding spaces | Complementary Learning Systems — McClelland, McNaughton & O'Reilly 1995; pattern separation / completion |
| Bitemporal validity | as in [Graphiti / Zep](https://github.com/getzep/graphiti) |
| Associative spread | as in [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| Source grading | Admiralty scale (NATO STANAG 2511) — two axes, because reliability and credibility are independent |

Where the biology and the engineering disagree, the engineering wins. Human memory is reconstructive *because neurons are expensive and can't be indexed*, not because reconstruction is desirable — and its known consequences are confabulation and source amnesia. OWL keeps the useful half of the metaphor and drops the rest.

---

## How it compares

|  | Typical agent memory | OWL |
| --- | --- | --- |
| Forgetting | delete rows, or never | decay the **index**; the record is immutable |
| Summarisation | on a timer | only what it can **prove** it can reconstruct *(Tier 2)* |
| "I don't know" | indistinguishable from bad matches | first-class, ~0.1 ms, no model call |
| Model-generated content | mixed into the same store | monotone epistemic tag; can't be laundered |
| Staleness | caller's problem | learned per-claim-class half-life |
| The user's knowledge | not modelled | forgetting curve + false-belief detection |
| Confidentiality | application convention | enforced in the store |
| Import someone else's memory | inherit their guesses as facts | automatic epistemic demotion |
| Install | Docker + Postgres + a model | `git clone` + `install.bat` |

OWL is a **component**, not an agent runtime. If you want tools, channels, identity, and a heartbeat, look at [Hexis](https://github.com/QuixiAI/Hexis) or [Letta](https://github.com/letta-ai/letta) — you could sensibly run OWL inside either.

---

## Examples

```bash
python examples/00_tier0_field_notes.py    # states of knowing, partitions, decay
python examples/01_theory_of_mind.py       # transactive memory, half-life, false belief
python examples/02_handover.py             # export, dry run, graft, demotion
python examples/03_semantic.py             # paraphrase gap, two spaces, growing into Tier 1
python examples/04_forward_direction.py    # decisions, blast radius, triage, poisoning defence
python examples/05_trust_loop.py           # source independence, attributed belief, commitments

```

---

## Status

**Alpha, and honest about it.** 465 tests, ~20 seconds, no GPU, no network, no dependencies.

**Working now** — provenance and the monotonicity invariant · FSRS salience · six-state FOK triage · event segmentation · interference detection and record fusion · information-flow partitions · bitemporal recall · epistemic half-life · verbatim protection · transactive memory and false-belief detection · negative memory · handover packs · prospective memory · heterogeneous entity graph · two-space semantic recall · decision–consequence graph · blast radius and retroactive revaluation · load-bearing criticality · retrieval receipts · poisoning defence · adversarial self-audit · source independence · attributed belief · commitment lifecycle · **ambient operation: non-blocking capture, read-only recall, session prefix, anticipatory retrieval, named diagnostics, reviewable handover, multi-operator convergence** · **partition-sharded storage**.

**Ambient operation, in one paragraph.** Capture is 2 ms with an 8B encoder
attached (`defer_embedding=True`) because embedding moved off the hot path —
and a memory captured-but-not-embedded reports as `provisional` rather than
absent, since "not yet" and "not there" are different claims. Recall runs
against a store opened read-only, on read-only media, or while another
process writes; where it can't reinforce what it returned it says so in
`Recall.degraded`, alongside "no embedder" and "embedder raised". Twenty-two
named checks (`python -m owl doctor mind.owl --json`) each carry a remedy,
and seventeen have a test that drives them red — the remaining five
(`epistemics.monotonic`, `defence.self_audit`,
`defence.quarantine_reviewed`, `decisions.impacts_acknowledged`,
`store.liveness`) are asserted only in the green direction and are the next
gap to close. `prefix()` puts consequence in front of a
session — shifted decisions before due commitments before open loops — under
a hard token budget, dropping whole tiers rather than truncating. `watch()`
is anticipatory retrieval that ships **off**, with a session cap, a cooldown,
never-twice-for-the-same-thing, and a `verdict()` that can tell you to turn
it off. Packs export a markdown review copy showing what the *recipient* will
see after demotion. And `converge()` promotes a claim only when independent
**origins** agree — three operators quoting one sitrep are one source.

**Speed, and what it is allowed to cost.** Every optimisation in this
engine has the same failure mode — *a wrong answer delivered faster* — and
that is invisible, so each ships with a test that pins the behaviour before
the speed. A vocabulary filter answers `DONT_KNOW` without touching SQLite;
vectors are a view over the stored buffer rather than a per-query
deserialise; recall is cached on `(query, partition, clock bucket, write generation)` and **any** write invalidates **everything**, because working
out which cached queries a write could have affected is where cache bugs
live and a stale answer looks exactly like a fresh one. `tend()` scopes to
what changed, with a periodic full sweep regardless — a dirty-set bug loses
consolidation *silently*, and the full pass is the thing that would surface
it. Storage is sharded by partition, so the unit of work is the shard and
not the table.

Building these found four real bugs, and the two worth naming were both
invisible:

* **The vocabulary filter went stale and lost memories.** It was built once
on first recall and never updated, reasoning that adding a term only
moves a Bloom filter towards "possibly present" — the safe direction.
True of adding to the *filter*; the code added to the *store* and not the
filter, which is the other direction. Every term written after the first
recall read as definitely absent, so the posting scan was skipped and a
memory the store held came back `DONT_KNOW`. No embedder needed to
reproduce, nothing raised, indistinguishable from ordinary forgetting.
* **Ranking depended on which index SQLite chose.** Candidates tied on
score were left in whatever order rows arrived in. Adding the shard index
changed the plan and the returned chunks reordered with every score
identical. Ties now break on content — the only tiebreak available that
is derived from the memory rather than from the store. `observed_at` was
tried first and is wall-clock, so two memories written in the same loop
tie in one run and not the next; the same test caught that one round
later, which is the argument for the test.

**Not yet** — encryption at rest · time-travel replay · jointly-edited ledger · external benchmarks (LoCoMo, LongMemEval, HaluMem).

**Consolidation is deterministic.** Same content, same communities, same schemas, same verdict — 100 consecutive runs identical. Nobody else guarantees this, and without it *"why did you forget that?"* has no answer. Label propagation breaks ties by sorted id rather than randomly; community ids are derived, never uuid; and a no-op pass is genuinely a no-op rather than silently bumping a generation counter.

**Community identity survives splits and merges.** Recomputing clusters every cycle churns their ids, so every composite derived from an old one points at something that no longer exists and provenance chains break silently. A community keeps its name if its core persists; a split gives the name to the larger side and records lineage on the other.

**Compression is earned, not assumed.** Nothing is compressed until the system reconstructs it from cue plus neighbours and scores the result against the original — recall-weighted and asymmetric, because adding material is verbose while dropping it is destructive, and losing a *number* scores zero rather than 0.9. Below the floor the content stays verbatim. Verbatim claims are never compressed at all, however well they round-trip.

**Embedder:** ships a GGUF adapter for llama.cpp that **reads the model's own
conventions** rather than assuming one family's. Embedding models do not agree
on pooling or query instructions, and getting it wrong is *silent* — the
vectors come back the right shape and retrieval is merely bad:

|  | pooling | query instruction |
| --- | --- | --- |
| `bge-m3` | CLS | none |
| `Qwen3-Embedding` | **last** | **required** |
| `bge-*-en` | CLS | "Represent this sentence…" |
| `e5` | mean | `query:` / `passage:` |

The instruction is asymmetric — query side only — which is the first reason
beyond pattern separation for OWL's READ/WRITE space split to exist. A causal
model never falls back to CLS pooling, because position 0 has attended to
nothing.

**Gate parameters are measured, not hard-coded.** Which signal tells you a
match is real is a property of the encoder. Measured on the same 22-document
corpus:

```
bge-m3      related 0.426-0.691   unrelated 0.415-0.490   bands OVERLAP  -> margin carries
Qwen3-8B    related 0.360-0.531   unrelated 0.275-0.353   bands SEPARATE -> level carries

```

The absolute scale moves too. A `noise_floor` of 0.40 is reasonable for the
first and sits **above two of the four true matches** for the second — real
memories silently discarded, nothing erroring.

```bat
validate.bat "model.gguf" --calibrate

```

measures both bands and writes `model.gguf.owlcal.json` next to the model. The
adapter loads it automatically, so the numbers travel with the file. Running
uncalibrated is a loud warning, not a silent default.

**Two backgrounds, not one.** A raw cosine means nothing without knowing where
the encoder puts text that has nothing to do with anything — and that zero is
in *two different places*, because Qwen3-style models prefix the query side
only, which displaces query vectors relative to document vectors:

```
query -> doc, unrelated   mean 0.216   p95 0.333    <- the recall gate's zero
doc   -> doc, unrelated   mean 0.406   p95 0.529    <- fusion's zero

```

Judging the first against the second is a category error that cost a round of
this project: it reported *negative* headroom for an encoder scoring AUC 0.993
and recommended re-quantising a model that was working correctly. The two are
measured and stored separately now, and `fusion.plan(..., calibration=c)`
rescales dedupe/cluster thresholds into the document space — 0.75 is meant to
sit three quarters of the way from chance to identical, not a fifth.

**Separability, not headroom.** Headroom is a minimum over four probes, so one
weak probe becomes a verdict on the whole encoder. AUC uses every comparison:

```
usable headroom    +0.027   (worst case, n=4)
separability AUC    0.993   (all 600 comparisons)

```

Both are true. Only the second is a judgement about the model. Nothing else in
the field publishes this per-model, and it costs one sweep.

`validate.bat <model.gguf>` must pass before any Tier 1 number is trustworthy.

**The scoreboard.** `bench/scoreboard.py` scores the axes the field does not:

```
FLAGSHIP
  Rescue@10                    1.000   current fact still ranks after supersession
  Inverse-Rescue@10            1.000   superseded wording still retrievable
HONESTY
  Confabulation rate           0.000   absent facts answered confidently
  Epistemic leakage                0   model-generated nodes reachable as fact
  Source attribution           1.000   claims traced to the correct origin
FORWARD
  Consequence recall           1.000   affected decisions surfaced
  Blast-radius completeness    1.000   contaminated conclusions demoted
ADVERSARIAL
  Flooding resistance          1.000   50 docs / 1 origin earns nothing
  Injection containment        1.000   5/5 caught, 0/6 benign notes flagged
SUBSTRATE
  Fusion false-merge           1.000   6/6 real dupes kept, 0 strangers merged
                                       (uncalibrated: 153)
TEMPORAL
  Staleness accuracy           1.000   volatile flagged, durable not

```

The flagship pair belongs together. iai-pme reports **Rescue@10 = 1.000** and
honestly discloses that **inverse-Rescue regressed to 0.71**. OWL scores ~1.0 on
both — not by tuning, but because the substrate is append-only and the old row
was never rewritten. Nothing that mutates in place can hold that line.

Every metric has a **negative control** in `tests/test_scoreboard.py` proving it
can fail. A benchmark that always passes measures nothing, and building these
found two real bugs: pruned nodes leaking back through associative spread, and
`self_audit` never checking that a node's `kind` and epistemic tag agree.

**Tier 1 is validated on real hardware** — `bge-m3-Q6_K` through llama.cpp,
1024-dim, 24-document corpus:

```
pattern separation   READ 1.000 / WRITE 0.886     episodes stay distinct
identifier recall    "GX-4419" -> the right serial, not a paraphrase
cross-lingual        French and Spanish queries hit their English targets
engine faithful      4/4    OWL returned what the encoder ranked
encoder top-1        2/4    BGE-M3's own judgement, not OWL's
throughput           46 ms/text on CPU

```

That last pair is the honest bit. On two probes the **encoder itself** ranks a
different document first; OWL faithfully returns the encoder's ordering,
because ranking is deliberately monotone in similarity. Those are the model's
disagreement, not the engine's, and the validator now attributes them that way
rather than reporting a failure it can't fix.

**Not benchmarked externally yet.** LoCoMo, LongMemEval and HaluMem are the targets. I expect OWL to *lose* on raw QA recall — it retrieves 4–7 chunks by design, following both Cowan's working-memory limit and the lost-in-the-middle degradation — and to win on source attribution, confabulation rate on absent facts, and calibration. Those last three aren't currently scored by anything, which is arguably the more interesting problem.

---

## Design rules

The ten lines the codebase is held to:

1. Decay the index, never the evidence.
2. Never overwrite. Supersede.
3. Provenance is transitive and monotone.
4. Only compress what you can prove you can rebuild.
5. Surprise raises priority, never confidence.
6. Store the exception; compress the rule.
7. Decorrelate errors; don't blindfold nodes.
8. Reward calibration, not confidence.
9. Memory computes what is known. It never decides how to say it.
10. Measure it, or cut it.
11. Provenance points backwards; decisions point forwards. Build both.
12. Quarantine, never refuse — a blocked write is evidence of an attack.

And two the code enforces mechanically: **nothing calls `time.time()**` (the clock is injected, so a 400-day forgetting curve tests in milliseconds), and **nothing writes to the database except the single writer**.

---

## Contributing

Useful and self-contained, roughly in order:

* **A `Reasoner` adapter** — llama.cpp with GBNF grammars, or any OpenAI-compatible endpoint. Unlocks Tier 2.
* **An ANN index** behind `VectorIndex` — the only place that knows how similarity is computed.
* **Benchmark harness** for LoCoMo / LongMemEval.
* **A Postgres `Store**` — the protocol is small and SQLite-specific SQL is confined to one file.
* **Adversarial tests.** If you can make it assert something it was never told, that's the most valuable issue you can file.

Correctness tests must stay **fast, and green without a GPU or the network**. A memory system whose correctness suite doesn't get run is one that corrupts itself quietly for six months.

---

## Scope

The test any proposed feature has to pass: **does this help answer "how do you know that?"**

If not, it belongs in the host application. OWL is deliberately not an agent framework, not a RAG pipeline, and not a chat loop.

---

MIT · built for offline, air-gapped, and low-resource environments where being wrong has a cost