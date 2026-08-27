# O.W.L. — Build Plan

**Decisions locked:** OWL is a standalone library. Host-agnostic — not ATK-specific, not Athena-specific. Built and validated in isolation before anything imports it.
**Companions:** `OWL_Critique_and_Improvements.md` (why), `OWL_v2_Project_Brief.md` (what).

---

## 0. The decision that makes everything else easy

You've picked "standalone library." That choice hands you a design principle you should now use ruthlessly:

> **The library boundary is the deterministic/stochastic boundary.**

Everything OWL does that doesn't need a model — provenance, monotonicity, decay scoring, tier transitions, interference detection, scheduling, prospective-memory triggers, the whole evaluation harness — is deterministic Python over SQLite. It runs in milliseconds, tests in CI, needs no GPU, and behaves identically on every machine.

Everything that needs a model — reconstructive compression, grafting, hypothesis generation, elaborative encoding — sits behind a **protocol** that the host supplies.

This is not just tidiness. It's what makes the project *finishable*. The hardest bugs in a memory system are in the deterministic part (is provenance intact? did decay do the right thing after 90 simulated days? did dedup merge two things it shouldn't have?), and those are exactly the bugs that are impossible to find when every test run also involves a 5 GB model load and a stochastic output. Separate them and you can run your entire correctness suite in under a second.

It also gives you a genuinely strong adoption story, which matters if you want others to use this.

---

## 1. Where OWL sits relative to Hexis

The Gemini transcript got Hexis wrong, and it's worth correcting because the actual design is good and one part of it is worth taking.

**What Hexis actually is:** *"The Database Is the Brain."* PostgreSQL (pgvector + Apache AGE + btree_gist) is the system of record for all cognitive state. Python workers are **stateless**. Cognitive logic lives substantially in PLpgSQL — 15.6% of the repo. Plus an autonomous heartbeat on an OODA loop with **energy budgets** per action, multi-provider LLM support, 80+ tools, and messaging channels.

The transcript's read — "it treats cognition as a filing cabinet," "ACID means immutable" — is wrong on both counts. ACID is about transaction guarantees, not immutability, and stateless-workers-over-a-stateful-store is a strong pattern, not a rigid one.

**Take from Hexis:**

- **State in the store, workers stateless.** This is the right call and you should adopt it wholesale. It means you can kill the process mid-consolidation, restart, and lose nothing; you can inspect the entire mind with a SQL client; you can swap the model without migrating anything. It's also what makes OWL testable without inference.
- **Energy budgets.** This is where the transcript's "compute economy" came from, and it's a sound primitive: every autonomous action costs something from a finite budget, so autonomy is bounded by construction rather than by hoping the loop terminates.
- **`hexis doctor`.** A health-check command. Cheap to build, disproportionately good for adoption — it turns "it doesn't work" issues into self-diagnosis.

**Don't take from Hexis:**

- **Postgres + AGE + RabbitMQ + Docker.** Hexis's quick start requires Docker Desktop, Ollama, and a running Postgres. That is a completely reasonable choice for what Hexis is — a full agent runtime — and a fatal one for a library you want other people to depend on.

**That's your wedge, and it's a real one:**

| | Hexis | OWL |
|---|---|---|
| Install | Docker Desktop + Postgres + Ollama | `pip install owl-engine` |
| Scope | Full agent runtime — memory, tools, channels, identity | Memory engine only |
| Store | Postgres + pgvector + AGE | SQLite (Postgres optional, same protocol) |
| Needs a model to be useful | Yes | **No** — Tier 0 works with zero models |
| Headline property | Continuity and identity | **Provenance** — always answers "how do you know?" |

You are not competing with Hexis. Hexis is an agent; OWL is a component. Someone could sensibly run OWL *inside* Hexis. Position it that way and you get goodwill instead of a rivalry — and 586 stars' worth of an audience that already cares about this exact problem.

---

## 2. Public API — three verbs

Keep the surface tiny. Almost all memory libraries fail by exposing their internals; the ones people adopt expose four or five calls.

```python
from owl import Owl

mind = Owl.open("./mind.owl")          # zero config. SQLite file. That's the whole setup.

# ─── WRITE ────────────────────────────────────────────────
ref = mind.observe(
    "The Henderson filing was submitted 2026-03-14.",
    origin="document",                              # user_utterance | document | tool_output
    source_ref="file://henderson/filing.pdf#page=3",
)

# ─── READ ─────────────────────────────────────────────────
r = mind.recall("when was Henderson filed?", budget=5)

r.state          # KNOW | KNOW_WHERE | TIP_OF_TONGUE | DONT_KNOW   ← check this first
r.chunks         # <= budget items, each with .content and .provenance
r.why(chunk)     # full derivation chain back to primary sources

# ─── MAINTAIN ─────────────────────────────────────────────
mind.tend()                    # one synchronous maintenance pass, returns a report
mind.tend(budget_seconds=30)   # bounded
```

Four more calls cover everything else:

```python
mind.period("henderson-matter")     # context manager — scopes observations to a period
mind.intend(when=..., then=...)     # prospective memory
mind.namespace("logical_math")      # isolated sub-store  ← this is how Athena uses it
mind.doctor()                       # health check, borrowed from Hexis
```

**`r.state` is the most important line in the API.** It's the Feeling-of-Knowing gate, and putting it first in the returned object trains every caller to check "do I actually know this?" before consuming the chunks. `DONT_KNOW` returns in ~5 ms with zero chunks and no model call. Most memory libraries make "I have nothing" indistinguishable from "here are five bad matches"; making it a first-class enum is a small design decision with a large downstream effect on how honest the systems built on OWL end up being.

### Protocols the host supplies

```python
from typing import Protocol

class Embedder(Protocol):
    def embed(self, texts: list[str], space: Space) -> "ndarray": ...
    # space is WRITE (pattern separation) or READ (pattern completion)

class Reasoner(Protocol):
    def complete(self, prompt: str, *, grammar: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.7) -> str: ...

class Store(Protocol): ...     # SQLite ships; Postgres is a contributed impl

class Clock(Protocol):
    def now(self) -> float: ...
```

**The `Clock` protocol is the highest-leverage twelve lines in the codebase.** Never call `time.time()` anywhere in OWL. With an injectable clock you test a 90-day forgetting curve in three milliseconds:

```python
def test_forgetting_curve(mind, clock):
    mind.observe("the safe combination is 41-19-72", origin="user_utterance")
    clock.advance(days=90)
    assert mind.recall("safe combination").state is State.DONT_KNOW
    # ...and the substrate is still intact:
    assert mind.substrate_count() == 1
```

Without it, decay is untestable and you will ship it broken. With it, decay is the *easiest* thing in the system to test. Build this on day one.

### Capability tiers — the adoption story

```
pip install owl-engine              # Tier 0 — no models, no GPU, works immediately
pip install owl-engine[embed]       # Tier 1 — + local embeddings (ONNX / sentence-transformers)
pip install owl-engine[llama]       # Tier 2 — + llama.cpp reasoner
pip install owl-engine[openai]      # Tier 2 — + any OpenAI-compatible endpoint
```

| Tier | Needs | You get |
|---|---|---|
| **0** | nothing | Provenance graph, monotonicity, FSRS decay, lexical interference detection, periods, intentions, the eval harness. Pure Python + stdlib SQLite. |
| **1** | `Embedder` | Semantic recall, two-space separation/completion, real interference detection, FOK density estimation |
| **2** | `Reasoner` | Reconstructive compression, elaborative encoding, grafting, conflict resolution, hypothesis generation |

Tier 0 must be genuinely useful on its own. A provenance-tracked, decaying, self-deduplicating note store with no ML dependency and no config is something people will `pip install` on a whim — and that's how a library gets its first hundred users. Tiers 1 and 2 are what they upgrade to after it's already in their project.

Degradation must be graceful and *loud*: if no `Reasoner` is configured, `tend()` runs the deterministic passes, skips the rest, and says so in its report. Never silently no-op.

---

## 3. Repo layout

```
owl-engine/
├─ owl/
│  ├─ __init__.py          # Owl, State, Origin — the entire public API
│  ├─ protocols.py         # Embedder, Reasoner, Store, Clock
│  ├─ schema.sql           # the three-layer schema
│  ├─ provenance.py        # derivation graph + assert_monotonic()
│  ├─ salience.py          # FSRS (DSR) scoring
│  ├─ metamemory.py        # the FOK gate
│  ├─ encode.py            # observe path
│  ├─ recall.py            # recall path
│  ├─ interference.py      # confusability sweep, RIF demotion
│  ├─ periods.py           # Self-Memory System hierarchy
│  ├─ intentions.py        # prospective memory
│  ├─ tend/
│  │  ├─ scheduler.py      # sleep pressure (Process S + C)
│  │  ├─ nrem.py           # reconstructive compression, dedup, conflict resolution
│  │  └─ rem.py            # hypothesis generation  [Tier 2, flag-gated, default OFF]
│  ├─ store/
│  │  ├─ base.py
│  │  ├─ sqlite.py         # single-writer queue lives here
│  │  └─ postgres.py       # later; contributed
│  └─ adapters/
│     ├─ llamacpp.py
│     ├─ openai_compat.py
│     └─ onnx_embed.py
├─ tests/
│  ├─ conftest.py          # FakeClock, FakeReasoner, FakeEmbedder, temp mind fixture
│  ├─ test_provenance.py
│  ├─ test_monotonicity.py # property-based (hypothesis)
│  ├─ test_forgetting.py
│  ├─ test_confabulation.py
│  ├─ test_interference.py
│  └─ test_isolation.py    # namespace boundary enforcement
├─ bench/
│  └─ harness.py           # the §12 evaluation suite, runnable on a corpus
├─ examples/
│  ├─ 00_tier0_notes.py    # works with zero deps — the front-door demo
│  ├─ 01_with_embeddings.py
│  └─ 02_athena_adapter.py
└─ pyproject.toml
```

**Hard rules:**

- `owl/__init__.py` exports **at most eight names**. Everything else is internal.
- Nothing in `owl/` imports torch, transformers, llama_cpp, or numpy at module level. Tier 0 must import in under 100 ms.
- No module outside `store/sqlite.py` writes to the database. Every mutation goes through the single-writer queue.
- No module calls `time.time()`. Ever. Use the injected clock.

**Package name:** `owl-engine` looks available on PyPI. Grab it now — it costs nothing to register a placeholder and it's the kind of thing that's gone in six months. `owl` itself is almost certainly taken.

---

## 4. Milestones

### M0 — Foundation *(a weekend)*
**No models. No embeddings. No inference. This is the whole point.**

1. `schema.sql` — the three-layer schema, with an `UPDATE` trigger on `observation` that raises. Immutability enforced by SQLite, not by convention or code review.
2. `provenance.py` — `assert_monotonic()`: confidence never exceeds the parents' minimum, epistemic tag never decreases along a derivation edge.
3. `store/sqlite.py` — the single-writer queue. One `asyncio.Queue` (or one thread + `queue.Queue` if you'd rather stay sync). No exceptions to this.
4. `conftest.py` — `FakeClock`, `FakeReasoner` (returns canned strings), `FakeEmbedder` (deterministic hash vectors).
5. Three tests: monotonicity fuzz, substrate immutability, forgetting curve under simulated time.

**Exit criterion:** `pytest` green in under one second, and you can `mind.observe()` / `mind.recall()` with lexical matching only.

You now have something real. It's already better than most agent memory layers, and it has zero dependencies.

### M1 — Memory that works *(~2 weeks)*
FSRS scoring. FOK gate. Period hierarchy. Lexical interference sweep. `doctor()`. `examples/00_tier0_notes.py`.

**Ship this.** Tag v0.1.0, push to PyPI, write a short README that leads with provenance and `pip install owl-engine` with no other setup. Post it where the Hexis audience is. Getting real users at M1 is worth more than three more months of solo building — you'll find out immediately which parts of the API are wrong, and API mistakes are the expensive kind to fix later.

### M2 — Semantic tier *(~2 weeks)*
`Embedder` protocol + ONNX adapter. Two embedding spaces (separation on write, completion on read). Real confusability detection. Retrieval-induced-forgetting demotion. `[embed]` extra.

### M3 — Reasoner tier *(~3 weeks)*
`Reasoner` protocol + llama.cpp and OpenAI-compatible adapters. **Reconstructive compression** — the mechanism from the critique, and the thing that makes OWL distinctive rather than just tidy. Elaborative encoding. Conflict resolution. GBNF grammars for the llama.cpp path.

**This is the release that earns attention.** "A memory system that only forgets what it has proven it can reconstruct" is a one-sentence pitch nobody else has.

### M4 — Athena adapter *(~1 week)*
See §5. First real consumer. Expect to change the API here — that's what a first consumer is for.

### M5 — Autonomy *(later, flag-gated, default off)*
Sleep-pressure scheduler. NREM/REM split. Prospective memory queue. Quarantined hypotheses with mandatory falsifiers. Energy budgets, borrowed from Hexis.

**Do not build M5 before M4.** The dream engine is the most fun part and the least load-bearing, and if you build it early it will eat the schedule and you'll have an elaborate hypothesis generator sitting on an unvalidated memory core.

---

## 5. How Athena consumes it

Athena's current memory layer is one `AthenaMemoryManager` per persona, each with its own SQLite file. That maps onto OWL almost exactly — which is a good sign that the abstraction is right.

```python
mind = Owl.open("athena.owl", embedder=..., reasoner=...)

personas = {name: mind.namespace(name) for name in EIGHT_INTELLIGENCES}
composer = mind.namespace("composer")
constitutional = mind.namespace("constitutional")
```

**Namespaces are a first-class OWL feature, not an Athena special case.** Any multi-agent system needs isolated per-agent memory, so this is generally useful. And it upgrades Athena's central claim: right now the isolation contract is asserted by `test_persona_isolation_prompt_default` checking that `shared_context=None` is passed. Under OWL, **the store layer refuses cross-namespace reads** — isolation becomes a property the persistence layer enforces rather than a call convention a future refactor could quietly break. That's a stronger claim for the paper, and it's free.

### Five things OWL fixes in Athena's current implementation

**1. `access_boost = log(1 + access_count) / 10` has the same defect as the v1 decay formula.** It knows only *how many* times a memory was used, never *when*. Ten accesses in one hour and ten spread over a year score identically. FSRS's `access_log` fixes this and gives you the spacing effect for free.

**2. `ATHENA_SQLITE_TIMEOUT = 600.0` is a symptom, not a fix.** A ten-minute busy timeout means writers are contending and the current answer is to wait them out. The single-writer queue removes the contention rather than absorbing it. This will also make the failure mode legible — right now a deadlock looks like a ten-minute hang.

**3. The hashed bag-of-words fallback.** You were right to rename it from "TF-IDF" once you found the formula wasn't real IDF — that's exactly the kind of honesty the project has going for it. But it's a weak retrieval signal. OWL's `[embed]` extra ships a small ONNX model, so the fallback stops being the default path.

**4. `META_INTROSPECTION` is a source-monitoring fix, and you found it empirically.** You noticed that meta-responses were being stored as ordinary memories and then retrieved as "explanations of past explanations," and you fixed it with a type tag plus a default filter. That is precisely the source-monitoring failure from the critique — you hit it in the wild and patched the specific case. OWL generalizes it: instead of eight memory types with ad-hoc retrieval rules, one `origin` + `epistemic_tag` pair with a general policy. `DREAM_FRAGMENT`, `BACKGROUND_THOUGHT`, and `META_INTROSPECTION` all become `origin=model_inference` with different producers, and the "don't feed this back as fact" rule applies to all three by construction rather than one at a time. The fact that you discovered the problem independently is decent evidence the abstraction is the right one.

**5. Your eight memory types map cleanly and become richer.**

| Athena type | OWL representation |
|---|---|
| `STANDARD` | `observation` + derived summary |
| `COGNITIVE_TENSION` | `derived(kind='conflict')` — now *generated* by the interference sweep, not just recorded |
| `DOUBT` | low confidence + `TIP_OF_TONGUE` FOK state |
| `ERROR` | `derived(kind='correction')` with a `supersedes` edge — your error autobiography becomes queryable history |
| `CURIOSITY` | `intention` with `open_loop=1` (Zeigarnik bonus) |
| `BACKGROUND_THOUGHT` | `origin=model_inference`, `epistemic_tag=inferred` |
| `DREAM_FRAGMENT` | `derived(kind='hypothesis')` + **mandatory falsifier** |
| `META_INTROSPECTION` | `origin=model_inference`, `producer='explainer'` |

### On Gardner — a paper problem, not an architecture problem

I criticized MI in the review before I'd seen how central it is to Athena. Having read the paper, my position is narrower and I think more useful:

**Athena does not need MI to be true.** What Athena needs is specialists whose *errors are decorrelated*. MI is being used as a prompt-diversity generator, and it's a decent one — eight sharply different framings that a single system prompt wouldn't produce. The architecture works or fails on decorrelation, which is measurable and which doesn't depend on Gardner being right about human cognition.

**But the arxiv paper claims more than that,** starting with the title: *"Grounded in Gardner's Theory of Multiple Intelligences."* A reviewer will reach for Visser, Ashton & Vernon (2006) — who built two tests per intelligence, ran them on 200 adults, and found the cognitive domains loading on a single *g* factor rather than separating — and that becomes the review, regardless of how good the rest of the work is. You'd lose the paper on a claim the system doesn't actually rest on.

**Cheap fix, no code changes:** reframe MI in §2.1 as an explicit *design heuristic for generating diverse specialist priors*, note the empirical status honestly in one paragraph, and state the falsifiable claim you're actually testing — that cognitively-isolated, differently-primed specialists produce measurably more decorrelated outputs than prompt-differentiated agents sharing context. That claim is yours, it's testable, and it survives MI being wrong. It also makes the paper *stronger*, because "we use a contested framework as a generator and test the property we actually care about" is a more sophisticated position than "we implement Gardner."

---

## 6. What I'd do first, concretely

This week, in order:

1. **Register `owl-engine` on PyPI.** Empty placeholder. Five minutes.
2. **Write `tests/test_forgetting.py` before writing `salience.py`.** The test above, verbatim. It will fail. That failing test is your spec.
3. **Write `schema.sql`** with the immutability trigger. Verify by hand that `UPDATE observation SET content='x'` raises.
4. **Write `provenance.py`** and a `hypothesis`-based property test that fuzzes derivation graphs and asserts no reachable state violates monotonicity.
5. **Then** the single-writer store.

Note what's absent: no model, no embeddings, no llama.cpp, no Athena. If you find yourself installing torch in week one, you've drifted.

**The discipline that matters:** OWL's correctness suite must stay green in under a second, forever. The moment it needs a GPU, you'll stop running it, and a memory system whose correctness suite doesn't run is a memory system that quietly corrupts itself for six months before you notice.

---

## 7. Two risks worth naming

**Scope creep toward "agent."** OWL will constantly be tempted to grow tools, planning, a chat loop, an identity model. Hexis already occupies that space well. OWL's value is being the piece Hexis-like systems are missing, and it stays valuable by staying small. My suggested test for any proposed feature: *does this help answer "how do you know that?"* If not, it belongs in the host.

**Dreaming eating the schedule.** M5 is the most interesting part of the design and the least load-bearing. It's flag-gated and default-off in the brief for a reason. Build it last, run the ablation from the eval harness, and be genuinely willing to cut it. A memory engine with excellent hygiene and no dreaming is a good library; one with elaborate dreaming on an unvalidated core is a demo.

---

## Next step

I can generate the M0 skeleton — `schema.sql`, `provenance.py`, `protocols.py`, `store/sqlite.py`, and the four tests — as actual runnable code. Say the word and I'll build it into the OWL folder.

## Sources

- [QuixiAI/Hexis](https://github.com/QuixiAI/Hexis) — README, architecture, and stack
- [FSRS — Free Spaced Repetition Scheduler (DSR model)](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler)
- [Visser, Ashton & Vernon 2006 — Beyond g: Putting multiple intelligences theory to the test](https://www.sciencedirect.com/science/article/abs/pii/S0160289606000201)
