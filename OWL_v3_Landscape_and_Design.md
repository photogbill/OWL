# O.W.L. v3 — Landscape Survey & What Changed

**Companion to:** `owl-engine/` (running code, 24 tests, ~2 s, no GPU)
**Supersedes:** the v2 brief's architecture sections where they conflict.

---

## Why v3 exists

Two things changed after reading the ATK notes and surveying the field.

**From ATK:** Athena-in-ATK is not an analyst tool. It's a companion for an exhausted aid worker at 3am, and the notes state the requirement plainly — *"what's said to Athena stays private: NOT fed to the work graph, reports, or intercept log."* That is an architectural constraint, not a policy note, and it belongs in the persistence layer where a refactor can't quietly break it. v2 had namespaces; v3 has an **information-flow lattice with sealed partitions**, and the boundary is enforced by SQL, not by convention.

**From the field:** agent memory is now a mature area with its own benchmarks (LoCoMo, LongMemEval, BEAM) and several good systems. Some of what v2 proposed as novel isn't; some of what the field has built is better than what v2 specified. This document reconciles both.

---

# Part I — The landscape

| System | Core idea | Worth taking | Where it falls short |
|---|---|---|---|
| **MemGPT / Letta** | OS-style virtual context: main context = RAM, external = disk, LLM pages itself via function calls | The agent-facing tool interface — memory as something an agent can *query about itself* | The LLM decides what to evict. That's unreliable and expensive, and it makes forgetting non-deterministic |
| **Mem0** | Extract → decide ADD / UPDATE / DELETE / NOOP over a vector store, plus a graph variant | Explicit memory *operations* instead of blind append; huge adoption (41k stars, AWS Agent SDK) | LLM-judged UPDATE destroys history in place. No provenance, no way to ask "what did you used to think?" |
| **Zep / Graphiti** | Temporal knowledge graph; time is a first-class dimension; edges are *invalidated*, not deleted | **Bi-temporality** — the single best idea in the field, and v2 didn't have it | Requires a graph service; no metamemory; retrieval still assumes the answer is in there somewhere |
| **A-MEM** | Zettelkasten: atomic notes with LLM-generated keywords/tags, autonomous link generation, memory evolution with supersede detection | **Supersede rather than overwrite** — same conclusion v2 reached independently | Evolution rewrites note context with no provenance floor, so drift is unbounded |
| **HippoRAG / HippoRAG 2** | Open KG + Personalized PageRank from query-seeded nodes; explicitly modelled on hippocampal indexing | **Graph propagation for multi-hop** — recovers links flat vector search misses, no second model call | Indexing theory used as inspiration only; no pattern *separation* on write, which is the half that prevents interference |
| **EM-LLM** | Segment the token stream into episodes at **Bayesian surprise** boundaries, refined graph-theoretically; two-stage retrieval (similarity + temporal contiguity) | **Event segmentation** — and the finding that LLM surprise tracks human-perceived event boundaries | Operates inside the context window; not a persistent store |
| **Generative Agents** (Park et al.) | Memory stream scored by recency + importance + relevance; periodic reflection trees | LLM-scored **importance at write time**; reflection as a scheduled operation | Importance is a single LLM guess with no calibration; reflections enter the stream as ordinary memories |
| **MemoryBank** | Ebbinghaus-inspired forgetting curve | First to take forgetting seriously | Naive exponential — the exact defect in the v1 brief |
| **RAPTOR** | Recursive abstractive summarisation into a tree; retrieve at multiple levels | Multi-scale retrieval | Summarises blindly; no check that the summary preserves anything |
| **GraphRAG** | Leiden community detection + community summaries for *global* questions | **Discovered structure** — answers "what are the themes?" which vector search structurally cannot | Expensive; static; full re-index on update |
| **Titans** | Neural long-term memory learning to memorise at test time, gated by surprise (gradient magnitude) | Independent validation that **surprise is the right gate** | Architecture-level; not applicable to a llama.cpp deployment |

**The honest read:** Zep leads on temporal reasoning (published LongMemEval numbers put it well ahead of Mem0, though the two teams dispute each other's configurations — treat all vendor benchmarks sceptically). Mem0 leads on adoption. HippoRAG leads on multi-hop. Nobody leads on the thing OWL is for.

## What nobody does

These are OWL's actual differentiators, and they survived the survey:

1. **Provenance as an enforced invariant.** Zep has bi-temporal edges; Mem0 has operations; A-MEM has supersede detection. None enforces *monotone epistemic tagging* — the property that a node derived from a hypothesis is permanently a hypothesis, and abstraction cannot launder speculation into fact. Every one of these systems writes LLM-generated content into the same store as source content and retrieves it identically later.

2. **Reconstructive compression.** Everyone summarises on a schedule or a size trigger. Nobody first checks whether the summary can regenerate the original. This is the difference between compression and quiet data loss.

3. **Metamemory as a pre-retrieval gate.** SELF-RAG is closest, but it's a trained model emitting reflection tokens mid-generation. Nothing does cheap triage *before* spending anything — and nothing treats `DONT_KNOW` as a valuable output.

4. **Interference management.** Everyone models decay (or nothing). Nobody actively detects and resolves confusable neighbours, despite interference being the better-supported account of forgetting since Underwood (1957).

5. **Information-flow partitions.** Multi-tenancy exists everywhere. *Non-transitive, direction-sensitive, sealed* partitions with store-level enforcement do not.

6. **Pattern separation on write.** HippoRAG invokes hippocampal indexing but embeds everything in one space, which collapses similar items together — causing the interference problem it then works around. Separating on write and completing on read is the actual mechanism.

---

# Part II — What v3 adds

Seven mechanisms, all now in the code.

### 1. Bitemporal validity *(from Zep/Graphiti — the best idea I found)*

Every observation carries **two** timelines:

- `observed_at` / `superseded_at` — *system time*: when we learned it
- `valid_from` / `valid_to` — *world time*: when the fact was actually true

```python
mind.recall("route alpha", as_of=t0 + 1*DAY)   # -> "Route Alpha is open"
mind.recall("route alpha", as_of=t0 + 8*DAY)   # -> "closed by flooding"
```

For ATK this is not a nicety. "Was the bridge out when we routed that convoy?" and "is the bridge out now?" are different questions, and a system that conflates them will confidently answer the wrong one. v2 had only ingestion order. *(`tests/test_bitemporal.py`)*

### 2. Information-flow partitions *(new — required by ATK)*

A lattice, not a namespace. Flow is **denied by default**, **directional**, and **transitive along declared edges only**.

```python
mind.partition("intake", flows_to=["analysis"])   # intake -> analysis, not back
mind.partition("athena", sealed=True)             # no outflow, ever
```

A sealed partition is refused at creation if it tries to declare outflow. Reads compute the transitive inbound closure in the store layer, so a violation is impossible rather than merely discouraged.

This also upgrades Athena's central claim. Today the isolation contract is a test asserting `shared_context=None` is passed — a call convention. Under OWL it's a property of persistence. *(`tests/test_partitions.py`)*

### 3. Event segmentation at surprise boundaries *(from EM-LLM + Zacks & Tversky)*

Chunking by size is an artifact of the tokenizer. OWL declares an episode boundary when surprise exceeds a running z-threshold, so episodes are as long as the material stays coherent:

```
              surprise=1.00  The north well pump failed again this morning.
              surprise=0.75  The pump gasket is worn and the well is the only...
              surprise=0.50  Well water sampling showed contamination...
              surprise=0.75  Ahmed says pump parts for the well arrive Thursday.
-- BOUNDARY-- surprise=1.00  Separate matter entirely: vaccine cold chain...
              surprise=0.56  Refrigerator logs show a cold chain excursion...
```

One implementation note that cost me a debugging cycle and is worth recording: **do not clear the context window at a boundary.** Doing so makes the next observation look maximally surprising against an empty context and fires a spurious second boundary immediately after every real one. The sliding window ages the old episode out by itself. *(`owl/segmentation.py`, `tests/test_segmentation.py`)*

### 4. Associative spread *(from HippoRAG)*

One-step Personalized-PageRank-style propagation over a co-occurrence graph, seeded by the lexical/semantic candidates. Multi-hop association at Tier 0 cost — a few SQL queries, no model call. Successor-representation edges (what tends to be retrieved *next*) feed the same spread, which gives prefetch for free. *(`Owl._expand_assoc`)*

### 5. Affect-gated retrieval *(new — required by Athena)*

Observations carry an `affect` marker. A companion context can decline to surface distressing material unbidden:

```python
mind.recall("ward smell", partition="athena", suppress_affect_above=0.5)
```

It filters *presentation only*. Nothing is deleted, and the memory stays fully addressable on an explicit request. The design constraint from the ATK notes — attunement without amplification, never mirroring distress back — becomes a retrieval parameter rather than a prompt instruction that a model may or may not follow. *(`tests/test_partitions.py`)*

### 6. Confidence–epistemic coupling *(caught by the demo)*

The demo surfaced a real bug: a node created with `kind="hypothesis"` was stored as `epistemic="inferred"` because the caller didn't pass the tag. Downstream that reads as a *conclusion*. Now `kind` and `epistemic_tag` are not permitted to disagree — a hypothesis is forced to `hypothesized` regardless of what the caller claims.

Small fix, and exactly the class of thing that survives into production and later produces a confident assertion of something the system made up.

### 7. Coverage-weighted scoring *(caught by a test)*

The confabulation test failed on first run, and the cause is worth naming because it's the obvious implementation:

> Normalising candidate scores to the best candidate makes the top hit score 1.0 **no matter how bad it is.**

A store containing only *"the water tanker arrives Tuesday"* answered *"when does the **fuel** tanker arrive"* with `KNOW`. That is confabulation by ranking artifact, and every top-k-with-normalised-scores retriever has some version of it. The fix: score = **query coverage × match quality**, so a node can only reach `KNOW` if it covers most of what was actually asked. *(`tests/test_confabulation.py`)*

---

# Part III — What's built

```
owl-engine/                      1,818 lines Python + 189 lines SQL
├── owl/
│   ├── __init__.py              Owl facade: observe / recall / tend / derive / why
│   ├── protocols.py             Embedder, Reasoner, Store, Clock + value types
│   ├── schema.sql               3 layers + partitions + bitemporal + episodes
│   ├── provenance.py            the monotonicity invariant
│   ├── salience.py              FSRS (DSR) + ACT-R alternative
│   ├── metamemory.py            the FOK gate
│   ├── segmentation.py          surprise-boundary event segmentation
│   ├── lexical.py               Tier-0 index (real IDF, not the self-referential kind)
│   └── store/sqlite.py          single-writer queue + flow-control closure
├── tests/                       24 tests, ~2 s, no GPU
└── examples/00_tier0_field_notes.py
```

**Verified working** (`python examples/00_tier0_field_notes.py`):

```
Q: who runs the clinic          -> KNOW        (2.7 ms)
Q: what is the helicopter tail  -> DONT_KNOW   (0.1 ms)   <- no model call

work   -> 'triage decisions' : dont_know       <- sealed boundary holds
athena -> 'triage decisions' : know

*NOT FACT* [hypothesized] conf=0.70  The clinic can sustain operations...
*NOT FACT* [inferred    ] conf=0.90  Fuel resupply is currently viable.
FACT       [observed    ] conf=1.00  Route Alpha is open as of this morning.

after 400 idle days: tip_of_tongue,  tiers {'cold': 8},  substrate rows: 8
```

That last line is the whole design in one place: retrieval degraded, the index re-tiered, **nothing was deleted**.

Note the `DONT_KNOW` at 0.1 ms — 27× faster than the `KNOW` path, because the FOK gate returns before touching the index at all. On a 16 GB box where the alternative is a model load, that's the difference between a tool and a batch job.

---

# Part IV — What's next

**M2 — semantic tier.** ONNX embedder adapter. Two spaces: separation on write (embed content *plus* distinguishing context, so near-duplicates are pushed apart), completion on read. Replaces the `_blend_semantic` stub.

**M3 — reasoner tier.** llama.cpp + OpenAI-compatible adapters with GBNF grammar constraints. **Reconstructive compression** — the distinctive mechanism, and the one-sentence pitch: *a memory system that only forgets what it has proven it can reconstruct.*

**M4 — community detection.** Leiden over the assoc graph plus period summaries, for the global questions vector search can't answer. Borrowed from GraphRAG, but incremental rather than full-reindex.

**M5 — Athena adapter.** Eight sealed partitions, one per persona; composer reads all, personas read only themselves. Athena's isolation guarantee becomes a store property.

**M6 — sleep, behind a flag, default off.** Two-phase NREM/REM on sleep pressure. Quarantined hypotheses with mandatory falsifiers. Run the ablation; be willing to cut it.

**Benchmarks to target:** LoCoMo and LongMemEval, so the comparison against Mem0/Zep is on published ground rather than assertion. Expect OWL to lose on raw QA recall at first — it retrieves less by design (4–7 chunks) — and to win decisively on the axes nobody currently scores: source attribution accuracy, confabulation rate on absent facts, and calibration. Those are worth proposing as benchmark extensions in their own right.

---

## Two honest cautions

**Scope.** OWL is now doing more than v2 specified, and each addition was justified. That's how libraries become unmaintainable. The test I'd hold it to: *does this help answer "how do you know that?"* Bitemporality, partitions, and provenance pass. If community detection or dreaming can't pass it, they belong in the host.

**Don't put OWL under Athena-the-companion first.** Put it under the ATK *work* side, where a wrong answer is visible and correctable. The companion path has an unforgiving failure mode — a system that misremembers what someone told it at 3am, or worse surfaces it in the wrong context, does real harm. Earn that deployment with the eval harness green and a few months of work-side use behind it.

## Sources

- [QuixiAI/Hexis](https://github.com/QuixiAI/Hexis)
- [EM-LLM: Human-inspired Episodic Memory for Infinite Context LLMs](https://arxiv.org/abs/2407.09450)
- [HippoRAG 2 / From RAG to Memory: Non-Parametric Continual Learning](https://arxiv.org/abs/2502.14802)
- [HippoRAG (OSU-NLP-Group)](https://github.com/osu-nlp-group/hipporag)
- [FSRS — Free Spaced Repetition Scheduler](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler)
- [Mem0 vs Zep (Graphiti) comparison, 2026](https://vectorize.io/articles/mem0-vs-zep)
- [Agent memory architectures 2026: Mem0 vs Zep vs Letta vs LangMem](https://maidul-haque.vercel.app/blog/agent-memory-architectures-2026/)
- [State of AI Agent Memory 2026 (Mem0)](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
