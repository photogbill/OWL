# Competitive analysis — where OWL actually stands

Reviewed: MIRIX, MemoryLLM/M+, LycheeMemory, Memanto, MiniRAG, Mem0, Zep/Graphiti, A-MEM, HippoRAG, EM-LLM, GraphRAG, Letta, Hexis.
Not reviewed: `alibaba/zvec`, `CodeAbra/iai-personal-memory-engine` (search quota exhausted mid-review — flagged rather than guessed at).

The building philosophy is *innovate, improve, outperform — don't reinvent the wheel.* That's right, but it needs a sharpened edge, because you cannot outperform everyone on everything and trying to is how a focused library becomes a mediocre framework. What follows is the honest version.

---

## 1. The uncomfortable finding: Memanto

**Memanto's "Six Gaps" table is, substantially, OWL's pitch.** Published, 1.7k stars, an arXiv paper (2604.22085), `pip install memanto`:

| # | Their gap | Their answer |
|---|---|---|
| 2 | No temporal decay | Versioning, recency signals, temporal queries |
| **3** | **No provenance — can't tell explicit facts from inferred patterns or outdated info** | **Confidence + provenance metadata on every memory** |
| 5 | No writeback — contradictions silently coexist | Conflict detection, explicit versioning, **no silent overwrites** |

And they claim **89.8% LongMemEval / 87.1% LoCoMo, beating Mem0, Zep, and Letta.**

You need to know this before you write another line of README copy. "Nobody tracks provenance" is no longer true, and repeating it will get the project dismissed by anyone who has read Memanto's front page.

**What survives, and it's a real distinction:**

> Memanto has provenance as **metadata**. OWL has provenance as an **enforced invariant.**

Metadata is annotation: it can be wrong, ignored, or overwritten, and nothing in the system prevents a summariser from writing `confidence: 0.95` on a node derived from a guess. OWL's rule is checked on every write, clamped automatically, and fuzz-tested in CI:

```
confidence(node) <= min(confidence(parents))
epistemic(node)  >= max(epistemic(parents))
```

The difference shows up exactly where it matters — after a hundred cycles of abstraction, when the question is whether anything can still be traced. It also produces capabilities annotation cannot: safe handover (epistemic demotion), `KNEW_ONCE`, and the guarantee that dream/inference content can never present as fact.

**But that's a subtler sell than "we have provenance and they don't."** Lead with what only OWL does: enforcement, the six FOK states, transactive memory, sealed partitions, and zero dependencies.

**Also worth noting:** Memanto's on-prem path requires Docker (the Moorcheh engine runs as a container). LycheeMem needs a server. MIRIX needs Docker + Postgres. **OWL's `pip install owl-engine` with no daemon, no container, and no model remains genuinely unmatched** in this set.

---

## 2. What each system contributes, and what OWL took

| System | The real contribution | Taken? |
|---|---|---|
| **Memanto** | Provenance/decay/conflict as a *product thesis*; 13 typed categories; zero-latency ingestion; OKF portable export | Validation, not mechanism. Their export-to-markdown idea is worth revisiting for `.owlpack` |
| **MIRIX** | **Knowledge Vault** — a separate store for content that must stay verbatim | ✅ **Built** (§3) |
| **LycheeMemory** | **Record Fusion Engine** — dedupe → union-find cluster → composite → hierarchy, *zero LLM calls*; decontextualisation; `failure_pattern` and `tool_affordance` types | ✅ **Fusion built** (§4). Decontextualisation is the biggest remaining gap (§6) |
| **MemoryLLM / M+** | Parametric memory *inside* the weights; and the **`nuc` evaluation** — inject N unrelated contexts between write and read | ✅ **Benchmark built** (§5). Parametric memory is not adoptable (needs training + GPU) |
| **MiniRAG** | Heterogeneous entity/chunk graph; and the finding that graph-RAG goes **negative** below ~7B | ✅ Built last round |
| **Zep/Graphiti** | Bi-temporality | ✅ Built |
| **HippoRAG** | Personalised PageRank over an entity graph | ✅ Built |
| **EM-LLM** | Event segmentation at surprise boundaries | ✅ Built |
| **A-MEM** | Supersede-not-overwrite | ✅ Built (arrived at independently) |
| **Letta/MemGPT** | Self-paging; agent-facing tool interface | ✗ Deliberately not — LLM-decided eviction is non-deterministic and untestable |

---

## 3. Verbatim protection *(from MIRIX's Knowledge Vault)*

MIRIX keeps a separate store for credentials and identifiers. The underlying insight generalises further than they take it:

> **Some content is worthless unless exact.** A grid reference, a serial, a dosage, a frequency, an account number. It must never be summarised, paraphrased, fused, or compressed — and a memory system that gets this wrong is *dangerous*, not merely unhelpful.

`verbatim` is now a claim class, detected at write time and treated as a **protection** rather than a category:

- infinite half-life — an exact string does not become less exact
- excluded from fusion entirely, whatever the cosine says
- excluded from composite construction
- (and when Tier 2 lands, excluded from reconstructive compression)

```python
classify("Grid 31U DQ 48251 11932")     # -> verbatim
classify("Give 250 mg every six hours") # -> verbatim
classify("Net control is 145.500 MHz")  # -> verbatim
classify("The clinic has twelve beds")  # -> capacity
```

For ATK specifically this is the difference between a tool and a liability. A summariser that helpfully rewrites "31U DQ 48251 11932" as "the northern grid square" has destroyed the only part that mattered.

---

## 4. Record fusion *(from LycheeMemory)*

Their Record Fusion Engine is the best-engineered thing I found in this batch, and its virtue is what it *doesn't* use:

1. **Dedupe** — cosine > 0.85, near-duplicates soft-expired
2. **Cluster** — cosine > 0.75, union-find over the similarity graph
3. **Composite** — each component becomes one denser node
4. **Hierarchy** — the same pass runs over composites, growing a tree upward

**Zero LLM calls. Pure arithmetic.** Which, given MiniRAG's finding that model-dependent pipelines invert below ~7B, is not a limitation — it's the correct engineering choice for local deployment.

OWL's interference sweep found confusable *pairs*. It now *resolves* them: `owl/fusion.py` implements the algorithm over OWL's vectors, with two additions —

- A composite is an ordinary `derived` node, so **monotonicity applies**: it can never be more certain than its least certain member, and `why()` walks into it normally. LycheeMem's composites carry no such guarantee.
- **Verbatim content is excluded from fusion**, per §3.

The whole algorithm takes `(a, b, similarity)` tuples rather than reaching into the store, so it's testable with five tuples and no database.

---

## 5. The interference benchmark *(from MemoryLLM)* — and a caught mistake

MemoryLLM's knowledge-retention eval has one parameter worth stealing: **`nuc`, the number of unrelated contexts injected between writing a fact and asking for it.** Almost nobody evaluates this way. LoCoMo and LongMemEval measure whether an answer can be found at all — not whether it survives being *buried*.

It also tests OWL's central claim directly: if interference beats decay, accuracy should fall faster for *confusable* distractors than for merely numerous ones.

**And the first run said the thesis was wrong.**

```
 distractors |  unrelated |  confusable
         200 |        40% |         80%     <- backwards
```

Volume appeared to beat interference. Before writing that up, I checked whether the benchmark measured what it claimed. It did not: **it was measuring the toy embedder.** At Tier 0, all five targets survive 200 distractors at 100%. The toy's eight hand-built concept axes saturate, and max-fusion then prefers the bad semantic score over the good lexical one.

The benchmark now runs both tiers and labels the toy row *"NOT a valid semantic model; diagnostic only"*, with the incident recorded in the source. It's exactly the failure the file exists to catch — **a benchmark that measures the harness rather than the system** — and it would have produced a confident, wrong, published claim that OWL's core thesis was refuted by its own numbers.

Current honest reading: **neither condition bites at Tier 0 by 200 distractors.** The lexical index isn't saturated yet. The thesis is untested, not confirmed. It needs a real ONNX embedder and larger `n` before anyone believes either answer.

---

## 6. What OWL is still missing

**Decontextualisation (LycheeMem) — the biggest remaining gap.** They expand pronouns and context-dependent phrases at write time, so every record is standalone: *"he said it'd arrive Thursday"* becomes *"Ahmed said the pump gasket would arrive Thursday 14 March."* OWL stores raw content, which means a large fraction of conversational memories are useless out of context. This is a genuine retrieval-quality deficit and it needs a model, so it's Tier 2 — but a partial deterministic version (resolving pronouns against the episode's entity mentions) is achievable at Tier 0 and worth trying first.

**`failure_pattern` as a first-class type (LycheeMem).** OWL has negative memory for *absence* ("I looked, it isn't there") but not for *failure* ("we tried this and it didn't work, here's why"). For an analyst toolkit that's arguably the more valuable of the two.

**Action-outcome loop (LycheeMem).** They log `retrieval_count`, `action_success_count`, and later user feedback, closing a lightweight loop without training. OWL has a `calibration` table but nothing writes to it. That's a dangling thread and should either be wired or removed.

**Portable markdown export (Memanto's OKF).** `.owlpack` is JSON, which is inspectable but not *readable*. A markdown rendering would make handover packs reviewable by a human before transfer — which matters, because the pack format's whole justification is that you can read what you're handing someone.

---

## 7. On "outperform everyone"

The philosophy is right. It needs one qualifier to stay useful.

**You cannot outperform this field on retrieval accuracy alone, and you shouldn't try.** Memanto claims 89.8% LongMemEval with an information-theoretic engine and a paid cloud tier behind it. LycheeMem has an ACL paper and a trained reranker. Those teams are optimising a metric full-time.

**What nobody is competing on is the epistemics** — enforcement rather than annotation, six honest answer states, modelling the user's knowledge rather than only the machine's, safe transplant between operators, and confidentiality as a store property. That's a defensible position because it comes from a different question: not *"can you find it?"* but *"should you believe it, and how do you know?"*

Concretely, I'd propose two things:

1. **Run LoCoMo and LongMemEval anyway** — not to win, but so the numbers are yours and honest. Expect to lose on raw recall; OWL retrieves 4–7 chunks by design.
2. **Propose the benchmarks nobody runs.** Confabulation rate on absent facts, source-attribution accuracy, calibration (Brier/ECE), and interference resistance under `nuc`. If OWL is right about what matters, the way to prove it is to publish the scoreboard the field is missing — and to run *other systems* on it. That's a genuine contribution independent of whether OWL wins.

The wheel worth not reinventing is retrieval. The wheel nobody has built is the scoreboard.
