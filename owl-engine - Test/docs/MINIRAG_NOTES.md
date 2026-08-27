# Notes on MiniRAG (HKUDS, ACL 2026)

[arXiv:2501.06713](https://arxiv.org/abs/2501.06713) · [repo](https://github.com/HKUDS/MiniRAG)

What OWL took, what it left, and the finding in their paper that should
change how anyone builds a memory layer for local models.

---

## The finding that matters most is a negative one

MiniRAG's own benchmark table contains a result they state modestly and that
almost nobody quotes. Look at what happens to a sophisticated graph-RAG
pipeline as the model gets smaller:

| Model | NaiveRAG | LightRAG | MiniRAG |
|---|---|---|---|
| gpt-4o-mini (LiHua-World) | 46.6% | **56.9%** | 54.1% |
| Phi-3.5-mini (LiHua-World) | 41.2% | 39.8% | **53.3%** |
| Qwen2.5-3B (MultiHop) | 39.5% | **21.9%** | **48.6%** |
| GLM-Edge-1.5B (MultiHop) | 44.4% | *fails entirely* | **51.4%** |

With a capable model, LightRAG beats naive RAG comfortably. With a 3B model it
scores **half** what naive RAG does. With a 1.5B model it cannot produce usable
output at all.

> **Sophisticated retrieval pipelines that lean on model comprehension are
> negative value below a certain model size.** They don't degrade gracefully;
> they invert.

This is the single most important thing in the paper for OWL, and it is a
direct warning about **Tier 2**. Reconstructive compression, grafting, and
hypothesis generation all delegate judgement to the model. On a 16 GB box
running a quantised local model, "the clever version" may be *worse* than
doing nothing — and worse in a way that is invisible unless it's measured.

**Consequences adopted:**

- Tier 2 features ship behind flags, default off, with an ablation before
  they're recommended.
- Anything that can be done by topology or arithmetic is done that way, even
  when a model call would be easier to write.
- `tend()` reports what it skipped rather than silently degrading — so the
  operator can see which tier they are actually running.

This also retroactively justifies OWL's structure: the deterministic core
isn't a stepping stone to the model-powered version. For small local models it
may *be* the better version.

---

## What OWL took

### 1. Heterogeneous graph indexing — entities and observations in one structure

MiniRAG's core mechanism: put named entities and text chunks in a single
graph so retrieval walks topology instead of relying on semantic
understanding.

OWL had co-occurrence edges between observations, which connect things
retrieved together or sharing vocabulary. That cannot connect two notes five
weeks apart that concern the same person but share no wording — which is
exactly the multi-hop case field work produces constantly.

```python
mind.link(nid, mentions=[("Dr Warsame", "person")],
               relations=[("Dr Warsame", "signed", "cold chain log")])
```

The retrieval walk is two hops, and the **second hop is where the value is**.
One hop only recovers notes that already name the query entity — which
lexical search had anyway. Two hops reaches the note that mentions the *cold
chain log* and never mentions Warsame at all. Building this, the first version
did one hop and the test failed; the failure was the useful part.

### 2. Answer-type prediction

MiniRAG predicts what kind of thing the answer is and uses it to steer graph
traversal. OWL uses the same signal to sharpen the Feeling-of-Knowing gate: if
a query asks for a *person* and the store holds no person entity anywhere,
a strong lexical score is probably topical overlap rather than an answer.

Two guardrails, both tested:

- It **demotes, never vetoes.** The type predictor is a regex heuristic and
  must not be able to override a genuine hit.
- Absence of an entity graph returns `None`, not `False`. *A signal that is
  absent must not look like a signal that fired* — this is the easiest way to
  turn a heuristic into a bug.

### 3. Paths as dense context

A relationship path is far denser than the notes it came from. ATK's own
measurements put graph paths at roughly 300–600 tokens against 2000+ for
equivalent chunk RAG. On a slow local model that difference is most of the
latency budget, and it composes with OWL's 4–7 chunk cap, because a path *is*
a dense chunk.

```
Ahmed --[drives for]--> Warsame --[runs]--> Bardera clinic
```

---

## What OWL added that MiniRAG does not have

**Every edge carries its evidence.** In MiniRAG a relation is asserted by the
extractor and stands on its own. In OWL, `relation` has an `evidence_node`
column pointing at the observation that justifies it:

```python
step.evidence_node       # -> obs_...
mind._node_row(step.evidence_node)["source_ref"]   # -> "day9"
```

So a retrieved path is self-documenting, `why()` traverses it like any other
derivation, and a path can never assert a connection with no basis. For a
system whose entire pitch is "it can tell you how it knows", an unattributed
edge would be a hole straight through the middle.

---

## What OWL did not take

**The extraction pipeline.** MiniRAG uses the LLM for entity extraction.
OWL doesn't extract at all — `link()` accepts entities from the host. Three
reasons: extraction needs a model (so it can't live in Tier 0), it's
domain-specific, and hosts usually already do it. ATK builds an entity/
relationship link chart from every message it processes; that work should be
reused, not duplicated. ATK's decision to build graph memory over its own
extraction *rather than* adopting MiniRAG was the right call — a second model
and a duplicate graph store to get ideas you can implement in a few hundred
lines is a bad trade.

**The framework, the 10+ graph database backends, and the storage layer.**
OWL is one SQLite file and no dependencies. That is the product.

---

## Worth stealing later

**LiHua-World** — their benchmark dataset: a year of one person's chat records
with single-hop, multi-hop, and summary questions, built specifically for
on-device scenarios. That's a far better fit for OWL than LoCoMo or
LongMemEval, because it tests exactly the shape of memory a personal assistant
accumulates. It's also the only benchmark I've seen where the *summary*
questions would exercise OWL's period hierarchy.

