"""Tier 1: what semantics buys you, and what it doesn't.

Uses the toy embedder from the test suite so this runs with no model and no
downloads. Swap in OnnxEmbedder or STEmbedder for real work.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests"))
from owl import Owl, State
from owl.vectors import unpack, dot
from test_semantic import ToyEmbedder

DAY = 86400.0
class Clock:
    def __init__(s): s.t = 1_700_000_000.0
    def now(s): return s.t
    def advance(s, days): s.t += days * DAY

tmp = tempfile.mkdtemp()
NOTES = [
    "The clinic generator runs on depot fuel.",
    "North well pump needs a 40mm gasket.",
    "Route Alpha is open as of this morning.",
]

print("=" * 70); print("1. THE PARAPHRASE GAP")
print("=" * 70)
q = "how is the health facility powered"
for label, emb in [("Tier 0 (lexical)", None), ("Tier 1 (semantic)", ToyEmbedder())]:
    with Owl.open(os.path.join(tmp, f"{label[5]}.owl"), clock=Clock(),
                  embedder=emb) as m:
        for n in NOTES:
            m.observe(n, source_ref="survey")
        r = m.recall(q)
        print(f"  {label:20s} {r.state.value.upper():12s} {r.reason[:44]}")
print(f'\n  query: "{q}"')
print('  target: "The clinic generator runs on depot fuel."')
print("  Zero shared content words. This is the gap Tier 0 cannot close.")

print("\n" + "=" * 70); print("2. PATTERN SEPARATION — the write space")
print("=" * 70)
c = Clock()
with Owl.open(os.path.join(tmp, "sep.owl"), clock=c,
              embedder=ToyEmbedder()) as m:
    a = m.observe("Weekly supply meeting held; stock levels reviewed.",
                  source_ref="week1")
    c.advance(7)
    b = m.observe("Weekly supply meeting held; stock levels reviewed.",
                  source_ref="week2")
    def v(nid, sp):
        return unpack(m._s.one("SELECT data FROM vector WHERE node_id=? "
                               "AND space=?", (nid, sp))["data"])
    print(f"  identical text, different weeks:")
    print(f"    READ  space similarity = {dot(v(a,'read'),  v(b,'read')):.3f}"
          "   (meaning: same)")
    print(f"    WRITE space similarity = {dot(v(a,'write'), v(b,'write')):.3f}"
          "   (episodes: distinct)")
    print("\n  Standard RAG has only the first number, so the two collapse")
    print("  together and cause the interference it then works around.")

print("\n" + "=" * 70); print("3. WHERE EMBEDDINGS ARE WEAK")
print("=" * 70)
with Owl.open(os.path.join(tmp, "id.owl"), clock=Clock(),
              embedder=ToyEmbedder()) as m:
    m.observe("Generator serial is GX-4419.", source_ref="asset-reg")
    m.observe("Generator serial is GX-4491.", source_ref="asset-reg")
    top = m.recall("GX-4419", budget=2).chunks[0]
    print(f"  query 'GX-4419' -> {top.content}")
    print("  No embedding model reliably separates GX-4419 from GX-4491.")
    print("  Fusion is max-of-normalised, not a weighted sum: either signal")
    print("  firing hard is good evidence, and averaging them would lose both.")

print("\n" + "=" * 70); print("4. GROWING INTO IT")
print("=" * 70)
path = os.path.join(tmp, "grow.owl")
with Owl.open(path, clock=Clock()) as m:
    for n in NOTES:
        m.observe(n, source_ref="survey")
    print(f"  ran at tier {m.tier} for a month, {m._scalar('SELECT COUNT(*) FROM observation')} observations")
with Owl.open(path, clock=Clock(), embedder=ToyEmbedder()) as m:
    print(f"  attached an embedder -> tier {m.tier}, "
          f"doctor flags {len(m.doctor()['problems'])} problem(s)")
    print(f"  reindex() backfilled {m.reindex()} nodes; "
          f"{m._vec.count()} vectors, {len(m.doctor()['problems'])} problems")
    print("  Nothing had to be re-ingested. The substrate was always complete.")
