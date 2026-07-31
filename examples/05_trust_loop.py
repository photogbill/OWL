"""Trust as a living system. Tier 0 — no model, no GPU.

Three mechanisms that form one self-maintaining loop:
  source independence -- corroboration counts ORIGINS, not documents
  attributed belief   -- the claim is separate from the person making it
  commitments         -- promises resolve, and the outcome revalues everything
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from owl import Owl

DAY = 86400.0
class Clock:
    def __init__(s): s.t = 1_700_000_000.0
    def now(s): return s.t
    def advance(s, d): s.t += d * DAY

c = Clock()
with Owl.open(os.path.join(tempfile.mkdtemp(), "trust.owl"), clock=c) as m:

    print("=" * 70); print("1. SOURCE FLOODING EARNS NOTHING")
    print("=" * 70)
    flood = [m.observe("The depot is empty.", origin="document",
                       source_ref=f"file://attacker/doc{i}.pdf")
             for i in range(20)]
    r = m.independent_sources(flood)
    print(f"  20 documents asserting the same thing")
    print(f"     independent origins : {r['independent']}")
    print(f"     corroboration weight: {r['weight']}   ({r['note']})")

    real = [m.observe("The bridge at Km 42 is out.", origin="document",
                      source_ref="file://survey/report.pdf"),
            m.observe("The bridge at Km 42 is out.", origin="user_utterance",
                      source_ref="conv:ahmed:3"),
            m.observe("The bridge at Km 42 is out.", origin="document",
                      source_ref="https://reliefweb.int/sitrep/9")]
    r = m.independent_sources(real)
    print(f"\n  3 documents from genuinely different places")
    print(f"     independent origins : {r['independent']}")
    print(f"     corroboration weight: {r['weight']}")
    print(f"     {r['clusters']}")

    print("\n" + "=" * 70); print("2. WHO SAID IT")
    print("=" * 70)
    n = m.observe("Ahmed reports the depot restocks every Tuesday.",
                  origin="user_utterance", source_ref="conv:ahmed:1",
                  reliability="B", credibility=2)
    m.claimed("Ahmed", "the depot restocks every Tuesday", node_id=n)
    for w in m.who_claims("the depot restocks every Tuesday"):
        print(f"  {w['who']}  grade={w['grade']}  accuracy={w['accuracy']}")
    print(f"  effective grade of that source: {m.effective_grade(n)}")

    print("\n" + "=" * 70); print("3. PROMISES RESOLVE")
    print("=" * 70)
    for i in range(4):
        cm = m.committed("Ahmed", f"deliver fuel batch {i}",
                         due=c.now() + DAY, node_id=n)
        c.advance(2)
        due = m.due_commitments()
        rec = m.resolve_commitment(cm, kept=False, note="did not arrive")
        print(f"  promise {i}: BROKEN   -> {rec.describe()}")

    print("\n" + "=" * 70); print("4. THE LOOP CLOSES")
    print("=" * 70)
    print(f"  Ahmed's grade is now derived from outcomes, not assigned.")
    print(f"  Every source he spoke through was revalued automatically:")
    print(f"     effective grade of conv:ahmed:1 : {m.effective_grade(n)}")
    print(f"     (it was B/2 before he broke four promises)")
    print()
    print("  Nobody else closes this. Memanto has `commitment` as a memory")
    print("  TYPE; the lifecycle is where the value is.")
