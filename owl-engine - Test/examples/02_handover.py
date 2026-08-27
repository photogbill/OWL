"""Handover: inheriting the ledger of the person who was there before you.

Still Tier 0 -- no model, no GPU, no dependencies.
"""
import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from owl import Owl, State

DAY = 86400.0
class Clock:
    def __init__(s): s.t = 1_700_000_000.0
    def now(s): return s.t
    def advance(s, days): s.t += days * DAY

tmp = tempfile.mkdtemp()
pack_path = os.path.join(tmp, "bardera.owlpack")

# ── Operator 1: Ferrand, six months in the field ────────────────────────
c1 = Clock()
with Owl.open(os.path.join(tmp, "ferrand.owl"), clock=c1) as m:
    m.partition("work")
    m.partition("private", sealed=True)

    a = m.observe("North well pump needs a 40mm gasket; parts via Kismayo depot.",
                  origin="user_utterance", source_ref="day31", partition="work",
                  reliability="B", credibility=2)
    m.observe("Dr Warsame runs the Bardera clinic and speaks English and Somali.",
              origin="user_utterance", source_ref="day3", partition="work",
              reliability="A", credibility=1)
    m.observe("Route Alpha floods above 40mm rainfall; use the Km-58 track.",
              origin="user_utterance", source_ref="day44", partition="work",
              reliability="B", credibility=2)
    m.derive("The clinic's water problem is upstream of the outbreak, not caused by it.",
             parents=[a], kind="hypothesis", producer="ferrand-analysis",
             partition="work",
             falsifier="compare clinic intake dates against the pump repair log")
    m.tell("ferrand", a, channel="generated")
    m.record_absence("diesel supplier in bardera", scope="local vendors",
                     partition="work",
                     reason="canvassed all six vendors, none stock diesel")
    m.intend("re-test the well once the gasket is fitted",
             on_event="gasket delivery", partition="work")
    m.observe("Some nights I don't cope with this well.", partition="private",
              affect=0.9)

    man = m.export_pack(pack_path, partition="work", exporter="ferrand",
                        label="Bardera — 6 month handover",
                        notes="Water, clinic, routes. Diesel is a dead end.")

print("=" * 72); print("1. THE PACK — plain JSON, readable before you send it")
print("=" * 72)
print(f"  {json.dumps(man['counts'])}")
print(f"  exporter={man['exporter']}  checksum={man['checksum'][:16]}...")
try:
    with Owl.open(os.path.join(tmp, "f2.owl")) as x:
        pass
except Exception:
    pass
print("  sealed 'private' partition: not in the pack, and cannot be forced")

# ── Operator 2: lands today, knows nobody ───────────────────────────────
c2 = Clock()
with Owl.open(os.path.join(tmp, "bill.owl"), clock=c2) as m:
    m.partition("work")

    print("\n" + "=" * 72); print("2. DRY RUN — a handover is a trust decision")
    print("=" * 72)
    for k, v in m.inspect_pack(pack_path).items():
        print(f"  {k:26s} {v}")

    print("\n" + "=" * 72); print("3. GRAFT — every tag shifts down one rank")
    print("=" * 72)
    stats = m.graft(pack_path, as_source="prev:ferrand", partition="work")
    for k, v in stats.items():
        print(f"  {k:16s} {v}")

    print("\n" + "=" * 72); print("4. WHAT DAY ONE NOW LOOKS LIKE")
    print("=" * 72)
    for q in ["who runs the clinic",
              "north well pump gasket",
              "route alpha flooding",
              "diesel supplier in bardera"]:
        r = m.recall(q, partition="work")
        print(f"\n  Q: {q}")
        print(f"     {r.state.value.upper()}")
        if r.chunks and r.chunks[0].content:
            c = r.chunks[0]
            print(f"       {c.content[:60]}")
            print(f"       src={c.provenance.source_ref[:44]}")
            print(f"       reliability={c.reliability}/{c.credibility} "
                  f"(downgraded: we weren't there)")
        else:
            print(f"       {r.reason}")

    print("\n" + "=" * 72); print("5. WHAT DIDN'T COME ACROSS")
    print("=" * 72)
    hyp = m.recall("water problem upstream outbreak", partition="work")
    print(f"  Ferrand's hypothesis: {hyp.state.value}")
    print("  -> his guess is nothing to me. Dropped, not inherited as fact.")
    print(f"\n  Open loops inherited: "
          f"{m._scalar('SELECT COUNT(*) FROM intention')}")
    print(f"  Dead ends inherited : "
          f"{m._scalar('SELECT COUNT(*) FROM absence')}  "
          f"(no re-canvassing six vendors)")
