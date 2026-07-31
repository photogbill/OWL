"""Theory of Mind + epistemic half-life. Still Tier 0 -- no model, no GPU.

Every memory system in the field models what the MACHINE knows.
This models what the PERSON knows, and where the two have diverged.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from owl import Owl, State

DAY = 86400.0
class Clock:
    def __init__(s): s.t = 1_700_000_000.0
    def now(s): return s.t
    def advance(s, days): s.t += days * DAY

clock = Clock()
with Owl.open(os.path.join(tempfile.mkdtemp(), "tom.owl"), clock=clock) as m:
    m.partition("work")

    brief = m.observe("Checkpoint protocol: radio ahead, headlights off at 200m.",
                      origin="document", source_ref="security-brief-v3",
                      partition="work", reliability="B", credibility=2)
    route = m.observe("Route Alpha is open.", origin="document",
                      source_ref="sitrep-1", partition="work",
                      claim_class="status", reliability="C", credibility=3)
    name  = m.observe("Dr Warsame runs the Bardera clinic and speaks Somali.",
                      origin="user_utterance", source_ref="day1",
                      partition="work")

    m.tell("bill", brief, channel="briefing")      # skimmed once
    m.tell("bill", route, channel="conversation")  # discussed
    m.tell("bill", name,  channel="generated")     # he said it himself

    print("=" * 70); print("1. TRANSACTIVE MEMORY — modelling HIS forgetting")
    print("=" * 70)
    for label, nid in [("checkpoint protocol (skimmed)", brief),
                       ("route status (discussed)", route),
                       ("Dr Warsame (he said it)", name)]:
        print(f"  day 0   {label:32s} retention={m.knows('bill', nid).retrievability:.2f}")
    clock.advance(21)
    print()
    for label, nid in [("checkpoint protocol (skimmed)", brief),
                       ("route status (discussed)", route),
                       ("Dr Warsame (he said it)", name)]:
        h = m.knows("bill", nid)
        print(f"  day 21  {label:32s} retention={h.retrievability:.2f}"
              f"{'   <- AT RISK' if h.at_risk else ''}")
    print("\n  The system still holds all three perfectly. That's the point:")
    print("  the memory dynamics worth modelling are the human's, not its own.")

    print("\n" + "=" * 70); print("2. EPISTEMIC HALF-LIFE — findable vs still true")
    print("=" * 70)
    for q in ["checkpoint protocol", "route alpha", "warsame clinic"]:
        c = m.recall(q, partition="work").chunks[0]
        verdict = "TRUST" if c.trustworthy else "STALE"
        print(f"  {verdict:5s} {c.claim_class:9s} findable={c.retrievability:.2f} "
              f"stale={c.staleness:.2f}  {c.content[:38]}")
    print("\n  'Route Alpha is open' is still perfectly findable and almost")
    print("  certainly no longer true. No other system separates these.")

    print("\n" + "=" * 70); print("3. FALSE BELIEF — Sally-Anne, made operational")
    print("=" * 70)
    m.observe("Route Alpha is closed by flooding.", origin="document",
              source_ref="sitrep-2", partition="work", claim_class="status",
              supersedes=route, reliability="B", credibility=2)
    for d in m.divergence("bill", partition="work"):
        print(f"  [{d.direction}] severity={d.severity:.2f}")
        print(f"    he holds : {m._node_row(d.held_node)['content']}")
        print(f"    record   : {m._node_row(d.truth_node)['content']}")
        print(f"    {d.note}")
    print("\n  Computed from the exposure log and the bitemporal record.")
    print("  No model call. And it resolves SYMMETRICALLY — if he'd been at")
    print("  the checkpoint an hour ago, the RECORD would be flagged stale.")

    print("\n" + "=" * 70); print("4. ANSWERS THAT AREN'T 'I DON'T KNOW'")
    print("=" * 70)
    m.record_absence("diesel supplier in bardera", scope="local vendors",
                     partition="work",
                     reason="canvassed all six vendors, none stock diesel")
    old = m.observe("Warehouse alarm bypass is star-seven-four.",
                    origin="document", source_ref="file://handover.pdf#p9",
                    partition="work")
    m._s.write(lambda c: c.execute(
        "UPDATE mem_index SET tier='pruned' WHERE node_id=?", (old,)))

    for q in ["diesel supplier in bardera", "warehouse alarm bypass",
              "helicopter tail number"]:
        r = m.recall(q, partition="work")
        print(f"  {r.state.value.upper():20s} {q}")
        print(f"    -> {r.reason}")
