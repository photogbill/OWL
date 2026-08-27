"""Tier 0 demo: no model, no embeddings, no GPU, no dependencies.

Scenario: an aid worker's first three days somewhere they know nobody.
"""
import sys, tempfile, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from owl import Owl, State

DAY = 86400.0


class Clock:
    def __init__(self): self.t = 1_700_000_000.0
    def now(self): return self.t
    def advance(self, days): self.t += days * DAY


clock = Clock()
path = os.path.join(tempfile.mkdtemp(), "field.owl")

with Owl.open(path, clock=clock) as mind:
    # Work and companion memory are separate universes.
    mind.partition("work")
    mind.partition("athena", sealed=True)

    with mind.period("bardera-deployment", partition="work"):
        for note in [
            "Clinic at Bardera has 12 beds and one oxygen concentrator.",
            "Oxygen concentrator at the clinic is intermittent; needs a filter.",
            "Dr Warsame runs the Bardera clinic and speaks English and Somali.",
            "Fuel for the clinic generator comes from the depot on Route Alpha.",
            "Route Alpha is open as of this morning.",
            "Entirely different: the vaccine cold chain fridge logged an excursion.",
            "Cold chain excursion Tuesday night may have spoiled the measles batch.",
        ]:
            mind.observe(note, origin="user_utterance",
                         source_ref="fieldnotes/day1", partition="work")

    mind.observe("I did not sleep. I keep going back over the triage decisions.",
                 origin="user_utterance", source_ref="athena/night1",
                 partition="athena", affect=0.8)

    print("=" * 68)
    print("1. THREE STATES OF KNOWING")
    print("=" * 68)
    for q in ["who runs the clinic",
              "oxygen concentrator filter",
              "what is the helicopter tail number"]:
        r = mind.recall(q, partition="work")
        print(f"\n  Q: {q}")
        print(f"     -> {r.state.value.upper():14s} ({r.latency_ms:.1f} ms)  {r.reason}")
        for c in r.chunks[:2]:
            print(f"        - {c.content[:58]}")
            print(f"          src={c.provenance.source_ref} "
                  f"epistemic={c.provenance.epistemic.value} R={c.retrievability:.2f}")

    print("\n" + "=" * 68)
    print("2. CONFIDENTIALITY BOUNDARY (enforced by the store)")
    print("=" * 68)
    print(f"  work   -> 'triage decisions' : "
          f"{mind.recall('triage decisions sleep', partition='work').state.value}")
    print(f"  athena -> 'triage decisions' : "
          f"{mind.recall('triage decisions sleep', partition='athena').state.value}")

    print("\n" + "=" * 68)
    print("3. PROVENANCE — 'how do you know that?'")
    print("=" * 68)
    obs = mind.recall("route alpha", partition="work").chunks[0].node_id
    concl = mind.derive("Fuel resupply is currently viable.", parents=[obs],
                        kind="abstraction", producer="analyst",
                        confidence=0.9, partition="work")
    hyp = mind.derive("The clinic can sustain operations for two weeks.",
                      parents=[concl], kind="hypothesis", producer="rem-phase",
                      falsifier="Check depot stock levels against daily burn rate")
    for node in mind.why(hyp):
        flag = "FACT " if node["presentable_as_fact"] else "*NOT FACT*"
        print(f"  {flag} [{node['epistemic']:<12}] conf={node['confidence']:.2f} "
              f"{node['content'][:46]}")
    print("\n  Note: the hypothesis could not be created without a falsifier,")
    print("  and its confidence was clamped to its parent's ceiling.")

    print("\n" + "=" * 68)
    print("4. DECAY — 400 days later")
    print("=" * 68)
    before = mind.recall("oxygen concentrator", partition="work")
    clock.advance(400)
    report = mind.tend(partition="work")
    after = mind.recall("oxygen concentrator", partition="work")
    print(f"  before: {before.state.value}   after 400 idle days: {after.state.value}")
    print(f"  index tiers: {report['tiers']}   retiered: {report['retiered']}")
    print(f"  substrate rows still present: "
          f"{mind._scalar('SELECT COUNT(*) FROM observation')}  <- nothing deleted")
    print(f"  skipped (no model): {report['skipped']}")

    print("\n" + "=" * 68)
    d = mind.doctor()
    print(f"5. DOCTOR: healthy={d['healthy']} tier={d['tier']} "
          f"observations={d['observations']} episodes={d['episodes']}")
    print("=" * 68)
