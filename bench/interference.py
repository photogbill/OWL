"""Interference-resistance benchmark.

Methodology borrowed from MemoryLLM's knowledge-retention evaluation, whose
key parameter is `nuc` -- the *number of unrelated contexts* injected between
writing a fact and asking for it. That is the right way to test a memory
system, and almost nobody does it: LoCoMo and LongMemEval measure whether the
answer can be found at all, not whether it survives being buried.

It also happens to test OWL's central claim directly. OWL asserts that
INTERFERENCE, not decay, is what kills retrieval (Underwood 1957 onward). If
that is right, accuracy should fall as confusable neighbours accumulate --
and should fall *much faster* for confusable distractors than for merely
numerous ones. If both curves look the same, the claim is wrong and the
de-interference machinery is not earning its place.

Run:  python bench/interference.py
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests"))

from owl import Owl, State

DAY = 86400.0


class Clock:
    def __init__(self): self.t = 1_700_000_000.0
    def now(self): return self.t
    def advance(self, days=0.0, hours=0.0): self.t += days * DAY + hours * 3600


TARGETS = [
    ("The generator serial is GX-4419.", "generator serial"),
    ("Dr Warsame runs the Bardera clinic.", "who runs the Bardera clinic"),
    ("The depot access code is 4471.", "depot access code"),
    ("Route Alpha floods above 40mm rainfall.", "route alpha flooding threshold"),
    ("The clinic has twelve beds.", "how many beds does the clinic have"),
]

# Distractors that share NO vocabulary with the targets.
UNRELATED = [
    "The radio mast was repainted last spring.",
    "Coffee supplies were restocked on Thursday.",
    "The perimeter fence needs new wire on the east side.",
    "Two volunteers arrived from the regional office.",
    "The satellite uplink was tested and works.",
]

# Distractors deliberately confusable with the targets -- same shape, same
# vocabulary, different content. This is the condition that should hurt.
CONFUSABLE = [
    "The generator serial is GX-4491.",
    "Dr Warsan runs the Baardheere clinic.",
    "The depot access code is 4417.",
    "Route Bravo floods above 40mm rainfall.",
    "The clinic has twenty beds.",
]


def run(n_distractors: int, *, confusable: bool, embedder=None,
        tend: bool = False, seed: int = 0) -> float:
    rng = random.Random(seed)
    clock = Clock()
    path = os.path.join(tempfile.mkdtemp(), "bench.owl")
    pool = CONFUSABLE if confusable else UNRELATED
    hits = 0
    with Owl.open(path, clock=clock, embedder=embedder) as mind:
        for text, _ in TARGETS:
            mind.observe(text, source_ref="target")
            clock.advance(hours=1)
        for i in range(n_distractors):
            base = pool[i % len(pool)]
            mind.observe(f"{base} (entry {i})", source_ref=f"noise{i}")
            clock.advance(hours=1)
        if tend:
            mind.tend()
        for text, query in TARGETS:
            r = mind.recall(query, budget=3)
            if r.chunks and r.chunks[0].content.startswith(text[:24]):
                hits += 1
    return hits / len(TARGETS)


def _table(label: str, embedder, tend: bool = False) -> list[tuple]:
    print(f"\n{label}")
    header = (f"{'distractors':>12} | {'unrelated':>10} | {'confusable':>11} "
              f"| {'delta':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for n in (0, 10, 25, 50, 100, 200):
        u = run(n, confusable=False, embedder=embedder, tend=tend)
        c = run(n, confusable=True, embedder=embedder, tend=tend)
        rows.append((n, u, c))
        print(f"{n:>12} | {u:>10.0%} | {c:>11.0%} | {c - u:>+7.0%}")
    return rows


def main() -> int:
    print("=" * 68)
    print("INTERFERENCE-RESISTANCE  (MemoryLLM `nuc` methodology)")
    print("=" * 68)
    print("top-1 exact-target accuracy over 5 planted facts")

    t0 = _table("TIER 0 -- lexical only, no model", None)

    # A first version of this benchmark ran only with the toy embedder from
    # the test suite and reported that VOLUME beat INTERFERENCE -- i.e. that
    # OWL's central thesis was wrong. It was measuring the toy embedder.
    # Tier 0 answers all five targets under 200 distractors; the toy's eight
    # hand-built concept axes simply saturate, and max-fusion then takes the
    # (bad) semantic score over the (good) lexical one.
    #
    # Recorded here because it is the exact failure mode this file exists to
    # catch: a benchmark that measures the harness rather than the system.
    # A real ONNX embedder is required before the Tier 1 row means anything.
    try:
        from test_semantic import ToyEmbedder
        _table("TIER 1 -- TOY embedder (NOT a valid semantic model; "
               "diagnostic only)", ToyEmbedder())
    except Exception:                                     # noqa: BLE001
        print("\n  (toy embedder unavailable; skipping diagnostic row)")

    print("\nInterpretation  [Tier 0]")
    print("-" * 68)
    u_drop = t0[0][1] - t0[-1][1]
    c_drop = t0[0][2] - t0[-1][2]
    print(f"  unrelated  distractors cost {u_drop:+.0%} accuracy")
    print(f"  confusable distractors cost {c_drop:+.0%} accuracy")
    if c_drop > u_drop + 0.1:
        print("\n  -> Interference dominates volume, as the design predicts.")
        print("     Age-based pruning would not help here: the store is not")
        print("     old, it is CONFUSED. That is what the de-interference")
        print("     sweep and pattern separation exist to address.")
    elif c_drop <= u_drop and c_drop == 0:
        print("\n  -> Neither condition bites at this scale. The lexical index")
        print("     is not yet saturated. Raise n, or move to a real embedder")
        print("     where semantic collapse is the binding constraint.")
    elif c_drop <= u_drop:
        print("\n  -> Volume dominates, NOT interference. If this reproduces")
        print("     with a real embedder and real data, the interference")
        print("     thesis is wrong for this workload and the machinery")
        print("     should be cut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
