"""B9 -- what the approximate index actually costs you.

The plan's acceptance criterion reads "identical results to brute force at
>=10x speed on 100k nodes". Half of that is unachievable by construction:
an approximate index is approximate. Reporting it as met would be the same
species of claim this project keeps catching in its own benchmarks.

So this measures both halves honestly and prints them together. Speed is
worth having only if you can see what you paid for it.

    python bench/ann_recall.py            10k, quick
    python bench/ann_recall.py --n 100000 the plan's actual scale (slow)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from owl import ann


def corpus(n, dim=64, seed=11, clusters=40):
    """Clustered, not uniform. Real embedding spaces are lumpy, and IVF's
    whole premise is that structure exists to exploit -- benchmarking it on
    uniform noise measures the worst case and calls it typical."""
    rng = random.Random(seed)

    def unit(v):
        m = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / m for x in v]

    cents = [unit([rng.gauss(0, 1) for _ in range(dim)])
             for _ in range(clusters)]
    items = []
    for i in range(n):
        c = cents[i % clusters]
        v = [0.75 * a + 0.25 * rng.gauss(0, 1) for a in c]
        items.append((f"n{i}", unit(v)))
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)

    print("=" * 70)
    print(f"ANN vs BRUTE FORCE  --  {args.n} vectors, k={args.k}")
    print("=" * 70)

    items = corpus(args.n)
    rng = random.Random(5)
    queries = [items[rng.randrange(len(items))][1] for _ in range(args.queries)]

    t = time.perf_counter()
    for q in queries:
        ann.brute(q, items, top_k=args.k)
    brute_ms = (time.perf_counter() - t) / len(queries) * 1000

    t = time.perf_counter()
    idx = ann.IvfIndex().build(items)
    build_s = time.perf_counter() - t
    print(f"  built {len(idx.centroids)} lists in {build_s:.1f}s")
    print(f"  brute force            {brute_ms:8.2f} ms/query   (exact)\n")

    print(f"  {'nprobe':>7} {'ms/query':>10} {'speedup':>9} "
          f"{'recall@k':>9} {'top-3':>7} {'top-1':>7}")
    print("  " + "-" * 56)
    best = None
    for p in (1, 2, 4, 8, 16, 32):
        if p > len(idx.centroids):
            break
        t = time.perf_counter()
        for q in queries:
            idx.search(q, top_k=args.k, nprobe=p)
        ms = (time.perf_counter() - t) / len(queries) * 1000
        r = ann.recall_at_k(idx, items, queries, k=args.k, nprobe=p)
        speed = brute_ms / ms if ms else 0
        print(f"  {p:7d} {ms:10.2f} {speed:8.1f}x {r['recall_at_k']:9.3f} "
              f"{r['top3_agreement']:7.3f} {r['top1_agreement']:7.3f}")
        # Keyed on top-3, because that is what OWL acts on. recall@10 is
        # the stricter number and is printed beside it, unhidden.
        if r["top3_agreement"] >= 0.95 and (best is None or speed > best[1]):
            best = (p, speed, r["top3_agreement"], r["recall_at_k"])

    print()
    if best:
        print(f"  At nprobe={best[0]}: {best[1]:.1f}x faster, "
              f"{best[2]:.1%} of the top-3 retained "
              f"(recall@{args.k} {best[3]:.1%}).")
        print("  The head survives; the tail does not. That is the trade, "
              "stated rather than buried.")
    else:
        print("  Nothing reached 95% top-3 agreement. At this corpus size "
              "brute force is the right")
        print("  answer, which is why it is the default.")
    print("\n  'Identical results at 10x speed' is not achievable -- an "
          "approximate index is\n  approximate. This is the honest version "
          "of that criterion: pick a row.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
