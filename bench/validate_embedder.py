"""Validate a real embedder end-to-end. RUN THIS ON YOUR MACHINE.

    python bench/validate_embedder.py "embedding model/bge-m3-Q6_K.gguf"

Every Tier 1 number OWL reports is meaningless until this passes with a real
model. The toy embedder in the test suite exists only to exercise plumbing --
it once "disproved" OWL's central thesis because a benchmark was measuring the
toy rather than the system.

Checks, in order of what they would catch:
  1. dimensionality and normalisation
  2. paraphrase recall  -- the gap Tier 0 cannot close
  3. discrimination     -- unrelated text must NOT match
  4. pattern separation -- identical text, different context, must land apart
  5. identifier recall  -- max-fusion must not lose GX-4419 to GX-4491
  6. cross-lingual      -- BGE-M3 is multilingual; verify it, don't assume
  7. throughput
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from owl import Owl, State
from owl import metamemory
from owl.protocols import Space
from owl.vectors import dot, pack, unpack

DAY = 86400.0


class Clock:
    def __init__(self): self.t = 1_700_000_000.0
    def now(self): return self.t
    def advance(self, days=0.0): self.t += days * DAY


PARAPHRASE = [
    ("The clinic generator runs on depot fuel.",
     "how is the health facility powered"),
    ("North well pump needs a 40mm gasket.",
     "what part is the borehole missing"),
    ("Dr Warsame runs the Bardera clinic and speaks Somali.",
     "who is in charge of the medical centre"),
    ("Route Alpha floods above 40mm rainfall.",
     "which road becomes impassable in heavy rain"),
]

UNRELATED_QUERIES = [
    "what is the helicopter tail number",
    "when does the school term start",
    "how much does a satellite phone cost",
]

CROSS_LINGUAL = [
    # Deliberately NOT about the north well. An earlier fixture used "the
    # water pump at the north well is broken", which is a near-duplicate of
    # the borehole paraphrase target -- so "what part is the borehole
    # missing" had two defensible answers and the test was measuring an
    # ambiguity I had built.
    ("The vaccine cold chain refrigerator logged an excursion.",
     "le réfrigérateur de la chaîne du froid a enregistré un écart"),
    ("The clinic has twelve beds.", "la clínica tiene doce camas"),
]

# A gate that judges a match against the DISTRIBUTION of its competitors
# needs a distribution. Eight documents is not one; real stores hold
# hundreds. This filler is ordinary operational prose, unrelated to every
# probe, and exists so the benchmark measures the system rather than the
# smallness of the fixture.
FILLER = [
    "The perimeter fence needs new wire on the eastern approach.",
    "Two volunteers arrived from the regional office on Monday.",
    "Coffee and dry goods were restocked from the market.",
    "The satellite uplink was tested and reached the relay.",
    "Tent seventeen was re-pegged after the wind.",
    "The radio mast was repainted during the dry spell.",
    "Waste disposal moved to the far side of the compound.",
    "A new logbook was opened for vehicle movements.",
    "The water bladder was cleaned and refilled.",
    "Solar panels on block C were wiped down.",
    "The gate roster was revised for the holiday week.",
    "Spare tyres were counted and recorded as four.",
    "The generator run-hours meter was photographed.",
    "Latrine servicing is now contracted weekly.",
    "The staff notice board was moved indoors.",
    "Drainage was cleared along the northern wall.",
]


def calibrate(model_path: str) -> int:
    """Measure what the gate parameters SHOULD be for this encoder.

    Gate thresholds are encoder-specific. Hard-coding them from four data
    points -- or worse, from a mock -- is guesswork. This measures the real
    distributions on a realistic corpus and prints the numbers to use.
    """
    from owl.adapters import gguf_embed
    if not gguf_embed.available():
        print("  [BLOCKED] llama-cpp-python is not installed. Run install.bat")
        return 3
    from owl.vectors import unit

    print("=" * 70)
    print("ENCODER CALIBRATION SWEEP")
    print("=" * 70)
    emb = gguf_embed.GgufEmbedder(model_path, n_ctx=2048)
    corpus = [t for t, _ in PARAPHRASE] + [t for t, _ in CROSS_LINGUAL] + FILLER
    C = [unit(v) for v in emb.embed(corpus, Space.WRITE)]
    print(f"  corpus: {len(corpus)} documents\n")

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    rows = []
    # Every query->doc cosine EXCEPT the intended target. This is the
    # population the recall gate is actually judging against, measured the
    # way retrieval runs: prefixed query, bare document.
    qd_background: list[float] = []
    for target, q in PARAPHRASE:
        qv = unit(emb.embed([q], Space.READ)[0])
        sims = sorted(((cos(qv, c), t) for c, t in zip(C, corpus)),
                      reverse=True)
        best = sims[0][0]
        hit = next(s for s, t in sims if t == target)
        tail = [s for s, _ in sims[int(len(sims) * 0.4):]]
        bg = sum(tail) / len(tail)
        qd_background += [s for s, t in sims if t != target]
        rows.append(("RELATED", q, hit, best, bg))
    for q in UNRELATED_QUERIES:
        qv = unit(emb.embed([q], Space.READ)[0])
        sims = sorted((cos(qv, c) for c in C), reverse=True)
        tail = sims[int(len(sims) * 0.4):]
        bg = sum(tail) / len(tail)
        qd_background += sims          # nothing here is a true match
        rows.append(("UNRELATED", q, sims[0], sims[0], bg))

    print(f"  {'kind':10s} {'target':>7} {'best':>7} {'bg':>7} {'margin':>7}  query")
    print("  " + "-" * 74)
    rel_m, unrel_m = [], []
    for kind, q, hit, best, bg in rows:
        margin = hit - bg
        (rel_m if kind == "RELATED" else unrel_m).append(margin)
        print(f"  {kind:10s} {hit:7.3f} {best:7.3f} {bg:7.3f} {margin:7.3f}  "
              f"{q[:34]}")

    print()
    rel_levels = [hit for k, _, hit, _, _ in rows if k == "RELATED"]
    unrel_levels = [hit for k, _, hit, _, _ in rows if k == "UNRELATED"]
    print(f"  related levels     {min(rel_levels):.3f} .. "
          f"{max(rel_levels):.3f}")
    print(f"  unrelated levels   {min(unrel_levels):.3f} .. "
          f"{max(unrel_levels):.3f}")
    print(f"  related margins    {min(rel_m):.3f} .. {max(rel_m):.3f}")
    print(f"  unrelated margins  {min(unrel_m):.3f} .. {max(unrel_m):.3f}")

    from owl.adapters import calibration as cal
    import time as _t

    # Two backgrounds on two scales. Judging query->doc scores against
    # doc->doc pairs measures the query prefix, not the model -- it once
    # reported -0.169 headroom for an encoder that was working.
    qbg, qbg95 = cal.background(qd_background)
    daniso, daniso95 = cal.anisotropy(C)
    npairs = len(C) * (len(C) - 1) // 2
    print(f"\n  query -> doc, unrelated   mean {qbg:.3f}   p95 {qbg95:.3f}"
          f"   ({len(qd_background)} pairs)   <- the recall gate's zero")
    print(f"  doc   -> doc, unrelated   mean {daniso:.3f}   p95 "
          f"{daniso95:.3f}   ({npairs} pairs)   <- fusion's zero")
    auc = cal.separability(rel_levels, qd_background)
    print(f"  usable headroom           {min(rel_levels) - qbg95:+.3f}"
          "   (worst case: weakest true match over query->doc p95)")
    print(f"  separability AUC          {auc:.3f}"
          f"   (all {len(rel_levels) * len(qd_background)} comparisons)")

    c = cal.derive(model=emb.meta.name, corpus_size=len(corpus),
                   related=rel_levels, unrelated=unrel_levels,
                   rel_margins=rel_m, unrel_margins=unrel_m, now=_t.time(),
                   aniso=qbg, aniso_p95=qbg95,
                   doc_aniso=daniso, doc_aniso_p95=daniso95, auc=auc)
    print()
    for n in c.notes:
        print(f"  - {n}")
    print(f"\n  MEASURED PARAMETERS")
    print(f"    noise_floor   {c.noise_floor}")
    print(f"    search_floor  {c.search_floor}")
    print(f"    level_weight  {c.level_weight}    (separator: {c.separator})")
    written = c.save(model_path)
    print(f"\n  written to {written.name}")
    print("  The adapter loads this automatically from now on -- these are "
          "properties")
    print("  of the encoder, not of the engine, so they travel with the "
          "model file.")
    return 0


def main(model_path: str) -> int:
    print("=" * 70)
    print("EMBEDDER VALIDATION")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"  [FAIL] no such file: {model_path}")
        print("\n  Pass the path to a .gguf embedding model, e.g.")
        print('    python bench/validate_embedder.py "embedding model/bge-m3-Q6_K.gguf"')
        return 2

    from owl.adapters import gguf_embed
    if not gguf_embed.available():
        print("  [BLOCKED] llama-cpp-python is not installed in this "
              "interpreter.\n")
        print(f"  running under: {sys.prefix}")
        if sys.prefix == sys.base_prefix:
            print("  ...which is SYSTEM Python, not a venv.\n")
            print("  Fix (recommended) -- set up the isolated environment:")
            print("      install.bat                 (Windows)")
            print("      ./install.sh                (Linux / macOS)")
            print("  then:")
            print('      validate.bat "embedding model\\bge-m3-Q6_K.gguf"')
        else:
            print("\n  Fix -- install a PREBUILT wheel into this venv:")
            print("      python -m pip install llama-cpp-python \\")
            print("        --extra-index-url "
                  "https://abetlen.github.io/llama-cpp-python/whl/cpu")
            print("      (swap cpu for cu124 / cu121 / metal as appropriate;")
            print("       `python scripts/detect_backend.py --explain` picks)")
        print("\n  Tier 0 is unaffected and needs none of this:")
        print("      python -m pytest tests -q")
        return 3

    GgufEmbedder = gguf_embed.GgufEmbedder
    t0 = time.time()
    try:
        emb = GgufEmbedder(model_path, n_ctx=2048)
        print(f"  {emb.describe()}")
    except Exception as exc:                              # noqa: BLE001
        print(f"  [FAIL] could not load the model: "
              f"{exc.__class__.__name__}: {exc}")
        print("\n  Common causes:")
        print("    - the .gguf is not an EMBEDDING model (needs a bert/"
              "embedding arch)")
        print("    - the file is truncated or still downloading")
        print("    - not enough RAM for the chosen n_ctx")
        return 4
    print(f"  loaded {emb.name}  dim={emb.dim}  ({time.time() - t0:.1f}s)\n")

    fails: list[str] = []

    # 1 ─ scale and separation of the encoder's own similarity range
    v = emb.embed(["a sanity check sentence"], Space.READ)[0]
    norm = sum(x * x for x in v) ** 0.5
    size_gb = os.path.getsize(model_path) / 1e9
    if size_gb > 2.0:
        print(f"\n  NOTE: {size_gb:.1f} GB model. On CPU this will be far "
              "slower than a small")
        print("        encoder -- check section 7 before using it for bulk "
              "ingest.")
    print()
    print(f"0. POOLING")
    pooled = getattr(emb, "pooled_by_llama", None)
    if getattr(emb, "calibration", None) is None:
        print("   [WARN] NOT CALIBRATED. The defaults were measured on a")
        print("          DIFFERENT model, and both the absolute scale and")
        print("          which signal carries differ between encoders.")
        print(f"          Run:  validate.bat \"{os.path.basename(model_path)}\""
              " --calibrate")
        print()
    if pooled is True:
        print("   llama.cpp applied the model's own pooling  (correct)")
    elif pooled is False:
        print(f"   llama.cpp returned PER-TOKEN output; adapter pooled with "
              f"'{emb.pooling}'")
        print("   BGE-M3 is trained for CLS. If this says 'mean', rankings "
              "will look")
        print("   like noise rather than like a weak model.")
        if getattr(emb, "pooling", "") != "cls":
            fails.append("mean-pooling a CLS-trained encoder")
    else:
        print("   unknown")
    print()
    print(f"1. ENCODER CALIBRATION")
    print(f"   raw vector norm      {norm:.3f}"
          + ("  (unnormalised - OWL normalises at the boundary)"
             if abs(norm - 1.0) > 0.05 else "  (unit)"))
    if len(v) != emb.dim:
        fails.append("dimension mismatch")

    from owl.vectors import unit as _unit

    def _cos(a, b):
        ua, ub = _unit(a), _unit(b)
        return sum(x * y for x, y in zip(ua, ub))

    corpus = [t for t, _ in PARAPHRASE]
    C = emb.embed(corpus, Space.WRITE)
    rel, unrel = [], []
    for text, q in PARAPHRASE:
        qv = emb.embed([q], Space.READ)[0]
        for t, cv in zip(corpus, C):
            (rel if t == text else unrel).append(_cos(qv, cv))
    for q in UNRELATED_QUERIES:
        qv = emb.embed([q], Space.READ)[0]
        unrel.extend(_cos(qv, cv) for cv in C)
    hi_unrel, lo_rel = max(unrel), min(rel)
    mean_rel = sum(rel) / len(rel)
    mean_unrel = sum(unrel) / len(unrel)
    floor = getattr(emb, "noise_floor", 0.40)
    print(f"   related cosines      {min(rel):.3f} .. {max(rel):.3f}"
          f"   (mean {mean_rel:.3f})")
    print(f"   unrelated cosines    {min(unrel):.3f} .. {hi_unrel:.3f}"
          f"   (mean {mean_unrel:.3f})")
    print(f"   noise cutoff         {floor:.3f}")

    if hi_unrel < lo_rel:
        print("   -> bands are cleanly separated (unusual; most encoders "
              "overlap)")
    else:
        print(f"   -> bands OVERLAP by {hi_unrel - lo_rel:.3f}. Expected: no "
              "single threshold")
        print("      separates them, which is why discrimination uses MARGIN "
              "above background,")
        print("      not an absolute cutoff.")

    if lo_rel < floor:
        print(f"   -> WARNING: the cutoff {floor:.3f} discards true matches "
              f"down at {lo_rel:.3f}.")
        print(f"      Pass noise_floor={max(0.0, round(lo_rel - 0.05, 2))} "
              "to GgufEmbedder.")
        fails.append(f"noise cutoff {floor} discards true matches")
    separation = mean_rel - mean_unrel
    print(f"   margin headroom      {separation:+.3f}  "
          f"(mean related - mean unrelated)")
    if separation < 0.08:
        fails.append(f"encoder gives only {separation:.3f} headroom - "
                     "margin cannot discriminate")

    # 2/3 ─ paraphrase and discrimination
    clock = Clock()
    path = os.path.join(tempfile.mkdtemp(), "val.owl")
    with Owl.open(path, clock=clock, embedder=emb) as mind:
        for text, _ in PARAPHRASE:
            mind.observe(text, source_ref="survey")
        for text, _ in CROSS_LINGUAL:
            mind.observe(text, source_ref="notes")
        mind.observe("Generator serial is GX-4419.", source_ref="asset")
        mind.observe("Generator serial is GX-4491.", source_ref="asset")
        for i, text in enumerate(FILLER):
            mind.observe(text, source_ref=f"daily/{i}")
        print(f"   corpus: {len(PARAPHRASE) + len(CROSS_LINGUAL) + 2 + len(FILLER)}"
              " documents")

        # Hubness is measured over the corpus, so it needs a maintenance
        # pass before it can help.
        mind.tend()

        print("\n2. PARAPHRASE RECALL (the gap Tier 0 cannot close)")
        print("   RAW  - the encoder's own ranking, unmodified")
        print("   OWL  - after hubness correction and answer-type affinity")
        print("   If OWL beats RAW, the corrections are earning their place.\n")

        from owl.vectors import unit as _u
        corpus_texts = [t for t, _ in PARAPHRASE] + \
                       [t for t, _ in CROSS_LINGUAL] + FILLER
        CV = [_u(v) for v in emb.embed(corpus_texts, Space.WRITE)]

        raw_top1 = owl_top1 = owl_top3 = 0
        for text, query in PARAPHRASE:
            qv = _u(emb.embed([query], Space.READ)[0])
            sims = sorted(((sum(a * b for a, b in zip(qv, c)), t)
                           for c, t in zip(CV, corpus_texts)), reverse=True)
            raw_rank = next(i for i, (_, t) in enumerate(sims, 1) if t == text)
            raw_top1 += raw_rank == 1

            r = mind.recall(query, budget=5, group_by=None)
            returned = [c.content for c in r.chunks]
            owl_rank = returned.index(text) + 1 if text in returned else 0
            owl_top1 += owl_rank == 1
            owl_top3 += 1 <= owl_rank <= 3

            arrow = ("=" if owl_rank == raw_rank else
                     "better" if owl_rank and owl_rank < raw_rank else "worse")
            print(f"   RAW@{raw_rank}  OWL@{owl_rank or '-'}  {arrow:6s} "
                  f"{query[:40]:42s} -> {r.state.value}")
            if raw_rank != 1:
                print(f"            encoder's own pick: "
                      f"{sims[0][1][:44]!r}")
            if not owl_rank:
                # WHY it was dropped, not just that it was. A miss here is
                # either the gate (score below KNOW_WHERE_SCORE) or hubness
                # (the target discounted for being close to everything), and
                # those have opposite fixes. Guessing between them cost a
                # round of this project.
                target_sim = next(s for s, t in sims if t == text)
                lvl = metamemory.level_of(
                    target_sim, getattr(emb, "noise_floor", 0.40),
                    getattr(emb, "ceiling", 1.0))
                hub = mind.hubness_of(text)
                print(f"            dropped: sim {target_sim:.3f} -> level "
                      f"{lvl:.3f} vs KNOW_WHERE {metamemory.KNOW_WHERE_SCORE}"
                      f"; hubness discount {hub * 0.6:.3f}")

        n = len(PARAPHRASE)
        print(f"\n   raw encoder top-1  {raw_top1}/{n}")
        print(f"   OWL top-1          {owl_top1}/{n}   top-3 {owl_top3}/{n}")
        if owl_top1 < raw_top1:
            fails.append(f"corrections made ranking WORSE "
                         f"({owl_top1} vs raw {raw_top1})")
        elif owl_top1 > raw_top1:
            print(f"   -> the corrections recovered "
                  f"{owl_top1 - raw_top1} probe(s) the encoder ranked below "
                  "a hub")
        else:
            print("   -> corrections neutral on this corpus")

        print("\n3. DISCRIMINATION (unrelated must NOT match)")
        # Two grades, because they are different failures. KNOW on an absent
        # fact is confabulation -- the thing this engine exists to prevent.
        # KNOW_WHERE is milder but not "ok": it claims a neighbourhood for
        # something that has none, and reporting it as a pass hid a real
        # difference between encoders behind an identical green line.
        false_pos = soft = 0
        for q in UNRELATED_QUERIES:
            r = mind.recall(q)
            if r.state is State.KNOW:
                false_pos += 1
                label = "FALSE POSITIVE"
            elif r.state in (State.KNOW_WHERE, State.FAMILIAR):
                soft += 1
                label = "soft FP       "
            else:
                label = "ok            "
            print(f"   {label} {q[:40]:42s} -> {r.state.value}")
        if false_pos:
            fails.append(f"{false_pos} false positives (confabulation)")
        if soft:
            print(f"   -> {soft} soft: claimed a neighbourhood for something "
                  "absent. Not fatal,")
            print("      but an encoder with none of these is strictly "
                  "better behaved.")

        print("\n5. IDENTIFIER RECALL (max-fusion must not lose these)")
        r = mind.recall("GX-4419", budget=2)
        ok = bool(r.chunks) and "GX-4419" in r.chunks[0].content
        print(f"   {'OK  ' if ok else 'MISS'} GX-4419 -> "
              f"{r.chunks[0].content if r.chunks else 'nothing'}")
        if not ok:
            fails.append("identifier recall")

        print("\n6. CROSS-LINGUAL (BGE-M3 is multilingual — verify it)")
        for en, other in CROSS_LINGUAL:
            r = mind.recall(other, budget=3)
            ok = bool(r.chunks) and r.chunks[0].content == en
            print(f"   {'OK  ' if ok else 'MISS'} {other[:44]:46s} -> "
                  f"{r.state.value}")

    # 4 ─ pattern separation
    print("\n4. PATTERN SEPARATION (write space must push duplicates apart)")
    c2 = Clock()
    with Owl.open(os.path.join(tempfile.mkdtemp(), "sep.owl"), clock=c2,
                  embedder=emb) as m:
        a = m.observe("Weekly supply meeting held; stock levels reviewed.",
                      source_ref="week1")
        c2.advance(7)
        b = m.observe("Weekly supply meeting held; stock levels reviewed.",
                      source_ref="week2")

        def vec(nid, sp):
            return unpack(m._s.one(
                "SELECT data FROM vector WHERE node_id=? AND space=?",
                (nid, sp))["data"])

        rs = dot(vec(a, "read"), vec(b, "read"))
        ws = dot(vec(a, "write"), vec(b, "write"))
        print(f"   READ  similarity = {rs:.3f}  (meaning: same)")
        print(f"   WRITE similarity = {ws:.3f}  (episodes: distinct)")
        if rs - ws < 0.05:
            print("   -> separation vanished. If the raw norm above is far "
                  "from 1.0,")
            print("      the context component is being swamped by an "
                  "unnormalised vector.")
            fails.append(f"separation cosmetic (read {rs:.3f} write {ws:.3f})")

    # 7 ─ throughput
    print("\n7. THROUGHPUT")
    batch = ["field note number %d about supplies and logistics" % i
             for i in range(16)]
    times = []
    for _ in range(3):
        t = time.time()
        emb.embed(batch, Space.WRITE)
        times.append(time.time() - t)
    per = statistics.median(times) / len(batch) * 1000
    print(f"   {per:.1f} ms/text  ({16 / statistics.median(times):.0f} texts/s)")

    print("\n" + "=" * 70)
    if fails:
        print("FAILED:")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("ALL CHECKS PASSED — Tier 1 numbers are now trustworthy.")
    return 0


def compare(paths: list[str]) -> int:
    """Side by side: what does a smaller model actually cost you?

    Quality comes from the sidecars, which already hold every measured
    number, so only throughput has to be run live. The point is to make the
    trade explicit -- "8x faster for 0.02 AUC" is a decision; "the big one
    is better" is not.
    """
    from owl.adapters import calibration as cal, gguf_embed
    if not gguf_embed.available():
        print("  [BLOCKED] llama-cpp-python is not installed. Run install.bat")
        return 3

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    cols = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [FAIL] no such file: {p}")
            return 2
        c = cal.load(p)
        if c is None:
            print(f"  [FAIL] {os.path.basename(p)} is not calibrated.")
            print(f'         Run: validate.bat "{p}" --calibrate')
            return 2
        emb = gguf_embed.GgufEmbedder(p, n_ctx=2048)
        batch = [f"field note number {i} about supplies and logistics"
                 for i in range(16)]
        times = []
        for _ in range(3):
            t = time.time()
            emb.embed(batch, Space.WRITE)
            times.append(time.time() - t)
        ms = statistics.median(times) / len(batch) * 1000
        emb.close()
        cols.append((os.path.basename(p)[:-5][:26], c, ms, emb.meta.dim))

    def row(label, fn):
        print(f"  {label:24s}" + "".join(f"{fn(c):>28s}" for c in cols))

    print(f"  {'':24s}" + "".join(f"{n:>28s}" for n, _, _, _ in cols))
    print("  " + "-" * (24 + 28 * len(cols) - 2))
    row("separability AUC", lambda c: f"{c[1].separability:.4f}")
    row("usable headroom", lambda c: f"{c[1].headroom:+.3f}")
    row("floor .. ceiling",
        lambda c: (f"{c[1].noise_floor} .. {c[1].ceiling}"
                   + ("  STALE" if c[1].ceiling == 1.0 else "")))
    row("separator", lambda c: c[1].separator)
    row("query->doc background", lambda c: f"{c[1].anisotropy:.3f}")
    row("doc->doc background", lambda c: f"{c[1].doc_anisotropy:.3f}")
    row("dimensions", lambda c: f"{c[3]} ({c[3] * 4 // 1024} KB/memory)")
    row("throughput", lambda c: f"{c[2]:.1f} ms/text")

    if len(cols) == 2:
        (_, a, ta, _), (_, b, tb, _) = cols
        print()
        speed = ta / tb if tb else 0
        d_auc = b.separability - a.separability
        print(f"  the second is {speed:.1f}x {'faster' if speed > 1 else 'slower'}"
              f" for {d_auc:+.4f} AUC")
        # AUC is a probability, so differences near the ceiling matter more
        # than they look: 0.993 -> 0.980 nearly triples the error rate.
        ea, eb = 1 - a.separability, 1 - b.separability
        if ea > 0 and eb > 0:
            print(f"  error rate {ea * 100:.2f}% -> {eb * 100:.2f}% "
                  f"({eb / ea:.1f}x)")
        # AUC scores RANKING. OWL's headline claim is the GATE -- six honest
        # states rather than always returning something -- and the two come
        # apart. An encoder can rank well on average and still have no clean
        # boundary between a true match and noise, which shows up as
        # separator "neither" and negative headroom, not in the AUC.
        for name, c, _, _ in cols:
            if c.separator == "neither" or c.headroom <= 0:
                print(f"\n  {name}: separator '{c.separator}', headroom "
                      f"{c.headroom:+.3f}")
                print("    Ranks well, but no clean KNOW / DONT_KNOW boundary "
                      "-- borderline queries")
                print("    land in KNOW_WHERE. Honest, and weaker than the "
                      "AUC alone suggests.")
    for name, c, _, _ in cols:
        if c.ceiling == 1.0:
            print(f"\n  [STALE] {name} was calibrated before ceiling was "
                  "measured. Re-run")
            print("          --calibrate; until then its scores are divided "
                  "by a range")
            print("          the encoder cannot reach and good matches read "
                  "as DONT_KNOW.")
    print("\n  Neither number decides this alone. Ingest volume decides it.")
    return 0


def calibrate_all(folder: str) -> int:
    """Calibrate every .gguf in a folder.

    Sidecars live next to the model and do not survive a clean checkout, so
    a workflow that reinstalls from scratch loses them every time -- and a
    missing sidecar is silent, while a STALE one is worse: it reinstated a
    fixed bug here twice. One command, all models, no per-file bookkeeping.
    """
    import glob
    models = sorted(glob.glob(os.path.join(folder, "*.gguf")))
    if not models:
        print(f"  no .gguf files in {folder!r}")
        return 2
    print(f"  {len(models)} model(s) in {folder!r}\n")
    rc = 0
    for m in models:
        print("#" * 70)
        print(f"#  {os.path.basename(m)}")
        print("#" * 70)
        rc |= calibrate(m)
        print()
    return rc


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--calibrate-all" in sys.argv:
        raise SystemExit(calibrate_all(args[0] if args else "embedding model"))
    if not args:
        print(__doc__)
        raise SystemExit(2)
    if "--compare" in sys.argv:
        raise SystemExit(compare(args))
    if "--calibrate" in sys.argv:
        raise SystemExit(calibrate(args[0]))
    raise SystemExit(main(args[0]))
