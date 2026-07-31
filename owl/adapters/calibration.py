"""Measured gate parameters, stored next to the model they describe.

Which signal tells you a match is real is a property of the ENCODER, not of
the engine. Measured on the same 22-document corpus:

    bge-m3      absolute bands OVERLAP by 0.088   -> margin is the signal
    Qwen3-8B    absolute bands separate           -> level is the signal
                (unrelated max 0.353 < related min 0.360)

And the absolute scale moves too. bge-m3 put related text at 0.43-0.69;
Qwen3 puts it at 0.36-0.53. A `noise_floor` of 0.40 is reasonable for the
first and silently discards half the true matches for the second -- nothing
errors, retrieval is just quietly worse.

So these cannot be constants in the source. `validate_embedder.py
--calibrate` measures them and writes `<model>.owlcal.json`; the adapter
loads it automatically. Calibrate once, and the numbers travel with the
model file.

ANISOTROPY, and why a raw cosine means nothing on its own
---------------------------------------------------------
0.36 is a good score in a space where unrelated text sits at 0.05 and a
worthless one in a space where unrelated text sits at 0.34. Last-token
pooling on a causal model concentrates embeddings into a narrow cone, and
aggressive quantisation compresses it further -- so the same threshold can
be generous on one model and vacuous on the next.

So the sweep measures the background directly, from hundreds of pairs
rather than a handful of probes:

    headroom = weakest true match - background 95th percentile

TWO BACKGROUNDS, NOT ONE -- a mistake this file made once
----------------------------------------------------------
The first version of this measured document-to-document cosines and used
them to judge query-to-document scores. On Qwen3 that produced a headroom
of -0.169 and a confident recommendation to re-quantise a model that was
working correctly.

The two are not on the same scale. Qwen3-Embedding puts an instruction
prefix on the QUERY side only, which deliberately displaces query vectors
relative to document vectors. Doc-doc pairs get identical treatment on
both sides and sit high in the cone; query-doc pairs do not. Comparing one
to the other measures the prefix, not the model.

    retrieval_background   query -> doc, unrelated pairs
                           judges the recall gate: noise_floor, KNOW vs
                           KNOW_WHERE, whether a match is real

    doc_background         doc -> doc, unrelated pairs
                           judges FUSION: dedupe, clustering, interference.
                           A 0.75 cluster threshold assumes unrelated text
                           sits near zero. When it sits at 0.41, that
                           threshold is barely above chance and fusion will
                           merge memories that have nothing to do with each
                           other.

Both are worth knowing. Neither substitutes for the other.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUFFIX = ".owlcal.json"


@dataclass
class Calibration:
    model: str = ""
    measured_at: float = 0.0
    corpus_size: int = 0
    # what to feed the gate
    noise_floor: float = 0.40
    search_floor: float = 0.15
    level_weight: float = 0.25
    margin_scale: float = 0.30
    # The highest cosine a TRUE match actually reached. Not 1.0, ever, and
    # scores must be scaled by this rather than by an imaginary 1.0.
    ceiling: float = 1.0
    # what was seen, so the numbers can be argued with
    related: tuple[float, float] = (0.0, 0.0)
    unrelated: tuple[float, float] = (0.0, 0.0)
    related_margins: tuple[float, float] = (0.0, 0.0)
    unrelated_margins: tuple[float, float] = (0.0, 0.0)
    separator: str = "unknown"          # "level" | "margin" | "neither"
    # where the space sits -- two scales, see the note above
    anisotropy: float = 0.0             # query->doc, unrelated: mean
    anisotropy_p95: float = 0.0         # query->doc, unrelated: p95
    headroom: float = 0.0               # weakest true match - anisotropy_p95
    separability: float = 0.0           # AUC: true match vs unrelated pair
    doc_anisotropy: float = 0.0         # doc->doc, unrelated: mean
    doc_anisotropy_p95: float = 0.0     # doc->doc, unrelated: p95
    notes: list[str] = field(default_factory=list)

    def fusion_thresholds(self, dedupe: float, cluster: float
                          ) -> tuple[float, float]:
        """Rescale fusion thresholds into this encoder's actual range.

        0.75 means "three quarters of the way from unrelated to identical".
        In a space where unrelated documents already sit at 0.53, the raw
        0.75 is only a fifth of the way up and clusters near-strangers.
        """
        return (rescale(dedupe, self.doc_anisotropy_p95),
                rescale(cluster, self.doc_anisotropy_p95))

    def save(self, model_path: str | Path) -> Path:
        p = Path(str(model_path) + SUFFIX)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p


def load(model_path: str | Path) -> Calibration | None:
    p = Path(str(model_path) + SUFFIX)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    known = {f for f in Calibration.__dataclass_fields__}
    c = Calibration(**{k: v for k, v in d.items() if k in known})
    if "ceiling" not in d:
        # Predates the ceiling measurement, so it would silently supply 1.0
        # -- the exact bug the field was added to fix, and one that reappears
        # every time an old sidecar outlives the code. Telling the user to
        # re-calibrate was not good enough: they did, and a stale copy came
        # back and reintroduced it invisibly.
        #
        # It does not need re-measuring. `derive()` sets ceiling = max of the
        # related band, and that band is already in the file, so the upgrade
        # is exact and offline.
        top = c.related[1] if c.related and len(c.related) > 1 else 0.0
        if 0.0 < top <= 1.0:
            c.ceiling = top
            c.notes = c.notes + [
                f"ceiling backfilled to {top} from the recorded related band "
                "(sidecar predates the field; no re-calibration needed)"]
        else:
            c.notes = c.notes + [
                "ceiling NOT measured and not recoverable -- re-run "
                "--calibrate; scores are being scaled against a similarity "
                "no encoder reaches"]
    if "doc_anisotropy" not in d and c.anisotropy:
        # Written before query->doc and doc->doc were separated, so its
        # `anisotropy` is a doc->doc number being used as a query->doc one.
        # Drop the diagnostic rather than report a false verdict; the gate
        # parameters themselves are unaffected.
        c.anisotropy = c.anisotropy_p95 = c.headroom = 0.0
        c.notes = c.notes + ["background dropped: measured before the "
                             "query/doc scale split -- re-run --calibrate"]
    return c


def rescale(threshold: float, background: float) -> float:
    """Move a [0,1] threshold into a space whose floor is `background`.

    A threshold written as 0.75 encodes an intent -- "much more similar
    than chance" -- not a physical constant. Preserve the intent.
    """
    if not 0.0 < background < 1.0:
        return threshold
    return round(background + threshold * (1.0 - background), 3)


def _spread(sims: list[float]) -> tuple[float, float]:
    if len(sims) < 3:
        return 0.0, 0.0
    sims = sorted(sims)
    return (sum(sims) / len(sims),
            sims[min(len(sims) - 1, int(len(sims) * 0.95))])


def anisotropy(vectors: list[list[float]]) -> tuple[float, float]:
    """Mean and p95 cosine between unrelated unit vectors, all pairs.

    DOCUMENT space only. Both sides get identical treatment, so this is the
    baseline for fusion and clustering -- NOT for the retrieval gate, whose
    queries carry an instruction prefix and live elsewhere in the cone.
    Use `background()` for that.
    """
    n = len(vectors)
    if n < 3:
        return 0.0, 0.0
    return _spread([sum(x * y for x, y in zip(vectors[i], vectors[j]))
                    for i in range(n) for j in range(i + 1, n)])


def separability(related: list[float], bg: list[float]) -> float:
    """P(a true match outscores a random unrelated pair). Plain AUC.

    Headroom is a MINIMUM over four probes, so one weak probe sets it for
    the whole encoder. That is the right statistic for placing a floor --
    the floor must clear the worst true match -- and the wrong one for
    judging the encoder, where it reads as a verdict on evidence of n=1.

    AUC uses every comparison instead: 4 probes against 150 background
    pairs is 600 of them. An encoder can have thin headroom and excellent
    separability at the same time, and Qwen3 does.
    """
    import bisect
    if not related or len(bg) < 3:
        return 0.0
    bg = sorted(bg)
    total = 0.0
    for r in related:
        lo = bisect.bisect_left(bg, r)
        hi = bisect.bisect_right(bg, r)
        total += lo + 0.5 * (hi - lo)
    return round(total / (len(related) * len(bg)), 4)


def background(query_doc_sims: list[float]) -> tuple[float, float]:
    """Mean and p95 of query->document cosines for UNRELATED pairs.

    The right zero for the recall gate, because it is measured the way
    retrieval actually runs: prefixed query on one side, bare document on
    the other.
    """
    return _spread(query_doc_sims)


def derive(*, model: str, corpus_size: int, related: list[float],
           unrelated: list[float], rel_margins: list[float],
           unrel_margins: list[float], now: float,
           aniso: float = 0.0, aniso_p95: float = 0.0,
           doc_aniso: float = 0.0, doc_aniso_p95: float = 0.0,
           auc: float = 0.0) -> Calibration:
    """Turn measurements into parameters, and say which signal is carrying.

    Deliberately conservative on the floor: it sits BELOW the weakest true
    match, because discarding a real memory is worse than considering a
    weak candidate that the gate will reject anyway.
    """
    lo_rel, hi_rel = min(related), max(related)
    lo_un, hi_un = min(unrelated), max(unrelated)
    lo_relm, hi_relm = min(rel_margins), max(rel_margins)
    lo_unm, hi_unm = min(unrel_margins), max(unrel_margins)

    notes: list[str] = []
    level_sep = lo_rel - hi_un
    margin_sep = lo_relm - hi_unm

    if level_sep > 0 and level_sep >= margin_sep:
        separator, level_weight = "level", 0.75
        notes.append(
            f"absolute level separates cleanly ({hi_un:.3f} < {lo_rel:.3f}); "
            "weighting it as the primary signal")
    elif margin_sep > 0:
        separator, level_weight = "margin", 0.25
        notes.append(
            f"margin separates ({hi_unm:.3f} < {lo_relm:.3f}); weighting it "
            "as the primary signal")
    else:
        # Neither separates. Lean on whichever overlaps LESS, and say so.
        separator = "neither"
        level_weight = 0.55 if level_sep >= margin_sep else 0.25
        notes.append(
            f"neither signal separates these probes (level overlap "
            f"{-level_sep:+.3f}, margin overlap {-margin_sep:+.3f}); "
            "borderline queries will land in KNOW_WHERE, which is the "
            "honest outcome")

    # How far the weakest true match sits above where random prose sits.
    # This is the encoder's verdict on itself, and no threshold can argue
    # with it.
    headroom = round(lo_rel - aniso_p95, 3) if aniso_p95 else 0.0
    if aniso_p95:
        notes.append(
            f"unrelated query->doc pairs sit at {aniso:.3f} mean / "
            f"{aniso_p95:.3f} p95; the weakest true match clears that by "
            f"{headroom:+.3f}")
        if auc >= 0.95 and headroom > 0:
            notes.append(
                f"separability AUC {auc:.3f} -- a true match outscores a "
                f"random unrelated pair {auc * 100:.1f}% of the time. This "
                "encoder is discriminating well; the thin headroom above is "
                "one weak probe, not a weak model")
        elif auc >= 0.95:
            # Do NOT call a negative headroom "thin". The weakest probe is
            # not near the noise band, it is INSIDE it -- and saying
            # otherwise is the same species of error as every other bug this
            # session: a reassuring summary detached from the measurement.
            notes.append(
                f"separability AUC {auc:.3f}, but headroom is NEGATIVE "
                f"({headroom:+.3f}): the weakest probe scores below the p95 "
                "of unrelated pairs. Across all comparisons the encoder "
                "separates well, so that probe is an outlier rather than the "
                "norm -- but it will be answered DONT_KNOW, and any query "
                "resembling it will be too")
        elif auc >= 0.85:
            notes.append(
                f"separability AUC {auc:.3f} -- workable, but borderline "
                "queries will land in KNOW_WHERE rather than KNOW. That is "
                "the honest outcome, not a bug")
        elif auc > 0:
            notes.append(
                f"WARNING: separability AUC {auc:.3f}. A true match barely "
                "outscores an unrelated one, and no floor fixes that -- "
                "check pooling and the query prefix first, then the quant")
        elif headroom <= 0.0:
            notes.append(
                "WARNING: a true match scores no better than an unrelated "
                "one. No floor fixes that -- check pooling and the query "
                "prefix first, then try a lighter quant")

    if doc_aniso_p95:
        d_ded, d_clu = rescale(0.85, doc_aniso_p95), rescale(0.75,
                                                             doc_aniso_p95)
        notes.append(
            f"document->document background is {doc_aniso:.3f} mean / "
            f"{doc_aniso_p95:.3f} p95 -- a DIFFERENT scale from the above, "
            "because the query prefix displaces query vectors")
        if doc_aniso >= 0.60:
            notes.append(
                f"fusion thresholds rescaled to {d_ded}/{d_clu}. At this "
                "background the raw 0.75 is inside the range unrelated "
                "documents reach, so the naive setting WOULD merge strangers")
        elif doc_aniso >= 0.25:
            notes.append(
                f"fusion thresholds rescaled to {d_ded}/{d_clu}. No merge "
                "is wrong at this background -- the tail does not reach 0.75 "
                "-- but 0.75 is meant to sit three quarters of the way from "
                "chance to identical and here it sits about a fifth, so the "
                "margin is much thinner than the number suggests")

    # Never above the weakest true match. The cushion is proportional to the
    # USABLE range rather than a flat 0.08, because in a cone-shaped space
    # 0.08 can be half the range that carries any signal at all.
    if aniso_p95:
        cushion = max(0.03, 0.10 * max(0.0, hi_rel - aniso_p95))
    else:
        cushion = 0.08
    noise_floor = round(max(0.05, lo_rel - cushion), 2)
    if noise_floor > lo_rel:                       # paranoia
        noise_floor = round(lo_rel * 0.9, 2)
    # Always well below the noise floor. Anchoring it to the unrelated band
    # put it at 0.31 for bge-m3, which would starve the candidate set and
    # trip the "too few candidates to estimate a background" path -- the
    # exact double-penalty already fixed once.
    search_floor = round(max(0.02, noise_floor * 0.5), 2)
    notes.append(f"noise_floor set below the weakest true match ({lo_rel:.3f})"
                 " so real memories are never silently discarded")

    # The top of the scale, measured. Scores are (sim - floor)/(ceiling -
    # floor); using 1.0 here compressed every match into the bottom third of
    # its range and returned DONT_KNOW for the encoder's own top-ranked hit.
    ceiling = round(min(1.0, hi_rel), 3)
    notes.append(
        f"scores scaled to the range this encoder actually reaches "
        f"({noise_floor} .. {ceiling}); a true match at {lo_rel:.3f} is near "
        "the bottom of that range, not near zero as 1.0-scaling implied")

    return Calibration(
        model=model, measured_at=now, corpus_size=corpus_size,
        noise_floor=noise_floor, search_floor=search_floor,
        level_weight=level_weight, ceiling=ceiling,
        related=(round(lo_rel, 3), round(hi_rel, 3)),
        unrelated=(round(lo_un, 3), round(hi_un, 3)),
        related_margins=(round(lo_relm, 3), round(hi_relm, 3)),
        unrelated_margins=(round(lo_unm, 3), round(hi_unm, 3)),
        separator=separator,
        anisotropy=round(aniso, 3), anisotropy_p95=round(aniso_p95, 3),
        headroom=headroom, separability=auc,
        doc_anisotropy=round(doc_aniso, 3),
        doc_anisotropy_p95=round(doc_aniso_p95, 3), notes=notes)
