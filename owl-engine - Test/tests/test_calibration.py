"""Gate parameters are properties of the ENCODER, and must travel with it."""
import math
import random
import time

from owl.adapters.calibration import Calibration, anisotropy, derive, load


def _qwen():
    return derive(model="Qwen3-Embedding-8B", corpus_size=22,
                  related=[0.531, 0.360, 0.430, 0.497],
                  unrelated=[0.285, 0.275, 0.353],
                  rel_margins=[0.303, 0.109, 0.256, 0.284],
                  unrel_margins=[0.107, 0.115, 0.243], now=time.time())


def _bge():
    return derive(model="bge-m3", corpus_size=22,
                  related=[0.691, 0.426, 0.486, 0.603],
                  unrelated=[0.415, 0.469, 0.490],
                  rel_margins=[0.304, 0.067, 0.166, 0.306],
                  unrel_margins=[0.106, 0.142, 0.198], now=time.time())


def test_the_floor_never_sits_above_a_true_match():
    """The bug this exists to prevent: a default of 0.40 is above two of
    Qwen3's four related probes, so real memories are silently discarded
    and nothing errors."""
    for c, weakest in ((_qwen(), 0.360), (_bge(), 0.426)):
        assert c.noise_floor < weakest, c.model


def test_search_floor_always_leaves_headroom():
    """Anchoring it to the unrelated band put it at 0.31 for bge-m3, which
    starves the candidate set and trips the 'too few candidates' path."""
    for c in (_qwen(), _bge()):
        assert c.search_floor <= c.noise_floor * 0.6, c.model


def test_it_identifies_which_signal_actually_separates():
    """Measured on the SAME corpus: for Qwen3 absolute level separates, for
    bge-m3 neither does. A single hard-coded weighting cannot serve both."""
    q, b = _qwen(), _bge()
    assert q.separator == "level" and q.level_weight > 0.5
    assert b.separator == "neither"
    assert q.level_weight != b.level_weight


def test_notes_explain_the_numbers():
    for c in (_qwen(), _bge()):
        assert c.notes and any("noise_floor" in n for n in c.notes)


def test_roundtrip_through_the_sidecar(tmp_path):
    model = tmp_path / "fake-model.gguf"
    model.write_bytes(b"not really a model")
    c = _qwen()
    written = c.save(model)
    assert written.name.endswith(".owlcal.json")

    back = load(model)
    assert back is not None
    assert back.noise_floor == c.noise_floor
    assert back.level_weight == c.level_weight
    assert back.separator == c.separator


def test_missing_or_corrupt_sidecar_is_not_fatal(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    assert load(model) is None
    (tmp_path / "m.gguf.owlcal.json").write_text("{ not json")
    assert load(model) is None


def test_unknown_fields_are_ignored(tmp_path):
    """A sidecar written by a newer version must not break an older one."""
    import json
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    (tmp_path / "m.gguf.owlcal.json").write_text(
        json.dumps({"noise_floor": 0.31, "some_future_field": 42}))
    c = load(model)
    assert c is not None and c.noise_floor == 0.31


def _cone(n, dim, tightness, seed=7):
    """n unit vectors packed into a cone -- the shape last-token pooling on
    a causal model produces, and that quantisation tightens further."""
    rng = random.Random(seed)
    axis = [rng.gauss(0, 1) for _ in range(dim)]
    m = math.sqrt(sum(x * x for x in axis))
    axis = [x / m for x in axis]
    out = []
    for _ in range(n):
        # normalise the noise FIRST -- a raw gaussian vector has norm
        # sqrt(dim), so mixing it against a unit axis drowns the axis
        # entirely and every cone comes out isotropic.
        noise = [rng.gauss(0, 1) for _ in range(dim)]
        m = math.sqrt(sum(x * x for x in noise))
        noise = [x / m for x in noise]
        v = [tightness * a + (1 - tightness) * z for a, z in zip(axis, noise)]
        m = math.sqrt(sum(x * x for x in v))
        out.append([x / m for x in v])
    return out


def test_anisotropy_measures_the_shape_of_the_space():
    """A raw cosine is meaningless without knowing where the floor is: 0.36
    is strong in an isotropic space and worthless in a tight cone."""
    wide_mean, wide_p95 = anisotropy(_cone(20, 64, tightness=0.05))
    tight_mean, tight_p95 = anisotropy(_cone(20, 64, tightness=0.60))
    assert wide_mean < 0.25 < tight_mean
    assert tight_p95 > wide_p95
    assert anisotropy([[1.0, 0.0]]) == (0.0, 0.0)      # too few to say


def test_it_refuses_to_call_a_collapsed_space_calibrated():
    """A true match must outscore an unrelated one. When it doesn't, no
    threshold fixes it and the sweep has to say so rather than derive a
    confident-looking floor."""
    c = derive(model="collapsed", corpus_size=22,
               related=[0.36, 0.40], unrelated=[0.35, 0.34],
               rel_margins=[0.02, 0.03], unrel_margins=[0.02, 0.03],
               now=time.time(), aniso=0.33, aniso_p95=0.37)
    assert c.headroom < 0
    assert any("no better than an unrelated" in n for n in c.notes)


def test_query_doc_and_doc_doc_are_not_the_same_scale():
    """The bug this exists to prevent, and it shipped once.

    Qwen3 puts an instruction prefix on queries only, so query->doc pairs
    sit far lower than doc->doc pairs. Judging the first against the second
    reported -0.169 headroom and recommended re-quantising a model that was
    working correctly. Real numbers from that run."""
    c = derive(model="Qwen3-Embedding-8B", corpus_size=22,
               related=[0.531, 0.360, 0.430, 0.497],
               unrelated=[0.285, 0.275, 0.353],
               rel_margins=[0.303, 0.109, 0.256, 0.284],
               unrel_margins=[0.107, 0.115, 0.243], now=time.time(),
               aniso=0.19, aniso_p95=0.33,        # query -> doc
               doc_aniso=0.406, doc_aniso_p95=0.529)   # doc -> doc
    assert c.headroom > 0, "the encoder does separate; don't tell them it doesn't"
    assert not any("no better than an unrelated" in n for n in c.notes)
    assert c.doc_anisotropy_p95 > c.anisotropy_p95
    assert any("DIFFERENT scale" in n for n in c.notes)


def test_fusion_thresholds_move_into_the_measured_space():
    """0.75 means 'much more similar than chance'. Where chance is 0.53,
    the literal 0.75 clusters near-strangers."""
    from owl import fusion
    c = derive(model="q", corpus_size=22, related=[0.36, 0.53],
               unrelated=[0.28, 0.35], rel_margins=[0.11, 0.30],
               unrel_margins=[0.11, 0.24], now=time.time(),
               aniso=0.19, aniso_p95=0.33,
               doc_aniso=0.406, doc_aniso_p95=0.529)
    ded, clu = c.fusion_thresholds(0.85, 0.75)
    assert 0.92 < ded < 0.94 and 0.87 < clu < 0.89

    # a pair at 0.78 is "clearly related" by default and chance in this space
    pairs = [("a", "b", 0.78)]
    assert fusion.plan(pairs).clusters                       # naive: merges
    assert not fusion.plan(pairs, calibration=c).clusters     # measured: no

    # an unmeasured calibration must not silently move anything
    blank = Calibration()
    assert blank.fusion_thresholds(0.85, 0.75) == (0.85, 0.75)


def test_the_cushion_scales_to_the_usable_range():
    """A flat 0.08 below the weakest match is a rounding error in a wide
    space and half the signal in a narrow one."""
    narrow = derive(model="narrow", corpus_size=22,
                    related=[0.36, 0.53], unrelated=[0.28, 0.35],
                    rel_margins=[0.11, 0.30], unrel_margins=[0.11, 0.24],
                    now=time.time(), aniso=0.30, aniso_p95=0.34)
    # still below the weakest true match, but not needlessly far below it
    assert narrow.noise_floor < 0.36
    assert narrow.noise_floor > 0.36 - 0.08
    assert narrow.headroom > 0


def test_anisotropy_is_optional():
    """Sweeps predating the measurement must still derive."""
    c = _qwen()
    assert c.anisotropy == 0.0 and c.headroom == 0.0
    assert c.noise_floor < 0.360


def test_separability_survives_one_weak_probe():
    """Qwen3's real numbers. Headroom is a min over 4 probes and reads as
    +0.027 -- 'thin'. AUC over all 600 comparisons says the encoder is
    discriminating near-perfectly. Both are true; only one is a verdict on
    the model."""
    from owl.adapters.calibration import separability
    rng = random.Random(3)
    # 150 unrelated query->doc cosines, mean 0.216 / p95 0.333
    bg = [max(0.0, rng.gauss(0.216, 0.071)) for _ in range(150)]
    related = [0.531, 0.360, 0.430, 0.497]
    auc = separability(related, bg)
    assert auc > 0.95, auc
    assert min(related) - sorted(bg)[142] < 0.06      # headroom IS thin

    c = derive(model="q", corpus_size=22, related=related,
               unrelated=[0.285, 0.275, 0.353],
               rel_margins=[0.303, 0.109, 0.256, 0.284],
               unrel_margins=[0.107, 0.115, 0.243], now=time.time(),
               aniso=0.216, aniso_p95=0.333, auc=auc)
    assert any("one weak probe, not a weak model" in n for n in c.notes)


def test_separability_still_condemns_a_genuinely_bad_encoder():
    from owl.adapters.calibration import separability
    rng = random.Random(4)
    bg = [rng.gauss(0.40, 0.10) for _ in range(150)]
    auc = separability([0.41, 0.38, 0.44, 0.40], bg)
    assert auc < 0.85
    c = derive(model="bad", corpus_size=22, related=[0.41, 0.38],
               unrelated=[0.39, 0.42], rel_margins=[0.01, 0.02],
               unrel_margins=[0.01, 0.02], now=time.time(),
               aniso=0.40, aniso_p95=0.55, auc=auc)
    assert any("WARNING" in n for n in c.notes)


def test_separability_degrades_gracefully():
    from owl.adapters.calibration import separability
    assert separability([], [0.1, 0.2, 0.3]) == 0.0
    assert separability([0.5], [0.1]) == 0.0
    assert separability([0.9], [0.1, 0.2, 0.3]) == 1.0


def test_background_uses_the_query_side_population():
    from owl.adapters.calibration import background
    mean, p95 = background([0.1, 0.2, 0.15, 0.3, 0.9])
    assert 0.3 < mean < 0.4 and p95 == 0.9
    assert background([0.1, 0.2]) == (0.0, 0.0)


def test_scores_scale_to_the_encoders_real_ceiling():
    """The regression this exists to prevent, with the numbers that caused
    it. Qwen3's best true match over 24 documents is 0.531. Dividing by
    (1.0 - 0.33) maps a correct answer at 0.430 to 0.149 -- below
    KNOW_WHERE_SCORE -- so the hit the ENCODER ranked first came back
    DONT_KNOW, and the validator reported OWL 2/4 against raw 3/4."""
    from owl.metamemory import KNOW_WHERE_SCORE, level_of
    floor, ceiling = 0.33, 0.531

    assert level_of(0.430, floor, 1.0) < KNOW_WHERE_SCORE      # the bug
    assert level_of(0.430, floor, ceiling) > KNOW_WHERE_SCORE   # the fix

    # ordering is untouched -- this rescales, it does not rerank
    scaled = [level_of(s, floor, ceiling)
              for s in (0.360, 0.430, 0.497, 0.531)]
    assert scaled == sorted(scaled)
    assert scaled[-1] == 1.0

    # and the unrelated probes stay out: the satellite-phone query hit
    # 0.353, which is 0.007 from a true match. Nothing separates those two,
    # and admitting the weak one would admit the noise as well.
    assert level_of(0.353, floor, ceiling) < KNOW_WHERE_SCORE
    assert level_of(0.285, floor, ceiling) == 0.0


def test_ceiling_defaults_are_backwards_compatible():
    from owl.metamemory import level_of
    for sim in (0.2, 0.5, 0.9):
        assert level_of(sim, 0.4) == level_of(sim, 0.4, 1.0)
    # a degenerate ceiling must not divide by ~zero
    assert 0.0 <= level_of(0.5, 0.49, 0.495) <= 1.0


def test_a_sidecar_without_a_ceiling_repairs_itself(tmp_path):
    """A stale sidecar must not silently reintroduce the 1.0-scaling bug.

    Telling the user to re-calibrate was not enough -- they did, an older
    copy came back, and the regression returned invisibly. derive() sets
    ceiling to the top of the related band, and that band is already in the
    file, so the upgrade is exact and needs no model.
    """
    import json
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    (tmp_path / "m.gguf.owlcal.json").write_text(json.dumps({
        "noise_floor": 0.33, "level_weight": 0.75,
        "related": [0.36, 0.531], "separability": 0.9933}))
    c = load(model)
    assert c.ceiling == 0.531
    assert any("backfilled" in n for n in c.notes)

    from owl.metamemory import KNOW_WHERE_SCORE, level_of
    assert level_of(0.430, c.noise_floor, c.ceiling) > KNOW_WHERE_SCORE

    # unrecoverable stays loud rather than guessing
    (tmp_path / "m.gguf.owlcal.json").write_text(json.dumps({"noise_floor": 0.4}))
    bad = load(model)
    assert bad.ceiling == 1.0
    assert any("not recoverable" in n for n in bad.notes)


def test_derive_records_the_ceiling():
    c = _qwen()
    assert c.ceiling == 0.531
    assert any("actually reaches" in n for n in c.notes)


def test_vectors_from_a_different_encoder_are_never_compared(tmp_path):
    """Swapping models on an existing store must not silently return
    nonsense. Two encoders put vectors in unrelated coordinate systems, and
    a cosine between them is a number, not an error."""
    from owl import Owl
    from owl.protocols import Space

    class Toy:
        is_semantic = True

        def __init__(self, name, dim, seed):
            self.name, self.dim, self.seed = name, dim, seed

        def embed(self, texts, space):
            out = []
            for t in texts:
                rng = random.Random(f"{self.seed}:{t}")
                out.append([rng.gauss(0, 1) for _ in range(self.dim)])
            return out

    path = tmp_path / "drift.owl"
    a = Owl.open(str(path), embedder=Toy("model-a", 64, 1))
    with a:
        a.observe("The clinic generator runs on depot fuel.")
        a.observe("Route Alpha floods above 40mm rainfall.")

    b = Owl.open(str(path), embedder=Toy("model-b", 128, 2))
    with b:
        r = b.recall("how is the health facility powered")
        assert all(v["model"] == "model-a" for v in b._s.query(
            "SELECT DISTINCT model FROM vector WHERE space=?",
            (Space.READ.value,)))
        # nothing from model-a may surface, and the user must be told why
        assert not r.chunks
        assert any("different model" in w for w in b._warnings), b._warnings


def test_level_weight_reaches_the_gate():
    from owl.metamemory import semantic_density
    # identical inputs, different weighting -> different verdict
    level_led = semantic_density(0.50, 0.28, background=0.30, level_weight=0.75)
    margin_led = semantic_density(0.50, 0.28, background=0.30, level_weight=0.25)
    assert level_led != margin_led
