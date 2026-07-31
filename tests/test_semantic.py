"""Tier 1: two embedding spaces, separation on write, completion on read."""
import math

import pytest

from owl import Owl, State
from owl.adapters.hashing import HashingEmbedder
from owl.protocols import Space
from owl.vectors import pack, unpack, dot


class ToyEmbedder:
    """A tiny semantic embedder: hand-built concept axes, deterministic.

    Real models are not available offline in CI, and a real model would make
    this suite slow. What matters here is that the two-space PLUMBING is
    exercised -- that write and read vectors differ, that separation pushes
    duplicates apart, and that completion matches paraphrase.
    """
    is_semantic = True
    name = "toy"
    dim = 8
    # A bag-of-concepts projection puts unrelated text near zero, unlike a
    # real sentence encoder. The floor is a property of the ENCODER.
    noise_floor = 0.20
    search_floor = 0.05
    AXES = [
        {"pump", "well", "water", "borehole", "gasket"},
        {"clinic", "medical", "health", "beds", "doctor", "facility"},
        {"fuel", "generator", "diesel", "power", "powered", "electricity"},
        {"route", "road", "bridge", "convoy", "track"},
        {"open", "closed", "broken", "working", "failed"},
        {"monday", "tuesday", "thursday", "march", "week", "meeting"},
        {"warsame", "ahmed", "ferrand", "bill"},
        {"bardera", "kismayo", "depot", "north", "south"},
    ]

    def embed(self, texts, space):
        from owl.lexical import tokenize
        out = []
        for t in texts:
            toks = set(tokenize(t))
            v = [float(len(toks & ax)) for ax in self.AXES]
            extra = len(toks) - sum(v)
            v = [x + 0.01 * extra for x in v]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture
def smind(clock, tmp_path):
    m = Owl.open(tmp_path / "sem.owl", clock=clock, embedder=ToyEmbedder())
    yield m
    m.close()


def test_paraphrase_recall_needs_semantics(smind, mind):
    """The gap Tier 0 cannot close: no shared content words."""
    text = "The clinic generator runs on depot fuel."
    q = "how is the health facility powered"

    mind.observe(text, source_ref="survey")
    assert mind.recall(q).state is State.DONT_KNOW, "lexical should miss this"

    smind.observe(text, source_ref="survey")
    assert smind.recall(q).state is not State.DONT_KNOW


def test_write_and_read_vectors_differ(smind):
    nid = smind.observe("North well pump needs a gasket.", source_ref="day3")
    w = smind._s.one("SELECT data FROM vector WHERE node_id=? AND space='write'",
                     (nid,))
    r = smind._s.one("SELECT data FROM vector WHERE node_id=? AND space='read'",
                     (nid,))
    assert w is not None and r is not None
    assert w["data"] != r["data"], (
        "write must carry distinguishing context; read must not")


def test_pattern_separation_pushes_duplicates_apart(smind, clock):
    """Two structurally identical entries from different days/sources must NOT
    collapse onto each other -- that collapse is what causes interference."""
    a = smind.observe("Weekly supply meeting held; stock levels reviewed.",
                      source_ref="week1")
    clock.advance(days=7)
    b = smind.observe("Weekly supply meeting held; stock levels reviewed.",
                      source_ref="week2")

    def vec(nid, space):
        return unpack(smind._s.one(
            "SELECT data FROM vector WHERE node_id=? AND space=?",
            (nid, space))["data"])

    read_sim = dot(vec(a, "read"), vec(b, "read"))
    write_sim = dot(vec(a, "write"), vec(b, "write"))
    assert read_sim > 0.98, "identical text must mean the same thing"
    # Separation must be REAL, not a rounding artefact. Prepending "[week2]"
    # to the text and hoping a mean-pooled encoder notices is a wish, not a
    # mechanism -- so the context component is concatenated structurally.
    assert read_sim - write_sim > 0.05, (
        f"separation is cosmetic: read {read_sim:.3f} write {write_sim:.3f}")
    # ...but not so aggressive that related material stops being related.
    assert write_sim > 0.6, (
        f"over-separated at {write_sim:.3f}: same subject should stay close")


def test_context_signatures_are_near_orthogonal():
    from owl.vectors import CONTEXT_DIM, context_signature
    a = context_signature(CONTEXT_DIM, "work", "per1", "epi1", "week1", 100)
    b = context_signature(CONTEXT_DIM, "work", "per1", "epi2", "week2", 107)
    same = context_signature(CONTEXT_DIM, "work", "per1", "epi1", "week1", 100)
    assert abs(dot_list(a, same) - 1.0) < 1e-6, "must be deterministic"
    assert abs(dot_list(a, b)) < 0.35, "distinct contexts must be ~orthogonal"


def test_context_dim_is_large_enough_for_reliable_separation():
    """Regression guard for a real bug.

    The signature dim was originally derived from the encoder's dim
    (`max(8, dim // 8)`). With a small encoder that gave 8-dim signatures
    whose worst-case |cos| was 0.93, so two identical texts from different
    episodes scored 0.99 in the write space and separation silently vanished
    -- intermittently, which is the worst way for it to fail.
    """
    from owl.vectors import CONTEXT_DIM, context_signature
    assert CONTEXT_DIM >= 64, "too few dims for reliable near-orthogonality"
    worst = 0.0
    for i in range(300):
        a = context_signature(CONTEXT_DIM, "w", f"p{i}", f"e{i}", "s1", i)
        b = context_signature(CONTEXT_DIM, "w", f"p{i}", f"e{i+1}", "s2", i + 7)
        worst = max(worst, abs(dot_list(a, b)))
    assert worst < 0.40, f"worst |cos| {worst:.3f} - separation is unreliable"


def dot_list(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_exact_identifiers_still_win(smind):
    """Max-fusion, not weighted sum: embeddings are weakest exactly where
    lexical is strongest -- no model reliably separates GX-4419 from GX-4491."""
    smind.observe("Generator serial is GX-4419.", source_ref="asset-reg")
    smind.observe("Generator serial is GX-4491.", source_ref="asset-reg")
    r = smind.recall("GX-4419", budget=2)
    assert r.chunks and "GX-4419" in r.chunks[0].content


def test_hashing_fallback_is_not_counted_as_semantic(tmp_path, clock):
    m = Owl.open(tmp_path / "h.owl", clock=clock, embedder=HashingEmbedder())
    try:
        m.observe("anything at all here")
        assert m.tier == 0, "a bag-of-words projection is not semantic recall"
        # Match the check ID, not its prose. IDs are the stable contract --
        # asserting on wording makes the message unimprovable.
        rep = m.doctor()
        assert any(p.startswith("embedder.semantic") for p in rep["problems"]), \
            rep["problems"]
    finally:
        m.close()


def test_reindex_backfills_a_tier0_store(tmp_path, clock):
    path = tmp_path / "grow.owl"
    m0 = Owl.open(path, clock=clock)
    for i in range(5):
        m0.observe(f"Clinic note {i}: beds and doctor rounds reviewed.")
    m0.close()

    m1 = Owl.open(path, clock=clock, embedder=ToyEmbedder())
    try:
        assert m1._vec.count() == 0
        assert m1.reindex() == 5
        assert m1._vec.count() == 10          # two spaces per node
        assert not m1.doctor()["problems"]
    finally:
        m1.close()


def test_embedder_failure_never_loses_a_write(tmp_path, clock):
    class Broken:
        is_semantic = True
        name = "broken"
        dim = 4
        def embed(self, texts, space):
            raise RuntimeError("model unloaded")

    m = Owl.open(tmp_path / "b.owl", clock=clock, embedder=Broken())
    try:
        nid = m.observe("The bridge at Km 42 is out.")
        assert m._node_row(nid) is not None, "write was lost on embedder failure"
        assert m.recall("bridge km 42").state is not State.DONT_KNOW
        assert any("embedder failed" in w for w in m.tend().get("warnings", []))
    finally:
        m.close()


def test_pack_normalises():
    v = unpack(pack([3.0, 4.0]))
    assert abs(math.sqrt(v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-6


class RealisticEmbedder:
    """Behaves like a real sentence encoder, and that is the point.

    Two properties the toy lacked, each of which hid a live bug:

      * UNNORMALISED output (norm ~25, as BGE-M3 returns through llama.cpp).
        Against that the 0.35-weighted context component is 1.5% of the write
        vector and pattern separation silently vanishes.

      * A COMMON COMPONENT. Real encoders put unrelated short texts around
        0.5, not 0 -- every sentence shares "is English prose". Max-
        normalising semantic scores then flattens everything to ~1.0 and
        swamps the lexical signal.

    Built from a deterministic token projection plus a fixed shared
    direction, weighted so unrelated text lands near 0.5 and related text
    near 0.8.

    WHAT THIS CANNOT DO, and it matters: it is a bag-of-tokens hash, so
    "related by MEANING with no shared words" -- exactly what a paraphrase
    is -- scores near zero. It reproduces the SCALE and CLUSTERING behaviour
    of a real encoder, which is what caught the two bugs it exists for. It
    cannot stand in for one when calibrating semantic thresholds; that
    needs the real model, which is what `bench/validate_embedder.py
    --calibrate` is for. Tuning gate parameters against this mock would be
    tuning against an artefact.
    """
    is_semantic = True
    name = "realistic"
    dim = 96
    noise_floor = 0.40      # where 'unrelated' sits for this encoder
    search_floor = 0.15     # where to stop looking - a different job
    COMMON = 0.50         # weight of the shared direction
    SCALE = 25.3          # deliberately off the unit sphere

    def __init__(self):
        import hashlib
        self._h = hashlib

    def _tok_vec(self, text):
        from owl.lexical import tokenize
        v = [0.0] * self.dim
        for t in tokenize(text):
            d = self._h.blake2b(t.encode(), digest_size=8).digest()
            for k in range(3):                       # 3 hashes per token
                idx = int.from_bytes(d[k * 2:k * 2 + 2], "little") % self.dim
                v[idx] += 1.0 if d[6] & (1 << k) else -1.0
        return v

    def _common(self):
        d = self._h.blake2b(b"shared-prose-direction", digest_size=64).digest()
        return [(d[i % 64] / 127.5) - 1.0 for i in range(self.dim)]

    def embed(self, texts, space):
        from owl.vectors import unit
        common = unit(self._common())
        out = []
        for t in texts:
            v = unit(self._tok_vec(t))
            mixed = [(1 - self.COMMON) * a + self.COMMON * b
                     for a, b in zip(v, common)]
            out.append([x * self.SCALE for x in unit(mixed)])
        return out


@pytest.fixture
def rmind(clock, tmp_path):
    m = Owl.open(tmp_path / "real.owl", clock=clock, embedder=RealisticEmbedder())
    yield m
    m.close()


def test_unnormalised_embeddings_do_not_kill_separation(rmind, clock):
    """The bug the real model exposed: with norm-25 vectors the context
    component was 1.5% of the write vector and separation scored 0.9998."""
    a = rmind.observe("Weekly supply meeting held; stock levels reviewed.",
                      source_ref="week1")
    clock.advance(days=7)
    b = rmind.observe("Weekly supply meeting held; stock levels reviewed.",
                      source_ref="week2")

    def vec(nid, sp):
        return unpack(rmind._s.one(
            "SELECT data FROM vector WHERE node_id=? AND space=?",
            (nid, sp))["data"])

    rs = dot(vec(a, "read"), vec(b, "read"))
    ws = dot(vec(a, "write"), vec(b, "write"))
    assert rs > 0.98, "identical text must mean the same thing"
    assert rs - ws > 0.05, (
        f"separation vanished with unnormalised input: read {rs:.4f} "
        f"write {ws:.4f}")


def test_unit_normalises_regardless_of_input_scale():
    from owl.vectors import unit
    for scale in (0.001, 1.0, 25.3, 1e6):
        v = unit([3.0 * scale, 4.0 * scale])
        assert abs(math.sqrt(v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9
    assert unit([0.0, 0.0]) == [0.0, 0.0]


def test_exact_identifiers_survive_tightly_clustered_cosines(rmind):
    """With max-normalised semantic scores every top hit scored 1.0 and
    swamped lexical, so a query for the exact string 'GX-4419' returned an
    unrelated note about generator fuel."""
    rmind.observe("Generator serial is GX-4419.", source_ref="asset-reg")
    rmind.observe("Generator serial is GX-4491.", source_ref="asset-reg")
    rmind.observe("The clinic generator runs on depot fuel.", source_ref="survey")
    rmind.observe("Fuel is delivered to the generator weekly.", source_ref="survey")

    r = rmind.recall("GX-4419", budget=3)
    assert r.chunks and "GX-4419" in r.chunks[0].content, (
        f"got {r.chunks[0].content if r.chunks else 'nothing'!r}")


def test_unrelated_text_stays_below_the_noise_floor(rmind):
    """A real encoder puts unrelated text around 0.5. If that counts as a
    match, everything matches."""
    rmind.observe("The clinic has twelve beds.", source_ref="survey")
    rmind.observe("Route Alpha floods above 40mm rainfall.", source_ref="survey")
    r = rmind.recall("cryptographic key rotation schedule")
    assert r.state is State.DONT_KNOW, r.reason


def test_noise_floor_is_a_property_of_the_embedder():
    """Hard-coding one number makes an engine that works with one model and
    silently fails with another."""
    from owl.adapters.gguf_embed import GgufEmbedder
    from owl.adapters.hashing import HashingEmbedder
    for cls in (RealisticEmbedder, ToyEmbedder, HashingEmbedder, GgufEmbedder):
        assert hasattr(cls, "noise_floor"), cls.__name__
        assert 0.0 <= cls.noise_floor < 0.6, (
            f"{cls.__name__}: the floor is a NOISE CUTOFF, not a decision "
            "threshold - set it high and true matches are discarded")


def test_margin_beats_a_fixed_threshold():
    """Real measurement from BGE-M3: the two bands overlap, so no threshold
    separates them. What separates them is whether the best match rises
    above its background."""
    from owl.metamemory import semantic_density

    # a genuine oblique paraphrase: lower absolute score, big margin
    real = semantic_density(0.69, 0.40, background=0.35)
    # an accidental collision: higher-looking score, no margin
    noise = semantic_density(0.514, 0.40, background=0.46)
    assert real > noise * 2, f"real={real} noise={noise}"

    # and a weak-but-standout match still beats a strong-but-flat one
    weak_standout = semantic_density(0.45, 0.40, background=0.22)
    strong_flat = semantic_density(0.52, 0.40, background=0.50)
    assert weak_standout > strong_flat


def test_background_needs_a_distribution_to_exist(rmind):
    """With fewer than three candidates there is no distribution, and
    inventing one is worse than admitting it: two unrelated notes sitting
    just above the cutoff looked like a strong margin."""
    rmind.observe("The clinic has twelve beds.", source_ref="survey")
    rmind.observe("Route Alpha floods above 40mm rainfall.", source_ref="survey")
    rmind.recall("how many beds does the clinic have")
    assert rmind._last_background is None, "too few candidates to estimate"

    for i in range(6):
        rmind.observe(f"Depot logistics note {i} about convoy scheduling.",
                      source_ref=f"log{i}")
    rmind.recall("convoy scheduling depot")
    assert rmind._last_background is not None


def test_gate_calibration_against_measured_bge_m3():
    """Pinned to numbers MEASURED from bge-m3-Q6_K, not invented.

    The gate has to separate these seven distributions. Earlier versions
    failed on the last two -- an accidental collision among tightly clustered
    unrelated text scored as high as a genuine oblique paraphrase.
    """
    from owl.metamemory import COVERAGE_FLOOR, semantic_density

    def background(sims, floor):
        o = sorted(sims)
        if len(o) < 3:
            return None
        tail = o[:max(1, int(len(o) * 0.6))]
        return sum(tail) / len(tail)

    cases = [
        ("tiny store, perfect match", [0.734], 0.20, True),
        ("tiny store, weak match", [0.528, 0.508], 0.40, False),
        ("clinic 'who' query", [0.52, 0.50, 0.48, 0.36, 0.33, 0.31], 0.40, True),
        ("two good answers", [0.55, 0.52, 0.34, 0.32, 0.30, 0.29], 0.40, True),
        ("clear paraphrase", [0.69, 0.40, 0.38, 0.35, 0.33, 0.30], 0.40, True),
        ("oblique paraphrase", [0.426, 0.31, 0.29, 0.27, 0.25, 0.22], 0.40, True),
        ("unrelated flat", [0.514, 0.49, 0.47, 0.46, 0.44, 0.42], 0.40, False),
        ("unrelated low", [0.35, 0.33, 0.31, 0.30, 0.28, 0.25], 0.40, False),
    ]
    for label, sims, floor, want in cases:
        d = semantic_density(max(sims), floor, background(sims, floor))
        assert (d >= COVERAGE_FLOOR) is want, f"{label}: density {d:.3f}"


def test_ranking_is_monotone_in_similarity(rmind):
    """The margin test belongs to the GATE, never to the ordering. Feeding it
    into ranking crushed the runner-up whenever several candidates were
    plausible -- on BGE-M3 it buried the only note containing a person under
    "The clinic has twelve beds"."""
    docs = ["The clinic generator runs on depot fuel.",
            "The clinic has twelve beds.",
            "Dr Warsame runs the Bardera clinic.",
            "Route Alpha floods above 40mm rainfall.",
            "North well pump needs a gasket.",
            "Generator serial is GX-4419."]
    for d in docs:
        rmind.observe(d, source_ref="survey")

    q = "clinic generator depot fuel"
    qv = rmind._embed([q], Space.READ)[0]
    allowed = {r["node_id"] for r in rmind._s.query(
        "SELECT node_id FROM mem_index")}
    raw = dict(rmind._vec.search(qv, space=Space.READ, allowed=allowed,
                                 top_k=40, floor=rmind.embedder.noise_floor))
    scored, _ = rmind._blend_semantic(q, {}, {"default": "full"})
    common = [n for n in scored if n in raw]
    by_raw = sorted(common, key=lambda n: -raw[n])
    by_score = sorted(common, key=lambda n: -scored[n])
    assert by_raw == by_score, "the gate reordered the candidates"


def test_hubness_penalises_documents_close_to_everything(rmind):
    """A hub is the nearest neighbour of many unrelated queries. Its cosine
    carries less information than the same number from a discriminating
    document -- which is why a bland filler note outranked the real answer
    to "what part is the borehole missing"."""
    from owl.protocols import Space

    specific = ["North well pump needs a 40mm gasket.",
                "Route Alpha floods above 40mm rainfall.",
                "Dr Warsame runs the Bardera clinic.",
                "Generator serial is GX-4419.",
                "The vaccine cold chain logged an excursion.",
                "Fuel arrives from the Kismayo depot."]
    for t in specific:
        rmind.observe(t, source_ref="survey")
    # deliberately bland, sharing a little with everything
    hub = rmind.observe("The site was checked and the equipment was fine.",
                        source_ref="daily")
    for i in range(6):
        rmind.observe(f"Routine daily note {i} about the site and equipment.",
                      source_ref=f"daily{i}")

    scored = rmind._vec.recompute_hubness(Space.READ)
    assert scored > 0
    rows = {r["node_id"]: r["hubness"] for r in rmind._s.query(
        "SELECT node_id,hubness FROM vector WHERE space='read'")}
    assert max(rows.values()) > 0.0, "no document was identified as a hub"
    # hubness is relative to the corpus mean, so a typical doc scores zero
    assert min(rows.values()) == 0.0


def test_hubness_is_relative_not_absolute(rmind):
    """A global offset would change no ranking. Only documents ABOVE the
    corpus mean are discounted."""
    from owl.protocols import Space
    for i in range(12):
        rmind.observe(f"Equally typical note number {i} about supplies.",
                      source_ref=f"n{i}")
    rmind._vec.recompute_hubness(Space.READ)
    vals = [r["hubness"] for r in rmind._s.query(
        "SELECT hubness FROM vector WHERE space='read'")]
    assert max(vals) < 0.25, "a uniform corpus should have no strong hubs"


def test_hubness_can_be_switched_off(rmind):
    from owl.protocols import Space
    for i in range(10):
        rmind.observe(f"Note {i} about depot logistics and scheduling.",
                      source_ref=f"n{i}")
    rmind._vec.recompute_hubness(Space.READ)
    qv = rmind._embed(["depot logistics"], Space.READ)[0]
    on = rmind._vec.search(qv, space=Space.READ, floor=0.0, hubness=True)
    off = rmind._vec.search(qv, space=Space.READ, floor=0.0, hubness=False)
    assert len(on) == len(off)
    # Scores must differ even where the ordering happens to coincide.
    assert dict(on) != dict(off) or all(h == 0 for h in [
        r["hubness"] for r in rmind._s.query(
            "SELECT hubness FROM vector WHERE space='read'")])
