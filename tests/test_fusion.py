"""Record fusion and the verbatim protection."""
from owl import State
from owl.epistemics import classify
from owl.fusion import UnionFind, plan


def test_union_find_clusters_transitively():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")
    comps = {tuple(sorted(v)) for v in uf.components().values()}
    assert ("a", "b", "c") in comps and ("d",) in comps


def test_plan_dedupes_then_clusters():
    p = plan([("a", "b", 0.93), ("c", "d", 0.80), ("x", "y", 0.40)])
    assert p.duplicates == [("a", "b")]
    assert p.clusters == [["c", "d"]]


def test_verbatim_content_is_never_fused():
    p = plan([("k", "m", 0.99)], verbatim={"k"})
    assert p.is_empty and p.skipped_verbatim == 1


def test_verbatim_classification():
    """Content that is worthless unless exact."""
    for text in ["Grid 31U DQ 48251 11932",
                 "The generator serial is GX-4419.",
                 "Give 250 mg every six hours.",
                 "Net control is on 145.500 MHz.",
                 "The bypass sequence is star-seven-four."]:
        assert classify(text) == "verbatim", text
    assert classify("The clinic has twelve beds.") != "verbatim"


def test_verbatim_never_goes_stale(mind, clock):
    """An exact string does not become less exact."""
    mind.observe("Depot grid is 31U DQ 48251 11932.", source_ref="d1")
    clock.advance(days=400)
    c = mind.recall("depot grid").chunks[0]
    assert c.claim_class == "verbatim"
    assert c.staleness == 0.0


def test_composite_is_never_more_certain_than_its_members(mind):
    a = mind.observe("Supply run completed on Tuesday.", source_ref="d1")
    b = mind.observe("Supply run completed Tuesday afternoon.", source_ref="d2")
    weak = mind.derive("Supplies are adequate.", parents=[a], kind="abstraction",
                       producer="analyst", confidence=0.4)
    cid = mind._make_composite([a, b, weak], "default", 0)
    if cid:
        row = mind._node_row(cid)
        assert row["confidence"] <= 0.4
        assert row["epistemic"] in ("inferred", "hypothesized")


def test_tend_reports_fusion(mind):
    for i in range(4):
        mind.observe(f"Weekly stock check completed, run {i}.", source_ref=f"w{i}")
    report = mind.tend()
    assert "fusion" in report
    assert set(report["fusion"]) >= {"merged", "composites", "protected"}
