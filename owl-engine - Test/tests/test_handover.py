"""Memory transplant. The demotion rule is what makes it safe."""
import json
import pytest
from owl import Epistemic, PartitionError, State
from owl.handover import HandoverError, demote


def test_demotion_ladder():
    assert demote(Epistemic.OBSERVED) is Epistemic.REPORTED
    assert demote(Epistemic.INFERRED) is Epistemic.HYPOTHESIZED
    assert demote(Epistemic.HYPOTHESIZED) is None      # dropped
    assert demote(Epistemic.OBSERVED, 2) is Epistemic.INFERRED


def _seed(mind):
    mind.partition("work")
    a = mind.observe("The north well pump needs a 40mm gasket.",
                     origin="user_utterance", source_ref="day3",
                     partition="work", reliability="B", credibility=2)
    b = mind.observe("Dr Warsame runs the Bardera clinic.",
                     origin="user_utterance", source_ref="day1",
                     partition="work")
    mind.derive("Pump failure is the likely contamination source.",
                parents=[a], kind="hypothesis", producer="analysis",
                partition="work",
                falsifier="compare clinic intake dates to the repair log")
    mind.tell("ferrand", a, channel="generated")
    mind.record_absence("diesel supplier in bardera", scope="local vendors",
                        partition="work", reason="canvassed six vendors, none")
    mind.intend("re-check the well after the gasket arrives",
                on_event="gasket delivery", partition="work")
    return a, b


def test_round_trip_demotes_everything_one_rank(mind, tmp_path):
    a, b = _seed(mind)
    pack = tmp_path / "bardera.owlpack"
    man = mind.export_pack(pack, partition="work", exporter="ferrand",
                           label="Bardera handover")
    assert man["counts"]["observations"] == 2

    plan = mind.inspect_pack(pack)
    assert plan["observations_admitted_as"] == "reported"
    assert plan["derived_dropped"] == 1, "their hypothesis is nothing to me"

    mind.partition("incoming")
    stats = mind.graft(pack, as_source="prev:ferrand", partition="incoming")
    assert stats["observations"] == 2 and stats["dropped"] == 1
    assert stats["absences"] == 1 and stats["intentions"] == 1

    r = mind.recall("north well gasket", partition="incoming")
    assert r.state is not State.DONT_KNOW
    c = r.chunks[0]
    assert c.provenance.epistemic is Epistemic.OBSERVED  # substrate is a record
    assert c.provenance.source_ref.startswith("prev:ferrand::")
    assert c.reliability >= "C", "an import can never be grade A: we weren't there"


def test_sealed_partitions_never_export(mind, tmp_path):
    mind.partition("athena", sealed=True)
    mind.observe("I keep replaying the fourth one.", partition="athena",
                 affect=0.9)
    with pytest.raises(PartitionError, match="never be exported"):
        mind.export_pack(tmp_path / "x.owlpack", partition="athena",
                         exporter="bill")


def test_affect_and_suppressed_material_never_travels(mind, tmp_path):
    mind.partition("work")
    ok = mind.observe("Convoy manifest: 3 trucks.", partition="work")
    mind.observe("The smell in the ward still wakes me up.",
                 partition="work", affect=0.85)
    gone = mind.observe("A thing I asked you to drop.", partition="work")
    mind.suppress(gone, reason="user asked")

    pack = tmp_path / "w.owlpack"
    mind.export_pack(pack, partition="work", exporter="bill")
    body = json.loads(pack.read_text())
    contents = [o["content"] for o in body["observations"]]
    assert "Convoy manifest: 3 trucks." in contents
    assert not any("smell in the ward" in c for c in contents)
    assert not any("asked you to drop" in c for c in contents)


def test_tampered_pack_is_refused(mind, tmp_path):
    _seed(mind)
    pack = tmp_path / "t.owlpack"
    mind.export_pack(pack, partition="work", exporter="ferrand")
    body = json.loads(pack.read_text())
    body["observations"][0]["content"] = "The well is fine, no action needed."
    pack.write_text(json.dumps(body))
    with pytest.raises(HandoverError, match="checksum"):
        mind.graft(pack, as_source="prev:ferrand")


def test_independent_operators_corroborate(mind, tmp_path):
    mind.partition("work")
    mind.observe("The bridge at Km 42 is out.", partition="work",
                 source_ref="ferrand", reliability="C", credibility=3)
    pack = tmp_path / "c.owlpack"
    mind.export_pack(pack, partition="work", exporter="ferrand")

    mind.partition("mine")
    mind.observe("The bridge at Km 42 is out.", partition="mine",
                 source_ref="me", reliability="C", credibility=3)
    stats = mind.graft(pack, as_source="prev:ferrand", partition="mine")
    assert stats["corroborated"] == 1, (
        "two independent operators seeing the same thing is corroboration, "
        "not a duplicate")


def test_exposure_history_travels(mind, tmp_path):
    a, _ = _seed(mind)
    pack = tmp_path / "e.owlpack"
    mind.export_pack(pack, partition="work", exporter="ferrand")
    mind.partition("incoming")
    stats = mind.graft(pack, as_source="prev:ferrand", partition="incoming")
    assert stats["exposures"] >= 1, (
        "you inherit not just what they knew but what they'd been told")
