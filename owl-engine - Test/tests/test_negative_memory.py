"""Absence, and the answers that are not 'I don't know'."""
from owl import State


def test_recorded_absence_beats_dont_know(mind):
    plain = mind.recall("diesel supplier in bardera")
    assert plain.state is State.DONT_KNOW

    mind.record_absence("diesel supplier in bardera", scope="local vendors",
                        reason="canvassed all six vendors, none stock diesel")
    again = mind.recall("diesel supplier in bardera")
    assert again.state is State.SEARCHED_AND_ABSENT
    assert again.informative
    assert "canvassed" in again.reason


def test_absence_outranks_a_weak_lexical_match(mind):
    """Expensive knowledge must not be buried by a shared keyword."""
    mind.observe("Bardera clinic supply list reviewed.", source_ref="notes")
    mind.record_absence("diesel supplier in bardera", scope="local vendors",
                        reason="canvassed all six vendors, none stock diesel")
    r = mind.recall("diesel supplier in bardera")
    assert r.state is State.SEARCHED_AND_ABSENT
    # ...and it still shows what related material does exist.
    assert r.chunks and "supply list" in r.chunks[0].content


def test_knew_once_is_not_dont_know(mind, clock):
    nid = mind.observe("Warehouse alarm bypass sequence is star-seven-four.",
                       origin="document", source_ref="file://handover.pdf#p9")
    mind._s.write(lambda c: c.execute(
        "UPDATE mem_index SET tier='pruned' WHERE node_id=?", (nid,)))
    r = mind.recall("warehouse alarm bypass sequence")
    assert r.state is State.KNEW_ONCE
    assert r.informative and not r
    assert r.chunks[0].provenance.source_ref == "file://handover.pdf#p9"
    assert r.chunks[0].content == "", "detail should not be fabricated"


def test_suppression_demotes_without_deleting(mind):
    nid = mind.observe("The third tent. I am not doing this again tonight.",
                       origin="user_utterance", source_ref="athena/night2")
    mind.suppress(nid, reason="user asked me to let it go")
    assert mind._node_row(nid) is not None, "suppression must not delete"
    row = mind._s.one("SELECT suppress_reason FROM mem_index WHERE node_id=?", (nid,))
    assert row["suppress_reason"] == "user asked me to let it go"
