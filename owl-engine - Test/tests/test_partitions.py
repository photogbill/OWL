"""Information-flow control. Athena's confidentiality boundary, enforced.

The ATK design states it plainly: what is said to Athena stays private -- NOT
fed to the work graph, reports, or the intercept log. That is an architectural
requirement, not a policy note, so it belongs in the persistence layer where a
future refactor cannot quietly break it.
"""
import pytest
from owl import PartitionError, State


def test_sealed_partition_never_leaks(mind):
    mind.partition("work")
    mind.partition("athena", sealed=True)

    mind.observe("Convoy manifest: 3 trucks, 40 cases ORS.",
                 partition="work", source_ref="manifest.pdf")
    mind.observe("I keep seeing the child from the third tent when I close my eyes.",
                 partition="athena", origin="user_utterance", affect=0.9)

    leaked = mind.recall("child tent", partition="work")
    assert leaked.state is State.DONT_KNOW, "companion memory leaked into work"

    own = mind.recall("child tent", partition="athena")
    assert own.state is not State.DONT_KNOW, "companion cannot recall its own memory"


def test_flow_is_directional_not_symmetric(mind):
    mind.partition("intake", flows_to=["analysis"])
    mind.observe("Border crossing closed at 1400.", partition="intake")
    assert mind.recall("border crossing", partition="analysis").state is not State.DONT_KNOW
    mind.observe("Analyst conclusion: rerouting required.", partition="analysis")
    assert mind.recall("analyst conclusion rerouting", partition="intake").state is State.DONT_KNOW


def test_sealed_partition_cannot_declare_outflow(mind):
    with pytest.raises(PartitionError):
        mind.partition("private", sealed=True, flows_to=["work"])


def test_affect_suppression_filters_without_deleting(mind):
    mind.partition("athena", sealed=True)
    nid = mind.observe("The smell in the ward still wakes me up.",
                       partition="athena", affect=0.85)
    quiet = mind.recall("ward smell", partition="athena",
                        suppress_affect_above=0.5)
    assert quiet.state is State.DONT_KNOW
    # Not deleted -- still addressable on an explicit request.
    direct = mind.recall("ward smell", partition="athena")
    assert any(c.node_id == nid for c in direct.chunks)


def test_one_way_membrane_full(mind):
    """Athena remembers everything Athena-related AND knows the work context;
    nothing she thinks reaches the work arena."""
    mind.partition("work")
    mind.partition("athena", sealed=True, reads_from=["work"])

    mind.observe("Clinic triage ran 14 hours; four did not make it.",
                 partition="work", source_ref="fieldnotes/day9")
    mind.observe("I keep replaying the fourth one.",
                 partition="athena", origin="user_utterance", affect=0.9)

    # Athena sees both sides.
    assert mind.recall("clinic triage hours", partition="athena").state \
        is not State.DONT_KNOW
    assert mind.recall("keep replaying", partition="athena").state \
        is not State.DONT_KNOW
    # Work sees only work. Ever.
    assert mind.recall("keep replaying", partition="work").state is State.DONT_KNOW


def test_graded_permeability_admits_summaries_only(mind):
    mind.partition("work")
    mind.partition("athena", sealed=True, summary_reads_from=["work"])

    raw = mind.observe("Patient 4, aged six, arrested at 0340 during triage.",
                       partition="work", source_ref="fieldnotes/day9")
    mind.derive("Day nine at the clinic was sustained and costly.",
                parents=[raw], kind="summary", producer="nrem",
                partition="work", confidence=0.9)

    # The abstraction crosses...
    assert mind.recall("day nine clinic costly", partition="athena").state \
        is not State.DONT_KNOW
    # ...the raw detail does not.
    detail = mind.recall("patient aged six arrested triage", partition="athena")
    assert all("aged six" not in c.content for c in detail.chunks), \
        "raw traumatic detail crossed a summary-only boundary"


def test_cannot_read_from_a_sealed_partition(mind):
    mind.partition("athena", sealed=True)
    with pytest.raises(PartitionError):
        mind.partition("work", reads_from=["athena"])
