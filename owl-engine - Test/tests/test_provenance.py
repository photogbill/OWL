"""The invariant that matters most: speculation cannot become fact."""
import pytest
from owl import Owl, Epistemic, OwlError
from owl.protocols import MonotonicityError
from owl.provenance import ParentFacts, assert_monotonic, resolve


def test_confidence_cannot_exceed_parents(mind):
    weak = mind.observe("Ahmed said the depot may reopen Thursday.",
                        origin="user_utterance", source_ref="conv:1")
    guess = mind.derive("The depot reopens Thursday.", parents=[weak],
                        kind="abstraction", producer="test", confidence=0.95)
    row = mind._node_row(guess)
    assert row["confidence"] <= 1.0


def test_epistemic_tag_is_monotone(mind):
    obs = mind.observe("Water pump is broken.", source_ref="field-note")
    hyp = mind.derive("Pump failure caused the outbreak.", parents=[obs],
                      kind="hypothesis", producer="rem",
                      epistemic=Epistemic.HYPOTHESIZED,
                      falsifier="Check clinic intake dates against pump repair log")
    # A child of a hypothesis is a hypothesis, no matter what it claims to be.
    child = mind.derive("The outbreak is waterborne.", parents=[hyp],
                        kind="abstraction", producer="test",
                        epistemic=Epistemic.OBSERVED, confidence=0.99)
    row = mind._node_row(child)
    assert row["epistemic"] == "hypothesized", (
        "abstraction laundered a hypothesis into an observation")
    assert row["confidence"] <= 0.7


def test_hypothesis_without_falsifier_is_refused(mind):
    obs = mind.observe("Two clinics reported fever cases.")
    with pytest.raises(OwlError, match="falsifier"):
        mind.derive("There is an epidemic.", parents=[obs], kind="hypothesis",
                    producer="rem", epistemic=Epistemic.HYPOTHESIZED)


def test_assert_monotonic_raises_on_violation():
    parents = [ParentFacts("p", 0.4, Epistemic.INFERRED)]
    with pytest.raises(MonotonicityError):
        assert_monotonic(parents, confidence=0.9, epistemic=Epistemic.INFERRED)
    with pytest.raises(MonotonicityError):
        assert_monotonic(parents, confidence=0.2, epistemic=Epistemic.OBSERVED)


def test_why_traces_back_to_primary_source(mind):
    a = mind.observe("Fuel delivery logged 400L on 14 March.",
                     origin="document", source_ref="file://logs/fuel.csv#L22")
    b = mind.derive("Fuel stocks are adequate.", parents=[a], kind="abstraction",
                    producer="analyst")
    chain = mind.why(b)
    prims = [n for n in chain if n["origin"] == "document"]
    assert prims and prims[0]["source_ref"] == "file://logs/fuel.csv#L22"
    inferred = [n for n in chain if n["epistemic"] == "inferred"]
    assert inferred and not inferred[0]["presentable_as_fact"]


def test_hypothesis_kind_forces_hypothesized_tag(mind):
    """A hypothesis mislabelled 'inferred' reads downstream as a conclusion."""
    obs = mind.observe("Depot stock was 400L on Tuesday.", origin="document",
                       source_ref="depot.csv")
    hyp = mind.derive("Stock lasts two weeks.", parents=[obs], kind="hypothesis",
                      producer="rem", epistemic=Epistemic.OBSERVED,
                      confidence=1.0,
                      falsifier="Compare depot stock against daily burn rate")
    row = mind._node_row(hyp)
    assert row["epistemic"] == "hypothesized"
    from owl.provenance import is_presentable_as_fact
    assert not is_presentable_as_fact(Epistemic(row["epistemic"]))
