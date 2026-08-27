"""Retrieval receipts — a retrieval you can't reconstruct can't be audited."""
from owl import State


def test_every_recall_leaves_a_receipt(mind):
    mind.observe("The clinic has twelve beds.", source_ref="survey")
    mind.recall("how many beds")
    r = mind.receipts_for()
    assert r and r[0]["query"] == "how many beds"
    assert r[0]["state"] in (s.value for s in State)
    assert r[0]["returned"], "returned set must be recorded"
    assert r[0]["returned"][0]["src"] == "survey", "provenance in the receipt"


def test_dont_know_is_receipted_too(mind):
    mind.observe("Unrelated content.", source_ref="d1")
    mind.recall("helicopter tail number")
    r = mind.receipts_for()
    assert r[0]["state"] == "dont_know"
    assert r[0]["returned"] == []
    assert r[0]["reason"]


def test_receipt_records_near_misses(mind):
    for i in range(6):
        mind.observe(f"Fuel delivery log entry {i} for the depot.",
                     source_ref=f"log{i}")
    mind.recall("fuel delivery depot", budget=2)
    r = mind.receipts_for()[0]
    assert len(r["returned"]) == 2
    assert r["rejected"], "what was considered and rejected must be recorded"


def test_receipts_can_be_disabled(mind):
    mind.receipts = False
    mind.observe("Something.", source_ref="d1")
    mind.recall("something")
    assert not mind.receipts_for()
