"""Evidence is immutable. Enforced by the database, not by convention."""
import sqlite3
import pytest


def test_observation_cannot_be_updated(mind):
    nid = mind.observe("The bridge at Km 42 is out.")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        mind._s.write(lambda c: c.execute(
            "UPDATE observation SET content='the bridge is fine' WHERE id=?", (nid,)))
    row = mind._node_row(nid)
    assert row["content"] == "The bridge at Km 42 is out."


def test_observation_cannot_be_deleted_without_redaction(mind):
    nid = mind.observe("Sensitive detail.")
    with pytest.raises(sqlite3.IntegrityError, match="redaction"):
        mind._s.write(lambda c: c.execute(
            "DELETE FROM observation WHERE id=?", (nid,)))


def test_redaction_permits_deletion(mind):
    nid = mind.observe("Please forget this.")
    def _w(c):
        c.execute("INSERT INTO redaction VALUES(?,?,?)",
                  (nid, mind.clock.now(), "user request"))
        c.execute("DELETE FROM observation WHERE id=?", (nid,))
    mind._s.write(_w)
    assert mind._node_row(nid) is None
