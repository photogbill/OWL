"""An intention can be finished, not only made.

`intend()` had no way out except a commitment resolving, so an operator
who did the thing could not say so and it stayed due for ever -- and a
reminder list that never empties is one nobody reads.
"""
import os
import tempfile
import time

import pytest

from owl import Owl


def _mind():
    return Owl.open(os.path.join(tempfile.mkdtemp(), "i.owl"))


def test_a_done_intention_stops_being_due():
    with _mind() as m:
        iid = m.intend("Re-check Route Alpha after the rain",
                       when=time.time() - 60)
        assert [d["id"] for d in m.due()] == [iid]
        assert m.complete_intention(iid) is True
        assert m.due() == []
        assert m.prefix()["empty"] or "Route Alpha" not in m.prefix()["text"]


def test_a_dropped_intention_is_recorded_as_expired_not_deleted():
    with _mind() as m:
        iid = m.intend("Call the depot", when=time.time() - 60)
        assert m.complete_intention(iid, status="expired")
        row = m._s.one("SELECT status FROM intention WHERE id=?", (iid,))
        assert row["status"] == "expired"


def test_finishing_twice_or_a_wrong_id_is_not_a_second_write():
    with _mind() as m:
        iid = m.intend("Call the depot", when=time.time() - 60)
        assert m.complete_intention(iid)
        assert m.complete_intention(iid) is False
        assert m.complete_intention("int_nope") is False


def test_only_the_two_honest_endings_are_accepted():
    with _mind() as m:
        iid = m.intend("Call the depot", when=time.time() - 60)
        with pytest.raises(ValueError):
            m.complete_intention(iid, status="pending")
