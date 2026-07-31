"""What did we hold to be true on date X?"""
from owl import State

DAY = 86400.0


def test_as_of_respects_world_time_validity(mind, clock):
    t0 = clock.now()
    mind.observe("Route Alpha is open.", source_ref="sitrep-1",
                 valid_from=t0, valid_to=t0 + 5 * DAY)
    mind.observe("Route Alpha is closed by flooding.", source_ref="sitrep-2",
                 valid_from=t0 + 5 * DAY)
    clock.advance(days=10)

    early = mind.recall("route alpha", as_of=t0 + 1 * DAY)
    assert early.chunks and "open" in early.chunks[0].content.lower()

    late = mind.recall("route alpha", as_of=t0 + 8 * DAY)
    assert late.chunks and "closed" in late.chunks[0].content.lower()
