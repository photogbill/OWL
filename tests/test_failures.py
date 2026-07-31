"""Tried and failed - distinct from 'looked and it wasn't there'."""


def test_a_prior_failure_surfaces_before_re_proposing(mind, clock):
    mind.failed("route the convoy via the Km-58 track",
                reason="track is impassable after rain; lost a truck",
                context="March flooding")
    clock.advance(days=20)

    hits = mind.prior_failures("should we route the convoy via Km-58 track?")
    assert hits, "the same rejected option was about to be re-proposed"
    assert "impassable" in hits[0]["reason"]
    assert hits[0]["days_ago"] >= 20


def test_repeat_failures_accumulate_rather_than_duplicate(mind):
    for _ in range(3):
        fid = mind.failed("call the depot directly", reason="no answer")
    hits = mind.prior_failures("call the depot directly")
    assert len(hits) == 1
    assert hits[0]["recurrence"] == 3


def test_unrelated_proposals_are_not_blocked(mind):
    mind.failed("route via Km-58", reason="impassable")
    assert not mind.prior_failures("order more measles vaccine")


def test_conditions_change_and_a_failure_can_be_retired(mind):
    fid = mind.failed("route via Km-58", reason="impassable after rain")
    assert mind.prior_failures("route via Km-58")
    mind.supersede_failure(fid, because="track was regraded in May")
    assert not mind.prior_failures("route via Km-58")
