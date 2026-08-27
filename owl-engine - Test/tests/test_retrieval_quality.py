"""Diversity, the familiarity/recollection split, and token cost."""
from owl import State
from owl.metamemory import recollection_score


def test_group_by_stops_one_document_filling_the_answer(mind):
    """Five chunks from one file look like five pieces of evidence and are
    one."""
    for i in range(8):
        mind.observe(f"Fuel delivery log entry {i}: depot diesel movement.",
                     origin="document", source_ref="file://logbook.pdf")
    for i in range(2):
        mind.observe(f"Field note {i}: depot diesel collected by convoy.",
                     origin="user_utterance", source_ref=f"conv:day{i}")

    # budget 4 is fillable from the caps alone (2 + 1 + 1), so the cap holds
    r = mind.recall("depot diesel", budget=4, group_by="source", per_group=2)
    by_src = {}
    for c in r.chunks:
        by_src[c.provenance.source_ref] = by_src.get(c.provenance.source_ref, 0) + 1
    assert max(by_src.values()) <= 2, f"one source dominated: {by_src}"
    assert len(by_src) == 3, "the answer should span all three sources"

    # With a larger budget the log book takes more slots -- correctly, since
    # there is nothing else left to show. The cap shapes the order; it does
    # not withhold material.
    wide = mind.recall("depot diesel", budget=8, group_by="source",
                       per_group=2)
    assert len(wide.chunks) > len(r.chunks)


def test_group_cap_never_starves_a_small_store(mind):
    """The cap shapes the ORDER. When the budget cannot be filled from
    distinct groups, overflow backfills rather than returning less -- a
    diversity rule that silently shrinks answers is worse than no rule."""
    for i in range(5):
        mind.observe(f"Depot note {i} about diesel.", origin="document",
                     source_ref="file://only.pdf")
    r = mind.recall("depot diesel", budget=4, group_by="source", per_group=1)
    assert len(r.chunks) == 4, "overflow must backfill the budget"


def test_group_by_can_be_switched_off(mind):
    for i in range(6):
        mind.observe(f"Depot note {i} about diesel.", source_ref="file://a.pdf")
    r = mind.recall("depot diesel", budget=4, group_by=None)
    assert len(r.chunks) == 4


def test_recollection_is_more_than_familiarity():
    thin = recollection_score(has_episode=False, has_period=False,
                              n_neighbours=0, has_provenance=False,
                              decontextualised=False)
    rich = recollection_score(has_episode=True, has_period=True,
                              n_neighbours=4, has_provenance=True,
                              decontextualised=True)
    assert thin < 0.2 < rich
    assert rich <= 1.0


def test_familiar_is_a_distinct_honest_state(mind):
    """Seen before, but nothing anchors it: no siblings, no period, no
    provenance, no links. Returning chunks as if placed would overstate what
    is actually held."""
    mind.observe("Kalonji ridge is mentioned in passing somewhere.",
                 origin="document")
    # Partial match: enough to feel familiar, not enough to be a direct hit,
    # and nothing anchoring it.
    r = mind.recall("kalonji plateau elevation")
    assert r.state is State.FAMILIAR, r.reason
    assert "cannot place it" in r.reason
    assert r.informative and bool(r)


def test_an_episode_of_one_is_not_context(mind):
    """Every observation is assigned an episode, so counting bare membership
    handed out free recollection and made FAMILIAR unreachable."""
    lone = mind.observe("A single isolated remark.", origin="document")
    assert mind._recollection(mind._node_row(lone)) < 0.20


def test_context_lifts_a_memory_out_of_merely_familiar(mind, clock):
    with mind.period("bardera"):
        a = mind.observe("Kalonji ridge is the northern approach.",
                         origin="document", source_ref="file://survey.pdf")
        mind.link(a, mentions=[("Kalonji ridge", "place")])
        clock.advance(seconds=30)
        mind.observe("The ridge road was graded in May.",
                     origin="document", source_ref="file://survey.pdf")
    r = mind.recall("kalonji ridge")
    assert r.state is not State.FAMILIAR, (
        "an anchored memory should be recollected, not merely familiar")


def test_recall_reports_its_token_cost(mind):
    for i in range(5):
        mind.observe("A reasonably long field note about depot diesel "
                     f"logistics and convoy scheduling, entry {i}.",
                     source_ref=f"note{i}")
    r = mind.recall("depot diesel", budget=5)
    assert r.tokens > 0
    assert r.tokens >= sum(len(c.content) // 4 for c in r.chunks) * 0.8


def test_token_budget_is_respected(mind):
    for i in range(6):
        mind.observe("A long field note about depot diesel logistics and "
                     f"convoy scheduling with plenty of detail, entry {i}.",
                     source_ref=f"note{i}")
    tight = mind.recall("depot diesel", budget=6, token_budget=40)
    loose = mind.recall("depot diesel", budget=6)
    assert len(tight.chunks) < len(loose.chunks)
    assert tight.tokens <= loose.tokens
