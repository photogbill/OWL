"""Heterogeneous graph: entities bridge what vocabulary cannot."""
from owl import State
from owl.entities import canonicalise, predict_answer_type


def test_entity_bridges_notes_with_no_shared_words(mind, clock):
    """The multi-hop case flat retrieval structurally misses."""
    a = mind.observe("Dr Warsame signed off on the cold chain log.",
                     source_ref="day3")
    mind.link(a, mentions=[("Dr Warsame", "person"),
                           ("cold chain log", "artifact")],
              relations=[("Dr Warsame", "signed", "cold chain log")])

    clock.advance(days=35)
    b = mind.observe("Measles vials were discarded after the excursion.",
                     source_ref="day38")
    mind.link(b, mentions=[("cold chain log", "artifact"),
                           ("measles vials", "artifact")],
              relations=[("cold chain log", "records", "measles vials")])

    # No shared content words between the query target and note b.
    r = mind.recall("Dr Warsame", budget=5)
    ids = {c.node_id for c in r.chunks}
    assert a in ids
    assert b in ids, "entity bridge failed: the two notes share no vocabulary"


def test_relations_carry_their_evidence(mind):
    """MiniRAG's edges are asserted. OWL's point at what justifies them."""
    nid = mind.observe("Ahmed drives the depot truck for Warsame.",
                       source_ref="day9")
    mind.link(nid, mentions=[("Ahmed", "person"), ("Warsame", "person")],
              relations=[("Ahmed", "drives for", "Warsame")])
    paths = mind.paths("Ahmed")
    assert paths
    step = paths[0].steps[0]
    assert step.evidence_node == nid
    assert mind._node_row(step.evidence_node)["source_ref"] == "day9"


def test_path_is_denser_than_its_evidence(mind):
    nid1 = mind.observe(
        "Following the meeting on Tuesday, it was confirmed that Ahmed, who "
        "has driven for the programme since March, operates the depot truck "
        "on behalf of Dr Warsame at the Bardera clinic.", source_ref="d1")
    mind.link(nid1, mentions=[("Ahmed", "person"), ("Warsame", "person")],
              relations=[("Ahmed", "drives for", "Warsame")])
    nid2 = mind.observe(
        "It should further be noted that Dr Warsame retains overall "
        "responsibility for the Bardera clinic and its associated cold "
        "chain equipment throughout the current reporting period.",
        source_ref="d2")
    mind.link(nid2, mentions=[("Warsame", "person"), ("Bardera clinic", "org")],
              relations=[("Warsame", "runs", "Bardera clinic")])

    path = mind.paths("Ahmed", max_hops=3)[0]
    evidence_len = sum(len(mind._node_row(e)["content"]) for e in path.evidence)
    assert len(path.render()) < evidence_len / 3, (
        f"path {len(path.render())} chars vs evidence {evidence_len}")


def test_entities_dedupe_conservatively():
    assert canonicalise("Dr. Warsame") == canonicalise("Warsame")
    assert canonicalise("The Bardera Clinic") == canonicalise("bardera clinic")
    # Merging two distinct people is far worse than carrying a duplicate.
    assert canonicalise("Ahmed Hassan") != canonicalise("Ahmed Hussein")


def test_answer_type_prediction():
    assert predict_answer_type("who runs the clinic") == "person"
    assert predict_answer_type("how many beds are there") == "quantity"
    assert predict_answer_type("when does the convoy leave") == "time"
    assert predict_answer_type("what is the generator serial") == "identifier"
    assert predict_answer_type("summarise the situation") is None


def test_answer_type_demotes_but_never_vetoes(mind):
    """A heuristic must not be able to veto a genuine hit."""
    nid = mind.observe("The clinic operates a cold chain refrigerator.",
                       source_ref="d1")
    mind.link(nid, mentions=[("cold chain refrigerator", "artifact")])
    r = mind.recall("who runs the clinic")          # asks for a person
    assert r.state is State.KNOW_WHERE, (
        "strong topical match with no person entity should demote, not vetoed")
    assert r.chunks, "the hit must still be returned"


def test_no_entity_graph_means_no_signal_not_a_negative(mind):
    """An absent signal must not look like a signal that fired."""
    mind.observe("Dr Warsame runs the Bardera clinic.", source_ref="d1")
    assert mind.recall("who runs the clinic").state is State.KNOW


def test_answer_type_affinity_prefers_text_that_can_answer(mind):
    """The case this exists for: a bi-encoder finds these nearly equally
    similar (both about a clinic, both use "runs"), but only one contains a
    person and so only one can answer "who is in charge"."""
    from owl.entities import content_affinity, predict_answer_type
    want = predict_answer_type("who is in charge of the medical centre")
    assert want == "person"
    assert content_affinity("Dr Warsame runs the Bardera clinic.", want) > 1.0
    assert content_affinity("The clinic generator runs on depot fuel.",
                            want) < 1.0


def test_affinity_does_not_mistake_place_names_for_people():
    """A bare Title Case pair is not enough: 'Route Alpha', 'Grid North' and
    'North Well' all match the naive rule."""
    from owl.entities import content_affinity
    for text in ("Route Alpha floods above 40mm rainfall.",
                 "Grid North was resurveyed last week.",
                 "North well pump needs a gasket.",
                 "Camp Bravo relocated in May."):
        assert content_affinity(text, "person") < 1.0, text
    for text in ("Dr Warsame runs the clinic.",
                 "Ahmed Hassan collected the parts."):
        assert content_affinity(text, "person") > 1.0, text


def test_affinity_is_a_multiplier_never_a_filter(mind):
    """The predictor is a regex heuristic and must not be able to veto a
    genuine match."""
    from owl.entities import content_affinity
    assert content_affinity("anything at all", None) == 1.0
    assert content_affinity("anything at all", "unmodelled_type") == 1.0
    assert content_affinity("no person here", "person") > 0.0

    nid = mind.observe("The clinic generator runs on depot fuel.",
                       source_ref="survey")
    r = mind.recall("who runs the clinic")
    assert r.chunks and r.chunks[0].node_id == nid, (
        "the only candidate must still be returned")
