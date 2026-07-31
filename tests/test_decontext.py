"""Make a memory readable cold - without guessing."""
from owl import Epistemic, State
from owl.decontext import expand, needs_context, resolve_date

AT = 1_700_000_000.0          # 2023-11-14, a Tuesday
DAY = 86400.0


def test_relative_dates_anchor_to_when_it_was_said():
    assert resolve_date("tomorrow", AT) == "2023-11-15"
    assert resolve_date("yesterday", AT) == "2023-11-13"
    assert resolve_date("Thursday", AT) == "2023-11-16"
    assert resolve_date("last Friday", AT) == "2023-11-10"


def test_pronouns_resolve_against_recent_entities():
    e = expand("He said it would arrive Thursday.", at=AT,
               candidates=[("the gasket", "artifact"), ("Ahmed", "person")])
    assert "Ahmed said the gasket" in e.text
    assert "2023-11-16" in e.text
    assert e.standalone


def test_ambiguity_is_refused_not_guessed():
    """A wrong substitution is invisible to the reader; an unresolved
    pronoun is not."""
    e = expand("He told her about it.", at=AT,
               candidates=[("Ahmed", "person"), ("Fatima", "person")])
    assert e.text == "He told her about it.", "must not guess between two people"
    assert not e.standalone
    assert "He" in e.unresolved


def test_one_referent_is_not_reused_for_two_pronouns():
    """Without this, 'they will deliver it' resolved both pronouns to the
    same org and produced 'the depot will deliver the depot'."""
    e = expand("They will deliver it next week.", at=AT,
               candidates=[("the depot", "org"), ("the pump", "artifact")])
    assert "the depot will deliver the pump" in e.text

    e2 = expand("They will deliver it next week.", at=AT,
                candidates=[("the depot", "org")])
    assert "deliver it" in e2.text, "nothing left to resolve to -> refuse"
    assert "it" in e2.unresolved


def test_needs_context_is_a_cheap_precheck():
    assert needs_context("He said it arrives Thursday.")
    assert not needs_context("The clinic in Bardera has twelve beds.")


def test_expansion_is_a_derived_node_never_a_rewrite(mind, clock):
    """The raw utterance is evidence. The expansion is a derivation."""
    a = mind.observe("Ahmed inspected the gasket.", origin="user_utterance",
                     source_ref="conv:1")
    mind.link(a, mentions=[("Ahmed", "person"), ("the gasket", "artifact")])
    clock.advance(seconds=60)
    b = mind.observe("He said it would arrive Thursday.",
                     origin="user_utterance", source_ref="conv:1")

    res = mind.decontextualise(b)
    assert res["changed"]
    assert "Ahmed" in res["text"] and "the gasket" in res["text"]

    # the observation is untouched
    assert mind._node_row(b)["content"] == "He said it would arrive Thursday."
    d = mind._node_row(res["derived"])
    assert d["kind"] == "decontext"


def test_expansion_improves_recall(mind, clock):
    a = mind.observe("Ahmed inspected the north well gasket.",
                     origin="user_utterance", source_ref="conv:1")
    mind.link(a, mentions=[("Ahmed", "person"),
                           ("the north well gasket", "artifact")])
    clock.advance(seconds=60)
    mind.observe("He said it would arrive Thursday.",
                 origin="user_utterance", source_ref="conv:1")
    mind.tend()

    r = mind.recall("when did Ahmed say the gasket arrives")
    assert r.state is not State.DONT_KNOW
    assert any("Ahmed" in c.content and "gasket" in c.content
               for c in r.chunks)


def test_standalone_text_is_skipped(mind):
    n = mind.observe("The clinic in Bardera has twelve beds.",
                     source_ref="survey")
    res = mind.decontextualise(n)
    assert not res["changed"] and "standalone" in res["reason"]


def test_tend_reports_the_pass(mind):
    mind.observe("He said it arrives Thursday.", source_ref="conv:1")
    rep = mind.tend()
    assert "decontext" in rep
    assert set(rep["decontext"]) == {"scanned", "expanded", "refused"}
