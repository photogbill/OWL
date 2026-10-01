"""v5 #10 -- cross-lingual claim identity.

ATK translates: a radio intercept in Somali, a Pashto document, an Arabic
leaflet. Two failures are easy and both are silent:

  * the translation is stored as a NEW fact, so a report and its own
    translation corroborate each other -- one source counted twice;
  * the original is replaced by the translation, so the evidence is gone and
    nobody can check the rendering against it.

A translation here is a derived node whose only parent is the original. The
original stays the evidence; the translation is a reading of it.
"""
import os
import tempfile

import pytest

from owl import Owl, OwlError

SOMALI = "Isbitaalka Bardera wuxuu leeyahay shidaal ku filan saddex maalmood."
ENGLISH = "The Bardera hospital has fuel for three days."


def _mind():
    return Owl.open(os.path.join(tempfile.mkdtemp(), "x.owl"))


def test_the_language_is_recorded_and_unknown_is_not_english():
    with _mind() as m:
        so = m.observe(SOMALI, origin="tool_output", source_ref="rf:1:146e6",
                       lang="so")
        plain = m.observe("Route Alpha is open.", origin="document",
                          source_ref="sitrep")
        assert m.language_of(so) == "so"
        assert m.language_of(plain) is None


def test_a_misspelt_language_is_refused_not_invented():
    with _mind() as m:
        with pytest.raises(ValueError):
            m.observe(SOMALI, lang="Somali language")
        so = m.observe(SOMALI, lang="SO")
        assert m.language_of(so) == "so", "tags are normalised"


def test_a_translation_is_a_reading_and_the_original_stays_the_evidence():
    with _mind() as m:
        so = m.observe(SOMALI, origin="tool_output", source_ref="rf:1:146e6",
                       lang="so")
        en = m.translation(so, ENGLISH, lang="en", producer="atk.translate")
        assert m.language_of(en) == "en"
        assert m.original_of(en) == so
        assert m.original_of(so) == so
        chain = m.why(en)
        assert chain[0]["kind"] == "translation"
        assert any(step["id"] == so for step in chain), \
            "why() on a translation must reach the original"
        assert [t["id"] for t in m.translations_of(so)] == [en]


def test_a_report_and_its_own_translation_are_ONE_source():
    with _mind() as m:
        so = m.observe(SOMALI, origin="tool_output", source_ref="rf:1:146e6",
                       lang="so")
        m.translation(so, ENGLISH, lang="en", producer="atk.translate")
        c = m.corroborated(ENGLISH)
        assert c["independent"] == 1, "a translation corroborated its source"
        assert c["documents"] == 1
        assert c["via_translation"] == 1


def test_the_same_claim_from_an_independent_source_in_another_language_corroborates():
    with _mind() as m:
        so = m.observe(SOMALI, origin="tool_output", source_ref="rf:1:146e6",
                       lang="so")
        m.translation(so, ENGLISH, lang="en", producer="atk.translate")
        m.observe(ENGLISH, origin="document", source_ref="unhcr-sitrep-14",
                  lang="en")
        c = m.corroborated(ENGLISH)
        assert c["independent"] == 2, c
        assert c["languages"] == ["en", "so"]


def test_two_translations_of_one_report_are_still_one_source():
    with _mind() as m:
        so = m.observe(SOMALI, origin="tool_output", source_ref="rf:1:146e6",
                       lang="so")
        m.translation(so, ENGLISH, lang="en", producer="atk.translate")
        m.translation(so, ENGLISH, lang="en", producer="analyst")
        c = m.corroborated(ENGLISH)
        assert c["independent"] == 1 and c["documents"] == 1


def test_a_translation_that_changes_a_dose_is_refused():
    with _mind() as m:
        fr = m.observe("Donner 250 mg de paracétamol toutes les six heures.",
                       origin="document", source_ref="msf-protocol", lang="fr")
        with pytest.raises(OwlError) as e:
            m.translation(fr, "Give 250 g of paracetamol every six hours.",
                          lang="en", producer="atk.translate")
        assert "dimensional integrity" in str(e.value)
        ok = m.translation(fr, "Give 250 mg of paracetamol every six hours.",
                           lang="en", producer="atk.translate")
        assert m.original_of(ok) == fr


def test_a_translation_cannot_outrank_what_it_translates():
    with _mind() as m:
        obs = m.observe(SOMALI, origin="tool_output", source_ref="rf:1",
                        lang="so")
        hyp = m.derive("Waxaa laga yaabaa in shidaalku dhammaado.",
                       parents=[obs], kind="hypothesis", producer="t",
                       confidence=0.3,
                       falsifier="a fuel reading above three days on Friday")
        tr = m.translation(hyp, "The fuel may run out.", lang="en",
                           producer="atk.translate", confidence=0.95)
        row = m._node_row(tr)
        assert row["confidence"] <= 0.3
        assert row["epistemic"] == "hypothesized", \
            "a translated hypothesis is still a hypothesis"


def test_translating_into_the_same_language_is_a_paraphrase_and_refused():
    with _mind() as m:
        en = m.observe(ENGLISH, lang="en")
        with pytest.raises(OwlError):
            m.translation(en, "Bardera hospital: three days of fuel.",
                          lang="en", producer="t")


def test_an_english_question_finds_somali_evidence_through_its_translation():
    with _mind() as m:
        so = m.observe(SOMALI, origin="tool_output", source_ref="rf:1:146e6",
                       lang="so")
        m.translation(so, ENGLISH, lang="en", producer="atk.translate")
        r = m.recall("how many days of fuel does the Bardera hospital have")
        assert r.chunks, "nothing found"
        assert so in {m.original_of(c.node_id) for c in r.chunks}


# ── migration: an ATK ledger written before 'translation' existed ────────

def _old_store(path):
    """A store as the previous engine wrote it: the same schema, with the
    derived-kind list one item shorter. Built from the real schema text so
    the test drifts with it."""
    import sqlite3
    from pathlib import Path as P
    import owl as owl_pkg
    ddl = (P(owl_pkg.__file__).parent / "schema.sql").read_text()
    old = ddl.replace(",\n                     'translation')", ")")
    assert old != ddl, "the schema's kind list moved; update this test"
    c = sqlite3.connect(path)
    c.executescript(old)
    c.close()


def test_an_older_store_is_widened_on_open_and_keeps_everything():
    import sqlite3
    from owl import migrations
    path = os.path.join(tempfile.mkdtemp(), "old.owl")
    _old_store(path)
    c = sqlite3.connect(path)
    assert "translation" not in migrations.derived_kinds(c)
    c.close()
    with Owl.open(path) as m:
        assert m._s.migration.get("derived_kinds_added") == ["translation"]
        so = m.observe(SOMALI, lang="so", source_ref="rf:1")
        summary = m.derive("Bardera: fuel for three days.", parents=[so],
                           kind="summary", producer="t", confidence=0.8)
        en = m.translation(so, ENGLISH, lang="en", producer="t")
        assert m.original_of(en) == so and m._node_row(summary)
    with Owl.open(path) as m:                      # idempotent second open
        assert not (m._s.migration or {}).get("derived_kinds_added")
        assert m._node_row(summary)["content"].startswith("Bardera")
        assert m.original_of(en) == so
        rep = m.doctor()
        assert rep["healthy"], rep["report"]


def test_rows_written_before_the_migration_survive_it():
    """The rebuild copies every row. Written with the old schema, read back
    after the widening."""
    import sqlite3
    path = os.path.join(tempfile.mkdtemp(), "old.owl")
    _old_store(path)
    with Owl.open(path) as m:
        pass                                        # first open migrates
    # Simulate data that existed before: write with a fresh old store,
    # through the engine, then widen.
    path2 = os.path.join(tempfile.mkdtemp(), "old2.owl")
    _old_store(path2)
    c = sqlite3.connect(path2)
    c.execute("INSERT INTO partition(name,sealed,created_at) "
              "VALUES('default',0,0)")
    c.execute("INSERT INTO derived(id,partition,created_at,kind,epistemic_tag,"
              "producer,content,confidence) VALUES('der_x','default',1,"
              "'summary','inferred','t','kept across the rebuild',0.5)")
    c.execute("INSERT INTO period(id,partition,label,opened_at,summary_id) "
              "VALUES('per_x','default','p',0,'der_x')")
    c.commit()
    c.close()
    with Owl.open(path2) as m:
        assert m._s.migration.get("derived_kinds_added") == ["translation"]
        row = m._s.one("SELECT content FROM derived WHERE id='der_x'")
        assert row["content"] == "kept across the rebuild"
        per = m._s.one("SELECT summary_id FROM period WHERE id='per_x'")
        assert per["summary_id"] == "der_x"


def test_the_schema_and_the_migration_agree_on_the_kinds():
    import sqlite3
    from owl import migrations
    with _mind() as m:
        c = sqlite3.connect(m._s.path)
        assert tuple(migrations.derived_kinds(c)) == migrations.DERIVED_KINDS
        c.close()


def test_a_handover_carries_the_languages_and_the_translation_link():
    d = tempfile.mkdtemp()
    with Owl.open(os.path.join(d, "a.owl")) as a:
        so = a.observe(SOMALI, origin="tool_output", source_ref="rf:1",
                       lang="so")
        a.translation(so, ENGLISH, lang="en", producer="atk.translate")
        a.export_pack(os.path.join(d, "x.owlpack"), exporter="op-a")
    with Owl.open(os.path.join(d, "b.owl")) as b:
        b.graft(os.path.join(d, "x.owlpack"), as_source="op-a")
        tr = b._s.one("SELECT id FROM derived WHERE kind='translation'")["id"]
        orig = b.original_of(tr)
        assert orig.startswith("obs_") and b.language_of(orig) == "so"
        assert b.language_of(tr) == "en"


def test_tampered_language_tags_are_refused():
    import json
    from owl.handover import HandoverError
    d = tempfile.mkdtemp()
    pack = os.path.join(d, "x.owlpack")
    with Owl.open(os.path.join(d, "a.owl")) as a:
        a.observe(SOMALI, lang="so", source_ref="rf:1")
        a.export_pack(pack, exporter="op-a")
    body = json.loads(open(pack).read())
    body["langs"] = {k: "en" for k in body["langs"]}
    open(pack, "w").write(json.dumps(body))
    with Owl.open(os.path.join(d, "b.owl")) as b:
        with pytest.raises(HandoverError):
            b.graft(pack, as_source="op-a")


# ── unit words are English; figures are universal (2026-10-01) ───────────

def test_a_translation_out_of_english_keeps_its_figures_not_its_unit_words():
    """Found from ATK: English "4000 litres" into Somali "4000 litir" was
    refused as a stripped unit, because 'litir' is not an English unit
    word. Every translation out of English with a quantity in it was."""
    with _mind() as m:
        en = m.observe("The depot holds 4000 litres of diesel.", lang="en",
                       source_ref="sitrep")
        so = m.translation(en, "Bakhaarku wuxuu hayaa 4000 litir oo naafto.",
                           lang="so", producer="t")
        assert m.language_of(so) == "so"
        with pytest.raises(OwlError) as e:
            m.translation(en, "Bakhaarku wuxuu hayaa 400 litir oo naafto.",
                          lang="so", producer="t")
        assert "4000" in str(e.value) and "missing" in str(e.value)


def test_an_abbreviated_unit_is_still_checked_in_any_language():
    with _mind() as m:
        en = m.observe("Give 250 mg every six hours.", lang="en")
        with pytest.raises(OwlError):
            m.translation(en, "Bixi 250 g lix saacadood kasta.", lang="so",
                          producer="t")
        ok = m.translation(en, "Bixi 250 mg lix saacadood kasta.",
                           lang="so", producer="t")
        assert m.original_of(ok) == en


def test_arabic_digits_are_figures_too():
    with _mind() as m:
        en = m.observe("The depot holds 4000 litres.", lang="en")
        ar = m.translation(en, "يحتوي المستودع على ٤٠٠٠ لتر.", lang="ar",
                           producer="t")
        assert m.language_of(ar) == "ar"


def test_into_english_the_old_check_is_unchanged():
    with _mind() as m:
        fr = m.observe("Le dépôt contient 4000 litres.", lang="fr")
        with pytest.raises(OwlError) as e:
            m.translation(fr, "The depot holds 4000.", lang="en",
                          producer="t")
        assert "lost its unit" in str(e.value)


def test_a_figure_written_the_way_another_language_writes_it_is_the_figure():
    """Found in review: "4.000 Liter", "4 000" and "2,5" were refused as
    changed or missing figures in correct translations."""
    with _mind() as m:
        en = m.observe("The depot holds 4,000 litres; 2.5 litres per hour.",
                       lang="en")
        m.translation(en, "Das Depot hat 4.000 Liter; 2,5 Liter pro Stunde.",
                      lang="de", producer="t")
        m.translation(en, "Le dépôt contient 4 000 litres ; 2,5 litres par "
                          "heure.", lang="fr", producer="t")
        with pytest.raises(OwlError):
            m.translation(en, "Das Depot hat 400 Liter; 2,5 Liter pro "
                              "Stunde.", lang="de", producer="t")


def test_a_decimal_comma_is_never_read_as_a_thousands_comma():
    with _mind() as m:
        en = m.observe("Give 25 litres.", lang="en")
        with pytest.raises(OwlError):
            m.translation(en, "Gib 2,5 Liter.", lang="de", producer="t")
