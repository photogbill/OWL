"""Embedding models do not share conventions, and getting it wrong is silent."""
from pathlib import Path

import pytest

from owl.adapters.gguf_meta import GgufMeta, conventions, read

MODELS = Path(__file__).resolve().parents[1] / "embedding model"


def test_causal_models_are_recognised():
    assert not GgufMeta(architecture="bert").is_causal
    assert GgufMeta(architecture="qwen3").is_causal
    assert GgufMeta(architecture="llama").is_causal


def test_a_causal_model_never_falls_back_to_cls_pooling():
    """Position 0 of a causal model has attended to nothing. Taking it as the
    sentence embedding is meaningless."""
    c = conventions(GgufMeta(architecture="qwen3", pooling="unknown"))
    assert c["pooling"] == "last"
    c = conventions(GgufMeta(architecture="bert", pooling="unknown"))
    assert c["pooling"] == "cls"


def test_declared_pooling_wins_over_the_guess():
    assert conventions(GgufMeta(architecture="bert",
                                pooling="mean"))["pooling"] == "mean"


def test_query_instruction_is_per_family():
    """Qwen3-Embedding requires one; bge-m3 explicitly does not. Applying the
    wrong convention costs accuracy silently."""
    qwen = conventions(GgufMeta(architecture="qwen3",
                                name="Qwen3 Embedding 8B"))
    assert qwen["query_prefix"].startswith("Instruct:")

    m3 = conventions(GgufMeta(architecture="bert", name="bge-m3-Q6_K"))
    assert m3["query_prefix"] == ""

    en = conventions(GgufMeta(architecture="bert", name="bge-large-en-v1.5"))
    assert "Represent this sentence" in en["query_prefix"]

    e5 = conventions(GgufMeta(architecture="bert", name="multilingual-e5-large"))
    assert e5["query_prefix"] == "query: " and e5["doc_prefix"] == "passage: "


def test_instructions_are_asymmetric():
    """An instruction on BOTH sides defeats its purpose."""
    for name, arch in (("Qwen3 Embedding 8B", "qwen3"),
                       ("bge-large-en-v1.5", "bert")):
        c = conventions(GgufMeta(architecture=arch, name=name))
        assert c["query_prefix"] and not c["doc_prefix"], name


def test_unknown_family_applies_nothing():
    c = conventions(GgufMeta(architecture="mystery", name="something-new"))
    assert c["query_prefix"] == ""
    assert "unknown family" in c["reason"]


def test_a_non_gguf_file_does_not_explode(tmp_path):
    f = tmp_path / "not.gguf"
    f.write_bytes(b"this is not a gguf file at all")
    assert read(f).architecture == "unknown"


@pytest.mark.skipif(not MODELS.exists(), reason="no local models")
def test_reads_the_real_models():
    for path in MODELS.glob("*.gguf"):
        m = read(path)
        assert m.architecture != "unknown", path.name
        assert m.dim > 0 and m.context > 0
        c = conventions(m)
        if m.is_causal:
            assert c["pooling"] != "cls", (
                f"{path.name}: CLS pooling on a causal model")
