from pathlib import Path

import pytest

from data_analyst_agent.prompts import RULES, build_system_prompt, system_prompt_text
from data_analyst_agent.retrieval import load_data_dictionary, retrieve_columns, tokens

DICTIONARY = Path(__file__).resolve().parent.parent / "data" / "data_dictionary.yaml"


@pytest.fixture(scope="module")
def dictionary():
    return load_data_dictionary(DICTIONARY)


def names(notes):
    return [n.name for n in notes]


# -- retrieval --------------------------------------------------------------- #


def test_tokens_stems_and_drops_stopwords():
    assert tokens("What are the Regions' unit_prices?") == {"region", "unit", "price"}


@pytest.mark.parametrize(
    "question,expected_first",
    [
        ("Average revenue in each region?", {"region", "revenue"}),
        ("Is North different from South?", {"region"}),
        ("Do Corporate customers order more?", {"segment"}),
        ("What is the price per unit?", {"unit_price"}),
        ("Which orders are missing revenue?", {"revenue"}),
    ],
)
def test_retrieve_ranks_relevant_column_first(dictionary, question, expected_first):
    notes = retrieve_columns(question, dictionary)
    top = notes[0].score
    assert {n.name for n in notes if n.score == top} == expected_first


def test_retrieve_respects_k_and_ignores_unrelated(dictionary):
    assert len(retrieve_columns("units price revenue region segment order", dictionary, k=2)) == 2
    assert retrieve_columns("What's the weather in Paris?", dictionary) == []
    assert retrieve_columns("", dictionary) == []


def test_note_render_includes_derivation(dictionary):
    note = retrieve_columns("missing revenue", dictionary)[0]
    text = note.render()
    assert "revenue = units * unit_price" in text and "may contain missing values" in text


def test_load_rejects_non_dictionary(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("just: a mapping\n")
    with pytest.raises(ValueError, match="columns"):
        load_data_dictionary(bad)


# -- prompts ----------------------------------------------------------------- #


def test_system_prompt_has_rules_schema_and_notes(dictionary):
    notes = retrieve_columns("missing revenue by region", dictionary)
    blocks = build_system_prompt(dictionary, notes)
    assert len(blocks) == 2 and all(b["type"] == "text" for b in blocks)
    stable, per_question = blocks[0]["text"], blocks[1]["text"]

    assert stable.startswith(RULES)
    for col in dictionary["columns"]:
        assert f"- {col} [" in stable
    assert "Dataset: sales (600 rows)" in stable
    assert "- revenue (float)" in per_question and "derivation" in per_question


def test_stable_block_does_not_depend_on_question(dictionary):
    a = build_system_prompt(dictionary, retrieve_columns("revenue by region", dictionary))
    b = build_system_prompt(dictionary, retrieve_columns("units by segment", dictionary))
    assert a[0] == b[0] and a[1] != b[1]


def test_no_notes_message(dictionary):
    text = system_prompt_text(build_system_prompt(dictionary, []))
    assert "none matched" in text


@pytest.mark.parametrize(
    "phrase",
    [
        "Call describe_data first",
        "one step at a time",
        "If a tool returns an error",
        "must appear in a tool result",
        "clarifying question",
        "cannot be answered from this dataset",
    ],
)
def test_rules_cover_required_behaviours(phrase):
    assert phrase in RULES
