import pytest

from evals.scoring import (
    ItemScore,
    collect_numbers,
    extract_number_spans,
    extract_numbers,
    is_clarification,
    is_refusal,
    numeric_match,
    score_item,
    summarise,
    ungrounded_numbers,
)

# -- extract_numbers ----------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Mean revenue is 562.67 (n=144).", [562.67, 144]),
        ("Total: $262,272.00 across 7,900 units", [262272.0, 7900]),
        ("t = -3.87, p = 1.3e-4", [-3.87, 1.3e-4]),
        ("A share of 12.5% and .25 of orders", [12.5, 0.25]),
        ("Order ORD-0042 in region2", [42]),  # digits glued to letters are not numbers
        ("No figures here.", []),
        ("", []),
        ("Ends with a number 600.", [600]),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_extract_number_spans_metadata():
    spans = extract_number_spans("p < 0.001, mean 562.7, share 12%")
    assert [(s.value, s.decimals, s.threshold, s.percent) for s in spans] == [
        (0.001, 3, True, False),
        (562.7, 1, False, False),
        (12.0, 0, False, True),
    ]
    assert extract_number_spans("1.2e-5")[0].decimals == 6


# -- numeric_match --------------------------------------------------------------- #


def test_numeric_match():
    assert numeric_match(562.67, [100, 562.7], tolerance=0.5)
    assert not numeric_match(562.67, [561.0], tolerance=0.5)
    assert numeric_match(600, [600.0])
    assert not numeric_match(600, [601])
    assert not numeric_match(1.0, [])


# -- refusal / clarification ---------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "I can't answer that: the dataset has no cost column, so profit margin cannot be computed.",
        "The data does not contain dates, so a monthly trend is not possible.",
        "That is outside the scope of this dataset.",
        "I'm unable to forecast revenue from this data.",
        "There is no customer column in this dataset.",
    ],
)
def test_is_refusal_true(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", ["East has the highest mean revenue at 562.67.", "", None])
def test_is_refusal_false(text):
    assert not is_refusal(text)


def test_is_clarification():
    assert is_clarification("By what measure do you mean best: mean revenue, total revenue or units?")
    assert is_clarification("Could you clarify which metric you care about?")
    assert not is_clarification("Could you clarify. I will assume revenue.")  # no question asked
    assert not is_clarification("East is best by mean revenue (562.67).")
    assert not is_clarification(None)


# -- grounding ------------------------------------------------------------------ #


def test_collect_numbers_walks_nested_results_and_skips_bools():
    result = {"rows": 600, "ok": True, "groups": [{"mean": 1.5, "group": {"region": "East"}}], "p": None}
    assert sorted(collect_numbers(result)) == [1.5, 600.0]


TOOL_RESULTS = [
    {"groups": [{"group": {"region": "East"}, "mean": 562.6666666, "n": 144}]},
    {"r": 0.9419526, "p_value": 3.0e-274, "n": 576},
    {"share": 0.125},
]


@pytest.mark.parametrize(
    "answer",
    [
        "East: 562.67 (n=144).",
        "East averages about 563 per order.",
        "East averages 562.7.",
        "r = 0.94 (n = 576), p < 0.001.",
        "12.5% of orders.",
    ],
)
def test_grounded_answers(answer):
    assert ungrounded_numbers(answer, TOOL_RESULTS) == []


@pytest.mark.parametrize(
    "answer,bad",
    [
        ("East averages 600.", ["600"]),
        ("East averages 562.6.", ["562.6"]),  # wrong rounding
        ("East is 20% above South.", ["20%"]),  # derived figure not from a tool
        ("r = 0.95.", ["0.95"]),
    ],
)
def test_ungrounded_answers(answer, bad):
    assert ungrounded_numbers(answer, TOOL_RESULTS) == bad


def test_numbers_from_question_are_allowed():
    assert ungrounded_numbers("Top 3 regions are...", [], question="What are the top 3 regions?") == []


# -- score_item ----------------------------------------------------------------- #


def run(answer, *, steps=None, stopped="answered", question="q"):
    return {"question": question, "answer": answer, "stopped_reason": stopped, "steps": steps or []}


AGG_STEP = {"tool": "aggregate", "ok": True, "result": TOOL_RESULTS[0]}
ANSWER_ITEM = {
    "id": "top_region",
    "kind": "answer",
    "question": "Which region has the highest mean revenue?",
    "expect_substrings": ["East"],
    "expect_numbers": [{"value": 562.67, "tolerance": 0.5}],
}


def test_score_answer_pass():
    s = score_item(ANSWER_ITEM, run("East, at 562.67 per order (n=144).", steps=[AGG_STEP]))
    assert s.passed and s.reasons == [] and s.steps == 1


def test_score_answer_wrong_number_and_text():
    s = score_item(ANSWER_ITEM, run("West, at 415.83.", steps=[AGG_STEP]))
    assert not s.passed
    assert "missing number 562.67 (+/-0.5)" in s.reasons and "missing text 'East'" in s.reasons
    assert s.ungrounded == ["415.83"]


def test_score_answer_right_number_but_invented():
    # Correct figure, but no tool ever produced it: still a fail.
    s = score_item(ANSWER_ITEM, run("East, at 562.67 per order."))
    assert not s.passed and s.ungrounded == ["562.67"]


def test_score_answer_not_finished():
    s = score_item(ANSWER_ITEM, run(None, stopped="step_budget"))
    assert not s.passed and "stopped: step_budget" in s.reasons


def test_score_counts_tool_errors():
    failed = {"tool": "aggregate", "ok": False, "error": {"message": "x"}}
    s = score_item(ANSWER_ITEM, run("East, 562.67.", steps=[failed, AGG_STEP]))
    assert s.passed and s.tool_errors == 1


def test_score_refuse_and_clarify():
    refuse = {"id": "profit", "kind": "refuse", "question": "Profit margin?"}
    assert score_item(refuse, run("There is no cost column, so margin cannot be calculated.")).passed
    s = score_item(refuse, run("Profit margin is 23%."))
    assert not s.passed and s.reasons == ["did not decline"]

    clarify = {"id": "best", "kind": "clarify", "question": "Which region is best?"}
    assert score_item(clarify, run("Best by which metric: mean revenue or total units?")).passed
    assert not score_item(clarify, run("East.")).passed


def test_score_unknown_kind():
    assert score_item({"id": "x", "kind": "vibes"}, run("hi")).reasons == ["unknown item kind 'vibes'"]


def test_summarise():
    scores = [
        ItemScore("a", "answer", True, steps=2),
        ItemScore("b", "answer", False, ["x"], ungrounded=["5"], steps=4, tool_errors=1),
        ItemScore("c", "refuse", True),
    ]
    s = summarise(scores)
    assert s["passed"] == 2 and s["total"] == 3 and s["pass_rate"] == pytest.approx(2 / 3)
    assert s["by_kind"] == {"answer": {"total": 2, "passed": 1}, "refuse": {"total": 1, "passed": 1}}
    assert s["ungrounded_items"] == 1 and s["mean_steps"] == 2 and s["tool_errors"] == 1
    assert summarise([])["pass_rate"] == 0.0
