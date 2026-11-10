"""The eval harness end to end with MockLLM, plus checks that the eval set itself is sound."""

import json

import pytest

from data_analyst_agent.llm import MockLLM
from data_analyst_agent.tools import Toolbox
from evals.run_eval import format_scorecard, load_items, main
from evals.scoring import extract_numbers, score_item

from .test_sample_data import CSV


def test_eval_set_is_well_formed():
    items = load_items()
    kinds = {i["kind"] for i in items}
    assert kinds == {"answer", "refuse", "clarify"}
    for item in items:
        assert item["question"].strip()
        if item["kind"] == "answer":
            assert item.get("expect_numbers") or item.get("expect_substrings"), item["id"]


def test_load_items_filters_and_rejects_unknown_ids():
    assert [i["id"] for i in load_items(ids=["forecast", "row_count"])] == ["row_count", "forecast"]
    with pytest.raises(ValueError, match="Unknown eval id"):
        load_items(ids=["nope"])


# Tool calls that produce every expected figure, checked against the real data.
# If an expected value in the YAML is wrong, the matching case here fails.
GROUND_TRUTH_CALLS = {
    "row_count": ("describe_data", {}),
    "missing_revenue": ("column_stats", {"column": "revenue"}),
    "overall_mean_revenue": ("column_stats", {"column": "revenue"}),
    "top_region_by_mean_revenue": ("aggregate", {"group_by": ["region"], "value": "revenue", "agg": "mean"}),
    "corporate_mean_revenue": ("aggregate", {"group_by": ["segment"], "value": "revenue", "agg": "mean"}),
    "mean_units_by_segment": ("aggregate", {"group_by": ["segment"], "value": "units", "agg": "mean"}),
    "total_units": ("column_stats", {"column": "units"}),
    "total_recorded_revenue": ("column_stats", {"column": "revenue"}),
    "median_revenue": ("column_stats", {"column": "revenue"}),
    "units_revenue_correlation": ("correlation", {"x": "units", "y": "revenue"}),
    "north_vs_south_welch": ("hypothesis_test", {"value": "revenue", "group_col": "region", "group_a": "North",
                                                 "group_b": "South", "test": "welch_t"}),
    "south_orders": ("aggregate", {"group_by": ["region"], "value": "units", "agg": "count"}),
}


def test_every_answer_item_has_a_ground_truth_call():
    answer_ids = {i["id"] for i in load_items() if i["kind"] == "answer"}
    assert answer_ids == set(GROUND_TRUTH_CALLS)


@pytest.mark.parametrize("item", [i for i in load_items() if i["kind"] == "answer"], ids=lambda i: i["id"])
def test_expected_values_are_correct(item, tmp_path):
    tool, args = GROUND_TRUTH_CALLS[item["id"]]
    result = Toolbox.from_csv(CSV, output_dir=tmp_path).run(tool, args)
    numbers = extract_numbers(json.dumps(result))
    for exp in item.get("expect_numbers", []):
        assert any(abs(n - exp["value"]) <= exp.get("tolerance", 0) for n in numbers), (exp, result)


def test_perfect_mock_agent_passes_every_answer_item(tmp_path):
    """A scripted agent that makes the right call and quotes the tool exactly scores 100%."""
    items = [i for i in load_items() if i["kind"] == "answer"]

    def factory(item):
        tool, args = GROUND_TRUTH_CALLS[item["id"]]

        def answer(messages):
            payload = json.loads(messages[-1]["content"][0]["content"])
            # Quote every figure the tool returned, plus the expected labels.
            figures = ", ".join(f"{n:g}" if abs(n) >= 1e-3 or n == 0 else f"{n:.3e}"
                                for n in extract_numbers(json.dumps(payload)))
            return f"{' '.join(item.get('expect_substrings', []))} {figures}".strip()

        return MockLLM([{"tool": tool, "input": args}, answer])

    code = main(["--ids", *[i["id"] for i in items], "--output-dir", str(tmp_path)], llm_factory=factory)
    assert code == 0


def test_harness_scores_refusal_clarify_and_failures(tmp_path, capsys):
    scripts = {
        "profit_margin": ["There is no cost column, so profit margin cannot be calculated."],
        "best_region": ["Best by which measure: mean revenue, total revenue or units?"],
        # wrong + ungrounded, and still wrong after the runtime correction
        "row_count": [{"tool": "describe_data"}, "There are 650 orders.", "There are 640 orders."],
        "total_units": [{"tool": "describe_data"}] * 5,  # never answers
    }
    out_json = tmp_path / "eval.json"
    code = main(
        ["--ids", *scripts, "--max-steps", "2", "--output-dir", str(tmp_path), "--json-out", str(out_json)],
        llm_factory=lambda item: MockLLM(scripts[item["id"]]),
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "passed 2/4 (50%)" in out
    assert "stopped: ungrounded" in out and "ungrounded numbers: ['640']" in out
    assert "stopped: step_budget" in out

    data = json.loads(out_json.read_text())
    assert data["summary"]["by_kind"]["refuse"] == {"total": 1, "passed": 1}
    row = next(r for r in data["items"] if r["item"]["id"] == "row_count")
    assert row["result"]["steps"][0]["tool"] == "describe_data" and not row["score"]["passed"]


def test_llm_error_on_one_item_does_not_stop_the_run(tmp_path, capsys):
    code = main(["--ids", "forecast", "--output-dir", str(tmp_path)], llm_factory=lambda item: MockLLM([]))
    assert code == 1
    assert "stopped: error: MockLLM script exhausted" in capsys.readouterr().out


def test_format_scorecard_lists_every_item():
    items = load_items(ids=["forecast", "best_region"])
    scores = [score_item(i, {"answer": "I can't forecast; dates are not available.", "stopped_reason": "answered",
                             "steps": []}) for i in items]
    card = format_scorecard(scores)
    assert "forecast" in card and "best_region" in card and "passed 1/2" in card
