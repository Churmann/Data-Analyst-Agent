import json

import pytest

from data_analyst_agent.agent import Agent
from data_analyst_agent.cli import main
from data_analyst_agent.llm import LLMResponse, MockLLM
from data_analyst_agent.retrieval import load_data_dictionary
from data_analyst_agent.tools import Toolbox

from .test_retrieval_and_prompts import DICTIONARY
from .test_sample_data import CSV


@pytest.fixture
def toolbox(tmp_path):
    return Toolbox.from_csv(CSV, output_dir=tmp_path)


@pytest.fixture(scope="module")
def dictionary():
    return load_data_dictionary(DICTIONARY)


def make_agent(script, toolbox, dictionary, **kwargs):
    llm = MockLLM(script)
    return Agent(llm, toolbox, dictionary, **kwargs), llm


def tool_results(message):
    return [b for b in message["content"] if b["type"] == "tool_result"]


def test_happy_path_threads_history(toolbox, dictionary):
    agent, llm = make_agent(
        [
            {"tool": "describe_data", "text": "First, check the data."},
            {"tool": "aggregate", "input": {"group_by": ["region"], "value": "revenue", "agg": "mean"}},
            "East has the highest mean revenue (562.67, n=144).",
        ],
        toolbox,
        dictionary,
    )
    result = agent.run("Which region has the highest mean revenue?")

    assert result.stopped_reason == "answered"
    assert result.answer.startswith("East")
    assert result.turns == 3 and [s.tool for s in result.steps] == ["describe_data", "aggregate"]
    assert all(s.ok for s in result.steps)
    assert result.steps[0].thought == "First, check the data."
    east = next(g for g in result.steps[1].result["groups"] if g["group"]["region"] == "East")
    assert east["mean"] == pytest.approx(1688 / 3)
    assert {"region", "revenue"} <= set(result.retrieved_columns)

    # Final request: user, (assistant, tool_result) x 2, in that order.
    final = llm.calls[-1]["messages"]
    assert [m["role"] for m in final] == ["user", "assistant", "user", "assistant", "user"]
    for assistant, results in ((final[1], final[2]), (final[3], final[4])):
        use_ids = [b["id"] for b in assistant["content"] if b["type"] == "tool_use"]
        assert [r["tool_use_id"] for r in tool_results(results)] == use_ids
    payload = json.loads(tool_results(final[4])[0]["content"])
    assert payload["agg"] == "mean"

    # System prompt and tools were sent every turn.
    assert all(c["tools"] == Toolbox.specs() for c in llm.calls)
    assert "Column notes most relevant" in llm.calls[0]["system"][1]["text"]


def test_tool_error_is_fed_back_and_agent_recovers(toolbox, dictionary):
    def fix_after_error(messages):
        result = tool_results(messages[-1])[0]
        assert result["is_error"] is True
        error = json.loads(result["content"])
        assert error["error_type"] == "invalid_input" and "Unknown column 'sales'" in error["message"]
        return {"tool": "column_stats", "input": {"column": "revenue"}}

    agent, _ = make_agent(
        [{"tool": "column_stats", "input": {"column": "sales"}}, fix_after_error, "Mean revenue is 455.33."],
        toolbox,
        dictionary,
    )
    result = agent.run("What is average sales?")
    assert result.stopped_reason == "answered"
    assert [s.ok for s in result.steps] == [False, True]
    assert result.steps[0].error["error_type"] == "invalid_input"
    assert result.steps[0].result is None
    assert len(result.tool_errors) == 1


def test_unknown_tool_is_recoverable(toolbox, dictionary):
    agent, _ = make_agent([{"tool": "run_sql", "input": {"q": "x"}}, "I can't run SQL."], toolbox, dictionary)
    result = agent.run("q")
    assert result.stopped_reason == "answered"
    assert "Unknown tool 'run_sql'" in result.steps[0].error["message"]


def test_internal_tool_failure_does_not_crash(toolbox, dictionary, monkeypatch):
    def boom(**_):
        raise RuntimeError("disk full")

    monkeypatch.setitem(toolbox._tools, "describe_data", boom)
    agent, _ = make_agent([{"tool": "describe_data"}, "Could not inspect the data."], toolbox, dictionary)
    result = agent.run("q")
    assert result.steps[0].error == {"error_type": "internal", "message": "RuntimeError: disk full"}
    assert result.stopped_reason == "answered"


def test_step_budget_stops_a_model_that_never_answers(toolbox, dictionary):
    agent, llm = make_agent([{"tool": "describe_data"}] * 10, toolbox, dictionary, max_steps=3)
    result = agent.run("q")
    assert result.stopped_reason == "step_budget"
    assert result.answer is None
    assert len(result.steps) == 3
    assert result.turns == 4  # 3 tool turns + 1 last chance to answer


def test_model_can_answer_on_the_turn_after_budget(toolbox, dictionary):
    agent, _ = make_agent([{"tool": "describe_data"}, {"tool": "describe_data"}, "600 rows."],
                          toolbox, dictionary, max_steps=2)
    result = agent.run("How many rows?")
    assert result.stopped_reason == "answered" and result.answer == "600 rows."


def test_parallel_calls_beyond_budget_get_error_results(toolbox, dictionary):
    batch = [{"tool": "describe_data"}, {"tool": "column_stats", "input": {"column": "units"}},
             {"tool": "column_stats", "input": {"column": "revenue"}}]
    agent, llm = make_agent([batch, "done"], toolbox, dictionary, max_steps=2)
    result = agent.run("q")
    assert len(result.steps) == 2
    results = tool_results(llm.calls[1]["messages"][-1])
    assert len(results) == 3  # every tool_use answered
    assert json.loads(results[2]["content"])["error_type"] == "step_budget"
    assert result.answer == "done"


def test_on_step_callback_and_json(toolbox, dictionary):
    seen = []
    agent, _ = make_agent([{"tool": "describe_data"}, "ok"], toolbox, dictionary, on_step=seen.append)
    result = agent.run("q")
    assert [s.tool for s in seen] == ["describe_data"]
    data = json.loads(result.to_json())
    assert data["stopped_reason"] == "answered" and data["steps"][0]["result"]["rows"] == 600


# -- runtime grounding check ---------------------------------------------------- #


def last_user_text(messages):
    content = messages[-1]["content"]
    assert messages[-1]["role"] == "user" and isinstance(content, str)
    return content


def test_ungrounded_answer_is_corrected(toolbox, dictionary):
    seen_correction = []

    def corrected(messages):
        seen_correction.append(last_user_text(messages))
        return "There are 600 orders."

    agent, llm = make_agent(
        [{"tool": "describe_data"}, "There are 650 orders.", corrected], toolbox, dictionary
    )
    result = agent.run("How many orders are there?")

    assert result.answer == "There are 600 orders."
    assert result.stopped_reason == "answered"
    assert result.repairs == 1 and result.ungrounded == []
    assert result.turns == 3 and len(result.steps) == 1  # the correction did not use a tool step
    # The model was told exactly which figure had no support.
    assert "650" in seen_correction[0]
    # The rejected answer stays in the history, followed by the correction.
    history = llm.calls[-1]["messages"]
    assert history[-2]["content"] == [{"type": "text", "text": "There are 650 orders."}]


def test_still_ungrounded_after_correction(toolbox, dictionary):
    agent, llm = make_agent(
        [{"tool": "describe_data"}, "There are 650 orders.", "There are 640 orders."], toolbox, dictionary
    )
    result = agent.run("How many orders are there?")

    assert result.stopped_reason == "ungrounded"
    assert result.answer == "There are 640 orders."  # kept so the caller can see what was rejected
    assert result.ungrounded == ["640"] and result.repairs == 1
    assert llm.remaining == 0  # no extra turn after the last allowed correction


def test_grounded_first_answer_needs_no_correction(toolbox, dictionary):
    agent, llm = make_agent([{"tool": "describe_data"}, "There are 600 orders."], toolbox, dictionary)
    result = agent.run("How many orders are there?")
    assert result.stopped_reason == "answered"
    assert result.turns == 2 and result.repairs == 0 and result.ungrounded == []


def test_numbers_from_the_question_are_allowed(toolbox, dictionary):
    agent, _ = make_agent(
        [{"tool": "describe_data"}, "Yes: 600 orders, which is more than 500."], toolbox, dictionary
    )
    result = agent.run("Are there more than 500 orders?")
    assert result.stopped_reason == "answered" and result.repairs == 0


def test_failed_steps_do_not_ground_figures(toolbox, dictionary):
    # The bad call's error text is not a source; only successful results count.
    agent, _ = make_agent(
        [{"tool": "column_stats", "input": {"column": "sales"}}, "Mean sales are 455.33."],
        toolbox, dictionary, max_repairs=0,
    )
    result = agent.run("Average sales?")
    assert result.stopped_reason == "ungrounded" and result.ungrounded == ["455.33"]


def test_max_repairs_zero_reports_immediately(toolbox, dictionary):
    agent, llm = make_agent([{"tool": "describe_data"}, "There are 650 orders."], toolbox, dictionary, max_repairs=0)
    result = agent.run("How many orders?")
    assert result.stopped_reason == "ungrounded" and result.repairs == 0 and result.turns == 2


def test_turns_are_bounded_by_steps_plus_repairs(toolbox, dictionary):
    # Worst case: uses every tool step, then keeps answering with invented figures.
    script = [{"tool": "describe_data"}] * 3 + ["650 orders."] * 10
    agent, _ = make_agent(script, toolbox, dictionary, max_steps=3, max_repairs=2)
    result = agent.run("How many orders?")
    assert result.stopped_reason == "ungrounded"
    assert result.turns == 3 + 2 + 1  # max_steps + max_repairs + final answer


def test_no_stale_ungrounded_figures_when_stopped_by_budget(toolbox, dictionary):
    # Rejected answer, then the model asks for a tool with the budget spent:
    # no answer is returned, so no figures should be reported against it.
    script = [{"tool": "describe_data"}, "There are 650 orders.", {"tool": "describe_data"}]
    agent, _ = make_agent(script, toolbox, dictionary, max_steps=1)
    result = agent.run("How many orders?")
    assert result.stopped_reason == "step_budget" and result.answer is None
    assert result.repairs == 1 and result.ungrounded == []


def test_max_repairs_must_not_be_negative(toolbox, dictionary):
    with pytest.raises(ValueError):
        Agent(MockLLM([]), toolbox, dictionary, max_repairs=-1)


class RefusingLLM(MockLLM):
    def complete(self, system, messages, tools):
        return LLMResponse(text="", raw_content=[], stop_reason="refusal")


def test_refusal_stops_without_answer(toolbox, dictionary):
    result = Agent(RefusingLLM([]), toolbox, dictionary).run("q")
    assert result.stopped_reason == "refused" and result.answer is None and result.steps == []


def test_max_steps_must_be_positive(toolbox, dictionary):
    with pytest.raises(ValueError):
        Agent(MockLLM([]), toolbox, dictionary, max_steps=0)


# -- CLI ---------------------------------------------------------------------- #


def test_cli_answers_and_prints_steps(tmp_path, capsys):
    llm = MockLLM([{"tool": "describe_data"}, "The dataset has 600 rows."])
    code = main(["--question", "How many rows?", "--verbose", "--output-dir", str(tmp_path)], llm=llm)
    out, err = capsys.readouterr()
    assert code == 0
    assert out.strip() == "The dataset has 600 rows."
    assert "[1] describe_data({}) -> ok" in err and "stopped: answered" in err


def test_cli_json_and_budget_exit_code(tmp_path, capsys):
    llm = MockLLM([{"tool": "describe_data"}] * 3)
    code = main(["-q", "x", "--max-steps", "1", "--json", "--output-dir", str(tmp_path)], llm=llm)
    out, _ = capsys.readouterr()
    assert code == 2 and json.loads(out)["stopped_reason"] == "step_budget"


def test_cli_warns_on_ungrounded_answer(tmp_path, capsys):
    llm = MockLLM([{"tool": "describe_data"}, "There are 650 orders.", "There are 640 orders."])
    code = main(["-q", "How many orders?", "--output-dir", str(tmp_path)], llm=llm)
    out = capsys.readouterr().out
    assert code == 2
    assert "There are 640 orders." in out and "do not match any tool result: 640" in out


def test_cli_missing_csv(tmp_path, capsys):
    code = main(["-q", "x", "--csv", str(tmp_path / "nope.csv")], llm=MockLLM([]))
    assert code == 1 and "error:" in capsys.readouterr().err
