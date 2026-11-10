from types import SimpleNamespace as NS

import pytest

from data_analyst_agent.llm import AnthropicLLM, LLMError, MockLLM, ToolResult
from data_analyst_agent.tools import Toolbox

SYSTEM = [{"type": "text", "text": "sys"}]
TOOLS = Toolbox.specs()


# -- MockLLM ----------------------------------------------------------------- #


def test_mock_replays_tool_calls_then_answer():
    llm = MockLLM([
        {"tool": "describe_data"},
        {"tool": "column_stats", "input": {"column": "revenue"}, "text": "Checking revenue."},
        "Mean revenue is 455.33.",
    ])
    r1 = llm.complete(SYSTEM, [llm.user_message("q")], TOOLS)
    assert [c.name for c in r1.tool_calls] == ["describe_data"] and r1.stop_reason == "tool_use"
    r2 = llm.complete(SYSTEM, [], TOOLS)
    assert r2.text == "Checking revenue." and r2.tool_calls[0].input == {"column": "revenue"}
    assert r2.raw_content[0] == {"type": "text", "text": "Checking revenue."}
    r3 = llm.complete(SYSTEM, [], TOOLS)
    assert r3.text == "Mean revenue is 455.33." and not r3.wants_tools
    assert llm.remaining == 0


def test_mock_unique_ids_and_parallel_calls():
    llm = MockLLM([[{"tool": "describe_data"}, {"tool": "column_stats", "input": {"column": "units"}}]])
    r = llm.complete(SYSTEM, [], TOOLS)
    assert len({c.id for c in r.tool_calls}) == 2
    assert [b["type"] for b in r.raw_content] == ["tool_use", "tool_use"]


def test_mock_callable_step_sees_history():
    def react(messages):
        last = messages[-1]["content"][0]
        return "recovered" if last.get("is_error") else "no error seen"

    llm = MockLLM([react])
    msg = llm.tool_results_message([ToolResult("toolu_1", "Unknown column 'x'", is_error=True)])
    assert llm.complete(SYSTEM, [msg], TOOLS).text == "recovered"


def test_mock_records_a_snapshot_of_each_request():
    llm = MockLLM(["a"])
    history = [llm.user_message("q")]
    llm.complete(SYSTEM, history, TOOLS)
    history.append({"role": "assistant", "content": "mutated later"})
    assert len(llm.calls[0]["messages"]) == 1


def test_mock_exhausted_raises():
    with pytest.raises(LLMError, match="exhausted"):
        MockLLM([]).complete(SYSTEM, [], TOOLS)


def test_mock_rejects_bad_step():
    with pytest.raises(LLMError, match="Invalid MockLLM step"):
        MockLLM([{"no_tool_key": 1}]).complete(SYSTEM, [], TOOLS)


# -- history builders --------------------------------------------------------- #


def test_tool_results_message_shape():
    llm = MockLLM([])
    msg = llm.tool_results_message([
        ToolResult("a", {"mean": 1.5}),
        ToolResult("b", "bad column", is_error=True),
    ])
    assert msg["role"] == "user"
    ok, err = msg["content"]
    assert ok == {"type": "tool_result", "tool_use_id": "a", "content": '{"mean": 1.5}'}
    assert err["is_error"] is True and err["content"] == "bad column"


# -- AnthropicLLM (fake client, no network) ----------------------------------- #


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.sent = None
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **params):
        self.sent = params
        return self.response


def _response(content, stop_reason="tool_use"):
    return NS(content=content, stop_reason=stop_reason, usage=NS(input_tokens=100, output_tokens=20))


def test_anthropic_request_params():
    client = FakeClient(_response([NS(type="text", text="hi")], "end_turn"))
    llm = AnthropicLLM(client=client)
    llm.complete(SYSTEM, [{"role": "user", "content": "q"}], TOOLS)
    sent = client.sent
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == SYSTEM and sent["tools"] == TOOLS
    assert sent["output_config"] == {"effort": "high"}
    assert sent["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert sent["fallbacks"] == "default" and sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in sent and "temperature" not in sent


def test_anthropic_without_fallbacks_uses_plain_endpoint():
    client = FakeClient(_response([NS(type="text", text="hi")], "end_turn"))
    AnthropicLLM(client=client, fallbacks=False).complete(SYSTEM, [], TOOLS)
    assert "betas" not in client.sent and "fallbacks" not in client.sent


def test_anthropic_normalises_thinking_text_and_tool_use():
    thinking = NS(type="thinking", thinking="", signature="sig")
    content = [
        thinking,
        NS(type="text", text="Let me look."),
        NS(type="tool_use", id="toolu_1", name="column_stats", input={"column": "revenue"}),
    ]
    r = AnthropicLLM(client=FakeClient(_response(content))).complete(SYSTEM, [], TOOLS)
    assert r.text == "Let me look."
    assert r.tool_calls[0].id == "toolu_1" and r.tool_calls[0].input == {"column": "revenue"}
    assert r.raw_content[0] is thinking  # passed back unchanged
    assert r.usage == {"input_tokens": 100, "output_tokens": 20}
    assert AnthropicLLM().assistant_message(r) == {"role": "assistant", "content": content}


def test_anthropic_refusal_drops_tool_calls():
    content = [NS(type="tool_use", id="t", name="describe_data", input={})]
    r = AnthropicLLM(client=FakeClient(_response(content, "refusal"))).complete(SYSTEM, [], TOOLS)
    assert r.stop_reason == "refusal" and r.tool_calls == []


def test_anthropic_sdk_errors_become_llm_errors():
    import anthropic

    class NoCredentials(FakeClient):
        def _create(self, **params):
            raise anthropic.AnthropicError("no credentials")

    with pytest.raises(LLMError, match="no credentials"):
        AnthropicLLM(client=NoCredentials(None)).complete(SYSTEM, [], TOOLS)


def test_anthropic_truncated_tool_call_raises():
    content = [NS(type="tool_use", id="t", name="plot", input={})]
    with pytest.raises(LLMError, match="max_tokens"):
        AnthropicLLM(client=FakeClient(_response(content, "max_tokens"))).complete(SYSTEM, [], TOOLS)
