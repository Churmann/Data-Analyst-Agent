"""The agent loop: plan, act, observe, repeat.

Each turn the model sees the system prompt, the conversation so far and the
tool schemas. If it asks for tools, they run and their results (or structured
errors) go back as tool_result messages; if it answers in plain text, the run
ends. A step budget caps the number of tool executions so every run
terminates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from ..llm import LLM, LLMResponse, ToolCall, ToolResult
from ..prompts import build_system_prompt
from ..retrieval import ColumnNote, retrieve_columns
from ..tools import Toolbox, ToolError

StoppedReason = Literal["answered", "step_budget", "refused"]


@dataclass
class Step:
    """One tool execution."""

    index: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    thought: str = ""  # text the model wrote in the same turn, before calling the tool
    call_id: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str | None
    stopped_reason: StoppedReason
    steps: list[Step] = field(default_factory=list)
    turns: int = 0
    retrieved_columns: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tool_errors(self) -> list[Step]:
        return [s for s in self.steps if not s.ok]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class Agent:
    """Answers one question at a time over a Toolbox, using any LLM adapter.

    Args:
        llm: the model adapter (AnthropicLLM in production, MockLLM in tests).
        toolbox: the analysis tools, bound to the dataset.
        dictionary: the parsed data dictionary, used for retrieval and the prompt.
        max_steps: maximum number of tool executions per question.
        retrieve_k: how many column notes to put in the system prompt.
        on_step: called after every tool execution (the CLI uses it for --verbose).
    """

    def __init__(
        self,
        llm: LLM,
        toolbox: Toolbox,
        dictionary: dict[str, Any],
        *,
        max_steps: int = 12,
        retrieve_k: int = 3,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.toolbox = toolbox
        self.dictionary = dictionary
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k
        self.on_step = on_step

    def run(self, question: str) -> AgentResult:
        """Run the loop for one question and return the answer with its full trace.

        Raises LLMError if the model call itself fails; tool failures never raise.
        """
        # Ground the prompt in the column notes that match this question.
        notes: list[ColumnNote] = retrieve_columns(question, self.dictionary, k=self.retrieve_k)
        system = build_system_prompt(self.dictionary, notes)
        tools = self.toolbox.specs()

        # Working memory: the full message history, resent every turn. It only
        # ever grows (assistant turns are appended exactly as returned), so the
        # model always sees every call it made and every result it got.
        messages: list[dict[str, Any]] = [self.llm.user_message(question)]
        result = AgentResult(
            question=question,
            answer=None,
            stopped_reason="step_budget",
            retrieved_columns=[n.name for n in notes],
        )

        while True:
            # Plan: the model reads everything so far and decides what to do.
            response = self.llm.complete(system, messages, tools)
            result.turns += 1
            _add_usage(result.usage, response)
            messages.append(self.llm.assistant_message(response))

            if response.stop_reason == "refusal":
                result.answer = response.text or None
                result.stopped_reason = "refused"
                return result

            if not response.wants_tools:
                result.answer = response.text
                result.stopped_reason = "answered"
                return result

            # After the budget is used up the model still gets one more turn to
            # answer from what it has; if it asks for tools again, stop here.
            remaining = self.max_steps - len(result.steps)
            if remaining <= 0:
                return result

            # Act + observe. Every tool_use in the turn gets a tool_result; calls
            # past the budget are answered with an error rather than run.
            tool_results: list[ToolResult] = []
            for i, call in enumerate(response.tool_calls):
                if i < remaining:
                    step = self._execute(call, len(result.steps) + 1, response.text)
                    result.steps.append(step)
                    if self.on_step:
                        self.on_step(step)
                    payload = step.result if step.ok else step.error
                    tool_results.append(ToolResult(call.id, payload, is_error=not step.ok))
                else:
                    skipped = {"error_type": "step_budget", "message": "Not run: the step budget is spent."}
                    tool_results.append(ToolResult(call.id, skipped, is_error=True))
            messages.append(self.llm.tool_results_message(tool_results))

    def _execute(self, call: ToolCall, index: int, thought: str) -> Step:
        """Run one tool call and record it. Failures become structured errors, never exceptions."""
        step = Step(index=index, tool=call.name, arguments=dict(call.input), ok=False, thought=thought, call_id=call.id)
        try:
            step.result = self.toolbox.run(call.name, call.input)
            step.ok = True
        except ToolError as e:
            step.error = {"error_type": "invalid_input", "message": str(e)}
        except Exception as e:  # a bug in a tool must not kill the run
            step.error = {"error_type": "internal", "message": f"{type(e).__name__}: {e}"}
        return step


def _add_usage(total: dict[str, int], response: LLMResponse) -> None:
    """Accumulate token counts across turns."""
    for key, value in response.usage.items():
        total[key] = total.get(key, 0) + int(value)
