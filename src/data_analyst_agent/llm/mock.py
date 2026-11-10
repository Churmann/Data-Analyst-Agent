"""Scripted LLM for offline tests: no network, no API key.

A script is a list of steps, one per model turn:

* ``"some text"``                          final answer (ends the run)
* ``{"tool": "name", "input": {...}}``     one tool call
* ``[{"tool": ...}, {"tool": ...}]``       several tool calls in one turn
* ``{"text": "...", "tool": ..., ...}``    tool call with leading text
* ``callable(messages) -> step``           decide based on history so far,
                                            e.g. to react to a tool error
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable

from .base import LLM, LLMError, LLMResponse, SystemPrompt, ToolCall

Step = str | dict[str, Any] | list[dict[str, Any]] | Callable[[list[dict[str, Any]]], Any]


class MockLLM(LLM):
    def __init__(self, script: Iterable[Step]) -> None:
        self.script: list[Step] = list(script)
        self.calls: list[dict[str, Any]] = []  # what the agent sent each turn
        self._next_id = 0

    @property
    def remaining(self) -> int:
        return len(self.script) - len(self.calls)

    def complete(
        self,
        system: SystemPrompt,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        turn = len(self.calls)
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if turn >= len(self.script):
            raise LLMError(f"MockLLM script exhausted after {len(self.script)} turn(s).")

        step = self.script[turn]
        if callable(step):
            step = step(messages)
        return self._to_response(step)

    def _to_response(self, step: Any) -> LLMResponse:
        if isinstance(step, str):
            return LLMResponse(
                text=step,
                raw_content=[{"type": "text", "text": step}],
                stop_reason="end_turn",
            )

        items = step if isinstance(step, list) else [step]
        if not items or not all(isinstance(i, dict) and "tool" in i for i in items):
            raise LLMError(f"Invalid MockLLM step: {step!r}")

        text = " ".join(i["text"] for i in items if i.get("text"))
        raw: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        calls = []
        for item in items:
            self._next_id += 1
            call = ToolCall(id=f"toolu_mock_{self._next_id:03d}", name=item["tool"], input=item.get("input", {}))
            calls.append(call)
            raw.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
        return LLMResponse(text=text, tool_calls=calls, raw_content=raw, stop_reason="tool_use")
