"""Provider-agnostic types for talking to an LLM with tool use.

The agent loop only sees ``LLMResponse`` and the ``LLM`` interface. Each
provider adapter is responsible for turning its own wire format into these
types, and for building the history messages it expects (``user_message``,
``tool_results_message``), so the loop never constructs provider-specific
payloads itself.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

SystemPrompt = str | list[dict[str, Any]]


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked to run."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a ToolCall, to send back to the model."""

    call_id: str
    content: Any
    is_error: bool = False

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, allow_nan=False)


@dataclass
class LLMResponse:
    """One normalised model turn.

    ``raw_content`` is the assistant message content exactly as the provider
    returned it. Append it to the history unchanged (via
    ``assistant_message``) so provider-specific blocks such as thinking
    blocks survive round trips.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(Exception):
    """The model call failed or returned something the agent cannot use."""


class LLM(ABC):
    """Interface every provider adapter implements."""

    @abstractmethod
    def complete(
        self,
        system: SystemPrompt,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Run one model turn over the given history."""

    # History builders. The defaults use the Anthropic Messages shape, which
    # the mock also uses; other providers override these.

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        return {"role": "assistant", "content": response.raw_content}

    def tool_results_message(self, results: list[ToolResult]) -> dict[str, Any]:
        """All results for one assistant turn, in a single user message."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.as_text(),
                    **({"is_error": True} if r.is_error else {}),
                }
                for r in results
            ],
        }
