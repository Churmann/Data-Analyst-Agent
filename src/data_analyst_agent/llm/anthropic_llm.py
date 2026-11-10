"""Anthropic Messages API adapter."""

from __future__ import annotations

from typing import Any

from .base import LLM, LLMError, LLMResponse, SystemPrompt, ToolCall

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM(LLM):
    """Calls the Messages API with native tool use.

    Thinking is always on for the default model; depth is set with ``effort``.
    ``fallbacks="default"`` lets the API re-run a request on another model if
    the first one declines it, instead of the turn simply stopping.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "high",
        max_tokens: int = 16000,
        one_tool_per_turn: bool = True,
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.one_tool_per_turn = one_tool_per_turn
        self.fallbacks = fallbacks
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so offline use needs no SDK or key

            self._client = anthropic.Anthropic()
        return self._client

    def request_params(
        self,
        system: SystemPrompt,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
            "output_config": {"effort": self.effort},
        }
        if tools:
            params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": self.one_tool_per_turn}
        if self.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def complete(
        self,
        system: SystemPrompt,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        import anthropic

        params = self.request_params(system, messages, tools)
        try:
            client = self.client
            if self.fallbacks:
                response = client.beta.messages.create(**params)
            else:
                response = client.messages.create(**params)
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Could not reach the API: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.AnthropicError as e:  # e.g. no credentials configured
            raise LLMError(f"{type(e).__name__}: {e}") from e

        return self.normalise(response)

    @staticmethod
    def normalise(response: Any) -> LLMResponse:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
            }

        stop_reason = response.stop_reason
        if stop_reason == "refusal":
            calls = []  # never run tools from a declined turn
        elif stop_reason == "max_tokens" and calls:
            raise LLMError("Response hit max_tokens mid tool call; the tool input may be truncated.")

        return LLMResponse(
            text="\n".join(t for t in texts if t).strip(),
            tool_calls=calls,
            # SDK block objects are passed back as-is so thinking blocks stay intact.
            raw_content=list(response.content),
            stop_reason=stop_reason,
            usage=usage,
        )
