"""Provider-agnostic LLM layer: one interface, an Anthropic adapter and a scripted mock."""

from .anthropic_llm import DEFAULT_MODEL, AnthropicLLM
from .base import LLM, LLMError, LLMResponse, ToolCall, ToolResult
from .mock import MockLLM

__all__ = [
    "DEFAULT_MODEL",
    "LLM",
    "AnthropicLLM",
    "LLMError",
    "LLMResponse",
    "MockLLM",
    "ToolCall",
    "ToolResult",
]
