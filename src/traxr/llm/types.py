"""LLM response types shared by every :class:`~traxr.llm.LLMClient`.

``metadata`` defaults are normalized in ``__post_init__`` so callers may
pass ``None``.
"""

from dataclasses import dataclass, field
from typing import Any

__all__ = ["LLMResponse", "LLMToolResponse"]


@dataclass
class LLMResponse:
    """Response from a plain-text LLM call (:meth:`LLMClient.generate`)."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str = "unknown"
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMToolResponse:
    """Response from a tool-enabled LLM call (:meth:`LLMClient.generate_with_tools`).

    ``tool_calls`` holds structured tool invocations. The built-in reference
    agent consumes objects exposing ``tool_name``, ``operation``,
    ``arguments`` and ``call_id`` attributes (duck-typed; see
    ``traxr.mas.tools.tool_schema.StructuredToolCall``).
    """

    content: str | None
    tool_calls: list[Any]
    prompt_tokens: int
    completion_tokens: int
    model: str = "unknown"
    finish_reason: str = "stop"  # "stop" or "tool_calls"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Whether the response contains structured tool calls."""
        return bool(self.tool_calls)
