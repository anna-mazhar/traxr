"""The public ``LLMClient`` protocol for the built-in reference agent.

Implement these two methods to bring any provider to
the built-in agent path::

    class MyClient:
        def generate(self, prompt, response_type="default", context=None):
            ...
        def generate_with_tools(self, prompt, tools, response_type="default",
                                context=None, system_prompt_override=None):
            ...

External agents do not use this protocol — they own their LLM and are
captured at the SDK boundary by ``traxr.instrument()`` instead.
"""

from typing import Any, Protocol, runtime_checkable

from traxr.llm.types import LLMResponse, LLMToolResponse

__all__ = ["LLMClient"]


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients usable with :func:`traxr.agents.builtin_agent`.

    ``response_type`` is a routing hint (``"plan"``, ``"route"``,
    ``"data_analyst"``, ``"synthesize"``, ...) used to select system prompts
    or, for the deterministic stub, scripted replies.
    """

    def generate(
        self,
        prompt: str,
        response_type: str = "default",
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a plain-text response."""
        ...

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        response_type: str = "default",
        context: dict[str, Any] | None = None,
        system_prompt_override: str | None = None,
    ) -> LLMToolResponse:
        """Generate a response that may include structured tool calls.

        ``tools`` is a list of tool schemas (``ToolSchema`` for the built-in
        agent). Returned ``tool_calls`` items expose ``tool_name``,
        ``operation``, ``arguments`` and ``call_id``.
        """
        ...
