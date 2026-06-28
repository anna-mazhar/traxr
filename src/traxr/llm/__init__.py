"""LLM boundary for the built-in reference agent.

Public surface: the :class:`LLMClient` protocol, the response types it
returns, the :class:`OpenAICompatibleClient` (``[openai]`` extra — the class
imports ``openai`` lazily and raises ``OptionalDependencyError`` when it is
missing), and the :class:`DeterministicLLMStub` (no keys, no network).
"""

from traxr.llm.openai_compat import OpenAICompatibleClient
from traxr.llm.protocol import LLMClient
from traxr.llm.stub import DeterministicLLMStub
from traxr.llm.types import LLMResponse, LLMToolResponse

__all__ = [
    "DeterministicLLMStub",
    "LLMClient",
    "LLMResponse",
    "LLMToolResponse",
    "OpenAICompatibleClient",
]
