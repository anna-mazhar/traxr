"""Agent adapters and the built-in reference agent facade.

``builtin_agent()`` lands here in M3; the ``Task``/``AgentRunner`` contract
and ``from_langgraph()`` follow in M3b/M4b.
"""

from traxr.agents.builtin import BuiltinAgent, builtin_agent

__all__ = ["BuiltinAgent", "builtin_agent"]
