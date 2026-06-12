"""Agent adapters and the built-in reference agent facade.

The ``Task``/``AgentRunner`` contract and ``invoke_agent`` harness live in
:mod:`traxr.agents.task`; ``from_langgraph()`` (Tier 1 capture) in
:mod:`traxr.agents.langgraph`.
"""

from traxr.agents.builtin import BuiltinAgent, builtin_agent
from traxr.agents.langgraph import from_langgraph
from traxr.agents.task import AgentRunner, Task, invoke_agent

__all__ = [
    "AgentRunner",
    "BuiltinAgent",
    "Task",
    "builtin_agent",
    "from_langgraph",
    "invoke_agent",
]
