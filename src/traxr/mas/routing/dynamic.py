"""Dynamic router implementation."""

from typing import Dict, Optional

from .base import Router
from ..core.state import SharedState
from ..agents.base import BaseAgent


class DynamicRouter(Router):
    """Router that uses a routing agent (SupervisorRouter) to choose the next agent."""

    def __init__(self, router_agent: BaseAgent):
        """Initialize with a router agent.

        Args:
            router_agent: The routing agent that makes routing decisions
                         (typically SupervisorRouter)
        """
        self._router_agent = router_agent
        self._last_chosen: Optional[str] = None

    @property
    def name(self) -> str:
        return "dynamic"

    @property
    def router_agent(self) -> BaseAgent:
        """Get the router agent."""
        return self._router_agent

    @property
    def last_chosen(self) -> Optional[str]:
        """Get the last chosen agent name."""
        return self._last_chosen

    def get_next_agent(
        self,
        state: SharedState,
        agents: Dict[str, BaseAgent],
        current_step: int,
    ) -> Optional[BaseAgent]:
        """Use router agent to choose next agent."""
        # Check if we have a final answer
        if state.has_final_answer():
            return None

        # Use router agent to choose
        next_agent_name = self._router_agent.choose_next(state, current_step)
        self._last_chosen = next_agent_name

        return agents.get(next_agent_name)

    def reset(self) -> None:
        """Reset router state."""
        self._last_chosen = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "router_agent": self._router_agent.name,
            "last_chosen": self._last_chosen,
            "available_agents": self._router_agent.available_agents,
        }
