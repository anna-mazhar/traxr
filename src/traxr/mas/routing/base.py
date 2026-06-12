"""Base router interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..core.state import SharedState
from ..agents.base import BaseAgent


class Router(ABC):
    """Abstract base class for agent routers."""

    @abstractmethod
    def get_next_agent(
        self,
        state: SharedState,
        agents: Dict[str, BaseAgent],
        current_step: int,
    ) -> Optional[BaseAgent]:
        """Get the next agent to execute.

        Args:
            state: Current shared state
            agents: Dictionary of available agents by name
            current_step: Current step number

        Returns:
            Next agent to execute, or None if episode should end
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset router state for a new episode."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Router name for serialization."""
        pass
