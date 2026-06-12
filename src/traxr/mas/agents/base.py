"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Dict, Any

from ..core.types import AgentRole
from ..core.state import SharedState, MemoryAccessTracker
from ..core.outputs import AgentOutput
from ..retrieval.items import RetrievalResult

if TYPE_CHECKING:
    from ..tools.base import ToolExecutor


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm: Optional[Any] = None,  # OpenAI or Tinker client
        tool_executor: Optional["ToolExecutor"] = None,
    ):
        self._name = name
        self._role = role
        self._llm = llm
        self._tool_executor = tool_executor

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    @property
    def role(self) -> AgentRole:
        """Agent role."""
        return self._role

    @property
    def tool_executor(self) -> Optional["ToolExecutor"]:
        """Tool executor for this agent."""
        return self._tool_executor

    @property
    def uses_retrieval(self) -> bool:
        """Whether this agent uses retrieval."""
        return False

    @property
    def uses_tools(self) -> bool:
        """Whether this agent uses tools."""
        return False

    @abstractmethod
    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Execute one step of the agent.

        Args:
            state: Shared state for reading/writing
            memory_tracker: Tracker for memory access
            retrieval_result: Optional retrieval results (if agent uses retrieval)
            step_num: Current step number

        Returns:
            AgentOutput with the agent's action and content
        """
        pass

    def get_tool_schemas(self) -> List:
        """Get tool schemas for structured tool invocation via generate_with_tools().

        Override in subclasses that use tools to return the relevant ToolSchema
        objects. The default returns an empty list (no structured tool calling).
        """
        return []

    def get_retrieval_query(self, state: SharedState) -> Optional[str]:
        """Get retrieval query if this agent uses retrieval.

        Override in subclasses that use retrieval.
        """
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self._name,
            "role": self._role.name,
            "uses_retrieval": self.uses_retrieval,
        }
