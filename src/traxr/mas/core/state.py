"""Shared state management for multi-agent episodes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Any
from datetime import datetime, timezone
import uuid

from .types import RetrievalID


@dataclass
class TaskInput:
    """Input for an episode task."""
    task_id: str
    query: str
    context: Optional[str] = None
    expected_answer: Optional[str] = None  # For evaluation
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "query": self.query,
            "context": self.context,
            "expected_answer": self.expected_answer,
            "metadata": self.metadata,
        }


@dataclass
class MemoryEntry:
    """An entry in shared memory written by an agent."""
    id: str
    agent_name: str
    content: str
    entry_type: str  # "note", "critique", "answer", etc.
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cited_retrieval_ids: List[RetrievalID] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = "preliminary"  # "preliminary", "failed_attempt", "verified", "final"
    confidence: float = 0.5  # 0.0 to 1.0, indicates reliability of this entry

    @classmethod
    def create(
        cls,
        agent_name: str,
        content: str,
        entry_type: str,
        step: int,
        cited_retrieval_ids: Optional[List[RetrievalID]] = None,
        metadata: Optional[dict] = None,
        status: str = "preliminary",
        confidence: float = 0.5,
    ) -> "MemoryEntry":
        """Create a new memory entry with generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            agent_name=agent_name,
            content=content,
            entry_type=entry_type,
            step=step,
            cited_retrieval_ids=cited_retrieval_ids or [],
            metadata=metadata or {},
            status=status,
            confidence=confidence,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "content": self.content,
            "entry_type": self.entry_type,
            "step": self.step,
            "timestamp": self.timestamp,
            "cited_retrieval_ids": list(self.cited_retrieval_ids),
            "metadata": self.metadata,
            "status": self.status,
            "confidence": self.confidence,
        }


class StateListener(ABC):
    """Abstract listener for state changes."""

    @abstractmethod
    def on_memory_write_attempt(
        self, entry: MemoryEntry, context: dict
    ) -> tuple[bool, Optional[str]]:
        """Called before a memory write.

        Args:
            entry: The entry to be written
            context: Additional context (current step, agent, etc.)

        Returns:
            Tuple of (allowed, reason). If not allowed, reason explains why.
        """
        pass

    @abstractmethod
    def on_memory_written(self, entry: MemoryEntry) -> None:
        """Called after a memory entry is successfully written."""
        pass


class SharedState:
    """Shared state accessible by all agents during an episode."""

    def __init__(self, task_input: TaskInput):
        self.task_input = task_input
        self._memory: List[MemoryEntry] = []
        self._listeners: List[StateListener] = []
        self._final_answer: Optional[str] = None
        self._current_step: int = 0

    def add_listener(self, listener: StateListener) -> None:
        """Add a state listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: StateListener) -> None:
        """Remove a state listener."""
        self._listeners.remove(listener)

    @property
    def current_step(self) -> int:
        """Get current step number."""
        return self._current_step

    def set_current_step(self, step: int) -> None:
        """Set current step (called by runner)."""
        self._current_step = step

    def write_memory(
        self,
        agent_name: str,
        content: str,
        entry_type: str,
        cited_retrieval_ids: Optional[List[RetrievalID]] = None,
        metadata: Optional[dict] = None,
        status: str = "preliminary",
        confidence: float = 0.5,
    ) -> tuple[bool, Optional[str], Optional[MemoryEntry]]:
        """Write an entry to shared memory.

        Returns:
            Tuple of (success, rejection_reason, entry).
            If rejected, entry is None.
        """
        entry = MemoryEntry.create(
            agent_name=agent_name,
            content=content,
            entry_type=entry_type,
            step=self._current_step,
            cited_retrieval_ids=cited_retrieval_ids,
            metadata=metadata,
            status=status,
            confidence=confidence,
        )

        # Check with listeners
        context = {"step": self._current_step, "agent_name": agent_name}
        for listener in self._listeners:
            allowed, reason = listener.on_memory_write_attempt(entry, context)
            if not allowed:
                return False, reason, None

        # Write successful
        self._memory.append(entry)

        # Notify listeners
        for listener in self._listeners:
            listener.on_memory_written(entry)

        return True, None, entry

    def read_memory(
        self,
        entry_types: Optional[List[str]] = None,
        agent_names: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """Read entries from shared memory with optional filtering."""
        result = self._memory

        if entry_types is not None:
            result = [e for e in result if e.entry_type in entry_types]

        if agent_names is not None:
            result = [e for e in result if e.agent_name in agent_names]

        return list(result)

    def get_all_memory(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        return list(self._memory)

    def set_final_answer(self, answer: str) -> None:
        """Set the final answer for the episode."""
        self._final_answer = answer

    def get_final_answer(self) -> Optional[str]:
        """Get the final answer if set."""
        return self._final_answer

    def has_final_answer(self) -> bool:
        """Check if a final answer has been set."""
        return self._final_answer is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_input": self.task_input.to_dict(),
            "memory": [e.to_dict() for e in self._memory],
            "final_answer": self._final_answer,
            "current_step": self._current_step,
        }


class MemoryAccessTracker:
    """Wraps SharedState to track which memory entries an agent reads.

    Enables taint propagation from tainted memory to subsequent decisions.
    """

    def __init__(self, state: SharedState, agent_name: str):
        self._state = state
        self._agent_name = agent_name
        self._read_entry_ids: Set[str] = set()

    def read_memory(
        self,
        entry_types: Optional[List[str]] = None,
        agent_names: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """Read memory and track which entries were accessed."""
        entries = self._state.read_memory(entry_types, agent_names)
        for entry in entries:
            self._read_entry_ids.add(entry.id)
        return entries

    def get_all_memory(self) -> List[MemoryEntry]:
        """Get all memory and track access."""
        entries = self._state.get_all_memory()
        for entry in entries:
            self._read_entry_ids.add(entry.id)
        return entries

    def get_read_entry_ids(self) -> Set[str]:
        """Get IDs of all memory entries that were read."""
        return set(self._read_entry_ids)

    @property
    def task_input(self) -> TaskInput:
        """Access task input without tracking."""
        return self._state.task_input

    @property
    def current_step(self) -> int:
        """Access current step without tracking."""
        return self._state.current_step

    def has_final_answer(self) -> bool:
        """Check if final answer exists."""
        return self._state.has_final_answer()
