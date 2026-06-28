"""Core module for retrieval-contamination experiments."""

from .types import (
    RetrievalID,
    make_retrieval_id,
    RetrievalCondition,
    CitationType,
    AgentRole,
    CostProxy,
)
from .episode_spec import EpisodeSpec, TerminationCriteria, ExperimentCondition
from .state import TaskInput, MemoryEntry, SharedState, StateListener, MemoryAccessTracker
from .outputs import CitationRecord, AgentOutput
from .runner import EpisodeRunner, StepResult, EpisodeResult

__all__ = [
    "RetrievalID",
    "make_retrieval_id",
    "RetrievalCondition",
    "CitationType",
    "AgentRole",
    "CostProxy",
    "EpisodeSpec",
    "TerminationCriteria",
    "ExperimentCondition",
    "TaskInput",
    "MemoryEntry",
    "SharedState",
    "StateListener",
    "MemoryAccessTracker",
    "CitationRecord",
    "AgentOutput",
    "EpisodeRunner",
    "StepResult",
    "EpisodeResult",
]
