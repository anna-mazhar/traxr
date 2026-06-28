"""Core types for retrieval-contamination experiments."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NewType
import hashlib


# Stable content-hash identifier for retrieval items
RetrievalID = NewType("RetrievalID", str)


def make_retrieval_id(content: str) -> RetrievalID:
    """Create a stable RetrievalID from content hash."""
    return RetrievalID(hashlib.sha256(content.encode()).hexdigest()[:16])


class RetrievalCondition(Enum):
    """Experimental conditions for retrieval interventions."""
    NORMAL = auto()     # Pass-through, no intervention
    NULL = auto()       # Returns empty results


class CitationType(Enum):
    """How an agent used a retrieval item."""
    QUOTED = auto()      # Direct quote
    PARAPHRASED = auto() # Paraphrased content
    REFERENCED = auto()  # Referenced without content


class AgentRole(Enum):
    """Role categories for agents."""
    RESEARCHER = auto()   # Queries retrieval, writes notes
    CRITIC = auto()       # Reviews content, no direct retrieval
    SYNTHESIZER = auto()  # Produces final output
    ROUTER = auto()       # Chooses next agent


@dataclass
class CostProxy:
    """Tracks computational cost proxies for a run."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_calls: int = 0
    total_steps: int = 0

    def add_tokens(self, prompt: int, completion: int) -> None:
        """Add token counts."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def add_retrieval_call(self) -> None:
        """Record a retrieval call."""
        self.retrieval_calls += 1

    def increment_steps(self) -> None:
        """Increment step counter."""
        self.total_steps += 1

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retrieval_calls": self.retrieval_calls,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostProxy":
        """Create from dictionary."""
        return cls(
            total_tokens=data.get("total_tokens", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            retrieval_calls=data.get("retrieval_calls", 0),
            total_steps=data.get("total_steps", 0),
        )
