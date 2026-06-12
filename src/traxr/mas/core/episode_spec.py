"""Episode specification and experiment conditions."""

from dataclasses import dataclass, field
from typing import Optional, List
import hashlib
import json

from .types import RetrievalCondition


@dataclass(frozen=True)
class TerminationCriteria:
    """Criteria for terminating an episode."""
    max_steps: int = 10
    max_tokens: Optional[int] = None
    require_final_answer: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "require_final_answer": self.require_final_answer,
        }


@dataclass(frozen=True)
class EpisodeSpec:
    """Immutable specification for an episode.

    Same EpisodeSpec + different ExperimentCondition = counterfactual experiment.
    """
    task_id: str
    seed: int
    agent_sequence: tuple[str, ...]  # Agent names for reference (dynamic routing chooses order)
    termination: TerminationCriteria = field(default_factory=TerminationCriteria)

    def spec_hash(self) -> str:
        """Unique identifier for baseline comparisons."""
        data = {
            "task_id": self.task_id,
            "seed": self.seed,
            "agent_sequence": list(self.agent_sequence),
            "termination": self.termination.to_dict(),
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "agent_sequence": list(self.agent_sequence),
            "termination": self.termination.to_dict(),
            "spec_hash": self.spec_hash(),
        }


@dataclass
class ExperimentCondition:
    """Experimental variables that can vary for the same EpisodeSpec."""
    retrieval_condition: RetrievalCondition = RetrievalCondition.NORMAL

    def condition_name(self) -> str:
        """Human-readable name for this condition combination."""
        return f"dynamic_{self.retrieval_condition.name.lower()}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "retrieval_condition": self.retrieval_condition.name,
            "condition_name": self.condition_name(),
        }
