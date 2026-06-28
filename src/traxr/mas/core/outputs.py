"""Agent output types with citation tracking."""

from dataclasses import dataclass, field
from typing import Optional, List

from .types import RetrievalID, CitationType


@dataclass
class CitationRecord:
    """Tracks how an agent used a retrieval item."""
    retrieval_id: RetrievalID
    citation_type: CitationType
    excerpt: Optional[str] = None  # The quoted/paraphrased text if applicable

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "retrieval_id": self.retrieval_id,
            "citation_type": self.citation_type.name,
            "excerpt": self.excerpt,
        }


@dataclass
class AgentOutput:
    """Output from an agent's step execution."""
    agent_name: str
    action: str  # "write_note", "write_critique", "final_answer", "route", etc.
    content: str
    citations: List[CitationRecord] = field(default_factory=list)
    memory_entry_id: Optional[str] = None  # ID of written memory entry if any
    is_final_answer: bool = False
    next_agent_suggestion: Optional[str] = None  # For router agents
    metadata: dict = field(default_factory=dict)

    # Cost tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def get_cited_retrieval_ids(self) -> List[RetrievalID]:
        """Get all retrieval IDs cited in this output."""
        return [c.retrieval_id for c in self.citations]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "action": self.action,
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "memory_entry_id": self.memory_entry_id,
            "is_final_answer": self.is_final_answer,
            "next_agent_suggestion": self.next_agent_suggestion,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "metadata": self.metadata,
        }
