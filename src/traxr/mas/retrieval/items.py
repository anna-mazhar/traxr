"""Retrieval item types."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.types import RetrievalID, make_retrieval_id


@dataclass
class RetrievalItem:
    """A single item returned from retrieval."""
    content: str
    score: float
    source: str  # Document/source identifier
    retrieval_id: RetrievalID = field(init=False)
    is_injected: bool = False  # True if this was injected by a condition (reserved for future use)
    is_oracle: bool = False  # True if this is oracle-quality (reserved for future use)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Generate retrieval ID from content."""
        self.retrieval_id = make_retrieval_id(self.content)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "retrieval_id": self.retrieval_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "is_injected": self.is_injected,
            "is_oracle": self.is_oracle,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalItem":
        """Create from dictionary."""
        item = cls(
            content=data["content"],
            score=data["score"],
            source=data["source"],
            is_injected=data.get("is_injected", False),
            is_oracle=data.get("is_oracle", False),
            metadata=data.get("metadata", {}),
        )
        return item


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    query: str
    items: List[RetrievalItem]
    total_available: int  # Total items in index (for stats)
    filtered_count: int = 0  # Items filtered by gatekeeper

    def get_retrieval_ids(self) -> List[RetrievalID]:
        """Get all retrieval IDs in this result."""
        return [item.retrieval_id for item in self.items]

    def get_injected_ids(self) -> List[RetrievalID]:
        """Get retrieval IDs of injected items."""
        return [item.retrieval_id for item in self.items if item.is_injected]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "items": [item.to_dict() for item in self.items],
            "total_available": self.total_available,
            "filtered_count": self.filtered_count,
        }
