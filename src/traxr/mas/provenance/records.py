"""Provenance record types."""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from datetime import datetime, timezone

from ..core.types import RetrievalID


@dataclass
class ProvenanceRecord:
    """Records the provenance of a piece of information."""
    id: str
    source_type: str  # "retrieval", "memory", "agent_generation"
    source_id: str  # RetrievalID or MemoryEntry.id
    step_introduced: int
    agent_name: str
    content_hash: str
    parent_ids: List[str] = field(default_factory=list)  # Parent provenance records
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "step_introduced": self.step_introduced,
            "agent_name": self.agent_name,
            "content_hash": self.content_hash,
            "parent_ids": list(self.parent_ids),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
