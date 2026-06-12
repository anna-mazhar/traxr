"""Provenance tracking for information flow."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import uuid
import hashlib

from ..core.types import RetrievalID
from ..core.state import MemoryEntry
from ..core.outputs import AgentOutput
from ..retrieval.items import RetrievalItem
from .records import ProvenanceRecord


class ProvenanceTracker:
    """Tracks provenance of all information in the system."""

    def __init__(self):
        self._records: Dict[str, ProvenanceRecord] = {}
        self._retrieval_to_provenance: Dict[RetrievalID, str] = {}
        self._memory_to_provenance: Dict[str, str] = {}

    def track_retrieval(
        self,
        item: RetrievalItem,
        step: int,
        agent_name: str,
    ) -> ProvenanceRecord:
        """Track provenance of a retrieval item."""
        # Check if already tracked
        if item.retrieval_id in self._retrieval_to_provenance:
            return self._records[self._retrieval_to_provenance[item.retrieval_id]]

        record = ProvenanceRecord(
            id=str(uuid.uuid4())[:8],
            source_type="retrieval",
            source_id=item.retrieval_id,
            step_introduced=step,
            agent_name=agent_name,
            content_hash=hashlib.sha256(item.content.encode()).hexdigest()[:16],
            metadata={
                "score": item.score,
                "source": item.source,
                "is_injected": item.is_injected,
                "is_oracle": item.is_oracle,
            },
        )

        self._records[record.id] = record
        self._retrieval_to_provenance[item.retrieval_id] = record.id

        return record

    def track_memory_write(
        self,
        entry: MemoryEntry,
        cited_retrieval_ids: List[RetrievalID],
        read_memory_ids: Set[str],
    ) -> ProvenanceRecord:
        """Track provenance of a memory write."""
        # Gather parent provenance IDs
        parent_ids = []

        # From cited retrievals
        for rid in cited_retrieval_ids:
            if rid in self._retrieval_to_provenance:
                parent_ids.append(self._retrieval_to_provenance[rid])

        # From read memory entries
        for mid in read_memory_ids:
            if mid in self._memory_to_provenance:
                parent_ids.append(self._memory_to_provenance[mid])

        record = ProvenanceRecord(
            id=str(uuid.uuid4())[:8],
            source_type="memory",
            source_id=entry.id,
            step_introduced=entry.step,
            agent_name=entry.agent_name,
            content_hash=hashlib.sha256(entry.content.encode()).hexdigest()[:16],
            parent_ids=parent_ids,
            metadata={
                "entry_type": entry.entry_type,
            },
        )

        self._records[record.id] = record
        self._memory_to_provenance[entry.id] = record.id

        return record

    def get_provenance_chain(self, record_id: str) -> List[ProvenanceRecord]:
        """Get full provenance chain for a record."""
        if record_id not in self._records:
            return []

        chain = []
        visited = set()
        to_visit = [record_id]

        while to_visit:
            current_id = to_visit.pop(0)
            if current_id in visited:
                continue

            visited.add(current_id)
            record = self._records.get(current_id)
            if record:
                chain.append(record)
                to_visit.extend(record.parent_ids)

        return chain

    def get_retrieval_provenance(self, retrieval_id: RetrievalID) -> Optional[ProvenanceRecord]:
        """Get provenance record for a retrieval ID."""
        prov_id = self._retrieval_to_provenance.get(retrieval_id)
        return self._records.get(prov_id) if prov_id else None

    def get_memory_provenance(self, memory_id: str) -> Optional[ProvenanceRecord]:
        """Get provenance record for a memory entry."""
        prov_id = self._memory_to_provenance.get(memory_id)
        return self._records.get(prov_id) if prov_id else None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "records": {k: v.to_dict() for k, v in self._records.items()},
            "retrieval_to_provenance": dict(self._retrieval_to_provenance),
            "memory_to_provenance": dict(self._memory_to_provenance),
        }
