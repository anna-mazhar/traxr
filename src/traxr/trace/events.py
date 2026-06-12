"""Structured trace events for cross-run divergence analysis.

Any non-empty string is a valid ``event_type`` (unknown types get a
graceful fallback signature in :mod:`traxr.trace.registry`). Per-type
key-field comparison is delegated to the registry.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from traxr.errors import MalformedEventError


@dataclass
class TraceEvent:
    """A single alignable event in an execution trace.

    Events are aligned across clean/perturbed runs by
    (step_num, event_type, occurrence_order) for divergence detection.

    Payload schemas for the built-in event types:
        routing_decision: {chosen_agent: str, reasoning_hash: str}
        tool_invocation:  {tool_name: str, operation: str, arguments: dict,
                           success: bool, output_hash: str, output_preview: str}
        memory_write:     {entry_id: str, entry_type: str, content_hash: str,
                           confidence: float}
        memory_read:      {entry_ids: list[str], entry_types: list[str]}
        retrieval_shown:  {query: str, item_count: int, item_hashes: list[str]}
        agent_output:     {action: str, content_hash: str, is_final_answer: bool,
                           citation_ids: list[str]}
        final_answer:     {answer: str, answer_hash: str}
        tool_failure:     {tool_name: str, error: str}
        agent_halt:       {reason: str}

    External (capture-layer) event types and their payloads are documented in
    :mod:`traxr.trace.registry`. Custom types are allowed; register a
    signature builder via :func:`traxr.trace.registry.register_signature` to
    lift them out of the ``unknown:{event_type}`` fallback.
    """

    event_type: str
    sequence_index: int
    step_num: int
    agent_name: str
    payload: dict[str, Any]
    content_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type:
            raise MalformedEventError(
                f"event_type must be a non-empty string, got {self.event_type!r}"
            )

    def semantic_equals(self, other: "TraceEvent") -> bool:
        """Check if two events are semantically equivalent.

        Uses content_hash for fast path, then falls back to
        key-field comparison per event type (registry-driven).
        """
        if self.event_type != other.event_type:
            return False
        if self.content_hash == other.content_hash:
            return True
        return self._key_field_compare(other)

    def _key_field_compare(self, other: "TraceEvent") -> bool:
        """Compare events by their key fields (registry-driven, type-specific)."""
        from traxr.trace import registry

        return registry.key_fields_equal(self.event_type, self.payload, other.payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @staticmethod
    def compute_content_hash(payload: dict[str, Any]) -> str:
        """Compute a deterministic hash of a payload dict."""
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
