"""Trace collector for accumulating events during a run.

``emit()`` is thread-safe (concurrent agents must not corrupt sequence
indices).
"""

import threading
from typing import Any

from traxr.trace.events import TraceEvent


class TraceCollector:
    """Collects TraceEvents during a run.

    Create one per run (baseline or perturbation), then pass it through the
    runner/capture layer so events are emitted at each execution point.

    Usage:
        collector = TraceCollector(run_label="baseline")
        # ... emit events during the run ...
        events = collector.events
    """

    def __init__(self, run_label: str):
        """Initialize collector.

        Args:
            run_label: Identifier for this run (e.g., "baseline", "column_swap").
        """
        self.run_label = run_label
        self.events: list[TraceEvent] = []
        self._sequence_counter = 0
        self._emit_lock = threading.Lock()

    def emit(
        self,
        event_type: str,
        step_num: int,
        agent_name: str,
        payload: dict[str, Any],
    ) -> TraceEvent:
        """Create and store a new trace event (thread-safe).

        Args:
            event_type: Any non-empty string (built-in, external, or custom).
            step_num: Episode step number.
            agent_name: Name of the agent involved.
            payload: Type-specific event data.

        Returns:
            The created TraceEvent.

        Raises:
            MalformedEventError: If event_type is empty or not a string.
        """
        content_hash = TraceEvent.compute_content_hash(payload)

        with self._emit_lock:
            event = TraceEvent(
                event_type=event_type,
                sequence_index=self._sequence_counter,
                step_num=step_num,
                agent_name=agent_name,
                payload=payload,
                content_hash=content_hash,
            )
            self.events.append(event)
            self._sequence_counter += 1
        return event

    @property
    def event_count(self) -> int:
        """Number of events collected."""
        return len(self.events)

    def get_events_by_type(self, event_type: str) -> list[TraceEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_step(self, step_num: int) -> list[TraceEvent]:
        """Get all events for a given step."""
        return [e for e in self.events if e.step_num == step_num]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trace for output."""
        return {
            "run_label": self.run_label,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceCollector":
        """Reconstruct a TraceCollector from serialized data.

        Useful for loading saved traces for offline divergence analysis.
        """
        collector = cls(run_label=data["run_label"])
        for event_data in data.get("events", []):
            event = TraceEvent(**event_data)
            collector.events.append(event)
            collector._sequence_counter = max(collector._sequence_counter, event.sequence_index + 1)
        return collector
