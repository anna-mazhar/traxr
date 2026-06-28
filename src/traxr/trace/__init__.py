"""Trace capture: events, collector, and the event-type registry.

Three deliberate design choices:

1. ``event_type`` is open to any non-empty string;
   :class:`~traxr.errors.MalformedEventError` is raised
   only for empty/non-string types.
2. ``TraceCollector.emit()`` is thread-safe (concurrent agents must not
   corrupt sequence indices).
3. Per-event-type behavior (signatures, divergence classifiers, key-field
   comparison) lives in :mod:`traxr.trace.registry` instead of hard-coded
   ``if event_type == ...`` chains. The analyzer-golden gate pins the
   built-in vocabulary's behavior exactly.
"""

from traxr.trace.collector import TraceCollector
from traxr.trace.events import TraceEvent
from traxr.trace.registry import (
    BUILTIN_EVENT_TYPES,
    EXTERNAL_EVENT_TYPES,
    STRUCTURAL_DIVERGENCE_TYPES,
    EventTypeSpec,
    register_signature,
)

__all__ = [
    "BUILTIN_EVENT_TYPES",
    "EXTERNAL_EVENT_TYPES",
    "STRUCTURAL_DIVERGENCE_TYPES",
    "EventTypeSpec",
    "TraceCollector",
    "TraceEvent",
    "register_signature",
]
