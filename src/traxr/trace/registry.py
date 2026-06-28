"""Event-type registry: signature builders + divergence classifiers.

Signature builders, divergence classifiers, and key-field comparison are
registry dispatch rather than hard-coded ``if event_type == ...`` chains.
The built-in vocabulary's behavior is pinned exactly by the analyzer-golden
gate (``make analyzer-goldens``).

Three vocabularies:

* **Built-in** (emitted by the bundled reference agent): the original 7
  (``routing_decision``, ``tool_invocation``, ``memory_write``,
  ``memory_read``, ``retrieval_shown``, ``agent_output``, ``final_answer``)
  plus the additive ``tool_failure`` and ``agent_halt``.
* **External** (emitted by the Tier 0/Tier 1 capture layer and the harness):

  ============== =====================================================
  event_type     signature
  ============== =====================================================
  llm_call       ``llm:{model}:{finish_reason}:{tool_call_names or -}``
  tool_request   ``tool_req:{tool_name}`` (args stay out of signatures)
  tool_result    ``tool_res:{tool_name}`` (never call_id — provider IDs
                 differ across runs)
  agent_error    ``agent_error:{exc_type}``
  ============== =====================================================

* **Custom** (via ``traxr.emit()``): fall back to ``unknown:{event_type}``
  with a one-time :class:`~traxr.errors.UnknownEventTypeWarning`, upgradeable
  through :func:`register_signature`.

Signatures are *structural*: they never include argument values, hashes, or
other lexical content — lexical noise would saturate ``d_norm``, against the
paper's structural-divergence philosophy.
"""

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from traxr.errors import MalformedEventError, UnknownEventTypeWarning

Payload = Mapping[str, Any]
SignatureFn = Callable[[Payload], str]
ClassifierFn = Callable[[Payload, Payload], "str | None"]
KeyFieldsFn = Callable[[Payload, Payload], bool]

#: Divergence type when an event appears only in the perturbed trace.
EVENT_MISSING_IN_CLEAN = "event_missing_in_clean"
#: Divergence type when an event appears only in the clean trace.
EVENT_MISSING_IN_PERTURBED = "event_missing_in_perturbed"
#: Divergence type when aligned events have different event types.
EVENT_TYPE_DIFFERS = "event_type_differs"

#: Alignment-level divergence types that exist independent of any event type.
_ALIGNMENT_DIVERGENCE_TYPES = frozenset(
    {EVENT_MISSING_IN_CLEAN, EVENT_MISSING_IN_PERTURBED, EVENT_TYPE_DIFFERS}
)

#: Structural divergence types: control flow or execution state changes.
#: Mutated in place as event types register; importers always see the live set.
STRUCTURAL_DIVERGENCE_TYPES: set[str] = set(_ALIGNMENT_DIVERGENCE_TYPES)


@dataclass(frozen=True)
class EventTypeSpec:
    """Everything the analyzer needs to know about one event type.

    Attributes:
        signature: Builds the structural signature string from the payload.
        classifier: Compares two *same-type* payloads and returns a structural
            divergence type, or ``None`` if the difference is lexical only.
            ``None`` (no classifier) means same-type pairs never diverge
            structurally (e.g. ``memory_read``, ``final_answer``).
        key_fields_equal: Type-specific semantic equality on payloads, used by
            ``TraceEvent.semantic_equals`` when content hashes differ.
        structural_types: Divergence-type strings this classifier can return;
            added to :data:`STRUCTURAL_DIVERGENCE_TYPES` on registration.
        missing_in_clean_type: Divergence type reported when an event of this
            type appears only in the perturbed trace. Defaults to the generic
            ``event_missing_in_clean``; ``agent_error`` overrides it with
            ``agent_error_introduced``.
    """

    signature: SignatureFn
    classifier: ClassifierFn | None = None
    key_fields_equal: KeyFieldsFn | None = None
    structural_types: frozenset[str] = frozenset()
    missing_in_clean_type: str = EVENT_MISSING_IN_CLEAN


_REGISTRY: dict[str, EventTypeSpec] = {}
_warned_unknown_types: set[str] = set()


def register(event_type: str, spec: EventTypeSpec) -> None:
    """Register (or override) the spec for an event type."""
    if not isinstance(event_type, str) or not event_type:
        raise MalformedEventError(f"event_type must be a non-empty string, got {event_type!r}")
    _REGISTRY[event_type] = spec
    STRUCTURAL_DIVERGENCE_TYPES.update(spec.structural_types)
    STRUCTURAL_DIVERGENCE_TYPES.add(spec.missing_in_clean_type)
    # A registered type is no longer unknown; let the warning fire again if
    # the registration is somehow removed later.
    _warned_unknown_types.discard(event_type)


def register_signature(
    event_type: str,
    signature: SignatureFn,
    *,
    classifier: ClassifierFn | None = None,
    structural_types: frozenset[str] | set[str] = frozenset(),
    key_fields_equal: KeyFieldsFn | None = None,
) -> None:
    """Upgrade a custom event type from the ``unknown:{event_type}`` fallback.

    Args:
        event_type: The custom event type emitted via ``traxr.emit()``.
        signature: ``payload -> str`` structural signature builder.
        classifier: Optional ``(clean_payload, perturbed_payload) -> str | None``
            returning a structural divergence type (or ``None`` for
            lexical-only differences).
        structural_types: The divergence-type strings the classifier can
            return (so ``t*`` typing recognizes them as structural).
        key_fields_equal: Optional semantic-equality predicate on payloads.
    """
    if not callable(signature):
        raise MalformedEventError(
            f"signature for {event_type!r} must be callable, got {signature!r}"
        )
    register(
        event_type,
        EventTypeSpec(
            signature=signature,
            classifier=classifier,
            key_fields_equal=key_fields_equal,
            structural_types=frozenset(structural_types),
        ),
    )


def signature_for(event_type: str, payload: Payload) -> str:
    """Structural signature for an event (registry-driven).

    Unknown event types fall back to ``unknown:{event_type}`` and emit a
    one-time :class:`~traxr.errors.UnknownEventTypeWarning` per type.
    """
    spec = _REGISTRY.get(event_type)
    if spec is not None:
        return spec.signature(payload)
    if event_type not in _warned_unknown_types:
        _warned_unknown_types.add(event_type)
        warnings.warn(
            f"Unknown trace event type {event_type!r}: falling back to signature "
            f"'unknown:{event_type}'. Its payload will not contribute structure to "
            "divergence metrics — register a signature builder via "
            "traxr.register_signature() to upgrade it.",
            UnknownEventTypeWarning,
            stacklevel=2,
        )
    return f"unknown:{event_type}"


def classify_divergence(event_type: str, clean: Payload, perturbed: Payload) -> str | None:
    """Classify the structural divergence between two same-type payloads.

    Returns a structural divergence type, or ``None`` if the difference is
    lexical only (or the event type has no registered classifier).
    """
    spec = _REGISTRY.get(event_type)
    if spec is None or spec.classifier is None:
        return None
    return spec.classifier(clean, perturbed)


def key_fields_equal(event_type: str, p1: Payload, p2: Payload) -> bool:
    """Type-specific semantic equality on payloads (``False`` for unknown types)."""
    spec = _REGISTRY.get(event_type)
    if spec is None or spec.key_fields_equal is None:
        return False
    return spec.key_fields_equal(p1, p2)


def missing_in_clean_type(event_type: str) -> str:
    """Divergence type when an event of this type appears only in the perturbed trace."""
    spec = _REGISTRY.get(event_type)
    if spec is None:
        return EVENT_MISSING_IN_CLEAN
    return spec.missing_in_clean_type


def is_registered(event_type: str) -> bool:
    """Whether an event type has a registered spec."""
    return event_type in _REGISTRY


# ---------------------------------------------------------------------------
# Built-in vocabulary. The analyzer-golden gate
# (scripts/check_analyzer_goldens.py) pins these signatures, classifiers,
# and key fields to the committed golden outputs.
# ---------------------------------------------------------------------------


def _classify_routing_decision(cp: Payload, pp: Payload) -> str | None:
    if cp.get("chosen_agent") != pp.get("chosen_agent"):
        return "different_agent_routed"
    return None  # Same agent, different reasoning = lexical only.


def _classify_tool_invocation(cp: Payload, pp: Payload) -> str | None:
    # Structural differences in order of importance.
    if cp.get("tool_name") != pp.get("tool_name"):
        return "different_tool"
    if cp.get("operation") != pp.get("operation"):
        return "different_operation"
    if cp.get("success") != pp.get("success"):
        return "tool_success_failure_change"
    return None  # Same tool, operation, success — only lexical differences.


def _classify_memory_write(cp: Payload, pp: Payload) -> str | None:
    if cp.get("entry_type") != pp.get("entry_type"):
        return "different_entry_type"
    return None


def _classify_agent_output(cp: Payload, pp: Payload) -> str | None:
    if cp.get("action") != pp.get("action"):
        return "different_action"
    return None


register(
    "routing_decision",
    EventTypeSpec(
        signature=lambda p: f"route:{p.get('chosen_agent', '?')}",
        classifier=_classify_routing_decision,
        key_fields_equal=lambda p1, p2: p1.get("chosen_agent") == p2.get("chosen_agent"),
        structural_types=frozenset({"different_agent_routed"}),
    ),
)

register(
    "tool_invocation",
    EventTypeSpec(
        signature=lambda p: (
            f"tool:{p.get('tool_name', '?')}:{p.get('operation', '?')}"
            f":{'ok' if p.get('success', True) else 'fail'}"
        ),
        classifier=_classify_tool_invocation,
        key_fields_equal=lambda p1, p2: (
            p1.get("tool_name") == p2.get("tool_name")
            and p1.get("operation") == p2.get("operation")
            and p1.get("arguments") == p2.get("arguments")
            and p1.get("output_hash") == p2.get("output_hash")
        ),
        structural_types=frozenset(
            {"different_tool", "different_operation", "tool_success_failure_change"}
        ),
    ),
)

register(
    "memory_write",
    EventTypeSpec(
        signature=lambda p: f"mem_write:{p.get('entry_type', '?')}",
        classifier=_classify_memory_write,
        key_fields_equal=lambda p1, p2: (
            p1.get("entry_type") == p2.get("entry_type")
            and p1.get("content_hash") == p2.get("content_hash")
        ),
        structural_types=frozenset({"different_entry_type"}),
    ),
)

# The signature is count-agnostic on purpose: a change in *how many* entries
# an agent read / items it was shown is treated as lexical, not structural, so
# it does not move d_norm while leaving t*/control-flow/is_match blind to it
# (the three would otherwise disagree — d_norm rising while the typed metrics
# report a match). The *number of read/retrieval events* still registers
# structurally via insertion/deletion. FUTURE: a later version may treat
# read/retrieval cardinality as structural by registering a classifier that
# returns "different_memory_read_count" / "different_item_count" (and the count
# back in the signature) so all four metrics agree it diverged.
register(
    "memory_read",
    EventTypeSpec(
        signature=lambda p: "mem_read",
        key_fields_equal=lambda p1, p2: (
            set(p1.get("entry_ids", [])) == set(p2.get("entry_ids", []))
        ),
    ),
)

register(
    "retrieval_shown",
    EventTypeSpec(
        signature=lambda p: "retrieval",
        key_fields_equal=lambda p1, p2: (
            p1.get("query") == p2.get("query")
            and set(p1.get("item_hashes", [])) == set(p2.get("item_hashes", []))
        ),
    ),
)

register(
    "agent_output",
    EventTypeSpec(
        signature=lambda p: f"output:{p.get('action', '?')}:{p.get('is_final_answer', False)}",
        classifier=_classify_agent_output,
        key_fields_equal=lambda p1, p2: (
            p1.get("action") == p2.get("action")
            and p1.get("content_hash") == p2.get("content_hash")
        ),
        structural_types=frozenset({"different_action"}),
    ),
)

register(
    "final_answer",
    EventTypeSpec(
        # Answer differences are captured in the report's answer_changed field.
        signature=lambda p: "final_answer",
        key_fields_equal=lambda p1, p2: p1.get("answer_hash") == p2.get("answer_hash"),
    ),
)

# Additive first-class built-ins (new in traxr; the analyzer already inferred
# failures/termination from tool_invocation success and step counts).

register(
    "tool_failure",
    EventTypeSpec(
        signature=lambda p: f"tool_failure:{p.get('tool_name', '?')}",
        # A different tool failing is the existing "different_tool" change.
        classifier=lambda cp, pp: (
            "different_tool" if cp.get("tool_name") != pp.get("tool_name") else None
        ),
        key_fields_equal=lambda p1, p2: p1.get("tool_name") == p2.get("tool_name"),
        structural_types=frozenset({"different_tool"}),
    ),
)

register(
    "agent_halt",
    EventTypeSpec(
        signature=lambda p: f"agent_halt:{p.get('reason', '?')}",
        key_fields_equal=lambda p1, p2: p1.get("reason") == p2.get("reason"),
    ),
)

#: The built-in vocabulary: the original 7 plus tool_failure/agent_halt.
BUILTIN_EVENT_TYPES = frozenset(
    {
        "routing_decision",
        "tool_invocation",
        "memory_write",
        "memory_read",
        "retrieval_shown",
        "agent_output",
        "final_answer",
        "tool_failure",
        "agent_halt",
    }
)


# ---------------------------------------------------------------------------
# External vocabulary — emitted by the Tier 0 wrapper (instrument()), the
# Tier 1 LangGraph adapter, and the AgentRunner harness.
# ---------------------------------------------------------------------------


def _llm_call_signature(p: Payload) -> str:
    tool_call_names = ",".join(p.get("tool_call_names", []) or [])
    return f"llm:{p.get('model', '?')}:{p.get('finish_reason', '?')}:{tool_call_names or '-'}"


def _classify_llm_call(cp: Payload, pp: Payload) -> str | None:
    if list(cp.get("tool_call_names", []) or []) != list(pp.get("tool_call_names", []) or []):
        return "different_tool_requested"
    if cp.get("finish_reason") != pp.get("finish_reason"):
        return "llm_finish_reason_change"
    return None  # Same structure, different content/model = lexical only.


register(
    "llm_call",
    EventTypeSpec(
        signature=_llm_call_signature,
        classifier=_classify_llm_call,
        key_fields_equal=lambda p1, p2: (
            p1.get("model") == p2.get("model")
            and p1.get("finish_reason") == p2.get("finish_reason")
            and list(p1.get("tool_call_names", []) or [])
            == list(p2.get("tool_call_names", []) or [])
        ),
        structural_types=frozenset({"different_tool_requested", "llm_finish_reason_change"}),
    ),
)

register(
    "tool_request",
    EventTypeSpec(
        # Args stay out of the signature (payload carries a hash only):
        # lexical noise in signatures would saturate d_norm.
        signature=lambda p: f"tool_req:{p.get('tool_name', '?')}",
        classifier=lambda cp, pp: (
            "different_tool_requested" if cp.get("tool_name") != pp.get("tool_name") else None
        ),
        key_fields_equal=lambda p1, p2: p1.get("tool_name") == p2.get("tool_name"),
        structural_types=frozenset({"different_tool_requested"}),
    ),
)

register(
    "tool_result",
    EventTypeSpec(
        # Never call_id — provider-generated, differs across runs.
        signature=lambda p: f"tool_res:{p.get('tool_name', '?')}",
        classifier=lambda cp, pp: (
            "different_tool_result" if cp.get("tool_name") != pp.get("tool_name") else None
        ),
        key_fields_equal=lambda p1, p2: p1.get("tool_name") == p2.get("tool_name"),
        structural_types=frozenset({"different_tool_result"}),
    ),
)

register(
    "agent_error",
    EventTypeSpec(
        signature=lambda p: f"agent_error:{p.get('exc_type', '?')}",
        key_fields_equal=lambda p1, p2: p1.get("exc_type") == p2.get("exc_type"),
        # An error appearing only in the perturbed trace is its own
        # structural divergence type (not just a generic missing event).
        missing_in_clean_type="agent_error_introduced",
    ),
)

#: The v1 external vocabulary (Tier 0 + Tier 1 + harness).
EXTERNAL_EVENT_TYPES = frozenset({"llm_call", "tool_request", "tool_result", "agent_error"})
