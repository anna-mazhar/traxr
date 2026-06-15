"""TraceEvent: open event_type validation, hashing, semantic equality."""

import pytest

from traxr.errors import MalformedEventError
from traxr.trace.events import TraceEvent


def make_event(event_type="routing_decision", payload=None, **kwargs):
    payload = payload if payload is not None else {"chosen_agent": "researcher"}
    defaults = {
        "event_type": event_type,
        "sequence_index": 0,
        "step_num": 1,
        "agent_name": "router",
        "payload": payload,
        "content_hash": TraceEvent.compute_content_hash(payload),
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)


class TestOpenEventTypeValidation:
    def test_builtin_type_accepted(self):
        assert make_event("tool_invocation", {"tool_name": "csv_tool"}).event_type == (
            "tool_invocation"
        )

    def test_custom_type_accepted(self):
        # The closed 7-type whitelist is gone: any non-empty string works.
        event = make_event("my_custom_hook", {"anything": 1})
        assert event.event_type == "my_custom_hook"

    def test_empty_string_rejected(self):
        with pytest.raises(MalformedEventError, match="non-empty string"):
            make_event("")

    @pytest.mark.parametrize("bad", [None, 7, b"routing_decision", ["routing_decision"]])
    def test_non_string_rejected(self, bad):
        with pytest.raises(MalformedEventError, match="non-empty string"):
            make_event(bad)


class TestContentHash:
    def test_deterministic(self):
        payload = {"a": 1, "b": [1, 2]}
        assert TraceEvent.compute_content_hash(payload) == TraceEvent.compute_content_hash(
            {"a": 1, "b": [1, 2]}
        )

    def test_key_order_insensitive(self):
        assert TraceEvent.compute_content_hash({"a": 1, "b": 2}) == (
            TraceEvent.compute_content_hash({"b": 2, "a": 1})
        )

    def test_different_payloads_differ(self):
        assert TraceEvent.compute_content_hash({"a": 1}) != TraceEvent.compute_content_hash(
            {"a": 2}
        )


class TestSemanticEquals:
    def test_different_types_never_equal(self):
        a = make_event("routing_decision")
        b = make_event("agent_output", {"action": "critique"})
        assert not a.semantic_equals(b)

    def test_identical_events_are_equal(self):
        a = make_event()
        b = make_event()
        assert a.semantic_equals(b)

    def test_hash_collision_does_not_force_equality(self):
        """M2: equal content_hash must NOT short-circuit to equal.

        ``content_hash`` is a truncated 64-bit digest, so a collision is
        possible; equality must still be decided by the registry key fields.
        Here we force the collision and assert the differing key fields win.
        """
        a = make_event(payload={"chosen_agent": "researcher"})
        b = make_event(payload={"chosen_agent": "critic"}, content_hash=a.content_hash)
        assert a.content_hash == b.content_hash  # forced collision
        assert not a.semantic_equals(b)  # key fields differ → not equal

    def test_routing_decision_key_fields(self):
        a = make_event(payload={"chosen_agent": "researcher", "reasoning_hash": "r1"})
        b = make_event(payload={"chosen_agent": "researcher", "reasoning_hash": "r2"})
        c = make_event(payload={"chosen_agent": "critic", "reasoning_hash": "r1"})
        assert a.semantic_equals(b)  # different reasoning = lexical only
        assert not a.semantic_equals(c)

    def test_tool_invocation_key_fields(self):
        base = {
            "tool_name": "csv_tool",
            "operation": "read",
            "arguments": {"path": "a.csv"},
            "output_hash": "h1",
        }
        a = make_event("tool_invocation", {**base, "output_preview": "x"})
        b = make_event("tool_invocation", {**base, "output_preview": "y"})
        c = make_event("tool_invocation", {**base, "output_hash": "h2"})
        assert a.semantic_equals(b)
        assert not a.semantic_equals(c)

    def test_memory_read_key_fields_set_semantics(self):
        a = make_event("memory_read", {"entry_ids": ["m1", "m2"], "entry_types": ["note"]})
        b = make_event("memory_read", {"entry_ids": ["m2", "m1"], "entry_types": ["other"]})
        assert a.semantic_equals(b)

    def test_final_answer_key_fields(self):
        a = make_event("final_answer", {"answer": "$4.2M", "answer_hash": "h"})
        b = make_event("final_answer", {"answer": "4.2 million", "answer_hash": "h"})
        c = make_event("final_answer", {"answer": "$4.2M", "answer_hash": "h2"})
        assert a.semantic_equals(b)
        assert not a.semantic_equals(c)

    def test_external_llm_call_key_fields(self):
        a = make_event(
            "llm_call",
            {
                "model": "gpt-4o",
                "finish_reason": "stop",
                "tool_call_names": [],
                "prompt_hash": "p1",
            },
        )
        b = make_event(
            "llm_call",
            {
                "model": "gpt-4o",
                "finish_reason": "stop",
                "tool_call_names": [],
                "prompt_hash": "p2",
            },
        )
        c = make_event(
            "llm_call",
            {"model": "gpt-4o", "finish_reason": "tool_calls", "tool_call_names": ["csv_tool"]},
        )
        assert a.semantic_equals(b)
        assert not a.semantic_equals(c)

    def test_unknown_type_with_different_hash_not_equal(self):
        a = make_event("my_custom_hook", {"x": 1})
        b = make_event("my_custom_hook", {"x": 2})
        assert not a.semantic_equals(b)


def test_to_dict_round_trips():
    event = make_event()
    data = event.to_dict()
    assert TraceEvent(**data) == event
