"""Event-type registry: signatures, classifiers, fallback, upgrade path.

Pre/post-refactor identity for the built-in vocabulary is proven by the
analyzer-golden gate (tests/unit/test_analyzer_goldens.py +
`make analyzer-goldens`); these tests pin the registry's contract directly.
"""

import pytest

from traxr.errors import MalformedEventError, UnknownEventTypeWarning
from traxr.trace import registry
from traxr.trace.registry import (
    BUILTIN_EVENT_TYPES,
    EXTERNAL_EVENT_TYPES,
    STRUCTURAL_DIVERGENCE_TYPES,
    classify_divergence,
    signature_for,
)


class TestBuiltinSignatures:
    """Exact signature strings of the built-in vocabulary."""

    @pytest.mark.parametrize(
        ("event_type", "payload", "expected"),
        [
            ("routing_decision", {"chosen_agent": "researcher"}, "route:researcher"),
            ("routing_decision", {}, "route:?"),
            (
                "tool_invocation",
                {"tool_name": "csv_tool", "operation": "read", "success": True},
                "tool:csv_tool:read:ok",
            ),
            (
                "tool_invocation",
                {"tool_name": "csv_tool", "operation": "read", "success": False},
                "tool:csv_tool:read:fail",
            ),
            ("tool_invocation", {}, "tool:?:?:ok"),  # success defaults to True
            ("memory_write", {"entry_type": "note"}, "mem_write:note"),
            ("memory_read", {"entry_ids": ["m1", "m2"]}, "mem_read:2"),
            ("memory_read", {}, "mem_read:0"),
            ("retrieval_shown", {"item_count": 3}, "retrieval:3"),
            ("retrieval_shown", {}, "retrieval:0"),
            (
                "agent_output",
                {"action": "critique", "is_final_answer": False},
                "output:critique:False",
            ),
            ("agent_output", {"action": "answer", "is_final_answer": True}, "output:answer:True"),
            ("final_answer", {"answer": "$4.2M", "answer_hash": "a1"}, "final_answer"),
        ],
    )
    def test_original_seven(self, event_type, payload, expected):
        assert signature_for(event_type, payload) == expected

    def test_additive_builtins(self):
        assert signature_for("tool_failure", {"tool_name": "csv_tool"}) == "tool_failure:csv_tool"
        assert signature_for("agent_halt", {"reason": "max_steps"}) == "agent_halt:max_steps"
        assert signature_for("agent_halt", {}) == "agent_halt:?"

    def test_builtin_vocabulary_is_registered(self):
        assert BUILTIN_EVENT_TYPES == {
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
        for etype in BUILTIN_EVENT_TYPES:
            assert registry.is_registered(etype)


class TestExternalSignatures:
    """Signatures of the v1 external (capture-layer) vocabulary."""

    def test_llm_call(self):
        assert (
            signature_for(
                "llm_call",
                {"model": "gpt-4o", "finish_reason": "stop", "tool_call_names": []},
            )
            == "llm:gpt-4o:stop:-"
        )
        assert (
            signature_for(
                "llm_call",
                {
                    "model": "gpt-4o",
                    "finish_reason": "tool_calls",
                    "tool_call_names": ["csv_tool", "calculator"],
                },
            )
            == "llm:gpt-4o:tool_calls:csv_tool,calculator"
        )
        assert signature_for("llm_call", {}) == "llm:?:?:-"

    def test_tool_request_ignores_arguments(self):
        # Args stay out of signatures (payload hash only) — lexical noise
        # would saturate d_norm.
        sig_a = signature_for("tool_request", {"tool_name": "csv_tool", "arguments_hash": "h1"})
        sig_b = signature_for("tool_request", {"tool_name": "csv_tool", "arguments_hash": "h2"})
        assert sig_a == sig_b == "tool_req:csv_tool"

    def test_tool_result_ignores_call_id(self):
        # Never call_id — provider-generated, differs across runs.
        sig_a = signature_for("tool_result", {"tool_name": "csv_tool", "call_id": "call_1"})
        sig_b = signature_for("tool_result", {"tool_name": "csv_tool", "call_id": "call_2"})
        assert sig_a == sig_b == "tool_res:csv_tool"

    def test_agent_error(self):
        assert signature_for("agent_error", {"exc_type": "TimeoutError"}) == (
            "agent_error:TimeoutError"
        )

    def test_external_vocabulary_is_registered(self):
        assert EXTERNAL_EVENT_TYPES == {"llm_call", "tool_request", "tool_result", "agent_error"}
        for etype in EXTERNAL_EVENT_TYPES:
            assert registry.is_registered(etype)


class TestBuiltinClassifiers:
    def test_routing_decision(self):
        assert (
            classify_divergence("routing_decision", {"chosen_agent": "a"}, {"chosen_agent": "b"})
            == "different_agent_routed"
        )
        assert (
            classify_divergence(
                "routing_decision",
                {"chosen_agent": "a", "reasoning_hash": "r1"},
                {"chosen_agent": "a", "reasoning_hash": "r2"},
            )
            is None
        )

    def test_tool_invocation_priority_order(self):
        clean = {"tool_name": "csv_tool", "operation": "read", "success": True}
        # tool_name change wins over operation and success changes.
        assert (
            classify_divergence(
                "tool_invocation",
                clean,
                {"tool_name": "pdf_tool", "operation": "write", "success": False},
            )
            == "different_tool"
        )
        assert (
            classify_divergence(
                "tool_invocation",
                clean,
                {"tool_name": "csv_tool", "operation": "write", "success": False},
            )
            == "different_operation"
        )
        assert (
            classify_divergence(
                "tool_invocation",
                clean,
                {"tool_name": "csv_tool", "operation": "read", "success": False},
            )
            == "tool_success_failure_change"
        )
        assert classify_divergence("tool_invocation", clean, dict(clean)) is None

    def test_memory_write_and_agent_output(self):
        assert (
            classify_divergence("memory_write", {"entry_type": "note"}, {"entry_type": "claim"})
            == "different_entry_type"
        )
        assert (
            classify_divergence("agent_output", {"action": "critique"}, {"action": "summarize"})
            == "different_action"
        )

    @pytest.mark.parametrize(
        "etype", ["memory_read", "retrieval_shown", "final_answer", "agent_halt"]
    )
    def test_types_without_classifier_never_structurally_diverge(self, etype):
        assert classify_divergence(etype, {"x": 1}, {"x": 2}) is None

    def test_tool_failure_reuses_different_tool(self):
        assert (
            classify_divergence("tool_failure", {"tool_name": "a"}, {"tool_name": "b"})
            == "different_tool"
        )
        assert classify_divergence("tool_failure", {"tool_name": "a"}, {"tool_name": "a"}) is None


class TestExternalClassifiers:
    def test_llm_call_tool_change_wins_over_finish_reason(self):
        clean = {"model": "gpt-4o", "finish_reason": "tool_calls", "tool_call_names": ["a"]}
        assert (
            classify_divergence(
                "llm_call",
                clean,
                {"model": "gpt-4o", "finish_reason": "stop", "tool_call_names": ["b"]},
            )
            == "different_tool_requested"
        )
        assert (
            classify_divergence(
                "llm_call",
                clean,
                {"model": "gpt-4o", "finish_reason": "length", "tool_call_names": ["a"]},
            )
            == "llm_finish_reason_change"
        )
        # Model/content changes alone are lexical.
        assert (
            classify_divergence(
                "llm_call",
                clean,
                {"model": "gpt-4o-mini", "finish_reason": "tool_calls", "tool_call_names": ["a"]},
            )
            is None
        )

    def test_tool_request_and_result(self):
        assert (
            classify_divergence("tool_request", {"tool_name": "a"}, {"tool_name": "b"})
            == "different_tool_requested"
        )
        assert (
            classify_divergence("tool_result", {"tool_name": "a"}, {"tool_name": "b"})
            == "different_tool_result"
        )
        assert classify_divergence("tool_request", {"tool_name": "a"}, {"tool_name": "a"}) is None

    def test_agent_error_missing_in_clean_override(self):
        assert registry.missing_in_clean_type("agent_error") == "agent_error_introduced"
        # Everything else keeps the generic type.
        assert registry.missing_in_clean_type("routing_decision") == "event_missing_in_clean"
        assert registry.missing_in_clean_type("never_registered") == "event_missing_in_clean"


class TestStructuralDivergenceTypes:
    def test_original_members_preserved(self):
        assert {
            "different_agent_routed",
            "different_tool",
            "different_operation",
            "tool_success_failure_change",
            "event_missing_in_clean",
            "event_missing_in_perturbed",
            "event_type_differs",
            "different_entry_type",
            "different_action",
        } <= STRUCTURAL_DIVERGENCE_TYPES

    def test_external_members_added(self):
        assert {
            "different_tool_requested",
            "llm_finish_reason_change",
            "different_tool_result",
            "agent_error_introduced",
        } <= STRUCTURAL_DIVERGENCE_TYPES

    def test_analyzer_reexports_live_set(self):
        from traxr.metrics.analyzer import STRUCTURAL_DIVERGENCE_TYPES as reexported

        assert reexported is STRUCTURAL_DIVERGENCE_TYPES


class TestUnknownTypeFallback:
    def test_fallback_signature_and_one_time_warning(self, clean_registry):
        with pytest.warns(UnknownEventTypeWarning, match="custom_hook"):
            assert signature_for("custom_hook", {"x": 1}) == "unknown:custom_hook"

        # Second call for the same type: no warning.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert signature_for("custom_hook", {"x": 2}) == "unknown:custom_hook"

        # A different unknown type warns again (one-time is per type).
        with pytest.warns(UnknownEventTypeWarning, match="other_hook"):
            assert signature_for("other_hook", {}) == "unknown:other_hook"

    def test_unknown_type_payload_never_contributes_structure(self, clean_registry):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnknownEventTypeWarning)
            assert signature_for("custom_hook", {"x": 1}) == signature_for(
                "custom_hook", {"x": 999}
            )


class TestRegisterSignatureUpgrade:
    def test_upgrade_replaces_fallback_and_silences_warning(self, clean_registry):
        import warnings

        registry.register_signature("custom_hook", lambda p: f"hook:{p.get('kind', '?')}")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert signature_for("custom_hook", {"kind": "fetch"}) == "hook:fetch"

    def test_upgrade_with_classifier_extends_structural_types(self, clean_registry):
        registry.register_signature(
            "custom_hook",
            lambda p: f"hook:{p.get('kind', '?')}",
            classifier=lambda cp, pp: (
                "different_hook_kind" if cp.get("kind") != pp.get("kind") else None
            ),
            structural_types={"different_hook_kind"},
        )
        assert "different_hook_kind" in STRUCTURAL_DIVERGENCE_TYPES
        assert (
            classify_divergence("custom_hook", {"kind": "a"}, {"kind": "b"})
            == "different_hook_kind"
        )
        assert classify_divergence("custom_hook", {"kind": "a"}, {"kind": "a"}) is None

    def test_upgrade_with_key_fields(self, clean_registry):
        registry.register_signature(
            "custom_hook",
            lambda p: "hook",
            key_fields_equal=lambda p1, p2: p1.get("kind") == p2.get("kind"),
        )
        assert registry.key_fields_equal("custom_hook", {"kind": "a", "noise": 1}, {"kind": "a"})
        assert not registry.key_fields_equal("custom_hook", {"kind": "a"}, {"kind": "b"})

    def test_non_callable_signature_rejected(self, clean_registry):
        with pytest.raises(MalformedEventError, match="callable"):
            registry.register_signature("custom_hook", "hook:{kind}")

    def test_empty_event_type_rejected(self, clean_registry):
        with pytest.raises(MalformedEventError, match="non-empty string"):
            registry.register_signature("", lambda p: "x")

    def test_key_fields_equal_unknown_type_is_false(self, clean_registry):
        assert not registry.key_fields_equal("never_registered", {"a": 1}, {"a": 1})
