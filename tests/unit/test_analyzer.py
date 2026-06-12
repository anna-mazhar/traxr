"""Divergence analyzer: edit distance, t*, control flow, external vocabulary.

Test category 3 (edit distance + t*, hand-crafted sequences with known
distances) and the analyzer side of category 6 (external event types produce
the right divergence types). Hand-crafted expectations follow the source
oracle (``verify_edit_distance.py``, plain Wagner-Fischer).
"""

from typing import Any

import pytest

from traxr.metrics.analyzer import (
    STRUCTURAL_DIVERGENCE_TYPES,
    ControlFlowChanges,
    TraceDivergenceAnalyzer,
)
from traxr.trace.collector import TraceCollector


def make_trace(label: str, events: list[tuple[int, str, dict[str, Any]]]) -> TraceCollector:
    """Build a TraceCollector from (step_num, event_type, payload) triples."""
    collector = TraceCollector(run_label=label)
    for step_num, event_type, payload in events:
        collector.emit(event_type, step_num=step_num, agent_name="agent", payload=payload)
    return collector


def analyze(
    clean_events: list[tuple[int, str, dict[str, Any]]],
    perturbed_events: list[tuple[int, str, dict[str, Any]]],
):
    return TraceDivergenceAnalyzer().analyze(
        make_trace("baseline", clean_events),
        make_trace("perturbed", perturbed_events),
        task_id="t1",
    )


class TestEditDistance:
    """Hand-crafted signature sequences with known Wagner-Fischer distances."""

    analyzer = TraceDivergenceAnalyzer()

    def test_identical_sequences(self):
        result = self.analyzer._compute_edit_distance(["a", "b", "c"], ["a", "b", "c"])
        assert result.distance == 0
        assert result.normalized == 0.0
        assert (result.substitutions, result.insertions, result.deletions) == (0, 0, 0)

    def test_both_empty(self):
        result = self.analyzer._compute_edit_distance([], [])
        assert result.distance == 0
        assert result.normalized == 0.0
        assert (result.baseline_length, result.perturbed_length) == (0, 0)

    def test_empty_baseline_is_all_insertions(self):
        result = self.analyzer._compute_edit_distance([], ["a", "b", "c"])
        assert result.distance == 3
        assert result.normalized == 1.0
        assert (result.substitutions, result.insertions, result.deletions) == (0, 3, 0)

    def test_empty_perturbed_is_all_deletions(self):
        result = self.analyzer._compute_edit_distance(["a", "b"], [])
        assert result.distance == 2
        assert result.normalized == 1.0
        assert (result.substitutions, result.insertions, result.deletions) == (0, 0, 2)

    def test_single_substitution(self):
        result = self.analyzer._compute_edit_distance(["a", "b", "c"], ["a", "x", "c"])
        assert result.distance == 1
        assert result.normalized == pytest.approx(1 / 3)
        assert (result.substitutions, result.insertions, result.deletions) == (1, 0, 0)

    def test_single_insertion(self):
        result = self.analyzer._compute_edit_distance(["a", "b"], ["a", "b", "c"])
        assert result.distance == 1
        assert result.normalized == pytest.approx(1 / 3)
        assert result.insertions == 1

    def test_single_deletion(self):
        result = self.analyzer._compute_edit_distance(["a", "b", "c"], ["a", "c"])
        assert result.distance == 1
        assert result.normalized == pytest.approx(1 / 3)
        assert result.deletions == 1

    def test_kitten_sitting_distance_three(self):
        # The classic Levenshtein example: 2 substitutions + 1 insertion.
        result = self.analyzer._compute_edit_distance(list("kitten"), list("sitting"))
        assert result.distance == 3
        assert result.normalized == pytest.approx(3 / 7)
        assert result.substitutions + result.insertions + result.deletions == 3

    def test_fully_disjoint_is_saturated(self):
        result = self.analyzer._compute_edit_distance(["a", "b"], ["x", "y", "z"])
        assert result.distance == 3
        assert result.normalized == 1.0

    def test_decomposition_sums_to_distance(self):
        result = self.analyzer._compute_edit_distance(
            ["route:a", "tool:t:read:ok", "final_answer"],
            ["route:b", "tool:t:read:fail", "mem_write:note", "final_answer"],
        )
        assert result.substitutions + result.insertions + result.deletions == result.distance


class TestFirstDivergence:
    """t* — first structural divergence step, type, and normalized position."""

    def test_identical_traces_have_no_divergence(self):
        events = [
            (1, "routing_decision", {"chosen_agent": "researcher"}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
            (3, "final_answer", {"answer": "42", "answer_hash": "h"}),
        ]
        report = analyze(events, list(events))
        assert report.edit_distance is not None
        assert report.edit_distance.distance == 0
        assert report.first_divergence_step is None
        assert report.first_divergence_type is None
        assert report.divergence_normalized_position is None
        assert report.answer_changed is False

    def test_known_tstar_step_and_type(self):
        clean = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
            (3, "routing_decision", {"chosen_agent": "analyst"}),
            (4, "final_answer", {"answer": "42", "answer_hash": "h"}),
        ]
        perturbed = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
            (3, "routing_decision", {"chosen_agent": "critic"}),  # <- t* here
            (4, "final_answer", {"answer": "42", "answer_hash": "h"}),
        ]
        report = analyze(clean, perturbed)
        assert report.first_divergence_step == 3
        assert report.first_divergence_type == "different_agent_routed"
        assert report.divergence_normalized_position == pytest.approx(3 / 4)

    def test_lexical_difference_is_not_divergence(self):
        clean = [(1, "routing_decision", {"chosen_agent": "planner", "reasoning_hash": "r1"})]
        perturbed = [(1, "routing_decision", {"chosen_agent": "planner", "reasoning_hash": "r2"})]
        report = analyze(clean, perturbed)
        assert report.first_divergence_step is None
        assert report.first_divergence_type is None

    def test_missing_event_in_perturbed(self):
        clean = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
        ]
        report = analyze(clean, clean[:1])
        assert report.first_divergence_step == 2
        assert report.first_divergence_type == "event_missing_in_perturbed"

    def test_missing_event_in_clean(self):
        perturbed = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "memory_write", {"entry_type": "note"}),
        ]
        report = analyze(perturbed[:1], perturbed)
        assert report.first_divergence_step == 2
        assert report.first_divergence_type == "event_missing_in_clean"


class TestExternalVocabulary:
    """External event types get full t* typing through the registry."""

    def test_llm_call_tool_change_is_different_tool_requested(self):
        clean = [
            (
                1,
                "llm_call",
                {"model": "m", "finish_reason": "tool_calls", "tool_call_names": ["search"]},
            )
        ]
        perturbed = [
            (
                1,
                "llm_call",
                {"model": "m", "finish_reason": "tool_calls", "tool_call_names": ["calc"]},
            )
        ]
        report = analyze(clean, perturbed)
        assert report.first_divergence_type == "different_tool_requested"
        assert "different_tool_requested" in STRUCTURAL_DIVERGENCE_TYPES

    def test_llm_finish_reason_change(self):
        clean = [
            (1, "llm_call", {"model": "m", "finish_reason": "tool_calls", "tool_call_names": []})
        ]
        perturbed = [
            (1, "llm_call", {"model": "m", "finish_reason": "stop", "tool_call_names": []})
        ]
        report = analyze(clean, perturbed)
        assert report.first_divergence_type == "llm_finish_reason_change"
        assert "llm_finish_reason_change" in STRUCTURAL_DIVERGENCE_TYPES

    def test_tool_result_name_change(self):
        clean = [(1, "tool_result", {"tool_name": "search"})]
        perturbed = [(1, "tool_result", {"tool_name": "calc"})]
        report = analyze(clean, perturbed)
        assert report.first_divergence_type == "different_tool_result"
        assert "different_tool_result" in STRUCTURAL_DIVERGENCE_TYPES

    def test_agent_error_introduced(self):
        clean = [(1, "llm_call", {"model": "m", "finish_reason": "stop", "tool_call_names": []})]
        perturbed = clean + [(2, "agent_error", {"exc_type": "TimeoutError"})]
        report = analyze(clean, perturbed)
        assert report.first_divergence_step == 2
        assert report.first_divergence_type == "agent_error_introduced"
        assert "agent_error_introduced" in STRUCTURAL_DIVERGENCE_TYPES

    def test_external_traces_get_d_norm(self):
        clean = [
            (
                1,
                "llm_call",
                {"model": "m", "finish_reason": "tool_calls", "tool_call_names": ["search"]},
            ),
            (2, "tool_request", {"tool_name": "search"}),
            (3, "tool_result", {"tool_name": "search"}),
        ]
        perturbed = [
            (
                1,
                "llm_call",
                {"model": "m", "finish_reason": "tool_calls", "tool_call_names": ["calc"]},
            ),
            (2, "tool_request", {"tool_name": "calc"}),
            (3, "tool_result", {"tool_name": "calc"}),
        ]
        report = analyze(clean, perturbed)
        assert report.edit_distance is not None
        assert report.edit_distance.distance == 3
        assert report.edit_distance.normalized == 1.0


class TestControlFlowChanges:
    def test_reroutes_counted(self):
        clean = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "routing_decision", {"chosen_agent": "analyst"}),
        ]
        perturbed = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "routing_decision", {"chosen_agent": "critic"}),
        ]
        report = analyze(clean, perturbed)
        assert report.control_flow_changes is not None
        assert report.control_flow_changes.reroutes == 1

    def test_routing_cycles_and_extended_execution(self):
        clean = [(1, "routing_decision", {"chosen_agent": "planner"})]
        perturbed = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "routing_decision", {"chosen_agent": "planner"}),
            (3, "routing_decision", {"chosen_agent": "planner"}),
        ]
        cf = analyze(clean, perturbed).control_flow_changes
        assert cf is not None
        assert cf.extra_routing_cycles == 2
        assert cf.extended_execution is True
        assert cf.early_termination is False

    def test_missing_cycles_and_early_termination(self):
        clean = [
            (1, "routing_decision", {"chosen_agent": "planner"}),
            (2, "routing_decision", {"chosen_agent": "planner"}),
        ]
        perturbed = [(1, "routing_decision", {"chosen_agent": "planner"})]
        cf = analyze(clean, perturbed).control_flow_changes
        assert cf is not None
        assert cf.missing_routing_cycles == 1
        assert cf.early_termination is True

    def test_tool_failures_introduced_and_avoided(self):
        clean = [
            (1, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": False}),
        ]
        perturbed = [
            (1, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": False}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
        ]
        cf = analyze(clean, perturbed).control_flow_changes
        assert cf is not None
        assert cf.tool_failures_introduced == 1
        assert cf.tool_failures_avoided == 1
        assert cf.tool_state_changes == 2

    def test_tool_calls_added_and_removed(self):
        clean = [(1, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True})]
        perturbed = clean + [
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
        ]
        cf = analyze(clean, perturbed).control_flow_changes
        assert cf is not None
        assert cf.tool_calls_added == 1
        cf_rev = analyze(perturbed, clean).control_flow_changes
        assert cf_rev is not None
        assert cf_rev.tool_calls_removed == 1

    def test_total_changes_composite(self):
        cf = ControlFlowChanges(
            reroutes=1,
            extra_routing_cycles=2,
            tool_failures_introduced=1,
            early_termination=True,
        )
        assert cf.total_changes == 5
        assert cf.to_dict()["total_changes"] == 5


class TestOutcomeAndReport:
    def test_answer_changed(self):
        clean = [(1, "final_answer", {"answer": "42", "answer_hash": "h1"})]
        perturbed = [(1, "final_answer", {"answer": "41", "answer_hash": "h2"})]
        report = analyze(clean, perturbed)
        assert report.baseline_answer == "42"
        assert report.perturbed_answer == "41"
        assert report.answer_changed is True

    def test_missing_final_answer_is_none(self):
        report = analyze([(1, "routing_decision", {"chosen_agent": "a"})], [])
        assert report.baseline_answer is None
        assert report.perturbed_answer is None
        assert report.answer_changed is False

    def test_perturbation_type_is_run_label(self):
        report = analyze([], [])
        assert report.perturbation_type == "perturbed"
        assert report.task_id == "t1"

    def test_summary_and_detail_dicts(self):
        clean = [(1, "routing_decision", {"chosen_agent": "a"})]
        perturbed = [(1, "routing_decision", {"chosen_agent": "b"})]
        report = analyze(clean, perturbed)
        summary = report.to_summary_dict()
        assert "aligned_pairs" not in summary
        assert summary["edit_distance"]["distance"] == 1
        detail = report.to_detail_dict()
        assert len(detail["aligned_pairs"]) == 1
        assert detail["aligned_pairs"][0]["divergence_type"] == "different_agent_routed"
        assert detail["aligned_pairs"][0]["is_match"] is False

    def test_trace_statistics(self):
        clean = [
            (1, "routing_decision", {"chosen_agent": "a"}),
            (2, "tool_invocation", {"tool_name": "csv", "operation": "read", "success": True}),
        ]
        report = analyze(clean, [])
        assert report.baseline_steps == 2
        assert report.perturbed_steps == 0
        assert report.baseline_events == 2
        assert report.perturbed_events == 0
        assert report.baseline_routing_turns == 1
        assert report.baseline_tool_calls == 1
