"""Manifestation classification: every fine category, thresholds, paper groups.

Test category 4: each fine category + the 4 paper groups via constructed
PairMetrics; threshold boundaries (edn 0.5 / 0.2 / ~0, total_changes 3).
"""

import math

import pytest

from traxr.metrics.manifest import (
    BEHAVIORAL_DETOURS_GROUP,
    CATASTROPHIC_FAILURE,
    COMBINED_DISRUPTION_GROUP,
    EARLY_TERMINATION,
    FINE_CATEGORIES,
    LOOP_OR_EXTENDED_EXECUTION,
    NO_OBSERVABLE_EFFECT,
    NO_OBSERVABLE_EFFECT_GROUP,
    OUTCOME_CHANGE_UNCATEGORIZED,
    PAPER_GROUPS,
    SILENT_CORRUPTION_GROUP,
    SILENT_SEMANTIC_CORRUPTION,
    STRATEGY_REROUTE,
    STRUCTURAL_DIVERGENCE_RECOVERED,
    STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE,
    PairMetrics,
    classify_manifestation,
    to_paper_group,
)


class TestFineCategories:
    """One constructed PairMetrics per fine category."""

    def test_catastrophic_failure_null_answer_high_divergence(self):
        m = PairMetrics(
            answer_changed=True, perturbed_answer_is_null=True, edit_distance_normalized=0.7
        )
        assert classify_manifestation(m) == CATASTROPHIC_FAILURE

    def test_catastrophic_failure_null_answer_early_termination(self):
        m = PairMetrics(answer_changed=True, perturbed_answer_is_null=True, early_termination=True)
        assert classify_manifestation(m) == CATASTROPHIC_FAILURE

    def test_early_termination(self):
        m = PairMetrics(answer_changed=False, early_termination=True, total_changes=1)
        assert classify_manifestation(m) == EARLY_TERMINATION

    def test_loop_or_extended_execution_via_changes(self):
        m = PairMetrics(extended_execution=True, total_changes=3)
        assert classify_manifestation(m) == LOOP_OR_EXTENDED_EXECUTION

    def test_loop_or_extended_execution_via_edn(self):
        m = PairMetrics(extended_execution=True, total_changes=1, edit_distance_normalized=0.2)
        assert classify_manifestation(m) == LOOP_OR_EXTENDED_EXECUTION

    def test_silent_semantic_corruption(self):
        m = PairMetrics(answer_changed=True, total_changes=0, edit_distance_normalized=0.0)
        assert classify_manifestation(m) == SILENT_SEMANTIC_CORRUPTION

    def test_strategy_reroute(self):
        m = PairMetrics(reroutes=2, total_changes=2, edit_distance_normalized=0.1)
        assert classify_manifestation(m) == STRATEGY_REROUTE

    def test_structural_divergence_with_outcome_change(self):
        m = PairMetrics(answer_changed=True, total_changes=1, edit_distance_normalized=0.1)
        assert classify_manifestation(m) == STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE

    def test_structural_divergence_recovered(self):
        m = PairMetrics(answer_changed=False, total_changes=1, edit_distance_normalized=0.1)
        assert classify_manifestation(m) == STRUCTURAL_DIVERGENCE_RECOVERED

    def test_no_observable_effect(self):
        m = PairMetrics(edit_distance_normalized=0.0)
        assert classify_manifestation(m) == NO_OBSERVABLE_EFFECT

    def test_outcome_change_uncategorized_fallback(self):
        # Answer flipped with no structural signal at all (no edit distance).
        m = PairMetrics(answer_changed=True, total_changes=0, edit_distance_normalized=None)
        assert classify_manifestation(m) == OUTCOME_CHANGE_UNCATEGORIZED


class TestThresholdBoundaries:
    def test_catastrophic_edn_boundary_at_0_5(self):
        at = PairMetrics(
            answer_changed=True, perturbed_answer_is_null=True, edit_distance_normalized=0.5
        )
        below = PairMetrics(
            answer_changed=True, perturbed_answer_is_null=True, edit_distance_normalized=0.4999
        )
        assert classify_manifestation(at) == CATASTROPHIC_FAILURE
        assert classify_manifestation(below) != CATASTROPHIC_FAILURE

    def test_early_termination_beats_null_answer_below_threshold(self):
        # Null answer + early termination is catastrophic regardless of edn.
        m = PairMetrics(
            answer_changed=True,
            perturbed_answer_is_null=True,
            early_termination=True,
            edit_distance_normalized=0.1,
        )
        assert classify_manifestation(m) == CATASTROPHIC_FAILURE

    def test_loop_edn_boundary_at_0_2(self):
        at = PairMetrics(extended_execution=True, edit_distance_normalized=0.2)
        below = PairMetrics(extended_execution=True, edit_distance_normalized=0.1999)
        assert classify_manifestation(at) == LOOP_OR_EXTENDED_EXECUTION
        assert classify_manifestation(below) != LOOP_OR_EXTENDED_EXECUTION

    def test_loop_total_changes_boundary_at_3(self):
        at = PairMetrics(extended_execution=True, total_changes=3)
        below = PairMetrics(extended_execution=True, total_changes=2)
        assert classify_manifestation(at) == LOOP_OR_EXTENDED_EXECUTION
        assert classify_manifestation(below) != LOOP_OR_EXTENDED_EXECUTION

    def test_silent_corruption_requires_near_zero_edn(self):
        tiny = PairMetrics(answer_changed=True, total_changes=0, edit_distance_normalized=1e-10)
        nonzero = PairMetrics(answer_changed=True, total_changes=0, edit_distance_normalized=0.01)
        assert classify_manifestation(tiny) == SILENT_SEMANTIC_CORRUPTION
        assert classify_manifestation(nonzero) == STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE

    def test_silent_corruption_requires_zero_changes(self):
        m = PairMetrics(answer_changed=True, total_changes=1, edit_distance_normalized=0.0)
        assert classify_manifestation(m) != SILENT_SEMANTIC_CORRUPTION

    def test_nan_edn_means_no_edit_distance(self):
        # Source coerced via pandas: NaN and None both mean "unavailable".
        m = PairMetrics(answer_changed=True, total_changes=0, edit_distance_normalized=float("nan"))
        assert classify_manifestation(m) == OUTCOME_CHANGE_UNCATEGORIZED
        assert math.isnan(float("nan"))  # sanity: NaN never equals 0

    def test_priority_early_termination_before_reroute(self):
        m = PairMetrics(early_termination=True, reroutes=2, total_changes=3)
        assert classify_manifestation(m) == EARLY_TERMINATION

    def test_priority_reroute_before_structural(self):
        m = PairMetrics(reroutes=1, total_changes=1, edit_distance_normalized=0.3)
        assert classify_manifestation(m) == STRATEGY_REROUTE


class TestPaperGroups:
    @pytest.mark.parametrize(
        ("fine", "group"),
        [
            (CATASTROPHIC_FAILURE, COMBINED_DISRUPTION_GROUP),
            (EARLY_TERMINATION, COMBINED_DISRUPTION_GROUP),
            (LOOP_OR_EXTENDED_EXECUTION, BEHAVIORAL_DETOURS_GROUP),
            (STRATEGY_REROUTE, BEHAVIORAL_DETOURS_GROUP),
            (STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE, BEHAVIORAL_DETOURS_GROUP),
            (STRUCTURAL_DIVERGENCE_RECOVERED, BEHAVIORAL_DETOURS_GROUP),
            (SILENT_SEMANTIC_CORRUPTION, SILENT_CORRUPTION_GROUP),
            (OUTCOME_CHANGE_UNCATEGORIZED, SILENT_CORRUPTION_GROUP),
            (NO_OBSERVABLE_EFFECT, NO_OBSERVABLE_EFFECT_GROUP),
        ],
    )
    def test_rollup(self, fine, group):
        assert to_paper_group(fine) == group

    def test_every_fine_category_maps_to_a_paper_group(self):
        for fine in FINE_CATEGORIES:
            assert to_paper_group(fine) in PAPER_GROUPS

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown manifestation category"):
            to_paper_group("not_a_category")
