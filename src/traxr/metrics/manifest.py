"""Manifestation taxonomy: how a perturbation's effect manifested.

Classification takes a typed :class:`PairMetrics` dataclass (no pandas
dependency). The priority-ordered rules and thresholds include NaN handling
for the normalized edit distance.

Fine categories (priority-ordered rules + fallback):

==========================================  ==========================================
fine category                               condition
==========================================  ==========================================
``catastrophic_failure``                    null answer + (early termination or d_norm >= 0.5)
``early_termination``                       perturbed finished before baseline
``loop_or_extended_execution``              extended + (total_changes >= 3 or d_norm >= 0.2)
``silent_semantic_corruption``              answer changed, d_norm ~ 0, no control-flow changes
``strategy_reroute``                        reroutes > 0
``structural_divergence_with_outcome_change``  divergence + answer changed
``structural_divergence_recovered``         divergence but same answer
``no_observable_effect``                    nothing changed
``outcome_change_uncategorized``            fallback
==========================================  ==========================================

``to_paper_group`` rolls the fine categories up to the paper's 4 groups
(silent corruption / behavioral detours / combined disruption / no observable
effect). The source maps 6 of the 9 fine categories explicitly
(``paper_stats.py``); the remaining 3 are assigned here by definition:
structural divergences (recovered or with outcome change) are behavioral
detours from the baseline path, and an uncategorized outcome change — answer
flipped with no structural signal — is silent corruption.
"""

import math
from dataclasses import dataclass

# Fine categories
CATASTROPHIC_FAILURE = "catastrophic_failure"
EARLY_TERMINATION = "early_termination"
LOOP_OR_EXTENDED_EXECUTION = "loop_or_extended_execution"
SILENT_SEMANTIC_CORRUPTION = "silent_semantic_corruption"
STRATEGY_REROUTE = "strategy_reroute"
STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE = "structural_divergence_with_outcome_change"
STRUCTURAL_DIVERGENCE_RECOVERED = "structural_divergence_recovered"
NO_OBSERVABLE_EFFECT = "no_observable_effect"
OUTCOME_CHANGE_UNCATEGORIZED = "outcome_change_uncategorized"

FINE_CATEGORIES: tuple[str, ...] = (
    CATASTROPHIC_FAILURE,
    EARLY_TERMINATION,
    LOOP_OR_EXTENDED_EXECUTION,
    SILENT_SEMANTIC_CORRUPTION,
    STRATEGY_REROUTE,
    STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE,
    STRUCTURAL_DIVERGENCE_RECOVERED,
    NO_OBSERVABLE_EFFECT,
    OUTCOME_CHANGE_UNCATEGORIZED,
)

# Paper groups
SILENT_CORRUPTION_GROUP = "silent_corruption"
BEHAVIORAL_DETOURS_GROUP = "behavioral_detours"
COMBINED_DISRUPTION_GROUP = "combined_disruption"
NO_OBSERVABLE_EFFECT_GROUP = "no_observable_effect"

PAPER_GROUPS: tuple[str, ...] = (
    SILENT_CORRUPTION_GROUP,
    BEHAVIORAL_DETOURS_GROUP,
    COMBINED_DISRUPTION_GROUP,
    NO_OBSERVABLE_EFFECT_GROUP,
)

_PAPER_GROUP_BY_CATEGORY: dict[str, str] = {
    CATASTROPHIC_FAILURE: COMBINED_DISRUPTION_GROUP,
    EARLY_TERMINATION: COMBINED_DISRUPTION_GROUP,
    LOOP_OR_EXTENDED_EXECUTION: BEHAVIORAL_DETOURS_GROUP,
    STRATEGY_REROUTE: BEHAVIORAL_DETOURS_GROUP,
    STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE: BEHAVIORAL_DETOURS_GROUP,
    STRUCTURAL_DIVERGENCE_RECOVERED: BEHAVIORAL_DETOURS_GROUP,
    SILENT_SEMANTIC_CORRUPTION: SILENT_CORRUPTION_GROUP,
    OUTCOME_CHANGE_UNCATEGORIZED: SILENT_CORRUPTION_GROUP,
    NO_OBSERVABLE_EFFECT: NO_OBSERVABLE_EFFECT_GROUP,
}

#: One-line, neutral descriptions of each fine category — the single source for
#: the report legend (:meth:`traxr.results.ExperimentResults.to_report`). These
#: describe what the category *means*, not whether it is good or bad.
MANIFESTATION_DESCRIPTIONS: dict[str, str] = {
    CATASTROPHIC_FAILURE: "Null answer alongside an early stop or large trace divergence.",
    EARLY_TERMINATION: "The perturbed run finished before the baseline did.",
    LOOP_OR_EXTENDED_EXECUTION: "The perturbed run ran longer, with extra control-flow changes.",
    SILENT_SEMANTIC_CORRUPTION: "The answer changed while the trace looked unchanged.",
    STRATEGY_REROUTE: "The agent routed to a different agent at some aligned step.",
    STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE: "The trace diverged and the answer changed.",
    STRUCTURAL_DIVERGENCE_RECOVERED: "The trace diverged but the answer was unchanged.",
    NO_OBSERVABLE_EFFECT: "No change in either the trace or the answer.",
    OUTCOME_CHANGE_UNCATEGORIZED: "The answer changed with no structural signal to explain it.",
}


@dataclass(frozen=True)
class PairMetrics:
    """The per-pair signals manifestation classification needs.

    A typed replacement for the flattened ``pd.Series`` row the source used;
    field names match the source's row keys.
    """

    answer_changed: bool = False
    perturbed_answer_is_null: bool = False
    early_termination: bool = False
    extended_execution: bool = False
    reroutes: int = 0
    total_changes: int = 0
    edit_distance_normalized: float | None = None


def classify_manifestation(metrics: PairMetrics) -> str:
    """Simple rule-based manifestation taxonomy.

    Priority order matters (preserved exactly from the source).
    """
    answer_changed = bool(metrics.answer_changed)
    perturbed_null = bool(metrics.perturbed_answer_is_null)
    early_term = bool(metrics.early_termination)
    extended = bool(metrics.extended_execution)
    reroutes = metrics.reroutes or 0
    total_changes = metrics.total_changes or 0
    edn = metrics.edit_distance_normalized
    # Source coerced via pandas: None/NaN both mean "no edit distance available".
    edn = float(edn) if edn is not None and not math.isnan(edn) else None

    # catastrophic / failure-like first
    if perturbed_null and (early_term or (edn is not None and edn >= 0.5)):
        return CATASTROPHIC_FAILURE

    if early_term:
        return EARLY_TERMINATION

    if extended and (total_changes >= 3 or (edn is not None and edn >= 0.2)):
        return LOOP_OR_EXTENDED_EXECUTION

    # silent corruption: outcome changes with no structural signal
    if answer_changed and (edn == 0 or (edn is not None and edn <= 1e-9)) and total_changes == 0:
        return SILENT_SEMANTIC_CORRUPTION

    # strategy / routing shift
    if reroutes > 0:
        return STRATEGY_REROUTE

    # structural divergence without answer flip
    if (edn is not None and edn > 0) or total_changes > 0:
        if answer_changed:
            return STRUCTURAL_DIVERGENCE_WITH_OUTCOME_CHANGE
        return STRUCTURAL_DIVERGENCE_RECOVERED

    # no observable effect
    if not answer_changed:
        return NO_OBSERVABLE_EFFECT

    # fallback
    return OUTCOME_CHANGE_UNCATEGORIZED


def to_paper_group(fine_category: str) -> str:
    """Roll a fine category up to one of the paper's 4 groups."""
    try:
        return _PAPER_GROUP_BY_CATEGORY[fine_category]
    except KeyError:
        raise ValueError(
            f"Unknown manifestation category {fine_category!r}; expected one of {FINE_CATEGORIES}"
        ) from None
