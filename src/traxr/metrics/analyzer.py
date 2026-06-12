"""Trace divergence analyzer for comparing clean vs perturbed execution traces.

Per-event-type signatures and divergence classifiers are registry-driven
(:mod:`traxr.trace.registry`), so external and custom event vocabularies
get full ``d_norm`` *and* ``t*`` typing. The analyzer-golden gate
(``make analyzer-goldens``) pins built-in behavior exactly.

Focuses on STRUCTURAL divergence: control flow and execution state changes
that indicate meaningful behavioral differences, not just textual variation.

Key metrics:
- Edit Distance: Minimum edits (substitutions, insertions, deletions) to transform
  one trace into another, normalized to [0,1] (``d_norm``). Robust to cascading
  mismatches.
- First Divergence: Where structural divergence first occurs (``t*``).
- Trace Statistics: Step counts, routing turns, tool calls for pattern comparison.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

from traxr.trace import registry
from traxr.trace.collector import TraceCollector
from traxr.trace.events import TraceEvent

# Structural divergence types: control flow or execution state changes.
# Re-exported from the registry (the live set, updated as types register).
STRUCTURAL_DIVERGENCE_TYPES = registry.STRUCTURAL_DIVERGENCE_TYPES


@dataclass
class EditDistanceResult:
    """Result of computing edit distance between two traces."""

    distance: int  # Raw edit distance (number of edits)
    normalized: float  # Normalized to [0, 1] where 0 = identical, 1 = completely different
    substitutions: int  # Number of substitution edits
    insertions: int  # Number of insertion edits (event in perturbed but not baseline)
    deletions: int  # Number of deletion edits (event in baseline but not perturbed)
    baseline_length: int
    perturbed_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "normalized": round(self.normalized, 4),
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "baseline_length": self.baseline_length,
            "perturbed_length": self.perturbed_length,
        }


@dataclass
class ControlFlowChanges:
    """Counts of specific structural control-flow changes.

    These are discrete, countable changes to execution flow that indicate
    how a perturbation affected the agent system's behavior.
    """

    # Agent routing changes
    reroutes: int = 0  # Different agent chosen at same step

    # Execution pattern changes
    extra_routing_cycles: int = 0  # Additional routing decisions in perturbed trace
    missing_routing_cycles: int = 0  # Fewer routing decisions in perturbed trace

    # Tool execution changes
    tool_failures_introduced: int = 0  # Tool succeeded in baseline, failed in perturbed
    tool_failures_avoided: int = 0  # Tool failed in baseline, succeeded in perturbed
    tool_calls_added: int = 0  # Extra tool calls in perturbed trace
    tool_calls_removed: int = 0  # Missing tool calls in perturbed trace

    # Termination changes
    early_termination: bool = False  # Perturbed finished before baseline step count
    extended_execution: bool = False  # Perturbed ran longer than baseline

    @property
    def total_changes(self) -> int:
        """Total number of control flow changes (composite metric)."""
        return (
            self.reroutes
            + self.extra_routing_cycles
            + self.missing_routing_cycles
            + self.tool_failures_introduced
            + self.tool_failures_avoided
            + self.tool_calls_added
            + self.tool_calls_removed
            + (1 if self.early_termination else 0)
            + (1 if self.extended_execution else 0)
        )

    @property
    def tool_state_changes(self) -> int:
        """Total tool success/failure state changes."""
        return self.tool_failures_introduced + self.tool_failures_avoided

    def to_dict(self) -> dict[str, Any]:
        return {
            "reroutes": self.reroutes,
            "extra_routing_cycles": self.extra_routing_cycles,
            "missing_routing_cycles": self.missing_routing_cycles,
            "tool_failures_introduced": self.tool_failures_introduced,
            "tool_failures_avoided": self.tool_failures_avoided,
            "tool_calls_added": self.tool_calls_added,
            "tool_calls_removed": self.tool_calls_removed,
            "early_termination": self.early_termination,
            "extended_execution": self.extended_execution,
            # Composite metrics
            "total_changes": self.total_changes,
            "tool_state_changes": self.tool_state_changes,
        }


@dataclass
class AlignedEventPair:
    """A pair of events from clean and perturbed traces, aligned by position."""

    index: int
    clean_event: TraceEvent | None
    perturbed_event: TraceEvent | None
    is_match: bool  # True if structurally equivalent (lexical differences ignored)
    divergence_type: str | None = None  # One of STRUCTURAL_DIVERGENCE_TYPES, or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "clean_event": self.clean_event.to_dict() if self.clean_event else None,
            "perturbed_event": (self.perturbed_event.to_dict() if self.perturbed_event else None),
            "is_match": self.is_match,
            "divergence_type": self.divergence_type,
        }


@dataclass
class DivergenceReport:
    """Structural divergence analysis between clean and perturbed traces.

    Primary metric: Normalized Edit Distance
    - Captures overall process disruption in one number [0, 1]
    - Robust to cascading mismatches (uses optimal sequence alignment)
    - Decomposed into substitutions/insertions/deletions

    Secondary metrics:
    - First divergence point (where structural change begins)
    - Trace statistics (step counts, routing turns, tool calls)
    """

    task_id: str
    perturbation_type: str

    # === PRIMARY: EDIT DISTANCE (normalized sequence alignment) ===
    # Captures overall trace divergence magnitude in [0, 1]
    # 0 = identical traces, 1 = completely different
    edit_distance: EditDistanceResult | None = None

    # === FIRST DIVERGENCE POINT ===
    # Where structural divergence first occurs
    first_divergence_step: int | None = None
    first_divergence_type: str | None = None
    divergence_normalized_position: float | None = None  # step / total_steps

    # === CONTROL FLOW CHANGES (discrete counts) ===
    # Specific structural changes: reroutes, tool failures, termination changes
    control_flow_changes: ControlFlowChanges | None = None

    # === TRACE STATISTICS ===
    baseline_steps: int = 0
    perturbed_steps: int = 0
    baseline_events: int = 0
    perturbed_events: int = 0
    baseline_routing_turns: int = 0
    perturbed_routing_turns: int = 0
    baseline_tool_calls: int = 0
    perturbed_tool_calls: int = 0

    # === OUTCOME ===
    baseline_answer: str | None = None
    perturbed_answer: str | None = None
    answer_changed: bool = False

    # Full alignment details (omit from summary, include in details)
    aligned_pairs: list[AlignedEventPair] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        """Compact summary without full aligned_pairs (for summary.json)."""
        return {
            "task_id": self.task_id,
            "perturbation_type": self.perturbation_type,
            # Primary metric: edit distance
            "edit_distance": self.edit_distance.to_dict() if self.edit_distance else None,
            # Control flow changes (discrete counts)
            "control_flow_changes": (
                self.control_flow_changes.to_dict() if self.control_flow_changes else None
            ),
            # First divergence point
            "first_divergence_step": self.first_divergence_step,
            "first_divergence_type": self.first_divergence_type,
            "divergence_normalized_position": self.divergence_normalized_position,
            # Trace statistics
            "baseline_steps": self.baseline_steps,
            "perturbed_steps": self.perturbed_steps,
            "baseline_events": self.baseline_events,
            "perturbed_events": self.perturbed_events,
            "baseline_routing_turns": self.baseline_routing_turns,
            "perturbed_routing_turns": self.perturbed_routing_turns,
            "baseline_tool_calls": self.baseline_tool_calls,
            "perturbed_tool_calls": self.perturbed_tool_calls,
            # Outcome
            "baseline_answer": self.baseline_answer,
            "perturbed_answer": self.perturbed_answer,
            "answer_changed": self.answer_changed,
        }

    def to_detail_dict(self) -> dict[str, Any]:
        """Full detail including aligned_pairs (for details.json)."""
        d = self.to_summary_dict()
        d["aligned_pairs"] = [p.to_dict() for p in self.aligned_pairs]
        return d


class TraceDivergenceAnalyzer:
    """Compares two execution traces to measure divergence.

    Primary metric: Normalized Edit Distance
    - Computes minimum edits (substitutions, insertions, deletions) to transform
      baseline trace into perturbed trace
    - Uses structural signatures (ignoring IDs, timestamps, lexical content)
    - Normalized to [0, 1] for cross-task comparability

    Secondary metrics:
    - First divergence point (step where traces first differ structurally)
    - Trace statistics (step counts, routing turns, tool calls)
    """

    def analyze(
        self,
        clean_trace: TraceCollector,
        perturbed_trace: TraceCollector,
        task_id: str = "",
    ) -> DivergenceReport:
        """Analyze divergence between baseline and perturbed traces.

        Args:
            clean_trace: Baseline (unperturbed) trace.
            perturbed_trace: Perturbed trace.
            task_id: Task identifier for the report.

        Returns:
            DivergenceReport with edit distance and other metrics.
        """
        # Compute structural signatures for each trace
        baseline_signatures = [self._event_to_signature(e) for e in clean_trace.events]
        perturbed_signatures = [self._event_to_signature(e) for e in perturbed_trace.events]

        # Compute edit distance (primary metric)
        edit_result = self._compute_edit_distance(baseline_signatures, perturbed_signatures)

        # Find first divergence point using alignment
        aligned = self._align_traces(clean_trace.events, perturbed_trace.events)
        first_div_step, first_div_type = self._find_first_divergence(aligned)

        # Trace statistics
        clean_steps = {e.step_num for e in clean_trace.events}
        perturbed_steps_set = {e.step_num for e in perturbed_trace.events}
        baseline_steps = len(clean_steps)
        perturbed_steps_count = len(perturbed_steps_set)
        total_steps = max(baseline_steps, perturbed_steps_count, 1)

        baseline_routing_turns = len(clean_trace.get_events_by_type("routing_decision"))
        perturbed_routing_turns = len(perturbed_trace.get_events_by_type("routing_decision"))
        baseline_tool_calls = len(clean_trace.get_events_by_type("tool_invocation"))
        perturbed_tool_calls = len(perturbed_trace.get_events_by_type("tool_invocation"))

        # Compute control flow changes (discrete counts)
        control_flow = self._compute_control_flow_changes(
            aligned,
            baseline_steps,
            perturbed_steps_count,
            baseline_routing_turns,
            perturbed_routing_turns,
            baseline_tool_calls,
            perturbed_tool_calls,
        )

        # Normalized position of first divergence
        normalized_position: float | None = None
        if first_div_step is not None:
            normalized_position = first_div_step / total_steps

        # Outcome comparison
        baseline_answer = self._extract_final_answer(clean_trace)
        perturbed_answer = self._extract_final_answer(perturbed_trace)
        answer_changed = baseline_answer != perturbed_answer

        return DivergenceReport(
            task_id=task_id,
            perturbation_type=perturbed_trace.run_label,
            edit_distance=edit_result,
            control_flow_changes=control_flow,
            first_divergence_step=first_div_step,
            first_divergence_type=first_div_type,
            divergence_normalized_position=normalized_position,
            baseline_steps=baseline_steps,
            perturbed_steps=perturbed_steps_count,
            baseline_events=len(clean_trace.events),
            perturbed_events=len(perturbed_trace.events),
            baseline_routing_turns=baseline_routing_turns,
            perturbed_routing_turns=perturbed_routing_turns,
            baseline_tool_calls=baseline_tool_calls,
            perturbed_tool_calls=perturbed_tool_calls,
            baseline_answer=baseline_answer,
            perturbed_answer=perturbed_answer,
            answer_changed=answer_changed,
            aligned_pairs=aligned,
        )

    def _event_to_signature(self, event: TraceEvent) -> str:
        """Convert event to structural signature (registry-driven).

        Signatures capture control flow and execution state (which agent was
        routed, which tool ran and whether it succeeded, ...) and ignore IDs,
        timestamps, and lexical content. Unknown event types fall back to
        ``unknown:{event_type}`` with a one-time warning.
        """
        return registry.signature_for(event.event_type, event.payload)

    def _compute_edit_distance(self, seq1: list[str], seq2: list[str]) -> EditDistanceResult:
        """Compute edit distance with backtracking to identify edit types.

        Uses Wagner-Fischer algorithm with backtracking to count
        substitutions, insertions, and deletions.
        """
        m, n = len(seq1), len(seq2)

        # Handle empty sequences
        if m == 0 and n == 0:
            return EditDistanceResult(
                distance=0,
                normalized=0.0,
                substitutions=0,
                insertions=0,
                deletions=0,
                baseline_length=0,
                perturbed_length=0,
            )
        if m == 0:
            return EditDistanceResult(
                distance=n,
                normalized=1.0,
                substitutions=0,
                insertions=n,
                deletions=0,
                baseline_length=0,
                perturbed_length=n,
            )
        if n == 0:
            return EditDistanceResult(
                distance=m,
                normalized=1.0,
                substitutions=0,
                insertions=0,
                deletions=m,
                baseline_length=m,
                perturbed_length=0,
            )

        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # deletion
                        dp[i][j - 1],  # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        # Backtrack to count edit types
        substitutions = 0
        insertions = 0
        deletions = 0
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
                # Match - no edit
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                # Substitution
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                # Insertion (element in perturbed but not baseline)
                insertions += 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                # Deletion (element in baseline but not perturbed)
                deletions += 1
                i -= 1
            else:
                # Should not happen, but handle gracefully
                break

        distance = dp[m][n]
        max_len = max(m, n)
        normalized = distance / max_len if max_len > 0 else 0.0

        return EditDistanceResult(
            distance=distance,
            normalized=normalized,
            substitutions=substitutions,
            insertions=insertions,
            deletions=deletions,
            baseline_length=m,
            perturbed_length=n,
        )

    def _find_first_divergence(
        self, aligned: list[AlignedEventPair]
    ) -> tuple[int | None, str | None]:
        """Find the first structural divergence in aligned pairs."""
        for pair in aligned:
            div_type = pair.divergence_type
            if div_type and div_type in STRUCTURAL_DIVERGENCE_TYPES:
                step = None
                if pair.clean_event:
                    step = pair.clean_event.step_num
                elif pair.perturbed_event:
                    step = pair.perturbed_event.step_num
                return step, div_type
        return None, None

    def _align_traces(
        self,
        clean_events: list[TraceEvent],
        perturbed_events: list[TraceEvent],
    ) -> list[AlignedEventPair]:
        """Align events across two traces.

        Strategy: Match by (step_num, event_type) in occurrence order.
        This handles cases where traces diverge in routing (different agents
        called at different steps) while still aligning comparable events.
        """
        # Group events by step_num
        clean_by_step = self._group_by_step(clean_events)
        perturbed_by_step = self._group_by_step(perturbed_events)

        all_steps = sorted(set(clean_by_step.keys()) | set(perturbed_by_step.keys()))
        aligned: list[AlignedEventPair] = []
        pair_index = 0

        for step in all_steps:
            clean_step_events = clean_by_step.get(step, [])
            perturbed_step_events = perturbed_by_step.get(step, [])

            step_pairs = self._align_step_events(
                clean_step_events, perturbed_step_events, pair_index
            )
            aligned.extend(step_pairs)
            pair_index += len(step_pairs)

        return aligned

    def _align_step_events(
        self,
        clean_events: list[TraceEvent],
        perturbed_events: list[TraceEvent],
        start_index: int,
    ) -> list[AlignedEventPair]:
        """Align events within a single step by event_type matching.

        Within a step, match events by type in order. E.g., if both
        traces have 2 tool_invocations at step 3, pair them 1-to-1.

        Uses STRUCTURAL comparison only - lexical differences are ignored.
        """
        pairs: list[AlignedEventPair] = []
        idx = start_index

        # Group by event_type within step
        clean_by_type: dict[str, list[TraceEvent]] = defaultdict(list)
        perturbed_by_type: dict[str, list[TraceEvent]] = defaultdict(list)

        for e in clean_events:
            clean_by_type[e.event_type].append(e)
        for e in perturbed_events:
            perturbed_by_type[e.event_type].append(e)

        all_types = sorted(set(clean_by_type.keys()) | set(perturbed_by_type.keys()))

        for etype in all_types:
            c_list = clean_by_type.get(etype, [])
            p_list = perturbed_by_type.get(etype, [])

            # Pair up in order
            max_len = max(len(c_list), len(p_list))
            for i in range(max_len):
                c_evt = c_list[i] if i < len(c_list) else None
                p_evt = p_list[i] if i < len(p_list) else None

                div_type: str | None
                if c_evt and p_evt:
                    # Check for structural divergence only
                    div_type = self._classify_divergence(c_evt, p_evt)
                    # is_match = True if no structural divergence (div_type is None)
                    is_match = div_type is None
                elif c_evt:
                    is_match = False
                    div_type = registry.EVENT_MISSING_IN_PERTURBED
                else:
                    is_match = False
                    # Registry-driven: agent_error appearing only in the
                    # perturbed trace reports "agent_error_introduced".
                    div_type = registry.missing_in_clean_type(etype)

                pairs.append(
                    AlignedEventPair(
                        index=idx,
                        clean_event=c_evt,
                        perturbed_event=p_evt,
                        is_match=is_match,
                        divergence_type=div_type,
                    )
                )
                idx += 1

        return pairs

    def _classify_divergence(self, clean: TraceEvent, perturbed: TraceEvent) -> str | None:
        """Classify the type of divergence between two events (registry-driven).

        Returns a structural divergence type, or None if only lexical difference.
        """
        if clean.event_type != perturbed.event_type:
            return registry.EVENT_TYPE_DIFFERS
        return registry.classify_divergence(clean.event_type, clean.payload, perturbed.payload)

    def _group_by_step(self, events: list[TraceEvent]) -> dict[int, list[TraceEvent]]:
        """Group events by step_num."""
        groups: dict[int, list[TraceEvent]] = defaultdict(list)
        for e in events:
            groups[e.step_num].append(e)
        return groups

    def _extract_final_answer(self, trace: TraceCollector) -> str | None:
        """Extract the final answer from a trace, if present."""
        final_events = trace.get_events_by_type("final_answer")
        if final_events:
            return cast(str | None, final_events[-1].payload.get("answer"))
        return None

    def _compute_control_flow_changes(
        self,
        aligned: list[AlignedEventPair],
        baseline_steps: int,
        perturbed_steps: int,
        baseline_routing_turns: int,
        perturbed_routing_turns: int,
        baseline_tool_calls: int,
        perturbed_tool_calls: int,
    ) -> ControlFlowChanges:
        """Compute discrete control flow changes between traces.

        Args:
            aligned: Aligned event pairs.
            baseline_steps: Number of steps in baseline.
            perturbed_steps: Number of steps in perturbed.
            baseline_routing_turns: Routing decisions in baseline.
            perturbed_routing_turns: Routing decisions in perturbed.
            baseline_tool_calls: Tool invocations in baseline.
            perturbed_tool_calls: Tool invocations in perturbed.

        Returns:
            ControlFlowChanges with counts of each change type.
        """
        changes = ControlFlowChanges()

        # === Reroutes (different agent chosen at same step) ===
        for pair in aligned:
            if pair.divergence_type == "different_agent_routed":
                changes.reroutes += 1

        # === Routing cycle changes ===
        if perturbed_routing_turns > baseline_routing_turns:
            changes.extra_routing_cycles = perturbed_routing_turns - baseline_routing_turns
        elif perturbed_routing_turns < baseline_routing_turns:
            changes.missing_routing_cycles = baseline_routing_turns - perturbed_routing_turns

        # === Tool call count changes ===
        if perturbed_tool_calls > baseline_tool_calls:
            changes.tool_calls_added = perturbed_tool_calls - baseline_tool_calls
        elif perturbed_tool_calls < baseline_tool_calls:
            changes.tool_calls_removed = baseline_tool_calls - perturbed_tool_calls

        # === Tool success/failure state changes ===
        # Look at aligned tool_invocation events where success status differs
        for pair in aligned:
            if pair.divergence_type == "tool_success_failure_change":
                c_success = (
                    pair.clean_event.payload.get("success", True) if pair.clean_event else True
                )
                p_success = (
                    pair.perturbed_event.payload.get("success", True)
                    if pair.perturbed_event
                    else True
                )
                if c_success and not p_success:
                    changes.tool_failures_introduced += 1
                elif not c_success and p_success:
                    changes.tool_failures_avoided += 1

        # === Termination changes ===
        if perturbed_steps < baseline_steps:
            changes.early_termination = True
        elif perturbed_steps > baseline_steps:
            changes.extended_execution = True

        return changes
