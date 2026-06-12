"""Hypothesis properties of the divergence analyzer.

Over BOTH the built-in and the external event vocabularies:

* ``0 <= d_norm <= 1``
* identity: a trace compared with itself has zero distance and no ``t*``
* symmetry: edit distance is symmetric in its arguments
* ``t*`` in-bounds: the first-divergence step exists in one of the traces and
  its normalized position is in ``(0, 1]``
* oracle: distance matches an independent Wagner-Fischer implementation
  (mirrors the source repo's ``verify_edit_distance.py``)
"""

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from traxr.metrics.analyzer import TraceDivergenceAnalyzer
from traxr.trace.collector import TraceCollector

_AGENTS = ["planner", "researcher", "analyst"]
_TOOLS = ["csv_tool", "search", "calculator"]

# Payload strategies per event type, spanning the built-in vocabulary
# (original 7 + tool_failure/agent_halt) and the external Tier 0/1 vocabulary.
_BUILTIN_PAYLOADS: dict[str, st.SearchStrategy[dict[str, Any]]] = {
    "routing_decision": st.fixed_dictionaries({"chosen_agent": st.sampled_from(_AGENTS)}),
    "tool_invocation": st.fixed_dictionaries(
        {
            "tool_name": st.sampled_from(_TOOLS),
            "operation": st.sampled_from(["read", "write", "query"]),
            "success": st.booleans(),
        }
    ),
    "memory_write": st.fixed_dictionaries({"entry_type": st.sampled_from(["note", "fact"])}),
    "memory_read": st.fixed_dictionaries(
        {"entry_ids": st.lists(st.sampled_from(["m1", "m2", "m3"]), max_size=3, unique=True)}
    ),
    "retrieval_shown": st.fixed_dictionaries({"item_count": st.integers(0, 5)}),
    "agent_output": st.fixed_dictionaries(
        {
            "action": st.sampled_from(["draft", "critique", "answer"]),
            "is_final_answer": st.booleans(),
        }
    ),
    "final_answer": st.fixed_dictionaries({"answer": st.sampled_from(["42", "$4.2M", ""])}),
    "tool_failure": st.fixed_dictionaries({"tool_name": st.sampled_from(_TOOLS)}),
    "agent_halt": st.fixed_dictionaries({"reason": st.sampled_from(["max_steps", "budget"])}),
}

_EXTERNAL_PAYLOADS: dict[str, st.SearchStrategy[dict[str, Any]]] = {
    "llm_call": st.fixed_dictionaries(
        {
            "model": st.sampled_from(["gpt-4o", "gpt-4o-mini"]),
            "finish_reason": st.sampled_from(["stop", "tool_calls", "length"]),
            "tool_call_names": st.lists(st.sampled_from(_TOOLS), max_size=2),
        }
    ),
    "tool_request": st.fixed_dictionaries({"tool_name": st.sampled_from(_TOOLS)}),
    "tool_result": st.fixed_dictionaries({"tool_name": st.sampled_from(_TOOLS)}),
    "agent_error": st.fixed_dictionaries(
        {"exc_type": st.sampled_from(["ValueError", "TimeoutError"])}
    ),
}


def _events_strategy(
    payloads: dict[str, st.SearchStrategy[dict[str, Any]]],
) -> st.SearchStrategy[list[tuple[str, dict[str, Any]]]]:
    event = st.sampled_from(sorted(payloads)).flatmap(
        lambda etype: st.tuples(st.just(etype), payloads[etype])
    )
    return st.lists(event, max_size=12)


builtin_events = _events_strategy(_BUILTIN_PAYLOADS)
external_events = _events_strategy(_EXTERNAL_PAYLOADS)
mixed_events = _events_strategy({**_BUILTIN_PAYLOADS, **_EXTERNAL_PAYLOADS})
trace_pairs = st.one_of(
    st.tuples(builtin_events, builtin_events),
    st.tuples(external_events, external_events),
    st.tuples(mixed_events, mixed_events),
)


def _make_trace(label: str, events: list[tuple[str, dict[str, Any]]]) -> TraceCollector:
    collector = TraceCollector(run_label=label)
    for i, (etype, payload) in enumerate(events):
        collector.emit(etype, step_num=i + 1, agent_name="agent", payload=payload)
    return collector


def _reference_levenshtein(seq1: list[str], seq2: list[str]) -> int:
    """Independent Wagner-Fischer oracle (no backtrace), per verify_edit_distance.py."""
    m, n = len(seq1), len(seq2)
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
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


_analyzer = TraceDivergenceAnalyzer()


@given(trace_pairs)
def test_d_norm_bounded(pair):
    clean, perturbed = pair
    report = _analyzer.analyze(_make_trace("baseline", clean), _make_trace("p", perturbed))
    assert report.edit_distance is not None
    assert 0.0 <= report.edit_distance.normalized <= 1.0
    assert report.edit_distance.distance >= 0


@given(st.one_of(builtin_events, external_events, mixed_events))
def test_identity(events):
    report = _analyzer.analyze(_make_trace("baseline", events), _make_trace("p", events))
    assert report.edit_distance is not None
    assert report.edit_distance.distance == 0
    assert report.edit_distance.normalized == 0.0
    assert report.first_divergence_step is None
    assert report.first_divergence_type is None
    assert report.answer_changed is False


@given(trace_pairs)
def test_symmetry(pair):
    clean, perturbed = pair
    forward = _analyzer.analyze(_make_trace("baseline", clean), _make_trace("p", perturbed))
    backward = _analyzer.analyze(_make_trace("baseline", perturbed), _make_trace("p", clean))
    assert forward.edit_distance is not None and backward.edit_distance is not None
    assert forward.edit_distance.distance == backward.edit_distance.distance
    assert forward.edit_distance.normalized == backward.edit_distance.normalized


@given(trace_pairs)
def test_tstar_in_bounds(pair):
    clean, perturbed = pair
    clean_trace = _make_trace("baseline", clean)
    perturbed_trace = _make_trace("p", perturbed)
    report = _analyzer.analyze(clean_trace, perturbed_trace)
    if report.first_divergence_step is None:
        assert report.divergence_normalized_position is None
        return
    all_steps = {e.step_num for e in clean_trace.events} | {
        e.step_num for e in perturbed_trace.events
    }
    assert report.first_divergence_step in all_steps
    assert report.divergence_normalized_position is not None
    assert 0.0 < report.divergence_normalized_position <= 1.0


@given(trace_pairs)
def test_distance_matches_reference_oracle(pair):
    clean, perturbed = pair
    clean_trace = _make_trace("baseline", clean)
    perturbed_trace = _make_trace("p", perturbed)
    report = _analyzer.analyze(clean_trace, perturbed_trace)
    sigs_clean = [_analyzer._event_to_signature(e) for e in clean_trace.events]
    sigs_perturbed = [_analyzer._event_to_signature(e) for e in perturbed_trace.events]
    assert report.edit_distance is not None
    assert report.edit_distance.distance == _reference_levenshtein(sigs_clean, sigs_perturbed)


@given(trace_pairs)
def test_edit_decomposition_sums_to_distance(pair):
    clean, perturbed = pair
    report = _analyzer.analyze(_make_trace("baseline", clean), _make_trace("p", perturbed))
    ed = report.edit_distance
    assert ed is not None
    assert ed.substitutions + ed.insertions + ed.deletions == ed.distance
