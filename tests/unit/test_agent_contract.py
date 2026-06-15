"""The ``Task``/``AgentRunner`` contract + ``invoke_agent`` harness (category 8)."""

import dataclasses
from pathlib import Path

import pytest
from _openai_mock import MockOpenAIServer, completion, tool_call

from traxr.agents import Task, invoke_agent
from traxr.capture import instrument
from traxr.errors import AgentContractError, RunBudgetExceeded
from traxr.metrics.analyzer import TraceDivergenceAnalyzer
from traxr.trace.collector import TraceCollector

from _openai_mock import make_fixture_agent  # isort: skip

TASK = Task(question="What is in the file?", files=(Path("sample.csv"),))


def fixture_agent_responses():
    return [
        completion("", tool_calls=[tool_call("call_f1", "lookup", '{"q": "x"}')]),
        completion("the answer is 42"),
    ]


def test_task_is_frozen_with_defaults():
    assert TASK.run_label == "baseline"
    assert dict(TASK.metadata) == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        TASK.question = "changed"  # type: ignore[misc]


def test_invoke_agent_happy_path_emits_final_answer_last():
    server = MockOpenAIServer(fixture_agent_responses())
    agent = make_fixture_agent(instrument(server.client()))
    collector = TraceCollector(run_label="baseline")

    answer = invoke_agent(agent, TASK, collector)

    assert answer == "the answer is 42"
    assert [e.event_type for e in collector.events] == [
        "llm_call",
        "tool_request",
        "tool_result",
        "llm_call",
        "final_answer",
    ]
    final = collector.events[-1]
    assert final.agent_name == "harness"
    assert final.payload.keys() == {"answer_hash"}  # hash-only by default


def test_invoke_agent_store_llm_content_includes_answer():
    server = MockOpenAIServer([completion("plain answer")])
    agent = make_fixture_agent(instrument(server.client()))
    collector = TraceCollector(run_label="baseline")

    invoke_agent(agent, TASK, collector, store_llm_content=True)

    assert collector.events[-1].payload["answer"] == "plain answer"


def test_external_agent_report_answer_fields_use_hash():
    """M1: with hash-only payloads (store_llm_content=False, the default), the
    DivergenceReport answer fields are still populated and answer_changed is
    correct — the analyzer falls back to answer_hash."""
    baseline = TraceCollector(run_label="baseline")
    perturbed = TraceCollector(run_label="perturbed")
    invoke_agent(lambda task: "forty two", TASK, baseline)
    invoke_agent(lambda task: "thirteen", TASK, perturbed)

    report = TraceDivergenceAnalyzer().analyze(baseline, perturbed, task_id="t")
    assert report.baseline_answer is not None
    assert report.perturbed_answer is not None
    assert report.baseline_answer != report.perturbed_answer  # differing hashes
    assert report.answer_changed is True

    # Identical answers -> identical hashes -> no change reported.
    same = TraceCollector(run_label="perturbed")
    invoke_agent(lambda task: "forty two", TASK, same)
    report_same = TraceDivergenceAnalyzer().analyze(baseline, same, task_id="t")
    assert report_same.answer_changed is False


def test_non_str_return_raises_agent_contract_error():
    collector = TraceCollector(run_label="baseline")
    with pytest.raises(AgentContractError, match="got 'dict'"):
        invoke_agent(lambda task: {"answer": 4}, TASK, collector)  # type: ignore[arg-type, return-value]


def test_crash_emits_agent_error_and_partial_trace_is_analyzable():
    clean_server = MockOpenAIServer(fixture_agent_responses())
    clean_collector = TraceCollector(run_label="baseline")
    invoke_agent(make_fixture_agent(instrument(clean_server.client())), TASK, clean_collector)

    crash_server = MockOpenAIServer([completion("partial")])
    crash_client = instrument(crash_server.client())

    def crashing_agent(task: Task) -> str:
        crash_client.chat.completions.create(model="mock-model", messages=[])
        raise ValueError("tool exploded")

    crashed_collector = TraceCollector(run_label="perturbed")
    with pytest.raises(ValueError, match="tool exploded"):
        invoke_agent(crashing_agent, TASK, crashed_collector)

    assert [e.event_type for e in crashed_collector.events] == ["llm_call", "agent_error"]
    error_event = crashed_collector.events[-1]
    assert error_event.payload["exc_type"] == "ValueError"
    assert error_event.payload["message"] == "tool exploded"

    # Metrics stay computable on the partial trace.
    report = TraceDivergenceAnalyzer().analyze(clean_collector, crashed_collector)
    assert report.edit_distance is not None
    assert 0.0 <= report.edit_distance.normalized <= 1.0


def test_budget_exceeded_inside_run_is_recorded_as_agent_error():
    server = MockOpenAIServer([completion("1"), completion("2")])
    agent = make_fixture_agent(instrument(server.client()))
    collector = TraceCollector(run_label="baseline")

    def greedy_agent(task: Task) -> str:
        agent(task)
        return agent(task)

    with pytest.raises(RunBudgetExceeded):
        invoke_agent(greedy_agent, TASK, collector, max_llm_calls_per_run=1)

    assert [e.event_type for e in collector.events] == ["llm_call", "agent_error"]
    assert collector.events[-1].payload["exc_type"] == "RunBudgetExceeded"


def test_invoke_agent_passes_task_through_unchanged():
    seen: list[Task] = []

    def agent(task: Task) -> str:
        seen.append(task)
        return "ok"

    collector = TraceCollector(run_label="baseline")
    task = Task(
        question="q",
        files=(Path("a.csv"), Path("b.pdf")),
        run_label="row_shuffle",
        metadata={"item_id": "x1"},
    )
    invoke_agent(agent, task, collector)
    assert seen == [task]
