"""M3 smoke: the reference agent answers a fixture CSV question end-to-end
under the DeterministicLLMStub, with a well-formed trace and zero API keys."""

import pandas as pd
import pytest

from traxr.agents import builtin_agent
from traxr.llm import DeterministicLLMStub
from traxr.trace import TraceCollector

QUESTION = "How many rows of data does the table contain?"


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """The no-key path must really need no keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_reference_agent_answers_csv_question_end_to_end(fixtures_dir):
    csv_path = fixtures_dir / "sample.csv"
    expected = str(len(pd.read_csv(csv_path)))

    stub = DeterministicLLMStub("identity", final_answer=expected)
    factory = builtin_agent(llm=stub)
    agent = factory()
    collector = TraceCollector(run_label="baseline")

    answer = agent.run([csv_path], QUESTION, expected_answer=expected, collector=collector)

    # Correct answer, derived from a real python-tool run over the real file.
    assert answer == expected
    invocation = collector.get_events_by_type("tool_invocation")[0]
    assert f"Answer: {expected}" in invocation.payload["output_preview"]

    # Well-formed trace: monotone ordering, the expected event families.
    types = {e.event_type for e in collector.events}
    assert {
        "routing_decision",
        "agent_output",
        "tool_invocation",
        "memory_write",
        "final_answer",
    } <= types
    indices = [e.sequence_index for e in collector.events]
    assert indices == list(range(len(indices)))
    assert collector.get_events_by_type("final_answer")[-1].payload["answer"] == expected


def test_paired_runs_are_structurally_deterministic(fixtures_dir):
    """Two identity runs produce the same structural trace (controlled variable)."""
    csv_path = fixtures_dir / "sample.csv"

    def run():
        stub = DeterministicLLMStub("identity", final_answer="5")
        agent = builtin_agent(llm=stub)()
        collector = TraceCollector(run_label="run")
        answer = agent.run([csv_path], QUESTION, collector=collector)
        structure = [(e.event_type, e.step_num, e.agent_name) for e in collector.events]
        return answer, structure

    answer_a, structure_a = run()
    answer_b, structure_b = run()
    assert answer_a == answer_b == "5"
    assert structure_a == structure_b
