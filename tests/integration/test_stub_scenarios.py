"""Category 5 — trace pipeline under the stub (built-in agent).

Validates that the extracted EpisodeRunner emits well-ordered events through
the M1 TraceCollector and that every DeterministicLLMStub scenario produces
its intended trace shape (the M3 validation checklist item M4 builds on).

Only routing_decision and agent_output fire on every step; the other event
types are conditional — assertions are per-scenario expected sets, never
"all event types every run".
"""

from pathlib import Path

import pytest

from traxr.agents import builtin_agent
from traxr.llm import DeterministicLLMStub
from traxr.llm.stub import SCENARIOS, ScriptedToolCall, StubReply, _plan_reply
from traxr.trace import TraceCollector

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
QUESTION = "How many rows of data does the table contain?"
ROW_COUNT = "5"  # tests/fixtures/sample.csv has 5 data rows
WRONG = "999"


def run_scenario(csv_path: Path, scenario: str) -> tuple[str | None, TraceCollector]:
    stub = DeterministicLLMStub(scenario, final_answer=ROW_COUNT, wrong_answer=WRONG)
    agent = builtin_agent(llm=stub)()
    collector = TraceCollector(run_label=scenario)
    answer = agent.run([csv_path], QUESTION, collector=collector)
    return answer, collector


@pytest.fixture(scope="module")
def scenario_runs() -> dict[str, tuple[str | None, TraceCollector]]:
    """One run per scenario, shared across this module's assertions."""
    csv_path = FIXTURES_DIR / "sample.csv"
    return {scenario: run_scenario(csv_path, scenario) for scenario in SCENARIOS}


def routing_sequence(collector: TraceCollector) -> list[str]:
    return [
        e.payload["chosen_agent"] for e in collector.events if e.event_type == "routing_decision"
    ]


def event_types(collector: TraceCollector) -> list[str]:
    return [e.event_type for e in collector.events]


# Required payload keys (and spot-checked types) per built-in event type,
# matching the schema documented on traxr.trace.events.TraceEvent.
PAYLOAD_SCHEMAS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "routing_decision": {"chosen_agent": str, "reasoning_hash": str},
    "tool_invocation": {
        "tool_name": str,
        "operation": str,
        "arguments": dict,
        "success": bool,
        "output_hash": str,
        "output_preview": str,
    },
    "memory_write": {
        "entry_id": str,
        "entry_type": str,
        "content_hash": str,
        "confidence": (int, float),
    },
    "memory_read": {"entry_ids": list, "entry_types": list},
    "retrieval_shown": {"query": str, "item_count": int, "item_hashes": list},
    "agent_output": {
        "action": str,
        "content_hash": str,
        "is_final_answer": bool,
        "citation_ids": list,
    },
    "final_answer": {"answer": str, "answer_hash": str},
    "tool_failure": {"tool_name": str, "error": str},
    "agent_halt": {"reason": str},
}


class TestTraceWellFormedness:
    def test_events_are_well_ordered(self, scenario_runs):
        for scenario, (_, collector) in scenario_runs.items():
            assert collector.event_count > 0, scenario
            indices = [e.sequence_index for e in collector.events]
            assert indices == list(range(len(indices))), scenario
            steps = [e.step_num for e in collector.events]
            assert steps == sorted(steps), f"{scenario}: step_num not monotone"

    def test_every_payload_matches_its_schema(self, scenario_runs):
        for scenario, (_, collector) in scenario_runs.items():
            for event in collector.events:
                schema = PAYLOAD_SCHEMAS.get(event.event_type)
                assert schema is not None, f"{scenario}: unexpected type {event.event_type}"
                for key, expected_type in schema.items():
                    assert key in event.payload, f"{scenario}/{event.event_type}: missing {key}"
                    assert isinstance(event.payload[key], expected_type), (
                        f"{scenario}/{event.event_type}.{key}: {type(event.payload[key]).__name__}"
                    )
                assert event.content_hash
                assert event.agent_name

    def test_routing_decision_and_agent_output_fire_every_step(self, scenario_runs):
        """The two unconditional per-step event types (others are conditional)."""
        _, collector = scenario_runs["identity"]
        executed_steps = sorted(
            {e.step_num for e in collector.events if e.event_type == "agent_output"}
        )
        for step in executed_steps:
            step_types = {e.event_type for e in collector.get_events_by_step(step)}
            assert "routing_decision" in step_types
            assert "agent_output" in step_types


class TestScenarioShapes:
    def test_identity_shape(self, scenario_runs):
        answer, collector = scenario_runs["identity"]
        assert answer == ROW_COUNT
        assert routing_sequence(collector) == ["planner", "data_analyst", "synthesizer"]

        finals = collector.get_events_by_type("final_answer")
        assert len(finals) == 1
        assert finals[0].payload["answer"] == ROW_COUNT

        invocations = collector.get_events_by_type("tool_invocation")
        assert len(invocations) >= 1
        assert all(e.payload["success"] for e in invocations)
        assert invocations[0].payload["tool_name"] == "python"

        assert not collector.get_events_by_type("agent_halt")
        assert not collector.get_events_by_type("tool_failure")

    def test_wrong_answer_same_route_different_answer(self, scenario_runs):
        identity_answer, identity_collector = scenario_runs["identity"]
        wrong_answer, wrong_collector = scenario_runs["wrong_answer"]

        # Same structural path (the dangerous case: silent corruption) ...
        assert routing_sequence(wrong_collector) == routing_sequence(identity_collector)
        assert event_types(wrong_collector) == event_types(identity_collector)
        # ... but a different final answer.
        assert wrong_answer == WRONG
        assert wrong_answer != identity_answer

    def test_reroute_inserts_critic(self, scenario_runs):
        answer, collector = scenario_runs["reroute"]
        route = routing_sequence(collector)
        assert "critic" in route
        assert route != routing_sequence(scenario_runs["identity"][1])
        assert answer == ROW_COUNT
        assert len(collector.get_events_by_type("final_answer")) == 1

    def test_halt_early_emits_agent_halt_without_final_answer(self, scenario_runs):
        answer, collector = scenario_runs["halt_early"]
        assert answer is None
        assert not collector.get_events_by_type("final_answer")

        halts = collector.get_events_by_type("agent_halt")
        assert len(halts) == 1
        assert halts[0].payload["reason"] == "token_budget_exhausted"
        assert halts[0].agent_name == "runner"

        # Strictly shorter than the identity run.
        identity_route = routing_sequence(scenario_runs["identity"][1])
        assert len(routing_sequence(collector)) <= len(identity_route)
        assert collector.event_count < scenario_runs["identity"][1].event_count

    def test_loop_extends_execution_with_repeated_routing(self, scenario_runs):
        answer, collector = scenario_runs["loop"]
        route = routing_sequence(collector)
        identity_route = routing_sequence(scenario_runs["identity"][1])

        assert len(route) > len(identity_route)
        assert route.count("researcher") >= 3
        # Extended but still concludes.
        assert answer == ROW_COUNT
        assert len(collector.get_events_by_type("final_answer")) == 1


class TestToolFailureEmission:
    def test_failing_tool_call_emits_tool_failure(self, fixtures_dir):
        """A crashing python.run produces tool_invocation(success=False) + tool_failure."""
        script = {
            "plan": [
                _plan_reply(
                    [
                        {
                            "id": "s1",
                            "description": "Extract the requested value from the table",
                            "agent": "data_analyst",
                            "dependencies": [],
                        },
                        {
                            "id": "s2",
                            "description": "Produce the final answer",
                            "agent": "synthesizer",
                            "dependencies": ["s1"],
                        },
                    ],
                    "Analyze the table, then produce the final answer.",
                )
            ],
            "data_analyst": [
                StubReply(
                    tool_calls=[
                        ScriptedToolCall(
                            tool_name="python",
                            operation="run",
                            arguments={"code": 'raise ValueError("boom")'},
                        )
                    ]
                )
            ],
            "synthesize": [StubReply(content="n/a")],
            "route": [StubReply(content="NEXT_AGENT: synthesizer")],
            "default": [StubReply(content="Proceeding with the task as planned.")],
        }
        stub = DeterministicLLMStub(script=script)
        agent = builtin_agent(llm=stub, max_steps=4)()
        collector = TraceCollector(run_label="tool-failure")
        agent.run([fixtures_dir / "sample.csv"], QUESTION, collector=collector)

        failed_invocations = [
            e for e in collector.get_events_by_type("tool_invocation") if not e.payload["success"]
        ]
        failures = collector.get_events_by_type("tool_failure")
        assert failed_invocations, "expected a failed tool_invocation"
        assert failures, "expected tool_failure to be emitted"
        assert failures[0].payload["tool_name"] == "python"
        assert "ValueError" in failures[0].payload["error"]
        # tool_failure directly follows its failed invocation.
        first_failure = failures[0]
        preceding = collector.events[first_failure.sequence_index - 1]
        assert preceding.event_type == "tool_invocation"
        assert preceding.payload["success"] is False
