"""Unit tests for DeterministicLLMStub (scripting mechanics, no agent runs)."""

import json

import pytest

from traxr.llm import DeterministicLLMStub, LLMClient, LLMResponse, LLMToolResponse
from traxr.llm.stub import SCENARIOS, ScriptedToolCall, StubReply


class TestScenarioConstruction:
    def test_all_documented_scenarios_construct(self):
        assert set(SCENARIOS) == {"identity", "wrong_answer", "reroute", "halt_early", "loop"}
        for scenario in SCENARIOS:
            stub = DeterministicLLMStub(scenario)
            assert stub.scenario == scenario
            assert stub.model == f"stub:{scenario}"

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown stub scenario"):
            DeterministicLLMStub("nonsense")

    def test_conforms_to_llm_client_protocol(self):
        assert isinstance(DeterministicLLMStub(), LLMClient)


class TestScriptKeying:
    def test_keyed_by_response_type_and_call_index(self):
        script = {
            "route": [StubReply(content="first"), StubReply(content="second")],
            "default": [StubReply(content="fallback")],
        }
        stub = DeterministicLLMStub(script=script)
        assert stub.generate("p", response_type="route").content == "first"
        assert stub.generate("p", response_type="route").content == "second"
        # Last reply repeats once the script is exhausted.
        assert stub.generate("p", response_type="route").content == "second"
        # Unscripted types fall back to "default".
        assert stub.generate("p", response_type="synthesize").content == "fallback"

    def test_counters_are_per_response_type(self):
        stub = DeterministicLLMStub("identity")
        plan_1 = stub.generate("p", response_type="plan")
        synth_1 = stub.generate("p", response_type="synthesize")
        plan_2 = stub.generate("p", response_type="plan")
        assert plan_1.content == plan_2.content  # single plan reply repeats
        assert synth_1.content != plan_1.content

    def test_reset_call_count_restores_initial_state(self):
        stub = DeterministicLLMStub("identity")
        first = stub.generate("p", response_type="plan").content
        stub.generate("p", response_type="plan")
        assert stub.call_count == 2
        stub.reset_call_count()
        assert stub.call_count == 0
        assert stub.generate("p", response_type="plan").content == first

    def test_deterministic_across_instances(self):
        a = DeterministicLLMStub("identity", final_answer="7")
        b = DeterministicLLMStub("identity", final_answer="7")
        for rt in ("plan", "data_analyst", "synthesize", "route"):
            assert (
                a.generate("x", response_type=rt).content
                == b.generate("x", response_type=rt).content
            )


class TestResponseShapes:
    def test_generate_returns_llm_response(self):
        response = DeterministicLLMStub().generate("prompt", response_type="synthesize")
        assert isinstance(response, LLMResponse)
        assert response.finish_reason == "stop"
        assert response.prompt_tokens > 0 and response.completion_tokens > 0

    def test_generate_with_tools_returns_scripted_tool_calls(self):
        stub = DeterministicLLMStub("identity")
        response = stub.generate_with_tools("prompt", tools=[], response_type="data_analyst")
        assert isinstance(response, LLMToolResponse)
        assert response.has_tool_calls
        assert response.finish_reason == "tool_calls"
        call = response.tool_calls[0]
        assert isinstance(call, ScriptedToolCall)
        # Duck-type surface consumed by the reference agent.
        assert call.tool_name == "python"
        assert call.operation == "run"
        assert "code" in call.arguments
        assert call.call_id

    def test_text_only_reply_has_stop_finish_reason(self):
        stub = DeterministicLLMStub("identity")
        response = stub.generate_with_tools("prompt", tools=[], response_type="synthesize")
        assert not response.has_tool_calls
        assert response.finish_reason == "stop"

    def test_generate_with_retrieval_uses_same_script(self):
        stub = DeterministicLLMStub("identity", final_answer="42")
        response = stub.generate_with_retrieval("prompt", ["doc"], response_type="synthesize")
        assert response.content == "42"


class TestScenarioScripts:
    def test_plan_replies_are_parseable_execution_plans(self):
        """Every scenario's plan reply must parse via the MAS plan parser."""
        from traxr.mas.planning.plan_types import ExecutionPlan

        for scenario in SCENARIOS:
            stub = DeterministicLLMStub(scenario)
            content = stub.generate("p", response_type="plan").content
            plan = ExecutionPlan.from_json(content)
            assert plan.subtasks, scenario
            data = json.loads(content)
            assert {st["agent"] for st in data["subtasks"]} <= {
                "data_analyst",
                "critic",
                "researcher",
                "synthesizer",
            }

    def test_wrong_answer_differs_from_identity(self):
        identity = DeterministicLLMStub("identity", final_answer="5")
        wrong = DeterministicLLMStub("wrong_answer", final_answer="5", wrong_answer="999")
        assert identity.generate("p", response_type="synthesize").content == "5"
        assert wrong.generate("p", response_type="synthesize").content == "999"

    def test_halt_early_plan_reply_blows_token_budget(self):
        stub = DeterministicLLMStub("halt_early")
        response = stub.generate("p", response_type="plan")
        assert response.completion_tokens >= 1_000_000

    def test_loop_plan_repeats_researcher(self):
        stub = DeterministicLLMStub("loop")
        data = json.loads(stub.generate("p", response_type="plan").content)
        researcher_subtasks = [st for st in data["subtasks"] if st["agent"] == "researcher"]
        assert len(researcher_subtasks) >= 3
