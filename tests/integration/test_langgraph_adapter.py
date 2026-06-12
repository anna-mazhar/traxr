"""LangGraph adapter (category 11): event mapping, reroutes, suppression.

Runs against the pinned ``[langgraph]`` extra with ``GenericFakeChatModel``
— deterministic, offline. Skip-safe when the extra is not installed.
"""

import warnings
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from traxr.agents import Task, from_langgraph, invoke_agent  # noqa: E402
from traxr.agents.langgraph import _get_handler_cls  # noqa: E402
from traxr.capture import CaptureSession, bind_session, instrument  # noqa: E402
from traxr.errors import AgentContractError, TokenUnavailableWarning  # noqa: E402
from traxr.metrics.analyzer import TraceDivergenceAnalyzer  # noqa: E402
from traxr.trace.collector import TraceCollector  # noqa: E402


def fake_model(*messages):
    return GenericFakeChatModel(messages=iter(messages))


def run_with_capture(runner, question="q", files=()):
    collector = TraceCollector(run_label="run")
    task = Task(question=question, files=tuple(Path(f) for f in files))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TokenUnavailableWarning)
        answer = invoke_agent(runner, task, collector)
    return answer, collector


def linear_graph():
    """planner -> analyst, one fake-model call per node."""
    model = fake_model(AIMessage(content="plan made"), AIMessage(content="the answer is 42"))

    def planner(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    def analyst(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("planner", planner)
    graph.add_node("analyst", analyst)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", END)
    return graph.compile()


def test_linear_graph_event_mapping():
    answer, collector = run_with_capture(from_langgraph(linear_graph()))

    assert answer == "the answer is 42"
    assert [(e.event_type, e.step_num) for e in collector.events] == [
        ("routing_decision", 1),
        ("llm_call", 1),
        ("routing_decision", 2),
        ("llm_call", 2),
        ("final_answer", 2),
    ]
    route_one, llm_one = collector.events[0], collector.events[1]
    assert route_one.payload == {"chosen_agent": "planner", "via": "langgraph_node"}
    assert llm_one.agent_name == "planner"
    assert llm_one.payload["model"] == "generic-fake-chat-model"
    assert llm_one.payload["finish_reason"] == "stop"
    assert llm_one.payload["usage"] is None  # the fake model reports none
    assert "content" not in llm_one.payload  # hash-only by default


def test_default_input_builder_mentions_question_and_files():
    seen = {}

    class DuckGraph:
        def invoke(self, graph_input, config=None):
            seen["input"] = graph_input
            return {"messages": [AIMessage(content="ok")]}

    runner = from_langgraph(DuckGraph())
    run_with_capture(runner, question="What is the total?", files=["data/sample.csv"])
    text = seen["input"]["messages"][0][1]
    assert "What is the total?" in text
    assert "sample.csv" in text


def test_tool_graph_emits_tool_invocation_with_fidelity():
    @tool
    def lookup(q: str) -> str:
        """Look up a value."""
        return "42"

    model = fake_model(
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c1"}]),
        AIMessage(content="done"),
    )

    def agent(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([lookup]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    answer, collector = run_with_capture(from_langgraph(graph.compile()))

    assert answer == "done"
    llm_calls = collector.get_events_by_type("llm_call")
    assert llm_calls[0].payload["tool_call_names"] == ["lookup"]
    (tool_event,) = collector.get_events_by_type("tool_invocation")
    assert tool_event.payload["tool_name"] == "lookup"
    assert tool_event.payload["operation"] == "call"
    assert tool_event.payload["success"] is True
    assert tool_event.payload["output_hash"]
    routed = [e.payload["chosen_agent"] for e in collector.get_events_by_type("routing_decision")]
    assert routed == ["agent", "tools", "agent"]


def test_tool_error_maps_to_failed_tool_invocation():
    collector = TraceCollector(run_label="run")
    handler = _get_handler_cls()(CaptureSession(collector))
    metadata = {"langgraph_node": "tools", "langgraph_step": 3}
    handler.on_tool_start({"name": "boom"}, "x", run_id="r1", metadata=metadata)
    handler.on_tool_error(ValueError("nope"), run_id="r1")

    (event,) = collector.events
    assert event.event_type == "tool_invocation"
    assert event.payload["success"] is False
    assert event.payload["tool_name"] == "boom"
    assert event.step_num == 3


def test_usage_metadata_propagates_to_llm_call():
    message = AIMessage(
        content="counted",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    def node(state: MessagesState):
        return {"messages": [fake_model(message).invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    _, collector = run_with_capture(from_langgraph(graph.compile()))

    (llm_event,) = collector.get_events_by_type("llm_call")
    assert llm_event.payload["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def routing_graph():
    """START routes to table_agent or text_agent based on the question."""

    def route(state: MessagesState) -> str:
        text = str(state["messages"][0].content)
        return "table_agent" if "table" in text else "text_agent"

    def table_agent(state: MessagesState):
        return {"messages": [fake_model(AIMessage(content="from table")).invoke(state["messages"])]}

    def text_agent(state: MessagesState):
        return {"messages": [fake_model(AIMessage(content="from text")).invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("table_agent", table_agent)
    graph.add_node("text_agent", text_agent)
    graph.add_conditional_edges(START, route)
    graph.add_edge("table_agent", END)
    graph.add_edge("text_agent", END)
    return graph.compile()


def test_reroute_detected_across_perturbed_pair():
    # Same graph; the "perturbation" flips the routing condition — the
    # existing analyzer must see it with zero changes.
    _, clean = run_with_capture(from_langgraph(routing_graph()), question="use the table")
    _, perturbed = run_with_capture(from_langgraph(routing_graph()), question="plain prose now")

    report = TraceDivergenceAnalyzer().analyze(clean, perturbed, task_id="reroute")
    assert report.first_divergence_type == "different_agent_routed"
    assert report.control_flow_changes is not None
    assert report.control_flow_changes.reroutes >= 1


def test_tier0_double_count_suppression():
    pytest.importorskip("openai")  # the [langgraph]-only CI job runs without it
    from _openai_mock import MODEL, MockOpenAIServer, completion

    server = MockOpenAIServer([completion("via raw client")], cycle=True)
    client = instrument(server.client())

    def node(state: MessagesState):
        response = client.chat.completions.create(model=MODEL, messages=[])
        return {"messages": [AIMessage(content=response.choices[0].message.content or "")]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    answer, collector = run_with_capture(from_langgraph(graph.compile()))

    assert answer == "via raw client"
    # The instrumented client's call is NOT double-counted during the graph run.
    assert collector.get_events_by_type("llm_call") == []
    # Control: outside the graph the same client emits normally.
    control = TraceCollector(run_label="control")
    with bind_session(CaptureSession(control)):
        client.chat.completions.create(model=MODEL, messages=[])
    assert len(control.get_events_by_type("llm_call")) == 1


def test_input_builder_and_output_extractor_hooks():
    class DuckGraph:
        def invoke(self, graph_input, config=None):
            return {"verdict": graph_input["q"].upper()}

    runner = from_langgraph(
        DuckGraph(),
        input_builder=lambda task: {"q": task.question},
        output_extractor=lambda result: result["verdict"],
    )
    answer, _ = run_with_capture(runner, question="fine")
    assert answer == "FINE"


def test_default_output_extractor_rejects_non_messages_state():
    class DuckGraph:
        def invoke(self, graph_input, config=None):
            return {"not_messages": True}

    with pytest.raises(AgentContractError, match="output_extractor"):
        run_with_capture(from_langgraph(DuckGraph()))


def test_runner_without_session_is_passthrough():
    runner = from_langgraph(linear_graph())
    task = Task(question="standalone", files=())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TokenUnavailableWarning)
        assert runner(task) == "the answer is 42"


def test_missing_extra_raises_optional_dependency_error(monkeypatch):
    import sys

    from traxr.agents import langgraph as adapter_module
    from traxr.errors import OptionalDependencyError

    monkeypatch.setattr(adapter_module, "_handler_cls", None)
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", None)
    with pytest.raises(OptionalDependencyError, match=r"traxr\[langgraph\]"):
        from_langgraph(object())
