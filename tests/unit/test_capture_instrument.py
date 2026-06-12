"""Tier 0 ``instrument()`` — sync interception paths (category 7), offline."""

import pytest
from _openai_mock import USAGE, MockOpenAIServer, completion, tool_call

from traxr.capture import CaptureSession, bind_session, instrument, suppress_tier0
from traxr.errors import RunBudgetExceeded, TokenUnavailableWarning
from traxr.trace.collector import TraceCollector


def make_session(**kwargs):
    collector = TraceCollector(run_label="baseline")
    return CaptureSession(collector, **kwargs), collector


def test_sync_create_emits_llm_call():
    server = MockOpenAIServer([completion("the answer is 4")])
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session):
        response = client.chat.completions.create(
            model="mock-model", messages=[{"role": "user", "content": "2+2?"}]
        )

    assert response.choices[0].message.content == "the answer is 4"
    assert [e.event_type for e in collector.events] == ["llm_call"]
    event = collector.events[0]
    assert event.step_num == 1
    assert event.payload["model"] == "mock-model"
    assert event.payload["finish_reason"] == "stop"
    assert event.payload["tool_call_names"] == []
    assert event.payload["usage"] == USAGE
    # Hash-only by default: no raw content in the payload.
    assert "content" not in event.payload
    assert event.payload["content_hash"]


def test_tool_call_roundtrip_events_and_id_mapping():
    server = MockOpenAIServer(
        [
            completion(
                "",
                tool_calls=[tool_call("call_abc", "lookup", '{"q": "x"}')],
            ),
            completion("final answer"),
        ]
    )
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session):
        first = client.chat.completions.create(
            model="mock-model", messages=[{"role": "user", "content": "q"}]
        )
        tc = first.choices[0].message.tool_calls[0]
        client.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call(tc.id, "lookup", '{"q": "x"}')],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": "result"},
            ],
        )

    kinds = [(e.event_type, e.step_num) for e in collector.events]
    assert kinds == [
        ("llm_call", 1),
        ("tool_request", 1),
        ("tool_result", 1),  # joined back to the requesting llm_call's step
        ("llm_call", 2),
    ]
    request_event = collector.events[1]
    assert request_event.payload["tool_name"] == "lookup"
    assert request_event.payload["call_id"] == "call_abc"
    assert "arguments" not in request_event.payload  # hash-only default
    result_event = collector.events[2]
    assert result_event.payload["tool_name"] == "lookup"  # via call_id map
    assert "success" not in result_event.payload  # Tier 0 can't know it
    assert collector.events[0].payload["tool_call_names"] == ["lookup"]


def test_tool_result_dedupe_and_unknown_call_id():
    history = [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "call_unseen", "content": "r"},
    ]
    server = MockOpenAIServer([completion("a"), completion("b")])
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session):
        client.chat.completions.create(model="mock-model", messages=history)
        # Same tool message re-sent: must not produce a second tool_result.
        client.chat.completions.create(model="mock-model", messages=history)

    tool_results = collector.get_events_by_type("tool_result")
    assert len(tool_results) == 1
    assert tool_results[0].payload["tool_name"] == "?"  # no tool_request seen


def test_budget_exceeded_preserves_partial_trace():
    server = MockOpenAIServer([completion("1"), completion("2"), completion("3")])
    client = instrument(server.client())
    session, collector = make_session(max_llm_calls_per_run=2)
    with bind_session(session):
        client.chat.completions.create(model="mock-model", messages=[])
        client.chat.completions.create(model="mock-model", messages=[])
        with pytest.raises(RunBudgetExceeded):
            client.chat.completions.create(model="mock-model", messages=[])

    assert len(collector.get_events_by_type("llm_call")) == 2
    assert len(server.requests) == 2  # the over-budget call never went out


def test_missing_usage_warns_token_unavailable():
    server = MockOpenAIServer([completion("a", usage=None)])
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session), pytest.warns(TokenUnavailableWarning):
        client.chat.completions.create(model="mock-model", messages=[])
    assert collector.events[0].payload["usage"] is None


def test_store_llm_content_opt_in():
    server = MockOpenAIServer(
        [completion("raw text", tool_calls=[tool_call("c1", "lookup", '{"q": 1}')])]
    )
    client = instrument(server.client())
    session, collector = make_session(store_llm_content=True)
    with bind_session(session):
        client.chat.completions.create(model="mock-model", messages=[])

    llm_event, request_event = collector.events
    assert llm_event.payload["content"] == "raw text"
    assert request_event.payload["arguments"] == '{"q": 1}'


def test_no_session_is_pure_passthrough():
    server = MockOpenAIServer([completion("standalone")])
    client = instrument(server.client())
    response = client.chat.completions.create(model="mock-model", messages=[])
    assert response.choices[0].message.content == "standalone"
    assert len(server.requests) == 1


def test_tier1_suppression_flag_silences_tier0():
    server = MockOpenAIServer([completion("a")])
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session), suppress_tier0():
        client.chat.completions.create(model="mock-model", messages=[])
    assert collector.events == []


def test_instrument_is_idempotent():
    server = MockOpenAIServer([completion("a")])
    client = server.client()
    instrument(client)
    wrapped_once = client.chat.completions.create
    instrument(client)
    assert client.chat.completions.create is wrapped_once

    session, collector = make_session()
    with bind_session(session):
        client.chat.completions.create(model="mock-model", messages=[])
    assert len(collector.get_events_by_type("llm_call")) == 1


def test_instrument_rejects_non_client():
    with pytest.raises(TypeError, match="chat.completions.create"):
        instrument(object())
