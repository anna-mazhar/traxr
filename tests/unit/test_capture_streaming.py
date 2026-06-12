"""Tier 0 streaming capture — delta reassembly, usage injection, abandonment."""

import pytest
from _openai_mock import USAGE, MockOpenAIServer, chunk

from traxr.capture import CaptureSession, bind_session, instrument
from traxr.errors import TokenUnavailableWarning
from traxr.trace.collector import TraceCollector


def make_session(**kwargs):
    collector = TraceCollector(run_label="baseline")
    return CaptureSession(collector, **kwargs), collector


def content_chunks():
    return [
        chunk(content="Hel"),
        chunk(content="lo "),
        chunk(content="world"),
        chunk(finish_reason="stop"),
        chunk(usage=USAGE),
    ]


def test_streaming_reassembles_content_and_usage():
    server = MockOpenAIServer([content_chunks()])
    client = instrument(server.client())
    session, collector = make_session(store_llm_content=True)
    with bind_session(session):
        stream = client.chat.completions.create(model="mock-model", messages=[], stream=True)
        seen = [c for c in stream]

    assert len(seen) == 5  # chunks pass through unchanged
    assert [e.event_type for e in collector.events] == ["llm_call"]
    payload = collector.events[0].payload
    assert payload["content"] == "Hello world"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"] == USAGE
    # include_usage was injected into the outgoing request.
    assert server.requests[0]["stream_options"] == {"include_usage": True}


def test_streaming_tool_call_argument_delta_reassembly():
    chunks = [
        chunk(
            tool_call_deltas=[
                {"index": 0, "id": "call_s1", "function": {"name": "lookup", "arguments": ""}}
            ]
        ),
        chunk(tool_call_deltas=[{"index": 0, "function": {"arguments": '{"q": '}}]),
        chunk(tool_call_deltas=[{"index": 0, "function": {"arguments": '"x"}'}}]),
        chunk(finish_reason="tool_calls"),
        chunk(usage=USAGE),
    ]
    server = MockOpenAIServer([chunks])
    client = instrument(server.client())
    session, collector = make_session(store_llm_content=True)
    with bind_session(session):
        for _ in client.chat.completions.create(model="mock-model", messages=[], stream=True):
            pass

    assert [e.event_type for e in collector.events] == ["llm_call", "tool_request"]
    llm_event, request_event = collector.events
    assert llm_event.payload["finish_reason"] == "tool_calls"
    assert llm_event.payload["tool_call_names"] == ["lookup"]
    assert request_event.payload["call_id"] == "call_s1"
    assert request_event.payload["arguments"] == '{"q": "x"}'
    # The reassembled request joins the call_id map like the non-streaming path.
    assert session.tool_call_names["call_s1"] == ("lookup", 1)


def test_abandoned_stream_emits_abandoned_llm_call_once():
    server = MockOpenAIServer([content_chunks()])
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session):
        stream = client.chat.completions.create(model="mock-model", messages=[], stream=True)
        next(stream)  # consume one chunk, then drop the stream
        with pytest.warns(TokenUnavailableWarning):  # usage chunk never arrived
            stream.close()
        stream.close()  # closing again must not double-emit

    llm_calls = collector.get_events_by_type("llm_call")
    assert len(llm_calls) == 1
    assert llm_calls[0].payload["finish_reason"] == "abandoned"


def test_streaming_without_usage_warns():
    chunks = [chunk(content="hi"), chunk(finish_reason="stop")]
    server = MockOpenAIServer([chunks])
    client = instrument(server.client())
    session, collector = make_session()
    with bind_session(session), pytest.warns(TokenUnavailableWarning):
        for _ in client.chat.completions.create(
            model="mock-model", messages=[], stream=True, stream_options={}
        ):
            pass
    assert collector.events[0].payload["usage"] is None


def test_user_stream_options_left_untouched():
    server = MockOpenAIServer([content_chunks()])
    client = instrument(server.client())
    session, _ = make_session()
    with bind_session(session):
        for _ in client.chat.completions.create(
            model="mock-model",
            messages=[],
            stream=True,
            stream_options={"include_usage": False},
        ):
            pass
    assert server.requests[0]["stream_options"] == {"include_usage": False}
