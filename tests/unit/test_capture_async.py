"""Tier 0 async capture — AsyncOpenAI create + streaming, contextvar propagation."""

import asyncio

from _openai_mock import USAGE, MockOpenAIServer, chunk, completion, tool_call

from traxr.capture import CaptureSession, bind_session, instrument
from traxr.trace.collector import TraceCollector


def make_session(**kwargs):
    collector = TraceCollector(run_label="baseline")
    return CaptureSession(collector, **kwargs), collector


def test_async_create_emits_events():
    server = MockOpenAIServer(
        [completion("async answer", tool_calls=[tool_call("call_a", "lookup", "{}")])]
    )
    client = instrument(server.async_client())
    session, collector = make_session()

    async def run():
        return await client.chat.completions.create(model="mock-model", messages=[])

    with bind_session(session):
        response = asyncio.run(run())

    assert response.choices[0].message.content == "async answer"
    assert [e.event_type for e in collector.events] == ["llm_call", "tool_request"]
    assert collector.events[0].payload["usage"] == USAGE
    assert collector.events[1].payload["tool_name"] == "lookup"


def test_async_streaming_reassembly():
    chunks = [
        chunk(content="as"),
        chunk(content="ync"),
        chunk(finish_reason="stop"),
        chunk(usage=USAGE),
    ]
    server = MockOpenAIServer([chunks])
    client = instrument(server.async_client())
    session, collector = make_session(store_llm_content=True)

    async def run():
        stream = await client.chat.completions.create(model="mock-model", messages=[], stream=True)
        async for _ in stream:
            pass

    with bind_session(session):
        asyncio.run(run())

    payload = collector.events[0].payload
    assert payload["content"] == "async"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"] == USAGE
    assert server.requests[0]["stream_options"] == {"include_usage": True}


def test_contextvar_binding_propagates_into_asyncio_tasks():
    server = MockOpenAIServer([completion("from task")])
    client = instrument(server.async_client())
    session, collector = make_session()

    async def run():
        # The create call happens inside a child task; contextvars propagate
        # into asyncio tasks natively, so no global fallback is needed.
        task = asyncio.create_task(client.chat.completions.create(model="mock-model", messages=[]))
        await task

    with bind_session(session):
        asyncio.run(run())

    assert len(collector.get_events_by_type("llm_call")) == 1
