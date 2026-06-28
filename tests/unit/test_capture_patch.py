"""``patch_openai()`` — class-level capture parity with ``instrument()``."""

import asyncio

from _openai_mock import MockOpenAIServer, completion, tool_call

from traxr.capture import CaptureSession, bind_session, instrument, patch_openai
from traxr.trace.collector import TraceCollector


def two_call_responses():
    return [
        completion("", tool_calls=[tool_call("call_p", "lookup", '{"q": 1}')]),
        completion("done"),
    ]


def drive_two_calls(client):
    client.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "q"}])
    client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "tool", "tool_call_id": "call_p", "content": "result"}],
    )


def trace_shape(collector):
    return [
        (e.event_type, e.step_num, e.payload.get("tool_name"), e.payload.get("finish_reason"))
        for e in collector.events
    ]


def test_patch_openai_parity_with_instrument():
    instrumented_collector = TraceCollector(run_label="baseline")
    server = MockOpenAIServer(two_call_responses())
    with bind_session(CaptureSession(instrumented_collector)):
        drive_two_calls(instrument(server.client()))

    patched_collector = TraceCollector(run_label="baseline")
    server = MockOpenAIServer(two_call_responses())
    with patch_openai(), bind_session(CaptureSession(patched_collector)):
        # Client constructed *inside* the patch context, never instrumented.
        drive_two_calls(server.client())

    assert trace_shape(patched_collector) == trace_shape(instrumented_collector)
    assert len(patched_collector.events) == 4  # llm, tool_req, tool_res, llm


def test_patch_openai_restores_original_create():
    server = MockOpenAIServer([completion("a"), completion("b")])
    collector = TraceCollector(run_label="baseline")
    with bind_session(CaptureSession(collector)):
        with patch_openai():
            server.client().chat.completions.create(model="mock-model", messages=[])
        # Outside the patch context: no capture.
        server.client().chat.completions.create(model="mock-model", messages=[])

    assert len(collector.get_events_by_type("llm_call")) == 1


def test_patch_openai_covers_async_clients():
    server = MockOpenAIServer([completion("async via patch")])
    collector = TraceCollector(run_label="baseline")

    async def run():
        client = server.async_client()
        return await client.chat.completions.create(model="mock-model", messages=[])

    with patch_openai(), bind_session(CaptureSession(collector)):
        response = asyncio.run(run())

    assert response.choices[0].message.content == "async via patch"
    assert len(collector.get_events_by_type("llm_call")) == 1
