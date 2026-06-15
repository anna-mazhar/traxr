"""Session binding, thread fallback, concurrency detection, emit() escape hatch."""

import threading
import warnings

import httpx
import openai
import pytest
from _openai_mock import MockOpenAIServer, completion

import traxr
from traxr.capture import CaptureSession, bind_session, current_session, instrument
from traxr.capture.context import suppress_tier0, tier0_suppressed
from traxr.errors import ConcurrentTraceWarning
from traxr.trace.collector import TraceCollector


def make_session(**kwargs):
    collector = TraceCollector(run_label="baseline")
    return CaptureSession(collector, **kwargs), collector


def test_bind_session_sets_and_restores():
    assert current_session() is None
    session, _ = make_session()
    with bind_session(session):
        assert current_session() is session
        inner, _ = make_session()
        with bind_session(inner):
            assert current_session() is inner
        assert current_session() is session
    assert current_session() is None


def test_thread_fallback_reaches_global_session():
    # Contextvars don't propagate into user-spawned threads; the emit must
    # land in the bound session via the process-global fallback.
    server = MockOpenAIServer([completion("threaded")])
    client = instrument(server.client())
    session, collector = make_session()

    def worker():
        client.chat.completions.create(model="mock-model", messages=[])

    with bind_session(session), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # second emitting thread warns; tested below
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert len(collector.get_events_by_type("llm_call")) == 1


def test_emission_from_two_threads_flags_concurrency_once():
    session, _ = make_session()
    with bind_session(session), pytest.warns(ConcurrentTraceWarning) as record:
        session.emit("llm_call", {})

        def worker():
            session.emit("llm_call", {})
            session.emit("llm_call", {})  # second emit must not warn again

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert session.concurrent_detected is True
    concurrency_warnings = [w for w in record if w.category is ConcurrentTraceWarning]
    assert len(concurrency_warnings) == 1
    # H3: stacklevel=2 lands deterministically at the detecting call site in
    # the capture module, regardless of the (varying) caller depth.
    assert concurrency_warnings[0].filename.endswith("capture/context.py")


def test_overlapping_in_flight_calls_flag_concurrency():
    barrier = threading.Barrier(2, timeout=5)
    session, collector = make_session()

    def handler(request: httpx.Request) -> httpx.Response:
        barrier.wait()  # both calls in flight before either returns
        return httpx.Response(200, json=completion("ok"))

    client = instrument(
        openai.OpenAI(
            api_key="test-key",
            base_url="http://traxr.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    def worker():
        client.chat.completions.create(model="mock-model", messages=[])

    with bind_session(session), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConcurrentTraceWarning)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert session.concurrent_detected is True
    assert len(collector.get_events_by_type("llm_call")) == 2


def test_emit_escape_hatch_records_at_current_step():
    session, collector = make_session()
    with bind_session(session):
        session.begin_llm_call()
        session.end_llm_call()
        traxr.emit("plan_revision", {"plan_id": "p1"})

    event = collector.events[0]
    assert event.event_type == "plan_revision"
    assert event.step_num == 1
    assert event.agent_name == "user"
    assert event.payload == {"plan_id": "p1"}


def test_emit_is_noop_outside_a_run():
    traxr.emit("plan_revision", {"plan_id": "p1"})  # must not raise


def test_suppress_tier0_flag():
    assert tier0_suppressed() is False
    with suppress_tier0():
        assert tier0_suppressed() is True
    assert tier0_suppressed() is False
