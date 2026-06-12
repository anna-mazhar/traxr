"""TraceCollector: emission ordering, queries, serialization, thread safety."""

import threading

from traxr.trace.collector import TraceCollector


def test_emit_assigns_contiguous_sequence_indices():
    c = TraceCollector(run_label="baseline")
    e0 = c.emit("routing_decision", step_num=1, agent_name="router", payload={"chosen_agent": "a"})
    e1 = c.emit("agent_output", step_num=1, agent_name="a", payload={"action": "x"})
    assert (e0.sequence_index, e1.sequence_index) == (0, 1)
    assert c.event_count == 2


def test_emit_computes_content_hash():
    c = TraceCollector(run_label="baseline")
    payload = {"chosen_agent": "a"}
    event = c.emit("routing_decision", step_num=1, agent_name="router", payload=payload)
    from traxr.trace.events import TraceEvent

    assert event.content_hash == TraceEvent.compute_content_hash(payload)


def test_get_events_by_type_and_step():
    c = TraceCollector(run_label="baseline")
    c.emit("routing_decision", step_num=1, agent_name="router", payload={"chosen_agent": "a"})
    c.emit("agent_output", step_num=1, agent_name="a", payload={"action": "x"})
    c.emit("routing_decision", step_num=2, agent_name="router", payload={"chosen_agent": "b"})
    assert len(c.get_events_by_type("routing_decision")) == 2
    assert len(c.get_events_by_type("final_answer")) == 0
    assert [e.event_type for e in c.get_events_by_step(1)] == ["routing_decision", "agent_output"]


def test_to_dict_from_dict_round_trip():
    c = TraceCollector(run_label="baseline")
    c.emit("routing_decision", step_num=1, agent_name="router", payload={"chosen_agent": "a"})
    c.emit("final_answer", step_num=2, agent_name="synth", payload={"answer": "42"})

    restored = TraceCollector.from_dict(c.to_dict())
    assert restored.run_label == "baseline"
    assert restored.events == c.events
    # Sequence counter resumes after the highest restored index.
    next_event = restored.emit(
        "agent_halt", step_num=3, agent_name="synth", payload={"reason": "done"}
    )
    assert next_event.sequence_index == 2


def test_emit_is_thread_safe():
    c = TraceCollector(run_label="baseline")
    n_threads, per_thread = 8, 200
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            c.emit("llm_call", step_num=tid, agent_name=f"agent{tid}", payload={"i": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = n_threads * per_thread
    assert c.event_count == total
    # No duplicated or skipped sequence indices under concurrency.
    assert sorted(e.sequence_index for e in c.events) == list(range(total))
