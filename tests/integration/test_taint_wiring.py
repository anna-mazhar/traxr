"""Regression for finding N-H1: the live EpisodeRunner path must register every
written memory entry with the taint tracker.

`runner.run()` previously passed ``memory_entry=None`` to
``TaintTracker.on_agent_output``, so ``_written_memory_ids`` stayed empty on real
episodes and ``get_taint_in_notes_rate()`` was structurally always 0.0 (its
denominator never populated). This test reconstructs the runner the way
``agents/builtin.py`` does, runs a stub episode that writes memory notes, and
asserts the tracker now records exactly the entries the trace reports written.
"""

from pathlib import Path

from traxr.agents import builtin_agent
from traxr.llm import DeterministicLLMStub
from traxr.trace import TraceCollector

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
QUESTION = "How many rows of data does the table contain?"


def test_written_memory_entries_are_registered_with_taint_tracker():
    # Imports are deferred (mirroring builtin.py) to resolve the mas package
    # import cycle once the public agent path has initialized it.
    csv_path = FIXTURES_DIR / "sample.csv"
    stub = DeterministicLLMStub("identity", final_answer="5", wrong_answer="999")
    agent = builtin_agent(llm=stub)()

    from traxr.mas.agents.specialized_agents import create_specialized_agents
    from traxr.mas.core.episode_spec import (
        EpisodeSpec,
        ExperimentCondition,
        TerminationCriteria,
    )
    from traxr.mas.core.runner import EpisodeRunner
    from traxr.mas.core.state import TaskInput

    task_input = TaskInput(
        task_id="t",
        query=QUESTION,
        expected_answer="5",
        metadata={"file_path": str(csv_path), "file_name": csv_path.name},
    )
    tools = agent._create_tools(task_input)
    agents = create_specialized_agents(llm=stub, tool_executor=tools)
    retrieval = agent._create_retrieval(task_input)
    spec = EpisodeSpec(
        task_id="t",
        seed=agent._seed,
        agent_sequence=tuple(agents),
        termination=TerminationCriteria(max_steps=agent._max_steps, max_tokens=agent._max_tokens),
    )
    collector = TraceCollector(run_label="taint_wiring")
    result = EpisodeRunner(agents=agents, retrieval=retrieval, llm_stub=stub).run(
        spec=spec,
        condition=ExperimentCondition(),
        task_input=task_input,
        trace_collector=collector,
    )

    written_in_trace = {
        e.payload["entry_id"] for e in collector.events if e.event_type == "memory_write"
    }
    assert written_in_trace, "stub scenario should write at least one memory note"
    # N-H1: every written entry is now recorded (was empty when run() passed None).
    assert result.taint_tracker._written_memory_ids == written_in_trace
    # The rate is well-defined (here 0.0: no tainted retrieval, but a populated denominator).
    assert 0.0 <= result.taint_tracker.get_taint_in_notes_rate() <= 1.0
