"""Taint tracking: get_taint_in_notes_rate is a well-defined rate in [0, 1].

Regression for code-review finding M3: the rate divided tainted *written*
entries by memory ids *read*, so it could exceed 1.0, or read 0.0 when taint
sources existed but nothing was read.
"""

from traxr.mas.core.outputs import AgentOutput
from traxr.mas.core.state import MemoryEntry
from traxr.mas.provenance.taint import TaintTracker


def _note(mid: str, agent: str = "critic", step: int = 1) -> MemoryEntry:
    return MemoryEntry(id=mid, agent_name=agent, content="c", entry_type="note", step=step)


def _write(tracker: TaintTracker, mid: str, step: int, read_memory_ids: set[str]) -> None:
    tracker.on_agent_output(
        step=step,
        output=AgentOutput(agent_name="critic", action="write_note", content="c"),
        memory_entry=_note(mid, step=step),
        read_memory_ids=read_memory_ids,
    )


def test_rate_is_zero_when_nothing_written() -> None:
    assert TaintTracker().get_taint_in_notes_rate() == 0.0


def test_rate_well_defined_with_zero_memory_reads() -> None:
    """Old code returned 0.0 here (no on_memory_read calls); the rate must
    reflect that one of two written notes is tainted."""
    tracker = TaintTracker()
    tracker.state.add_tainted_memory("seed", "seed source")
    _write(tracker, "m1", step=1, read_memory_ids={"seed"})  # reads tainted -> tainted write
    _write(tracker, "m2", step=2, read_memory_ids=set())  # clean write
    rate = tracker.get_taint_in_notes_rate()
    assert rate == 0.5
    assert 0.0 <= rate <= 1.0


def test_rate_never_exceeds_one_with_more_writes_than_reads() -> None:
    """Old code (tainted_written / reads) gave 2/1 = 2.0 here."""
    tracker = TaintTracker()
    tracker.on_memory_read(step=1, memory_ids={"r1"}, agent_name="critic")
    tracker.state.add_tainted_memory("seed", "seed source")
    _write(tracker, "m1", step=1, read_memory_ids={"seed"})
    _write(tracker, "m2", step=1, read_memory_ids={"seed"})
    rate = tracker.get_taint_in_notes_rate()
    assert rate == 1.0
    assert 0.0 <= rate <= 1.0
