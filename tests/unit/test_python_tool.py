"""Hardened PythonTool tests — subprocess isolation and the timeout guard."""

import time

import pandas as pd
import pytest

from traxr.mas.tools.python_tool import PythonTool


@pytest.fixture()
def tool() -> PythonTool:
    tool = PythonTool(timeout=15.0)
    tool.set_context({"df": pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})})
    return tool


class TestExecution:
    def test_successful_run_captures_stdout_and_variables(self, tool):
        result = tool.execute("run", code='x = len(df)\nprint(f"Answer: {x}")')
        assert result.success
        assert result.output["stdout"] == "Answer: 3\n"
        assert result.output["variables"] == {"x": "3"}
        assert result.output["stderr"] is None

    def test_context_dataframe_round_trips_to_subprocess(self, tool):
        result = tool.execute("run", code="print(df['b'].sum())")
        assert result.success
        assert "15.0" in result.output["stdout"]

    def test_exception_returns_error_with_traceback(self, tool):
        result = tool.execute("run", code="1 / 0")
        assert not result.success
        assert result.output is None
        assert "ZeroDivisionError" in result.error
        assert "Traceback" in result.error

    def test_unknown_operation_fails_cleanly(self, tool):
        result = tool.execute("nope")
        assert not result.success
        assert "Unknown operation" in result.error

    def test_non_ascii_code_and_output_round_trip(self, tool):
        """N-L1: non-ASCII source and output survive the code/result file
        roundtrip (written and read as UTF-8, not the platform default)."""
        result = tool.execute("run", code='msg = "héllo — naïve ✓"\nprint(msg)')
        assert result.success
        assert result.output["stdout"] == "héllo — naïve ✓\n"
        assert result.output["variables"]["msg"] == "héllo — naïve ✓"

    def test_unpicklable_context_is_skipped_and_reported(self):
        tool = PythonTool(timeout=15.0)
        tool.set_context({"ok": 41, "bad": lambda: None})
        result = tool.execute("run", code="print(ok + 1)")
        assert result.success
        assert result.output["stdout"] == "42\n"
        assert result.metadata["skipped_context_keys"] == ["bad"]


class TestTimeout:
    def test_hanging_script_times_out_cleanly(self):
        """The M3 exit-gate test: an infinite loop must not hang the run."""
        tool = PythonTool(timeout=1.5)
        start = time.monotonic()
        result = tool.execute("run", code="while True:\n    pass")
        elapsed = time.monotonic() - start

        assert not result.success
        assert "Timeout" in result.error
        assert "1.5" in result.error
        assert result.metadata.get("timed_out") is True
        # Killed promptly — not hanging until some outer harness gives up.
        assert elapsed < 10

    def test_tool_remains_usable_after_a_timeout(self):
        tool = PythonTool(timeout=1.5)
        assert not tool.execute("run", code="while True:\n    pass").success
        follow_up = tool.execute("run", code="print('alive')")
        assert follow_up.success
        assert follow_up.output["stdout"] == "alive\n"
