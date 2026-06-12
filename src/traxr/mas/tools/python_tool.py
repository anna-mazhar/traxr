"""Python execution tool for pandas operations.

Hardened during extraction (build plan, Known issues #2): the original ran
LLM-generated code through a raw in-process ``exec()``, so an infinite loop
hung the whole experiment. Code now runs in a subprocess with a timeout;
context (DataFrames, file content, ...) is pickled across, and the output
contract ({"stdout", "stderr", "variables"}) is unchanged.
"""

import json
import pickle
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import BaseTool, ToolResult

# Driver executed inside the subprocess. Re-imports pandas/numpy itself
# (modules are not pickled) and mirrors the original in-process semantics:
# captured stdout/stderr plus str() of newly created variables.
_DRIVER = """\
import io
import json
import pickle
import sys
import traceback
import types
from contextlib import redirect_stdout, redirect_stderr

ctx_path, code_path, out_path = sys.argv[1:4]

with open(ctx_path, "rb") as f:
    context = pickle.load(f)
with open(code_path) as f:
    code = f.read()

import pandas as pd
import numpy as np

exec_globals = {"pd": pd, "np": np}
exec_globals.update(context)
base_keys = set(exec_globals) | {"__builtins__"}

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        exec(code, exec_globals)
except Exception as e:
    result = {
        "success": False,
        "error": (
            f"Execution error: {type(e).__name__}: {e}\\n\\n"
            f"Traceback:\\n{traceback.format_exc()}"
        ),
        "stderr": stderr_capture.getvalue(),
    }
else:
    variables = {}
    for k, v in exec_globals.items():
        if k in base_keys or k.startswith("_") or isinstance(v, types.ModuleType):
            continue
        try:
            variables[k] = str(v)
        except Exception:
            variables[k] = "<unprintable>"
    result = {
        "success": True,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "variables": variables,
    }

with open(out_path, "w") as f:
    json.dump(result, f)
"""


class PythonTool(BaseTool):
    """Tool for executing Python code with pandas/numpy in a subprocess.

    Provides a process-isolated execution environment with access to:
    - pandas as pd
    - numpy as np
    - Pre-loaded variables (passed via set_context; pickled across)

    Args:
        timeout: Seconds before the subprocess is killed and the run is
            reported as a failed execution.
    """

    def __init__(self, timeout: float = 30.0):
        super().__init__(name="python")
        self.timeout = timeout
        self.context: Dict[str, Any] = {
            "pd": pd,
            "np": np,
        }

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute Python operation."""
        if operation == "run":
            return self._run(**kwargs)
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation '{operation}'. Available: ['run']"
            )

    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return ["run"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="python",
            description="Execute Python code with pandas and numpy. Pre-loaded DataFrames are available in the execution context.",
            operations={
                "run": OperationSchema(
                    name="run",
                    description="Execute Python code. Has access to pandas (pd), numpy (np), and any pre-loaded DataFrames. Print output is captured in stdout.",
                    parameters=[
                        ToolParameterSchema(name="code", type="string", description="Python code to execute"),
                    ],
                ),
            },
        )

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set execution context (e.g., DataFrames).

        Args:
            context: Dictionary of variables available in execution
        """
        self.context.update(context)

    def _picklable_context(self):
        """Split context into picklable values and skipped key names.

        Modules (pd/np) are dropped silently — the subprocess re-imports
        them. Other unpicklable values are reported via metadata.
        """
        picklable: Dict[str, Any] = {}
        skipped: List[str] = []
        for key, value in self.context.items():
            if isinstance(value, types.ModuleType):
                continue
            try:
                pickle.dumps(value)
            except Exception:
                skipped.append(key)
            else:
                picklable[key] = value
        return picklable, skipped

    def _run(self, code: str) -> ToolResult:
        """Execute Python code in a subprocess with a timeout.

        Args:
            code: Python code string

        Returns:
            ToolResult with execution output
        """
        context, skipped_keys = self._picklable_context()

        with tempfile.TemporaryDirectory(prefix="traxr-python-tool-") as tmp:
            tmp_path = Path(tmp)
            ctx_path = tmp_path / "context.pkl"
            code_path = tmp_path / "code.py"
            driver_path = tmp_path / "driver.py"
            out_path = tmp_path / "result.json"

            with open(ctx_path, "wb") as f:
                pickle.dump(context, f)
            code_path.write_text(code)
            driver_path.write_text(_DRIVER)

            try:
                proc = subprocess.run(
                    [sys.executable, str(driver_path), str(ctx_path), str(code_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        f"Execution error: Timeout: code did not finish within "
                        f"{self.timeout} seconds (possible infinite loop); the "
                        "subprocess was terminated"
                    ),
                    metadata={"code_length": len(code), "timed_out": True},
                )

            if not out_path.exists():
                # Driver itself crashed (e.g. unpicklable context edge case).
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        "Execution error: python subprocess produced no result "
                        f"(exit code {proc.returncode}). Stderr: {proc.stderr[-2000:]}"
                    ),
                    metadata={"code_length": len(code)},
                )

            with open(out_path) as f:
                result = json.load(f)

        if not result.get("success"):
            error_msg = result.get("error", "Execution error: unknown failure")
            stderr_content = result.get("stderr") or ""
            if stderr_content:
                error_msg += f"\nStderr: {stderr_content}"
            return ToolResult(
                success=False,
                output=None,
                error=error_msg,
            )

        stdout_text = result.get("stdout") or ""
        stderr_text = result.get("stderr") or ""
        result_vars = result.get("variables") or None

        output = {
            "stdout": stdout_text if stdout_text else None,
            "stderr": stderr_text if stderr_text else None,
            "variables": result_vars,
        }

        metadata: Dict[str, Any] = {"code_length": len(code)}
        if skipped_keys:
            metadata["skipped_context_keys"] = skipped_keys

        return ToolResult(
            success=True,
            output=output,
            metadata=metadata,
        )
