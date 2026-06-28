"""Specialized agent implementations for GAIA benchmark tasks.

These agents use real LLM (OpenAIClient) for intelligent decision-making.
"""

import re
from typing import Optional, List

from .base import BaseAgent
from .confidence import (
    calculate_tabular_data_confidence,
    calculate_document_confidence,
    calculate_visual_confidence,
    calculate_audio_confidence,
    calculate_web_search_confidence,
    calculate_execution_confidence,
    calculate_answer_specificity_confidence,
    ConfidenceResult,
    ConfidenceFactors,
)
from ..core.types import AgentRole, CitationType
from ..core.state import SharedState, MemoryAccessTracker
from ..core.outputs import AgentOutput, CitationRecord
from ..retrieval.items import RetrievalResult
from ..utils.code_extraction import extract_python_code, validate_python_syntax

import logging

logger = logging.getLogger(__name__)


class DataAnalystAgent(BaseAgent):
    """Agent specialized in analyzing tabular data (CSV, Excel files).

    Expert capabilities for handling complex, messy, real-world spreadsheets:
    - Automatic structure detection (header rows, section markers, blank rows)
    - Merged cell handling via forward-fill
    - Color-coded data interpretation
    - Hidden row/column awareness
    - Multi-sheet support
    - Self-correction loop with rich error context (up to 5 retries)

    Uses structured tool calling via generate_with_tools() so the LLM
    provides code as a structured tool argument rather than free text.
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="data_analyst",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )

    @property
    def uses_tools(self) -> bool:
        return True

    def get_tool_schemas(self) -> list:
        """Return python tool schema for structured tool calling."""
        if self._tool_executor:
            python_tool = self._tool_executor.get_tool("python")
            if python_tool:
                return [python_tool.get_schema()]
        return []

    def _build_structural_context(self, excel_tool, schema, shape, columns, sample_data) -> str:
        """Build a rich structural context string from the ExcelTool.

        This gives the LLM a deep understanding of the spreadsheet structure
        before it writes any analysis code.
        """
        parts = []

        # Basic info
        parts.append(f"Shape: {shape[0]} rows x {shape[1]} columns")
        parts.append(f"Columns (integer indices, NO headers assumed): {columns}")
        parts.append(f"Sheet names: {excel_tool.sheet_names}")
        parts.append(f"Number of sheets: {len(excel_tool.sheet_names)}")

        # Structural summary (blank rows, section markers, colors, etc.)
        struct_result = self._tool_executor.execute("excel", "get_structural_summary")
        if struct_result.success:
            summary = struct_result.output

            # Potential header rows
            if summary.get("potential_header_rows"):
                parts.append("\n--- POTENTIAL HEADER ROWS ---")
                for h in summary["potential_header_rows"]:
                    parts.append(f"  Row {h['row']}: {h['values']}")
            else:
                parts.append("\nNo obvious header row detected in first 20 rows.")

            # Blank rows
            if summary.get("blank_row_count", 0) > 0:
                parts.append(f"\nBlank rows found: {summary['blank_row_count']} (indices: {summary['blank_rows'][:10]}...)")
                parts.append("  -> These may be section separators. Use dropna(how='all') to remove, but check for section markers first.")

            # Section markers
            if summary.get("section_markers"):
                parts.append("\n--- SECTION MARKERS DETECTED ---")
                for m in summary["section_markers"][:10]:
                    parts.append(f"  Row {m['row']}: {m['values']}")
                parts.append("  -> These standalone values are likely category/section headers. Use forward-fill to propagate.")

            # Trailing phantom rows
            if summary.get("trailing_empty_rows", 0) > 0:
                parts.append(f"\nTrailing empty rows: {summary['trailing_empty_rows']} (phantom rows at end of sheet)")

            # Color information
            if summary.get("has_meaningful_colors"):
                parts.append("\n--- COLOR DATA AVAILABLE ---")
                parts.append(f"  Unique colors: {summary.get('unique_colors', [])[:10]}")
                if summary.get("color_distribution"):
                    parts.append(f"  Color distribution (color -> row count): {summary['color_distribution']}")
                parts.append("  -> DataFrame 'df_colors' is available with hex codes aligned to 'df'. Use it to filter/group by color.")
            else:
                parts.append("\nNo meaningful cell colors detected.")

            # Merged cells
            merged_count = summary.get("merged_cell_count", 0)
            if merged_count > 0:
                parts.append(f"\nMerged cells: {merged_count} ranges found")
                parts.append("  -> Merged cells show as NaN in all but top-left cell. Use .ffill() to propagate values.")

            # Hidden rows/cols
            if summary.get("has_hidden_rows") or summary.get("has_hidden_cols"):
                parts.append(f"\nHidden rows: {summary.get('has_hidden_rows', False)}, Hidden columns: {summary.get('has_hidden_cols', False)}")
                parts.append("  -> Hidden content may contain relevant data. Check if the question requires it.")

            # Comments
            if summary.get("has_comments"):
                parts.append("\nCell comments detected — may contain instructions or caveats.")

            # Column analysis
            if summary.get("column_analysis"):
                parts.append("\n--- COLUMN ANALYSIS ---")
                for col_name, info in summary["column_analysis"].items():
                    if isinstance(info, dict):
                        parts.append(f"  Col {col_name}: {info.get('non_null', 0)} non-null, "
                                     f"{info.get('numeric', 0)} numeric, {info.get('string', 0)} string, "
                                     f"sample: {info.get('sample', [])}")
                    else:
                        parts.append(f"  Col {col_name}: {info}")

        # First 20 rows (raw data)
        parts.append(f"\n--- FIRST 20 ROWS (raw, no headers) ---")
        if sample_data:
            for i, row in enumerate(sample_data[:20]):
                parts.append(f"  Row {i}: {row}")

        # Tail (last 5 rows to show data end)
        tail_result = self._tool_executor.execute("excel", "tail", n=5)
        if tail_result.success and tail_result.output:
            parts.append(f"\n--- LAST 5 ROWS ---")
            for i, row in enumerate(tail_result.output):
                parts.append(f"  {row}")

        return "\n".join(parts)

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Analyze tabular data using ExcelTool with self-correction."""
        # Read existing memory
        existing_analyses = memory_tracker.read_memory(entry_types=["analysis"])

        # Check if ExcelTool is available
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for data analysis",
                metadata={"error": "no_tool_executor"},
            )

        excel_tool = self._tool_executor.get_tool("excel")
        if not excel_tool:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="ExcelTool not available",
                metadata={"error": "no_excel_tool"},
            )

        # Get schema
        schema_result = self._tool_executor.execute("excel", "get_schema")
        if not schema_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to load Excel schema: {schema_result.error}",
                metadata={"error": "schema_failed"},
            )

        schema = schema_result.output
        columns = schema['columns']
        shape = schema['shape']

        # Show more rows (20) to better understand structure
        head_result = self._tool_executor.execute("excel", "head", n=20)
        sample_data = head_result.output if head_result.success else []

        # Build comprehensive structural context
        structural_context = self._build_structural_context(
            excel_tool, schema, shape, columns, sample_data
        )

        # Determine data availability mode
        has_colors = excel_tool.df_colors is not None and not excel_tool.df_colors.empty
        has_text_data = shape[0] > 0 and shape[1] > 0

        # Get actual column info for clear prompting
        actual_columns = list(excel_tool.df.columns)
        columns_are_strings = all(isinstance(c, str) for c in actual_columns)
        headers_already_set = columns_are_strings and len(actual_columns) > 0

        # Classify the spreadsheet type
        if has_text_data and has_colors:
            data_mode = "TEXT_AND_COLORS"
            color_shape = excel_tool.df_colors.shape
            header_warning = "IMPORTANT: Headers are already set. Do NOT modify df.columns!" if headers_already_set else ""
            data_description = f"""=== DATA AVAILABLE ===
MODE: Text data WITH color formatting

`df` (text data): {shape[0]} rows × {shape[1]} columns
  - Column names: {actual_columns}
  - Access columns using: df['{actual_columns[0]}'] (string names)
  {header_warning}

`df_colors` (color data): {color_shape[0]} rows × {color_shape[1]} columns
  - Contains 6-digit hex color codes (e.g., '00FF00' for green)
  - Use when task mentions colors or visual formatting
"""
        elif has_colors and not has_text_data:
            data_mode = "COLORS_ONLY"
            color_shape = excel_tool.df_colors.shape
            data_description = f"""=== DATA AVAILABLE ===
MODE: Color-coded grid (NO text data)

WARNING: `df` is EMPTY - DO NOT USE IT!

`df_colors` (the ONLY data source): {color_shape[0]} rows × {color_shape[1]} columns
  - This spreadsheet uses ONLY colors to represent data (like a visual map/grid)
  - Each cell contains a 6-digit hex color code (e.g., '00FF00' for green)
  - Common colors: '00FF00'=green, 'FF0000'=red, 'FFFFFF'=white, '000000'=black
  - Treat each cell as a pixel/plot in a grid

CRITICAL: You MUST use `df_colors` for ALL analysis. Examples:
  # Count cells of each color
  df_colors.stack().value_counts()

  # Find positions of green cells
  green_positions = np.argwhere(df_colors.values == '00FF00')

  # Check adjacency/connectivity of colored regions
  from scipy import ndimage  # if needed for connected components
"""
        else:
            data_mode = "TEXT_ONLY"
            header_warning = "IMPORTANT: Headers are already set. Do NOT modify df.columns!" if headers_already_set else ""
            data_description = f"""=== DATA AVAILABLE ===
MODE: Standard text/numeric data (no colors)

`df` (text data): {shape[0]} rows × {shape[1]} columns
  - Column names: {actual_columns}
  - Access columns using: df['{actual_columns[0] if actual_columns else 0}']
  {header_warning}
"""

        # Self-correction loop: Try up to 5 times
        max_attempts = 5
        attempt_history = []
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0

        for attempt in range(max_attempts):
            if attempt == 0:
                prompt = f"""{data_description}

=== STRUCTURAL ANALYSIS ===
{structural_context}

=== TASK ===
{state.task_input.query}

=== INSTRUCTIONS ===
{"USE `df_colors` ONLY - `df` is empty!" if data_mode == "COLORS_ONLY" else "Based on the structural analysis above:"}
1. {"Work with df_colors as your data source" if data_mode == "COLORS_ONLY" else "Filter/query the data to find the relevant rows"}
2. {"Analyze the color grid to answer the question" if data_mode == "COLORS_ONLY" else "Clean data if needed: remove blank rows, strip whitespace, coerce types"}
3. {"For spatial/connectivity questions, treat the grid as a 2D map" if data_mode == "COLORS_ONLY" else "Extract the answer from the filtered data"}
4. Solve the task and print the answer with prefix: print(f"Answer: {{answer}}")
5. Optionally print verification data AFTER the answer line.

IMPORTANT: Always print the final answer as: print(f"Answer: {{your_answer}}")

Write complete, executable Python code.
Use the python tool to execute your code."""
            else:
                last = attempt_history[-1]

                # Build attempt summary
                attempt_summary = []
                for i, hist in enumerate(attempt_history, 1):
                    level = hist.get('success_level', 'unknown')
                    if level == 'syntax_error':
                        attempt_summary.append(f"  Attempt {i}: Syntax error — code didn't parse")
                    elif level == 'execution_error':
                        attempt_summary.append(f"  Attempt {i}: Runtime error — {hist['error'][:200]}")
                    elif level == 'no_output':
                        attempt_summary.append(f"  Attempt {i}: Ran successfully but produced no print() output")
                    elif level == 'partial_success':
                        attempt_summary.append(f"  Attempt {i}: Produced output: {hist.get('result', '')[:100]}")

                attempts_info = "\n".join(attempt_summary)

                # Build mode-specific retry instructions
                if data_mode == "COLORS_ONLY":
                    mode_reminder = """CRITICAL REMINDER: This is a COLORS-ONLY spreadsheet!
- `df` is EMPTY - do NOT use it!
- Use `df_colors` for ALL analysis
- df_colors contains hex color codes (e.g., '00FF00' for green)"""
                elif data_mode == "TEXT_AND_COLORS":
                    mode_reminder = f"""REMINDER: Both `df` (text data) and `df_colors` (color codes) are available.
Column names in df: {actual_columns}
DO NOT modify df.columns - headers are already set!"""
                else:
                    mode_reminder = f"""Column names in df: {actual_columns}
Access columns using: df['{actual_columns[0] if actual_columns else 0}']
DO NOT modify df.columns - headers are already set!"""

                prompt = f"""{mode_reminder}

=== STRUCTURAL ANALYSIS ===
{structural_context}

=== TASK ===
{state.task_input.query}

=== PREVIOUS ATTEMPTS ===
{attempts_info}

=== LAST ATTEMPT CODE ===
```python
{last['code']}
```
Error/Issue: {last['error']}

=== INSTRUCTIONS ===
Analyze what went wrong. Common fixes:
{"- REMEMBER: Use df_colors, NOT df!" if data_mode == "COLORS_ONLY" else f"- If KeyError: check column names are exactly: {actual_columns}"}
- If empty result from filter: print unique values of the filter column to debug
- If type error: use pd.to_numeric() or .astype(str) before comparing
- If no output: make sure you have print() statements

IMPORTANT: Always print the final answer as: print(f"Answer: {{your_answer}}")

Write corrected, complete Python code. Use the python tool to execute your code."""

            # Get LLM code via structured tool calling
            tool_schemas = self.get_tool_schemas()
            plan_response = self._llm.generate_with_tools(
                prompt, tools=tool_schemas, response_type="data_analyst"
            )

            # Track cumulative tokens
            cumulative_prompt_tokens += plan_response.prompt_tokens
            cumulative_completion_tokens += plan_response.completion_tokens

            # Extract code from tool call or fall back to text extraction
            code = None
            if plan_response.has_tool_calls:
                for call in plan_response.tool_calls:
                    if call.operation == "run" and "code" in call.arguments:
                        code = call.arguments["code"]
                        break

            # Fallback: extract from text response
            if not code and plan_response.content:
                code = extract_python_code(plan_response.content)

            # Handle case where no code was extracted at all
            if not code or not code.strip():
                attempt_history.append({
                    'attempt': attempt + 1,
                    'code': code or '',
                    'error': "No code extracted from LLM response",
                    'result': None,
                    'produced_output': False,
                    'success_level': 'syntax_error',
                    'prompt_tokens': plan_response.prompt_tokens,
                    'completion_tokens': plan_response.completion_tokens,
                })
                continue

            # Validate syntax
            is_valid, syntax_error = validate_python_syntax(code)
            if not is_valid:
                attempt_history.append({
                    'attempt': attempt + 1,
                    'code': code,
                    'error': f"Syntax error: {syntax_error}",
                    'result': None,
                    'produced_output': False,
                    'success_level': 'syntax_error',
                    'prompt_tokens': plan_response.prompt_tokens,
                    'completion_tokens': plan_response.completion_tokens,
                })
                continue

            # Execute
            python_result = self._tool_executor.execute("python", "run", code=code)

            if python_result.success and python_result.output.get('stdout'):
                # Success
                break
            else:
                if not python_result.success:
                    success_level = 'execution_error'
                    produced_output = False
                elif python_result.output.get('stdout'):
                    success_level = 'partial_success'
                    produced_output = True
                else:
                    success_level = 'no_output'
                    produced_output = False

                attempt_history.append({
                    'attempt': attempt + 1,
                    'code': code,
                    'error': python_result.error if not python_result.success else "No output produced (missing print() statement?)",
                    'result': python_result.output.get('stdout') if python_result.success else None,
                    'produced_output': produced_output,
                    'success_level': success_level,
                    'prompt_tokens': plan_response.prompt_tokens,
                    'completion_tokens': plan_response.completion_tokens,
                })

                if attempt == max_attempts - 1:
                    python_result = python_result
                    code = code

        # Build final analysis output
        analysis_parts = []
        analysis_parts.append(f"Task: {state.task_input.query}")
        analysis_parts.append(f"\nData: {shape[0]} rows x {shape[1]} columns, {len(excel_tool.sheet_names)} sheet(s)")

        if existing_analyses:
            analysis_parts.append(f"(Previous analyses: {len(existing_analyses)})")

        if len(attempt_history) > 0:
            analysis_parts.append(f"\nRequired {len(attempt_history) + 1} attempt(s)")
            for hist in attempt_history:
                analysis_parts.append(f"  Attempt {hist['attempt']}: {hist['success_level']} — {hist['error'][:150]}")

        analysis_parts.append(f"\nFinal code:")
        analysis_parts.append(f"```python\n{code}\n```")

        if python_result.success:
            stdout = python_result.output.get('stdout', '')
            stderr = python_result.output.get('stderr', '')

            analysis_parts.append(f"\nResult:")
            if stdout:
                analysis_parts.append(stdout.strip())
            if stderr:
                analysis_parts.append(f"\nWarnings: {stderr.strip()}")

            if stdout:
                lines = stdout.strip().split('\n')
                # Look for line starting with "Answer:" prefix
                final_answer = "No output"
                for line in lines:
                    if line.strip().startswith("Answer:"):
                        final_answer = line.strip().replace("Answer:", "").strip()
                        break
                # Fallback to first non-empty line if no "Answer:" prefix found
                if final_answer == "No output":
                    non_empty = [l.strip() for l in lines if l.strip()]
                    final_answer = non_empty[0] if non_empty else "No output"
                analysis_parts.append(f"\nFinal Answer: {final_answer}")
        else:
            analysis_parts.append(f"\nExecution Error: {python_result.error}")
            analysis_parts.append("\nFallback — raw data:")
            head_result = self._tool_executor.execute("excel", "head", n=20)
            if head_result.success:
                analysis_parts.append(f"First 20 rows: {head_result.output}")

        final_analysis = "\n".join(analysis_parts)

        # Calculate confidence using multiple signals
        stdout = python_result.output.get('stdout', '') if python_result.success else ''

        # Count null values if we can get the info
        null_count = 0
        try:
            if excel_tool and hasattr(excel_tool, 'df') and excel_tool.df is not None:
                null_count = excel_tool.df.isnull().sum().sum()
        except Exception:
            pass

        confidence_result = calculate_tabular_data_confidence(
            shape=shape,
            null_count=null_count,
            execution_success=python_result.success,
            attempts=len(attempt_history) + 1,
            output_text=stdout,
            raw_text=structural_context,
            llm=self._llm,
        )
        logger.debug(f"[DataAnalyst] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Status based on confidence
        if not python_result.success:
            entry_status = "failed_attempt"
        elif confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        entry_confidence = confidence_result.score

        # Write to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=final_analysis,
            entry_type="analysis",
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "analysis_type": "tabular_data",
                "code_executed": code,
                "execution_success": python_result.success,
                "attempts": len(attempt_history) + 1,
                "has_colors": has_colors,
            },
            status=entry_status,
            confidence=entry_confidence,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_analysis",
            content=final_analysis,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            # Use cumulative tokens across all retry attempts
            prompt_tokens=cumulative_prompt_tokens,
            completion_tokens=cumulative_completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "data_analysis",
                "llm_generated_code": True,
                "code_length": len(code),
                "attempts": len(attempt_history) + 1,
                # For enhanced tracing
                "code_executed": code,
                "code_output": python_result.output.get('stdout', '') if python_result.success else None,
                "code_error": python_result.error if not python_result.success else None,
                "tool_calls": [
                    {
                        "tool_name": "python",
                        "operation": "run",
                        "params": {"code_length": len(code)},
                        "success": python_result.success,
                        "output": python_result.output.get('stdout', '')[:500] if python_result.success else None,
                        "error": python_result.error,
                    }
                ],
                # Detailed retry/attempt history for analysis
                "attempt_history": [
                    {
                        "attempt": h["attempt"],
                        "success_level": h["success_level"],
                        "error": h.get("error", "")[:200],
                        "prompt_tokens": h.get("prompt_tokens", 0),
                        "completion_tokens": h.get("completion_tokens", 0),
                    }
                    for h in attempt_history
                ],
                "cumulative_prompt_tokens": cumulative_prompt_tokens,
                "cumulative_completion_tokens": cumulative_completion_tokens,
                # Confidence assessment for routing decisions
                **confidence_result.to_dict(),
            },
        )


class PythonAgent(BaseAgent):
    """Agent specialized in Python code execution and numerical computation.

    Capabilities:
    - Write and execute Python code using PythonTool
    - Perform complex calculations
    - Data processing and manipulation with pandas
    - Algorithmic problem solving

    Uses structured tool calling via generate_with_tools() so the LLM
    provides code as a structured tool argument rather than free text.
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="python_executor",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )

    @property
    def uses_tools(self) -> bool:
        return True

    def get_tool_schemas(self) -> list:
        """Return python tool schema for structured tool calling."""
        if self._tool_executor:
            python_tool = self._tool_executor.get_tool("python")
            if python_tool:
                return [python_tool.get_schema()]
        return []

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Execute Python code using PythonTool."""
        # Read existing memory for context
        calculations = memory_tracker.read_memory(entry_types=["calculation"])
        analyses = memory_tracker.read_memory(entry_types=["analysis"])

        # Check if PythonTool is available
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for Python execution",
                metadata={"error": "no_tool_executor"},
            )

        python_tool = self._tool_executor.get_tool("python")
        if not python_tool:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="PythonTool not available",
                metadata={"error": "no_python_tool"},
            )

        prompt_parts = [
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
        ]

        if calculations:
            prompt_parts.append("Previous calculations:")
            for calc in calculations[-3:]:
                prompt_parts.append(f"  - {calc.content[:100]}...")
            prompt_parts.append("")

        if analyses:
            prompt_parts.append("Context from data_analyst:")
            for analysis in analyses[-2:]:
                # Show brief summary only - keep prompt small
                lines = analysis.content.split('\n')
                summary = lines[0] if lines else analysis.content[:100]
                prompt_parts.append(f"  {summary}")
            prompt_parts.append("")

        # Check what data is available
        excel_tool = self._tool_executor.get_tool("excel") if self._tool_executor else None
        python_tool = self._tool_executor.get_tool("python") if self._tool_executor else None

        has_df = False
        has_colors = False
        has_file_content = False
        has_file_code = False
        has_document_data = False

        if python_tool:
            has_df = 'df' in python_tool.context and python_tool.context.get('df') is not None
            has_colors = 'df_colors' in python_tool.context
            has_file_content = 'file_content' in python_tool.context
            has_file_code = 'file_code' in python_tool.context
            has_document_data = 'document_data' in python_tool.context

        # Tell LLM what variables are available
        prompt_parts.append("AVAILABLE VARIABLES:")

        if has_df:
            df = python_tool.context.get('df')
            if df is not None and hasattr(df, 'shape'):
                prompt_parts.append(f"  - df: DataFrame with {df.shape[0]} rows × {df.shape[1]} columns")
            else:
                prompt_parts.append("  - df: DataFrame (empty or not loaded)")

        if has_colors:
            df_colors = python_tool.context.get('df_colors')
            if df_colors is not None and hasattr(df_colors, 'shape'):
                prompt_parts.append(f"  - df_colors: DataFrame with {df_colors.shape[0]} rows × {df_colors.shape[1]} columns (hex color codes)")
                if not has_df or (has_df and python_tool.context.get('df') is not None and python_tool.context.get('df').empty):
                    prompt_parts.append("    NOTE: df is empty but df_colors has data - use df_colors for color-based analysis!")

        if has_document_data:
            doc_data = python_tool.context.get('document_data', {})
            doc_summary = python_tool.context.get('document_summary', '')
            prompt_parts.append(f"  - document_data: Dictionary with parsed document sections")
            prompt_parts.append(f"  - document_summary: String with document structure overview")
            prompt_parts.append("")
            prompt_parts.append("DOCUMENT STRUCTURE:")
            if doc_summary:
                for line in doc_summary.split('\n'):
                    prompt_parts.append(f"  {line}")
            prompt_parts.append("")
            prompt_parts.append("PARSED DATA (document_data):")
            for key, value in doc_data.items():
                if value is None:
                    prompt_parts.append(f"  {key}: (empty/not specified)")
                elif isinstance(value, list):
                    prompt_parts.append(f"  {key}: {value}")
                elif isinstance(value, dict):
                    prompt_parts.append(f"  {key}:")
                    for k, v in value.items():
                        prompt_parts.append(f"    {k}: {v}")
                else:
                    prompt_parts.append(f"  {key}: {value[:200]}..." if len(str(value)) > 200 else f"  {key}: {value}")
            prompt_parts.append("")
            prompt_parts.append("USE document_data dictionary directly in your code!")

        if has_file_content and not has_document_data:
            content = python_tool.context.get('file_content', '')
            prompt_parts.append(f"  - file_content: String variable containing the entire text file ({len(content)} chars)")

            # For small documents (<2000 chars), show the FULL content
            # For larger ones, show a longer preview
            if len(content) < 2000:
                prompt_parts.append(f"    FULL CONTENT of file_content:")
                prompt_parts.append(f"    ```")
                for line in content.split('\n'):
                    prompt_parts.append(f"    {line}")
                prompt_parts.append(f"    ```")
            else:
                # Show first 20 lines for larger files
                preview_lines = content.split('\n')[:20]
                prompt_parts.append(f"    PREVIEW of file_content (first 20 lines):")
                prompt_parts.append(f"    ```")
                for line in preview_lines:
                    prompt_parts.append(f"    {line[:100]}{'...' if len(line) > 100 else ''}")
                prompt_parts.append(f"    ```")

            prompt_parts.append(f"    USE THIS VARIABLE DIRECTLY - do NOT define your own file_content!")

        if has_file_code:
            code = python_tool.context.get('file_code', '')
            preview_lines = code.split('\n')[:5]
            prompt_parts.append(f"  - file_code: Python code from the attached .py file ({len(code)} chars)")
            prompt_parts.append(f"    PREVIEW:")
            for line in preview_lines[:3]:
                prompt_parts.append(f"    {line[:80]}")
            prompt_parts.append(f"    To execute the file's code: exec(file_code)")

        # Zip archive context - show extracted file paths
        has_zip = 'zip_extract_dir' in python_tool.context and python_tool.context.get('zip_extract_dir')
        if has_zip:
            extract_dir = python_tool.context.get('zip_extract_dir')
            file_list = python_tool.context.get('zip_file_list', [])
            prompt_parts.append(f"  - zip_extract_dir: '{extract_dir}' (temp directory with extracted files)")
            prompt_parts.append(f"  - zip_file_list: list of {len(file_list)} extracted files")
            prompt_parts.append("")
            prompt_parts.append("EXTRACTED ZIP FILES (use these exact paths):")
            for f in file_list:
                full_path = f"{extract_dir}/{f['filename']}"
                prompt_parts.append(f"  - '{full_path}' ({f['file_size']} bytes)")
            prompt_parts.append("")
            prompt_parts.append("IMPORTANT: The zip has been extracted. Use the full paths above to access files.")
            prompt_parts.append("Example: tree = ET.parse('{}/your_file.xml')".format(extract_dir))

        if not has_df and not has_colors and not has_file_content and not has_file_code and not has_document_data and not has_zip:
            prompt_parts.append("  - No pre-loaded data available. Write standalone code.")

        prompt_parts.append("")

        # Show last error if any (so agent can avoid repeating the same mistake)
        last_error = state.task_input.metadata.get('last_error')
        if last_error:
            prompt_parts.append("PREVIOUS ERROR (avoid this mistake):")
            prompt_parts.append(f"  Type: {last_error.get('error_type', 'unknown')}")
            if last_error.get('missing_file'):
                prompt_parts.append(f"  Missing file: {last_error.get('missing_file')}")
            if last_error.get('missing_key'):
                prompt_parts.append(f"  Missing key: {last_error.get('missing_key')}")
            if last_error.get('available_keys'):
                prompt_parts.append(f"  Available keys: {last_error.get('available_keys')[:10]}")
            prompt_parts.append(f"  Suggestion: {last_error.get('suggestion', 'N/A')}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Write Python code to solve this task.",
            "",
            "IMPORTANT:",
            "- Use ONLY the variables listed above - they are pre-loaded",
            "- Do NOT try to load files with pd.read_excel() or open()",
            "- Use print() to output the final answer",
            "- Use the python tool to execute your code",
        ])

        prompt = "\n".join(prompt_parts)

        # Use structured tool calling
        tool_schemas = self.get_tool_schemas()
        response = self._llm.generate_with_tools(
            prompt, tools=tool_schemas, response_type="python_executor"
        )

        # Extract code from tool call or fall back to text extraction
        code = None
        if response.has_tool_calls:
            for call in response.tool_calls:
                if call.operation == "run" and "code" in call.arguments:
                    code = call.arguments["code"]
                    break

        # Fallback: extract code from text response
        if not code and response.content:
            code = extract_python_code(response.content)

        # Check if code extraction detected a refusal/apology
        if not code or not code.strip():
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"LLM refused to write code. Response was:\n{(response.content or '')[:500]}",
                metadata={"error": "refusal_detected", "llm_response": response.content},
            )

        # Execute the code using PythonTool
        exec_result = self._tool_executor.execute("python", "run", code=code)

        # Build final output combining code + execution results
        output_parts = [f"Code:\n```python\n{code}\n```"]

        if exec_result.success:
            if exec_result.output.get("stdout"):
                output_parts.append(f"\nOutput:\n{exec_result.output['stdout']}")
            if exec_result.output.get("stderr"):
                output_parts.append(f"\nWarnings:\n{exec_result.output['stderr']}")
        else:
            output_parts.append(f"\nExecution Error: {exec_result.error}")

        final_content = "\n".join(output_parts)

        # Calculate confidence using proper execution and answer quality assessment
        # Start with execution confidence
        exec_confidence_score, exec_signals = calculate_execution_confidence(
            success=exec_result.success,
            attempts=1,
            error_message=exec_result.error if not exec_result.success else None,
            warnings=[exec_result.output.get('stderr', '')] if exec_result.success and exec_result.output.get('stderr') else None,
        )

        # Assess answer specificity from output
        output_text = exec_result.output.get('stdout', '') if exec_result.success else ''
        spec_confidence_score, spec_signals = calculate_answer_specificity_confidence(output_text)

        # Combine into ConfidenceResult-like structure
        all_signals = exec_signals + spec_signals
        confidence_factors = ConfidenceFactors(
            parsing_quality=1.0,  # Code was generated successfully
            data_completeness=1.0 if exec_result.success else 0.3,
            result_consistency=1.0,
            answer_specificity=spec_confidence_score,
            execution_success=exec_confidence_score,
        )
        confidence_score = confidence_factors.overall()
        confidence_result = ConfidenceResult(
            score=confidence_score,
            factors=confidence_factors,
            signals=all_signals,
        )
        logger.debug(f"[PythonExecutor] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if not exec_result.success:
            entry_status = "failed_attempt"
        elif confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        # Write calculation to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=final_content,
            entry_type="calculation",
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "code_type": "python",
                "execution_success": exec_result.success,
            },
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_calculation",
            content=final_content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "python_execution",
                "tool_executed": exec_result.success,
                # For enhanced tracing
                "code_executed": code,
                "code_output": exec_result.output.get('stdout', '') if exec_result.success else None,
                "code_error": exec_result.error if not exec_result.success else None,
                "tool_calls": [
                    {
                        "tool_name": "python",
                        "operation": "run",
                        "params": {"code_length": len(code)},
                        "success": exec_result.success,
                        "output": exec_result.output.get('stdout', '')[:500] if exec_result.success else None,
                        "error": exec_result.error,
                    }
                ],
                **confidence_result.to_dict(),
            },
        )


class CalculatorAgent(BaseAgent):
    """Agent specialized in mathematical calculations and numerical operations.

    Capabilities:
    - Perform arithmetic operations using CalculatorTool
    - Formula evaluation with safe eval
    - Statistical calculations (sum, mean, min, max)
    - Rounding and precision control

    Uses structured tool calling via generate_with_tools() for reliable
    tool invocation instead of regex-based parsing.
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="calculator",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )

    @property
    def uses_tools(self) -> bool:
        return True

    def get_tool_schemas(self) -> list:
        """Return calculator tool schema for structured tool calling."""
        if self._tool_executor:
            calc_tool = self._tool_executor.get_tool("calculator")
            if calc_tool:
                return [calc_tool.get_schema()]
        return []

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Perform mathematical calculations using CalculatorTool."""
        # Read memory for numerical data
        all_entries = memory_tracker.get_all_memory()

        # Check if CalculatorTool is available
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for calculations",
                metadata={"error": "no_tool_executor"},
            )

        calc_tool = self._tool_executor.get_tool("calculator")
        if not calc_tool:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="CalculatorTool not available",
                metadata={"error": "no_calculator_tool"},
            )

        prompt_parts = [
            "You are a mathematical computation expert with access to a calculator tool.",
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
            "Available information from memory:",
        ]

        for entry in all_entries:
            if entry.entry_type in ["analysis", "note", "calculation", "visual_analysis", "audio_analysis"]:
                prompt_parts.append(f"  [{entry.entry_type}] {entry.content[:500]}...")

        prompt_parts.extend([
            "",
            "Instructions:",
            "1. Identify the mathematical expression or computation needed",
            "2. Use the calculator tool to perform the computation",
            "3. If no computation is needed, explain why",
        ])

        prompt = "\n".join(prompt_parts)

        # Use structured tool calling
        tool_schemas = self.get_tool_schemas()
        response = self._llm.generate_with_tools(
            prompt, tools=tool_schemas, response_type="calculator"
        )

        # Execute any tool calls from the response
        tool_calls = []
        calc_result = None
        operation_used = None

        if response.has_tool_calls:
            for call in response.tool_calls:
                calc_result = self._tool_executor.execute(
                    call.tool_name, call.operation, **call.arguments
                )
                operation_used = f"{call.operation}({call.arguments})"
                tool_calls.append({
                    "tool_name": call.tool_name,
                    "operation": call.operation,
                    "params": call.arguments,
                    "success": calc_result.success if calc_result else False,
                    "output": str(calc_result.output) if calc_result and calc_result.success else None,
                    "error": calc_result.error if calc_result and not calc_result.success else None,
                })

        # Build final output
        output_parts = []
        if response.content:
            output_parts.append(response.content)

        if calc_result and calc_result.success:
            output_parts.append(f"\nCalculation Result: {calc_result.output}")
        elif calc_result:
            output_parts.append(f"\nCalculation Error: {calc_result.error}")

        final_content = "\n".join(output_parts) if output_parts else "No calculation performed"

        # Write calculation to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=final_content,
            entry_type="calculation",
            metadata={
                "step": step_num,
                "calculation_type": "mathematical",
                "tool_operation": operation_used,
                "tool_calls": tool_calls,
            },
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_calculation",
            content=final_content,
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "calculation",
                "tool_used": operation_used is not None,
                "tool_calls": tool_calls,
            },
        )


class VisualAnalystAgent(BaseAgent):
    """Agent specialized in analyzing visual content (images, charts, graphs).

    Capabilities:
    - Get images via ImageTool
    - Describe visual content using vision model
    - Analyze charts and graphs
    - Extract information from images
    - Interpret spatial relationships
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="visual_analyst",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )
        self._vision_llm = self._resolve_vision_llm(llm)

    @staticmethod
    def _resolve_vision_llm(llm):
        """Return a vision-capable LLM based on the provider type.

        - OpenAI: uses gpt-4o (natively supports vision)
        - Other (stub, etc.): returns the original LLM as-is

        (The TinkerClient branch was left behind during extraction —
        heavy optional deps; see the build plan's reuse ledger.)
        """
        try:
            from traxr.llm.openai_compat import OpenAICompatibleClient as OpenAIClient
            if isinstance(llm, OpenAIClient):
                # OpenAI's generate_with_image already upgrades to gpt-4o
                # internally, but we pin it explicitly so the model choice
                # is visible in logs/metadata.
                vision = OpenAIClient(
                    model="gpt-4o",
                    seed=llm.seed,
                    api_key=llm.client.api_key,
                )
                return vision
        except ImportError:
            pass

        # Stub or unknown — return as-is
        return llm

    def _get_gpt4o_fallback(self):
        """Create a GPT-4o client for vision fallback when Tinker VL fails.

        Returns OpenAIClient with gpt-4o model, or None if OPENAI_API_KEY not available.
        """
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            from traxr.llm.openai_compat import OpenAICompatibleClient as OpenAIClient
            return OpenAIClient(
                model="gpt-4o",
                seed=getattr(self._llm, 'seed', None),
                api_key=api_key,
            )
        except ImportError:
            return None

    @property
    def uses_tools(self) -> bool:
        return True

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Analyze visual content using ImageTool and a vision model."""
        existing_visual = memory_tracker.read_memory(entry_types=["visual_analysis"])

        # Check if ImageTool is available
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for visual analysis",
                metadata={"error": "no_tool_executor"},
            )

        image_tool = self._tool_executor.get_tool("image")
        if not image_tool:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="ImageTool not available",
                metadata={"error": "no_image_tool"},
            )

        # Get image metadata and base64
        metadata_result = self._tool_executor.execute("image", "get_metadata")
        base64_result = self._tool_executor.execute("image", "get_base64")

        # Track tool calls for divergence analysis
        tool_calls = [
            {
                "tool_name": "image",
                "operation": "get_metadata",
                "params": {},
                "success": metadata_result.success,
                "output": str(metadata_result.output)[:200] if metadata_result.success else None,
                "error": metadata_result.error if not metadata_result.success else None,
            },
            {
                "tool_name": "image",
                "operation": "get_base64",
                "params": {},
                "success": base64_result.success,
                "output": "[base64 image data]" if base64_result.success else None,
                "error": base64_result.error if not base64_result.success else None,
            },
        ]

        if not base64_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to load image: {base64_result.error}",
                metadata={"error": "image_load_failed"},
            )

        # Extract base64 data and media type
        image_data = base64_result.output
        image_base64 = image_data["base64"]
        media_type = image_data["media_type"]

        # Build prompt for vision analysis
        image_info = f"{metadata_result.output['format']} image, {metadata_result.output['width']}x{metadata_result.output['height']} pixels" if metadata_result.success else "Image"

        prompt = f"""Task: {state.task_input.query}

Image information: {image_info}

Analyze the visual content and extract information relevant to answering the task question. Focus on:
1. Text, numbers, or labels visible in the image
2. Charts, graphs, or data visualizations
3. Key visual elements and their relationships
4. Any patterns or details that help answer the question

Provide a detailed visual analysis with specific observations:"""

        # Use vision-capable LLM (resolved at init to gpt-4o / qwen3-vl)
        vision_llm = self._vision_llm
        vision_model = getattr(vision_llm, 'model', 'unknown')
        logger.debug(f"[VisualAnalyst] Using vision model: {vision_model}")

        if hasattr(vision_llm, 'generate_with_image'):
            response = vision_llm.generate_with_image(
                prompt=prompt,
                image_base64=image_base64,
                media_type=media_type,
                response_type="visual_analysis",
            )
            vision_used = True

            # Check if response is empty or an error - fallback to GPT-4o if using Tinker
            needs_fallback = (
                not response.content or
                not response.content.strip() or
                response.content.startswith("[Tinker Vision")
            )

            if needs_fallback and "qwen" in vision_model.lower():
                logger.debug(f"[VisualAnalyst] Tinker VL returned empty/error, falling back to GPT-4o")
                fallback_llm = self._get_gpt4o_fallback()
                if fallback_llm and hasattr(fallback_llm, 'generate_with_image'):
                    response = fallback_llm.generate_with_image(
                        prompt=prompt,
                        image_base64=image_base64,
                        media_type=media_type,
                        response_type="visual_analysis",
                    )
                    vision_model = "gpt-4o (fallback)"
                    logger.debug(f"[VisualAnalyst] GPT-4o fallback response: {len(response.content)} chars")
                else:
                    logger.debug(f"[VisualAnalyst] GPT-4o fallback not available (no OPENAI_API_KEY?)")

            # Log response status
            if not response.content or not response.content.strip():
                logger.debug(f"[VisualAnalyst] Warning: Vision model returned empty content")
            elif response.content.startswith("["):
                logger.debug(f"[VisualAnalyst] Vision error: {response.content[:200]}")
            else:
                logger.debug(f"[VisualAnalyst] Vision analysis received ({len(response.content)} chars)")
        else:
            # Fallback for non-vision LLMs (e.g., stub)
            logger.debug(f"[VisualAnalyst] Warning: No generate_with_image method, using text-only fallback")
            response = self._llm.generate(prompt, response_type="research")
            vision_used = False
            vision_model = None

        # Calculate confidence based on visual analysis quality
        confidence_result = calculate_visual_confidence(
            image_loaded=True,  # We got here, so image was loaded
            response_text=response.content,
            response_length=len(response.content) if response.content else 0,
        )
        logger.debug(f"[VisualAnalyst] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        # Write analysis to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="visual_analysis",
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "image_dimensions": f"{metadata_result.output['width']}x{metadata_result.output['height']}" if metadata_result.success else None,
                "vision_model_used": vision_used,
                "vision_model": vision_model,
            },
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_visual_analysis",
            content=response.content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "visual_analysis",
                "image_loaded": True,
                "vision_model_used": vision_used,
                "vision_model": vision_model,
                "tool_calls": tool_calls,
                **confidence_result.to_dict(),
            },
        )


class FactCheckerAgent(BaseAgent):
    """Agent specialized in verifying factual claims and cross-referencing.

    Capabilities:
    - Verify factual claims
    - Cross-reference information
    - Identify contradictions
    - Assess information reliability
    """

    def __init__(self, llm):
        super().__init__(
            name="fact_checker",
            role=AgentRole.CRITIC,
            llm=llm,
        )

    @property
    def uses_retrieval(self) -> bool:
        return True  # May need to retrieve additional evidence

    def get_retrieval_query(self, state: SharedState) -> Optional[str]:
        """Generate retrieval query for fact-checking."""
        return f"{state.task_input.query} [Requesting verification sources]"

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Verify facts and check consistency."""
        # Read all memory to check facts
        all_entries = memory_tracker.get_all_memory()

        prompt_parts = [
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
            "Instructions:",
            "1. Review claims made by other agents",
            "2. Cross-reference information for consistency",
            "3. Identify any contradictions or uncertainties",
            "4. Verify calculations and logic",
            "5. Flag areas needing more evidence",
            "",
            "Information to verify:",
        ]

        for entry in all_entries:
            if entry.entry_type in ["note", "analysis", "calculation"]:
                prompt_parts.append(f"\n[{entry.agent_name} - {entry.entry_type}]:")
                prompt_parts.append(f"{entry.content[:300]}...")

        # Add retrieval sources if available
        citations = []
        if retrieval_result and retrieval_result.items:
            prompt_parts.append("\n\nAdditional verification sources:")
            for idx, item in enumerate(retrieval_result.items):
                prompt_parts.append(f"\n[Source {idx+1}]:\n{item.content[:500]}")
                citations.append(
                    CitationRecord(
                        retrieval_id=item.retrieval_id,
                        citation_type=CitationType.REFERENCED,
                        excerpt=item.content[:100] + "..." if len(item.content) > 100 else item.content,
                    )
                )

        prompt_parts.append("\nProvide your fact-check analysis:")
        prompt = "\n".join(prompt_parts)

        response = self._llm.generate(prompt, response_type="critique")

        # Write fact-check to memory (marked as verified with high confidence)
        cited_ids = [c.retrieval_id for c in citations]
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="fact_check",
            cited_retrieval_ids=cited_ids,
            metadata={"step": step_num, "verified_count": len(all_entries)},
            status="verified",
            confidence=0.9,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_fact_check",
            content=response.content,
            citations=citations,
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "fact_checking",
            },
        )


class DocumentAnalystAgent(BaseAgent):
    """Agent specialized in iterative document analysis using ReAct-style reasoning.

    Capabilities:
    - Iteratively query documents (get_summary, get_section, search)
    - Build understanding through multiple read passes
    - Re-read sections with new context as understanding evolves
    - Support both DocumentTool (.docx, .txt) and PDFTool (.pdf)
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="document_analyst",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )
        self._max_iterations = 8

    @property
    def uses_tools(self) -> bool:
        return True

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Analyze document using iterative ReAct-style reasoning."""
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for document analysis",
                metadata={"error": "no_tool_executor"},
            )

        # Check which document tool is available
        doc_tool = self._tool_executor.get_tool("document")
        pdf_tool = self._tool_executor.get_tool("pdf")
        pptx_tool = self._tool_executor.get_tool("pptx")

        if doc_tool:
            return self._analyze_with_document_tool(state, memory_tracker, step_num)
        elif pdf_tool:
            return self._analyze_with_pdf_tool(state, memory_tracker, step_num)
        elif pptx_tool:
            return self._analyze_with_pptx_tool(state, memory_tracker, step_num)
        else:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No document or PDF tool available",
                metadata={"error": "no_document_tool"},
            )

    def _analyze_with_document_tool(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        step_num: int,
    ) -> AgentOutput:
        """Iterative analysis using DocumentTool with ReAct loop."""
        # Track all tool calls for divergence analysis
        tool_calls = []

        # Get initial document summary
        summary_result = self._tool_executor.execute("document", "get_summary")
        tool_calls.append({
            "tool_name": "document",
            "operation": "get_summary",
            "params": {},
            "success": summary_result.success,
            "output": str(summary_result.output)[:200] if summary_result.success else None,
            "error": summary_result.error if not summary_result.success else None,
        })

        if not summary_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to get document summary: {summary_result.error}",
                metadata={"error": "summary_failed", "tool_calls": tool_calls},
            )

        # Track conversation history for iterative reasoning
        observations = [f"DOCUMENT SUMMARY:\n{summary_result.output}"]
        actions_taken = []

        # Initial prompt explaining available actions
        system_context = """You are analyzing a document to answer a question. You can iteratively query the document.

AVAILABLE ACTIONS:
- THINK: <your reasoning> - reason step-by-step about what you know and need to find
- GET_SECTION: <section_name> - retrieve full content of a section
- SEARCH: <keyword> - search for specific terms in the document
- GET_DATA: - get all structured data as a dictionary
- VERIFY: <claim> - re-check your reasoning before concluding
- ANSWER: <your final answer> - when you have enough information

REASONING APPROACH:
1. First understand the document structure (use GET_DATA to see all data at once)
2. Identify what the question is PRECISELY asking (e.g., "who did X" vs "who received X" are different)
3. Count items in related lists - mismatches often reveal the answer
4. Consider if list positions or ordering has meaning
5. Before answering, VERIFY your conclusion makes sense

Always output ONE action per line. Use THINK often to reason through the problem."""

        # ReAct loop
        iteration = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_answer = None

        while iteration < self._max_iterations:
            iteration += 1

            # Build prompt with history
            prompt_parts = [
                system_context,
                f"\nTASK: {state.task_input.query}",
                "\n--- OBSERVATIONS SO FAR ---",
            ]
            for obs in observations:
                prompt_parts.append(obs)
            prompt_parts.append("\n--- YOUR TURN ---")
            prompt_parts.append("What is your next action? (THINK, GET_SECTION, SEARCH, GET_DATA, or ANSWER)")

            prompt = "\n".join(prompt_parts)

            # Get LLM response
            response = self._llm.generate(prompt, response_type="document_analyst")
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            response_text = response.content.strip()
            actions_taken.append(response_text)

            # Log the iteration
            logger.debug(f"  [DocAnalyst] Iteration {iteration}/{self._max_iterations}")

            # Parse ALL actions in the response (LLM may output multiple)
            # Split by newlines and process each action
            lines = response_text.split('\n')
            found_answer = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                line_upper = line.upper()

                if line_upper.startswith("ANSWER:") or line_upper.startswith("ANSWER "):
                    # Final answer
                    final_answer = line.split(":", 1)[-1].strip() if ":" in line else line[7:].strip()
                    logger.debug(f"    → ANSWER: {final_answer[:100]}{'...' if len(final_answer) > 100 else ''}")
                    found_answer = True
                    break

                elif line_upper.startswith("THINK:") or line_upper.startswith("THINK "):
                    # Thinking - add to observations
                    thought = line[6:].strip() if line.upper().startswith("THINK:") else line[5:].strip()
                    logger.debug(f"    → THINK: {thought[:100]}{'...' if len(thought) > 100 else ''}")
                    observations.append(f"THOUGHT: {line}")

                elif line_upper.startswith("GET_SECTION:") or line_upper.startswith("GET_SECTION "):
                    # Get a specific section
                    section_name = line.split(":", 1)[-1].strip() if ":" in line else line[12:].strip()
                    logger.debug(f"    → GET_SECTION: {section_name}")
                    result = self._tool_executor.execute("document", "get_section", title=section_name)
                    tool_calls.append({
                        "tool_name": "document",
                        "operation": "get_section",
                        "params": {"title": section_name},
                        "success": result.success,
                        "output": str(result.output)[:200] if result.success else None,
                        "error": result.error if not result.success else None,
                    })
                    if result.success:
                        observations.append(f"SECTION '{section_name}':\n{result.output}")
                        logger.debug(f"      ✓ Found section with {len(str(result.output))} chars")
                    else:
                        observations.append(f"SECTION '{section_name}': Not found - {result.error}")
                        logger.debug(f"      ✗ Not found: {result.error}")

                elif line_upper.startswith("SEARCH:") or line_upper.startswith("SEARCH "):
                    # Search for keyword
                    query = line.split(":", 1)[-1].strip() if ":" in line else line[7:].strip()
                    logger.debug(f"    → SEARCH: {query}")
                    result = self._tool_executor.execute("document", "search", query=query)
                    tool_calls.append({
                        "tool_name": "document",
                        "operation": "search",
                        "params": {"query": query},
                        "success": result.success,
                        "output": str(result.output)[:200] if result.success else None,
                        "error": result.error if not result.success else None,
                    })
                    if result.success:
                        observations.append(f"SEARCH '{query}':\n{result.output}")
                        match_info = result.metadata.get('match_count', 'unknown') if result.metadata else 'unknown'
                        logger.debug(f"      ✓ Found {match_info} matches")
                    else:
                        observations.append(f"SEARCH '{query}': {result.error}")
                        logger.debug(f"      ✗ Error: {result.error}")

                elif line_upper.startswith("GET_DATA"):
                    # Get structured data
                    logger.debug(f"    → GET_DATA")
                    result = self._tool_executor.execute("document", "get_structured_data")
                    tool_calls.append({
                        "tool_name": "document",
                        "operation": "get_structured_data",
                        "params": {},
                        "success": result.success,
                        "output": str(result.output)[:200] if result.success else None,
                        "error": result.error if not result.success else None,
                    })
                    if result.success:
                        # Format the data nicely
                        data = result.output
                        data_str = "STRUCTURED DATA:\n"
                        for key, value in data.items():
                            if value is None:
                                data_str += f"  {key}: (empty)\n"
                            elif isinstance(value, list):
                                data_str += f"  {key}: {value}\n"
                            elif isinstance(value, dict):
                                data_str += f"  {key}:\n"
                                for k, v in value.items():
                                    data_str += f"    {k}: {v}\n"
                            else:
                                data_str += f"  {key}: {value}\n"
                        observations.append(data_str)
                        logger.debug(f"      ✓ Retrieved {len(data)} sections")
                    else:
                        observations.append(f"GET_DATA: {result.error}")
                        logger.debug(f"      ✗ Error: {result.error}")

                elif line_upper.startswith("VERIFY:") or line_upper.startswith("VERIFY "):
                    # Verification step - add to observations for LLM to re-evaluate
                    claim = line.split(":", 1)[-1].strip() if ":" in line else line[7:].strip()
                    logger.debug(f"    → VERIFY: {claim[:80]}{'...' if len(claim) > 80 else ''}")
                    observations.append(f"VERIFYING: {claim}\n(Review your reasoning above to confirm this is correct)")

            if found_answer:
                break

        # If no explicit answer, use last response
        if not final_answer:
            final_answer = actions_taken[-1] if actions_taken else "Unable to analyze document"

        # Build final output
        analysis_summary = f"Document Analysis ({iteration} iterations):\n\n{final_answer}"

        # Calculate confidence based on document quality signals
        # Gather text from observations for quality assessment
        all_text = "\n".join(observations)
        confidence_result = calculate_document_confidence(
            text=all_text,
            page_count=1,  # Document tool doesn't expose page count
            extraction_success=True,  # We got here, so extraction worked
            output_text=final_answer,
            llm=self._llm,
        )
        logger.debug(f"[DocumentAnalyst] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        # Write to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=analysis_summary,
            entry_type="document_summary",
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "iterations": iteration,
                "actions": len(actions_taken),
            },
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_document_summary",
            content=analysis_summary,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "document_analysis",
                "iterations": iteration,
                "actions_taken": actions_taken,
                "tool_calls": tool_calls,
                **confidence_result.to_dict(),
            },
        )

    def _analyze_with_pdf_tool(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        step_num: int,
    ) -> AgentOutput:
        """Fallback: analyze PDF with single-pass extraction."""
        tool_calls = []

        # Get PDF metadata and page count
        metadata_result = self._tool_executor.execute("pdf", "get_metadata")
        tool_calls.append({
            "tool_name": "pdf",
            "operation": "get_metadata",
            "params": {},
            "success": metadata_result.success,
            "output": str(metadata_result.output)[:200] if metadata_result.success else None,
            "error": metadata_result.error if not metadata_result.success else None,
        })

        page_count_result = self._tool_executor.execute("pdf", "get_page_count")
        tool_calls.append({
            "tool_name": "pdf",
            "operation": "get_page_count",
            "params": {},
            "success": page_count_result.success,
            "output": page_count_result.output if page_count_result.success else None,
            "error": page_count_result.error if not page_count_result.success else None,
        })
        page_count = page_count_result.output if page_count_result.success else 0

        # Extract all text from PDF
        max_pages = min(20, page_count)
        all_text_result = self._tool_executor.execute("pdf", "extract_all_text", max_pages=max_pages)
        tool_calls.append({
            "tool_name": "pdf",
            "operation": "extract_all_text",
            "params": {"max_pages": max_pages},
            "success": all_text_result.success,
            "output": f"[{len(all_text_result.output)} chars]" if all_text_result.success else None,
            "error": all_text_result.error if not all_text_result.success else None,
        })

        if not all_text_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to extract PDF text: {all_text_result.error}",
                metadata={"error": "pdf_extraction_failed", "tool_calls": tool_calls},
            )

        full_text = all_text_result.output

        # Build analysis prompt
        prompt = f"""Task: {state.task_input.query}

PDF Document ({page_count} pages, {max_pages} extracted):

{full_text}

Analyze the document and provide a detailed answer to the task. Be precise with details."""

        response = self._llm.generate(prompt, response_type="research")

        # Calculate confidence based on PDF extraction quality
        confidence_result = calculate_document_confidence(
            text=full_text,
            page_count=page_count,
            extraction_success=True,
            output_text=response.content,
            llm=self._llm,
        )
        logger.debug(f"[DocumentAnalyst/PDF] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="document_summary",
            cited_retrieval_ids=[],
            metadata={"step": step_num, "pages_analyzed": max_pages},
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_document_summary",
            content=response.content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "specialty": "document_analysis",
                "pages_analyzed": max_pages,
                "tool_calls": tool_calls,
                **confidence_result.to_dict(),
            },
        )

    def _analyze_with_pptx_tool(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        step_num: int,
    ) -> AgentOutput:
        """Analyze PowerPoint presentation using PPTXTool."""
        tool_calls = []

        # Get summary of the presentation
        summary_result = self._tool_executor.execute("pptx", "get_summary")
        tool_calls.append({
            "tool_name": "pptx",
            "operation": "get_summary",
            "params": {},
            "success": summary_result.success,
            "output": str(summary_result.output)[:200] if summary_result.success else None,
            "error": summary_result.error if not summary_result.success else None,
        })

        if not summary_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to get PowerPoint summary: {summary_result.error}",
                metadata={"error": "pptx_summary_failed", "tool_calls": tool_calls},
            )

        summary = summary_result.output
        slide_count = summary.get('slide_count', 0)

        # Get all text from the presentation
        text_result = self._tool_executor.execute("pptx", "get_text")
        tool_calls.append({
            "tool_name": "pptx",
            "operation": "get_text",
            "params": {},
            "success": text_result.success,
            "output": f"[{len(text_result.output)} chars]" if text_result.success else None,
            "error": text_result.error if not text_result.success else None,
        })

        if not text_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to extract PowerPoint text: {text_result.error}",
                metadata={"error": "pptx_text_failed", "tool_calls": tool_calls},
            )

        full_text = text_result.output

        # Build analysis prompt
        prompt = f"""Task: {state.task_input.query}

PowerPoint Presentation Summary:
- Total slides: {slide_count}
- Title: {summary.get('title', 'Unknown')}

Full Text Content (by slide):
{full_text}

Analyze the presentation content and provide a detailed answer to the task.
Be precise with details, counts, and specific content from the slides.
If asked about specific slides, identify them by their slide number."""

        response = self._llm.generate(prompt, response_type="research")

        # Calculate confidence based on PPTX extraction quality
        confidence_result = calculate_document_confidence(
            text=full_text,
            page_count=slide_count,  # Use slide count as proxy for page count
            extraction_success=True,
            output_text=response.content,
            llm=self._llm,
        )
        logger.debug(f"[DocumentAnalyst/PPTX] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="document_summary",
            cited_retrieval_ids=[],
            metadata={"step": step_num, "slides_analyzed": slide_count},
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_document_summary",
            content=response.content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "specialty": "pptx_analysis",
                "slides_analyzed": slide_count,
                "tool_calls": tool_calls,
                **confidence_result.to_dict(),
            },
        )


class PlannerAgent(BaseAgent):
    """Agent specialized in task decomposition and creating execution plans.

    Creates execution plans with:
    - Subtasks breakdown
    - Agent assignments for each subtask
    - Dependencies and ordering
    - Adaptation based on progress
    """

    def __init__(self, llm):
        super().__init__(
            name="planner",
            role=AgentRole.RESEARCHER,
            llm=llm,
        )

    @property
    def uses_retrieval(self) -> bool:
        return False  # Plans based on task and current state

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Create or update execution plan with subtasks and agent assignments."""
        # Read existing memory to understand progress
        all_entries = memory_tracker.get_all_memory()
        existing_plans = memory_tracker.read_memory(entry_types=["plan"])

        prompt_parts = [
            "You are an execution planning expert for multi-agent systems.",
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
        ]

        # Add file context
        if state.task_input.metadata.get('file_name'):
            file_name = state.task_input.metadata.get('file_name', '')
            file_type = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
            prompt_parts.append(f"Attached file: {file_name} (type: {file_type})")

            # For zip files, show the extracted contents
            if file_type == 'zip' and state.task_input.metadata.get('zip_file_list'):
                extract_dir = state.task_input.metadata.get('zip_extract_dir', '')
                file_list = state.task_input.metadata.get('zip_file_list', [])
                prompt_parts.append("")
                prompt_parts.append("ZIP ALREADY EXTRACTED - contents:")
                for f in file_list:
                    ext = f['filename'].split('.')[-1].lower() if '.' in f['filename'] else 'unknown'
                    full_path = f"{extract_dir}/{f['filename']}"
                    prompt_parts.append(f"  - {f['filename']} ({ext}, {f['file_size']} bytes) → {full_path}")
                prompt_parts.append("")
                prompt_parts.append("IMPORTANT: The zip is already extracted. Agents should use python_executor")
                prompt_parts.append("with the full paths above to read the files directly. Do NOT try to extract again.")
            prompt_parts.append("")

        # Show available agents
        prompt_parts.append("Available specialized agents:")
        prompt_parts.append("  - data_analyst: Analyzes tabular data from CSV/Excel files with DATA (rows/columns)")
        prompt_parts.append("  - python_executor: Executes Python code - USE FOR: .py files, .txt files, computations, algorithms")
        prompt_parts.append("  - calculator: Performs simple mathematical calculations (eval, sum, mean)")
        prompt_parts.append("  - visual_analyst: Analyzes images (PNG, JPG, GIF, WEBP, BMP ONLY) - extracts text, charts, visual info")
        prompt_parts.append("  - document_analyst: Analyzes documents (PDF, DOCX) with iterative reasoning")
        prompt_parts.append("  - web_researcher: Searches the web and fetches URL content")
        prompt_parts.append("  - audio_analyst: Transcribes and analyzes audio files (MP3, WAV)")
        prompt_parts.append("  - fact_checker: Verifies factual claims against known sources")
        prompt_parts.append("  - researcher: Gathers information from internal retrieval system")
        prompt_parts.append("  - critic: Critically reviews work for errors")
        prompt_parts.append("  - synthesizer: Combines all gathered information into final answer - ALWAYS use as last step")
        prompt_parts.append("  - generalist: Multi-tool agent for tasks requiring COMBINED capabilities (file + web + code)")
        prompt_parts.append("")
        prompt_parts.append("FILE TYPE ROUTING RULES:")
        prompt_parts.append("  - .xlsx/.csv with DATA → data_analyst")
        prompt_parts.append("  - .xlsx with ONLY colors (no data rows) → python_executor (use df_colors)")
        prompt_parts.append("  - .py (Python code) → python_executor (execute the code)")
        prompt_parts.append("  - .txt/.json/.md → python_executor (file_content variable available)")
        prompt_parts.append("  - .docx (Word document) → document_analyst (iterative analysis with ReAct reasoning)")
        prompt_parts.append("  - .png/.jpg/.gif/.webp/.bmp → visual_analyst")
        prompt_parts.append("  - .mp3/.wav/.ogg/.flac → audio_analyst")
        prompt_parts.append("  - .pdf → document_analyst")
        prompt_parts.append("  - .pptx → document_analyst (text extracted from slides)")
        prompt_parts.append("  - .zip → python_executor (ZIP already extracted, use full paths from context)")
        prompt_parts.append("  - Tasks requiring BOTH file analysis AND web search → generalist")
        prompt_parts.append("")

        # File inspection context (concrete facts about file structure)
        if state.task_input.metadata.get('file_inspection_context'):
            prompt_parts.append(state.task_input.metadata['file_inspection_context'])
            prompt_parts.append("")
            prompt_parts.append("IMPORTANT: Plan based on the ACTUAL file structure above, not assumptions.")
            prompt_parts.append("")

        # Context from previous work
        if all_entries and not existing_plans:
            # First plan
            prompt_parts.append("This is the initial planning phase.")
        elif existing_plans:
            # Re-planning
            prompt_parts.append("REPLANNING - Previous plan needs revision.")

            # Check for quality-triggered replan context
            replan_context = state.task_input.metadata.get('replan_context')
            if replan_context:
                prompt_parts.append("")
                prompt_parts.append("REPLAN REASON (address this issue):")
                prompt_parts.append(f"  Trigger: {replan_context.get('trigger', 'unknown')}")
                prompt_parts.append(f"  Reason: {replan_context.get('reason', 'quality issues')}")
                if replan_context.get('failed_agent'):
                    prompt_parts.append(f"  Failed agent: {replan_context.get('failed_agent')}")
                if replan_context.get('failed_subtask'):
                    prompt_parts.append(f"  Failed subtask: {replan_context.get('failed_subtask')}")
                if replan_context.get('issues'):
                    issues_summary = ", ".join(i[0] for i in replan_context['issues'] if isinstance(i, tuple))
                    prompt_parts.append(f"  Issues detected: {issues_summary}")

                # Provide confidence-specific context
                trigger = replan_context.get('trigger', '')
                replan_mode = replan_context.get('replan_mode', 'full')

                if trigger == 'low_confidence':
                    confidence_score = replan_context.get('confidence_score', 0)
                    prompt_parts.append(f"  Confidence score: {confidence_score:.2f}")
                    prompt_parts.append("")
                    prompt_parts.append("The previous agent reported low confidence in its output.")

                # Handle partial replanning (continue mode)
                if replan_mode == 'continue':
                    completed_subtasks = replan_context.get('completed_subtasks', [])
                    prompt_parts.append("")
                    prompt_parts.append("PARTIAL REPLAN MODE: Keep completed work, only replan remaining steps.")
                    prompt_parts.append("")

                    if completed_subtasks:
                        prompt_parts.append("COMPLETED SUBTASKS (preserve these - do NOT redo):")
                        for st in completed_subtasks:
                            output = st.get('output_summary', 'completed')[:100]
                            prompt_parts.append(f"  ✓ {st['id']}: {st['description']} ({st['agent']}) -> {output}")
                        prompt_parts.append("")

                    prompt_parts.append("IMPORTANT: Plan ONLY the remaining steps needed to complete the task.")
                    prompt_parts.append("- Reference completed subtask IDs (e.g., s1, s2) in dependencies if needed")
                    prompt_parts.append("- Use a DIFFERENT approach for the failed step (different agent or method)")
                    prompt_parts.append("- Start your new subtask IDs after the completed ones (e.g., if s1,s2 done, start from s3)")
                    prompt_parts.append("")
                else:
                    # Full replan
                    prompt_parts.append("Design an alternative approach that avoids the same failure mode.")
                    prompt_parts.append("")
                    prompt_parts.append("IMPORTANT: Create a NEW plan that addresses the issues above.")
                    prompt_parts.append("Consider: using a different agent, breaking into smaller steps, or adding verification.")
                    prompt_parts.append("")

            prompt_parts.append("Current progress:")
            entry_types = {}
            by_agent = {}
            for entry in all_entries:
                entry_types[entry.entry_type] = entry_types.get(entry.entry_type, 0) + 1
                by_agent[entry.agent_name] = by_agent.get(entry.agent_name, 0) + 1

            for agent_name, count in by_agent.items():
                prompt_parts.append(f"  - {agent_name}: {count} entries")

        prompt_parts.extend([
            "",
            "INSTRUCTIONS:",
            "1. Analyze the task and identify what needs to be done",
            "2. Break the task into clear subtasks with dependency tracking",
            "3. For EACH subtask, assign the most appropriate specialist agent",
            "4. Consider file type when assigning agents",
            "5. Specify dependencies: which subtasks must complete before each one can start",
            "6. Independent subtasks should have NO dependencies on each other",
            "",
            "OUTPUT FORMAT (JSON):",
            "Respond with ONLY a JSON object in this exact format:",
            '```json',
            '{',
            '  "reasoning": "2-3 sentences explaining your strategy",',
            '  "subtasks": [',
            '    {"id": "s1", "description": "...", "agent": "agent_name", "dependencies": []},',
            '    {"id": "s2", "description": "...", "agent": "agent_name", "dependencies": ["s1"]},',
            '    {"id": "s3", "description": "...", "agent": "agent_name", "dependencies": ["s1"]},',
            '    {"id": "s4", "description": "...", "agent": "agent_name", "dependencies": ["s2", "s3"]}',
            '  ]',
            '}',
            '```',
            "",
            "Example (spreadsheet analysis):",
            '```json',
            '{',
            '  "reasoning": "This is a spreadsheet analysis task. We need to extract data, verify it, then synthesize.",',
            '  "subtasks": [',
            '    {"id": "s1", "description": "Extract and analyze spreadsheet data", "agent": "data_analyst", "dependencies": []},',
            '    {"id": "s2", "description": "Verify the analysis results", "agent": "fact_checker", "dependencies": ["s1"]},',
            '    {"id": "s3", "description": "Produce final answer", "agent": "synthesizer", "dependencies": ["s2"]}',
            '  ]',
            '}',
            '```',
            "",
            "Example (multi-source research with parallel steps):",
            '```json',
            '{',
            '  "reasoning": "Need both file analysis and web research, which can run in parallel before synthesis.",',
            '  "subtasks": [',
            '    {"id": "s1", "description": "Analyze attached document", "agent": "document_analyst", "dependencies": []},',
            '    {"id": "s2", "description": "Search web for supporting information", "agent": "web_researcher", "dependencies": []},',
            '    {"id": "s3", "description": "Produce final answer combining all findings", "agent": "synthesizer", "dependencies": ["s1", "s2"]}',
            '  ]',
            '}',
            '```',
            "",
            "IMPORTANT RULES:",
            "- ALWAYS end with 'synthesizer' to produce the final answer",
            "- ONLY use document_analyst for PDF files - NEVER for compiling or formatting results",
            "- For image analysis tasks, use visual_analyst (NOT document_analyst)",
            "- For audio files, use audio_analyst",
            "- Use dependencies to express which subtasks must complete before others",
            "- Independent subtasks (e.g., file analysis + web search) should have empty dependencies []",
            "",
            "Now create the execution plan as JSON:"
        ])

        prompt = "\n".join(prompt_parts)

        response = self._llm.generate(prompt, response_type="plan")

        # Parse structured plan from response (with fallback)
        from ..planning.plan_types import ExecutionPlan as StructuredPlan
        structured_plan = StructuredPlan.from_json(
            response.content, created_at_step=step_num
        )

        # Write plan to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="plan",
            metadata={
                "step": step_num,
                "is_replan": len(existing_plans) > 0,
                "structured_plan": structured_plan.to_dict(),
            },
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_plan",
            content=response.content,
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "planning",
                "is_replan": len(existing_plans) > 0,
                "structured_plan": structured_plan.to_dict(),
                "subtask_count": len(structured_plan.subtasks),
                "plan_id": structured_plan.plan_id,
            },
        )


class CriticAgent(BaseAgent):
    """Agent specialized in critical review and quality assessment.

    Capabilities:
    - Review work from other agents
    - Identify gaps or inconsistencies
    - Assess quality and completeness
    - Provide recommendations for Router
    """

    def __init__(self, llm):
        super().__init__(
            name="critic",
            role=AgentRole.CRITIC,
            llm=llm,
        )

    @property
    def uses_retrieval(self) -> bool:
        return False  # Works from memory

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Critique step: review work and provide assessment."""
        # Read all memory for review
        all_entries = memory_tracker.get_all_memory()

        prompt_parts = [
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
            "WORK TO REVIEW:",
        ]

        for entry in all_entries:
            if entry.entry_type not in ["plan", "critique"]:  # Don't review plans/critiques
                preview = entry.content[:200].replace('\n', ' ')
                prompt_parts.append(f"  [{entry.agent_name} - {entry.entry_type}]")
                prompt_parts.append(f"    {preview}...")

        prompt_parts.extend([
            "",
            "INSTRUCTIONS:",
            "1. Assess the quality and completeness of the work",
            "2. Identify any gaps, inconsistencies, or concerns",
            "3. Determine if sufficient evidence exists to answer the task",
            "4. Provide a recommendation for what should happen next",
            "",
            "OUTPUT FORMAT:",
            "Assessment: <your critique>",
            "",
            "Recommendation: <suggest what Router should do next>",
            "(Options: gather more info, verify with fact_checker, ready to synthesize, etc.)",
        ])

        prompt = "\n".join(prompt_parts)
        response = self._llm.generate(prompt, response_type="critique")

        # Extract recommendation from response
        recommendation = None
        if "Recommendation:" in response.content:
            parts = response.content.split("Recommendation:")
            if len(parts) > 1:
                recommendation = parts[1].strip().split('\n')[0]

        # Write critique to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="critique",
            metadata={
                "step": step_num,
                "reviewed_count": len(all_entries),
                "recommendation": recommendation,
            },
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_critique",
            content=response.content,
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "reviewed_count": len(all_entries),
                "recommendation": recommendation,
                "specialty": "critical_review",
            },
        )


class ResearcherAgent(BaseAgent):
    """General research agent that queries retrieval and writes notes."""

    def __init__(self, llm):
        super().__init__(
            name="researcher",
            role=AgentRole.RESEARCHER,
            llm=llm,
        )

    @property
    def uses_retrieval(self) -> bool:
        return True

    def get_retrieval_query(self, state: SharedState) -> Optional[str]:
        """Generate retrieval query from task input."""
        return state.task_input.query

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Research step: query retrieval and write notes."""
        # Read existing memory for context
        existing_notes = memory_tracker.read_memory(entry_types=["note"])

        # Build prompt
        prompt_parts = [
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
        ]

        if existing_notes:
            prompt_parts.append(f"Existing notes: {len(existing_notes)} entries")

        # Process retrieval results
        citations = []
        retrieval_content = []
        if retrieval_result and retrieval_result.items:
            for item in retrieval_result.items:
                retrieval_content.append(item.content)
                citations.append(
                    CitationRecord(
                        retrieval_id=item.retrieval_id,
                        citation_type=CitationType.REFERENCED,
                        excerpt=item.content[:50] + "..." if len(item.content) > 50 else item.content,
                    )
                )

        # Generate response
        prompt = "\n".join(prompt_parts)
        if retrieval_content:
            response = self._llm.generate_with_retrieval(
                prompt,
                retrieval_content,
                response_type="research",
            )
        else:
            response = self._llm.generate(prompt, response_type="research")

        # Write note to memory
        cited_ids = [c.retrieval_id for c in citations]
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="note",
            cited_retrieval_ids=cited_ids,
            metadata={"step": step_num},
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_note",
            content=response.content,
            citations=citations,
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "retrieval_count": len(retrieval_content),
            },
        )


class SynthesizerAgent(BaseAgent):
    """Agent that reads memory and produces final answer.

    Uses ContextBudget to prevent context overflow on long episodes.
    When total entry content fits within the token budget, entries are
    included in full. When it exceeds the budget, entries are truncated
    proportionally based on priority weights and recency.
    """

    def __init__(self, llm, max_context_tokens: int = 12000):
        super().__init__(
            name="synthesizer",
            role=AgentRole.SYNTHESIZER,
            llm=llm,
        )
        self._max_context_tokens = max_context_tokens

    @property
    def uses_retrieval(self) -> bool:
        return False

    def _format_entry(self, entry, max_chars: int) -> str:
        """Format a single memory entry, truncating content if needed."""
        content = entry.content
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        # Build prefix
        if entry.status == "failed_attempt":
            prefix = f"  [FAILED ATTEMPT - Low confidence] From {entry.agent_name}: "
        else:
            confidence_label = f"[Confidence: {entry.confidence:.1f}] " if entry.confidence != 0.5 else ""
            prefix = f"  From {entry.agent_name} {confidence_label}: "

        suffix = "... [truncated]" if truncated else ""
        return f"{prefix}{content}{suffix}"

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Synthesize step: combine all information into final answer."""
        from ..core.context_budget import ContextBudget

        # Read all memory
        all_entries = memory_tracker.get_all_memory()

        # Compute context budget allocations
        budget = ContextBudget(max_tokens=self._max_context_tokens)
        needs_truncation = not budget.fits_in_budget(all_entries)

        # Build allocation lookup: entry.id -> max_chars
        char_limits = {}
        if needs_truncation:
            allocations = budget.allocate(all_entries)
            for alloc in allocations:
                char_limits[alloc.entry.id] = alloc.max_chars

        def get_max_chars(entry):
            """Get character limit for an entry (full length if no truncation needed)."""
            if not needs_truncation:
                return len(entry.content)
            return char_limits.get(entry.id, len(entry.content))

        # Organize by entry type - include all relevant types from specialized agents
        entries_by_type = {
            "plan": [],
            "analysis": [],
            "calculation": [],
            "visual_analysis": [],
            "audio_analysis": [],
            "document_summary": [],
            "fact_check": [],
            "note": [],
            "critique": [],
            "web_research": [],
        }

        for entry in all_entries:
            if entry.entry_type in entries_by_type:
                entries_by_type[entry.entry_type].append(entry)

        # Build prompt with all available information
        prompt_parts = [
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
            "You are synthesizing the final answer from the work of multiple specialized agents.",
            "Review all the information below and provide a clear, direct answer to the task.",
            "",
        ]

        if needs_truncation:
            prompt_parts.append("[Note: Some entries were truncated to fit context budget]")
            prompt_parts.append("")

        # Add plan if exists
        if entries_by_type["plan"]:
            prompt_parts.append("EXECUTION PLAN:")
            for entry in entries_by_type["plan"]:
                prompt_parts.append(f"  {self._format_entry(entry, get_max_chars(entry))}")
            prompt_parts.append("")

        # Add analyses from specialized agents (sorted by quality)
        if entries_by_type["analysis"]:
            prompt_parts.append("DATA ANALYSIS:")
            status_order = {"verified": 3, "final": 3, "preliminary": 2, "failed_attempt": 1}
            sorted_analyses = sorted(
                entries_by_type["analysis"],
                key=lambda e: (status_order.get(e.status, 2), e.confidence),
                reverse=True
            )
            for entry in sorted_analyses:
                prompt_parts.append(self._format_entry(entry, get_max_chars(entry)))
            prompt_parts.append("")

        if entries_by_type["calculation"]:
            prompt_parts.append("CALCULATIONS:")
            status_order = {"verified": 3, "final": 3, "preliminary": 2, "failed_attempt": 1}
            sorted_calcs = sorted(
                entries_by_type["calculation"],
                key=lambda e: (status_order.get(e.status, 2), e.confidence),
                reverse=True
            )
            for entry in sorted_calcs:
                prompt_parts.append(self._format_entry(entry, get_max_chars(entry)))
            prompt_parts.append("")

        # Simple sections — same pattern for each
        section_map = {
            "visual_analysis": "VISUAL ANALYSIS:",
            "document_summary": "DOCUMENT SUMMARIES:",
            "audio_analysis": "AUDIO TRANSCRIPTION/ANALYSIS:",
            "web_research": "WEB RESEARCH:",
            "fact_check": "FACT CHECKS:",
            "note": "NOTES:",
            "critique": "CRITIQUES:",
        }

        for entry_type, header in section_map.items():
            if entries_by_type[entry_type]:
                prompt_parts.append(header)
                for entry in entries_by_type[entry_type]:
                    prompt_parts.append(self._format_entry(entry, get_max_chars(entry)))
                prompt_parts.append("")

        prompt_parts.append("SYNTHESIS INSTRUCTIONS:")
        prompt_parts.append("- Prioritize entries marked with higher confidence scores")
        prompt_parts.append("- Verified information (from fact_checker) has highest reliability")
        prompt_parts.append("- Failed attempts show what DIDN'T work - don't base your answer on them")
        prompt_parts.append("- If multiple answers conflict, trust the one with higher confidence")
        prompt_parts.append("")
        prompt_parts.append("Based on all the information above, provide the final answer to the task.")
        prompt_parts.append("Be direct and concise. Answer exactly what is asked.")

        prompt = "\n".join(prompt_parts)
        response = self._llm.generate(
            prompt,
            response_type="synthesize",
            context={"answer": f"Answer to: {state.task_input.query}"},
        )

        # Strip <think>...</think> blocks from the response (Qwen chain-of-thought)
        final_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

        # Set final answer
        state.set_final_answer(final_content)

        # Also write to memory for record (marked as final answer)
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=final_content,
            entry_type="answer",
            metadata={"step": step_num, "is_final": True},
            status="final",
            confidence=0.8,
        )

        return AgentOutput(
            agent_name=self.name,
            action="final_answer",
            content=final_content,
            is_final_answer=True,
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "total_entries": len(all_entries),
                "analysis_count": len(entries_by_type["analysis"]),
                "calculation_count": len(entries_by_type["calculation"]),
                "note_count": len(entries_by_type["note"]),
                "critique_count": len(entries_by_type["critique"]),
                "context_truncated": needs_truncation,
            },
        )


class WebResearcherAgent(BaseAgent):
    """Agent specialized in web research - searching and extracting information from the internet.

    Capabilities:
    - Search the web using WebSearchTool
    - Fetch and extract content from URLs using WebFetchTool
    - Follow links to gather more information
    - Synthesize findings from multiple web sources

    Uses structured tool calling via generate_with_tools() for reliable
    tool invocation. Operates in a 2-turn loop:
    1. Search turn: LLM picks queries and calls web_search
    2. Fetch+Analyze turn: Given results, LLM optionally fetches URLs and provides analysis
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="web_researcher",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )

    @property
    def uses_tools(self) -> bool:
        return True

    def get_tool_schemas(self) -> list:
        """Return web tool schemas for structured tool calling."""
        schemas = []
        if self._tool_executor:
            for tool_name in ["web_search", "web_fetch"]:
                tool = self._tool_executor.get_tool(tool_name)
                if tool:
                    schemas.append(tool.get_schema())
        return schemas

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Perform web research using search and fetch tools."""
        existing_research = memory_tracker.read_memory(entry_types=["web_research"])

        # Check if web tools are available
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for web research",
                metadata={"error": "no_tool_executor"},
            )

        web_search = self._tool_executor.get_tool("web_search")
        if not web_search:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="WebSearchTool not available",
                metadata={"error": "no_web_search_tool"},
            )

        task_query = state.task_input.query
        tool_schemas = self.get_tool_schemas()

        # Track all tool calls and token counts
        all_tool_calls = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # --- Turn 1: Search ---
        search_prompt_parts = [
            "You are a web research expert. Use the web_search tool to find information.",
            f"Task: {task_query}",
            "",
        ]

        if existing_research:
            search_prompt_parts.append(f"Previous research ({len(existing_research)} entries):")
            for entry in existing_research[-2:]:
                search_prompt_parts.append(f"  - {entry.content[:200]}...")
            search_prompt_parts.append("")

        search_prompt_parts.extend([
            "Search for 1-2 focused queries that would help answer the task.",
            "Use the web_search tool to perform each search.",
        ])

        search_prompt = "\n".join(search_prompt_parts)
        search_response = self._llm.generate_with_tools(
            search_prompt, tools=tool_schemas, response_type="research"
        )
        total_prompt_tokens += search_response.prompt_tokens
        total_completion_tokens += search_response.completion_tokens

        # Execute search tool calls
        all_results = []
        search_queries = []

        if search_response.has_tool_calls:
            for call in search_response.tool_calls:
                result = self._tool_executor.execute(
                    call.tool_name, call.operation, **call.arguments
                )
                query = call.arguments.get("query", "")
                search_queries.append(query)
                all_tool_calls.append({
                    "tool_name": call.tool_name,
                    "operation": call.operation,
                    "params": call.arguments,
                    "success": result.success if result else False,
                    "output_count": len(result.output) if result and result.success and isinstance(result.output, list) else 0,
                })
                if result and result.success and result.output:
                    if isinstance(result.output, list):
                        all_results.extend(result.output)
        else:
            # Fallback: use task query directly
            search_queries = [task_query]
            result = self._tool_executor.execute("web_search", "search", query=task_query)
            all_tool_calls.append({
                "tool_name": "web_search",
                "operation": "search",
                "params": {"query": task_query},
                "success": result.success if result else False,
                "output_count": len(result.output) if result and result.success and isinstance(result.output, list) else 0,
            })
            if result and result.success and result.output:
                if isinstance(result.output, list):
                    all_results.extend(result.output)

        if not all_results:
            return AgentOutput(
                agent_name=self.name,
                action="write_web_research",
                content=f"No search results found for queries: {search_queries}",
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                metadata={"error": "no_results", "queries": search_queries, "tool_calls": all_tool_calls},
            )

        # --- Turn 2: Fetch + Analyze ---
        analysis_parts = [
            "You are a web research expert analyzing search results.",
            f"Task: {task_query}",
            "",
            "Search Results:",
        ]

        for i, result in enumerate(all_results[:10], 1):
            analysis_parts.append(f"{i}. {result.get('title', 'No title')}")
            analysis_parts.append(f"   URL: {result.get('url', '')}")
            analysis_parts.append(f"   {result.get('snippet', '')}")
            analysis_parts.append("")

        analysis_parts.extend([
            "",
            "You may use web_fetch to retrieve the full content of any promising URLs above.",
            "After fetching (or if not needed), provide your analysis citing sources.",
        ])

        analysis_prompt = "\n".join(analysis_parts)
        analysis_response = self._llm.generate_with_tools(
            analysis_prompt, tools=tool_schemas, response_type="research"
        )
        total_prompt_tokens += analysis_response.prompt_tokens
        total_completion_tokens += analysis_response.completion_tokens

        # Execute any fetch tool calls
        fetched_content = []
        if analysis_response.has_tool_calls:
            for call in analysis_response.tool_calls:
                result = self._tool_executor.execute(
                    call.tool_name, call.operation, **call.arguments
                )
                all_tool_calls.append({
                    "tool_name": call.tool_name,
                    "operation": call.operation,
                    "params": call.arguments,
                    "success": result.success if result else False,
                })
                if result and result.success and result.output:
                    content = result.output
                    if isinstance(content, dict):
                        fetched_content.append({
                            "url": call.arguments.get("url", ""),
                            "title": content.get("title", ""),
                            "content": content.get("content", "")[:2000],
                        })

        # If we fetched content, do a final analysis pass
        if fetched_content:
            final_parts = [
                f"Task: {task_query}",
                "",
                "Fetched Page Content:",
            ]
            for content in fetched_content:
                final_parts.append(f"\n--- {content['title']} ---")
                final_parts.append(f"URL: {content['url']}")
                final_parts.append(content['content'][:1500])
                final_parts.append("")

            final_parts.append("\nBased on all the search results and fetched content above, extract the relevant information to answer the task. Be specific and cite sources.")

            final_prompt = "\n".join(final_parts)
            final_response = self._llm.generate(final_prompt, response_type="research")
            total_prompt_tokens += final_response.prompt_tokens
            total_completion_tokens += final_response.completion_tokens
            final_content = final_response.content
        else:
            # Use the analysis response content
            final_content = analysis_response.content or "No analysis produced"

        # Calculate confidence based on web search results quality
        total_content_length = sum(
            len(fc.get('content', '')) for fc in fetched_content
        ) if fetched_content else len(final_content)

        confidence_result = calculate_web_search_confidence(
            results_count=len(all_results),
            results_relevant=len(fetched_content),  # Fetched URLs as proxy for relevance
            fetch_success=True,
            content_length=total_content_length,
        )
        logger.debug(f"[WebResearcher] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        # Write research to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=final_content,
            entry_type="web_research",
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "search_queries": search_queries,
                "result_count": len(all_results),
                "urls_fetched": len(fetched_content),
                "tool_calls": all_tool_calls,
            },
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_web_research",
            content=final_content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "web_research",
                "search_queries": search_queries,
                "result_count": len(all_results),
                "urls_fetched": len(fetched_content),
                "tool_calls": all_tool_calls,
                **confidence_result.to_dict(),
            },
        )


class AudioAnalystAgent(BaseAgent):
    """Agent specialized in analyzing audio content (MP3, WAV files).

    Capabilities:
    - Transcribe audio using AudioTool (Whisper API)
    - Extract key information from transcriptions
    - Analyze spoken content for answers
    """

    def __init__(self, llm, tool_executor=None):
        super().__init__(
            name="audio_analyst",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )

    @property
    def uses_tools(self) -> bool:
        return True

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Analyze audio content using AudioTool."""
        existing_audio = memory_tracker.read_memory(entry_types=["audio_analysis"])

        # Check if AudioTool is available
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for audio analysis",
                metadata={"error": "no_tool_executor"},
            )

        audio_tool = self._tool_executor.get_tool("audio")
        if not audio_tool:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="AudioTool not available",
                metadata={"error": "no_audio_tool"},
            )

        # Get audio metadata and transcription
        metadata_result = self._tool_executor.execute("audio", "get_metadata")
        logger.debug(f"[AudioAnalystAgent] Audio metadata result: success={metadata_result.success}, output={metadata_result.output}, error={metadata_result.error}")
        transcribe_result = self._tool_executor.execute("audio", "transcribe")
        logger.debug(f"[AudioAnalystAgent] Audio transcription result: {transcribe_result}")

        # Track tool calls for divergence analysis
        tool_calls = [
            {
                "tool_name": "audio",
                "operation": "get_metadata",
                "params": {},
                "success": metadata_result.success,
                "output": str(metadata_result.output)[:200] if metadata_result.success else None,
                "error": metadata_result.error if not metadata_result.success else None,
            },
            {
                "tool_name": "audio",
                "operation": "transcribe",
                "params": {},
                "success": transcribe_result.success,
                "output": f"[{len(transcribe_result.output)} chars]" if transcribe_result.success else None,
                "error": transcribe_result.error if not transcribe_result.success else None,
            },
        ]

        if not transcribe_result.success:
            return AgentOutput(
                agent_name=self.name,
                action="error",
                content=f"Failed to transcribe audio: {transcribe_result.error}",
                metadata={"error": "transcription_failed", "tool_calls": tool_calls},
            )

        transcription = transcribe_result.output

        # Build analysis prompt
        audio_info = ""
        if metadata_result.success:
            meta = metadata_result.output
            audio_info = f"Audio: {meta.get('format', 'unknown')} format, {meta.get('duration', 'unknown')} seconds"

        prompt = f"""Task: {state.task_input.query}

{audio_info}

Audio Transcription:
---
{transcription}
---

Analyze the transcription above and extract information relevant to answering the task question. Be specific about what was said and who said it if identifiable.

Analysis:"""

        response = self._llm.generate(prompt, response_type="research")

        # Calculate confidence based on audio transcription quality
        audio_duration = None
        if metadata_result.success:
            audio_duration = metadata_result.output.get('duration')

        confidence_result = calculate_audio_confidence(
            transcription=transcription,
            transcription_success=True,
            audio_duration=audio_duration,
        )
        logger.debug(f"[AudioAnalyst] Confidence: {confidence_result.score:.2f} | Signals: {confidence_result.signals}")

        # Determine status based on confidence
        if confidence_result.score >= 0.7:
            entry_status = "high_confidence"
        elif confidence_result.score >= 0.4:
            entry_status = "preliminary"
        else:
            entry_status = "low_confidence"

        # Write analysis to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=response.content,
            entry_type="audio_analysis",
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "transcription_length": len(transcription),
                "audio_metadata": metadata_result.output if metadata_result.success else None,
            },
            status=entry_status,
            confidence=confidence_result.score,
        )

        return AgentOutput(
            agent_name=self.name,
            action="write_audio_analysis",
            content=response.content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "audio_analysis",
                "transcription_length": len(transcription),
                "tool_calls": tool_calls,
                **confidence_result.to_dict(),
            },
        )


def create_specialized_agents(llm, tool_executor=None) -> dict:
    """Create all specialized agents with the given LLM.

    Args:
        llm: LLM instance (OpenAIClient for real experiments)
        tool_executor: Optional ToolExecutor for agents that use tools

    Returns:
        Dictionary mapping agent names to agent instances
    """
    from .generalist_agent import GeneralistAgent

    return {
        "planner": PlannerAgent(llm=llm),
        "data_analyst": DataAnalystAgent(llm=llm, tool_executor=tool_executor),
        "python_executor": PythonAgent(llm=llm, tool_executor=tool_executor),
        "calculator": CalculatorAgent(llm=llm, tool_executor=tool_executor),
        "visual_analyst": VisualAnalystAgent(llm=llm, tool_executor=tool_executor),
        "fact_checker": FactCheckerAgent(llm=llm),
        "document_analyst": DocumentAnalystAgent(llm=llm, tool_executor=tool_executor),
        "researcher": ResearcherAgent(llm=llm),
        "critic": CriticAgent(llm=llm),
        "synthesizer": SynthesizerAgent(llm=llm),
        "web_researcher": WebResearcherAgent(llm=llm, tool_executor=tool_executor),
        "audio_analyst": AudioAnalystAgent(llm=llm, tool_executor=tool_executor),
        "generalist": GeneralistAgent(llm=llm, tool_executor=tool_executor),
    }
