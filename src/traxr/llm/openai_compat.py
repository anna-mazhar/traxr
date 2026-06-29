"""OpenAI-compatible LLM client for the built-in reference agent.

Supports ``base_url`` so one client covers OpenAI, Azure proxies, Ollama,
vLLM, Together, Groq, LM Studio, OpenRouter, and proxied Anthropic.

Defaults are tuned for the controlled-variable invariant: ``temperature=0``
and a fixed ``seed`` so paired clean/perturbed runs differ only in the data.

The ``openai`` package is imported lazily; install with
``pip install "traxr[openai]"``.
"""

import os
from typing import Any

from traxr.errors import LLMConnectionError, OptionalDependencyError
from traxr.llm.types import LLMResponse, LLMToolResponse

__all__ = ["OpenAICompatibleClient"]

# System prompts per response_type, extracted unchanged from the source
# client (originally adapted from agent_library.json / Captain Agent).
_DATA_ANALYST_PROMPT = """## Your role
You are an expert Excel/CSV data analyst. You write Python code (pandas + openpyxl) to extract precise answers from spreadsheets. You handle messy, real-world files that have irregular structures.

## Critical rules
1. The DataFrame `df` is ALREADY loaded with `header=None` (raw cell values, no assumed headers). Column names are integers (0, 1, 2, ...).
2. NEVER call `pd.read_excel()` or `open()` — the data is already loaded.
3. ALWAYS print the final answer with prefix: print(f"Answer: {your_answer}")
4. Write ONE complete, executable code block. No pseudo-code, no explanations outside the code block.
5. When the question asks for a value "from the spreadsheet", return the EXACT value as it appears — preserve original casing, spacing, and format.

## Step-by-step approach for EVERY task
1. **Inspect first**: Print `df.head(20)` or `df.iloc[:20]` mentally to understand the structure before writing analysis code.
2. **Find the real header row**: Row 0 is often NOT the header. Look for the first row where most cells contain string labels that describe columns. Set it as header:
   ```python
   header_row = <detected_row>
   df.columns = df.iloc[header_row]
   df = df.iloc[header_row + 1:].reset_index(drop=True)
   ```
3. **Clean the data**: After setting headers:
   - `df.columns = df.columns.astype(str).str.strip()` to clean header whitespace
   - `df = df.dropna(how='all')` to remove fully blank rows
   - `df = df[df.iloc[:, 0].notna()]` if first column should never be empty (filter phantom rows)
   - For string columns: `df[col] = df[col].astype(str).str.strip()`
   - For numeric columns: `df[col] = pd.to_numeric(df[col], errors='coerce')`
   - For date columns: `df[col] = pd.to_datetime(df[col], errors='coerce')`
4. **Handle section markers**: If you see standalone values in a column (a category name with no data in other columns), these are section headers. Use forward-fill:
   ```python
   df['section'] = df['column_with_markers'].where(df['other_column'].isna()).ffill()
   ```
5. **Handle merged cells**: Merged cells appear as NaN in all but the top-left cell. Use `.ffill()` to propagate values downward or rightward as appropriate.
6. **Compute answer**: Filter, aggregate, sort, or extract as needed.
7. **Output**: First print(f"Answer: {answer}"), then optionally print verification data.

## Handling tricky Excel patterns
- **Blank separator rows**: Use `df.dropna(how='all')` to remove them, but only AFTER identifying section markers.
- **Numbers stored as text**: Use `pd.to_numeric(col, errors='coerce')` before any arithmetic.
- **Mixed date formats**: Use `pd.to_datetime(col, format='mixed', errors='coerce')`.
- **Multiple tables on one sheet**: Look for blank rows/columns that separate tables. Extract each sub-table by slicing the DataFrame.
- **Color-coded data**: If `df_colors` is available, it's a DataFrame of hex color codes aligned with `df`. Use it to filter or group by color:
  ```python
  mask = df_colors[col_idx] == 'FF0000'  # red cells
  red_rows = df[mask]
  ```
- **Hidden rows/columns**: If structural_summary mentions hidden rows, they may contain relevant data or should be excluded depending on the question.
- **Trailing phantom rows**: `ws.max_row` in Excel often overshoots. Always `dropna(how='all')` to trim.
- **Column with all NaN**: Skip it — it's likely a blank separator column.

## Common mistakes to avoid
- Do NOT assume column names exist. They are integers unless you explicitly set them.
- Do NOT forget to convert types before comparing or aggregating (e.g., comparing strings to numbers will silently fail).
- Do NOT use `.str` accessor on non-string columns without first converting with `.astype(str)`.
- Do NOT return a DataFrame as the answer — always extract and print(f"Answer: {specific_value}").
- If filtering returns empty results, print the unique values of the filter column to debug.
- ALWAYS use the "Answer: " prefix for your final answer so it can be extracted correctly."""

_PYTHON_EXECUTOR_PROMPT = """## Your role
You write and execute Python code to solve computational problems. Your code runs immediately in a live Python environment.

## Task and skill instructions
- Write executable Python code in ```python code blocks```
- Variables and DataFrames mentioned in the task are already loaded in your environment
- Use print() to output your final answer
- Your code must be complete and run without errors

## Important
- If the task mentions data is "already loaded", trust that it exists - don't say you need the data
- Write actual working code, not pseudo-code or explanations
- Include necessary imports (pandas, numpy, etc.)
- If code fails, analyze the error and write corrected code"""

_CALCULATOR_PROMPT = """## Your role
As Math_Expert, you bring an in-depth understanding of mathematical calculations. You are a reliable resource for tackling complex computational problems.

## Task and skill instructions
- Your primary task involves applying your knowledge to perform mathematical calculations.
- Your expertise in Python programming is noteworthy, especially your proficiency with scientific libraries such as numpy and scipy.
- You excel in mathematical problem-solving, particularly when it comes to finding approximate solutions where exact calculations prove to be formidable.
- Unit conversion is second nature to you.

## Useful instructions for task-solving
- Follow the instruction provided by the user.
- Solve the task step by step if you need to.
- If a plan is not provided, explain your plan first.
- When you find an answer, verify the answer carefully.

## How to use code?
- Suggest python code (in a python coding block) for the Computer_terminal to execute.
- When using code, you must indicate the script type in the code block.
- Do not suggest incomplete code which requires users to modify.
- Use 'print' function for the output when relevant.
- If the result indicates there is an error, fix the error and output the code again."""

_CRITIQUE_PROMPT = """## Your role
Verification_Expert is a proficient individual with expertise in verifying and validating results. You bring meticulous attention to detail and a rigorous approach to verification.

## Task and skill instructions
- Crucially, apply a stringent verification process to all outcomes.
- This involves cross-checking results against theoretical expectations, running test cases, and validating solutions with alternative methods.
- Ensure the utmost accuracy and reliability of the work produced.

## Useful instructions
- Follow the instruction provided by the user.
- When you find an answer, verify the answer carefully.
- Include verifiable evidence in your response if possible."""

_TEXT_SYSTEM_PROMPTS: dict[str, str] = {
    "data_analyst": _DATA_ANALYST_PROMPT,
    "python_executor": _PYTHON_EXECUTOR_PROMPT,
    "calculator": _CALCULATOR_PROMPT,
    "research": (
        "You are a research assistant. Analyze the provided information and "
        "write clear, factual notes with citations to sources."
    ),
    "critique": _CRITIQUE_PROMPT,
    "synthesize": (
        "You are a synthesis expert. Combine all available information into a "
        "clear, concise final answer."
    ),
    "route": (
        "You are a task router. Determine which agent should handle the next "
        "step based on the current state."
    ),
    "plan": (
        "You are a task planner. Break down complex tasks into logical "
        "subtasks and assign appropriate agents."
    ),
    "default": "You are a helpful assistant.",
}

_TOOL_SYSTEM_PROMPTS: dict[str, str] = {
    "data_analyst": (
        "You are an expert data analyst. Use the provided tools to analyze "
        "data and answer questions."
    ),
    "python_executor": "You are a Python code executor. Use tools to run code and computations.",
    "calculator": "You are a math expert. Use calculator tools for precise computations.",
    "research": "You are a research assistant. Use search and fetch tools to find information.",
    "default": (
        "You are a helpful assistant. Use the provided tools when they would "
        "help answer the question."
    ),
}

_VISION_SYSTEM_PROMPTS: dict[str, str] = {
    "research": (
        "You are a visual analyst. Analyze the image carefully and extract "
        "all relevant information to answer the question."
    ),
    "visual_analysis": (
        "You are an expert visual analyst. Examine the image in detail, "
        "identifying key elements, patterns, text, charts, or any visual "
        "information relevant to the task."
    ),
    "default": "You are a helpful assistant with vision capabilities.",
}


def _import_openai() -> Any:
    """Import the optional ``openai`` package or fail with the pip extra."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise OptionalDependencyError(
            "The 'openai' package is required for OpenAICompatibleClient. "
            'Install it with: pip install "traxr[openai]"'
        ) from exc
    return openai


class OpenAICompatibleClient:
    """:class:`~traxr.llm.LLMClient` for any OpenAI-compatible endpoint.

    Args:
        model: Model name as known to the endpoint.
        api_key: API key; falls back to the ``OPENAI_API_KEY`` environment
            variable. Local servers (Ollama, LM Studio, ...) usually accept
            any non-empty string.
        base_url: Endpoint base URL (``None`` = api.openai.com).
        seed: Sampling seed forwarded to the endpoint (reproducibility).
        temperature: Sampling temperature; defaults to ``0.0`` for the
            controlled-variable invariant. ``None`` omits the parameter.
        max_retries: How many times the OpenAI SDK retries a transient
            failure (network error, ``429``, ``5xx``) before raising. Forwarded
            to ``openai.OpenAI(max_retries=...)``; defaults to ``2`` (the SDK
            default). Set ``0`` to disable automatic retries.

    Raises:
        OptionalDependencyError: If the ``openai`` package is not installed.
        LLMConnectionError: If no API key is available.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int = 42,
        temperature: float | None = 0.0,
        max_retries: int = 2,
    ):
        openai = _import_openai()
        self._openai = openai

        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.base_url = base_url
        self.max_retries = max_retries
        self._call_count = 0

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMConnectionError(
                "API key required for OpenAICompatibleClient. Pass api_key= or set "
                "the OPENAI_API_KEY environment variable. For local OpenAI-compatible "
                "servers (Ollama, vLLM, LM Studio, ...) any non-empty string works, "
                'e.g. OpenAICompatibleClient(base_url="http://localhost:11434/v1", '
                'api_key="local").'
            )

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)

    def _completion_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Common chat.completions kwargs (seed; temperature only if set)."""
        kwargs["model"] = kwargs.get("model", self.model)
        kwargs["seed"] = self.seed
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    def generate(
        self,
        prompt: str,
        response_type: str = "default",
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a plain-text response."""
        self._call_count += 1
        system_prompt = _TEXT_SYSTEM_PROMPTS.get(response_type, _TEXT_SYSTEM_PROMPTS["default"])

        try:
            response = self.client.chat.completions.create(
                **self._completion_kwargs(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
        except (self._openai.AuthenticationError, self._openai.APIConnectionError) as exc:
            raise LLMConnectionError(
                f"Could not reach the LLM endpoint (model={self.model!r}, "
                f"base_url={self.base_url or 'api.openai.com'!r}): {exc}"
            ) from exc
        except self._openai.APIError as exc:
            return LLMResponse(
                content=f"[OpenAI API Error: {exc}]",
                prompt_tokens=0,
                completion_tokens=0,
                model=self.model,
                finish_reason="error",
                metadata={"error": str(exc)},
            )

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            finish_reason=response.choices[0].finish_reason or "stop",
            metadata={
                "call_count": self._call_count,
                "response_type": response_type,
                "seed": self.seed,
            },
        )

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        response_type: str = "default",
        context: dict[str, Any] | None = None,
        system_prompt_override: str | None = None,
    ) -> LLMToolResponse:
        """Generate a response with native OpenAI function calling."""
        import json

        # Lazy: pulls in the (internal, heavier) reference-agent tool schemas.
        from traxr.mas.tools.tool_schema import StructuredToolCall, schemas_to_openai_tools

        self._call_count += 1
        system_prompt = system_prompt_override or _TOOL_SYSTEM_PROMPTS.get(
            response_type, _TOOL_SYSTEM_PROMPTS["default"]
        )

        openai_tools = schemas_to_openai_tools(tools)
        if not openai_tools:
            # No valid tools — fall back to text-only generation.
            response = self.generate(prompt, response_type, context)
            return LLMToolResponse(
                content=response.content,
                tool_calls=[],
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                model=self.model,
                finish_reason="stop",
                metadata=response.metadata,
            )

        try:
            response = self.client.chat.completions.create(
                **self._completion_kwargs(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tools=openai_tools,
                    tool_choice="auto",
                )
            )
        except (self._openai.AuthenticationError, self._openai.APIConnectionError) as exc:
            raise LLMConnectionError(
                f"Could not reach the LLM endpoint (model={self.model!r}, "
                f"base_url={self.base_url or 'api.openai.com'!r}): {exc}"
            ) from exc
        except self._openai.APIError as exc:
            return LLMToolResponse(
                content=f"[OpenAI API Error: {exc}]",
                tool_calls=[],
                prompt_tokens=0,
                completion_tokens=0,
                model=self.model,
                finish_reason="error",
                metadata={"error": str(exc)},
            )

        choice = response.choices[0]
        message = choice.message

        structured_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                function = getattr(tc, "function", None)
                if function is None:
                    continue
                try:
                    args = json.loads(function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                structured_calls.append(
                    StructuredToolCall.from_openai_function_name(
                        function_name=function.name,
                        arguments=args,
                        call_id=tc.id or "",
                    )
                )

        usage = response.usage
        return LLMToolResponse(
            content=message.content,
            tool_calls=structured_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            finish_reason=choice.finish_reason or "stop",
            metadata={
                "call_count": self._call_count,
                "response_type": response_type,
                "seed": self.seed,
            },
        )

    def generate_with_retrieval(
        self,
        prompt: str,
        retrieval_content: list[str],
        response_type: str = "research",
    ) -> LLMResponse:
        """Generate a response that incorporates retrieval content."""
        if retrieval_content:
            retrieval_section = "\n\n".join(
                f"[Source {i + 1}]: {content}" for i, content in enumerate(retrieval_content)
            )
            full_prompt = (
                f"{prompt}\n\nRetrieved Information:\n{retrieval_section}\n\n"
                "Please analyze the above sources and provide your response "
                "with citations to the source numbers."
            )
        else:
            full_prompt = f"{prompt}\n\n(No retrieved information available)"

        return self.generate(full_prompt, response_type)

    def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        media_type: str = "image/png",
        response_type: str = "research",
    ) -> LLMResponse:
        """Generate a response that analyzes an image (vision-capable models).

        Retained from the extracted client for interface fidelity; the
        built-in agent's image path is inert in v1.
        """
        self._call_count += 1

        vision_model = self.model
        if "gpt-4" in self.model.lower() and "vision" not in self.model.lower():
            if "turbo" in self.model.lower() or "o" in self.model.lower():
                vision_model = self.model  # GPT-4 Turbo / GPT-4o support vision
            else:
                vision_model = "gpt-4o"

        system_prompt = _VISION_SYSTEM_PROMPTS.get(response_type, _VISION_SYSTEM_PROMPTS["default"])

        try:
            response = self.client.chat.completions.create(
                **self._completion_kwargs(
                    model=vision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{image_base64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=4096,
                )
            )
        except (self._openai.AuthenticationError, self._openai.APIConnectionError) as exc:
            raise LLMConnectionError(
                f"Could not reach the LLM endpoint (model={vision_model!r}, "
                f"base_url={self.base_url or 'api.openai.com'!r}): {exc}"
            ) from exc
        except self._openai.APIError as exc:
            return LLMResponse(
                content=f"[OpenAI Vision API Error: {exc}]",
                prompt_tokens=0,
                completion_tokens=0,
                model=vision_model,
                finish_reason="error",
                metadata={"error": str(exc)},
            )

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=vision_model,
            finish_reason=response.choices[0].finish_reason or "stop",
            metadata={
                "call_count": self._call_count,
                "response_type": response_type,
                "seed": self.seed,
                "vision_model": vision_model,
            },
        )

    def reset_call_count(self) -> None:
        """Reset the call counter (determinism between paired runs)."""
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of LLM calls made by this client instance."""
        return self._call_count

    def close(self) -> None:
        """Close the underlying HTTP client to release connections."""
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
