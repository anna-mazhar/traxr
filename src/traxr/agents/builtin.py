"""``builtin_agent()`` — factory facade over the reference MAS EpisodeRunner.

The built-in agent is the no-key path: pair it with
:class:`~traxr.llm.DeterministicLLMStub` for offline runs, or with
:class:`~traxr.llm.OpenAICompatibleClient` for real models.

Handles per-file-type tool registration and the perturbation injection
consumption point (``metadata["injected_pdf_content"]`` →
``PDFTool.inject_perturbed_content()``).
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from traxr.errors import OptionalDependencyError
from traxr.llm.protocol import LLMClient
from traxr.trace.collector import TraceCollector

__all__ = ["BuiltinAgent", "builtin_agent"]

logger = logging.getLogger(__name__)

_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".log"}


def _import_mas() -> None:
    """Import the reference MAS or fail with the missing pip extra."""
    try:
        import traxr.mas.core.runner  # noqa: F401
    except ImportError as exc:
        raise OptionalDependencyError(
            "The built-in reference agent needs pandas (and openpyxl/pdfplumber "
            "for xlsx/pdf files). Install the extras with: "
            'pip install "traxr[pandas,document]"'
        ) from exc


class BuiltinAgent:
    """One run-scoped instance of the built-in reference multi-agent system.

    Create instances via :func:`builtin_agent`; each instance carries fresh
    router/plan state and should be used for exactly one run.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        enable_web_tools: bool = False,
        enable_python_tool: bool = True,
        max_steps: int = 12,
        max_tokens: int | None = 100_000,
        python_tool_timeout: float = 30.0,
        seed: int = 42,
    ):
        _import_mas()
        self._llm = llm
        self._enable_web_tools = enable_web_tools
        self._enable_python_tool = enable_python_tool
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._python_tool_timeout = python_tool_timeout
        self._seed = seed
        #: Token/step cost of the most recent run (set by :meth:`run`),
        #: read by the experiment runner for token-overhead metrics.
        self.last_cost: Any = None

    def run(
        self,
        files: list[str | Path] | tuple[str | Path, ...],
        question: str,
        *,
        expected_answer: str | None = None,
        collector: TraceCollector | None = None,
        metadata: dict[str, Any] | None = None,
        task_id: str = "task",
    ) -> str | None:
        """Run one episode over ``files`` + ``question``; return the final answer.

        Args:
            files: Input artifact paths. v1 supports exactly one file
                (CSV/XLSX/TXT/MD/PDF).
            question: The task question.
            expected_answer: Optional reference answer (recorded, not used
                for control flow).
            collector: Optional :class:`~traxr.trace.TraceCollector`; when
                given, the runner emits routing/tool/memory/answer events
                into it.
            metadata: Extra ``TaskInput.metadata`` entries. The perturbation
                engine's built-in-agent delivery path passes
                ``injected_pdf_content`` here (consumed by
                ``PDFTool.inject_perturbed_content``).
            task_id: Identifier recorded in the episode spec.
        """
        from traxr.mas.agents.specialized_agents import create_specialized_agents
        from traxr.mas.core.episode_spec import (
            EpisodeSpec,
            ExperimentCondition,
            TerminationCriteria,
        )
        from traxr.mas.core.runner import EpisodeRunner
        from traxr.mas.core.state import TaskInput

        if len(files) != 1:
            raise ValueError(
                f"The built-in agent supports exactly one input file in v1, got "
                f"{len(files)}. Multi-file tasks are on the roadmap."
            )
        file_path = Path(files[0])
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        task_metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": file_path.name,
        }
        if metadata:
            task_metadata.update(metadata)

        task_input = TaskInput(
            task_id=task_id,
            query=question,
            expected_answer=expected_answer,
            metadata=task_metadata,
        )

        if hasattr(self._llm, "reset_call_count"):
            self._llm.reset_call_count()

        tool_executor = self._create_tools(task_input)
        agents = create_specialized_agents(llm=self._llm, tool_executor=tool_executor)
        retrieval = self._create_retrieval(task_input)

        spec = EpisodeSpec(
            task_id=task_id,
            seed=self._seed,
            agent_sequence=tuple(agents),
            termination=TerminationCriteria(
                max_steps=self._max_steps,
                max_tokens=self._max_tokens,
            ),
        )

        runner = EpisodeRunner(agents=agents, retrieval=retrieval, llm_stub=self._llm)
        result = runner.run(
            spec=spec,
            condition=ExperimentCondition(),
            task_input=task_input,
            trace_collector=collector,
        )
        self.last_cost = result.cost
        return result.final_answer

    __call__ = run

    def _create_tools(self, task_input: Any) -> Any:
        """Per-file-type tool registration (mined from experiment_runner)."""
        from traxr.mas.tools.base import ToolExecutor
        from traxr.mas.tools.calculator_tool import CalculatorTool
        from traxr.mas.tools.python_tool import PythonTool

        executor = ToolExecutor()
        file_path = Path(task_input.metadata["file_path"])
        file_ext = file_path.suffix.lower()

        def make_python_tool() -> Any:
            return PythonTool(timeout=self._python_tool_timeout)

        if file_ext in _TABULAR_EXTENSIONS:
            from traxr.mas.tools.excel_tool import ExcelTool

            excel_tool = ExcelTool(file_path=str(file_path))
            load_result = excel_tool.execute("load")
            if not load_result.success:
                # CSV fallback mined from experiment_runner: ExcelTool's load
                # path is Excel-engine based, so plain CSVs land here.
                try:
                    import pandas as pd

                    df = pd.read_csv(file_path)
                    if not df.empty:
                        excel_tool.df = df
                        load_result.success = True
                        logger.info("CSV fallback succeeded: %s", df.shape)
                except Exception as csv_err:
                    logger.warning("CSV fallback failed: %s", csv_err)
            if not load_result.success:
                logger.warning("Failed to load tabular file: %s", load_result.error)
                task_input.metadata["file_load_error"] = load_result.error
            executor.register_tool(excel_tool)

            if self._enable_python_tool:
                context: dict[str, Any] = {}
                if excel_tool.df is not None and not excel_tool.df.empty:
                    context["df"] = excel_tool.df
                if excel_tool.df_colors is not None and not excel_tool.df_colors.empty:
                    context["df_colors"] = excel_tool.df_colors
                if context:
                    python_tool = make_python_tool()
                    python_tool.set_context(context)
                    executor.register_tool(python_tool)

        elif file_ext == ".pdf":
            from traxr.mas.tools.pdf_tool import PDFTool

            pdf_tool = PDFTool(file_path=str(file_path))
            load_result = pdf_tool.execute("load")
            if not load_result.success:
                logger.warning("Failed to load PDF file: %s", load_result.error)

            # Perturbation injection consumption point (built-in-agent PDF
            # delivery): perturbed text rides in via task metadata and
            # replaces the tool's extracted content.
            injected_content = task_input.metadata.get("injected_pdf_content")
            if injected_content:
                pdf_tool.inject_perturbed_content(injected_content)
                logger.info(
                    "Injected perturbed content into PDFTool (%d chars)",
                    len(injected_content),
                )

            executor.register_tool(pdf_tool)

            # Extract text for PythonTool context (returns injected content
            # when present).
            text_result = pdf_tool.execute("extract_all_text", max_pages=20)
            pdf_text = text_result.output if text_result.success else ""
            task_input.metadata["file_type"] = ".pdf"
            task_input.metadata["pdf_text"] = pdf_text
            if self._enable_python_tool:
                python_tool = make_python_tool()
                python_tool.set_context(
                    {
                        "pdf_text": pdf_text,
                        "file_path": str(file_path),
                        "file_name": task_input.metadata.get("file_name", ""),
                    }
                )
                executor.register_tool(python_tool)

        elif file_ext in _TEXT_EXTENSIONS:
            try:
                text_content = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to read text file: %s", exc)
            else:
                task_input.metadata["file_content"] = text_content
                task_input.metadata["file_type"] = file_ext
                if self._enable_python_tool:
                    key = "file_code" if file_ext == ".py" else "file_content"
                    python_tool = make_python_tool()
                    python_tool.set_context(
                        {
                            key: text_content,
                            "file_path": str(file_path),
                            "file_name": task_input.metadata.get("file_name", ""),
                        }
                    )
                    executor.register_tool(python_tool)
        else:
            logger.warning(
                "No specialized tool for extension %r; registering general tools only",
                file_ext,
            )

        # General-purpose tools.
        if self._enable_python_tool and not executor.get_tool("python"):
            executor.register_tool(make_python_tool())
        executor.register_tool(CalculatorTool())

        # Web tools are opt-in (default OFF): keeps runs offline/deterministic.
        if self._enable_web_tools:
            from traxr.mas.tools.web_fetch_tool import WebFetchTool
            from traxr.mas.tools.web_search_tool import WebSearchTool

            executor.register_tool(WebSearchTool(backend="duckduckgo"))
            executor.register_tool(WebFetchTool())

        return executor

    def _create_retrieval(self, task_input: Any) -> Any:
        """Keyword retrieval over text-file content (mined from experiment_runner).

        Tabular and PDF files are tool-handled and stay out of retrieval.
        """
        from traxr.mas.retrieval.items import RetrievalItem
        from traxr.mas.retrieval.keyword_retrieval import InMemoryRetrieval

        retrieval = InMemoryRetrieval(seed=self._seed)
        text_content = task_input.metadata.get("file_content")
        if text_content:
            retrieval.add_item(
                RetrievalItem(
                    content=text_content,
                    score=0.0,
                    source=task_input.metadata.get("file_name", "attachment"),
                )
            )
        return retrieval


def builtin_agent(
    llm: LLMClient,
    *,
    enable_web_tools: bool = False,
    enable_python_tool: bool = True,
    max_steps: int = 12,
    max_tokens: int | None = 100_000,
    python_tool_timeout: float = 30.0,
    seed: int = 42,
) -> Callable[[], BuiltinAgent]:
    """Build a factory of :class:`BuiltinAgent` instances over the reference MAS.

    The returned zero-argument factory is called once per run (clean baseline
    or perturbation), so every run gets fresh router/plan state while sharing
    the same LLM client and configuration.

    Args:
        llm: Any :class:`~traxr.llm.LLMClient` (OpenAICompatibleClient,
            DeterministicLLMStub, or your own implementation).
        enable_web_tools: Register web_search/web_fetch tools. Default OFF —
            opt-in only, keeps runs offline and deterministic.
        enable_python_tool: Register the (subprocess-sandboxed) python tool.
        max_steps: Episode step budget.
        max_tokens: Episode token budget (``None`` disables the check).
        python_tool_timeout: Seconds before LLM-written code is killed.
        seed: Seed recorded in the episode spec and used by retrieval.

    Returns:
        A zero-argument callable producing fresh :class:`BuiltinAgent`
        instances.

    Raises:
        OptionalDependencyError: If the reference agent's dependencies
            (pandas, ...) are not installed.

    Example::

        from traxr.llm import DeterministicLLMStub
        from traxr.agents import builtin_agent

        factory = builtin_agent(llm=DeterministicLLMStub("identity"))
        agent = factory()
        answer = agent.run(["data.csv"], "How many rows does the table have?")
    """
    _import_mas()

    def factory() -> BuiltinAgent:
        return BuiltinAgent(
            llm,
            enable_web_tools=enable_web_tools,
            enable_python_tool=enable_python_tool,
            max_steps=max_steps,
            max_tokens=max_tokens,
            python_tool_timeout=python_tool_timeout,
            seed=seed,
        )

    return factory
