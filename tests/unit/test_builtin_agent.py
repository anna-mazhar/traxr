"""Unit tests for the builtin_agent factory facade (wiring, flags, injection)."""

import sys

import pytest

from traxr.agents import BuiltinAgent, builtin_agent
from traxr.errors import OptionalDependencyError
from traxr.llm import DeterministicLLMStub
from traxr.llm.stub import StubReply
from traxr.trace import TraceCollector


def make_task_input(file_path, metadata=None):
    from traxr.mas.core.state import TaskInput

    task_metadata = {"file_path": str(file_path), "file_name": file_path.name}
    task_metadata.update(metadata or {})
    return TaskInput(task_id="t", query="q", metadata=task_metadata)


@pytest.fixture()
def agent() -> BuiltinAgent:
    return builtin_agent(llm=DeterministicLLMStub("identity"))()


class TestFactory:
    def test_factory_returns_fresh_instances(self):
        factory = builtin_agent(llm=DeterministicLLMStub("identity"))
        assert factory() is not factory()

    def test_missing_mas_dependency_raises_optional_dependency_error(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "traxr.mas.core.runner", raising=False)
        monkeypatch.setitem(sys.modules, "traxr.mas.core.runner", None)
        with pytest.raises(OptionalDependencyError, match="pandas"):
            builtin_agent(llm=DeterministicLLMStub("identity"))


class TestInputValidation:
    def test_multiple_files_rejected(self, agent, fixtures_dir):
        with pytest.raises(ValueError, match="exactly one input file"):
            agent.run([fixtures_dir / "sample.csv", fixtures_dir / "sample.txt"], "q")

    def test_missing_file_rejected(self, agent, tmp_path):
        with pytest.raises(FileNotFoundError):
            agent.run([tmp_path / "nope.csv"], "q")


class TestToolWiring:
    def test_web_tools_off_by_default(self, agent, fixtures_dir):
        executor = agent._create_tools(make_task_input(fixtures_dir / "sample.csv"))
        assert executor.get_tool("web_search") is None
        assert executor.get_tool("web_fetch") is None
        assert executor.get_tool("excel") is not None
        assert executor.get_tool("python") is not None
        assert executor.get_tool("calculator") is not None

    def test_web_tools_opt_in(self, fixtures_dir):
        agent = builtin_agent(llm=DeterministicLLMStub("identity"), enable_web_tools=True)()
        executor = agent._create_tools(make_task_input(fixtures_dir / "sample.csv"))
        assert executor.get_tool("web_search") is not None
        assert executor.get_tool("web_fetch") is not None

    def test_python_tool_can_be_disabled(self, fixtures_dir):
        agent = builtin_agent(llm=DeterministicLLMStub("identity"), enable_python_tool=False)()
        executor = agent._create_tools(make_task_input(fixtures_dir / "sample.csv"))
        assert executor.get_tool("python") is None

    def test_python_tool_timeout_is_plumbed(self, fixtures_dir):
        agent = builtin_agent(llm=DeterministicLLMStub("identity"), python_tool_timeout=3.5)()
        executor = agent._create_tools(make_task_input(fixtures_dir / "sample.csv"))
        assert executor.get_tool("python").timeout == 3.5

    def test_unknown_extension_gets_general_tools_only(self, agent, tmp_path):
        weird = tmp_path / "input.xyz"
        weird.write_text("data")
        executor = agent._create_tools(make_task_input(weird))
        assert executor.get_tool("excel") is None
        assert executor.get_tool("pdf") is None
        assert executor.get_tool("python") is not None
        assert executor.get_tool("calculator") is not None

    def test_text_file_content_loaded_into_python_context(self, agent, fixtures_dir):
        task_input = make_task_input(fixtures_dir / "sample.txt")
        executor = agent._create_tools(task_input)
        assert "file_content" in task_input.metadata
        python_tool = executor.get_tool("python")
        assert python_tool.context["file_content"] == task_input.metadata["file_content"]


class TestPdfInjectionConsumption:
    """The consumer half of the built-in-agent PDF delivery path (M4 producer)."""

    def test_injected_pdf_content_reaches_the_pdf_tool(self, agent, fixtures_dir):
        perturbed = "PERTURBED PDF TEXT: the answer is 7."
        task_input = make_task_input(
            fixtures_dir / "sample.pdf", metadata={"injected_pdf_content": perturbed}
        )
        executor = agent._create_tools(task_input)

        pdf_tool = executor.get_tool("pdf")
        assert pdf_tool.has_injected_content
        extracted = pdf_tool.execute("extract_all_text")
        assert extracted.success
        assert extracted.output == perturbed
        # The python-tool context sees the injected text too.
        assert task_input.metadata["pdf_text"] == perturbed

    def test_without_injection_pdf_text_is_real_extraction(self, agent, fixtures_dir):
        task_input = make_task_input(fixtures_dir / "sample.pdf")
        executor = agent._create_tools(task_input)
        assert not executor.get_tool("pdf").has_injected_content
        assert "pdf_text" in task_input.metadata


class TestRetrieval:
    def test_text_content_is_retrievable(self, agent, fixtures_dir):
        task_input = make_task_input(fixtures_dir / "sample.txt")
        agent._create_tools(task_input)  # populates file_content metadata
        retrieval = agent._create_retrieval(task_input)
        assert retrieval.get_total_items() == 1

    def test_tabular_files_stay_out_of_retrieval(self, agent, fixtures_dir):
        task_input = make_task_input(fixtures_dir / "sample.csv")
        agent._create_tools(task_input)
        retrieval = agent._create_retrieval(task_input)
        assert retrieval.get_total_items() == 0


class TestTextFileRun:
    def test_text_file_end_to_end_with_retrieval_shown(self, fixtures_dir):
        """Researcher route over a text file emits retrieval_shown events."""
        script = {
            "plan": [
                StubReply(
                    content=(
                        '{"reasoning": "Read the document, then conclude.", "subtasks": ['
                        '{"id": "s1", "description": "Gather background information",'
                        ' "agent": "researcher", "dependencies": []},'
                        '{"id": "s2", "description": "Produce the final answer",'
                        ' "agent": "synthesizer", "dependencies": ["s1"]}]}'
                    )
                )
            ],
            "research": [StubReply(content="Recorded the document contents for the question.")],
            "synthesize": [StubReply(content="done")],
            "route": [StubReply(content="NEXT_AGENT: synthesizer")],
            "default": [StubReply(content="Proceeding with the task as planned.")],
        }
        stub = DeterministicLLMStub(script=script)
        agent = builtin_agent(llm=stub, max_steps=6)()
        collector = TraceCollector(run_label="text")
        answer = agent.run(
            [fixtures_dir / "sample.txt"], "What does the document say?", collector=collector
        )
        assert answer == "done"
        shown = collector.get_events_by_type("retrieval_shown")
        assert shown, "expected retrieval_shown for the researcher route"
        assert shown[0].payload["item_count"] >= 1
