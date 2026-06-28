"""Category 9 — LLM boundary tests for the built-in path.

Protocol conformance (mocked OpenAICompatibleClient + stub), base_url
plumbing, missing key -> LLMConnectionError, missing package ->
OptionalDependencyError. Fully offline: chat.completions.create is mocked.
"""

import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from traxr.errors import LLMConnectionError, OptionalDependencyError
from traxr.llm import DeterministicLLMStub, LLMClient, OpenAICompatibleClient


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch):
    """Tests must not depend on (or leak through) a real key in the env."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def make_client(**kwargs) -> OpenAICompatibleClient:
    kwargs.setdefault("api_key", "test-key")
    return OpenAICompatibleClient(**kwargs)


def fake_chat_response(content="hello", tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


class TestConstruction:
    def test_missing_api_key_raises_llm_connection_error(self):
        with pytest.raises(LLMConnectionError, match="API key required"):
            OpenAICompatibleClient()

    def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        client = OpenAICompatibleClient()
        assert client.client.api_key == "env-key"

    def test_missing_openai_package_raises_optional_dependency_error(self, monkeypatch):
        # Setting the sys.modules entry to None makes `import openai` fail.
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(OptionalDependencyError, match=r"traxr\[openai\]"):
            OpenAICompatibleClient(api_key="k")

    def test_base_url_is_plumbed_to_the_sdk_client(self):
        client = make_client(base_url="http://localhost:11434/v1", model="llama3")
        assert client.base_url == "http://localhost:11434/v1"
        assert str(client.client.base_url).startswith("http://localhost:11434/v1")

    def test_default_base_url_is_openai(self):
        client = make_client()
        assert client.base_url is None
        assert "api.openai.com" in str(client.client.base_url)

    def test_controlled_variable_defaults(self):
        client = make_client()
        assert client.temperature == 0.0
        assert client.seed == 42


class TestProtocolConformance:
    def test_openai_compatible_client_conforms(self):
        assert isinstance(make_client(), LLMClient)

    def test_stub_conforms(self):
        assert isinstance(DeterministicLLMStub(), LLMClient)


class TestGenerate:
    def test_generate_maps_response_fields(self, monkeypatch):
        client = make_client(model="test-model")
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return fake_chat_response(content="the answer")

        monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
        response = client.generate("what?", response_type="synthesize")

        assert response.content == "the answer"
        assert response.prompt_tokens == 11
        assert response.completion_tokens == 7
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        # Controlled-variable invariant: seed + temperature forwarded.
        assert captured["seed"] == 42
        assert captured["temperature"] == 0.0
        assert captured["messages"][0]["role"] == "system"
        assert "synthesis expert" in captured["messages"][0]["content"]
        assert captured["messages"][1] == {"role": "user", "content": "what?"}

    def test_generate_call_count_increments_and_resets(self, monkeypatch):
        client = make_client()
        monkeypatch.setattr(
            client.client.chat.completions, "create", lambda **kw: fake_chat_response()
        )
        client.generate("a")
        client.generate("b")
        assert client.call_count == 2
        client.reset_call_count()
        assert client.call_count == 0

    def test_api_error_returns_error_response(self, monkeypatch):
        import openai

        client = make_client()
        request = httpx.Request("POST", "http://test/v1/chat/completions")

        def fail(**kwargs):
            raise openai.APIError("boom", request=request, body=None)

        monkeypatch.setattr(client.client.chat.completions, "create", fail)
        response = client.generate("x")
        assert response.finish_reason == "error"
        assert "boom" in response.content

    def test_connection_error_raises_llm_connection_error(self, monkeypatch):
        import openai

        client = make_client(base_url="http://localhost:1/v1")
        request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")

        def fail(**kwargs):
            raise openai.APIConnectionError(request=request)

        monkeypatch.setattr(client.client.chat.completions, "create", fail)
        with pytest.raises(LLMConnectionError, match="localhost:1"):
            client.generate("x")


class TestGenerateWithTools:
    def make_tool_schema(self):
        from traxr.mas.tools.python_tool import PythonTool

        return PythonTool(timeout=1.0).get_schema()

    def test_tool_calls_parsed_into_structured_calls(self, monkeypatch):
        client = make_client()
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="python__run",
                arguments=json.dumps({"code": "print(1)"}),
            ),
        )
        monkeypatch.setattr(
            client.client.chat.completions,
            "create",
            lambda **kw: fake_chat_response(
                content=None, tool_calls=[tool_call], finish_reason="tool_calls"
            ),
        )
        response = client.generate_with_tools(
            "run it", tools=[self.make_tool_schema()], response_type="data_analyst"
        )
        assert response.has_tool_calls
        call = response.tool_calls[0]
        assert call.tool_name == "python"
        assert call.operation == "run"
        assert call.arguments == {"code": "print(1)"}
        assert call.call_id == "call_1"
        assert response.finish_reason == "tool_calls"

    def test_no_tools_falls_back_to_text_generation(self, monkeypatch):
        client = make_client()
        monkeypatch.setattr(
            client.client.chat.completions,
            "create",
            lambda **kw: fake_chat_response(content="plain"),
        )
        response = client.generate_with_tools("q", tools=[])
        assert response.content == "plain"
        assert response.tool_calls == []
