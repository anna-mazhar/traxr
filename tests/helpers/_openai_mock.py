"""Offline OpenAI-client doubles for the capture-layer tests.

Real ``openai`` SDK clients over ``httpx.MockTransport`` — every request is
served from canned chat.completion JSON (or SSE chunk lists for streaming),
so the full parse path the Tier 0 wrapper depends on is exercised with zero
network and zero keys. Also home of the fixture external agent (a plain
function satisfying the ``AgentRunner`` contract).
"""

import json
from typing import Any

import httpx
import openai

MODEL = "mock-model"
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def completion(
    content: str = "ok",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = USAGE,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """A canned chat.completion response body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """A canned tool_calls entry."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def chunk(
    *,
    content: str | None = None,
    tool_call_deltas: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """A canned chat.completion.chunk body (streaming)."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_call_deltas is not None:
        delta["tool_calls"] = tool_call_deltas
    body: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": []
        if usage is not None and not delta and finish_reason is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


class MockOpenAIServer:
    """Serves a fixed sequence of canned responses; records request bodies.

    Each item in ``responses`` is either a completion dict (JSON response)
    or a list of chunk dicts (served as an SSE stream). With ``cycle=True``
    the sequence repeats — for multi-run experiments where every run makes
    the same calls.
    """

    def __init__(self, responses: list[Any], *, cycle: bool = False):
        self._responses = list(responses)
        self._cycle = cycle
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        index = len(self.requests) - 1
        if self._cycle:
            index %= len(self._responses)
        spec = self._responses[index]
        if isinstance(spec, list):
            payload = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in spec)
            payload += b"data: [DONE]\n\n"
            return httpx.Response(
                200, content=payload, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=spec)

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="test-key",
            base_url="http://traxr.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )

    def async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://traxr.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )


def make_fixture_agent(client: openai.OpenAI) -> Any:
    """The fixture external agent: a plain ``(Task) -> str`` function.

    Makes deterministic chat.completions calls through ``client``, resolving
    any tool calls with a canned tool message, and returns the final content.
    """

    def agent(task: Any) -> str:
        question = f"{task.question}\nFiles: {', '.join(f.name for f in task.files)}"
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        response = client.chat.completions.create(model=MODEL, messages=messages)
        message = response.choices[0].message
        while message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        tool_call(tc.id, tc.function.name, tc.function.arguments or "")
                        for tc in message.tool_calls
                    ],
                }
            )
            for tc in message.tool_calls:
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": "tool-output-42"}
                )
            response = client.chat.completions.create(model=MODEL, messages=messages)
            message = response.choices[0].message
        return message.content or ""

    return agent
