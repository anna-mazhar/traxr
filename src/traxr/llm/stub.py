"""Deterministic scripted LLM stub — zero network, zero keys.

``DeterministicLLMStub`` returns canned responses keyed by
``(response_type, call_index)``. Five built-in scenarios shape the content
that the reference agent's ``SupervisorRouter``, ``PlannerAgent``,
``DataAnalystAgent`` and ``SynthesizerAgent`` parse, so each scenario
produces a distinct, reproducible trace shape:

==============  =============================================================
Scenario        Intended trace shape (validated in tests/integration)
==============  =============================================================
``identity``    plan -> data_analyst (one successful python tool call) ->
                synthesizer; exactly one ``final_answer`` event.
``wrong_answer``  Same routing shape as ``identity``, but the synthesizer
                returns a different final answer (silent corruption probe).
``reroute``     plan inserts a ``critic`` review step: routing path differs
                from ``identity`` while still concluding with an answer.
``halt_early``  The first reply reports a huge token usage, tripping the
                episode token budget: run ends with ``agent_halt`` and no
                ``final_answer``.
``loop``        plan chains repeated ``researcher`` subtasks: many repeated
                routing decisions (extended execution) before concluding.
==============  =============================================================

Custom scripts can be passed via ``script=`` (mapping ``response_type`` to a
list of replies); when a response type runs out of scripted replies the last
one repeats, and unscripted types fall back to ``"default"``.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from traxr.llm.types import LLMResponse, LLMToolResponse

__all__ = ["DeterministicLLMStub", "StubReply", "ScriptedToolCall", "SCENARIOS"]

SCENARIOS = ("identity", "wrong_answer", "reroute", "halt_early", "loop")

# A reply that never trips the router's rule-based quality checks
# (no "error:", "assuming", "not sure", "invalid", ... substrings).
_SAFE_DEFAULT_TEXT = "Proceeding with the task as planned."


@dataclass
class ScriptedToolCall:
    """Structured tool call returned by the stub.

    Duck-type compatible with ``traxr.mas.tools.tool_schema.StructuredToolCall``
    (the reference agent only accesses these four attributes), without
    importing the heavier ``traxr.mas`` package.
    """

    tool_name: str
    operation: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class StubReply:
    """One canned reply for a given ``(response_type, call_index)`` slot."""

    content: str = ""
    tool_calls: list[ScriptedToolCall] = field(default_factory=list)
    prompt_tokens: int = 25
    completion_tokens: int = 10


def _plan_reply(subtasks: list[dict[str, Any]], reasoning: str, **kwargs: Any) -> StubReply:
    """Build a planner reply in the JSON format ``ExecutionPlan.from_json`` parses."""
    return StubReply(
        content=json.dumps({"reasoning": reasoning, "subtasks": subtasks}),
        **kwargs,
    )


def _analyst_tool_reply(code: str) -> StubReply:
    """Build a data_analyst reply carrying one python.run tool call."""
    return StubReply(
        content="",
        tool_calls=[
            ScriptedToolCall(tool_name="python", operation="run", arguments={"code": code})
        ],
    )


# Code executed by the real PythonTool against the real DataFrame, so the
# printed answer reflects the (possibly perturbed) fixture data. The extra
# shape line feeds the data analyst's confidence heuristics.
_ANALYST_CODE = 'print(f"Answer: {len(df)}")\nprint(f"Table shape: {df.shape}")'

_IDENTITY_SUBTASKS: list[dict[str, Any]] = [
    {
        "id": "s1",
        "description": "Extract the requested value from the table",
        "agent": "data_analyst",
        "dependencies": [],
    },
    {
        "id": "s2",
        "description": "Produce the final answer",
        "agent": "synthesizer",
        "dependencies": ["s1"],
    },
]

_REROUTE_SUBTASKS: list[dict[str, Any]] = [
    {
        "id": "s1",
        "description": "Extract the requested value from the table",
        "agent": "data_analyst",
        "dependencies": [],
    },
    {
        "id": "s2",
        "description": "Critique the extracted value",
        "agent": "critic",
        "dependencies": ["s1"],
    },
    {
        "id": "s3",
        "description": "Produce the final answer",
        "agent": "synthesizer",
        "dependencies": ["s2"],
    },
]

_LOOP_SUBTASKS: list[dict[str, Any]] = [
    {
        "id": f"s{i}",
        "description": f"Gather background information, pass {i}",
        "agent": "researcher",
        "dependencies": [] if i == 1 else [f"s{i - 1}"],
    }
    for i in range(1, 5)
] + [
    {
        "id": "s5",
        "description": "Produce the final answer",
        "agent": "synthesizer",
        "dependencies": ["s4"],
    }
]

_RESEARCH_NOTE = (
    "Recorded background information from the retrieval store. The table "
    "holds the data needed for the question."
)
_CRITIQUE_TEXT = (
    "The analysis is consistent with the table and the computed value stands. No revision needed."
)


def _build_scenario_script(
    scenario: str, final_answer: str, wrong_answer: str
) -> dict[str, list[StubReply]]:
    """Build the per-response_type reply script for a built-in scenario."""
    base: dict[str, list[StubReply]] = {
        "plan": [
            _plan_reply(
                _IDENTITY_SUBTASKS,
                "Tabular question; analyze the table, then produce the final answer.",
            )
        ],
        "data_analyst": [_analyst_tool_reply(_ANALYST_CODE)],
        "synthesize": [StubReply(content=final_answer)],
        "route": [StubReply(content="NEXT_AGENT: synthesizer")],
        "research": [StubReply(content=_RESEARCH_NOTE)],
        "critique": [StubReply(content=_CRITIQUE_TEXT)],
        "default": [StubReply(content=_SAFE_DEFAULT_TEXT)],
    }

    if scenario == "identity":
        return base

    if scenario == "wrong_answer":
        base["synthesize"] = [StubReply(content=wrong_answer)]
        return base

    if scenario == "reroute":
        base["plan"] = [
            _plan_reply(
                _REROUTE_SUBTASKS,
                "Analyze the table, have the critic look at the result, then conclude.",
            )
        ]
        return base

    if scenario == "halt_early":
        # The plan reply alone blows the episode token budget, so the run
        # halts after the planning step without reaching a final answer.
        base["plan"] = [
            _plan_reply(
                _IDENTITY_SUBTASKS,
                "Tabular question; analyze the table, then produce the final answer.",
                completion_tokens=10_000_000,
            )
        ]
        return base

    if scenario == "loop":
        base["plan"] = [
            _plan_reply(
                _LOOP_SUBTASKS,
                "Gather background in several passes before concluding.",
            )
        ]
        return base

    raise ValueError(f"Unknown stub scenario {scenario!r}. Available: {', '.join(SCENARIOS)}")


class DeterministicLLMStub:
    """Scripted :class:`~traxr.llm.LLMClient` for tests, goldens, and demos.

    Args:
        scenario: One of :data:`SCENARIOS` (ignored when ``script`` is given).
        final_answer: The answer the ``identity``/``reroute``/``loop``
            synthesizer reply returns. Match it to the fixture data
            (e.g. the row count printed by the scripted analyst code).
        wrong_answer: The answer the ``wrong_answer`` scenario returns.
        script: Custom mapping ``response_type -> list[StubReply]``;
            overrides the scenario entirely.
    """

    def __init__(
        self,
        scenario: str = "identity",
        *,
        final_answer: str = "4",
        wrong_answer: str = "999",
        script: dict[str, list[StubReply]] | None = None,
    ):
        self.scenario = scenario
        self.model = f"stub:{scenario}"
        self._script = script or _build_scenario_script(scenario, final_answer, wrong_answer)
        self._counters: dict[str, int] = {}
        self._call_count = 0

    def _next_reply(self, response_type: str) -> StubReply:
        """Pop the next scripted reply for ``response_type`` (last one repeats)."""
        replies = self._script.get(response_type)
        if not replies:
            replies = self._script.get("default") or [StubReply(content=_SAFE_DEFAULT_TEXT)]
        index = self._counters.get(response_type, 0)
        self._counters[response_type] = index + 1
        return replies[min(index, len(replies) - 1)]

    def generate(
        self,
        prompt: str,
        response_type: str = "default",
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return the next scripted plain-text reply."""
        self._call_count += 1
        reply = self._next_reply(response_type)
        return LLMResponse(
            content=reply.content,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            model=self.model,
            finish_reason="stop",
            metadata={
                "call_count": self._call_count,
                "response_type": response_type,
                "scenario": self.scenario,
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
        """Return the next scripted reply, including any scripted tool calls."""
        self._call_count += 1
        reply = self._next_reply(response_type)
        tool_calls: list[Any] = list(reply.tool_calls)
        return LLMToolResponse(
            content=reply.content or None,
            tool_calls=tool_calls,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            model=self.model,
            finish_reason="tool_calls" if tool_calls else "stop",
            metadata={
                "call_count": self._call_count,
                "response_type": response_type,
                "scenario": self.scenario,
            },
        )

    def generate_with_retrieval(
        self,
        prompt: str,
        retrieval_content: list[str],
        response_type: str = "research",
    ) -> LLMResponse:
        """Retrieval-augmented text generation (same scripted replies)."""
        return self.generate(prompt, response_type)

    def reset_call_count(self) -> None:
        """Reset all call counters (determinism between paired runs)."""
        self._counters.clear()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of LLM calls made since the last reset."""
        return self._call_count
