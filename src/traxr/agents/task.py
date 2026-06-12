"""The ``Task``/``AgentRunner`` contract and the single-run invocation harness.

An external agent is any callable ``(Task) -> str``. The experiment runner
builds one :class:`Task` per run (clean baseline or perturbation) and invokes
the agent through :func:`invoke_agent`, which binds the capture session,
emits the ``final_answer`` event from the return value, and converts crashes
into ``agent_error`` events.
"""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traxr.capture.context import CaptureSession, bind_session
from traxr.errors import AgentContractError
from traxr.trace.collector import TraceCollector

__all__ = ["AgentRunner", "Task", "invoke_agent"]


@dataclass(frozen=True)
class Task:
    """One run's input, as handed to an :data:`AgentRunner`.

    Attributes:
        question: The task question.
        files: Input artifact paths. Clean runs receive the originals;
            perturbed runs receive copies with the ORIGINAL basenames in a
            fresh temp dir, so file names never leak the condition.
        run_label: ``"baseline"`` or the perturbation name (informational).
        metadata: Extra experiment context (never condition-revealing).
    """

    question: str
    files: tuple[Path, ...]
    run_label: str = "baseline"
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: The external-agent contract: any callable taking a Task and returning the
#: final answer as a string (non-str returns raise AgentContractError).
AgentRunner = Callable[[Task], str]


def invoke_agent(
    runner: AgentRunner,
    task: Task,
    collector: TraceCollector,
    *,
    max_llm_calls_per_run: int | None = None,
    store_llm_content: bool = False,
) -> str:
    """Run ``runner`` on ``task`` with Tier 0 capture bound to ``collector``.

    The harness — not the agent — emits the ``final_answer`` event from the
    return value. An agent exception is recorded as an ``agent_error`` event
    (the partial trace stays analyzable) and re-raised; the record-vs-raise
    policy is the experiment runner's concern.

    Args:
        runner: The agent callable.
        task: This run's input.
        collector: The run's trace collector.
        max_llm_calls_per_run: LLM-call budget enforced inside the Tier 0
            wrapper (the only honest runaway bound for code we don't own).
        store_llm_content: Include raw LLM/tool content in event payloads
            instead of hashes only.

    Returns:
        The agent's final answer.

    Raises:
        AgentContractError: If the agent returns a non-``str`` value.
    """
    session = CaptureSession(
        collector,
        max_llm_calls_per_run=max_llm_calls_per_run,
        store_llm_content=store_llm_content,
    )
    with bind_session(session):
        try:
            answer = runner(task)
        except Exception as exc:
            session.emit(
                "agent_error",
                {"exc_type": type(exc).__name__, "message": str(exc)},
                agent_name="harness",
            )
            raise
        if not isinstance(answer, str):
            raise AgentContractError(
                f"The agent must return the final answer as a str, got "
                f"{type(answer).__name__!r}. Convert structured output to a "
                f"string before returning."
            )
        payload: dict[str, Any] = {"answer_hash": hashlib.sha256(answer.encode()).hexdigest()[:16]}
        if store_llm_content:
            payload["answer"] = answer
        session.emit("final_answer", payload, agent_name="harness")
    return answer
