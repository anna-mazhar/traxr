"""Tier 1 capture: the LangGraph adapter.

``from_langgraph(compiled_graph)`` wraps a compiled LangGraph as an
:data:`~traxr.agents.AgentRunner`. Capture rides a
``BaseCallbackHandler`` passed through ``config={"callbacks": [...]}`` —
callbacks work with sync ``invoke`` and live in ``langchain-core``, a
stabler surface than the streaming APIs.

The mapping deliberately reuses the existing event vocabulary so reroute
counts and routing-cycle metrics work on LangGraph traces with zero
analyzer changes:

* node entry (``on_chain_start`` with ``langgraph_node`` metadata) →
  ``routing_decision`` ``{chosen_agent: <node>, via: "langgraph_node"}``
* ``on_tool_end`` / ``on_tool_error`` → ``tool_invocation`` (full
  success/failure fidelity, unlike Tier 0)
* ``on_llm_end`` → ``llm_call`` (same payload schema as Tier 0)

Tier 0 emission is suppressed for the duration of each graph invocation, so
a graph whose model is also an ``instrument()``-wrapped client does not
double-count its LLM calls. Note: that also means ``max_llm_calls_per_run``
(a Tier 0 wrapper bound) is not enforced for LangGraph runs in v1.

The default ``input_builder``/``output_extractor`` only fit canonical
messages-state graphs — for anything else, those two hooks are the real
contract.
"""

import hashlib
import warnings
from collections.abc import Callable
from typing import Any

from traxr.agents.task import AgentRunner, Task
from traxr.capture.context import CaptureSession, current_session, suppress_tier0
from traxr.errors import AgentContractError, OptionalDependencyError, TokenUnavailableWarning

__all__ = ["from_langgraph"]

_handler_cls: type[Any] | None = None


def from_langgraph(
    compiled_graph: Any,
    input_builder: Callable[[Task], Any] | None = None,
    output_extractor: Callable[[Any], str] | None = None,
) -> AgentRunner:
    """Wrap a compiled LangGraph as an ``AgentRunner`` with Tier 1 capture.

    Args:
        compiled_graph: The compiled graph (anything with
            ``invoke(input, config=...)``).
        input_builder: ``(Task) -> graph input``. Default builds a
            messages-state input from the question + file-path listing.
        output_extractor: ``(graph output) -> str``. Default takes the last
            message's content from a messages-state result.

    Raises:
        OptionalDependencyError: If langchain-core is not installed.
    """
    handler_cls = _get_handler_cls()  # fail fast on the missing extra

    def runner(task: Task) -> str:
        session = current_session()
        callbacks: list[Any] = []
        if session is not None:
            callbacks.append(handler_cls(session))
        graph_input = input_builder(task) if input_builder is not None else _default_input(task)
        with suppress_tier0():
            result = compiled_graph.invoke(graph_input, config={"callbacks": callbacks})
        if output_extractor is not None:
            return output_extractor(result)
        return _default_output(result)

    return runner


def _default_input(task: Task) -> dict[str, Any]:
    files = ", ".join(str(p) for p in task.files)
    text = task.question + (f"\n\nInput files: {files}" if files else "")
    return {"messages": [("user", text)]}


def _default_output(result: Any) -> str:
    try:
        message = result["messages"][-1]
    except (KeyError, IndexError, TypeError):
        raise AgentContractError(
            "from_langgraph's default output_extractor expects a messages-state "
            "result with a non-empty 'messages' list. Pass output_extractor= "
            "to extract the final answer from your graph's output."
        ) from None
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _get_handler_cls() -> type[Any]:
    """Build (once) the handler class; lazy so the extra stays optional."""
    global _handler_cls
    if _handler_cls is not None:
        return _handler_cls
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as exc:
        raise OptionalDependencyError(
            "from_langgraph() needs langchain-core (and langgraph). Install "
            'them with: pip install "traxr[langgraph]"'
        ) from exc

    class TraxrLangGraphHandler(BaseCallbackHandler):
        """Maps LangGraph callbacks onto one run's capture session."""

        def __init__(self, session: CaptureSession):
            self._session = session
            #: (node, langgraph_step, checkpoint_ns) already announced —
            #: inner runnable chains inherit the node metadata and must not
            #: produce duplicate routing_decision events.
            self._seen_nodes: set[tuple[Any, ...]] = set()
            #: run_id -> (model, step, node) recorded at *_start time.
            self._llm_runs: dict[Any, tuple[str, int, str]] = {}
            #: run_id -> (tool_name, step, node).
            self._tool_runs: dict[Any, tuple[str, int, str]] = {}
            self._step = 0  # fallback when metadata lacks langgraph_step

        # -- helpers ---------------------------------------------------------

        def _step_from(self, metadata: Any) -> int:
            step = (metadata or {}).get("langgraph_step")
            if isinstance(step, int):
                self._step = max(self._step, step)
                return step
            return self._step

        def _node_from(self, metadata: Any) -> str:
            return str((metadata or {}).get("langgraph_node") or "langgraph")

        def _record_llm_start(self, run_id: Any, metadata: Any, kwargs: dict[str, Any]) -> None:
            params = kwargs.get("invocation_params") or {}
            model = params.get("model") or params.get("model_name") or params.get("_type") or "?"
            if self._llm_runs:
                # A second LLM call started before the first finished: real
                # parallelism (the session's thread heuristic is bypassed —
                # LangGraph runs callbacks on executor threads even when
                # execution is sequential).
                self._session.note_concurrency()
            self._llm_runs[run_id] = (
                str(model),
                self._step_from(metadata),
                self._node_from(metadata),
            )

        # -- node entry → routing_decision ------------------------------------

        def on_chain_start(
            self,
            serialized: Any,
            inputs: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            tags: Any = None,
            metadata: Any = None,
            **kwargs: Any,
        ) -> None:
            node = (metadata or {}).get("langgraph_node")
            if not node:
                return  # the graph's own root chain / non-node runnables
            key = (
                node,
                (metadata or {}).get("langgraph_step"),
                (metadata or {}).get("langgraph_checkpoint_ns"),
            )
            if key in self._seen_nodes:
                return
            self._seen_nodes.add(key)
            self._session.emit(
                "routing_decision",
                {"chosen_agent": str(node), "via": "langgraph_node"},
                agent_name="langgraph",
                step_num=self._step_from(metadata),
                count_thread=False,
            )

        # -- chat/LLM calls → llm_call ----------------------------------------

        def on_chat_model_start(
            self,
            serialized: Any,
            messages: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            tags: Any = None,
            metadata: Any = None,
            **kwargs: Any,
        ) -> None:
            self._record_llm_start(run_id, metadata, kwargs)

        def on_llm_start(
            self,
            serialized: Any,
            prompts: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            tags: Any = None,
            metadata: Any = None,
            **kwargs: Any,
        ) -> None:
            self._record_llm_start(run_id, metadata, kwargs)

        def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
            model, step, node = self._llm_runs.pop(run_id, ("?", self._step, "langgraph"))
            generations = getattr(response, "generations", None) or [[]]
            generation = generations[0][0] if generations[0] else None
            message = getattr(generation, "message", None)
            content = getattr(message, "content", None) or getattr(generation, "text", "") or ""
            tool_call_names = [
                str(tc.get("name", "?")) if isinstance(tc, dict) else str(getattr(tc, "name", "?"))
                for tc in (getattr(message, "tool_calls", None) or [])
            ]
            finish_reason = (getattr(generation, "generation_info", None) or {}).get(
                "finish_reason"
            ) or "stop"
            usage = _usage_from(message, response)
            if usage is None:
                warnings.warn(
                    "No token usage was captured for an LLM call; token-overhead "
                    "metrics will be incomplete for this run.",
                    TokenUnavailableWarning,
                    stacklevel=2,
                )
            payload: dict[str, Any] = {
                "model": model,
                "finish_reason": finish_reason,
                "tool_call_names": tool_call_names,
                "usage": usage,
                "content_hash": _hash_text(str(content)),
            }
            if self._session.store_llm_content:
                payload["content"] = str(content)
            self._session.emit(
                "llm_call", payload, agent_name=node, step_num=step, count_thread=False
            )

        # -- tools → tool_invocation (full success/failure fidelity) ----------

        def on_tool_start(
            self,
            serialized: Any,
            input_str: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            tags: Any = None,
            metadata: Any = None,
            **kwargs: Any,
        ) -> None:
            name = (serialized or {}).get("name") or "?"
            self._tool_runs[run_id] = (
                str(name),
                self._step_from(metadata),
                self._node_from(metadata),
            )

        def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
            name, step, node = self._tool_runs.pop(run_id, ("?", self._step, "langgraph"))
            self._session.emit(
                "tool_invocation",
                {
                    "tool_name": name,
                    "operation": "call",
                    "success": True,
                    "output_hash": _hash_text(str(output)),
                },
                agent_name=node,
                step_num=step,
                count_thread=False,
            )

        def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
            name, step, node = self._tool_runs.pop(run_id, ("?", self._step, "langgraph"))
            self._session.emit(
                "tool_invocation",
                {
                    "tool_name": name,
                    "operation": "call",
                    "success": False,
                    "output_hash": _hash_text(type(error).__name__),
                },
                agent_name=node,
                step_num=step,
                count_thread=False,
            )

    _handler_cls = TraxrLangGraphHandler
    return _handler_cls


def _usage_from(message: Any, response: Any) -> dict[str, int] | None:
    """Token counts from usage_metadata (preferred) or llm_output, else None."""
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        return {
            "prompt_tokens": int(usage_metadata.get("input_tokens", 0)),
            "completion_tokens": int(usage_metadata.get("output_tokens", 0)),
            "total_tokens": int(usage_metadata.get("total_tokens", 0)),
        }
    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage")
    if token_usage:
        return {
            "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(token_usage.get("completion_tokens", 0)),
            "total_tokens": int(token_usage.get("total_tokens", 0)),
        }
    return None
