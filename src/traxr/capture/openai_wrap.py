"""``instrument()`` — Tier 0 capture at the OpenAI-client boundary.

Wraps ``chat.completions.create`` on the client *instance* the user passes
in (sync or async) and returns the same client. The wrapper operates on
parsed SDK objects — ``tool_calls``, ``finish_reason``, ``usage`` come for
free — touches no HTTP, and depends only on the ``create`` call surface, so
it works with any OpenAI-compatible SDK client (no ``openai`` import needed
here; :func:`~traxr.capture.patch.patch_openai` is the class-level fallback).

Per completed ``create`` call the wrapper emits to the active
:class:`~traxr.capture.context.CaptureSession`:

* best-effort ``tool_result`` events for unseen ``role=="tool"`` entries in
  the *request* messages (at the step of the ``llm_call`` that requested
  them — Tier 0 never knows tool success/failure, so the payload omits it),
* one ``llm_call`` event on response completion (for streams: when the
  stream is exhausted or closed; ``finish_reason="abandoned"`` if dropped),
* one ``tool_request`` event per ``message.tool_calls`` entry.

Payloads are hash-only by default; raw content/arguments are included only
when the session was created with ``store_llm_content=True``.
"""

import hashlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any

from traxr.capture.context import CaptureSession, current_session, tier0_suppressed
from traxr.errors import TokenUnavailableWarning

__all__ = ["instrument"]

_MARKER = "_traxr_instrumented"


def instrument(client: Any) -> Any:
    """Capture LLM calls made through ``client`` during Traxr runs.

    Wraps ``client.chat.completions.create`` in place (sync ``OpenAI`` or
    ``AsyncOpenAI``, including streaming) and returns the same client.
    Construct your agent with the instrumented client; outside a Traxr run
    the wrapper is a pure passthrough, so the agent keeps working standalone.

    Idempotent: instrumenting an already-instrumented client is a no-op.

    Raises:
        TypeError: If ``client`` has no ``chat.completions.create``.
    """
    try:
        completions = client.chat.completions
        original = completions.create
    except AttributeError as exc:
        raise TypeError(
            "instrument() expects an OpenAI-compatible client exposing "
            f"chat.completions.create, got {type(client).__name__!r}"
        ) from exc
    if getattr(original, _MARKER, False):
        return client
    completions.create = make_create_wrapper(original)
    return client


# ---------------------------------------------------------------------------
# Wrapper factory — shared with patch_openai(), which wraps the *unbound*
# class methods (``self`` then arrives as the first positional arg; the
# capture logic only ever inspects kwargs, which create() is keyword-only on).
# ---------------------------------------------------------------------------


def make_create_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a ``create`` callable (sync or async) with Tier 0 capture.

    Sync/async is decided per call by whether ``original`` returned an
    awaitable — the SDK's async ``create`` is a plain ``def`` returning a
    coroutine, so ``iscoroutinefunction`` cannot tell the variants apart.
    """

    def create(*args: Any, **kwargs: Any) -> Any:
        session = _active_session()
        if session is None:
            return original(*args, **kwargs)
        step = session.begin_llm_call()
        _emit_request_tool_results(session, kwargs)
        streaming = bool(kwargs.get("stream"))
        if streaming:
            _inject_usage_option(kwargs)
        try:
            result = original(*args, **kwargs)
        except BaseException:
            session.end_llm_call()
            raise
        if inspect.isawaitable(result):
            return _finish_async(result, session, step, kwargs, streaming)
        if streaming:
            # The create() call has returned; balance begin_llm_call now so a
            # held, unconsumed stream is not counted as an in-flight call and
            # does not trip a false ConcurrentTraceWarning (N-L3). The llm_call
            # event is still emitted when the stream is consumed/closed/GC'd.
            session.end_llm_call()
            return _SyncStreamCapture(result, session, step, kwargs)
        session.end_llm_call()
        _emit_completion(session, step, kwargs, result)
        return result

    setattr(create, _MARKER, True)
    return create


async def _finish_async(
    awaitable: Any,
    session: CaptureSession,
    step: int,
    kwargs: dict[str, Any],
    streaming: bool,
) -> Any:
    """Async continuation of the wrapper after ``original`` returned an awaitable."""
    try:
        response = await awaitable
    except BaseException:
        session.end_llm_call()
        raise
    if streaming:
        # Balance begin_llm_call at create-return time, not at stream finalize —
        # see the N-L3 note in create().
        session.end_llm_call()
        return _AsyncStreamCapture(response, session, step, kwargs)
    session.end_llm_call()
    _emit_completion(session, step, kwargs, response)
    return response


def _active_session() -> CaptureSession | None:
    """The session to emit to, or None for passthrough (no run / Tier 1 active)."""
    if tier0_suppressed():
        return None
    return current_session()


def _inject_usage_option(kwargs: dict[str, Any]) -> None:
    """Ask the provider to include usage in the final stream chunk when absent."""
    if not kwargs.get("stream_options"):
        kwargs["stream_options"] = {"include_usage": True}


# ---------------------------------------------------------------------------
# Event-builder core.
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _msg_get(message: Any, key: str) -> Any:
    """Read a field from a request message (plain dict or SDK model)."""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _emit_request_tool_results(session: CaptureSession, kwargs: dict[str, Any]) -> None:
    """Emit ``tool_result`` for unseen ``role=="tool"`` entries in the request.

    The tool name and step come from the ``tool_request`` that introduced the
    ``tool_call_id``; Tier 0 cannot observe success/failure, so the payload
    omits it.
    """
    for message in kwargs.get("messages") or []:
        if _msg_get(message, "role") != "tool":
            continue
        call_id = _msg_get(message, "tool_call_id")
        if not isinstance(call_id, str) or call_id in session.seen_tool_result_ids:
            continue
        session.seen_tool_result_ids.add(call_id)
        tool_name, request_step = session.tool_call_names.get(call_id, ("?", session.step_num))
        content = str(_msg_get(message, "content") or "")
        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "call_id": call_id,
            "content_hash": _hash_text(content),
        }
        if session.store_llm_content:
            payload["content"] = content
        session.emit("tool_result", payload, step_num=request_step)


def _emit_completion(
    session: CaptureSession,
    step: int,
    kwargs: dict[str, Any],
    response: Any,
) -> None:
    """Emit ``llm_call`` + per-tool-call ``tool_request`` from a parsed response."""
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    content = getattr(message, "content", None) or ""
    _emit_llm_call(
        session,
        step,
        model=getattr(response, "model", None) or kwargs.get("model") or "?",
        finish_reason=getattr(choice, "finish_reason", None) or "?",
        content=content,
        usage=_usage_dict(getattr(response, "usage", None)),
        tool_calls=[(tc.id, tc.function.name, tc.function.arguments or "") for tc in tool_calls],
    )


def _emit_llm_call(
    session: CaptureSession,
    step: int,
    *,
    model: str,
    finish_reason: str,
    content: str,
    usage: dict[str, int] | None,
    tool_calls: list[tuple[str, str, str]],
) -> None:
    """Shared terminal emission for non-streaming and reassembled streams."""
    if usage is None:
        warnings.warn(
            "No token usage was captured for an LLM call; token-overhead "
            "metrics will be incomplete for this run.",
            TokenUnavailableWarning,
            stacklevel=4,
        )
    payload: dict[str, Any] = {
        "model": model,
        "finish_reason": finish_reason,
        "tool_call_names": [name for _, name, _ in tool_calls],
        "usage": usage,
        "content_hash": _hash_text(content),
    }
    if session.store_llm_content:
        payload["content"] = content
    session.emit("llm_call", payload, step_num=step)
    for call_id, name, arguments in tool_calls:
        session.tool_call_names[call_id] = (name, step)
        request_payload: dict[str, Any] = {
            "tool_name": name,
            "call_id": call_id,
            "arguments_hash": _hash_text(arguments),
        }
        if session.store_llm_content:
            request_payload["arguments"] = arguments
        session.emit("tool_request", request_payload, step_num=step)


def _usage_dict(usage: Any) -> dict[str, int] | None:
    """Token counts from an SDK usage object, or None when unavailable."""
    if usage is None:
        return None
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Streaming delta reassembly.
# ---------------------------------------------------------------------------


class _StreamState:
    """Accumulates chunks into the data ``_emit_llm_call`` needs.

    Emits exactly once per stream — when exhausted, closed, or garbage
    collected — with ``finish_reason="abandoned"`` when the stream was
    dropped before its natural end.
    """

    def __init__(self, session: CaptureSession, step: int, kwargs: dict[str, Any]):
        self._session = session
        self._step = step
        self._model: str = kwargs.get("model") or "?"
        self._content_parts: list[str] = []
        #: tool_call index -> [call_id, name, list of argument fragments]
        self._tool_calls: dict[int, list[Any]] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, int] | None = None
        self._done = False

    def ingest(self, chunk: Any) -> None:
        usage = _usage_dict(getattr(chunk, "usage", None))
        if usage is not None:
            self._usage = usage
        model = getattr(chunk, "model", None)
        if model:
            self._model = model
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if getattr(delta, "content", None):
                    self._content_parts.append(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    entry = self._tool_calls.setdefault(tc.index, ["", "", []])
                    if getattr(tc, "id", None):
                        entry[0] = tc.id
                    function = getattr(tc, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            entry[1] += function.name
                        if getattr(function, "arguments", None):
                            entry[2].append(function.arguments)
            if getattr(choice, "finish_reason", None):
                self._finish_reason = choice.finish_reason

    def finalize(self, *, exhausted: bool) -> None:
        if self._done:
            return
        self._done = True
        finish_reason = self._finish_reason if exhausted else "abandoned"
        _emit_llm_call(
            self._session,
            self._step,
            model=self._model,
            finish_reason=finish_reason or "?",
            content="".join(self._content_parts),
            usage=self._usage,
            tool_calls=[
                (entry[0], entry[1], "".join(entry[2]))
                for _, entry in sorted(self._tool_calls.items())
            ],
        )
        # NB: end_llm_call() is balanced at create-return time (see the wrapper),
        # not here — finalize only emits the reassembled llm_call event.


class _SyncStreamCapture:
    """Iterator/context-manager shim around a sync SDK stream."""

    def __init__(self, stream: Any, session: CaptureSession, step: int, kwargs: dict[str, Any]):
        self._stream = stream
        self._state = _StreamState(session, step, kwargs)

    def __iter__(self) -> "_SyncStreamCapture":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._state.finalize(exhausted=True)
            raise
        self._state.ingest(chunk)
        return chunk

    def __enter__(self) -> "_SyncStreamCapture":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        self._state.finalize(exhausted=False)
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def __del__(self) -> None:
        # Best-effort: a dropped stream still records an abandoned llm_call.
        try:
            self._state.finalize(exhausted=False)
        except Exception:  # pragma: no cover - interpreter-shutdown GC
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamCapture:
    """Async-iterator/context-manager shim around an async SDK stream."""

    def __init__(self, stream: Any, session: CaptureSession, step: int, kwargs: dict[str, Any]):
        self._stream = stream
        self._state = _StreamState(session, step, kwargs)

    def __aiter__(self) -> "_AsyncStreamCapture":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._state.finalize(exhausted=True)
            raise
        self._state.ingest(chunk)
        return chunk

    async def __aenter__(self) -> "_AsyncStreamCapture":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._state.finalize(exhausted=False)
        # Prefer the async stream's aclose(); some SDK async streams expose only
        # aclose (no sync close), so reaching for "close" leaks the HTTP response.
        closer = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if closer is not None:
            result = closer()
            if inspect.isawaitable(result):
                await result

    def __del__(self) -> None:
        try:
            self._state.finalize(exhausted=False)
        except Exception:  # pragma: no cover - interpreter-shutdown GC
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
