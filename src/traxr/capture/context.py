"""Run binding for the Tier 0 capture layer.

The experiment harness binds one :class:`CaptureSession` around each agent
invocation; the ``instrument()``/``patch_openai()`` wrappers and the
``traxr.emit()`` escape hatch route events to whatever session is currently
bound. Resolution order:

1. The contextvar set by :func:`bind_session` (propagates into asyncio tasks
   natively).
2. The process-global fallback (contextvars do not auto-propagate into
   user-spawned threads; the global is correct because runs are sequential).
3. ``None`` — capture is a no-op, so instrumented agents keep working when
   run outside a Traxr experiment.
"""

import contextvars
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from traxr.errors import ConcurrentTraceWarning, RunBudgetExceeded
from traxr.trace.collector import TraceCollector
from traxr.trace.events import TraceEvent

__all__ = [
    "CaptureSession",
    "bind_session",
    "current_session",
    "emit",
    "suppress_tier0",
    "tier0_suppressed",
]


class CaptureSession:
    """Per-run capture state shared by the Tier 0 wrapper and the harness.

    Tracks the step counter (one step per ``llm_call``; its ``tool_request``s
    and subsequent ``tool_result``s share that step), the
    ``tool_call_id -> tool_name`` map, the LLM-call budget, and concurrency
    detection (emission from more than one thread, or overlapping in-flight
    ``create`` calls, sets ``concurrent_detected`` and warns
    :class:`~traxr.errors.ConcurrentTraceWarning` once per run).
    """

    def __init__(
        self,
        collector: TraceCollector,
        *,
        max_llm_calls_per_run: int | None = None,
        store_llm_content: bool = False,
    ):
        self.collector = collector
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.store_llm_content = store_llm_content
        self.llm_call_count = 0
        self.concurrent_detected = False
        #: tool_call_id -> (tool_name, step_num of the requesting llm_call),
        #: for joining tool_result events to names and steps.
        self.tool_call_names: dict[str, tuple[str, int]] = {}
        #: tool_call_ids already reported as tool_result this run.
        self.seen_tool_result_ids: set[str] = set()
        self._step = 0
        self._lock = threading.Lock()
        self._emit_threads: set[int] = set()
        self._in_flight = 0

    @property
    def step_num(self) -> int:
        """The current step (the step of the most recent ``llm_call``)."""
        return self._step

    def begin_llm_call(self) -> int:
        """Enforce the budget and allocate the step for one ``create`` call.

        Called by the wrapper before dispatching to the provider, so a
        budget-exceeding call is never made.

        Returns:
            The step number this call's events share.

        Raises:
            RunBudgetExceeded: When the call would exceed
                ``max_llm_calls_per_run``.
        """
        with self._lock:
            if (
                self.max_llm_calls_per_run is not None
                and self.llm_call_count >= self.max_llm_calls_per_run
            ):
                raise RunBudgetExceeded(
                    f"This run already made {self.llm_call_count} LLM calls, the "
                    f"configured max_llm_calls_per_run. The partial trace up to "
                    f"this point is preserved."
                )
            self.llm_call_count += 1
            self._in_flight += 1
            if self._in_flight > 1:
                self._note_concurrency_locked()
            self._step += 1
            return self._step

    def end_llm_call(self) -> None:
        """Mark one in-flight ``create`` call as finished."""
        with self._lock:
            self._in_flight -= 1

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        agent_name: str = "external",
        step_num: int | None = None,
    ) -> TraceEvent:
        """Emit an event to this run's collector at the given (or current) step."""
        with self._lock:
            self._emit_threads.add(threading.get_ident())
            if len(self._emit_threads) > 1:
                self._note_concurrency_locked()
            step = self._step if step_num is None else step_num
        return self.collector.emit(
            event_type=event_type,
            step_num=step,
            agent_name=agent_name,
            payload=payload,
        )

    def _note_concurrency_locked(self) -> None:
        """Record concurrent tracing (caller holds the lock); warn once per run.

        The once-per-run dedup is the lock-guarded ``concurrent_detected`` flag,
        not the (non-thread-safe) ``warnings`` registry. ``stacklevel`` is
        best-effort: this warns about a run-level condition, not a fixable line
        of user code, and the call depth varies (sync ``emit`` vs. the Tier 1
        ``note_concurrency`` wrapper fired from a callback dispatcher), so no
        fixed deep level is correct. ``stacklevel=2`` points just past this
        private helper at the detecting call site.
        """
        if self.concurrent_detected:
            return
        self.concurrent_detected = True
        warnings.warn(
            "Concurrent LLM calls detected while tracing a single run. Event "
            "order is scheduling-dependent, which inflates divergence metrics; "
            "the pair will be flagged order_nondeterministic. Use the noise "
            "floor to calibrate, or set require_sequential=True to fail fast.",
            ConcurrentTraceWarning,
            stacklevel=2,
        )


_session_var: contextvars.ContextVar[CaptureSession | None] = contextvars.ContextVar(
    "traxr_capture_session", default=None
)
_tier0_suppressed_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "traxr_tier0_suppressed", default=False
)

_global_lock = threading.Lock()
_global_session: CaptureSession | None = None


def current_session() -> CaptureSession | None:
    """The active capture session: contextvar, then global fallback, then None."""
    session = _session_var.get()
    if session is not None:
        return session
    return _global_session


@contextmanager
def bind_session(session: CaptureSession) -> Iterator[CaptureSession]:
    """Bind ``session`` as the active capture session for the enclosed run.

    Sets both the contextvar (asyncio-safe) and the process-global fallback
    (reached by user-spawned threads); both are restored on exit.
    """
    global _global_session
    token = _session_var.set(session)
    with _global_lock:
        previous = _global_session
        _global_session = session
    try:
        yield session
    finally:
        _session_var.reset(token)
        with _global_lock:
            _global_session = previous


@contextmanager
def suppress_tier0() -> Iterator[None]:
    """Suppress Tier 0 emission inside this context.

    Used by the Tier 1 LangGraph adapter so a graph whose model is also a
    Tier-0-instrumented client does not double-count ``llm_call`` events.
    """
    token = _tier0_suppressed_var.set(True)
    try:
        yield
    finally:
        _tier0_suppressed_var.reset(token)


def tier0_suppressed() -> bool:
    """Whether Tier 0 emission is suppressed in the current context."""
    return _tier0_suppressed_var.get()


def emit(
    event_type: str,
    payload: Mapping[str, Any] | None = None,
    *,
    agent_name: str = "user",
) -> None:
    """Manually emit a trace event from inside your agent (escape hatch).

    The event lands at the current step of the active run. Unregistered event
    types fall back to the ``unknown:{event_type}`` signature (with a one-time
    :class:`~traxr.errors.UnknownEventTypeWarning`); upgrade them via
    :func:`traxr.register_signature`.

    Outside a Traxr run this is a no-op — same passthrough principle as
    ``instrument()`` — so agents that call ``traxr.emit()`` keep working
    standalone.
    """
    session = current_session()
    if session is None:
        return
    session.emit(event_type, dict(payload or {}), agent_name=agent_name)
