"""Tier 0 capture: universal OpenAI-client interception.

``instrument()`` (instance-level, primary) and ``patch_openai()``
(class-level fallback) capture ``llm_call``/``tool_request``/``tool_result``
events at the SDK boundary; ``emit()`` is the manual escape hatch. Events
route to the run's collector via the session binding in
:mod:`traxr.capture.context`.
"""

from traxr.capture.context import (
    CaptureSession,
    bind_session,
    current_session,
    emit,
    suppress_tier0,
)
from traxr.capture.openai_wrap import instrument
from traxr.capture.patch import patch_openai

__all__ = [
    "CaptureSession",
    "bind_session",
    "current_session",
    "emit",
    "instrument",
    "patch_openai",
    "suppress_tier0",
]
