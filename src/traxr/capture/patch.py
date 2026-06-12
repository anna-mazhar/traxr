"""``patch_openai()`` — class-level Tier 0 capture fallback.

For agents that construct OpenAI clients internally, where the user cannot
thread an :func:`~traxr.capture.openai_wrap.instrument`-wrapped instance.
Patches ``Completions.create`` / ``AsyncCompletions.create`` at class level
for the duration of the context, sharing the instance wrapper's event-builder
core. Not composable with nested experiments — prefer ``instrument()``.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from traxr.capture.openai_wrap import make_create_wrapper
from traxr.errors import OptionalDependencyError

__all__ = ["patch_openai"]


@contextmanager
def patch_openai() -> Iterator[None]:
    """Capture LLM calls from *every* OpenAI client created inside the context.

    Raises:
        OptionalDependencyError: If the ``openai`` package is not installed.
    """
    try:
        from openai.resources.chat import completions as _completions
    except ImportError as exc:  # pragma: no cover - exercised without openai
        raise OptionalDependencyError(
            'patch_openai() needs the openai package. Install it with: pip install "traxr[openai]"'
        ) from exc

    sync_original = _completions.Completions.create
    async_original = _completions.AsyncCompletions.create
    _completions.Completions.create = make_create_wrapper(  # type: ignore[method-assign]
        sync_original
    )
    _completions.AsyncCompletions.create = make_create_wrapper(  # type: ignore[method-assign]
        async_original
    )
    try:
        yield
    finally:
        _completions.Completions.create = sync_original  # type: ignore[method-assign]
        _completions.AsyncCompletions.create = async_original  # type: ignore[method-assign]
