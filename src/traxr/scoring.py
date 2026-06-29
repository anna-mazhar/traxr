"""Answer scoring: normalized matching between answers.

``check_answer_match`` is the default scorer for ``task_success`` and the
normalizer behind ``answer_changed``. Deliberately simple for v1: normalized
string equality with numeric tolerance — bring your own scorer via
``ExperimentConfig(scorer=...)`` for anything fuzzier, e.g. a custom
function for verbose/free-text answers, or the opt-in ``llm_judge_match``
for semantic matching (non-deterministic, costs an extra LLM call).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traxr.llm.protocol import LLMClient

__all__ = ["check_answer_match", "llm_judge_match", "normalize_answer"]

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def normalize_answer(answer: str | None) -> str:
    """Normalize an answer for comparison.

    Lowercases, collapses internal whitespace, strips surrounding whitespace
    and quotes, and drops a trailing period. ``None`` normalizes to ``""``.

    >>> normalize_answer("  The  Answer. ")
    'the answer'
    >>> normalize_answer(None)
    ''
    """
    if answer is None:
        return ""
    text = " ".join(answer.split()).strip().strip("\"'").casefold()
    return text.removesuffix(".")


def check_answer_match(expected: str | None, actual: str | None) -> bool:
    """Whether ``actual`` matches ``expected`` after normalization.

    Numeric answers compare as floats (so ``"42"`` matches ``"42.0"``);
    everything else is normalized string equality.

    >>> check_answer_match("Paris", " PARIS. ")
    True
    >>> check_answer_match("1,000", "$1000")
    True
    >>> check_answer_match("42", "43")
    False
    """
    norm_expected = normalize_answer(expected)
    norm_actual = normalize_answer(actual)
    if norm_expected == norm_actual:
        return True
    cleaned_expected = norm_expected.replace(",", "").removeprefix("$").rstrip("%")
    cleaned_actual = norm_actual.replace(",", "").removeprefix("$").rstrip("%")
    if _NUMBER_RE.match(cleaned_expected) and _NUMBER_RE.match(cleaned_actual):
        return abs(float(cleaned_expected) - float(cleaned_actual)) < 1e-9
    return False


@lru_cache(maxsize=1)
def _default_judge_client() -> LLMClient:
    """Lazily build the default judge client (OpenAI, ``OPENAI_API_KEY``).

    Cached so ``llm_judge_match`` doesn't reconnect on every scored answer.
    Only constructed if a caller omits ``llm`` — pass your own client to
    use a different provider or avoid this default entirely.
    """
    from traxr.llm.openai_compat import OpenAICompatibleClient

    return OpenAICompatibleClient(model="gpt-4o-mini")


def llm_judge_match(
    expected: str | None, actual: str | None, llm: LLMClient | None = None
) -> bool:
    """Semantic match via LLM judge. Opt-in only, usable directly as
    ``ExperimentConfig(scorer=llm_judge_match)``.

    Unlike :func:`check_answer_match`, this is not deterministic: results
    can vary between runs and providers. Use it when literal/numeric
    matching is too strict for your question's expected phrasing.

    If ``llm`` is omitted, lazily constructs and caches a default
    ``OpenAICompatibleClient``; this reads ``OPENAI_API_KEY`` from the
    environment and makes a live network call per scored answer. Pass your
    own ``llm`` (e.g. via ``functools.partial``) to use another provider or
    avoid the implicit network/API-key dependency.
    """
    if expected is None or actual is None:
        return expected == actual
    if llm is None:
        llm = _default_judge_client()
    prompt = (
        f"Expected answer: {expected!r}\n"
        f"Candidate answer: {actual!r}\n\n"
        "Do these two answers reach the same core conclusion/bottom-line? "
        "Ignore differences in verbosity, formatting, or extra supporting "
        "detail present in only one of the two — focus only on whether the "
        "main conclusion matches. Reply with exactly one word: yes or no."
    )
    response = llm.generate(prompt, response_type="default")
    return response.content.strip().lower().startswith("yes")
