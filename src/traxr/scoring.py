"""Answer scoring: normalized matching between answers.

The default scorer for ``task_success`` and the normalizer behind
``answer_changed``. Deliberately simple for v1: normalized string equality
with numeric tolerance — bring your own scorer via
``ExperimentConfig(scorer=...)`` for anything fuzzier.
"""

import re

__all__ = ["check_answer_match", "normalize_answer"]

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
