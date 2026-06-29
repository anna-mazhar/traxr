"""Default scorer: normalized answer matching."""

from dataclasses import dataclass

import pytest

from traxr.scoring import check_answer_match, llm_judge_match, normalize_answer


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("  The Answer  ", "the answer"),
        ("42.", "42"),
        ('"quoted"', "quoted"),
        ("multi   space\ttabs", "multi space tabs"),
        (None, ""),
    ],
)
def test_normalize_answer(raw, normalized):
    assert normalize_answer(raw) == normalized


@pytest.mark.parametrize(
    ("expected", "actual", "match"),
    [
        ("Paris", "paris", True),
        ("Paris", " PARIS. ", True),
        ("42", "42.0", True),
        ("1,000", "1000", True),
        ("$5", "5", True),
        ("12%", "12", True),
        ("42", "43", False),
        ("Paris", "London", False),
        (None, None, True),  # both normalize to ""
        ("Paris", None, False),
    ],
)
def test_check_answer_match(expected, actual, match):
    assert check_answer_match(expected, actual) is match


@dataclass
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def generate(self, prompt, response_type="default", context=None):
        return _FakeLLMResponse(content=self._content)

    def generate_with_tools(self, prompt, tools, response_type="default", context=None, system_prompt_override=None):
        raise NotImplementedError


@pytest.mark.parametrize(
    ("judge_reply", "match"),
    [
        ("yes", True),
        ("Yes.", True),
        ("no", False),
        ("no, the candidate names a different region", False),
    ],
)
def test_llm_judge_match(judge_reply, match):
    assert llm_judge_match("EMEA", "some verbose answer", _FakeLLM(judge_reply)) is match


def test_llm_judge_match_none_short_circuits():
    llm = _FakeLLM("yes")
    assert llm_judge_match(None, None, llm) is True
    assert llm_judge_match("EMEA", None, llm) is False
    assert llm_judge_match(None, "EMEA", llm) is False


def test_llm_judge_match_uses_default_client_when_omitted(monkeypatch):
    import traxr.scoring as scoring_module

    monkeypatch.setattr(
        scoring_module, "_default_judge_client", lambda: _FakeLLM("yes")
    )
    assert llm_judge_match("EMEA", "some verbose answer") is True
