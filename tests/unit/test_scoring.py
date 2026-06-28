"""Default scorer: normalized answer matching."""

import pytest

from traxr.scoring import check_answer_match, normalize_answer


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
