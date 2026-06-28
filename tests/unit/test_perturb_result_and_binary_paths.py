"""Category 1 supplements: result serialization, engine binary routing, JSON tabular.

Image/audio perturbators themselves are copied-not-exported (backlog); only the
engine's routing/NULL/skip paths around them are gated here.
"""

import pytest

from traxr.perturb import PerturbationEngine, PerturbationResult, PerturbationType
from traxr.perturb.tabular import TabularPerturbator

# ---------------------------------------------------------------------------
# PerturbationResult helpers
# ---------------------------------------------------------------------------


def _result(**kwargs: object) -> PerturbationResult:
    defaults: dict = {
        "original_content": "a,b\n1,2",
        "corrupted_content": "b,a\n2,1",
        "perturbation_type": PerturbationType.COLUMN_SWAP,
        "description": "Swapped columns",
    }
    defaults.update(kwargs)
    return PerturbationResult(**defaults)


def test_diff_summary_applied_change() -> None:
    assert _result().diff_summary == "column_swap: Swapped columns"


def test_diff_summary_skip() -> None:
    r = _result(applied=False, skip_reason="too small")
    assert r.diff_summary == "No change: too small"


def test_diff_summary_no_observable_change() -> None:
    r = _result(corrupted_content="a,b\n1,2")
    assert not r.content_changed
    assert r.diff_summary == "No observable change"


def test_to_dict_excludes_content_and_tracks_lengths() -> None:
    d = _result(file_type="csv", file_name="sample.csv").to_dict()
    assert d["perturbation_type"] == "column_swap"
    assert d["content_changed"] is True
    assert d["original_length"] == len("a,b\n1,2")
    assert "original_content" not in d
    assert "corrupted_bytes_length" not in d


def test_to_dict_reports_binary_length() -> None:
    d = _result(corrupted_bytes=b"abc").to_dict()
    assert d["corrupted_bytes_length"] == 3


# ---------------------------------------------------------------------------
# Engine binary routing (image/audio): NULL + no-handler skip paths
# ---------------------------------------------------------------------------


def test_apply_image_null_content() -> None:
    result = PerturbationEngine(seed=1).apply_image(
        b"\x89PNG", "png", PerturbationType.NULL_CONTENT
    )
    assert result.applied
    assert result.corrupted_bytes == b""


def test_apply_image_unknown_type_is_recorded_skip() -> None:
    result = PerturbationEngine(seed=1).apply_image(b"data", "csv", PerturbationType.BLUR)
    assert not result.applied
    assert "No image perturbator" in (result.skip_reason or "")


def test_apply_audio_null_content() -> None:
    result = PerturbationEngine(seed=1).apply_audio(b"RIFF", "wav", PerturbationType.NULL_CONTENT)
    assert result.applied
    assert result.corrupted_bytes == b""


def test_apply_audio_unknown_type_is_recorded_skip() -> None:
    result = PerturbationEngine(seed=1).apply_audio(
        b"data", "txt", PerturbationType.BACKGROUND_NOISE
    )
    assert not result.applied
    assert "No audio perturbator" in (result.skip_reason or "")


def test_type_predicates() -> None:
    engine = PerturbationEngine()
    assert engine.is_image_type(".PNG")
    assert not engine.is_image_type("csv")
    assert engine.is_audio_type("wav")
    assert not engine.is_audio_type("png")
    assert engine.is_zip_type("zip")
    assert not engine.is_zip_type("csv")


# ---------------------------------------------------------------------------
# Tabular JSON path
# ---------------------------------------------------------------------------

JSON_CONTENT = (
    '[{"name": "a", "amount": "1"}, {"name": "b", "amount": "2"}, {"name": "c", "amount": "3"}]'
)


def test_json_round_trip_column_swap() -> None:
    result = TabularPerturbator(seed=3).apply(
        JSON_CONTENT, PerturbationType.COLUMN_SWAP, file_type="json"
    )
    assert result.applied
    import json

    rows = json.loads(result.corrupted_content)
    assert isinstance(rows, list) and len(rows) == 3
    assert {"name", "amount"} <= set(rows[0])  # keys survive the swap


def test_json_determinism() -> None:
    a = TabularPerturbator(seed=5).apply(
        JSON_CONTENT, PerturbationType.ROW_DUPLICATE, file_type="json"
    )
    b = TabularPerturbator(seed=5).apply(
        JSON_CONTENT, PerturbationType.ROW_DUPLICATE, file_type="json"
    )
    assert a.corrupted_hash == b.corrupted_hash


@pytest.mark.parametrize("content", ["not json", "{}", '["a", "b"]', "[]"])
def test_json_non_tabular_is_recorded_skip(content: str) -> None:
    result = TabularPerturbator(seed=1).apply(
        content, PerturbationType.COLUMN_SWAP, file_type="json"
    )
    assert not result.applied
    assert result.skip_reason


def test_can_handle() -> None:
    perturbator = TabularPerturbator()
    assert perturbator.can_handle("CSV")
    assert not perturbator.can_handle("pdf")
