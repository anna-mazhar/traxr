"""Category 1 tests: PerturbationEngine dispatch, NULL, history, helpers."""

from pathlib import Path

import pytest

from traxr.perturb import (
    PerturbationEngine,
    PerturbationType,
    get_all_perturbation_types,
    get_pdf_perturbation_types,
    get_tabular_perturbation_types,
)


@pytest.mark.parametrize("file_type", ["csv", "pdf", "txt", "zip", "xyz"])
def test_null_content_is_universal(file_type: str) -> None:
    result = PerturbationEngine(seed=42).apply(
        "some content here", file_type, PerturbationType.NULL_CONTENT
    )
    assert result.applied
    assert result.corrupted_content == ""
    assert result.content_changed


def test_unknown_file_type_is_recorded_skip() -> None:
    result = PerturbationEngine(seed=42).apply("a,b\n1,2", "xyz", PerturbationType.COLUMN_SWAP)
    assert not result.applied
    assert result.skip_reason == "No perturbator for file type: xyz"
    assert result.corrupted_content == result.original_content


def test_history_tracks_all_applications() -> None:
    engine = PerturbationEngine(seed=42)
    engine.apply("a,b\n1,2", "csv", PerturbationType.COLUMN_SWAP)
    engine.apply("text long enough here", "txt", PerturbationType.OCR_NOISE)
    history = engine.get_history()
    assert [r.perturbation_type for r in history] == [
        PerturbationType.COLUMN_SWAP,
        PerturbationType.OCR_NOISE,
    ]
    engine.clear_history()
    assert engine.get_history() == []


def test_apply_from_file_csv(fixtures_dir: Path) -> None:
    result = PerturbationEngine(seed=42).apply_from_file(
        str(fixtures_dir / "sample.csv"), PerturbationType.COLUMN_SWAP
    )
    assert result.applied
    assert result.file_type == "csv"
    assert result.file_name == "sample.csv"


def test_apply_from_file_xlsx_reads_as_tsv(fixtures_dir: Path) -> None:
    result = PerturbationEngine(seed=42).apply_from_file(
        str(fixtures_dir / "sample.xlsx"), PerturbationType.DATA_TYPE_CORRUPT
    )
    assert result.applied
    assert "\t" in result.corrupted_content  # TSV re-serialization
    assert "=== Sheet:" in result.original_content


def test_get_supported_perturbations_by_type() -> None:
    engine = PerturbationEngine(seed=42)
    csv_ops = engine.get_supported_perturbations("csv")
    assert set(get_tabular_perturbation_types()) == set(csv_ops)
    pdf_ops = engine.get_supported_perturbations(".pdf")  # dot is stripped
    assert set(get_pdf_perturbation_types()) == set(pdf_ops)
    assert engine.get_supported_perturbations("zip") == [PerturbationType.NULL_CONTENT]
    assert engine.get_supported_perturbations("xyz") == [PerturbationType.NULL_CONTENT]


def test_get_all_perturbation_types_is_complete() -> None:
    assert set(get_all_perturbation_types()) == set(PerturbationType)


def test_seed_is_propagated_to_handlers() -> None:
    """Engine seed determines handler behavior (same seed -> same output)."""
    a = PerturbationEngine(seed=5).apply("a,b,c\n1,2,3", "csv", PerturbationType.COLUMN_SWAP)
    b = PerturbationEngine(seed=5).apply("a,b,c\n1,2,3", "csv", PerturbationType.COLUMN_SWAP)
    assert a.corrupted_content == b.corrupted_content


def test_write_excel_without_openpyxl_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from traxr.errors import OptionalDependencyError

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openpyxl":
            raise ImportError("No module named 'openpyxl'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = PerturbationEngine(seed=42)
    with pytest.raises(OptionalDependencyError, match=r"traxr\[document\]"):
        engine.write_excel("a,b\n1,2", "/tmp/never-written.xlsx")
