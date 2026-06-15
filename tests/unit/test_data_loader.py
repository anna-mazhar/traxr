"""Category 2 tests: data loader — happy paths and typed error paths.

Valid CSV/XLSX/TXT/MD/PDF load; corrupt/empty/truncated -> InvalidArtifactError;
wrong extension -> ModalityMismatchError; docx/png -> UnsupportedModalityError;
missing optional dependency -> OptionalDependencyError.
"""

from pathlib import Path

import pytest

from traxr.data import loader
from traxr.data.loader import FileInspection, inspect_file, read_file
from traxr.errors import (
    InvalidArtifactError,
    ModalityMismatchError,
    OptionalDependencyError,
    UnsupportedModalityError,
)

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sample.csv", "sample.txt", "sample.md"])
def test_read_text_based_fixture(name: str, fixtures_dir: Path) -> None:
    artifact = read_file(fixtures_dir / name)
    assert artifact.file_type == Path(name).suffix.lstrip(".")
    assert artifact.content.strip()
    assert artifact.metadata["file_name"] == name
    assert artifact.metadata["truncated"] is False


def test_read_xlsx_fixture(fixtures_dir: Path) -> None:
    artifact = read_file(fixtures_dir / "sample.xlsx")
    assert artifact.file_type == "xlsx"
    assert "=== Sheet:" in artifact.content
    assert "\t" in artifact.content
    # N-L4: sheet_names is captured before the workbook is closed, so it stays in
    # step with the per-sheet sections written into the content.
    assert artifact.metadata["sheet_names"]
    for name in artifact.metadata["sheet_names"]:
        assert f"=== Sheet: {name} ===" in artifact.content


def test_read_pdf_fixture(fixtures_dir: Path) -> None:
    artifact = read_file(fixtures_dir / "sample.pdf")
    assert artifact.file_type == "pdf"
    assert "--- Page 1 ---" in artifact.content
    assert artifact.metadata["num_pages"] >= 1


def test_read_pdf_pdfplumber_fallback(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pdfplumber")
    monkeypatch.setattr(loader, "HAS_PYMUPDF", False)
    artifact = read_file(fixtures_dir / "sample.pdf")
    assert artifact.file_type == "pdf"
    assert artifact.metadata["num_pages"] >= 1


def test_max_chars_truncates_text(fixtures_dir: Path) -> None:
    artifact = read_file(fixtures_dir / "sample.txt", max_chars=10)
    assert len(artifact.content) == 10
    assert artifact.metadata["truncated"] is True


# ---------------------------------------------------------------------------
# InvalidArtifactError: missing / empty / corrupt / truncated
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidArtifactError, match="not found"):
        read_file(tmp_path / "nope.csv")


@pytest.mark.parametrize("name", ["empty.csv", "empty.txt"])
def test_empty_file_raises(name: str, fixtures_dir: Path) -> None:
    with pytest.raises(InvalidArtifactError, match="empty"):
        read_file(fixtures_dir / name)


@pytest.mark.parametrize("name", ["corrupt.pdf", "truncated.pdf"])
def test_unparseable_pdf_raises(name: str, fixtures_dir: Path) -> None:
    with pytest.raises(InvalidArtifactError):
        read_file(fixtures_dir / name)


def test_truncated_xlsx_raises(fixtures_dir: Path) -> None:
    with pytest.raises(InvalidArtifactError):
        read_file(fixtures_dir / "truncated.xlsx")


# ---------------------------------------------------------------------------
# ModalityMismatchError: wrong extension caught by content signature
# ---------------------------------------------------------------------------


def test_pdf_bytes_with_csv_extension_raises(fixtures_dir: Path) -> None:
    with pytest.raises(ModalityMismatchError, match="looks like"):
        read_file(fixtures_dir / "wrong_extension.csv")


def test_zip_bytes_with_pdf_extension_raises(fixtures_dir: Path) -> None:
    with pytest.raises(ModalityMismatchError, match="Office/zip"):
        read_file(fixtures_dir / "wrong_extension.pdf")


def test_pdf_bytes_with_xlsx_extension_raises(fixtures_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "fake.xlsx"
    bad.write_bytes((fixtures_dir / "sample.pdf").read_bytes())
    with pytest.raises(ModalityMismatchError, match="looks like a PDF"):
        read_file(bad)


# ---------------------------------------------------------------------------
# UnsupportedModalityError: outside the v1 set
# ---------------------------------------------------------------------------


def test_docx_raises_future_release_message(fixtures_dir: Path) -> None:
    with pytest.raises(UnsupportedModalityError, match="future release"):
        read_file(fixtures_dir / "sample.docx")


def test_png_raises_image_message(fixtures_dir: Path) -> None:
    with pytest.raises(UnsupportedModalityError, match="image/audio"):
        read_file(fixtures_dir / "sample.png")


@pytest.mark.parametrize(
    ("name", "match"),
    [("a.zip", "archives"), ("a.xyz", "Unknown file extension")],
)
def test_other_unsupported_extensions(name: str, match: str, tmp_path: Path) -> None:
    p = tmp_path / name
    p.write_bytes(b"data")
    with pytest.raises(UnsupportedModalityError, match=match):
        read_file(p)


# ---------------------------------------------------------------------------
# OptionalDependencyError: missing optional readers
# ---------------------------------------------------------------------------


def test_pdf_without_any_reader_raises(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "HAS_PYMUPDF", False)
    monkeypatch.setattr(loader, "HAS_PDFPLUMBER", False)
    with pytest.raises(OptionalDependencyError, match=r"traxr\[document\]"):
        read_file(fixtures_dir / "sample.pdf")


def test_xlsx_without_openpyxl_raises(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "HAS_OPENPYXL", False)
    with pytest.raises(OptionalDependencyError, match="openpyxl"):
        read_file(fixtures_dir / "sample.xlsx")


# ---------------------------------------------------------------------------
# inspect_file
# ---------------------------------------------------------------------------


def test_inspect_csv(fixtures_dir: Path) -> None:
    inspection = inspect_file(fixtures_dir / "sample.csv")
    assert inspection.file_type == "csv"
    assert inspection.columns
    assert inspection.row_count >= 1
    assert inspection.sample_rows
    assert not inspection.errors


def test_inspect_xlsx(fixtures_dir: Path) -> None:
    inspection = inspect_file(fixtures_dir / "sample.xlsx")
    assert inspection.file_type == "xlsx"
    assert inspection.sheet_names
    assert inspection.columns
    assert not inspection.errors


def test_inspect_txt_has_no_tabular_structure(fixtures_dir: Path) -> None:
    inspection = inspect_file(fixtures_dir / "sample.txt")
    assert inspection.file_type == "txt"
    assert inspection.columns == []
    assert inspection.file_size > 0


def test_inspect_is_best_effort_on_corrupt_structure(fixtures_dir: Path) -> None:
    inspection = inspect_file(fixtures_dir / "truncated.xlsx")
    assert inspection.errors  # recorded, not raised


def test_inspect_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidArtifactError, match="not found"):
        inspect_file(tmp_path / "nope.csv")


def test_inspect_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    p.write_bytes(b"data")
    with pytest.raises(UnsupportedModalityError):
        inspect_file(p)


def test_inspection_to_dict_round_trip(fixtures_dir: Path) -> None:
    d = inspect_file(fixtures_dir / "sample.csv").to_dict()
    assert d["file_type"] == "csv"
    assert "columns" in d and "sample_rows" in d
    assert "errors" not in d


def test_inspection_to_dict_omits_empty_sections() -> None:
    d = FileInspection(file_name="a.txt", file_type="txt", file_size=1).to_dict()
    assert "columns" not in d
    assert "sheet_names" not in d
