"""Category 2 tests: DataSource declaration and modality detection."""

from pathlib import Path

import pytest

from traxr.data import DataSource, ModalityType, detect_modality
from traxr.data.loader import LoadedArtifact
from traxr.errors import (
    InvalidArtifactError,
    ModalityMismatchError,
    UnsupportedModalityError,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sample.csv", ModalityType.TABULAR),
        ("sample.xlsx", ModalityType.TABULAR),
        ("sample.txt", ModalityType.DOCUMENT),
        ("sample.md", ModalityType.DOCUMENT),
        ("sample.pdf", ModalityType.DOCUMENT),
    ],
)
def test_detect_modality(name: str, expected: ModalityType) -> None:
    assert detect_modality(name) is expected


def test_detect_modality_unsupported_extension() -> None:
    with pytest.raises(UnsupportedModalityError):
        detect_modality("sample.png")


def test_from_path_detects_modality_and_file_type(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.csv")
    assert source.modality is ModalityType.TABULAR
    assert source.file_type == "csv"
    assert source.source_id == "sample.csv"


def test_from_path_accepts_matching_declared_modality(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.pdf", modality=ModalityType.DOCUMENT)
    assert source.modality is ModalityType.DOCUMENT


def test_from_path_declared_mismatch_raises(fixtures_dir: Path) -> None:
    with pytest.raises(ModalityMismatchError, match="declared tabular"):
        DataSource.from_path(fixtures_dir / "sample.pdf", modality=ModalityType.TABULAR)


def test_from_path_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidArtifactError, match="not found"):
        DataSource.from_path(tmp_path / "nope.csv")


def test_from_path_unsupported_extension_raises(fixtures_dir: Path) -> None:
    with pytest.raises(UnsupportedModalityError):
        DataSource.from_path(fixtures_dir / "sample.docx")


def test_load_is_lazy_and_forwards_opts(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.txt", max_chars=10)
    artifact = source.load()
    assert isinstance(artifact, LoadedArtifact)
    assert len(artifact.content) == 10
    assert artifact.metadata["truncated"] is True
