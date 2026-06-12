"""Category 1 tests: document (TXT/MD) perturbation operators."""

import re
from pathlib import Path

import pytest

from traxr.perturb import PerturbationEngine, PerturbationType

DOCUMENT_OPERATORS = [
    PerturbationType.OCR_NOISE,
    PerturbationType.NUMBER_CORRUPTION,
    PerturbationType.TEXT_REDACTION,
    PerturbationType.PARAGRAPH_SHUFFLE,
    PerturbationType.ENCODING_ERROR,
    PerturbationType.SECTION_REMOVAL,
]


@pytest.fixture()
def txt_content(fixtures_dir: Path) -> str:
    return (fixtures_dir / "sample.txt").read_text()


@pytest.fixture()
def md_content(fixtures_dir: Path) -> str:
    return (fixtures_dir / "sample.md").read_text()


def paragraphs_of(content: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]


@pytest.mark.parametrize("op", DOCUMENT_OPERATORS)
@pytest.mark.parametrize("file_type", ["txt", "md"])
def test_determinism_same_seed_identical_hashes(
    op: PerturbationType, file_type: str, txt_content: str, md_content: str
) -> None:
    content = txt_content if file_type == "txt" else md_content
    results = [PerturbationEngine(seed=11).apply(content, file_type, op) for _ in range(2)]
    assert results[0].corrupted_hash == results[1].corrupted_hash
    assert results[0].changes == results[1].changes


@pytest.mark.parametrize("op", DOCUMENT_OPERATORS)
def test_seed_variation_changes_output(op: PerturbationType, txt_content: str) -> None:
    outputs = {
        PerturbationEngine(seed=s).apply(txt_content, "txt", op).corrupted_hash for s in range(8)
    }
    assert len(outputs) > 1, f"{op.value} produced identical output across 8 seeds"


@pytest.mark.parametrize("op", DOCUMENT_OPERATORS)
def test_applied_with_changes_recorded(op: PerturbationType, txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(txt_content, "txt", op)
    assert result.applied
    assert result.skip_reason is None
    assert result.changes, f"{op.value} recorded no changes"
    assert result.content_changed


@pytest.mark.parametrize("op", DOCUMENT_OPERATORS)
def test_skip_on_insufficient_content(op: PerturbationType) -> None:
    result = PerturbationEngine(seed=42).apply("tiny", "txt", op)
    assert not result.applied
    assert result.skip_reason == "Insufficient content (need at least 10 characters)"
    assert result.corrupted_content == "tiny"


@pytest.mark.parametrize("op", DOCUMENT_OPERATORS)
def test_round_trip_to_disk_is_faithful(
    op: PerturbationType, txt_content: str, tmp_path: Path
) -> None:
    """Text output written back as a text file reads back identically."""
    result = PerturbationEngine(seed=42).apply(txt_content, "txt", op)
    out = tmp_path / "sample.txt"
    out.write_text(result.corrupted_content, encoding="utf-8")
    assert out.read_text(encoding="utf-8") == result.corrupted_content


# ---------------------------------------------------------------------------
# Intensity / semantics sanity per operator
# ---------------------------------------------------------------------------


def test_ocr_noise_intensity(txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(txt_content, "txt", PerturbationType.OCR_NOISE)
    # ~8% error rate on applicable chars: length stays in the same ballpark.
    assert 0.8 * len(txt_content) <= len(result.corrupted_content) <= 1.3 * len(txt_content)
    match = re.search(r"Applied (\d+) OCR-style", result.description)
    assert match is not None
    assert int(match.group(1)) > 0


def test_number_corruption_changes_appear_in_output(txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(
        txt_content, "txt", PerturbationType.NUMBER_CORRUPTION
    )
    assert len(result.changes) <= 10  # recorded changes are capped
    for change in result.changes:
        assert change["corrupted"] in result.corrupted_content


def test_text_redaction_markers_in_output(txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(txt_content, "txt", PerturbationType.TEXT_REDACTION)
    markers = {
        "date_redaction": "[DATE]",
        "number_redaction": "[REDACTED]",
        "name_redaction": "[NAME]",
    }
    assert len(result.changes) <= 15
    for change in result.changes:
        assert markers[change["type"]] in result.corrupted_content


def test_paragraph_shuffle_preserves_first_and_multiset(txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(
        txt_content, "txt", PerturbationType.PARAGRAPH_SHUFFLE
    )
    original = paragraphs_of(txt_content)
    corrupted = paragraphs_of(result.corrupted_content)
    assert corrupted[0] == original[0]  # title kept in place
    assert sorted(corrupted) == sorted(original)  # nothing lost or invented
    assert corrupted != original


def test_encoding_error_intensity(txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(txt_content, "txt", PerturbationType.ENCODING_ERROR)
    match = re.search(r"Introduced (\d+) encoding errors", result.description)
    assert match is not None
    count = int(match.group(1))
    # ~4-12% of alpha chars; sanity-bound it well below "everything".
    assert 0 < count < len(txt_content) * 0.3


def test_section_removal_drops_middle_paragraphs(txt_content: str) -> None:
    result = PerturbationEngine(seed=42).apply(txt_content, "txt", PerturbationType.SECTION_REMOVAL)
    original = paragraphs_of(txt_content)
    corrupted = paragraphs_of(result.corrupted_content)
    removed = len(original) - len(corrupted)
    assert removed == len(result.changes)
    assert 1 <= removed <= 2
    assert corrupted[0] == original[0]  # first kept
    assert corrupted[-1] == original[-1]  # last kept
    assert all(p in original for p in corrupted)


def test_paragraph_ops_skip_on_too_few_paragraphs() -> None:
    content = "Only one paragraph here, but long enough to pass the length check."
    for op in (PerturbationType.PARAGRAPH_SHUFFLE, PerturbationType.SECTION_REMOVAL):
        result = PerturbationEngine(seed=42).apply(content, "txt", op)
        assert result.applied  # parseable content, but nothing to do
        assert not result.content_changed
        assert result.changes == []
