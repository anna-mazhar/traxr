"""Category 1 tests: PDF in-place surgical editing — guard tests T1-T4 per operator.

T1 changed-span: each recorded replacement is present in the extracted text and
    the original is absent at that locus.
T2 untouched-region: per-page word lists, excluding words intersecting edited
    rects, are identical clean-vs-perturbed.
T3 non-noop: file hash != original AND extracted text != original.
T4 determinism: same seed -> identical extracted text twice.
"""

import hashlib
from pathlib import Path

import fitz
import pytest

from traxr.errors import OptionalDependencyError
from traxr.perturb import PerturbationType
from traxr.perturb.pdf_inplace import (
    PDF_INPLACE_OPERATORS,
    PDFInPlaceEditor,
    apply_pdf_inplace,
    extract_text,
)
from traxr.perturb.types import PerturbationResult

SPAN_OPERATORS = [
    PerturbationType.NUMBER_CORRUPTION,
    PerturbationType.TEXT_REDACTION,
]
ALL_OPERATORS = [
    PerturbationType.NUMBER_CORRUPTION,
    PerturbationType.TEXT_REDACTION,
    PerturbationType.SECTION_REMOVAL,
    PerturbationType.PAGE_REMOVAL,
    PerturbationType.PAGE_SHUFFLE,
]

Words = list[tuple[float, float, float, float, str]]


def words_per_page(path: Path) -> list[Words]:
    doc = fitz.open(path)
    pages = [[(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")] for page in doc]
    doc.close()
    return pages


def extracted(path: Path) -> str:
    doc = fitz.open(path)
    text = extract_text(doc)
    doc.close()
    return text


def applied_changes(result: PerturbationResult) -> list[dict]:
    return [c for c in result.changes if "skipped" not in c]


def run(
    fixtures_dir: Path, tmp_path: Path, op: PerturbationType, seed: int = 42
) -> tuple[Path, Path, PerturbationResult]:
    src = fixtures_dir / "sample.pdf"
    dst = tmp_path / f"perturbed_{op.value}_{seed}.pdf"
    result = apply_pdf_inplace(src, dst, op, seed=seed)
    return src, dst, result


@pytest.mark.parametrize("op", ALL_OPERATORS)
def test_operator_applies_on_fixture(
    op: PerturbationType, fixtures_dir: Path, tmp_path: Path
) -> None:
    _, dst, result = run(fixtures_dir, tmp_path, op)
    assert result.applied, f"{op.value} skipped: {result.skip_reason}"
    assert result.changes
    assert result.description
    assert dst.exists()


# ---------------------------------------------------------------------------
# T1 — changed-span
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", SPAN_OPERATORS)
def test_t1_changed_span_replacements_present(
    op: PerturbationType, fixtures_dir: Path, tmp_path: Path
) -> None:
    _, dst, result = run(fixtures_dir, tmp_path, op)
    text = extracted(dst)
    changes = applied_changes(result)
    assert changes
    doc = fitz.open(dst)
    for change in changes:
        assert change["corrupted"] in text, f"replacement {change['corrupted']!r} missing"
        # Locus check: at the edited rect, the bare original token is gone.
        page = doc[change["page"]]
        rect = fitz.Rect(change["rect"])
        locus_words = [w[4] for w in page.get_text("words") if fitz.Rect(w[:4]).intersects(rect)]
        assert change["original"] not in locus_words
    doc.close()


def test_t1_section_removal_blocks_absent(fixtures_dir: Path, tmp_path: Path) -> None:
    _, dst, result = run(fixtures_dir, tmp_path, PerturbationType.SECTION_REMOVAL)
    text = " ".join(extracted(dst).split())
    for change in result.changes:
        preview = change["content_preview"].removesuffix("...")
        assert preview not in text, f"removed section still present: {preview[:40]!r}"


def test_t1_page_removal_page_absent(fixtures_dir: Path, tmp_path: Path) -> None:
    src, dst, result = run(fixtures_dir, tmp_path, PerturbationType.PAGE_REMOVAL)
    (change,) = result.changes
    removed_idx = change["page"]
    markers = ["PAGEONE", "PAGETWO", "PAGETHREE", "PAGEFOUR"]
    text = extracted(dst)
    assert markers[removed_idx] not in text
    assert 1 <= removed_idx <= 2  # seeded *middle* page
    doc = fitz.open(dst)
    assert doc.page_count == 3
    doc.close()


def test_t1_page_shuffle_order_changed(fixtures_dir: Path, tmp_path: Path) -> None:
    src, dst, result = run(fixtures_dir, tmp_path, PerturbationType.PAGE_SHUFFLE)
    (change,) = result.changes
    order = change["new_order"]
    assert sorted(order) == [0, 1, 2, 3]
    assert order != [0, 1, 2, 3]
    markers = ["PAGEONE", "PAGETWO", "PAGETHREE", "PAGEFOUR"]
    doc = fitz.open(dst)
    for new_pos, old_pos in enumerate(order):
        assert markers[old_pos] in doc[new_pos].get_text()
    doc.close()


# ---------------------------------------------------------------------------
# T2 — untouched-region
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", SPAN_OPERATORS + [PerturbationType.SECTION_REMOVAL])
def test_t2_untouched_regions_identical(
    op: PerturbationType, fixtures_dir: Path, tmp_path: Path
) -> None:
    src, dst, result = run(fixtures_dir, tmp_path, op)
    edited_rects: dict[int, list[fitz.Rect]] = {}
    for change in applied_changes(result):
        edited_rects.setdefault(change["page"], []).append(fitz.Rect(change["rect"]))

    clean_pages = words_per_page(src)
    perturbed_pages = words_per_page(dst)
    assert len(clean_pages) == len(perturbed_pages)

    def untouched(words: Words, rects: list[fitz.Rect]) -> list[str]:
        return [w[4] for w in words if not any(fitz.Rect(w[:4]).intersects(r) for r in rects)]

    for page_no, (clean, perturbed) in enumerate(zip(clean_pages, perturbed_pages, strict=True)):
        rects = edited_rects.get(page_no, [])
        assert untouched(clean, rects) == untouched(perturbed, rects), (
            f"untouched words differ on page {page_no}"
        )


def test_t2_page_removal_surviving_pages_identical(fixtures_dir: Path, tmp_path: Path) -> None:
    src, dst, result = run(fixtures_dir, tmp_path, PerturbationType.PAGE_REMOVAL)
    removed_idx = result.changes[0]["page"]
    clean_pages = words_per_page(src)
    surviving = [p for i, p in enumerate(clean_pages) if i != removed_idx]
    assert words_per_page(dst) == surviving


def test_t2_page_shuffle_pages_identical_modulo_order(fixtures_dir: Path, tmp_path: Path) -> None:
    src, dst, result = run(fixtures_dir, tmp_path, PerturbationType.PAGE_SHUFFLE)
    order = result.changes[0]["new_order"]
    clean_pages = words_per_page(src)
    assert words_per_page(dst) == [clean_pages[i] for i in order]


# ---------------------------------------------------------------------------
# T3 — non-noop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ALL_OPERATORS)
def test_t3_non_noop(op: PerturbationType, fixtures_dir: Path, tmp_path: Path) -> None:
    src, dst, result = run(fixtures_dir, tmp_path, op)
    assert (
        hashlib.sha256(src.read_bytes()).hexdigest() != hashlib.sha256(dst.read_bytes()).hexdigest()
    )
    assert extracted(src) != extracted(dst)
    assert result.content_changed


# ---------------------------------------------------------------------------
# T4 — determinism (on extracted text, per seed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ALL_OPERATORS)
def test_t4_same_seed_identical_extracted_text(
    op: PerturbationType, fixtures_dir: Path, tmp_path: Path
) -> None:
    _, dst1, r1 = run(fixtures_dir, tmp_path / "a", op, seed=42)
    _, dst2, r2 = run(fixtures_dir, tmp_path / "b", op, seed=42)
    assert extracted(dst1) == extracted(dst2)
    assert r1.changes == r2.changes
    assert r1.corrupted_hash == r2.corrupted_hash


@pytest.mark.parametrize("op", ALL_OPERATORS)
def test_seed_variation_changes_selection(
    op: PerturbationType, fixtures_dir: Path, tmp_path: Path
) -> None:
    outputs = set()
    for seed in range(8):
        _, dst, result = run(fixtures_dir, tmp_path / str(seed), op, seed=seed)
        outputs.add(extracted(dst))
    assert len(outputs) > 1, f"{op.value} identical across 8 seeds"


# ---------------------------------------------------------------------------
# NULL_CONTENT, metadata scrub, skips, errors
# ---------------------------------------------------------------------------


def test_null_content_single_blank_page(fixtures_dir: Path, tmp_path: Path) -> None:
    _, dst, result = run(fixtures_dir, tmp_path, PerturbationType.NULL_CONTENT)
    assert result.applied
    assert result.corrupted_content == ""
    doc = fitz.open(dst)
    assert doc.page_count == 1
    assert doc[0].get_text().strip() == ""
    doc.close()


@pytest.mark.parametrize("op", ALL_OPERATORS + [PerturbationType.NULL_CONTENT])
def test_metadata_scrubbed(op: PerturbationType, fixtures_dir: Path, tmp_path: Path) -> None:
    _, dst, _ = run(fixtures_dir, tmp_path, op)
    doc = fitz.open(dst)
    metadata = doc.metadata or {}
    doc.close()
    for key in ("title", "author", "producer", "creator"):
        assert not metadata.get(key), f"{key} survived the scrub: {metadata[key]!r}"


def make_pdf(path: Path, num_pages: int) -> None:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 200), f"Tiny page {i + 1}", fontsize=11)
    doc.save(path)
    doc.close()


def test_page_removal_skips_below_three_pages(tmp_path: Path) -> None:
    src = tmp_path / "two.pdf"
    make_pdf(src, 2)
    dst = tmp_path / "out.pdf"
    result = apply_pdf_inplace(src, dst, PerturbationType.PAGE_REMOVAL)
    assert not result.applied
    assert result.skip_reason == "Need at least 3 pages to remove a middle page"
    assert dst.read_bytes() == src.read_bytes()  # unchanged copy delivered


def test_page_shuffle_skips_single_page(tmp_path: Path) -> None:
    src = tmp_path / "one.pdf"
    make_pdf(src, 1)
    dst = tmp_path / "out.pdf"
    result = apply_pdf_inplace(src, dst, PerturbationType.PAGE_SHUFFLE)
    assert not result.applied
    assert result.skip_reason == "Need at least 2 pages to shuffle"


def test_section_removal_skips_without_enough_blocks(tmp_path: Path) -> None:
    src = tmp_path / "sparse.pdf"
    make_pdf(src, 1)  # single text block
    dst = tmp_path / "out.pdf"
    result = apply_pdf_inplace(src, dst, PerturbationType.SECTION_REMOVAL)
    assert not result.applied
    assert result.skip_reason == "Not enough sections to remove"


def test_span_ops_skip_on_blank_pdf(tmp_path: Path) -> None:
    src = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(src)
    doc.close()
    for op in SPAN_OPERATORS:
        result = apply_pdf_inplace(src, tmp_path / f"{op.value}.pdf", op)
        assert not result.applied
        assert result.skip_reason == "No extractable text spans"


def test_non_inplace_operator_rejected(fixtures_dir: Path, tmp_path: Path) -> None:
    editor = PDFInPlaceEditor(seed=42)
    with pytest.raises(ValueError, match="not an in-place PDF operator"):
        editor.apply(fixtures_dir / "sample.pdf", tmp_path / "x.pdf", PerturbationType.OCR_NOISE)
    assert PerturbationType.OCR_NOISE not in PDF_INPLACE_OPERATORS


def test_missing_pymupdf_raises_typed_error(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "fitz", None)  # import fitz -> ImportError
    with pytest.raises(OptionalDependencyError, match=r"traxr\[document\]"):
        apply_pdf_inplace(
            fixtures_dir / "sample.pdf",
            tmp_path / "x.pdf",
            PerturbationType.NUMBER_CORRUPTION,
        )


def test_overflow_skip_is_recorded(fixtures_dir: Path, tmp_path: Path) -> None:
    """Spans whose replacement cannot fit are skipped, recorded, and never redacted."""
    from traxr.perturb import pdf_inplace as mod

    editor = PDFInPlaceEditor(seed=42)
    # Force every fit attempt to fail. Save the staticmethod DESCRIPTOR
    # (class __dict__), not the unwrapped function — restoring the bare
    # function would rebind it as an instance method and break later tests.
    original_fit = mod.PDFInPlaceEditor.__dict__["_fit_fontsize"]
    try:
        mod.PDFInPlaceEditor._fit_fontsize = staticmethod(  # type: ignore[method-assign]
            lambda fitz, text, width, original_size: None
        )
        src = fixtures_dir / "sample.pdf"
        dst = tmp_path / "out.pdf"
        result = editor.apply(src, dst, PerturbationType.NUMBER_CORRUPTION)
    finally:
        mod.PDFInPlaceEditor._fit_fontsize = original_fit  # type: ignore[method-assign]

    assert not result.applied  # every selected span was an overflow skip
    assert result.changes  # ...but the skips are recorded
    assert all(c.get("skipped") for c in result.changes)
    assert extracted(src) == extracted(dst)  # nothing redacted
