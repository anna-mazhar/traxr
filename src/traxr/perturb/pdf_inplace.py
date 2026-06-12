"""In-place surgical PDF editing (NEW in traxr; PyMuPDF).

For external agents the perturbed artifact must be a **real PDF on disk** —
the agent reads files itself, so content injection is impossible. This module
applies surgical edits to a copy of the PDF, leaving everything else
untouched.

Span selection reuses the text operators' logic at span granularity (same
regexes, same probabilities, same seeding — via the constants on
:class:`traxr.perturb.pdf.PDFPerturbator`).

Documented behavior divergences from the text-path operators:

* **Edits are capped to the recorded list.** The text operators record at most
  10/15 ``changes`` but apply more (the regex sub is uncapped). Here every
  actual edit is recorded, so every edit is verifiable (guard test T1).
* **Extraction fidelity, not visual fidelity.** Embedded fonts are subsets, so
  replacements are reinserted in base-14 ``helv`` at the span's size/color.
  Agents consume extracted text; visuals may differ.
* **Determinism is asserted on extracted text per seed, not bytes** (PDF IDs
  and dates are scrubbed but not byte-gated).

Overflow: replacements can lengthen spans and ``insert_textbox`` silently
clips. Mitigation: shrink fontsize stepwise (floor 60% of the original); if it
still cannot fit, that span is skipped with a recorded skip (and never
redacted).
"""

import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traxr.errors import OptionalDependencyError

from .pdf import PDFPerturbator
from .types import PerturbationResult, PerturbationType

# Operators deliverable as in-place PDF edits (external agents).
PDF_INPLACE_OPERATORS = (
    PerturbationType.NUMBER_CORRUPTION,
    PerturbationType.TEXT_REDACTION,
    PerturbationType.SECTION_REMOVAL,
    PerturbationType.PAGE_REMOVAL,
    PerturbationType.PAGE_SHUFFLE,
    PerturbationType.NULL_CONTENT,
)

# Selection constants shared with the text-path PDF operators.
_NUMBER_PATTERN = re.compile(PDFPerturbator.NUMBER_PATTERN)
_DATE_PATTERN = re.compile(PDFPerturbator.DATE_PATTERN)
_REDACT_NUMBER_PATTERN = re.compile(PDFPerturbator.REDACT_NUMBER_PATTERN)
_NAME_PATTERN = re.compile(PDFPerturbator.NAME_PATTERN)
_NUMBER_NOISE_PATTERNS = PDFPerturbator.NUMBER_NOISE_PATTERNS
_NUMBER_CORRUPT_P = 0.4  # pdf.py:240 — corrupt each number with p=0.4
_DATE_REDACT_P = 0.4  # pdf.py:275
_NUMBER_REDACT_P = 0.3  # pdf.py:290
_NAME_REDACT_P = 0.25  # pdf.py:304
_NUMBER_CORRUPTION_CAP = 10  # pdf.py:246 (changes cap; here = actual edit cap)
_REDACTION_CAP = 15  # pdf.py:276 (changes cap; here = actual edit cap)

_FONT_SHRINK_STEP = 0.95
_FONT_SHRINK_FLOOR = 0.6  # of the span's original size


def _require_fitz() -> Any:
    try:
        import fitz
    except ImportError as exc:
        raise OptionalDependencyError(
            "PyMuPDF is required for in-place PDF editing. "
            'Install with: pip install "traxr[document]"'
        ) from exc
    return fitz


@dataclass
class _SpanRec:
    """A text span with its location, ready for redact+reinsert editing."""

    page_no: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    color: int  # sRGB int as reported by get_text("dict")
    new_text: str = ""

    def __post_init__(self) -> None:
        if not self.new_text:
            self.new_text = self.text


@dataclass
class _Plan:
    """Selected edits plus recorded changes for one operator application."""

    edits: list[_SpanRec] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""
    skip_reason: str | None = None


def extract_text(doc: Any) -> str:
    """Extract plain text from an open PyMuPDF document, page by page."""
    return "\n".join(page.get_text() for page in doc)


class PDFInPlaceEditor:
    """Applies seeded surgical perturbations to a PDF file on disk.

    Flow per perturbation: open the source PDF -> span/page selection with the
    mined seeded logic -> redact + reinsert (or page ops) -> scrub metadata ->
    save to ``dst_path`` with ``garbage=3, deflate=True``. The source file is
    never modified. On a recorded skip, the source is copied unchanged so the
    destination artifact always exists.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def apply(
        self,
        src_path: str | Path,
        dst_path: str | Path,
        perturbation: PerturbationType,
    ) -> PerturbationResult:
        """Apply ``perturbation`` to ``src_path``, writing the result to ``dst_path``.

        Returns a :class:`PerturbationResult` whose ``original_content`` /
        ``corrupted_content`` are the **extracted texts** (the determinism
        contract for PDFs).
        """
        fitz = _require_fitz()
        src = Path(src_path)
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if perturbation not in PDF_INPLACE_OPERATORS:
            raise ValueError(
                f"{perturbation.value} is not an in-place PDF operator; "
                f"supported: {[p.value for p in PDF_INPLACE_OPERATORS]}"
            )

        rng = random.Random(self.seed)  # reset per apply, like the text path
        doc = fitz.open(src)
        try:
            original_text = extract_text(doc)

            if perturbation == PerturbationType.NULL_CONTENT:
                doc.close()
                blank = fitz.open()
                blank.new_page()
                self._scrub_and_save(blank, dst)
                blank.close()
                return PerturbationResult(
                    original_content=original_text,
                    corrupted_content="",
                    perturbation_type=perturbation,
                    description="PDF replaced with a single blank page",
                    file_type="pdf",
                    file_name=src.name,
                )

            if perturbation == PerturbationType.PAGE_REMOVAL:
                plan = self._plan_page_removal(doc, rng)
            elif perturbation == PerturbationType.PAGE_SHUFFLE:
                plan = self._plan_page_shuffle(doc, rng)
            elif perturbation == PerturbationType.NUMBER_CORRUPTION:
                plan = self._plan_number_corruption(fitz, doc, rng)
            elif perturbation == PerturbationType.TEXT_REDACTION:
                plan = self._plan_text_redaction(fitz, doc, rng)
            else:  # SECTION_REMOVAL
                plan = self._plan_section_removal(doc, rng)

            if plan.skip_reason is not None:
                doc.close()
                shutil.copyfile(src, dst)
                return PerturbationResult(
                    original_content=original_text,
                    corrupted_content=original_text,
                    perturbation_type=perturbation,
                    description="",
                    changes=plan.changes,
                    applied=False,
                    skip_reason=plan.skip_reason,
                    file_type="pdf",
                    file_name=src.name,
                )

            if perturbation == PerturbationType.PAGE_REMOVAL:
                doc.delete_page(plan.changes[0]["page"])
            elif perturbation == PerturbationType.PAGE_SHUFFLE:
                doc.select(plan.changes[0]["new_order"])
            elif perturbation == PerturbationType.SECTION_REMOVAL:
                self._apply_redactions(fitz, doc, plan.edits, reinsert=False)
            else:
                self._apply_redactions(fitz, doc, plan.edits, reinsert=True)

            self._scrub_and_save(doc, dst)
            doc.close()

            check = fitz.open(dst)
            corrupted_text = extract_text(check)
            check.close()

            return PerturbationResult(
                original_content=original_text,
                corrupted_content=corrupted_text,
                perturbation_type=perturbation,
                description=plan.description,
                changes=plan.changes,
                file_type="pdf",
                file_name=src.name,
            )
        finally:
            if not doc.is_closed:
                doc.close()

    # =====================================================================
    # Selection (mined logic at span/block/page granularity)
    # =====================================================================

    def _collect_spans(self, doc: Any) -> list[_SpanRec]:
        """Collect all text spans across the document in reading order."""
        spans: list[_SpanRec] = []
        for page_no, page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if not span["text"].strip():
                            continue
                        spans.append(
                            _SpanRec(
                                page_no=page_no,
                                bbox=tuple(span["bbox"]),
                                text=span["text"],
                                size=span["size"],
                                color=span["color"],
                            )
                        )
        return spans

    def _plan_number_corruption(self, fitz: Any, doc: Any, rng: random.Random) -> _Plan:
        """Same regex/probabilities as pdf.py:224-257, capped to recorded edits."""
        plan = _Plan()
        spans = self._collect_spans(doc)
        if not spans:
            plan.skip_reason = "No extractable text spans"
            return plan

        budget = _NUMBER_CORRUPTION_CAP
        for rec in spans:
            if budget <= 0:
                break

            def corrupt(match: re.Match[str], rec: _SpanRec = rec) -> str:
                nonlocal budget
                original = match.group(0)
                if budget <= 0:
                    return original
                # 40% chance to corrupt each number (pdf.py:240)
                if rng.random() > _NUMBER_CORRUPT_P:
                    return original
                prefix, suffix = rng.choice(_NUMBER_NOISE_PATTERNS)
                corrupted = f"{prefix}{original}{suffix}"
                budget -= 1
                plan.changes.append(
                    {
                        "type": "number_corruption",
                        "page": rec.page_no,
                        "rect": list(rec.bbox),
                        "original": original,
                        "corrupted": corrupted,
                    }
                )
                return corrupted

            rec.new_text = _NUMBER_PATTERN.sub(corrupt, rec.text)

        self._finalize_span_plan(fitz, plan, spans)
        plan.description = f"Added noise to {len(plan.changes)} numbers (in-place)"
        if not plan.edits and plan.skip_reason is None:
            plan.skip_reason = "No numbers selected for corruption"
        return plan

    def _plan_text_redaction(self, fitz: Any, doc: Any, rng: random.Random) -> _Plan:
        """Same three passes/probabilities as pdf.py:259-317, capped to recorded edits."""
        plan = _Plan()
        spans = self._collect_spans(doc)
        if not spans:
            plan.skip_reason = "No extractable text spans"
            return plan

        budget = _REDACTION_CAP

        def make_pass(pattern: re.Pattern[str], probability: float, kind: str, marker: str) -> None:
            nonlocal budget
            for rec in spans:
                if budget <= 0:
                    return

                def redact(match: re.Match[str], rec: _SpanRec = rec) -> str:
                    nonlocal budget
                    if budget <= 0:
                        return match.group(0)
                    if rng.random() < probability:
                        budget -= 1
                        plan.changes.append(
                            {
                                "type": kind,
                                "page": rec.page_no,
                                "rect": list(rec.bbox),
                                "original": match.group(0),
                                "corrupted": marker,
                            }
                        )
                        return marker
                    return match.group(0)

                rec.new_text = pattern.sub(redact, rec.new_text)

        # Pass order mirrors the text path: dates, then numbers, then names.
        make_pass(_DATE_PATTERN, _DATE_REDACT_P, "date_redaction", "[DATE]")
        make_pass(_REDACT_NUMBER_PATTERN, _NUMBER_REDACT_P, "number_redaction", "[REDACTED]")
        make_pass(_NAME_PATTERN, _NAME_REDACT_P, "name_redaction", "[NAME]")

        self._finalize_span_plan(fitz, plan, spans)
        plan.description = f"Redacted {len(plan.changes)} values (in-place)"
        if not plan.edits and plan.skip_reason is None:
            plan.skip_reason = "No values selected for redaction"
        return plan

    def _finalize_span_plan(self, fitz: Any, plan: _Plan, spans: list[_SpanRec]) -> None:
        """Keep only changed spans that fit after font shrinking; record skips.

        A span whose replacement cannot fit (even at the 60% fontsize floor)
        is dropped from the edit list — it is never redacted — and the skip is
        recorded on its change entries.
        """
        for rec in spans:
            if rec.new_text == rec.text:
                continue
            width = rec.bbox[2] - rec.bbox[0]
            fontsize = self._fit_fontsize(fitz, rec.new_text, width, rec.size)
            if fontsize is None:
                rec.new_text = rec.text  # never redact a span we cannot reinsert
                for change in plan.changes:
                    if change["rect"] == list(rec.bbox):
                        change["skipped"] = "replacement does not fit at >=60% fontsize"
                continue
            rec.size = fontsize
            plan.edits.append(rec)
        # Changes capped == applied: drop entries for spans skipped on overflow
        # from the *applied* count but keep them (flagged) for transparency.

    @staticmethod
    def _fit_fontsize(fitz: Any, text: str, width: float, original_size: float) -> float | None:
        """Shrink fontsize stepwise until ``text`` fits in ``width``.

        Returns the fontsize to use, or None if it cannot fit at >= 60% of the
        span's original size.
        """
        floor = original_size * _FONT_SHRINK_FLOOR
        fontsize = original_size
        while fontsize >= floor:
            if fitz.get_text_length(text, fontname="helv", fontsize=fontsize) <= width:
                return fontsize
            fontsize *= _FONT_SHRINK_STEP
        return None

    def _plan_section_removal(self, doc: Any, rng: random.Random) -> _Plan:
        """Block-granular mirror of pdf.py:410-459 (v1: single-block units only)."""
        plan = _Plan()
        blocks: list[tuple[int, tuple[float, float, float, float], str]] = []
        for page_no, page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                text = " ".join(
                    span["text"] for line in block["lines"] for span in line["spans"]
                ).strip()
                if text:
                    blocks.append((page_no, tuple(block["bbox"]), text))

        if len(blocks) < 3:
            plan.skip_reason = "Not enough sections to remove"
            return plan

        # Keep first and last blocks; remove 1-2 from the middle (pdf.py:429).
        num_to_remove = min(rng.randint(1, 2), len(blocks) - 2)
        removable_indices = list(range(1, len(blocks) - 1))
        indices_to_remove = rng.sample(
            removable_indices, min(num_to_remove, len(removable_indices))
        )
        indices_to_remove.sort(reverse=True)

        for idx in indices_to_remove:
            page_no, bbox, text = blocks[idx]
            plan.changes.append(
                {
                    "type": "section_removal",
                    "position": idx,
                    "page": page_no,
                    "rect": list(bbox),
                    "content_preview": text[:100] + "..." if len(text) > 100 else text,
                }
            )
            plan.edits.append(_SpanRec(page_no=page_no, bbox=bbox, text=text, size=0.0, color=0))

        plan.description = f"Removed {len(plan.changes)} section(s) (in-place)"
        return plan

    def _plan_page_removal(self, doc: Any, rng: random.Random) -> _Plan:
        """Delete a seeded middle page."""
        plan = _Plan()
        n = doc.page_count
        if n < 3:
            plan.skip_reason = "Need at least 3 pages to remove a middle page"
            return plan
        page_idx = rng.randrange(1, n - 1)
        plan.changes.append({"type": "page_removal", "page": page_idx, "page_count": n})
        plan.description = f"Removed page {page_idx + 1} of {n}"
        return plan

    def _plan_page_shuffle(self, doc: Any, rng: random.Random) -> _Plan:
        """Reorder pages with a seeded non-identity permutation."""
        plan = _Plan()
        n = doc.page_count
        if n < 2:
            plan.skip_reason = "Need at least 2 pages to shuffle"
            return plan
        order = list(range(n))
        attempts = 0
        while order == list(range(n)) and attempts < 10:
            rng.shuffle(order)
            attempts += 1
        if order == list(range(n)):
            plan.skip_reason = "Could not derive a non-identity page order"
            return plan
        plan.changes.append({"type": "page_shuffle", "new_order": order})
        plan.description = f"Shuffled {n} pages into order {order}"
        return plan

    # =====================================================================
    # Application
    # =====================================================================

    def _apply_redactions(self, fitz: Any, doc: Any, edits: list[_SpanRec], reinsert: bool) -> None:
        """Redact each edit's rect, then (optionally) reinsert the replacement.

        Reinsertion uses base-14 ``helv`` at the (possibly shrunk) span size
        and the span's color — extraction fidelity, not visual fidelity.
        """
        by_page: dict[int, list[_SpanRec]] = {}
        for rec in edits:
            by_page.setdefault(rec.page_no, []).append(rec)

        for page_no, recs in by_page.items():
            page = doc[page_no]
            for rec in recs:
                page.add_redact_annot(fitz.Rect(rec.bbox))
            page.apply_redactions()
            if not reinsert:
                continue
            for rec in recs:
                rect = fitz.Rect(rec.bbox)
                color = (
                    ((rec.color >> 16) & 255) / 255,
                    ((rec.color >> 8) & 255) / 255,
                    (rec.color & 255) / 255,
                )
                # Give the textbox vertical slack: redaction already cleared
                # the area, and insert_textbox clips on tight ascender room.
                slack = fitz.Rect(
                    rect.x0, rect.y0 - rec.size * 0.2, rect.x1 + 2, rect.y1 + rec.size
                )
                rv = page.insert_textbox(
                    slack,
                    rec.new_text,
                    fontname="helv",
                    fontsize=rec.size,
                    color=color,
                )
                if rv < 0:
                    # Last-resort: insert_text never clips; keeps T1 intact.
                    page.insert_text(
                        (rect.x0, rect.y1 - rec.size * 0.2),
                        rec.new_text,
                        fontname="helv",
                        fontsize=rec.size,
                        color=color,
                    )

    @staticmethod
    def _scrub_and_save(doc: Any, dst: Path) -> None:
        """Scrub metadata and save with full garbage collection."""
        doc.set_metadata({})
        doc.del_xml_metadata()
        doc.save(str(dst), garbage=3, deflate=True)


def apply_pdf_inplace(
    src_path: str | Path,
    dst_path: str | Path,
    perturbation: PerturbationType,
    seed: int = 42,
) -> PerturbationResult:
    """Convenience wrapper: one-shot in-place PDF perturbation."""
    return PDFInPlaceEditor(seed=seed).apply(src_path, dst_path, perturbation)
