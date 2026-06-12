"""Perturbation types and result structures.

Includes the PDF-native operator types ``PAGE_REMOVAL`` and
``PAGE_SHUFFLE`` (delivered by :mod:`traxr.perturb.pdf_inplace`).
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PerturbationType(Enum):
    """Individual perturbation strategies."""

    # Tabular perturbations
    COLUMN_SWAP = "column_swap"
    LABEL_CORRUPT = "label_corrupt"
    DATA_TYPE_CORRUPT = "data_type_corrupt"
    ROW_DUPLICATE = "row_duplicate"
    IRRELEVANT_COLUMNS = "irrelevant_columns"
    UNIT_CHANGE = "unit_change"

    # PDF perturbations
    OCR_NOISE = "ocr_noise"
    NUMBER_CORRUPTION = "number_corruption"
    TEXT_REDACTION = "text_redaction"
    PARAGRAPH_SHUFFLE = "paragraph_shuffle"
    ENCODING_ERROR = "encoding_error"
    SECTION_REMOVAL = "section_removal"

    # PDF-native perturbations (NEW in traxr; applied in-place via PyMuPDF)
    PAGE_REMOVAL = "page_removal"
    PAGE_SHUFFLE = "page_shuffle"

    # Image perturbations
    BLUR = "blur"
    NOISE = "noise"
    LOW_RESOLUTION = "low_resolution"
    PARTIAL_OCCLUSION = "partial_occlusion"
    CONTRAST_REDUCTION = "contrast_reduction"
    WATERMARK = "watermark"

    # Audio perturbations
    BACKGROUND_NOISE = "background_noise"
    SPEED_CHANGE = "speed_change"
    LOW_PASS_FILTER = "low_pass_filter"

    # Content-level perturbations
    NULL_CONTENT = "null_content"  # Return empty content


PERTURBATION_DESCRIPTIONS = {
    # Tabular perturbations
    PerturbationType.COLUMN_SWAP: "Swap two random columns in tabular data",
    PerturbationType.LABEL_CORRUPT: "Replace column headers with acronyms/synonyms",
    PerturbationType.DATA_TYPE_CORRUPT: "Add symbols to numeric values (e.g., ~, *)",
    PerturbationType.ROW_DUPLICATE: "Add duplicate rows with slight variations",
    PerturbationType.IRRELEVANT_COLUMNS: "Add 1-2 irrelevant noise columns",
    PerturbationType.UNIT_CHANGE: "Multiply numeric columns by unit conversion factor",
    # PDF perturbations
    PerturbationType.OCR_NOISE: "Simulate OCR errors (l->1, O->0, rn->m, etc.)",
    PerturbationType.NUMBER_CORRUPTION: "Add noise/symbols to numbers in text",
    PerturbationType.TEXT_REDACTION: "Replace key values with [REDACTED] markers",
    PerturbationType.PARAGRAPH_SHUFFLE: "Randomly reorder paragraphs",
    PerturbationType.ENCODING_ERROR: "Simulate character encoding errors",
    PerturbationType.SECTION_REMOVAL: "Remove a random section/paragraph from the document",
    # PDF-native perturbations
    PerturbationType.PAGE_REMOVAL: "Remove a seeded middle page from the PDF",
    PerturbationType.PAGE_SHUFFLE: "Reorder the pages of the PDF",
    # Image perturbations
    PerturbationType.BLUR: "Apply Gaussian blur to reduce image clarity",
    PerturbationType.NOISE: "Add random noise (salt & pepper) to the image",
    PerturbationType.LOW_RESOLUTION: "Reduce image resolution (downscale then upscale)",
    PerturbationType.PARTIAL_OCCLUSION: "Add random shapes to occlude parts of the image",
    PerturbationType.CONTRAST_REDUCTION: "Reduce image contrast (washed out appearance)",
    PerturbationType.WATERMARK: "Add a distracting watermark overlay",
    # Audio perturbations
    PerturbationType.BACKGROUND_NOISE: "Add background noise (white noise/static) to audio",
    PerturbationType.SPEED_CHANGE: "Change audio playback speed (faster or slower)",
    PerturbationType.LOW_PASS_FILTER: "Apply low-pass filter (muffled/phone quality)",
    # Content-level perturbations
    PerturbationType.NULL_CONTENT: "Return empty content (simulates missing file)",
}


@dataclass
class PerturbationResult:
    """Result of applying a perturbation."""

    # Core result
    original_content: str
    corrupted_content: str
    perturbation_type: PerturbationType

    # Description of what was changed
    description: str

    # Hashes for tracking
    original_hash: str = ""
    corrupted_hash: str = ""

    # Detailed diff info
    changes: list[dict[str, Any]] = field(default_factory=list)

    # Whether perturbation was actually applied
    applied: bool = True
    skip_reason: str | None = None

    # File metadata
    file_type: str = ""
    file_name: str = ""

    # Binary content for images (if applicable)
    corrupted_bytes: bytes | None = None

    def __post_init__(self) -> None:
        if not self.original_hash:
            self.original_hash = hashlib.sha256(self.original_content.encode()).hexdigest()[:16]
        if not self.corrupted_hash and self.corrupted_content:
            self.corrupted_hash = hashlib.sha256(self.corrupted_content.encode()).hexdigest()[:16]

    @property
    def content_changed(self) -> bool:
        """Check if content actually changed."""
        return self.original_hash != self.corrupted_hash

    @property
    def diff_summary(self) -> str:
        """Human-readable summary of the diff."""
        if not self.applied:
            return f"No change: {self.skip_reason}"
        if not self.content_changed:
            return "No observable change"
        return f"{self.perturbation_type.value}: {self.description}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "perturbation_type": self.perturbation_type.value,
            "description": self.description,
            "applied": self.applied,
            "skip_reason": self.skip_reason,
            "content_changed": self.content_changed,
            "original_hash": self.original_hash,
            "corrupted_hash": self.corrupted_hash,
            "file_type": self.file_type,
            "file_name": self.file_name,
            "changes": self.changes,
            # Don't include full content in dict - too large
            "original_length": len(self.original_content),
            "corrupted_length": len(self.corrupted_content),
        }
        # Add binary content info for images
        if self.corrupted_bytes is not None:
            result["corrupted_bytes_length"] = len(self.corrupted_bytes)
        return result
