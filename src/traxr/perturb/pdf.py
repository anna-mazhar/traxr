"""PDF/text document perturbation strategies.

Operates on extracted/whole text content: this is the faithful round-trip path
for TXT/MD and the content-injection path for PDFs consumed by the built-in
agent. The external-agent PDF path (real perturbed PDF on disk) lives in
:mod:`traxr.perturb.pdf_inplace`, which reuses this module's span-selection
constants.
"""

import random
import re
from typing import Any

from .types import PerturbationResult, PerturbationType


class PDFPerturbator:
    """Applies perturbations to PDF/text document content.

    Works on extracted text content from PDFs. Each perturbation is applied
    independently and deterministically based on the provided seed.
    """

    # OCR error patterns: original -> possible errors
    OCR_SUBSTITUTIONS = {
        # Letter-digit confusions
        "l": ["1", "|"],
        "I": ["1", "|", "l"],
        "O": ["0"],
        "o": ["0"],
        "S": ["5", "$"],
        "s": ["5"],
        "B": ["8"],
        "g": ["9"],
        "Z": ["2"],
        "z": ["2"],
        # Letter confusions
        "rn": ["m"],
        "m": ["rn"],
        "cl": ["d"],
        "d": ["cl"],
        "vv": ["w"],
        "w": ["vv"],
        "ii": ["u"],
        "u": ["ii"],
        "h": ["li"],
        "fi": ["fl"],
        "fl": ["fi"],
        # Punctuation
        ",": ["."],
        ".": [","],
        ":": [";"],
        ";": [":"],
    }

    # Characters that simulate encoding errors
    ENCODING_ERROR_CHARS = [
        "�",  # Replacement character (�)
        "??",  # Double question mark (common fallback)
        "###",  # Hash replacement
        "[?]",  # Bracketed question mark
        "\\x00",  # Null byte representation
        "_",  # Underscore (simple replacement)
        "~",  # Tilde (common encoding artifact)
        "^",  # Caret
        "|",  # Pipe
        "*",  # Asterisk
    ]

    # Number corruption patterns
    NUMBER_NOISE_PATTERNS = [
        ("~", ""),  # Prefix with ~
        ("", "*"),  # Suffix with *
        ("approx.", ""),  # Prefix with approx.
        ("", "?"),  # Suffix with ?
        ("(", ")"),  # Wrap in parentheses
    ]

    # Span-selection patterns (mined by pdf_inplace; keep in sync with the
    # regexes used in the methods below).
    NUMBER_PATTERN = r"(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?%?)"
    DATE_PATTERN = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    REDACT_NUMBER_PATTERN = r"\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b"
    NAME_PATTERN = r"(?<=[.!?]\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"

    SUPPORTED_TYPES = {"pdf", "txt", "text", "md", "markdown", "doc", "docx"}

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def can_handle(self, file_type: str) -> bool:
        """Check if this perturbator handles the file type."""
        return file_type.lower() in self.SUPPORTED_TYPES

    def apply(
        self,
        content: str,
        perturbation: PerturbationType,
        file_type: str = "pdf",
        file_name: str = "",
    ) -> PerturbationResult:
        """Apply a single perturbation to PDF/text content.

        Args:
            content: Extracted text content from PDF
            perturbation: Which perturbation to apply
            file_type: File type hint
            file_name: Original file name

        Returns:
            PerturbationResult with corrupted content and metadata
        """
        # Reset RNG for reproducibility
        self._rng = random.Random(self.seed)

        # Handle NULL case
        if perturbation == PerturbationType.NULL_CONTENT:
            return PerturbationResult(
                original_content=content,
                corrupted_content="",
                perturbation_type=perturbation,
                description="Content replaced with empty",
                file_type=file_type,
                file_name=file_name,
            )

        # Check minimum content
        if not content or len(content.strip()) < 10:
            return PerturbationResult(
                original_content=content,
                corrupted_content=content,
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="Insufficient content (need at least 10 characters)",
                file_type=file_type,
                file_name=file_name,
            )

        # Apply specific perturbation
        corrupted_content, description, changes = self._apply_perturbation(content, perturbation)

        return PerturbationResult(
            original_content=content,
            corrupted_content=corrupted_content,
            perturbation_type=perturbation,
            description=description,
            changes=changes,
            file_type=file_type,
            file_name=file_name,
        )

    def _apply_perturbation(
        self,
        content: str,
        perturbation: PerturbationType,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Apply perturbation and return (content, description, changes)."""
        if perturbation == PerturbationType.OCR_NOISE:
            return self._apply_ocr_noise(content)
        elif perturbation == PerturbationType.NUMBER_CORRUPTION:
            return self._apply_number_corruption(content)
        elif perturbation == PerturbationType.TEXT_REDACTION:
            return self._apply_text_redaction(content)
        elif perturbation == PerturbationType.PARAGRAPH_SHUFFLE:
            return self._apply_paragraph_shuffle(content)
        elif perturbation == PerturbationType.ENCODING_ERROR:
            return self._apply_encoding_error(content)
        elif perturbation == PerturbationType.SECTION_REMOVAL:
            return self._apply_section_removal(content)
        else:
            return content, "Unknown perturbation", []

    # =========================================================================
    # Individual Perturbation Implementations
    # =========================================================================

    def _apply_ocr_noise(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Simulate OCR errors in text.

        Applies character substitutions that mimic common OCR mistakes.
        """
        changes: list[dict[str, Any]] = []
        result = list(content)
        error_count = 0

        # Target ~5-10% of applicable characters
        error_rate = 0.08

        i = 0
        while i < len(result):
            # Check for multi-character patterns first
            for pattern, replacements in self.OCR_SUBSTITUTIONS.items():
                if len(pattern) > 1:
                    substring = "".join(result[i : i + len(pattern)])
                    if substring == pattern and self._rng.random() < error_rate:
                        replacement = self._rng.choice(replacements)
                        # Replace the pattern
                        for j in range(len(pattern)):
                            if i + j < len(result):
                                result[i + j] = ""
                        result[i] = replacement
                        error_count += 1
                        if len(changes) < 10:
                            changes.append(
                                {
                                    "type": "ocr_substitution",
                                    "position": i,
                                    "original": pattern,
                                    "replacement": replacement,
                                }
                            )
                        i += len(pattern)
                        break
            else:
                # Single character substitutions
                char = result[i]
                if char in self.OCR_SUBSTITUTIONS and self._rng.random() < error_rate:
                    replacement = self._rng.choice(self.OCR_SUBSTITUTIONS[char])
                    result[i] = replacement
                    error_count += 1
                    if len(changes) < 10:
                        changes.append(
                            {
                                "type": "ocr_substitution",
                                "position": i,
                                "original": char,
                                "replacement": replacement,
                            }
                        )
                i += 1

        corrupted = "".join(result)
        desc = f"Applied {error_count} OCR-style character substitutions"
        return corrupted, desc, changes

    def _apply_number_corruption(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Add noise/symbols to numbers in text.

        Targets numbers like prices, quantities, dates, etc.
        """
        changes: list[dict[str, Any]] = []

        # Find all numbers (including decimals, currency, percentages)
        number_pattern = self.NUMBER_PATTERN

        def corrupt_number(match: re.Match[str]) -> str:
            original = match.group(0)

            # 40% chance to corrupt each number
            if self._rng.random() > 0.4:
                return original

            prefix, suffix = self._rng.choice(self.NUMBER_NOISE_PATTERNS)
            corrupted = f"{prefix}{original}{suffix}"

            if len(changes) < 10:
                changes.append(
                    {
                        "type": "number_corruption",
                        "original": original,
                        "corrupted": corrupted,
                    }
                )

            return corrupted

        corrupted = re.sub(number_pattern, corrupt_number, content)
        desc = f"Added noise to {len(changes)} numbers"
        return corrupted, desc, changes

    def _apply_text_redaction(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Replace key values with [REDACTED] markers.

        Targets:
        - Numbers (prices, quantities, dates)
        - Potential names (capitalized words)
        - Dates
        """
        changes: list[dict[str, Any]] = []

        # Redact dates first (more specific pattern)
        date_pattern = self.DATE_PATTERN

        def redact_date(match: re.Match[str]) -> str:
            if self._rng.random() < 0.4:
                if len(changes) < 15:
                    changes.append(
                        {
                            "type": "date_redaction",
                            "original": match.group(0),
                        }
                    )
                return "[DATE]"
            return match.group(0)

        result = re.sub(date_pattern, redact_date, content)

        # Redact some numbers (not all) - exclude already redacted
        number_pattern = self.REDACT_NUMBER_PATTERN

        def redact_number(match: re.Match[str]) -> str:
            if self._rng.random() < 0.3:
                if len(changes) < 15:
                    changes.append(
                        {
                            "type": "number_redaction",
                            "original": match.group(0),
                        }
                    )
                return "[REDACTED]"
            return match.group(0)

        result = re.sub(number_pattern, redact_number, result)

        # Redact some potential names (Capitalized words after sentence boundaries)
        name_pattern = self.NAME_PATTERN

        def redact_name(match: re.Match[str]) -> str:
            if self._rng.random() < 0.25:
                if len(changes) < 15:
                    changes.append(
                        {
                            "type": "name_redaction",
                            "original": match.group(0),
                        }
                    )
                return "[NAME]"
            return match.group(0)

        result = re.sub(name_pattern, redact_name, result)

        desc = f"Redacted {len(changes)} values"
        return result, desc, changes

    def _apply_paragraph_shuffle(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Randomly reorder paragraphs in the document.

        Preserves the first paragraph (often a title/header) and shuffles the rest.
        """
        # Split into paragraphs (double newline or multiple newlines)
        paragraphs = re.split(r"\n\s*\n", content)

        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) < 3:
            return content, "Not enough paragraphs to shuffle", []

        # Keep first paragraph in place, shuffle the rest
        first_para = paragraphs[0]
        rest = paragraphs[1:]

        # Create shuffle mapping
        original_order = list(range(len(rest)))
        shuffled_order = original_order.copy()
        self._rng.shuffle(shuffled_order)

        # Apply shuffle
        shuffled_rest = [rest[i] for i in shuffled_order]

        # Reconstruct
        result_paragraphs = [first_para] + shuffled_rest
        corrupted = "\n\n".join(result_paragraphs)

        # Record changes
        changes: list[dict[str, Any]] = []
        for new_pos, old_pos in enumerate(shuffled_order):
            if new_pos != old_pos:
                changes.append(
                    {
                        "type": "paragraph_move",
                        "original_position": old_pos + 1,  # +1 because we kept first
                        "new_position": new_pos + 1,
                    }
                )

        desc = f"Shuffled {len(rest)} paragraphs ({len(changes)} moved)"
        return corrupted, desc, changes

    def _apply_encoding_error(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Simulate character encoding errors.

        Randomly replaces characters with encoding artifacts that commonly
        occur when text is decoded with the wrong character set.
        """
        changes: list[dict[str, Any]] = []
        result = list(content)
        error_count = 0

        # Target ~3-5% of characters
        error_rate = 0.04

        # Characters more likely to have encoding issues
        # (common ASCII that often get corrupted)
        target_chars = set("aeiouAEIOU'\".,;:-")

        for i in range(len(result)):
            char = result[i]

            # Higher error rate for special characters
            if char in target_chars:
                rate = error_rate * 3
            elif char.isalpha():
                rate = error_rate
            else:
                continue

            if self._rng.random() < rate:
                replacement = self._rng.choice(self.ENCODING_ERROR_CHARS)
                result[i] = replacement
                error_count += 1

                if len(changes) < 10:
                    changes.append(
                        {
                            "type": "encoding_error",
                            "position": i,
                            "original": char,
                            "replacement": replacement,
                        }
                    )

        corrupted = "".join(result)
        desc = f"Introduced {error_count} encoding errors"
        return corrupted, desc, changes

    def _apply_section_removal(self, content: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Remove a random section/paragraph from the document.

        Simulates incomplete document extraction or missing pages.
        Removes 1-2 paragraphs from the middle of the document.
        """
        # Split into paragraphs (double newline or multiple newlines)
        paragraphs = re.split(r"\n\s*\n", content)

        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) < 3:
            return content, "Not enough paragraphs to remove", []

        # Keep first and last paragraphs (often title/header and conclusion)
        # Remove 1-2 paragraphs from the middle
        num_to_remove = min(self._rng.randint(1, 2), len(paragraphs) - 2)

        # Select paragraphs to remove from the middle section
        middle_start = 1
        middle_end = len(paragraphs) - 1
        removable_indices = list(range(middle_start, middle_end))

        if not removable_indices:
            return content, "No removable paragraphs", []

        indices_to_remove = self._rng.sample(
            removable_indices, min(num_to_remove, len(removable_indices))
        )
        indices_to_remove.sort(reverse=True)  # Remove from end to preserve indices

        changes: list[dict[str, Any]] = []
        for idx in indices_to_remove:
            removed_para = paragraphs[idx]
            changes.append(
                {
                    "type": "section_removal",
                    "position": idx,
                    "content_preview": removed_para[:100] + "..."
                    if len(removed_para) > 100
                    else removed_para,
                }
            )
            paragraphs.pop(idx)

        # Reconstruct document
        corrupted = "\n\n".join(paragraphs)

        desc = f"Removed {len(changes)} paragraph(s) from the document"
        return corrupted, desc, changes
