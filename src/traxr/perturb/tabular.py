"""Tabular data perturbation strategies.

Deterministic/seeded behavior: per-instance seed and an RNG reset at the
top of every :meth:`TabularPerturbator.apply`.
"""

import csv
import io
import json
import math
import random
from collections.abc import Callable
from typing import Any

from .types import PerturbationResult, PerturbationType


def lossless_number(value: str) -> int | float | str:
    """Coerce a cell to ``int``/``float`` only if the conversion round-trips exactly.

    The clean file is copied verbatim, so the perturbed artifact must not
    silently re-type cells. Strings like ``"007"``, ``"1_000"``, ``"+5"``,
    ``"1e3"``, ``"NaN"``, and ``"inf"`` are preserved as text: re-typing them
    would confound divergence attribution, and ``float("NaN")`` would also emit
    invalid JSON. Genuine canonical numbers (``"42"``, ``"3.14"``) still convert.
    """
    s = value.strip()
    if not s:
        return value
    try:
        i = int(s)
        if str(i) == s:  # rejects "007", "+5", "1_000"
            return i
    except (ValueError, TypeError):
        pass
    try:
        f = float(s)
        if math.isfinite(f) and repr(f) == s:  # rejects NaN/inf and non-canonical forms
            return f
    except (ValueError, TypeError):
        pass
    return value


class TabularPerturbator:
    """Applies individual perturbations to tabular data (CSV, Excel).

    Each perturbation is applied independently and deterministically
    based on the provided seed.
    """

    # Column label corruptions (original -> alternatives)
    LABEL_CORRUPTIONS = {
        "name": ["nm", "title", "label"],
        "date": ["dt", "time", "when"],
        "amount": ["amt", "value", "sum"],
        "price": ["cost", "rate", "fee"],
        "total": ["tot", "sum", "aggregate"],
        "quantity": ["qty", "count", "num"],
        "description": ["desc", "details", "info"],
        "category": ["cat", "type", "class"],
        "status": ["state", "condition"],
        "email": ["mail", "e-mail", "address"],
        "phone": ["tel", "number", "contact"],
        "address": ["addr", "location", "place"],
        "id": ["identifier", "key", "code"],
        "revenue": ["rev", "income", "sales"],
        "cost": ["expense", "spend", "outlay"],
    }

    # Unit conversion multipliers
    UNIT_MULTIPLIERS = [
        (3.28084, "meters to feet"),
        (0.621371, "kilometers to miles"),
        (2.20462, "kilograms to pounds"),
        (0.264172, "liters to gallons"),
        (0.393701, "centimeters to inches"),
        (1.8, "Celsius delta scaling"),
    ]

    # Noise columns
    NOISE_COLUMNS: list[tuple[str, Callable[[random.Random], str]]] = [
        ("_noise_id", lambda rng: str(rng.randint(1000, 9999))),
        ("_flag", lambda rng: rng.choice(["X", "", "Y", "N"])),
        ("_legacy", lambda rng: ""),
        ("_temp", lambda rng: "???"),
    ]

    SUPPORTED_TYPES = {"csv", "xlsx", "xls", "excel", "tsv", "json"}

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
        file_type: str = "csv",
        file_name: str = "",
    ) -> PerturbationResult:
        """Apply a single perturbation to tabular content.

        Args:
            content: Raw tabular content (CSV or TSV format)
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

        # Parse content based on file type
        is_json = file_type.lower() == "json"

        if is_json:
            json_rows = self._parse_json(content)
            if json_rows is None:
                return PerturbationResult(
                    original_content=content,
                    corrupted_content=content,
                    perturbation_type=perturbation,
                    description="",
                    applied=False,
                    skip_reason="JSON is not tabular (not an array of objects)",
                    file_type=file_type,
                    file_name=file_name,
                )
            rows = json_rows
        else:
            delimiter = self._detect_delimiter(content)
            rows = self._parse_tabular(content, delimiter)

        if not rows or len(rows) < 2:
            return PerturbationResult(
                original_content=content,
                corrupted_content=content,
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="Insufficient data (need header + at least 1 row)",
                file_type=file_type,
                file_name=file_name,
            )

        # Apply specific perturbation
        corrupted_rows, description, changes = self._apply_perturbation(rows, perturbation)

        # Convert back to string
        if is_json:
            corrupted_content = self._to_json(corrupted_rows)
        else:
            corrupted_content = self._to_tabular(corrupted_rows, delimiter)

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
        rows: list[list[str]],
        perturbation: PerturbationType,
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Apply perturbation and return (rows, description, changes)."""
        if perturbation == PerturbationType.COLUMN_SWAP:
            return self._swap_columns(rows)
        elif perturbation == PerturbationType.LABEL_CORRUPT:
            return self._corrupt_labels(rows)
        elif perturbation == PerturbationType.DATA_TYPE_CORRUPT:
            return self._corrupt_data_types(rows)
        elif perturbation == PerturbationType.ROW_DUPLICATE:
            return self._add_duplicates(rows)
        elif perturbation == PerturbationType.IRRELEVANT_COLUMNS:
            return self._add_irrelevant_columns(rows)
        elif perturbation == PerturbationType.UNIT_CHANGE:
            return self._change_units(rows)
        else:
            return rows, "Unknown perturbation", []

    def _detect_delimiter(self, content: str) -> str:
        """Detect delimiter (tab or comma)."""
        lines = content.split("\n")[:5]
        tab_count = sum(line.count("\t") for line in lines)
        comma_count = sum(line.count(",") for line in lines)
        return "\t" if tab_count > comma_count else ","

    def _parse_tabular(self, content: str, delimiter: str) -> list[list[str]]:
        """Parse tabular content into rows."""
        lines = []
        for line in content.split("\n"):
            # Skip sheet headers from Excel
            if line.strip().startswith("===") and "===" in line:
                continue
            lines.append(line)

        content_cleaned = "\n".join(lines)
        reader = csv.reader(io.StringIO(content_cleaned), delimiter=delimiter)
        try:
            return [row for row in reader if any(cell.strip() for cell in row)]
        except csv.Error:
            # Unparseable pseudo-tabular content (e.g. a bare carriage return
            # in an unquoted field, found by the Hypothesis fuzz). No rows ->
            # apply() records an "insufficient data" skip instead of crashing.
            return []

    def _to_tabular(self, rows: list[list[str]], delimiter: str) -> str:
        """Convert rows back to tabular string."""
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        writer.writerows(rows)
        return output.getvalue()

    def _parse_json(self, content: str) -> list[list[str]] | None:
        """Parse JSON array of objects into rows (header + data).

        Returns None if JSON is not tabular (not an array of objects).
        """
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, RecursionError):
            # RecursionError: pathologically nested input (fuzz hardening);
            # treated the same as non-tabular JSON.
            return None

        # Must be an array
        if not isinstance(data, list) or len(data) == 0:
            return None

        # Must be array of objects (dicts)
        if not all(isinstance(item, dict) for item in data):
            return None

        # Extract all unique keys as header (preserve order from first object)
        all_keys = []
        seen_keys = set()
        for item in data:
            for key in item.keys():
                if key not in seen_keys:
                    all_keys.append(key)
                    seen_keys.add(key)

        if not all_keys:
            return None

        # Build rows: header + data rows
        rows = [all_keys]  # Header row
        for item in data:
            row = [str(item.get(key, "")) for key in all_keys]
            rows.append(row)

        return rows

    def _to_json(self, rows: list[list[str]]) -> str:
        """Convert rows back to JSON array of objects."""
        if not rows or len(rows) < 2:
            return "[]"

        header = rows[0]
        data = []

        for row in rows[1:]:
            obj: dict[str, Any] = {}
            for i, key in enumerate(header):
                value = row[i] if i < len(row) else ""
                # Preserve numeric types only when the coercion is lossless.
                obj[key] = lossless_number(value)
            data.append(obj)

        return json.dumps(data, indent=2, ensure_ascii=False)

    # =========================================================================
    # Individual Perturbation Implementations
    # =========================================================================

    def _swap_columns(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Swap two random columns."""
        if len(rows[0]) < 2:
            return rows, "No swap (single column)", []

        col1 = self._rng.randint(0, len(rows[0]) - 1)
        col2 = self._rng.randint(0, len(rows[0]) - 1)
        attempts = 0
        while col2 == col1 and attempts < 10:
            col2 = self._rng.randint(0, len(rows[0]) - 1)
            attempts += 1

        col1_name = rows[0][col1] if col1 < len(rows[0]) else f"col{col1}"
        col2_name = rows[0][col2] if col2 < len(rows[0]) else f"col{col2}"

        result = []
        for row in rows:
            new_row = row.copy()
            if col1 < len(new_row) and col2 < len(new_row):
                new_row[col1], new_row[col2] = new_row[col2], new_row[col1]
            result.append(new_row)

        changes: list[dict[str, Any]] = [
            {
                "type": "column_swap",
                "col1_idx": col1,
                "col1_name": col1_name,
                "col2_idx": col2,
                "col2_name": col2_name,
            }
        ]
        desc = f"Swapped '{col1_name}' (col {col1}) with '{col2_name}' (col {col2})"
        return result, desc, changes

    def _corrupt_labels(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Replace column headers with acronyms/synonyms."""
        header = rows[0]
        new_header = []
        changes: list[dict[str, Any]] = []

        for i, col_name in enumerate(header):
            lower = col_name.lower().strip()
            corrupted = None

            # Try to find a known corruption
            for key, alternatives in self.LABEL_CORRUPTIONS.items():
                if key in lower:
                    corrupted = self._rng.choice(alternatives)
                    break

            # Fallback: abbreviate
            if corrupted is None and len(col_name) > 3:
                corrupted = col_name[:3].upper()

            if corrupted and corrupted != col_name:
                changes.append(
                    {
                        "type": "label_change",
                        "col_idx": i,
                        "original": col_name,
                        "corrupted": corrupted,
                    }
                )
                new_header.append(corrupted)
            else:
                new_header.append(col_name)

        if not changes:
            return rows, "No labels changed", []

        desc = f"Changed {len(changes)} headers: " + ", ".join(
            f"'{c['original']}'->'{c['corrupted']}'" for c in changes[:3]
        )
        if len(changes) > 3:
            desc += f" (+{len(changes) - 3} more)"

        return [new_header] + rows[1:], desc, changes

    def _corrupt_data_types(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Add symbols to numeric values."""
        result = [rows[0]]
        changes: list[dict[str, Any]] = []
        corrupted_count = 0

        for row_idx, row in enumerate(rows[1:], start=1):
            new_row = []
            for col_idx, cell in enumerate(row):
                try:
                    num = float(cell)
                    # Apply corruption
                    if self._rng.random() < 0.5:
                        new_val = f"~{num}"
                    else:
                        new_val = f"{num:.1f}*"
                    new_row.append(new_val)
                    corrupted_count += 1
                    if len(changes) < 5:  # Limit tracked changes
                        changes.append(
                            {
                                "type": "data_type",
                                "row": row_idx,
                                "col": col_idx,
                                "original": cell,
                                "corrupted": new_val,
                            }
                        )
                except ValueError:
                    new_row.append(cell)
            result.append(new_row)

        desc = f"Added symbols to {corrupted_count} numeric cells"
        return result, desc, changes

    def _add_duplicates(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Add duplicate rows with slight variations."""
        result = [rows[0]]  # Header
        changes: list[dict[str, Any]] = []
        added_count = 0

        for row_idx, row in enumerate(rows[1:], start=1):
            result.append(row)

            # 30% chance to duplicate
            if self._rng.random() < 0.3:
                dup = row.copy()
                if dup and dup[-1]:
                    dup[-1] = dup[-1] + "*"
                result.append(dup)
                added_count += 1
                changes.append({"type": "duplicate", "original_row": row_idx, "variation": "*"})

        desc = f"Added {added_count} duplicate rows"
        return result, desc, changes

    def _add_irrelevant_columns(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Add irrelevant noise columns."""
        num_cols = self._rng.randint(1, 2)
        selected = self._rng.sample(self.NOISE_COLUMNS, min(num_cols, len(self.NOISE_COLUMNS)))

        result = []
        changes: list[dict[str, Any]] = []

        for i, row in enumerate(rows):
            new_row = row.copy()
            if i == 0:
                # Header row
                for col_name, _ in selected:
                    new_row.append(col_name)
                    changes.append({"type": "add_column", "name": col_name})
            else:
                # Data rows
                for _, generator in selected:
                    new_row.append(generator(self._rng))
            result.append(new_row)

        col_names = [c[0] for c in selected]
        desc = f"Added columns: {', '.join(col_names)}"
        return result, desc, changes

    def _change_units(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], str, list[dict[str, Any]]]:
        """Multiply a numeric column by a unit conversion factor."""
        # Find numeric columns
        numeric_cols = []
        for col_idx in range(len(rows[0])):
            numeric_count = 0
            total = 0
            for row in rows[1:]:
                if col_idx < len(row) and row[col_idx]:
                    total += 1
                    try:
                        float(row[col_idx])
                        numeric_count += 1
                    except ValueError:
                        pass
            if total > 0 and numeric_count / total > 0.5:
                numeric_cols.append(col_idx)

        if not numeric_cols:
            return rows, "No numeric columns found", []

        target_col = self._rng.choice(numeric_cols)
        multiplier, unit_desc = self._rng.choice(self.UNIT_MULTIPLIERS)
        col_name = rows[0][target_col] if target_col < len(rows[0]) else f"col{target_col}"

        result = [rows[0]]  # Keep header
        changed_count = 0

        for row in rows[1:]:
            new_row = row.copy()
            if target_col < len(new_row):
                try:
                    val = float(new_row[target_col])
                    new_row[target_col] = f"{val * multiplier:.2f}"
                    changed_count += 1
                except ValueError:
                    pass
            result.append(new_row)

        changes: list[dict[str, Any]] = [
            {
                "type": "unit_change",
                "col_idx": target_col,
                "col_name": col_name,
                "multiplier": multiplier,
                "unit_conversion": unit_desc,
                "values_changed": changed_count,
            }
        ]
        desc = f"Multiplied '{col_name}' by {multiplier:.2f} ({unit_desc})"
        return result, desc, changes
