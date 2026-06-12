"""Data source declaration and modality detection (NEW in traxr).

``DataSource.from_path`` declares an artifact and detects its modality from
the extension (via the loader's category sets). I/O is lazy: the file content
is only read by :meth:`DataSource.load`.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from traxr.errors import InvalidArtifactError, ModalityMismatchError

from .loader import (
    DOCUMENT_EXTENSIONS,
    TABULAR_EXTENSIONS,
    LoadedArtifact,
    file_type_for,
    read_file,
)


class ModalityType(Enum):
    """v1 complete modality set."""

    TABULAR = "tabular"  # CSV / XLSX
    DOCUMENT = "document"  # PDF / TXT / MD


def detect_modality(path: str | Path) -> ModalityType:
    """Infer the modality from the file extension.

    Raises:
        UnsupportedModalityError: extension outside the v1 set.
    """
    file_type_for(path)  # raises UnsupportedModalityError for unknown extensions
    ext = Path(path).suffix.lower()
    if ext in TABULAR_EXTENSIONS:
        return ModalityType.TABULAR
    assert ext in DOCUMENT_EXTENSIONS
    return ModalityType.DOCUMENT


@dataclass(frozen=True)
class DataSource:
    """A declared input artifact: path + detected modality, lazy content."""

    path: Path
    modality: ModalityType
    file_type: str  # csv / xlsx / txt / md / pdf
    opts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        modality: ModalityType | None = None,
        **opts: Any,
    ) -> "DataSource":
        """Declare a source, detecting modality from the extension.

        Args:
            path: Path to the artifact (must exist).
            modality: Optional declared modality; checked against detection.
            **opts: Loader options forwarded to ``read_file`` (e.g. ``max_chars``).

        Raises:
            InvalidArtifactError: the file does not exist.
            UnsupportedModalityError: extension outside the v1 set.
            ModalityMismatchError: declared modality != detected modality.
        """
        p = Path(path)
        if not p.exists():
            raise InvalidArtifactError(f"File not found: {p}")
        detected = detect_modality(p)
        if modality is not None and modality is not detected:
            raise ModalityMismatchError(
                f"'{p.name}' was declared {modality.value} but its extension "
                f"indicates {detected.value}."
            )
        return cls(path=p, modality=detected, file_type=file_type_for(p), opts=opts)

    @property
    def source_id(self) -> str:
        """Stable identifier for seed derivation (the file's basename)."""
        return self.path.name

    def load(self) -> LoadedArtifact:
        """Read the artifact's content (the lazy I/O step)."""
        return read_file(self.path, **self.opts)
