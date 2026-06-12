"""Data layer: artifact loading, inspection, and source declaration."""

from .loader import FileInspection, LoadedArtifact, inspect_file, read_file
from .sources import DataSource, ModalityType, detect_modality

__all__ = [
    "DataSource",
    "FileInspection",
    "LoadedArtifact",
    "ModalityType",
    "detect_modality",
    "inspect_file",
    "read_file",
]
