"""Shared pytest fixtures for the Traxr test suite."""

import sys
from pathlib import Path

import pytest

from traxr.trace import registry

# Shared test doubles (e.g. _openai_mock) importable from every test dir.
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    """Directory of committed tiny real artifacts."""
    return FIXTURES_DIR


@pytest.fixture
def clean_registry():
    """Snapshot and restore the event-type registry around a test.

    Lets tests register custom types or trigger one-time unknown-type
    warnings without leaking state into other tests.
    """
    saved_registry = dict(registry._REGISTRY)
    saved_structural = set(registry.STRUCTURAL_DIVERGENCE_TYPES)
    saved_warned = set(registry._warned_unknown_types)
    try:
        yield registry
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved_registry)
        registry.STRUCTURAL_DIVERGENCE_TYPES.clear()
        registry.STRUCTURAL_DIVERGENCE_TYPES.update(saved_structural)
        registry._warned_unknown_types.clear()
        registry._warned_unknown_types.update(saved_warned)
