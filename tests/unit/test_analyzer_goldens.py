"""Analyzer golden gate as a unit test: each case must reproduce its golden.

Mirrors `make analyzer-goldens` (scripts/check_analyzer_goldens.py) so the
behavior-preservation proof also runs inside `make test` / coverage. Fixtures
are read-only ground truth — if a case fails, fix the code, never the
fixture.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_analyzer_goldens.py"
_spec = importlib.util.spec_from_file_location("golden_check", _SCRIPT)
assert _spec is not None and _spec.loader is not None
golden_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden_check)

_MANIFEST = json.loads((golden_check.FIXTURE_DIR / "manifest.json").read_text())


def test_manifest_matches_fixture_files():
    found = {p.stem for p in golden_check.FIXTURE_DIR.glob("*.json")} - {"manifest"}
    assert found == set(_MANIFEST["cases"])


@pytest.mark.parametrize("case", sorted(_MANIFEST["cases"]))
def test_golden_case(case):
    mismatches = golden_check.run_case(golden_check.FIXTURE_DIR / f"{case}.json")
    assert mismatches == [], f"case {case} diverges from its golden output: {mismatches}"
