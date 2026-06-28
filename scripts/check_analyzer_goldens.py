"""Analyzer golden check: the analyzer must reproduce the committed goldens.

Loads each fixture in tests/fixtures/analyzer_goldens/ (inputs AND
ground-truth outputs committed once during initial development), runs the
traxr analyzer on the same inputs, and asserts the summary AND full detail
dicts (including aligned_pairs) match EXACTLY.

This is the analyzer's behavior-preservation proof. If it fails, fix the
code — never the fixtures.
"""

import json
from pathlib import Path
from typing import Any

from traxr.metrics.analyzer import TraceDivergenceAnalyzer
from traxr.trace.collector import TraceCollector

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "analyzer_goldens"


def _canonical(obj: Any) -> Any:
    """Normalize via a JSON round-trip so tuple/list and key-order differences vanish."""
    return json.loads(json.dumps(obj, sort_keys=True))


def _diff_paths(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Human-readable list of leaf differences between two canonical JSON values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                diffs.append(f"{path}.{key}: unexpected key (got {actual[key]!r})")
            elif key not in actual:
                diffs.append(f"{path}.{key}: missing (expected {expected[key]!r})")
            else:
                diffs.extend(_diff_paths(expected[key], actual[key], f"{path}.{key}"))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        diffs = []
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(actual)} != expected {len(expected)}")
        # Length mismatches are reported above; diff the common prefix.
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            diffs.extend(_diff_paths(e, a, f"{path}[{i}]"))
        return diffs
    if expected != actual:
        return [f"{path}: {actual!r} != expected {expected!r}"]
    return []


def run_case(fixture_path: Path) -> list[str]:
    """Run one fixture case; return a list of mismatch descriptions (empty = pass)."""
    data = json.loads(fixture_path.read_text())
    clean = TraceCollector.from_dict(data["input"]["clean_trace"])
    perturbed = TraceCollector.from_dict(data["input"]["perturbed_trace"])

    report = TraceDivergenceAnalyzer().analyze(clean, perturbed, task_id=data["input"]["task_id"])

    mismatches: list[str] = []
    for section, got in (
        ("summary", report.to_summary_dict()),
        ("detail", report.to_detail_dict()),
    ):
        expected = _canonical(data["output"][section])
        actual = _canonical(got)
        if expected != actual:
            diffs = _diff_paths(expected, actual, path=section)
            mismatches.extend(diffs[:20])
            if len(diffs) > 20:
                mismatches.append(f"{section}: ... and {len(diffs) - 20} more differences")

    ed = report.edit_distance
    assert ed is not None  # analyze() always sets it
    print(
        f"  {data['case']:<15} d={ed.distance} d_norm={ed.normalized:.4f} "
        f"t*={report.first_divergence_step} type={report.first_divergence_type} "
        f"answer_changed={report.answer_changed} -> " + ("OK" if not mismatches else "MISMATCH")
    )
    return mismatches


def main() -> int:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    print(f"analyzer-goldens: {len(manifest['cases'])} committed cases")

    case_paths = [p for p in sorted(FIXTURE_DIR.glob("*.json")) if p.name != "manifest.json"]
    expected_cases = set(manifest["cases"])
    found_cases = {p.stem for p in case_paths}
    if found_cases != expected_cases:
        print(
            f"analyzer-goldens: FAIL — fixture files {found_cases} "
            f"!= manifest cases {expected_cases}"
        )
        return 1

    failures: dict[str, list[str]] = {}
    for path in case_paths:
        mismatches = run_case(path)
        if mismatches:
            failures[path.stem] = mismatches

    if failures:
        print(
            f"\nanalyzer-goldens: FAIL — {len(failures)}/{len(case_paths)} "
            f"cases diverge from the goldens:"
        )
        for case, case_mismatches in failures.items():
            print(f"  case {case}:")
            for m in case_mismatches:
                print(f"    {m}")
        return 1

    print(
        f"analyzer-goldens: PASS — all {len(case_paths)} cases reproduce the golden outputs exactly"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
