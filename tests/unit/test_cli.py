"""CLI smoke tests: run / operators / argument validation (category 14)."""

import json
import sys
import warnings

import pytest

from traxr.cli import main


@pytest.fixture()
def csv_file(fixtures_dir):
    return str(fixtures_dir / "sample.csv")


def run_cli(*argv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main(list(argv))


def test_operators_lists_catalog(capsys):
    assert run_cli("operators") == 0
    out = capsys.readouterr().out
    assert "column_swap" in out
    assert "built-in agent only" in out


def test_run_stub_writes_results_json(csv_file, tmp_path, capsys):
    out_file = tmp_path / "results.json"
    code = run_cli(
        "run",
        "--stub",
        "--file",
        csv_file,
        "--question",
        "What is the total?",
        "--expected-answer",
        "42",
        "--perturbations",
        "column_swap",
        "--out",
        str(out_file),
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Traxr experiment summary" in printed
    data = json.loads(out_file.read_text())
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["perturbation"] == "column_swap"


def test_run_stub_writes_report(csv_file, tmp_path, capsys):
    common = (
        "run", "--stub", "--file", csv_file, "--question", "q",
        "--perturbations", "column_swap",
    )  # fmt: skip
    html = tmp_path / "report.html"
    assert run_cli(*common, "--report", str(html)) == 0
    assert "report written to" in capsys.readouterr().out
    assert html.read_text().startswith("<!DOCTYPE html>")

    md = tmp_path / "report.md"
    assert run_cli(*common, "--report", str(md)) == 0
    capsys.readouterr()
    assert md.read_text().startswith("# Traxr experiment report")


def test_run_stub_env_var(csv_file, capsys, monkeypatch):
    monkeypatch.setenv("TRAXR_STUB", "1")
    assert run_cli("run", "--file", csv_file, "--question", "q", "--dry-run") == 0
    assert "No LLM calls or agent invocations were made." in capsys.readouterr().out


def test_run_external_agent_from_module(csv_file, tmp_path, capsys, monkeypatch):
    agent_module = tmp_path / "fixture_cli_agent.py"
    agent_module.write_text("def agent(task):\n    return 'cli answer'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    code = run_cli(
        "run",
        "--agent",
        "fixture_cli_agent:agent",
        "--file",
        csv_file,
        "--question",
        "q",
        "--perturbations",
        "column_swap",
    )
    assert code == 0
    assert "Traxr experiment summary" in capsys.readouterr().out


def test_run_requires_exactly_one_agent_source(csv_file, capsys):
    assert run_cli("run", "--file", csv_file, "--question", "q") == 2
    assert "exactly one" in capsys.readouterr().err.lower()
    code = run_cli(
        "run", "--agent", "m:f", "--model", "gpt-x", "--file", csv_file, "--question", "q"
    )
    assert code == 2


def test_run_bad_agent_specs(csv_file, capsys):
    assert run_cli("run", "--agent", "noseparator", "--file", csv_file, "--question", "q") == 2
    assert "module:callable" in capsys.readouterr().err
    assert (
        run_cli(
            "run", "--agent", "definitely_missing_mod:fn", "--file", csv_file, "--question", "q"
        )
        == 2
    )
    assert "Could not import" in capsys.readouterr().err


def test_run_unknown_perturbation(csv_file, capsys):
    code = run_cli(
        "run",
        "--stub",
        "--file",
        csv_file,
        "--question",
        "q",
        "--perturbations",
        "not_an_op",
    )
    assert code == 2
    assert "Unknown perturbation" in capsys.readouterr().err


def test_console_entry_point_configured():
    from pathlib import Path

    # Plain string check: tomllib only exists on Python >= 3.11.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert 'traxr = "traxr.cli:main"' in pyproject.read_text()


def test_module_invocation_smoke(csv_file):
    # `python -m traxr.cli` path (argv parsing through real sys.argv plumbing).
    argv = sys.argv
    try:
        sys.argv = ["traxr", "operators"]
        assert main(None) == 0
    finally:
        sys.argv = argv
