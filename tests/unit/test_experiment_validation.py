"""Experiment fail-fast validation + dry-run planning (categories 10/13)."""

import pytest

from traxr.errors import (
    ExperimentConfigError,
    InvalidArtifactError,
    MatrixTooLargeError,
)
from traxr.experiment import Experiment, ExperimentConfig, ExperimentPlan
from traxr.llm import DeterministicLLMStub
from traxr.perturb.types import PerturbationType


def dummy_agent(task):
    return "answer"


@pytest.fixture()
def csv_file(fixtures_dir):
    return fixtures_dir / "sample.csv"


def test_exactly_one_agent_param_required(csv_file):
    with pytest.raises(ExperimentConfigError, match="Exactly one"):
        Experiment(files=csv_file, question="q")
    with pytest.raises(ExperimentConfigError, match="Exactly one"):
        Experiment(files=csv_file, question="q", agent=dummy_agent, llm=DeterministicLLMStub())


def test_builtin_agent_rejects_multiple_files(csv_file, fixtures_dir):
    with pytest.raises(ExperimentConfigError, match="exactly one input file"):
        Experiment(
            files=[csv_file, fixtures_dir / "sample.txt"],
            question="q",
            llm=DeterministicLLMStub(),
        )


def test_missing_file_fails_fast(tmp_path):
    with pytest.raises(InvalidArtifactError, match="not found"):
        Experiment(files=tmp_path / "missing.csv", question="q", agent=dummy_agent)


def test_no_files_fails_fast():
    with pytest.raises(ExperimentConfigError, match="At least one"):
        Experiment(files=[], question="q", agent=dummy_agent)


def test_duplicate_basenames_rejected(tmp_path):
    # Two inputs sharing a basename would collide in staging/labels/traces;
    # fail fast (before the existence check) with a clear error.
    with pytest.raises(ExperimentConfigError, match="unique basenames"):
        Experiment(
            files=[tmp_path / "a" / "report.csv", tmp_path / "b" / "report.csv"],
            question="q",
            agent=dummy_agent,
        )


def test_invalid_on_run_error():
    with pytest.raises(ExperimentConfigError, match="on_run_error"):
        ExperimentConfig(on_run_error="ignore")


def test_negative_noise_floor_runs():
    with pytest.raises(ExperimentConfigError, match="noise_floor_runs"):
        ExperimentConfig(noise_floor_runs=-1)


def test_perturbations_string_other_than_all(csv_file):
    exp = Experiment(
        files=csv_file,
        question="q",
        agent=dummy_agent,
        config=ExperimentConfig(perturbations="column_swap"),
    )
    with pytest.raises(ExperimentConfigError, match='"all" or a list'):
        exp.run(dry_run=True)


def test_unknown_perturbations_rejected(csv_file):
    exp = Experiment(
        files=csv_file,
        question="q",
        agent=dummy_agent,
        config=ExperimentConfig(perturbations=["definitely_not_an_operator"]),
    )
    with pytest.raises(ExperimentConfigError, match="Unknown perturbations"):
        exp.run(dry_run=True)


def test_inapplicable_perturbations_rejected(csv_file):
    exp = Experiment(
        files=csv_file,
        question="q",
        agent=dummy_agent,
        config=ExperimentConfig(perturbations=[PerturbationType.OCR_NOISE]),  # text-only op
    )
    with pytest.raises(ExperimentConfigError, match="traxr operators"):
        exp.run(dry_run=True)


def test_matrix_cap(csv_file):
    exp = Experiment(
        files=csv_file,
        question="q",
        agent=dummy_agent,
        config=ExperimentConfig(max_permutations=2),
    )
    with pytest.raises(MatrixTooLargeError, match="cap of 2"):
        exp.run(dry_run=True)


def test_dry_run_plan_counts_and_defaults(csv_file, capsys):
    exp = Experiment(files=csv_file, question="q", agent=dummy_agent)
    plan = exp.run(dry_run=True)
    assert isinstance(plan, ExperimentPlan)
    # 6 tabular operators + null_content; external default noise floor = 1.
    assert sum(1 for r in plan.runs if r.kind == "perturbation") == 7
    assert plan.noise_floor_runs == 1
    assert plan.capture_tier == "tier0"
    assert plan.budget == 50
    printed = capsys.readouterr().out
    assert "1 baseline + 7 perturbation run(s) (+1 noise-floor run(s))" in printed
    assert "No LLM calls or agent invocations were made." in printed


def test_dry_run_builtin_defaults(csv_file):
    exp = Experiment(files=csv_file, question="q", llm=DeterministicLLMStub())
    plan = exp.run(dry_run=True)
    assert isinstance(plan, ExperimentPlan)
    assert plan.noise_floor_runs == 0  # built-in default
    assert plan.capture_tier == "builtin"
    assert plan.budget is None
