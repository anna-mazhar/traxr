"""Built-in-agent experiments under the deterministic stub (category 10)."""

import warnings

from traxr.experiment import Experiment, ExperimentConfig
from traxr.llm import DeterministicLLMStub
from traxr.perturb.types import PerturbationType
from traxr.results import STATUS_OK


def run_quiet(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.run()


def test_builtin_stub_identity_end_to_end(fixtures_dir):
    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="What is the total?",
            expected_answer="42",
            llm=DeterministicLLMStub(scenario="identity", final_answer="42"),
            config=ExperimentConfig(
                perturbations=[PerturbationType.COLUMN_SWAP, PerturbationType.NULL_CONTENT]
            ),
        )
    )

    assert results.noise_floor is None  # built-in default: 0 floor runs
    assert results.noise_floor_runs == 0
    assert results.fingerprint["agent_kind"] == "builtin"
    assert results.fingerprint["capture_tier"] == "builtin"
    assert results.fingerprint["llm_client"] == "DeterministicLLMStub"
    assert "deterministic built-in path" in results.summary()
    assert len(results.pairs) == 2
    for pair in results.pairs:
        assert pair.status_baseline == STATUS_OK
        assert pair.status_perturbed == STATUS_OK
        assert pair.d_norm is not None and 0.0 <= pair.d_norm <= 1.0
        assert pair.task_success is True  # identity scenario answers "42" everywhere
        assert pair.within_noise_floor is None  # floor unmeasured
        assert pair.control_flow_changes is not None
    # The stub identity scenario is insensitive to column order but the
    # null_content pair must diverge (the agent sees an empty table).
    by_op = {p.perturbation: p for p in results.pairs}
    assert by_op["null_content"].d_norm > 0.0
    # Built-in traces carry stub token counts -> overhead is measurable.
    assert by_op["null_content"].token_overhead is not None


def test_builtin_wrong_answer_scenario_fails_scoring(fixtures_dir):
    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            expected_answer="42",
            llm=DeterministicLLMStub(scenario="wrong_answer", final_answer="42", wrong_answer="99"),
            config=ExperimentConfig(perturbations=[PerturbationType.COLUMN_SWAP]),
        )
    )
    (pair,) = results.pairs
    # Both runs give the same wrong answer: no flip, but task_success False.
    assert pair.task_success is False
    assert pair.answer_changed is False


def test_builtin_pdf_injection_delivery_end_to_end(fixtures_dir):
    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.pdf",
            question="What does the document say?",
            llm=DeterministicLLMStub(scenario="identity", final_answer="it says things"),
            config=ExperimentConfig(perturbations=[PerturbationType.OCR_NOISE]),
        )
    )
    (pair,) = results.pairs
    assert pair.delivery == "injection"
    assert pair.status_perturbed == STATUS_OK


def test_builtin_noise_floor_opt_in(fixtures_dir):
    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            llm=DeterministicLLMStub(scenario="identity", final_answer="42"),
            config=ExperimentConfig(
                perturbations=[PerturbationType.COLUMN_SWAP], noise_floor_runs=1
            ),
        )
    )
    # The stub is deterministic: the measured floor must be exactly zero.
    assert results.noise_floor == 0.0
    assert results.noise_floor_runs == 1
    (pair,) = results.pairs
    assert pair.within_noise_floor is (pair.d_norm == 0.0)


def test_builtin_crash_recorded(fixtures_dir, monkeypatch):
    from traxr.agents.builtin import BuiltinAgent

    def explode(self, *args, **kwargs):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(BuiltinAgent, "run", explode)
    experiment = Experiment(
        files=fixtures_dir / "sample.csv",
        question="q",
        llm=DeterministicLLMStub(),
        config=ExperimentConfig(perturbations=[PerturbationType.COLUMN_SWAP]),
    )
    results = run_quiet(experiment)
    (pair,) = results.pairs
    assert pair.status_baseline == "crashed"
    assert pair.status_perturbed == "crashed"
    assert results.traces["baseline"]["events"][-1]["event_type"] == "agent_error"


def test_keep_artifacts_preserves_run_dirs(fixtures_dir, tmp_path):
    import shutil
    from pathlib import Path

    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            llm=DeterministicLLMStub(scenario="identity", final_answer="42"),
            config=ExperimentConfig(
                perturbations=[PerturbationType.COLUMN_SWAP], keep_artifacts=True
            ),
        )
    )
    artifacts_dir = Path(results.fingerprint["artifacts_dir"])
    try:
        staged = list(artifacts_dir.rglob("sample.csv"))
        # Baseline copy + perturbed-run copy + the perturbed input itself.
        assert len(staged) >= 2
        labels = {p.parent.name for p in staged}
        assert "baseline" in labels
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
