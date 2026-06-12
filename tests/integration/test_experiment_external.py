"""External-agent experiments end to end (categories 10 + 13), fully offline."""

import warnings

import pytest
from _openai_mock import MODEL, MockOpenAIServer, completion

from traxr.capture import instrument
from traxr.errors import EmptyTraceWarning, PerturbationSkippedWarning
from traxr.experiment import Experiment, ExperimentConfig
from traxr.perturb.types import PerturbationType
from traxr.results import STATUS_CRASHED, STATUS_EMPTY, STATUS_OK

TWO_OPS = ExperimentConfig(
    perturbations=[PerturbationType.COLUMN_SWAP, PerturbationType.NULL_CONTENT]
)


def make_canned_agent():
    """One LLM call per run; canned answer — trace identical across runs."""
    server = MockOpenAIServer([completion("the answer is 42")], cycle=True)
    client = instrument(server.client())

    def agent(task):
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": task.question}]
        )
        return response.choices[0].message.content or ""

    return agent, server


def make_content_sensitive_agent():
    """One LLM call per run; the ANSWER derives from the file content."""
    server = MockOpenAIServer([completion("ack")], cycle=True)
    client = instrument(server.client())

    def agent(task):
        client.chat.completions.create(model=MODEL, messages=[])
        return f"rows={len(task.files[0].read_text().splitlines())}"

    return agent, server


def run_quiet(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.run()


def test_external_experiment_end_to_end(fixtures_dir):
    agent, server = make_canned_agent()
    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="What is the total?",
            expected_answer="the answer is 42",
            agent=agent,
            config=TWO_OPS,
        )
    )

    # External default: one noise-floor run; deterministic agent -> floor 0.
    assert results.noise_floor == 0.0
    assert results.noise_floor_runs == 1
    assert set(results.answers) == {
        "baseline",
        "noise_floor_1",
        "sample.csv::column_swap",
        "sample.csv::null_content",
    }
    assert len(results.pairs) == 2
    for pair in results.pairs:
        assert pair.status_perturbed == STATUS_OK
        assert pair.d_norm == 0.0
        assert pair.within_noise_floor is True
        assert pair.manifestation == "no_observable_effect"
        assert pair.answer_changed is False
        assert pair.task_success is True
        assert pair.token_overhead == pytest.approx(1.0)
        assert pair.order_nondeterministic is False
    assert results.fingerprint["model_ids"] == [MODEL]
    assert results.fingerprint["agent_kind"] == "external"
    # 1 baseline + 1 floor + 2 perturbed runs, one LLM call each.
    assert len(server.requests) == 4


def test_content_sensitive_agent_classified_silent_corruption(fixtures_dir):
    agent, _ = make_content_sensitive_agent()
    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="How many rows?",
            agent=agent,
            config=ExperimentConfig(perturbations=[PerturbationType.NULL_CONTENT]),
        )
    )
    (pair,) = results.pairs
    # Same trace, different answer: the silent-corruption signature.
    assert pair.d_norm == 0.0
    assert pair.answer_changed is True
    assert pair.manifestation == "silent_semantic_corruption"
    assert pair.task_success is None  # no expected_answer given
    assert pair.recovery is False


def test_agent_factory_called_once_per_run(fixtures_dir):
    calls = []

    def factory():
        agent, _ = make_canned_agent()
        calls.append(1)
        return agent

    run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            agent_factory=factory,
            config=TWO_OPS,
        )
    )
    assert len(calls) == 4  # baseline + noise floor + 2 perturbations


def test_run_twice_is_reproducible(fixtures_dir):
    def build():
        agent, _ = make_canned_agent()
        return Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            agent=agent,
            seed=11,
            config=TWO_OPS,
        )

    key = lambda r: [  # noqa: E731
        (p.perturbation, p.d_norm, p.answer_changed, p.manifestation) for p in r.pairs
    ]
    assert key(run_quiet(build())) == key(run_quiet(build()))


def test_crash_recorded_and_raise_mode(fixtures_dir):
    server = MockOpenAIServer([completion("ok")], cycle=True)
    client = instrument(server.client())

    def fragile_agent(task):
        client.chat.completions.create(model=MODEL, messages=[])
        if not task.files[0].read_text():
            raise RuntimeError("empty input file")
        return "fine"

    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            agent=fragile_agent,
            config=ExperimentConfig(perturbations=[PerturbationType.NULL_CONTENT]),
        )
    )
    (pair,) = results.pairs
    assert pair.status_baseline == STATUS_OK
    assert pair.status_perturbed == STATUS_CRASHED
    assert any("RuntimeError" in w for w in pair.warnings)
    perturbed_trace = results.traces["sample.csv::null_content"]
    assert perturbed_trace["events"][-1]["event_type"] == "agent_error"
    assert pair.recovery is None

    raising = Experiment(
        files=fixtures_dir / "sample.csv",
        question="q",
        agent=fragile_agent,
        config=ExperimentConfig(
            perturbations=[PerturbationType.NULL_CONTENT], on_run_error="raise"
        ),
    )
    with pytest.raises(RuntimeError, match="empty input file"):
        run_quiet(raising)


def test_budget_exceeded_mid_run_records_partial_trace(fixtures_dir):
    server = MockOpenAIServer([completion("a")], cycle=True)
    client = instrument(server.client())

    def chatty_agent(task):
        client.chat.completions.create(model=MODEL, messages=[])
        client.chat.completions.create(model=MODEL, messages=[])
        return "done"

    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            agent=chatty_agent,
            config=ExperimentConfig(
                perturbations=[PerturbationType.COLUMN_SWAP],
                max_llm_calls_per_run=1,
                noise_floor_runs=0,
            ),
        )
    )
    (pair,) = results.pairs
    assert pair.status_baseline == STATUS_CRASHED  # baseline hits the budget too
    assert any("RunBudgetExceeded" in w for w in pair.warnings)
    baseline_events = results.traces["baseline"]["events"]
    # Partial trace: the one allowed llm_call plus the recorded agent_error.
    assert [e["event_type"] for e in baseline_events] == ["llm_call", "agent_error"]


def test_zero_event_agent_flagged_empty(fixtures_dir):
    def untraceable_agent(task):
        return "no llm involved"

    experiment = Experiment(
        files=fixtures_dir / "sample.csv",
        question="q",
        agent=untraceable_agent,
        config=ExperimentConfig(perturbations=[PerturbationType.COLUMN_SWAP], noise_floor_runs=0),
    )
    with pytest.warns(EmptyTraceWarning):
        results = experiment.run()
    (pair,) = results.pairs
    assert pair.status_baseline == STATUS_EMPTY
    assert pair.status_perturbed == STATUS_EMPTY


def test_scorer_error_recorded_not_raised(fixtures_dir):
    agent, _ = make_canned_agent()

    def broken_scorer(expected, actual):
        raise ValueError("scorer bug")

    results = run_quiet(
        Experiment(
            files=fixtures_dir / "sample.csv",
            question="q",
            expected_answer="42",
            agent=agent,
            config=ExperimentConfig(
                perturbations=[PerturbationType.COLUMN_SWAP],
                scorer=broken_scorer,
                noise_floor_runs=0,
            ),
        )
    )
    (pair,) = results.pairs
    assert pair.task_success is None
    assert any("scorer raised ValueError" in w for w in pair.warnings)


def test_perturbation_skip_recorded(fixtures_dir, monkeypatch):
    from traxr import experiment as experiment_module
    from traxr.perturb.types import PerturbationResult

    def always_skip(spec, source_path, dst_path):
        return (
            PerturbationResult(
                original_content="x",
                corrupted_content="x",
                perturbation_type=spec.perturbation,
                description="",
                applied=False,
                skip_reason="not applicable to this fixture",
            ),
            {},
        )

    monkeypatch.setattr(experiment_module, "_deliver_perturbation", always_skip)
    agent, server = make_canned_agent()
    experiment = Experiment(
        files=fixtures_dir / "sample.csv",
        question="q",
        agent=agent,
        config=ExperimentConfig(perturbations=[PerturbationType.COLUMN_SWAP], noise_floor_runs=0),
    )
    with pytest.warns(PerturbationSkippedWarning, match="not applicable"):
        results = experiment.run()
    (pair,) = results.pairs
    assert pair.status_perturbed == "skipped"
    assert pair.d_norm is None
    assert len(server.requests) == 1  # only the baseline ran
