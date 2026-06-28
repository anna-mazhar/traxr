"""End-to-end self-check: a tiny offline experiment with expected metrics.

``python -m traxr.selfcheck`` (or ``traxr.selfcheck()``) runs a complete
stub-driven experiment on a generated CSV — zero network, zero keys — and
verifies the results have the expected shape. Degrades gracefully to a
metrics-only check when the built-in agent's optional dependencies (pandas)
are not installed, so the wheel smoke test passes on a bare install.
"""

import sys
import tempfile
import warnings
from pathlib import Path

__all__ = ["selfcheck"]

_CSV = "region,revenue\nnorth,10\nsouth,32\n"


def _check_metrics_core() -> None:
    """Analyzer + manifestation on synthetic traces (stdlib-only path)."""
    from traxr.metrics.analyzer import TraceDivergenceAnalyzer
    from traxr.metrics.manifest import PairMetrics, classify_manifestation
    from traxr.trace.collector import TraceCollector

    clean = TraceCollector(run_label="baseline")
    perturbed = TraceCollector(run_label="perturbed")
    for step, tool in enumerate(["load", "analyze", "answer"], start=1):
        clean.emit("llm_call", step, "agent", {"model": "m", "finish_reason": "stop"})
        clean.emit("tool_request", step, "agent", {"tool_name": tool})
    for step, tool in enumerate(["load", "retry", "answer"], start=1):
        perturbed.emit("llm_call", step, "agent", {"model": "m", "finish_reason": "stop"})
        perturbed.emit("tool_request", step, "agent", {"tool_name": tool})

    report = TraceDivergenceAnalyzer().analyze(clean, perturbed, task_id="selfcheck")
    assert report.edit_distance is not None
    d_norm = report.edit_distance.normalized
    assert 0.0 < d_norm < 1.0, f"unexpected d_norm {d_norm}"
    category = classify_manifestation(PairMetrics(edit_distance_normalized=d_norm))
    assert category == "structural_divergence_recovered", category
    print(f"selfcheck: metrics core OK (d_norm={d_norm:.4f}, {category})")


def _check_stub_experiment() -> None:
    """Full built-in-agent experiment under the deterministic stub."""
    from traxr.experiment import Experiment, ExperimentConfig
    from traxr.llm import DeterministicLLMStub
    from traxr.perturb.types import PerturbationType

    with tempfile.TemporaryDirectory(prefix="traxr-selfcheck-") as tmp:
        csv_path = Path(tmp) / "sales.csv"
        csv_path.write_text(_CSV)
        experiment = Experiment(
            files=csv_path,
            question="What is the total revenue?",
            expected_answer="42",
            llm=DeterministicLLMStub(scenario="identity", final_answer="42"),
            config=ExperimentConfig(perturbations=[PerturbationType.COLUMN_SWAP]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = experiment.run()
    from traxr.results import ExperimentResults

    assert isinstance(results, ExperimentResults)
    assert len(results.pairs) == 1, f"expected 1 pair, got {len(results.pairs)}"
    pair = results.pairs[0]
    assert pair.status_perturbed == "ok", pair.status_perturbed
    assert pair.d_norm is not None and 0.0 <= pair.d_norm <= 1.0
    assert pair.task_success is True, "stub identity scenario must score True"
    assert pair.manifestation is not None
    results.to_json(include_traces=False)  # serialization round-trip
    print(
        f"selfcheck: stub experiment OK (d_norm={pair.d_norm:.4f}, "
        f"manifestation={pair.manifestation}, task_success={pair.task_success})"
    )


def selfcheck() -> None:
    """Run the self-check; raises (non-zero exit under ``-m``) on failure."""
    _check_metrics_core()
    try:
        import pandas  # noqa: F401
    except ImportError:
        print(
            "selfcheck: built-in agent extras not installed — skipped the "
            'stub experiment (install "traxr[document]" + pandas to enable). PASS'
        )
        return
    _check_stub_experiment()
    print("selfcheck: PASS")


if __name__ == "__main__":
    try:
        selfcheck()
    except Exception as exc:  # pragma: no cover - exercised via exit-code tests
        print(f"selfcheck: FAIL — {exc}", file=sys.stderr)
        sys.exit(1)
