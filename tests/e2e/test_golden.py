"""Built-in-agent golden: fixed CSV + stub + fixed seed → byte-stable JSON.

Two assertions: (1) two fresh in-process runs serialize identically (the
canonical form excludes timestamps), and (2) the output matches the committed
snapshot with the volatile ``fingerprint.environment`` block normalized.
Never regenerate-to-match without understanding the diff.
"""

import json
import warnings
from pathlib import Path

from traxr.experiment import Experiment, ExperimentConfig
from traxr.llm import DeterministicLLMStub
from traxr.perturb.types import PerturbationType

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "goldens" / "builtin_stub.json"


def run_golden_experiment():
    experiment = Experiment(
        files=Path(__file__).resolve().parents[1] / "fixtures" / "sample.csv",
        question="What is the total revenue?",
        expected_answer="42",
        llm=DeterministicLLMStub(scenario="identity", final_answer="42"),
        seed=1234,
        config=ExperimentConfig(
            perturbations=[PerturbationType.COLUMN_SWAP, PerturbationType.NULL_CONTENT]
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = experiment.run()
    return results.to_json()


def normalize_environment(text: str) -> str:
    data = json.loads(text)
    data["fingerprint"]["environment"] = "NORMALIZED"
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def test_builtin_golden_run_twice_identical():
    assert run_golden_experiment() == run_golden_experiment()


def test_builtin_golden_matches_committed_snapshot():
    assert normalize_environment(run_golden_experiment()) == normalize_environment(
        SNAPSHOT.read_text()
    )
