"""External-agent golden: fixed CSV + mock-transport fixture agent + fixed
seed → byte-stable JSON (same contract as the built-in golden)."""

import json
import warnings
from pathlib import Path

from _openai_mock import MODEL, MockOpenAIServer, completion, tool_call

from traxr.capture import instrument
from traxr.experiment import Experiment, ExperimentConfig
from traxr.perturb.types import PerturbationType

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "goldens" / "external_mock.json"


def make_agent():
    """Two deterministic calls per run, incl. a tool round-trip."""
    server = MockOpenAIServer(
        [
            completion("", tool_calls=[tool_call("call_g1", "lookup", '{"q": "total"}')]),
            completion("the answer is 42"),
        ],
        cycle=True,
    )
    client = instrument(server.client())

    def agent(task):
        messages = [{"role": "user", "content": task.question}]
        first = client.chat.completions.create(model=MODEL, messages=messages)
        tc = first.choices[0].message.tool_calls[0]
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": "42"})
        second = client.chat.completions.create(model=MODEL, messages=messages)
        return second.choices[0].message.content or ""

    return agent


def run_golden_experiment():
    experiment = Experiment(
        files=Path(__file__).resolve().parents[1] / "fixtures" / "sample.csv",
        question="What is the total revenue?",
        expected_answer="the answer is 42",
        agent=make_agent(),
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


def test_external_golden_run_twice_identical():
    assert run_golden_experiment() == run_golden_experiment()


def test_external_golden_matches_committed_snapshot():
    assert normalize_environment(run_golden_experiment()) == normalize_environment(
        SNAPSHOT.read_text()
    )
