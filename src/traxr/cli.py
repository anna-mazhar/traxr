"""``traxr`` command-line interface: run / operators / selfcheck.

``traxr run`` wires an agent (``--agent module:callable`` for external
agents, ``--model``/``--base-url`` for the built-in reference agent) to data
files and runs the experiment; ``traxr operators`` prints the perturbation
catalog per agent kind; ``traxr selfcheck`` runs the offline end-to-end
check.
"""

import argparse
import importlib
import os
import sys
from typing import Any

from traxr.errors import TraxrError
from traxr.perturb.types import PerturbationType

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "operators":
            return _cmd_operators()
        assert args.command == "selfcheck"
        from traxr.selfcheck import selfcheck

        selfcheck()
        return 0
    except TraxrError as exc:
        print(f"traxr: error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traxr",
        description="Controlled-perturbation experiments for your own agent and your own data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a perturbation experiment.")
    run.add_argument(
        "--agent",
        help="External agent as module:callable (a (Task) -> str function).",
    )
    run.add_argument("--model", help="Built-in agent: model id on an OpenAI-compatible endpoint.")
    run.add_argument("--base-url", help="Built-in agent: endpoint base URL.")
    # Hidden: the no-key demo/test path (also honored via TRAXR_STUB=1).
    run.add_argument("--stub", action="store_true", help=argparse.SUPPRESS)
    run.add_argument(
        "--file",
        action="append",
        required=True,
        dest="files",
        help="Input data file (repeatable).",
    )
    run.add_argument("--question", required=True, help="The task question.")
    run.add_argument("--expected-answer", help="Reference answer for task_success scoring.")
    run.add_argument(
        "--perturbations",
        help=f"Comma-separated operator names (default: all). "
        f"Known: {', '.join(sorted(p.value for p in PerturbationType))}",
    )
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", help="Write results JSON here.")
    run.add_argument(
        "--report",
        help="Write a human-readable report here (format inferred from the "
        "extension: .html or .md).",
    )
    run.add_argument("--dry-run", action="store_true", help="Print the plan; run nothing.")
    run.add_argument("--keep-artifacts", action="store_true", help="Keep perturbed file copies.")
    run.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        metavar="N",
        help="External agent (--agent) budget: hard cap on LLM calls per run, "
        "enforced inside the capture wrapper (default 50). Use 0 or a negative "
        "value to disable.",
    )
    run.add_argument(
        "--max-retries",
        type=int,
        default=2,
        metavar="N",
        help="Built-in agent (--model) only: how many times the OpenAI SDK "
        "retries a transient failure before raising (default 2; 0 disables).",
    )

    sub.add_parser("operators", help="List the perturbation catalog per agent kind.")
    sub.add_parser("selfcheck", help="Run the offline end-to-end self-check.")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from traxr.experiment import Experiment, ExperimentConfig

    agent_kwargs = _resolve_agent(args)

    config_kwargs: dict[str, Any] = {
        "keep_artifacts": args.keep_artifacts,
    }
    if args.max_llm_calls is not None:
        # <= 0 disables the budget (ExperimentConfig treats None as "no cap").
        config_kwargs["max_llm_calls_per_run"] = (
            args.max_llm_calls if args.max_llm_calls > 0 else None
        )
    if args.perturbations:
        config_kwargs["perturbations"] = [
            _parse_perturbation(name) for name in args.perturbations.split(",")
        ]
    experiment = Experiment(
        files=args.files,
        question=args.question,
        expected_answer=args.expected_answer,
        seed=args.seed,
        config=ExperimentConfig(**config_kwargs),
        **agent_kwargs,
    )
    outcome = experiment.run(dry_run=args.dry_run)
    if args.dry_run:
        return 0  # run() already printed the plan

    from traxr.results import ExperimentResults

    assert isinstance(outcome, ExperimentResults)
    print(outcome.summary())
    if args.out:
        outcome.to_json(args.out)
        print(f"results written to {args.out}")
    if args.report:
        fmt = "html" if args.report.lower().endswith((".html", ".htm")) else "md"
        from pathlib import Path

        Path(args.report).write_text(outcome.to_report(fmt), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


def _resolve_agent(args: argparse.Namespace) -> dict[str, Any]:
    use_stub = args.stub or os.environ.get("TRAXR_STUB") == "1"
    chosen = [
        name
        for name, on in (("--agent", args.agent), ("--model", args.model), ("--stub", use_stub))
        if on
    ]
    if len(chosen) != 1:
        raise TraxrError(
            "Pass exactly one of --agent module:callable (external agent) or "
            "--model [--base-url] (built-in agent over an OpenAI-compatible "
            f"endpoint); got {chosen or 'neither'}."
        )
    if args.agent:
        return {"agent": _load_agent(args.agent)}
    if use_stub:
        from traxr.llm import DeterministicLLMStub

        return {"llm": DeterministicLLMStub()}
    from traxr.llm import OpenAICompatibleClient

    return {
        "llm": OpenAICompatibleClient(
            model=args.model, base_url=args.base_url, max_retries=args.max_retries
        )
    }


def _load_agent(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise TraxrError(f"--agent must look like module:callable, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TraxrError(f"Could not import agent module {module_name!r}: {exc}") from exc
    try:
        agent = getattr(module, attr)
    except AttributeError:
        raise TraxrError(f"Module {module_name!r} has no attribute {attr!r}") from None
    if not callable(agent):
        raise TraxrError(f"--agent {spec!r} resolved to a non-callable {type(agent).__name__}")
    return agent


def _parse_perturbation(name: str) -> PerturbationType:
    try:
        return PerturbationType(name.strip())
    except ValueError:
        raise TraxrError(
            f"Unknown perturbation {name.strip()!r}. Known: "
            f"{', '.join(sorted(p.value for p in PerturbationType))}"
        ) from None


def _cmd_operators() -> int:
    from traxr.perturb.matrix import (
        PDF_INJECTION_ONLY_OPERATORS,
        TABULAR_OPERATORS,
        TEXT_OPERATORS,
    )
    from traxr.perturb.pdf_inplace import PDF_INPLACE_OPERATORS

    def section(title: str, ops: tuple[PerturbationType, ...], note: str = "") -> None:
        print(title + (f"  ({note})" if note else ""))
        for op in ops:
            print(f"  {op.value}")
        print()

    null = (PerturbationType.NULL_CONTENT,)
    print("Perturbation catalog (v1)\n")
    section("tabular (csv/xlsx) — any agent", TABULAR_OPERATORS + null)
    section("text (txt/md) — any agent", TEXT_OPERATORS + null)
    section("pdf — any agent", tuple(PDF_INPLACE_OPERATORS), "surgical in-place edits")
    section(
        "pdf — built-in agent only",
        PDF_INJECTION_ONLY_OPERATORS,
        "whole-text-flow, delivered via content injection",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
