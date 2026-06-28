"""The experiment runner: paired clean/perturbed runs over your agent + data.

``Experiment`` validates inputs, enumerates the perturbation matrix per file
(agent-kind-aware), runs the clean baseline (plus noise-floor re-runs),
applies each perturbation with an identically re-derived seed, runs the
agent in a fresh temp dir with original basenames, and turns each paired
trace into a :class:`~traxr.results.PairResult`.

``run(dry_run=True)`` prints the full execution plan with zero LLM calls and
zero agent invocations — token spend cannot be honestly estimated up front.
"""

import shutil
import tempfile
import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traxr.agents.builtin import BuiltinAgent
from traxr.agents.task import AgentRunner, Task, invoke_agent
from traxr.capture.context import CaptureSession
from traxr.data.loader import read_file
from traxr.data.sources import DataSource
from traxr.errors import (
    ControlledVariableError,
    EmptyTraceWarning,
    ExperimentConfigError,
    NonDeterminismWarning,
    PerturbationSkippedWarning,
)
from traxr.llm.protocol import LLMClient
from traxr.metrics.analyzer import ControlFlowChanges, TraceDivergenceAnalyzer
from traxr.metrics.cost import CostProxy
from traxr.metrics.manifest import PairMetrics, classify_manifestation, to_paper_group
from traxr.perturb.engine import PerturbationEngine
from traxr.perturb.matrix import (
    AgentKind,
    DeliveryPath,
    MatrixConfig,
    PermutationSpec,
    build_matrix,
)
from traxr.perturb.pdf_inplace import apply_pdf_inplace
from traxr.perturb.types import PerturbationResult, PerturbationType
from traxr.results import (
    STATUS_CRASHED,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_SKIPPED,
    ExperimentResults,
    PairResult,
)
from traxr.scoring import check_answer_match, normalize_answer
from traxr.trace.collector import TraceCollector

__all__ = ["Experiment", "ExperimentConfig", "ExperimentPlan", "PlannedRun"]

Scorer = Callable[[str | None, str | None], bool]


@dataclass(frozen=True)
class ExperimentConfig:
    """Knobs for :class:`Experiment` (frozen — the controlled-variable invariant).

    Attributes:
        perturbations: ``"all"`` or an explicit operator list.
        max_steps / max_tokens / enable_web_tools / enable_python_tool:
            Built-in-agent knobs (ignored for external agents).
        max_llm_calls_per_run: External-agent budget, enforced inside the
            Tier 0 wrapper — the only honest runaway bound for code we
            don't own.
        store_llm_content: Include raw LLM/tool content in trace payloads
            (hashes only by default; final answers are always stored raw).
        require_sequential: Raise instead of warn when concurrent LLM calls
            are detected during a run.
        scorer: ``(expected, actual) -> bool`` for ``task_success``.
        on_run_error: ``"record"`` keeps a CRASHED run record and continues;
            ``"raise"`` propagates the agent's exception.
        keep_artifacts: Keep the per-run temp dirs (perturbed file copies).
        noise_floor_runs: Clean re-runs measuring the nondeterminism floor.
            ``None`` means the agent-kind default: 1 for external agents,
            0 for the built-in agent.
        max_permutations: Matrix size cap (:class:`MatrixTooLargeError`).
    """

    perturbations: str | Sequence[PerturbationType] = "all"
    max_steps: int = 12
    max_tokens: int | None = 100_000
    max_llm_calls_per_run: int | None = 50
    store_llm_content: bool = False
    require_sequential: bool = False
    enable_web_tools: bool = False
    enable_python_tool: bool = True
    scorer: Scorer = check_answer_match
    on_run_error: str = "record"
    keep_artifacts: bool = False
    noise_floor_runs: int | None = None
    max_permutations: int = 64

    def __post_init__(self) -> None:
        if self.on_run_error not in ("record", "raise"):
            raise ExperimentConfigError(
                f'on_run_error must be "record" or "raise", got {self.on_run_error!r}'
            )
        if self.noise_floor_runs is not None and self.noise_floor_runs < 0:
            raise ExperimentConfigError("noise_floor_runs must be >= 0")


@dataclass(frozen=True)
class PlannedRun:
    """One planned agent invocation in a dry run."""

    label: str
    kind: str  # baseline | noise_floor | perturbation
    source_id: str | None = None
    perturbation: str | None = None
    delivery: str | None = None


@dataclass(frozen=True)
class ExperimentPlan:
    """The execution plan ``run(dry_run=True)`` returns (no agent ran)."""

    runs: tuple[PlannedRun, ...]
    agent_kind: str
    capture_tier: str
    budget: int | None
    noise_floor_runs: int

    def describe(self) -> str:
        perturbation_runs = sum(1 for r in self.runs if r.kind == "perturbation")
        floor = f" (+{self.noise_floor_runs} noise-floor run(s))" if self.noise_floor_runs else ""
        budget = (
            f"budget {self.budget} LLM calls/run" if self.budget else "no per-run LLM-call budget"
        )
        lines = [
            f"Traxr dry run: 1 baseline + {perturbation_runs} perturbation run(s){floor}, "
            f"{self.capture_tier} capture, {budget}.",
            "No LLM calls or agent invocations were made.",
        ]
        for run in self.runs:
            if run.kind == "perturbation":
                lines.append(f"  - {run.label} [{run.delivery}]")
            else:
                lines.append(f"  - {run.label}")
        return "\n".join(lines)


@dataclass
class _RunRecord:
    """Internal: one executed run."""

    label: str
    collector: TraceCollector
    answer: str | None = None
    status: str = STATUS_OK
    cost: CostProxy = field(default_factory=CostProxy)
    concurrent: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class Experiment:
    """A controlled-perturbation experiment over one agent and its data.

    Exactly one of ``agent`` (a stateless :data:`~traxr.agents.AgentRunner`
    callable, reused across runs), ``agent_factory`` (zero-arg factory called
    once per run — the fresh-state path), or ``llm`` (the built-in reference
    agent over your :class:`~traxr.llm.LLMClient`) must be given.
    """

    def __init__(
        self,
        files: str | Path | Sequence[str | Path],
        question: str,
        expected_answer: str | None = None,
        agent: AgentRunner | None = None,
        agent_factory: Callable[[], AgentRunner] | None = None,
        llm: LLMClient | None = None,
        seed: int = 42,
        config: ExperimentConfig | None = None,
    ):
        params = (("agent", agent), ("agent_factory", agent_factory), ("llm", llm))
        given = [name for name, value in params if value is not None]
        if len(given) != 1:
            raise ExperimentConfigError(
                "Exactly one of agent / agent_factory / llm must be given, "
                f"got {len(given)}: {given or 'none'}. Pass your callable as "
                "agent=, a per-run factory as agent_factory=, or an LLM "
                "client as llm= for the built-in reference agent."
            )
        if isinstance(files, (str, Path)):
            files = [files]
        if not files:
            raise ExperimentConfigError("At least one input file is required.")
        self.files = tuple(Path(f) for f in files)
        # Fail fast on duplicate basenames: the basename is each file's
        # source_id, which keys per-run staging, perturbation labels, and the
        # trace dict — so two inputs sharing a basename would silently overwrite
        # each other's staged file and trace. FUTURE: support duplicate
        # basenames by making source_id path-aware end-to-end (sources.py,
        # matrix labels, staging dirs, trace keys).
        basenames = [p.name for p in self.files]
        duplicate_basenames = sorted({n for n in basenames if basenames.count(n) > 1})
        if duplicate_basenames:
            raise ExperimentConfigError(
                "Input files must have unique basenames (the basename identifies "
                "each file in staging, perturbation labels, and traces). "
                f"Duplicates: {duplicate_basenames}."
            )
        self.question = question
        self.expected_answer = expected_answer
        self.seed = seed
        self.config = config or ExperimentConfig()
        self._agent = agent
        self._agent_factory = agent_factory
        self._llm = llm
        self.agent_kind = AgentKind.BUILTIN if llm is not None else AgentKind.EXTERNAL
        if self.agent_kind is AgentKind.BUILTIN and len(self.files) != 1:
            raise ExperimentConfigError(
                f"The built-in agent supports exactly one input file in v1, got {len(self.files)}."
            )
        # Fail fast on missing files / unsupported modalities.
        self.sources = tuple(DataSource.from_path(f) for f in self.files)

    # -- public API ----------------------------------------------------------

    def run(self, dry_run: bool = False) -> "ExperimentResults | ExperimentPlan":
        """Run the experiment (or, with ``dry_run=True``, just plan it)."""
        specs = self._build_specs()
        noise_floor_runs = self._noise_floor_runs()
        if dry_run:
            plan = self._build_plan(specs, noise_floor_runs)
            print(plan.describe())
            return plan

        snapshot = (self.seed, self.config)
        artifacts_root = Path(tempfile.mkdtemp(prefix="traxr-"))
        try:
            baseline = self._run_once("baseline", artifacts_root)
            floor, floor_records = self._measure_noise_floor(
                baseline, noise_floor_runs, artifacts_root
            )
            pairs: list[PairResult] = []
            perturbed_records: list[_RunRecord] = []
            for spec in specs:
                if (self.seed, self.config) != snapshot:
                    raise ControlledVariableError(
                        "Experiment seed/config changed between paired runs; "
                        "results would not be comparable."
                    )
                pair, record = self._run_pair(spec, baseline, floor, artifacts_root)
                pairs.append(pair)
                if record is not None:
                    perturbed_records.append(record)

            all_records = [baseline, *floor_records, *perturbed_records]
            results = ExperimentResults(
                pairs=pairs,
                traces={r.label: r.collector.to_dict() for r in all_records},
                answers={r.label: r.answer for r in all_records},
                fingerprint=self._fingerprint(specs, noise_floor_runs, all_records),
                noise_floor=floor,
                noise_floor_runs=noise_floor_runs,
            )
            if self.config.keep_artifacts:
                results.fingerprint["artifacts_dir"] = str(artifacts_root)
            return results
        finally:
            if not self.config.keep_artifacts:
                shutil.rmtree(artifacts_root, ignore_errors=True)

    # -- planning ------------------------------------------------------------

    def _build_specs(self) -> list[PermutationSpec]:
        matrix_config = MatrixConfig(
            agent_kind=self.agent_kind,
            base_seed=self.seed,
            max_permutations=self.config.max_permutations,
        )
        specs: list[PermutationSpec] = []
        for source in self.sources:
            specs.extend(build_matrix(source, matrix_config))
        if isinstance(self.config.perturbations, str) and self.config.perturbations != "all":
            raise ExperimentConfigError(
                f'perturbations must be "all" or a list of PerturbationType, '
                f"got {self.config.perturbations!r}"
            )
        if not isinstance(self.config.perturbations, str):
            wanted = set(self.config.perturbations)
            unknown = wanted - set(PerturbationType)
            if unknown:
                raise ExperimentConfigError(
                    f"Unknown perturbations: {sorted(str(u) for u in unknown)}"
                )
            specs = [s for s in specs if s.perturbation in wanted]
            if not specs:
                raise ExperimentConfigError(
                    "None of the requested perturbations apply to the given "
                    "files for this agent kind. Run `traxr operators` for the "
                    "catalog."
                )
        return specs

    def _noise_floor_runs(self) -> int:
        if self.config.noise_floor_runs is not None:
            return self.config.noise_floor_runs
        return 1 if self.agent_kind is AgentKind.EXTERNAL else 0

    def _capture_tier(self) -> str:
        return "tier0" if self.agent_kind is AgentKind.EXTERNAL else "builtin"

    def _build_plan(self, specs: list[PermutationSpec], noise_floor_runs: int) -> ExperimentPlan:
        runs = [PlannedRun(label="baseline", kind="baseline")]
        runs += [
            PlannedRun(label=f"noise_floor_{i + 1}", kind="noise_floor")
            for i in range(noise_floor_runs)
        ]
        runs += [
            PlannedRun(
                label=_spec_label(spec),
                kind="perturbation",
                source_id=spec.source_id,
                perturbation=spec.perturbation.value,
                delivery=spec.delivery.value,
            )
            for spec in specs
        ]
        budget = (
            self.config.max_llm_calls_per_run if self.agent_kind is AgentKind.EXTERNAL else None
        )
        return ExperimentPlan(
            runs=tuple(runs),
            agent_kind=self.agent_kind.value,
            capture_tier=self._capture_tier(),
            budget=budget,
            noise_floor_runs=noise_floor_runs,
        )

    # -- execution -----------------------------------------------------------

    def _stage_files(
        self, artifacts_root: Path, label: str, perturbed: tuple[Path, Path] | None = None
    ) -> list[Path]:
        """Copy inputs into a fresh run dir with their ORIGINAL basenames.

        ``perturbed`` optionally maps one source path to a pre-perturbed file
        that replaces its clean copy.
        """
        run_dir = artifacts_root / label.replace("/", "_").replace(":", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        for source in self.files:
            dst = run_dir / source.name
            if perturbed is not None and source == perturbed[0]:
                shutil.copy2(perturbed[1], dst)
            else:
                shutil.copy2(source, dst)
            staged.append(dst)
        return staged

    def _run_once(
        self,
        label: str,
        artifacts_root: Path,
        perturbed: tuple[Path, Path] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _RunRecord:
        staged = self._stage_files(artifacts_root, label, perturbed)
        record = _RunRecord(label=label, collector=TraceCollector(run_label=label))
        if self.agent_kind is AgentKind.EXTERNAL:
            self._run_external(record, staged, label)
        else:
            self._run_builtin(record, staged, label, metadata)
        # External runs always carry the harness-emitted final_answer; "empty"
        # means the agent's own LLM calls were invisible to Tier 0 capture.
        empty = (
            len(record.collector.get_events_by_type("llm_call")) == 0
            if self.agent_kind is AgentKind.EXTERNAL
            else record.collector.event_count == 0
        )
        if empty and record.status == STATUS_OK:
            record.status = STATUS_EMPTY
            message = (
                f"Run '{label}' produced no trace events — the agent's LLM "
                "calls were not captured (see the 'is my agent traceable?' docs)."
            )
            record.warnings.append(message)
            warnings.warn(message, EmptyTraceWarning, stacklevel=3)
        if record.concurrent and self.config.require_sequential:
            raise ControlledVariableError(
                f"Concurrent LLM calls detected in run '{label}' with "
                "require_sequential=True; event order is scheduling-dependent."
            )
        self._print_run_tokens(record)
        return record

    def _run_external(self, record: _RunRecord, staged: list[Path], label: str) -> None:
        runner = self._agent if self._agent is not None else self._agent_factory()  # type: ignore[misc]
        session = CaptureSession(
            record.collector,
            max_llm_calls_per_run=self.config.max_llm_calls_per_run,
            store_llm_content=self.config.store_llm_content,
        )
        task = Task(
            question=self.question,
            files=tuple(staged),
            run_label=label,
            metadata={},
        )
        try:
            record.answer = invoke_agent(runner, task, record.collector, session=session)
        except Exception as exc:
            if self.config.on_run_error == "raise":
                raise
            record.status = STATUS_CRASHED
            record.error = traceback.format_exc()
            record.warnings.append(f"agent crashed: {type(exc).__name__}: {exc}")
        record.concurrent = session.concurrent_detected
        record.cost = _cost_from_trace(record.collector)

    def _run_builtin(
        self,
        record: _RunRecord,
        staged: list[Path],
        label: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        assert self._llm is not None
        agent = BuiltinAgent(
            llm=self._llm,
            enable_web_tools=self.config.enable_web_tools,
            enable_python_tool=self.config.enable_python_tool,
            max_steps=self.config.max_steps,
            max_tokens=self.config.max_tokens,
            seed=self.seed,
        )
        try:
            record.answer = agent.run(
                tuple(staged),
                self.question,
                expected_answer=self.expected_answer,
                collector=record.collector,
                metadata=metadata,
                task_id=label,
            )
        except Exception as exc:
            if self.config.on_run_error == "raise":
                raise
            record.status = STATUS_CRASHED
            record.error = traceback.format_exc()
            record.warnings.append(f"agent crashed: {type(exc).__name__}: {exc}")
            record.collector.emit(
                "agent_error",
                step_num=0,
                agent_name="harness",
                payload={"exc_type": type(exc).__name__, "message": str(exc)},
            )
        if agent.last_cost is not None:
            record.cost = CostProxy(
                total_tokens=agent.last_cost.total_tokens,
                prompt_tokens=agent.last_cost.prompt_tokens,
                completion_tokens=agent.last_cost.completion_tokens,
                retrieval_calls=agent.last_cost.retrieval_calls,
                total_steps=agent.last_cost.total_steps,
            )

    def _measure_noise_floor(
        self, baseline: _RunRecord, noise_floor_runs: int, artifacts_root: Path
    ) -> tuple[float | None, list[_RunRecord]]:
        """Re-run the clean baseline; baseline-vs-itself d_norm IS the floor."""
        if noise_floor_runs == 0:
            return None, []
        analyzer = TraceDivergenceAnalyzer()
        floor = 0.0
        records = []
        for i in range(noise_floor_runs):
            rerun = self._run_once(f"noise_floor_{i + 1}", artifacts_root)
            records.append(rerun)
            report = analyzer.analyze(baseline.collector, rerun.collector, task_id="noise_floor")
            d_norm = report.edit_distance.normalized if report.edit_distance else 0.0
            if d_norm > 0:
                warnings.warn(
                    f"Clean re-run {i + 1} diverged from the baseline "
                    f"(d_norm={d_norm:.4f}) — the agent is nondeterministic; "
                    "pairs at or below this floor are flagged within_noise_floor.",
                    NonDeterminismWarning,
                    stacklevel=2,
                )
            floor = max(floor, d_norm)
        return floor, records

    def _run_pair(
        self,
        spec: PermutationSpec,
        baseline: _RunRecord,
        floor: float | None,
        artifacts_root: Path,
    ) -> tuple[PairResult, _RunRecord | None]:
        label = _spec_label(spec)
        source_path = next(p for p in self.files if p.name == spec.source_id)
        work_dir = artifacts_root / "perturbed_inputs"
        work_dir.mkdir(exist_ok=True)
        perturbed_file = work_dir / f"{label.replace('::', '_')}_{spec.source_id}"

        result, metadata = _deliver_perturbation(spec, source_path, perturbed_file)
        if not result.applied:
            message = (
                f"Perturbation {spec.perturbation.value} skipped for "
                f"'{spec.source_id}': {result.skip_reason}"
            )
            warnings.warn(message, PerturbationSkippedWarning, stacklevel=2)
            return (
                PairResult(
                    item_id=spec.source_id,
                    perturbation=spec.perturbation.value,
                    delivery=spec.delivery.value,
                    status_baseline=baseline.status,
                    status_perturbed=STATUS_SKIPPED,
                    warnings=[message],
                ),
                None,
            )

        perturbed_arg = (
            (source_path, perturbed_file) if spec.delivery is not DeliveryPath.INJECTION else None
        )
        record = self._run_once(label, artifacts_root, perturbed=perturbed_arg, metadata=metadata)
        return self._build_pair(spec, baseline, record, floor), record

    # -- pair metrics ----------------------------------------------------------

    def _build_pair(
        self,
        spec: PermutationSpec,
        baseline: _RunRecord,
        perturbed: _RunRecord,
        floor: float | None,
    ) -> PairResult:
        report = TraceDivergenceAnalyzer().analyze(
            baseline.collector, perturbed.collector, task_id=spec.source_id
        )
        d_norm = report.edit_distance.normalized if report.edit_distance else None
        control_flow = report.control_flow_changes or ControlFlowChanges()

        pair_warnings = [*baseline.warnings, *perturbed.warnings]
        # Reuse config.scorer for baseline-vs-perturbed comparison too, so a
        # semantic scorer (e.g. an LLM judge) treats "same answer" consistently
        # for both answer_changed and task_success rather than only the latter.
        try:
            answer_changed = not bool(self.config.scorer(baseline.answer, perturbed.answer))
        except Exception as exc:
            answer_changed = normalize_answer(baseline.answer) != normalize_answer(perturbed.answer)
            pair_warnings.append(f"scorer raised {type(exc).__name__} comparing baseline/perturbed: {exc}")

        manifestation = classify_manifestation(
            PairMetrics(
                answer_changed=answer_changed,
                perturbed_answer_is_null=not normalize_answer(perturbed.answer),
                early_termination=control_flow.early_termination,
                extended_execution=control_flow.extended_execution,
                reroutes=control_flow.reroutes,
                total_changes=control_flow.total_changes,
                edit_distance_normalized=d_norm,
            )
        )

        task_success: bool | None = None
        if self.expected_answer is not None and perturbed.status == STATUS_OK:
            try:
                task_success = bool(self.config.scorer(self.expected_answer, perturbed.answer))
            except Exception as exc:
                pair_warnings.append(f"scorer raised {type(exc).__name__}: {exc}")

        diverged = bool(d_norm) or control_flow.total_changes > 0
        # Recovery answers "did the answer survive the perturbation?" and is only
        # defined when the perturbation had an observable effect — a structural
        # divergence OR an answer change (the latter covers silent semantic
        # corruption: same trace, different answer). A pure no-op (not diverged
        # and answer unchanged) is left None so it is excluded from
        # recovery_rate()'s denominator, which the prior code wrongly diluted.
        recovery: bool | None = None
        if (
            baseline.status == STATUS_OK
            and perturbed.status == STATUS_OK
            and (diverged or answer_changed)
        ):
            recovery = not answer_changed

        token_overhead: float | None = None
        if baseline.cost.total_tokens > 0 and perturbed.cost.total_tokens > 0:
            token_overhead = perturbed.cost.total_tokens / baseline.cost.total_tokens

        return PairResult(
            item_id=spec.source_id,
            perturbation=spec.perturbation.value,
            delivery=spec.delivery.value,
            status_baseline=baseline.status,
            status_perturbed=perturbed.status,
            d_norm=d_norm,
            t_star=report.first_divergence_step,
            t_star_norm=report.divergence_normalized_position,
            divergence_type=report.first_divergence_type,
            control_flow_changes=control_flow.to_dict(),
            task_success=task_success,
            answer_changed=answer_changed,
            recovery=recovery,
            token_overhead=token_overhead,
            manifestation=manifestation,
            paper_group=to_paper_group(manifestation),
            within_noise_floor=(None if floor is None or d_norm is None else d_norm <= floor),
            order_nondeterministic=baseline.concurrent or perturbed.concurrent,
            warnings=pair_warnings,
        )

    # -- misc ------------------------------------------------------------------

    def _print_run_tokens(self, record: _RunRecord) -> None:
        if record.cost.total_tokens > 0:
            print(f"[traxr] {record.label}: {record.cost.total_tokens} tokens")

    def _fingerprint(
        self,
        specs: list[PermutationSpec],
        noise_floor_runs: int,
        records: list[_RunRecord],
    ) -> dict[str, Any]:
        from traxr import __version__  # local import: traxr/__init__ imports this module

        models = sorted(
            {
                str(e.payload.get("model"))
                for r in records
                for e in r.collector.get_events_by_type("llm_call")
                if e.payload.get("model")
            }
        )
        return {
            "agent_kind": self.agent_kind.value,
            "capture_tier": self._capture_tier(),
            "seed": self.seed,
            "files": [p.name for p in self.files],
            "question": self.question,
            "expected_answer": self.expected_answer,
            "perturbations": [s.perturbation.value for s in specs],
            "scorer": getattr(self.config.scorer, "__qualname__", repr(self.config.scorer)),
            "max_llm_calls_per_run": self.config.max_llm_calls_per_run,
            "noise_floor_runs": noise_floor_runs,
            "llm_client": type(self._llm).__name__ if self._llm is not None else None,
            "model_ids": models,
            "environment": {"traxr_version": __version__},
        }


def _spec_label(spec: PermutationSpec) -> str:
    return f"{spec.source_id}::{spec.perturbation.value}"


def _cost_from_trace(collector: TraceCollector) -> CostProxy:
    """Token/step cost for an external run, from the captured usage.

    ``total_steps`` is the number of captured ``llm_call`` events, not
    ``session.llm_call_count`` — the latter is only incremented by the Tier 0
    wrapper's ``begin_llm_call`` and stays 0 for Tier 1 (LangGraph) runs, where
    Tier 0 is suppressed. Counting the emitted events is correct for both tiers.
    """
    llm_calls = collector.get_events_by_type("llm_call")
    cost = CostProxy(total_steps=len(llm_calls))
    for event in llm_calls:
        usage = event.payload.get("usage")
        if usage:
            cost.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    return cost


def _deliver_perturbation(
    spec: PermutationSpec, source_path: Path, dst_path: Path
) -> tuple[PerturbationResult, dict[str, Any]]:
    """Produce the perturbed artifact (or injection payload) for one spec.

    Returns the engine result plus the ``Task``/built-in metadata to pass to
    the run (the injection delivery path carries the perturbed text in
    ``metadata["injected_pdf_content"]`` instead of touching the file).
    """
    if spec.delivery is DeliveryPath.PDF_INPLACE:
        return apply_pdf_inplace(source_path, dst_path, spec.perturbation, seed=spec.seed), {}

    engine = PerturbationEngine(seed=spec.seed)
    if spec.delivery is DeliveryPath.INJECTION:
        # PDF text must come from the extractor, not raw file bytes.
        extracted = read_file(source_path).content
        result = engine.apply(
            content=extracted,
            file_type="pdf",
            perturbation=spec.perturbation,
            file_name=source_path.name,
        )
        metadata = {"injected_pdf_content": result.corrupted_content} if result.applied else {}
        return result, metadata

    result = engine.apply_from_file(str(source_path), spec.perturbation)
    if result.applied:
        if dst_path.suffix.lower() in (".xlsx", ".xls"):
            engine.write_excel(result.corrupted_content, str(dst_path))
        else:
            # Match the UTF-8 read path (perturb/engine.py); some perturbations
            # inject non-ASCII (e.g. ENCODING_ERROR's U+FFFD), which would raise
            # UnicodeEncodeError or write mojibake under a non-UTF-8 locale.
            dst_path.write_text(result.corrupted_content, encoding="utf-8")
    return result, {}
