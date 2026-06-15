"""Experiment results: per-pair metrics, aggregates, and exporters.

``PairResult`` is one clean-vs-perturbed comparison; ``ExperimentResults``
holds all pairs plus the raw traces and environment fingerprint.
Serialization is canonical — wall-clock timestamps are excluded — so
stub/mock-transport experiments produce byte-stable JSON snapshots.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from traxr.errors import OptionalDependencyError

__all__ = [
    "STATUS_CRASHED",
    "STATUS_EMPTY",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "ExperimentResults",
    "PairResult",
]

#: Run statuses recorded per side of a pair.
STATUS_OK = "ok"
STATUS_CRASHED = "crashed"
STATUS_EMPTY = "empty"
STATUS_SKIPPED = "skipped"  # perturbation not applicable; agent never ran


@dataclass
class PairResult:
    """Metrics for one clean-vs-perturbed pair."""

    item_id: str
    perturbation: str
    delivery: str
    status_baseline: str = STATUS_OK
    status_perturbed: str = STATUS_OK
    d_norm: float | None = None
    t_star: int | None = None
    t_star_norm: float | None = None
    divergence_type: str | None = None
    control_flow_changes: dict[str, Any] | None = None
    task_success: bool | None = None
    answer_changed: bool | None = None
    recovery: bool | None = None
    token_overhead: float | None = None
    manifestation: str | None = None
    paper_group: str | None = None
    within_noise_floor: bool | None = None
    order_nondeterministic: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        """Whether this pair produced metrics (the perturbed run happened)."""
        return self.status_perturbed != STATUS_SKIPPED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Everything one ``Experiment.run()`` produced.

    Attributes:
        pairs: One :class:`PairResult` per permutation.
        traces: ``run_label -> serialized trace`` (collector ``to_dict()``).
        answers: ``run_label -> raw final answer`` (stored raw by design —
            scoring and ``answer_changed`` need them; see the security docs).
        fingerprint: Environment/config fingerprint for reproducibility.
        noise_floor: Baseline-vs-itself ``d_norm`` (None when unmeasured).
        noise_floor_runs: How many clean re-runs measured the floor.
    """

    pairs: list[PairResult]
    traces: dict[str, dict[str, Any]]
    answers: dict[str, str | None]
    fingerprint: dict[str, Any]
    noise_floor: float | None = None
    noise_floor_runs: int = 0

    # -- aggregates ---------------------------------------------------------

    def manifestation_prevalence(self) -> dict[str, float]:
        """Fraction of scored pairs per fine manifestation category."""
        scored = [p for p in self.pairs if p.scored and p.manifestation]
        if not scored:
            return {}
        prevalence: dict[str, float] = {}
        for pair in scored:
            assert pair.manifestation is not None
            prevalence[pair.manifestation] = prevalence.get(pair.manifestation, 0) + 1
        return {k: v / len(scored) for k, v in sorted(prevalence.items())}

    def divergence_summary(self) -> dict[str, float | int | None]:
        """Count / mean / max of ``d_norm`` and mean ``t*_norm`` over measured pairs."""
        d_norms = [p.d_norm for p in self.pairs if p.d_norm is not None]
        t_norms = [p.t_star_norm for p in self.pairs if p.t_star_norm is not None]
        return {
            "pairs_measured": len(d_norms),
            "mean_d_norm": sum(d_norms) / len(d_norms) if d_norms else None,
            "max_d_norm": max(d_norms) if d_norms else None,
            "mean_t_star_norm": sum(t_norms) / len(t_norms) if t_norms else None,
        }

    def recovery_rate(self) -> float | None:
        """Fraction of diverged pairs whose answer survived (recovery=True)."""
        diverged = [p.recovery for p in self.pairs if p.recovery is not None]
        if not diverged:
            return None
        return sum(diverged) / len(diverged)

    def token_overhead_summary(self) -> dict[str, float | None]:
        """Mean / max token-inflation ratio over pairs with usage data."""
        ratios = [p.token_overhead for p in self.pairs if p.token_overhead is not None]
        return {
            "mean": sum(ratios) / len(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        }

    # -- exporters ----------------------------------------------------------

    def to_dict(self, *, include_traces: bool = True) -> dict[str, Any]:
        """Canonical dict form (timestamps excluded; see :meth:`to_json`)."""
        payload: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "noise_floor": self.noise_floor,
            "noise_floor_runs": self.noise_floor_runs,
            "pairs": [p.to_dict() for p in self.pairs],
            "answers": self.answers,
            "aggregates": {
                "manifestation_prevalence": self.manifestation_prevalence(),
                "divergence_summary": self.divergence_summary(),
                "recovery_rate": self.recovery_rate(),
                "token_overhead_summary": self.token_overhead_summary(),
            },
        }
        if include_traces:
            payload["traces"] = {
                label: _strip_timestamps(trace) for label, trace in self.traces.items()
            }
        return payload

    def to_json(self, path: Any = None, *, include_traces: bool = True) -> str:
        """Canonical JSON: sorted keys, timestamps excluded — byte-stable
        for deterministic (stub / mock-transport) experiments.

        Args:
            path: Optional file path to also write the JSON to.
            include_traces: Include the full serialized traces.
        """
        text = json.dumps(self.to_dict(include_traces=include_traces), sort_keys=True, indent=2)
        text += "\n"
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_dataframe(self) -> Any:
        """The pairs as a pandas DataFrame (needs the ``[pandas]`` extra)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise OptionalDependencyError(
                'to_dataframe() needs pandas. Install it with: pip install "traxr[pandas]"'
            ) from exc
        return pd.DataFrame([p.to_dict() for p in self.pairs])

    def to_report(self, fmt: str = "md") -> str:
        """Human-readable report (``"md"`` or ``"html"``)."""
        if fmt not in ("md", "html"):
            raise ValueError(f"to_report() supports 'md' and 'html', got {fmt!r}")
        md = self._markdown_report()
        if fmt == "md":
            return md
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Traxr report</title></head><body><pre>\n"
            f"{md}"
            "\n</pre></body></html>\n"
        )

    def summary(self) -> str:
        """Compact printable summary (with a prominent noise-floor caveat)."""
        scored = [p for p in self.pairs if p.scored]
        skipped = len(self.pairs) - len(scored)
        lines = [
            "Traxr experiment summary",
            f"  agent kind: {self.fingerprint.get('agent_kind', '?')} "
            f"(capture: {self.fingerprint.get('capture_tier', '?')})",
            f"  pairs: {len(self.pairs)} ({len(scored)} scored, {skipped} skipped)",
        ]
        if self.noise_floor is not None:
            lines.append(
                f"  noise floor (d_norm): {self.noise_floor:.4f} "
                f"over {self.noise_floor_runs} clean re-run(s)"
            )
        elif self.fingerprint.get("agent_kind") == "external":
            lines.append(
                "  noise floor: UNMEASURED — divergence below the (unknown) "
                "sampling-noise floor may be reported as contamination. Set "
                "noise_floor_runs >= 1 for external agents."
            )
        else:
            lines.append("  noise floor: unmeasured (deterministic built-in path)")
        div = self.divergence_summary()
        if div["pairs_measured"]:
            assert div["mean_d_norm"] is not None and div["max_d_norm"] is not None
            lines.append(
                f"  d_norm: mean {div['mean_d_norm']:.4f}, max {div['max_d_norm']:.4f} "
                f"over {div['pairs_measured']} pair(s)"
            )
        recovery = self.recovery_rate()
        if recovery is not None:
            lines.append(f"  recovery rate: {recovery:.0%}")
        tokens = self.token_overhead_summary()
        if tokens["mean"] is not None:
            lines.append(f"  token overhead: mean {tokens['mean']:.2f}x")
        for category, fraction in self.manifestation_prevalence().items():
            lines.append(f"    {category}: {fraction:.0%}")
        if any(p.order_nondeterministic for p in self.pairs):
            lines.append(
                "  ⚠ concurrent LLM calls detected in at least one run — event "
                "order is scheduling-dependent (order_nondeterministic pairs)."
            )
        return "\n".join(lines)

    def _markdown_report(self) -> str:
        lines = ["# Traxr experiment report", "", "## Fingerprint", ""]
        for key in sorted(self.fingerprint):
            lines.append(f"- **{key}**: {self.fingerprint[key]}")
        lines += ["", "## Summary", "", "```", self.summary(), "```", "", "## Pairs", ""]
        lines.append(
            "| item | perturbation | status | d_norm | t* | manifestation "
            "| answer_changed | token_overhead |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in self.pairs:
            d_norm = f"{p.d_norm:.4f}" if p.d_norm is not None else "—"
            overhead = f"{p.token_overhead:.2f}x" if p.token_overhead is not None else "—"
            lines.append(
                f"| {p.item_id} | {p.perturbation} | {p.status_perturbed} | {d_norm} "
                f"| {p.t_star if p.t_star is not None else '—'} | {p.manifestation or '—'} "
                f"| {p.answer_changed if p.answer_changed is not None else '—'} | {overhead} |"
            )
        lines.append("")
        return "\n".join(lines)


def _strip_timestamps(trace: dict[str, Any]) -> dict[str, Any]:
    """Trace dict without per-event wall-clock timestamps (canonical form)."""
    out = dict(trace)
    out["events"] = [
        {k: v for k, v in event.items() if k != "timestamp"} for event in trace.get("events", [])
    ]
    return out
