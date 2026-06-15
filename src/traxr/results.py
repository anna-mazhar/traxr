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
        """Human-readable report (``"md"`` or ``"html"``).

        ``"md"`` is a plain markdown document (good for terminals and PRs).
        ``"html"`` is a single self-contained file — inline styles, no scripts,
        no external assets — that embeds the :mod:`traxr.viz` figures when
        matplotlib is installed and degrades gracefully when it is not.
        """
        if fmt not in ("md", "html"):
            raise ValueError(f"to_report() supports 'md' and 'html', got {fmt!r}")
        if fmt == "md":
            return self._markdown_report()
        return self._html_report()

    # -- report building blocks --------------------------------------------

    def _t_star_cell(self, pair: "PairResult") -> str:
        """First divergence anchored to the perturbed trace: ``step N · type``."""
        if pair.t_star is None:
            return "—"
        trace = self.traces.get(f"{pair.item_id}::{pair.perturbation}")
        event_type = _event_type_at_step(trace, pair.t_star) if trace else None
        return f"step {pair.t_star} · {event_type}" if event_type else f"step {pair.t_star}"

    def _noise_floor_status(self) -> str:
        """One-line noise-floor status (shared by summary and report).

        Keeps the prominent UNMEASURED caveat for external agents and reports
        how many measured pairs fall within a measured floor.
        """
        if self.noise_floor is not None:
            measured = [p for p in self.pairs if p.d_norm is not None]
            within = [p for p in measured if p.within_noise_floor]
            line = f"{self.noise_floor:.4f} over {self.noise_floor_runs} clean re-run(s)"
            if measured:
                line += f"; {len(within)} of {len(measured)} measured pair(s) within floor"
            return line
        if self.fingerprint.get("agent_kind") == "external":
            return (
                "UNMEASURED — divergence below the (unknown) sampling-noise floor "
                "may be reported as contamination. Set noise_floor_runs >= 1 for "
                "external agents."
            )
        return "unmeasured (deterministic built-in path)"

    def summary(self) -> str:
        """Compact printable summary.

        Reads top-down as a diagnosis: how many pairs ran, *how* perturbations
        manifested, how far the traces diverged (against the noise floor),
        whether the answer survived, and what it cost.
        """
        scored = [p for p in self.pairs if p.scored]
        skipped = len(self.pairs) - len(scored)
        lines = [
            "Traxr experiment summary",
            f"  agent kind: {self.fingerprint.get('agent_kind', '?')} "
            f"(capture: {self.fingerprint.get('capture_tier', '?')})",
            f"  pairs: {len(self.pairs)} ({len(scored)} scored, {skipped} skipped)",
        ]
        prevalence = self.manifestation_prevalence()
        if prevalence:
            lines.append("  manifestations:")
            for category, fraction in prevalence.items():
                lines.append(f"    {fraction:>4.0%}  {category}")
        div = self.divergence_summary()
        if div["pairs_measured"]:
            assert div["mean_d_norm"] is not None and div["max_d_norm"] is not None
            lines.append(
                f"  d_norm: mean {div['mean_d_norm']:.4f}, max {div['max_d_norm']:.4f} "
                f"over {div['pairs_measured']} pair(s)"
            )
        lines.append(f"  noise floor (d_norm): {self._noise_floor_status()}")
        recovery = self.recovery_rate()
        if recovery is not None:
            lines.append(f"  recovery rate: {recovery:.0%} of diverged pairs kept their answer")
        tokens = self.token_overhead_summary()
        if tokens["mean"] is not None:
            lines.append(f"  token overhead: mean {tokens['mean']:.2f}x")
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
            "| item | perturbation | status | d_norm | <=floor | t* | manifestation "
            "| answer_changed | recovery | control flow | tokens |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in self.pairs:
            d_norm = f"{p.d_norm:.4f}" if p.d_norm is not None else "—"
            overhead = f"{p.token_overhead:.2f}x" if p.token_overhead is not None else "—"
            lines.append(
                f"| {p.item_id} | {p.perturbation} | {p.status_perturbed} | {d_norm} "
                f"| {_noise_floor_mark(p)} | {self._t_star_cell(p)} | {p.manifestation or '—'} "
                f"| {_tristate(p.answer_changed)} | {_tristate(p.recovery)} "
                f"| {_control_flow_summary(p.control_flow_changes)} | {overhead} |"
            )
        lines.append("")
        lines += self._manifestation_legend_md()
        return "\n".join(lines)

    def _manifestation_legend_md(self) -> list[str]:
        from traxr.metrics.manifest import MANIFESTATION_DESCRIPTIONS

        prevalence = self.manifestation_prevalence()
        if not prevalence:
            return []
        out = ["## Manifestations", ""]
        for category, fraction in prevalence.items():
            desc = MANIFESTATION_DESCRIPTIONS.get(category, "")
            out.append(f"- **{category}** ({fraction:.0%}) — {desc}")
        out.append("")
        return out

    def _html_report(self) -> str:
        from html import escape

        from traxr.metrics.manifest import MANIFESTATION_DESCRIPTIONS

        div = self.divergence_summary()
        scored = [p for p in self.pairs if p.scored]
        skipped = len(self.pairs) - len(scored)

        overview: list[tuple[str, str]] = [
            (
                "Agent",
                f"{self.fingerprint.get('agent_kind', '?')} "
                f"(capture: {self.fingerprint.get('capture_tier', '?')})",
            ),
            ("Pairs", f"{len(self.pairs)} ({len(scored)} scored, {skipped} skipped)"),
        ]
        if div["pairs_measured"]:
            overview.append(
                (
                    "Trace divergence (d_norm)",
                    f"mean {div['mean_d_norm']:.4f}, max {div['max_d_norm']:.4f} "
                    f"over {div['pairs_measured']} pair(s)",
                )
            )
        overview.append(("Noise floor", self._noise_floor_status()))
        recovery = self.recovery_rate()
        if recovery is not None:
            overview.append(
                ("Recovery rate", f"{recovery:.0%} of diverged pairs kept their answer")
            )
        tokens = self.token_overhead_summary()
        if tokens["mean"] is not None:
            overview.append(("Token overhead", f"mean {tokens['mean']:.2f}×"))

        parts = [_HTML_HEAD]
        parts.append("<h1>Traxr experiment report</h1>")

        # Overview
        parts.append("<section><h2>Overview</h2><dl class='overview'>")
        for term, value in overview:
            parts.append(f"<dt>{escape(term)}</dt><dd>{escape(value)}</dd>")
        parts.append("</dl></section>")

        # Per-pair table
        headers = [
            "item",
            "perturbation",
            "status",
            "d_norm",
            "≤floor",
            "first divergence (t*)",
            "manifestation",
            "answer changed",
            "recovery",
            "control flow",
            "tokens",
        ]
        parts.append("<section><h2>Per-pair results</h2><table><thead><tr>")
        parts += [f"<th>{escape(h)}</th>" for h in headers]
        parts.append("</tr></thead><tbody>")
        for p in self.pairs:
            cells = [
                p.item_id,
                p.perturbation,
                p.status_perturbed,
                f"{p.d_norm:.4f}" if p.d_norm is not None else "—",
                _noise_floor_mark(p),
                self._t_star_cell(p),
                p.manifestation or "—",
                _tristate(p.answer_changed),
                _tristate(p.recovery),
                _control_flow_summary(p.control_flow_changes),
                f"{p.token_overhead:.2f}×" if p.token_overhead is not None else "—",
            ]
            parts.append("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>")
        parts.append("</tbody></table></section>")

        # Manifestation legend (only categories that occurred)
        prevalence = self.manifestation_prevalence()
        if prevalence:
            parts.append("<section><h2>Manifestations</h2><dl class='legend'>")
            for category, fraction in prevalence.items():
                desc = MANIFESTATION_DESCRIPTIONS.get(category, "")
                parts.append(
                    f"<dt><span class='pct'>{fraction:.0%}</span> {escape(category)}</dt>"
                    f"<dd>{escape(desc)}</dd>"
                )
            parts.append("</dl></section>")

        # Embedded figures (graceful when matplotlib / [viz] is absent)
        figures = _render_figures_svg(self)
        if figures:
            parts.append("<section><h2>Figures</h2><div class='figures'>")
            parts += [f"<figure>{svg}</figure>" for svg in figures]
            parts.append("</div></section>")

        # Fingerprint (collapsed)
        parts.append("<section><details><summary>Fingerprint</summary><dl class='fingerprint'>")
        for key in sorted(self.fingerprint):
            parts.append(f"<dt>{escape(key)}</dt><dd>{escape(str(self.fingerprint[key]))}</dd>")
        parts.append("</dl></details></section>")

        parts.append("</body></html>\n")
        return "\n".join(parts)


_HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traxr report</title>
<style>
:root { color-scheme: light; }
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem;
  color: #1a1a1a; background: #fff; }
h1 { font-size: 1.5rem; margin-bottom: 1.5rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 0.75rem;
  border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
dl.overview, dl.fingerprint { display: grid;
  grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; margin: 0; }
dl.overview dt, dl.fingerprint dt { font-weight: 600; color: #555; }
dl.overview dd, dl.fingerprint dd { margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem;
  border-bottom: 1px solid #eee; vertical-align: top; }
th { border-bottom: 2px solid #ccc; white-space: nowrap; }
dl.legend dt { font-weight: 600; margin-top: 0.5rem; }
dl.legend dd { margin: 0 0 0.25rem; color: #555; }
.pct { display: inline-block; min-width: 3ch; color: #1f77b4; font-variant-numeric: tabular-nums; }
.figures { display: flex; flex-wrap: wrap; gap: 1rem; }
.figures figure { margin: 0; flex: 1 1 280px; }
.figures svg { max-width: 100%; height: auto; }
details summary { cursor: pointer; font-weight: 600; color: #555; }
</style></head><body>"""


def _tristate(value: bool | None) -> str:
    """Render an optional boolean as ``yes`` / ``no`` / ``—``."""
    if value is None:
        return "—"
    return "yes" if value else "no"


def _noise_floor_mark(pair: "PairResult") -> str:
    """Per-pair noise-floor marker: ``✓`` within floor, ``·`` above, ``—`` unknown."""
    if pair.within_noise_floor is None:
        return "—"
    return "✓" if pair.within_noise_floor else "·"


def _control_flow_summary(cfc: dict[str, Any] | None) -> str:
    """Compact, neutral rendering of the control-flow deltas (``—`` if none)."""
    if not cfc:
        return "—"
    parts: list[str] = []

    def count(key: str, label: str) -> None:
        n = cfc.get(key) or 0
        if n:
            parts.append(f"{n}× {label}")

    count("reroutes", "reroute")
    if cfc.get("tool_failures_introduced"):
        parts.append("tool ok→fail")
    if cfc.get("tool_failures_avoided"):
        parts.append("tool fail→ok")
    count("tool_calls_added", "+tool call")
    count("tool_calls_removed", "−tool call")
    count("extra_routing_cycles", "+routing")
    count("missing_routing_cycles", "−routing")
    if cfc.get("early_termination"):
        parts.append("early stop")
    if cfc.get("extended_execution"):
        parts.append("ran longer")
    return ", ".join(parts) if parts else "—"


def _event_type_at_step(trace: dict[str, Any], step: int) -> str | None:
    """The ``event_type`` of the first event at ``step_num == step`` (or None)."""
    for event in trace.get("events", []):
        if event.get("step_num") == step:
            etype = event.get("event_type")
            return etype if isinstance(etype, str) else None
    return None


def _render_figures_svg(results: "ExperimentResults") -> list[str]:
    """Inline SVG for the three viz plots; empty list if matplotlib is absent."""
    try:
        from traxr import viz
    except Exception:
        return []
    return viz.render_svgs(results)


def _strip_timestamps(trace: dict[str, Any]) -> dict[str, Any]:
    """Trace dict without per-event wall-clock timestamps (canonical form)."""
    out = dict(trace)
    out["events"] = [
        {k: v for k, v in event.items() if k != "timestamp"} for event in trace.get("events", [])
    ]
    return out
