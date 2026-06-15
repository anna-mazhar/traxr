"""Plots over :class:`~traxr.results.ExperimentResults` (``[viz]`` extra).

Three small, deliberately boring matplotlib views: per-pair ``d_norm`` bars,
the ``t*/T`` (normalized first-divergence position) histogram, and the
manifestation breakdown. Each returns the Axes so callers can restyle; pass
``ax=`` to compose them into a figure.
"""

from typing import Any

from traxr.errors import OptionalDependencyError
from traxr.results import ExperimentResults

__all__ = ["plot_d_norm", "plot_manifestations", "plot_t_star", "render_svgs"]

_NOISE_FLOOR_STYLE = {"color": "#d62728", "linestyle": "--", "linewidth": 1.0}


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise OptionalDependencyError(
            'traxr.viz needs matplotlib. Install it with: pip install "traxr[viz]"'
        ) from exc
    return plt


def _axes(plt: Any, ax: Any) -> Any:
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=(7, 4))
    return ax


def plot_d_norm(results: ExperimentResults, ax: Any = None) -> Any:
    """Per-pair normalized edit distance, with the noise floor when measured."""
    plt = _require_matplotlib()
    ax = _axes(plt, ax)
    scored = [p for p in results.pairs if p.d_norm is not None]
    labels = [p.perturbation for p in scored]
    values = [p.d_norm for p in scored]
    ax.bar(range(len(values)), values, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("d_norm")
    ax.set_ylim(0, 1)
    ax.set_title("Trace divergence per perturbation")
    if results.noise_floor is not None:
        ax.axhline(results.noise_floor, label="noise floor", **_NOISE_FLOOR_STYLE)
        ax.legend()
    return ax


def plot_t_star(results: ExperimentResults, ax: Any = None, bins: int = 10) -> Any:
    """Histogram of normalized first-divergence positions (``t*/T``)."""
    plt = _require_matplotlib()
    ax = _axes(plt, ax)
    positions = [p.t_star_norm for p in results.pairs if p.t_star_norm is not None]
    ax.hist(positions, bins=bins, range=(0, 1), color="#1f77b4", edgecolor="white")
    ax.set_xlabel("t* / T (0 = diverged immediately)")
    ax.set_ylabel("pairs")
    ax.set_title("Where divergence begins")
    return ax


def plot_manifestations(results: ExperimentResults, ax: Any = None) -> Any:
    """Manifestation-category prevalence over scored pairs."""
    plt = _require_matplotlib()
    ax = _axes(plt, ax)
    prevalence = results.manifestation_prevalence()
    labels = list(prevalence)
    values = [prevalence[k] for k in labels]
    ax.barh(range(len(labels)), values, color="#1f77b4")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("fraction of scored pairs")
    ax.set_xlim(0, 1)
    ax.set_title("How perturbations manifested")
    ax.invert_yaxis()
    return ax


def render_svgs(
    results: ExperimentResults, *, figsize: tuple[float, float] = (5.0, 3.2)
) -> list[str]:
    """Inline-SVG strings for the three plots, for embedding in HTML reports.

    Returns an empty list when matplotlib (the ``[viz]`` extra) is unavailable,
    so :meth:`~traxr.results.ExperimentResults.to_report` can degrade
    gracefully. Builds figures via the pyplot-free ``Figure`` API, so it needs
    no display/backend and mutates no global state.
    """
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return []
    import io

    svgs: list[str] = []
    for plot in (plot_d_norm, plot_t_star, plot_manifestations):
        fig = Figure(figsize=figsize)
        plot(results, ax=fig.subplots())
        fig.tight_layout()
        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        svgs.append(_svg_body(buf.getvalue()))
    return svgs


def _svg_body(svg_text: str) -> str:
    """Drop the XML/DOCTYPE preamble so the ``<svg>`` embeds inline in HTML5."""
    idx = svg_text.find("<svg")
    return svg_text[idx:] if idx != -1 else svg_text
