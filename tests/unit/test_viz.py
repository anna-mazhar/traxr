"""traxr.viz smoke tests (Agg backend, no display)."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from traxr.results import ExperimentResults, PairResult  # noqa: E402
from traxr.viz import (  # noqa: E402
    plot_d_norm,
    plot_manifestations,
    plot_t_star,
    render_svgs,
)


@pytest.fixture()
def results():
    pairs = [
        PairResult(
            item_id="f.csv",
            perturbation="column_swap",
            delivery="round_trip",
            d_norm=0.2,
            t_star_norm=0.5,
            manifestation="strategy_reroute",
        ),
        PairResult(
            item_id="f.csv",
            perturbation="null_content",
            delivery="round_trip",
            d_norm=0.8,
            t_star_norm=0.1,
            manifestation="catastrophic_failure",
        ),
    ]
    return ExperimentResults(
        pairs=pairs,
        traces={},
        answers={},
        fingerprint={"agent_kind": "external"},
        noise_floor=0.05,
        noise_floor_runs=1,
    )


def test_plot_d_norm_renders_bars_and_floor(results):
    ax = plot_d_norm(results)
    assert len(ax.patches) == 2
    assert ax.get_ylim() == (0.0, 1.0)
    assert any(line.get_ydata()[0] == 0.05 for line in ax.lines)  # the floor line
    matplotlib.pyplot.close("all")


def test_plot_t_star_histogram(results):
    ax = plot_t_star(results)
    assert sum(p.get_height() for p in ax.patches) == 2  # both pairs binned
    matplotlib.pyplot.close("all")


def test_plot_manifestations_categories(results):
    ax = plot_manifestations(results)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) == {"strategy_reroute", "catastrophic_failure"}
    matplotlib.pyplot.close("all")


def test_compose_on_existing_axes(results):
    fig, axes = matplotlib.pyplot.subplots(1, 3)
    assert plot_d_norm(results, ax=axes[0]) is axes[0]
    assert plot_t_star(results, ax=axes[1]) is axes[1]
    assert plot_manifestations(results, ax=axes[2]) is axes[2]
    matplotlib.pyplot.close("all")


def test_render_svgs_returns_three_inline_svgs(results):
    svgs = render_svgs(results)
    assert len(svgs) == 3
    assert all(s.startswith("<svg") for s in svgs)  # XML/DOCTYPE preamble stripped


def test_render_svgs_without_matplotlib_returns_empty(monkeypatch):
    import sys

    # Simulate the [viz] extra being absent: Figure import fails -> [].
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    empty = ExperimentResults(pairs=[], traces={}, answers={}, fingerprint={})
    assert render_svgs(empty) == []


def test_missing_matplotlib_raises(monkeypatch):
    import sys

    from traxr.errors import OptionalDependencyError

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(OptionalDependencyError, match=r"traxr\[viz\]"):
        plot_d_norm(ExperimentResults(pairs=[], traces={}, answers={}, fingerprint={}))
