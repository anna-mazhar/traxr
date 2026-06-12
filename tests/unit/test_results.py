"""ExperimentResults: aggregates, exporters, summary caveats (category 14)."""

import json

import pytest

from traxr.results import STATUS_SKIPPED, ExperimentResults, PairResult


def make_results(pairs, *, agent_kind="external", noise_floor=None, noise_floor_runs=0):
    return ExperimentResults(
        pairs=pairs,
        traces={"baseline": {"run_label": "baseline", "event_count": 0, "events": []}},
        answers={"baseline": "42"},
        fingerprint={"agent_kind": agent_kind, "capture_tier": "tier0", "seed": 42},
        noise_floor=noise_floor,
        noise_floor_runs=noise_floor_runs,
    )


def pair(**kwargs):
    defaults = dict(item_id="f.csv", perturbation="column_swap", delivery="round_trip")
    defaults.update(kwargs)
    return PairResult(**defaults)


SAMPLE_PAIRS = [
    pair(
        d_norm=0.2,
        t_star_norm=0.5,
        manifestation="strategy_reroute",
        recovery=True,
        token_overhead=1.5,
    ),
    pair(
        perturbation="null_content",
        d_norm=0.6,
        t_star_norm=0.25,
        manifestation="catastrophic_failure",
        recovery=False,
        token_overhead=2.5,
    ),
    pair(perturbation="unit_change", status_perturbed=STATUS_SKIPPED),
]


def test_manifestation_prevalence_over_scored_pairs_only():
    prevalence = make_results(SAMPLE_PAIRS).manifestation_prevalence()
    # 2 scored pairs -> 50/50; the skipped pair is excluded from the denominator.
    assert prevalence == {"catastrophic_failure": 0.5, "strategy_reroute": 0.5}
    assert sum(prevalence.values()) == pytest.approx(1.0)


def test_divergence_recovery_token_aggregates():
    results = make_results(SAMPLE_PAIRS)
    div = results.divergence_summary()
    assert div["pairs_measured"] == 2
    assert div["mean_d_norm"] == pytest.approx(0.4)
    assert div["max_d_norm"] == pytest.approx(0.6)
    assert div["mean_t_star_norm"] == pytest.approx(0.375)
    assert results.recovery_rate() == pytest.approx(0.5)
    assert results.token_overhead_summary() == {
        "mean": pytest.approx(2.0),
        "max": pytest.approx(2.5),
    }


def test_empty_aggregates_are_none_or_empty():
    results = make_results([pair(status_perturbed=STATUS_SKIPPED)])
    assert results.manifestation_prevalence() == {}
    assert results.divergence_summary()["mean_d_norm"] is None
    assert results.recovery_rate() is None
    assert results.token_overhead_summary() == {"mean": None, "max": None}


def test_to_json_shape_and_timestamp_stripping(tmp_path):
    results = make_results(SAMPLE_PAIRS)
    results.traces["baseline"]["events"] = [
        {"event_type": "llm_call", "sequence_index": 0, "timestamp": "2026-06-12T00:00:00"}
    ]
    out = tmp_path / "results.json"
    text = results.to_json(out)
    assert out.read_text() == text
    data = json.loads(text)
    assert set(data) == {
        "aggregates",
        "answers",
        "fingerprint",
        "noise_floor",
        "noise_floor_runs",
        "pairs",
        "traces",
    }
    assert "timestamp" not in data["traces"]["baseline"]["events"][0]
    assert len(data["pairs"]) == 3
    assert json.loads(results.to_json(include_traces=False)).get("traces") is None


def test_summary_caveats_unmeasured_external_floor():
    summary = make_results(SAMPLE_PAIRS).summary()
    assert "UNMEASURED" in summary
    assert "noise_floor_runs" in summary

    measured = make_results(SAMPLE_PAIRS, noise_floor=0.05, noise_floor_runs=2).summary()
    assert "0.0500" in measured and "UNMEASURED" not in measured

    builtin = make_results(SAMPLE_PAIRS, agent_kind="builtin").summary()
    assert "deterministic built-in path" in builtin


def test_summary_flags_order_nondeterminism():
    results = make_results([pair(d_norm=0.0, order_nondeterministic=True)])
    assert "concurrent LLM calls detected" in results.summary()


def test_to_report_md_and_html():
    results = make_results(SAMPLE_PAIRS)
    md = results.to_report("md")
    assert md.startswith("# Traxr experiment report")
    assert "| f.csv | column_swap |" in md
    html = results.to_report("html")
    assert html.startswith("<!DOCTYPE html>") and "column_swap" in html
    with pytest.raises(ValueError, match="md.*html"):
        results.to_report("pdf")


def test_to_dataframe_shape():
    pd = pytest.importorskip("pandas")
    df = make_results(SAMPLE_PAIRS).to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "d_norm" in df.columns and "manifestation" in df.columns
