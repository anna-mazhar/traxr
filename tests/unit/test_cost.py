"""Cost proxies, token-overhead comparison, and the baseline store."""

import json

from traxr.experiment import _cost_from_trace
from traxr.metrics.cost import BaselineStore, CostComparison, CostProxy
from traxr.trace.collector import TraceCollector


class TestCostFromTrace:
    def test_total_steps_counts_llm_call_events(self):
        collector = TraceCollector("run")
        collector.emit(
            "llm_call", 1, "ext", {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        )
        collector.emit("tool_request", 1, "ext", {"tool_name": "x"})
        collector.emit(
            "llm_call", 2, "ext", {"usage": {"prompt_tokens": 20, "completion_tokens": 7}}
        )
        cost = _cost_from_trace(collector)
        assert cost.total_steps == 2  # two llm_call events; tool_request is not a step
        assert cost.total_tokens == 42

    def test_total_steps_independent_of_session_counter(self):
        # Tier 1 (LangGraph) emits llm_call events without ever calling
        # begin_llm_call, so step count must come from the events themselves.
        collector = TraceCollector("run")
        collector.emit("llm_call", 1, "langgraph", {"usage": None})
        cost = _cost_from_trace(collector)
        assert cost.total_steps == 1
        assert cost.total_tokens == 0


class TestCostProxy:
    def test_add_tokens_accumulates(self):
        cost = CostProxy()
        cost.add_tokens(prompt=100, completion=50)
        cost.add_tokens(prompt=10, completion=5)
        assert cost.prompt_tokens == 110
        assert cost.completion_tokens == 55
        assert cost.total_tokens == 165

    def test_counters(self):
        cost = CostProxy()
        cost.add_retrieval_call()
        cost.increment_steps()
        cost.increment_steps()
        assert cost.retrieval_calls == 1
        assert cost.total_steps == 2

    def test_dict_round_trip(self):
        cost = CostProxy(
            total_tokens=150,
            prompt_tokens=100,
            completion_tokens=50,
            retrieval_calls=2,
            total_steps=7,
        )
        assert CostProxy.from_dict(cost.to_dict()) == cost

    def test_from_dict_defaults_missing_keys(self):
        assert CostProxy.from_dict({}) == CostProxy()


class TestCostComparison:
    def test_token_overhead_is_the_headline_ratio(self):
        this = CostProxy(total_tokens=300, total_steps=10, retrieval_calls=4)
        baseline = CostProxy(total_tokens=200, total_steps=5, retrieval_calls=2)
        cmp = CostComparison.compute(this, baseline)
        assert cmp.token_inflation_ratio == 1.5
        assert cmp.step_inflation_ratio == 2.0
        assert cmp.retrieval_inflation_ratio == 2.0
        assert cmp.tokens_baseline == 200

    def test_no_baseline_defaults_ratios_to_one(self):
        cmp = CostComparison.compute(CostProxy(total_tokens=300, total_steps=10), None)
        assert cmp.token_inflation_ratio == 1.0
        assert cmp.tokens_this_run == 300
        assert cmp.tokens_baseline == 0

    def test_zero_baseline_is_safe(self):
        cmp = CostComparison.compute(CostProxy(total_tokens=300), CostProxy(total_tokens=0))
        assert cmp.token_inflation_ratio == 1.0

    def test_to_dict_shape(self):
        d = CostComparison.compute(CostProxy(total_tokens=10), CostProxy(total_tokens=10)).to_dict()
        assert d["this_run"]["tokens"] == 10
        assert d["baseline"]["tokens"] == 10
        assert d["inflation_ratios"]["tokens"] == 1.0


class TestBaselineStore:
    def test_in_memory_set_get(self):
        store = BaselineStore()
        cost = CostProxy(total_tokens=100)
        assert not store.has_baseline("h1")
        assert store.get_baseline("h1") is None
        store.set_baseline("h1", cost)
        assert store.has_baseline("h1")
        assert store.get_baseline("h1") == cost
        assert store.get_all_spec_hashes() == ["h1"]

    def test_persistence_round_trip(self, tmp_path):
        path = tmp_path / "baselines.json"
        store = BaselineStore(store_path=path)
        store.set_baseline("h1", CostProxy(total_tokens=42, total_steps=3))

        reloaded = BaselineStore(store_path=path)
        baseline = reloaded.get_baseline("h1")
        assert baseline is not None
        assert baseline.total_tokens == 42
        assert baseline.total_steps == 3

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "baselines.json"
        BaselineStore(store_path=path).set_baseline("h1", CostProxy())
        assert path.exists()

    def test_corrupt_store_starts_empty(self, tmp_path, caplog):
        path = tmp_path / "baselines.json"
        path.write_text("{not json")
        store = BaselineStore(store_path=path)
        assert store.get_all_spec_hashes() == []

    def test_clear_persists(self, tmp_path):
        path = tmp_path / "baselines.json"
        store = BaselineStore(store_path=path)
        store.set_baseline("h1", CostProxy(total_tokens=1))
        store.clear()
        assert json.loads(path.read_text()) == {}
        assert not BaselineStore(store_path=path).has_baseline("h1")

    def test_to_dict(self):
        store = BaselineStore()
        store.set_baseline("h1", CostProxy(total_tokens=5))
        assert store.to_dict()["h1"]["total_tokens"] == 5
