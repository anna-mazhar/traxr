"""Cost proxies and clean-run baselines.

The headline derived metric is **token overhead** =
``perturbed_tokens / baseline_tokens``
(``CostComparison.token_inflation_ratio``).

For external agents, token counts come from the Tier 0 wrapper's captured
``usage``; when usage is unavailable the capture layer emits
``TokenUnavailableWarning`` (M3b).

Left behind in the source: ``ContaminationMetrics`` and ``RunSummary``
(taint-tracking and ``EpisodeResult`` coupled — internal to the built-in MAS).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostProxy:
    """Tracks computational cost proxies for a run."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_calls: int = 0
    total_steps: int = 0

    def add_tokens(self, prompt: int, completion: int) -> None:
        """Add token counts."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def add_retrieval_call(self) -> None:
        """Record a retrieval call."""
        self.retrieval_calls += 1

    def increment_steps(self) -> None:
        """Increment step counter."""
        self.total_steps += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retrieval_calls": self.retrieval_calls,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostProxy":
        """Create from dictionary."""
        return cls(
            total_tokens=data.get("total_tokens", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            retrieval_calls=data.get("retrieval_calls", 0),
            total_steps=data.get("total_steps", 0),
        )


@dataclass
class CostComparison:
    """Compares costs between this run and the clean-run baseline."""

    # This run
    tokens_this_run: int = 0
    steps_this_run: int = 0
    retrieval_calls_this_run: int = 0

    # Normal baseline
    tokens_baseline: int = 0
    steps_baseline: int = 0
    retrieval_calls_baseline: int = 0

    # Inflation ratios (token_inflation_ratio is the paper's token overhead)
    token_inflation_ratio: float = 1.0
    step_inflation_ratio: float = 1.0
    retrieval_inflation_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "this_run": {
                "tokens": self.tokens_this_run,
                "steps": self.steps_this_run,
                "retrieval_calls": self.retrieval_calls_this_run,
            },
            "baseline": {
                "tokens": self.tokens_baseline,
                "steps": self.steps_baseline,
                "retrieval_calls": self.retrieval_calls_baseline,
            },
            "inflation_ratios": {
                "tokens": self.token_inflation_ratio,
                "steps": self.step_inflation_ratio,
                "retrieval_calls": self.retrieval_inflation_ratio,
            },
        }

    @classmethod
    def compute(cls, this_cost: CostProxy, baseline_cost: CostProxy | None) -> "CostComparison":
        """Compute cost comparison (ratios default to 1.0 without a baseline)."""
        if baseline_cost is None:
            return cls(
                tokens_this_run=this_cost.total_tokens,
                steps_this_run=this_cost.total_steps,
                retrieval_calls_this_run=this_cost.retrieval_calls,
            )

        def safe_ratio(a: int, b: int) -> float:
            return a / b if b > 0 else 1.0

        return cls(
            tokens_this_run=this_cost.total_tokens,
            steps_this_run=this_cost.total_steps,
            retrieval_calls_this_run=this_cost.retrieval_calls,
            tokens_baseline=baseline_cost.total_tokens,
            steps_baseline=baseline_cost.total_steps,
            retrieval_calls_baseline=baseline_cost.retrieval_calls,
            token_inflation_ratio=safe_ratio(this_cost.total_tokens, baseline_cost.total_tokens),
            step_inflation_ratio=safe_ratio(this_cost.total_steps, baseline_cost.total_steps),
            retrieval_inflation_ratio=safe_ratio(
                this_cost.retrieval_calls, baseline_cost.retrieval_calls
            ),
        )


class BaselineStore:
    """Persists clean-run (baseline) cost metrics for each spec_hash."""

    def __init__(self, store_path: str | Path | None = None):
        """Initialize baseline store.

        Args:
            store_path: Path to the JSON file for persistence.
                       If None, uses in-memory only.
        """
        self._store_path = Path(store_path) if store_path else None
        self._baselines: dict[str, CostProxy] = {}

        # Load existing baselines if file exists
        if self._store_path and self._store_path.exists():
            self._load()

    def _load(self) -> None:
        """Load baselines from file."""
        if not self._store_path:
            return

        try:
            with open(self._store_path, encoding="utf-8") as f:
                data = json.load(f)
                for spec_hash, cost_data in data.items():
                    self._baselines[spec_hash] = CostProxy.from_dict(cost_data)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(
                "Could not load baseline store from %s; starting empty.", self._store_path
            )

    def _save(self) -> None:
        """Save baselines to file."""
        if not self._store_path:
            return

        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {spec_hash: cost.to_dict() for spec_hash, cost in self._baselines.items()}
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_baseline(self, spec_hash: str) -> CostProxy | None:
        """Get baseline cost for a spec_hash.

        Args:
            spec_hash: The episode spec hash

        Returns:
            CostProxy if baseline exists, None otherwise
        """
        return self._baselines.get(spec_hash)

    def set_baseline(self, spec_hash: str, cost: CostProxy) -> None:
        """Set baseline cost for a spec_hash.

        Args:
            spec_hash: The episode spec hash
            cost: The cost metrics from a clean (baseline) run
        """
        self._baselines[spec_hash] = cost
        self._save()

    def has_baseline(self, spec_hash: str) -> bool:
        """Check if baseline exists for spec_hash."""
        return spec_hash in self._baselines

    def clear(self) -> None:
        """Clear all baselines."""
        self._baselines.clear()
        self._save()

    def get_all_spec_hashes(self) -> list[str]:
        """Get all spec hashes with baselines."""
        return list(self._baselines.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {spec_hash: cost.to_dict() for spec_hash, cost in self._baselines.items()}
