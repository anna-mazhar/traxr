"""Retrieval condition applicators for experimental interventions.

Note: File corruption is handled by the perturbations module,
which applies perturbations directly to files before they're loaded.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..core.types import RetrievalCondition
from .items import RetrievalResult


class RetrievalConditionApplicator(ABC):
    """Abstract base class for retrieval condition applicators."""

    @property
    @abstractmethod
    def condition(self) -> RetrievalCondition:
        """The condition type this applicator implements."""
        pass

    @abstractmethod
    def apply(
        self,
        result: RetrievalResult,
        seed: int,
        context: Optional[dict] = None,
    ) -> RetrievalResult:
        """Apply the condition to retrieval results.

        Args:
            result: Original retrieval result
            seed: Episode seed for determinism
            context: Optional context (query, task info, etc.)

        Returns:
            Modified retrieval result
        """
        pass


class NormalCondition(RetrievalConditionApplicator):
    """Pass-through condition - no modification."""

    @property
    def condition(self) -> RetrievalCondition:
        return RetrievalCondition.NORMAL

    def apply(
        self,
        result: RetrievalResult,
        seed: int,
        context: Optional[dict] = None,
    ) -> RetrievalResult:
        """Return result unchanged."""
        return result


class NullCondition(RetrievalConditionApplicator):
    """Returns empty retrieval results."""

    @property
    def condition(self) -> RetrievalCondition:
        return RetrievalCondition.NULL

    def apply(
        self,
        result: RetrievalResult,
        seed: int,
        context: Optional[dict] = None,
    ) -> RetrievalResult:
        """Return empty result."""
        return RetrievalResult(
            query=result.query,
            items=[],
            total_available=result.total_available,
            filtered_count=len(result.items),  # All items were filtered
        )


def get_condition_applicator(condition: RetrievalCondition) -> RetrievalConditionApplicator:
    """Factory function to get the appropriate condition applicator."""
    applicators = {
        RetrievalCondition.NORMAL: NormalCondition,
        RetrievalCondition.NULL: NullCondition,
    }
    return applicators[condition]()
