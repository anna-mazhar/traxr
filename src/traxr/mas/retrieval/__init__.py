"""Retrieval module for retrieval-contamination experiments.

This module provides:
- RetrievalItem/RetrievalResult: Data structures for retrieval
- InMemoryRetrieval: Keyword-based in-memory retrieval
- Condition applicators: NORMAL, NULL

Note: File perturbations are handled by the perturbations module,
which applies perturbations directly to files before they're loaded by tools.
"""

from .items import RetrievalItem, RetrievalResult
from .base import RetrievalComponent
from .conditions import (
    RetrievalConditionApplicator,
    NormalCondition,
    NullCondition,
    get_condition_applicator,
)
from .keyword_retrieval import InMemoryRetrieval

__all__ = [
    "RetrievalItem",
    "RetrievalResult",
    "RetrievalComponent",
    "RetrievalConditionApplicator",
    "NormalCondition",
    "NullCondition",
    "get_condition_applicator",
    "InMemoryRetrieval",
]
