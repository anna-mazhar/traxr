"""Divergence, manifestation, and cost metrics.

* :mod:`traxr.metrics.analyzer` — ``d_norm`` (normalized structural edit
  distance), ``t*`` (first divergence point), and control-flow change counts.
* :mod:`traxr.metrics.manifest` — manifestation taxonomy over a typed
  :class:`~traxr.metrics.manifest.PairMetrics`.
* :mod:`traxr.metrics.cost` — token/step cost proxies and the clean-run
  baseline store.
"""

from traxr.metrics.analyzer import (
    STRUCTURAL_DIVERGENCE_TYPES,
    AlignedEventPair,
    ControlFlowChanges,
    DivergenceReport,
    EditDistanceResult,
    TraceDivergenceAnalyzer,
)
from traxr.metrics.cost import BaselineStore, CostComparison, CostProxy
from traxr.metrics.manifest import (
    FINE_CATEGORIES,
    MANIFESTATION_GROUPS,
    PairMetrics,
    classify_manifestation,
    to_manifestation_group,
)

__all__ = [
    "FINE_CATEGORIES",
    "MANIFESTATION_GROUPS",
    "STRUCTURAL_DIVERGENCE_TYPES",
    "AlignedEventPair",
    "BaselineStore",
    "ControlFlowChanges",
    "CostComparison",
    "CostProxy",
    "DivergenceReport",
    "EditDistanceResult",
    "PairMetrics",
    "TraceDivergenceAnalyzer",
    "classify_manifestation",
    "to_manifestation_group",
]
