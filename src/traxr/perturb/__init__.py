"""Perturbation operators and engine.

v1 exposes the tabular (6) + document (6) + PDF-native (2) + NULL operators.
PDF delivery is split into two paths: in-place surgical editing for external
agents (:mod:`traxr.perturb.pdf_inplace`) and content injection for the
built-in agent (the :class:`PDFPerturbator` text path).

``image``/``audio`` perturbators are copied for fidelity but **not exported**
(backlog).
"""

from .engine import (
    PerturbationEngine,
    get_all_perturbation_types,
    get_pdf_perturbation_types,
    get_tabular_perturbation_types,
)
from .matrix import (
    AgentKind,
    DeliveryPath,
    MatrixConfig,
    PermutationSpec,
    build_matrix,
    derive_seed,
    supported_operators,
)
from .pdf import PDFPerturbator
from .pdf_inplace import PDF_INPLACE_OPERATORS, PDFInPlaceEditor, apply_pdf_inplace
from .tabular import TabularPerturbator
from .types import PERTURBATION_DESCRIPTIONS, PerturbationResult, PerturbationType

__all__ = [
    "PERTURBATION_DESCRIPTIONS",
    "PDF_INPLACE_OPERATORS",
    "AgentKind",
    "DeliveryPath",
    "MatrixConfig",
    "PDFInPlaceEditor",
    "PDFPerturbator",
    "PermutationSpec",
    "PerturbationEngine",
    "PerturbationResult",
    "PerturbationType",
    "TabularPerturbator",
    "apply_pdf_inplace",
    "build_matrix",
    "derive_seed",
    "get_all_perturbation_types",
    "get_pdf_perturbation_types",
    "get_tabular_perturbation_types",
    "supported_operators",
]
