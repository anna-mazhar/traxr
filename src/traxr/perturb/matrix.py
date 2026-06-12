"""Permutation matrix: enumerate perturbations for a source and agent kind.

NEW in traxr. v1 = single-operator permutations. The enumeration is aware of
the PDF delivery split: external agents get the in-place surgical operators
(real perturbed PDF on disk); the built-in agent additionally gets the
whole-text-flow operators via the verified content-injection mechanism.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum

from traxr.data.sources import DataSource, ModalityType
from traxr.errors import MatrixTooLargeError

from .pdf_inplace import PDF_INPLACE_OPERATORS
from .types import PerturbationType


class AgentKind(Enum):
    """Which kind of agent consumes the perturbed artifact."""

    EXTERNAL = "external"
    BUILTIN = "builtin"


class DeliveryPath(Enum):
    """How a perturbed artifact reaches the agent."""

    ROUND_TRIP = "round_trip"  # parse -> perturb -> re-serialize to disk
    PDF_INPLACE = "pdf_inplace"  # surgical PyMuPDF edits on a PDF copy
    INJECTION = "injection"  # built-in agent only: injected extracted content


# Operator catalogs (v1 supported + tested sets).
TABULAR_OPERATORS = (
    PerturbationType.COLUMN_SWAP,
    PerturbationType.LABEL_CORRUPT,
    PerturbationType.DATA_TYPE_CORRUPT,
    PerturbationType.ROW_DUPLICATE,
    PerturbationType.IRRELEVANT_COLUMNS,
    PerturbationType.UNIT_CHANGE,
)

TEXT_OPERATORS = (
    PerturbationType.OCR_NOISE,
    PerturbationType.NUMBER_CORRUPTION,
    PerturbationType.TEXT_REDACTION,
    PerturbationType.PARAGRAPH_SHUFFLE,
    PerturbationType.ENCODING_ERROR,
    PerturbationType.SECTION_REMOVAL,
)

# Whole-text-flow operators that cannot be applied surgically to a PDF;
# deliverable only to the built-in agent via content injection.
PDF_INJECTION_ONLY_OPERATORS = (
    PerturbationType.OCR_NOISE,
    PerturbationType.PARAGRAPH_SHUFFLE,
    PerturbationType.ENCODING_ERROR,
)


@dataclass(frozen=True)
class PermutationSpec:
    """One clean-vs-perturbed pair to run: a single operator on one source."""

    source_id: str
    perturbation: PerturbationType
    seed: int
    delivery: DeliveryPath


@dataclass(frozen=True)
class MatrixConfig:
    """Configuration for :func:`build_matrix`."""

    agent_kind: AgentKind = AgentKind.EXTERNAL
    base_seed: int = 42
    max_permutations: int = 64

    def __post_init__(self) -> None:
        if self.max_permutations < 1:
            raise ValueError("max_permutations must be >= 1")


def derive_seed(base_seed: int, source_id: str, perturbation: PerturbationType) -> int:
    """Deterministic per-permutation seed (stable across runs and platforms)."""
    digest = hashlib.sha256(f"{base_seed}:{source_id}:{perturbation.value}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def supported_operators(
    source: DataSource, agent_kind: AgentKind
) -> list[tuple[PerturbationType, DeliveryPath]]:
    """Enumerate (operator, delivery) pairs supported for a source and agent kind."""
    pairs: list[tuple[PerturbationType, DeliveryPath]] = []

    if source.modality is ModalityType.TABULAR:
        pairs.extend((op, DeliveryPath.ROUND_TRIP) for op in TABULAR_OPERATORS)
        pairs.append((PerturbationType.NULL_CONTENT, DeliveryPath.ROUND_TRIP))
    elif source.file_type in ("txt", "md"):
        pairs.extend((op, DeliveryPath.ROUND_TRIP) for op in TEXT_OPERATORS)
        pairs.append((PerturbationType.NULL_CONTENT, DeliveryPath.ROUND_TRIP))
    else:  # pdf
        pairs.extend((op, DeliveryPath.PDF_INPLACE) for op in PDF_INPLACE_OPERATORS)
        if agent_kind is AgentKind.BUILTIN:
            pairs.extend((op, DeliveryPath.INJECTION) for op in PDF_INJECTION_ONLY_OPERATORS)

    return pairs


def build_matrix(source: DataSource, config: MatrixConfig | None = None) -> list[PermutationSpec]:
    """Build the v1 single-operator permutation matrix for one source.

    Raises:
        MatrixTooLargeError: if the enumeration exceeds
            ``config.max_permutations``.
    """
    if config is None:
        config = MatrixConfig()
    specs = [
        PermutationSpec(
            source_id=source.source_id,
            perturbation=op,
            seed=derive_seed(config.base_seed, source.source_id, op),
            delivery=delivery,
        )
        for op, delivery in supported_operators(source, config.agent_kind)
    ]

    if len(specs) > config.max_permutations:
        raise MatrixTooLargeError(
            f"Permutation matrix for '{source.source_id}' has {len(specs)} entries, "
            f"exceeding the cap of {config.max_permutations}. Raise "
            "MatrixConfig.max_permutations deliberately if this is intended."
        )

    return specs
