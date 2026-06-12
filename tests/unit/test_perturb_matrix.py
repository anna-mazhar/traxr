"""Category 1 tests: permutation matrix — agent-kind-aware enumeration."""

from pathlib import Path

import pytest

from traxr.data import DataSource
from traxr.errors import MatrixTooLargeError
from traxr.perturb import PerturbationType
from traxr.perturb.matrix import (
    PDF_INJECTION_ONLY_OPERATORS,
    TABULAR_OPERATORS,
    TEXT_OPERATORS,
    AgentKind,
    DeliveryPath,
    MatrixConfig,
    build_matrix,
    derive_seed,
    supported_operators,
)
from traxr.perturb.pdf_inplace import PDF_INPLACE_OPERATORS

# ---------------------------------------------------------------------------
# derive_seed
# ---------------------------------------------------------------------------


def test_derive_seed_is_deterministic() -> None:
    a = derive_seed(42, "sample.csv", PerturbationType.COLUMN_SWAP)
    b = derive_seed(42, "sample.csv", PerturbationType.COLUMN_SWAP)
    assert a == b
    assert 0 <= a < 2**32


def test_derive_seed_varies_with_each_input() -> None:
    base = derive_seed(42, "sample.csv", PerturbationType.COLUMN_SWAP)
    assert derive_seed(43, "sample.csv", PerturbationType.COLUMN_SWAP) != base
    assert derive_seed(42, "other.csv", PerturbationType.COLUMN_SWAP) != base
    assert derive_seed(42, "sample.csv", PerturbationType.ROW_DUPLICATE) != base


# ---------------------------------------------------------------------------
# supported_operators — modality x agent kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sample.csv", "sample.xlsx"])
@pytest.mark.parametrize("kind", list(AgentKind))
def test_tabular_operators_round_trip_for_any_agent(
    name: str, kind: AgentKind, fixtures_dir: Path
) -> None:
    pairs = supported_operators(DataSource.from_path(fixtures_dir / name), kind)
    assert {op for op, _ in pairs} == set(TABULAR_OPERATORS) | {PerturbationType.NULL_CONTENT}
    assert {d for _, d in pairs} == {DeliveryPath.ROUND_TRIP}


@pytest.mark.parametrize("name", ["sample.txt", "sample.md"])
@pytest.mark.parametrize("kind", list(AgentKind))
def test_text_operators_round_trip_for_any_agent(
    name: str, kind: AgentKind, fixtures_dir: Path
) -> None:
    pairs = supported_operators(DataSource.from_path(fixtures_dir / name), kind)
    assert {op for op, _ in pairs} == set(TEXT_OPERATORS) | {PerturbationType.NULL_CONTENT}
    assert {d for _, d in pairs} == {DeliveryPath.ROUND_TRIP}


def test_pdf_external_agent_gets_only_inplace_operators(fixtures_dir: Path) -> None:
    pairs = supported_operators(
        DataSource.from_path(fixtures_dir / "sample.pdf"), AgentKind.EXTERNAL
    )
    assert pairs == [(op, DeliveryPath.PDF_INPLACE) for op in PDF_INPLACE_OPERATORS]


def test_pdf_builtin_agent_adds_injection_operators(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.pdf")
    external = supported_operators(source, AgentKind.EXTERNAL)
    builtin = supported_operators(source, AgentKind.BUILTIN)
    assert builtin[: len(external)] == external
    assert builtin[len(external) :] == [
        (op, DeliveryPath.INJECTION) for op in PDF_INJECTION_ONLY_OPERATORS
    ]


# ---------------------------------------------------------------------------
# build_matrix
# ---------------------------------------------------------------------------


def test_build_matrix_specs_are_deterministic(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.csv")
    first = build_matrix(source)
    second = build_matrix(source)
    assert first == second
    assert len(first) == len(TABULAR_OPERATORS) + 1


def test_build_matrix_uses_derived_per_permutation_seeds(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.txt")
    config = MatrixConfig(base_seed=7)
    specs = build_matrix(source, config)
    for spec in specs:
        assert spec.source_id == "sample.txt"
        assert spec.seed == derive_seed(7, "sample.txt", spec.perturbation)
    assert len({spec.seed for spec in specs}) == len(specs)


def test_build_matrix_base_seed_changes_all_seeds(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.csv")
    seeds_a = [s.seed for s in build_matrix(source, MatrixConfig(base_seed=1))]
    seeds_b = [s.seed for s in build_matrix(source, MatrixConfig(base_seed=2))]
    assert all(a != b for a, b in zip(seeds_a, seeds_b, strict=True))


def test_build_matrix_cap_raises(fixtures_dir: Path) -> None:
    source = DataSource.from_path(fixtures_dir / "sample.csv")
    with pytest.raises(MatrixTooLargeError, match="max_permutations"):
        build_matrix(source, MatrixConfig(max_permutations=2))


def test_matrix_config_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        MatrixConfig(max_permutations=0)
