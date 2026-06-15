"""Perturbation delivery paths — incl. both PDF non-identical-artifact guards.

Known risk: each PDF path can silently no-op if mishandled, so both get
explicit guards here: the external in-place file must differ on disk, and
the built-in injection payload must differ from the extracted text.
"""

import hashlib
from pathlib import Path

import pytest

from traxr.data.loader import read_file
from traxr.experiment import _deliver_perturbation
from traxr.perturb.matrix import (
    PDF_INJECTION_ONLY_OPERATORS,
    DeliveryPath,
    PermutationSpec,
)
from traxr.perturb.pdf_inplace import PDF_INPLACE_OPERATORS
from traxr.perturb.types import PerturbationType


def spec_for(perturbation, delivery, source_id="sample.pdf", seed=7):
    return PermutationSpec(
        source_id=source_id, perturbation=perturbation, seed=seed, delivery=delivery
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_writes_perturbed_csv(fixtures_dir, tmp_path):
    source = fixtures_dir / "sample.csv"
    dst = tmp_path / "sample.csv"
    spec = spec_for(PerturbationType.COLUMN_SWAP, DeliveryPath.ROUND_TRIP, "sample.csv")
    result, metadata = _deliver_perturbation(spec, source, dst)
    assert result.applied
    assert metadata == {}
    assert dst.exists()
    assert dst.read_text() != source.read_text()


def test_round_trip_non_ascii_is_written_utf8(fixtures_dir, tmp_path):
    """H2: a perturbation that injects non-ASCII (ENCODING_ERROR -> U+FFFD) must
    write the perturbed file as UTF-8, not the platform-default encoding (which
    raises UnicodeEncodeError or writes mojibake under e.g. cp1252 / LANG=C)."""
    source = fixtures_dir / "sample.txt"
    dst = tmp_path / "sample.txt"
    spec = spec_for(PerturbationType.ENCODING_ERROR, DeliveryPath.ROUND_TRIP, "sample.txt", seed=7)
    result, _ = _deliver_perturbation(spec, source, dst)
    assert result.applied
    raw = dst.read_bytes()
    assert any(b > 127 for b in raw)  # non-ASCII really was injected
    # Bytes are the UTF-8 encoding of the content (locale-independent), and the
    # file round-trips through UTF-8 exactly.
    assert raw == result.corrupted_content.encode("utf-8")
    assert dst.read_text(encoding="utf-8") == result.corrupted_content


def test_round_trip_null_content_writes_empty_file(fixtures_dir, tmp_path):
    source = fixtures_dir / "sample.csv"
    dst = tmp_path / "sample.csv"
    spec = spec_for(PerturbationType.NULL_CONTENT, DeliveryPath.ROUND_TRIP, "sample.csv")
    result, _ = _deliver_perturbation(spec, source, dst)
    assert result.applied
    assert dst.read_text() == ""


@pytest.mark.parametrize("perturbation", PDF_INPLACE_OPERATORS, ids=lambda p: p.value)
def test_pdf_inplace_guard_applied_means_different_bytes(fixtures_dir, tmp_path, perturbation):
    """External-agent PDF guard: an applied in-place edit must change the file."""
    source = fixtures_dir / "sample.pdf"
    dst = tmp_path / "sample.pdf"
    spec = spec_for(perturbation, DeliveryPath.PDF_INPLACE)
    result, metadata = _deliver_perturbation(spec, source, dst)
    assert metadata == {}
    if result.applied:
        assert dst.exists()
        assert file_hash(dst) != file_hash(source), (
            f"{perturbation.value} reported applied but produced an identical PDF"
        )
    else:
        assert result.skip_reason


def test_pdf_inplace_at_least_one_operator_applies(fixtures_dir, tmp_path):
    applied = []
    for perturbation in PDF_INPLACE_OPERATORS:
        dst = tmp_path / f"{perturbation.value}.pdf"
        result, _ = _deliver_perturbation(
            spec_for(perturbation, DeliveryPath.PDF_INPLACE), fixtures_dir / "sample.pdf", dst
        )
        applied.append(result.applied)
    assert any(applied), "every in-place PDF operator skipped on the fixture PDF"


@pytest.mark.parametrize("perturbation", PDF_INJECTION_ONLY_OPERATORS, ids=lambda p: p.value)
def test_pdf_injection_guard_payload_differs_from_extraction(fixtures_dir, tmp_path, perturbation):
    """Built-in-agent PDF guard: the injection payload must differ from the
    clean extracted text (otherwise the perturbation silently no-ops)."""
    source = fixtures_dir / "sample.pdf"
    dst = tmp_path / "sample.pdf"
    spec = spec_for(perturbation, DeliveryPath.INJECTION)
    result, metadata = _deliver_perturbation(spec, source, dst)
    if result.applied:
        injected = metadata["injected_pdf_content"]
        assert injected != read_file(source).content, (
            f"{perturbation.value} injection payload identical to the clean extraction"
        )
        assert not dst.exists()  # injection never touches the staged file
    else:
        assert metadata == {}


def test_delivery_is_deterministic_per_seed(fixtures_dir, tmp_path):
    """Controlled-variable invariant: same spec seed -> identical artifact."""
    source = fixtures_dir / "sample.csv"
    spec = spec_for(PerturbationType.LABEL_CORRUPT, DeliveryPath.ROUND_TRIP, "sample.csv")
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    result_a, _ = _deliver_perturbation(spec, source, first)
    result_b, _ = _deliver_perturbation(spec, source, second)
    assert result_a.applied and result_b.applied
    assert first.read_text() == second.read_text()
