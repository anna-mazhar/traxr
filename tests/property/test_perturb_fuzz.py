"""Hypothesis fuzz: the perturbation engine never crashes on generated input.

M2 exit-gate property suite. ``PerturbationEngine.apply`` must be total over
arbitrary text content, arbitrary file-type strings (known and unknown), every
operator, and any seed — always returning a well-formed
:class:`PerturbationResult` (applied, or skipped with a recorded reason).
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from traxr.perturb import PerturbationEngine, PerturbationResult, PerturbationType

# Text-content operators + a sample of out-of-domain ones (binary image/audio
# operators routed at text handlers must degrade to a no-op, never crash).
OPERATORS = st.sampled_from(list(PerturbationType))

KNOWN_FILE_TYPES = ["csv", "tsv", "xlsx", "json", "txt", "md", "pdf", "docx"]
FILE_TYPES = st.one_of(
    st.sampled_from(KNOWN_FILE_TYPES),
    st.text(min_size=0, max_size=8),  # unknown types -> recorded skip
)

CONTENT = st.text(max_size=2000)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


@settings(max_examples=300, deadline=None)
@given(content=CONTENT, file_type=FILE_TYPES, perturbation=OPERATORS, seed=SEEDS)
def test_engine_never_crashes(
    content: str, file_type: str, perturbation: PerturbationType, seed: int
) -> None:
    engine = PerturbationEngine(seed=seed)
    result = engine.apply(content=content, file_type=file_type, perturbation=perturbation)

    assert isinstance(result, PerturbationResult)
    assert result.perturbation_type is perturbation
    assert result.original_content == content
    assert isinstance(result.corrupted_content, str)
    if not result.applied:
        assert result.skip_reason  # every skip carries a recorded reason
    assert engine.get_history()[-1] is result


@settings(max_examples=100, deadline=None)
@given(content=CONTENT, file_type=FILE_TYPES, perturbation=OPERATORS, seed=SEEDS)
def test_engine_is_deterministic_per_seed(
    content: str, file_type: str, perturbation: PerturbationType, seed: int
) -> None:
    first = PerturbationEngine(seed=seed).apply(
        content=content, file_type=file_type, perturbation=perturbation
    )
    second = PerturbationEngine(seed=seed).apply(
        content=content, file_type=file_type, perturbation=perturbation
    )
    assert first.corrupted_content == second.corrupted_content
    assert first.corrupted_hash == second.corrupted_hash
    assert first.applied == second.applied
