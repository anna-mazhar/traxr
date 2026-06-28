"""Unit tests for the typed exception/warning hierarchy in traxr.errors."""

import pytest

import traxr
from traxr import errors

EXCEPTION_TYPES = [
    errors.UnsupportedModalityError,
    errors.ModalityMismatchError,
    errors.InvalidArtifactError,
    errors.OptionalDependencyError,
    errors.LLMConnectionError,
    errors.AgentContractError,
    errors.RunBudgetExceeded,
    errors.ExperimentConfigError,
    errors.MatrixTooLargeError,
    errors.MalformedEventError,
    errors.ControlledVariableError,
]

WARNING_TYPES = [
    errors.EmptyTraceWarning,
    errors.NonDeterminismWarning,
    errors.ConcurrentTraceWarning,
    errors.TokenUnavailableWarning,
    errors.PerturbationSkippedWarning,
]


def test_package_exposes_version_and_errors() -> None:
    assert traxr.__version__
    assert traxr.errors is errors


@pytest.mark.parametrize("exc_type", EXCEPTION_TYPES)
def test_exceptions_subclass_traxr_error(exc_type: type) -> None:
    assert issubclass(exc_type, errors.TraxrError)
    assert issubclass(exc_type, Exception)


@pytest.mark.parametrize("warn_type", WARNING_TYPES)
def test_warnings_subclass_traxr_warning(warn_type: type) -> None:
    assert issubclass(warn_type, errors.TraxrWarning)
    assert issubclass(warn_type, UserWarning)


def test_traxr_error_is_catchable_base() -> None:
    with pytest.raises(errors.TraxrError, match="missing dep"):
        raise errors.OptionalDependencyError("missing dep: install traxr[document]")


def test_all_matches_module_contents() -> None:
    for name in errors.__all__:
        assert hasattr(errors, name)
