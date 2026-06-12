"""Typed exception and warning hierarchy for Traxr.

Principle: fail loud with actionable messages; never silently produce wrong
metrics. User-configuration errors fail fast; data-dependent failures are
skipped and recorded (as warnings collected per pair).

All exceptions derive from :class:`TraxrError`; all warnings derive from
:class:`TraxrWarning`.
"""

__all__ = [
    "TraxrError",
    "UnsupportedModalityError",
    "ModalityMismatchError",
    "InvalidArtifactError",
    "OptionalDependencyError",
    "LLMConnectionError",
    "AgentContractError",
    "RunBudgetExceeded",
    "ExperimentConfigError",
    "MatrixTooLargeError",
    "MalformedEventError",
    "ControlledVariableError",
    "TraxrWarning",
    "EmptyTraceWarning",
    "NonDeterminismWarning",
    "ConcurrentTraceWarning",
    "TokenUnavailableWarning",
    "PerturbationSkippedWarning",
]


class TraxrError(Exception):
    """Base class for all Traxr exceptions."""


class UnsupportedModalityError(TraxrError):
    """Unknown extension or unsupported modality.

    Raised for docx/pptx/image/audio inputs in v1. The message names what IS
    supported (CSV/XLSX/TXT/MD/PDF) and points at the roadmap for the rest.
    """


class ModalityMismatchError(TraxrError):
    """Declared modality does not match the detected modality of the file."""


class InvalidArtifactError(TraxrError):
    """Input file is missing, unreadable, or corrupt."""


class OptionalDependencyError(TraxrError):
    """A required optional dependency is not installed.

    Raised when openai/PyMuPDF/pdfplumber/openpyxl/matplotlib/langchain-core
    is needed but missing. The message names the pip extra that provides it
    (e.g. ``pip install "traxr[document]"``).
    """


class LLMConnectionError(TraxrError):
    """Missing/invalid API key or unreachable base_url (built-in agent path)."""


class AgentContractError(TraxrError):
    """The user's agent violated the AgentRunner contract.

    For example, it returned a non-``str`` value.
    """


class RunBudgetExceeded(TraxrError):
    """The agent exceeded ``max_llm_calls_per_run``.

    Raised inside the agent by the Tier 0 capture wrapper.
    """


class ExperimentConfigError(TraxrError):
    """Invalid experiment configuration.

    Raised when both or neither of ``agent``/``agent_factory``/``llm`` are
    resolvable, or other fail-fast configuration problems.
    """


class MatrixTooLargeError(TraxrError):
    """The permutation matrix exceeds the configured cap."""


class MalformedEventError(TraxrError):
    """Malformed or missing event payload, or empty/non-string event_type."""


class ControlledVariableError(TraxrError):
    """Experiment configuration was mutated between paired runs."""


class TraxrWarning(UserWarning):
    """Base class for all Traxr warnings (non-fatal, collected per pair)."""


class EmptyTraceWarning(TraxrWarning):
    """A run produced no trace events."""


class NonDeterminismWarning(TraxrWarning):
    """Paired runs differ beyond the measured noise floor's expectation."""


class ConcurrentTraceWarning(TraxrWarning):
    """Concurrent LLM calls detected while tracing a single run."""


class TokenUnavailableWarning(TraxrWarning):
    """Token usage could not be captured for one or more LLM calls."""


class PerturbationSkippedWarning(TraxrWarning):
    """A perturbation was not applicable and was skipped (recorded).

    Corresponds to the engine reporting ``applied=False`` with a
    ``skip_reason``.
    """
