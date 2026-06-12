"""Traxr — controlled-perturbation experiments for your own agent and your own data.

Point your agent at your data, run paired clean/perturbed experiments, and get
back contamination/divergence metrics (``d_norm``, ``t*``, manifestation,
token overhead).
"""

from traxr import errors

__version__ = "0.1.0.dev0"

from traxr.agents import AgentRunner, Task, builtin_agent
from traxr.capture import emit, instrument
from traxr.experiment import Experiment, ExperimentConfig
from traxr.llm import DeterministicLLMStub, LLMClient, OpenAICompatibleClient
from traxr.perturb.types import PerturbationType
from traxr.results import ExperimentResults, PairResult
from traxr.trace.registry import register_signature


def __getattr__(name: str) -> object:
    # Lazy so `python -m traxr.selfcheck` doesn't re-import the module it is
    # already executing (avoids the runpy double-import RuntimeWarning).
    if name == "selfcheck":
        from traxr.selfcheck import selfcheck

        return selfcheck
    raise AttributeError(f"module 'traxr' has no attribute {name!r}")


# Curated public API. Placeholder: from_langgraph lands in M4b.
# patch_openai stays under traxr.capture.
__all__ = [
    "AgentRunner",
    "DeterministicLLMStub",
    "Experiment",
    "ExperimentConfig",
    "ExperimentResults",
    "LLMClient",
    "OpenAICompatibleClient",
    "PairResult",
    "PerturbationType",
    "Task",
    "__version__",
    "builtin_agent",
    "emit",
    "errors",
    "instrument",
    "register_signature",
    "selfcheck",
]
