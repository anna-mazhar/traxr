"""Traxr — controlled-perturbation experiments for your own agent and your own data.

Point your agent at your data, run paired clean/perturbed experiments, and get
back contamination/divergence metrics (``d_norm``, ``t*``, manifestation,
token overhead).
"""

from traxr import errors
from traxr.agents import builtin_agent
from traxr.llm import DeterministicLLMStub, LLMClient, OpenAICompatibleClient
from traxr.perturb.types import PerturbationType
from traxr.trace.registry import register_signature

__version__ = "0.1.0.dev0"

# Curated public API. Placeholder: Experiment, ExperimentResults, Task,
# instrument, from_langgraph, emit land across M3b–M4b.
__all__ = [
    "DeterministicLLMStub",
    "LLMClient",
    "OpenAICompatibleClient",
    "PerturbationType",
    "__version__",
    "builtin_agent",
    "errors",
    "register_signature",
]
