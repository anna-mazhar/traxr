"""Traxr — controlled-perturbation experiments for your own agent and your own data.

Point your agent at your data, run paired clean/perturbed experiments, and get
back contamination/divergence metrics (``d_norm``, ``t*``, manifestation,
token overhead).
"""

from traxr import errors
from traxr.trace.registry import register_signature

__version__ = "0.1.0.dev0"

# Curated public API. Placeholder: Experiment, ExperimentResults, Task,
# instrument, from_langgraph, builtin_agent, emit, OpenAICompatibleClient,
# LLMClient, PerturbationType land across M2–M4b.
__all__ = [
    "__version__",
    "errors",
    "register_signature",
]
