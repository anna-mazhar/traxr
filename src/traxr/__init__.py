"""Traxr — controlled-perturbation experiments for your own agent and your own data.

Point your agent at your data, run paired clean/perturbed experiments, and get
back contamination/divergence metrics (``d_norm``, ``t*``, manifestation,
token overhead).
"""

from traxr import errors
from traxr.agents import AgentRunner, Task, builtin_agent
from traxr.capture import emit, instrument
from traxr.llm import DeterministicLLMStub, LLMClient, OpenAICompatibleClient
from traxr.perturb.types import PerturbationType
from traxr.trace.registry import register_signature

__version__ = "0.1.0.dev0"

# Curated public API. Placeholder: Experiment, ExperimentResults, and
# from_langgraph land in M4/M4b. patch_openai stays under traxr.capture.
__all__ = [
    "AgentRunner",
    "DeterministicLLMStub",
    "LLMClient",
    "OpenAICompatibleClient",
    "PerturbationType",
    "Task",
    "__version__",
    "builtin_agent",
    "emit",
    "errors",
    "instrument",
    "register_signature",
]
