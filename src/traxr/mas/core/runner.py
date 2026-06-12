"""Episode runner orchestrating agent execution."""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timezone

from .types import CostProxy
from .episode_spec import EpisodeSpec, ExperimentCondition
from .state import TaskInput, SharedState, MemoryAccessTracker
from .outputs import AgentOutput
from ..agents.base import BaseAgent
from ..agents.supervisor_router import SupervisorRouter
from ..routing.dynamic import DynamicRouter
from ..retrieval.base import RetrievalComponent
from ..retrieval.items import RetrievalResult
from ..retrieval.conditions import get_condition_applicator
from ..provenance.tracker import ProvenanceTracker
from ..provenance.taint import TaintTracker
from traxr.trace import TraceCollector

import logging

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single step execution."""
    step_num: int
    agent_name: str
    agent_role: str
    output: AgentOutput
    retrieval_result: Optional[RetrievalResult]
    read_memory_ids: set
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "step_num": self.step_num,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "output": self.output.to_dict(),
            "retrieval_result": self.retrieval_result.to_dict() if self.retrieval_result else None,
            "read_memory_ids": list(self.read_memory_ids),
            "timestamp": self.timestamp,
        }


@dataclass
class EpisodeResult:
    """Result of a complete episode."""
    spec: EpisodeSpec
    condition: ExperimentCondition
    steps: List[StepResult]
    final_answer: Optional[str]
    cost: CostProxy
    taint_tracker: TaintTracker
    provenance_tracker: ProvenanceTracker
    start_time: str
    end_time: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "spec": self.spec.to_dict(),
            "condition": self.condition.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "cost": self.cost.to_dict(),
            "taint": self.taint_tracker.to_dict(),
            "provenance": self.provenance_tracker.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class EpisodeRunner:
    """Orchestrates episode execution with agents, retrieval, and tracking."""

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        retrieval: RetrievalComponent,
        llm_stub=None,  # Optional LLM for router
    ):
        self._agents = agents
        self._retrieval = retrieval
        self._llm_stub = llm_stub

    def run(
        self,
        spec: EpisodeSpec,
        condition: ExperimentCondition,
        task_input: TaskInput,
        step_callback: Optional[Callable[[StepResult], None]] = None,
        trace_collector: Optional[TraceCollector] = None,
    ) -> EpisodeResult:
        """Run a complete episode.

        Args:
            spec: Episode specification
            condition: Experimental condition
            task_input: Input for the task
            step_callback: Optional callback called after each step
            trace_collector: Optional collector for trace divergence analysis

        Returns:
            EpisodeResult with all execution data
        """
        start_time = datetime.now(timezone.utc).isoformat()

        # Initialize state
        state = SharedState(task_input)
        cost = CostProxy()
        taint_tracker = TaintTracker()
        provenance_tracker = ProvenanceTracker()

        # Set up router
        router = self._create_router(spec)

        # Get condition applicator
        condition_applicator = get_condition_applicator(condition.retrieval_condition)

        # Run episode
        steps: List[StepResult] = []
        step_num = 0
        halt_reason: Optional[str] = None

        while step_num < spec.termination.max_steps:
            state.set_current_step(step_num)

            # Get next agent
            agent = router.get_next_agent(state, self._agents, step_num)
            if agent is None:
                halt_reason = "no_agent_selected"
                break

            # Emit routing decision trace event
            if trace_collector:
                trace_collector.emit(
                    event_type="routing_decision",
                    step_num=step_num,
                    agent_name="router",
                    payload={
                        "chosen_agent": agent.name,
                        "reasoning_hash": "",
                    },
                )

            # Check token limit
            if spec.termination.max_tokens and cost.total_tokens >= spec.termination.max_tokens:
                halt_reason = "token_budget_exhausted"
                break

            # Execute agent step
            step_result = self._execute_step(
                agent=agent,
                state=state,
                step_num=step_num,
                spec=spec,
                condition_applicator=condition_applicator,
                taint_tracker=taint_tracker,
                provenance_tracker=provenance_tracker,
                cost=cost,
                trace_collector=trace_collector,
            )

            steps.append(step_result)

            # Log agent execution to update subtask status in the plan
            if hasattr(router, 'router_agent') and hasattr(router.router_agent, 'log_agent_execution'):
                router.router_agent.log_agent_execution(
                    agent_name=agent.name,
                    action=step_result.output.action,
                    result=step_result.output.content or "",
                )

            # Update cost
            cost.add_tokens(
                step_result.output.prompt_tokens,
                step_result.output.completion_tokens,
            )
            cost.increment_steps()

            # Callback
            if step_callback:
                step_callback(step_result)

            # Check for final answer
            if step_result.output.is_final_answer:
                break

            step_num += 1

        # Emit agent_halt when the episode ends without a final answer
        # (router stop, token budget, or max_steps exhaustion).
        if trace_collector and not state.has_final_answer():
            trace_collector.emit(
                event_type="agent_halt",
                step_num=step_num,
                agent_name="runner",
                payload={"reason": halt_reason or "max_steps_exhausted"},
            )

        end_time = datetime.now(timezone.utc).isoformat()

        return EpisodeResult(
            spec=spec,
            condition=condition,
            steps=steps,
            final_answer=state.get_final_answer(),
            cost=cost,
            taint_tracker=taint_tracker,
            provenance_tracker=provenance_tracker,
            start_time=start_time,
            end_time=end_time,
        )

    def _create_router(
        self,
        spec: EpisodeSpec,
    ) -> DynamicRouter:
        """Create the dynamic router for agent selection."""
        available_agents = list(self._agents.keys())
        router_agent = SupervisorRouter(
            available_agents,
            llm=self._llm_stub,
            max_turns=spec.termination.max_steps,
        )
        return DynamicRouter(router_agent)

    def _execute_step(
        self,
        agent: BaseAgent,
        state: SharedState,
        step_num: int,
        spec: EpisodeSpec,
        condition_applicator,
        taint_tracker: TaintTracker,
        provenance_tracker: ProvenanceTracker,
        cost: CostProxy,
        trace_collector: Optional[TraceCollector] = None,
    ) -> StepResult:
        """Execute a single agent step."""
        # Create memory access tracker
        memory_tracker = MemoryAccessTracker(state, agent.name)

        # Handle retrieval if agent uses it
        retrieval_result = None
        if agent.uses_retrieval:
            query = agent.get_retrieval_query(state)
            if query:
                # Get raw retrieval
                raw_result = self._retrieval.retrieve(query)
                cost.add_retrieval_call()

                # Apply condition
                context = {
                    "expected_answer": state.task_input.expected_answer,
                    "oracle_content": state.task_input.metadata.get("oracle_content"),
                }
                retrieval_result = condition_applicator.apply(
                    raw_result, spec.seed, context
                )

                # Track provenance and taint
                for item in retrieval_result.items:
                    provenance_tracker.track_retrieval(item, step_num, agent.name)

                taint_tracker.on_retrieval_shown(
                    step_num, retrieval_result.items, agent.name
                )

                # Emit retrieval_shown trace event
                if trace_collector and retrieval_result.items:
                    trace_collector.emit(
                        event_type="retrieval_shown",
                        step_num=step_num,
                        agent_name=agent.name,
                        payload={
                            "query": retrieval_result.query,
                            "item_count": len(retrieval_result.items),
                            "item_hashes": [
                                str(item.retrieval_id)
                                for item in retrieval_result.items
                            ],
                        },
                    )

        # Execute agent
        logger.debug(f"\n[{agent.name.upper()}] Executing...")
        if retrieval_result and retrieval_result.items:
            logger.debug(f"[{agent.name.upper()}] Retrieved {len(retrieval_result.items)} items")

        output = agent.step(
            state=state,
            memory_tracker=memory_tracker,
            retrieval_result=retrieval_result,
            step_num=step_num,
        )

        # Show output
        if output.content:
            logger.debug(f"[{agent.name.upper()}] Output: {output.content}")
        if output.is_final_answer:
            logger.debug(f"[{agent.name.upper()}] ✓ Final answer produced")

        # Get read memory IDs
        read_memory_ids = memory_tracker.get_read_entry_ids()

        # Emit memory_read trace event
        if trace_collector and read_memory_ids:
            # Get entry types for the read entries
            all_entries = state.read_memory()
            read_types = []
            for entry in all_entries:
                if entry.id in read_memory_ids:
                    read_types.append(entry.entry_type)
            trace_collector.emit(
                event_type="memory_read",
                step_num=step_num,
                agent_name=agent.name,
                payload={
                    "entry_ids": sorted(read_memory_ids),
                    "entry_types": sorted(set(read_types)),
                },
            )

        # Track memory reads for taint
        taint_tracker.on_memory_read(step_num, read_memory_ids, agent.name)

        # Emit tool_invocation trace events from agent output metadata
        if trace_collector and output.metadata.get("tool_calls"):
            for tc in output.metadata["tool_calls"]:
                output_val = tc.get("output", "")
                output_hash = hashlib.sha256(
                    str(output_val).encode()
                ).hexdigest()[:16]
                trace_collector.emit(
                    event_type="tool_invocation",
                    step_num=step_num,
                    agent_name=agent.name,
                    payload={
                        "tool_name": tc.get("tool_name", ""),
                        "operation": tc.get("operation", ""),
                        "arguments": tc.get("params", {}),
                        "success": tc.get("success", True),
                        "output_hash": output_hash,
                        "output_preview": str(output_val)[:200],
                    },
                )
                # Emit tool_failure alongside the failed invocation
                if not tc.get("success", True):
                    trace_collector.emit(
                        event_type="tool_failure",
                        step_num=step_num,
                        agent_name=agent.name,
                        payload={
                            "tool_name": tc.get("tool_name", ""),
                            "error": str(tc.get("error") or ""),
                        },
                    )

        # Track provenance and taint for output
        if output.memory_entry_id:
            # Get the memory entry that was written
            entries = state.read_memory()
            entry = next(
                (e for e in entries if e.id == output.memory_entry_id), None
            )
            if entry:
                provenance_tracker.track_memory_write(
                    entry,
                    output.get_cited_retrieval_ids(),
                    read_memory_ids,
                )

            # Emit memory_write trace event
            if trace_collector:
                content_hash = hashlib.sha256(
                    output.content.encode()
                ).hexdigest()[:16]
                entry_type = ""
                confidence = 0.0
                if entry:
                    entry_type = entry.entry_type
                    confidence = entry.confidence
                trace_collector.emit(
                    event_type="memory_write",
                    step_num=step_num,
                    agent_name=agent.name,
                    payload={
                        "entry_id": output.memory_entry_id,
                        "entry_type": entry_type,
                        "content_hash": content_hash,
                        "confidence": confidence,
                    },
                )

        taint_tracker.on_agent_output(
            step_num, output, None, read_memory_ids
        )

        # Emit agent_output trace event
        if trace_collector:
            content_hash = hashlib.sha256(
                output.content.encode()
            ).hexdigest()[:16]
            trace_collector.emit(
                event_type="agent_output",
                step_num=step_num,
                agent_name=agent.name,
                payload={
                    "action": output.action,
                    "content_hash": content_hash,
                    "is_final_answer": output.is_final_answer,
                    "citation_ids": [
                        str(rid) for rid in output.get_cited_retrieval_ids()
                    ],
                },
            )

        # Emit final_answer trace event
        if trace_collector and output.is_final_answer:
            answer = state.get_final_answer() or output.content
            answer_hash = hashlib.sha256(
                str(answer).encode()
            ).hexdigest()[:16]
            trace_collector.emit(
                event_type="final_answer",
                step_num=step_num,
                agent_name=agent.name,
                payload={
                    "answer": str(answer),
                    "answer_hash": answer_hash,
                },
            )

        return StepResult(
            step_num=step_num,
            agent_name=agent.name,
            agent_role=agent.role.name if agent.role else "",
            output=output,
            retrieval_result=retrieval_result,
            read_memory_ids=read_memory_ids,
        )
