"""Structured plan data types with dependency-aware execution.

Provides Subtask and ExecutionPlan for DAG-based task orchestration.
The Router queries get_ready_subtasks() to find subtasks whose
dependencies are satisfied, enabling parallel-safe execution ordering.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Subtask:
    """A single subtask within an execution plan."""

    id: str
    description: str
    agent: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | completed | failed | skipped
    output_summary: Optional[str] = None
    max_retries: int = 1
    retry_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "agent": self.agent,
            "dependencies": self.dependencies,
            "status": self.status,
            "output_summary": self.output_summary,
        }


class ExecutionPlan:
    """DAG-based execution plan with dependency tracking.

    Subtasks form a directed acyclic graph (DAG) where each subtask may
    depend on zero or more other subtasks. The Router uses this to
    determine which subtasks are ready to execute.

    Usage:
        plan = ExecutionPlan.from_json(llm_response)
        ready = plan.get_ready_subtasks()     # subtasks with met deps
        plan.mark_completed("s1", "Found 42 items")
        ready = plan.get_ready_subtasks()     # now s2/s3 may be ready
    """

    def __init__(
        self,
        subtasks: List[Subtask],
        reasoning: str = "",
        plan_id: Optional[str] = None,
        created_at_step: int = 0,
    ):
        self.subtasks = subtasks
        self.reasoning = reasoning
        self.plan_id = plan_id or str(uuid.uuid4())[:8]
        self.created_at_step = created_at_step
        self._subtask_map: Dict[str, Subtask] = {s.id: s for s in subtasks}

    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """Get a subtask by ID."""
        return self._subtask_map.get(subtask_id)

    def get_ready_subtasks(self) -> List[Subtask]:
        """Get subtasks whose dependencies are all completed/skipped.

        Returns subtasks that are 'pending' and whose dependencies have
        all been marked completed or skipped.
        """
        ready = []
        for subtask in self.subtasks:
            if subtask.status != "pending":
                continue
            deps_met = all(
                self._subtask_map.get(dep_id, Subtask(id=dep_id, description="", agent="")).status
                in ("completed", "skipped")
                for dep_id in subtask.dependencies
            )
            if deps_met:
                ready.append(subtask)
        return ready

    def get_current_subtask(self) -> Optional[Subtask]:
        """Get the subtask currently in_progress, if any."""
        for subtask in self.subtasks:
            if subtask.status == "in_progress":
                return subtask
        return None

    def get_next_subtask(self) -> Optional[Subtask]:
        """Get the next subtask to execute (first ready subtask).

        Convenience method for sequential execution.
        """
        ready = self.get_ready_subtasks()
        return ready[0] if ready else None

    def mark_in_progress(self, subtask_id: str) -> bool:
        """Mark a subtask as in progress."""
        subtask = self._subtask_map.get(subtask_id)
        if subtask and subtask.status == "pending":
            subtask.status = "in_progress"
            return True
        return False

    def mark_completed(self, subtask_id: str, output_summary: Optional[str] = None) -> bool:
        """Mark a subtask as completed with optional output summary."""
        subtask = self._subtask_map.get(subtask_id)
        if subtask and subtask.status in ("pending", "in_progress"):
            subtask.status = "completed"
            subtask.output_summary = output_summary
            return True
        return False

    def mark_failed(self, subtask_id: str, reason: Optional[str] = None) -> bool:
        """Mark a subtask as failed.

        If the subtask has retries remaining, it is reset to pending instead.
        """
        subtask = self._subtask_map.get(subtask_id)
        if not subtask:
            return False

        subtask.retry_count += 1
        if subtask.retry_count < subtask.max_retries:
            subtask.status = "pending"
            subtask.output_summary = f"Retry {subtask.retry_count}: {reason}"
        else:
            subtask.status = "failed"
            subtask.output_summary = reason
        return True

    def mark_skipped(self, subtask_id: str, reason: Optional[str] = None) -> bool:
        """Mark a subtask as skipped (e.g., not needed)."""
        subtask = self._subtask_map.get(subtask_id)
        if subtask and subtask.status == "pending":
            subtask.status = "skipped"
            subtask.output_summary = reason
            return True
        return False

    @property
    def is_complete(self) -> bool:
        """Check if all subtasks are in a terminal state."""
        return all(
            s.status in ("completed", "failed", "skipped")
            for s in self.subtasks
        )

    @property
    def has_pending(self) -> bool:
        """Check if any subtasks are still pending or in progress."""
        return any(
            s.status in ("pending", "in_progress")
            for s in self.subtasks
        )

    @property
    def is_stuck(self) -> bool:
        """Check if the plan is stuck (has pending but no ready subtasks).

        This happens when remaining subtasks have unmet dependencies
        that will never be resolved (e.g., a dependency failed).
        """
        if not self.has_pending:
            return False
        return len(self.get_ready_subtasks()) == 0 and self.get_current_subtask() is None

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "reasoning": self.reasoning,
            "created_at_step": self.created_at_step,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "is_complete": self.is_complete,
        }

    def to_text_summary(self) -> str:
        """Human-readable plan summary for prompts."""
        lines = [f"Plan (id={self.plan_id}):"]
        if self.reasoning:
            lines.append(f"  Reasoning: {self.reasoning}")
        for s in self.subtasks:
            status_icon = {
                "pending": " ",
                "in_progress": ">",
                "completed": "x",
                "failed": "!",
                "skipped": "-",
            }.get(s.status, "?")
            deps = f" (deps: {', '.join(s.dependencies)})" if s.dependencies else ""
            lines.append(f"  [{status_icon}] {s.id}: {s.description} -> {s.agent}{deps}")
        return "\n".join(lines)

    @classmethod
    def from_json(cls, json_str: str, created_at_step: int = 0) -> "ExecutionPlan":
        """Parse an ExecutionPlan from JSON output by the LLM.

        Expected format:
        {
            "reasoning": "...",
            "subtasks": [
                {"id": "s1", "description": "...", "agent": "...", "dependencies": []},
                {"id": "s2", "description": "...", "agent": "...", "dependencies": ["s1"]}
            ]
        }

        Falls back gracefully on parse errors.
        """
        import re

        # Strip <think>...</think> blocks (Qwen chain-of-thought reasoning)
        cleaned_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL).strip()

        try:
            data = json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_str, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return cls._fallback_parse(json_str, created_at_step)
            else:
                # Try to find JSON object anywhere in the string
                json_match = re.search(r'\{[^{}]*"subtasks"\s*:\s*\[.*?\]\s*\}', cleaned_str, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        return cls._fallback_parse(json_str, created_at_step)
                else:
                    return cls._fallback_parse(json_str, created_at_step)

        reasoning = data.get("reasoning", "")
        subtasks = []
        for st_data in data.get("subtasks", []):
            subtasks.append(Subtask(
                id=st_data.get("id", f"s{len(subtasks) + 1}"),
                description=st_data.get("description", ""),
                agent=st_data.get("agent", ""),
                dependencies=st_data.get("dependencies", []),
            ))

        return cls(
            subtasks=subtasks,
            reasoning=reasoning,
            created_at_step=created_at_step,
        )

    @classmethod
    def _fallback_parse(cls, text: str, created_at_step: int = 0) -> "ExecutionPlan":
        """Fallback parser for non-JSON plan text.

        Handles the existing format:
            Subtask 1: description -> agent_name
        """
        import re
        subtasks = []
        reasoning_lines = []
        in_subtasks = False

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Look for subtask/step pattern with arrow
            if any(indicator in line.lower() for indicator in ["subtask", "step"]):
                in_subtasks = True
                if "→" in line or "->" in line:
                    parts = line.split("→" if "→" in line else "->")
                    if len(parts) == 2:
                        desc = parts[0].strip().lstrip("0123456789. ")
                        # Remove "Subtask N:" prefix
                        desc = re.sub(r'^(?:sub)?task\s*\d*[:.]\s*', '', desc, flags=re.IGNORECASE)
                        agent = parts[1].strip().lower()
                        subtask_id = f"s{len(subtasks) + 1}"
                        # Sequential dependency on previous subtask
                        deps = [f"s{len(subtasks)}"] if subtasks else []
                        subtasks.append(Subtask(
                            id=subtask_id,
                            description=desc,
                            agent=agent,
                            dependencies=deps,
                        ))
            elif not in_subtasks:
                reasoning_lines.append(line)

        reasoning = " ".join(reasoning_lines) if reasoning_lines else "Execution plan created"

        return cls(
            subtasks=subtasks,
            reasoning=reasoning,
            created_at_step=created_at_step,
        )

    @classmethod
    def from_subtask_list(
        cls,
        subtask_dicts: List[Dict[str, str]],
        reasoning: str = "",
        created_at_step: int = 0,
    ) -> "ExecutionPlan":
        """Create from the existing router's subtask list format.

        This provides backward compatibility with the existing
        SupervisorRouter.ExecutionPlan format:
            [{"subtask": "description", "agent": "agent_name"}, ...]
        """
        subtasks = []
        for i, st_dict in enumerate(subtask_dicts):
            subtask_id = f"s{i + 1}"
            # Sequential dependencies by default
            deps = [f"s{i}"] if i > 0 else []
            subtasks.append(Subtask(
                id=subtask_id,
                description=st_dict.get("subtask", ""),
                agent=st_dict.get("agent", ""),
                dependencies=deps,
            ))

        return cls(
            subtasks=subtasks,
            reasoning=reasoning,
            created_at_step=created_at_step,
        )
