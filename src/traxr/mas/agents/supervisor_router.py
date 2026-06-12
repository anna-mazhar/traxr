"""Supervisor Router - Central brain for multi-agent orchestration.

Architecture:
- Router is the central controller that makes ALL decisions
- First invokes Planner to create execution plan
- Executes plan step-by-step using dependency-aware scheduling
- Router's LLM judges: completion, conflicts, next steps
- Always calls Synthesizer when concluding

Uses planning.plan_types.ExecutionPlan for DAG-based dependency tracking.
The Router queries get_ready_subtasks() to find subtasks whose dependencies
are satisfied, enabling parallel-safe execution ordering.
"""

from typing import Optional, List, Dict, Any
from collections import defaultdict

from .base import BaseAgent
from ..core.types import AgentRole
from ..core.state import SharedState, MemoryAccessTracker
from ..core.outputs import AgentOutput
from ..retrieval.items import RetrievalResult
from ..planning.plan_types import ExecutionPlan, Subtask

import logging

logger = logging.getLogger(__name__)


class SupervisorRouter(BaseAgent):
    """Router that acts as central brain for multi-agent orchestration.

    Flow:
    1. Router invokes Planner to create execution plan (DAG of subtasks)
    2. Router uses get_ready_subtasks() to find subtasks with met dependencies
    3. Router picks the next ready subtask and dispatches to assigned agent
    4. After agent completes, Router marks subtask completed and checks progress
    5. Router detects conflicts and re-invokes Planner if needed
    6. Router judges when task is complete
    7. Router always calls Synthesizer to conclude
    """

    def __init__(
        self,
        available_agents: List[str],
        llm,  # Required: OpenAI or Tinker client
        max_turns: int = 10,
    ):
        super().__init__(
            name="router",
            role=AgentRole.ROUTER,
            llm=llm,
        )
        self._available_agents = available_agents
        self._max_turns = max_turns

        # Router's state
        self._current_plan: Optional[ExecutionPlan] = None
        self._execution_log: List[Dict[str, Any]] = []
        self._current_turn = 0

        # Track which subtask is currently being executed
        self._current_subtask_id: Optional[str] = None

        # Stuck detection
        self._agent_call_counts = defaultdict(int)
        self._last_memory_hash = None
        self._last_chosen_agent: Optional[str] = None

        # Quality-triggered replanning
        self._pending_replan_reason: Optional[Dict[str, Any]] = None

        # Track confidence-triggered replan (only allow once per task)
        self._confidence_replan_done: bool = False

    @property
    def available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self._available_agents)

    def _needs_planning(self, state: SharedState) -> bool:
        """Check if we need to invoke (or re-invoke) Planner."""
        memory = state.get_all_memory()

        # Need planning if:
        # 1. No plan exists yet (first step)
        if self._current_plan is None:
            return True

        # 2. No plan in memory yet (memory-based check)
        has_plan = any(e.entry_type == "plan" for e in memory)
        if not has_plan:
            return True

        # 3. Plan is stuck (has pending subtasks but none are ready)
        if self._current_plan.is_stuck:
            logger.debug("[Router] Plan is stuck - dependencies cannot be resolved, need replan")
            return True

        # 4. Quality-triggered replan (set by _evaluate_subtask_output)
        if self._pending_replan_reason:
            reason = self._pending_replan_reason.get("reason", "quality issues")
            logger.debug(f"[Router] Quality-triggered replan: {reason}")
            # Store context in state metadata for planner to use
            state.task_input.metadata['replan_context'] = {
                "trigger": "quality_evaluation",
                "reason": reason,
                "failed_agent": self._pending_replan_reason.get("agent"),
                "failed_subtask": self._pending_replan_reason.get("subtask"),
                "issues": self._pending_replan_reason.get("issues", []),
            }
            # Clear the flag so we don't replan infinitely
            self._pending_replan_reason = None
            return True

        return False

    def _build_router_context(self, state: SharedState) -> str:
        """Build comprehensive context for Router's LLM decision."""
        memory = state.get_all_memory()

        # Organize memory by agent and type
        memory_by_agent = defaultdict(list)
        for entry in memory:
            memory_by_agent[entry.agent_name].append(entry)

        context_parts = [
            "=== ROUTER CONTEXT ===",
            "",
            f"TASK: {state.task_input.query}",
            f"TURN: {self._current_turn}/{self._max_turns}",
            "",
        ]

        # File info
        if state.task_input.metadata.get('file_name'):
            file_name = state.task_input.metadata.get('file_name', '')
            file_type = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
            context_parts.append(f"ATTACHED FILE: {file_name} (type: {file_type})")

            # For zip files, show the extracted contents
            if file_type == 'zip' and state.task_input.metadata.get('zip_file_list'):
                extract_dir = state.task_input.metadata.get('zip_extract_dir', '')
                file_list = state.task_input.metadata.get('zip_file_list', [])
                context_parts.append("ZIP CONTENTS (already extracted):")
                for f in file_list:
                    ext = f['filename'].split('.')[-1].lower() if '.' in f['filename'] else 'unknown'
                    context_parts.append(f"  - {extract_dir}/{f['filename']} ({ext})")
            context_parts.append("")

        # Show last error if any (for error-aware routing)
        last_error = state.task_input.metadata.get('last_error')
        if last_error:
            context_parts.append("LAST ERROR DETECTED:")
            context_parts.append(f"  Type: {last_error.get('error_type', 'unknown')}")
            context_parts.append(f"  Suggestion: {last_error.get('suggestion', 'N/A')}")
            context_parts.append("")

        # Current plan — use structured plan summary
        if self._current_plan:
            context_parts.append(self._current_plan.to_text_summary())
            context_parts.append("")

        # Execution log (what's been done)
        if self._execution_log:
            context_parts.append("EXECUTION LOG:")
            for i, log_entry in enumerate(self._execution_log, 1):
                agent_name = log_entry['agent']
                action = log_entry['action']
                result_preview = log_entry['result'][:200] if log_entry['result'] else "N/A"
                recommendation = log_entry.get('recommendation', 'None')
                context_parts.append(f"  Turn {i}:")
                context_parts.append(f"    Agent: {agent_name}")
                context_parts.append(f"    Action: {action}")
                context_parts.append(f"    Result: {result_preview}...")
                context_parts.append(f"    Recommendation: {recommendation}")
            context_parts.append("")

        # Extract any "Final Answer:" from memory for easy visibility
        final_answers = []
        for entry in memory:
            if "Final Answer:" in entry.content:
                # Extract the line with Final Answer
                for line in entry.content.split('\n'):
                    if "Final Answer:" in line:
                        final_answers.append(f"{entry.agent_name}: {line.strip()}")
                        break

        if final_answers:
            context_parts.append("FINAL ANSWERS IDENTIFIED:")
            for ans in final_answers:
                context_parts.append(f"  {ans}")
            context_parts.append("")

        # Full memory details by agent
        if memory:
            context_parts.append("MEMORY BY AGENT:")
            for agent_name, entries in memory_by_agent.items():
                context_parts.append(f"  {agent_name} ({len(entries)} entries):")
                for entry in entries:
                    preview = entry.content[:300].replace('\n', ' ')
                    context_parts.append(f"    [{entry.entry_type}] {preview}...")
            context_parts.append("")

        # Available agents
        context_parts.append("AVAILABLE AGENTS:")
        agent_descriptions = {
            "planner": "Creates execution plan with subtasks and agent assignments",
            "data_analyst": "Analyzes tabular data (CSV, Excel)",
            "python_executor": "Executes complex computations: multi-step calculations, data processing, algorithms, scientific computing. Use for any calculation beyond single arithmetic operations.",
            "calculator": "Basic arithmetic only (single operations like 2+2, 15*3, 100/4). Use ONLY for simple calculations. For multi-step math or complex computations, use python_executor instead.",
            "visual_analyst": "Analyzes images and charts",
            "document_analyst": "Processes documents (PDF, text)",
            "web_researcher": "Searches web for current information, external research, or when information is not in the retrieval database. Use for current events and online content.",
            "audio_analyst": "Transcribes and analyzes audio files (MP3, WAV)",
            "fact_checker": "Verifies factual claims",
            "researcher": "Queries retrieval database for stored information. Use ONLY when retrieval system is explicitly available. For web search, use web_researcher instead.",
            "critic": "Critically reviews work",
            "synthesizer": "Produces final answer (use to conclude)",
            "generalist": "Multi-tool agent for tasks requiring combined file + web + code capabilities",
        }

        for agent in self._available_agents:
            desc = agent_descriptions.get(agent, "Specialized agent")
            # Mark which agents have been used
            used = "✓ USED" if any(log['agent'] == agent for log in self._execution_log) else ""
            context_parts.append(f"  - {agent}: {desc} {used}")

        context_parts.append("")

        return "\n".join(context_parts)

    def _detect_stuck(self, state: SharedState, last_agent: Optional[str] = None) -> Optional[str]:
        """Detect if execution is stuck in a loop without progress.

        Returns: Intervention action if stuck, None otherwise
        """
        if not last_agent:
            return None

        # Get current memory state
        memory = state.get_all_memory()
        memory_content = "".join([e.content for e in memory])
        current_hash = hash(memory_content)

        # Increment counter for this agent
        self._agent_call_counts[last_agent] += 1
        call_count = self._agent_call_counts[last_agent]

        # Check if memory actually changed (progress indicator)
        memory_changed = current_hash != self._last_memory_hash
        self._last_memory_hash = current_hash

        # Debug logging
        logger.debug(f"[Router] Stuck detection: {last_agent} count={call_count}, memory_changed={memory_changed}, all_counts={dict(self._agent_call_counts)}")

        # Stuck conditions
        if call_count >= 3 and not memory_changed:
            # Same agent 3+ times with no new information = definitely stuck
            logger.debug(f"\n[Router] STUCK DETECTED: {last_agent} called {call_count} times without progress")
            return "escalate_replan"
        elif call_count >= 5:
            # Same agent 5+ times, even if memory changes = probably stuck (e.g., repeated failures)
            logger.debug(f"\n[Router] STUCK DETECTED: {last_agent} called {call_count} times (too many attempts)")
            return "escalate_replan"
        elif call_count >= 2 and not memory_changed:
            # 2 calls with no progress = add critic to review
            logger.debug(f"\n[Router] Potential stuck detected: adding critic review")
            return "add_critic"

        return None

    def _detect_error_pattern(self, state: SharedState) -> Optional[Dict[str, Any]]:
        """Detect specific error patterns in recent memory entries.

        Returns actionable error info for error-aware replanning, or None.
        """
        memory = state.get_all_memory()
        if not memory:
            return None

        # Check recent entries (last 3) for error patterns
        recent_entries = sorted(memory, key=lambda e: e.timestamp, reverse=True)[:3]

        for entry in recent_entries:
            content = entry.content.lower()

            # FileNotFoundError - wrong file path
            if 'filenotfounderror' in content or 'no such file or directory' in content:
                # Extract the file name from the error
                import re
                match = re.search(r"'([^']+)'", entry.content)
                missing_file = match.group(1) if match else "unknown"

                # Check if we have zip extract info
                zip_dir = state.task_input.metadata.get('zip_extract_dir')
                zip_files = state.task_input.metadata.get('zip_file_list', [])

                suggestion = "Check file path - the file may be in a different location."
                if zip_dir and zip_files:
                    file_names = [f['filename'] for f in zip_files]
                    suggestion = f"Use full paths from extracted zip: {zip_dir}. Available files: {file_names}"

                return {
                    "error_type": "FileNotFoundError",
                    "missing_file": missing_file,
                    "suggestion": suggestion,
                    "action": "replan_with_correct_paths",
                }

            # KeyError in JSON/dict - wrong key
            if 'keyerror' in content:
                import re
                match = re.search(r"KeyError[:\s]*'([^']+)'", entry.content, re.IGNORECASE)
                missing_key = match.group(1) if match else "unknown"

                # Get actual keys from file inspection
                inspection = state.task_input.metadata.get('file_inspection')
                available_keys = []
                if inspection and hasattr(inspection, 'json_keys'):
                    available_keys = inspection.json_keys

                return {
                    "error_type": "KeyError",
                    "missing_key": missing_key,
                    "available_keys": available_keys,
                    "suggestion": f"Key '{missing_key}' not found. Available keys: {available_keys[:10]}",
                    "action": "replan_with_correct_keys",
                }

            # No DataFrame loaded
            if 'no dataframe' in content or 'df is none' in content or 'df is empty' in content:
                return {
                    "error_type": "NoDataFrame",
                    "suggestion": "DataFrame not loaded. Check file format or try different parsing method.",
                    "action": "try_alternative_parser",
                }

            # No search results
            if 'no results' in content and ('search' in content or 'web' in content):
                return {
                    "error_type": "NoSearchResults",
                    "suggestion": "Search returned no results. Try different search terms or check query.",
                    "action": "reformulate_search",
                }

            # Parse error
            if 'parseerror' in content or 'decode error' in content or 'invalid' in content:
                return {
                    "error_type": "ParseError",
                    "suggestion": "File parsing failed. Try alternative parser or check file format.",
                    "action": "try_alternative_parser",
                }

        return None

    def _check_task_answered(self, state: SharedState) -> bool:
        """Check if a specialist agent has already answered the task.

        Returns True if we should skip to synthesizer because the answer exists.

        Validation checks to prevent premature answer detection:
        - No assumptions introduced (e.g., "I assume", "assuming")
        - No errors or warnings in output
        - Answer appears complete and grounded in data
        """
        memory = state.get_all_memory()
        task_query = state.task_input.query.lower()

        # Agents that produce final answers (not planner/router/critic)
        answer_agents = {
            "data_analyst", "python_executor", "document_analyst",
            "audio_analyst", "visual_analyst", "web_researcher",
            "fact_checker", "researcher", "generalist"
        }

        for entry in memory:
            # Skip non-answer agents and short entries
            if entry.agent_name not in answer_agents:
                continue
            if len(entry.content) < 50:
                continue

            content_lower = entry.content.lower()

            # === VALIDATION: Check for disqualifying patterns ===

            # 1. Check for assumptions - if present, don't finalize
            assumption_phrases = [
                "i assume", "assuming", "i believe", "probably",
                "might be", "could be", "possibly", "i think",
                "not sure", "uncertain", "unclear", "guess"
            ]
            has_assumptions = any(phrase in content_lower for phrase in assumption_phrases)
            if has_assumptions:
                logger.debug(f"[Router] Answer from {entry.agent_name} contains assumptions - not finalizing")
                continue

            # 2. Check for errors or warnings
            error_phrases = [
                "error:", "warning:", "failed to", "could not",
                "unable to", "exception", "traceback", "not found",
                "not available", "missing", "invalid"
            ]
            has_errors = any(phrase in content_lower for phrase in error_phrases)
            if has_errors:
                logger.debug(f"[Router] Answer from {entry.agent_name} contains errors - not finalizing")
                continue

            # 3. Check for incomplete indicators
            incomplete_phrases = [
                "need more", "requires further", "incomplete",
                "todo", "to be determined", "tbd", "work in progress",
                "next step", "should also"
            ]
            is_incomplete = any(phrase in content_lower for phrase in incomplete_phrases)
            if is_incomplete:
                logger.debug(f"[Router] Answer from {entry.agent_name} appears incomplete - not finalizing")
                continue

            # === Check for explicit answer indicators ===
            answer_phrases = [
                "the answer is", "in total,", "in conclusion,",
                "therefore,", "the result is", "final answer:",
                "the ingredients are", "the following"
            ]

            has_answer_phrase = any(phrase in content_lower for phrase in answer_phrases)

            # Check for question-specific patterns
            # "How many" questions - look for numbers with context
            if "how many" in task_query:
                import re
                # Pattern: number followed by relevant noun (slides, items, etc.)
                if re.search(r'\b\d+\s+(slides?|items?|times?|results?|files?|rows?|entries?|mentions?)\b', content_lower):
                    logger.debug(f"\n[Router] Task appears answered by {entry.agent_name} (found count pattern)")
                    return True
                # Pattern: "In total, X" or "total of X"
                if re.search(r'(in total[,:]?\s*\*?\*?\d+|total of \d+|\*\*\d+\s+\w+\*\*)', content_lower):
                    logger.debug(f"\n[Router] Task appears answered by {entry.agent_name} (found total pattern)")
                    return True

            # "What/List/Give" questions about ingredients, items, etc.
            if any(q in task_query for q in ["what are", "list", "give me", "ingredients"]):
                # Look for bullet points or comma-separated lists
                if entry.content.count('\n-') >= 2 or entry.content.count('\n•') >= 2:
                    logger.debug(f"\n[Router] Task appears answered by {entry.agent_name} (found list)")
                    return True
                if has_answer_phrase:
                    logger.debug(f"\n[Router] Task appears answered by {entry.agent_name} (found answer phrase)")
                    return True

            # Generic: if agent explicitly states an answer
            if has_answer_phrase:
                # Make sure it's substantive (not just "the answer is unknown")
                if "unknown" not in content_lower and "cannot" not in content_lower and "unable" not in content_lower:
                    logger.debug(f"\n[Router] Task appears answered by {entry.agent_name} (explicit answer)")
                    return True

        return False

    def _evaluate_subtask_output(
        self,
        agent_name: str,
        output: str,
        subtask_description: str,
    ) -> Dict[str, Any]:
        """Evaluate subtask output quality using Tier 1 rule-based checks.

        Returns evaluation dict with:
        - quality: "good", "warning", "poor"
        - issues: list of detected issues
        - should_replan: bool
        - replan_reason: str (if should_replan)
        """
        if not output:
            return {
                "quality": "poor",
                "issues": ["empty_output"],
                "should_replan": True,
                "replan_reason": f"Agent {agent_name} produced no output for: {subtask_description}",
            }

        output_lower = output.lower()
        issues = []

        # 1. Check for assumptions
        assumption_indicators = [
            "i assume", "assuming that", "presumably", "probably",
            "i think", "might be", "could be", "i believe",
            "if i had to guess", "my guess is", "likely to be",
            "it seems like", "appears to be", "possibly"
        ]
        assumptions_found = [ind for ind in assumption_indicators if ind in output_lower]
        if assumptions_found:
            issues.append(("assumptions", assumptions_found[:3]))  # Limit to first 3

        # 2. Check for errors/failures
        error_indicators = [
            "error:", "failed to", "could not", "unable to",
            "exception:", "traceback", "filenotfounderror",
            "keyerror", "valueerror", "typeerror", "indexerror",
            "timeout", "permission denied", "access denied",
            "not found", "does not exist", "no such file"
        ]
        errors_found = [ind for ind in error_indicators if ind in output_lower]
        if errors_found:
            issues.append(("errors", errors_found[:3]))

        # 3. Check for incompleteness
        incomplete_indicators = [
            "need more information", "requires further", "incomplete",
            "partial result", "couldn't find all", "only found some",
            "missing data", "not enough", "insufficient",
            "todo:", "to be determined", "tbd", "work in progress",
            "will need to", "should also check", "next step would be"
        ]
        incomplete_found = [ind for ind in incomplete_indicators if ind in output_lower]
        if incomplete_found:
            issues.append(("incomplete", incomplete_found[:3]))

        # 4. Check for low confidence
        low_confidence_indicators = [
            "not sure", "uncertain", "unclear", "ambiguous",
            "i don't know", "cannot determine", "no way to tell",
            "hard to say", "difficult to determine", "inconclusive"
        ]
        low_conf_found = [ind for ind in low_confidence_indicators if ind in output_lower]
        if low_conf_found:
            issues.append(("low_confidence", low_conf_found[:3]))

        # Determine overall quality and replan decision
        error_issues = [i for i in issues if i[0] == "errors"]
        assumption_issues = [i for i in issues if i[0] == "assumptions"]
        incomplete_issues = [i for i in issues if i[0] == "incomplete"]

        # Decision matrix
        should_replan = False
        replan_reason = None

        # Errors always trigger replan
        if error_issues:
            should_replan = True
            replan_reason = f"Agent {agent_name} encountered errors: {error_issues[0][1]}"
            quality = "poor"
        # Multiple issue types = poor quality, replan
        elif len(issues) >= 2:
            should_replan = True
            issue_types = [i[0] for i in issues]
            replan_reason = f"Agent {agent_name} output has multiple issues ({', '.join(issue_types)})"
            quality = "poor"
        # Single assumption or incomplete = warning, continue but note it
        elif assumption_issues or incomplete_issues:
            quality = "warning"
            # Don't replan for single warning, but track it
        elif issues:
            quality = "warning"
        else:
            quality = "good"

        return {
            "quality": quality,
            "issues": issues,
            "should_replan": should_replan,
            "replan_reason": replan_reason,
        }

    def _has_verification_pending(self) -> bool:
        """Check if the plan has verification/critic subtasks still pending.

        Used to prevent premature exit when verification steps remain.
        """
        if not self._current_plan:
            return False

        verification_agents = {"critic", "fact_checker", "verifier"}
        verification_keywords = ["verify", "check", "validate", "review", "confirm"]

        for subtask in self._current_plan.subtasks:
            if subtask.status == "pending":
                # Check if it's a verification agent
                if subtask.agent in verification_agents:
                    return True
                # Check if description mentions verification
                desc_lower = subtask.description.lower()
                if any(kw in desc_lower for kw in verification_keywords):
                    return True

        return False

    def _check_low_confidence_entries(
        self,
        state: SharedState,
        threshold: float = 0.65,
    ) -> Optional[Dict[str, Any]]:
        """Check recent memory entries for low confidence signals.

        This enables confidence-triggered replanning: when agents report
        low confidence in their outputs (e.g., due to parsing issues,
        missing data, or execution errors), the router can trigger
        replanning with alternative approaches.

        Args:
            state: Current shared state
            threshold: Confidence threshold below which to trigger replan

        Returns:
            Dict with replan info if low confidence detected, None otherwise
        """
        memory = state.get_all_memory()
        if not memory:
            return None

        # Check the most recent entry from a specialist agent (not router/planner)
        infrastructure_agents = {"router", "planner", "synthesizer", "critic"}

        # Find the most recent specialist entry
        for entry in reversed(memory):
            if entry.agent_name in infrastructure_agents:
                continue

            # Check if entry has confidence information
            confidence = entry.confidence
            status = entry.status

            # Trigger replan if confidence is below threshold
            if confidence < threshold:
                return {
                    "trigger": "low_confidence",
                    "agent": entry.agent_name,
                    "entry_type": entry.entry_type,
                    "confidence": confidence,
                    "status": status,
                    "reason": f"Agent {entry.agent_name} reported low confidence ({confidence:.2f}): {status}",
                }

            # Also check status field for explicit low confidence markers
            if status in ("low_confidence", "failed_attempt"):
                return {
                    "trigger": "low_confidence_status",
                    "agent": entry.agent_name,
                    "entry_type": entry.entry_type,
                    "confidence": confidence,
                    "status": status,
                    "reason": f"Agent {entry.agent_name} marked output as {status}",
                }

            # Only check the most recent specialist entry
            break

        return None

    def _get_next_from_plan(self) -> Optional[str]:
        """Get next agent from the dependency-aware plan.

        Uses get_ready_subtasks() to find subtasks whose dependencies are
        all satisfied. Returns the agent for the first ready subtask,
        or None if no subtasks are ready.
        """
        if not self._current_plan:
            return None

        # If plan is complete, route to synthesizer
        if self._current_plan.is_complete:
            logger.debug("[Router] All plan subtasks complete - routing to synthesizer")
            return "synthesizer"

        # Get ready subtasks (dependencies met)
        ready = self._current_plan.get_ready_subtasks()

        if not ready:
            # Check if there's a subtask currently in progress
            current = self._current_plan.get_current_subtask()
            if current:
                logger.debug(f"[Router] Subtask {current.id} still in progress ({current.agent})")
                return None  # Let the LLM decide
            return None

        # Pick first ready subtask
        next_subtask = ready[0]
        self._current_subtask_id = next_subtask.id
        self._current_plan.mark_in_progress(next_subtask.id)

        agent = next_subtask.agent

        # Validate agent exists
        if agent not in self._available_agents:
            logger.debug(f"[Router] Plan assigned unknown agent '{agent}', skipping subtask {next_subtask.id}")
            self._current_plan.mark_skipped(next_subtask.id, f"Agent '{agent}' not available")
            # Try the next ready subtask recursively
            return self._get_next_from_plan()

        logger.debug(f"[Router] Plan: executing subtask {next_subtask.id} "
              f"({next_subtask.description[:60]}) → {agent}")
        return agent

    def choose_next(
        self,
        state: SharedState,
        current_step: int,
    ) -> str:
        """Router's LLM makes the orchestration decision.

        Returns: Next agent to invoke
        """
        self._current_turn = current_step

        # First, check if Planner just created a plan and update our state
        memory = state.get_all_memory()
        if memory:
            latest = memory[-1]
            if latest.agent_name == "planner" and latest.entry_type == "plan":
                # Only update if this is a NEW plan (not already processed)
                if self._current_plan is None or latest.step > self._current_plan.created_at_step:
                    self.update_plan(latest.content, latest.step, latest.metadata, state)

        # Build comprehensive context
        context = self._build_router_context(state)

        # CRITICAL: Check for stuck state BEFORE planning check
        # This prevents infinite planner loops if planning keeps failing
        stuck_intervention = self._detect_stuck(state, self._last_chosen_agent)
        if stuck_intervention == "escalate_replan":
            # If planner itself is stuck, don't call planner again - conclude instead
            if self._last_chosen_agent == "planner":
                logger.debug("[Router] Planner stuck in loop - forcing conclusion with Synthesizer")
                self._agent_call_counts.clear()
                self._last_chosen_agent = "synthesizer"
                return "synthesizer"
            else:
                logger.debug("[Router] Escalating to replanning due to stuck state")
                self._agent_call_counts.clear()  # Reset counters after intervention
                self._last_chosen_agent = "planner"
                return "planner"
        elif stuck_intervention == "add_critic":
            if "critic" in self._available_agents:
                logger.debug("[Router] Adding critic to review progress")
                self._agent_call_counts.clear()  # Reset counters after intervention
                self._last_chosen_agent = "critic"
                return "critic"

        # CHECK: Error-aware replanning - detect specific errors and respond appropriately
        error_info = self._detect_error_pattern(state)
        if error_info:
            error_type = error_info.get("error_type", "unknown")
            suggestion = error_info.get("suggestion", "")
            action = error_info.get("action", "")

            logger.debug(f"\n[Router] Detected error pattern: {error_type}")
            logger.debug(f"  Suggestion: {suggestion}")

            # Store error info in state metadata for planner to use
            state.task_input.metadata['last_error'] = error_info

            if action == "replan_with_correct_paths":
                # FileNotFoundError - need to replan with correct file paths
                logger.debug("[Router] Error-aware: Replanning with correct file paths")
                self._agent_call_counts.clear()
                self._last_chosen_agent = "planner"
                return "planner"
            elif action == "try_alternative_parser":
                # Parse error - let python_executor try a different approach
                if "python_executor" in self._available_agents:
                    logger.debug("[Router] Error-aware: Trying python_executor with alternative approach")
                    self._agent_call_counts.clear()
                    self._last_chosen_agent = "python_executor"
                    return "python_executor"
            elif action == "reformulate_search":
                # No search results - try web_researcher again with different query
                if "web_researcher" in self._available_agents:
                    logger.debug("[Router] Error-aware: Reformulating search query")
                    # Don't clear counts - let stuck detection catch if it keeps failing
                    self._last_chosen_agent = "web_researcher"
                    return "web_researcher"
            # For other errors, proceed normally - the LLM will handle it

        # CHECK: Low confidence detection - agents report confidence in their outputs
        # When confidence is low, trigger replanning with alternative approach (ONCE only)
        low_confidence_info = self._check_low_confidence_entries(state)
        if low_confidence_info:
            logger.debug(f"\n[Router] Detected low confidence output:")
            logger.debug(f"  Agent: {low_confidence_info.get('agent')}")
            logger.debug(f"  Confidence: {low_confidence_info.get('confidence', 0):.2f}")
            logger.debug(f"  Status: {low_confidence_info.get('status')}")
            logger.debug(f"  Reason: {low_confidence_info.get('reason')}")

            # Only allow ONE confidence-triggered replan per task
            if self._confidence_replan_done:
                logger.debug("[Router] Confidence replan already done - continuing without replan")
                # Clear any pending replan reason to prevent _needs_planning() from triggering
                self._pending_replan_reason = None
                # Clear replan context from metadata
                state.task_input.metadata.pop('replan_context', None)
                # Continue with current plan - don't trigger another replan
            else:
                # Mark that we've done a confidence replan
                self._confidence_replan_done = True

                # Gather completed subtasks to preserve progress
                completed_subtasks = []
                if self._current_plan:
                    for subtask in self._current_plan.subtasks:
                        if subtask.status == "completed":
                            completed_subtasks.append({
                                "id": subtask.id,
                                "description": subtask.description,
                                "agent": subtask.agent,
                                "output_summary": subtask.output_summary,
                            })

                # Store context for planner to use - include completed work
                state.task_input.metadata['replan_context'] = {
                    "trigger": "low_confidence",
                    "reason": low_confidence_info.get("reason"),
                    "failed_agent": low_confidence_info.get("agent"),
                    "confidence_score": low_confidence_info.get("confidence"),
                    "entry_status": low_confidence_info.get("status"),
                    "replan_mode": "continue",  # Signal to replan remaining steps only
                    "completed_subtasks": completed_subtasks,
                }

                # Set pending replan reason (checked by _needs_planning)
                self._pending_replan_reason = low_confidence_info

                logger.debug("[Router] Confidence-triggered: Replanning remaining steps (preserving completed work)")
                self._agent_call_counts.clear()
                self._last_chosen_agent = "planner"
                return "planner"

        # CHECK: Has a specialist agent already answered the task?
        # Only skip to synthesizer if:
        # 1. Task appears answered AND
        # 2. No verification/critic subtasks are pending in the plan
        if self._check_task_answered(state):
            if self._has_verification_pending():
                logger.debug(f"[Router] Task appears answered but verification steps pending - continuing plan")
            elif self._last_chosen_agent != "synthesizer":
                logger.debug(f"[Router] Skipping to synthesizer - task answered and verified")
                self._last_chosen_agent = "synthesizer"
                return "synthesizer"

        # Check if we need planning (but only if not already stuck in planning loop)
        if self._needs_planning(state):
            logger.debug(f"\n[Router Turn {current_step}] Need planning - invoking Planner")
            self._last_chosen_agent = "planner"
            return "planner"

        # Check budget
        if current_step >= self._max_turns - 1:
            logger.debug(f"\n[Router Turn {current_step}] Budget exhausted - concluding with Synthesizer")
            self._last_chosen_agent = "synthesizer"
            return "synthesizer"

        # Try dependency-aware plan dispatch first
        plan_agent = self._get_next_from_plan()
        if plan_agent:
            # Plan has a ready subtask — use it directly
            self._last_chosen_agent = plan_agent
            return plan_agent

        # Fall back to LLM-based decision when plan doesn't have a clear next step
        # (e.g., all subtasks in progress, or LLM needs to judge completion)
        prompt_parts = [
            context,
            "=== ROUTER DECISION ===",
            "",
            "As the central orchestrator, analyze the execution so far and decide the next action.",
            "",
            "DECISION CRITERIA:",
            "1. COMPLETION: Do we have enough evidence to answer the task?",
            "   - Check if any agent has produced a 'Final Answer:' in their output",
            "   - If a clear answer exists → choose 'synthesizer' to conclude",
            "   - If NO clear answer yet → continue executing plan",
            "",
            "2. CONFLICT DETECTION: Are there contradictions or issues in agent outputs?",
            "   - If YES → choose 'planner' to revise strategy",
            "   - If NO → proceed normally",
            "",
            "3. PLAN EXECUTION: Follow the execution plan unless you have reason to deviate",
            "   - Consider agent recommendations",
            "   - Adapt if circumstances changed",
            "",
            "4. BUDGET: Be mindful of remaining turns",
            f"   - Current: {current_step}/{self._max_turns}",
            "   - If running out, prioritize getting to synthesizer",
            "",
            "INSTRUCTIONS:",
            "- Review the 'FINAL ANSWERS IDENTIFIED' section first",
            "- If a final answer exists, the task is likely complete → choose synthesizer",
            "- Review memory and agent outputs carefully",
            "- Detect any conflicts or gaps in information",
            "- Choose the most appropriate next agent",
            "- ALWAYS end with 'synthesizer' when concluding",
            "",
            "OUTPUT FORMAT:",
            "First, provide your reasoning in 2-3 sentences.",
            "Then on a new line write: NEXT_AGENT: <agent_name>",
            "",
            "Example 1 (task complete):",
            "The agent has produced a Final Answer to the task. The information is clear and unambiguous. We have sufficient evidence to conclude.",
            "NEXT_AGENT: synthesizer",
            "",
            "Example 2 (task incomplete):",
            "The agent has analyzed the data but has not yet produced a final answer. We need to continue with the plan.",
            "NEXT_AGENT: data_analyst",
        ]

        prompt = "\n".join(prompt_parts)

        # Get Router's LLM decision
        response = self._llm.generate(
            prompt,
            response_type="route",
        )

        # Parse response
        content = response.content.strip()

        # Extract agent name
        chosen_agent = None
        reasoning = ""
        if "NEXT_AGENT:" in content:
            parts = content.split("NEXT_AGENT:")
            if len(parts) > 1:
                chosen_agent = parts[1].strip().split()[0].lower()
                reasoning = parts[0].strip()

        # Fallback parsing
        if not chosen_agent:
            # Try to find agent name in response
            for agent in self._available_agents:
                if agent in content.lower():
                    chosen_agent = agent
                    reasoning = content
                    break

        # Final fallback
        if not chosen_agent:
            logger.debug(f"[Router Warning] Could not parse agent from response: {content[:100]}")
            # Default to synthesizer if near budget
            if current_step >= self._max_turns - 2:
                chosen_agent = "synthesizer"
                reasoning = "Budget exhausted - forcing conclusion"
            else:
                chosen_agent = "planner"
                reasoning = "Unclear state - re-planning"

        logger.debug(f"\n[Router Turn {current_step}] Decision:")
        logger.debug(f"  Reasoning: {reasoning}")
        logger.debug(f"  Next Agent: {chosen_agent}")

        # Track this choice for stuck detection
        self._last_chosen_agent = chosen_agent

        return chosen_agent

    def log_agent_execution(
        self,
        agent_name: str,
        action: str,
        result: str,
        recommendation: Optional[str] = None,
    ):
        """Log what an agent did and update plan subtask status.

        After an agent completes, this marks the corresponding subtask
        as completed (or failed) in the plan, enabling downstream
        subtasks to become ready.
        """
        self._execution_log.append({
            "turn": self._current_turn,
            "agent": agent_name,
            "action": action,
            "result": result,
            "recommendation": recommendation,
        })

        # Update plan subtask status with quality evaluation
        if self._current_plan and self._current_subtask_id:
            subtask = self._current_plan.get_subtask(self._current_subtask_id)
            if subtask and subtask.agent == agent_name:
                # Evaluate output quality (Tier 1 rule-based checks)
                evaluation = self._evaluate_subtask_output(
                    agent_name=agent_name,
                    output=result or "",
                    subtask_description=subtask.description,
                )

                # Log evaluation results
                quality = evaluation["quality"]
                issues = evaluation["issues"]
                if issues:
                    issue_summary = ", ".join(f"{i[0]}" for i in issues)
                    logger.debug(f"[Router] Subtask output quality: {quality} (issues: {issue_summary})")
                else:
                    logger.debug(f"[Router] Subtask output quality: {quality}")

                # Determine success based on action/result AND quality
                is_failure = (
                    action in ("error", "skip")
                    or quality == "poor"
                )

                if is_failure:
                    failure_reason = evaluation.get("replan_reason") or (
                        result[:200] if result else "Agent reported failure"
                    )
                    self._current_plan.mark_failed(
                        self._current_subtask_id,
                        reason=failure_reason
                    )
                    logger.debug(f"[Router] Subtask {self._current_subtask_id} marked failed")

                    # Set up quality-triggered replan if needed
                    if evaluation["should_replan"]:
                        self._pending_replan_reason = {
                            "reason": evaluation["replan_reason"],
                            "agent": agent_name,
                            "subtask": subtask.description,
                            "issues": issues,
                        }
                else:
                    output_summary = result[:200] if result else None
                    self._current_plan.mark_completed(
                        self._current_subtask_id,
                        output_summary=output_summary
                    )
                    logger.debug(f"[Router] Subtask {self._current_subtask_id} marked completed")

                    # Even if completed, warn about quality issues
                    if quality == "warning" and issues:
                        logger.debug(f"[Router] Warning: Output has minor issues but proceeding")

                self._current_subtask_id = None

    def update_plan(self, plan_content: str, step_num: int, metadata: Optional[Dict] = None, state: Optional[SharedState] = None):
        """Update current execution plan from Planner's output.

        Tries to use the structured plan from metadata first (parsed by PlannerAgent),
        then falls back to parsing the plan text directly.

        In "continue" replan mode, preserves completed subtasks from the previous plan.
        """
        # Check if we're in "continue" replan mode - preserve completed work
        preserved_subtasks = []
        replan_context = None
        if state:
            replan_context = state.task_input.metadata.get('replan_context')
        if replan_context and replan_context.get('replan_mode') == 'continue':
            # Save completed subtasks from current plan
            if self._current_plan:
                for subtask in self._current_plan.subtasks:
                    if subtask.status == "completed":
                        preserved_subtasks.append(subtask)
                logger.debug(f"[Router] Preserving {len(preserved_subtasks)} completed subtasks from previous plan")

            # Clear replan context after using it
            state.task_input.metadata.pop('replan_context', None)

        # Try to use structured plan from metadata (set by PlannerAgent)
        if metadata and "structured_plan" in metadata:
            plan_data = metadata["structured_plan"]
            subtasks = []
            for st_data in plan_data.get("subtasks", []):
                subtasks.append(Subtask(
                    id=st_data.get("id", f"s{len(subtasks) + 1}"),
                    description=st_data.get("description", ""),
                    agent=st_data.get("agent", ""),
                    dependencies=st_data.get("dependencies", []),
                ))
            self._current_plan = ExecutionPlan(
                subtasks=subtasks,
                reasoning=plan_data.get("reasoning", ""),
                created_at_step=step_num,
            )
        else:
            # Parse from text using ExecutionPlan parser (handles JSON and fallback)
            self._current_plan = ExecutionPlan.from_json(
                plan_content, created_at_step=step_num
            )

        # Merge preserved subtasks if in continue mode
        if preserved_subtasks:
            preserved_map = {s.id: s for s in preserved_subtasks}

            # Update status of matching subtasks in the new plan
            for subtask in self._current_plan.subtasks:
                if subtask.id in preserved_map:
                    # Copy completed status and output from preserved subtask
                    preserved = preserved_map[subtask.id]
                    subtask.status = preserved.status
                    subtask.output_summary = preserved.output_summary
                    logger.debug(f"[Router] Preserved status for {subtask.id}: {subtask.status}")

            # Also add any preserved subtasks that aren't in the new plan
            new_ids = {s.id for s in self._current_plan.subtasks}
            unique_preserved = [s for s in preserved_subtasks if s.id not in new_ids]
            if unique_preserved:
                self._current_plan.subtasks = unique_preserved + self._current_plan.subtasks
                logger.debug(f"[Router] Added {len(unique_preserved)} preserved subtasks to new plan")

            # Rebuild subtask map
            self._current_plan._subtask_map = {s.id: s for s in self._current_plan.subtasks}

        # Reset subtask tracking for new plan
        self._current_subtask_id = None

        logger.debug(f"\n[Router] Plan Updated (id={self._current_plan.plan_id}):")
        logger.debug(f"  Reasoning: {self._current_plan.reasoning}")
        logger.debug(f"  Subtasks: {len(self._current_plan.subtasks)}")
        for s in self._current_plan.subtasks:
            status_icon = "✓" if s.status == "completed" else " "
            deps = f" (deps: {', '.join(s.dependencies)})" if s.dependencies else ""
            logger.debug(f"    [{status_icon}] {s.id}. {s.description} → {s.agent}{deps}")

    def reset(self):
        """Reset for new episode."""
        self._current_plan = None
        self._execution_log = []
        self._current_turn = 0
        self._current_subtask_id = None
        self._agent_call_counts = defaultdict(int)
        self._last_memory_hash = None
        self._last_chosen_agent = None
        self._confidence_replan_done = False
        self._pending_replan_reason = None

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Router step: make orchestration decision."""
        # Read memory
        all_entries = memory_tracker.get_all_memory()

        # Update plan if Planner just executed
        if all_entries:
            latest = all_entries[-1]
            if latest.agent_name == "planner" and latest.entry_type == "plan":
                self.update_plan(latest.content, step_num, latest.metadata, state)

        # Make decision
        next_agent = self.choose_next(state, step_num)

        # Create lightweight response for logging
        response = self._llm.generate(
            f"Routing decision at step {step_num}",
            response_type="route",
            context={"agent": next_agent},
        )

        # Build metadata
        metadata = {
            "chosen_agent": next_agent,
            "turn": self._current_turn,
            "budget_remaining": self._max_turns - step_num,
            "has_plan": self._current_plan is not None,
        }

        # Include plan status in metadata
        if self._current_plan:
            metadata["plan_id"] = self._current_plan.plan_id
            metadata["plan_complete"] = self._current_plan.is_complete
            metadata["plan_stuck"] = self._current_plan.is_stuck
            metadata["current_subtask_id"] = self._current_subtask_id

        return AgentOutput(
            agent_name=self.name,
            action="route",
            content=f"Routing to {next_agent}",
            next_agent_suggestion=next_agent,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata=metadata,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "available_agents": self._available_agents,
            "max_turns": self._max_turns,
            "current_turn": self._current_turn,
            "execution_log_length": len(self._execution_log),
        })
        if self._current_plan:
            base["current_plan"] = self._current_plan.to_dict()
        return base
