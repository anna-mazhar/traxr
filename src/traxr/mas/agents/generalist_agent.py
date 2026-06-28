"""GeneralistAgent — multi-tool agent with universal tool access.

Unlike specialized agents that are bound to a single tool, the GeneralistAgent
has access to ALL registered tools via ToolExecutor.get_all_schemas(). It uses
a multi-turn ReAct loop to iteratively invoke tools and reason about results
until it arrives at an answer.

This is useful for GAIA tasks that require combined capabilities
(e.g., file analysis + web search + calculation in a single step).
"""

from typing import Optional, List, Dict, Any

from .base import BaseAgent
from ..core.types import AgentRole
from ..core.state import SharedState, MemoryAccessTracker
from ..core.outputs import AgentOutput
from ..retrieval.items import RetrievalResult
from ..tools.tool_schema import StructuredToolResult

import logging

logger = logging.getLogger(__name__)


class GeneralistAgent(BaseAgent):
    """Agent with access to all registered tools via multi-turn ReAct loop.

    Flow:
    1. Build prompt with task, memory context, and all available tool schemas
    2. Call LLM with generate_with_tools()
    3. Execute any returned tool calls, collect results
    4. Append results to conversation history
    5. Repeat until LLM returns text-only (no more tool calls) or max iterations
    6. Write final analysis to memory

    Because the LLM sees ALL tool schemas, it can combine file analysis,
    web search, code execution, and calculation in a single agent step.
    """

    def __init__(self, llm, tool_executor=None, max_iterations: int = 8):
        super().__init__(
            name="generalist",
            role=AgentRole.RESEARCHER,
            llm=llm,
            tool_executor=tool_executor,
        )
        self._max_iterations = max_iterations

    @property
    def uses_tools(self) -> bool:
        return True

    def get_tool_schemas(self) -> list:
        """Return ALL tool schemas from the ToolExecutor."""
        if self._tool_executor:
            return self._tool_executor.get_all_schemas()
        return []

    def step(
        self,
        state: SharedState,
        memory_tracker: MemoryAccessTracker,
        retrieval_result: Optional[RetrievalResult] = None,
        step_num: int = 0,
    ) -> AgentOutput:
        """Execute multi-turn ReAct loop with all available tools."""
        # Check tools
        if not self._tool_executor:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tools available for generalist agent",
                metadata={"error": "no_tool_executor"},
            )

        tool_schemas = self.get_tool_schemas()
        if not tool_schemas:
            return AgentOutput(
                agent_name=self.name,
                action="skip",
                content="No tool schemas available",
                metadata={"error": "no_tool_schemas"},
            )

        # Read existing memory for context
        all_entries = memory_tracker.get_all_memory()

        # Build initial prompt
        prompt_parts = [
            "You are a versatile research agent with access to multiple tools.",
            f"Task: {state.task_input.query}",
            f"Step: {step_num}",
            "",
        ]

        # File context
        if state.task_input.metadata.get("file_name"):
            file_name = state.task_input.metadata["file_name"]
            file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "unknown"
            prompt_parts.append(f"Attached file: {file_name} (type: {file_type})")
            prompt_parts.append("")

        # Existing memory context
        if all_entries:
            prompt_parts.append("EXISTING INFORMATION:")
            for entry in all_entries:
                preview = entry.content[:300].replace("\n", " ")
                prompt_parts.append(f"  [{entry.agent_name} — {entry.entry_type}] {preview}...")
            prompt_parts.append("")

        # Available data variables (from python tool context)
        python_tool = self._tool_executor.get_tool("python") if self._tool_executor else None
        if python_tool and hasattr(python_tool, "context"):
            ctx = python_tool.context
            available = []
            if ctx.get("df") is not None and hasattr(ctx["df"], "shape"):
                available.append(f"df: DataFrame {ctx['df'].shape}")
            if ctx.get("df_colors") is not None and hasattr(ctx["df_colors"], "shape"):
                available.append(f"df_colors: DataFrame {ctx['df_colors'].shape} (hex color codes)")
            if ctx.get("file_content"):
                available.append(f"file_content: str ({len(ctx['file_content'])} chars)")
            if ctx.get("file_code"):
                available.append(f"file_code: str ({len(ctx['file_code'])} chars)")
            if ctx.get("document_data"):
                available.append(f"document_data: dict with parsed sections")

            # Zip archive context - show extracted files
            if ctx.get("zip_extract_dir") and ctx.get("zip_file_list"):
                extract_dir = ctx["zip_extract_dir"]
                file_list = ctx["zip_file_list"]
                available.append(f"zip_extract_dir: '{extract_dir}' (temp directory with extracted files)")
                available.append(f"zip_file_list: list of {len(file_list)} files")

                # Show actual file paths for direct access
                prompt_parts.append("EXTRACTED ZIP CONTENTS (files already extracted and ready to use):")
                for f in file_list:
                    full_path = f"{extract_dir}/{f['filename']}"
                    prompt_parts.append(f"  - {full_path} ({f['file_size']} bytes)")
                prompt_parts.append("")
                prompt_parts.append("NOTE: The zip file has already been extracted. Use the paths above directly.")
                prompt_parts.append("For example, to read an XML file: ET.parse('{}/filename.xml')".format(extract_dir))
                prompt_parts.append("")

            if available:
                prompt_parts.append("PRE-LOADED VARIABLES (available in python tool):")
                for v in available:
                    prompt_parts.append(f"  - {v}")
                prompt_parts.append("")

        prompt_parts.extend([
            "INSTRUCTIONS:",
            "- Use the available tools to gather information and solve the task",
            "- You may use multiple tools in sequence",
            "- When you have enough information, provide your final analysis as text",
            "- Be precise and cite specific data from tool outputs",
            "",
            "Solve the task step by step.",
        ])

        system_prompt = "\n".join(prompt_parts)

        # Multi-turn ReAct loop
        conversation_history: List[Dict[str, Any]] = []
        all_tool_call_records: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_content = None

        for iteration in range(self._max_iterations):
            # Build prompt for this iteration
            if iteration == 0:
                current_prompt = system_prompt
            else:
                # Append tool results to prompt for next iteration
                history_parts = [system_prompt, "", "--- CONVERSATION HISTORY ---"]
                for turn in conversation_history:
                    if turn["role"] == "assistant":
                        if turn.get("text"):
                            history_parts.append(f"\nAssistant: {turn['text']}")
                        if turn.get("tool_calls"):
                            for tc in turn["tool_calls"]:
                                history_parts.append(
                                    f"\n[Called {tc['tool']}.{tc['operation']}({tc['args']})]"
                                )
                    elif turn["role"] == "tool_result":
                        history_parts.append(f"\nResult: {turn['content'][:1000]}")

                history_parts.append("\n--- YOUR TURN ---")
                history_parts.append("Continue solving the task. Use tools if needed, or provide your final analysis.")
                current_prompt = "\n".join(history_parts)

            # Call LLM with all tool schemas
            response = self._llm.generate_with_tools(
                current_prompt,
                tools=tool_schemas,
                response_type="generalist",
            )
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            logger.debug(f"  [Generalist] Iteration {iteration + 1}/{self._max_iterations}: "
                  f"{'tool_calls' if response.has_tool_calls else 'text-only'}")

            # Record assistant turn in conversation history
            assistant_turn: Dict[str, Any] = {"role": "assistant", "text": response.content}

            if response.has_tool_calls:
                # Execute each tool call
                tool_call_records = []
                for call in response.tool_calls:
                    result = self._tool_executor.execute(
                        call.tool_name, call.operation, **call.arguments
                    )

                    # Build structured result for conversation history
                    structured_result = StructuredToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        operation=call.operation,
                        success=result.success,
                        output=result.output,
                        error=result.error,
                    )

                    tool_call_records.append({
                        "tool": call.tool_name,
                        "operation": call.operation,
                        "args": call.arguments,
                    })

                    all_tool_call_records.append({
                        "iteration": iteration + 1,
                        "tool_name": call.tool_name,
                        "operation": call.operation,
                        "params": call.arguments,  # Use "params" for consistency with other agents
                        "success": result.success,
                        "output": str(result.output)[:500] if result.success else None,
                        "error": result.error,
                    })

                    # Add result to conversation
                    result_content = structured_result.to_message_content()
                    conversation_history.append({
                        "role": "tool_result",
                        "content": result_content,
                    })

                    logger.debug(f"    → {call.tool_name}.{call.operation}: "
                          f"{'OK' if result.success else f'ERROR: {result.error}'}")

                assistant_turn["tool_calls"] = tool_call_records
                conversation_history.append(assistant_turn)
            else:
                # No tool calls — LLM is done reasoning
                conversation_history.append(assistant_turn)
                final_content = response.content
                break

        # If we exhausted iterations without a text-only response, use the last content
        if final_content is None:
            # Collect any text from the last assistant turn
            if conversation_history:
                for turn in reversed(conversation_history):
                    if turn["role"] == "assistant" and turn.get("text"):
                        final_content = turn["text"]
                        break
            if final_content is None:
                final_content = "Exhausted iterations without producing a final analysis."

        # Determine entry type based on what tools were used
        tools_used = set(r["tool_name"] for r in all_tool_call_records)
        if "excel" in tools_used or "python" in tools_used:
            entry_type = "analysis"
        elif "web_search" in tools_used or "web_fetch" in tools_used:
            entry_type = "web_research"
        elif "calculator" in tools_used:
            entry_type = "calculation"
        elif "image" in tools_used:
            entry_type = "visual_analysis"
        elif "audio" in tools_used:
            entry_type = "audio_analysis"
        elif "document" in tools_used or "pdf" in tools_used or "pptx" in tools_used:
            entry_type = "document_summary"
        else:
            entry_type = "note"

        # Confidence based on iteration count and tool success
        successful_calls = sum(1 for r in all_tool_call_records if r["success"])
        total_calls = len(all_tool_call_records)
        if total_calls == 0:
            entry_confidence = 0.3  # No tools used
        elif successful_calls == total_calls:
            entry_confidence = 0.8  # All tools succeeded
        else:
            entry_confidence = 0.5  # Mixed success

        # Write to memory
        success, reason, entry = state.write_memory(
            agent_name=self.name,
            content=final_content,
            entry_type=entry_type,
            cited_retrieval_ids=[],
            metadata={
                "step": step_num,
                "iterations": min(iteration + 1, self._max_iterations),
                "tools_used": list(tools_used),
                "tool_call_count": total_calls,
                "successful_calls": successful_calls,
            },
            status="preliminary",
            confidence=entry_confidence,
        )

        return AgentOutput(
            agent_name=self.name,
            action=f"write_{entry_type}",
            content=final_content,
            citations=[],
            memory_entry_id=entry.id if entry else None,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            metadata={
                "write_success": success,
                "write_reason": reason,
                "specialty": "generalist",
                "iterations": min(iteration + 1, self._max_iterations),
                "tools_used": list(tools_used),
                "tool_calls": all_tool_call_records,
                "entry_type": entry_type,
            },
        )
