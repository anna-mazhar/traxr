"""Structured tool-use schema definitions for function calling.

This module defines the schema types used to describe tool capabilities
to LLM clients, enabling structured (non-regex) tool invocation.

The OpenAI client converts these to native function-calling format.
The Tinker client embeds them as JSON in the system prompt.
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameterSchema:
    """Schema for a single tool parameter."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format (used by OpenAI function calling)."""
        schema: Dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class OperationSchema:
    """Schema for a single tool operation (maps to one OpenAI function)."""

    name: str
    description: str
    parameters: List[ToolParameterSchema] = field(default_factory=list)

    def to_openai_function(self, tool_name: str) -> Dict[str, Any]:
        """Convert to OpenAI function-calling format.

        The function name is "{tool_name}_{operation_name}" to avoid
        collisions across tools.
        """
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        function_def: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": f"{tool_name}__{self.name}",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required:
            function_def["function"]["parameters"]["required"] = required

        return function_def

    def to_prompt_description(self, tool_name: str) -> str:
        """Convert to human-readable description for prompt-based invocation."""
        params_desc = []
        for p in self.parameters:
            req = " (required)" if p.required else " (optional)"
            params_desc.append(f"    - {p.name}: {p.type}{req} — {p.description}")
        params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"
        return f"  {tool_name}.{self.name}: {self.description}\n{params_str}"


@dataclass
class ToolSchema:
    """Complete schema for a tool, containing all its operations."""

    name: str
    description: str
    operations: Dict[str, OperationSchema] = field(default_factory=dict)

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all operations to OpenAI function-calling tool list."""
        return [
            op.to_openai_function(self.name)
            for op in self.operations.values()
        ]

    def to_prompt_description(self) -> str:
        """Convert to human-readable description for prompt-based invocation."""
        ops = "\n".join(
            op.to_prompt_description(self.name)
            for op in self.operations.values()
        )
        return f"## {self.name}\n{self.description}\n\nOperations:\n{ops}"


@dataclass
class StructuredToolCall:
    """A structured tool invocation returned by the LLM."""

    tool_name: str
    operation: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def from_openai_function_name(
        cls, function_name: str, arguments: Dict[str, Any], call_id: str = ""
    ) -> "StructuredToolCall":
        """Parse from OpenAI function-calling format.

        OpenAI function names are "{tool_name}__{operation_name}".
        Also handles fallback for "tool.operation" format in case of edge cases.
        """
        if "__" in function_name:
            tool_name, operation = function_name.split("__", 1)
        elif "." in function_name:
            # Fallback: handle "tool.operation" format
            # (should not happen with OpenAI, but defensive coding)
            tool_name, operation = function_name.split(".", 1)
        else:
            tool_name = function_name
            operation = "default"

        return cls(
            tool_name=tool_name,
            operation=operation,
            arguments=arguments,
            call_id=call_id or str(uuid.uuid4())[:8],
        )


@dataclass
class StructuredToolResult:
    """Result of executing a structured tool call."""

    call_id: str
    tool_name: str
    operation: str
    success: bool
    output: Any = None
    error: Optional[str] = None

    def to_message_content(self) -> str:
        """Format result for inclusion in LLM conversation history."""
        if self.success:
            return f"[{self.tool_name}.{self.operation}] Result: {self.output}"
        else:
            return f"[{self.tool_name}.{self.operation}] Error: {self.error}"


def schemas_to_openai_tools(schemas: List[ToolSchema]) -> List[Dict[str, Any]]:
    """Convert a list of ToolSchemas to OpenAI tools format."""
    tools = []
    for schema in schemas:
        tools.extend(schema.to_openai_tools())
    return tools


def schemas_to_prompt(schemas: List[ToolSchema]) -> str:
    """Convert a list of ToolSchemas to a prompt description.

    Used for models that don't support native function calling (e.g., Tinker/Qwen).
    """
    header = (
        "# Available Tools\n\n"
        "You can invoke tools by outputting a JSON block in this exact format:\n"
        "```tool_call\n"
        '{"tool": "<tool_name>", "operation": "<operation>", "arguments": {<args>}}\n'
        "```\n\n"
        "You may make multiple tool calls. After each tool call, you will receive "
        "the result before generating your next response.\n\n"
    )
    descriptions = "\n\n".join(s.to_prompt_description() for s in schemas)
    return header + descriptions
