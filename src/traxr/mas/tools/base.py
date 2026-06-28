"""Base tool interface for agent tool usage."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .tool_schema import ToolSchema


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Base class for agent tools."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a tool operation.

        Args:
            operation: Operation name (e.g., "load", "filter", "sort")
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with success status and output
        """
        pass

    @abstractmethod
    def get_available_operations(self) -> List[str]:
        """Get list of available operations for this tool."""
        pass

    def get_schema(self) -> "ToolSchema":
        """Get structured schema describing this tool's capabilities.

        Returns a ToolSchema that can be converted to OpenAI function-calling
        format or embedded in prompts for models without native tool support.

        Subclasses should override this to provide accurate schemas. The
        default implementation returns an empty schema.
        """
        from .tool_schema import ToolSchema
        return ToolSchema(
            name=self.name,
            description=f"Tool: {self.name}",
            operations={},
        )


class ToolExecutor:
    """Manages tool execution for agents."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool for use."""
        self._tools[tool.name] = tool

    def execute(self, tool_name: str, operation: str, **kwargs) -> ToolResult:
        """Execute a tool operation.

        Args:
            tool_name: Name of the tool
            operation: Operation to perform
            **kwargs: Operation parameters

        Returns:
            ToolResult
        """
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found"
            )

        tool = self._tools[tool_name]
        return tool.execute(operation, **kwargs)

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a registered tool."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> List["ToolSchema"]:
        """Get schemas from all registered tools.

        Returns:
            List of ToolSchema objects for all tools with non-empty schemas.
        """
        schemas = []
        for tool in self._tools.values():
            schema = tool.get_schema()
            if schema.operations:
                schemas.append(schema)
        return schemas
