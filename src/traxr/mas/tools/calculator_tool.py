"""Calculator tool for deterministic mathematical operations."""

import math
from typing import List, Union
from .base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations.

    Provides deterministic operations:
    - eval: Evaluate mathematical expression
    - sum: Sum of numbers
    - mean: Average of numbers
    - min/max: Min/max of numbers
    - round: Round number to decimals
    """

    def __init__(self) -> None:
        super().__init__(name="calculator")
        # Safe math context
        self.math_context = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute calculator operation."""
        operations = {
            "eval": self._eval,
            "sum": self._sum,
            "mean": self._mean,
            "min": self._min,
            "max": self._max,
            "round": self._round,
        }

        if operation not in operations:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation '{operation}'. Available: {list(operations.keys())}"
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error in {operation}: {str(e)}"
            )

    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return ["eval", "sum", "mean", "min", "max", "round"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="calculator",
            description="Mathematical calculator for evaluating expressions and computing statistics.",
            operations={
                "eval": OperationSchema(
                    name="eval",
                    description="Evaluate a mathematical expression (supports +, -, *, /, sqrt, log, exp, sin, cos, tan, pi, e).",
                    parameters=[
                        ToolParameterSchema(name="expression", type="string", description="Mathematical expression to evaluate, e.g. '2 + 3 * 4' or 'sqrt(16) + log(100)'"),
                    ],
                ),
                "sum": OperationSchema(
                    name="sum",
                    description="Compute the sum of a list of numbers.",
                    parameters=[ToolParameterSchema(name="numbers", type="array", description="List of numbers to sum")],
                ),
                "mean": OperationSchema(
                    name="mean",
                    description="Compute the arithmetic mean (average) of a list of numbers.",
                    parameters=[ToolParameterSchema(name="numbers", type="array", description="List of numbers to average")],
                ),
                "min": OperationSchema(
                    name="min",
                    description="Find the minimum value in a list of numbers.",
                    parameters=[ToolParameterSchema(name="numbers", type="array", description="List of numbers")],
                ),
                "max": OperationSchema(
                    name="max",
                    description="Find the maximum value in a list of numbers.",
                    parameters=[ToolParameterSchema(name="numbers", type="array", description="List of numbers")],
                ),
                "round": OperationSchema(
                    name="round",
                    description="Round a number to a specified number of decimal places.",
                    parameters=[
                        ToolParameterSchema(name="number", type="number", description="Number to round"),
                        ToolParameterSchema(name="decimals", type="integer", description="Number of decimal places", required=False, default=0),
                    ],
                ),
            },
        )

    def _eval(self, expression: str) -> ToolResult:
        """Evaluate mathematical expression.

        Args:
            expression: Math expression string (e.g., "2 + 3 * 4")

        Returns:
            ToolResult with computed value
        """
        try:
            result = eval(expression, {"__builtins__": {}}, self.math_context)
            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Evaluation error: {str(e)}"
            )

    def _sum(self, numbers: List[Union[int, float]]) -> ToolResult:
        """Sum of numbers."""
        result = sum(numbers)
        return ToolResult(
            success=True,
            output=result,
            metadata={"count": len(numbers)}
        )

    def _mean(self, numbers: List[Union[int, float]]) -> ToolResult:
        """Mean/average of numbers."""
        if not numbers:
            return ToolResult(success=False, output=None, error="Empty list")
        result = sum(numbers) / len(numbers)
        return ToolResult(
            success=True,
            output=result,
            metadata={"count": len(numbers)}
        )

    def _min(self, numbers: List[Union[int, float]]) -> ToolResult:
        """Minimum of numbers."""
        if not numbers:
            return ToolResult(success=False, output=None, error="Empty list")
        result = min(numbers)
        return ToolResult(success=True, output=result)

    def _max(self, numbers: List[Union[int, float]]) -> ToolResult:
        """Maximum of numbers."""
        if not numbers:
            return ToolResult(success=False, output=None, error="Empty list")
        result = max(numbers)
        return ToolResult(success=True, output=result)

    def _round(self, number: Union[int, float], decimals: int = 0) -> ToolResult:
        """Round number to specified decimals."""
        result = round(number, decimals)
        return ToolResult(
            success=True,
            output=result,
            metadata={"decimals": decimals}
        )
