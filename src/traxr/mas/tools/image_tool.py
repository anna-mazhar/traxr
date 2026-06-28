"""Image tool for structured image access."""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolResult

# Lazy import PIL to avoid logging module conflict
HAS_PIL = None

def _check_pil():
    """Check if PIL is available (lazy check)."""
    global HAS_PIL
    if HAS_PIL is None:
        try:
            from PIL import Image
            HAS_PIL = True
        except ImportError:
            HAS_PIL = False
    return HAS_PIL


class ImageTool(BaseTool):
    """Tool for image file operations with structured access.

    Provides deterministic operations:
    - load: Load image and get metadata
    - get_base64: Get base64-encoded image (for vision models)
    - get_metadata: Get image metadata (dimensions, format, mode)
    - get_dimensions: Get width and height
    """

    def __init__(self, file_path: Optional[str] = None):
        """Initialize ImageTool.

        Args:
            file_path: Path to image file (can be set later via load)
        """
        super().__init__(name="image")
        self.file_path = file_path
        self.image = None  # PIL Image object
        self.metadata = {}

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute image operation."""
        operations = {
            "load": self._load,
            "get_base64": self._get_base64,
            "get_metadata": self._get_metadata,
            "get_dimensions": self._get_dimensions,
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
        return ["load", "get_base64", "get_metadata", "get_dimensions"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="image",
            description="Image tool for loading images, getting base64 encoding (for vision models), and reading metadata.",
            operations={
                "load": OperationSchema(
                    name="load",
                    description="Load an image file and get basic info (format, dimensions, mode).",
                    parameters=[
                        ToolParameterSchema(name="file_path", type="string", description="Path to image file", required=False),
                    ],
                ),
                "get_base64": OperationSchema(
                    name="get_base64",
                    description="Get base64-encoded image data for use with vision models.",
                    parameters=[
                        ToolParameterSchema(name="format", type="string", description="Output format (png, jpeg, etc.)", required=False),
                    ],
                ),
                "get_metadata": OperationSchema(
                    name="get_metadata",
                    description="Get image metadata (format, mode, width, height, file size).",
                    parameters=[],
                ),
                "get_dimensions": OperationSchema(
                    name="get_dimensions",
                    description="Get image width and height.",
                    parameters=[],
                ),
            },
        )

    def _load(self, file_path: Optional[str] = None) -> ToolResult:
        """Load image file.

        Args:
            file_path: Path to image file (optional if set in __init__)

        Returns:
            ToolResult with image info
        """
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            return ToolResult(
                success=False,
                output=None,
                error="No file_path provided"
            )

        if not _check_pil():
            return ToolResult(
                success=False,
                output=None,
                error="Pillow required. Install with: pip install Pillow"
            )

        # Load image
        from PIL import Image
        self.image = Image.open(self.file_path)
        self.metadata = {
            "format": self.image.format,
            "mode": self.image.mode,
            "width": self.image.width,
            "height": self.image.height,
            "size_bytes": Path(self.file_path).stat().st_size,
        }

        return ToolResult(
            success=True,
            output=f"Loaded {self.metadata['format']} image: {self.metadata['width']}x{self.metadata['height']}",
            metadata=self.metadata
        )

    def _get_base64(self, format: Optional[str] = None) -> ToolResult:
        """Get base64-encoded image.

        Args:
            format: Output format (png, jpeg, etc.). Defaults to original format.

        Returns:
            ToolResult with base64 string and media type
        """
        if self.image is None:
            return ToolResult(success=False, output=None, error="No image loaded")

        # Use original format if not specified
        output_format = format or self.metadata.get("format", "PNG")

        # Read file and encode
        with open(self.file_path, 'rb') as f:
            image_data = f.read()

        base64_string = base64.b64encode(image_data).decode('utf-8')

        # Determine media type
        media_types = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'GIF': 'image/gif',
            'WEBP': 'image/webp',
            'BMP': 'image/bmp',
        }
        media_type = media_types.get(output_format.upper(), 'image/png')

        return ToolResult(
            success=True,
            output={
                "base64": base64_string,
                "media_type": media_type,
                "data_url": f"data:{media_type};base64,{base64_string}",
            },
            metadata={
                "format": output_format,
                "size_bytes": len(image_data),
            }
        )

    def _get_metadata(self) -> ToolResult:
        """Get image metadata."""
        if self.image is None:
            return ToolResult(success=False, output=None, error="No image loaded")

        return ToolResult(
            success=True,
            output=self.metadata,
            metadata=self.metadata
        )

    def _get_dimensions(self) -> ToolResult:
        """Get image dimensions."""
        if self.image is None:
            return ToolResult(success=False, output=None, error="No image loaded")

        dimensions = {
            "width": self.metadata["width"],
            "height": self.metadata["height"],
        }

        return ToolResult(
            success=True,
            output=dimensions,
            metadata=dimensions
        )

    def __del__(self):
        """Clean up image."""
        if self.image is not None:
            self.image.close()
