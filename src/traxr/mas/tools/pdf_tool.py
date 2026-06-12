"""PDF tool for structured document operations."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .base import BaseTool, ToolResult

# Optional imports
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class PDFTool(BaseTool):
    """Tool for PDF file operations with structured access.

    Provides deterministic operations:
    - load: Load PDF and get metadata
    - get_page_count: Get number of pages
    - get_page_text: Get text from specific page
    - extract_all_text: Get all text from PDF
    - extract_tables: Extract tables from page (if using pdfplumber)
    - get_metadata: Get PDF metadata
    """

    def __init__(self, file_path: Optional[str] = None):
        """Initialize PDFTool.

        Args:
            file_path: Path to PDF file (can be set later via load)
        """
        super().__init__(name="pdf")
        self.file_path = file_path
        self.doc = None  # PyMuPDF document object
        self.page_count = 0
        self.metadata = {}
        # Perturbation support: injected content overrides extracted text
        self._injected_content: Optional[str] = None
        self._injection_active: bool = False

    def inject_perturbed_content(self, content: str) -> None:
        """Inject perturbed content to override PDF text extraction.

        This allows perturbation experiments to modify the extracted text
        while keeping the same file type and processing path.

        Args:
            content: Perturbed text content to return instead of actual PDF text
        """
        self._injected_content = content
        self._injection_active = True

    def clear_injection(self) -> None:
        """Clear any injected content, returning to normal PDF extraction."""
        self._injected_content = None
        self._injection_active = False

    @property
    def has_injected_content(self) -> bool:
        """Check if perturbed content has been injected."""
        return self._injection_active and self._injected_content is not None

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute PDF operation."""
        operations = {
            "load": self._load,
            "get_page_count": self._get_page_count,
            "get_page_text": self._get_page_text,
            "extract_all_text": self._extract_all_text,
            "extract_tables": self._extract_tables,
            "get_metadata": self._get_metadata,
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
        return ["load", "get_page_count", "get_page_text", "extract_all_text", "extract_tables", "get_metadata"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="pdf",
            description="PDF document tool for loading, reading pages, extracting text, tables, and metadata.",
            operations={
                "load": OperationSchema(
                    name="load",
                    description="Load a PDF file and get basic info (page count, title, author).",
                    parameters=[
                        ToolParameterSchema(name="file_path", type="string", description="Path to PDF file", required=False),
                    ],
                ),
                "get_page_count": OperationSchema(
                    name="get_page_count",
                    description="Get the number of pages in the loaded PDF.",
                    parameters=[],
                ),
                "get_page_text": OperationSchema(
                    name="get_page_text",
                    description="Get text content from a specific page.",
                    parameters=[
                        ToolParameterSchema(name="page_num", type="integer", description="Page number (0-indexed)"),
                    ],
                ),
                "extract_all_text": OperationSchema(
                    name="extract_all_text",
                    description="Extract all text from the PDF, optionally limiting to a maximum number of pages.",
                    parameters=[
                        ToolParameterSchema(name="max_pages", type="integer", description="Maximum pages to extract (all if not specified)", required=False),
                    ],
                ),
                "extract_tables": OperationSchema(
                    name="extract_tables",
                    description="Extract tables from a specific page (requires pdfplumber).",
                    parameters=[
                        ToolParameterSchema(name="page_num", type="integer", description="Page number (0-indexed)"),
                    ],
                ),
                "get_metadata": OperationSchema(
                    name="get_metadata",
                    description="Get PDF metadata (title, author, creation date, etc.).",
                    parameters=[],
                ),
            },
        )

    def _load(self, file_path: Optional[str] = None) -> ToolResult:
        """Load PDF file.

        Args:
            file_path: Path to PDF file (optional if set in __init__)

        Returns:
            ToolResult with PDF info
        """
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            return ToolResult(
                success=False,
                output=None,
                error="No file_path provided"
            )

        if not HAS_PYMUPDF:
            return ToolResult(
                success=False,
                output=None,
                error="PyMuPDF required. Install with: pip install PyMuPDF"
            )

        # Load PDF
        self.doc = fitz.open(self.file_path)
        self.page_count = len(self.doc)
        self.metadata = self.doc.metadata

        return ToolResult(
            success=True,
            output=f"Loaded PDF with {self.page_count} pages",
            metadata={
                "page_count": self.page_count,
                "title": self.metadata.get("title", ""),
                "author": self.metadata.get("author", ""),
                "subject": self.metadata.get("subject", ""),
            }
        )

    def _get_page_count(self) -> ToolResult:
        """Get number of pages in PDF."""
        if self.doc is None:
            return ToolResult(success=False, output=None, error="No PDF loaded")

        return ToolResult(
            success=True,
            output=self.page_count,
            metadata={"page_count": self.page_count}
        )

    def _get_page_text(self, page_num: int) -> ToolResult:
        """Get text from specific page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            ToolResult with page text
        """
        if self.doc is None:
            return ToolResult(success=False, output=None, error="No PDF loaded")

        if page_num < 0 or page_num >= self.page_count:
            return ToolResult(
                success=False,
                output=None,
                error=f"Page {page_num} out of range (0-{self.page_count-1})"
            )

        # If perturbed content is injected, return portion for this page
        if self._injection_active and self._injected_content:
            # Split injected content by page markers or divide evenly
            lines = self._injected_content.split('\n')
            lines_per_page = max(1, len(lines) // self.page_count)
            start = page_num * lines_per_page
            end = start + lines_per_page if page_num < self.page_count - 1 else len(lines)
            text = '\n'.join(lines[start:end])
            return ToolResult(
                success=True,
                output=text,
                metadata={
                    "page_num": page_num,
                    "char_count": len(text),
                    "perturbed": True,
                }
            )

        page = self.doc[page_num]
        text = page.get_text()

        return ToolResult(
            success=True,
            output=text,
            metadata={
                "page_num": page_num,
                "char_count": len(text),
            }
        )

    def _extract_all_text(self, max_pages: Optional[int] = None) -> ToolResult:
        """Extract all text from PDF.

        Args:
            max_pages: Maximum number of pages to extract (None = all)

        Returns:
            ToolResult with all text
        """
        if self.doc is None:
            return ToolResult(success=False, output=None, error="No PDF loaded")

        # If perturbed content is injected, return it directly
        if self._injection_active and self._injected_content:
            return ToolResult(
                success=True,
                output=self._injected_content,
                metadata={
                    "pages_extracted": self.page_count,
                    "total_pages": self.page_count,
                    "char_count": len(self._injected_content),
                    "perturbed": True,
                }
            )

        pages_to_extract = min(max_pages, self.page_count) if max_pages else self.page_count

        text_parts = []
        for page_num in range(pages_to_extract):
            page = self.doc[page_num]
            text = page.get_text()
            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        full_text = "\n\n".join(text_parts)

        return ToolResult(
            success=True,
            output=full_text,
            metadata={
                "pages_extracted": pages_to_extract,
                "total_pages": self.page_count,
                "char_count": len(full_text),
            }
        )

    def _extract_tables(self, page_num: int) -> ToolResult:
        """Extract tables from specific page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            ToolResult with list of tables (each table is list of rows)
        """
        if not HAS_PDFPLUMBER:
            return ToolResult(
                success=False,
                output=None,
                error="pdfplumber required for table extraction. Install with: pip install pdfplumber"
            )

        if self.doc is None:
            return ToolResult(success=False, output=None, error="No PDF loaded")

        if page_num < 0 or page_num >= self.page_count:
            return ToolResult(
                success=False,
                output=None,
                error=f"Page {page_num} out of range (0-{self.page_count-1})"
            )

        # Use pdfplumber for table extraction
        with pdfplumber.open(self.file_path) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()

        if not tables:
            return ToolResult(
                success=True,
                output=[],
                metadata={"page_num": page_num, "table_count": 0}
            )

        return ToolResult(
            success=True,
            output=tables,
            metadata={
                "page_num": page_num,
                "table_count": len(tables),
            }
        )

    def _get_metadata(self) -> ToolResult:
        """Get PDF metadata."""
        if self.doc is None:
            return ToolResult(success=False, output=None, error="No PDF loaded")

        metadata = {
            "page_count": self.page_count,
            "title": self.metadata.get("title", ""),
            "author": self.metadata.get("author", ""),
            "subject": self.metadata.get("subject", ""),
            "creator": self.metadata.get("creator", ""),
            "producer": self.metadata.get("producer", ""),
            "creation_date": self.metadata.get("creationDate", ""),
        }

        return ToolResult(
            success=True,
            output=metadata,
            metadata=metadata
        )

    def __del__(self):
        """Clean up PDF document."""
        if self.doc is not None:
            self.doc.close()
