"""Document tool for intelligent document parsing and retrieval.

Provides structured access to documents:
- get_summary: See document structure at a glance
- get_section: Retrieve specific section content
- search: Find content by keyword
- get_structured_data: Get parsed data as dictionary
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from .base import BaseTool, ToolResult


@dataclass
class DocumentSection:
    """A section of a document."""
    title: str
    content: str
    items: List[str] = field(default_factory=list)
    section_type: str = "text"  # text, list, key_value, empty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "type": self.section_type,
            "item_count": len(self.items) if self.items else None,
            "content_length": len(self.content)
        }


class DocumentTool(BaseTool):
    """Tool for structured document access.

    Operations:
    - get_summary: Overview of document structure
    - get_section: Get content of a specific section
    - get_all_sections: List all section titles
    - search: Search for keywords in document
    - get_structured_data: Get parsed key-value data
    """

    def __init__(self, file_path: Optional[str] = None):
        super().__init__(name="document")
        self.file_path = file_path
        self.raw_text: str = ""
        self.sections: List[DocumentSection] = []
        self._loaded = False
        # Perturbation support: injected content overrides loaded text
        self._injected_content: Optional[str] = None
        self._injection_active: bool = False

    def inject_perturbed_content(self, content: str) -> None:
        """Inject perturbed content to override document text extraction.

        This allows perturbation experiments to modify the extracted text
        while keeping the same file type and processing path.

        Args:
            content: Perturbed text content to use instead of actual document text
        """
        self._injected_content = content
        self._injection_active = True

    def clear_injection(self) -> None:
        """Clear any injected content, returning to normal document extraction."""
        self._injected_content = None
        self._injection_active = False

    @property
    def has_injected_content(self) -> bool:
        """Check if perturbed content has been injected."""
        return self._injection_active and self._injected_content is not None

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a document operation."""
        operations = {
            "load": self._load,
            "get_summary": self._get_summary,
            "get_section": self._get_section,
            "get_all_sections": self._get_all_sections,
            "search": self._search,
            "get_structured_data": self._get_structured_data,
            "get_raw": self._get_raw,
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
            return ToolResult(success=False, output=None, error=str(e))

    def get_available_operations(self) -> List[str]:
        return ["load", "get_summary", "get_section", "get_all_sections",
                "search", "get_structured_data", "get_raw"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="document",
            description="Document tool for parsing and querying text documents (.txt, .md, .docx). Provides section-based access, keyword search, and structured data extraction.",
            operations={
                "load": OperationSchema(
                    name="load",
                    description="Load and parse a document file into sections.",
                    parameters=[
                        ToolParameterSchema(name="file_path", type="string", description="Path to document file", required=False),
                    ],
                ),
                "get_summary": OperationSchema(
                    name="get_summary",
                    description="Get an overview of the document structure (sections, types, sizes).",
                    parameters=[],
                ),
                "get_section": OperationSchema(
                    name="get_section",
                    description="Get content of a specific section by title (case-insensitive partial match).",
                    parameters=[
                        ToolParameterSchema(name="title", type="string", description="Section title to find"),
                    ],
                ),
                "get_all_sections": OperationSchema(
                    name="get_all_sections",
                    description="List all section titles and their types.",
                    parameters=[],
                ),
                "search": OperationSchema(
                    name="search",
                    description="Search for a keyword in the document. Returns matching sections and items.",
                    parameters=[
                        ToolParameterSchema(name="query", type="string", description="Keyword to search for"),
                        ToolParameterSchema(name="context_lines", type="integer", description="Number of context lines around matches", required=False, default=2),
                    ],
                ),
                "get_structured_data": OperationSchema(
                    name="get_structured_data",
                    description="Get the document as a structured dictionary with sections as keys.",
                    parameters=[],
                ),
                "get_raw": OperationSchema(
                    name="get_raw",
                    description="Get the raw document text.",
                    parameters=[],
                ),
            },
        )

    def _load(self, file_path: Optional[str] = None) -> ToolResult:
        """Load and parse a document."""
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            return ToolResult(success=False, output=None, error="No file path provided")

        path = Path(self.file_path)
        if not path.exists():
            return ToolResult(success=False, output=None, error=f"File not found: {self.file_path}")

        ext = path.suffix.lower()

        # Check for injected perturbed content (for perturbation experiments)
        if self._injection_active and self._injected_content:
            text = self._injected_content
            self.raw_text = text
            self.sections = self._parse_sections(text)
            self._loaded = True
            return ToolResult(
                success=True,
                output=f"Loaded document: {path.name} ({len(self.sections)} sections, {len(text)} chars) [perturbed]",
                metadata={"sections": len(self.sections), "chars": len(text), "perturbed": True}
            )

        # Load text based on file type
        if ext == '.docx':
            text = self._load_docx(self.file_path)
        elif ext == '.txt' or ext == '.md':
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            return ToolResult(success=False, output=None, error=f"Unsupported file type: {ext}")

        self.raw_text = text
        self.sections = self._parse_sections(text)
        self._loaded = True

        return ToolResult(
            success=True,
            output=f"Loaded document: {path.name} ({len(self.sections)} sections, {len(text)} chars)",
            metadata={"sections": len(self.sections), "chars": len(text)}
        )

    def _load_docx(self, file_path: str) -> str:
        """Load text from a .docx file."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            return '\n'.join(p.text for p in doc.paragraphs)
        except ImportError:
            # Fallback: .docx is a ZIP with XML
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as z:
                if 'word/document.xml' in z.namelist():
                    xml = z.read('word/document.xml').decode('utf-8')
                    parts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml)
                    return ' '.join(parts)
            return ""

    def _parse_sections(self, text: str) -> List[DocumentSection]:
        """Parse text into sections."""
        lines = text.split('\n')
        sections = []
        current_title = None
        current_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if self._is_header(stripped, lines, i):
                # Save previous section
                if current_title is not None:
                    section = self._create_section(current_title, current_lines)
                    sections.append(section)
                current_title = stripped.rstrip(':')
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_title is not None:
            sections.append(self._create_section(current_title, current_lines))
        elif current_lines:
            sections.append(self._create_section("Content", current_lines))

        return sections

    def _is_header(self, line: str, all_lines: List[str], idx: int) -> bool:
        """Check if line is a section header.

        Simple heuristics - not perfect, but provides useful structure:
        - Short line (1-2 words) with blank before
        - Line ending with colon (single word only)
        """
        if not line or len(line) > 40:
            return False

        words = line.split()

        # "Section:" pattern - single word ending with colon
        if line.endswith(':') and len(words) == 1:
            return True

        # Skip lines with colon in middle (key: value data)
        if ':' in line:
            return False

        # Short line (1-2 words) with blank before
        if len(words) <= 2:
            has_blank_before = idx == 0 or not all_lines[idx - 1].strip()
            has_blank_after = idx + 1 < len(all_lines) and not all_lines[idx + 1].strip()
            if has_blank_before and has_blank_after:
                return True

        return False

    def _create_section(self, title: str, lines: List[str]) -> DocumentSection:
        """Create a section from lines."""
        # Strip blank lines
        while lines and not lines[0].strip():
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines = lines[:-1]

        content = '\n'.join(lines)
        non_empty = [l.strip() for l in lines if l.strip()]

        if not non_empty:
            return DocumentSection(title=title, content="", items=[], section_type="empty")

        # Detect type
        kv_count = sum(1 for l in non_empty if ':' in l and len(l.split(':')[0]) < 30)
        if kv_count >= len(non_empty) * 0.7:
            return DocumentSection(title=title, content=content, items=non_empty, section_type="key_value")

        avg_words = sum(len(l.split()) for l in non_empty) / len(non_empty)
        if avg_words <= 5 and len(non_empty) >= 2:
            return DocumentSection(title=title, content=content, items=non_empty, section_type="list")

        return DocumentSection(title=title, content=content, items=non_empty, section_type="text")

    def _get_summary(self) -> ToolResult:
        """Get document structure summary."""
        if not self._loaded:
            return ToolResult(success=False, output=None, error="Document not loaded. Call 'load' first.")

        lines = ["DOCUMENT STRUCTURE:", ""]
        for i, section in enumerate(self.sections, 1):
            if section.section_type == "list":
                preview = ", ".join(section.items[:3])
                if len(section.items) > 3:
                    preview += f"... ({len(section.items)} items)"
                lines.append(f"{i}. {section.title} [LIST]: {preview}")
            elif section.section_type == "key_value":
                lines.append(f"{i}. {section.title} [KEY-VALUE]: {len(section.items)} entries")
            elif section.section_type == "empty":
                lines.append(f"{i}. {section.title} [EMPTY]")
            else:
                words = len(section.content.split())
                lines.append(f"{i}. {section.title} [TEXT]: {words} words")

        return ToolResult(success=True, output="\n".join(lines))

    def _get_section(self, title: str) -> ToolResult:
        """Get content of a specific section."""
        if not self._loaded:
            return ToolResult(success=False, output=None, error="Document not loaded")

        title_lower = title.lower()
        for section in self.sections:
            if title_lower in section.title.lower():
                result = {
                    "title": section.title,
                    "type": section.section_type,
                    "content": section.content if section.section_type == "text" else None,
                    "items": section.items if section.items else None
                }
                return ToolResult(success=True, output=result)

        available = [s.title for s in self.sections]
        return ToolResult(
            success=False,
            output=None,
            error=f"Section '{title}' not found. Available: {available}"
        )

    def _get_all_sections(self) -> ToolResult:
        """List all section titles."""
        if not self._loaded:
            return ToolResult(success=False, output=None, error="Document not loaded")

        sections = [{"title": s.title, "type": s.section_type} for s in self.sections]
        return ToolResult(success=True, output=sections)

    def _search(self, query: str, context_lines: int = 2) -> ToolResult:
        """Search for keyword in document."""
        if not self._loaded:
            return ToolResult(success=False, output=None, error="Document not loaded")

        query_lower = query.lower()
        results = []

        for section in self.sections:
            if query_lower in section.title.lower():
                results.append({
                    "match_type": "section_title",
                    "section": section.title,
                    "content": section.content[:200] + "..." if len(section.content) > 200 else section.content
                })

            for item in section.items:
                if query_lower in item.lower():
                    results.append({
                        "match_type": "item",
                        "section": section.title,
                        "item": item
                    })

        if not results:
            return ToolResult(success=True, output=f"No matches found for '{query}'")

        return ToolResult(success=True, output=results, metadata={"match_count": len(results)})

    def _get_structured_data(self) -> ToolResult:
        """Get document as structured dictionary."""
        if not self._loaded:
            return ToolResult(success=False, output=None, error="Document not loaded")

        data = {}
        for section in self.sections:
            key = section.title.lower().replace(" ", "_")

            if section.section_type == "empty":
                data[key] = None
            elif section.section_type == "list":
                data[key] = section.items
            elif section.section_type == "key_value":
                kv = {}
                for item in section.items:
                    if ':' in item:
                        k, v = item.split(':', 1)
                        kv[k.strip()] = v.strip()
                data[key] = kv
            else:
                data[key] = section.content

        return ToolResult(success=True, output=data)

    def _get_raw(self) -> ToolResult:
        """Get raw document text."""
        if not self._loaded:
            return ToolResult(success=False, output=None, error="Document not loaded")

        return ToolResult(success=True, output=self.raw_text)
