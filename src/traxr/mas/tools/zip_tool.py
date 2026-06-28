"""Zip archive tool for handling zip file attachments.

Provides operations to inspect and extract content from zip archives,
making nested files accessible to the agent system.
"""

import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolResult

import logging

logger = logging.getLogger(__name__)


class ZipTool(BaseTool):
    """Tool for zip archive operations.

    Provides deterministic operations for listing, inspecting, and extracting
    content from zip archives. Designed to handle GAIA tasks where the
    attachment is a zip file containing multiple files.

    Key features:
    - List all files in the archive with sizes
    - Extract to a temporary directory for tool access
    - Read individual text files directly from archive
    - Get archive metadata and structure
    """

    def __init__(self, file_path: Optional[str] = None):
        """Initialize ZipTool.

        Args:
            file_path: Path to zip file (can be set later via load)
        """
        super().__init__(name="zip")
        self.file_path = file_path
        self._archive: Optional[zipfile.ZipFile] = None
        self._extract_dir: Optional[Path] = None
        self._file_list: List[Dict[str, Any]] = []
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute zip operation."""
        operations = {
            "load": self._load,
            "list_files": self._list_files,
            "get_info": self._get_info,
            "read_file": self._read_file,
            "extract_all": self._extract_all,
            "extract_file": self._extract_file,
            "get_structure": self._get_structure,
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
        return [
            "load", "list_files", "get_info", "read_file",
            "extract_all", "extract_file", "get_structure",
        ]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema

        # Update description based on whether file is already loaded
        if self._archive is not None:
            base_desc = (
                "Zip archive tool - FILE ALREADY LOADED AND EXTRACTED. "
                "Use list_files, read_file, get_structure to access contents. "
                "Do NOT call 'load' again."
            )
            load_desc = "Already loaded - DO NOT CALL. Use list_files or read_file instead."
        else:
            base_desc = "Zip archive tool for listing, inspecting, and extracting files from zip archives."
            load_desc = "Load a zip archive and list its contents."

        return ToolSchema(
            name="zip",
            description=base_desc,
            operations={
                "load": OperationSchema(
                    name="load",
                    description=load_desc,
                    parameters=[
                        ToolParameterSchema(
                            name="file_path",
                            type="string",
                            description="Path to zip file (not needed if already loaded)",
                            required=False
                        ),
                    ],
                ),
                "list_files": OperationSchema(
                    name="list_files",
                    description="List all files in the archive with their sizes.",
                    parameters=[],
                ),
                "get_info": OperationSchema(
                    name="get_info",
                    description="Get detailed information about a specific file in the archive.",
                    parameters=[
                        ToolParameterSchema(
                            name="filename",
                            type="string",
                            description="Path to file within the archive",
                            required=True
                        ),
                    ],
                ),
                "read_file": OperationSchema(
                    name="read_file",
                    description="Read the content of a text file directly from the archive.",
                    parameters=[
                        ToolParameterSchema(
                            name="filename",
                            type="string",
                            description="Path to file within the archive",
                            required=True
                        ),
                        ToolParameterSchema(
                            name="max_chars",
                            type="integer",
                            description="Maximum characters to read (default: 50000)",
                            required=False,
                            default=50000
                        ),
                    ],
                ),
                "extract_all": OperationSchema(
                    name="extract_all",
                    description="Extract all files to a temporary directory. Returns the extraction path.",
                    parameters=[],
                ),
                "extract_file": OperationSchema(
                    name="extract_file",
                    description="Extract a specific file to a temporary directory. Returns the file path.",
                    parameters=[
                        ToolParameterSchema(
                            name="filename",
                            type="string",
                            description="Path to file within the archive",
                            required=True
                        ),
                    ],
                ),
                "get_structure": OperationSchema(
                    name="get_structure",
                    description="Get the directory structure of the archive as a tree.",
                    parameters=[],
                ),
            },
        )

    def _load(self, file_path: Optional[str] = None) -> ToolResult:
        """Load a zip archive.

        Args:
            file_path: Path to zip file (optional if set in __init__)

        Returns:
            ToolResult with archive info
        """
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            return ToolResult(
                success=False,
                output=None,
                error="No file_path provided"
            )

        path = Path(self.file_path)
        if not path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {self.file_path}"
            )

        if not zipfile.is_zipfile(self.file_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Not a valid zip file: {self.file_path}"
            )

        try:
            self._archive = zipfile.ZipFile(self.file_path, 'r')
            self._file_list = []

            for info in self._archive.infolist():
                self._file_list.append({
                    "filename": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "is_dir": info.is_dir(),
                    "date_time": f"{info.date_time[0]}-{info.date_time[1]:02d}-{info.date_time[2]:02d} "
                                 f"{info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
                })

            # Count files vs directories
            file_count = sum(1 for f in self._file_list if not f["is_dir"])
            dir_count = sum(1 for f in self._file_list if f["is_dir"])
            total_size = sum(f["file_size"] for f in self._file_list)
            compressed_size = sum(f["compress_size"] for f in self._file_list)

            # Get file types
            extensions = set()
            for f in self._file_list:
                if not f["is_dir"]:
                    ext = Path(f["filename"]).suffix.lower()
                    if ext:
                        extensions.add(ext)

            logger.debug(f"[ZipTool] Loaded archive: {file_count} files, {dir_count} directories")

            return ToolResult(
                success=True,
                output=f"Loaded zip archive with {file_count} files and {dir_count} directories",
                metadata={
                    "file_count": file_count,
                    "directory_count": dir_count,
                    "total_size_bytes": total_size,
                    "compressed_size_bytes": compressed_size,
                    "compression_ratio": round(compressed_size / total_size, 3) if total_size > 0 else 1.0,
                    "file_extensions": sorted(extensions),
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to load zip file: {str(e)}"
            )

    def _list_files(self) -> ToolResult:
        """List all files in the archive."""
        if not self._archive:
            return ToolResult(
                success=False,
                output=None,
                error="No archive loaded. Call 'load' first."
            )

        # Filter out directories for cleaner output
        files_only = [f for f in self._file_list if not f["is_dir"]]

        return ToolResult(
            success=True,
            output=files_only,
            metadata={"file_count": len(files_only)}
        )

    def _get_info(self, filename: str) -> ToolResult:
        """Get detailed information about a file in the archive."""
        if not self._archive:
            return ToolResult(
                success=False,
                output=None,
                error="No archive loaded. Call 'load' first."
            )

        try:
            info = self._archive.getinfo(filename)

            # Detect file type
            ext = Path(filename).suffix.lower()
            file_type = self._detect_file_type(ext)

            return ToolResult(
                success=True,
                output={
                    "filename": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "compression_ratio": round(info.compress_size / info.file_size, 3) if info.file_size > 0 else 1.0,
                    "is_dir": info.is_dir(),
                    "date_time": f"{info.date_time[0]}-{info.date_time[1]:02d}-{info.date_time[2]:02d} "
                                 f"{info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
                    "crc": info.CRC,
                    "compress_type": info.compress_type,
                    "file_type": file_type,
                    "extension": ext,
                },
                metadata={"filename": filename}
            )

        except KeyError:
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found in archive: {filename}"
            )

    def _read_file(self, filename: str, max_chars: int = 50000) -> ToolResult:
        """Read content of a text file from the archive.

        Args:
            filename: Path to file within the archive
            max_chars: Maximum characters to read

        Returns:
            ToolResult with file content
        """
        if not self._archive:
            return ToolResult(
                success=False,
                output=None,
                error="No archive loaded. Call 'load' first."
            )

        try:
            info = self._archive.getinfo(filename)
            if info.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Cannot read directory: {filename}"
                )

            # Check file type - only read text-like files directly
            ext = Path(filename).suffix.lower()
            binary_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pdf',
                                 '.xlsx', '.xls', '.doc', '.docx', '.pptx', '.ppt',
                                 '.mp3', '.mp4', '.wav', '.zip', '.tar', '.gz', '.exe'}

            if ext in binary_extensions:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Cannot read binary file directly. Use 'extract_file' to extract it first: {filename}"
                )

            # Read and decode
            with self._archive.open(filename) as f:
                raw_bytes = f.read(max_chars * 4)  # Read more bytes to account for encoding

            # Try different encodings
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                try:
                    content = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Could not decode file with any known encoding: {filename}"
                )

            # Truncate if needed
            truncated = False
            if len(content) > max_chars:
                content = content[:max_chars]
                truncated = True

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "filename": filename,
                    "chars_read": len(content),
                    "truncated": truncated,
                    "original_size": info.file_size,
                }
            )

        except KeyError:
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found in archive: {filename}"
            )

    def _extract_all(self) -> ToolResult:
        """Extract all files to a temporary directory.

        Returns:
            ToolResult with extraction path
        """
        if not self._archive:
            return ToolResult(
                success=False,
                output=None,
                error="No archive loaded. Call 'load' first."
            )

        # Create temp directory if not exists
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="zip_extract_")
            self._extract_dir = Path(self._temp_dir.name)

        # Extract all
        self._archive.extractall(self._extract_dir)

        # List extracted files
        extracted_files = []
        for f in self._file_list:
            if not f["is_dir"]:
                full_path = self._extract_dir / f["filename"]
                extracted_files.append({
                    "filename": f["filename"],
                    "full_path": str(full_path),
                    "exists": full_path.exists(),
                })

        logger.debug(f"[ZipTool] Extracted {len(extracted_files)} files to {self._extract_dir}")

        return ToolResult(
            success=True,
            output=str(self._extract_dir),
            metadata={
                "extract_dir": str(self._extract_dir),
                "files_extracted": len(extracted_files),
                "extracted_files": extracted_files,
            }
        )

    def _extract_file(self, filename: str) -> ToolResult:
        """Extract a specific file to a temporary directory.

        Args:
            filename: Path to file within the archive

        Returns:
            ToolResult with the extracted file path
        """
        if not self._archive:
            return ToolResult(
                success=False,
                output=None,
                error="No archive loaded. Call 'load' first."
            )

        try:
            info = self._archive.getinfo(filename)
            if info.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Cannot extract directory as file: {filename}"
                )

            # Create temp directory if not exists
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(prefix="zip_extract_")
                self._extract_dir = Path(self._temp_dir.name)

            # Extract the file
            extracted_path = self._archive.extract(filename, self._extract_dir)

            logger.debug(f"[ZipTool] Extracted {filename} to {extracted_path}")

            return ToolResult(
                success=True,
                output=extracted_path,
                metadata={
                    "filename": filename,
                    "extracted_path": extracted_path,
                    "file_size": info.file_size,
                }
            )

        except KeyError:
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found in archive: {filename}"
            )

    def _get_structure(self) -> ToolResult:
        """Get the directory structure of the archive as a tree."""
        if not self._archive:
            return ToolResult(
                success=False,
                output=None,
                error="No archive loaded. Call 'load' first."
            )

        # Build tree structure
        tree: Dict[str, Any] = {}
        for f in self._file_list:
            parts = Path(f["filename"]).parts
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1 and not f["is_dir"]:
                    # This is a file
                    current[part] = {
                        "_type": "file",
                        "_size": f["file_size"],
                        "_ext": Path(part).suffix.lower(),
                    }
                else:
                    # This is a directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        # Convert tree to readable format
        def format_tree(node: Dict, prefix: str = "") -> List[str]:
            lines = []
            items = sorted(node.items(), key=lambda x: (x[1].get("_type") == "file", x[0]))
            for i, (name, value) in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                if isinstance(value, dict) and value.get("_type") == "file":
                    size_str = self._format_size(value["_size"])
                    lines.append(f"{prefix}{connector}{name} ({size_str})")
                else:
                    lines.append(f"{prefix}{connector}{name}/")
                    extension = "    " if is_last else "│   "
                    lines.extend(format_tree(value, prefix + extension))
            return lines

        tree_lines = format_tree(tree)
        tree_str = "\n".join(tree_lines)

        return ToolResult(
            success=True,
            output=tree_str,
            metadata={
                "file_count": sum(1 for f in self._file_list if not f["is_dir"]),
                "directory_count": sum(1 for f in self._file_list if f["is_dir"]),
            }
        )

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def _detect_file_type(self, extension: str) -> str:
        """Detect file type category from extension."""
        ext = extension.lower().lstrip(".")

        categories = {
            "tabular": {"csv", "xlsx", "xls", "tsv"},
            "document": {"pdf", "doc", "docx", "txt", "md", "rtf"},
            "presentation": {"pptx", "ppt"},
            "image": {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "svg"},
            "audio": {"mp3", "wav", "ogg", "flac", "m4a", "aac"},
            "video": {"mp4", "avi", "mov", "mkv", "webm"},
            "code": {"py", "js", "ts", "java", "cpp", "c", "h", "go", "rs", "rb"},
            "data": {"json", "xml", "yaml", "yml", "toml"},
            "archive": {"zip", "tar", "gz", "rar", "7z"},
        }

        for category, extensions in categories.items():
            if ext in extensions:
                return category

        return "unknown"

    def get_extracted_path(self, filename: str) -> Optional[str]:
        """Get the full path to an extracted file.

        Useful for other tools that need to access extracted files.

        Args:
            filename: Path within the archive

        Returns:
            Full path to extracted file, or None if not extracted
        """
        if self._extract_dir is None:
            return None
        full_path = self._extract_dir / filename
        if full_path.exists():
            return str(full_path)
        return None

    def cleanup(self):
        """Clean up temporary extraction directory."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._extract_dir = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
        if self._archive:
            try:
                self._archive.close()
            except Exception:
                pass
