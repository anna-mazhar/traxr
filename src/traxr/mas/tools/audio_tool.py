"""Audio tool for processing audio files (MP3, WAV, etc.).

Uses OpenAI Whisper API for transcription.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from .base import BaseTool, ToolResult

# Optional imports
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Try to import audio libraries for metadata
try:
    from mutagen import File as MutagenFile
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False


class AudioTool(BaseTool):
    """Tool for processing audio files.

    Provides operations:
    - load: Load an audio file
    - transcribe: Transcribe audio to text using Whisper
    - get_metadata: Get audio file metadata (duration, format, etc.)
    """

    # Supported audio formats
    SUPPORTED_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.flac', '.ogg'}

    def __init__(
        self,
        file_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
    ):
        """Initialize AudioTool.

        Args:
            file_path: Path to audio file
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Whisper model to use (default: whisper-1)
        """
        super().__init__(name="audio")
        self.file_path = file_path
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        self._transcription = None
        self._metadata = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            if not HAS_OPENAI:
                raise ImportError("openai package required. Install with: pip install openai")
            if not self._api_key:
                raise ValueError("OpenAI API key required for transcription")
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute audio tool operation."""
        operations = {
            "load": self._load,
            "transcribe": self._transcribe,
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
        return ["load", "transcribe", "get_metadata"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="audio",
            description="Audio tool for loading audio files, transcribing speech to text via Whisper, and reading metadata.",
            operations={
                "load": OperationSchema(
                    name="load",
                    description="Load an audio file (mp3, wav, flac, ogg, etc.) and get basic info.",
                    parameters=[
                        ToolParameterSchema(name="file_path", type="string", description="Path to audio file", required=False),
                    ],
                ),
                "transcribe": OperationSchema(
                    name="transcribe",
                    description="Transcribe audio to text using OpenAI Whisper API.",
                    parameters=[
                        ToolParameterSchema(name="language", type="string", description="Language code (e.g. 'en', 'es', 'fr')", required=False),
                        ToolParameterSchema(name="prompt", type="string", description="Optional prompt to guide transcription", required=False),
                    ],
                ),
                "get_metadata": OperationSchema(
                    name="get_metadata",
                    description="Get audio file metadata (duration, sample rate, channels, bitrate).",
                    parameters=[],
                ),
            },
        )

    def _load(self, file_path: Optional[str] = None) -> ToolResult:
        """Load an audio file.

        Args:
            file_path: Optional path to audio file (uses constructor path if not provided)

        Returns:
            ToolResult with file info
        """
        path = file_path or self.file_path
        if not path:
            return ToolResult(
                success=False,
                output=None,
                error="No file path provided"
            )

        path = Path(path)
        if not path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {path}"
            )

        # Check format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unsupported format: {path.suffix}. Supported: {self.SUPPORTED_FORMATS}"
            )

        self.file_path = str(path)
        self._transcription = None  # Reset transcription
        self._metadata = None

        # Get basic file info
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        return ToolResult(
            success=True,
            output={
                "file_path": str(path),
                "file_name": path.name,
                "format": path.suffix.lower(),
                "size_mb": round(file_size_mb, 2),
            },
            metadata={"loaded": True}
        )

    def _get_metadata(self) -> ToolResult:
        """Get audio file metadata.

        Returns:
            ToolResult with metadata (duration, format, etc.)
        """
        if not self.file_path:
            return ToolResult(
                success=False,
                output=None,
                error="No file loaded. Use 'load' operation first."
            )

        if self._metadata:
            return ToolResult(success=True, output=self._metadata)

        path = Path(self.file_path)
        metadata = {
            "file_name": path.name,
            "format": path.suffix.lower().replace(".", ""),
            "size_bytes": path.stat().st_size,
        }

        # Try to get detailed metadata using mutagen
        if HAS_MUTAGEN:
            try:
                audio = MutagenFile(self.file_path)
                if audio is not None:
                    if hasattr(audio, 'info'):
                        info = audio.info
                        if hasattr(info, 'length'):
                            metadata["duration"] = round(info.length, 2)
                        if hasattr(info, 'sample_rate'):
                            metadata["sample_rate"] = info.sample_rate
                        if hasattr(info, 'channels'):
                            metadata["channels"] = info.channels
                        if hasattr(info, 'bitrate'):
                            metadata["bitrate"] = info.bitrate
            except Exception:
                pass  # Silently fail if metadata extraction fails

        self._metadata = metadata

        return ToolResult(
            success=True,
            output=metadata,
            metadata={"has_mutagen": HAS_MUTAGEN}
        )

    def _transcribe(
        self,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> ToolResult:
        """Transcribe audio to text using OpenAI Whisper.

        Args:
            language: Optional language code (e.g., 'en', 'es', 'fr')
            prompt: Optional prompt to guide transcription

        Returns:
            ToolResult with transcription text
        """
        if not self.file_path:
            return ToolResult(
                success=False,
                output=None,
                error="No file loaded. Use 'load' operation first."
            )

        # Return cached transcription if available
        if self._transcription:
            return ToolResult(
                success=True,
                output=self._transcription,
                metadata={"cached": True}
            )

        if not HAS_OPENAI:
            return ToolResult(
                success=False,
                output=None,
                error="openai package required. Install with: pip install openai"
            )

        try:
            client = self._get_client()

            # Prepare transcription request
            with open(self.file_path, "rb") as audio_file:
                kwargs = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": "text",
                }

                if language:
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt

                transcription = client.audio.transcriptions.create(**kwargs)

            # Store and return transcription
            self._transcription = transcription

            return ToolResult(
                success=True,
                output=transcription,
                metadata={
                    "model": self.model,
                    "language": language,
                    "file": Path(self.file_path).name,
                    "char_count": len(transcription),
                }
            )

        except openai.APIError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Whisper API error: {str(e)}"
            )

    def get_transcription(self) -> Optional[str]:
        """Get cached transcription if available."""
        return self._transcription
