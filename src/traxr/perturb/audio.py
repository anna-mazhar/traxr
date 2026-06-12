"""Audio perturbation strategies."""

import io
import random
from typing import List, Dict, Any, Tuple, Optional

from .types import PerturbationType, PerturbationResult

# Optional imports
try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import signal

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioPerturbator:
    """Applies perturbations to audio files.

    Each perturbation is applied independently and deterministically
    based on the provided seed.

    Requires: pydub, numpy, scipy
    Install with: pip install pydub numpy scipy
    Also requires ffmpeg installed on system:
        macOS: brew install ffmpeg
        Ubuntu: sudo apt install ffmpeg
    """

    SUPPORTED_TYPES = {"mp3", "wav", "ogg", "flac", "m4a", "aac", "wma"}

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def can_handle(self, file_type: str) -> bool:
        """Check if this perturbator handles the file type."""
        return file_type.lower() in self.SUPPORTED_TYPES

    def apply(
        self,
        content: bytes,
        perturbation: PerturbationType,
        file_type: str = "mp3",
        file_name: str = "",
    ) -> PerturbationResult:
        """Apply a single perturbation to audio content.

        Args:
            content: Raw audio bytes
            perturbation: Which perturbation to apply
            file_type: File type hint (mp3, wav, etc.)
            file_name: Original file name

        Returns:
            PerturbationResult with corrupted audio bytes
        """
        # Reset RNG for reproducibility
        self._rng = random.Random(self.seed)

        # Check dependencies
        if not HAS_PYDUB:
            return PerturbationResult(
                original_content="[binary audio data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="pydub required. Install with: pip install pydub (also requires ffmpeg)",
                file_type=file_type,
                file_name=file_name,
            )

        if not HAS_NUMPY:
            return PerturbationResult(
                original_content="[binary audio data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="numpy required. Install with: pip install numpy",
                file_type=file_type,
                file_name=file_name,
            )

        if not HAS_SCIPY and perturbation == PerturbationType.LOW_PASS_FILTER:
            return PerturbationResult(
                original_content="[binary audio data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="scipy required for low-pass filter. Install with: pip install scipy",
                file_type=file_type,
                file_name=file_name,
            )

        # Handle NULL case
        if perturbation == PerturbationType.NULL_CONTENT:
            return PerturbationResult(
                original_content="[binary audio data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="Audio replaced with empty content",
                file_type=file_type,
                file_name=file_name,
                corrupted_bytes=b"",
            )

        # Load audio
        try:
            audio = AudioSegment.from_file(io.BytesIO(content), format=file_type)
        except Exception as e:
            return PerturbationResult(
                original_content="[binary audio data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason=f"Failed to load audio: {str(e)}",
                file_type=file_type,
                file_name=file_name,
            )

        # Apply specific perturbation
        corrupted_audio, description, changes = self._apply_perturbation(audio, perturbation)

        # Convert back to bytes
        output_buffer = io.BytesIO()
        # Use original format or default to mp3
        output_format = file_type.lower()
        if output_format not in ("mp3", "wav", "ogg", "flac"):
            output_format = "mp3"

        corrupted_audio.export(output_buffer, format=output_format)
        corrupted_bytes = output_buffer.getvalue()

        return PerturbationResult(
            original_content="[binary audio data]",
            corrupted_content=f"[perturbed audio: {len(corrupted_bytes)} bytes]",
            perturbation_type=perturbation,
            description=description,
            changes=changes,
            file_type=file_type,
            file_name=file_name,
            corrupted_bytes=corrupted_bytes,
        )

    def _apply_perturbation(
        self,
        audio: "AudioSegment",
        perturbation: PerturbationType,
    ) -> Tuple["AudioSegment", str, List[Dict]]:
        """Apply perturbation and return (audio, description, changes)."""
        if perturbation == PerturbationType.BACKGROUND_NOISE:
            return self._apply_background_noise(audio)
        elif perturbation == PerturbationType.SPEED_CHANGE:
            return self._apply_speed_change(audio)
        elif perturbation == PerturbationType.LOW_PASS_FILTER:
            return self._apply_low_pass_filter(audio)
        else:
            return audio, "Unknown perturbation", []

    # =========================================================================
    # Individual Perturbation Implementations
    # =========================================================================

    def _apply_background_noise(
        self, audio: "AudioSegment"
    ) -> Tuple["AudioSegment", str, List[Dict]]:
        """Add background noise (white noise) to audio.

        Generates white noise and mixes it with the original audio.
        """
        # Get audio parameters
        sample_rate = audio.frame_rate
        channels = audio.channels
        duration_ms = len(audio)
        num_samples = int(sample_rate * duration_ms / 1000)

        # Generate white noise
        # Use seeded numpy random for reproducibility
        np_rng = np.random.RandomState(self.seed)
        noise_samples = np_rng.normal(0, 1, num_samples * channels)

        # Normalize noise to 16-bit range
        # 0.3 = noise amplitude multiplier (higher = more noise)
        noise_samples = (noise_samples * 32767 * 0.3).astype(np.int16)

        # Create noise AudioSegment
        noise_bytes = noise_samples.tobytes()
        noise_audio = AudioSegment(
            data=noise_bytes,
            sample_width=2,  # 16-bit
            frame_rate=sample_rate,
            channels=channels,
        )

        # Adjust noise volume relative to original
        # -10 dB = noticeable noise, -5 dB = heavy noise, 0 dB = equal to original
        noise_level_db = -10
        noise_audio = noise_audio + noise_level_db

        # Mix noise with original
        noisy_audio = audio.overlay(noise_audio)

        changes = [
            {
                "type": "background_noise",
                "noise_type": "white_noise",
                "noise_level_db": noise_level_db,
                "duration_ms": duration_ms,
            }
        ]
        desc = f"Added white noise at {noise_level_db} dB below original"
        return noisy_audio, desc, changes

    def _apply_speed_change(self, audio: "AudioSegment") -> Tuple["AudioSegment", str, List[Dict]]:
        """Change audio playback speed.

        Speeds up or slows down audio (also affects pitch).
        """
        # Random speed factor between 0.7x and 1.4x
        speed_factor = self._rng.uniform(0.7, 1.4)

        # Avoid factors too close to 1.0 (no noticeable change)
        if 0.95 < speed_factor < 1.05:
            speed_factor = 1.3  # Default to faster

        # Change speed by altering frame rate
        original_frame_rate = audio.frame_rate
        new_frame_rate = int(original_frame_rate * speed_factor)

        # Create new audio with altered frame rate, then convert back
        speed_changed = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
        # Convert back to original frame rate (this actually changes the speed)
        speed_changed = speed_changed.set_frame_rate(original_frame_rate)

        original_duration = len(audio)
        new_duration = len(speed_changed)

        changes = [
            {
                "type": "speed_change",
                "speed_factor": speed_factor,
                "original_duration_ms": original_duration,
                "new_duration_ms": new_duration,
            }
        ]

        if speed_factor > 1:
            desc = (
                f"Sped up audio by {speed_factor:.2f}x ({original_duration}ms -> {new_duration}ms)"
            )
        else:
            desc = f"Slowed down audio by {speed_factor:.2f}x ({original_duration}ms -> {new_duration}ms)"

        return speed_changed, desc, changes

    def _apply_low_pass_filter(
        self, audio: "AudioSegment"
    ) -> Tuple["AudioSegment", str, List[Dict]]:
        """Apply low-pass filter to create muffled/phone quality audio.

        Removes high frequencies above a cutoff point.
        """
        # Get audio as numpy array
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        channels = audio.channels

        # Reshape for stereo if needed
        if channels == 2:
            samples = samples.reshape((-1, 2))

        # Low-pass filter cutoff frequency (Hz)
        # Phone quality is typically around 3000-4000 Hz
        cutoff_freq = self._rng.randint(2000, 4000)

        # Design Butterworth low-pass filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # Ensure normalized cutoff is valid (between 0 and 1)
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))

        b, a = signal.butter(4, normalized_cutoff, btype="low")

        # Apply filter
        if channels == 2:
            filtered_left = signal.filtfilt(b, a, samples[:, 0])
            filtered_right = signal.filtfilt(b, a, samples[:, 1])
            filtered_samples = np.column_stack((filtered_left, filtered_right))
            filtered_samples = filtered_samples.flatten()
        else:
            filtered_samples = signal.filtfilt(b, a, samples)

        # Convert back to int16
        filtered_samples = np.clip(filtered_samples, -32768, 32767).astype(np.int16)

        # Create new AudioSegment
        filtered_audio = AudioSegment(
            data=filtered_samples.tobytes(),
            sample_width=audio.sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )

        changes = [
            {
                "type": "low_pass_filter",
                "cutoff_frequency_hz": cutoff_freq,
                "filter_order": 4,
                "sample_rate": sample_rate,
            }
        ]
        desc = f"Applied low-pass filter at {cutoff_freq} Hz (muffled/phone quality)"
        return filtered_audio, desc, changes


def get_audio_perturbation_types() -> List[PerturbationType]:
    """Get perturbation types for audio data."""
    return [
        PerturbationType.BACKGROUND_NOISE,
        PerturbationType.SPEED_CHANGE,
        PerturbationType.LOW_PASS_FILTER,
        PerturbationType.NULL_CONTENT,
    ]
