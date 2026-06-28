"""Image perturbation strategies."""

import io
import random
from typing import List, Dict, Any, Tuple, Optional

from .types import PerturbationType, PerturbationResult

# Optional imports
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ImagePerturbator:
    """Applies perturbations to image files.

    Each perturbation is applied independently and deterministically
    based on the provided seed.

    Requires: PIL (Pillow) and numpy
    Install with: pip install Pillow numpy
    """

    SUPPORTED_TYPES = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}

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
        file_type: str = "png",
        file_name: str = "",
    ) -> PerturbationResult:
        """Apply a single perturbation to image content.

        Args:
            content: Raw image bytes
            perturbation: Which perturbation to apply
            file_type: File type hint (png, jpg, etc.)
            file_name: Original file name

        Returns:
            PerturbationResult with corrupted image bytes
        """
        # Reset RNG for reproducibility
        self._rng = random.Random(self.seed)

        # Check dependencies
        if not HAS_PIL:
            return PerturbationResult(
                original_content="[binary image data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="PIL (Pillow) required. Install with: pip install Pillow",
                file_type=file_type,
                file_name=file_name,
            )

        if not HAS_NUMPY and perturbation == PerturbationType.NOISE:
            return PerturbationResult(
                original_content="[binary image data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason="numpy required for noise perturbation. Install with: pip install numpy",
                file_type=file_type,
                file_name=file_name,
            )

        # Handle NULL case
        if perturbation == PerturbationType.NULL_CONTENT:
            return PerturbationResult(
                original_content="[binary image data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="Image replaced with empty content",
                file_type=file_type,
                file_name=file_name,
                corrupted_bytes=b"",
            )

        # Load image
        try:
            img = Image.open(io.BytesIO(content))
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
        except Exception as e:
            return PerturbationResult(
                original_content="[binary image data]",
                corrupted_content="",
                perturbation_type=perturbation,
                description="",
                applied=False,
                skip_reason=f"Failed to load image: {str(e)}",
                file_type=file_type,
                file_name=file_name,
            )

        # Apply specific perturbation
        corrupted_img, description, changes = self._apply_perturbation(img, perturbation)

        # Convert back to bytes
        output_buffer = io.BytesIO()
        # Use original format or default to PNG
        output_format = file_type.upper()
        if output_format == "JPG":
            output_format = "JPEG"
        if output_format not in ("PNG", "JPEG", "GIF", "WEBP", "BMP", "TIFF"):
            output_format = "PNG"

        corrupted_img.save(output_buffer, format=output_format)
        corrupted_bytes = output_buffer.getvalue()

        return PerturbationResult(
            original_content="[binary image data]",
            corrupted_content=f"[perturbed image: {len(corrupted_bytes)} bytes]",
            perturbation_type=perturbation,
            description=description,
            changes=changes,
            file_type=file_type,
            file_name=file_name,
            corrupted_bytes=corrupted_bytes,
        )

    def _apply_perturbation(
        self,
        img: "Image.Image",
        perturbation: PerturbationType,
    ) -> Tuple["Image.Image", str, List[Dict]]:
        """Apply perturbation and return (image, description, changes)."""
        if perturbation == PerturbationType.BLUR:
            return self._apply_blur(img)
        elif perturbation == PerturbationType.NOISE:
            return self._apply_noise(img)
        elif perturbation == PerturbationType.LOW_RESOLUTION:
            return self._apply_low_resolution(img)
        elif perturbation == PerturbationType.PARTIAL_OCCLUSION:
            return self._apply_partial_occlusion(img)
        elif perturbation == PerturbationType.CONTRAST_REDUCTION:
            return self._apply_contrast_reduction(img)
        elif perturbation == PerturbationType.WATERMARK:
            return self._apply_watermark(img)
        else:
            return img, "Unknown perturbation", []

    # =========================================================================
    # Individual Perturbation Implementations
    # =========================================================================

    def _apply_blur(self, img: "Image.Image") -> Tuple["Image.Image", str, List[Dict]]:
        """Apply Gaussian blur to reduce image clarity.

        Uses a blur radius proportional to image size for consistent effect.
        """
        # Calculate blur radius based on image size (1-2% of smaller dimension)
        min_dim = min(img.width, img.height)
        blur_radius = max(1, int(min_dim * 0.015))

        # Apply Gaussian blur
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        changes = [
            {
                "type": "blur",
                "radius": blur_radius,
                "image_size": (img.width, img.height),
            }
        ]
        desc = f"Applied Gaussian blur with radius {blur_radius}px"
        return blurred, desc, changes

    def _apply_noise(self, img: "Image.Image") -> Tuple["Image.Image", str, List[Dict]]:
        """Add salt and pepper noise to the image.

        Randomly sets pixels to black or white.
        """
        # Convert to numpy array
        img_array = np.array(img)

        # Noise density (proportion of pixels to affect)
        noise_density = 0.05

        # Create noise mask
        total_pixels = img_array.shape[0] * img_array.shape[1]
        num_salt = int(total_pixels * noise_density / 2)
        num_pepper = int(total_pixels * noise_density / 2)

        # Add salt (white pixels)
        coords_salt = [
            np.array([self._rng.randint(0, img_array.shape[0] - 1) for _ in range(num_salt)]),
            np.array([self._rng.randint(0, img_array.shape[1] - 1) for _ in range(num_salt)]),
        ]
        img_array[coords_salt[0], coords_salt[1]] = 255

        # Add pepper (black pixels)
        coords_pepper = [
            np.array([self._rng.randint(0, img_array.shape[0] - 1) for _ in range(num_pepper)]),
            np.array([self._rng.randint(0, img_array.shape[1] - 1) for _ in range(num_pepper)]),
        ]
        img_array[coords_pepper[0], coords_pepper[1]] = 0

        # Convert back to PIL Image
        noisy = Image.fromarray(img_array)

        changes = [
            {
                "type": "noise",
                "noise_type": "salt_and_pepper",
                "density": noise_density,
                "pixels_affected": num_salt + num_pepper,
            }
        ]
        desc = f"Added salt & pepper noise ({noise_density * 100:.1f}% density, {num_salt + num_pepper} pixels)"
        return noisy, desc, changes

    def _apply_low_resolution(self, img: "Image.Image") -> Tuple["Image.Image", str, List[Dict]]:
        """Reduce resolution by downscaling then upscaling.

        This simulates loss of detail that can't be recovered.
        """
        original_size = img.size

        # Downscale to 25% of original size
        scale_factor = 0.25
        small_size = (
            max(1, int(img.width * scale_factor)),
            max(1, int(img.height * scale_factor)),
        )

        # Downscale (loses detail)
        small_img = img.resize(small_size, Image.Resampling.BILINEAR)

        # Upscale back to original size (pixelated result)
        low_res = small_img.resize(original_size, Image.Resampling.BILINEAR)

        changes = [
            {
                "type": "low_resolution",
                "original_size": original_size,
                "downscaled_to": small_size,
                "scale_factor": scale_factor,
            }
        ]
        desc = f"Reduced resolution: {original_size} -> {small_size} -> {original_size}"
        return low_res, desc, changes

    def _apply_partial_occlusion(self, img: "Image.Image") -> Tuple["Image.Image", str, List[Dict]]:
        """Add random rectangles to occlude parts of the image.

        Simulates partial obstruction or damage.
        """
        # Work on a copy
        occluded = img.copy()
        draw = ImageDraw.Draw(occluded)

        # Add 5-8 random rectangles
        num_rectangles = self._rng.randint(5, 8)
        changes = []

        for i in range(num_rectangles):
            # Random rectangle size (10-20% of image dimension)
            rect_width = self._rng.randint(int(img.width * 0.10), int(img.width * 0.20))
            rect_height = self._rng.randint(int(img.height * 0.10), int(img.height * 0.20))

            # Random position
            x = self._rng.randint(0, max(0, img.width - rect_width))
            y = self._rng.randint(0, max(0, img.height - rect_height))

            # Random gray color (dark to medium gray for visibility)
            gray_value = self._rng.randint(40, 120)
            color = (gray_value, gray_value, gray_value)

            # Draw rectangle
            draw.rectangle([x, y, x + rect_width, y + rect_height], fill=color)

            changes.append(
                {
                    "type": "occlusion_rectangle",
                    "position": (x, y),
                    "size": (rect_width, rect_height),
                    "color": color,
                }
            )

        desc = f"Added {num_rectangles} occluding rectangles"
        return occluded, desc, changes

    def _apply_contrast_reduction(
        self, img: "Image.Image"
    ) -> Tuple["Image.Image", str, List[Dict]]:
        """Reduce image contrast for a washed-out appearance.

        Also slightly reduces color saturation.
        """
        # Reduce contrast to 40-60% of original
        contrast_factor = self._rng.uniform(0.4, 0.6)
        enhancer = ImageEnhance.Contrast(img)
        reduced = enhancer.enhance(contrast_factor)

        # Also reduce saturation slightly
        saturation_factor = self._rng.uniform(0.6, 0.8)
        enhancer = ImageEnhance.Color(reduced)
        reduced = enhancer.enhance(saturation_factor)

        changes = [
            {
                "type": "contrast_reduction",
                "contrast_factor": contrast_factor,
                "saturation_factor": saturation_factor,
            }
        ]
        desc = (
            f"Reduced contrast to {contrast_factor:.1%} and saturation to {saturation_factor:.1%}"
        )
        return reduced, desc, changes

    def _apply_watermark(self, img: "Image.Image") -> Tuple["Image.Image", str, List[Dict]]:
        """Add a distracting watermark overlay.

        Creates a semi-transparent text watermark across the image.
        """
        # Work on a copy with alpha channel
        watermarked = img.copy()
        if watermarked.mode != "RGBA":
            watermarked = watermarked.convert("RGBA")

        # Create watermark layer
        watermark_layer = Image.new("RGBA", watermarked.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark_layer)

        # Watermark text options
        watermark_texts = [
            "SAMPLE",
            "DRAFT",
            "CONFIDENTIAL",
            "DO NOT COPY",
            "PREVIEW",
            "WATERMARK",
        ]
        text = self._rng.choice(watermark_texts)

        # Calculate font size based on image size
        font_size = max(20, min(img.width, img.height) // 8)

        # Try to use a default font, fall back to default
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw watermark diagonally across image multiple times
        # More visible gray watermark
        watermark_color = (100, 100, 100, 160)

        # Calculate diagonal pattern
        num_repeats_x = max(1, img.width // (text_width + 50)) + 1
        num_repeats_y = max(1, img.height // (text_height + 100)) + 1

        for i in range(num_repeats_y + 1):
            for j in range(num_repeats_x + 1):
                x = j * (text_width + 50) - text_width // 2
                y = i * (text_height + 100) - text_height // 2

                # Create rotated text
                txt_img = Image.new("RGBA", (text_width + 20, text_height + 20), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((10, 10), text, font=font, fill=watermark_color)

                # Rotate
                txt_img = txt_img.rotate(30, expand=True, fillcolor=(0, 0, 0, 0))

                # Paste onto watermark layer
                watermark_layer.paste(txt_img, (x, y), txt_img)

        # Composite
        watermarked = Image.alpha_composite(watermarked, watermark_layer)

        # Convert back to RGB if original was RGB
        if img.mode == "RGB":
            watermarked = watermarked.convert("RGB")

        changes = [
            {
                "type": "watermark",
                "text": text,
                "font_size": font_size,
                "opacity": 100,
                "pattern": "diagonal_repeat",
            }
        ]
        desc = f"Added '{text}' watermark (diagonal pattern, font size {font_size})"
        return watermarked, desc, changes


def get_image_perturbation_types() -> List[PerturbationType]:
    """Get perturbation types for image data."""
    return [
        PerturbationType.BLUR,
        PerturbationType.NOISE,
        PerturbationType.LOW_RESOLUTION,
        PerturbationType.PARTIAL_OCCLUSION,
        PerturbationType.CONTRAST_REDUCTION,
        PerturbationType.WATERMARK,
        PerturbationType.NULL_CONTENT,
    ]
