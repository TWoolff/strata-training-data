"""Post-render style transforms applied to rendered color images.

Provides image-space augmentations that convert flat 3D renders into
different art styles. Segmentation masks are unaffected — only the
color image is modified.

Currently implemented:
- Pixel art (downscale + palette reduction + upscale)
"""

from __future__ import annotations

import logging

from PIL import Image

from .config import (
    PIXEL_ART_DOWNSCALE_SIZE,
    PIXEL_ART_PALETTE_SIZE,
)

logger = logging.getLogger(__name__)


def apply_pixel_art(
    image: Image.Image,
    *,
    downscale_size: int = PIXEL_ART_DOWNSCALE_SIZE,
    palette_size: int = PIXEL_ART_PALETTE_SIZE,
) -> Image.Image:
    """Transform a rendered image into pixel art style.

    Flow: separate alpha -> downscale RGB with nearest-neighbor ->
    quantize to reduced palette -> upscale back to original size ->
    restore original alpha mask.

    Args:
        image: Input RGBA image (typically 512x512).
        downscale_size: Target resolution for the downscale step (e.g. 64, 128).
        palette_size: Number of colors in the reduced palette (e.g. 16, 24, 32).

    Returns:
        Transformed RGBA image at the same resolution as the input.
    """
    original_size = image.size
    rgba = image.convert("RGBA")

    # Separate alpha channel (preserve transparency untouched)
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    # Downscale with nearest-neighbor (creates chunky pixel grid)
    small = rgb.resize((downscale_size, downscale_size), Image.NEAREST)

    # Reduce color palette using PIL adaptive quantization
    quantized = small.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
    small_rgb = quantized.convert("RGB")

    # Upscale back to original resolution with nearest-neighbor (preserves pixel grid)
    big = small_rgb.resize(original_size, Image.NEAREST)

    # Restore original alpha channel
    result = Image.merge("RGBA", (*big.split(), a))

    logger.info(
        "Pixel art: %dx%d -> %dx%d (%d colors) -> %dx%d",
        original_size[0],
        original_size[1],
        downscale_size,
        downscale_size,
        palette_size,
        original_size[0],
        original_size[1],
    )
    return result


def apply_post_render_style(image: Image.Image, style: str) -> Image.Image:
    """Apply a post-render art style transform to a color image.

    Routes to the appropriate style function. For styles not yet
    implemented (painterly, sketch), returns the image unchanged.

    Args:
        image: Input RGBA image.
        style: Style name ("pixel", "painterly", "sketch").

    Returns:
        Transformed RGBA image (same resolution as input).
    """
    if style == "pixel":
        return apply_pixel_art(image)
    # Future: painterly, sketch
    logger.warning("Post-render style '%s' not yet implemented, returning original", style)
    return image
