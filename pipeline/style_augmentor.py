"""Post-render style transforms applied to rendered color images.

Provides image-space augmentations that convert flat 3D renders into
different art styles. Segmentation masks are unaffected — only the
color image is modified.

Currently implemented:
- Pixel art (downscale + palette reduction + upscale)
- Painterly (bilateral filter + color jitter + noise grain)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from .config import (
    PAINTERLY_BILATERAL_D,
    PAINTERLY_DEFAULT_STRENGTH,
    PAINTERLY_HUE_JITTER,
    PAINTERLY_NOISE_SIGMA,
    PAINTERLY_PASSES,
    PAINTERLY_SAT_JITTER,
    PAINTERLY_SIGMA_COLOR,
    PAINTERLY_SIGMA_SPACE,
    PAINTERLY_VAL_JITTER,
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


def apply_painterly(
    image: Image.Image,
    *,
    strength: str = PAINTERLY_DEFAULT_STRENGTH,
    seed: int = 0,
) -> Image.Image:
    """Transform a rendered image into a painterly/soft style.

    Flow: separate alpha → bilateral filter (multi-pass) → color jitter
    in HSV space → add Gaussian noise → restore alpha.

    Args:
        image: Input RGBA image (typically 512x512).
        strength: Filter strength — "light", "medium", or "heavy".
            Controls the number of bilateral filter passes.
        seed: Random seed for deterministic jitter and noise.

    Returns:
        Transformed RGBA image at the same resolution as the input.
    """
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    # PIL RGB → OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    # --- Bilateral filter (edge-preserving blur) ---
    num_passes = PAINTERLY_PASSES.get(strength, PAINTERLY_PASSES[PAINTERLY_DEFAULT_STRENGTH])
    for _ in range(num_passes):
        img_bgr = cv2.bilateralFilter(
            img_bgr,
            d=PAINTERLY_BILATERAL_D,
            sigmaColor=PAINTERLY_SIGMA_COLOR,
            sigmaSpace=PAINTERLY_SIGMA_SPACE,
        )

    # --- Color jitter in HSV space ---
    rng = np.random.default_rng(seed)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)
    h_shift = rng.integers(-PAINTERLY_HUE_JITTER, PAINTERLY_HUE_JITTER + 1)
    s_shift = rng.integers(-PAINTERLY_SAT_JITTER, PAINTERLY_SAT_JITTER + 1)
    v_shift = rng.integers(-PAINTERLY_VAL_JITTER, PAINTERLY_VAL_JITTER + 1)

    # Hue wraps around 0–179 in OpenCV
    h = ((h.astype(np.int16) + h_shift) % 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + s_shift, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + v_shift, 0, 255).astype(np.uint8)

    img_hsv = cv2.merge([h, s, v])
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # --- Gaussian noise ---
    noise = rng.normal(0, PAINTERLY_NOISE_SIGMA * 255, img_bgr.shape)
    img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # OpenCV BGR → PIL RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = Image.fromarray(img_rgb)

    # Restore original alpha channel
    result = Image.merge("RGBA", (*result_rgb.split(), a))

    logger.info(
        "Painterly: strength=%s (%d passes), seed=%d",
        strength,
        num_passes,
        seed,
    )
    return result


def apply_post_render_style(
    image: Image.Image,
    style: str,
    *,
    seed: int = 0,
) -> Image.Image:
    """Apply a post-render art style transform to a color image.

    Routes to the appropriate style function. For styles not yet
    implemented (sketch), returns the image unchanged.

    Args:
        image: Input RGBA image.
        style: Style name ("pixel", "painterly", "sketch").
        seed: Random seed for deterministic transforms (used by painterly).

    Returns:
        Transformed RGBA image (same resolution as input).
    """
    if style == "pixel":
        return apply_pixel_art(image)
    if style == "painterly":
        return apply_painterly(image, seed=seed)
    # Future: sketch
    logger.warning("Post-render style '%s' not yet implemented, returning original", style)
    return image
