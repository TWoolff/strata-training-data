"""Post-render style transforms applied to rendered color images.

Provides image-space augmentations that convert flat 3D renders into
different art styles. Segmentation masks are unaffected — only the
color image is modified.

Currently implemented:
- Pixel art (downscale + palette reduction + upscale)
- Painterly (bilateral filter + color jitter + noise grain)
- Sketch/lineart (edge detection + thick outlines + optional wobble)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from .config import (
    INK_WASH_BILATERAL_D,
    INK_WASH_EDGE_THICKNESS,
    INK_WASH_EDGE_THRESHOLD1,
    INK_WASH_EDGE_THRESHOLD2,
    INK_WASH_SATURATION,
    INK_WASH_SIGMA_COLOR,
    INK_WASH_SIGMA_SPACE,
    INK_WASH_TINT,
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
    SKETCH_BG_COLOR,
    SKETCH_BLUR_KSIZE,
    SKETCH_CANNY_THRESHOLD1,
    SKETCH_CANNY_THRESHOLD2,
    SKETCH_ENABLE_WOBBLE,
    SKETCH_LINE_THICKNESS,
    SKETCH_WOBBLE_RANGE,
    WATERCOLOR_BILATERAL_D,
    WATERCOLOR_BILATERAL_PASSES,
    WATERCOLOR_EDGE_COLOR,
    WATERCOLOR_EDGE_THICKNESS,
    WATERCOLOR_EDGE_THRESHOLD1,
    WATERCOLOR_EDGE_THRESHOLD2,
    WATERCOLOR_GRAIN_SIGMA,
    WATERCOLOR_SAT_BOOST,
    WATERCOLOR_SIGMA_COLOR,
    WATERCOLOR_SIGMA_SPACE,
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


def apply_sketch(
    image: Image.Image,
    *,
    seed: int = 0,
    enable_wobble: bool = SKETCH_ENABLE_WOBBLE,
) -> Image.Image:
    """Transform a rendered image into a sketch/lineart style.

    Flow: separate alpha → grayscale → Gaussian blur → Canny edge detection
    → dilate edges → invert (black lines on cream background) → optional
    wobble → apply original alpha mask.

    Args:
        image: Input RGBA image (typically 512x512).
        seed: Random seed for deterministic wobble displacement.
        enable_wobble: Whether to apply hand-drawn wobble effect.

    Returns:
        Transformed RGBA image at the same resolution as the input.
    """
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    # PIL RGB → grayscale
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)

    # Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (SKETCH_BLUR_KSIZE, SKETCH_BLUR_KSIZE), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, SKETCH_CANNY_THRESHOLD1, SKETCH_CANNY_THRESHOLD2)

    # Dilate edges for thicker lines
    kernel = np.ones((SKETCH_LINE_THICKNESS, SKETCH_LINE_THICKNESS), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Optional wobble: displace edge pixels for hand-drawn feel
    if enable_wobble and SKETCH_WOBBLE_RANGE > 0:
        rng = np.random.default_rng(seed)
        h, w = edges.shape
        # Create displacement maps for x and y
        dx = rng.integers(-SKETCH_WOBBLE_RANGE, SKETCH_WOBBLE_RANGE + 1, size=(h, w)).astype(
            np.float32
        )
        dy = rng.integers(-SKETCH_WOBBLE_RANGE, SKETCH_WOBBLE_RANGE + 1, size=(h, w)).astype(
            np.float32
        )
        # Build coordinate maps for remap
        map_x = (np.arange(w)[np.newaxis, :] + dx).astype(np.float32)
        map_y = (np.arange(h)[:, np.newaxis] + dy).astype(np.float32)
        edges = cv2.remap(edges, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)

    # Invert: black lines on cream/white background
    bg = np.full((*edges.shape, 3), SKETCH_BG_COLOR, dtype=np.uint8)
    # Where edges are detected, set to black; otherwise keep background
    line_mask = edges > 0
    bg[line_mask] = (0, 0, 0)

    # Convert to PIL and restore original alpha
    result_rgb = Image.fromarray(bg, mode="RGB")
    result = Image.merge("RGBA", (*result_rgb.split(), a))

    logger.info(
        "Sketch: blur=%d, canny=(%d,%d), thickness=%d, wobble=%s, seed=%d",
        SKETCH_BLUR_KSIZE,
        SKETCH_CANNY_THRESHOLD1,
        SKETCH_CANNY_THRESHOLD2,
        SKETCH_LINE_THICKNESS,
        enable_wobble,
        seed,
    )
    return result


def apply_ink_wash(
    image: Image.Image,
    *,
    seed: int = 0,
) -> Image.Image:
    """Transform a rendered image into an anime ink wash / sumi-e style.

    Flow: separate alpha → bilateral filter → partial desaturation (warm wash)
    → Canny edges overlaid in dark ink → restore alpha.

    Args:
        image: Input RGBA image (typically 512x512).
        seed: Random seed (unused currently, reserved for future jitter).

    Returns:
        Transformed RGBA image at the same resolution as the input.
    """
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    img_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    # Bilateral filter for ink-like edge preservation with soft fills
    img_bgr = cv2.bilateralFilter(
        img_bgr,
        d=INK_WASH_BILATERAL_D,
        sigmaColor=INK_WASH_SIGMA_COLOR,
        sigmaSpace=INK_WASH_SIGMA_SPACE,
    )

    # Partial desaturation: reduce saturation to create ink wash feel
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] *= INK_WASH_SATURATION
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Blend with warm paper tint in shadow areas
    tint_bgr = np.array([INK_WASH_TINT[2], INK_WASH_TINT[1], INK_WASH_TINT[0]], dtype=np.float32)
    gray_mask = (255 - img_bgr[:, :, 0].astype(np.float32)) / 255.0  # dark areas get more tint
    for c in range(3):
        img_bgr[:, :, c] = np.clip(
            img_bgr[:, :, c].astype(np.float32) * (1 - gray_mask * 0.3)
            + tint_bgr[c] * gray_mask * 0.3,
            0,
            255,
        ).astype(np.uint8)

    # Canny edges overlaid as ink lines
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, INK_WASH_EDGE_THRESHOLD1, INK_WASH_EDGE_THRESHOLD2)
    kernel = np.ones((INK_WASH_EDGE_THICKNESS, INK_WASH_EDGE_THICKNESS), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Darken pixels where edges detected (ink lines)
    edge_mask = edges > 0
    img_bgr[edge_mask] = np.clip(img_bgr[edge_mask].astype(np.int16) - 120, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = Image.fromarray(img_rgb)
    result = Image.merge("RGBA", (*result_rgb.split(), a))

    logger.info(
        "Ink wash: saturation=%.2f, edge_threshold=(%d,%d)",
        INK_WASH_SATURATION,
        INK_WASH_EDGE_THRESHOLD1,
        INK_WASH_EDGE_THRESHOLD2,
    )
    return result


def apply_watercolor(
    image: Image.Image,
    *,
    seed: int = 0,
) -> Image.Image:
    """Transform a rendered image into a soft watercolor illustration style.

    Flow: separate alpha → multi-pass bilateral filter (paint-like blurring)
    → saturation boost → paper grain noise → thin warm edge overlay → restore alpha.

    Args:
        image: Input RGBA image (typically 512x512).
        seed: Random seed for deterministic grain and jitter.

    Returns:
        Transformed RGBA image at the same resolution as the input.
    """
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))

    img_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    # Multi-pass bilateral filter — creates the paint-blended look
    for _ in range(WATERCOLOR_BILATERAL_PASSES):
        img_bgr = cv2.bilateralFilter(
            img_bgr,
            d=WATERCOLOR_BILATERAL_D,
            sigmaColor=WATERCOLOR_SIGMA_COLOR,
            sigmaSpace=WATERCOLOR_SIGMA_SPACE,
        )

    # Saturation boost (watercolors are vibrant)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * WATERCOLOR_SAT_BOOST, 0, 255)
    img_hsv = img_hsv.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Paper grain: low-sigma Gaussian noise for texture
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, WATERCOLOR_GRAIN_SIGMA * 255, img_bgr.shape)
    img_bgr = np.clip(img_bgr.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Thin warm-colored edges (not harsh black — soft watercolor outline)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, WATERCOLOR_EDGE_THRESHOLD1, WATERCOLOR_EDGE_THRESHOLD2)
    kernel = np.ones((WATERCOLOR_EDGE_THICKNESS, WATERCOLOR_EDGE_THICKNESS), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_mask = edges > 0
    edge_bgr = np.array(
        [WATERCOLOR_EDGE_COLOR[2], WATERCOLOR_EDGE_COLOR[1], WATERCOLOR_EDGE_COLOR[0]],
        dtype=np.uint8,
    )
    # Blend edge color at 70% opacity over existing pixels
    img_bgr[edge_mask] = np.clip(
        img_bgr[edge_mask].astype(np.float32) * 0.3 + edge_bgr.astype(np.float32) * 0.7,
        0,
        255,
    ).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = Image.fromarray(img_rgb)
    result = Image.merge("RGBA", (*result_rgb.split(), a))

    logger.info(
        "Watercolor: passes=%d, sat_boost=%.2f, grain_sigma=%.3f",
        WATERCOLOR_BILATERAL_PASSES,
        WATERCOLOR_SAT_BOOST,
        WATERCOLOR_GRAIN_SIGMA,
    )
    return result


def apply_post_render_style(
    image: Image.Image,
    style: str,
    *,
    seed: int = 0,
) -> Image.Image:
    """Apply a post-render art style transform to a color image.

    Routes to the appropriate style function.

    Args:
        image: Input RGBA image.
        style: Style name ("pixel", "painterly", "sketch", "ink_wash", "watercolor").
        seed: Random seed for deterministic transforms.

    Returns:
        Transformed RGBA image (same resolution as input).
    """
    if style == "pixel":
        return apply_pixel_art(image)
    if style == "painterly":
        return apply_painterly(image, seed=seed)
    if style == "sketch":
        return apply_sketch(image, seed=seed)
    if style == "ink_wash":
        return apply_ink_wash(image, seed=seed)
    if style == "watercolor":
        return apply_watercolor(image, seed=seed)
    logger.warning("Post-render style '%s' not yet implemented, returning original", style)
    return image
