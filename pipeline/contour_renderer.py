"""Freestyle contour pair rendering for contour-removal training data.

Renders each character × pose × angle twice — once with Blender Freestyle
lines enabled (``with_contours``) and once without (``without_contours``).
The pixel difference yields a binary contour mask.

Depends on Blender (``bpy``); must run in ``blender --background`` mode.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

from .config import (
    CONTOUR_DIFF_THRESHOLD,
    CONTOUR_EDGE_BORDER,
    CONTOUR_EDGE_CREASE,
    CONTOUR_EDGE_MARK,
    CONTOUR_EDGE_MATERIAL_BOUNDARY,
    CONTOUR_EDGE_SILHOUETTE,
    CONTOUR_FREESTYLE_THICKNESS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Freestyle setup / teardown
# ---------------------------------------------------------------------------


def enable_freestyle(scene: bpy.types.Scene, *, thickness: float | None = None) -> None:
    """Enable Freestyle line rendering on *scene* with contour edge settings.

    Configures a single Freestyle line set with edge type flags from
    ``config.CONTOUR_EDGE_*`` constants.

    Args:
        scene: The Blender scene.
        thickness: Line thickness in pixels.  Defaults to
            ``CONTOUR_FREESTYLE_THICKNESS``.
    """
    if thickness is None:
        thickness = CONTOUR_FREESTYLE_THICKNESS

    scene.render.use_freestyle = True
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True

    # Get or create the line set.
    if view_layer.freestyle_settings.linesets:
        lineset = view_layer.freestyle_settings.linesets[0]
    else:
        lineset = view_layer.freestyle_settings.linesets.new("contour")

    # Edge selection flags.
    lineset.select_silhouette = CONTOUR_EDGE_SILHOUETTE
    lineset.select_crease = CONTOUR_EDGE_CREASE
    lineset.select_material_boundary = CONTOUR_EDGE_MATERIAL_BOUNDARY
    lineset.select_border = CONTOUR_EDGE_BORDER
    lineset.select_edge_mark = CONTOUR_EDGE_MARK

    # Line appearance.
    lineset.linestyle.thickness = thickness
    lineset.linestyle.color = (0.0, 0.0, 0.0)

    logger.debug(
        "Freestyle enabled: thickness=%.1f, sil=%s, crease=%s, mat=%s",
        thickness,
        CONTOUR_EDGE_SILHOUETTE,
        CONTOUR_EDGE_CREASE,
        CONTOUR_EDGE_MATERIAL_BOUNDARY,
    )


def disable_freestyle(scene: bpy.types.Scene) -> None:
    """Disable Freestyle line rendering on *scene*.

    Args:
        scene: The Blender scene.
    """
    scene.render.use_freestyle = False
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = False
    logger.debug("Freestyle disabled")


# ---------------------------------------------------------------------------
# Contour mask computation
# ---------------------------------------------------------------------------


def compute_contour_mask(
    with_contours: Image.Image,
    without_contours: Image.Image,
    *,
    threshold: int = CONTOUR_DIFF_THRESHOLD,
) -> np.ndarray:
    """Compute a binary contour mask from a Freestyle on/off pair.

    A pixel is classified as a contour pixel when the maximum absolute
    channel difference between the two images exceeds *threshold*.

    Args:
        with_contours: RGB(A) image rendered with Freestyle lines.
        without_contours: RGB(A) image rendered without Freestyle lines.
        threshold: Per-channel difference threshold (0–255).

    Returns:
        2-D ``uint8`` array (same H×W as inputs).  Values are 0 (no contour)
        or 255 (contour pixel).
    """
    arr_with = np.array(with_contours.convert("RGB"), dtype=np.int16)
    arr_without = np.array(without_contours.convert("RGB"), dtype=np.int16)

    diff = np.abs(arr_with - arr_without)
    mask = np.max(diff, axis=2) > threshold
    return (mask * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Render pair
# ---------------------------------------------------------------------------


def render_contour_pair(
    scene: bpy.types.Scene,
    output_dir: Path,
    file_prefix: str,
    *,
    thickness: float | None = None,
    threshold: int = CONTOUR_DIFF_THRESHOLD,
) -> tuple[Path, Path, Path]:
    """Render a Freestyle contour on/off pair and compute the contour mask.

    The caller must have already set up the camera, materials, and EEVEE
    color render settings (via ``renderer.setup_color_render``).  This
    function only toggles Freestyle and fires two render passes.

    Args:
        scene: The Blender scene (camera + materials already configured).
        output_dir: Root output directory (``contours/`` subdirectory is used).
        file_prefix: Filename prefix, e.g. ``"char01_pose_00_front"``.
        thickness: Freestyle line thickness (pixels).
        threshold: Contour mask difference threshold.

    Returns:
        Tuple of three paths:
        ``(with_contours_path, without_contours_path, contour_mask_path)``.
    """
    contour_dir = output_dir / "contours"
    contour_dir.mkdir(parents=True, exist_ok=True)

    with_path = contour_dir / f"{file_prefix}_with_contours.png"
    without_path = contour_dir / f"{file_prefix}_without_contours.png"
    mask_path = contour_dir / f"{file_prefix}_contour_mask.png"

    # --- Render WITH Freestyle ---
    enable_freestyle(scene, thickness=thickness)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_with = Path(tmp.name)
    scene.render.filepath = str(tmp_with)
    bpy.ops.render.render(write_still=True)
    img_with = Image.open(tmp_with).copy()
    img_with.save(with_path, format="PNG", compress_level=9)
    tmp_with.unlink(missing_ok=True)

    # --- Render WITHOUT Freestyle ---
    disable_freestyle(scene)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_without = Path(tmp.name)
    scene.render.filepath = str(tmp_without)
    bpy.ops.render.render(write_still=True)
    img_without = Image.open(tmp_without).copy()
    img_without.save(without_path, format="PNG", compress_level=9)
    tmp_without.unlink(missing_ok=True)

    # --- Compute contour mask ---
    mask_arr = compute_contour_mask(img_with, img_without, threshold=threshold)
    Image.fromarray(mask_arr, mode="L").save(mask_path, format="PNG", compress_level=9)

    logger.info(
        "Contour pair: %s — %d contour pixels",
        file_prefix,
        int(np.count_nonzero(mask_arr)),
    )

    return with_path, without_path, mask_path
