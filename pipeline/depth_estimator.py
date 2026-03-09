"""Monocular depth estimation for draw order enrichment.

Uses Depth Anything v2 (small) ONNX model to estimate per-pixel relative
depth from a single image.  The depth map is normalized to [0, 255] and
saved as ``draw_order.png`` — a grayscale image where 0 = farthest and
255 = nearest (matching the Strata draw order convention).

Pure Python — no Blender dependency.  Requires ``onnxruntime``.

Usage::

    from pipeline.depth_estimator import load_depth_model, enrich_example

    model = load_depth_model("models/depth_anything_v2_vits.onnx")
    enrich_example(model, Path("output/anime_seg/example_001"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model input size for Depth Anything v2 small.
DEPTH_INPUT_SIZE = (518, 518)


def load_depth_model(
    model_path: str | Path,
    *,
    device: str = "cpu",
) -> Any:
    """Load the Depth Anything v2 ONNX model.

    Args:
        model_path: Path to the ONNX model file.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        ONNX InferenceSession.
    """
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=providers)
    logger.info("Depth model loaded: %s (device=%s)", model_path, device)
    return session


def estimate_depth(
    session: Any,
    image: np.ndarray,
) -> np.ndarray:
    """Estimate relative depth from a single image.

    Args:
        session: ONNX InferenceSession for Depth Anything v2.
        image: BGR image (OpenCV format), any size.

    Returns:
        Depth map as float32 array, same H×W as input, values in [0, 1]
        where 1 = nearest.
    """
    h, w = image.shape[:2]

    # Preprocess: resize, normalize, CHW, batch.
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, DEPTH_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    # ImageNet normalization.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    blob = normalized.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    # Inference.
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: blob})[0]

    # Result shape varies: [1, 1, H, W] or [1, H, W].
    depth = result.squeeze()

    # Resize back to original dimensions.
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] — invert so that closer = higher value.
    d_min, d_max = depth_resized.min(), depth_resized.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_resized - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_resized)

    # Depth Anything outputs: large value = far. Invert for Strata convention.
    depth_norm = 1.0 - depth_norm

    return depth_norm.astype(np.float32)


def enrich_example(
    session: Any,
    example_dir: Path,
) -> bool:
    """Add draw_order.png to a Strata example directory.

    Reads ``image.png``, runs depth estimation, saves ``draw_order.png``,
    and updates ``metadata.json``.

    Args:
        session: ONNX InferenceSession for Depth Anything v2.
        example_dir: Path to the example directory.

    Returns:
        True on success, False on failure.
    """
    image_path = example_dir / "image.png"
    if not image_path.is_file():
        logger.warning("No image.png in %s", example_dir)
        return False

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to read %s", image_path)
            return False

        depth = estimate_depth(session, image)

        # Convert to 8-bit grayscale.
        draw_order = (depth * 255).clip(0, 255).astype(np.uint8)

        # Mask out background (transparent pixels should be 0 in draw order).
        image_rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image_rgba is not None and image_rgba.shape[2] == 4:
            alpha = image_rgba[:, :, 3]
            draw_order[alpha < 128] = 0

        cv2.imwrite(str(example_dir / "draw_order.png"), draw_order)

        # Update metadata.
        meta_path = example_dir / "metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["has_draw_order"] = True
            if "draw_order" in meta.get("missing_annotations", []):
                meta["missing_annotations"].remove("draw_order")
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        return True

    except Exception as exc:
        logger.warning("Depth estimation failed for %s: %s", example_dir, exc)
        return False
