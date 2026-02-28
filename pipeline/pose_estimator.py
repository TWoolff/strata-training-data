"""2D pose estimation via RTMPose for image-only datasets.

Maps COCO 17-point keypoints to Strata's 19-region skeleton.  Uses rtmlib
(ONNX backend) for inference — no Blender dependency.  Designed as a
post-processing enrichment step that runs after ingest adapters.

COCO 17-point keypoint indices:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import JOINT_BBOX_PADDING, NUM_JOINT_REGIONS, REGION_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO keypoint indices
# ---------------------------------------------------------------------------

COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

# Confidence penalty for interpolated joints (no direct COCO keypoint).
INTERPOLATION_CONFIDENCE_FACTOR: float = 0.8

# Default minimum confidence to mark a joint as visible.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3

# Extrapolation factors for hands and feet.
# COCO's most distal keypoints are wrists/ankles, but Strata's "hand" and
# "foot" regions are beyond those joints.  We extrapolate along the limb.
HAND_EXTRAPOLATION: float = 0.3  # 30% past wrist along elbow→wrist direction
FOOT_EXTRAPOLATION: float = 0.25  # 25% past ankle along knee→ankle direction


# ---------------------------------------------------------------------------
# COCO → Strata mapping (pure function, no model dependency)
# ---------------------------------------------------------------------------


def _midpoint(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Return the midpoint of two 2D points."""
    return (a + b) / 2.0


def _lerp(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
) -> np.ndarray:
    """Linear interpolation between two points at parameter *t*."""
    return a + (b - a) * t


def coco_to_strata(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    image_size: tuple[int, int],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Map COCO 17-point keypoints to Strata 19-region joints.

    Produces all 19 body-region joints (regions 1–19).  Each joint is placed
    on the body part it names: ``upper_arm`` at the elbow, ``lower_arm`` at
    the wrist, ``hand`` extrapolated past the wrist, etc.  5 are directly
    mapped from COCO keypoints; 14 are interpolated or extrapolated with a
    reduced confidence factor.

    Args:
        keypoints: Array of shape ``(17, 2)`` with (x, y) pixel coords.
        confidences: Array of shape ``(17,)`` with per-keypoint confidence.
        image_size: ``(width, height)`` of the source image.
        confidence_threshold: Minimum confidence for visible=True.

    Returns:
        Dict keyed by Strata region name, each value containing
        ``position``, ``confidence``, and ``visible`` fields.
    """
    kp = keypoints  # (17, 2)
    cf = confidences  # (17,)
    w, h = image_size

    joints: dict[str, dict[str, Any]] = {}

    # Helper to compute midpoint confidence
    def mid_conf(*indices: int) -> float:
        return float(min(cf[i] for i in indices))

    # --- Shoulder midpoint (used by multiple interpolations) ---
    shoulder_mid = _midpoint(kp[COCO_LEFT_SHOULDER], kp[COCO_RIGHT_SHOULDER])
    shoulder_mid_conf = mid_conf(COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER)

    # --- Hip midpoint ---
    hip_mid = _midpoint(kp[COCO_LEFT_HIP], kp[COCO_RIGHT_HIP])
    hip_mid_conf = mid_conf(COCO_LEFT_HIP, COCO_RIGHT_HIP)

    # --- Direct mappings (5 joints with 1:1 COCO keypoint) ---
    #   shoulder = at the shoulder, upper_arm = at the elbow,
    #   lower_arm = at the wrist, upper_leg = at the knee,
    #   lower_leg = at the ankle.
    direct: list[tuple[str, int]] = [
        ("head", COCO_NOSE),
        ("shoulder_l", COCO_LEFT_SHOULDER),
        ("shoulder_r", COCO_RIGHT_SHOULDER),
        ("upper_arm_l", COCO_LEFT_ELBOW),     # elbow
        ("upper_arm_r", COCO_RIGHT_ELBOW),     # elbow
        ("lower_arm_l", COCO_LEFT_WRIST),      # wrist
        ("lower_arm_r", COCO_RIGHT_WRIST),     # wrist
        ("upper_leg_l", COCO_LEFT_KNEE),       # knee
        ("upper_leg_r", COCO_RIGHT_KNEE),      # knee
        ("lower_leg_l", COCO_LEFT_ANKLE),      # ankle
        ("lower_leg_r", COCO_RIGHT_ANKLE),     # ankle
    ]

    def joint(pos: np.ndarray, conf: float) -> dict[str, Any]:
        return _make_joint(pos, conf, w, h, confidence_threshold)

    for region_name, coco_idx in direct:
        joints[region_name] = joint(kp[coco_idx], float(cf[coco_idx]))

    # Hips: midpoint of both COCO hips
    joints["hips"] = joint(hip_mid, hip_mid_conf)

    # --- Interpolated / extrapolated regions (7 regions, confidence × 0.8) ---
    icf = INTERPOLATION_CONFIDENCE_FACTOR

    # neck: midpoint of nose ↔ shoulder midpoint
    neck_pos = _midpoint(kp[COCO_NOSE], shoulder_mid)
    neck_conf = mid_conf(COCO_NOSE, COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER)
    joints["neck"] = joint(neck_pos, neck_conf * icf)

    # chest: 1/3 from shoulder midpoint toward hip midpoint
    chest_pos = _lerp(shoulder_mid, hip_mid, 1.0 / 3.0)
    torso_conf = min(shoulder_mid_conf, hip_mid_conf)
    joints["chest"] = joint(chest_pos, torso_conf * icf)

    # spine: 2/3 from shoulder midpoint toward hip midpoint
    spine_pos = _lerp(shoulder_mid, hip_mid, 2.0 / 3.0)
    joints["spine"] = joint(spine_pos, torso_conf * icf)

    # hand_l: extrapolate past left wrist along elbow→wrist direction
    hand_l_pos = _lerp(kp[COCO_LEFT_ELBOW], kp[COCO_LEFT_WRIST], 1.0 + HAND_EXTRAPOLATION)
    hand_l_conf = mid_conf(COCO_LEFT_ELBOW, COCO_LEFT_WRIST)
    joints["hand_l"] = joint(hand_l_pos, hand_l_conf * icf)

    # hand_r: extrapolate past right wrist along elbow→wrist direction
    hand_r_pos = _lerp(kp[COCO_RIGHT_ELBOW], kp[COCO_RIGHT_WRIST], 1.0 + HAND_EXTRAPOLATION)
    hand_r_conf = mid_conf(COCO_RIGHT_ELBOW, COCO_RIGHT_WRIST)
    joints["hand_r"] = joint(hand_r_pos, hand_r_conf * icf)

    # foot_l: extrapolate past left ankle along knee→ankle direction
    foot_l_pos = _lerp(kp[COCO_LEFT_KNEE], kp[COCO_LEFT_ANKLE], 1.0 + FOOT_EXTRAPOLATION)
    foot_l_conf = mid_conf(COCO_LEFT_KNEE, COCO_LEFT_ANKLE)
    joints["foot_l"] = joint(foot_l_pos, foot_l_conf * icf)

    # foot_r: extrapolate past right ankle along knee→ankle direction
    foot_r_pos = _lerp(kp[COCO_RIGHT_KNEE], kp[COCO_RIGHT_ANKLE], 1.0 + FOOT_EXTRAPOLATION)
    foot_r_conf = mid_conf(COCO_RIGHT_KNEE, COCO_RIGHT_ANKLE)
    joints["foot_r"] = joint(foot_r_pos, foot_r_conf * icf)

    return joints


def _make_joint(
    position: np.ndarray,
    confidence: float,
    img_w: int,
    img_h: int,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Build a single joint dict, clamping position to image bounds.

    Args:
        position: (x, y) pixel coordinates.
        confidence: Detection confidence [0, 1].
        img_w: Image width.
        img_h: Image height.
        confidence_threshold: Minimum confidence for visible=True.

    Returns:
        Joint dict with ``position``, ``confidence``, ``visible`` fields.
    """
    x = round(float(position[0]))
    y = round(float(position[1]))

    visible = confidence >= confidence_threshold

    # Clamp to image bounds
    clamped_x = max(0, min(x, img_w - 1))
    clamped_y = max(0, min(y, img_h - 1))

    return {
        "position": [clamped_x, clamped_y],
        "confidence": round(confidence, 4),
        "visible": visible,
    }


# ---------------------------------------------------------------------------
# Bounding box (reuses logic from joint_extractor.py)
# ---------------------------------------------------------------------------


def _compute_bbox(
    joints: dict[str, dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Compute 2D bounding box from visible joint positions.

    Args:
        joints: Strata joint dict (region_name → joint info).
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        ``[x_min, y_min, x_max, y_max]`` in pixel coordinates.
    """
    visible_points = [
        j["position"] for j in joints.values() if j["visible"] and j["position"] != [-1, -1]
    ]

    if not visible_points:
        return [0, 0, image_width, image_height]

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width = x_max - x_min
    height = y_max - y_min
    pad_x = max(int(width * JOINT_BBOX_PADDING), 5)
    pad_y = max(int(height * JOINT_BBOX_PADDING), 5)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image_width, x_max + pad_x)
    y_max = min(image_height, y_max + pad_y)

    return [x_min, y_min, x_max, y_max]


# ---------------------------------------------------------------------------
# Joint data builder (matches joint_extractor.py output schema)
# ---------------------------------------------------------------------------


def build_joint_data(
    character_id: str,
    strata_joints: dict[str, dict[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Build the full joints.json dict matching the pipeline schema.

    Args:
        character_id: Example identifier (e.g. ``fbanimehq_0003_000257``).
        strata_joints: Dict returned by :func:`coco_to_strata`.
        image_size: ``(width, height)`` of the image.

    Returns:
        Dict ready for JSON serialization, compatible with
        ``joint_extractor.save_joints()`` output.
    """
    w, h = image_size
    bbox = _compute_bbox(strata_joints, w, h)

    return {
        "character_id": character_id,
        "pose_name": "default",
        "source_animation": "",
        "source_frame": 0,
        "image_size": [w, h],
        "joints": strata_joints,
        "bbox": bbox,
    }


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def load_model(
    det_model_path: str | Path,
    pose_model_path: str | Path,
    *,
    device: str = "cpu",
    backend: str = "onnxruntime",
    det_input_size: tuple[int, int] = (640, 640),
    pose_input_size: tuple[int, int] = (192, 256),
) -> Any:
    """Load RTMPose Body model via rtmlib.

    Args:
        det_model_path: Path or URL to the detection ONNX model.
        pose_model_path: Path or URL to the pose ONNX model.
        device: ``"cpu"`` or ``"cuda"``.
        backend: ONNX runtime backend (default ``"onnxruntime"``).
        det_input_size: Detection model input size ``(w, h)``.
        pose_input_size: Pose model input size ``(w, h)``.

    Returns:
        rtmlib Body instance.
    """
    from rtmlib import Body  # type: ignore[import-untyped]

    model = Body(
        det=str(det_model_path),
        det_input_size=det_input_size,
        pose=str(pose_model_path),
        pose_input_size=pose_input_size,
        backend=backend,
        device=device,
    )

    logger.info(
        "Loaded RTMPose model: det=%s, pose=%s, device=%s",
        det_model_path,
        pose_model_path,
        device,
    )
    return model


def estimate_pose(
    model: Any,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run pose estimation on a single image.

    If multiple people are detected, returns the one with the largest
    bounding box (most prominent figure).

    Args:
        model: rtmlib Body instance from :func:`load_model`.
        image: BGR image as numpy array (OpenCV format).

    Returns:
        ``(keypoints, confidences)`` where keypoints is ``(17, 2)``
        and confidences is ``(17,)``, or None if no person detected.
    """
    keypoints, scores = model(image)

    if keypoints is None or len(keypoints) == 0:
        return None

    # keypoints shape: (num_persons, 17, 2)
    # scores shape: (num_persons, 17)
    if len(keypoints) == 1:
        return keypoints[0], scores[0]

    # Multiple detections: pick the one with the largest bounding box area
    best_idx = 0
    best_area = 0.0
    for i in range(len(keypoints)):
        kp = keypoints[i]
        x_min, x_max = float(kp[:, 0].min()), float(kp[:, 0].max())
        y_min, y_max = float(kp[:, 1].min()), float(kp[:, 1].max())
        area = (x_max - x_min) * (y_max - y_min)
        if area > best_area:
            best_area = area
            best_idx = i

    return keypoints[best_idx], scores[best_idx]


# ---------------------------------------------------------------------------
# High-level enrichment function
# ---------------------------------------------------------------------------


def enrich_example(
    model: Any,
    example_dir: Path,
    image_size: tuple[int, int],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Run pose estimation on a single example and write joints.json.

    Reads ``image.png`` from *example_dir*, runs inference, maps COCO
    keypoints to Strata joints, and writes ``joints.json``.  Also updates
    ``metadata.json`` to set ``has_joints: true``.

    Args:
        model: rtmlib Body instance from :func:`load_model`.
        example_dir: Path to a single example directory containing
            ``image.png`` and ``metadata.json``.
        image_size: ``(width, height)`` of the images.
        confidence_threshold: Minimum confidence for visible=True.

    Returns:
        True if joints were written, False on error.
    """
    import cv2  # type: ignore[import-untyped]

    image_path = example_dir / "image.png"
    if not image_path.is_file():
        logger.warning("No image.png in %s", example_dir)
        return False

    # Read image (BGR for OpenCV/rtmlib)
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Failed to read image: %s", image_path)
        return False

    # Run inference
    result = estimate_pose(model, img)
    if result is None:
        logger.warning("No person detected in %s", image_path)
        return False

    keypoints, confidences = result
    character_id = example_dir.name

    # Map COCO → Strata
    strata_joints = coco_to_strata(
        keypoints, confidences, image_size, confidence_threshold=confidence_threshold
    )

    # Validate joint count
    expected_names = {REGION_NAMES[rid] for rid in range(1, NUM_JOINT_REGIONS + 1)}
    actual_names = set(strata_joints.keys())
    if actual_names != expected_names:
        missing = expected_names - actual_names
        logger.error("Missing joints for %s: %s", character_id, missing)
        return False

    # Build and write joints.json
    joint_data = build_joint_data(character_id, strata_joints, image_size)

    joints_path = example_dir / "joints.json"
    joints_path.write_text(
        json.dumps(joint_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Update metadata.json
    meta_path = example_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["has_joints"] = True
            if "joints" in meta.get("missing_annotations", []):
                meta["missing_annotations"] = [
                    a for a in meta["missing_annotations"] if a != "joints"
                ]
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to update metadata for %s: %s", character_id, exc)

    return True
