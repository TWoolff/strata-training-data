#!/usr/bin/env python3
"""Generate 22-class segmentation labels from SAM 3D Body mesh predictions.

SAM 3D Body outputs 3D mesh vertices + 70 keypoints for each detected human.
This script projects the mesh back to 2D and assigns body part labels based on
the nearest joint to each face, creating perfect anatomy-based segmentation.

The MHR body model has fixed topology (~10K vertices), so vertex→body part
assignment is consistent across all predictions. We determine body part per
vertex by finding which of the 70 keypoints is closest in 3D space, then map
to Strata's 22-class schema.

Usage::

    # Run SAM 3D Body first, save outputs as .npz
    python scripts/sam3d_body_to_seg.py \
        --input-dir ./output/sam3d_body_raw/ \
        --output-dir ./output/sam3d_body_seg/ \
        --image-dir ./data_cloud/sora_diverse/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MHR 70 keypoints → Strata 22-class mapping
# MHR keypoint indices (from SAM 3D Body's mhr70 format)
# Groups of keypoints that belong to each Strata region
MHR_TO_STRATA = {
    # Head region (face, eyes, ears, nose, jaw)
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1,  # head keypoints
    # Neck
    5: 2,  # neck
    # Shoulders
    6: 6,   # left shoulder → shoulder_l
    7: 10,  # right shoulder → shoulder_r
    # Upper arms
    8: 7,   # left upper arm → upper_arm_l
    9: 11,  # right upper arm → upper_arm_r
    # Elbows / Forearms
    10: 8,  # left elbow → forearm_l
    11: 12, # right elbow → forearm_r
    # Wrists / Hands
    12: 9,  # left wrist → hand_l
    13: 13, # right wrist → hand_r
    # Spine / Chest
    14: 3, 15: 4,  # chest, spine
    # Hips
    16: 5,  # pelvis → hips
    # Upper legs
    17: 14, # left hip → upper_leg_l
    18: 17, # right hip → upper_leg_r
    # Knees / Lower legs
    19: 15, # left knee → lower_leg_l
    20: 18, # right knee → lower_leg_r
    # Ankles / Feet
    21: 16, # left ankle → foot_l
    22: 19, # right ankle → foot_r
}

# For keypoints 23-69 (hands, face detail), map to parent region
# Left hand fingers (23-41) → hand_l (9)
# Right hand fingers (42-61) → hand_r (13)
# Face detail (62-69) → head (1)
for i in range(23, 42):
    MHR_TO_STRATA[i] = 9   # hand_l
for i in range(42, 62):
    MHR_TO_STRATA[i] = 13  # hand_r
for i in range(62, 70):
    MHR_TO_STRATA[i] = 1   # head

# Strata class names for reference
STRATA_CLASSES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]


def assign_vertex_labels(
    vertices_3d: np.ndarray,
    keypoints_3d: np.ndarray,
) -> np.ndarray:
    """Assign a Strata body part label to each mesh vertex.

    For each vertex, find the nearest keypoint in 3D space, then map
    that keypoint to the corresponding Strata class.

    Args:
        vertices_3d: [N, 3] mesh vertices in 3D.
        keypoints_3d: [K, 3] keypoint positions in 3D (K=70 for MHR).

    Returns:
        [N] uint8 array of Strata class IDs (0-21).
    """
    n_verts = vertices_3d.shape[0]
    n_kps = min(keypoints_3d.shape[0], 70)

    # Compute distance from each vertex to each keypoint
    # vertices_3d: [N, 3], keypoints_3d: [K, 3]
    dists = np.linalg.norm(
        vertices_3d[:, None, :] - keypoints_3d[None, :n_kps, :],
        axis=2,
    )  # [N, K]

    # Find nearest keypoint for each vertex
    nearest_kp = np.argmin(dists, axis=1)  # [N]

    # Map to Strata class
    labels = np.zeros(n_verts, dtype=np.uint8)
    for vi in range(n_verts):
        kp_idx = nearest_kp[vi]
        labels[vi] = MHR_TO_STRATA.get(kp_idx, 0)

    return labels


def project_vertices_to_2d(
    vertices_3d: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Project 3D vertices to 2D image coordinates.

    Uses weak perspective projection (SAM 3D Body's default).

    Args:
        vertices_3d: [N, 3] vertices.
        cam_t: [3] camera translation.
        focal_length: focal length in pixels.
        img_w, img_h: image dimensions.

    Returns:
        [N, 2] pixel coordinates.
    """
    # Apply camera translation
    verts = vertices_3d + cam_t[None, :]

    # Perspective projection
    x = verts[:, 0] / verts[:, 2] * focal_length + img_w / 2
    y = verts[:, 1] / verts[:, 2] * focal_length + img_h / 2

    return np.stack([x, y], axis=1)


def rasterize_mesh_labels(
    vertices_2d: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Rasterize labeled mesh faces to a 2D segmentation mask.

    For each triangle face, fill the pixels inside with the majority
    vertex label of that face.

    Args:
        vertices_2d: [N, 2] projected vertex positions.
        faces: [F, 3] face indices.
        vertex_labels: [N] body part labels per vertex.
        img_w, img_h: output mask dimensions.

    Returns:
        [H, W] uint8 segmentation mask.
    """
    import cv2

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Z-buffer for depth sorting (use mean vertex Z as depth)
    z_buffer = np.full((img_h, img_w), float('inf'), dtype=np.float32)

    for face in faces:
        v0, v1, v2 = face
        pts = vertices_2d[[v0, v1, v2]].astype(np.int32)

        # Majority label for this face
        face_labels = vertex_labels[[v0, v1, v2]]
        label = np.bincount(face_labels).argmax()

        # Fill triangle
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], int(label))

    return mask


def process_sam3d_output(
    output: dict,
    faces: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Convert a single SAM 3D Body output to a segmentation mask.

    Args:
        output: dict with pred_vertices, pred_keypoints_3d, pred_cam_t, focal_length
        faces: [F, 3] mesh face indices (shared MHR topology)
        img_w, img_h: original image dimensions

    Returns:
        [H, W] uint8 segmentation mask (Strata 22-class).
    """
    vertices_3d = output["pred_vertices"]      # [N, 3]
    keypoints_3d = output["pred_keypoints_3d"] # [70, 3]
    cam_t = output["pred_cam_t"]               # [3]
    focal_length = output["focal_length"]      # scalar

    # Assign body part labels to each vertex
    vertex_labels = assign_vertex_labels(vertices_3d, keypoints_3d)

    # Project to 2D
    vertices_2d = project_vertices_to_2d(
        vertices_3d, cam_t, focal_length, img_w, img_h,
    )

    # Rasterize to segmentation mask
    seg_mask = rasterize_mesh_labels(
        vertices_2d, faces, vertex_labels, img_w, img_h,
    )

    return seg_mask


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM 3D Body outputs to 22-class segmentation labels"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with SAM 3D Body .npz outputs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for segmentation.png files")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Original image directory (for dimensions)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.input_dir.glob("*.npz"))
    logger.info("Found %d SAM 3D Body outputs", len(npz_files))

    for npz_path in npz_files:
        name = npz_path.stem
        try:
            data = np.load(npz_path, allow_pickle=True)

            # Get image dimensions
            if args.image_dir:
                img_path = args.image_dir / f"{name}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                else:
                    img_w, img_h = 512, 512
            else:
                img_w, img_h = 512, 512

            output = {k: data[k] for k in data.files}
            faces = data.get("faces", None)

            seg_mask = process_sam3d_output(output, faces, img_w, img_h)

            out_path = args.output_dir / f"{name}_seg.png"
            Image.fromarray(seg_mask, mode="L").save(out_path)
            logger.info("Saved %s (%d classes present)", out_path.name,
                       len(np.unique(seg_mask)))

        except Exception as e:
            logger.warning("Error on %s: %s", name, e)

    logger.info("Done!")


if __name__ == "__main__":
    main()
