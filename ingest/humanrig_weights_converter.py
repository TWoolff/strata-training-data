#!/usr/bin/env python3
"""Convert HumanRig raw vertex/bone data into pipeline weight JSON format.

HumanRig raw format (per example):
    vertices.json: {"0": {"coord": [x,y,z], "weight": [22-float array]}, ...}
    bone_3d.json:  {"Hips": [x,y,z], "Spine": [x,y,z], ...}  (22 bones)
    extrinsic.npy: 4x4 camera extrinsic matrix
    intrinsics.npy: 3x3 camera intrinsic matrix

Pipeline weight format (target):
    weights.json: {"vertex_count": N, "image_size": [W,H], "vertices": [
        {"position": [px, py], "weights": {"bone_name": weight}}, ...
    ]}

The 22-element weight array maps to HumanRig's bone order (from bone_3d.json):
    0:Hips, 1:Spine, 2:Spine1, 3:Spine2, 4:Neck, 5:Head,
    6:LeftShoulder, 7:LeftArm, 8:LeftForeArm, 9:LeftHand,
    10:RightShoulder, 11:RightArm, 12:RightForeArm, 13:RightHand,
    14:LeftUpLeg, 15:LeftLeg, 16:LeftFoot, 17:LeftToeBase,
    18:RightUpLeg, 19:RightLeg, 20:RightFoot, 21:RightToeBase

Usage:
    # Convert and add weights to existing bucket-format humanrig data
    python ingest/humanrig_weights_converter.py \
        --raw_dir /Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final \
        --output_dir ./output/humanrig_weights \
        --image_size 512

    # Then upload:
    #   rclone copy ./output/humanrig_weights/ hetzner:strata-training-data/humanrig/ --transfers 32 --fast-list --size-only -P
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# HumanRig 22-bone order (from bone_3d.json key order)
HUMANRIG_BONES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]

# Map HumanRig bone names to Strata 20-bone names
# Spine1/Spine2 merge into "spine", LeftToeBase/RightToeBase merge into foot
HUMANRIG_TO_STRATA = {
    "Hips": "hips",
    "Spine": "spine",
    "Spine1": "spine",       # merge into spine
    "Spine2": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "shoulder_l",
    "LeftArm": "upper_arm_l",
    "LeftForeArm": "forearm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "shoulder_r",
    "RightArm": "upper_arm_r",
    "RightForeArm": "forearm_r",
    "RightHand": "hand_r",
    "LeftUpLeg": "upper_leg_l",
    "LeftLeg": "lower_leg_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "foot_l",   # merge into foot
    "RightUpLeg": "upper_leg_r",
    "RightLeg": "lower_leg_r",
    "RightFoot": "foot_r",
    "RightToeBase": "foot_r",  # merge into foot
}


def project_3d_to_2d(
    coords_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    target_size: int = 512,
    source_size: int = 1024,
) -> np.ndarray:
    """Project 3D coordinates to 2D pixel positions.

    Args:
        coords_3d: (N, 3) array of 3D coordinates
        extrinsic: (4, 4) camera extrinsic matrix
        intrinsic: (3, 3) camera intrinsic matrix
        target_size: output image size (we rescale from source_size)
        source_size: original rendered image size (1024 for HumanRig)

    Returns:
        (N, 2) array of [x, y] pixel positions in target_size space
    """
    n = coords_3d.shape[0]
    # Homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    coords_h = np.hstack([coords_3d, ones])  # (N, 4)

    # Apply extrinsic: world -> camera
    cam_coords = (extrinsic @ coords_h.T).T  # (N, 4)
    cam_xyz = cam_coords[:, :3]  # (N, 3)

    # Apply intrinsic: camera -> pixel
    pixel_h = (intrinsic @ cam_xyz.T).T  # (N, 3)

    # Perspective divide
    z = pixel_h[:, 2:3]
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    pixel_2d = pixel_h[:, :2] / z  # (N, 2)

    # Rescale from source to target size
    scale = target_size / source_size
    pixel_2d *= scale

    return pixel_2d


def convert_example(
    raw_dir: Path,
    example_id: str,
    target_size: int = 512,
) -> dict | None:
    """Convert one HumanRig example to pipeline weight format.

    Returns the weight dict or None if data is missing/invalid.
    """
    vertices_path = raw_dir / example_id / "vertices.json"
    bone_path = raw_dir / example_id / "bone_3d.json"
    extrinsic_path = raw_dir / example_id / "extrinsic.npy"
    intrinsic_path = raw_dir / example_id / "intrinsics.npy"

    for p in [vertices_path, bone_path, extrinsic_path, intrinsic_path]:
        if not p.exists():
            return None

    # Load data
    raw_verts = json.loads(vertices_path.read_text(encoding="utf-8"))
    extrinsic = np.load(extrinsic_path)
    intrinsic = np.load(intrinsic_path)

    # Build bone index from bone_3d.json key order (matches weight array order)
    bone_data = json.loads(bone_path.read_text(encoding="utf-8"))
    bone_names = list(bone_data.keys())

    # Collect 3D coords and weights
    n_verts = len(raw_verts)
    coords_3d = np.zeros((n_verts, 3), dtype=np.float64)
    weight_arrays = np.zeros((n_verts, len(bone_names)), dtype=np.float64)

    for i in range(n_verts):
        key = str(i)
        if key not in raw_verts:
            continue
        v = raw_verts[key]
        coords_3d[i] = v["coord"]
        weight_arrays[i] = v["weight"]

    # Project to 2D
    pixel_2d = project_3d_to_2d(coords_3d, extrinsic, intrinsic, target_size)

    # Build pipeline vertices
    vertices = []
    for i in range(n_verts):
        px, py = int(round(pixel_2d[i, 0])), int(round(pixel_2d[i, 1]))

        # Convert 22-bone weights to Strata 20-bone names (merging where needed)
        strata_weights: dict[str, float] = {}
        for bi, bone_name in enumerate(bone_names):
            w = float(weight_arrays[i, bi])
            if w < 1e-6:
                continue
            strata_name = HUMANRIG_TO_STRATA.get(bone_name)
            if strata_name is None:
                continue
            strata_weights[strata_name] = strata_weights.get(strata_name, 0.0) + w

        # Round small weights
        strata_weights = {
            k: round(v, 6) for k, v in strata_weights.items() if v > 1e-6
        }

        if not strata_weights:
            continue

        vertices.append({
            "position": [px, py],
            "weights": strata_weights,
        })

    if not vertices:
        return None

    return {
        "vertex_count": len(vertices),
        "image_size": [target_size, target_size],
        "character_id": f"humanrig_{example_id}",
        "vertices": vertices,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert HumanRig raw vertex data to pipeline weight format"
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final"),
        help="Path to HumanRig raw data (contains numbered subdirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/humanrig_weights"),
        help="Output directory for converted weight JSONs",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Target image size for 2D projection (default: 512)",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        help="Skip examples that already have weights.json",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    if not raw_dir.is_dir():
        logger.error("Raw directory not found: %s", raw_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all example IDs
    example_ids = sorted(
        [d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    logger.info("Found %d examples in %s", len(example_ids), raw_dir)

    converted = 0
    skipped = 0
    failed = 0

    for i, eid in enumerate(example_ids):
        # Output goes into per-example dir matching bucket format
        # Raw dirs are unpadded ints ("0", "100"), bucket uses 5-digit zero-padded
        padded_id = f"{int(eid):05d}"
        out_example_dir = args.output_dir / f"humanrig_{padded_id}_front"
        out_path = out_example_dir / "weights.json"

        if args.only_new and out_path.exists():
            skipped += 1
            continue

        result = convert_example(raw_dir, eid, args.image_size)
        if result is None:
            failed += 1
            continue

        out_example_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f)

        converted += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "Progress: %d/%d — %d converted, %d skipped, %d failed",
                i + 1, len(example_ids), converted, skipped, failed,
            )

    logger.info(
        "Done: %d converted, %d skipped, %d failed out of %d total",
        converted, skipped, failed, len(example_ids),
    )


if __name__ == "__main__":
    main()
