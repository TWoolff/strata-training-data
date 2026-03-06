#!/usr/bin/env python3
"""Convert UniRig raw NPZ data into pipeline weight JSON format.

UniRig raw format (per example):
    raw_data.npz with keys:
        vertices: (V, 3) float16 — 3D vertex positions
        skin: (V, N) float16 — per-vertex skinning weights for N bones
        names: (N,) str — bone/joint names
        joints: (N, 3) float16 — 3D joint positions

Pipeline weight format (target):
    weights.json: {"vertex_count": V, "image_size": [512, 512], "vertices": [
        {"position": [px, py], "weights": {"bone_name": weight}}, ...
    ]}

Uses the same orthographic camera setup as the Blender adapter (unirig_adapter.py)
to project 3D vertices to 2D pixel positions matching the rendered images.

Usage:
    python ingest/unirig_weights_converter.py \
        --raw_dir /Volumes/TAMWoolff/data/preprocessed/unirig/processed/rigxl \
        --output_dir ./output/unirig_weights

    # Then upload:
    #   rclone copy ./output/unirig_weights/ hetzner:strata-training-data/unirig/ --transfers 32 --fast-list --size-only -P
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from ingest.unirig_skeleton_mapper import map_joint_name
from pipeline.config import REGION_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Strata bone names (20 bones) — must match training/data/transforms.py BONE_ORDER
STRATA_BONES = [
    "hips", "spine", "chest", "neck", "head",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "hair_back",
]
STRATA_BONE_SET = set(STRATA_BONES)

RESOLUTION = 512


def ortho_project_front(
    vertices_3d: np.ndarray,
    resolution: int = RESOLUTION,
) -> np.ndarray:
    """Project 3D vertices to 2D using orthographic front view.

    Replicates the camera setup from unirig_adapter.py:
    - Front view (azimuth=0): camera looks along -Y axis
    - X maps to horizontal, Z maps to vertical
    - ortho_scale = max(height, width, depth) * 1.1
    - Centered on bounding box center

    Args:
        vertices_3d: (V, 3) array of 3D positions
        resolution: output image resolution

    Returns:
        (V, 2) array of [x, y] pixel positions
    """
    xs = vertices_3d[:, 0]
    ys = vertices_3d[:, 1]
    zs = vertices_3d[:, 2]

    cx = (xs.min() + xs.max()) / 2
    cz = (zs.min() + zs.max()) / 2

    height = zs.max() - zs.min()
    width = xs.max() - xs.min()
    depth = ys.max() - ys.min()
    ortho_scale = max(height, width, depth) * 1.1

    if ortho_scale < 1e-6:
        ortho_scale = 1.0

    # Normalize to [0, 1] range centered on bbox
    # X: left-to-right in image
    # Z: top-to-bottom in image (flip Z since image Y grows downward)
    norm_x = (xs - cx) / ortho_scale + 0.5
    norm_z = -(zs - cz) / ortho_scale + 0.5

    # Scale to pixel coordinates
    px = norm_x * resolution
    pz = norm_z * resolution

    return np.stack([px, pz], axis=1)


def convert_example(
    npz_path: Path,
    resolution: int = RESOLUTION,
) -> dict | None:
    """Convert one UniRig example to pipeline weight format.

    Returns the weight dict or None if data is missing/invalid.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except (OSError, ValueError):
        return None

    required_keys = {"vertices", "skin", "names"}
    if not required_keys.issubset(set(data.keys())):
        return None

    vertices_3d = data["vertices"].astype(np.float64)
    skin = data["skin"].astype(np.float64)
    bone_names = [str(n) for n in data["names"]]

    n_verts = vertices_3d.shape[0]
    n_bones = len(bone_names)

    if n_verts == 0 or n_bones == 0:
        return None
    if skin.shape != (n_verts, n_bones):
        return None

    # Map bone names to Strata region names
    # bone_index -> strata_bone_name (or None if unmapped/not a body bone)
    bone_to_strata: list[str | None] = []
    for bname in bone_names:
        jm = map_joint_name(bname)
        if jm.region_name is not None and jm.region_name in STRATA_BONE_SET:
            bone_to_strata.append(jm.region_name)
        else:
            bone_to_strata.append(None)

    # Check we have at least some mapped bones
    mapped_count = sum(1 for b in bone_to_strata if b is not None)
    if mapped_count < 5:
        return None

    # Project to 2D
    pixel_2d = ortho_project_front(vertices_3d, resolution)

    # Build pipeline vertices
    vertices = []
    for vi in range(n_verts):
        px = int(round(pixel_2d[vi, 0]))
        py = int(round(pixel_2d[vi, 1]))

        # Skip vertices outside image bounds
        if px < 0 or px >= resolution or py < 0 or py >= resolution:
            continue

        # Aggregate weights by Strata bone name
        strata_weights: dict[str, float] = {}
        for bi in range(n_bones):
            w = skin[vi, bi]
            if w < 1e-6:
                continue
            sname = bone_to_strata[bi]
            if sname is None:
                continue
            strata_weights[sname] = strata_weights.get(sname, 0.0) + w

        # Round and filter (cast to float for JSON serialization)
        strata_weights = {
            k: round(float(v), 6) for k, v in strata_weights.items() if v > 1e-6
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
        "image_size": [resolution, resolution],
        "vertices": vertices,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert UniRig raw NPZ data to pipeline weight format"
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/Volumes/TAMWoolff/data/preprocessed/unirig/processed/rigxl"),
        help="Path to UniRig rigxl directory (contains numbered subdirs with raw_data.npz)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/unirig_weights"),
        help="Output directory for converted weight JSONs",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        help="Target image resolution (default: 512)",
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

    # Discover all example IDs (subdirs containing raw_data.npz)
    example_ids = sorted(
        d.name for d in raw_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "raw_data.npz").exists()
    )
    logger.info("Found %d examples in %s", len(example_ids), raw_dir)

    converted = 0
    skipped = 0
    failed = 0

    for i, eid in enumerate(example_ids):
        # Output matches bucket structure: {eid}/front/weights.json
        out_dir = args.output_dir / eid / "front"
        out_path = out_dir / "weights.json"

        if args.only_new and out_path.exists():
            skipped += 1
            continue

        npz_path = raw_dir / eid / "raw_data.npz"
        result = convert_example(npz_path, args.resolution)

        if result is None:
            failed += 1
            continue

        result["character_id"] = f"unirig_{eid}"
        out_dir.mkdir(parents=True, exist_ok=True)
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
