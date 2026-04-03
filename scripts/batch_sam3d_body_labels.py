#!/usr/bin/env python3
"""Batch-generate anatomy segmentation labels using SAM 3D Body.

Runs SAM 3D Body on all illustrated character images, projects the 3D mesh
back to 2D to create perfect anatomy-based 22-class segmentation labels.

These labels can then be added to SAM 3 seg fine-tuning data.

Requires: SAM 3D Body installed at /workspace/sam-3d-body with checkpoints.

Usage::

    python scripts/batch_sam3d_body_labels.py \
        --input-dir ./data_cloud/sora_diverse \
        --output-dir ./data_cloud/sam3d_body_labels \
        --checkpoint-path /workspace/sam-3d-body/checkpoints/dinov3/model.ckpt \
        --mhr-path /workspace/sam-3d-body/checkpoints/dinov3/assets/mhr_model.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Add SAM 3D Body to path
sys.path.insert(0, "/workspace/sam-3d-body")

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, mhr_path: str, device: torch.device):
    """Load SAM 3D Body model."""
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    model, model_cfg = load_sam_3d_body(
        checkpoint_path, device=device, mhr_path=mhr_path
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    return estimator


def output_to_seg_mask(
    output: dict,
    faces: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Convert SAM 3D Body output to 22-class segmentation mask."""
    from scripts.sam3d_body_to_seg import (
        assign_vertex_labels,
        project_vertices_to_2d,
        rasterize_mesh_labels,
    )

    vertices_3d = output["pred_vertices"]
    keypoints_3d = output["pred_keypoints_3d"]
    cam_t = output["pred_cam_t"]
    focal_length = float(output["focal_length"])

    vertex_labels = assign_vertex_labels(vertices_3d, keypoints_3d)
    vertices_2d = project_vertices_to_2d(
        vertices_3d, cam_t, focal_length, img_w, img_h
    )
    seg_mask = rasterize_mesh_labels(
        vertices_2d, faces, vertex_labels, img_w, img_h
    )
    return seg_mask


def main():
    parser = argparse.ArgumentParser(
        description="Batch SAM 3D Body → segmentation labels"
    )
    parser.add_argument("--input-dirs", type=str, nargs="+", required=True,
                        help="Directories with illustrated character images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for labeled examples")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--mhr-path", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0=all)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Skip images where SAM 3D Body confidence is too low")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SAM 3D Body model...")
    estimator = load_model(args.checkpoint_path, args.mhr_path, device)
    faces = estimator.faces  # Shared MHR topology

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    examples = []
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)
        for ex_dir in sorted(input_dir.iterdir()):
            if not ex_dir.is_dir():
                continue
            img_path = ex_dir / "image.png"
            if img_path.exists():
                examples.append((img_path, ex_dir.name, input_dir.name))

    logger.info("Found %d images", len(examples))

    if args.max_images > 0:
        examples = examples[:args.max_images]

    processed = 0
    skipped = 0
    failed = 0

    for i, (img_path, ex_name, ds_name) in enumerate(examples):
        try:
            # Run SAM 3D Body
            outputs = estimator.process_one_image(
                str(img_path), bbox_thr=0.0, use_mask=False
            )

            if not outputs:
                skipped += 1
                continue

            # Use first detection
            output = outputs[0]

            # Get image dimensions
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Convert to segmentation mask
            seg_mask = output_to_seg_mask(output, faces, img_w, img_h)

            # Check quality: at least 4 distinct regions, reasonable coverage
            n_classes = len(np.unique(seg_mask))
            foreground_frac = np.count_nonzero(seg_mask) / seg_mask.size

            if n_classes < 4 or foreground_frac < 0.05:
                skipped += 1
                continue

            # Save in seg-compatible format
            out_dir = output_dir / f"{ds_name}_{ex_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Copy original image
            import shutil
            shutil.copy2(img_path, out_dir / "image.png")

            # Save segmentation mask
            Image.fromarray(seg_mask, mode="L").save(out_dir / "segmentation.png")

            # Save metadata
            import json
            meta = {
                "source": "sam3d_body",
                "original_dataset": ds_name,
                "original_example": ex_name,
                "n_classes": int(n_classes),
                "foreground_fraction": float(foreground_frac),
                "keypoints_3d_available": True,
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f)

            # Also save the raw SAM 3D Body output for future use
            np.savez_compressed(
                out_dir / "sam3d_body_output.npz",
                pred_vertices=output["pred_vertices"],
                pred_keypoints_3d=output["pred_keypoints_3d"],
                pred_keypoints_2d=output["pred_keypoints_2d"],
                pred_cam_t=output["pred_cam_t"],
                focal_length=output["focal_length"],
                faces=faces,
            )

            processed += 1

        except Exception as e:
            logger.warning("Error on %s/%s: %s", ds_name, ex_name, e)
            failed += 1

        if (i + 1) % 100 == 0:
            logger.info(
                "Progress: %d/%d (processed=%d, skipped=%d, failed=%d)",
                i + 1, len(examples), processed, skipped, failed,
            )

    logger.info(
        "Done! processed=%d, skipped=%d, failed=%d (total=%d)",
        processed, skipped, failed, len(examples),
    )


if __name__ == "__main__":
    main()
