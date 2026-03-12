"""Generate 22-class segmentation masks using SAM2 + joint-conditioned region assignment.

SAM2 produces precise segment boundaries but has no concept of body regions.
Joint positions (from joints.json) provide semantic assignment: each SAM2
segment is assigned to the body region whose joint falls inside it.

This produces much better pseudo-labels than our trained seg model (0.545 mIoU),
because SAM2's boundaries are sharper and joint-based assignment is deterministic.

Usage::

    # Process a dataset with joints.json files
    python scripts/run_sam2_pseudolabel.py \
        --input-dir ./data_cloud/gemini_diverse \
        --sam2-checkpoint ./models/sam2.1_hiera_large.pt \
        --sam2-config sam2.1_hiera_l \
        --device cuda

    # Only process examples missing segmentation.png
    python scripts/run_sam2_pseudolabel.py \
        --input-dir ./data_cloud/anime_seg \
        --sam2-checkpoint ./models/sam2.1_hiera_large.pt \
        --sam2-config sam2.1_hiera_l \
        --only-missing --device cuda

    # Use smaller model on Mac
    python scripts/run_sam2_pseudolabel.py \
        --input-dir ./data_cloud/gemini_diverse \
        --sam2-checkpoint ./models/sam2.1_hiera_small.pt \
        --sam2-config sam2.1_hiera_s \
        --device mps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_components

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region definitions (matches pipeline/config.py)
# ---------------------------------------------------------------------------

REGION_NAME_TO_ID: dict[str, int] = {
    "background": 0,
    "head": 1,
    "neck": 2,
    "chest": 3,
    "spine": 4,
    "hips": 5,
    "shoulder_l": 6,
    "upper_arm_l": 7,
    "forearm_l": 8,
    "hand_l": 9,
    "shoulder_r": 10,
    "upper_arm_r": 11,
    "forearm_r": 12,
    "hand_r": 13,
    "upper_leg_l": 14,
    "lower_leg_l": 15,
    "foot_l": 16,
    "upper_leg_r": 17,
    "lower_leg_r": 18,
    "foot_r": 19,
    "accessory": 20,
    "hair_back": 21,
}

# Skeleton connectivity for interpolating regions between joints.
# Each bone connects two joints: regions along the bone get assigned
# based on proximity to the nearest endpoint.
SKELETON_BONES: list[tuple[str, str]] = [
    ("head", "neck"),
    ("neck", "chest"),
    ("chest", "spine"),
    ("spine", "hips"),
    ("chest", "shoulder_l"),
    ("shoulder_l", "upper_arm_l"),
    ("upper_arm_l", "forearm_l"),
    ("forearm_l", "hand_l"),
    ("chest", "shoulder_r"),
    ("shoulder_r", "upper_arm_r"),
    ("upper_arm_r", "forearm_r"),
    ("forearm_r", "hand_r"),
    ("hips", "upper_leg_l"),
    ("upper_leg_l", "lower_leg_l"),
    ("lower_leg_l", "foot_l"),
    ("hips", "upper_leg_r"),
    ("upper_leg_r", "lower_leg_r"),
    ("lower_leg_r", "foot_r"),
]

# Minimum joint confidence to use for region assignment
MIN_JOINT_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Joint loading
# ---------------------------------------------------------------------------


def load_joints(joints_path: Path) -> dict[str, tuple[int, int, float]]:
    """Load joints from joints.json.

    Returns:
        Dict mapping region_name -> (x, y, confidence).
        Only includes joints with confidence >= MIN_JOINT_CONFIDENCE.
    """
    data = json.loads(joints_path.read_text(encoding="utf-8"))
    joints_data = data.get("joints", {})
    joints: dict[str, tuple[int, int, float]] = {}
    for name, info in joints_data.items():
        conf = info.get("confidence", 0.0)
        if conf < MIN_JOINT_CONFIDENCE:
            continue
        pos = info["position"]
        joints[name] = (int(pos[0]), int(pos[1]), conf)
    return joints


# ---------------------------------------------------------------------------
# SAM2 mask generation
# ---------------------------------------------------------------------------


def generate_sam2_masks(
    image_np: np.ndarray,
    mask_generator,
) -> list[dict]:
    """Run SAM2 automatic mask generation on an image.

    Args:
        image_np: RGB uint8 array [H, W, 3].
        mask_generator: SAM2AutomaticMaskGenerator instance.

    Returns:
        List of mask dicts sorted by quality (best first).
        Each has 'segmentation' (bool [H,W]), 'area', 'predicted_iou',
        'stability_score'.
    """
    masks = mask_generator.generate(image_np)

    # Sort by quality: predicted_iou * stability_score, descending
    masks.sort(
        key=lambda m: m["predicted_iou"] * m["stability_score"],
        reverse=True,
    )
    return masks


# ---------------------------------------------------------------------------
# Region assignment
# ---------------------------------------------------------------------------


def assign_regions_spatial(
    sam2_masks: list[dict],
    image_shape: tuple[int, int],
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Assign SAM2 segments to body regions using spatial heuristics (no joints).

    Uses vertical position of each segment's centroid to assign regions:
    - Top ~15%: head (1)
    - 15-20%: neck (2)
    - 20-40%: chest (3) or shoulders/arms based on horizontal position
    - 40-50%: spine (4)
    - 50-55%: hips (5)
    - 55-75%: upper legs (14/17) based on left/right
    - 75-90%: lower legs (15/18)
    - 90-100%: feet (16/19)
    Arms are assigned based on horizontal position (left third / right third).

    Args:
        sam2_masks: SAM2 masks sorted by quality (best first).
        image_shape: (H, W) of the image.
        alpha: Optional alpha channel [H, W] uint8.

    Returns:
        Segmentation mask [H, W] uint8 with region IDs (0-21).
    """
    h, w = image_shape
    result = np.zeros((h, w), dtype=np.uint8)
    assigned = np.zeros((h, w), dtype=bool)

    # Find foreground bounding box from alpha
    if alpha is not None:
        fg_mask = alpha > 0
        if fg_mask.any():
            fg_ys, fg_xs = np.where(fg_mask)
            bbox_top, bbox_bot = fg_ys.min(), fg_ys.max()
            bbox_left, bbox_right = fg_xs.min(), fg_xs.max()
        else:
            bbox_top, bbox_bot, bbox_left, bbox_right = 0, h - 1, 0, w - 1
    else:
        bbox_top, bbox_bot, bbox_left, bbox_right = 0, h - 1, 0, w - 1

    fg_h = max(bbox_bot - bbox_top, 1)
    fg_w = max(bbox_right - bbox_left, 1)
    mid_x = bbox_left + fg_w / 2

    for mask_info in sam2_masks:
        seg_mask = mask_info["segmentation"]
        available = seg_mask & ~assigned
        if available.sum() < 10:
            continue

        ys, xs = np.where(available)
        centroid_y = ys.mean()
        centroid_x = xs.mean()

        # Normalize position relative to foreground bounding box
        rel_y = (centroid_y - bbox_top) / fg_h  # 0=top, 1=bottom
        rel_x = (centroid_x - mid_x) / (fg_w / 2)  # -1=left, 0=center, 1=right
        is_left = rel_x < -0.15
        is_right = rel_x > 0.15

        # Assign region based on vertical position
        if rel_y < 0.15:
            region_id = 1  # head
        elif rel_y < 0.20:
            region_id = 2  # neck
        elif rel_y < 0.40:
            if is_left:
                region_id = 7  # upper_arm_l
            elif is_right:
                region_id = 11  # upper_arm_r
            else:
                region_id = 3  # chest
        elif rel_y < 0.50:
            if is_left:
                region_id = 8  # forearm_l
            elif is_right:
                region_id = 12  # forearm_r
            else:
                region_id = 4  # spine
        elif rel_y < 0.55:
            if is_left:
                region_id = 9  # hand_l
            elif is_right:
                region_id = 13  # hand_r
            else:
                region_id = 5  # hips
        elif rel_y < 0.75:
            region_id = 14 if rel_x < 0 else 17  # upper_leg_l/r
        elif rel_y < 0.90:
            region_id = 15 if rel_x < 0 else 18  # lower_leg_l/r
        else:
            region_id = 16 if rel_x < 0 else 19  # foot_l/r

        result[available] = region_id
        assigned[available] = True

    if alpha is not None:
        result[alpha == 0] = 0

    return result


def assign_regions(
    sam2_masks: list[dict],
    joints: dict[str, tuple[int, int, float]],
    image_shape: tuple[int, int],
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Assign SAM2 segments to body regions using joint positions.

    Strategy:
    1. For each SAM2 segment, find which joints fall inside it.
    2. If exactly one joint: assign that region.
    3. If multiple joints: assign the region of the joint closest to
       the segment's centroid.
    4. If no joints: assign the region of the nearest joint (Euclidean
       distance from segment centroid to all joint positions).

    Overlapping SAM2 masks are resolved by priority (highest quality first).

    Args:
        sam2_masks: SAM2 masks sorted by quality (best first).
        joints: Dict mapping region_name -> (x, y, confidence).
        image_shape: (H, W) of the image.
        alpha: Optional alpha channel [H, W] uint8. Background where alpha==0.

    Returns:
        Segmentation mask [H, W] uint8 with region IDs (0-21).
    """
    h, w = image_shape
    result = np.zeros((h, w), dtype=np.uint8)  # 0 = background
    assigned = np.zeros((h, w), dtype=bool)  # track which pixels are assigned

    if not joints:
        logger.warning("No valid joints — falling back to spatial assignment")
        return assign_regions_spatial(sam2_masks, image_shape, alpha)

    # Precompute joint positions as arrays for distance calculations
    joint_names = list(joints.keys())
    joint_positions = np.array([(joints[n][0], joints[n][1]) for n in joint_names])  # [N, 2] (x, y)

    for mask_info in sam2_masks:
        seg_mask = mask_info["segmentation"]  # bool [H, W]

        # Skip pixels already assigned by a higher-quality mask
        available = seg_mask & ~assigned
        if available.sum() < 10:
            continue

        # Find segment centroid
        ys, xs = np.where(available)
        centroid_x = xs.mean()
        centroid_y = ys.mean()

        # Find which joints fall inside this segment
        joints_inside: list[tuple[str, float]] = []
        for name, (jx, jy, conf) in joints.items():
            if 0 <= jy < h and 0 <= jx < w and seg_mask[jy, jx]:
                joints_inside.append((name, conf))

        if len(joints_inside) == 1:
            region_name = joints_inside[0][0]
        elif len(joints_inside) > 1:
            # Multiple joints — pick the one closest to centroid
            best_name = None
            best_dist = float("inf")
            for name, _ in joints_inside:
                jx, jy, _ = joints[name]
                dist = (jx - centroid_x) ** 2 + (jy - centroid_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            region_name = best_name
        else:
            # No joints inside — assign nearest joint by distance
            centroid = np.array([centroid_x, centroid_y])
            distances = np.sqrt(((joint_positions - centroid) ** 2).sum(axis=1))
            nearest_idx = distances.argmin()
            region_name = joint_names[nearest_idx]

        region_id = REGION_NAME_TO_ID.get(region_name, 0)
        result[available] = region_id
        assigned[available] = True

    # Set background where alpha is transparent
    if alpha is not None:
        result[alpha == 0] = 0

    return result


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def cleanup_mask(mask: np.ndarray, min_region_size: int = 50) -> np.ndarray:
    """Remove small isolated regions by absorbing into surrounding region.

    For each region ID, find connected components smaller than min_region_size
    and replace them with the most common neighboring region.
    """
    result = mask.copy()
    h, w = mask.shape

    for region_id in range(1, 22):  # skip background
        region_mask = result == region_id
        if not region_mask.any():
            continue

        labeled, n_components = connected_components(region_mask)
        for comp_id in range(1, n_components + 1):
            component = labeled == comp_id
            if component.sum() >= min_region_size:
                continue

            # Find neighboring pixels (dilate by 1)
            ys, xs = np.where(component)
            neighbors = set()
            for y, x in zip(ys, xs):
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not component[ny, nx]:
                        neighbors.add((ny, nx))

            if not neighbors:
                continue

            # Replace with most common neighbor region
            neighbor_regions = [result[ny, nx] for ny, nx in neighbors]
            if neighbor_regions:
                most_common = max(set(neighbor_regions), key=neighbor_regions.count)
                result[component] = most_common

    return result


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    mask: np.ndarray,
    joints: dict[str, tuple[int, int, float]],
    alpha: np.ndarray | None = None,
) -> dict:
    """Compute quality metrics for a pseudo-labeled mask.

    Returns:
        Dict with quality stats for logging/filtering.
    """
    unique_regions = set(np.unique(mask)) - {0}
    fg_pixels = (mask > 0).sum() if alpha is None else ((mask > 0) & (alpha > 0)).sum()
    total_fg = (alpha > 0).sum() if alpha is not None else (mask > 0).sum()

    # Check how many joints fall inside their assigned region
    joints_matched = 0
    joints_total = len(joints)
    h, w = mask.shape
    for name, (jx, jy, _) in joints.items():
        expected_id = REGION_NAME_TO_ID.get(name, 0)
        if 0 <= jy < h and 0 <= jx < w and mask[jy, jx] == expected_id:
            joints_matched += 1

    return {
        "num_regions": len(unique_regions),
        "fg_coverage": float(fg_pixels / max(total_fg, 1)),
        "joints_matched": joints_matched,
        "joints_total": joints_total,
        "joints_match_rate": joints_matched / max(joints_total, 1),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_directory(
    input_dir: Path,
    sam2_checkpoint: str,
    sam2_config: str,
    device: str = "cuda",
    only_missing: bool = False,
    points_per_side: int = 32,
    min_region_size: int = 50,
) -> dict:
    """Process all examples in a directory with SAM2 pseudo-labeling.

    Args:
        input_dir: Directory with per-example subdirs containing image.png + joints.json.
        sam2_checkpoint: Path to SAM2 checkpoint file.
        sam2_config: SAM2 model config name (e.g., "sam2.1_hiera_l").
        device: Device to run on ("cuda", "mps", "cpu").
        only_missing: Only process examples without segmentation.png.
        points_per_side: SAM2 grid density (16=fast, 32=quality).
        min_region_size: Minimum connected component size to keep.

    Returns:
        Dict with processing stats.
    """
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Build SAM2
    logger.info("Loading SAM2 from %s (config: %s) on %s", sam2_checkpoint, sam2_config, device)
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )
    logger.info("SAM2 loaded, points_per_side=%d", points_per_side)

    # Discover examples (joints.json is optional — spatial fallback used if missing)
    examples: list[Path] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        image_path = child / "image.png"
        if not image_path.exists():
            continue
        if only_missing and (child / "segmentation.png").exists():
            continue
        examples.append(child)

    logger.info("Found %d examples to process in %s", len(examples), input_dir)
    if not examples:
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    stats = {"total": len(examples), "processed": 0, "skipped": 0, "failed": 0}
    all_metrics: list[dict] = []
    t0 = time.time()

    for i, example_dir in enumerate(examples):
        try:
            # Load image
            img_pil = Image.open(example_dir / "image.png").convert("RGBA")
            alpha = np.array(img_pil)[:, :, 3]
            img_rgb = np.array(img_pil.convert("RGB"))

            # Load joints (optional — spatial fallback if missing)
            joints_path = example_dir / "joints.json"
            has_joints = joints_path.exists()
            if has_joints:
                joints = load_joints(joints_path)
                if len(joints) < 3:
                    joints = {}
                    has_joints = False
            else:
                joints = {}

            # Generate SAM2 masks
            sam2_masks = generate_sam2_masks(img_rgb, mask_generator)

            # Assign regions (joint-based or spatial fallback)
            seg_mask = assign_regions(sam2_masks, joints, img_rgb.shape[:2], alpha=alpha)

            # Cleanup small regions
            seg_mask = cleanup_mask(seg_mask, min_region_size=min_region_size)

            # Quality metrics
            seg_source = "sam2_joint" if has_joints else "sam2_spatial"
            unique_regions = set(np.unique(seg_mask)) - {0}
            num_regions = len(unique_regions)

            if has_joints:
                metrics = compute_quality_metrics(seg_mask, joints, alpha=alpha)
            else:
                # Spatial mode: no joints to match against
                fg_pixels = (seg_mask > 0).sum() if alpha is None else ((seg_mask > 0) & (alpha > 0)).sum()
                total_fg = (alpha > 0).sum() if alpha is not None else (seg_mask > 0).sum()
                metrics = {
                    "num_regions": num_regions,
                    "fg_coverage": float(fg_pixels / max(total_fg, 1)),
                    "joints_matched": 0,
                    "joints_total": 0,
                    "joints_match_rate": 1.0,  # no joints to fail
                }
            metrics["example_id"] = example_dir.name
            metrics["source"] = seg_source
            all_metrics.append(metrics)

            # Skip low-quality results (spatial mode: only check region count)
            min_regions = 3 if has_joints else 2
            if num_regions < min_regions:
                logger.debug(
                    "Low quality for %s (regions=%d) — skipping",
                    example_dir.name, num_regions,
                )
                stats["skipped"] += 1
                continue
            if has_joints and metrics["joints_match_rate"] < 0.3:
                logger.debug(
                    "Low joint match for %s (%.1f%%) — skipping",
                    example_dir.name, metrics["joints_match_rate"] * 100,
                )
                stats["skipped"] += 1
                continue

            # Save segmentation mask
            seg_img = Image.fromarray(seg_mask, mode="L")
            seg_img.save(example_dir / "segmentation.png")

            # Update metadata
            meta_path = example_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = {}
            meta["segmentation_source"] = seg_source
            meta["sam2_quality"] = {
                "num_regions": num_regions,
                "joints_match_rate": round(metrics["joints_match_rate"], 3),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            stats["processed"] += 1

        except Exception:
            logger.exception("Failed to process %s", example_dir.name)
            stats["failed"] += 1

        # Progress logging
        if (i + 1) % 100 == 0 or i + 1 == len(examples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info(
                "  %d/%d done (%.1f img/s) — %d processed, %d skipped, %d failed",
                i + 1, len(examples), rate,
                stats["processed"], stats["skipped"], stats["failed"],
            )

    elapsed = time.time() - t0

    # Save stats
    stats_path = input_dir / "sam2_pseudolabel_stats.json"
    stats_data = {
        **stats,
        "elapsed_seconds": round(elapsed, 1),
        "rate_img_per_sec": round(len(examples) / max(elapsed, 1), 2),
        "points_per_side": points_per_side,
        "min_region_size": min_region_size,
    }

    # Aggregate quality metrics
    if all_metrics:
        match_rates = [m["joints_match_rate"] for m in all_metrics]
        region_counts = [m["num_regions"] for m in all_metrics]
        stats_data["quality_summary"] = {
            "mean_joints_match_rate": round(np.mean(match_rates), 3),
            "median_joints_match_rate": round(float(np.median(match_rates)), 3),
            "mean_num_regions": round(np.mean(region_counts), 1),
            "min_num_regions": int(np.min(region_counts)),
        }

    stats_path.write_text(json.dumps(stats_data, indent=2), encoding="utf-8")
    logger.info("Stats saved to %s", stats_path)
    logger.info(
        "Done: %d processed, %d skipped, %d failed in %.0fs",
        stats["processed"], stats["skipped"], stats["failed"], elapsed,
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 22-class seg masks using SAM2 + joint positions"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--sam2-checkpoint", type=str, required=True, help="SAM2 checkpoint path")
    parser.add_argument(
        "--sam2-config", type=str, default="sam2.1_hiera_l",
        help="SAM2 config name (sam2.1_hiera_s, sam2.1_hiera_l, etc.)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, mps, cpu)")
    parser.add_argument("--only-missing", action="store_true", help="Skip examples with existing seg")
    parser.add_argument(
        "--points-per-side", type=int, default=32,
        help="SAM2 grid density (16=fast ~4x, 32=quality)",
    )
    parser.add_argument(
        "--min-region-size", type=int, default=50,
        help="Min connected component size to keep",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    process_directory(
        input_dir=Path(args.input_dir),
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        only_missing=args.only_missing,
        points_per_side=args.points_per_side,
        min_region_size=args.min_region_size,
    )


if __name__ == "__main__":
    main()
