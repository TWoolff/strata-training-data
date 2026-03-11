"""Test HumanRig joint reprojection accuracy.

Compares reprojected front-view joints (bone_3d.json + camera matrices → 2D)
against ground-truth bone_2d.json to verify the reprojection math is correct.
Also generates overlay visualizations at all 5 camera angles.

Usage:
    python3 scripts/test_joint_reprojection.py [--samples N] [--output_dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest.humanrig_adapter import (
    ANGLE_CONFIGS,
    ORIGINAL_RESOLUTION,
    STRATA_RESOLUTION,
    _build_strata_joints,
    _load_bone_2d,
    _load_bone_3d,
    _load_camera,
    _project_joints,
)

HUMANRIG_ROOT = Path(
    "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final"
)

# Joint colors for visualization (one per region)
JOINT_COLORS = [
    (255, 0, 0),      # head
    (255, 128, 0),    # neck
    (255, 255, 0),    # chest
    (128, 255, 0),    # spine
    (0, 255, 0),      # hips
    (0, 255, 128),    # shoulder_l
    (0, 255, 255),    # upper_arm_l
    (0, 128, 255),    # forearm_l
    (0, 0, 255),      # hand_l
    (128, 0, 255),    # shoulder_r
    (255, 0, 255),    # upper_arm_r
    (255, 0, 128),    # forearm_r
    (128, 0, 128),    # hand_r
    (255, 128, 128),  # upper_leg_l
    (128, 255, 128),  # lower_leg_l
    (128, 128, 255),  # foot_l
    (255, 128, 255),  # upper_leg_r
    (128, 255, 255),  # lower_leg_r
    (255, 255, 128),  # foot_r
]

# Skeleton connections (pairs of region indices 0-18)
SKELETON_CONNECTIONS = [
    (0, 1),   # head → neck
    (1, 2),   # neck → chest
    (2, 3),   # chest → spine
    (3, 4),   # spine → hips
    (1, 5),   # neck → shoulder_l
    (5, 6),   # shoulder_l → upper_arm_l
    (6, 7),   # upper_arm_l → forearm_l
    (7, 8),   # forearm_l → hand_l
    (1, 9),   # neck → shoulder_r
    (9, 10),  # shoulder_r → upper_arm_r
    (10, 11), # upper_arm_r → forearm_r
    (11, 12), # forearm_r → hand_r
    (4, 13),  # hips → upper_leg_l
    (13, 14), # upper_leg_l → lower_leg_l
    (14, 15), # lower_leg_l → foot_l
    (4, 16),  # hips → upper_leg_r
    (16, 17), # upper_leg_r → lower_leg_r
    (17, 18), # lower_leg_r → foot_r
]


def compare_front_reprojection(sample_dir: Path) -> dict:
    """Compare reprojected front joints against bone_2d.json ground truth.

    Returns dict with per-joint and aggregate error stats.
    """
    bone_2d = _load_bone_2d(sample_dir / "bone_2d.json")
    bone_3d = _load_bone_3d(sample_dir / "bone_3d.json")
    extrinsic, intrinsic = _load_camera(
        sample_dir / "extrinsic.npy",
        sample_dir / "intrinsics.npy",
    )

    # Reproject at 0° (front) — should match bone_2d.json
    projected = _project_joints(
        bone_3d, extrinsic, intrinsic,
        azimuth_deg=0,
        output_resolution=ORIGINAL_RESOLUTION,  # Compare at original 1024px
        original_resolution=ORIGINAL_RESOLUTION,
    )

    errors = {}
    for joint_name, gt_xy in bone_2d.items():
        if joint_name not in projected:
            errors[joint_name] = {"error_px": float("inf"), "gt": gt_xy, "proj": None}
            continue
        proj_xy = projected[joint_name]
        dx = proj_xy[0] - gt_xy[0]
        dy = proj_xy[1] - gt_xy[1]
        err = (dx**2 + dy**2) ** 0.5
        errors[joint_name] = {
            "error_px": round(err, 2),
            "gt": gt_xy,
            "proj": [round(proj_xy[0], 1), round(proj_xy[1], 1)],
        }

    all_errors = [v["error_px"] for v in errors.values() if v["error_px"] != float("inf")]
    return {
        "sample_id": sample_dir.name,
        "per_joint": errors,
        "mean_error_px": round(np.mean(all_errors), 2) if all_errors else float("inf"),
        "max_error_px": round(max(all_errors), 2) if all_errors else float("inf"),
        "num_joints": len(all_errors),
    }


def draw_skeleton_overlay(
    img: Image.Image,
    joints: list[dict],
    radius: int = 5,
) -> Image.Image:
    """Draw joints and skeleton connections on an image."""
    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Draw connections first (behind joints)
    for i, j in SKELETON_CONNECTIONS:
        if i >= len(joints) or j >= len(joints):
            continue
        ji, jj = joints[i], joints[j]
        if not ji.get("visible", True) or not jj.get("visible", True):
            continue
        x1, y1 = ji["x"], ji["y"]
        x2, y2 = jj["x"], jj["y"]
        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 180), width=2)

    # Draw joints
    for idx, joint in enumerate(joints):
        if not joint.get("visible", True):
            continue
        x, y = joint["x"], joint["y"]
        color = JOINT_COLORS[idx % len(JOINT_COLORS)]
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(*color, 255),
            outline=(255, 255, 255, 255),
        )

    return overlay


def generate_multi_angle_visualization(
    sample_dir: Path,
    output_path: Path,
) -> None:
    """Generate a visualization showing reprojected joints at all 5 angles.

    Creates a horizontal strip: front image with joints, then 4 blank canvases
    with joint skeletons showing the reprojected positions at each angle.
    """
    bone_3d = _load_bone_3d(sample_dir / "bone_3d.json")
    extrinsic, intrinsic = _load_camera(
        sample_dir / "extrinsic.npy",
        sample_dir / "intrinsics.npy",
    )

    res = STRATA_RESOLUTION
    panels = []

    for angle_label, azimuth_deg in ANGLE_CONFIGS:
        projected = _project_joints(
            bone_3d, extrinsic, intrinsic,
            azimuth_deg=azimuth_deg,
            output_resolution=res,
        )
        strata_joints = _build_strata_joints(projected, res)

        # Use front image for front view, gray background for others
        if angle_label == "front":
            try:
                img = Image.open(sample_dir / "front.png").convert("RGBA")
                img = img.resize((res, res), Image.LANCZOS)
            except OSError:
                img = Image.new("RGBA", (res, res), (64, 64, 64, 255))
        else:
            img = Image.new("RGBA", (res, res), (64, 64, 64, 255))

        overlay = draw_skeleton_overlay(img, strata_joints, radius=6)
        panels.append((angle_label, overlay))

    # Compose horizontal strip with labels
    margin = 4
    label_h = 20
    total_w = len(panels) * res + (len(panels) - 1) * margin
    total_h = res + label_h
    canvas = Image.new("RGBA", (total_w, total_h), (32, 32, 32, 255))

    draw = ImageDraw.Draw(canvas)
    for i, (label, panel) in enumerate(panels):
        x = i * (res + margin)
        canvas.paste(panel, (x, label_h))
        draw.text((x + res // 2 - 30, 2), label, fill=(255, 255, 255, 255))

    canvas.save(output_path)
    print(f"  Saved visualization: {output_path}")


def generate_front_comparison(
    sample_dir: Path,
    output_path: Path,
) -> None:
    """Generate side-by-side comparison: bone_2d (green) vs reprojected (red)."""
    bone_2d = _load_bone_2d(sample_dir / "bone_2d.json")
    bone_3d = _load_bone_3d(sample_dir / "bone_3d.json")
    extrinsic, intrinsic = _load_camera(
        sample_dir / "extrinsic.npy",
        sample_dir / "intrinsics.npy",
    )

    projected = _project_joints(
        bone_3d, extrinsic, intrinsic,
        azimuth_deg=0,
        output_resolution=ORIGINAL_RESOLUTION,
    )

    try:
        img = Image.open(sample_dir / "front.png").convert("RGBA")
    except OSError:
        img = Image.new("RGBA", (ORIGINAL_RESOLUTION, ORIGINAL_RESOLUTION), (64, 64, 64, 255))

    draw = ImageDraw.Draw(img)
    radius = 8

    for joint_name, gt_xy in bone_2d.items():
        # Ground truth in green
        x, y = gt_xy
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 255, 0, 200),
            outline=(255, 255, 255, 255),
        )

        # Reprojected in red
        if joint_name in projected:
            px, py = projected[joint_name]
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=(255, 0, 0, 200),
                outline=(255, 255, 255, 255),
            )
            # Line connecting gt to reprojected
            draw.line([(x, y), (px, py)], fill=(255, 255, 0, 180), width=2)

    # Resize to 512 for display
    img = img.resize((STRATA_RESOLUTION, STRATA_RESOLUTION), Image.LANCZOS)
    img.save(output_path)
    print(f"  Saved front comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test HumanRig joint reprojection")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/joint_reprojection_test",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(HUMANRIG_ROOT),
        help="HumanRig dataset root",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Discover sample directories
    sample_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name),
    )
    print(f"Found {len(sample_dirs)} sample directories")

    if not sample_dirs:
        print("No samples found!")
        sys.exit(1)

    # Pick evenly spaced samples
    n = min(args.samples, len(sample_dirs))
    step = max(1, len(sample_dirs) // n)
    selected = [sample_dirs[i * step] for i in range(n)]

    print(f"\nTesting {n} samples for front-view reprojection accuracy...")
    print("=" * 70)

    all_results = []
    for sample_dir in selected:
        result = compare_front_reprojection(sample_dir)
        all_results.append(result)

        print(f"\nSample {result['sample_id']}:")
        print(f"  Mean error: {result['mean_error_px']:.2f} px (at 1024px)")
        print(f"  Max error:  {result['max_error_px']:.2f} px")

        # Show worst joints
        worst = sorted(
            result["per_joint"].items(),
            key=lambda x: x[1]["error_px"],
            reverse=True,
        )[:3]
        for jname, jdata in worst:
            print(f"  Worst: {jname} = {jdata['error_px']:.2f} px (gt={jdata['gt']}, proj={jdata['proj']})")

        # Generate visualizations
        generate_front_comparison(
            sample_dir,
            output_dir / f"sample_{result['sample_id']}_front_comparison.png",
        )
        generate_multi_angle_visualization(
            sample_dir,
            output_dir / f"sample_{result['sample_id']}_multi_angle.png",
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mean_errors = [r["mean_error_px"] for r in all_results]
    max_errors = [r["max_error_px"] for r in all_results]
    print(f"Samples tested: {len(all_results)}")
    print(f"Mean error across samples: {np.mean(mean_errors):.2f} px (at 1024px)")
    print(f"Mean error at 512px:       {np.mean(mean_errors) / 2:.2f} px")
    print(f"Max error across samples:  {max(max_errors):.2f} px (at 1024px)")
    print(f"Max error at 512px:        {max(max_errors) / 2:.2f} px")

    threshold = 5.0  # px at 1024
    if np.mean(mean_errors) < threshold:
        print(f"\n✓ PASS: Mean error {np.mean(mean_errors):.2f}px < {threshold}px threshold")
        print("  Reprojection math is accurate enough for training data.")
    else:
        print(f"\n✗ FAIL: Mean error {np.mean(mean_errors):.2f}px >= {threshold}px threshold")
        print("  Reprojection has significant distortions — investigate camera model.")

    print(f"\nVisualizations saved to: {output_dir}/")
    print("  *_front_comparison.png — Green=ground_truth, Red=reprojected, Yellow=error line")
    print("  *_multi_angle.png — Skeleton at all 5 camera angles")


if __name__ == "__main__":
    main()
