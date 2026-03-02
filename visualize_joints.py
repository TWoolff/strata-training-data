"""Visualize joint annotations overlaid on character images.

Draws colored keypoints and skeleton connections on a copy of the image.

Usage::

    python visualize_joints.py --input_dir ./output/fbanimehq_phase1 --limit 10
    python visualize_joints.py --example ./output/fbanimehq_phase1/fbanimehq_0000_000117
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Strata skeleton connections (pairs of joint names to draw bones between)
SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    ("head", "neck"),
    ("neck", "chest"),
    ("chest", "spine"),
    ("spine", "hips"),
    # Left arm
    ("neck", "shoulder_l"),
    ("shoulder_l", "upper_arm_l"),
    ("upper_arm_l", "forearm_l"),
    ("forearm_l", "hand_l"),
    # Right arm
    ("neck", "shoulder_r"),
    ("shoulder_r", "upper_arm_r"),
    ("upper_arm_r", "forearm_r"),
    ("forearm_r", "hand_r"),
    # Left leg
    ("hips", "upper_leg_l"),
    ("upper_leg_l", "lower_leg_l"),
    ("lower_leg_l", "foot_l"),
    # Right leg
    ("hips", "upper_leg_r"),
    ("upper_leg_r", "lower_leg_r"),
    ("lower_leg_r", "foot_r"),
]

# Colors per joint (BGR for OpenCV)
JOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "head": (0, 255, 255),         # yellow
    "neck": (0, 200, 200),         # dark yellow
    "chest": (0, 165, 255),        # orange
    "spine": (0, 100, 255),        # dark orange
    "hips": (0, 0, 255),           # red
    "shoulder_l": (255, 200, 0),   # cyan-ish
    "upper_arm_l": (255, 150, 0),  # blue-cyan
    "forearm_l": (255, 100, 0),  # blue
    "hand_l": (255, 50, 0),        # dark blue
    "shoulder_r": (200, 0, 255),   # magenta
    "upper_arm_r": (150, 0, 255),  # pink-magenta
    "forearm_r": (100, 0, 255),  # pink
    "hand_r": (50, 0, 255),        # dark pink
    "upper_leg_l": (0, 255, 0),    # green
    "lower_leg_l": (0, 200, 0),    # dark green
    "foot_l": (0, 150, 0),         # darker green
    "upper_leg_r": (50, 255, 50),  # light green
    "lower_leg_r": (50, 200, 50),  # medium green
    "foot_r": (50, 150, 50),       # olive green
}

BONE_COLOR = (200, 200, 200)  # light gray for skeleton lines


def draw_joints(image: np.ndarray, joints_data: dict) -> np.ndarray:
    """Draw joints and skeleton on a copy of the image."""
    canvas = image.copy()
    joints = joints_data.get("joints", {})

    # Draw skeleton connections first (so dots render on top)
    for name_a, name_b in SKELETON_CONNECTIONS:
        ja = joints.get(name_a)
        jb = joints.get(name_b)
        if not ja or not jb:
            continue
        if not ja.get("visible") or not jb.get("visible"):
            continue
        pt_a = tuple(ja["position"])
        pt_b = tuple(jb["position"])
        avg_conf = (ja["confidence"] + jb["confidence"]) / 2
        alpha = max(0.3, min(1.0, avg_conf))
        color = tuple(int(c * alpha) for c in BONE_COLOR)
        cv2.line(canvas, pt_a, pt_b, color, 2, cv2.LINE_AA)

    # Draw joint dots + labels
    for name, joint in joints.items():
        if not joint.get("visible"):
            continue
        pos = tuple(joint["position"])
        conf = joint["confidence"]
        color = JOINT_COLORS.get(name, (255, 255, 255))
        radius = 5 if conf >= 0.5 else 3
        cv2.circle(canvas, pos, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pos, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw label with background for readability
        label = name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tx = pos[0] + radius + 3
        ty = pos[1] + th // 2
        # Keep label within image bounds
        if tx + tw > canvas.shape[1]:
            tx = pos[0] - radius - 3 - tw
        if ty - th < 0:
            ty = pos[1] + th + radius
        # Dark background rectangle
        cv2.rectangle(
            canvas,
            (tx - 1, ty - th - 1),
            (tx + tw + 1, ty + baseline + 1),
            (0, 0, 0),
            -1,
        )
        cv2.putText(canvas, label, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    # Draw bbox if present
    bbox = joints_data.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (128, 128, 128), 1)

    return canvas


def visualize_example(example_dir: Path, output_dir: Path | None = None) -> Path | None:
    """Visualize joints for a single example. Returns path to output image."""
    image_path = example_dir / "image.png"
    joints_path = example_dir / "joints.json"

    if not image_path.exists() or not joints_path.exists():
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    with open(joints_path) as f:
        joints_data = json.load(f)

    canvas = draw_joints(image, joints_data)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{example_dir.name}_joints.png"
    else:
        out_path = example_dir / "joints_overlay.png"

    cv2.imwrite(str(out_path), canvas)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize joint annotations on images.")
    parser.add_argument(
        "--example",
        type=Path,
        help="Single example directory to visualize.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Root directory containing example subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for overlay images (default: alongside originals).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of examples to visualize (0 = all).",
    )
    args = parser.parse_args()

    if not args.example and not args.input_dir:
        parser.error("Provide --example or --input_dir")

    if args.example:
        result = visualize_example(args.example, args.output_dir)
        if result:
            print(f"Saved: {result}")
        else:
            print(f"Failed: {args.example}", file=sys.stderr)
        return

    # Batch mode
    examples = sorted(
        p.parent for p in args.input_dir.rglob("joints.json")
    )
    if args.limit > 0:
        examples = examples[: args.limit]

    print(f"Visualizing {len(examples)} examples...")
    for i, ex in enumerate(examples):
        result = visualize_example(ex, args.output_dir)
        if result and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(examples)}")

    print(f"Done — {len(examples)} overlays saved.")


if __name__ == "__main__":
    main()
