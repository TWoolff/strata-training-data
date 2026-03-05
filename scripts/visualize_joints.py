"""Visualize joint predictions overlaid on character images.

Generates annotated images showing joint positions with labels and
confidence values. Useful for verifying RTMPose enrichment quality.

Usage::

    python3 scripts/visualize_joints.py \
        --input_dir /Volumes/TAMWoolff/data/output/curated_diverse \
        --output_dir /Volumes/TAMWoolff/data/output/curated_diverse_viz

    # Only first 10:
    python3 scripts/visualize_joints.py \
        --input_dir /Volumes/TAMWoolff/data/output/curated_diverse \
        --output_dir /Volumes/TAMWoolff/data/output/curated_diverse_viz \
        --max_images 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Skeleton connections for drawing limb lines.
SKELETON = [
    ("head", "neck"),
    ("neck", "chest"),
    ("chest", "spine"),
    ("spine", "hips"),
    # Left arm
    ("chest", "shoulder_l"),
    ("shoulder_l", "upper_arm_l"),
    ("upper_arm_l", "forearm_l"),
    ("forearm_l", "hand_l"),
    # Right arm
    ("chest", "shoulder_r"),
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

# Colors per body region (BGR).
JOINT_COLORS = {
    "head": (0, 255, 255),          # yellow
    "neck": (0, 200, 200),          # dark yellow
    "chest": (0, 165, 255),         # orange
    "spine": (0, 128, 255),         # dark orange
    "hips": (0, 0, 255),            # red
    "shoulder_l": (255, 200, 0),    # cyan-ish
    "upper_arm_l": (255, 150, 0),
    "forearm_l": (255, 100, 0),
    "hand_l": (255, 50, 0),
    "shoulder_r": (200, 0, 255),    # magenta-ish
    "upper_arm_r": (150, 0, 255),
    "forearm_r": (100, 0, 255),
    "hand_r": (50, 0, 255),
    "upper_leg_l": (0, 255, 0),     # green
    "lower_leg_l": (0, 200, 0),
    "foot_l": (0, 150, 0),
    "upper_leg_r": (255, 0, 150),   # pink
    "lower_leg_r": (255, 0, 100),
    "foot_r": (255, 0, 50),
}


def draw_joints(image: np.ndarray, joints: dict) -> np.ndarray:
    """Draw joints, labels, and skeleton on an image.

    Args:
        image: BGR image (will be copied).
        joints: Dict of region_name -> {position, confidence, visible}.

    Returns:
        Annotated BGR image.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Build lookup of positions.
    positions: dict[str, tuple[int, int]] = {}
    for name, data in joints.items():
        pos = data.get("position", [0, 0])
        x, y = int(pos[0]), int(pos[1])
        positions[name] = (x, y)

    # Draw skeleton lines first (behind dots).
    for name_a, name_b in SKELETON:
        if name_a in joints and name_b in joints:
            a_vis = joints[name_a].get("visible", False)
            b_vis = joints[name_b].get("visible", False)
            if a_vis and b_vis:
                pt_a = positions[name_a]
                pt_b = positions[name_b]
                cv2.line(vis, pt_a, pt_b, (200, 200, 200), 2, cv2.LINE_AA)

    # Draw joints with labels.
    for name, data in joints.items():
        pos = data.get("position", [0, 0])
        conf = data.get("confidence", 0.0)
        visible = data.get("visible", False)
        x, y = int(pos[0]), int(pos[1])

        if not visible:
            continue

        color = JOINT_COLORS.get(name, (255, 255, 255))

        # Draw filled circle.
        cv2.circle(vis, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)

        # Label with name and confidence.
        label = f"{name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label to avoid going off-screen.
        lx = min(x + 8, w - tw - 2)
        ly = max(y - 5, th + 2)

        # Background rectangle for readability.
        cv2.rectangle(vis, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 2), (0, 0, 0), -1)
        cv2.putText(vis, label, (lx, ly), font, font_scale, color, thickness, cv2.LINE_AA)

    return vis


def process_example(example_dir: Path, output_dir: Path) -> bool:
    """Generate joint visualization for one example."""
    image_path = example_dir / "image.png"
    joints_path = example_dir / "joints.json"

    if not image_path.is_file() or not joints_path.is_file():
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return False

    # If RGBA, composite on white background for visibility.
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 240.0)  # light gray background
        composited = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    else:
        composited = image[:, :, :3]

    joints_data = json.loads(joints_path.read_text(encoding="utf-8"))
    joints = joints_data.get("joints", joints_data)
    annotated = draw_joints(composited, joints)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{example_dir.name}_joints.png"
    cv2.imwrite(str(out_path), annotated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize joint predictions on images.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=0, help="Max images (0=all).")
    args = parser.parse_args()

    examples = sorted(
        p.parent for p in args.input_dir.rglob("joints.json")
    )

    if args.max_images > 0:
        examples = examples[: args.max_images]

    if not examples:
        print("No examples with joints.json found.")
        sys.exit(1)

    print(f"Generating {len(examples)} joint visualizations...")

    done = 0
    for i, ex in enumerate(examples):
        if process_example(ex, args.output_dir):
            done += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(examples):
            logger.info("Progress: %d/%d", i + 1, len(examples))

    print(f"Done: {done} visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
