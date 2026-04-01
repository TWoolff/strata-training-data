"""Dataset for fine-tuning Apple SHARP (single image → 3D Gaussians).

Loads turnaround sheet characters with 5 known views (front, 3/4, side,
back 3/4, back) and provides (input_view, target_views, camera_poses) tuples
for training with a differentiable Gaussian splat rendering loss.

Each character has views at known angles around the Y axis:
    front=0°, three_quarter=45°, side=90°, back_three_quarter=135°, back=180°

During training:
    1. Pick a random source view as input
    2. Model predicts Gaussians from that single view
    3. Render Gaussians from all other camera angles
    4. Compare rendered views to GT views (L1 + perceptual loss)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Turnaround sheet view angles (degrees around Y axis, OpenCV convention)
VIEW_ANGLES: dict[str, float] = {
    "front": 0.0,
    "threequarter": 45.0,
    "side": 90.0,
    "back_threequarter": 135.0,
    "back": 180.0,
}

# File name variants for each view
VIEW_FILE_NAMES: dict[str, list[str]] = {
    "front": ["front.png"],
    "threequarter": ["threequarter.png", "three_quarter.png"],
    "side": ["side.png"],
    "back_threequarter": ["back_threequarter.png", "back_three_quarter.png"],
    "back": ["back.png"],
}

# demo_pairs format: pairs share views from the same character
# We group pairs by their source images to reconstruct per-character view sets
DEMO_PAIR_VIEWS = ["front", "threequarter", "back"]


@dataclass
class SharpDatasetConfig:
    """Configuration for SharpDataset."""
    dataset_dirs: list[Path] = field(default_factory=list)
    resolution: int = 512
    internal_resolution: int = 1536  # SHARP's internal processing resolution
    focal_length_mm: float = 30.0  # Default focal length
    sensor_width_mm: float = 36.0  # Full-frame equivalent
    min_views: int = 3  # Minimum views per character to include
    split_seed: int = 42
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.85, "val": 0.10, "test": 0.05}
    )


class CameraPose(NamedTuple):
    """Camera pose for a view."""
    extrinsics: torch.Tensor  # [4, 4] world-to-camera
    intrinsics: torch.Tensor  # [4, 4] camera intrinsics
    angle_deg: float  # Y rotation in degrees


class SharpExample(NamedTuple):
    """A single training example."""
    input_image: torch.Tensor  # [3, H, W] RGB normalized [0, 1]
    input_pose: CameraPose
    target_images: list[torch.Tensor]  # list of [3, H, W]
    target_poses: list[CameraPose]
    disparity_factor: torch.Tensor  # scalar
    character_id: str


def _make_camera_pose(
    angle_deg: float,
    focal_length_px: float,
    image_size: int,
    radius: float = 3.0,
) -> CameraPose:
    """Create camera extrinsics/intrinsics for a turnaround view.

    Camera orbits around Y axis at given angle, looking at origin.
    Uses OpenCV convention (x right, y down, z forward).
    """
    angle_rad = math.radians(angle_deg)

    # Camera position on circle around Y axis
    cam_x = radius * math.sin(angle_rad)
    cam_z = radius * math.cos(angle_rad)
    cam_pos = np.array([cam_x, 0.0, cam_z])

    # Look at origin
    forward = -cam_pos / np.linalg.norm(cam_pos)  # camera looks toward origin
    up = np.array([0.0, -1.0, 0.0])  # y down in OpenCV
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    # Rotation matrix (world to camera)
    R = np.stack([right, up, forward], axis=0)  # [3, 3]

    # Translation
    t = -R @ cam_pos  # [3]

    # Extrinsics [4, 4]
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    # Intrinsics [4, 4]
    cx = image_size / 2.0
    cy = image_size / 2.0
    intrinsics = np.array([
        [focal_length_px, 0, cx, 0],
        [0, focal_length_px, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    return CameraPose(
        extrinsics=torch.from_numpy(extrinsics),
        intrinsics=torch.from_numpy(intrinsics),
        angle_deg=angle_deg,
    )


def _find_view_file(char_dir: Path, view_name: str) -> Path | None:
    """Find the image file for a given view name."""
    for filename in VIEW_FILE_NAMES.get(view_name, []):
        path = char_dir / filename
        if path.exists():
            return path
    return None


def _load_image_rgb(path: Path, resolution: int) -> torch.Tensor:
    """Load image as RGB tensor [3, H, W] normalized to [0, 1].

    Handles RGBA by compositing onto white background (matching SHARP's
    expectation of photos with backgrounds).
    """
    img = Image.open(path).convert("RGBA")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 4]

    # Composite onto white background
    alpha = arr[:, :, 3:4]
    rgb = arr[:, :, :3] * alpha + (1.0 - alpha)  # white bg

    return torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]


class SharpDataset(torch.utils.data.Dataset):
    """Dataset of turnaround sheet characters for SHARP fine-tuning.

    Each item returns a source view and all available target views with
    their camera poses.
    """

    def __init__(
        self,
        config: SharpDatasetConfig,
        split: str = "train",
    ):
        self.config = config
        self.split = split
        self.resolution = config.resolution

        # Compute focal length in pixels
        self.focal_length_px = (
            config.focal_length_mm / config.sensor_width_mm * config.resolution
        )

        # Discover characters with multiple views
        self.characters: list[dict] = []
        self._discover_characters()

        # Split
        self._apply_split(split, config.split_seed, config.split_ratios)

        logger.info(
            "SharpDataset[%s]: %d characters, focal_length_px=%.1f",
            split, len(self.characters), self.focal_length_px,
        )

    def _discover_characters(self):
        """Find all characters with enough views across dataset directories."""
        for ds_dir in self.config.dataset_dirs:
            ds_dir = Path(ds_dir)
            if not ds_dir.exists():
                logger.warning("Dataset dir not found: %s", ds_dir)
                continue

            for char_dir in sorted(ds_dir.iterdir()):
                if not char_dir.is_dir() or char_dir.name.startswith("."):
                    continue

                # Find available views
                views = {}
                for view_name in VIEW_ANGLES:
                    path = _find_view_file(char_dir, view_name)
                    if path is not None:
                        views[view_name] = path

                if len(views) >= self.config.min_views:
                    self.characters.append({
                        "char_id": f"{ds_dir.name}/{char_dir.name}",
                        "views": views,
                    })

    def _apply_split(self, split: str, seed: int, ratios: dict[str, float]):
        """Split characters into train/val/test."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.characters))

        n = len(self.characters)
        n_train = int(n * ratios.get("train", 0.85))
        n_val = int(n * ratios.get("val", 0.10))

        if split == "train":
            indices = indices[:n_train]
        elif split == "val":
            indices = indices[n_train:n_train + n_val]
        else:
            indices = indices[n_train + n_val:]

        self.characters = [self.characters[i] for i in indices]

    def __len__(self) -> int:
        return len(self.characters)

    def __getitem__(self, idx: int) -> dict:
        """Return a training example.

        For training: randomly pick one view as input, rest as targets.
        For val/test: use front view as input, rest as targets.

        Returns dict with:
            input_image: [3, H, W] RGB
            input_angle: float (degrees)
            target_images: [N, 3, H, W] RGB
            target_angles: [N] float (degrees)
            disparity_factor: scalar
            extrinsics: [N+1, 4, 4] (input + targets)
            intrinsics: [4, 4]
        """
        char = self.characters[idx]
        views = char["views"]
        view_names = list(views.keys())

        # Pick input view
        if self.split == "train":
            input_idx = np.random.randint(len(view_names))
        else:
            # Prefer front view for evaluation
            input_idx = view_names.index("front") if "front" in view_names else 0

        input_name = view_names[input_idx]
        target_names = [n for n in view_names if n != input_name]

        # Load images
        input_image = _load_image_rgb(views[input_name], self.resolution)
        target_images = torch.stack([
            _load_image_rgb(views[n], self.resolution)
            for n in target_names
        ])

        # Camera poses
        input_angle = VIEW_ANGLES[input_name]
        target_angles = torch.tensor([VIEW_ANGLES[n] for n in target_names])

        # All extrinsics (input first, then targets)
        all_angles = [input_angle] + [VIEW_ANGLES[n] for n in target_names]
        all_extrinsics = torch.stack([
            _make_camera_pose(a, self.focal_length_px, self.resolution).extrinsics
            for a in all_angles
        ])

        intrinsics = _make_camera_pose(
            0.0, self.focal_length_px, self.resolution
        ).intrinsics

        disparity_factor = torch.tensor(
            self.focal_length_px / self.resolution
        ).float()

        return {
            "input_image": input_image,
            "input_angle": input_angle,
            "target_images": target_images,
            "target_angles": target_angles,
            "disparity_factor": disparity_factor,
            "extrinsics": all_extrinsics,
            "intrinsics": intrinsics,
            "char_id": char["char_id"],
        }
