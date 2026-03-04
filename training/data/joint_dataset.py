"""PyTorch Dataset for joint refinement training.

Loads training examples from pipeline output, maps joint names to the 20-slot
BONE_ORDER, generates synthetic geometric estimates with calibrated noise, and
computes ground-truth offsets in the dx-first ``[2, 20]`` layout matching the
Rust runtime.

Supports the same dual directory layouts as :class:`SegmentationDataset`:

1. **Flat layout**::

       dataset/images/{char_id}_pose_{nn}_{style}.png
       dataset/joints/{char_id}_pose_{nn}.json

2. **Per-example layout**::

       dataset/{example_id}/image.png
       dataset/{example_id}/joints.json

Pure Python + PIL/NumPy/Torch (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from training.data.split_loader import character_id_from_example, load_or_generate_splits
from training.data.transforms import (
    BONE_TO_INDEX,
    FLIP_JOINT_SWAP,
    PIPELINE_TO_BONE,
    normalize_imagenet,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS: int = 20
DEFAULT_RESOLUTION: int = 512
GEO_NOISE_STD: float = 0.03  # ~2-5% of image, calibrated to centroid-of-boundary error

# Regex to strip style suffix from flat-layout image filenames.
_STYLE_SUFFIXES = re.compile(r"_(flat|cel|pixel|painterly|sketch|unlit)$")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class JointDatasetConfig:
    """Configuration for JointDataset."""

    resolution: int = DEFAULT_RESOLUTION
    augment: bool = True
    horizontal_flip: bool = True
    color_jitter: dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
        }
    )
    random_rotation: float = 0.0  # No rotation for joints (breaks position labels)
    random_scale: tuple[float, float] = (1.0, 1.0)
    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    geo_noise_std: float = GEO_NOISE_STD

    @classmethod
    def from_dict(cls, d: dict) -> JointDatasetConfig:
        """Build config from a flat or nested dict (e.g. parsed YAML)."""
        aug = d.get("augmentation", {})
        data = d.get("data", {})
        ratios = data.get("split_ratios", {})
        return cls(
            resolution=data.get("resolution", DEFAULT_RESOLUTION),
            horizontal_flip=aug.get("horizontal_flip", True),
            color_jitter=aug.get(
                "color_jitter",
                {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
            ),
            split_seed=data.get("split_seed", 42),
            split_ratios=(
                ratios.get("train", 0.8),
                ratios.get("val", 0.1),
                ratios.get("test", 0.1),
            ),
            geo_noise_std=data.get("geo_noise_std", GEO_NOISE_STD),
        )


# ---------------------------------------------------------------------------
# Example descriptor
# ---------------------------------------------------------------------------


@dataclass
class _JointExample:
    """Paths for a single joint training example."""

    image_path: Path
    joints_path: Path
    example_id: str


# ---------------------------------------------------------------------------
# Layout detection & discovery
# ---------------------------------------------------------------------------


def _discover_flat(dataset_dir: Path) -> list[_JointExample]:
    """Discover examples from flat layout (images/ + joints/ subdirs)."""
    images_dir = dataset_dir / "images"
    joints_dir = dataset_dir / "joints"

    if not images_dir.is_dir() or not joints_dir.is_dir():
        return []

    examples: list[_JointExample] = []
    for img_path in sorted(images_dir.glob("*.png")):
        stem = img_path.stem
        # Strip style suffix to get the joints stem
        joints_stem = _STYLE_SUFFIXES.sub("", stem)
        joints_path = joints_dir / f"{joints_stem}.json"

        if not joints_path.exists():
            continue

        examples.append(
            _JointExample(image_path=img_path, joints_path=joints_path, example_id=stem)
        )

    return examples


def _discover_per_example(dataset_dir: Path) -> list[_JointExample]:
    """Discover examples from per-example layout ({id}/image.png)."""
    examples: list[_JointExample] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue

        image_path = child / "image.png"
        joints_path = child / "joints.json"

        if not image_path.exists() or not joints_path.exists():
            continue

        examples.append(
            _JointExample(image_path=image_path, joints_path=joints_path, example_id=child.name)
        )

    return examples


def _detect_layout(dataset_dir: Path) -> str:
    """Return ``"flat"`` or ``"per_example"`` based on directory contents."""
    if (dataset_dir / "images").is_dir():
        return "flat"
    for child in dataset_dir.iterdir():
        if child.is_dir() and (child / "image.png").exists():
            return "per_example"
    return "flat"


# ---------------------------------------------------------------------------
# Joint data parsing
# ---------------------------------------------------------------------------


def parse_joints_json(
    joints_path: Path,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a joints.json file into 20-slot position and visibility arrays.

    Supports two formats:

    1. **RTMPose / pipeline dict format**::

           {"joints": {"head": {"position": [x, y], "visible": true}, ...},
            "image_size": [w, h]}

    2. **HumanRig list format**::

           [{"name": "head", "x": 256.5, "y": 128.3, "visible": true}, ...]

    Args:
        joints_path: Path to the joints JSON file.
        resolution: Image resolution for normalizing positions.

    Returns:
        ``(positions, visible)`` where positions is ``[20, 2]`` float32
        normalized to ``[0, 1]`` and visible is ``[20]`` float32 (1.0/0.0).
    """
    data = json.loads(joints_path.read_text(encoding="utf-8"))

    positions = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    visible = np.zeros(NUM_JOINTS, dtype=np.float32)

    if isinstance(data, list):
        # HumanRig list format: [{"name": ..., "x": ..., "y": ..., "visible": ...}]
        img_w, img_h = resolution, resolution
        for joint_info in data:
            name = joint_info.get("name", "")
            bone_name = PIPELINE_TO_BONE.get(name, name)
            if bone_name not in BONE_TO_INDEX:
                continue
            idx = BONE_TO_INDEX[bone_name]
            positions[idx, 0] = joint_info.get("x", 0) / img_w
            positions[idx, 1] = joint_info.get("y", 0) / img_h
            visible[idx] = 1.0 if joint_info.get("visible", True) else 0.0
    else:
        # RTMPose / pipeline dict format
        joints_dict = data.get("joints", {})
        image_size = data.get("image_size", [resolution, resolution])
        img_w, img_h = image_size[0], image_size[1]

        for pipeline_name, joint_info in joints_dict.items():
            bone_name = PIPELINE_TO_BONE.get(pipeline_name, pipeline_name)
            if bone_name not in BONE_TO_INDEX:
                continue
            idx = BONE_TO_INDEX[bone_name]
            pos = joint_info.get("position", [0, 0])
            is_visible = joint_info.get("visible", True)
            positions[idx, 0] = pos[0] / img_w
            positions[idx, 1] = pos[1] / img_h
            visible[idx] = 1.0 if is_visible else 0.0

    # Slot 19 (hair_back) is always absent
    visible[BONE_TO_INDEX["hair_back"]] = 0.0

    return positions, visible


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class JointDataset:
    """PyTorch Dataset for joint refinement training data.

    Returns per example:
    - ``image``: ``[3, 512, 512]`` float32, ImageNet-normalized
    - ``gt_positions``: ``[20, 2]`` float32, normalized [0,1] xy per joint
    - ``gt_visible``: ``[20]`` float32, 1 if visible
    - ``geo_positions``: ``[20, 2]`` float32, noisy geometric estimates
    - ``gt_offsets``: ``[2, 20]`` float32, gt - geo (dx-first layout matching Rust)

    Args:
        dataset_dirs: One or more dataset root directories.
        split: ``"train"``, ``"val"``, or ``"test"``.
        augment: Enable augmentations.
        config: Optional ``JointDatasetConfig`` or raw dict.
    """

    def __init__(
        self,
        dataset_dirs: list[Path],
        split: str = "train",
        augment: bool = True,
        config: JointDatasetConfig | dict | None = None,
    ) -> None:
        import torch

        self._torch = torch

        if isinstance(config, dict):
            self.config = JointDatasetConfig.from_dict(config)
        elif config is None:
            self.config = JointDatasetConfig()
        else:
            self.config = config

        self.config.augment = augment
        self.split = split

        # Load character-level splits
        splits = load_or_generate_splits(
            dataset_dirs,
            seed=self.config.split_seed,
            ratios=self.config.split_ratios,
        )
        allowed_chars: set[str] = set(splits.get(split, []))

        # Discover examples across all directories
        all_examples: list[_JointExample] = []
        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                logger.warning("Dataset directory does not exist: %s", dataset_dir)
                continue
            layout = _detect_layout(dataset_dir)
            if layout == "flat":
                all_examples.extend(_discover_flat(dataset_dir))
            else:
                all_examples.extend(_discover_per_example(dataset_dir))

        # Filter by split
        self.examples = [
            ex for ex in all_examples if character_id_from_example(ex.example_id) in allowed_chars
        ]

        logger.info(
            "JointDataset[%s]: %d examples (%d before split filter)",
            split,
            len(self.examples),
            len(all_examples),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        """Load and return a single training example.

        Returns:
            Dict with keys: ``image``, ``gt_positions``, ``gt_visible``,
            ``geo_positions``, ``gt_offsets``.
        """
        torch = self._torch
        ex = self.examples[index]
        rng = np.random.default_rng()

        # Load image (RGBA → RGB)
        img = Image.open(ex.image_path).convert("RGBA").convert("RGB")

        res = self.config.resolution
        if img.size != (res, res):
            img = img.resize((res, res), Image.BILINEAR)

        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Load joints
        gt_positions, gt_visible = parse_joints_json(ex.joints_path, res)

        # Augmentation: horizontal flip
        if self.split == "train" and self.config.augment:
            if self.config.horizontal_flip and rng.random() < 0.5:
                img_np, gt_positions, gt_visible = _flip_joint_example(
                    img_np, gt_positions, gt_visible
                )

            # Color jitter (image only)
            cj = self.config.color_jitter
            if cj:
                img_np = _color_jitter(img_np, cj, rng)

        # Generate synthetic geometric estimates
        noise = rng.normal(0.0, self.config.geo_noise_std, size=gt_positions.shape).astype(
            np.float32
        )
        # Only add noise to visible joints
        noise *= gt_visible[:, np.newaxis]
        geo_positions = np.clip(gt_positions + noise, 0.0, 1.0)

        # Compute offsets: gt - geo, in [2, 20] layout (dx-first)
        offset_xy = gt_positions - geo_positions  # [20, 2]
        gt_offsets = offset_xy.T.copy()  # [2, 20] — row 0 = dx, row 1 = dy

        # Convert to tensors
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [3, H, W]
        img_tensor = normalize_imagenet(img_tensor)

        return {
            "image": img_tensor,
            "gt_positions": torch.from_numpy(gt_positions),  # [20, 2]
            "gt_visible": torch.from_numpy(gt_visible),  # [20]
            "geo_positions": torch.from_numpy(geo_positions),  # [20, 2]
            "gt_offsets": torch.from_numpy(gt_offsets),  # [2, 20]
        }


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def _flip_joint_example(
    img: np.ndarray,
    positions: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontally flip image and swap L/R joints in 20-slot arrays.

    Args:
        img: ``[H, W, 3]`` float32 image.
        positions: ``[20, 2]`` normalized joint positions.
        visible: ``[20]`` visibility flags.

    Returns:
        Flipped ``(img, positions, visible)``.
    """
    img = np.flip(img, axis=1).copy()

    new_positions = positions.copy()
    new_visible = visible.copy()

    # Mirror x coordinates
    new_positions[:, 0] = 1.0 - new_positions[:, 0]

    # Swap L/R joint slots
    for left_name, right_name in FLIP_JOINT_SWAP.items():
        if left_name not in BONE_TO_INDEX or right_name not in BONE_TO_INDEX:
            continue
        l_idx = BONE_TO_INDEX[left_name]
        r_idx = BONE_TO_INDEX[right_name]
        if l_idx >= r_idx:
            continue  # Only swap once per pair

        new_positions[[l_idx, r_idx]] = new_positions[[r_idx, l_idx]]
        new_visible[[l_idx, r_idx]] = new_visible[[r_idx, l_idx]]

    return img, new_positions, new_visible


def _color_jitter(
    img: np.ndarray,
    cj: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply random color jitter to an image.

    Args:
        img: ``[H, W, 3]`` float32 in [0, 1].
        cj: Color jitter config dict.
        rng: Numpy random generator.
    """
    brightness = cj.get("brightness", 0.0)
    if brightness > 0:
        factor = rng.uniform(max(0, 1 - brightness), 1 + brightness)
        img = img * factor

    contrast = cj.get("contrast", 0.0)
    if contrast > 0:
        factor = rng.uniform(max(0, 1 - contrast), 1 + contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean

    saturation = cj.get("saturation", 0.0)
    if saturation > 0:
        factor = rng.uniform(max(0, 1 - saturation), 1 + saturation)
        gray = img.mean(axis=2, keepdims=True)
        img = (img - gray) * factor + gray

    return np.clip(img, 0.0, 1.0)
