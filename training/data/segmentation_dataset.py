"""PyTorch Dataset for segmentation training with dual layout support.

Loads training examples from two directory layouts:

1. **Flat layout** (from ``pipeline/exporter.py``)::

       dataset/images/{char_id}_pose_{nn}_{style}.png
       dataset/masks/{char_id}_pose_{nn}.png
       dataset/draw_order/{char_id}_pose_{nn}.png

2. **Per-example layout** (from ingest adapters)::

       dataset/{example_id}/image.png
       dataset/{example_id}/segmentation.png
       dataset/{example_id}/draw_order.png

Pipeline produces 20-class masks (IDs 0-19). The Rust runtime expects
22 classes (0=bg, 1-19=body, 20=unused, 21=accessory). This dataset maps
class 21 for examples flagged with ``has_accessories: true`` in metadata.

Pure Python + PIL/NumPy/Torch (no Blender dependency).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from training.data.split_loader import character_id_from_example, load_or_generate_splits
from training.data.transforms import flip_mask, normalize_imagenet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 22  # 20 pipeline regions + unused (20) + accessory (21)
ACCESSORY_CLASS: int = 21
DEFAULT_RESOLUTION: int = 512

# Regex to strip style suffix from flat-layout image filenames.
# "char_pose_00_flat.png" → "char_pose_00"
# "char_pose_00_three_quarter_flat.png" → "char_pose_00_three_quarter"
_STYLE_SUFFIXES = re.compile(r"_(flat|cel|pixel|painterly|sketch|unlit)$")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for SegmentationDataset."""

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
    random_rotation: float = 10.0
    random_scale: tuple[float, float] = (0.9, 1.1)
    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)

    @classmethod
    def from_dict(cls, d: dict) -> DatasetConfig:
        """Build config from a flat or nested dict (e.g. parsed YAML)."""
        aug = d.get("augmentation", {})
        data = d.get("data", {})
        ratios = data.get("split_ratios", {})
        scale = aug.get("random_scale", [0.9, 1.1])
        return cls(
            resolution=data.get("resolution", DEFAULT_RESOLUTION),
            horizontal_flip=aug.get("horizontal_flip", True),
            color_jitter=aug.get(
                "color_jitter",
                {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
            ),
            random_rotation=aug.get("random_rotation", 10.0),
            random_scale=(scale[0], scale[1]),
            split_seed=data.get("split_seed", 42),
            split_ratios=(
                ratios.get("train", 0.8),
                ratios.get("val", 0.1),
                ratios.get("test", 0.1),
            ),
        )


# ---------------------------------------------------------------------------
# Example descriptor
# ---------------------------------------------------------------------------


@dataclass
class _Example:
    """Paths for a single training example."""

    image_path: Path
    mask_path: Path
    draw_order_path: Path | None
    metadata_path: Path | None
    example_id: str


# ---------------------------------------------------------------------------
# Layout detection & discovery
# ---------------------------------------------------------------------------


def _detect_layout(dataset_dir: Path) -> str:
    """Return ``"flat"`` or ``"per_example"`` based on directory contents."""
    if (dataset_dir / "images").is_dir():
        return "flat"
    # Check for per-example subdirectories containing image.png
    for child in dataset_dir.iterdir():
        if child.is_dir() and (child / "image.png").exists():
            return "per_example"
    return "flat"  # fallback (empty directory)


def _discover_flat(dataset_dir: Path) -> list[_Example]:
    """Discover examples from flat layout (images/ + masks/ subdirs)."""
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    draw_order_dir = dataset_dir / "draw_order"

    if not images_dir.is_dir():
        return []

    examples: list[_Example] = []
    for img_path in sorted(images_dir.glob("*.png")):
        stem = img_path.stem
        # Strip style suffix to get the mask stem
        mask_stem = _STYLE_SUFFIXES.sub("", stem)
        mask_path = masks_dir / f"{mask_stem}.png"

        if not mask_path.exists():
            logger.debug("No mask for %s — skipping", img_path.name)
            continue

        draw_order_path = draw_order_dir / f"{mask_stem}.png"
        if not draw_order_path.exists():
            draw_order_path = None

        examples.append(
            _Example(
                image_path=img_path,
                mask_path=mask_path,
                draw_order_path=draw_order_path,
                metadata_path=None,
                example_id=stem,
            )
        )

    return examples


def _discover_per_example(dataset_dir: Path) -> list[_Example]:
    """Discover examples from per-example layout ({id}/image.png)."""
    examples: list[_Example] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue

        image_path = child / "image.png"
        mask_path = child / "segmentation.png"

        if not image_path.exists() or not mask_path.exists():
            continue

        draw_order_path = child / "draw_order.png"
        metadata_path = child / "metadata.json"

        examples.append(
            _Example(
                image_path=image_path,
                mask_path=mask_path,
                draw_order_path=draw_order_path if draw_order_path.exists() else None,
                metadata_path=metadata_path if metadata_path.exists() else None,
                example_id=child.name,
            )
        )

    return examples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SegmentationDataset:
    """PyTorch Dataset for segmentation training data.

    Supports both flat and per-example directory layouts. Filters examples
    by character-level split to prevent data leakage.

    Args:
        dataset_dirs: One or more dataset root directories.
        split: ``"train"``, ``"val"``, or ``"test"``.
        augment: Enable augmentations (overrides config for this instance).
        config: Optional ``DatasetConfig`` or raw dict.
    """

    def __init__(
        self,
        dataset_dirs: list[Path],
        split: str = "train",
        augment: bool = True,
        config: DatasetConfig | dict | None = None,
    ) -> None:
        import torch

        self._torch = torch

        if isinstance(config, dict):
            self.config = DatasetConfig.from_dict(config)
        elif config is None:
            self.config = DatasetConfig()
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
        all_examples: list[_Example] = []
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
            "SegmentationDataset[%s]: %d examples (%d before split filter)",
            split,
            len(self.examples),
            len(all_examples),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        """Load and return a single training example.

        Returns:
            Dict with keys: ``image``, ``segmentation``, ``draw_order``,
            ``has_draw_order``, ``confidence_target``.
        """
        torch = self._torch
        ex = self.examples[index]

        # Load image (RGBA → RGB)
        img = Image.open(ex.image_path).convert("RGBA")
        alpha = np.array(img)[:, :, 3]  # preserve alpha for confidence
        img = img.convert("RGB")

        # Resize if needed
        res = self.config.resolution
        if img.size != (res, res):
            img = img.resize((res, res), Image.BILINEAR)
            alpha = np.array(Image.fromarray(alpha).resize((res, res), Image.NEAREST))

        # Load mask
        mask = Image.open(ex.mask_path).convert("L")
        if mask.size != (res, res):
            mask = mask.resize((res, res), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        # Load draw order (optional)
        has_draw_order = ex.draw_order_path is not None
        if has_draw_order:
            draw_order_img = Image.open(ex.draw_order_path).convert("L")
            if draw_order_img.size != (res, res):
                draw_order_img = draw_order_img.resize((res, res), Image.BILINEAR)
            draw_order_np = np.array(draw_order_img, dtype=np.float32) / 255.0
        else:
            draw_order_np = np.zeros((res, res), dtype=np.float32)

        # Convert image to numpy for augmentation
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Augmentations (train split only, if enabled)
        if self.split == "train" and self.config.augment:
            img_np, mask_np, draw_order_np, alpha = self._augment(
                img_np, mask_np, draw_order_np, alpha
            )

        # Confidence target: 1.0 where image has alpha > 0 or mask > 0
        confidence = np.where((alpha > 0) | (mask_np > 0), 1.0, 0.0).astype(np.float32)

        # Convert to tensors
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [3, H, W]
        img_tensor = normalize_imagenet(img_tensor)

        seg_tensor = torch.from_numpy(mask_np)  # [H, W] int64
        draw_order_tensor = torch.from_numpy(draw_order_np).unsqueeze(0)  # [1, H, W]
        confidence_tensor = torch.from_numpy(confidence).unsqueeze(0)  # [1, H, W]

        return {
            "image": img_tensor,
            "segmentation": seg_tensor,
            "draw_order": draw_order_tensor,
            "has_draw_order": has_draw_order,
            "confidence_target": confidence_tensor,
        }

    # -----------------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------------

    def _augment(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        draw_order: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply training augmentations.

        Args:
            img: ``[H, W, 3]`` float32 image in [0, 1].
            mask: ``[H, W]`` int64 region IDs.
            draw_order: ``[H, W]`` float32 depth in [0, 1].
            alpha: ``[H, W]`` uint8 alpha channel.

        Returns:
            Augmented (img, mask, draw_order, alpha).
        """
        rng = np.random.default_rng()

        # Horizontal flip (50% chance) with L/R region swap
        if self.config.horizontal_flip and rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = flip_mask(mask.astype(np.uint8)).astype(np.int64)
            draw_order = np.flip(draw_order, axis=1).copy()
            alpha = np.flip(alpha, axis=1).copy()

        # Color jitter (image only)
        cj = self.config.color_jitter
        if cj:
            img = self._color_jitter(
                img,
                brightness=cj.get("brightness", 0.0),
                contrast=cj.get("contrast", 0.0),
                saturation=cj.get("saturation", 0.0),
                hue=cj.get("hue", 0.0),
                rng=rng,
            )

        # Random rotation + scale via affine transform
        rotation = self.config.random_rotation
        scale_range = self.config.random_scale
        if rotation > 0 or scale_range != (1.0, 1.0):
            angle = rng.uniform(-rotation, rotation) if rotation > 0 else 0.0
            scale = rng.uniform(scale_range[0], scale_range[1])
            img, mask, draw_order, alpha = self._affine_transform(
                img, mask, draw_order, alpha, angle=angle, scale=scale
            )

        return img, mask, draw_order, alpha

    @staticmethod
    def _color_jitter(
        img: np.ndarray,
        *,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply random color jitter to an image.

        Args:
            img: ``[H, W, 3]`` float32 in [0, 1].
        """
        # Brightness
        if brightness > 0:
            factor = rng.uniform(max(0, 1 - brightness), 1 + brightness)
            img = img * factor

        # Contrast
        if contrast > 0:
            factor = rng.uniform(max(0, 1 - contrast), 1 + contrast)
            mean = img.mean()
            img = (img - mean) * factor + mean

        # Saturation
        if saturation > 0:
            factor = rng.uniform(max(0, 1 - saturation), 1 + saturation)
            gray = img.mean(axis=2, keepdims=True)
            img = (img - gray) * factor + gray

        # Hue shift via cv2 HSV conversion (cv2 is already a dependency)
        if hue > 0:
            import cv2

            shift = rng.uniform(-hue, hue)
            hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] / 360.0 + shift) % 1.0 * 360.0
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _affine_transform(
        img: np.ndarray,
        mask: np.ndarray,
        draw_order: np.ndarray,
        alpha: np.ndarray,
        *,
        angle: float,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply rotation and scale via OpenCV affine transform.

        Args:
            angle: Rotation angle in degrees.
            scale: Scale factor.
        """
        import cv2

        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        mat = cv2.getRotationMatrix2D(center, angle, scale)

        img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(
            mask.astype(np.float32), mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        ).astype(np.int64)
        draw_order = cv2.warpAffine(draw_order, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        alpha = cv2.warpAffine(alpha, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        return img, mask, draw_order, alpha
