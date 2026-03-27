"""PyTorch Dataset for segmentation training with dual layout support.

Loads training examples from two directory layouts:

1. **Flat layout** (from ``pipeline/exporter.py``)::

       dataset/images/{char_id}_pose_{nn}_{style}.png
       dataset/masks/{char_id}_pose_{nn}.png

2. **Per-example layout** (from ingest adapters)::

       dataset/{example_id}/image.png
       dataset/{example_id}/segmentation.png
       dataset/{example_id}/depth.png       (optional, Marigold LCM)
       dataset/{example_id}/normals.png     (optional, Marigold LCM)

Pipeline produces 20-class masks (IDs 0-19). The Rust runtime expects
22 classes (0=bg, 1-19=body, 20=unused, 21=accessory). This dataset maps
class 21 for examples flagged with ``has_accessories: true`` in metadata.

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
from training.data.transforms import flip_mask, normalize_imagenet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 22  # 20 pipeline regions + unused (20) + accessory (21)
ACCESSORY_CLASS: int = 21

# Class 20 ("unused") is not used by Strata's rigging pipeline — remap to
# background so the model doesn't waste capacity on it.
LABEL_REMAP: dict[int, int] = {20: 0}
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

    boundary_softening_radius: int = 0  # 0 = disabled, 2-3 = recommended
    dataset_weights: dict[str, float] = field(default_factory=dict)
    train_only_datasets: list[str] = field(default_factory=list)
    frozen_splits_file: str = ""

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
            boundary_softening_radius=d.get("loss", {}).get("boundary_softening_radius", 0),
            dataset_weights=data.get("dataset_weights", {}),
            train_only_datasets=data.get("train_only_datasets", []),
            frozen_splits_file=data.get("frozen_splits_file", ""),
        )


# ---------------------------------------------------------------------------
# Example descriptor
# ---------------------------------------------------------------------------


@dataclass
class _Example:
    """Paths for a single training example."""

    image_path: Path
    mask_path: Path
    depth_path: Path | None
    normals_path: Path | None
    metadata_path: Path | None
    example_id: str
    dataset_weight: float = 1.0


# ---------------------------------------------------------------------------
# Layout detection & discovery
# ---------------------------------------------------------------------------


def _detect_layout(dataset_dir: Path) -> str:
    """Return ``"flat"`` or ``"per_example"`` based on directory contents."""
    if (dataset_dir / "images").is_dir():
        return "flat"
    # Check for per-example subdirectories containing image.png (flat or nested)
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "image.png").exists():
            return "per_example"
        # Nested layout: {id}/{view}/image.png (e.g. UniRig 00000/front/)
        for sub in child.iterdir():
            if sub.is_dir() and (sub / "image.png").exists():
                return "per_example"
        break  # only check the first child
    return "flat"  # fallback (empty directory)


def _discover_flat(dataset_dir: Path) -> list[_Example]:
    """Discover examples from flat layout (images/ + masks/ subdirs)."""
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

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

        examples.append(
            _Example(
                image_path=img_path,
                mask_path=mask_path,
                depth_path=None,
                normals_path=None,
                metadata_path=None,
                example_id=stem,
            )
        )

    return examples


def _discover_per_example(dataset_dir: Path) -> list[_Example]:
    """Discover examples from per-example layout.

    Supports both flat (``{id}/image.png``) and nested
    (``{id}/{view}/image.png``) layouts (e.g. UniRig ``00000/front/``).
    """
    examples: list[_Example] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue

        # Collect candidate dirs: the child itself, plus any subdirs (nested layout)
        candidate_dirs = [child]
        if not (child / "image.png").exists():
            candidate_dirs = [
                sub for sub in sorted(child.iterdir())
                if sub.is_dir() and (sub / "image.png").exists()
            ]

        for cand in candidate_dirs:
            image_path = cand / "image.png"
            mask_path = cand / "segmentation.png"

            if not image_path.exists() or not mask_path.exists():
                continue

            depth_path = cand / "depth.png"
            normals_path = cand / "normals.png"
            metadata_path = cand / "metadata.json"

            # For nested layout, use "parent_view" as example_id (e.g. "00000_front")
            if cand != child:
                example_id = f"{child.name}_{cand.name}"
            else:
                example_id = child.name

            examples.append(
                _Example(
                    image_path=image_path,
                    mask_path=mask_path,
                    depth_path=depth_path if depth_path.exists() else None,
                    normals_path=normals_path if normals_path.exists() else None,
                    metadata_path=metadata_path if metadata_path.exists() else None,
                    example_id=example_id,
                )
            )

    return examples


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def _load_quality_rejects(dataset_dirs: list[Path]) -> set[str]:
    """Load rejected example IDs from quality_filter.json in each dataset dir."""
    rejected: set[str] = set()
    for dataset_dir in dataset_dirs:
        qf_path = dataset_dir / "quality_filter.json"
        if not qf_path.exists():
            continue
        try:
            data = json.loads(qf_path.read_text(encoding="utf-8"))
            reject_dict = data.get("rejected", {})
            rejected.update(reject_dict.keys())
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read quality filter: %s", qf_path)
    return rejected


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

        # Identify train-only directories
        train_only_names = set(self.config.train_only_datasets)
        train_only_dirs = [d for d in dataset_dirs if d.name in train_only_names]

        # Frozen splits file (if configured)
        frozen_file = (
            Path(self.config.frozen_splits_file)
            if self.config.frozen_splits_file
            else None
        )

        # Load character-level splits
        splits = load_or_generate_splits(
            dataset_dirs,
            seed=self.config.split_seed,
            ratios=self.config.split_ratios,
            train_only_dirs=train_only_dirs,
            frozen_splits_file=frozen_file,
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
                dir_examples = _discover_flat(dataset_dir)
            else:
                dir_examples = _discover_per_example(dataset_dir)

            # Assign per-dataset loss weight
            weight = self.config.dataset_weights.get(dataset_dir.name, 1.0)
            for ex in dir_examples:
                ex.dataset_weight = weight
            if weight != 1.0 and dir_examples:
                logger.info(
                    "  %s: %d examples, dataset_weight=%.2f",
                    dataset_dir.name, len(dir_examples), weight,
                )

            all_examples.extend(dir_examples)

        # Filter by split
        self.examples = [
            ex for ex in all_examples if character_id_from_example(ex.example_id) in allowed_chars
        ]
        n_after_split = len(self.examples)

        # Filter by quality_filter.json (reject bad segmentation masks)
        rejected_ids = _load_quality_rejects(dataset_dirs)
        if rejected_ids:
            self.examples = [
                ex for ex in self.examples if ex.example_id not in rejected_ids
            ]
            n_rejected = n_after_split - len(self.examples)
            if n_rejected > 0:
                logger.info(
                    "Quality filter removed %d examples (%d reject IDs loaded)",
                    n_rejected,
                    len(rejected_ids),
                )

        # Count examples with depth/normals for logging
        n_depth = sum(1 for ex in self.examples if ex.depth_path is not None)
        n_normals = sum(1 for ex in self.examples if ex.normals_path is not None)

        logger.info(
            "SegmentationDataset[%s]: %d examples (%d before split filter), "
            "%d with depth, %d with normals",
            split,
            len(self.examples),
            len(all_examples),
            n_depth,
            n_normals,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        """Load and return a single training example.

        Returns:
            Dict with keys: ``image``, ``segmentation``, ``depth``,
            ``has_depth``, ``normals``, ``has_normals``, ``confidence_target``.
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

        # Clamp out-of-range class IDs to ignore_index (-1).
        mask_np[mask_np >= NUM_CLASSES] = -1

        # Remap dead classes (e.g. class 20 "unused" → background).
        for src, dst in LABEL_REMAP.items():
            mask_np[mask_np == src] = dst

        # Load depth (optional, Marigold LCM grayscale uint8)
        has_depth = ex.depth_path is not None
        if has_depth:
            depth_img = Image.open(ex.depth_path).convert("L")
            if depth_img.size != (res, res):
                depth_img = depth_img.resize((res, res), Image.BILINEAR)
            depth_np = np.array(depth_img, dtype=np.float32) / 255.0
        else:
            depth_np = np.zeros((res, res), dtype=np.float32)

        # Load normals (optional, Marigold LCM RGB uint8 encoding [-1,1] as [0,255])
        has_normals = ex.normals_path is not None
        if has_normals:
            normals_img = Image.open(ex.normals_path).convert("RGB")
            if normals_img.size != (res, res):
                normals_img = normals_img.resize((res, res), Image.BILINEAR)
            # Convert from uint8 [0, 255] to float [-1, 1]
            normals_np = np.array(normals_img, dtype=np.float32) / 255.0 * 2.0 - 1.0  # [H, W, 3]
        else:
            normals_np = np.zeros((res, res, 3), dtype=np.float32)

        # Convert image to numpy for augmentation
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Augmentations (train split only, if enabled)
        if self.split == "train" and self.config.augment:
            img_np, mask_np, depth_np, normals_np, alpha = self._augment(
                img_np, mask_np, depth_np, normals_np, alpha
            )

        # Confidence target: 1.0 where image has alpha > 0 or mask > 0
        confidence = np.where((alpha > 0) | (mask_np > 0), 1.0, 0.0).astype(np.float32)

        # Boundary softening (optional): load precomputed or compute on the fly
        soft_seg_np = None
        bsr = getattr(self.config, "boundary_softening_radius", 0)
        if bsr > 0 and self.split == "train":
            # Try loading precomputed soft targets first
            soft_path = ex.mask_path.parent / "soft_segmentation.npy"
            if soft_path.exists():
                soft_seg_np = np.load(soft_path).astype(np.float32)
            else:
                soft_seg_np = self._soften_boundaries(
                    mask_np, radius=bsr, exclude_classes=self.SOFTENING_EXCLUDE_CLASSES,
                )

        # Convert to tensors
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [3, H, W]
        img_tensor = normalize_imagenet(img_tensor)

        seg_tensor = torch.from_numpy(mask_np)  # [H, W] int64
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]
        normals_tensor = torch.from_numpy(normals_np.transpose(2, 0, 1))  # [3, H, W]
        confidence_tensor = torch.from_numpy(confidence).unsqueeze(0)  # [1, H, W]

        result = {
            "image": img_tensor,
            "segmentation": seg_tensor,
            "depth": depth_tensor,
            "has_depth": has_depth,
            "normals": normals_tensor,
            "has_normals": has_normals,
            "confidence_target": confidence_tensor,
            "dataset_weight": ex.dataset_weight,
        }
        if soft_seg_np is not None:
            result["soft_segmentation"] = torch.from_numpy(soft_seg_np)  # [C, H, W]
        return result

    # -----------------------------------------------------------------------
    # Boundary softening
    # -----------------------------------------------------------------------

    # Classes to exclude from boundary softening (small/thin regions that
    # get blurred into neighbors).  Neck regressed 0.62→0.45 in run 20.
    SOFTENING_EXCLUDE_CLASSES: set[int] = {2, 21}  # neck, accessory/hair_back

    @staticmethod
    def _soften_boundaries(
        mask: np.ndarray,
        radius: int = 2,
        sigma: float = 1.0,
        exclude_classes: set[int] | None = None,
    ) -> np.ndarray:
        """Create soft one-hot targets with Gaussian-blurred boundaries.

        Interior pixels retain hard one-hot labels. Pixels within ``radius``
        of a body-part boundary get a soft distribution over neighboring
        classes. Background (0) boundaries are not softened.

        Args:
            mask: ``[H, W]`` int64 region IDs (0=bg, 1-21=body parts, -1=ignore).
            radius: Dilation radius for boundary detection.
            sigma: Gaussian sigma for softening.
            exclude_classes: Class IDs to keep as hard labels even at boundaries.

        Returns:
            ``[num_classes, H, W]`` float32 soft target distribution.
        """
        from scipy.ndimage import gaussian_filter

        if exclude_classes is None:
            exclude_classes = set()

        h, w = mask.shape
        num_classes = NUM_CLASSES

        # Build one-hot encoding (ignore index → all zeros)
        one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
        for c in range(num_classes):
            one_hot[c] = (mask == c).astype(np.float32)

        # Gaussian blur each class channel
        soft = np.zeros_like(one_hot)
        for c in range(num_classes):
            if one_hot[c].any():
                soft[c] = gaussian_filter(one_hot[c], sigma=sigma)

        # Normalize to sum to 1 at each pixel
        total = soft.sum(axis=0, keepdims=True).clip(min=1e-8)
        soft = soft / total

        # Only soften at body-part boundaries (not interior, not background-only)
        # Detect boundary: pixels where a dilation differs from the original
        from scipy.ndimage import maximum_filter, minimum_filter

        fg_mask = (mask > 0) & (mask < num_classes)  # foreground pixels
        dilated = maximum_filter(mask, size=2 * radius + 1)
        eroded = minimum_filter(mask, size=2 * radius + 1)
        boundary = (dilated != eroded) & fg_mask

        # Exclude specified classes from softening (keep hard labels)
        if exclude_classes:
            for c in exclude_classes:
                boundary = boundary & (mask != c)

        # At non-boundary pixels, revert to hard one-hot
        for c in range(num_classes):
            soft[c] = np.where(boundary, soft[c], one_hot[c])

        # Re-normalize boundary pixels
        if boundary.any():
            boundary_total = soft[:, boundary].sum(axis=0, keepdims=True).clip(min=1e-8)
            soft[:, boundary] = soft[:, boundary] / boundary_total

        # Ignore index pixels → all zeros
        ignore_mask = (mask < 0) | (mask >= num_classes)
        soft[:, ignore_mask] = 0.0

        return soft

    # -----------------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------------

    def _augment(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        normals: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply training augmentations.

        Args:
            img: ``[H, W, 3]`` float32 image in [0, 1].
            mask: ``[H, W]`` int64 region IDs.
            depth: ``[H, W]`` float32 depth in [0, 1].
            normals: ``[H, W, 3]`` float32 normals in [-1, 1].
            alpha: ``[H, W]`` uint8 alpha channel.

        Returns:
            Augmented (img, mask, depth, normals, alpha).
        """
        rng = np.random.default_rng()

        # Horizontal flip (50% chance) with L/R region swap
        if self.config.horizontal_flip and rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = flip_mask(mask.astype(np.uint8)).astype(np.int64)
            depth = np.flip(depth, axis=1).copy()
            normals = np.flip(normals, axis=1).copy()
            # Flip the X component of normals when horizontally flipping
            normals[:, :, 0] = -normals[:, :, 0]
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
            img, mask, depth, normals, alpha = self._affine_transform(
                img, mask, depth, normals, alpha, angle=angle, scale=scale
            )

        return img, mask, depth, normals, alpha

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
        depth: np.ndarray,
        normals: np.ndarray,
        alpha: np.ndarray,
        *,
        angle: float,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply rotation and scale via OpenCV affine transform."""
        import cv2

        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        mat = cv2.getRotationMatrix2D(center, angle, scale)

        img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(
            mask.astype(np.float32), mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        ).astype(np.int64)
        depth = cv2.warpAffine(depth, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        normals = cv2.warpAffine(normals, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        alpha = cv2.warpAffine(alpha, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        return img, mask, depth, normals, alpha
