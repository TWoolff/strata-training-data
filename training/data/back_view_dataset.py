"""PyTorch Dataset for back view generation training.

Loads paired front + three-quarter + back view triplets produced by
``prepare_back_view_pairs.py``.  Concatenates front and three-quarter RGBA
along the channel dimension to produce an 8-channel input, with the back
view as the target.

Expected layout::

    dataset_dir/
        pair_00000/
            front.png           (512x512 RGBA)
            three_quarter.png   (512x512 RGBA)
            back.png            (512x512 RGBA)
        pair_00001/
            ...

Pure Python + PIL/NumPy/Torch (no Blender dependency).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTION: int = 512
VIEW_NAMES: tuple[str, ...] = ("front", "three_quarter", "back")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BackViewDatasetConfig:
    """Configuration for BackViewDataset."""

    dataset_dirs: list[Path] = field(default_factory=list)
    resolution: int = DEFAULT_RESOLUTION
    split: str = "train"
    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    horizontal_flip: bool = True
    color_jitter: dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.02,
        }
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BackViewDataset(torch.utils.data.Dataset):
    """Dataset for back view generation training.

    Each sample returns:
        - ``image``: ``[8, H, W]`` float32 (front RGBA + 3/4 RGBA concatenated)
        - ``target``: ``[4, H, W]`` float32 (back view RGBA)

    Args:
        config: Dataset configuration.
    """

    def __init__(self, config: BackViewDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.pairs: list[Path] = []
        self._discover_pairs()

    def _discover_pairs(self) -> None:
        """Find all valid pair directories and filter by split."""
        all_pairs: list[Path] = []

        for dataset_dir in self.config.dataset_dirs:
            if not dataset_dir.is_dir():
                logger.warning("Dataset directory not found: %s", dataset_dir)
                continue

            for child in sorted(dataset_dir.iterdir()):
                if not child.is_dir() or not child.name.startswith("pair_"):
                    continue
                # Verify all 3 views exist
                if all((child / f"{v}.png").exists() for v in VIEW_NAMES):
                    all_pairs.append(child)

        if not all_pairs:
            logger.warning("No valid pairs found in %s", self.config.dataset_dirs)
            return

        # Split by pair index (character-level split would require char ID extraction
        # from pair directory, but pairs are already deduplicated by char+pose)
        splits = _split_pairs(
            all_pairs,
            seed=self.config.split_seed,
            ratios=self.config.split_ratios,
        )

        self.pairs = splits.get(self.config.split, [])
        logger.info(
            "BackViewDataset split=%s: %d pairs (of %d total)",
            self.config.split,
            len(self.pairs),
            len(all_pairs),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair_dir = self.pairs[idx]

        # Load all 3 views as RGBA numpy arrays [H, W, 4] in [0, 1]
        views = {}
        for name in VIEW_NAMES:
            img = Image.open(pair_dir / f"{name}.png").convert("RGBA")
            if img.size != (self.config.resolution, self.config.resolution):
                img = img.resize((self.config.resolution, self.config.resolution), Image.LANCZOS)
            views[name] = img

        # Augmentation (synchronized across all views)
        if self.config.split == "train":
            views = self._augment(views)

        # Convert to tensors [C, H, W] in [0, 1]
        tensors = {}
        for name, img in views.items():
            arr = np.array(img, dtype=np.float32) / 255.0
            tensors[name] = torch.from_numpy(arr).permute(2, 0, 1)  # [4, H, W]

        # Concatenate front + three_quarter as input
        input_tensor = torch.cat([tensors["front"], tensors["three_quarter"]], dim=0)  # [8, H, W]

        return {
            "image": input_tensor,
            "target": tensors["back"],
        }

    def _augment(self, views: dict[str, Image.Image]) -> dict[str, Image.Image]:
        """Apply synchronized augmentation to all views."""
        cfg = self.config

        # Synchronized horizontal flip
        if cfg.horizontal_flip and random.random() > 0.5:
            views = {name: img.transpose(Image.FLIP_LEFT_RIGHT) for name, img in views.items()}

        # Synchronized color jitter (same transform applied to all views)
        jitter = cfg.color_jitter
        if jitter:
            brightness = 1.0 + random.uniform(
                -jitter.get("brightness", 0), jitter.get("brightness", 0)
            )
            contrast = 1.0 + random.uniform(-jitter.get("contrast", 0), jitter.get("contrast", 0))
            saturation = 1.0 + random.uniform(
                -jitter.get("saturation", 0), jitter.get("saturation", 0)
            )

            result = {}
            for name, img in views.items():
                # Split alpha to preserve it
                rgb = img.convert("RGB")
                alpha = img.split()[3]

                rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
                rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
                rgb = ImageEnhance.Color(rgb).enhance(saturation)

                result[name] = Image.merge("RGBA", (*rgb.split(), alpha))
            views = result

        return views


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _split_pairs(
    pairs: list[Path],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[Path]]:
    """Split pair directories into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
