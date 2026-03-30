"""Dataset for unified view synthesis training.

Loads multi-view character data from turnaround sheet pairs and 3D-rendered
back view pairs. For each character, generates training triplets:
(source_A, source_B, target) with angle labels.

Supports two data layouts:

1. **Turnaround pairs** (from Gemini turnaround sheets):
   ```
   pair_demo_001/
   ├── front.png
   ├── three_quarter.png
   └── back.png
   ```

2. **Legacy back view pairs** (from 3D renders):
   ```
   pair_00001/
   ├── front.png
   ├── three_quarter.png
   └── back.png
   ```

Both layouts are identical — the dataset treats them the same.

Pure Python + PyTorch (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.models.view_synthesis_model import VIEW_ANGLES

logger = logging.getLogger(__name__)

# Map filenames to view names
FILENAME_TO_VIEW: dict[str, str] = {
    "front.png": "front",
    "three_quarter.png": "threequarter",
    "back.png": "back",
    "side.png": "side",
    "back_three_quarter.png": "back_threequarter",
    "front_three_quarter.png": "frontthreequarter",
}


@dataclass
class ViewSynthesisConfig:
    """Configuration for ViewSynthesisDataset."""
    dataset_dirs: list[Path] = field(default_factory=list)
    resolution: int = 512
    split: str = "train"
    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.85, 0.10, 0.05)
    horizontal_flip: bool = False
    color_jitter: dict[str, float] = field(default_factory=dict)
    dataset_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> ViewSynthesisConfig:
        data = d.get("data", {})
        aug = d.get("augmentation", {})
        ratios = data.get("split_ratios", {})
        return cls(
            dataset_dirs=[Path(p) for p in data.get("dataset_dirs", [])],
            resolution=data.get("resolution", 512),
            split_seed=data.get("split_seed", 42),
            split_ratios=(
                ratios.get("train", 0.85),
                ratios.get("val", 0.10),
                ratios.get("test", 0.05),
            ),
            horizontal_flip=aug.get("horizontal_flip", False),
            color_jitter=aug.get("color_jitter", {}),
            dataset_weights=data.get("dataset_weights", {}),
        )


@dataclass
class _Character:
    """A character with multiple view images."""
    char_id: str
    views: dict[str, Path]  # view_name → image path
    dataset_weight: float = 1.0


class ViewSynthesisDataset:
    """Dataset that generates (source_A, source_B, target_angle) → target triplets.

    For each character with N views, generates C(N,2) × (N-2) triplets:
    pick any 2 views as sources, any remaining view as target.

    Args:
        config: Dataset configuration.
        split: "train", "val", or "test".
    """

    def __init__(self, config: ViewSynthesisConfig, split: str = "train") -> None:
        self.config = config
        self.split = split
        self.resolution = config.resolution

        # Discover characters
        characters = self._discover_characters(config.dataset_dirs, config.dataset_weights)

        # Split by character
        characters = self._split_characters(characters, split, config.split_seed, config.split_ratios)

        # Generate all valid triplets
        self.triplets = self._generate_triplets(characters)

        logger.info(
            "ViewSynthesisDataset[%s]: %d triplets from %d characters",
            split, len(self.triplets), len(characters),
        )

    def _discover_characters(
        self, dataset_dirs: list[Path], dataset_weights: dict[str, float],
    ) -> list[_Character]:
        """Find all characters with 2+ views across dataset directories."""
        characters = []

        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                logger.warning("Dataset dir not found: %s", dataset_dir)
                continue

            ds_name = dataset_dir.name
            weight = dataset_weights.get(ds_name, 1.0)

            for pair_dir in sorted(dataset_dir.iterdir()):
                if not pair_dir.is_dir() or pair_dir.name.startswith("."):
                    continue

                views = {}
                for filename, view_name in FILENAME_TO_VIEW.items():
                    img_path = pair_dir / filename
                    if img_path.exists():
                        views[view_name] = img_path

                if len(views) >= 2:
                    characters.append(_Character(
                        char_id=f"{ds_name}/{pair_dir.name}",
                        views=views,
                        dataset_weight=weight,
                    ))

        return characters

    def _split_characters(
        self,
        characters: list[_Character],
        split: str,
        seed: int,
        ratios: tuple[float, float, float],
    ) -> list[_Character]:
        """Split characters into train/val/test by character ID."""
        rng = random.Random(seed)
        indices = list(range(len(characters)))
        rng.shuffle(indices)

        n = len(characters)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        return [characters[i] for i in selected]

    def _generate_triplets(
        self, characters: list[_Character],
    ) -> list[tuple[Path, float, Path, float, Path, float, float]]:
        """Generate all valid (src_A, angle_A, src_B, angle_B, target, angle_target, weight) triplets."""
        triplets = []

        for char in characters:
            view_names = list(char.views.keys())
            if len(view_names) < 3:
                # Need at least 3 views to make a triplet (2 source + 1 target)
                # For characters with only 2 views, use both as source → predict one of them
                if len(view_names) == 2:
                    a, b = view_names
                    # Predict each from the other
                    triplets.append((
                        char.views[a], VIEW_ANGLES.get(a, 0.0),
                        char.views[b], VIEW_ANGLES.get(b, 0.8),
                        char.views[b], VIEW_ANGLES.get(b, 0.8),
                        char.dataset_weight,
                    ))
                    triplets.append((
                        char.views[b], VIEW_ANGLES.get(b, 0.8),
                        char.views[a], VIEW_ANGLES.get(a, 0.0),
                        char.views[a], VIEW_ANGLES.get(a, 0.0),
                        char.dataset_weight,
                    ))
                continue

            # For 3+ views: pick 2 as source, 1 as target
            for target_name in view_names:
                source_names = [v for v in view_names if v != target_name]
                for src_a, src_b in combinations(source_names, 2):
                    triplets.append((
                        char.views[src_a], VIEW_ANGLES.get(src_a, 0.0),
                        char.views[src_b], VIEW_ANGLES.get(src_b, 0.2),
                        char.views[target_name], VIEW_ANGLES.get(target_name, 0.8),
                        char.dataset_weight,
                    ))

        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        src_a_path, angle_a, src_b_path, angle_b, target_path, angle_target, weight = self.triplets[idx]
        res = self.resolution

        def load_rgba(path: Path) -> np.ndarray:
            img = Image.open(path).convert("RGBA")
            if img.size != (res, res):
                img = img.resize((res, res), Image.LANCZOS)
            return np.array(img, dtype=np.float32) / 255.0  # [H, W, 4]

        src_a = load_rgba(src_a_path)
        src_b = load_rgba(src_b_path)
        target = load_rgba(target_path)

        # Color jitter (train only)
        if self.split == "train" and self.config.color_jitter:
            jit = self.config.color_jitter
            brightness = 1.0 + random.uniform(-jit.get("brightness", 0), jit.get("brightness", 0))
            contrast = 1.0 + random.uniform(-jit.get("contrast", 0), jit.get("contrast", 0))
            for arr in [src_a, src_b, target]:
                arr[:, :, :3] = np.clip(arr[:, :, :3] * brightness, 0, 1)
                mean = arr[:, :, :3].mean()
                arr[:, :, :3] = np.clip((arr[:, :, :3] - mean) * contrast + mean, 0, 1)

        # Convert to tensors [C, H, W]
        src_a_t = torch.from_numpy(src_a.transpose(2, 0, 1))  # [4, H, W]
        src_b_t = torch.from_numpy(src_b.transpose(2, 0, 1))  # [4, H, W]
        target_t = torch.from_numpy(target.transpose(2, 0, 1))  # [4, H, W]

        # Angle map: constant value broadcast to [1, H, W]
        angle_map = torch.full((1, res, res), angle_target, dtype=torch.float32)

        # Concatenate input: [9, H, W] = src_A (4) + src_B (4) + angle (1)
        input_t = torch.cat([src_a_t, src_b_t, angle_map], dim=0)

        return {
            "image": input_t,       # [9, H, W]
            "target": target_t,     # [4, H, W]
            "dataset_weight": weight,
        }
