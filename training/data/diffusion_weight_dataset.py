"""PyTorch Dataset for diffusion-enhanced weight prediction training (Model 4).

Extends :class:`WeightDataset` with precomputed encoder features from the
segmentation model. Each example provides:

- Standard 31-dim per-vertex features (same as Model 3)
- Precomputed encoder features sampled at vertex positions
- Ground truth skinning weights

Encoder features are precomputed by ``precompute_encoder_features.py`` and
stored as ``.npy`` files alongside the weight data.

Pure Python + NumPy/Torch (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from training.data.split_loader import character_id_from_example, load_or_generate_splits
from training.data.weight_dataset import (
    MAX_VERTICES,
    _detect_layout,
    _discover_flat,
    _discover_per_example,
    _parse_joint_positions,
    _WeightExample,
    build_features,
)

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_CHANNELS: int = 960


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DiffusionWeightDatasetConfig:
    """Configuration for DiffusionWeightDataset."""

    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    max_vertices: int = MAX_VERTICES
    encoder_channels: int = DEFAULT_ENCODER_CHANNELS
    encoder_features_dirs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> DiffusionWeightDatasetConfig:
        """Build config from a parsed YAML config dict."""
        data = d.get("data", {})
        model = d.get("model", {})
        ratios = data.get("split_ratios", {})
        return cls(
            split_seed=data.get("split_seed", 42),
            split_ratios=(
                ratios.get("train", 0.8),
                ratios.get("val", 0.1),
                ratios.get("test", 0.1),
            ),
            max_vertices=data.get("max_vertices", MAX_VERTICES),
            encoder_channels=model.get("encoder_channels", DEFAULT_ENCODER_CHANNELS),
            encoder_features_dirs=data.get("encoder_features_dirs", []),
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class _DiffusionWeightExample:
    """Paths for a single diffusion weight training example."""

    weights_path: Path
    joints_path: Path
    encoder_features_path: Path
    example_id: str


class DiffusionWeightDataset:
    """PyTorch Dataset for diffusion-enhanced weight prediction training.

    Returns per example:

    - ``features``: ``[31, N, 1]`` float32, standard per-vertex features
    - ``diffusion_features``: ``[C, N, 1]`` float32, encoder features at vertex positions
    - ``weights_target``: ``[20, N]`` float32, ground truth per-bone weights
    - ``confidence_target``: ``[N]`` float32, 1.0 where vertex has weight data
    - ``num_vertices``: int, actual vertex count before padding

    Args:
        dataset_dirs: One or more dataset root directories (with weights/joints).
        encoder_features_dirs: Directories containing precomputed ``.npy`` encoder features.
        split: ``"train"``, ``"val"``, or ``"test"``.
        config: Optional config or raw dict.
    """

    def __init__(
        self,
        dataset_dirs: list[Path],
        encoder_features_dirs: list[Path] | None = None,
        split: str = "train",
        config: DiffusionWeightDatasetConfig | dict | None = None,
    ) -> None:
        import torch

        self._torch = torch

        if isinstance(config, dict):
            self.config = DiffusionWeightDatasetConfig.from_dict(config)
        elif config is None:
            self.config = DiffusionWeightDatasetConfig()
        else:
            self.config = config

        self.split = split

        # Resolve encoder features dirs from config if not passed directly
        if encoder_features_dirs is None:
            encoder_features_dirs = [Path(p) for p in self.config.encoder_features_dirs]

        # Build lookup: example_id → encoder features .npy path
        self._encoder_lookup: dict[str, Path] = {}
        for ef_dir in encoder_features_dirs:
            if not ef_dir.is_dir():
                logger.warning("Encoder features dir not found: %s", ef_dir)
                continue
            for npy_path in ef_dir.glob("*.npy"):
                self._encoder_lookup[npy_path.stem] = npy_path

        # Load character-level splits
        splits = load_or_generate_splits(
            dataset_dirs,
            seed=self.config.split_seed,
            ratios=self.config.split_ratios,
        )
        allowed_chars: set[str] = set(splits.get(split, []))

        # Discover weight examples across all directories
        all_weight_examples: list[_WeightExample] = []
        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                logger.warning("Dataset directory does not exist: %s", dataset_dir)
                continue
            layout = _detect_layout(dataset_dir)
            if layout == "flat":
                all_weight_examples.extend(_discover_flat(dataset_dir))
            else:
                all_weight_examples.extend(_discover_per_example(dataset_dir))

        # Filter by split AND encoder feature availability
        self.examples: list[_DiffusionWeightExample] = []
        skipped_no_encoder = 0
        for ex in all_weight_examples:
            if character_id_from_example(ex.example_id) not in allowed_chars:
                continue
            encoder_path = self._encoder_lookup.get(ex.example_id)
            if encoder_path is None:
                skipped_no_encoder += 1
                continue
            self.examples.append(
                _DiffusionWeightExample(
                    weights_path=ex.weights_path,
                    joints_path=ex.joints_path,
                    encoder_features_path=encoder_path,
                    example_id=ex.example_id,
                )
            )

        if skipped_no_encoder > 0:
            logger.info(
                "Skipped %d examples without precomputed encoder features",
                skipped_no_encoder,
            )

        logger.info(
            "DiffusionWeightDataset[%s]: %d examples (from %d weight examples)",
            split,
            len(self.examples),
            len(all_weight_examples),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        """Load and return a single training example.

        Returns:
            Dict with keys: ``features``, ``diffusion_features``,
            ``weights_target``, ``confidence_target``, ``num_vertices``.
        """
        torch = self._torch
        ex = self.examples[index]
        max_verts = self.config.max_vertices

        # Load weight data
        weight_data = json.loads(ex.weights_path.read_text(encoding="utf-8"))
        vertices = weight_data.get("vertices", [])
        image_size = weight_data.get("image_size", [512, 512])

        # Load joint positions
        joint_positions = _parse_joint_positions(ex.joints_path)

        # Build standard 31-dim features
        features, weights_target, confidence_target, num_verts = build_features(
            vertices,
            joint_positions,
            (image_size[0], image_size[1]),
            max_vertices=max_verts,
        )

        # Load precomputed encoder features [C, N_actual]
        raw_encoder = np.load(ex.encoder_features_path)  # [N_actual, C] or [C, N_actual]

        # Normalize to [C, N, 1] with zero-padding
        if raw_encoder.ndim == 2 and raw_encoder.shape[0] != self.config.encoder_channels:
            # Stored as [N_actual, C] — transpose
            raw_encoder = raw_encoder.T  # [C, N_actual]

        c = raw_encoder.shape[0]
        n_enc = raw_encoder.shape[1]
        n_use = min(n_enc, max_verts)

        encoder_features = np.zeros((c, max_verts, 1), dtype=np.float32)
        encoder_features[:, :n_use, 0] = raw_encoder[:, :n_use]

        return {
            "features": torch.from_numpy(features),  # [31, N, 1]
            "diffusion_features": torch.from_numpy(encoder_features),  # [C, N, 1]
            "weights_target": torch.from_numpy(weights_target),  # [20, N]
            "confidence_target": torch.from_numpy(confidence_target),  # [N]
            "num_vertices": num_verts,
        }
