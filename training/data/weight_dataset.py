"""PyTorch Dataset for per-vertex weight prediction training.

Loads pipeline weight JSON files and constructs 31-dimensional per-vertex
feature vectors matching the Rust runtime's ``build_feature_tensor()`` layout
(``strata/src-tauri/src/ai/weights.rs``).

Each "sample" is a full mesh (up to ``MAX_VERTICES`` vertices), not a single
image. The model predicts per-vertex bone weights from these features.

**Feature vector layout (31 dimensions per vertex):**

- ``[0-1]``   Normalized position (x, y) in [0, 1]
- ``[2-21]``  Normalized distance to each of 20 bones
- ``[22-25]`` Top-4 heat diffusion weights (zeroed — runtime-only data)
- ``[26-29]`` Top-4 heat diffusion bone indices (zeroed — runtime-only data)
- ``[30]``    Region label (normalized by 21, zeroed if unavailable)

Supports the same dual directory layouts as other datasets:

1. **Flat layout**::

       dataset/weights/{char_id}_pose_{nn}.json
       dataset/joints/{char_id}_pose_{nn}.json

2. **Per-example layout**::

       dataset/{example_id}/weights.json
       dataset/{example_id}/joints.json

Pure Python + NumPy/Torch (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.data.split_loader import character_id_from_example, load_or_generate_splits
from training.data.transforms import BONE_ORDER, BONE_TO_INDEX, PIPELINE_TO_BONE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_VERTICES: int = 2048
NUM_BONES: int = 20
NUM_FEATURES: int = 31


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WeightDatasetConfig:
    """Configuration for WeightDataset."""

    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    max_vertices: int = MAX_VERTICES

    @classmethod
    def from_dict(cls, d: dict) -> WeightDatasetConfig:
        """Build config from a parsed YAML config dict."""
        data = d.get("data", {})
        ratios = data.get("split_ratios", {})
        return cls(
            split_seed=data.get("split_seed", 42),
            split_ratios=(
                ratios.get("train", 0.8),
                ratios.get("val", 0.1),
                ratios.get("test", 0.1),
            ),
            max_vertices=data.get("max_vertices", MAX_VERTICES),
        )


# ---------------------------------------------------------------------------
# Example descriptor
# ---------------------------------------------------------------------------


@dataclass
class _WeightExample:
    """Paths for a single weight training example."""

    weights_path: Path
    joints_path: Path | None
    example_id: str


# ---------------------------------------------------------------------------
# Layout detection & discovery
# ---------------------------------------------------------------------------


def _discover_flat(dataset_dir: Path) -> list[_WeightExample]:
    """Discover examples from flat layout (weights/ + joints/ subdirs)."""
    weights_dir = dataset_dir / "weights"
    joints_dir = dataset_dir / "joints"

    if not weights_dir.is_dir() or not joints_dir.is_dir():
        return []

    examples: list[_WeightExample] = []
    for weights_path in sorted(weights_dir.glob("*.json")):
        stem = weights_path.stem
        joints_path = joints_dir / f"{stem}.json"

        if not joints_path.exists():
            continue

        examples.append(
            _WeightExample(
                weights_path=weights_path,
                joints_path=joints_path,
                example_id=stem,
            )
        )

    return examples


def _discover_per_example(dataset_dir: Path) -> list[_WeightExample]:
    """Discover examples from per-example layout.

    Supports two structures:
    - ``{id}/weights.json`` + ``{id}/joints.json``
    - ``{id}/{view}/weights.json`` (nested views, e.g. UniRig ``{id}/front/``)

    When ``joints.json`` is absent, ``joints_path`` is set to ``None`` and
    joint positions are derived from the weight data at load time.
    """
    examples: list[_WeightExample] = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue

        # Direct layout: {id}/weights.json
        weights_path = child / "weights.json"
        if weights_path.exists():
            joints_path = child / "joints.json"
            examples.append(
                _WeightExample(
                    weights_path=weights_path,
                    joints_path=joints_path if joints_path.exists() else None,
                    example_id=child.name,
                )
            )
            continue

        # Nested view layout: {id}/{view}/weights.json (e.g. UniRig front/)
        for view_dir in sorted(child.iterdir()):
            if not view_dir.is_dir():
                continue
            weights_path = view_dir / "weights.json"
            if not weights_path.exists():
                continue
            joints_path = view_dir / "joints.json"
            examples.append(
                _WeightExample(
                    weights_path=weights_path,
                    joints_path=joints_path if joints_path.exists() else None,
                    # Use parent dir name as example_id so character_id_from_example
                    # matches the split_loader's discovery (which uses child.name)
                    example_id=child.name,
                )
            )

    return examples


def _detect_layout(dataset_dir: Path) -> str:
    """Return ``"flat"`` or ``"per_example"`` based on directory contents."""
    if (dataset_dir / "weights").is_dir():
        return "flat"
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        # Direct: {id}/weights.json
        if (child / "weights.json").exists():
            return "per_example"
        # Nested view: {id}/{view}/weights.json (e.g. UniRig)
        for view_dir in child.iterdir():
            if view_dir.is_dir() and (view_dir / "weights.json").exists():
                return "per_example"
        break  # Only check first child for detection
    return "flat"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _parse_joint_positions(
    joints_path: Path,
) -> dict[str, tuple[float, float]]:
    """Parse joints.json and return {bone_name: (x_pixel, y_pixel)}.

    Maps pipeline joint names to BONE_ORDER names.
    """
    data = json.loads(joints_path.read_text(encoding="utf-8"))
    joints_dict = data.get("joints", {})

    positions: dict[str, tuple[float, float]] = {}
    for pipeline_name, joint_info in joints_dict.items():
        bone_name = PIPELINE_TO_BONE.get(pipeline_name, pipeline_name)
        if bone_name not in BONE_TO_INDEX:
            continue
        pos = joint_info.get("position", [0, 0])
        positions[bone_name] = (float(pos[0]), float(pos[1]))

    return positions


def _derive_joint_positions(
    vertices: list[dict],
) -> dict[str, tuple[float, float]]:
    """Derive approximate joint positions from weight data.

    Computes the weighted centroid of vertices for each bone — vertices
    with higher weight for a bone are closer to that bone's joint.
    Used when joints.json is unavailable (e.g. UniRig data).
    """
    bone_sum_x: dict[str, float] = {}
    bone_sum_y: dict[str, float] = {}
    bone_sum_w: dict[str, float] = {}

    for vert in vertices:
        vx, vy = float(vert["position"][0]), float(vert["position"][1])
        for bone_name, weight in vert.get("weights", {}).items():
            if bone_name not in BONE_TO_INDEX:
                continue
            w = float(weight)
            bone_sum_x[bone_name] = bone_sum_x.get(bone_name, 0.0) + vx * w
            bone_sum_y[bone_name] = bone_sum_y.get(bone_name, 0.0) + vy * w
            bone_sum_w[bone_name] = bone_sum_w.get(bone_name, 0.0) + w

    positions: dict[str, tuple[float, float]] = {}
    for bone_name in bone_sum_w:
        total_w = bone_sum_w[bone_name]
        if total_w > 1e-6:
            positions[bone_name] = (
                bone_sum_x[bone_name] / total_w,
                bone_sum_y[bone_name] / total_w,
            )

    return positions


def build_features(
    vertices: list[dict],
    joint_positions: dict[str, tuple[float, float]],
    image_size: tuple[int, int],
    max_vertices: int = MAX_VERTICES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Construct the 31-dim per-vertex feature tensor.

    Args:
        vertices: List of vertex dicts from weight_extractor output, each with
            ``"position"`` ``[x, y]`` and ``"weights"`` ``{region_name: weight}``.
        joint_positions: Bone name → (x_pixel, y_pixel) from joints.json.
        image_size: ``(width, height)`` for normalizing positions.
        max_vertices: Maximum vertex count (zero-padded).

    Returns:
        ``(features, weights_target, confidence_target, num_vertices)`` where:
        - ``features``: ``[NUM_FEATURES, N, 1]`` float32
        - ``weights_target``: ``[NUM_BONES, N]`` float32, per-bone GT weights
        - ``confidence_target``: ``[N]`` float32, 1.0 where vertex has weights
        - ``num_vertices``: actual vertex count before padding
    """
    n = min(len(vertices), max_vertices)

    features = np.zeros((NUM_FEATURES, max_vertices, 1), dtype=np.float32)
    weights_target = np.zeros((NUM_BONES, max_vertices), dtype=np.float32)
    confidence_target = np.zeros(max_vertices, dtype=np.float32)

    # Compute bounding box for position normalization
    if n > 0:
        xs = [v["position"][0] for v in vertices[:n]]
        ys = [v["position"][1] for v in vertices[:n]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1.0)
        range_y = max(max_y - min_y, 1.0)
    else:
        min_x = min_y = 0.0
        range_x = range_y = 1.0

    # Precompute bone positions (in pixel coords)
    bone_positions = [joint_positions.get(name, (0.0, 0.0)) for name in BONE_ORDER]

    # Max distance for normalization (bounding box diagonal)
    max_dist = max(math.sqrt(range_x**2 + range_y**2), 1.0)

    for vi in range(n):
        vert = vertices[vi]
        vx, vy = float(vert["position"][0]), float(vert["position"][1])

        # [0-1] Normalized position
        features[0, vi, 0] = (vx - min_x) / range_x
        features[1, vi, 0] = (vy - min_y) / range_y

        # [2-21] Distance to each bone (normalized)
        for bi, (bx, by) in enumerate(bone_positions):
            dx = vx - bx
            dy = vy - by
            dist = math.sqrt(dx * dx + dy * dy)
            features[2 + bi, vi, 0] = dist / max_dist

        # [22-29] Heat diffusion features — zeroed (runtime-only data)
        # Already zero from initialization

        # [30] Region label — zeroed (not available from pipeline weight data)
        # Already zero from initialization

        # Ground truth weights
        gt_weights = vert.get("weights", {})
        if gt_weights:
            # Normalize GT weights to sum to 1.0
            total = sum(gt_weights.values())
            for region_name, weight in gt_weights.items():
                if region_name in BONE_TO_INDEX:
                    idx = BONE_TO_INDEX[region_name]
                    w = weight / total if total > 0 else 0.0
                    weights_target[idx, vi] = w
            confidence_target[vi] = 1.0

    return features, weights_target, confidence_target, n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WeightDataset:
    """PyTorch Dataset for per-vertex weight prediction training.

    Returns per example:
    - ``features``: ``[31, N, 1]`` float32, per-vertex feature tensor
    - ``weights_target``: ``[20, N]`` float32, ground truth per-bone weights
    - ``confidence_target``: ``[N]`` float32, 1.0 where vertex has weight data
    - ``num_vertices``: int, actual vertex count before padding

    Args:
        dataset_dirs: One or more dataset root directories.
        split: ``"train"``, ``"val"``, or ``"test"``.
        config: Optional ``WeightDatasetConfig`` or raw dict.
    """

    def __init__(
        self,
        dataset_dirs: list[Path],
        split: str = "train",
        config: WeightDatasetConfig | dict | None = None,
    ) -> None:
        import torch

        self._torch = torch

        if isinstance(config, dict):
            self.config = WeightDatasetConfig.from_dict(config)
        elif config is None:
            self.config = WeightDatasetConfig()
        else:
            self.config = config

        self.split = split

        # Load character-level splits
        splits = load_or_generate_splits(
            dataset_dirs,
            seed=self.config.split_seed,
            ratios=self.config.split_ratios,
        )
        allowed_chars: set[str] = set(splits.get(split, []))

        # Discover examples across all directories
        all_examples: list[_WeightExample] = []
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
            "WeightDataset[%s]: %d examples (%d before split filter)",
            split,
            len(self.examples),
            len(all_examples),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        """Load and return a single training example.

        Returns:
            Dict with keys: ``features``, ``weights_target``,
            ``confidence_target``, ``num_vertices``.
        """
        torch = self._torch
        ex = self.examples[index]

        # Load weight data
        weight_data = json.loads(ex.weights_path.read_text(encoding="utf-8"))
        vertices = weight_data.get("vertices", [])
        image_size = weight_data.get("image_size", [512, 512])

        # Load joint positions for bone distance computation
        if ex.joints_path is not None:
            joint_positions = _parse_joint_positions(ex.joints_path)
        else:
            joint_positions = _derive_joint_positions(vertices)

        # Build feature tensor
        features, weights_target, confidence_target, num_verts = build_features(
            vertices,
            joint_positions,
            (image_size[0], image_size[1]),
            max_vertices=self.config.max_vertices,
        )

        return {
            "features": torch.from_numpy(features),  # [31, N, 1]
            "weights_target": torch.from_numpy(weights_target),  # [20, N]
            "confidence_target": torch.from_numpy(confidence_target),  # [N]
            "num_vertices": num_verts,
        }
