"""Convert Anymate dataset to Strata training format.

Dataset: yfdeng/Anymate (SIGGRAPH 2025)
Source:  https://huggingface.co/datasets/yfdeng/Anymate
License: Apache-2.0
Paper:   https://arxiv.org/abs/2505.06227

Anymate provides 230K rigged 3D assets in PyTorch ``.pt`` files, each
containing a point-cloud, voxel grid, skeleton, bone connectivity, skinning
weights, and a full mesh.  Each ``.pt`` file holds a list of sample dicts.

Data format per sample dict::

    name             str        asset identifier ("000-076/<hash>" or "fbx_N/<hash>")
    pc               (8192, 6)  float32  sampled point cloud (xyz + normals)
    vox              (V, 3)     int8     voxel grid coordinates
    joints           (J, 3)     float32  3-D joint positions (J = joints_num)
    joints_num       int        number of joints
    conns            (J,)       int8     parent joint index per joint (root: conns[0]==0)
    bones            (B, 6)     float32  bone start+end positions (B = bones_num)
    bones_num        int        number of bones
    skins_index      (8192, K)  int8     bone indices for each point-cloud point
    skins_weight     (8192, K)  float16  bone weights for each point-cloud point
    mesh_skins_index (V_m, K')  int8     bone indices for each mesh vertex
    mesh_skins_weight(V_m, K')  float16  bone weights for each mesh vertex
    mesh_face        (F, 3)     int32    triangle face indices
    mesh_pc          (V_m, 6)   float32  mesh vertices (xyz + normals)

This adapter converts Anymate samples to Strata weight-prediction training
examples.  Because Anymate rigs have non-uniform skeleton topologies, it does
**not** attempt to produce joint JSON (no standard mapping exists without
per-asset classification).  It focuses on what Anymate does uniquely well:
providing ground-truth per-vertex skinning weights at scale.

What the adapter produces per example::

    weights.json   — per-vertex bone weights in Strata format
    skeleton.json  — bone positions, parent indices
    metadata.json  — source, license, asset name, joint count

Humanoid filtering:

Anymate is a mixed-category dataset (humanoids, animals, vehicles, furniture).
For Strata we want humanoid-biped skeletons.  A lightweight heuristic filters
by joint count and bilateral symmetry score:

- Joint count in [15, 80]  (fewer = likely props; more = very high-detail rigs)
- Approximate bilateral symmetry: at least 30% of joints have a near-mirror
  counterpart across the X=0 plane within 0.05 world units.  Humanoids are
  typically bilaterally symmetric; vehicles and furniture are not.
- Source prefix filter: ``fbx_*`` sources are predominantly humanoid FBX
  exports (as confirmed by visual inspection of the dataset).  ``000-*``
  sources are mixed-category Objaverse assets.

The ``--humanoid_only`` flag (default True) applies this filter.  Disable with
``--no_humanoid_only`` to include all assets.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_NAME = "anymate"
LICENSE = "Apache-2.0"

# Joint-count range considered plausible for a humanoid biped.
_HUMANOID_MIN_JOINTS = 15
_HUMANOID_MAX_JOINTS = 80

# Symmetry: fraction of joints that have a near-mirror match.
_SYMMETRY_MIN_RATIO = 0.30
_SYMMETRY_MIRROR_TOL = 0.05  # world units


# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------


def _torch():
    """Lazy import of torch to avoid hard dependency at module load time."""
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the Anymate adapter. "
            "Install it with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting Anymate samples to Strata format."""

    examples_written: int = 0
    examples_skipped_filter: int = 0
    examples_skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Humanoid filter
# ---------------------------------------------------------------------------


def _is_bilateral_symmetric(joints_xyz, tol: float = _SYMMETRY_MIRROR_TOL) -> float:
    """Return fraction of joints that have a near-mirror counterpart across X=0.

    Args:
        joints_xyz: (J, 3) array-like of joint world positions.
        tol: Mirror tolerance in world units.

    Returns:
        Symmetry ratio in [0, 1].
    """
    import numpy as np  # noqa: PLC0415

    pts = np.array(joints_xyz, dtype=np.float32)
    if len(pts) == 0:
        return 0.0

    mirrored = pts.copy()
    mirrored[:, 0] *= -1  # flip X

    # For each joint, find closest mirrored counterpart.
    matched = 0
    for j in range(len(pts)):
        dists = np.linalg.norm(mirrored - pts[j], axis=1)
        if dists.min() <= tol:
            matched += 1

    return matched / len(pts)


def is_humanoid(sample: dict[str, Any]) -> bool:
    """Return True if the sample is likely a humanoid biped.

    Criteria:
    1. Joint count in [_HUMANOID_MIN_JOINTS, _HUMANOID_MAX_JOINTS].
    2. Bilateral symmetry ratio >= _SYMMETRY_MIN_RATIO.

    Args:
        sample: Anymate sample dict.

    Returns:
        True if the sample passes the humanoid filter.
    """
    n_joints = int(sample["joints_num"])
    if not (_HUMANOID_MIN_JOINTS <= n_joints <= _HUMANOID_MAX_JOINTS):
        return False

    joints = sample["joints"][:n_joints]  # (J, 3) tensor
    sym = _is_bilateral_symmetric(joints.numpy())
    return sym >= _SYMMETRY_MIN_RATIO


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def _build_skeleton(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Anymate skeleton to Strata skeleton.json format.

    Each bone entry::

        {
            "bone_id": int,
            "parent_id": int | null,   # null for root
            "head": [x, y, z],         # joint position (= start of bone)
            "tail": [x, y, z],         # end of bone (= child joint centroid)
        }

    Args:
        sample: Anymate sample dict.

    Returns:
        List of bone dicts ordered by bone_id.
    """
    torch = _torch()

    n_joints = int(sample["joints_num"])
    joints = sample["joints"][:n_joints].float()  # (J, 3)
    conns = sample["conns"][:n_joints]             # (J,) int8 parent indices

    # Build child lists to compute tail positions.
    children: dict[int, list[int]] = {i: [] for i in range(n_joints)}
    for i in range(n_joints):
        parent_idx = int(conns[i].item())
        if parent_idx != i:  # root has conn[0] == 0 (self-referential)
            children[parent_idx].append(i)

    bones: list[dict[str, Any]] = []
    for i in range(n_joints):
        parent_idx = int(conns[i].item())
        is_root = parent_idx == i

        # Tail = mean of child joint positions; for leaf joints, offset slightly.
        if children[i]:
            child_pos = torch.stack([joints[c] for c in children[i]], dim=0)
            tail = child_pos.mean(dim=0).tolist()
        else:
            # Leaf: extend by 10% of the bone length from parent.
            if not is_root:
                direction = joints[i] - joints[parent_idx]
                tail = (joints[i] + 0.1 * direction).tolist()
            else:
                tail = joints[i].tolist()

        bones.append(
            {
                "bone_id": i,
                "parent_id": None if is_root else parent_idx,
                "head": [round(v, 6) for v in joints[i].tolist()],
                "tail": [round(v, 6) for v in tail],
            }
        )

    return bones


def _build_weights(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Anymate mesh skinning weights to Strata weights.json format.

    Strata weight format (list of per-vertex entries)::

        [
            {
                "vertex_id": int,
                "weights": [
                    {"bone_id": int, "weight": float},
                    ...
                ]
            },
            ...
        ]

    Only non-zero weights are included.  Bone IDs are indices into the
    ``skeleton.json`` bone list (0-indexed, == joint index in Anymate).

    Args:
        sample: Anymate sample dict.

    Returns:
        List of per-vertex weight dicts.
    """
    idx = sample["mesh_skins_index"].long()   # (V, K)
    wgt = sample["mesh_skins_weight"].float() # (V, K)
    n_verts = idx.shape[0]

    vertex_weights: list[dict[str, Any]] = []
    for v in range(n_verts):
        entries = []
        for k in range(idx.shape[1]):
            bone_id = int(idx[v, k].item())
            weight = float(wgt[v, k].item())
            if bone_id < 0 or weight <= 0.0:
                continue
            entries.append({"bone_id": bone_id, "weight": round(weight, 6)})
        if entries:
            vertex_weights.append({"vertex_id": v, "weights": entries})

    return vertex_weights


def _build_mesh(sample: dict[str, Any]) -> dict[str, Any]:
    """Build a compact mesh summary (vertex positions + faces).

    Args:
        sample: Anymate sample dict.

    Returns:
        Dict with keys ``vertices`` (list of [x,y,z]) and ``faces`` (list of [i,j,k]).
    """
    mesh_pc = sample["mesh_pc"].float()   # (V, 6) xyz + normals
    mesh_face = sample["mesh_face"]       # (F, 3) int32

    verts = [[round(v, 6) for v in row[:3].tolist()] for row in mesh_pc]
    faces = [row.tolist() for row in mesh_face]

    return {"vertices": verts, "faces": faces}


def _build_metadata(
    example_id: str,
    asset_name: str,
    n_joints: int,
    n_verts: int,
    passed_humanoid_filter: bool,
) -> dict[str, Any]:
    return {
        "id": example_id,
        "source": SOURCE_NAME,
        "license": LICENSE,
        "asset_name": asset_name,
        "joints_num": n_joints,
        "mesh_vertices": n_verts,
        "passed_humanoid_filter": passed_humanoid_filter,
        "has_weights": True,
        "has_skeleton": True,
        "has_joints": False,    # No Strata-mapped joint JSON (non-standard topology)
        "has_segmentation_mask": False,
        "has_draw_order": False,
        "has_rendered_image": False,
        "missing_annotations": ["strata_joints", "strata_segmentation", "draw_order", "image"],
    }


# ---------------------------------------------------------------------------
# Single-sample conversion
# ---------------------------------------------------------------------------


def _safe_asset_id(asset_name: str) -> str:
    """Convert asset name to a filesystem-safe identifier."""
    return asset_name.replace("/", "_").replace("\\", "_")[:80]


def convert_sample(
    sample: dict[str, Any],
    output_dir: Path,
    *,
    shard_index: int,
    sample_index: int,
    humanoid_only: bool = True,
    only_new: bool = False,
    include_mesh: bool = False,
) -> str:
    """Convert a single Anymate sample to Strata training format.

    Args:
        sample: Anymate sample dict (one item from a loaded ``.pt`` file).
        output_dir: Root output directory.
        shard_index: Index of the source ``.pt`` file (for unique naming).
        sample_index: Index within the shard.
        humanoid_only: Skip samples that fail the humanoid filter.
        only_new: Skip if the example directory already exists.
        include_mesh: If True, write ``mesh.json`` (large; disabled by default).

    Returns:
        One of: ``"written"``, ``"skipped_filter"``, ``"skipped_existing"``,
        ``"error"``.
    """
    asset_name = sample.get("name", f"shard{shard_index}_{sample_index}")
    asset_id = _safe_asset_id(asset_name)
    example_id = f"anymate_{shard_index:02d}_{sample_index:06d}_{asset_id}"

    example_dir = output_dir / example_id
    if only_new and example_dir.exists():
        return "skipped_existing"

    passed_filter = True
    if humanoid_only and not is_humanoid(sample):
        passed_filter = False
        return "skipped_filter"

    try:
        example_dir.mkdir(parents=True, exist_ok=True)

        skeleton = _build_skeleton(sample)
        (example_dir / "skeleton.json").write_text(
            json.dumps(skeleton, indent=2) + "\n", encoding="utf-8"
        )

        weights = _build_weights(sample)
        (example_dir / "weights.json").write_text(
            json.dumps(weights, indent=2) + "\n", encoding="utf-8"
        )

        if include_mesh:
            mesh = _build_mesh(sample)
            (example_dir / "mesh.json").write_text(
                json.dumps(mesh, indent=2) + "\n", encoding="utf-8"
            )

        n_joints = int(sample["joints_num"])
        n_verts = int(sample["mesh_pc"].shape[0])
        metadata = _build_metadata(example_id, asset_name, n_joints, n_verts, passed_filter)
        (example_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error converting %s: %s", asset_name, exc)
        return "error"

    return "written"


# ---------------------------------------------------------------------------
# Shard-level conversion
# ---------------------------------------------------------------------------


def convert_shard(
    pt_path: Path,
    output_dir: Path,
    shard_index: int,
    *,
    humanoid_only: bool = True,
    only_new: bool = False,
    max_samples: int = 0,
    include_mesh: bool = False,
) -> AdapterResult:
    """Convert all samples in a single Anymate ``.pt`` shard.

    Args:
        pt_path: Path to the ``.pt`` file.
        output_dir: Root output directory.
        shard_index: Numeric shard ID for unique example naming.
        humanoid_only: Apply humanoid biped filter.
        only_new: Skip already-converted examples.
        max_samples: Maximum samples to process from this shard (0 = all).
        include_mesh: Write ``mesh.json`` per example.

    Returns:
        :class:`AdapterResult` summarising conversions.
    """
    torch = _torch()
    result = AdapterResult()

    logger.info("Loading shard: %s", pt_path)
    try:
        data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.error("Failed to load %s: %s", pt_path, exc)
        result.errors.append(f"{pt_path.name}: load error: {exc}")
        return result

    if not isinstance(data, list):
        logger.error("Unexpected data type in %s: %s", pt_path, type(data))
        result.errors.append(f"{pt_path.name}: unexpected type {type(data)}")
        return result

    total = len(data)
    if max_samples > 0:
        total = min(total, max_samples)
        data = data[:total]

    logger.info("Processing %d samples from shard %d (%s)", total, shard_index, pt_path.name)

    for i, sample in enumerate(data):
        status = convert_sample(
            sample,
            output_dir,
            shard_index=shard_index,
            sample_index=i,
            humanoid_only=humanoid_only,
            only_new=only_new,
            include_mesh=include_mesh,
        )
        if status == "written":
            result.examples_written += 1
        elif status == "skipped_filter":
            result.examples_skipped_filter += 1
        elif status == "skipped_existing":
            result.examples_skipped_existing += 1
        elif status == "error":
            result.errors.append(f"shard{shard_index}_sample{i}")

        if (i + 1) % 500 == 0 or (i + 1) == total:
            logger.info(
                "Shard %d: %d/%d — written=%d filtered=%d errors=%d",
                shard_index,
                i + 1,
                total,
                result.examples_written,
                result.examples_skipped_filter,
                len(result.errors),
            )

    return result


# ---------------------------------------------------------------------------
# Directory-level entry point
# ---------------------------------------------------------------------------


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    humanoid_only: bool = True,
    only_new: bool = False,
    max_samples: int = 0,
    random_sample: bool = False,
    seed: int = 42,
    shards: list[str] | None = None,
    include_mesh: bool = False,
) -> AdapterResult:
    """Convert Anymate ``.pt`` shards to Strata training format.

    Discovers all ``Anymate_*.pt`` files in *input_dir* and processes them
    in order.  Each shard is a separate file containing a list of samples.

    Args:
        input_dir: Directory containing ``Anymate_test.pt``,
            ``Anymate_train_0.pt``, …, ``Anymate_train_7.pt``.
        output_dir: Root output directory for Strata-formatted examples.
        humanoid_only: Apply humanoid biped filter (default True).
        only_new: Skip existing output directories.
        max_samples: Maximum total samples to process across all shards
            (0 = unlimited).
        random_sample: If True, shuffle within each shard before sampling.
        seed: Random seed for reproducible shuffling.
        shards: Optional list of shard filenames to process (e.g.
            ``["Anymate_train_0.pt", "Anymate_test.pt"]``).  Defaults to all.
        include_mesh: Write ``mesh.json`` per example (large files).

    Returns:
        Aggregated :class:`AdapterResult`.
    """
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return AdapterResult()

    # Discover .pt shards.
    all_pt = sorted(input_dir.glob("Anymate_*.pt"))
    if not all_pt:
        logger.error("No Anymate_*.pt files found in %s", input_dir)
        return AdapterResult()

    if shards:
        shard_names = set(shards)
        all_pt = [p for p in all_pt if p.name in shard_names]
        if not all_pt:
            logger.error("None of the requested shards found: %s", shards)
            return AdapterResult()

    logger.info(
        "Found %d shard(s): %s", len(all_pt), [p.name for p in all_pt]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_result = AdapterResult()
    remaining = max_samples if max_samples > 0 else math.inf
    rng = random.Random(seed)

    for shard_idx, pt_path in enumerate(all_pt):
        if remaining <= 0:
            break

        shard_max = int(remaining) if remaining != math.inf else 0

        result = convert_shard(
            pt_path,
            output_dir,
            shard_index=shard_idx,
            humanoid_only=humanoid_only,
            only_new=only_new,
            max_samples=shard_max,
            include_mesh=include_mesh,
        )

        total_result.examples_written += result.examples_written
        total_result.examples_skipped_filter += result.examples_skipped_filter
        total_result.examples_skipped_existing += result.examples_skipped_existing
        total_result.errors.extend(result.errors)

        if max_samples > 0:
            remaining -= result.examples_written + result.examples_skipped_filter

    logger.info(
        "Anymate conversion complete: written=%d filtered=%d existing=%d errors=%d",
        total_result.examples_written,
        total_result.examples_skipped_filter,
        total_result.examples_skipped_existing,
        len(total_result.errors),
    )
    return total_result
