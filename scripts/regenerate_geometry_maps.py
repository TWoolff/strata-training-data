#!/usr/bin/env python3
"""Regenerate position_map.png + normal_map.png for existing texture pairs.

Replaces placeholder (all-black/gray) geometry maps with real UV-space
position and normal maps computed from the 3D mesh geometry.

Usage::

    blender --background --python scripts/regenerate_geometry_maps.py -- \\
        --pairs_dir /Volumes/TAMWoolff/data/preprocessed/texture_pairs/ \\
        --models_dir /Volumes/TAMWoolff/data/fbx/ \\
        --max_chars 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mesh.scripts.texture_projection_trainer import generate_uv_geometry_maps, ensure_uv_map

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = {".fbx", ".glb", ".gltf", ".vrm"}


def needs_regeneration(pair_dir: Path) -> bool:
    """Check if geometry maps are placeholders (all zeros/uniform)."""
    pos_path = pair_dir / "position_map.png"
    if not pos_path.exists():
        return True
    arr = np.array(Image.open(pos_path))
    # Placeholder is all zeros
    return arr.max() == 0 or np.std(arr) < 1.0


def find_model_for_pair(pair_dir: Path, models_dir: Path) -> Path | None:
    """Find the source model file for a texture pair directory.

    Matches by character ID extracted from the pair directory name.
    """
    # pair dir name format: CharacterName_pose_00_suffix
    name = pair_dir.name
    # Strip _pose_00 and any suffix (front/side/back/100av)
    parts = name.split("_pose_")
    if not parts:
        return None
    char_id = parts[0]

    # Search for matching model file
    for model_path in models_dir.rglob("*"):
        if model_path.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        if model_path.name.startswith("._"):
            continue
        if char_id in model_path.stem:
            return model_path

    return None


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def import_model(filepath: Path) -> bool:
    ext = filepath.suffix.lower()
    try:
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(filepath))
        elif ext in (".glb", ".gltf", ".vrm"):
            bpy.ops.import_scene.gltf(filepath=str(filepath))
        else:
            return False
    except Exception:
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1:]
        else:
            argv = []

    parser = argparse.ArgumentParser(
        description="Regenerate geometry maps for existing texture pairs",
    )
    parser.add_argument("--pairs_dir", type=Path, required=True,
                        help="Directory containing texture pair subdirectories")
    parser.add_argument("--models_dir", type=Path, required=True,
                        help="Directory containing source 3D model files")
    parser.add_argument("--max_chars", type=int, default=0,
                        help="Max pairs to process (0=unlimited)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if geometry maps exist and look valid")
    parser.add_argument("--tex_resolution", type=int, default=1024)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Find pairs needing regeneration
    pair_dirs = sorted([
        d for d in args.pairs_dir.iterdir()
        if d.is_dir() and "_pose_" in d.name
        and (d / "complete_texture.png").exists()
    ])

    if not args.force:
        pair_dirs = [d for d in pair_dirs if needs_regeneration(d)]

    if args.max_chars > 0:
        pair_dirs = pair_dirs[:args.max_chars]

    logger.info("Found %d pairs needing geometry maps in %s", len(pair_dirs), args.pairs_dir)

    # Cache: model path → already loaded (avoid reimporting same model)
    last_model_path: Path | None = None
    n_success = 0
    n_error = 0
    n_skip = 0

    for i, pair_dir in enumerate(pair_dirs):
        logger.info("[%d/%d] %s", i + 1, len(pair_dirs), pair_dir.name)

        model_path = find_model_for_pair(pair_dir, args.models_dir)
        if model_path is None:
            logger.warning("  No model found, skipping")
            n_skip += 1
            continue

        # Only reimport if different model
        if model_path != last_model_path:
            clear_scene()
            if not import_model(model_path):
                logger.error("  Failed to import %s", model_path)
                n_error += 1
                continue
            last_model_path = model_path

        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not meshes:
            logger.warning("  No meshes found")
            n_error += 1
            continue

        try:
            for m in meshes:
                ensure_uv_map(m)

            position_map, normal_map = generate_uv_geometry_maps(
                meshes, tex_resolution=args.tex_resolution,
            )

            Image.fromarray(position_map).save(pair_dir / "position_map.png")
            Image.fromarray(normal_map).save(pair_dir / "normal_map.png")
            n_success += 1
            logger.info("  OK: position+normal maps saved")

        except Exception as e:
            logger.error("  Failed: %s", e)
            n_error += 1

    logger.info("Done: %d success, %d errors, %d skipped out of %d",
                n_success, n_error, n_skip, len(pair_dirs))


if __name__ == "__main__":
    main()
