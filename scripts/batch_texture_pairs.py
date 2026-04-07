#!/usr/bin/env python3
"""Batch-generate UV texture training pairs for ControlNet inpainting.

Uses existing texture files as ground truth — no rendering needed. For each
character, loads the FBX/GLB mesh + its texture PNG, computes which UV regions
are visible from partial view angles, and outputs training pairs.

Usage::

    blender --background --python scripts/batch_texture_pairs.py -- \\
        --input_dir /Volumes/TAMWoolff/data/fbx/ \\
        --output_dir ./output/texture_pairs/ \\
        --partial_angles "0" \\
        --max_chars 50

Supports FBX, GLB, GLTF, and VRM formats. Expects texture PNGs alongside
the model files (standard Meshy/Sketchfab export layout).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]

# Append project root so we can import our modules
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mesh.scripts.texture_projection_trainer import generate_pairs_from_existing_texture

logger = logging.getLogger(__name__)

# Supported 3D model formats
MODEL_EXTENSIONS = {".fbx", ".glb", ".gltf", ".vrm"}
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_textured_models(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find model files that have a matching texture file nearby.

    Looks for texture PNGs in the same directory as the model file.
    Skips files named *_metallic*, *_normal*, *_roughness* (PBR maps, not diffuse).

    Returns:
        List of (model_path, texture_path) tuples.
    """
    pairs: list[tuple[Path, Path]] = []

    for model_path in sorted(input_dir.rglob("*")):
        if model_path.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        if model_path.name.startswith("._"):
            continue

        # Look for texture in same directory
        model_dir = model_path.parent
        texture = _find_diffuse_texture(model_dir, model_path.stem)
        if texture is not None:
            pairs.append((model_path, texture))

    return pairs


def _find_diffuse_texture(directory: Path, model_stem: str) -> Path | None:
    """Find the diffuse/base color texture PNG in a directory.

    Prefers textures matching the model name. Skips PBR maps
    (metallic, normal, roughness, ao, emissive).
    """
    pbr_suffixes = {"_metallic", "_normal", "_roughness", "_ao", "_emissive", "_opacity"}

    candidates = []
    for f in directory.iterdir():
        if f.suffix.lower() not in TEXTURE_EXTENSIONS:
            continue
        if f.name.startswith("._"):
            continue
        # Skip PBR maps
        stem_lower = f.stem.lower()
        if any(stem_lower.endswith(s) for s in pbr_suffixes):
            continue
        candidates.append(f)

    if not candidates:
        return None

    # Prefer texture matching model name
    for c in candidates:
        if c.stem.startswith(model_stem):
            return c

    # Fall back to first candidate
    return candidates[0]


def clear_scene() -> None:
    """Remove all objects, meshes, materials, etc. from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def import_model(filepath: Path) -> bool:
    """Import a 3D model file into the current scene."""
    ext = filepath.suffix.lower()
    try:
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(filepath))
        elif ext in (".glb", ".gltf", ".vrm"):
            bpy.ops.import_scene.gltf(filepath=str(filepath))
        else:
            return False
    except Exception as e:
        logger.error("Failed to import %s: %s", filepath, e)
        return False
    return True


def get_mesh_objects() -> list[bpy.types.Object]:
    """Get all mesh objects in the scene."""
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def process_character(
    model_path: Path,
    texture_path: Path,
    output_dir: Path,
    tex_resolution: int,
    partial_angles: list[int],
) -> dict[str, Any] | None:
    """Process a single character and generate texture pairs from existing texture."""
    character_id = model_path.stem

    # Skip if already processed
    pair_dir = output_dir / f"{character_id}_pose_00"
    if (pair_dir / "complete_texture.png").exists() and (pair_dir / "position_map.png").exists():
        logger.info("Skipping %s (already processed)", character_id)
        return {"character_id": character_id, "status": "skipped"}

    clear_scene()
    if not import_model(model_path):
        return None

    meshes = get_mesh_objects()
    if not meshes:
        logger.warning("No meshes found in %s", model_path)
        return None

    logger.info("Processing %s (%d meshes, texture: %s)", character_id, len(meshes), texture_path.name)

    scene = bpy.context.scene
    scene.render.film_transparent = True

    try:
        metadata = generate_pairs_from_existing_texture(
            scene=scene,
            meshes=meshes,
            texture_path=texture_path,
            output_dir=output_dir,
            character_id=character_id,
            partial_angles=partial_angles,
            tex_resolution=tex_resolution,
            skip_geometry=True,
        )
        metadata["status"] = "success"
        metadata["source_file"] = str(model_path)
        return metadata
    except Exception as e:
        logger.error("Failed to generate pairs for %s: %s", character_id, e)
        traceback.print_exc()
        return {"character_id": character_id, "status": "error", "error": str(e)}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1:]
        else:
            argv = []

    parser = argparse.ArgumentParser(
        description="Batch-generate UV texture training pairs from existing textures",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        nargs="+",
        help="Input directory/directories containing 3D model + texture files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/texture_pairs"),
        help="Output directory for texture pairs (default: %(default)s)",
    )
    parser.add_argument(
        "--tex_resolution",
        type=int,
        default=1024,
        help="UV texture resolution (default: %(default)s)",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=0,
        help="Max characters to process (0=unlimited, default: %(default)s)",
    )
    parser.add_argument(
        "--partial_angles",
        type=str,
        default="0",
        help="Comma-separated partial view angles in degrees (default: '0' = front only)",
    )
    parser.add_argument(
        "--name_filter",
        type=str,
        default=None,
        help="Only process models whose path contains this string (e.g. 'Meshy_AI')",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    partial_angles = [int(a.strip()) for a in args.partial_angles.split(",")]
    logger.info("Partial view angles: %s", partial_angles)

    # Find model+texture pairs
    all_pairs: list[tuple[Path, Path]] = []
    for input_dir in args.input_dir:
        if not input_dir.is_dir():
            logger.warning("Input directory not found: %s", input_dir)
            continue
        pairs = find_textured_models(input_dir)
        logger.info("Found %d textured models in %s", len(pairs), input_dir)
        all_pairs.extend(pairs)

    # Apply name filter
    if args.name_filter:
        before = len(all_pairs)
        all_pairs = [(m, t) for m, t in all_pairs if args.name_filter in str(m)]
        logger.info("Name filter '%s': %d → %d models", args.name_filter, before, len(all_pairs))

    if not all_pairs:
        logger.error("No textured model files found")
        sys.exit(1)

    if args.max_chars > 0:
        all_pairs = all_pairs[:args.max_chars]

    logger.info("Processing %d characters → %s", len(all_pairs), args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    n_success = 0
    n_error = 0
    n_skipped = 0

    for i, (model_path, texture_path) in enumerate(all_pairs):
        logger.info("[%d/%d] %s", i + 1, len(all_pairs), model_path.name)
        result = process_character(
            model_path, texture_path, args.output_dir,
            args.tex_resolution, partial_angles,
        )

        if result is None:
            n_error += 1
            results.append({"source_file": str(model_path), "status": "error"})
        elif result.get("status") == "skipped":
            n_skipped += 1
            results.append(result)
        elif result.get("status") == "success":
            n_success += 1
            results.append(result)
        else:
            n_error += 1
            results.append(result)

    summary = {
        "total": len(all_pairs),
        "success": n_success,
        "skipped": n_skipped,
        "error": n_error,
        "results": results,
    }
    summary_path = args.output_dir / "generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Done: %d success, %d skipped, %d errors out of %d total",
        n_success, n_skipped, n_error, len(all_pairs),
    )


if __name__ == "__main__":
    main()
