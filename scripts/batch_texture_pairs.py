#!/usr/bin/env python3
"""Batch-generate UV texture training pairs for ControlNet inpainting.

Iterates over 3D character files (FBX/GLB/VRM), imports each into Blender,
and generates (partial_texture, complete_texture, inpainting_mask, position_map,
normal_map) triplets using ``texture_projection_trainer.py``.

Usage::

    blender --background --python scripts/batch_texture_pairs.py -- \\
        --input_dir /Volumes/TAMWoolff/data/raw/meshy_cc0_rigged/ \\
        --output_dir ./output/texture_pairs/ \\
        --tex_resolution 1024 \\
        --max_chars 0

Supports FBX, GLB, GLTF, and VRM formats.
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

from mesh.scripts.texture_projection_trainer import generate_texture_pairs

logger = logging.getLogger(__name__)

# Supported 3D model formats
MODEL_EXTENSIONS = {".fbx", ".glb", ".gltf", ".vrm"}


def find_model_files(input_dir: Path) -> list[Path]:
    """Recursively find all supported 3D model files."""
    files = []
    for ext in MODEL_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))

    # Filter out macOS resource fork files
    files = [f for f in files if not f.name.startswith("._")]

    # Sort for deterministic ordering
    files.sort()
    return files


def clear_scene() -> None:
    """Remove all objects, meshes, materials, etc. from the scene."""
    # Remove all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clean up orphaned data
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def import_model(filepath: Path) -> bool:
    """Import a 3D model file into the current scene.

    Returns True on success, False on failure.
    """
    ext = filepath.suffix.lower()

    try:
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(filepath))
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(filepath))
        elif ext == ".vrm":
            # VRM is GLTF-based
            bpy.ops.import_scene.gltf(filepath=str(filepath))
        else:
            logger.warning("Unsupported format: %s", filepath)
            return False
    except Exception as e:
        logger.error("Failed to import %s: %s", filepath, e)
        return False

    return True


def get_mesh_objects() -> list[bpy.types.Object]:
    """Get all mesh objects in the scene."""
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def process_character(
    filepath: Path,
    output_dir: Path,
    tex_resolution: int,
    partial_angles: list[int] | None = None,
) -> dict[str, Any] | None:
    """Process a single character file and generate texture pairs.

    Returns metadata dict on success, None on failure.
    """
    character_id = filepath.stem

    # Skip if already processed
    pair_dir = output_dir / f"{character_id}_pose_00"
    if (pair_dir / "complete_texture.png").exists() and (pair_dir / "position_map.png").exists():
        logger.info("Skipping %s (already processed)", character_id)
        return {"character_id": character_id, "status": "skipped"}

    # Clear scene and import
    clear_scene()
    if not import_model(filepath):
        return None

    meshes = get_mesh_objects()
    if not meshes:
        logger.warning("No meshes found in %s", filepath)
        return None

    logger.info("Processing %s (%d meshes)", character_id, len(meshes))

    # Set up basic render settings
    scene = bpy.context.scene
    # Blender 4.x uses BLENDER_EEVEE_NEXT, 5.0+ uses BLENDER_EEVEE
    if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True

    try:
        kwargs = dict(
            scene=scene,
            meshes=meshes,
            output_dir=output_dir,
            character_id=character_id,
            pose_id=0,
            tex_resolution=tex_resolution,
        )
        if partial_angles is not None:
            kwargs["partial_angles"] = partial_angles
        metadata = generate_texture_pairs(**kwargs)
        metadata["status"] = "success"
        metadata["source_file"] = str(filepath)
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
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []

    parser = argparse.ArgumentParser(
        description="Batch-generate UV texture training pairs",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        nargs="+",
        help="Input directory/directories containing 3D model files",
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
        default=None,
        help="Comma-separated partial view angles (default: config default 0,45,180). E.g. '0' for front-only.",
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

    # Find all model files
    all_files: list[Path] = []
    for input_dir in args.input_dir:
        if not input_dir.is_dir():
            logger.warning("Input directory not found: %s", input_dir)
            continue
        files = find_model_files(input_dir)
        logger.info("Found %d model files in %s", len(files), input_dir)
        all_files.extend(files)

    if not all_files:
        logger.error("No model files found")
        sys.exit(1)

    if args.max_chars > 0:
        all_files = all_files[: args.max_chars]

    # Parse partial angles
    partial_angles = None
    if args.partial_angles is not None:
        partial_angles = [int(a.strip()) for a in args.partial_angles.split(",")]
        logger.info("Using custom partial angles: %s", partial_angles)

    logger.info("Processing %d characters → %s", len(all_files), args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each character
    results: list[dict[str, Any]] = []
    n_success = 0
    n_error = 0
    n_skipped = 0

    for i, filepath in enumerate(all_files):
        logger.info("[%d/%d] %s", i + 1, len(all_files), filepath.name)
        result = process_character(filepath, args.output_dir, args.tex_resolution, partial_angles)

        if result is None:
            n_error += 1
            results.append({"source_file": str(filepath), "status": "error"})
        elif result.get("status") == "skipped":
            n_skipped += 1
            results.append(result)
        elif result.get("status") == "success":
            n_success += 1
            results.append(result)
        else:
            n_error += 1
            results.append(result)

    # Save summary
    summary = {
        "total": len(all_files),
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
        n_success, n_skipped, n_error, len(all_files),
    )
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
