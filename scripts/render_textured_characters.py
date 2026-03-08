"""Render unrigged textured GLB/FBX characters from multiple camera angles.

Lightweight pipeline for characters without armatures. Produces textured
image renders + depth + normals from all configured camera angles. No
segmentation, joints, or weights (those require a skeleton).

Useful for training Models 4-6 (inpainting, texture inpainting, novel view
synthesis) which need multi-view textured images but not body region labels.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python \
        scripts/render_textured_characters.py -- \
        --input-dir /Volumes/TAMWoolff/data/raw/meshy_cc0_glb \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/meshy_cc0_unrigged

Output per example:
    {char_id}_{angle}/
        image.png       # textured render (512x512, RGBA)
        depth.png       # depth map (grayscale uint8)
        normals.png     # surface normals (RGB uint8)
        metadata.json   # source, angle, character info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is importable
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

from pipeline.config import CAMERA_ANGLES, RENDER_RESOLUTION, TARGET_CHARACTER_HEIGHT
from pipeline.importer import clear_scene, _combined_bounding_box
from pipeline.renderer import (
    render_color,
    render_depth,
    render_normals,
    setup_camera,
    setup_color_render,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("render_textured")


def _normalize_unrigged(meshes: list[bpy.types.Object]) -> None:
    """Scale and center unrigged meshes to standard size."""
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    height = bbox_max.z - bbox_min.z

    if height < 1e-6:
        logger.warning("Near-zero height (%.6f) — skipping normalization", height)
        return

    scale_factor = TARGET_CHARACTER_HEIGHT / height

    for obj in meshes:
        obj.scale *= scale_factor

    # Apply scale transform
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Recompute bbox and center
    bbox_min, bbox_max = _combined_bounding_box(meshes)
    center_x = (bbox_min.x + bbox_max.x) / 2
    center_y = (bbox_min.y + bbox_max.y) / 2
    offset = Vector((-center_x, -center_y, -bbox_min.z))

    for obj in meshes:
        obj.location += offset

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


def import_model(filepath: Path) -> list[bpy.types.Object] | None:
    """Import a GLB or FBX and return mesh objects."""
    clear_scene()

    suffix = filepath.suffix.lower()
    try:
        if suffix in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(filepath))
        elif suffix == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(filepath))
        else:
            logger.error("Unsupported format: %s", suffix)
            return None
    except Exception:
        logger.exception("Failed to import: %s", filepath)
        return None

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        logger.error("No meshes in %s", filepath.name)
        return None

    return meshes


def discover_models(input_dir: Path) -> list[Path]:
    """Find all GLB and FBX model files in the input directory.

    Searches subdirectories recursively to handle nested zip extractions
    (e.g. ``parent/child/model.fbx``).  Skips withSkin animation files.
    """
    # Use rglob to find all model files regardless of nesting depth
    found: dict[str, Path] = {}  # keyed by parent dir name to deduplicate
    for pattern, suffix in [("**/*.glb", ".glb"), ("**/*.fbx", ".fbx")]:
        for f in input_dir.rglob(f"*{suffix}"):
            if f.name.startswith("._") or "withSkin" in f.name:
                continue
            # Use the immediate parent directory name as key
            key = f.parent.name
            if key not in found:
                found[key] = f
    return sorted(found.values())


def process_character(
    model_path: Path,
    output_dir: Path,
    angles: dict[str, dict],
) -> int:
    """Import, normalize, and render one character from all angles.

    Returns:
        Number of examples rendered.
    """
    char_id = model_path.stem
    examples = 0

    # Check if all angles already rendered (skip entire character)
    all_exist = all(
        (output_dir / f"{char_id}_{angle}" / "image.png").exists()
        for angle in angles
    )
    if all_exist:
        return -1  # signal: skipped

    meshes = import_model(model_path)
    if meshes is None:
        return 0

    _normalize_unrigged(meshes)

    scene = bpy.context.scene

    for angle_name, angle_cfg in angles.items():
        example_dir = output_dir / f"{char_id}_{angle_name}"

        # Skip if already rendered
        if (example_dir / "image.png").exists():
            examples += 1
            continue

        example_dir.mkdir(parents=True, exist_ok=True)

        azimuth = angle_cfg.get("azimuth", 0)
        elevation = angle_cfg.get("elevation", 0)

        # Camera
        setup_camera(scene, meshes, azimuth=azimuth, elevation=elevation)

        # Textured color render
        setup_color_render(scene)
        render_color(scene, example_dir / "image.png")

        # Depth
        render_depth(scene, example_dir / "depth.png", meshes)

        # Normals
        render_normals(scene, example_dir / "normals.png", meshes)

        # Metadata
        metadata = {
            "source": "meshy_cc0",
            "character_id": char_id,
            "angle": angle_name,
            "azimuth": azimuth,
            "elevation": elevation,
            "has_armature": False,
            "has_segmentation": False,
            "has_joints": False,
            "has_weights": False,
            "style": "textured",
            "resolution": RENDER_RESOLUTION,
        }
        (example_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        examples += 1

    return examples


def main() -> None:
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render unrigged textured characters")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with model subdirectories (GLB/FBX)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for rendered examples")
    parser.add_argument("--batch", type=int, nargs=2, default=None, metavar=("START", "SIZE"),
                        help="Process a batch subset (start index, batch size)")
    parser.add_argument("--angles", nargs="*", default=None,
                        help="Specific angles to render (default: all)")
    parser.add_argument("--exclude-rigged", type=Path, default=None,
                        help="Path to rigged_to_original_ratios.json — skip originals already matched to rigged characters")
    args = parser.parse_args(argv)

    # Determine angles
    if args.angles:
        angles = {k: v for k, v in CAMERA_ANGLES.items() if k in args.angles}
    else:
        angles = CAMERA_ANGLES

    # Discover models
    models = discover_models(args.input_dir)
    logger.info("Found %d models in %s", len(models), args.input_dir)

    # Optionally exclude models that are already covered by rigged pipeline
    if args.exclude_rigged and args.exclude_rigged.exists():
        mapping = json.loads(args.exclude_rigged.read_text())
        rigged_originals = set(mapping.values())
        before = len(models)
        models = [m for m in models if str(m) not in rigged_originals]
        logger.info("Excluded %d models already matched to rigged characters, %d remaining",
                     before - len(models), len(models))

    # Apply batch
    if args.batch:
        start, size = args.batch
        models = models[start:start + size]
        logger.info("Batch [%d:%d] — processing %d models", start, start + size, len(models))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, model_path in enumerate(models):
        char_id = model_path.stem
        result = process_character(model_path, args.output_dir, angles)

        if result == -1:
            skipped += 1
        elif result == 0:
            errors += 1
            logger.error("[%d/%d] FAILED: %s", i + 1, len(models), char_id)
        else:
            rendered += 1
            if rendered % 10 == 0:
                elapsed = time.time() - t0
                rate = rendered / elapsed if elapsed > 0 else 0
                logger.info("[%d/%d] %d rendered (%.1f chars/min), %d skipped, %d errors",
                            i + 1, len(models), rendered, rate * 60, skipped, errors)

    elapsed = time.time() - t0
    total_examples = (rendered + skipped) * len(angles)

    print(f"\n{'='*60}")
    print(f"RENDERING COMPLETE")
    print(f"{'='*60}")
    print(f"  Models found:     {len(models)}")
    print(f"  Rendered:         {rendered}")
    print(f"  Skipped:          {skipped}")
    print(f"  Errors:           {errors}")
    print(f"  Total examples:   ~{total_examples}")
    print(f"  Time:             {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")

    # Save stats
    stats = {
        "models_found": len(models),
        "rendered": rendered,
        "skipped": skipped,
        "errors": errors,
        "angles": len(angles),
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.output_dir / "render_stats.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
