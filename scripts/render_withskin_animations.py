"""Render training data from Meshy withSkin animation FBX files.

Processes ``walking_withSkin.fbx`` and ``running_withSkin.fbx`` files that
contain mesh + armature + baked animation in a single file.  For each
animation, samples keyframes evenly across the animation range and renders
from all configured camera angles.

Outputs per example: ``image.png``, ``segmentation.png``, ``joints.json``,
``draw_order.png``, ``depth.png``, ``normals.png``, ``metadata.json``.

Output structure::

    meshy_cc0/
      {char_id}_{anim}_f{frame}_{angle}/
        image.png
        segmentation.png
        joints.json
        draw_order.png
        depth.png
        normals.png
        metadata.json

Usage::

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python scripts/render_withskin_animations.py -- \
        --input-dir /Volumes/TAMWoolff/data/raw/meshy_cc0_rigged \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/meshy_cc0

"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

# Blender modules
import bpy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so pipeline imports work
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.bone_mapper import map_bones  # noqa: E402
from pipeline.config import (  # noqa: E402
    ALL_CAMERA_ANGLES,
    CAMERA_ANGLES,
    KEYFRAMES_PER_CLIP,
)
from pipeline.distortion_detector import check_distortion_in_scene  # noqa: E402
from pipeline.draw_order_extractor import extract_draw_order  # noqa: E402
from pipeline.importer import (  # noqa: E402
    ImportResult,
    _combined_bounding_box,
    _normalize_transforms,
    clear_scene,
)
from pipeline.joint_extractor import extract_joints  # noqa: E402
from pipeline.renderer import (  # noqa: E402
    assign_region_materials,
    convert_rgb_to_grayscale_mask,
    create_region_materials,
    render_color,
    render_depth,
    render_normals,
    render_segmentation,
    setup_camera,
    setup_color_render,
    setup_segmentation_render,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_withskin")

# ---------------------------------------------------------------------------
# Animation file patterns
# ---------------------------------------------------------------------------

ANIMATION_PATTERNS: list[str] = [
    "walking_withSkin.fbx",
    "running_withSkin.fbx",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_animation_fbx(input_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover all withSkin animation FBX files under the input directory.

    Args:
        input_dir: Root directory containing ``rigged_*/`` subdirectories.

    Returns:
        List of ``(char_id, anim_name, fbx_path)`` tuples, sorted by char_id
        then animation name for deterministic ordering.
    """
    results: list[tuple[str, str, Path]] = []

    for char_dir in sorted(input_dir.iterdir()):
        if not char_dir.is_dir() or not char_dir.name.startswith("rigged_"):
            continue

        char_id = char_dir.name  # e.g. "rigged_01abc..."

        for pattern in ANIMATION_PATTERNS:
            fbx_path = char_dir / pattern
            if fbx_path.is_file():
                # Extract animation name: "walking" from "walking_withSkin.fbx"
                anim_name = pattern.replace("_withSkin.fbx", "")
                results.append((char_id, anim_name, fbx_path))

    return results


def _get_animation_range() -> tuple[int, int]:
    """Get the animation frame range from the active armature's action.

    Returns:
        ``(frame_start, frame_end)`` as integers.
    """
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE" and obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            return int(action.frame_range[0]), int(action.frame_range[1])

    # Fallback to scene range
    scene = bpy.context.scene
    return scene.frame_start, scene.frame_end


def _sample_keyframes(frame_start: int, frame_end: int, count: int) -> list[int]:
    """Sample ``count`` evenly spaced keyframes across the animation range.

    Args:
        frame_start: First animation frame.
        frame_end: Last animation frame.
        count: Number of keyframes to sample.

    Returns:
        Sorted list of integer frame numbers.
    """
    total_frames = frame_end - frame_start
    if total_frames <= 0 or count <= 1:
        return [frame_start]

    step = total_frames / count
    frames = [frame_start + int(step * i) for i in range(count)]

    # Deduplicate and sort
    return sorted(set(frames))


def _import_withskin_fbx(fbx_path: Path) -> ImportResult | None:
    """Import a withSkin FBX and return structured references.

    Similar to ``importer.import_character`` but preserves the baked
    animation (does not reset to rest pose).

    Args:
        fbx_path: Path to the withSkin FBX file.

    Returns:
        An ``ImportResult``, or None if import fails.
    """
    clear_scene()

    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    except Exception:
        logger.exception("Failed to import FBX: %s", fbx_path)
        return None

    # Discover armature and meshes
    armatures: list[bpy.types.Object] = []
    meshes: list[bpy.types.Object] = []

    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            armatures.append(obj)
        elif obj.type == "MESH":
            meshes.append(obj)

    # Filter meshes to those parented to an armature
    if armatures:
        armature_names = {a.name for a in armatures}
        parented = [m for m in meshes if m.parent and m.parent.name in armature_names]
        if parented:
            stray = [m for m in meshes if m not in parented]
            for m in stray:
                bpy.data.objects.remove(m, do_unlink=True)
            meshes = parented

    if not armatures:
        logger.error("No armature in %s", fbx_path.name)
        return None
    if not meshes:
        logger.error("No mesh in %s", fbx_path.name)
        return None

    armature = armatures[0]

    # Normalize transforms (center, scale to standard height)
    # Set to rest frame first for consistent normalization
    scene = bpy.context.scene
    frame_start, _frame_end = _get_animation_range()
    scene.frame_set(frame_start)
    bpy.context.evaluated_depsgraph_get()

    _normalize_transforms(armature, meshes)

    char_id = fbx_path.parent.name  # "rigged_{task_id}"

    logger.info(
        "Imported withSkin FBX: %s (armature=%s, meshes=%d)",
        fbx_path.name,
        armature.name,
        len(meshes),
    )

    return ImportResult(
        character_id=char_id,
        armature=armature,
        meshes=meshes,
    )


def _save_metadata(
    output_dir: Path,
    char_id: str,
    anim_name: str,
    frame: int,
    angle_name: str,
    source_fbx: str,
) -> Path:
    """Save per-example metadata JSON.

    Args:
        output_dir: Example output directory.
        char_id: Character identifier.
        anim_name: Animation name (e.g. "walking").
        frame: Source animation frame number.
        angle_name: Camera angle name.
        source_fbx: Source FBX filename.

    Returns:
        Path to the saved metadata file.
    """
    metadata = {
        "character_id": char_id,
        "source": "meshy_cc0",
        "source_type": "withSkin_animation",
        "source_file": source_fbx,
        "animation": anim_name,
        "frame": frame,
        "camera_angle": angle_name,
        "style": "flat",
        "license": "CC0",
        "has_accessories": False,
    }
    path = output_dir / "metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _process_single_example(
    *,
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    mapping,
    region_materials: list,
    char_id: str,
    anim_name: str,
    frame: int,
    angle_name: str,
    angle_cfg: dict[str, int],
    output_dir: Path,
    source_fbx: str,
) -> bool:
    """Render all outputs for a single (character, animation, frame, angle) example.

    Args:
        scene: The Blender scene.
        armature: Character armature.
        meshes: Character meshes.
        mapping: BoneMapping from bone_mapper.
        region_materials: Segmentation materials.
        char_id: Character identifier.
        anim_name: Animation name.
        frame: Animation frame number.
        angle_name: Camera angle name.
        angle_cfg: Camera angle config dict with azimuth/elevation.
        output_dir: Root output directory for the dataset.
        source_fbx: Source FBX filename for metadata.

    Returns:
        True if the example was rendered successfully.
    """
    # Build example directory name
    example_name = f"{char_id}_{anim_name}_f{frame:03d}_{angle_name}"
    example_dir = output_dir / example_name

    # Skip if already rendered (resumability)
    if (example_dir / "image.png").exists() and (example_dir / "segmentation.png").exists():
        logger.debug("Skipping existing example: %s", example_name)
        return True

    example_dir.mkdir(parents=True, exist_ok=True)

    azimuth = float(angle_cfg["azimuth"])
    elevation = float(angle_cfg.get("elevation", 0))

    # Set animation frame
    scene.frame_set(frame)
    bpy.context.evaluated_depsgraph_get()

    # --- Camera setup ---
    old_cam = bpy.data.objects.get("strata_camera")
    if old_cam is not None:
        bpy.data.objects.remove(old_cam, do_unlink=True)
    camera = setup_camera(scene, meshes, azimuth=azimuth, elevation=elevation)

    # --- Assign segmentation materials ---
    for mesh_idx, mesh_obj in enumerate(meshes):
        assign_region_materials(
            mesh_obj,
            mesh_idx,
            mapping.vertex_to_region,
            region_materials,
        )

    # --- Segmentation render ---
    setup_segmentation_render(scene)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_rgb_path = Path(tmp.name)

    seg_out_path = example_dir / "segmentation.png"
    render_segmentation(scene, tmp_rgb_path)
    convert_rgb_to_grayscale_mask(tmp_rgb_path, seg_out_path)
    tmp_rgb_path.unlink(missing_ok=True)

    # --- Extract joints ---
    joint_data = extract_joints(
        scene,
        camera,
        armature,
        meshes,
        mapping.bone_to_region,
    )
    joints_path = example_dir / "joints.json"
    joints_path.write_text(
        json.dumps(joint_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # --- Draw order ---
    seg_mask_arr = np.array(Image.open(seg_out_path).convert("L"))
    draw_order_data = extract_draw_order(
        scene,
        camera,
        armature,
        meshes,
        mapping.bone_to_region,
        seg_mask_arr,
    )
    draw_order_path = example_dir / "draw_order.png"
    Image.fromarray(
        draw_order_data["draw_order_map"].astype(np.uint8), mode="L"
    ).save(draw_order_path, format="PNG", compress_level=9)

    # --- Color render (flat / emission style — no textures in withSkin) ---
    # The segmentation materials ARE the render — withSkin has no textures,
    # so we render the segmentation-colored view as the "image".
    # Re-use the seg materials which are emission-based per region.
    setup_color_render(scene)
    image_out_path = example_dir / "image.png"
    render_color(scene, image_out_path)

    # --- Depth render ---
    depth_out_path = example_dir / "depth.png"
    render_depth(scene, depth_out_path, meshes)

    # --- Normals render ---
    normals_out_path = example_dir / "normals.png"
    render_normals(scene, normals_out_path, meshes)

    # Re-assign segmentation materials (depth/normals swapped them out)
    for mesh_idx, mesh_obj in enumerate(meshes):
        assign_region_materials(
            mesh_obj,
            mesh_idx,
            mapping.vertex_to_region,
            region_materials,
        )

    # --- Metadata ---
    _save_metadata(example_dir, char_id, anim_name, frame, angle_name, source_fbx)

    logger.info("Rendered example: %s", example_name)
    return True


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_all(
    input_dir: Path,
    output_dir: Path,
    *,
    camera_angles: list[str] | None = None,
    num_keyframes: int = KEYFRAMES_PER_CLIP,
    skip_distorted: bool = True,
    batch_start: int = 0,
    batch_size: int = 0,
) -> dict[str, int]:
    """Process all withSkin animation FBX files and generate training data.

    Args:
        input_dir: Root directory with ``rigged_*/`` subdirectories.
        output_dir: Root output directory for the dataset.
        camera_angles: Camera angle names to render (default: all 33 angles).
        num_keyframes: Number of keyframes to sample per animation clip.
        skip_distorted: Skip animations that fail distortion detection.
        batch_start: Start index for batch processing (0-based).
        batch_size: Number of animations to process (0 = all remaining).

    Returns:
        Dict with processing statistics.
    """
    if camera_angles is None:
        camera_angles = ALL_CAMERA_ANGLES

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all animation FBX files
    all_anims = _discover_animation_fbx(input_dir)
    total_found = len(all_anims)
    logger.info("Found %d animation FBX files in %s", total_found, input_dir)

    if total_found == 0:
        return {"found": 0, "processed": 0, "skipped_distorted": 0, "errors": 0}

    # Apply batch slicing
    if batch_size > 0:
        all_anims = all_anims[batch_start : batch_start + batch_size]
    elif batch_start > 0:
        all_anims = all_anims[batch_start:]

    logger.info(
        "Processing batch: %d animations (start=%d, size=%d)",
        len(all_anims),
        batch_start,
        batch_size or len(all_anims),
    )

    stats = {
        "found": total_found,
        "batch_count": len(all_anims),
        "processed": 0,
        "examples_rendered": 0,
        "skipped_distorted": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    for anim_idx, (char_id, anim_name, fbx_path) in enumerate(all_anims):
        t0 = time.monotonic()
        logger.info(
            "[%d/%d] Processing %s/%s (%s)",
            anim_idx + 1,
            len(all_anims),
            char_id,
            anim_name,
            fbx_path.name,
        )

        try:
            # Import the withSkin FBX
            result = _import_withskin_fbx(fbx_path)
            if result is None:
                logger.error("Failed to import %s — skipping", fbx_path.name)
                stats["errors"] += 1
                continue

            scene = bpy.context.scene

            # Distortion check (on the already-imported scene)
            if skip_distorted:
                is_ok, reason = check_distortion_in_scene()
                if not is_ok:
                    logger.warning(
                        "Skipping %s/%s: distorted — %s",
                        char_id,
                        anim_name,
                        reason,
                    )
                    stats["skipped_distorted"] += 1
                    continue

            # Get animation range and sample keyframes
            frame_start, frame_end = _get_animation_range()
            keyframes = _sample_keyframes(frame_start, frame_end, num_keyframes)
            logger.info(
                "Animation range: %d-%d, sampled keyframes: %s",
                frame_start,
                frame_end,
                keyframes,
            )

            # Map bones (done once per FBX — bone mapping doesn't change per frame)
            mapping = map_bones(
                result.armature,
                result.meshes,
                char_id,
                source_dir=fbx_path.parent,
            )

            if not mapping.bone_to_region:
                logger.error(
                    "No bones mapped for %s — skipping", char_id
                )
                stats["errors"] += 1
                continue

            # Create segmentation materials (once per character)
            region_materials = create_region_materials()

            # Process each keyframe x angle
            examples_this_anim = 0
            for frame in keyframes:
                for angle_name in camera_angles:
                    angle_cfg = CAMERA_ANGLES[angle_name]

                    try:
                        rendered = _process_single_example(
                            scene=scene,
                            armature=result.armature,
                            meshes=result.meshes,
                            mapping=mapping,
                            region_materials=region_materials,
                            char_id=char_id,
                            anim_name=anim_name,
                            frame=frame,
                            angle_name=angle_name,
                            angle_cfg=angle_cfg,
                            output_dir=output_dir,
                            source_fbx=fbx_path.name,
                        )
                        if rendered:
                            examples_this_anim += 1
                    except Exception:
                        logger.exception(
                            "Error rendering %s/%s frame=%d angle=%s",
                            char_id,
                            anim_name,
                            frame,
                            angle_name,
                        )
                        stats["errors"] += 1

            stats["examples_rendered"] += examples_this_anim
            stats["processed"] += 1

            elapsed = time.monotonic() - t0
            logger.info(
                "Finished %s/%s: %d examples in %.1fs",
                char_id,
                anim_name,
                examples_this_anim,
                elapsed,
            )

        except Exception:
            logger.exception("Unexpected error processing %s", fbx_path)
            stats["errors"] += 1

        # Force garbage collection between characters
        gc.collect()

    logger.info("Processing complete. Stats: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (after Blender's ``--`` separator).

    Returns:
        Parsed arguments namespace.
    """
    # Blender passes everything after "--" as script args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render training data from Meshy withSkin animation FBX files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Volumes/TAMWoolff/data/raw/meshy_cc0_rigged"),
        help="Directory containing rigged_*/ subdirectories with withSkin FBX files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/TAMWoolff/data/preprocessed/meshy_cc0"),
        help="Output directory for generated training data.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs=2,
        metavar=("START", "SIZE"),
        default=(0, 0),
        help="Batch processing: START index and SIZE (0 = all remaining).",
    )
    parser.add_argument(
        "--skip-distorted",
        action="store_true",
        default=True,
        help="Skip animations that fail distortion detection (default: True).",
    )
    parser.add_argument(
        "--no-skip-distorted",
        action="store_false",
        dest="skip_distorted",
        help="Process all animations even if distorted.",
    )
    parser.add_argument(
        "--keyframes",
        type=int,
        default=KEYFRAMES_PER_CLIP,
        help=f"Number of keyframes to sample per animation (default: {KEYFRAMES_PER_CLIP}).",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="all",
        help="Comma-separated list of camera angle names, or 'all' (default: all).",
    )

    return parser.parse_args(argv)


def main() -> None:
    """Entry point when run as a Blender script."""
    args = parse_args()

    # Resolve camera angles
    if args.angles == "all":
        camera_angles = ALL_CAMERA_ANGLES
    else:
        camera_angles = [a.strip() for a in args.angles.split(",")]
        # Validate
        for a in camera_angles:
            if a not in CAMERA_ANGLES:
                logger.error("Unknown camera angle: %s", a)
                logger.error("Available angles: %s", ", ".join(CAMERA_ANGLES.keys()))
                sys.exit(1)

    batch_start, batch_size = args.batch

    logger.info("=" * 60)
    logger.info("withSkin Animation Rendering Pipeline")
    logger.info("=" * 60)
    logger.info("Input:       %s", args.input_dir)
    logger.info("Output:      %s", args.output_dir)
    logger.info("Keyframes:   %d per animation", args.keyframes)
    logger.info("Angles:      %d camera angles", len(camera_angles))
    logger.info("Batch:       start=%d, size=%d", batch_start, batch_size)
    logger.info("Distortion:  %s", "skip" if args.skip_distorted else "process all")
    logger.info("=" * 60)

    stats = process_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        camera_angles=camera_angles,
        num_keyframes=args.keyframes,
        skip_distorted=args.skip_distorted,
        batch_start=batch_start,
        batch_size=batch_size,
    )

    # Save stats summary
    stats_path = args.output_dir / "render_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Stats saved to %s", stats_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RENDERING COMPLETE")
    print("=" * 60)
    print(f"  Found:            {stats['found']} animation files")
    print(f"  Batch processed:  {stats['batch_count']}")
    print(f"  Successfully:     {stats['processed']}")
    print(f"  Examples:         {stats['examples_rendered']}")
    print(f"  Skipped (dist.):  {stats['skipped_distorted']}")
    print(f"  Errors:           {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# When run via `blender --python`, __name__ is "__main__"
# but Blender also evaluates the script at the module level.
# The main() call above handles both cases.
