"""Main entry point: orchestrate the full synthetic data pipeline.

Processes multiple characters from an input directory, iterates all poses
per character, logs progress, handles errors gracefully, and supports
incremental processing.

Usage::

    blender --background --python run_pipeline.py -- \\
      --input_dir ./data/fbx/ \\
      --pose_dir ./data/poses/ \\
      --output_dir ./output/segmentation/ \\
      --styles flat,cel,pixel,painterly,sketch,unlit \\
      --resolution 512 \\
      --poses_per_character 20
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

from .bone_mapper import BoneMapping, map_bones
from .config import (
    ENABLE_FLIP,
    ENABLE_SCALE,
    POST_RENDER_STYLES,
    RENDER_RESOLUTION,
    RENDER_TIME_STYLES,
    SCALE_FACTORS,
)
from .exporter import (
    ensure_output_dirs,
    save_class_map,
    save_joints,
    save_source_metadata,
    save_weights,
)
from .importer import clear_scene, import_character
from .joint_extractor import extract_joints
from .pose_applicator import (
    AugmentationInfo,
    PoseInfo,
    apply_pose,
    apply_scale,
    flip_image,
    flip_joints,
    flip_mask,
    list_poses,
    reset_pose,
    restore_scale,
)
from .renderer import (
    apply_style,
    assign_region_materials,
    convert_rgb_to_grayscale_mask,
    create_region_materials,
    render_color,
    render_segmentation,
    restore_style,
    setup_camera,
    setup_color_render,
    setup_segmentation_render,
)
from .style_augmentor import apply_post_render_style
from .weight_extractor import extract_weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed after the ``--`` separator.

    Blender places its own arguments before ``--`` and forwards everything
    after it to the Python script via ``sys.argv``.
    """
    try:
        separator_idx = sys.argv.index("--")
    except ValueError:
        # No separator — no script args (use defaults)
        script_args: list[str] = []
    else:
        script_args = sys.argv[separator_idx + 1 :]

    parser = argparse.ArgumentParser(
        description="Strata synthetic data pipeline — generate labeled training data.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./data/fbx"),
        help="Directory containing .fbx source characters.",
    )
    parser.add_argument(
        "--pose_dir",
        type=Path,
        default=Path("./data/poses"),
        help="Directory containing .fbx animation files for poses.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output/segmentation"),
        help="Root output directory for the generated dataset.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default="flat",
        help="Comma-separated art style names.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help="Render resolution in pixels (square).",
    )
    parser.add_argument(
        "--enable_flip",
        action="store_true",
        default=ENABLE_FLIP,
        help="Enable Y-axis (horizontal) flip augmentation.",
    )
    parser.add_argument(
        "--scale_factors",
        type=str,
        default=",".join(str(s) for s in SCALE_FACTORS),
        help="Comma-separated scale factors for scale augmentation (e.g. '0.85,1.0,1.15').",
    )
    parser.add_argument(
        "--enable_scale",
        action="store_true",
        default=ENABLE_SCALE,
        help="Enable scale variation augmentation.",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        default=False,
        help="Skip already-processed character+pose combinations.",
    )
    parser.add_argument(
        "--max_characters",
        type=int,
        default=0,
        help="Limit number of characters to process (0 = all).",
    )
    parser.add_argument(
        "--poses_per_character",
        type=int,
        default=0,
        help="Limit number of poses per character (0 = all).",
    )

    return parser.parse_args(script_args)


# ---------------------------------------------------------------------------
# Material backup / restore
# ---------------------------------------------------------------------------


def _backup_materials(
    meshes: list[bpy.types.Object],
) -> list[list[bpy.types.Material | None]]:
    """Store each mesh's current material slot list for later restoration."""
    return [[slot.material for slot in mesh_obj.material_slots] for mesh_obj in meshes]


def _restore_materials(
    meshes: list[bpy.types.Object],
    backup: list[list[bpy.types.Material | None]],
) -> None:
    """Restore previously backed-up materials to each mesh."""
    for mesh_obj, saved_mats in zip(meshes, backup, strict=True):
        mesh_obj.data.materials.clear()
        for mat in saved_mats:
            mesh_obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def _build_augmentation_list(
    enable_flip: bool,
    enable_scale: bool,
    scale_factors: list[float],
) -> list[AugmentationInfo]:
    """Build the list of augmentation variants to render per pose.

    Always includes the identity (no augmentation). Optionally adds
    flipped, scaled, and flipped+scaled variants.

    Args:
        enable_flip: Whether to generate flipped variants.
        enable_scale: Whether to generate scaled variants.
        scale_factors: Scale factors to apply (1.0 is identity, kept once).

    Returns:
        List of AugmentationInfo describing each variant.
    """
    augmentations: list[AugmentationInfo] = [AugmentationInfo()]  # identity

    # Non-identity scale factors
    extra_scales = [s for s in scale_factors if s != 1.0]

    if enable_scale:
        for sf in extra_scales:
            augmentations.append(AugmentationInfo(scale_factor=sf))

    if enable_flip:
        augmentations.append(AugmentationInfo(flipped=True))
        if enable_scale:
            for sf in extra_scales:
                augmentations.append(AugmentationInfo(flipped=True, scale_factor=sf))

    return augmentations


# ---------------------------------------------------------------------------
# Already-processed check
# ---------------------------------------------------------------------------


def _is_already_processed(
    output_dir: Path,
    char_id: str,
    pose_index: int,
) -> bool:
    """Check if a character+pose combination has already been processed.

    Uses mask file existence as the indicator.

    Args:
        output_dir: Root dataset output directory.
        char_id: Character identifier.
        pose_index: Zero-based pose index.

    Returns:
        True if the mask file already exists.
    """
    mask_path = output_dir / "masks" / f"{char_id}_pose_{pose_index:02d}.png"
    return mask_path.exists()


# ---------------------------------------------------------------------------
# Per-character result tracking
# ---------------------------------------------------------------------------


@dataclass
class CharacterResult:
    """Tracks processing results for a single character."""

    char_id: str
    poses_succeeded: int = 0
    poses_failed: int = 0
    poses_skipped: int = 0
    elapsed: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Per-character pipeline
# ---------------------------------------------------------------------------


def process_character(
    fbx_path: Path,
    output_dir: Path,
    poses: list[PoseInfo],
    pose_dir: Path,
    styles: list[str],
    resolution: int,
    *,
    char_num: int = 1,
    total_chars: int = 1,
    enable_flip: bool = False,
    enable_scale: bool = False,
    scale_factors: list[float] | None = None,
    only_new: bool = False,
) -> CharacterResult:
    """Run the full pipeline for a single character across all poses.

    Args:
        fbx_path: Path to the .fbx file.
        output_dir: Root dataset output directory.
        poses: List of poses to apply.
        pose_dir: Directory containing animation FBX files.
        styles: List of art style names to render.
        resolution: Render resolution in pixels.
        char_num: Current character number (1-based, for progress).
        total_chars: Total number of characters (for progress).
        enable_flip: Generate horizontally flipped augmentation variants.
        enable_scale: Generate scale variation augmentation variants.
        scale_factors: Scale factors for scale augmentation.
        only_new: Skip already-processed character+pose combinations.

    Returns:
        CharacterResult with per-pose success/failure/skip counts.
    """
    if scale_factors is None:
        scale_factors = [1.0]

    t_start = time.monotonic()
    char_id = fbx_path.stem
    result = CharacterResult(char_id=char_id)
    total_poses = len(poses)

    print(f"[{char_num}/{total_chars}] Importing {char_id}...")
    import_result = import_character(fbx_path)
    if import_result is None:
        result.error = "import failed (no armature or mesh)"
        result.elapsed = time.monotonic() - t_start
        print(f"[{char_num}/{total_chars}] {char_id} FAILED — {result.error}")
        return result

    armature = import_result.armature
    meshes = import_result.meshes
    scene = bpy.context.scene

    # --- Bone mapping ---
    print(f"[{char_num}/{total_chars}] {char_id} — mapping bones...")
    mapping = map_bones(
        armature,
        meshes,
        character_id=char_id,
        source_dir=fbx_path.parent,
    )
    if mapping.unmapped_bones:
        print(
            f"[{char_num}/{total_chars}] {char_id} WARNING: "
            f"{len(mapping.unmapped_bones)} unmapped bones: {mapping.unmapped_bones}"
        )

    # --- Store original materials for color pass ---
    original_materials = _backup_materials(meshes)

    # --- Create segmentation materials (reused across poses) ---
    region_materials = create_region_materials()

    # --- Build augmentation variants ---
    augmentations = _build_augmentation_list(
        enable_flip,
        enable_scale,
        scale_factors,
    )

    # --- Iterate poses ---
    for pose_idx, pose in enumerate(poses):
        pose_num = pose_idx + 1
        progress = f"[{char_num}/{total_chars}] {char_id} — {pose.name} ({pose_num}/{total_poses})"

        # --- Skip if already processed ---
        if only_new and _is_already_processed(output_dir, char_id, pose_idx):
            print(f"{progress} SKIPPED (already exists)")
            result.poses_skipped += 1
            continue

        try:
            _process_single_pose(
                scene=scene,
                armature=armature,
                meshes=meshes,
                mapping=mapping,
                original_materials=original_materials,
                region_materials=region_materials,
                augmentations=augmentations,
                pose=pose,
                pose_idx=pose_idx,
                pose_dir=pose_dir,
                output_dir=output_dir,
                char_id=char_id,
                styles=styles,
                resolution=resolution,
            )
            result.poses_succeeded += 1
            print(f"{progress} OK")
        except Exception:
            logger.exception("Error processing %s pose %s", char_id, pose.name)
            result.poses_failed += 1
            print(f"{progress} FAILED")
            # Reset pose to clean state before next attempt
            try:
                reset_pose(armature)
            except Exception:
                logger.debug("Failed to reset pose after error", exc_info=True)

    # --- Extract weights (T-pose, once per character) ---
    try:
        print(f"[{char_num}/{total_chars}] {char_id} — extracting weights...")
        # Ensure T-pose for weight extraction
        reset_pose(armature)

        # Assign seg materials for consistent state
        for mesh_idx, mesh_obj in enumerate(meshes):
            assign_region_materials(
                mesh_obj,
                mesh_idx,
                mapping.vertex_to_region,
                region_materials,
            )

        old_cam = bpy.data.objects.get("strata_camera")
        if old_cam is not None:
            bpy.data.objects.remove(old_cam, do_unlink=True)
        weight_camera = setup_camera(scene, meshes)
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution

        weight_data = extract_weights(
            scene,
            weight_camera,
            meshes,
            mapping.bone_to_region,
        )
        weight_data["character_id"] = char_id
        save_weights(weight_data, output_dir, char_id, 0, only_new=only_new)

        bpy.data.objects.remove(weight_camera, do_unlink=True)
    except Exception:
        logger.exception("Error extracting weights for %s", char_id)

    # --- Save source metadata (once per character) ---
    print(f"[{char_num}/{total_chars}] {char_id} — saving metadata...")
    has_overrides = mapping.mapping_stats.override > 0
    save_source_metadata(
        output_dir,
        char_id,
        source=_infer_source(char_id),
        name=char_id,
        bone_mapping="manual" if has_overrides else "auto",
        unmapped_bones=mapping.unmapped_bones,
        only_new=only_new,
    )

    result.elapsed = time.monotonic() - t_start
    print(
        f"[{char_num}/{total_chars}] {char_id} DONE — "
        f"{result.poses_succeeded} OK, {result.poses_failed} failed, "
        f"{result.poses_skipped} skipped ({result.elapsed:.1f}s)"
    )
    return result


def _process_single_pose(
    *,
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    mapping: BoneMapping,
    original_materials: list[list[bpy.types.Material | None]],
    region_materials: list[bpy.types.Material],
    augmentations: list[AugmentationInfo],
    pose: PoseInfo,
    pose_idx: int,
    pose_dir: Path,
    output_dir: Path,
    char_id: str,
    styles: list[str],
    resolution: int,
) -> None:
    """Process a single pose for a character (all augmentation variants).

    Args:
        scene: The Blender scene.
        armature: The character's armature object.
        meshes: Character mesh objects.
        mapping: BoneMapping from bone_mapper.
        original_materials: Backed-up original materials.
        region_materials: Segmentation region materials.
        augmentations: List of augmentation variants.
        pose: The pose to apply.
        pose_idx: Zero-based pose index (for filenames).
        pose_dir: Directory containing animation FBX files.
        output_dir: Root dataset output directory.
        char_id: Character identifier.
        styles: Art style names.
        resolution: Render resolution.
    """
    # Apply the pose (handles built-in T-pose/A-pose and animation FBX poses)
    if not apply_pose(armature, pose, pose_dir):
        raise RuntimeError(f"Failed to apply pose {pose.name} from {pose.source}")

    # Assign segmentation materials
    for mesh_idx, mesh_obj in enumerate(meshes):
        assign_region_materials(
            mesh_obj,
            mesh_idx,
            mapping.vertex_to_region,
            region_materials,
        )

    for aug in augmentations:
        # --- Apply scale if needed ---
        if aug.scale_factor != 1.0:
            apply_scale(armature, meshes, aug.scale_factor)

        # --- Camera (recompute for each scale/pose variant) ---
        old_cam = bpy.data.objects.get("strata_camera")
        if old_cam is not None:
            bpy.data.objects.remove(old_cam, do_unlink=True)
        camera = setup_camera(scene, meshes)

        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution

        # Pose name suffix for file naming
        pose_suffix = f"_pose_{pose_idx:02d}{aug.suffix}"

        # --- Segmentation render (RGB → grayscale mask) ---
        setup_segmentation_render(scene)

        mask_out_path = output_dir / "masks" / f"{char_id}{pose_suffix}.png"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_rgb_path = Path(tmp.name)

        render_segmentation(scene, tmp_rgb_path)
        convert_rgb_to_grayscale_mask(tmp_rgb_path, mask_out_path)
        tmp_rgb_path.unlink(missing_ok=True)

        # --- Extract joints ---
        joint_data = extract_joints(
            scene,
            camera,
            armature,
            meshes,
            mapping.bone_to_region,
        )

        # --- Color render (one pass per style) ---
        setup_color_render(scene)
        color_paths: dict[str, Path] = {}
        for style in styles:
            # Restore original materials before each style application
            _restore_materials(meshes, original_materials)

            # Apply render-time style (no-op for post-render styles)
            if style in RENDER_TIME_STYLES:
                apply_style(scene, meshes, style)

            image_out_path = output_dir / "images" / f"{char_id}{pose_suffix}_{style}.png"
            render_color(scene, image_out_path)

            # Apply post-render style transform (pixel art, painterly, sketch)
            if style in POST_RENDER_STYLES:
                img = Image.open(image_out_path)
                img = apply_post_render_style(img, style)
                img.save(image_out_path, format="PNG")

            color_paths[style] = image_out_path

            # Clean up scene-level style state
            if style in RENDER_TIME_STYLES:
                restore_style(scene, style)

        # --- Re-assign segmentation materials for next variant ---
        for mesh_idx, mesh_obj in enumerate(meshes):
            assign_region_materials(
                mesh_obj,
                mesh_idx,
                mapping.vertex_to_region,
                region_materials,
            )

        # --- Save metadata (with augmentation info) ---
        joint_data["augmentation"] = aug.to_dict()
        joint_data["pose_name"] = pose.name
        joint_data["pose_source"] = pose.source
        joint_data["pose_frame"] = pose.frame
        save_joints(joint_data, output_dir, char_id, pose_idx)

        # --- Generate flip variant from the rendered outputs ---
        if aug.flipped:
            flip_mask(mask_out_path, mask_out_path)
            joint_data = flip_joints(joint_data, resolution)
            joint_data["augmentation"] = aug.to_dict()
            save_joints(joint_data, output_dir, char_id, pose_idx)

            for _style, img_path in color_paths.items():
                img = Image.open(img_path)
                flipped_img = flip_image(img)
                flipped_img.save(img_path, format="PNG")

        # --- Restore scale ---
        if aug.scale_factor != 1.0:
            restore_scale(armature, meshes)

    # Reset pose for next iteration
    reset_pose(armature)


def _infer_source(char_id: str) -> str:
    """Infer the asset source from the character ID prefix."""
    lower = char_id.lower()
    if lower.startswith("mixamo"):
        return "mixamo"
    if lower.startswith("quaternius"):
        return "quaternius"
    if lower.startswith("kenney"):
        return "kenney"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Discover FBX files, process each, and report a summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    input_dir: Path = args.input_dir
    pose_dir: Path = args.pose_dir
    output_dir: Path = args.output_dir
    styles = [s.strip() for s in args.styles.split(",")]
    resolution: int = args.resolution
    enable_flip: bool = args.enable_flip
    enable_scale: bool = args.enable_scale
    scale_factors = [float(s.strip()) for s in args.scale_factors.split(",")]
    only_new: bool = args.only_new
    max_characters: int = args.max_characters
    poses_per_character: int = args.poses_per_character

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Discover FBX files
    fbx_files = sorted(input_dir.glob("*.fbx"))
    if not fbx_files:
        print(f"ERROR: No .fbx files found in {input_dir}")
        sys.exit(1)

    # Apply --max_characters limit
    if max_characters > 0:
        fbx_files = fbx_files[:max_characters]

    # Discover poses
    print("Indexing pose library...")
    poses = list_poses(pose_dir)

    # Apply --poses_per_character limit
    if poses_per_character > 0:
        poses = poses[:poses_per_character]

    total_chars = len(fbx_files)
    total_poses = len(poses)

    print()
    print("=" * 60)
    print(f"Characters:  {total_chars} (from {input_dir})")
    print(f"Poses:       {total_poses} (from {pose_dir})")
    print(f"Styles:      {styles}")
    print(f"Resolution:  {resolution}x{resolution}")
    print(f"Flip:        {enable_flip}")
    print(f"Scale:       {enable_scale} (factors: {scale_factors})")
    print(f"Only new:    {only_new}")
    print(f"Output:      {output_dir}")
    print("=" * 60)
    print()

    # Create output directories
    ensure_output_dirs(output_dir)

    # Save class map (once, shared across all characters)
    save_class_map(output_dir)

    # Process each character
    results: list[CharacterResult] = []
    t_total = time.monotonic()

    for char_idx, fbx_path in enumerate(fbx_files):
        char_num = char_idx + 1

        try:
            char_result = process_character(
                fbx_path,
                output_dir,
                poses,
                pose_dir,
                styles,
                resolution,
                char_num=char_num,
                total_chars=total_chars,
                enable_flip=enable_flip,
                enable_scale=enable_scale,
                scale_factors=scale_factors,
                only_new=only_new,
            )
        except Exception:
            logger.exception("Unhandled error processing %s", fbx_path.name)
            char_result = CharacterResult(
                char_id=fbx_path.stem,
                error="unhandled exception",
            )

        results.append(char_result)

        # Scene cleanup between characters
        try:
            clear_scene()
        except Exception:
            logger.debug("Scene cleanup failed between characters", exc_info=True)

        # Force garbage collection to free memory
        gc.collect()

    elapsed_total = time.monotonic() - t_total

    # --- Summary ---
    _print_summary(results, elapsed_total)

    # Exit code: 1 if any character had failures
    any_failed = any(r.error or r.poses_failed > 0 for r in results)
    if any_failed:
        sys.exit(1)


def _print_summary(results: list[CharacterResult], elapsed_total: float) -> None:
    """Print a final summary table of processing results.

    Args:
        results: Per-character results.
        elapsed_total: Total wall-clock time in seconds.
    """
    total_succeeded = sum(r.poses_succeeded for r in results)
    total_failed = sum(r.poses_failed for r in results)
    total_skipped = sum(r.poses_skipped for r in results)
    chars_with_errors = sum(1 for r in results if r.error or r.poses_failed > 0)
    chars_ok = len(results) - chars_with_errors

    print()
    print("=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print()

    # Per-character table
    print(f"{'Character':<30} {'OK':>5} {'Fail':>5} {'Skip':>5} {'Time':>8}  Status")
    print("-" * 75)
    for r in results:
        if r.error:
            status = f"ERROR: {r.error}"
        elif r.poses_failed > 0:
            status = "PARTIAL"
        else:
            status = "OK"
        print(
            f"{r.char_id:<30} {r.poses_succeeded:>5} {r.poses_failed:>5} "
            f"{r.poses_skipped:>5} {r.elapsed:>7.1f}s  {status}"
        )

    print("-" * 75)
    print(
        f"{'TOTAL':<30} {total_succeeded:>5} {total_failed:>5} "
        f"{total_skipped:>5} {elapsed_total:>7.1f}s"
    )
    print()
    print(f"Characters: {chars_ok} succeeded, {chars_with_errors} with errors")
    print(f"Poses:      {total_succeeded} rendered, {total_failed} failed, {total_skipped} skipped")
    print(f"Total time: {elapsed_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
