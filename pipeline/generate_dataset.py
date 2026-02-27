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
import contextlib
import gc
import logging
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image  # type: ignore[import-untyped]

from .accessory_detector import detect_accessories, hide_accessories
from .bone_mapper import BoneMapping, generate_override_template, map_bones
from .config import (
    ALL_CAMERA_ANGLES,
    CAMERA_ANGLES,
    DEFAULT_CAMERA_ANGLES,
    ENABLE_FLIP,
    ENABLE_SCALE,
    POST_RENDER_BASE_STYLE,
    RENDER_RESOLUTION,
    SCALE_FACTORS,
    STYLE_REGISTRY,
)
from .draw_order_extractor import extract_draw_order
from .exporter import (
    ensure_output_dirs,
    save_class_map,
    save_joints,
    save_measurement_profiles,
    save_measurements,
    save_source_metadata,
    save_weights,
)
from .importer import ImportResult, clear_scene, import_character
from .joint_extractor import extract_joints
from .manifest import generate_manifest
from .measurement_ground_truth import extract_mesh_measurements
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
from .splitter import generate_splits
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
    parser.add_argument(
        "--generate_overrides",
        action="store_true",
        default=False,
        help="Generate template override JSONs for characters with unmapped bones, then exit.",
    )
    parser.add_argument(
        "--spine_dir",
        type=Path,
        default=None,
        help="Directory containing Spine project files (.spine/.json + images).",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default=",".join(DEFAULT_CAMERA_ANGLES),
        help=(
            "Comma-separated camera angle names or 'all'. "
            f"Available: {', '.join(ALL_CAMERA_ANGLES)}. Default: front."
        ),
    )
    parser.add_argument(
        "--vroid_dir",
        type=Path,
        default=None,
        help="Directory containing VRM/VRoid character files (.vrm).",
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
    style_counts: Counter = field(default_factory=Counter)
    elapsed: float = 0.0
    error: str = ""
    measurements: dict | None = None


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
    camera_angles: list[str] | None = None,
    import_result: ImportResult | None = None,
) -> CharacterResult:
    """Run the full pipeline for a single character across all poses.

    Args:
        fbx_path: Path to the source file (.fbx or .vrm).
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
        camera_angles: Camera angle names to render (default: front only).
        import_result: Pre-imported character (e.g. from VRM importer).
            If None, imports from fbx_path via import_character().

    Returns:
        CharacterResult with per-pose success/failure/skip counts.
    """
    if scale_factors is None:
        scale_factors = [1.0]
    if camera_angles is None:
        camera_angles = list(DEFAULT_CAMERA_ANGLES)

    t_start = time.monotonic()
    result_char_id = fbx_path.stem if import_result is None else import_result.character_id
    result = CharacterResult(char_id=result_char_id)
    total_poses = len(poses)

    if import_result is None:
        print(f"[{char_num}/{total_chars}] Importing {result_char_id}...")
        import_result = import_character(fbx_path)
        if import_result is None:
            result.error = "import failed (no armature or mesh)"
            result.elapsed = time.monotonic() - t_start
            print(f"[{char_num}/{total_chars}] {result_char_id} FAILED — {result.error}")
            return result

    char_id = import_result.character_id
    armature = import_result.armature
    meshes = import_result.meshes
    scene = bpy.context.scene

    # --- Accessory detection and hiding ---
    accessory_result = detect_accessories(meshes)
    if accessory_result.has_accessories:
        body_meshes = hide_accessories(meshes, accessory_result)
        print(
            f"[{char_num}/{total_chars}] {char_id} — "
            f"hiding {len(accessory_result.accessories)} accessories: "
            f"{accessory_result.accessory_names}"
        )
    else:
        body_meshes = meshes

    if not body_meshes:
        result.error = "all meshes detected as accessories — no body meshes"
        result.elapsed = time.monotonic() - t_start
        print(f"[{char_num}/{total_chars}] {char_id} FAILED — {result.error}")
        return result

    # --- Bone mapping ---
    print(f"[{char_num}/{total_chars}] {char_id} — mapping bones...")
    mapping = map_bones(
        armature,
        body_meshes,
        character_id=char_id,
        source_dir=fbx_path.parent,
    )
    if mapping.unmapped_bones:
        print(
            f"[{char_num}/{total_chars}] {char_id} WARNING: "
            f"{len(mapping.unmapped_bones)} unmapped bones: {mapping.unmapped_bones}"
        )

    # --- Store original materials for color pass ---
    original_materials = _backup_materials(body_meshes)

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
            pose_style_counts = _process_single_pose(
                scene=scene,
                armature=armature,
                meshes=body_meshes,
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
                camera_angles=camera_angles,
            )
            result.poses_succeeded += 1
            result.style_counts += pose_style_counts
            print(f"{progress} OK ({sum(pose_style_counts.values())} images)")
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
        for mesh_idx, mesh_obj in enumerate(body_meshes):
            assign_region_materials(
                mesh_obj,
                mesh_idx,
                mapping.vertex_to_region,
                region_materials,
            )

        old_cam = bpy.data.objects.get("strata_camera")
        if old_cam is not None:
            bpy.data.objects.remove(old_cam, do_unlink=True)
        weight_camera = setup_camera(scene, body_meshes)
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution

        weight_data = extract_weights(
            scene,
            weight_camera,
            body_meshes,
            mapping.bone_to_region,
        )
        weight_data["character_id"] = char_id
        save_weights(weight_data, output_dir, char_id, 0, only_new=only_new)

        bpy.data.objects.remove(weight_camera, do_unlink=True)
    except Exception:
        logger.exception("Error extracting weights for %s", char_id)

    # --- Extract measurements (T-pose, once per character) ---
    try:
        print(f"[{char_num}/{total_chars}] {char_id} — extracting measurements...")
        measurement_data = extract_mesh_measurements(body_meshes, mapping.bone_to_region)
        measurement_data["character_id"] = char_id
        save_measurements(measurement_data, output_dir, char_id, only_new=only_new)
        result.measurements = measurement_data
    except Exception:
        logger.exception("Error extracting measurements for %s", char_id)

    # --- Save source metadata (once per character) ---
    print(f"[{char_num}/{total_chars}] {char_id} — saving metadata...")
    has_overrides = mapping.mapping_stats.override > 0
    accessory_metadata = accessory_result.to_metadata()
    save_source_metadata(
        output_dir,
        char_id,
        source=_infer_source(char_id),
        name=char_id,
        bone_mapping="manual" if has_overrides else "auto",
        unmapped_bones=mapping.unmapped_bones,
        has_accessories=accessory_metadata["has_accessories"],
        accessories=accessory_metadata["accessories"],
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
    camera_angles: list[str] | None = None,
) -> Counter:
    """Process a single pose for a character (all augmentation and angle variants).

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
        camera_angles: Camera angle names to render.

    Returns:
        Counter of style images produced (style_name → count).
    """
    if camera_angles is None:
        camera_angles = list(DEFAULT_CAMERA_ANGLES)

    style_counts: Counter = Counter()

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

    # Partition styles by type for the base-image optimization
    render_styles = [s for s in styles if STYLE_REGISTRY.get(s) == "render"]
    post_styles = [s for s in styles if STYLE_REGISTRY.get(s) == "post"]

    for aug in augmentations:
        # --- Apply scale if needed ---
        if aug.scale_factor != 1.0:
            apply_scale(armature, meshes, aug.scale_factor)

        # --- Camera angle loop ---
        for angle_name in camera_angles:
            angle_cfg = CAMERA_ANGLES[angle_name]
            azimuth = float(angle_cfg["azimuth"])

            # --- Camera (recompute for each scale/pose/angle variant) ---
            old_cam = bpy.data.objects.get("strata_camera")
            if old_cam is not None:
                bpy.data.objects.remove(old_cam, do_unlink=True)
            camera = setup_camera(scene, meshes, azimuth=azimuth)

            scene.render.resolution_x = resolution
            scene.render.resolution_y = resolution

            # File naming: angle infix is empty for "front" (backward compat)
            angle_infix = "" if angle_name == "front" else f"_{angle_name}"
            pose_suffix = f"_pose_{pose_idx:02d}{aug.suffix}{angle_infix}"

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

            # --- Extract draw order (per-pose per-angle) ---
            seg_mask_arr = np.array(Image.open(mask_out_path).convert("L"))
            draw_order_data = extract_draw_order(
                scene,
                camera,
                armature,
                meshes,
                mapping.bone_to_region,
                seg_mask_arr,
            )
            draw_order_out_path = output_dir / "draw_order" / f"{char_id}{pose_suffix}.png"
            Image.fromarray(draw_order_data["draw_order_map"].astype(np.uint8), mode="L").save(
                draw_order_out_path, format="PNG", compress_level=9
            )

            # --- Color render ---
            setup_color_render(scene)
            color_paths: dict[str, Path] = {}
            style_seed = hash((char_id, pose_idx, angle_name)) & 0xFFFFFFFF

            # --- Render-time styles (flat, cel, unlit) ---
            base_image: Image.Image | None = None
            for style in render_styles:
                print(f"    rendering {style} ({angle_name})...")
                _restore_materials(meshes, original_materials)
                apply_style(scene, meshes, style)

                image_out_path = (
                    output_dir / "images" / f"{char_id}{pose_suffix}_{style}.png"
                )
                render_color(scene, image_out_path)
                color_paths[style] = image_out_path
                style_counts[style] += 1

                # Cache the flat render as the base for post-render styles
                if style == POST_RENDER_BASE_STYLE and post_styles:
                    base_image = Image.open(image_out_path).copy()

                restore_style(scene, style)

            # --- Post-render styles ---
            if post_styles:
                if base_image is None:
                    print(
                        f"    rendering {POST_RENDER_BASE_STYLE} "
                        f"(base for post-render, {angle_name})..."
                    )
                    _restore_materials(meshes, original_materials)
                    apply_style(scene, meshes, POST_RENDER_BASE_STYLE)

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_base_path = Path(tmp.name)
                    render_color(scene, tmp_base_path)
                    base_image = Image.open(tmp_base_path).copy()
                    tmp_base_path.unlink(missing_ok=True)

                    restore_style(scene, POST_RENDER_BASE_STYLE)

                for style in post_styles:
                    print(f"    applying {style} post-render transform ({angle_name})...")
                    image_out_path = (
                        output_dir / "images" / f"{char_id}{pose_suffix}_{style}.png"
                    )
                    styled_image = apply_post_render_style(
                        base_image.copy(), style, seed=style_seed
                    )
                    styled_image.save(image_out_path, format="PNG")
                    color_paths[style] = image_out_path
                    style_counts[style] += 1

            # --- Re-assign segmentation materials for next angle ---
            for mesh_idx, mesh_obj in enumerate(meshes):
                assign_region_materials(
                    mesh_obj,
                    mesh_idx,
                    mapping.vertex_to_region,
                    region_materials,
                )

            # --- Save metadata (with augmentation and camera angle info) ---
            joint_data["augmentation"] = aug.to_dict()
            joint_data["pose_name"] = pose.name
            joint_data["pose_source"] = pose.source
            joint_data["pose_frame"] = pose.frame
            joint_data["camera_angle"] = angle_name
            joint_data["camera_azimuth"] = azimuth
            joint_data["character_id"] = char_id
            joint_data["pose_id"] = pose_idx
            save_joints(joint_data, output_dir, char_id, pose_idx)

            # --- Generate flip variant from the rendered outputs ---
            if aug.flipped:
                flip_mask(mask_out_path, mask_out_path)
                joint_data = flip_joints(joint_data, resolution)
                joint_data["augmentation"] = aug.to_dict()
                save_joints(joint_data, output_dir, char_id, pose_idx)

                # Flip draw order map (horizontal mirror)
                do_img = Image.open(draw_order_out_path)
                do_img.transpose(Image.FLIP_LEFT_RIGHT).save(
                    draw_order_out_path, format="PNG"
                )

                for _style, img_path in color_paths.items():
                    img = Image.open(img_path)
                    flipped_img = flip_image(img)
                    flipped_img.save(img_path, format="PNG")

        # --- Restore scale ---
        if aug.scale_factor != 1.0:
            restore_scale(armature, meshes)

    # Reset pose for next iteration
    reset_pose(armature)

    return style_counts


def _infer_source(char_id: str) -> str:
    """Infer the asset source from the character ID prefix."""
    lower = char_id.lower()
    if lower.startswith("mixamo"):
        return "mixamo"
    if lower.startswith("quaternius"):
        return "quaternius"
    if lower.startswith("kenney"):
        return "kenney"
    if lower.startswith("spine"):
        return "spine"
    if lower.startswith("vroid"):
        return "vroid"
    return "unknown"


# ---------------------------------------------------------------------------
# Override template generation
# ---------------------------------------------------------------------------


def _generate_override_templates(fbx_files: list[Path]) -> None:
    """Import each character, run bone mapping, and generate override templates.

    For characters with unmapped bones that don't already have an override file,
    writes a template JSON with bone names mapped to null.

    Args:
        fbx_files: List of FBX file paths to process.
    """
    total = len(fbx_files)
    templates_created = 0

    for idx, fbx_path in enumerate(fbx_files, 1):
        char_id = fbx_path.stem
        print(f"[{idx}/{total}] Scanning {char_id}...")

        try:
            import_result = import_character(fbx_path)
            if import_result is None:
                print(f"[{idx}/{total}] {char_id} — import failed, skipping")
                continue

            mapping = map_bones(
                import_result.armature,
                import_result.meshes,
                character_id=char_id,
                source_dir=fbx_path.parent,
            )

            if mapping.unmapped_bones:
                result = generate_override_template(
                    char_id,
                    mapping.unmapped_bones,
                    fbx_path.parent,
                )
                if result is not None:
                    templates_created += 1
                    print(
                        f"[{idx}/{total}] {char_id} — generated template "
                        f"({len(mapping.unmapped_bones)} unmapped bones)"
                    )
                else:
                    print(f"[{idx}/{total}] {char_id} — override file already exists")
            else:
                print(f"[{idx}/{total}] {char_id} — all bones mapped, no template needed")

        except Exception:
            logger.exception("Error scanning %s", char_id)
            print(f"[{idx}/{total}] {char_id} — ERROR")
        finally:
            with contextlib.suppress(Exception):
                clear_scene()
            gc.collect()

    print(f"\nDone — {templates_created} template(s) created out of {total} characters.")


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
    do_generate_overrides: bool = args.generate_overrides
    spine_dir: Path | None = args.spine_dir
    vroid_dir: Path | None = args.vroid_dir

    # Parse camera angles
    angles_raw = args.angles.strip()
    if angles_raw.lower() == "all":
        camera_angles = list(ALL_CAMERA_ANGLES)
    else:
        camera_angles = [a.strip() for a in angles_raw.split(",")]
        for a in camera_angles:
            if a not in CAMERA_ANGLES:
                print(f"ERROR: Unknown camera angle '{a}'. Available: {ALL_CAMERA_ANGLES}")
                sys.exit(1)

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

    # --- Generate override templates and exit ---
    if do_generate_overrides:
        _generate_override_templates(fbx_files)
        return

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
    print(f"Angles:      {camera_angles}")
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
                camera_angles=camera_angles,
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

    # --- Process Spine projects (if --spine_dir provided) ---
    if spine_dir is not None and spine_dir.is_dir():
        from .spine_parser import process_spine_directory

        print()
        print("=" * 60)
        print(f"Processing Spine projects from {spine_dir}...")
        print("=" * 60)

        spine_results = process_spine_directory(
            spine_dir,
            output_dir,
            resolution=resolution,
            styles=styles,
            only_new=only_new,
        )

        # Track Spine characters as CharacterResult entries for the summary
        for sr in spine_results:
            results.append(CharacterResult(
                char_id=sr.char_id,
                poses_succeeded=1,  # default pose
                style_counts=Counter({s: 1 for s in styles}),
            ))

        print(f"Spine: {len(spine_results)} characters processed")

    # --- Process VRoid/VRM characters (if --vroid_dir provided) ---
    if vroid_dir is not None and vroid_dir.is_dir():
        from .vroid_importer import import_vrm

        vrm_files = sorted(vroid_dir.glob("*.vrm"))
        if vrm_files:
            if max_characters > 0:
                vrm_files = vrm_files[:max_characters]

            print()
            print("=" * 60)
            print(f"Processing VRoid characters from {vroid_dir}...")
            print(f"VRoid characters: {len(vrm_files)}")
            print("=" * 60)

            for vrm_idx, vrm_path in enumerate(vrm_files):
                vrm_num = vrm_idx + 1 + total_chars

                try:
                    vrm_import = import_vrm(vrm_path)
                    if vrm_import is None:
                        results.append(CharacterResult(
                            char_id=f"vroid_{vrm_path.stem}",
                            error="VRM import failed",
                        ))
                        continue

                    char_result = process_character(
                        vrm_path,
                        output_dir,
                        poses,
                        pose_dir,
                        styles,
                        resolution,
                        char_num=vrm_num,
                        total_chars=total_chars + len(vrm_files),
                        enable_flip=enable_flip,
                        enable_scale=enable_scale,
                        scale_factors=scale_factors,
                        only_new=only_new,
                        camera_angles=camera_angles,
                        import_result=vrm_import,
                    )
                    results.append(char_result)

                except Exception:
                    logger.exception("Unhandled error processing VRM %s", vrm_path.name)
                    results.append(CharacterResult(
                        char_id=f"vroid_{vrm_path.stem}",
                        error="unhandled exception",
                    ))

                # Scene cleanup between characters
                try:
                    clear_scene()
                except Exception:
                    logger.debug("Scene cleanup failed between VRM characters", exc_info=True)

                gc.collect()

            print(f"VRoid: {len(vrm_files)} characters processed")
        else:
            print(f"WARNING: No .vrm files found in {vroid_dir}")

    elapsed_total = time.monotonic() - t_total

    # --- Aggregate measurement profiles ---
    measurement_profiles = {
        r.char_id: r.measurements
        for r in results
        if r.measurements is not None
    }
    if measurement_profiles:
        profiles_path = save_measurement_profiles(
            measurement_profiles, output_dir, only_new=only_new,
        )
        if profiles_path:
            print(f"Measurement profiles written to {profiles_path}")

    # --- Summary ---
    _print_summary(results, elapsed_total)

    # --- Generate manifest ---
    manifest_path = generate_manifest(
        output_dir,
        results=results,
        styles=styles,
        resolution=resolution,
        poses_per_character=poses_per_character,
    )
    print(f"\nManifest written to {manifest_path}")

    # --- Generate splits ---
    splits_path = generate_splits(output_dir)
    print(f"Splits written to {splits_path}")

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

    # Aggregate style distribution across all characters
    total_style_counts: Counter = Counter()
    for r in results:
        total_style_counts += r.style_counts
    total_images = sum(total_style_counts.values())

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
    print(f"Images:     {total_images} total")

    # Style distribution
    if total_style_counts:
        print()
        print("Style distribution:")
        for style in sorted(total_style_counts):
            count = total_style_counts[style]
            print(f"  {style:<12} {count:>6} images")

    print(f"\nTotal time: {elapsed_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
