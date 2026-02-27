"""Main entry point: orchestrate the full synthetic data pipeline.

Phase 1 scope: single character, T-pose only, flat style only.
Wires together all pipeline modules: import → bone map → materials →
camera → render (seg + color) → extract joints → export.

Usage::

    blender --background --python generate_dataset.py -- \\
      --input_dir ./source_characters/ \\
      --output_dir ./dataset/ \\
      --styles flat \\
      --resolution 512
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

from bone_mapper import map_bones
from config import ENABLE_FLIP, ENABLE_SCALE, RENDER_RESOLUTION, SCALE_FACTORS
from exporter import (
    ensure_output_dirs,
    save_class_map,
    save_joints,
    save_source_metadata,
)
from importer import import_character
from joint_extractor import extract_joints
from pose_applicator import (
    AugmentationInfo,
    apply_scale,
    flip_image,
    flip_joints,
    flip_mask,
    restore_scale,
)
from renderer import (
    assign_region_materials,
    convert_rgb_to_grayscale_mask,
    create_region_materials,
    render_color,
    render_segmentation,
    setup_camera,
    setup_color_render,
    setup_segmentation_render,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

POSE_INDEX = 0  # Phase 1: T-pose only


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
        default=Path("./source_characters"),
        help="Directory containing .fbx source characters.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./dataset"),
        help="Root output directory for the generated dataset.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default="flat",
        help="Comma-separated art style names (Phase 1: flat only).",
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

    return parser.parse_args(script_args)


# ---------------------------------------------------------------------------
# Material backup / restore
# ---------------------------------------------------------------------------


def _backup_materials(
    meshes: list[bpy.types.Object],
) -> list[list[bpy.types.Material | None]]:
    """Store each mesh's current material slot list for later restoration."""
    return [
        [slot.material for slot in mesh_obj.material_slots]
        for mesh_obj in meshes
    ]


def _restore_materials(
    meshes: list[bpy.types.Object],
    backup: list[list[bpy.types.Material | None]],
) -> None:
    """Restore previously backed-up materials to each mesh."""
    for mesh_obj, saved_mats in zip(meshes, backup):
        mesh_obj.data.materials.clear()
        for mat in saved_mats:
            mesh_obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Per-character pipeline
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


def process_character(
    fbx_path: Path,
    output_dir: Path,
    styles: list[str],
    resolution: int,
    *,
    enable_flip: bool = False,
    enable_scale: bool = False,
    scale_factors: list[float] | None = None,
) -> bool:
    """Run the full pipeline for a single character.

    Args:
        fbx_path: Path to the .fbx file.
        output_dir: Root dataset output directory.
        styles: List of art style names to render.
        resolution: Render resolution in pixels.
        enable_flip: Generate horizontally flipped augmentation variants.
        enable_scale: Generate scale variation augmentation variants.
        scale_factors: Scale factors for scale augmentation.

    Returns:
        True if processing succeeded, False otherwise.
    """
    if scale_factors is None:
        scale_factors = [1.0]

    t_start = time.monotonic()
    char_id = fbx_path.stem

    print(f"[{char_id}] Importing...")
    result = import_character(fbx_path)
    if result is None:
        print(f"[{char_id}] FAILED — import returned None (no armature or mesh)")
        return False

    armature = result.armature
    meshes = result.meshes
    scene = bpy.context.scene

    # --- Bone mapping ---
    print(f"[{char_id}] Mapping bones...")
    mapping = map_bones(
        armature,
        meshes,
        character_id=char_id,
        source_dir=fbx_path.parent,
    )
    if mapping.unmapped_bones:
        print(
            f"[{char_id}] WARNING: {len(mapping.unmapped_bones)} unmapped bones: "
            f"{mapping.unmapped_bones}"
        )

    # --- Store original materials for color pass ---
    original_materials = _backup_materials(meshes)

    # --- Assign segmentation materials ---
    print(f"[{char_id}] Assigning segmentation materials...")
    region_materials = create_region_materials()
    for mesh_idx, mesh_obj in enumerate(meshes):
        assign_region_materials(
            mesh_obj, mesh_idx, mapping.vertex_to_region, region_materials,
        )

    # --- Build augmentation variants ---
    augmentations = _build_augmentation_list(
        enable_flip, enable_scale, scale_factors,
    )
    print(f"[{char_id}] Augmentation variants: {len(augmentations)}")

    for aug in augmentations:
        aug_label = aug.suffix or " (original)"
        print(f"[{char_id}] Processing variant{aug_label}...")

        # --- Apply scale if needed ---
        if aug.scale_factor != 1.0:
            apply_scale(armature, meshes, aug.scale_factor)

        # --- Camera (recompute for each scale variant) ---
        # Remove previous camera if it exists
        old_cam = bpy.data.objects.get("strata_camera")
        if old_cam is not None:
            bpy.data.objects.remove(old_cam, do_unlink=True)
        camera = setup_camera(scene, meshes)

        # Override resolution if specified
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution

        # Pose name suffix for file naming
        pose_suffix = f"_pose_{POSE_INDEX:02d}{aug.suffix}"

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
            scene, camera, armature, meshes, mapping.bone_to_region,
        )

        # --- Restore original materials for color pass ---
        _restore_materials(meshes, original_materials)

        # --- Color render ---
        setup_color_render(scene)
        color_paths: dict[str, Path] = {}
        for style in styles:
            image_out_path = (
                output_dir / "images" / f"{char_id}{pose_suffix}_{style}.png"
            )
            render_color(scene, image_out_path)
            color_paths[style] = image_out_path

        # --- Re-assign segmentation materials for next variant ---
        for mesh_idx, mesh_obj in enumerate(meshes):
            assign_region_materials(
                mesh_obj, mesh_idx, mapping.vertex_to_region, region_materials,
            )

        # --- Save metadata (with augmentation info) ---
        joint_data["augmentation"] = aug.to_dict()
        save_joints(joint_data, output_dir, char_id, POSE_INDEX)

        # --- Generate flip variant from the rendered outputs ---
        if aug.flipped:
            print(f"[{char_id}] Applying 2D flip...")

            # Flip mask and swap L/R region IDs
            flip_mask(mask_out_path, mask_out_path)

            # Flip joint positions and swap L/R names
            joint_data = flip_joints(joint_data, resolution)
            joint_data["augmentation"] = aug.to_dict()
            save_joints(joint_data, output_dir, char_id, POSE_INDEX)

            # Flip color images
            for style, img_path in color_paths.items():
                img = Image.open(img_path)
                flipped_img = flip_image(img)
                flipped_img.save(img_path, format="PNG")

        # --- Restore scale ---
        if aug.scale_factor != 1.0:
            restore_scale(armature, meshes)

    # --- Save source metadata (once per character) ---
    print(f"[{char_id}] Saving source metadata...")
    has_overrides = mapping.mapping_stats.override > 0
    save_source_metadata(
        output_dir,
        char_id,
        source=_infer_source(char_id),
        name=char_id,
        bone_mapping="manual" if has_overrides else "auto",
        unmapped_bones=mapping.unmapped_bones,
    )

    elapsed = time.monotonic() - t_start
    print(f"[{char_id}] DONE in {elapsed:.1f}s")
    return True


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
    output_dir: Path = args.output_dir
    styles = [s.strip() for s in args.styles.split(",")]
    resolution: int = args.resolution
    enable_flip: bool = args.enable_flip
    enable_scale: bool = args.enable_scale
    scale_factors = [float(s.strip()) for s in args.scale_factors.split(",")]

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Discover FBX files
    fbx_files = sorted(input_dir.glob("*.fbx"))
    if not fbx_files:
        print(f"ERROR: No .fbx files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(fbx_files)} character(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Styles: {styles}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Flip augmentation: {enable_flip}")
    print(f"Scale augmentation: {enable_scale} (factors: {scale_factors})")
    print()

    # Create output directories
    ensure_output_dirs(output_dir)

    # Save class map (once, shared across all characters)
    save_class_map(output_dir)

    # Process each character
    succeeded = 0
    failed = 0
    t_total = time.monotonic()

    for fbx_path in fbx_files:
        try:
            ok = process_character(
                fbx_path, output_dir, styles, resolution,
                enable_flip=enable_flip,
                enable_scale=enable_scale,
                scale_factors=scale_factors,
            )
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception:
            logger.exception("Unhandled error processing %s", fbx_path.name)
            failed += 1

    elapsed_total = time.monotonic() - t_total

    # Summary
    print()
    print("=" * 60)
    print(f"Pipeline complete: {succeeded} succeeded, {failed} failed")
    print(f"Total time: {elapsed_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
