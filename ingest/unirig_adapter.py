"""Convert UniRig / Rig-XL dataset to Strata training format.

Dataset: VAST-AI-Research/UniRig (SIGGRAPH 2025)
Source:  https://github.com/VAST-AI-Research/UniRig
HF data: https://huggingface.co/VAST-AI/UniRig/tree/main/data/rigxl

The dataset provides:
- 14,000+ rigged 3D meshes as NPZ files (raw_data.npz per model)
- Per-vertex bone weights (skin matrix N×J)
- Skeleton: joint positions, parent indices, bone names

This adapter:
1. Scans the extracted processed/ directory for raw_data.npz files.
2. Loads each NPZ in Blender, reconstructs the mesh + skinning.
3. Filters non-humanoids (requires head, hips, symmetric limbs).
4. Assigns Strata region materials by vertex weight majority vote
   (same approach as the main FBX pipeline).
5. Renders front + back orthographic views of each character.
6. Saves image.png, segmentation.png, metadata.json per view.

Requires Blender 4.0+ (uses bpy). Run via:

    blender --background --python run_pipeline.py -- --adapter unirig ...

Or via run_ingest.py (see _run_unirig in run_ingest.py).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIRIG_SOURCE = "unirig"
STRATA_RESOLUTION = 512


def _eevee_engine() -> str:
    """Return the correct EEVEE engine enum for this Blender version.

    Blender 4.2–4.4 used ``BLENDER_EEVEE_NEXT``; Blender 5.0+ renamed it
    back to ``BLENDER_EEVEE``.
    """
    import bpy

    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
        return "BLENDER_EEVEE_NEXT"
    except TypeError:
        return "BLENDER_EEVEE"

# Humanoid filter: minimum fraction of vertices assigned to a known region.
# Non-humanoids (quadrupeds, objects) typically fail this threshold.
MIN_HUMANOID_COVERAGE = 0.60

# Camera angles to render (azimuth in degrees, 0=front, 180=back).
RENDER_ANGLES = [0, 180]

# Annotations not available from UniRig.
_MISSING_ANNOTATIONS = ["joints", "draw_order"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConversionStats:
    """Aggregate stats for a conversion run."""

    total: int = 0
    converted: int = 0
    skipped: int = 0
    errors: int = 0

    def summary(self) -> str:
        return (
            f"{self.converted}/{self.total} converted, "
            f"{self.skipped} skipped, {self.errors} errors"
        )


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------


def load_npz(npz_path: Path) -> dict | None:
    """Load a UniRig raw_data.npz file.

    Args:
        npz_path: Path to raw_data.npz.

    Returns:
        Dict with keys vertices, faces, names, parents, skin, joints, or None
        on failure.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        fields = {k: data[k][()] for k in data}
        # Required fields
        for key in ("vertices", "faces", "names", "skin"):
            if key not in fields:
                logger.debug("Missing key %r in %s", key, npz_path)
                return None
        return fields
    except Exception as exc:
        logger.warning("Failed to load %s: %s", npz_path, exc)
        return None


# ---------------------------------------------------------------------------
# Humanoid detection
# ---------------------------------------------------------------------------


def _is_humanoid(fields: dict, bone_to_region: dict[int, int]) -> bool:
    """Check whether a character is a humanoid biped.

    Uses the mapped bone regions to determine coverage. A character is
    humanoid if enough of its vertices are assigned to known Strata regions
    AND it has the required bilateral symmetry (arms + legs).

    Args:
        fields: Loaded NPZ fields dict.
        bone_to_region: Mapping from bone index → Strata region ID.

    Returns:
        True if the character passes humanoid checks.
    """
    skin = fields["skin"].astype(np.float32)  # (N, J)
    n_vertices = skin.shape[0]
    if n_vertices == 0:
        return False

    # Assign each vertex to its dominant bone
    dominant_bone = np.argmax(skin, axis=1)  # (N,)

    # Count vertices with a mapped region
    mapped = sum(
        1 for b in dominant_bone if bone_to_region.get(int(b)) is not None
    )
    coverage = mapped / n_vertices

    if coverage < MIN_HUMANOID_COVERAGE:
        logger.debug("Coverage %.2f < %.2f — skipping", coverage, MIN_HUMANOID_COVERAGE)
        return False

    # Require bilateral symmetry: at least one bone mapped to each arm and leg
    mapped_regions = {bone_to_region[int(b)] for b in dominant_bone if int(b) in bone_to_region}
    # IDs: upper_arm_l=7, upper_arm_r=11, upper_leg_l=14, upper_leg_r=17
    has_arms = 7 in mapped_regions and 11 in mapped_regions
    has_legs = 14 in mapped_regions and 17 in mapped_regions

    return has_arms and has_legs


# ---------------------------------------------------------------------------
# Blender mesh import and rendering
# ---------------------------------------------------------------------------


def _import_npz_to_blender(fields: dict, name: str) -> tuple | None:
    """Create a Blender mesh object from NPZ data.

    Args:
        fields: Loaded NPZ fields.
        name: Name for the Blender object.

    Returns:
        (mesh_obj, armature_obj) tuple or None on failure.
    """
    try:
        import contextlib

        import bmesh
        import bpy

        vertices = fields["vertices"].astype(np.float32)
        faces = fields["faces"].astype(np.int32)

        # Create mesh
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)

        # Build geometry via bmesh
        bm = bmesh.new()
        for v in vertices:
            bm.verts.new(v.tolist())
        bm.verts.ensure_lookup_table()
        for f in faces:
            with contextlib.suppress(Exception):
                bm.faces.new([bm.verts[int(i)] for i in f])
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        return obj, None
    except Exception as exc:
        logger.warning("Failed to import NPZ to Blender for %s: %s", name, exc)
        return None


def _assign_region_vertex_colors(
    obj,
    fields: dict,
    bone_to_region: dict[int, int],
    region_colors: dict[int, tuple],
) -> None:
    """Assign per-vertex region colors by dominant bone weight.

    Creates a vertex color layer 'strata_region' with the region color
    for each vertex's dominant bone.

    Args:
        obj: Blender mesh object.
        fields: Loaded NPZ fields.
        bone_to_region: Bone index → Strata region ID.
        region_colors: Region ID → (R, G, B) tuple (0–255 ints).
    """
    mesh = obj.data
    skin = fields["skin"].astype(np.float32)  # (N, J)
    dominant_bone = np.argmax(skin, axis=1)  # (N,)

    # Map vertex index → region color (linear float)
    def _to_linear(c: int) -> float:
        return (c / 255.0) ** 2.2  # sRGB → linear approx

    vertex_colors: list[tuple[float, float, float, float]] = []
    for bone_idx in dominant_bone:
        region_id = bone_to_region.get(int(bone_idx), 0)
        rgb = region_colors.get(region_id, (0, 0, 0))
        vertex_colors.append((
            _to_linear(rgb[0]),
            _to_linear(rgb[1]),
            _to_linear(rgb[2]),
            1.0,
        ))

    # Apply per-loop (Blender vertex colors are per-loop, not per-vertex)
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="strata_region")
    vcol_layer = mesh.vertex_colors["strata_region"]

    for loop in mesh.loops:
        vcol_layer.data[loop.index].color = vertex_colors[loop.vertex_index]


def _setup_orthographic_camera(obj, azimuth_deg: float = 0.0) -> None:
    """Position an orthographic camera to frame the object.

    Args:
        obj: Object to frame.
        azimuth_deg: Camera azimuth angle in degrees (0=front, 180=back).
    """
    import bpy
    import mathutils

    # Compute bounding box
    bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]

    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    height = max(zs) - min(zs)
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)

    ortho_scale = max(height, width, depth) * 1.1

    # Camera at azimuth_deg, looking at center
    az_rad = math.radians(azimuth_deg)
    cam_dist = max(height, width, depth) * 3
    cam_x = cx + cam_dist * math.sin(az_rad)
    cam_y = cy - cam_dist * math.cos(az_rad)
    cam_z = cz

    cam_data = bpy.data.cameras.new("cam_unirig")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_obj = bpy.data.objects.new("cam_unirig", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = (cam_x, cam_y, cam_z)

    # Point camera at center
    direction = mathutils.Vector((cx - cam_x, cy - cam_y, cz - cam_z))
    rot = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot.to_euler()

    bpy.context.scene.camera = cam_obj


def _render_segmentation(output_path: Path, resolution: int) -> bool:
    """Render a flat-shaded segmentation pass using vertex colors.

    Args:
        output_path: Path to save the PNG.
        resolution: Output resolution in pixels.

    Returns:
        True on success.
    """
    try:
        import bpy

        scene = bpy.context.scene
        scene.render.engine = _eevee_engine()
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

        # Use vertex color output — set up a simple material
        # that reads from the 'strata_region' vertex color layer
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            mat = bpy.data.materials.new("unirig_seg_mat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            vcol = nodes.new("ShaderNodeVertexColor")
            vcol.layer_name = "strata_region"
            emission = nodes.new("ShaderNodeEmission")
            output = nodes.new("ShaderNodeOutputMaterial")
            links.new(vcol.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            obj.data.materials.clear()
            obj.data.materials.append(mat)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        return True
    except Exception as exc:
        logger.warning("Render failed: %s", exc)
        return False


def _cleanup_blender_objects(names: list[str]) -> None:
    """Remove temporary Blender objects by name."""
    try:
        import bpy

        for name in names:
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
            if name in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes[name])
            if name in bpy.data.cameras:
                bpy.data.cameras.remove(bpy.data.cameras[name])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-example conversion
# ---------------------------------------------------------------------------


def _convert_one(
    npz_path: Path,
    output_dir: Path,
    region_colors: dict[int, tuple],
    only_new: bool = False,
) -> str:
    """Convert one raw_data.npz to Strata format.

    Args:
        npz_path: Path to raw_data.npz.
        output_dir: Root output directory.
        region_colors: Region ID → (R,G,B) color tuple.
        only_new: Skip if output already exists.

    Returns:
        "converted", "skipped", or "error".
    """
    from ingest.unirig_skeleton_mapper import map_skeleton

    character_id = npz_path.parent.name  # e.g. "00000"
    example_base = output_dir / character_id

    if only_new and (example_base / "front").is_dir():
        return "skipped"

    fields = load_npz(npz_path)
    if fields is None:
        return "error"

    names = [str(n) for n in fields["names"]]
    n_joints = len(names)

    # Map bones to Strata regions
    mapping = map_skeleton(character_id, names)
    if not mapping.validation.has_limbs:
        logger.debug("Non-humanoid %s — skipping", character_id)
        return "skipped"

    # Build bone_index → region_id lookup
    bone_to_region: dict[int, int] = {}
    for i, jm in enumerate(mapping.joint_mappings):
        if jm.region_id is not None:
            bone_to_region[i] = jm.region_id

    if not _is_humanoid(fields, bone_to_region):
        return "skipped"

    # Render each camera angle
    any_success = False
    for azimuth in RENDER_ANGLES:
        angle_name = "front" if azimuth == 0 else "back"
        view_dir = example_base / angle_name
        if only_new and view_dir.is_dir():
            any_success = True
            continue

        try:
            import bpy

            # Clear default scene objects
            bpy.ops.wm.read_homefile(use_empty=True)

            # Import mesh
            result = _import_npz_to_blender(fields, character_id)
            if result is None:
                continue
            obj, _ = result

            # Assign vertex region colors
            _assign_region_vertex_colors(obj, fields, bone_to_region, region_colors)

            # Set up camera
            _setup_orthographic_camera(obj, azimuth_deg=float(azimuth))

            # Render segmentation
            seg_path = view_dir / "segmentation.png"
            if not _render_segmentation(seg_path, STRATA_RESOLUTION):
                continue

            # For the color image, render with a simple diffuse material
            # (UniRig has no texture, so use a neutral gray)
            _apply_neutral_material(obj)
            color_path = view_dir / "image.png"
            _render_color(color_path, STRATA_RESOLUTION)

            # Write metadata
            char_names = sorted({
                jm.region_name
                for jm in mapping.joint_mappings
                if jm.region_name
            })
            meta: dict[str, Any] = {
                "example_id": f"{character_id}_{angle_name}",
                "source": UNIRIG_SOURCE,
                "character_id": character_id,
                "camera_angle": azimuth,
                "angle_name": angle_name,
                "n_vertices": int(fields["vertices"].shape[0]),
                "n_faces": int(fields["faces"].shape[0]),
                "n_joints": n_joints,
                "auto_match_rate": round(mapping.auto_match_rate, 4),
                "mapped_regions": char_names,
                "missing_annotations": _MISSING_ANNOTATIONS,
                "license": "research",
            }
            view_dir.mkdir(parents=True, exist_ok=True)
            with open(view_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            any_success = True

        except Exception as exc:
            logger.warning("Error rendering %s angle=%d: %s", character_id, azimuth, exc)

    return "converted" if any_success else "error"


def _apply_neutral_material(obj) -> None:
    """Apply a neutral gray diffuse material for color renders."""
    try:
        import bpy

        mat = bpy.data.materials.new("unirig_color_mat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        bsdf = nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        obj.data.materials.clear()
        obj.data.materials.append(mat)
    except Exception:
        pass


def _render_color(output_path: Path, resolution: int) -> bool:
    """Render a color pass with a simple sun + ambient setup."""
    try:
        import bpy

        scene = bpy.context.scene
        scene.render.engine = _eevee_engine()
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

        # Add a simple sun light
        sun_data = bpy.data.lights.new("sun", type="SUN")
        sun_data.energy = 2.0
        sun_obj = bpy.data.objects.new("sun", sun_data)
        bpy.context.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        bpy.data.objects.remove(sun_obj, do_unlink=True)
        bpy.data.lights.remove(sun_data)
        return True
    except Exception as exc:
        logger.warning("Color render failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def convert_directory(
    dataset_dir: Path,
    output_dir: Path,
    *,
    max_examples: int | None = None,
    only_new: bool = False,
) -> ConversionStats:
    """Convert all NPZ files in the UniRig processed/ directory.

    Expects the layout from extracting processed.7z:
        dataset_dir/
        └── processed/
            ├── 00000/
            │   └── raw_data.npz
            ├── 00001/
            │   └── raw_data.npz
            └── ...

    Or flat layout:
        dataset_dir/
        ├── 00000/
        │   └── raw_data.npz
        └── ...

    Args:
        dataset_dir: Root of the extracted dataset (contains processed/ or
            direct model subdirs).
        output_dir: Root output directory. Examples go in
            output_dir/{character_id}/{angle_name}/.
        max_examples: If set, stop after this many models.
        only_new: Skip already-converted examples.

    Returns:
        ConversionStats.
    """
    from pipeline.config import REGION_COLORS

    # Find all raw_data.npz files
    npz_files = sorted(dataset_dir.rglob("raw_data.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No raw_data.npz files found under {dataset_dir}")

    logger.info("Found %d NPZ files in %s", len(npz_files), dataset_dir)

    if max_examples is not None:
        npz_files = npz_files[:max_examples]

    # Convert REGION_COLORS from Blender float tuples to 0-255 int tuples
    region_colors_int: dict[int, tuple] = {}
    for region_id, color in REGION_COLORS.items():
        if isinstance(color[0], float):
            region_colors_int[region_id] = tuple(int(c * 255) for c in color[:3])
        else:
            region_colors_int[region_id] = tuple(color[:3])

    stats = ConversionStats(total=len(npz_files))

    for i, npz_path in enumerate(npz_files):
        try:
            result = _convert_one(npz_path, output_dir, region_colors_int, only_new=only_new)
            if result == "converted":
                stats.converted += 1
            elif result == "skipped":
                stats.skipped += 1
            else:
                stats.errors += 1
        except Exception:
            logger.exception("Unhandled error for %s", npz_path)
            stats.errors += 1

        if (i + 1) % 100 == 0:
            logger.info(
                "[unirig] %d/%d — %s",
                i + 1,
                stats.total,
                stats.summary(),
            )

    logger.info("[unirig] Done: %s", stats.summary())
    return stats
