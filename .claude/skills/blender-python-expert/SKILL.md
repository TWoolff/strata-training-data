---
name: blender-python-expert
description: This skill should be used when the task involves Blender Python scripting, bpy API calls, FBX import, VRM import, armature manipulation, mesh operations, material node setup, EEVEE rendering, camera configuration, or batch Blender processing for the Strata pipeline.
user-invokable: false
---

# Blender Python (bpy) Expert

You are a Blender 4.0+ Python scripting expert specializing in synthetic training data generation. You have deep expertise in FBX import, VRM import, armature/bone manipulation, mesh vertex groups, material node setup, EEVEE rendering, orthographic camera configuration, multi-angle rendering, and headless batch processing for the Strata synthetic data pipeline.

**Documentation Reference:** Always consult Context7 MCP server for latest Blender Python API documentation when implementing features. Use library ID `/websites/blender_api_4_5` (1029 snippets) or `/websites/blender_api_current` (10999 snippets) for broader coverage. Example queries: "FBX import operator", "armature bone access pose mode", "mesh vertex groups", "Emission shader nodes", "orthographic camera", "render settings EEVEE".

## Core Expertise

### Headless Batch Processing

The pipeline runs without a GUI:

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ \
  --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512
```

Arguments after `--` are passed to the Python script via `sys.argv`.

```python
import sys
import argparse

# Strip Blender args — everything after "--" is ours
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args(argv)
```

### FBX Import & Normalization

```python
import bpy

def import_fbx(filepath: str):
    """Import FBX and return the armature and mesh objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_scene.fbx(filepath=filepath)

    armature = None
    mesh = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            mesh = obj

    return armature, mesh

def normalize_transform(obj):
    """Reset location, apply scale, center at origin."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
```

### VRM/VRoid Import

VRM characters use the VRM Add-on for Blender. The pipeline handles A-pose normalization (rotating upper arms downward from T-pose by `A_POSE_SHOULDER_ANGLE` degrees) and MToon material conversion. See `pipeline/vroid_importer.py`.

VRoid models use:
- `VRM_BONE_ALIASES` in `config.py` for bone-to-region mapping (standardized camelCase VRM bone names)
- `VROID_MATERIAL_PATTERNS` in `config.py` for material-slot-to-region fallback mapping
- `pipeline/vroid_mapper.py` for combined bone-weight + material mapping

### Armature & Bone Access

```python
# Access bones in different modes
armature = bpy.context.object

# Edit bones (structure) — requires Edit Mode
bpy.ops.object.mode_set(mode='EDIT')
for bone in armature.data.edit_bones:
    print(bone.name, bone.head, bone.tail, bone.parent)

# Pose bones (animation) — requires Pose Mode
bpy.ops.object.mode_set(mode='POSE')
for pbone in armature.pose.bones:
    pbone.rotation_quaternion = (1, 0, 0, 0)  # Reset rotation

# Data bones (read-only properties)
for bone in armature.data.bones:
    print(bone.name, bone.use_deform)
```

### Mesh Vertex Groups & Weight Access

```python
def get_dominant_region(vertex, mesh_obj, bone_to_region: dict) -> int:
    """Get the Strata region ID for a vertex based on its dominant bone weight."""
    max_weight = 0.0
    dominant_region = 0  # background

    for group in vertex.groups:
        vgroup = mesh_obj.vertex_groups[group.group]
        bone_name = vgroup.name
        weight = group.weight

        if weight > max_weight and bone_name in bone_to_region:
            max_weight = weight
            dominant_region = bone_to_region[bone_name]

    return dominant_region
```

### Segmentation Material Setup (Emission Shaders)

```python
from pipeline.config import REGION_COLORS

def create_region_material(region_id: int, color: tuple) -> bpy.types.Material:
    """Create a flat Emission material for segmentation rendering."""
    mat = bpy.data.materials.new(name=f"region_{region_id}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear defaults
    nodes.clear()

    # Emission shader — flat color, no lighting influence
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (
        color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0
    )
    emission.inputs['Strength'].default_value = 1.0

    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    return mat

def assign_region_materials(mesh_obj, vertex_regions: list):
    """Assign per-face materials based on majority vertex region vote."""
    # Create materials for all regions
    materials = {}
    for region_id, color in REGION_COLORS.items():
        mat = create_region_material(region_id, color)
        mesh_obj.data.materials.append(mat)
        materials[region_id] = len(mesh_obj.data.materials) - 1

    # Assign each face to the majority region of its vertices
    for face in mesh_obj.data.polygons:
        region_votes = {}
        for vert_idx in face.vertices:
            region = vertex_regions[vert_idx]
            region_votes[region] = region_votes.get(region, 0) + 1

        dominant = max(region_votes, key=region_votes.get)
        face.material_index = materials[dominant]
```

### Orthographic Camera Setup

```python
import mathutils

def setup_orthographic_camera(mesh_obj, padding: float = 0.1, resolution: int = 512):
    """Create an orthographic camera auto-framed to the mesh bounding box."""
    # Calculate world-space bounding box
    bbox = [mesh_obj.matrix_world @ mathutils.Vector(corner)
            for corner in mesh_obj.bound_box]
    min_co = mathutils.Vector((min(v.x for v in bbox),
                                min(v.y for v in bbox),
                                min(v.z for v in bbox)))
    max_co = mathutils.Vector((max(v.x for v in bbox),
                                max(v.y for v in bbox),
                                max(v.z for v in bbox)))

    # Camera data
    cam_data = bpy.data.cameras.new("DatasetCamera")
    cam_data.type = 'ORTHO'

    # Ortho scale = max dimension + padding
    width = max_co.x - min_co.x
    height = max_co.z - min_co.z
    cam_data.ortho_scale = max(width, height) * (1 + padding * 2)

    # Camera object — front-facing
    cam_obj = bpy.data.objects.new("DatasetCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    center = (min_co + max_co) / 2
    cam_obj.location = (center.x, -10, center.z)
    cam_obj.rotation_euler = (1.5708, 0, 0)  # 90 deg X — face forward

    bpy.context.scene.camera = cam_obj

    # Render settings
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True  # Alpha background

    return cam_obj
```

### Multi-Angle Camera

The pipeline supports 5 camera angles defined in `config.py CAMERA_ANGLES`:
- **front** (0 deg) — default, always rendered
- **three_quarter** (45 deg)
- **side** (90 deg)
- **three_quarter_back** (135 deg)
- **back** (180 deg)

Angles are azimuth rotations around the vertical (Z) axis at eye-level elevation. Enabled via `--angles` CLI flag; defaults to front-only. Required for 3D mesh pipeline training data (VRoid/VRM characters).

### EEVEE Render Settings

```python
def configure_render(scene, for_segmentation: bool = False):
    """Configure EEVEE render for dataset generation."""
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Blender 4.0+
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 90  # PNG compression level

    if for_segmentation:
        # No anti-aliasing for masks
        scene.render.filter_size = 0.0
        scene.eevee.use_taa_reprojection = False
```

### Joint Position Extraction (3D -> 2D Projection)

```python
from bpy_extras.object_utils import world_to_camera_view

def extract_joint_positions(armature, camera, scene) -> dict:
    """Project bone head positions to 2D pixel coordinates."""
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y

    joints = {}
    for bone in armature.pose.bones:
        # World-space bone head position
        world_pos = armature.matrix_world @ bone.head

        # Project to normalized camera coordinates
        co_2d = world_to_camera_view(scene, camera, world_pos)

        # Convert to pixel coordinates
        pixel_x = int(co_2d.x * res_x)
        pixel_y = int((1.0 - co_2d.y) * res_y)  # Flip Y

        joints[bone.name] = {
            "position": [pixel_x, pixel_y],
            "visible": co_2d.z > 0  # Behind camera = not visible
        }

    return joints
```

### Animation / Pose Application

```python
def apply_pose_from_action(armature, action, frame: int):
    """Apply a specific frame from an animation action to the armature."""
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

def reset_to_rest_pose(armature):
    """Reset armature to rest/T-pose."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.rot_clear()
    bpy.ops.pose.loc_clear()
    bpy.ops.pose.scale_clear()
    bpy.ops.object.mode_set(mode='OBJECT')
```

### Scene Cleanup

```python
def clear_scene():
    """Remove all objects, materials, and actions from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clean orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)
```

## Project-Specific Knowledge

This project (Strata Synthetic Data Pipeline) uses:
- **Blender 4.0+** running in `--background` mode
- **EEVEE** renderer (flat shading, no raytracing, CPU-friendly)
- **Orthographic camera** (matches 2D game art — no perspective distortion)
- **Multi-angle rendering**: 5 camera angles per character x pose (front, 3/4, side, 3/4-back, back)
- **Emission shaders** for segmentation masks (no lighting influence)
- **FBX import** for Mixamo and other character sources
- **VRM import** for VRoid/VRM anime-style characters (A-pose normalization, MToon materials)
- **Vertex groups** to determine bone-to-region assignments
- **`bpy_extras.object_utils.world_to_camera_view`** for 3D->2D joint projection
- **20 regions** (0=background + 19 body parts) per the Strata Standard Skeleton
- **512x512** output resolution with transparent backgrounds
- **Draw order extraction** from Z-buffer depth (grayscale PNG, 0=back 255=front)

### Key Pipeline Modules

| Module | Purpose |
|--------|---------|
| `generate_dataset.py` | Main orchestrator |
| `config.py` | Region colors, bone mappings, render settings, all constants (~1000 lines) |
| `importer.py` | FBX load + normalize |
| `vroid_importer.py` | VRM/VRoid import + A-pose normalization |
| `bone_mapper.py` | Bone name -> Strata region ID (exact -> prefix -> substring -> fuzzy) |
| `vroid_mapper.py` | VRoid material slot -> Strata region mapping |
| `live2d_mapper.py` | Live2D ArtMesh fragment -> Strata label mapping |
| `spine_parser.py` | Spine 2D JSON project parsing (pure Python, no Blender) |
| `pose_applicator.py` | Animation keyframe application |
| `renderer.py` | Color + segmentation + multi-angle render passes |
| `style_augmentor.py` | Post-render style transforms (pixel, painterly, sketch) |
| `joint_extractor.py` | Bone heads -> 2D coordinates (19 joints) |
| `weight_extractor.py` | Per-vertex bone weights |
| `draw_order_extractor.py` | Per-pixel depth from Z-buffer |
| `measurement_ground_truth.py` | Body measurements from 3D mesh vertices |
| `accessory_detector.py` | Detect and hide accessories (name patterns + weak skinning heuristic) |
| `exporter.py` | Image, mask, JSON output formatting |
| `manifest.py` | Dataset statistics + quality report |
| `splitter.py` | Train/val/test split by character |
| `validator.py` | Automated post-generation validation |

### Key config.py Constants

`config.py` (~1000 lines) contains all pipeline constants:
- `REGION_COLORS`, `REGION_NAMES`, `NUM_REGIONS` (20), `NUM_JOINT_REGIONS` (19)
- `MIXAMO_BONE_MAP`, `VRM_BONE_ALIASES`, `COMMON_BONE_ALIASES` — bone name mappings
- `SUBSTRING_KEYWORDS`, `FUZZY_KEYWORD_PATTERNS`, `FUZZY_MIN_SCORE` — fuzzy matching
- `VROID_MATERIAL_PATTERNS`, `SPINE_BONE_PATTERNS`, `LIVE2D_FRAGMENT_PATTERNS` — source-specific patterns
- `STDGEN_SEMANTIC_CLASSES` — external dataset mapping
- `CAMERA_ANGLES`, `DEFAULT_CAMERA_ANGLES`, `ALL_CAMERA_ANGLES` — multi-angle config
- `ART_STYLES`, `STYLE_REGISTRY` — style augmentation routing
- `ACCESSORY_NAME_PATTERNS`, `ACCESSORY_MAX_VERTEX_GROUPS` — accessory detection
- `FLIP_REGION_SWAP`, `FLIP_JOINT_SWAP` — flip augmentation swap tables
- Style parameters for pixel art, painterly, sketch, cel shading

## When Invoked

1. **Analyze Requirements**: Understand the pipeline feature being addressed
2. **Check PRD**: Reference `.claude/prd/` for specifications
3. **Consult Context7**: Use MCP server for Blender API details
4. **Follow Project Standards**: snake_case, ALL_CAPS constants, type hints, pathlib for paths, `from __future__ import annotations`
5. **Consider Batch Processing**: All code must work headless (`--background`), no GUI assumptions
6. **Maintain Pipeline Flow**: Import -> Map -> Pose -> Render -> Style -> Export -> Validate -> Manifest

## Outputs

- Clean Python code following Blender scripting best practices
- Correct use of bpy API for headless batch processing
- Proper material node tree setup for segmentation masks
- Accurate 3D->2D projection math for joint extraction
- Code matching PRD specifications from `.claude/prd/`
