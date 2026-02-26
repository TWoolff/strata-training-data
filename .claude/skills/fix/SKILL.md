---
name: fix
description: Fix ALL found issues and recommendations from code reviews, linting, or quality checks. Use when user says "/fix", "fix the issues", or after receiving review feedback.
user-invokable: true
---

# Fix Issues and Recommendations

Systematically fix all identified issues and recommendations from code reviews, linting, or other quality assessment tools.

## When to Use

- After receiving a code review with issues to fix
- When linting reveals errors (ruff, mypy, pyright)
- After a PR review identifies problems
- When Blender script execution reveals errors
- To clean up technical debt identified in reviews

## Fix Process

### 1. Identify Issues
Review the context for:
- Python errors from running the pipeline
- Lint errors from `ruff` or type errors from `mypy`/`pyright`
- Code review feedback
- Blender script runtime errors (bpy API misuse, missing objects, mode errors)
- Dataset validation failures (mask mismatches, missing files, wrong dimensions)

### 2. Apply Fixes
For each issue:
- Understand the root cause
- Apply minimal, targeted fix
- Follow project patterns (see CLAUDE.md)

### 3. Verify Fixes
```bash
# Run linter
ruff check .

# Run type checker (if configured)
mypy *.py

# Run pipeline on a single test character
blender --background --python generate_dataset.py -- \
  --input_dir ./source_characters/ \
  --output_dir ./dataset/ \
  --styles flat \
  --resolution 512
```

### 4. Validate Output
- Check that output images are 512×512 with correct format
- Verify segmentation masks are single-channel grayscale with valid region IDs (0–17)
- Confirm joint JSON has all 17 joints with positions within image bounds
- Overlay mask on image to verify alignment

## Pipeline-Specific Fixes

### Common Issues

**Blender Mode Errors:**
Many bpy operations require specific modes (Object, Edit, Pose)

```python
# BAD — editing bones without Edit Mode
armature.data.edit_bones["Bone"].head = (0, 0, 0)  # RuntimeError!

# GOOD — switch to correct mode first
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='EDIT')
armature.data.edit_bones["Bone"].head = (0, 0, 0)
bpy.ops.object.mode_set(mode='OBJECT')
```

**Active Object Not Set:**
Many operators require an active object in the view layer

```python
# BAD — operator fails silently or errors
bpy.ops.object.transform_apply(location=True)

# GOOD — set active object first
bpy.context.view_layer.objects.active = mesh_obj
mesh_obj.select_set(True)
bpy.ops.object.transform_apply(location=True)
```

**Orphaned Data Blocks:**
Not cleaning up between characters causes memory leaks in batch runs

```python
# BAD — just deleting objects, leaving orphaned meshes/materials
bpy.ops.object.delete()

# GOOD — also purge orphaned data
bpy.ops.object.delete()
for block in bpy.data.meshes:
    if block.users == 0:
        bpy.data.meshes.remove(block)
for block in bpy.data.materials:
    if block.users == 0:
        bpy.data.materials.remove(block)
```

**Material Slot Indexing:**
Face material_index must match the slot order, not the region ID

```python
# BAD — using region ID as material index
face.material_index = region_id  # Wrong if materials aren't in order!

# GOOD — track slot positions
materials_slots = {}
for region_id, color in REGION_COLORS.items():
    mat = create_region_material(region_id, color)
    mesh_obj.data.materials.append(mat)
    materials_slots[region_id] = len(mesh_obj.data.materials) - 1

face.material_index = materials_slots[dominant_region]
```

**Segmentation Anti-Aliasing Artifacts:**
Masks must have zero anti-aliasing — any blending creates invalid region IDs

```python
# BAD — default render settings include AA
bpy.ops.render.render(write_still=True)

# GOOD — disable AA for segmentation pass
scene.render.filter_size = 0.0
scene.eevee.use_taa_reprojection = False
```

**Path Handling:**
Use pathlib consistently, handle cross-platform paths

```python
# BAD
output = output_dir + "/" + filename

# GOOD
from pathlib import Path
output = Path(output_dir) / filename
```

**Bone Name Matching:**
Case-insensitive, handle prefixes like "mixamorig:"

```python
# BAD — exact match only
if bone.name == "LeftArm":

# GOOD — strip prefix, case-insensitive
clean_name = bone.name.split(":")[-1].lower()
if "leftarm" in clean_name or ("left" in clean_name and "arm" in clean_name):
```

**JSON Serialization with NumPy:**
NumPy types aren't JSON-serializable by default

```python
# BAD — crashes on numpy.float32
json.dump({"weight": vertex_weight}, f)

# GOOD — convert to Python types
json.dump({"weight": float(vertex_weight)}, f)
```

### Quality Commands
```bash
# Lint Python files
ruff check .

# Format Python files
ruff format .

# Run pipeline on test character
blender --background --python generate_dataset.py -- \
  --input_dir ./source_characters/ \
  --output_dir ./dataset/ \
  --styles flat \
  --resolution 512

# Validate dataset outputs
python -c "
from PIL import Image
import json, glob

# Check mask dimensions and values
for mask_path in glob.glob('dataset/masks/*.png'):
    img = Image.open(mask_path)
    assert img.mode == 'L', f'{mask_path}: expected grayscale, got {img.mode}'
    assert img.size == (512, 512), f'{mask_path}: expected 512x512, got {img.size}'
    pixels = set(img.getdata())
    assert all(0 <= p <= 17 for p in pixels), f'{mask_path}: invalid region IDs {pixels}'

# Check joint files
for joint_path in glob.glob('dataset/joints/*.json'):
    with open(joint_path) as f:
        data = json.load(f)
    assert 'joints' in data, f'{joint_path}: missing joints key'
"
```

## Output

Report:
- Issues found
- Fixes applied
- Verification results
- Any remaining items
