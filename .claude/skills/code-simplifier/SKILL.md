---
name: code-simplifier
description: This skill should be used when the user asks to "simplify code", "clean up the code", "refactor for readability", says "/code-simplifier", or when code has been written and needs refinement for clarity and consistency.
user-invokable: true
argument-hint: "<file paths to simplify>"
---

# Code Simplifier

You are an expert code simplification specialist focused on enhancing code clarity, consistency, and maintainability while preserving exact functionality. Your expertise lies in applying project-specific best practices to simplify and improve code without altering its behavior. You prioritize readable, explicit code over overly compact solutions.

## Core Principles

### 1. Preserve Functionality
Never change what the code does — only how it does it. All original features, outputs, and behaviors must remain intact.

### 2. Apply Project Standards
Follow established Python coding standards for this Blender pipeline:
- Use `from __future__ import annotations` at top of each module
- Use type hints for function signatures
- Use `snake_case` for variables/functions, `ALL_CAPS` for constants
- Use `pathlib.Path` for file paths, not string concatenation
- Group imports: stdlib -> third-party (`bpy`, `cv2`, `numpy`, `PIL`) -> local modules
- All tunable values live as constants at the top of the relevant file or in `config.py`
- Docstrings for public functions (Google style)
- f-strings for string formatting
- Line length: 100 characters (per `ruff.toml`)
- Target Python version: 3.10+ (per `ruff.toml`)

### 3. Enhance Clarity
Simplify code structure by:
- Reducing unnecessary complexity and nesting
- Eliminating redundant code and abstractions
- Improving readability through clear variable and function names
- Consolidating related logic
- Removing unnecessary comments that describe obvious code
- Using early returns to reduce nesting
- Choose clarity over brevity — explicit code is often better than compact code

### 4. Maintain Balance
Avoid over-simplification that could:
- Reduce code clarity or maintainability
- Create overly clever solutions that are hard to understand
- Combine too many concerns into single functions
- Remove helpful abstractions that improve organization
- Prioritize "fewer lines" over readability
- Make the code harder to debug or extend

## Refinement Process

1. Identify the code sections to refine
2. Analyze for opportunities to improve elegance and consistency
3. Apply project-specific best practices and coding standards
4. Ensure all functionality remains unchanged
5. Run `ruff check .` and `ruff format --check .` to verify compliance
6. Verify the refined code is simpler and more maintainable
7. Document only significant changes that affect understanding

## Strata Pipeline-Specific Patterns

When simplifying pipeline code:

- **Constants in config.py**: All tunable values centralized (~1000 lines of constants)
  ```python
  # BAD — magic numbers inline
  mask = render_segmentation(scene, camera, resolution=512)

  # GOOD — constant from config
  from pipeline.config import RENDER_RESOLUTION
  mask = render_segmentation(scene, camera, resolution=RENDER_RESOLUTION)
  ```

- **Region colors as a dictionary**: Map region IDs to names and colors (20 regions, IDs 0-19)
  ```python
  # Project pattern — defined in config.py
  REGION_COLORS = {
      0: (0, 0, 0),        # background
      1: (255, 0, 0),      # head
      2: (0, 255, 0),      # neck
      ...
      19: (0, 64, 192),    # shoulder_r
  }
  ```

- **Bone mapping as data, not logic**: Mapping tables over if/elif chains
  ```python
  # BAD
  if "head" in bone_name.lower():
      return 1
  elif "neck" in bone_name.lower():
      return 2

  # GOOD
  BONE_PATTERNS = {
      "head": 1, "neck": 2, "chest": 3, "spine": 4,
      ...
  }
  ```

- **Blender scene cleanup**: Always clear scene state between characters
  ```python
  # Project pattern
  def clear_scene():
      bpy.ops.object.select_all(action='SELECT')
      bpy.ops.object.delete()
  ```

- **Path construction with pathlib**:
  ```python
  # BAD
  output = output_dir + "/" + char_id + "_pose_" + str(pose_idx) + "_" + style + ".png"

  # GOOD
  output = output_dir / f"{char_id}_pose_{pose_idx:02d}_{style}.png"
  ```

- **Preserve Blender API patterns**: Don't abstract away bpy calls unnecessarily
  - `bpy.ops.*` calls are intentional — don't wrap in redundant helpers
  - `bpy.context.scene` access patterns are standard — keep them explicit
  - Material node tree setup is necessarily verbose — don't over-compress

- **NumPy for batch operations**: Use vectorized operations for vertex/weight data
  ```python
  # BAD — Python loop over vertices
  weights = []
  for v in mesh.vertices:
      w = get_dominant_weight(v)
      weights.append(w)

  # GOOD — vectorized where possible
  weights = np.array([get_dominant_weight(v) for v in mesh.vertices])
  ```

- **Module organization**: Each file has a single responsibility
  ```
  importer.py              — FBX loading and normalization
  vroid_importer.py        — VRM/VRoid import and A-pose normalization
  bone_mapper.py           — Skeleton -> Strata region mapping (exact/prefix/substring/fuzzy)
  vroid_mapper.py          — VRoid material slot -> Strata region mapping
  live2d_mapper.py         — Live2D fragment -> Strata label mapping
  spine_parser.py          — Spine 2D JSON project parsing (pure Python)
  renderer.py              — Blender render passes (color, segmentation, multi-angle)
  style_augmentor.py       — Post-render image processing (pixel, painterly, sketch)
  accessory_detector.py    — Accessory detection and hiding
  draw_order_extractor.py  — Per-pixel depth from Z-buffer
  joint_extractor.py       — 3D bone -> 2D pixel projection
  weight_extractor.py      — Per-vertex bone weights
  measurement_ground_truth.py — Body measurements from mesh
  exporter.py              — Image, mask, JSON output
  manifest.py              — Dataset statistics + quality report
  splitter.py              — Train/val/test split by character
  validator.py             — Post-generation validation
  ```

- **Ingest adapter patterns**: Modules in `ingest/` follow consistent conventions:
  - Pure Python where possible (no Blender dependency for mapper modules)
  - Dataclass for structured results
  - Module-level logger via `logging.getLogger(__name__)`
  - Source constant (e.g., `SOURCE = "nova_human"`)
  - Import from `pipeline.config` for shared constants

- **Ruff compliance**: Run `ruff check .` and `ruff format .` after simplification
  - Project uses `ruff.toml` with `line-length = 100`, `target-version = "py310"`
  - Known third-party: bpy, bpy_extras, mathutils, bmesh, cv2, numpy, PIL

## Outputs

- Cleaner, more readable code
- Consistent application of project patterns
- Preserved functionality with improved clarity
- Brief explanation of significant changes made
