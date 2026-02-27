# Issue #25: Implement Spine project parser

## Understanding
- Build a parser for Spine 2D animation project files (.spine/.json + atlas) that extracts character images, bone positions, and part-to-region assignments into the Strata dataset format
- **Type:** New feature (Phase 5 — 2D Source Integration)
- Spine is a 2D skeletal animation tool. Projects contain part images, bone hierarchies, and slot assignments that map naturally to Strata's segmentation regions
- This bridges the domain gap between 3D synthetic data and real 2D art

## Approach
- Create a pure Python module `spine_parser.py` (no Blender dependency) following the `live2d_mapper.py` precedent
- Add Spine-specific bone/slot patterns to `config.py` (like `LIVE2D_FRAGMENT_PATTERNS`)
- Use Pillow for image compositing (loading atlas parts, applying transforms, assembling character)
- Reuse existing `exporter.py` functions for output (images, masks, joints, source metadata)
- Map Spine bone names → Strata regions using regex patterns (like Live2D mapper)
- Generate segmentation masks by painting each part image's pixels with its region ID
- Extract joint positions from Spine bone world transforms projected to pixel coords

### Design decisions
- **Pure Python**: No Blender dependency. Spine files are JSON + PNG atlas, processable with stdlib + Pillow
- **Single module**: `spine_parser.py` handles parsing, compositing, mask generation, and joint extraction
- **Pattern-based mapping**: Spine bone names are often human-readable (e.g., "left-upper-arm", "head"). Use regex patterns like Live2D mapper, plus fallback to the existing substring/fuzzy matching from config.py
- **Default pose only for v1**: Extract the assembled character at the default (setup) pose. Animation poses would require implementing Spine's IK/constraint system — out of scope
- **Multiple skins**: Each skin is treated as a separate visual variant (different char_id suffix)

### Spine JSON structure (reference)
```json
{
  "skeleton": {"hash": "...", "spine": "4.1.xx", "width": 200, "height": 400},
  "bones": [
    {"name": "root"},
    {"name": "hip", "parent": "root", "x": 0, "y": 200},
    {"name": "left-upper-leg", "parent": "hip", "x": -50, "y": -10, "rotation": -5}
  ],
  "slots": [
    {"name": "left-foot", "bone": "left-foot", "attachment": "left-foot"},
    {"name": "left-lower-leg", "bone": "left-lower-leg", "attachment": "left-lower-leg"}
  ],
  "skins": {
    "default": {
      "left-foot": {"left-foot": {"x": 10, "y": 5, "width": 40, "height": 50}},
      ...
    }
  }
}
```

### Coordinate system
- Spine uses Y-up with origin at skeleton root bone
- Need to compute world transforms by walking the bone hierarchy (parent → child)
- Bone transform: translation (x, y) + rotation + scale, relative to parent
- Final output: 512×512 with character centered and scaled to fill frame

## Files to Modify
1. **`pipeline/config.py`** — Add `SPINE_BONE_PATTERNS` (regex patterns mapping Spine bone names to Strata region names, like `LIVE2D_FRAGMENT_PATTERNS`)
2. **`pipeline/spine_parser.py`** (NEW) — Main parser module:
   - Parse Spine JSON (skeleton, bones, slots, skins)
   - Compute bone world transforms (walk hierarchy)
   - Map bones to Strata regions via pattern matching
   - Load atlas/part images
   - Composite character image from parts at their bone transforms
   - Generate segmentation mask (each part's pixels → its region ID)
   - Extract joint positions from bone world transforms → pixel coords
   - Output: assembled image, mask, joint JSON — all via exporter.py
3. **`pipeline/generate_dataset.py`** — Add `--spine_dir` CLI arg and call spine_parser for .spine/.json files

## Risks & Edge Cases
- **Spine format versions**: Spine 3.x vs 4.x have slightly different JSON schemas (e.g., skins as array vs object). Need to handle both
- **Missing atlas images**: Some Spine projects reference images that may not be present. Log warning, skip missing parts
- **Complex attachments**: Spine supports mesh attachments (deformed images), not just region attachments. For v1, only handle region (rectangular) attachments; log warnings for mesh/path/etc
- **Rotation + scale compositing**: Pillow's affine transform may introduce aliasing. Use BILINEAR for color, NEAREST for masks
- **Coordinate origin**: Spine's root is typically at the character's feet (Y-up). Need to compute bounding box and center/scale to 512×512
- **Skins as array (Spine 4.x)**: Newer Spine versions use `"skins": [{"name": "default", "attachments": {...}}]` instead of `"skins": {"default": {...}}`

## Open Questions
- None significant — the Spine JSON format is well-documented and the pipeline patterns are clear from the codebase

## Implementation Notes

### What was implemented
- **`pipeline/config.py`**: Added `SPINE_BONE_PATTERNS` — 34 regex patterns covering head, neck, shoulders, arms (forearm/upper/hand), legs (shin/thigh/foot), hips, torso, accessories. Includes Spine-specific "front-"/"rear-" prefix convention (front=left viewer side, rear=right).
- **`pipeline/spine_parser.py`** (new, ~1000 lines): Pure Python module with:
  - Data structures: `SpineBone`, `SpineSlot`, `SpineAttachment`, `SpineProject`, `SpineParseResult`, `Viewport`
  - JSON parsing: Handles both Spine 3.x (dict skins) and 4.x (array skins) formats
  - Bone hierarchy: Single-pass world transform computation (parent-before-child order)
  - Region mapping: Regex-based bone→region, with slot→region fallback via slot name
  - Image compositing: PIL-based, slot draw order, affine transforms (scale + rotation + flip)
  - Mask generation: Vectorized numpy slicing (not per-pixel loops)
  - Joint extraction: Bone world positions → pixel coords via shared Viewport
  - `process_spine_directory()`: Batch processing with multi-skin support, style augmentation, exporter integration
- **`pipeline/generate_dataset.py`**: Added `--spine_dir` CLI argument. Spine processing runs after FBX processing and results are merged into the summary/manifest/splits.

### Design decisions made during implementation
- **Viewport dataclass**: Extracted shared viewport computation into a `Viewport` dataclass with `spine_to_pixel()` method, eliminating code duplication between compositing and joint extraction
- **Only region attachments**: Mesh, path, point, clipping, and bounding box attachment types are skipped with debug logging. This is a v1 limitation.
- **BILINEAR for color images, numpy for masks**: Color compositing uses PIL's BILINEAR resampling; mask painting uses numpy boolean indexing for speed (no per-pixel Python loops)
- **Lazy import of style_augmentor**: `process_spine_directory` uses a deferred import to avoid pulling in OpenCV when not needed
- **`_infer_source` updated**: Added "spine" prefix detection so Spine characters get correct source attribution in metadata

### Known limitations / follow-up work
- Animation poses not supported (would need Spine animation timeline + IK solver)
- Mesh attachments (deformed images) not composited — only region (rectangular) attachments
- Atlas texture packing (single atlas PNG with sub-regions) not yet supported — expects individual part images in a directory
- No weight extraction for Spine characters (2D doesn't have per-vertex bone weights)
- No draw order map extraction (Spine draw order is slot-based, not depth-based)
