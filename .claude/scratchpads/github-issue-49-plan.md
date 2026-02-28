# Issue #49: Build Live2D Model Renderer for Training Data Generation

## Understanding
- Build a pure Python renderer that loads Live2D models (.moc3 + texture atlas) and produces training data in the same format as the Blender and Spine pipelines
- Output: composite image (512x512), segmentation mask (8-bit grayscale), draw order map (8-bit grayscale), metadata JSON
- Augmentation: horizontal flip, slight rotation (±5°), scale (±10%), color jitter — NOT style transfer
- Target: 400 models × 4 augmentations = ~1,600 training images
- Dependencies: #47 (Live2D mapper — CLOSED) and #42 (draw order format — CLOSED) are both done

## Key Risk: .moc3 SDK Licensing
The Live2D Cubism SDK has restrictive licensing. We CANNOT ship the SDK or depend on it directly.

**Approach: Minimal custom .moc3 parser is not practical** — .moc3 is a binary format with undocumented internals.

**Practical approach: Texture atlas + JSON metadata parser**
- Many Live2D community models distribute with texture atlases (.png) and a model JSON file (.model3.json) that references part textures
- The .model3.json contains ArtMesh names, draw order, and references to texture regions
- We parse the JSON metadata to discover fragments, load texture atlas, extract individual fragment images via atlas coordinates
- For models with only .moc3 (no JSON), we skip them or document that manual pre-processing is needed

**Alternative considered: cubism-rs** — Rust bindings for the SDK. License still applies.

**Decision: Build a texture-atlas-based renderer** that works with models distributed as:
1. A `.model3.json` file (references textures and lists ArtMesh groups)
2. One or more texture atlas PNGs
3. Optionally a `.physics3.json` file (ignored for static rendering)

For models where we only have pre-rendered fragment images (extracted by artists), we handle those as a simpler case: load fragment PNGs directly and composite.

## Approach
Design closely mirrors `spine_parser.py` — same architecture pattern:
1. Parse model metadata (JSON) to discover fragments
2. Load texture atlas and extract per-fragment images
3. Composite character at default pose (back-to-front draw order)
4. Build segmentation mask using `live2d_mapper.map_fragment()` for each ArtMesh
5. Build draw order map from explicit fragment render order indices
6. Extract joint positions (approximate from fragment centroids per region)
7. Apply augmentations (flip, rotation, scale, color jitter)
8. Save via existing `exporter` module

## Files to Modify

### New: `pipeline/live2d_renderer.py`
- Core renderer module (pure Python, no Blender dependency)
- Data structures: `Live2DModel`, `Live2DFragment`, `Live2DRenderResult`
- Functions:
  - `parse_model_json()` — Parse .model3.json
  - `load_texture_atlas()` — Load atlas PNGs
  - `extract_fragments()` — Cut fragment images from atlas
  - `composite_character()` — Assemble fragments in draw order
  - `build_segmentation_mask()` — Region IDs from fragment mapping
  - `build_draw_order_map()` — Normalized depth from render order
  - `extract_joints()` — Approximate 2D joint positions from region centroids
  - `apply_augmentations()` — Flip, rotate, scale, color jitter
  - `process_live2d_model()` — Single model entry point
  - `process_live2d_directory()` — Batch entry point

### Modified: `pipeline/config.py`
- Add `LIVE2D_AUGMENTATION_ROTATIONS`, `LIVE2D_AUGMENTATION_SCALES`, `LIVE2D_AUGMENTATION_COLOR_JITTER` constants

### Modified: `pipeline/generate_dataset.py`
- Add `--live2d_dir` CLI argument
- Wire up `live2d_renderer.process_live2d_directory()` call

### New: `tests/test_live2d_renderer.py`
- Test JSON parsing, fragment extraction, compositing, mask building, draw order, augmentation

## Risks & Edge Cases
- .model3.json format varies across Live2D SDK versions (3.x vs 4.x)
- Texture atlas may be packed (requires atlas.json coordinates) or unpacked (separate PNGs)
- Some models have multiple texture pages
- Fragment draw order may not be explicitly specified in all model files
- Deformation meshes (non-rectangular fragments) need UV-based extraction
- Some models may only have .moc3 without readable JSON — these are skipped

## Open Questions
- Should we support models distributed as pre-extracted fragment PNGs (no atlas)?
  → Yes, as a simpler code path
- How to handle models with no draw order info?
  → Use fragment array order as implicit draw order (common convention)
- Should augmentation be part of the renderer or a separate step?
  → Part of the renderer, matching the issue spec

## Implementation Notes

### What was implemented
- **`pipeline/live2d_renderer.py`** (~1000 lines): Full renderer module with:
  - `.model3.json` parsing + `.cdi3.json` display name loading
  - Pre-extracted fragment PNG discovery (searches parts/, textures/, images/, model root)
  - Fragment compositing with draw order normalization to [0, 255]
  - Segmentation mask via `live2d_mapper.map_fragment()` (CDI name → raw name fallback)
  - Joint extraction from mask region centroids (confidence 0.8 for in-bounds, 0.4 otherwise)
  - 4 augmentation variants: identity, flip (with L/R region swap), rotation+jitter, scale+jitter
  - Lazy import of `style_augmentor` (cv2 dependency) — only loaded when non-flat styles requested
  - Batch `process_live2d_directory()` that iterates model subdirectories

- **`pipeline/config.py`**: Added `LIVE2D_AUGMENTATION_ROTATIONS`, `LIVE2D_AUGMENTATION_SCALES`, `LIVE2D_AUGMENTATION_COLOR_JITTER`

- **`pipeline/generate_dataset.py`**: Added `--live2d_dir` CLI arg, processing block after VRoid

- **`tests/test_live2d_renderer.py`** (40 tests): Covers JSON parsing, fragment discovery, compositing, joint extraction, all augmentation types, end-to-end model + directory processing

### Design decisions
- Primary workflow is **pre-extracted fragment PNGs** (not atlas UV extraction) since most community models are distributed this way
- Models with only .moc3 + atlas but no fragment images are logged and skipped (binary .moc3 parsing out of scope due to SDK licensing)
- Augmentation variants use `pose_index` 0-3 for the 4 variants per model (identity, flip, rotation, scale)
- Joint confidence is 0.8 (not 1.0) since centroids are approximate vs. skeleton-based joints
- Atlas images >2048px are auto-skipped to avoid treating full texture sheets as fragments

### Follow-up work
- Review UI for unmapped fragments (issue R7)
- Support for atlas UV extraction when .moc3 binary format is reverse-engineered or SDK license resolved
- Integration with actual Live2D model collection from Booth.pm/DeviantArt/GitHub
