# Issue #95: Build AnimeRun, UniRig, and LinkTo-Anime ingest adapters

## Understanding
- Build three ingest adapters for external datasets that need conversion to Strata training format
- Type: new feature (three new modules + tests)
- Each adapter handles a different data type:
  - AnimeRun: contour/color frame pairs → contour training format
  - UniRig: bone hierarchies → Strata 20-bone skeleton mapping
  - LinkTo-Anime: skeleton + optical flow extraction from Mixamo-rigged data

## Approach
Follow existing adapter patterns from `nova_human_adapter.py` and `stdgen_semantic_mapper.py`:
- Pure Python (no Blender dependency) for all three adapters
- Module-level source constants, dataclass results, structured logging
- Standard metadata dicts with `missing_annotations` fields
- Graceful error handling (log warnings, don't crash, return None for invalid)
- Batch processing with `convert_directory()` pattern

### Adapter 1: `animerun_contour_adapter.py`
- **Input:** AnimeRun directory with `{split}/{scene}/contour/` and `{split}/{scene}/anime/` subdirs
- **Output:** `contour_pairs/` with (with_contours.png, without_contours.png, contour_mask.png) triples
- Process: load anime frame + contour frame, resize to 512×512, generate binary contour mask from difference, save triple + metadata
- Dataset structure: `train/` and `test/` splits, each with scene subdirectories containing `contour/`, `anime/`, `flow_fwd/`, `flow_bwd/` etc.

### Adapter 2: `unirig_skeleton_mapper.py`
- **Input:** UniRig processed data (joint names, joint positions, skinning weights, parent indices)
- **Output:** Mapping table from UniRig bone names → Strata region IDs + validation report
- Reuse `bone_mapper.py` fuzzy matching logic but adapted for UniRig's naming conventions
- UniRig data format: JSON/npz files with keys for vertices, joints, weights, parent indices, joint_names
- Primary value is validation data — report match rates, unmapped bones

### Adapter 3: `linkto_adapter.py`
- **Input:** LinkTo-Anime dataset (~29K frames with Mixamo skeleton + optical flow)
- **Output:** Extracted joint positions, optical flow fields, frame metadata
- Uses Mixamo skeletons → map directly via existing `MIXAMO_BONE_MAP`
- Dataset structure: video sequences with frames, flow data, skeleton annotations
- Extract 2D joint positions from skeleton data, pair with frames

## Files to Modify
### New files:
- `ingest/animerun_contour_adapter.py` — AnimeRun contour pair converter
- `ingest/unirig_skeleton_mapper.py` — UniRig skeleton mapping
- `ingest/linkto_adapter.py` — LinkTo-Anime skeleton + flow extraction
- `tests/test_animerun_contour_adapter.py` — Tests for AnimeRun adapter
- `tests/test_unirig_skeleton_mapper.py` — Tests for UniRig mapper
- `tests/test_linkto_adapter.py` — Tests for LinkTo-Anime adapter

### No existing files need modification
- Config constants already exist (`MIXAMO_BONE_MAP`, `VRM_BONE_ALIASES`, `REGION_NAMES`)
- No new constants needed in config.py for these adapters

## Risks & Edge Cases
- **AnimeRun:** Contour/anime frame pairs may not always be 1:1 matched; need to validate pair existence
- **AnimeRun:** Image sizes may vary across scenes; must handle non-square inputs
- **UniRig:** Bone naming highly variable across 14K models (Objaverse + VRoid mix); expect <80% auto-match for many
- **UniRig:** Data format may be npz or JSON — need to handle both gracefully
- **LinkTo-Anime:** Skeleton data format specifics not fully documented; may need to adapt parsing
- **LinkTo-Anime:** Optical flow stored as binary `.flo` files — standard Middlebury format
- **All:** Missing/corrupt files must be handled gracefully with logging, not crashes

## Open Questions
- UniRig data format specifics (npz vs JSON) — will implement for common formats and log unsupported
- LinkTo-Anime exact directory structure for skeleton annotations — will implement flexible discovery

## Implementation Notes

### What was implemented

**AnimeRun contour adapter** (`ingest/animerun_contour_adapter.py`):
- Scene discovery via `{split}/{scene}/contour/` + `anime/` directory presence
- Frame pair matching by stem (intersection of contour and anime filenames)
- Contour mask generation via grayscale pixel difference with configurable threshold (default 30)
- Outputs: `with_contours.png`, `without_contours.png`, `contour_mask.png`, `metadata.json`
- `convert_scene()` and `convert_directory()` batch processing with `max_frames`/`max_scenes` limits
- Hidden directory filtering, `only_new` skip logic

**UniRig skeleton mapper** (`ingest/unirig_skeleton_mapper.py`):
- 6-level matching chain: exact → alias → VRM → prefix-strip → substring → fuzzy keyword
- Reused matching logic from `pipeline/bone_mapper.py` without bpy dependency
- Alias table includes Blender-style names, generic rig names, and common abbreviations
- Skeleton validation: `has_root`, `has_head`, `has_limbs`, `has_symmetric_arms`, `has_symmetric_legs`
- Input formats: `.npz` (with `joint_names` key) and `.json` (with `joint_names` key)
- JSON export with full joint mapping details, validation, and region coverage
- Batch `convert_directory()` with `max_characters`, `only_new`, good/poor tracking (threshold 0.80)

**LinkTo-Anime adapter** (`ingest/linkto_adapter.py`):
- Sequence discovery from directory listing (each subdirectory = sequence)
- Frame discovery by globbing `*.png`/`*.jpg` in sequence directory
- Skeleton parsing from JSON (dict or list format, with or without `"joints"` wrapper) and `.npy`
- Optical flow loading from Middlebury `.flo` (magic number validated) and `.npy`
- Mixamo bone name lookup built from `MIXAMO_BONE_MAP` (both prefixed and stripped variants)
- Outputs: `image.png`, `joints.json` (19 joints with region IDs), `flow.npy`, `metadata.json`
- Missing annotation flags in metadata (`has_joints`, `has_flow`)
- Batch processing with `max_sequences`/`max_frames` limits

### Test coverage
- 25 tests for AnimeRun adapter (resize, mask generation, metadata, scene/pair discovery, conversion, batching)
- 34 tests for UniRig mapper (joint mapping all 6 methods, skeleton validation, data loading, JSON round-trip, batch conversion)
- 43 tests for LinkTo adapter (flow reading, skeleton parsing, frame discovery, conversion, batching, edge cases)
- Total: 102 new tests, all passing

### Design decisions
- Used `CONTOUR_DIFF_THRESHOLD = 30` as module-level constant (not in config.py since it's adapter-specific)
- UniRig mapper builds its own alias tables rather than importing from bone_mapper.py to avoid bpy dependency chain
- LinkTo adapter normalizes joint coordinates to [0, 1] range and rounds to 4 decimal places
- All three adapters follow the established pattern: dataclass results, module logger, `SOURCE` constant, `only_new` flag

### Follow-up work
- None identified — all three adapters are complete and tested
