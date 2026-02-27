# Issue #8: Implement file export with naming conventions and directory structure

## Understanding
- Create `exporter.py` — the file I/O module for the pipeline
- Save rendered images, segmentation masks, joint JSON, and per-character metadata
- Enforce strict directory layout and naming conventions from PRD §8
- Support incremental runs (`--only_new`) by skipping existing files
- Type: **new feature** (new module)

## Approach
- Single module with focused public functions matching the issue spec:
  `save_image()`, `save_mask()`, `save_joints()`, `save_source_metadata()`,
  `save_class_map()`, `ensure_output_dirs()`
- Pure Python (PIL, json, pathlib) — no Blender dependency, so it can be tested outside Blender
- Follow existing patterns: logging, type hints, Google-style docstrings, pathlib.Path

## Files to Modify
- **`exporter.py`** (new) — all export functions
- No changes needed to `config.py` (REGION_NAMES already has all data needed for class_map)

## Key Design Decisions
- `save_joints()` already exists in `joint_extractor.py:349` — the exporter's `save_joints()` will wrap the full PRD schema (character_id, pose_name, etc.) similar to joint_extractor but as a standalone exporter function. Actually, looking again, `joint_extractor.save_joints()` already saves the full schema. The exporter should provide a simpler pass-through that just handles directory creation and skip-if-exists logic. I'll make the exporter's `save_joints()` take already-serialized dict data and write it, leaving the schema construction to `joint_extractor`.
- Masks: accept PIL Image or numpy array, ensure mode "L" before saving
- Images: accept PIL Image, ensure mode "RGBA" before saving
- class_map.json: use REGION_NAMES from config.py, keys as strings per PRD schema
- Pose number formatting: `pose_{nn:02d}` (zero-padded 2 digits)

## Risks & Edge Cases
- Mask images arriving as RGB instead of grayscale — enforce mode "L" conversion
- Images arriving without alpha — enforce mode "RGBA" conversion
- Race conditions on directory creation — `exist_ok=True` handles this
- class_map.json should use 20 regions (0–19) per CLAUDE.md, not 18 per PRD (PRD is outdated — config has 20 including shoulders)

## Open Questions
- None — the issue spec is clear and config.py has all the data needed.

## Implementation Notes
- Implemented exactly as planned — single new file `exporter.py`, no changes to other modules
- All 6 public functions from the issue spec implemented: `ensure_output_dirs()`, `save_image()`, `save_mask()`, `save_joints()`, `save_weights()`, `save_source_metadata()`, `save_class_map()`
- Added 5 filename helper functions (`image_filename`, `mask_filename`, `joints_filename`, `weights_filename`, `source_filename`) for consistent naming
- `save_joints()` in exporter takes a pre-built dict (from `joint_extractor.extract_joints`) — no schema construction, just file I/O with skip-if-exists
- `save_mask()` accepts both PIL Image and numpy array for flexibility
- class_map.json includes all 20 regions (0–19) from `config.REGION_NAMES`
- Code simplifier confirmed no changes needed — module is clean as written
- Note: `joint_extractor.save_joints()` still exists with its own schema wrapping — the two functions serve different roles (schema construction vs. directory-aware file I/O)
