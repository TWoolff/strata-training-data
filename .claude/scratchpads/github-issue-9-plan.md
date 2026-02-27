# Issue #9: Build Phase 1 end-to-end pipeline

## Understanding
- Wire all Phase 1 modules into a single `generate_dataset.py` orchestrator
- Type: **new feature** — integration of existing modules
- Phase 1 scope: single character, T-pose only, flat style only
- Must run headless via: `blender --background --python generate_dataset.py -- --input_dir ./source_characters/ --output_dir ./dataset/`

## Approach
- Single `generate_dataset.py` file with:
  1. `parse_args()` — argparse after `sys.argv.index("--")` separator
  2. `process_character()` — per-character pipeline (try/except wrapped)
  3. `main()` — discover FBX files, iterate, report summary
- Follow PRD §9.2 pseudocode exactly for pipeline ordering
- Phase 1 simplifications: no pose_applicator (T-pose only), no style_augmentor (flat only), no weight_extractor
- Two render passes per character:
  1. Segmentation pass: assign region materials → render RGB → convert to grayscale mask
  2. Color pass: restore original materials → setup lighting → render flat color
- Need to store and restore original materials for the color pass (seg pass overwrites them)
- Use a temp file for the RGB segmentation render, then convert to grayscale mask

## Files to Modify
- `generate_dataset.py` — **NEW** — main orchestrator (the only file for this issue)

## Pipeline Flow (per character)
1. `importer.import_character(fbx_path)` → ImportResult
2. `bone_mapper.map_bones(armature, meshes, char_id, source_dir)` → BoneMapping
3. `renderer.create_region_materials()` → materials list
4. For each mesh: `renderer.assign_region_materials(mesh, idx, vertex_to_region, materials)`
5. `renderer.setup_camera(scene, meshes)` → camera
6. `renderer.setup_segmentation_render(scene)` → configure render settings
7. `renderer.render_segmentation(scene, temp_rgb_path)` → RGB mask
8. `renderer.convert_rgb_to_grayscale_mask(rgb_path, mask_output)` → grayscale mask
9. `joint_extractor.extract_joints(scene, camera, armature, meshes, bone_to_region)` → joint data
10. Restore original materials on meshes for color render
11. `renderer.setup_color_render(scene)` → configure color settings
12. `renderer.render_color(scene, image_output)` → color image
13. Export: save mask, joints, image, source metadata, class_map via exporter

## Original Materials Strategy
Before assigning segmentation materials, store a list of each mesh's material slots.
After the seg render, restore them for the color pass.

## Risks & Edge Cases
- Blender's `sys.argv` contains both Blender args and script args — must split at `"--"`
- Original materials need proper backup/restore (material slots are mutable references)
- Temp RGB segmentation file needs cleanup
- Characters with no armature or no meshes → `importer` returns None, skip gracefully
- Unmapped bones → logged as warnings but don't block processing
- The seg render produces RGB but we need grayscale — the converter handles nearest-color matching

## Open Questions
- None — all modules have clear APIs and the issue is well-specified

## Implementation Notes
- Implemented as planned — single `generate_dataset.py` with `parse_args()`, `process_character()`, `main()`
- Pipeline follows PRD §9.2 flow exactly: import → map → materials → camera → seg render → joints → color render → export
- Material backup/restore: `_backup_materials()` stores material references per mesh slot, `_restore_materials()` clears and re-appends after seg pass
- Temp file strategy: `tempfile.NamedTemporaryFile` for RGB seg render, immediately converted to grayscale mask, then deleted
- `setup_color_render()` moved outside the style loop (render settings are style-independent in Phase 1)
- `_infer_source()` helper detects asset source from char_id prefix (mixamo, quaternius, kenney)
- Metadata saved via exporter: joints JSON + source metadata JSON + class_map.json
- Images and masks written directly by renderer to final paths (no double-save through exporter)
- Error handling: each character wrapped in try/except in `main()`, plus early return in `process_character()` if import fails
- Summary report with timing printed at end of batch
