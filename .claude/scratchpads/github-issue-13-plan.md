# Issue #13: Add batch processing: multi-character loop with progress logging and error handling

## Understanding
- Extend `generate_dataset.py` to process multiple characters × multiple poses from an input directory
- Type: **new feature** — adds multi-pose iteration, incremental processing, and robust batch handling
- Currently the pipeline discovers all FBX files but only renders T-pose (POSE_INDEX=0). This issue adds:
  1. Pose library iteration via `pose_applicator.list_poses()` / `apply_pose()`
  2. `--only_new` flag to skip already-processed character+pose combinations
  3. `--max_characters` and `--poses_per_character` flags for testing subsets
  4. Per-character and per-pose error handling (log + skip, don't crash batch)
  5. Progress logging with counts and timing
  6. End-of-batch summary

## Approach
- Modify `process_character()` to accept a list of `PoseInfo` objects and iterate poses
- Move the current single-pose logic into a per-pose loop within `process_character()`
- The pose loop: for each pose, apply it, render seg+color, extract joints, then reset
- Weights are T-pose only (already handled — just need to gate it with `pose.name == "t_pose"`)
- `--only_new` check: look for existing mask file at `masks/{char_id}_pose_{nn}.png`
- Progress format: `[{char_num}/{total_chars}] {char_id} - {pose_name} ({pose_num}/{total_poses})`
- Scene cleanup between characters via `importer.clear_scene()`
- `gc.collect()` between characters for memory management
- Exit code 1 if any character failed

### Key Design Decision: Pose Indexing
- Currently uses `POSE_INDEX = 0` constant. Need to replace with actual pose index.
- Pose index = position in the poses list (0-based), used in filenames like `_pose_00`, `_pose_01`, etc.
- The `PoseInfo.name` is too long for filenames — use numeric index instead.

### Key Design Decision: Camera Per-Pose vs Per-Character
- Camera must be recomputed per-pose (character bounding box changes with pose)
- Weight extraction camera stays T-pose only (already correct)

## Files to Modify
- `pipeline/generate_dataset.py` — Main changes: add `--pose_dir`, `--only_new`, `--max_characters`, `--poses_per_character` CLI args; refactor `process_character()` to iterate poses; add progress logging and per-pose error handling; add summary reporting

## Risks & Edge Cases
- **Pose application failure**: Some animation FBX files may not have matching bone names → `apply_pose()` returns False → log and skip pose
- **Camera recomputation per pose**: Character bounding box changes with pose (arms out vs crouching) → must recompute camera for each pose
- **Memory between characters**: Blender can accumulate orphaned data blocks → `clear_scene()` + `gc.collect()` between characters
- **`--only_new` false positives**: If a previous run partially completed (mask exists but joints don't), `--only_new` will skip it. Acceptable for v1 — mask existence is the marker.
- **Augmentation × pose combinations**: Each pose gets all augmentation variants (flip, scale). The augmentation loop is already inside `process_character()` — it should be inside the pose loop.

## Open Questions
- None — the issue is well-specified and all dependencies (#9 pipeline, #10 pose_applicator) are complete.

## Implementation Notes
- Refactored `process_character()` to accept `poses: list[PoseInfo]` and `pose_dir: Path` — iterates all poses per character
- Extracted `_process_single_pose()` as a keyword-only helper — processes one pose with all its augmentation variants, raises on failure for clean error handling in the caller
- Added `CharacterResult` dataclass to track per-character success/failure/skip counts and timing
- `--only_new` uses `_is_already_processed()` checking mask file existence at `masks/{char_id}_pose_{nn}.png`
- `--max_characters` slices `fbx_files` list; `--poses_per_character` slices `poses` list (both after discovery)
- Pose discovery: `list_poses(pose_dir)` called once in `main()`, shared across all characters
- Weight extraction moved after the pose loop (T-pose, once per character) — resets pose before extracting
- Scene cleanup: `clear_scene()` + `gc.collect()` between characters in `main()` loop
- Progress format: `[1/10] mixamo_001 — walk_frame_07 (3/20) OK`
- Summary: `_print_summary()` prints per-character table with OK/Fail/Skip/Time columns + totals
- Exit code: `sys.exit(1)` if any character had errors or pose failures
- Removed `POSE_INDEX = 0` constant — pose index is now the position in the poses list
- Simplified pose application: single `apply_pose()` call handles all pose types (T-pose, A-pose, animation FBX) — no special-casing needed
- Also installed ruff linter via pipx and added `ruff.toml` project config
- Fixed pre-existing lint issues: `zip()` missing `strict=` in renderer.py
