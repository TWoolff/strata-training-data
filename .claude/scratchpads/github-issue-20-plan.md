# Issue #20: Add per-character bone mapping override JSON support

## Understanding
- The issue asks for per-character override JSON files that manually assign unmapped bones to Strata regions
- Type: **new feature** (partially implemented — override loading/application already works)
- Much of the core override functionality already exists in `bone_mapper.py`:
  - `_load_overrides()` loads `{char_id}_overrides.json` from the source directory
  - `_map_all_bones()` applies overrides as highest priority (step 1)
  - `MappingStats.override` tracks override count
  - `map_bones()` logs override stats

## What's Missing
1. **Override validation**: Region IDs must be 1–19 (not 0/background). Invalid entries should be logged as warnings and skipped.
2. **Nonexistent bone warnings**: If an override references a bone name not in the armature, log a warning.
3. **Template generation function**: Write `{bone_name: null}` JSON for each unmapped bone after mapping.
4. **`--generate_overrides` CLI flag**: When set, process all characters, run bone mapping, and generate template override JSONs for those with unmapped bones — then exit (no rendering needed).

## Approach
- Add validation to `_load_overrides()` in `bone_mapper.py` — filter out entries with region_id outside 1–19 range, logging warnings for each
- Add nonexistent bone check in `_map_all_bones()` — after processing overrides, warn if any override bone names don't exist in the armature
- Add `generate_override_template()` function in `bone_mapper.py` — takes unmapped bones list, writes `{bone: null}` JSON to the source directory
- Add `--generate_overrides` flag to `generate_dataset.py` — runs a lightweight loop (import → map → generate template) without rendering

## Files to Modify
- `pipeline/bone_mapper.py` — Add validation, nonexistent bone warnings, template generation function
- `pipeline/generate_dataset.py` — Add `--generate_overrides` flag and lightweight generate-only loop

## Risks & Edge Cases
- Override file with region_id = 0 (background) — should warn and skip
- Override file with region_id > 19 or negative — should warn and skip
- Override referencing a bone not in the armature — should warn (bone might be misnamed)
- `null` values in override JSON — should be skipped (template file format uses null as placeholder)
- Empty override file or empty dict — should be handled gracefully (already works)
- Template generation when no unmapped bones — should skip silently (no file written)

## Open Questions
- None — requirements are clear from the issue

## Implementation Notes
- The core override loading/application was already implemented in a prior issue (#3)
- Added per-entry validation in `_load_overrides()`: null values are silently skipped (template format), non-integer and out-of-range (outside 1-19) region IDs are warned and skipped
- Added armature bone name check in `_map_all_bones()`: builds a set of actual bone names and warns for any override referencing a nonexistent bone
- `generate_override_template()` is a new public function in `bone_mapper.py` — writes sorted `{bone: null}` JSON, skips if override file already exists
- `_generate_override_templates()` in `generate_dataset.py` is a lightweight loop (import → map → template) that runs when `--generate_overrides` is passed, then exits without rendering
- Code-simplifier confirmed no changes needed — both files follow project patterns
