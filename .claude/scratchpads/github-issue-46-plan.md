# Issue #46: Add strata_compatible column to CMU action labels

## Understanding
- Add a `strata_compatible` column to `animation/labels/cmu_action_labels.csv`
- Create a helper function that auto-evaluates BVH files for Strata compatibility
- Compatibility = clip uses only bones in Strata's 19-bone skeleton (no heavy finger/facial data)
- Type: new feature (data + script enhancement)

## Approach
1. Create `animation/labels/cmu_action_labels.csv` with the new column (the file doesn't exist yet)
2. Add a `check_strata_compatibility()` function to `bvh_to_strata.py` — it already has `STRATA_BONES`, `CMU_TO_STRATA`, and `_build_bone_map()` which are the exact tools needed
3. The function evaluates a BVH by:
   - Retargeting and checking how many bones map successfully
   - Computing rotation deltas per-joint across frames to find bones with "significant motion"
   - If significant motion is concentrated in Strata-mapped bones → `yes`
   - If significant motion requires unmapped bones (fingers, facial) → `no`
4. Add a CLI entry point so users can run `python -m animation.scripts.bvh_to_strata --check-compat path/to/file.bvh`
5. Seed 50+ clip evaluations in the CSV (using CMU naming conventions from the BVH corpus)

## Files to Modify
- `animation/scripts/bvh_to_strata.py` — Add `check_strata_compatibility()` function + CLI `--check-compat` flag
- `animation/labels/cmu_action_labels.csv` — Create with headers + 50+ entries
- `tests/test_bvh_to_strata.py` — Add tests for the compatibility checker

## Risks & Edge Cases
- BVH files with only skeleton (no motion) — should return `yes` by default (no problematic motion data)
- Clips where finger motion is present but minimal — threshold-based, so configurable
- CMU files with unusual naming — `_build_bone_map()` already handles this, we piggyback on it
- Rotation delta threshold: too low = false negatives (marks everything as incompatible), too high = false positives. Default 1.0 degree seems reasonable.

## Open Questions
- None — dependencies (#44 BVH parser, #45 retargeting) are both complete

## Implementation Notes

### Key design decision: silently-ignored bones
`_build_bone_map()` silently ignores finger/toe bones (via `_SILENTLY_IGNORED_SUFFIXES`), so they never appear in the `unmapped` list. The compatibility checker needed to scan ALL non-Strata bones — not just the `unmapped` list. Solved by building a `non_strata` list inline that includes both unmapped bones AND silently-ignored bones with channels.

### CLI uses subcommands
Used `argparse` subcommands (`check-compat`) instead of flags (`--check-compat`) to leave room for future subcommands without cluttering the flag namespace. Usage: `python -m animation.scripts.bvh_to_strata check-compat *.bvh`

### CSV seeded with 78 entries
Used CMU subject numbering conventions. Categories marked `no` include: basketball (dribble/shoot/pass depend on finger articulation), gestures (wave/point/beckon), sign language, instrument playing, facial animation, typing.

### Files modified
- `animation/scripts/bvh_to_strata.py` — Added `CompatibilityResult`, `check_strata_compatibility()`, `_bone_has_significant_rotation()`, CLI via `main()`
- `animation/labels/cmu_action_labels.csv` — Created with 78 labeled entries
- `tests/test_bvh_to_strata.py` — Added 9 tests in `TestStrataCompatibility` class (28 total, up from 19)
- `docs/labeling-guide.md` — Added "Strata Compatibility" section with criteria, automated checking, and category examples
