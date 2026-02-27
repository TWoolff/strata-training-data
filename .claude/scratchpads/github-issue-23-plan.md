# Issue #23: Implement automated dataset validation checks

## Understanding
- Build 7 automated validation checks that verify dataset integrity after generation
- Checks: mask completeness, mask uniqueness, joint bounds, joint count, file pairing, resolution, region distribution
- Must run independently of Blender (pure Python + Pillow + NumPy + JSON)
- Output: validation report to stdout + optional `validation_report.json`
- Exit code: 0 = all pass, 1 = failures found
- Support full dataset or character subset validation

## Approach
- Create `pipeline/validator.py` as a pure Python module (no Blender dependency, like `config.py` and `exporter.py`)
- Each of the 7 checks is a standalone function that takes a file path and returns pass/fail + details
- A top-level `validate_dataset()` function orchestrates all checks, iterates files, and builds the report
- Create `run_validation.py` as a standalone CLI entry point (mirrors `run_pipeline.py` pattern)
- File scanning: discover all images via glob, derive expected mask/joint paths from image filenames
- Use the existing naming conventions from `exporter.py` to parse filenames

### Design decisions
- Check functions return structured results (dataclass or typed dict) rather than just bool
- Report aggregation: per-check summary (pass count, fail count, failing files list)
- Character subset support via `--characters` CLI arg (filter by char_id prefix)
- Use `RENDER_RESOLUTION` from config as default expected resolution
- Joint count: 19 (NUM_JOINT_REGIONS from config), not 17 as the issue initially says — the codebase defines 19 body regions

## Files to Modify
- `pipeline/validator.py` — NEW: all 7 validation checks + report generation
- `run_validation.py` — NEW: standalone CLI entry point

## Risks & Edge Cases
- Large datasets (36K+ images) — use efficient numpy operations, avoid loading all masks into memory at once
- Missing directories — handle gracefully if images/masks/joints dirs don't exist
- Corrupt PNG files — wrap in try/except, report as failures
- Augmented files (flipped/scaled) have different suffix patterns — handle `_flip`/`_s085` suffixes
- Mask pixel values can be 0 (background) which is valid — only non-transparent image pixels need non-zero mask regions
- Joint positions clamped to [0, 511] — check against image_size field in JSON, not hardcoded 512
- Empty masks (all background/transparent) — these should still pass mask uniqueness if there's only background

## Open Questions
- None — requirements are clear from the issue, PRD, and existing codebase patterns

## Implementation Notes
- Created `pipeline/validator.py` with 3 dataclasses (`CheckFailure`, `CheckSummary`, `ValidationReport`) and 6 individual check functions + 1 orchestrator
- Check functions return `tuple[bool, str]` (passed, detail) — simple and composable
- `validate_dataset()` discovers files via glob, groups images by pose key, runs all checks per-pose
- Mask completeness only checked against the first image per pose (all style variants share the same mask)
- Resolution check covers both images AND masks
- File pairing records one pass/fail per image (for mask existence) and one per pose (for joint existence)
- Joint count validates both count (19) and exact name set against `EXPECTED_JOINT_NAMES`
- Joint bounds reads `image_size` from the JSON rather than hardcoding 512
- Region distribution uses vectorized `np.bincount` for performance on large masks
- Also updated `ruff.toml` with per-file E402 ignore for `run_*.py` (pre-existing pattern from `run_pipeline.py`)
- Code-simplifier pass: removed unused params from `check_joint_count`/`check_joint_bounds`, fixed double regex match with walrus operator, simplified regex to work on stems
