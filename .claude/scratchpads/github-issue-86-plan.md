# Issue #86: Multi-View Segmentation Consistency Validation

## Understanding
- Validate that segmentation masks and measurement extraction produce consistent results across different camera angles for the same character × pose
- Flag inconsistencies exceeding a configurable threshold (default: 10% deviation)
- Generate per-character consistency reports (pass/fail per region per angle pair)
- Aggregate statistics: which regions and angle pairs have highest inconsistency
- Type: **new feature** (new validation module)

## Approach

Create a new module `pipeline/multiview_validator.py` (not extending `validator.py`) because:
1. The existing `validator.py` validates individual files; multi-view consistency validates *relationships between* files
2. Different input data — needs measurement_2d JSON files grouped by character/pose, plus optionally 3D ground truth
3. Different output — consistency report per character, not per-file check results
4. Pure Python, no Blender dependency

**Three core consistency checks:**

### 1. Region Presence Consistency
For a character × pose across angles, verify that major body regions (head, chest, hips, etc.) are visible in the expected angles. Midline regions (head, neck, chest, spine, hips) should be visible from all angles. Limbs may be occluded in some views — only flag if visible in front but missing in a geometrically unreasonable angle.

### 2. Pixel Area Foreshortening Consistency
When a region rotates from front (0°) to side (90°), its apparent width should roughly follow: `width_at_angle ≈ width_front * |cos(θ)| + depth * |sin(θ)|`. Compare actual pixel counts against expected foreshortening within the threshold. Uses 2D measurement files for apparent dimensions.

### 3. Measurement vs. Ground Truth Consistency
If 3D ground truth is available, validate that front-view apparent width correlates with true_width and side-view apparent width correlates with true_depth. The ratio `apparent_width_front / apparent_width_side` should approximate `true_width / true_depth`.

**Report structure:**
- Per character × pose × region × angle-pair: pass/fail with deviation percentage
- Summary statistics: highest inconsistency regions and angle pairs
- JSON output for programmatic use + human-readable print

## Files to Modify
1. **NEW: `pipeline/multiview_validator.py`** — core validation logic
   - `validate_multiview_consistency()` — main entry point
   - `check_region_presence()` — region visibility across angles
   - `check_pixel_area_consistency()` — foreshortening check
   - `check_measurement_ratio()` — 2D vs 3D ground truth
   - Report dataclasses
2. **NEW: `tests/test_multiview_validator.py`** — comprehensive test suite
3. **`pipeline/__init__.py`** — no change needed (modules are imported directly)

## Risks & Edge Cases
- **Self-occlusion**: Arms/legs can hide behind torso in certain poses — should not be flagged as inconsistency. Handle by only checking regions visible in both angles of a pair.
- **Symmetrical characters**: Left/right swap between front and back views — use `FLIP_REGION_SWAP` to handle
- **Thin geometry**: Neck, shoulders may have very small pixel counts, leading to noisy ratios — use minimum pixel count threshold to skip unreliable measurements
- **Only front angle rendered**: If only front view exists, skip multi-view checks (no pairs to compare)
- **Extreme poses**: Poses with heavy bending may cause unusual foreshortening — threshold needs to be generous enough

## Open Questions
- None — the issue spec and PRD are clear enough to proceed.

## Implementation Notes
- Created as a standalone module (`pipeline/multiview_validator.py`) rather than extending `validator.py` — different concern (cross-file relationships vs single-file checks)
- Three checks implemented: `region_presence`, `pixel_area_consistency`, `measurement_ratio`
- `FLIP_REGION_SWAP` was initially imported for left/right back-view validation but removed during simplification as unused — the current checks don't need it since they compare same-named regions across angles
- `MIN_PIXEL_COUNT = 50` filters out noisy small regions from pixel area comparison
- `MIDLINE_REGIONS` set defines which regions must be visible from all angles (head, neck, chest, spine, hips)
- File discovery reads from `measurements_2d/` directory, grouping by `{char_id}_{pose}` key
- Ground truth loading is optional — measurement_ratio check only runs when 3D measurements exist
- 31 tests covering all three checks, report serialization, file I/O, integration with file discovery, character filtering, and custom thresholds
- All pre-existing tests continue passing (605 passed excluding pre-existing `test_proportion_clusterer` failures)
