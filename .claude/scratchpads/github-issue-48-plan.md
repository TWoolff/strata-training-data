# Issue #48: Build Live2D annotation review UI

## Understanding
- Build `pipeline/live2d_review_ui.py` — a Tkinter-based review tool for verifying/correcting auto-mapped Live2D fragment labels
- The auto-mapper (#47, `live2d_mapper.py`) maps ~70-80% correctly; ~20-30% needs manual correction
- Reviewer loads a model, sees fragments highlighted one at a time, confirms or corrects the label
- Updates `data/live2d/labels/live2d_mappings.csv` with `confirmed=manual` status
- PRD estimate: ~5 sec/fragment × 20 fragments/model × 400 models = ~11 hours total annotation

## Type
New feature (annotation tool)

## Approach

**Tkinter-based GUI** (Option B from issue) — provides visual feedback without external dependencies.

### UI Layout
- Left panel: Canvas displaying the composite image with fragment highlight overlay
- Right panel: Current fragment info (name, auto-assigned label) + label selection
- Bottom: Progress bar, navigation buttons (prev/next/skip), keyboard shortcuts

### Workflow
1. Load a model from its directory (pre-extracted fragment PNGs, same as renderer)
2. Display composite image as background
3. Iterate through each fragment:
   - Highlight the current fragment (color tint overlay, e.g. green semi-transparent)
   - Show the auto-assigned label from the mapper
   - User presses a key to confirm (Enter) or selects correct label from list
4. Save updates to `live2d_mappings.csv` with `confirmed=manual`
5. Track progress (fragments reviewed / total, models completed / total)
6. Resume from where left off (skip models/fragments already confirmed)

### Key Design Decisions
- **Fragment loading**: Reuse `_discover_fragment_images()` from `live2d_renderer.py` for discovery
- **Mapping**: Reuse `map_fragment()` from `live2d_mapper.py` for initial auto-labels
- **CSV format**: Same schema as `live2d_mapper.CSV_HEADER`: `model_id, fragment_name, strata_label, strata_region_id, confirmed`
- **Confirmed values**: `auto` (mapper), `pending` (unmapped by mapper), `manual` (human reviewed)
- **Region labels**: Use `REGION_NAMES` from config for the label selection UI
- **Keyboard shortcuts**: Number keys 0-9 for quick region selection, Enter to confirm, arrow keys for navigation
- **State persistence**: CSV is the state file — fragments with `confirmed=manual` are skipped on resume

### Architecture
- Pure Python + Tkinter (no Blender dependency), matching the pattern of `live2d_mapper.py` and `spine_parser.py`
- Single-file module `pipeline/live2d_review_ui.py`
- CLI entry point with `--model_dir` and `--csv_path` args
- Separation of data model (fragment loading, CSV I/O) from UI (Tkinter widgets)

## Files to Modify
- **NEW**: `pipeline/live2d_review_ui.py` — main review UI module
- **NEW**: `tests/test_live2d_review_ui.py` — tests for non-UI logic (CSV operations, fragment loading, state tracking)

## Risks & Edge Cases
- **No models downloaded yet**: The tool should handle empty directories gracefully
- **Missing fragment images**: Some models may have .model3.json but no pre-extracted PNGs (skip with warning)
- **Tkinter not available**: Some Python installs don't include tkinter; provide clear error message
- **Large images**: Fragment images could be large; resize for display while preserving original for reference
- **CSV consistency**: Multiple review sessions could interleave; use append-safe CSV I/O
- **Unmapped fragments**: Fragments with `confirmed=pending` need special attention in the UI (flag them)

## Open Questions
- None — the issue is well-specified. Implementing Option B (Tkinter) as recommended.

## Implementation Notes

### What was implemented
- `pipeline/live2d_review_ui.py` — Full Tkinter-based review UI with:
  - `ReviewState` dataclass for session navigation (advance, go_back, pending tracking, resume)
  - CSV load/create helpers that auto-map fragments when no CSV exists
  - Fragment confirmation and label correction functions
  - Image compositing and highlight overlay for visual review
  - Keyboard shortcuts: Enter=confirm, Space=apply selection, arrows=navigate, b=background, digits=region groups, Esc=save+quit
  - Progress tracking (reviewed/total fragments, models completed)
  - Resume support via CSV `confirmed=manual` filtering
- `tests/test_live2d_review_ui.py` — 22 tests covering ReviewState navigation, fragment update/confirm, CSV round-trip, fragment discovery, auto-mapper helper

### Design decisions
- Pure Python + Tkinter (no Blender dependency), matching live2d_mapper.py and spine_parser.py patterns
- Fragment discovery duplicates logic from `live2d_renderer._discover_fragment_images` intentionally — keeps review UI independent of renderer module
- Three `confirmed` status values: `auto` (mapper), `pending` (unmapped), `manual` (human reviewed)
- Saves CSV after every confirm/correction for crash safety
- `_auto_map_model` is a local helper (not reusing `live2d_mapper.map_model`) to avoid coupling test infrastructure

### Test results
- 22 new tests, all passing
- Full suite: 525 tests passing
- ruff check + format clean
