# Issue #51: Build PSD layer extractor for training data

## Understanding
- Build a script to extract layers from Photoshop PSD files, map layer names to Strata labels, and generate segmentation masks
- Type: **new feature** (standalone module, no Blender dependency)
- PSD files are an opportunistic data source — real artist PSDs provide style diversity even in small quantities
- Only PSDs with body-part-separated layers are useful (not rendering-concern layers like "lineart", "color")
- Expected volume: 50–100 files, each processed in seconds

## Approach
- Follow the `live2d_mapper.py` pattern closely — regex-based layer name mapping, dataclass results, CSV export for review
- Follow the `spine_parser.py` pattern for full processing — composite image + mask generation, batch directory processing
- Use `psd-tools` library for PSD parsing (handles layer extraction, compositing, blend modes)
- Pure Python, no Blender dependency — testable outside Blender
- Output through `exporter.py` for consistent file naming and directory structure

### Key Design Decisions
1. **Layer name patterns** → Add `PSD_LAYER_PATTERNS` to `config.py` (same format as `LIVE2D_FRAGMENT_PATTERNS`)
2. **Group layers** → Recurse into layer groups to find leaf layers; group names may also provide context
3. **Adjustment layers / blend modes** → Skip adjustment layers (Curves, Levels, etc.); use psd-tools' composite for blend mode handling
4. **Mask generation** → For each mapped leaf layer, render its alpha channel at the assigned region ID value
5. **Unmapped layers** → Flag for manual review via CSV export (same pattern as live2d_mapper)
6. **Single-pose assumption** → Each PSD = one pose (pose_00), unlike 3D characters with multiple poses

## Files to Modify
1. **`pipeline/config.py`** — Add `PSD_LAYER_PATTERNS: list[tuple[str, str]]` constant
2. **`pipeline/psd_extractor.py`** — NEW: Main module (~400 lines)
   - Data structures: `LayerMapping`, `PSDMapping`
   - Core: `map_layer()`, `map_psd()`
   - Processing: `process_psd_file()`, `process_psd_directory()`
   - CSV: `export_csv()`, `load_csv()`
   - Helper: `region_summary()`
3. **`requirements.txt`** — Add `psd-tools>=2.0`
4. **`tests/test_psd_extractor.py`** — NEW: Tests for mapping + processing

## Risks & Edge Cases
- **Non-body-part layers**: Most PSDs have rendering-concern layers (lineart, color, shading) — these should map to UNMAPPED and be flagged
- **Nested groups**: Layer groups can be deeply nested — need recursive traversal
- **Blend modes**: Composite image quality depends on psd-tools' blend mode support (generally good for common modes)
- **Layer visibility**: Some layers may be hidden by default — should process only visible layers
- **Large PSDs**: Some PSDs can be very large (100MB+) — processing should be memory-efficient
- **No body-part layers found**: If zero layers map, warn and skip (don't output empty masks)
- **Overlapping regions**: When multiple layers map to the same region, they should combine (logical OR of alpha)
- **Layer order = draw order**: PSD layer order from bottom to top provides natural draw order

## Open Questions
- None — the issue is well-specified and the patterns are clear from existing code

## Implementation Notes
- Implemented as planned — all files match the approach above
- `pipeline/psd_extractor.py` is 647 lines, closely mirroring `live2d_mapper.py` for mapping and `spine_parser.py` for processing
- `PSD_LAYER_PATTERNS` in `config.py` includes both body-part patterns AND rendering-concern patterns (lineart, shadow, etc.) mapped to background
- Adjustment layers (Curves, Levels, etc.) are skipped entirely via `_SKIP_LAYER_KINDS` frozenset
- `mapped_count` intentionally excludes background (region 0) — only counts body-region mappings (regions 1-19)
- 117 tests all passing, covering mapping, CSV round-trip, group exclusion, and visibility handling
- `psd-tools` import is deferred (inside `process_psd_file`) so the module loads without it for mapping-only use
