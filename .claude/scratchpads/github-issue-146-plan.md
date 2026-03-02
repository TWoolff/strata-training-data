# Issue #146: Build .moc3 binary parser and atlas fragment extractor

## Understanding
- Live2D models downloaded from GitHub (100+) have `.moc3` binary files + texture atlases but no pre-extracted fragment PNGs
- The existing `live2d_renderer.py` is fully built (compositing, masks, draw order, joints, augmentation) but discovers **zero fragments** because all models are atlas-only
- Need to parse `.moc3` binary to extract per-ArtMesh UV coordinates and triangle indices, then rasterize each mesh's region from the texture atlas
- This is a **new feature** (`.moc3` parser) + **modification** (renderer integration) + **config update** (Chinese patterns)

## Approach
1. **New module `pipeline/moc3_parser.py`** — Pure Python binary parser using `struct` module
   - Dataclasses: `Moc3ArtMesh`, `Moc3Model`
   - Single public function: `parse_moc3(path) -> Moc3Model`
   - Reads SOT (section offset table), CIT (count info table), then per-mesh arrays
   - Returns parsed ArtMesh list with UVs, triangle indices, draw_order, texture_no, parent part ID/name

2. **Modify `pipeline/live2d_renderer.py`** — Add `_extract_fragments_from_moc3()`
   - Called when `_discover_fragment_images()` returns empty but .moc3 + textures exist
   - For each ArtMesh: rasterize triangles from atlas using cv2.fillConvexPoly
   - Returns `list[tuple[str, Image.Image, int]]` (same format as existing fragment loading)
   - Fragment images are full-atlas-sized to preserve coordinate system

3. **Update `pipeline/config.py`** — Add Chinese regex patterns to `LIVE2D_FRAGMENT_PATTERNS`
   - 91+ models from CNbysec repo use Chinese naming in CDI display names
   - Patterns for: 头/头发/刘海 (head), 脖子/颈 (neck), 胸/上身 (chest), etc.

## Files to Modify
- `pipeline/moc3_parser.py` — **NEW** (~300 LOC)
- `pipeline/live2d_renderer.py` — Modify (~80 LOC): add `_extract_fragments_from_moc3()`, update `process_live2d_model()`
- `pipeline/config.py` — Modify (~20 LOC): add Chinese patterns to `LIVE2D_FRAGMENT_PATTERNS`
- `tests/test_moc3_parser.py` — **NEW** (~200 LOC)
- `tests/test_live2d_renderer.py` — Modify (~80 LOC): add tests for .moc3 extraction path

## Key Technical Details
- .moc3 binary: magic `b'MOC3'`, version byte at offset 4, SOT at 0x40 (160 uint32s)
- CIT at SOT[0]: parts_count=CIT[0], artmesh_count=CIT[4]
- SOT[33]=ArtMesh IDs (64-byte UTF-8), SOT[3]=Part IDs (64-byte UTF-8)
- SOT[34]=parentPartIndex, SOT[35]=uvBeginIndex, SOT[36]=vertexCount
- SOT[40]=drawOrder, SOT[41]=textureNo, SOT[45]=posIndexBegin, SOT[46]=posIndexCount
- SOT[78]=UV float32 pairs (0-1 range), SOT[79]=triangle uint16 indices
- Triangle indices can exceed vertex_count — apply `index % vertex_count`
- Verified on real xinnong_5.moc3: 1056 artmeshes, UVs in 0-1 range, valid triangles

## Fragment Rasterization Strategy
- Create full-atlas-sized RGBA canvas per ArtMesh (preserves coordinate system for compositor)
- For each triangle: use cv2.fillPoly to create a mask, then bulk-copy pixels from atlas
- Apply opacity from .moc3 data (multiply alpha channel)
- Skip meshes with vertex_count == 0 (trigger boxes, atmosphere effects)

## Risks & Edge Cases
- Truncated/corrupt .moc3 files — wrap in try/except, return None
- Meshes with 0 vertices — skip silently
- Missing CDI file — fall back to ArtMesh ID + Part ID for name resolution
- Triangle indices exceeding vertex_count — apply modulo
- Opacity 0 meshes — skip (not visible)
- Multiple texture pages — index by texture_no field
- Version differences (1-4) — confirmed identical SOT layout, no branching needed

## Open Questions
- None — format verified against real data, approach matches existing renderer patterns
