# Issue #27: Set up manual annotation pipeline with Label Studio

## Understanding
- Set up Label Studio with the Strata 20-region label set (19 body + background) for manual segmentation + keypoint annotation of 2D hand-drawn characters
- Build import scripts to load images into Label Studio and export scripts to convert annotations to the standard dataset format
- This bridges the gap between synthetic 3D data and real 2D art diversity
- Type: new feature / infrastructure

## Approach
- Create `annotation/` directory with scripts for Label Studio setup, import, and export
- Reuse existing `pipeline.exporter` and `pipeline.config` modules directly (they're pure Python, no Blender dependency)
- Label Studio config uses `<PolygonLabels>` for segmentation and `<KeyPointLabels>` for joints
- Export script parses Label Studio JSON export, rasterizes polygons to masks with `cv2.fillPoly()`, and extracts keypoint positions
- All output follows the exact same format as the synthetic pipeline so the validator passes

### Key design decisions:
1. **PolygonLabels over BrushLabels** — polygons give precise region boundaries and are easier to rasterize programmatically
2. **Standalone scripts** — no dependency on Label Studio SDK at runtime for export; just parse the JSON export file
3. **Reuse pipeline.exporter** — `save_mask()`, `save_joints()`, `save_source_metadata()`, `save_class_map()`, `ensure_output_dirs()` are all Blender-free
4. **Character ID convention** — `manual_{NNN}` format (e.g., `manual_001`)

## Files to Create
- `annotation/label_studio_config.xml` — Template with 19 body region PolygonLabels + 19 KeyPointLabels
- `annotation/import_images.py` — Resize images to 512×512, batch import into Label Studio via API
- `annotation/export_annotations.py` — Parse LS JSON → rasterize polygons → save masks + joints + source metadata
- `docs/annotation-guide.md` — Step-by-step guide for annotators

## Files to Modify
- `requirements.txt` — Add `label-studio-sdk>=1.0` (for import script API calls)

## Risks & Edge Cases
- Label Studio JSON export format varies by task type (polygon vs brush) — need to handle both polygon and rectangle annotations
- Image aspect ratios — non-square images need padding (letterbox with transparency) to maintain 512×512
- Overlapping polygons — later polygons should overwrite earlier ones (last-wins for region assignment)
- Missing keypoints — annotator might not mark all 19; export must handle partial annotation with visible=false
- Region color assignment in Label Studio XML must exactly match REGION_COLORS from config.py

## Open Questions
- None — requirements are clear from the issue and PRD

## Implementation Notes

### What was implemented
- `annotation/label_studio_config.xml` — PolygonLabels + KeyPointLabels with all 19 body regions, colors matching `REGION_COLORS` exactly (hex-converted)
- `annotation/import_images.py` — Resize + letterbox to 512x512 RGBA, upload via Label Studio REST API (stdlib `urllib.request`, no SDK dependency), or generate local task JSON
- `annotation/export_annotations.py` — Parse LS JSON export, rasterize polygons via `cv2.fillPoly()`, build joint data with `JOINT_BBOX_PADDING`, save via `pipeline.exporter` functions
- `annotation/__init__.py` — Package init
- `docs/annotation-guide.md` — Full annotator walkthrough with quality checklist
- `requirements.txt` — Added comment about `label-studio` being an optional dependency

### Design decisions made during implementation
- Used `urllib.request` instead of `label-studio-sdk` for API upload — avoids a heavyweight dependency for a simple multipart POST
- Export script takes the most recent annotation per task (last in annotations list) — handles re-annotation gracefully
- Image lookup falls back to stem matching when exact filename doesn't match — handles LS URL mangling
- Bbox padding uses `JOINT_BBOX_PADDING` from config.py for consistency with `joint_extractor.py`

### Follow-up work
- Validation can be run immediately with `python run_validation.py --dataset_dir <output>` — no annotation-specific validator needed
- Consider adding `manual` to source inference tables in `manifest.py` / `splitter.py` if those modules need to recognize the source type
