# Issue #171: Download InstaOrder dataset and build draw order prediction training data

## Understanding
- InstaOrder provides 2.9M pairwise depth/occlusion ordering annotations across 101K COCO images
- Annotations are in JSON format: `InstaOrder_train2017.json` and `InstaOrder_val2017.json`
- Each annotation contains per-image pairwise orderings between instances:
  - **depth**: `"idx1 < idx2"` (idx1 closer than idx2), `"idx1 = idx2"` (same depth), with `overlap` bool and `count`
  - **occlusion**: `"idx1 < idx2"` (idx1 occludes idx2), `"idx1 & idx2"` (mutual)
- Instance IDs reference COCO `instances_train2017.json` / `instances_val2017.json` for masks + bboxes
- License: CC BY-SA (commercially usable)
- Images are natural photos (not illustrations) but depth ordering semantics generalize

## Approach
The adapter converts InstaOrder's pairwise depth ordering annotations into per-pixel draw order maps:

1. **Load InstaOrder annotations** — parse the JSON to get per-image pairwise depth orderings
2. **Load COCO instance annotations** — get instance masks (RLE-encoded) and bboxes
3. **For each image**: resolve pairwise depth orderings into a global instance ranking via topological sort
4. **Generate per-pixel draw order map**: assign each instance's pixels a normalized depth value (0=back, 255=front)
5. **Output**: `image.png` (COCO image resized to 512×512), `draw_order.png` (per-pixel depth), `metadata.json`

This is the first ingest adapter that produces draw order data — a key differentiator.

### Draw order map generation logic:
- Parse pairwise `depth` orderings to build a DAG of instance depth relationships
- Topologically sort instances from back to front
- Assign each instance a depth value normalized to [0, 255] based on its rank
- Paint each instance's COCO mask pixels with its depth value
- Background pixels stay at 0

### Simplification for v1:
- Only use `depth` orderings (not `occlusion` — occlusion is about which object hides which, depth is about distance)
- Skip images with cyclic depth orderings (can't topologically sort)
- Skip images with fewer than 2 instances (no pairwise ordering to learn from)

## Files to Modify
1. **NEW** `ingest/instaorder_adapter.py` — main adapter
2. **NEW** `tests/test_instaorder_adapter.py` — ≥8 tests
3. **EDIT** `run_ingest.py` — register `instaorder` adapter
4. **EDIT** `.claude/prd/strata-training-data-checklist.md` — mark items complete

## Risks & Edge Cases
- Cyclic depth orderings: some images may have A<B, B<C, C<A → skip these
- COCO images may not all be available locally → graceful skip with counter
- Large annotation files (~100K images) → streaming/lazy loading
- pycocotools dependency for RLE mask decoding → already used by anime_instance_seg_adapter
- Some instances may have no depth ordering pairs → assign middle depth (127)
- Very many instances in one image → topological sort handles this fine

## Open Questions
- COCO 2017 images: train split is Flickr (various licenses), val split is CC BY 4.0. The adapter should note this in metadata.
- Do we need COCO images at all during adapter runtime? Yes — we need to resize and output them as `image.png`.
- pycocotools for RLE decoding — already a project dependency (used in anime_instance_seg_adapter).

## Implementation Notes
- **Adapter built**: `ingest/instaorder_adapter.py` (490 lines)
- **Tests**: 23 tests in `tests/test_instaorder_adapter.py`, all passing
- **Registered**: Added `instaorder` to `run_ingest.py` dispatcher + choices
- **Key design decisions**:
  - Uses Kahn's algorithm for topological sort (handles cycles gracefully → returns None)
  - Depth values normalized: rank 0 = back (value 1), rank n-1 = front (value 255), bg = 0
  - Minimum depth value for foreground instances is 1 (to distinguish from background 0)
  - Supports both COCO RLE and polygon segmentation formats
  - Uses `--split` CLI arg to select train/val (defaults to val)
  - Images with cyclic orderings or <2 instances are skipped and counted separately
- **First adapter with `has_draw_order: True`** — all other adapters output fg masks / joints only
- **pycocotools dependency**: Already in project, used for both RLE and polygon→RLE mask decoding
