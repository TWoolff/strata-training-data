# Issue #156: Ingest anime_instance_seg dataset (98K images, COCO annotations)

## Understanding
- This is a **run + upload task** — the adapter is already built and registered
- Dataset: dreMaz/AnimeInstanceSegmentationDataset — 91,082 train + 7,496 val images
- Data is already on external HD at `/Volumes/TAMWoolff/data/preprocessed/anime_instance_seg/anime_instance_dataset/`
- Adapter: `ingest/anime_instance_seg_adapter.py` registered as `--adapter anime_instance_seg`
- Adapter decodes COCO RLE masks → binary foreground masks (255=char, 0=bg)
- Output: ~98K example directories, each with `image.png`, `segmentation.png`, `metadata.json`

## Approach
1. Test with small batch (5 images) to verify adapter works end-to-end
2. Run full train split (~91K images)
3. Run full val split (~7.5K images)
4. Verify output counts and file format
5. Upload to Hetzner bucket under `anime_instance_seg/` prefix
6. Update checklist line 391

## Key Detail: Input Path
- The adapter's `convert_split()` expects `dataset_dir` pointing to `anime_instance_dataset/` (the dir containing `train/`, `val/`, `annotations/`)
- HD path: `/Volumes/TAMWoolff/data/preprocessed/anime_instance_seg/anime_instance_dataset/`
- NOT the parent `/Volumes/TAMWoolff/data/preprocessed/anime_instance_seg/`

## Files to Modify
- `.claude/prd/strata-training-data-checklist.md` — line 391, check the box and add counts
- No code changes needed — adapter already works

## Risks & Edge Cases
- Large dataset (~98K images) — ingestion will take significant time
- External HD read speeds may bottleneck I/O
- Some images may have no annotations or empty masks → adapter skips these (expected)
- Disk space: ~98K examples × 3 files × ~200KB avg ≈ ~60GB output needed

## Open Questions
- None — straightforward run + upload task
