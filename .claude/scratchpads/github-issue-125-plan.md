# Issue #125: Build segmentation dataset loader with dual layout support

## Understanding
- Build a PyTorch `Dataset` class for training the segmentation model
- Must support two directory layouts: flat (from pipeline/exporter.py) and per-example (from ingest adapters)
- Handle 20→22 class mapping (pipeline produces 0-19, Rust runtime expects 0-21 with 20=unused, 21=accessory)
- Integrate with existing transforms.py (flip_mask, normalize_imagenet) and split_loader.py
- Type: new feature

## Approach
- Single `SegmentationDataset` class in `training/data/segmentation_dataset.py`
- Auto-detect layout at init by checking for `images/` subdirectory vs `*/image.png` pattern
- Lazy loading: discover all example paths at init, load images on `__getitem__`
- Use PIL for image loading (matches pipeline conventions)
- Augmentations via torchvision.transforms for color jitter, rotation, scale
- Horizontal flip uses `flip_mask()` from transforms.py for L/R region swap
- Config from segmentation.yaml passed as optional dict

## Files to Modify
- **New: `training/data/segmentation_dataset.py`** — Main Dataset class
- **New: `tests/test_segmentation_dataset.py`** — Tests

## Key Design Decisions
1. **Layout detection**: Check if `images/` subdir exists → flat layout. Otherwise glob for `*/image.png` → per-example layout.
2. **Mask filename mapping (flat layout)**: Image `char_pose_00_flat.png` → mask `char_pose_00.png` (strip style suffix via `_pose_\d+` pattern)
3. **Draw order**: Optional — load if exists, return zeros + `has_draw_order=False` if not
4. **Accessory class (21)**: Check `metadata.json` for `has_accessories: true` if available; otherwise leave as pipeline IDs
5. **Confidence target**: 1.0 where mask > 0 (foreground), 0.0 for background pixels
6. **Augmentation pipeline**: Rotation/scale via affine transform applied to both image and mask (nearest for mask, bilinear for image)

## Risks & Edge Cases
- Flat layout: multiple style variants per pose — each image maps to same mask (by design)
- Per-example layout: some examples may lack segmentation.png (skip gracefully)
- Draw order may be absent in both layouts
- metadata.json may not exist (treat as no accessories)
- Empty dataset directories should produce empty dataset, not crash
- Image may be RGBA — need to convert to RGB (drop alpha)

## Open Questions
- None — issue specification is comprehensive
