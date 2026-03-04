# Issue #161: Download CoNR dataset and build Strata segmentation adapter

## Understanding
- Build an ingest adapter for the CoNR (Collaborative Neural Rendering) dataset
- CoNR has ~700K anime character images (hand-drawn from Danbooru + synthesized from MMD 3D models)
- The dataset is CC-BY 4.0 licensed
- Type: new feature (ingest adapter)

### Dataset Structure (from research)
The CoNR_Dataset (P2Oileen/CoNR_Dataset on GitHub) has three components:
1. **Annotations** (`CoNR_Dataset_annotation_only.tar.gz`): `.npz` files with body region labels (values 1-9). Original Danbooru images NOT included — must reconstruct download URL from filename hash: `https://cdn.donmai.us/original/{first2chars}/{next2chars}/{hash}.jpg`
2. **3D Models** (`CoNR_Dataset_3Dmodel_only_a*`): PMX/PMD MikuMikuDance models
3. **Motion** (`CoNR_Dataset_motion_only_a*`): VMD motion files

**Critical insight**: The dataset does NOT ship pre-rendered images. For Danbooru images, only annotation `.npz` files are provided. The adapter must:
- Read `.npz` annotation files
- Reconstruct Danbooru CDN URLs from the filename hash
- Download original images from Danbooru CDN
- Extract the 9-class body region segmentation from the `.npz`

**NPZ format**: `np.load("hash.jpg.npz")['label']` → shape `(H, W)` with integer values 1-9 (body surface regions). This is a coarse segmentation (9 classes vs Strata's 22), NOT mappable to Strata's anatomical regions since the classes are unlabeled surface correspondence IDs, not body parts.

### What Strata can use
- The images themselves (diverse anime character illustrations) — valuable for style diversity
- The foreground/background segmentation (label > 0 = foreground)
- The 9-class segmentation is NOT useful for Strata's 22-region taxonomy (classes are surface IDs, not body parts)

## Approach
Build a simple image-only adapter (like fbanimehq_adapter) that:
1. Discovers `.npz` annotation files in the annotation directory
2. Extracts foreground mask from annotation (label > 0)
3. Either downloads images from Danbooru CDN OR reads pre-downloaded images from a local directory
4. Resizes to 512x512 with aspect-ratio-preserving padding
5. Saves image.png + segmentation.png (binary fg/bg) + metadata.json

**Design decision**: Support two modes:
- **Local mode** (primary): User has pre-downloaded images alongside `.npz` files. Adapter reads `{hash}.jpg` images from a companion image directory.
- **Download mode**: Not implemented in v1 — downloading 700K images from Danbooru CDN is impractical in an adapter. Document in README that images must be pre-downloaded.

Since the 9-class segmentation is unlabeled surface IDs (not body parts), we'll only extract a binary foreground mask. The adapter pattern matches anime_seg_adapter (images + binary fg mask).

## Files to Modify
- **NEW** `ingest/conr_adapter.py` — Main adapter module
- **NEW** `tests/test_conr_adapter.py` — Test suite (≥8 tests)
- **EDIT** `run_ingest.py` — Register `conr` adapter

## Risks & Edge Cases
- Images may not exist locally (need clear error handling and skip counting)
- `.npz` files reference Danbooru images that may be deleted/unavailable
- Some images may have unusual dimensions or be corrupted
- The annotation labels use values 1-9; value 0 = background. Simple threshold works for fg mask
- Large dataset (~700K annotations) — progress logging essential

## Open Questions
- None blocking — the adapter pattern is well-established and this is a straightforward image+fg_mask adapter

## Implementation Notes

### What was implemented
- `ingest/conr_adapter.py` — Full adapter following fbanimehq_adapter pattern
- `tests/test_conr_adapter.py` — 32 tests across 8 test classes
- `run_ingest.py` — Registered `conr` adapter with `_run_conr` dispatch function

### Key design decisions
1. **Binary foreground mask only**: The 9-class CoNR labels are unlabeled surface correspondence IDs (not body parts), so we only extract fg/bg binary mask (values 1-9 = fg). Value 255 (present in ~55% of files) marks unlabeled/uncertain pixels and is treated as background.
2. **Local images required**: No Danbooru CDN download support in v1. User must pre-download images into `images/` directory alongside `annotation/`. This avoids rate limiting and keeps the adapter pure/offline.
3. **Flexible directory discovery**: Supports both `input_dir/annotation/` + `input_dir/images/` layout and pointing directly at annotation dir with sibling `images/`.
4. **Three-state return from `convert_example`**: Returns `True` (saved), `False` (skipped/error), or `None` (image not found) — enables `images_missing` counter in `AdapterResult`.
5. **Nearest-neighbor mask resize**: Preserves binary mask values (0/255) without interpolation artifacts.

### Notable fixes during implementation
- Added `pickle.UnpicklingError` to exception handling in `load_annotation()` — numpy throws this on corrupt `.npz` files
- Simplified `annotation_hash()` to use `Path(npz_path.stem).stem` instead of manual string slicing

### Dataset assessment (from real download)
- Downloaded all 10 files to `/Volumes/TAMWoolff/data/preprocessed/conr/CoNR_Dataset/` (~8.5 GB)
- Annotation archive contains 3,669 `.npz` files (not 700K — that's total including synthesized 3D renders)
- Real `.npz` format: key `label`, uint8, values 0 (bg), 1-9 (body regions), 255 (unlabeled)
- Fixed `label_to_fg_mask` to treat value 255 as background (was incorrectly using `label > 0`)
- Danbooru CDN images must be downloaded separately; some are 404 (deleted from Danbooru)
- Verified end-to-end: adapter produces correct 512x512 RGBA image + grayscale fg mask + metadata
- License note: README says "research use only" (not CC-BY 4.0 as issue stated)

### Remaining work
- [ ] Download Danbooru images for all 3,669 annotations (write download script)
- [ ] Run full ingest on downloaded images
- [ ] Upload converted output to bucket under `conr/` prefix
- [ ] Update checklist in `.claude/prd/strata-training-data-checklist.md`
