---
name: validate-dataset
description: Run automated validation checks on generated dataset output. Checks mask correctness, joint bounds, file pairing, resolution, and region distribution. Use after batch generation or to diagnose dataset quality issues.
user-invokable: true
argument-hint: "<dataset directory path (default: ./output/segmentation/)>"
---

# Validate Dataset

Run comprehensive validation checks on the generated dataset: $ARGUMENTS

If no path specified, default to `./output/segmentation/`.

## Quick Validation (Preferred)

The project has a dedicated validation CLI (`run_validation.py` using `pipeline/validator.py`). Use this first:

```bash
# Validate entire dataset
python run_validation.py --dataset_dir ./output/segmentation/

# Validate specific characters
python run_validation.py --dataset_dir ./output/segmentation/ --characters mixamo_001,mixamo_002

# Validate and save JSON report
python run_validation.py --dataset_dir ./output/segmentation/ --save_report

# Custom resolution (default: 512)
python run_validation.py --dataset_dir ./output/segmentation/ --resolution 1024
```

The CLI validator checks: mask completeness, mask uniqueness, joint bounds, joint count (19 per pose), file pairing, resolution, and region distribution. It imports `NUM_REGIONS` (20) and `NUM_JOINT_REGIONS` (19) from `pipeline/config.py`.

## Manual Validation Checks

These supplement the automated CLI tool for deeper investigation.

### 1. File Pairing
Every image in `images/` must have:
- A corresponding mask in `masks/` (shared across styles — strip the style suffix to match)
- A corresponding joint file in `joints/` (same base name as mask)
- A source metadata file in `sources/` for the character

```python
# Naming convention:
# Image:  images/{source}_{id}_pose_{nn}_{style}.png
# Mask:   masks/{source}_{id}_pose_{nn}.png  (shared across styles)
# Joints: joints/{source}_{id}_pose_{nn}.json
# Source: sources/{source}_{id}.json
```

Report any orphaned files (images without masks, masks without joints, etc.).

### 2. Resolution Check
- All images must be exactly 512x512 pixels
- All masks must be exactly 512x512 pixels
- Images must be RGBA PNG
- Masks must be 8-bit grayscale (mode 'L') PNG

### 3. Mask Validity
For each mask:
- Every pixel value must be in range 0-19 (valid region IDs — 20 total: background + 19 body regions)
- No mask should be all-zero (would mean entire character is background — mapping failed)
- No mask should be all-one-value (would mean everything mapped to one region — mapping failed)
- At least 3 distinct regions should be present (head + torso + limbs minimum)

### 4. Mask-Image Alignment
For each image-mask pair:
- Every non-transparent pixel in the image (alpha > 0) should have a non-zero region in the mask
- Every non-zero pixel in the mask should correspond to a non-transparent pixel in the image
- Report alignment percentage (target: >98%)

### 5. Joint Bounds
For each joint JSON file:
- Must contain a `joints` key with entries for all 19 body regions (regions 1-19, excluding background)
- All joint `position` values must be within image bounds [0, 512) x [0, 512)
- Each joint must have `position` (array of 2 ints) and `visible` (boolean) fields
- Must contain `image_size` field matching [512, 512]

### 6. Region Distribution
For each mask:
- No single region should dominate >60% of non-zero pixels
- Report the per-region pixel distribution as a percentage
- Flag any characters where a body part is unexpectedly absent (e.g., no legs in a standing pose)

### 7. Weight Files (if present)
For each weight JSON in `weights/`:
- Must have valid structure with vertex positions and bone weight maps
- All weights must be positive floats
- Weights per vertex should sum to approximately 1.0 (within 0.01 tolerance)

### 8. Manifest & Splits
- `manifest.json` must exist and be valid JSON
- `class_map.json` must map all 20 region IDs (0-19) to region names
- `splits.json` must exist with `train`, `val`, `test` keys
- Splits must be by character (all poses/styles of one character in same split)
- Split ratio should approximate 80/10/10 (within 5% tolerance)

## Output Report

Generate a summary report:

```
Dataset Validation Report
=========================
Path: ./output/segmentation/
Date: YYYY-MM-DD

Files:
  Images:  N
  Masks:   N
  Joints:  N
  Weights: N
  Sources: N

Checks:
  [PASS] File pairing — all N images have matching masks and joints
  [FAIL] Resolution — 3 images are not 512x512: <list>
  [PASS] Mask validity — all masks have valid region IDs (0-19)
  [WARN] Mask alignment — 2 masks have <95% alignment: <list>
  [PASS] Joint bounds — all 19 joints within image bounds
  [WARN] Region distribution — 1 character has >60% single region: <list>
  [PASS] Splits — 80.2% train / 9.8% val / 10.0% test

Summary: N passed, N warnings, N failures
```

List all failures with file paths so they can be investigated and fixed.
