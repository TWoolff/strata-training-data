# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's AI models (semantic segmentation, joint prediction, weight prediction, drawn pose estimation). Runs independently of the Strata codebase.

## What This Project Does

Takes rigged 3D characters (.fbx) and produces:
- **Color images** rendered in multiple art styles (flat, cel, pixel art, painterly, sketch, unlit)
- **Segmentation masks** (8-bit grayscale PNG, pixel value = region ID 0–19)
- **Joint position JSON** (bone heads projected to 2D screen coords)
- **Weight map JSON** (per-vertex bone weights)

One character × 20 poses × 6 styles = 120 training examples. Target: 300+ characters → 36,000+ images.

## Project Structure

```
strata-training-data/
├── CLAUDE.md
├── generate_dataset.py        # Main entry point, orchestrates pipeline
├── importer.py                # Load FBX, normalize scale/position
├── bone_mapper.py             # Map bones to Strata's 19 regions
├── pose_applicator.py         # Apply animation keyframes
├── renderer.py                # Render color + segmentation passes
├── style_augmentor.py         # Post-render style transforms (pixel art, painterly, sketch)
├── joint_extractor.py         # Project bone positions to 2D → JSON
├── weight_extractor.py        # Extract per-vertex bone weights → JSON
├── exporter.py                # Save images, masks, JSON metadata
├── config.py                  # Region colors, bone mappings, defaults
├── source_characters/         # Input .fbx files
├── pose_library/              # Animation .fbx clips
└── dataset/                   # Generated output
    ├── manifest.json
    ├── class_map.json
    ├── splits.json
    ├── images/                # {char}_{pose}_{style}.png (512×512)
    ├── masks/                 # {char}_{pose}.png (shared across styles)
    ├── joints/                # {char}_{pose}.json
    ├── weights/               # {char}_{pose}.json
    └── sources/               # Per-character metadata + license info
```

## Running the Pipeline

```bash
blender --background --python generate_dataset.py -- \
  --input_dir ./source_characters/ \
  --pose_dir ./pose_library/ \
  --output_dir ./dataset/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 \
  --poses_per_character 20
```

Requires Blender 4.0+ (uses bundled Python 3.10+). No GPU needed — EEVEE flat shading runs on CPU.

## Key Technical Details

### Strata Standard Skeleton (19 body regions + background)

| ID | Region | ID | Region |
|----|--------|----|--------|
| 0 | background | 10 | lower_arm_r |
| 1 | head | 11 | hand_r |
| 2 | neck | 12 | upper_leg_l |
| 3 | chest | 13 | lower_leg_l |
| 4 | spine | 14 | foot_l |
| 5 | hips | 15 | upper_leg_r |
| 6 | upper_arm_l | 16 | lower_leg_r |
| 7 | lower_arm_l | 17 | foot_r |
| 8 | hand_l | 18 | shoulder_l |
| 9 | upper_arm_r | 19 | shoulder_r |

### Bone Mapping Priority

1. Exact match → 2. Prefix match → 3. Substring match (case-insensitive) → 4. Manual override JSON

Mixamo characters should map 100% automatically. Non-Mixamo ~80% auto, ~20% manual via per-character override JSON.

### Segmentation Mask Rendering

- 20 Emission-only materials (one per region), no lighting
- Each mesh face assigned to region by majority vote of vertex bone weights
- No anti-aliasing, nearest-neighbor sampling — each pixel = exactly one region ID
- Output: 8-bit single-channel grayscale PNG (pixel value = region ID)

### Render Setup

- Orthographic camera (no perspective distortion)
- Front-facing, auto-framed to character bounding box + 10% padding
- 512×512 output resolution
- Transparent background (alpha channel)
- Minimal lighting: single directional + high ambient (~0.7)

### Style Augmentation

Render-time (Blender shaders): flat, cel/toon, unlit
Post-render (Python/PIL/OpenCV): pixel art, painterly, sketch/lineart

Masks are per-pose, NOT per-style — the same mask applies to all style variants of a pose.

### Dataset Splits

Split by **character**, not by image (prevents data leakage). 80% train / 10% val / 10% test.

## Dependencies

- Blender 4.0+ (with bundled Python)
- OpenCV, Pillow (post-render style augmentation)
- NumPy (weight extraction)

For model training (separate from generation):
- PyTorch 2.0+, MMSegmentation, Albumentations, ONNX/onnxruntime

## Asset Sources

- **Mixamo** (primary): ~100 free rigged humanoids + animation clips. Standard skeleton naming.
- **Sketchfab CC0/CC-BY**: ~100–200 curated characters. More variety, quality varies.
- **Quaternius/Kenney**: ~50 CC0 low-poly game assets.
- **Blender community**: ~30–50 various CC rigs.

Only use CC0, CC-BY, or CC-BY-SA licenses. Never CC-NC. Log every asset's license in `dataset/sources/`.

## Validation Checks

Automated (run after every batch):
- Every non-transparent pixel has a non-zero mask region
- No mask is all-one-region
- All joint positions within image bounds
- 19 joints per pose
- Every image has a corresponding mask and joint file
- All images exactly 512×512
- No single region >60% of pixels

## Conventions

- Naming pattern: `{source}_{id}_pose_{nn}_{style}.png` for images, `{source}_{id}_pose_{nn}.png` for masks
- JSON metadata uses the schema defined in the PRD (sections 8.2–8.3)
- v1 is humanoid bipeds only. Non-humanoid (quadruped, bird, serpentine) is future work.
- Accessories: hide for v1 (cleaner training data). Flag with `has_accessories: true` in metadata.
