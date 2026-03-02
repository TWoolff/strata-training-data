# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's AI models (semantic segmentation, joint prediction, weight prediction, drawn pose estimation). Also houses animation intelligence scripts and curated metadata for animation training. Runs independently of the Strata codebase.

## What This Project Does

**Segmentation pipeline** — Takes rigged 3D characters (.fbx) and produces:
- **Color images** rendered in multiple art styles (flat, cel, pixel art, painterly, sketch, unlit)
- **Segmentation masks** (8-bit grayscale PNG, pixel value = region ID 0–21)
- **Draw order maps** (per-pixel depth, grayscale PNG)
- **Joint position JSON** (bone heads projected to 2D screen coords)
- **Weight map JSON** (per-vertex bone weights)

One character × 20 poses × 6 styles = 120 training examples. Target: 300+ characters → 36,000+ images.

**Spine 2D pipeline** — Parses Spine JSON animation projects, composites characters from atlas textures, maps Spine bones to Strata regions via regex patterns. Pure Python (no Blender dependency). See `pipeline/spine_parser.py`.

**Planned data sources** (see PRD v1.1):
- **Live2D models** — 2D illustrated characters with pre-decomposed layers, providing style diversity + draw order ground truth
- **VRoid/VRM models** — Anime-style 3D characters for multi-angle rendering (front, 3/4, side, back)
- **PSD files** — Opportunistic collection of layered Photoshop files with natural segmentation
- **Pre-processed datasets** — NOVA-Human, StdGEN, AnimeRun, UniRig, LinkTo-Anime, FBAnimeHQ and others. ~355K images available for download and conversion via `ingest/` adapters.

**Animation intelligence** — Scripts and curated metadata for animation training data:
- BVH mocap parsing, retargeting to Strata 19-bone skeleton, action labeling, timing extraction
- Synthetic degradation for in-betweening model training pairs
- Hand-annotated labels, breakdowns, and timing norms (tracked in Git)

**Manual annotation** — Label Studio integration for human review of segmentation masks (see `annotation/`).

## Project Structure

```
strata-training-data/
├── run_pipeline.py                    # CLI entry point for Blender pipeline
├── run_validation.py                  # CLI entry point for dataset validation
├── ruff.toml                          # Linting configuration
├── pipeline/                          # Blender/Python segmentation pipeline (18 modules)
│   ├── generate_dataset.py            #   Main entry point, orchestrates pipeline
│   ├── config.py                      #   Region colors, bone mappings, defaults
│   ├── bone_mapper.py                 #   Map bones → Strata's 22 regions
│   ├── renderer.py                    #   Render color + segmentation + multi-angle passes
│   ├── draw_order_extractor.py        #   Compute per-pixel depth from render
│   ├── live2d_mapper.py               #   Fragment name → Strata label mapping
│   ├── spine_parser.py               #   Parse Spine 2D JSON projects (no Blender)
│   ├── accessory_detector.py         #   Detect and hide accessories for clean training data
│   ├── manifest.py                   #   Generate dataset statistics + quality report
│   ├── measurement_ground_truth.py   #   Extract body measurements from 3D meshes
│   ├── splitter.py                   #   Train/val/test split by character
│   └── ...                            #   importer, exporter, validator, style_augmentor, etc.
├── annotation/                        # Label Studio manual annotation pipeline
│   ├── import_images.py               #   Import rendered images into Label Studio
│   ├── export_annotations.py          #   Export reviewed annotations back to dataset
│   └── label_studio_config.xml        #   Label Studio project configuration
├── animation/                         # BVH parsing, retargeting, degradation scripts
│   ├── scripts/                       #   bvh_parser, bvh_to_strata, degrade_animation, etc.
│   ├── labels/                        #   ✅ Tracked — cmu_action_labels.csv
│   ├── breakdowns/                    #   ✅ Tracked — transcribed animation analyses
│   └── timing-norms/                  #   ✅ Tracked — extracted from Williams/Thomas books
├── tests/                             # Test suite
├── data/                              # ⛔ .gitignore — large files, re-downloadable
│   ├── fbx/                           #   Mixamo/Sketchfab FBX characters
│   ├── poses/                         #   Animation FBX clips
│   ├── mocap/                         #   CMU/SFU BVH files
│   ├── sprites/                       #   Sprite sheet source files
│   ├── live2d/                        #   Live2D .moc3 + textures; labels/ tracked
│   ├── vroid/                         #   VRM source files; labels/ tracked
│   └── psd/                           #   Layered Photoshop files (opportunistic)
├── output/                            # ⛔ .gitignore — generated renders + masks
└── docs/                              # data-sources, labeling-guide, taxonomy-comparison, etc.
```

**Tracked:** `pipeline/`, `annotation/`, `animation/`, `tests/`, `docs/`, `data/*/labels/`, `data/*/README.md`
**Ignored:** `data/` contents (large binaries, re-downloadable), `output/` (generated, reproducible)

**Planned directories** (from PRD v1.1):
- `ingest/` — Adapter scripts for converting pre-processed external datasets into Strata training format
- `mesh/` — 3D mesh pipeline scripts (proportion clustering, texture projection)
- `scripts/` (top-level) — Utility scripts for download verification, statistics, cross-source splits
- `data/preprocessed/` — Downloaded pre-processed datasets from external research teams

**Label file convention:** Per-source label mappings live in `data/{source}/labels/` (e.g., `data/live2d/labels/live2d_mappings.csv`, `data/vroid/labels/vroid_mappings.csv`).

## Running the Pipeline

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ \
  --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 \
  --poses_per_character 20
```

Requires Blender 4.0+ (uses bundled Python 3.10+). No GPU needed — EEVEE flat shading runs on CPU.

## Key Technical Details

### Strata Standard Skeleton (21 body regions + background = 22 classes)

| ID | Region | ID | Region |
|----|--------|----|--------|
| 0 | background | 11 | upper_arm_r |
| 1 | head | 12 | forearm_r |
| 2 | neck | 13 | hand_r |
| 3 | chest | 14 | upper_leg_l |
| 4 | spine | 15 | lower_leg_l |
| 5 | hips | 16 | foot_l |
| 6 | shoulder_l | 17 | upper_leg_r |
| 7 | upper_arm_l | 18 | lower_leg_r |
| 8 | forearm_l | 19 | foot_r |
| 9 | hand_l | 20 | accessory |
| 10 | shoulder_r | 21 | hair_back |

### Bone Mapping Priority

1. Exact match → 2. Prefix match → 3. Substring match (case-insensitive) → 4. Manual override JSON

Mixamo characters should map 100% automatically. Non-Mixamo ~80% auto, ~20% manual via per-character override JSON.

### Segmentation Mask Rendering

- 22 Emission-only materials (one per region), no lighting
- Each mesh face assigned to region by majority vote of vertex bone weights
- No anti-aliasing, nearest-neighbor sampling — each pixel = exactly one region ID
- Output: 8-bit single-channel grayscale PNG (pixel value = region ID)

### Output Format

Per training example:
```
example_001/
├── image.png           ← Character render (512×512)
├── segmentation.png    ← Per-pixel label IDs (grayscale)
├── draw_order.png      ← Per-pixel depth (grayscale, 0=back 255=front)
├── joints.json         ← 2D joint positions
└── metadata.json       ← Source type, style, pose name, camera angle, draw order values
```

Draw order is computed from vertex Z-depth relative to camera (Mixamo) or from explicit render order indices (Live2D). Normalized to [0, 1] range per frame.

### Render Setup

- Orthographic camera (no perspective distortion)
- Auto-framed to character bounding box + 10% padding
- 512×512 output resolution
- Transparent background (alpha channel)
- Minimal lighting: single directional + high ambient (~0.7)
- **Multi-angle rendering**: 5 camera angles per character×pose — front (0°), three-quarter (45°), side (90°), three-quarter-back (135°), back (180°). Enabled via `--angles` CLI flag; defaults to front-only. Required for 3D mesh pipeline training data.

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

## Asset Sources

- **Mixamo** (primary): ~100 free rigged humanoids + animation clips. Standard skeleton naming.
- **Sketchfab CC0/CC-BY**: ~100–200 curated characters. More variety, quality varies.
- **Quaternius/Kenney**: ~50 CC0 low-poly game assets.
- **Blender community**: ~30–50 various CC rigs.
- **CMU Graphics Lab**: 2,548 mocap clips (BVH) — for animation intelligence.
- **SFU Motion Capture**: BVH mocap data.
- **Spine 2D community**: Sprite characters with Spine JSON animation data from OpenGameArt, itch.io.
- **Live2D community** (planned): ~300–500 models from Booth.pm, DeviantArt, GitHub. Pre-decomposed ArtMesh fragments provide near-free segmentation ground truth for 2D illustrated characters.
- **VRoid Hub** (planned): 2,000–5,000 VRM models. Anime-style 3D characters renderable from any angle. Standardized humanoid skeleton maps to Strata labels via bone weights.
- **PSD files** (opportunistic): ~50–100 layered Photoshop files from OpenGameArt, itch.io. Layer structure provides natural segmentation annotations.
- **Pre-processed datasets** (planned): NOVA-Human (~204K images), StdGEN (10.8K annotated characters), AnimeRun (~8K contour pairs), UniRig (14K rigged meshes), LinkTo-Anime (~29K frames), FBAnimeHQ (~113K images). Downloaded and converted via `ingest/` adapters. See `docs/preprocessed-datasets.md`.

Only use CC0, CC-BY, or CC-BY-SA licenses. Never CC-NC. Full source list in `docs/data-sources.md`. Log every asset's license in output metadata.

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

