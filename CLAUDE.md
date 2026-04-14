# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 6 ONNX AI models. Runs independently of the Strata codebase.

## Model Targets & Current Scores

**Goal:** Uploaded 2D character illustrations look natural from all generated angles when rigged and posed.

| # | Model | Current | Ship Target | Stretch | Key Issue If Bad |
|---|-------|---------|-------------|---------|------------------|
| 1 | **Segmentation** | 0.6485 mIoU | **0.75** | 0.80 | Background fringe, bg between legs, character parts deleted |
| 2 | **Joint Refinement** | 0.00121 offset | **0.0005** | 0.0003 | Joints misplaced → limbs bend wrong |
| 3 | **Weight Prediction** | 0.0215 MAE | **0.01** | 0.005 | Mesh deformation artifacts when posed |
| 4 | **Inpainting** (2D bg) | 0.0028 val/l1 | **0.002** | 0.001 | Visible seams where character removed |
| 5 | **Texture Inpainting** (UV) | 0.1282 val/l1 | **0.08** | 0.05 | Back/sides of 3D char look wrong |
| 6 | **3D Mesh** | Blurry (old U-Net) | **Clean geometry** | PBR | Character looks flat, geometry wrong |

### Priority Order (April 14, 2026)

1. **Segmentation** — Dr. Li's See-Through training code **released April 14**. Plan: take her SAM-HQ encoder, replace 19 clothing heads with our 22 anatomy heads, fine-tune. #1 user complaint.
2. **Texture Inpainting** — v3 ControlNet at 0.1282 val/l1 but fails on illustrated styles (lichtung cat test). Next: **test StyleTex** (SIGGRAPH 2024, Apache 2.0) — SDS-based style transfer from reference image. Pretrained, no data collection needed.
3. **3D Mesh** — SAM 3D Objects validated. Needs integration + texture projection pipeline.
4. **Joint Refinement** — ViTPose++ fine-tune. Current model functional.
5. **Weight Prediction** — Study Puppeteer architecture. Current model functional.
6. **Inpainting** — Low priority. Current model works for most cases.

### Next A100 Runs Queued

**Run A: StyleTex test on lichtung cat (~1.5 hrs)**
- `training/run_styletex_test.sh` — clones StyleTex, SDS optimize UV texture with illustration as style reference
- Goal: see if style-aware texture generation solves watercolor cat problem
- Storage: 40 GB | Time: 45-75 min

**Run B: Dr. Li SAM-HQ seg fine-tune (timing TBD, code just released)**
- Fetch her training pipeline from see-through repo (v3 training scripts released April 14)
- Replace 19 clothing class heads with our 22 anatomy heads
- Train on our existing seg data with her encoder
- Expected: 0.65 → 0.72-0.78 mIoU

## Model Strategy

Strata's unique value: combining 3D reconstruction with illustrated character understanding for animation-ready rigging.

| # | Task | New Model | License | Status |
|---|------|-------|---------|--------|
| 1 | **Segmentation** | Dr. Li's SAM-HQ encoder + our 22-class decoder | Apache 2.0 | Training code April 12 |
| 2 | **3D Mesh** | SAM 3D Objects | SAM License (commercial OK) | Validated |
| 3 | **Skeleton** | SAM 3D Body / ViTPose++ | SAM / Apache 2.0 | Tested — good on humanoid |
| 4 | **Texture** | Multi-view projection + ControlNet inpainting | Our code | Training now |
| 5 | **Weights** | Puppeteer / current MLP | TBD | Study architecture |
| 6 | **Backup 3D** | TRELLIS.2 (4B params, MIT) | MIT | Needs HF access |

**New Strata Pipeline (vision):**
1. User imports front view illustration
2. SAM 3D Objects → 3D mesh geometry (discard blurry texture)
3. Project artist's illustration onto mesh (pixel-perfect where visible)
4. Optional side/back views → more coverage
5. ControlNet inpaints remaining UV gaps (~10-30% of surface)
6. Segmentation → anatomy regions, SAM 3D Body → skeleton, weight prediction
7. Export rigged, textured, animatable 3D character

## 22-Class Anatomy Schema

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

Maps directly to skeleton bones — each class = one bone's influence zone. Dr. Li's 19-class schema is clothing-based (topwear, legwear) and cannot drive animation.

## Project Layout

Key dirs: `pipeline/` (Blender rendering), `ingest/` (external dataset adapters), `training/` (model training + configs + run scripts), `scripts/` (utilities), `animation/` (BVH/mocap), `tests/`, `docs/`.

Entry points: `run_pipeline.py`, `run_validation.py`. Config: `pipeline/config.py`.

## Output Format

```
example_001/
├── image.png           ← 512×512 RGBA
├── segmentation.png    ← 8-bit grayscale (pixel value = region ID 0-21)
├── depth.png           ← Depth map (uint8, Marigold LCM)
├── normals.png         ← Surface normals (RGB uint8, Marigold LCM)
├── joints.json         ← 2D joint positions
└── metadata.json       ← Source, style, enrichment flags
```

## Running the Pipeline

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 --poses_per_character 20
```

Requires Blender 4.0+/5.0+. Orthographic camera, 512×512, transparent bg, 22 Emission materials for seg masks.

## Asset Sources & Licensing

Only CC0, CC-BY, CC-BY-SA. **Never CC-NC.**

**Prohibited:** Mixamo (Adobe ToS), Live2D (proprietary), ArtStation/curated_diverse (no AI permission), CartoonSegmentation (no license).

## Storage & Bucket

- **Hetzner bucket** (`strata-training-data` at `fsn1.your-objectstorage.com`) — training-ready output only
- **External HD** (`/Volumes/TAMWoolff/data/`) — raw source data, never delete originals
- **Local SSD** — avoid large datasets
- **Always use `rclone copy`** (never `sync`, never `aws s3 sync`)

```bash
rclone copy ./output/ hetzner:strata-training-data/output/ --transfers 32 --checkers 64 --fast-list --size-only -P
```

### Key Bucket Contents

Tars in `tars/` prefix: humanrig (16.8G), humanrig_posed (12G), meshy_cc0_textured_restructured (2.8G), flux_diverse_clean (300M), sora_diverse (380M), gemini_li_converted (223M), vroid_cc0 (203M), cvat_annotated (9M), soft_targets_precomputed (1.9G), demo_back_view_pairs (3.4G).

Texture inpainting tars in root: texture_pairs_front (4.9G, with geometry maps), texture_pairs_side (2.8G), texture_pairs_back (2.8G), texture_pairs_100avatars (113M). Total ~4,135 pairs.

Frozen val/test splits: `data_cloud/frozen_val_test.json`. All runs must use this file.

## A100 Training Workflow

1. Prep locally → upload to bucket → push code
2. Spin up A100 → `cloud_setup.sh lean`
3. Run script (downloads data, trains, uploads checkpoints)
4. Destroy instance
5. Download checkpoints to Mac → benchmark

## Key Learnings

**Segmentation (best: run 20, 0.6485 mIoU):**
- Boundary label softening = biggest lever (+8.8% mIoU). Precompute as `.npz` files.
- Softening hurts thin regions (neck, accessory). Exclude or reduce radius for small regions.
- Clothing-based labels (SAM, Dr. Li's 19-class) don't help — anatomy ≠ clothing.
- SAM 3D Body labels don't help — body mesh ≠ clothing silhouette.
- Bootstrapping loop works: stronger model → better pseudo-labels → better model.
- More illustrated data = best lever for diversity. Pseudo-label ceiling at ~2K chars.
- Class 20 remapped to background (unused by rigging pipeline).
- A100 is 40GB. Batch 16 for soft targets. Use frozen val/test splits.

**Texture Inpainting (best: run 4 / v3, 0.1282 val/l1):**
- ControlNet on SD 1.5 Inpainting, 9-channel input (noisy latent + mask + partial).
- Run 1: 500 pairs → 0.1509. Run 2: 2,891 pairs → 0.1520. Run 3: 2,891 pairs, 100 epochs → 0.1497.
- **Run 4 (v3): 1,244 pairs WITH real geometry maps, fine-tuned from v3 → 0.1282 val/l1, 0.418 SSIM.** −14% L1, +6% SSIM. Geometry maps are the biggest single lever.
- Single-view "hero" pipeline confirmed: 1 illustration → SAM 3D mesh → auto-detect camera angle → project illustration → ControlNet inpaint gaps. Cleaner UX than asking for multi-view input.
- Manual denoising loop needed for validation (diffusers pipeline incompatible with 9-ch ControlNet).
- Use `torch.amp.autocast` for mixed fp16/fp32 validation.

**Pipeline experiment (lichtung cat, April 14):**
- SAM 3D Objects mesh + 1 watercolor cat illustration → silhouette-IoU search found best camera angle (72°, IoU 0.77).
- Hero view projects pixel-perfect for ~30% of UV. Rest needs inpainting.
- Multi-view (front/back) approach failed — Gemini-generated views don't match mesh from "true" front/back angles.
- TPS landmark-based warping helps but needs 20+ landmarks per view to be useful.
- Single-view + strong inpainting is the best UX.
- **v3 ControlNet failed on illustrated style**: Meshy-trained model produced dark solid fill, not watercolor. Empty-prompt training means text prompts + CFG have no effect. Need style-aware approach.
- **Next attempt: StyleTex** — takes illustration as style reference, explicitly decouples style from content via CLIP manipulation.

**3D Reconstruction:**
- U-Net view synthesis deprecated (blurry). SHARP abandoned (research-only license).
- SAM 3D Objects = primary 3D mesh source. TRELLIS.2 (MIT) = backup.
- Texture from artist's illustration projected onto mesh. AI only fills gaps.

## Segmentation Run History

| Run | mIoU | Key change |
|-----|------|-----------|
| 8 | 0.4721 | Bootstrapping loop validated |
| 13a | 0.5425 | +toon_pseudo |
| 16 | 0.5808 | Frozen val deployed |
| 18 | 0.5750 | True baseline with frozen val |
| 20 | **0.6485** | Boundary softening (radius=2) |
| 21 | 0.6361 | Re-pseudo-label, no softening |
| 22/22b | 0.56-0.59 | SAM labels — don't help |

Best: **Run 20** (0.6485 test mIoU). Config: `training/configs/segmentation_a100_run20.yaml`.

## Quality Filter

`scripts/filter_seg_quality.py`: `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`. Checks `missing_head` and `missing_torso`.

## Bootstrapping Loop

Model → pseudo-label new data → quality filter → retrain. Each cycle improves both.

```bash
python scripts/ingest_gemini.py --input-dir /path/to/raw --output-dir /path/to/preprocessed --no-seg --only-new
# Then pseudo-label + quality filter + train on A100 (handled by run scripts)
```

## Strata Post-Processing (in `../strata/`)

6 improvements in segmentation.rs, joints.rs, weights.rs:
1. Alpha-aware preprocessing (composite RGBA onto black)
2. Confidence-gated seg cleanup (<0.4 → 5×5 majority, remove <16px components)
3. Bilinear logit upscaling (interpolate logits then argmax)
4. Laplacian weight smoothing (2 iterations, confidence-adaptive)
5. Adaptive joint offset cap (15% of bbox, clamped [0.02, 0.10])
6. Confidence-weighted MLP/heat blending for weights

## Later — Post Launch

- Interactive view correction paint tool
- Blueprint marketplace
- InnoFounder application (August 2026, 430K DKK)
- PreSeed Ventures / Accelerace / byFounders for investment
