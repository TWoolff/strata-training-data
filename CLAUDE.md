# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 6 ONNX AI models. Also houses animation intelligence scripts and curated metadata for animation training. Runs independently of the Strata codebase.

## Strata's 6 ONNX Models

Strata (Tauri/Rust/React desktop app at `../strata/`) uses 6 ONNX models defined in `src-tauri/src/ai/runtime.rs`:

| # | Model | ONNX File | Architecture | Training Pipeline | Status |
|---|-------|-----------|-------------|-------------------|--------|
| 1 | **Segmentation** | `segmentation.onnx` | DeepLabV3+ MobileNetV3 (multi-head: seg + depth + normals) | `training/train_segmentation.py` | Has pipeline + data |
| 2 | **Joint Refinement** | `joint_refinement.onnx` | MobileNetV3 + regression heads | `training/train_joints.py` | Has pipeline + data |
| 3 | **Weight Prediction** | `weight_prediction.onnx` | Per-vertex MLP with optional encoder features (31→128→256→128→20, +encoder branch) | `training/train_weights.py` | Has pipeline + data |
| 4 | **Inpainting** | `inpainting.onnx` | U-Net for occluded body regions | `training/train_inpainting.py` | Has pipeline + data |
| 5 | **Texture Inpainting** | `texture_inpainting.onnx` | Diffusion-based 3D texture fill | **Not yet built** | Needs pipeline + data |
| 6 | **Novel View Synthesis** | `novel_view.onnx` | Multi-view conditioned diffusion | **Not yet built** | Needs pipeline + data |

### Model Details

**1. Segmentation** — Input: [1,3,512,512] image. Outputs: 22-class body region logits + depth map (sigmoid, trained from Marigold depth labels) + surface normals [3-channel] + confidence mask + encoder_features (passed to model 3). Fine-tunes from ImageNet MobileNetV3. The depth and normals heads are distilled from Marigold LCM — one forward pass produces segmentation + depth + normals.

**2. Joint Refinement** — Input: [1,3,512,512] image. Outputs: [1,2,20] joint offsets (dx-first layout) + [1,20] confidence + [1,20] presence. Fine-tunes from ImageNet MobileNetV3. Falls back to geometric predictions if unavailable.

**3. Weight Prediction** — Input A (always): [1,31,2048,1] vertex features (position, bone distances, heat diffusion, region label). Input B (optional): encoder features from model 1, bilinearly sampled at vertex positions. Outputs: [1,20,2048,1] per-bone weights + [1,1,2048,1] confidence. Single MLP with an optional encoder feature branch — encoder features are projected and concatenated with vertex features. During training, the encoder branch is randomly dropped (entire-branch dropout) so the model learns to work with or without visual context. At inference, Strata passes encoder features when available (improves accuracy for unusual proportions like chibi, elongated limbs, loose clothing) or zeros when not. Replaces the former separate weight_prediction.onnx and diffusion_weight_prediction.onnx.

**4. Inpainting** — U-Net that fills occluded body regions in 2D paintings. Fallback: EdgeExtend (dilates visible edge pixels). **Training pipeline + paired data needed.**

**5. Texture Inpainting** — Diffusion model that fills unobserved texture regions when generating 3D mesh. **Training pipeline + data needed.**

**6. Novel View Synthesis** — Multi-view conditioned diffusion model. Generates any unseen view (back, side, 3/4, etc.) from one or more reference views. Fallback: PaletteFill (mirror + color adjustment). **Training pipeline + multi-view data needed.**

### Training Coverage

Models 1-4 have complete training pipelines with configs for local (4070 Ti), lean A100, and full A100 runs. Models 5-6 still need training pipelines built in this repo. All models are bundled in `../strata/src-tauri/models/` (~55MB total) and loaded lazily via the `ort` ONNX runtime crate.

## What This Project Does

Generates labeled training data (color images, segmentation masks, draw order maps, joint positions, weight maps) from rigged 3D characters across 6 art styles. Also includes: Spine 2D parser (`pipeline/spine_parser.py`), BVH mocap pipeline (`animation/`), Label Studio annotation (`annotation/`), and external dataset adapters (`ingest/`). Target: 300+ characters, 36K+ images. See `docs/preprocessed-datasets.md` for planned external data sources.

## Project Layout

Key dirs: `pipeline/` (Blender rendering + segmentation), `ingest/` (external dataset adapters), `annotation/` (Label Studio), `animation/` (BVH/mocap), `tests/`, `docs/`. Entry points: `run_pipeline.py`, `run_validation.py`. Config: `pipeline/config.py`.

**Tracked:** `pipeline/`, `ingest/`, `annotation/`, `animation/`, `tests/`, `docs/`, `data/*/labels/`. **Ignored:** `data/` (large binaries), `output/` (generated).

Label mappings live in `data/{source}/labels/` (e.g., `data/live2d/labels/live2d_mappings.csv`).

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
├── image.png           ← Character render (512×512, RGBA)
├── segmentation.png    ← Per-pixel label IDs (grayscale, 0-21)
├── draw_order.png      ← Per-pixel depth (grayscale, 0=back 255=front) — legacy, being replaced by depth.png
├── depth.png           ← Depth map (grayscale uint8, Marigold LCM) — replaces draw_order as depth ground truth
├── normals.png         ← Surface normals (RGB uint8, Marigold LCM) — encodes [-1,1] as [0,255]
├── joints.json         ← 2D joint positions
└── metadata.json       ← Source type, style, pose name, camera angle, enrichment flags
```

Draw order is computed from vertex Z-depth relative to camera (Mixamo) or from explicit render order indices (Live2D). Depth and normals are enriched via Marigold LCM (`run_normals_enrich.py`). The segmentation model's depth head will be retrained using depth.png labels in run 3.

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

## Asset Sources

Only use CC0, CC-BY, or CC-BY-SA licenses. Never CC-NC. Log every asset's license in output metadata. Full source list in `docs/data-sources.md`, external datasets in `docs/preprocessed-datasets.md`.

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

## Storage Policy

- **Hetzner bucket** — upload only training-ready output (validated `output/` data: images, masks, joints, metadata). No raw source assets.
- **External hard drive** — store all raw/source data locally: FBX characters, BVH mocap, VRM files, sprite sheets, Live2D models, pre-processed external datasets. Download and process from here; do not upload raw data to the cloud bucket.
- **Local SSD** — avoid storing large output datasets locally. Process on the external HD when possible and upload from there.

### Bucket Upload/Download — Use rclone

**Always use `rclone copy` for bucket operations.** Do NOT use `aws s3 sync` — it is extremely slow with many small files.

```bash
# Upload (copy only adds files, never deletes remote)
rclone copy ./output/dataset/ hetzner:strata-training-data/dataset/ \
  --transfers 32 --checkers 64 --fast-list --size-only -P

# Download
rclone copy hetzner:strata-training-data/dataset/ ./output/dataset/ \
  --transfers 32 --checkers 64 --fast-list --size-only -P
```

- Config: `~/.config/rclone/rclone.conf`, remote name: `hetzner`
- `-P` shows real-time progress with transfer speed
- `--size-only` skips slow checksum comparison (safe for our use case)
- **Never use `rclone sync`** — it deletes remote files not present locally

### Bucket Contents (as of March 11 2026, post-run 4)

Bucket: `strata-training-data` at `fsn1.your-objectstorage.com`. ~755K files, ~248 GiB total.

Many datasets that were previously loose files have been consolidated into `tars/` for fast A100 setup. Loose-file prefixes (anime_instance_seg, animerun*, conr, ingest/vroid_lite, instaorder, nova_human, meshy_cc0*, segmentation) were purged after tarring.

| Prefix | Files | Size | Source |
|--------|------:|-----:|--------|
| `animation/` | 18,628 | 66.7 GiB | 100STYLE retargeted mocap (100 styles × 8-10 contents) |
| `anime_seg/` | 64,984 | 2.6 GiB | SkyTNT anime-segmentation v1+v2, RTMPose joints enriched |
| `checkpoints/` | 16 | 708 MiB | Run 1 checkpoints (all models) |
| `checkpoints_run1/` | 1 | 203 MiB | Run 1 seg best.pt (for run 4 resume) |
| `checkpoints_run3/` | 21 | 808 MiB | Run 3 checkpoints (seg, joints, weights, diffusion_weights, inpainting) |
| `checkpoints_run4/` | — | — | Run 4 checkpoints (seg, inpainting) |
| ~~`curated_diverse/`~~ | — | — | **DELETED** (ArtStation, prohibited) |
| `encoder_features/` | 11,434 | 41.9 GiB | Precomputed seg encoder features for weight training (float16) |
| `fbanimehq/` | 304,889 | 11.4 GiB | FBAnimeHQ face/body crops |
| `gemini_diverse/` | 1,164 | 55 MiB | 221 Gemini pseudo-labeled examples (seg + normals + draw_order). 698 training images ingested locally, pending upload. |
| `humanrig/` | 262,983 | 16.2 GiB | HumanRig rendered chars + joints + weights + Marigold normals (run 3) |
| ~~`live2d/`~~ | — | — | **DELETED** (Live2D ToS, prohibited) |
| `logs/` | — | — | Training logs (runs 1-4) |
| `models/` | 10 | 186 MiB | ONNX exports — run 3 (seg, joints, weights, diffusion_weights, inpainting) |
| `models/onnx_run4/` | 4 | — | ONNX exports — run 4 (segmentation, joint_refinement, weight_prediction, inpainting) |
| `tars/` | 12 | 44.9 GiB | Tar-packed datasets for fast A100 setup (see below) |
| `unirig/` | 80,980 | 52.7 GiB | UniRig rigged meshes + 14,950 weight.json uploaded |

**Tar contents** (`tars/` prefix):
| Tar | Size | Contents |
|-----|-----:|---------|
| `humanrig.tar` | 16.8 GiB | HumanRig dataset |
| `fbanimehq.tar` | 14.2 GiB | FBAnimeHQ dataset |
| `meshy_cc0_textured.tar` | 4.3 GiB | Meshy CC0 textured renders |
| `meshy_cc0_unrigged.tar` | 4.0 GiB | Meshy CC0 unrigged renders |
| `anime_seg.tar` | 3.0 GiB | anime-segmentation v1+v2 |
| `meshy_cc0.tar` | 1.5 GiB | Meshy CC0 rigged renders |
| `segmentation.tar` | 426 MiB | Mixamo pipeline output |
| ~~`curated_diverse.tar`~~ | — | **DELETED** (prohibited) |
| ~~`live2d.tar`~~ | — | **DELETED** (prohibited) |


## A100 Training Run Workflow

Each training run follows this automated pattern:

1. **Prep locally** — enrich new data, upload to bucket, push code to GitHub
2. **Spin up A100** — `cloud_setup.sh lean` downloads data + installs deps
3. **One command** — `training/run_second.sh` (or run_third.sh, etc.) handles everything:
   - Seg enrichment (pseudo-label unlabeled datasets with trained Model 1)
   - Normals enrichment (Marigold surface normals on small datasets)
   - Train all models (seg → joints → weights → inpainting → ONNX export)
   - Upload checkpoints, ONNX models, logs, and enriched data back to bucket
4. **Destroy instance** — everything is safe in the bucket
5. **Download to Mac** — `rclone copy` checkpoints + ONNX models
6. **Run benchmark** — `python3 run_benchmark.py` compares with previous runs

### Key Scripts

| Script | Purpose |
|--------|---------|
| `training/cloud_setup.sh lean` | Set up A100: install deps, configure rclone, download data |
| `training/run_second.sh` | Second run (abandoned): enrich → train → upload |
| `training/run_third.sh` | Third run: Marigold enrich → train → upload → tar |
| `training/run_fourth.sh` | Fourth run: quality filter → seg fine-tune → inpainting → ONNX → upload |
| `training/run_four_five.sh` | Run 4.5: SAM2 pseudo-labels → loss weighting → seg fix → ONNX → upload |
| `training/run_fifth.sh` | Fifth run: joints retrain (posed GT) → seg fine-tune → weights refresh → ONNX → upload |
| `training/train_all.sh lean` | Train all 4 models sequentially + ONNX export |
| `run_seg_enrich.py` | Enrich datasets with 22-class seg using trained Model 1 |
| `run_normals_enrich.py` | Enrich datasets with surface normals (Marigold LCM) |
| `run_benchmark.py` | Benchmark all models on 7 Gemini test characters |

### Benchmark Test Set

7 curated Gemini-generated character images at `/Volumes/TAMWoolff/data/preprocessed/gemini/` (druid, golem, knight, rogue, samurai, sorceres, spacemarine). `run_benchmark.py` produces a 6-column overview grid (Original | Segmentation | Joints | Draw Order | Surface Normals | Depth) saved as `output/trainingNN_overview.png`. Auto-increments run number. Use `--no-normals` / `--no-depth` for faster runs.

### Surface Normals (Marigold)

All 2D image datasets should include `normals.png` — surface normal maps generated by Marigold LCM (`prs-eth/marigold-normals-lcm-v0-1`). ~0.5s/img on A100, ~4.5s on Mac MPS. These provide 3D surface orientation from 2D images, critical for models 6 (texture inpainting) and 7 (back view generation).

## First Training Run (March 5-7 2026, A100 Lean) — COMPLETE

All models trained on Lambda A100 via `train_all.sh lean`. Checkpoints, ONNX models, and logs uploaded to bucket + downloaded locally. Instance destroyed.

### Training Data (first run)

| Dataset | Examples | Annotations |
|---------|----------|-------------|
| `segmentation/` | 1,598 | 22-class seg + draw_order + joints |
| `live2d/` | 844 | 22-class seg + draw_order |
| `humanrig/` | 11,434 | 22-class seg + joints + weights |
| ~~`curated_diverse/`~~ | ~~748~~ | ~~fg/bg mask + draw_order~~ — **REMOVED: ArtStation, no AI training permission** |
| `anime_seg/` | ~14K | fg/bg mask + joints (RTMPose) |
| **Total loaded** | **25,494 train / 3,137 val** | |

## Second Training Run (March 7 2026, A100 Lean) — ABANDONED

Ran seg enrichment + Marigold normals/depth on ~5K images, then started segmentation training. Killed early — seg regressed to 0.38 mIoU (vs run 1's 0.545), likely due to noisy pseudo-labeled data. Other models skipped. Enriched datasets re-tarred and uploaded to bucket. Instance destroyed.

Fixes deployed during run 2 (available for run 3):
- Image discovery in occlusion pair generator fixed (was grabbing seg maps as source images)
- Pair generation capped at 15K source images (~45K pairs) to avoid filling disk
- rclone region=fsn1 added to cloud_setup.sh
- All datasets tar-packed in bucket for faster setup (~30min vs 5h)

### What's New for Third Run

- **Segmentation model v2**: depth + normals heads (Marigold-distilled), replacing draw_order
- **Weight data: 54 → ~27K examples** — HumanRig (11,434) + UniRig (14,950) + Mixamo (1,598)
- **Inpainting data loader fixed** — image discovery now uses `glob("*/image.png")` not `rglob("*.png")`
- **Marigold enrichment on unirig** — ~10K front views getting depth.png + normals.png
- **PRD for Strata runtime**: `docs/prd-segmentation-model-v2.md` — ONNX contract change documentation

## Model 1: Segmentation (multi-head: seg + depth + normals)
What it does: Takes a 512×512 character image → outputs 22-class body region map (head, chest, arms, legs, etc.), pixel-level depth map, 3-channel surface normals, and confidence mask. This is the foundation — it tells Strata which pixels belong to which body part, how deep they are, and which direction they face. The depth and normals heads are distilled from Marigold LCM (knowledge distillation: big diffusion model labels → small MobileNetV3 student). One forward pass, three outputs.

Score (run 1): **0.5453 mIoU** (epoch 94/100, March 6 2026). Run 2 regressed to 0.38 (killed early). Run 3 regressed to 0.3728 (noisy Meshy CC0 data).
Score (run 4): **0.4389 mIoU** (epoch 139/143, March 11 2026). Resumed from run 1 checkpoint, fine-tuned 50 epochs at 5e-5 LR with label smoothing 0.05. 30,086 train / 3,717 val examples. Did not recover run 1 quality — quality filter helped but dataset composition changed (no Mixamo/live2d, added Gemini diverse).

## Model 2: Joint Refinement
What it does: Takes a 512×512 character image → predicts 2D positions of 20 skeleton joints (hips, knees, elbows, etc.) + confidence per joint. Strata uses geometric fallback if the model isn't confident, but the CNN improves accuracy especially for unusual poses.

Score (run 1): **0.001287 mean_offset_error** (epoch 13/80, early stopped at 28, March 6 2026 A100 lean run). 110K training examples, 97.9% presence accuracy.
Score (run 3): **0.001206 mean_offset_error** (epoch 21/80, early stopped at 36, March 10 2026). Slight improvement.

## Model 3: Weight Prediction
What it does: Takes per-vertex features (position, bone distances, heat diffusion, region label — 31 features per vertex) + optionally encoder features from Model 1's segmentation backbone → predicts skinning weights for 20 bones. This determines how each mesh vertex deforms when bones move. It's a small MLP with an optional encoder feature branch. When encoder features are available (sampled at vertex positions from the segmentation model), they provide visual context about body proportions — improving accuracy for unusual characters (chibi, elongated limbs, loose clothing). The encoder branch uses dropout during training so the model works with or without visual context.

Previously split into two separate models (weight_prediction.onnx + diffusion_weight_prediction.onnx). Now merged into a single model with optional encoder input.

Score (run 1, geometry-only): **0.083958 MAE** (epoch 53/100, early stopped at 68, March 6 2026 A100 lean run). Only 54 training examples (Mixamo segmentation data). 98.7% confidence accuracy.
Score (run 1, with encoder features): **0.089449 MAE** (epoch 30/60, early stopped at 45). Slightly worse — encoder features don't add enough signal with so few weight examples.
Score (run 3, geometry-only): **0.023137 MAE** (epoch 18/100, early stopped at 33, March 10 2026). 12,027 training examples (HumanRig via UniRig split_loader fix). 3.6x improvement over run 1.

## Model 4: Inpainting
U-Net that takes a character image with occluded/missing body regions (e.g., arm hidden behind body) and fills in the missing pixels. Currently Strata falls back to "EdgeExtend" (dilates visible edge pixels outward), which looks rough. A trained inpainting model would produce much cleaner fills.

Score (run 1): **BROKEN** — data loader bug (`rglob("*.png")` grabbed masks as source images). Fixed in commit `011c3ba`.
Score (run 4): **0.0028 val/l1** (epoch 33/50, March 11 2026). 35,769 train / 4,542 val examples (from 44,668 total pairs). Perceptual loss enabled (weight=0.100). Training cut short at epoch 33 when instance was destroyed — still improving. Best val/l1 was 0.0028 at epoch 33.

## Model 5: Texture Inpainting
Diffusion model that fills unobserved texture regions when unwrapping a 2D character painting onto a 3D mesh. When you wrap a front-facing painting around a 3D model, the back/sides have no texture data — this model would generate plausible fills. Needs training pipeline + data (no pipeline exists yet).

Score:

## Model 6: Novel View Synthesis
Multi-view conditioned diffusion model that generates any unseen view (back, side, 3/4, etc.) from one or more reference views. Given a front-facing character painting, it can synthesize the back, sides, and any angle in between — producing consistent hair, clothing, and accessory details across all views. Currently Strata falls back to "PaletteFill" (mirror + color adjustment). Needs training pipeline + multi-view paired data (no pipeline exists yet).

Score:

## Third Training Run (March 9-10 2026, A100 Lean) — COMPLETE

Seg and joints trained from scratch. Weights trained with split_loader fix (UniRig discovery). Encoder features precomputed + diffusion weights trained. All checkpoints uploaded to bucket as `checkpoints_run3/`. Instance destroyed.

### Run 3 Results

| Model | Score | vs Run 1 | Notes |
|-------|-------|----------|-------|
| Segmentation | 0.3728 mIoU (epoch 93/100) | Regressed from 0.545 | Noisy Meshy CC0 auto-rig data |
| Joints | 0.001206 mean_err (epoch 21, early stop 36) | Improved from 0.001287 | Slight improvement |
| Weights | 0.023137 MAE (epoch 18, early stop 33) | **3.6x better** (was 0.084) | 12K examples via split_loader fix |
| Diffusion weights | 0.021646 MAE (epoch 42/60) | **Better than geometry-only** | 9,129 examples with encoder features |
| Inpainting | Not retrained | N/A | Scheduled for run 4 |

### Run 3 Fixes
- **split_loader.py**: Added nested view + weight-only dataset discovery (UniRig `{id}/front/weights.json`)
- **weight_dataset.py**: Fixed example_id mismatch — used `child.name` instead of `{child}_{view}` so split filter matches
- **precompute_encoder_features.py**: Added float16 saving + MAX_VERTICES=2048 cap to prevent disk exhaustion
- **Disk space**: Original float32 uncapped encoder features filled 200GB; capped float16 uses ~42GB for 11K examples

## Fourth Training Run (March 11 2026, A100 Lean) — COMPLETE

**Goal: Ship-ready models 1-4.** Script: `training/run_fourth.sh`

Seg resumed from run 1 checkpoint (0.545 mIoU) and fine-tuned 50 epochs. Inpainting trained for the first time with fixed data loader (35K pairs). ONNX models exported to `models/onnx_run4/`. All checkpoints uploaded to bucket as `checkpoints_run4/`. Instance destroyed.

### Run 4 Results

| Model | Score | vs Previous | Notes |
|-------|-------|-------------|-------|
| Segmentation | 0.4389 mIoU (epoch 139/143) | Regressed from run 1's 0.545 | Resumed from run 1, but different dataset composition |
| Joints | Not retrained | Keeps run 3 (0.001206) | Good enough |
| Weights | Not retrained | Keeps run 3 (0.023 MAE) | 3.6x better than run 1 |
| Inpainting | 0.0028 val/l1 (epoch 33/50) | **First successful training** | Cut short — instance destroyed while still improving |

### Run 4 Quality Filter Results
- humanrig: 11,434 total → 11,402 passed, 32 rejected (0.3%)
- unirig: 10,095 total → 9,286 passed, 809 rejected (8.0%)
- meshy_cc0: 0 masks found (no seg masks in lean download)
- meshy_cc0_textured: 0 masks found
- gemini_diverse: 221 total → 220 passed, 1 rejected (0.5%)

### Run 4 Enrichment
- Gemini diverse: 163 normals + 221 depth maps enriched via Marigold LCM (~4 img/s on A100)

### Run 4 Training Data
- Segmentation: 30,086 train / 3,717 val (from 47,654 before split filter), 1,471 with depth, 1,471 with normals
- Inpainting: 35,769 train / 4,542 val (from 44,668 total pairs)
- Datasets: humanrig, meshy_cc0, meshy_cc0_textured, anime_seg, fbanimehq, gemini_diverse
- Note: unirig not downloaded (42GB too large for lean mode)
- Note: live2d removed (prohibited license)

### Run 4 Seg Analysis
Resumed from run 1's 0.545 mIoU checkpoint but regressed to 0.4389. The dataset composition changed significantly from run 1 (no Mixamo/live2d ground truth, replaced with quality-filtered auto-rigged data + pseudo-labeled Gemini). The seg model needs higher-quality ground-truth data to improve — run 5's expanded Gemini dataset (698 images) and humanrig_posed (45K GT examples) should help.

### Run 4 Inpainting Analysis
First successful inpainting training — the fixed data loader found 44,668 valid pairs (vs ~3 in run 1). Training was progressing well (val/l1 dropped from 0.0087 → 0.0028 over 33 epochs) but was cut short when the instance was destroyed. The model was still improving — run 5 should continue training from this checkpoint.

## Run 4.5 — Seg Fix (PLANNED)

**Goal: Break past 0.545 mIoU with SAM2 pseudo-labels + per-dataset loss weighting.** Script: `training/run_four_five.sh`

Focused seg-only run (~3-4 hrs on A100). Joints, weights, inpainting NOT retrained.

### Why Seg Keeps Regressing
Run 1 hit 0.545 with Mixamo (1,598 GT masks) + live2d (844) — small but high-quality ground-truth. Those datasets were removed (prohibited licenses). Runs 2-4 replaced them with noisier auto-rigged + pseudo-labeled data, and seg regressed to 0.4389.

### Two Fixes

**1. SAM2 + joint-conditioned pseudo-labeling** (`scripts/run_sam2_pseudolabel.py`):
- SAM2 produces precise segment boundaries (sharper than our 0.545 model)
- Joint positions from joints.json assign each segment to a body region
- Applied to anime_seg (~14K) and gemini_diverse (~700) — replacing binary fg/bg and noisy model pseudo-labels
- Result: 22-class masks with SAM2-quality boundaries and joint-based region semantics

**2. Per-dataset loss weighting** (code change in `train_segmentation.py` + `segmentation_dataset.py`):
- Each dataset directory gets a configurable weight multiplier in the YAML config
- Ground-truth data (humanrig: 3.0) weighted higher than noisy data
- SAM2-labeled data gets intermediate weight (gemini_diverse: 2.5, anime_seg: 1.5)

**3. Drop fbanimehq** — 101K binary fg/bg examples excluded entirely. They can't teach 22-class regions and were drowning the signal even at low weight.

### Run 4.5 Strategy

| Step | Task | Est. Time |
|------|------|-----------|
| 0 | Download run 4 seg checkpoint + SAM2 model | ~5 min |
| 1 | Download datasets | ~10 min |
| 2 | SAM2 pseudo-label anime_seg + gemini_diverse | ~1-2 hrs |
| 3 | Quality filter (re-run with SAM2 masks) | ~5 min |
| 4 | Marigold normals/depth enrichment | ~10 min |
| 5 | Train seg (60 epochs, 3e-5 LR, loss weighting) | ~2-3 hrs |
| 6 | ONNX export (seg only) | ~2 min |
| 7 | Upload to bucket | ~5 min |

### Run 4.5 Training Data
- humanrig: ~11,400 (GT 22-class, weight 3.0)
- meshy_cc0 + meshy_cc0_textured: ~15,900 (quality-filtered, weight 1.0)
- anime_seg: ~14,000 (SAM2 pseudo-labeled 22-class, weight 1.5)
- gemini_diverse: ~700 (SAM2 pseudo-labeled 22-class, weight 2.5)
- fbanimehq: EXCLUDED (binary fg/bg only, no 22-class signal)
- **Total: ~42K examples** (was ~140K with fbanimehq — smaller but cleaner)

### Run 4.5 Config Differences from Run 4
- `dataset_weights`: humanrig 3.0, gemini_diverse 2.5, anime_seg 1.5
- fbanimehq excluded entirely
- `learning_rate`: 3e-5 (lower than run 4's 5e-5 for stability with weighted loss)
- `label_smoothing`: 0.03 (lower than run 4's 0.05 — SAM2 labels are cleaner)
- `epochs`: 60 (more patience for weighted loss convergence)
- `early_stopping_patience`: 20

### Key Files
- `scripts/run_sam2_pseudolabel.py` — SAM2 + joint-conditioned region assignment
- `training/configs/segmentation_a100_run4_5.yaml` — run 4.5 seg config
- `training/run_four_five.sh` — A100 orchestration (7 steps)

## Fifth Training Run — IN PROGRESS

**Goal: Ground-truth joints upgrade + Gemini domain expansion + encoder features refresh.** Script: `training/run_fifth.sh`

### Pre-Run 5 Tasks (local Mac, before A100)
1. [x] **Render HumanRig posed dataset** — 1,000 chars × 15 Mixamo FBX poses × 3 angles (front, three_quarter, side) = **45,000 ground-truth joint examples**. Script: `run_humanrig_posed.py`. Output: `/Volumes/TAMWoolff/data/preprocessed/humanrig_posed/`. IN PROGRESS on Mac.
2. [ ] **Upload humanrig_posed to bucket** — `rclone copy` to `humanrig_posed/` prefix. Tar-pack for fast A100 setup.
3. [x] **Generate more Gemini characters** — expanded from 221 → **698 training images** (target was 600). 698 raw images ingested via `scripts/ingest_gemini.py --only-new --no-seg`. Pseudo-labeling deferred to A100.
4. [ ] **Generate ~200 Gemini validation images** — held-out characters (never trained on). Can double as novel view benchmark for run 6+.
5. [x] **Create `training/run_fifth.sh`** — orchestration script with pre-flight checks, 8 steps.
6. [x] **Create `training/configs/joints_a100_run5.yaml`** — joints config with humanrig_posed dataset.

### Run 5 Strategy

| Step | Task | Est. Time |
|------|------|-----------|
| 0 | Download run 4 seg checkpoint + humanrig_posed tar + expanded Gemini | ~5 min |
| 1 | Verify all datasets present (fail-fast if humanrig_posed missing) | ~1 min |
| 2 | Quality filter + Marigold normals/depth on new data | ~30 min |
| 3 | **Train joints** — retrain with 45K posed GT examples | ~4-6 hrs |
| 4 | **Train seg** — resume from run 4, add expanded Gemini data (698 examples) | ~5 hrs |
| 5 | Recompute encoder features with new seg model + retrain weights | ~2-3 hrs |
| 6 | ONNX export (all 4 models) | ~5 min |
| 7 | Re-enrich Gemini data with new seg model (bootstrap for run 6) | ~5 min |
| 8 | Upload to bucket | ~5 min |

**Total estimated: ~12-14 hrs on A100.** Joints and seg are sequential (single GPU). Early stopping should help — run 3 joints stopped at epoch 36/80.

**NOT retraining in run 5**: Inpainting (keeps run 4 checkpoint).

### Run 5 Joints Data (the big upgrade)

| Dataset | Examples | Source | Quality |
|---------|--------:|--------|---------|
| HumanRig (T-pose, existing) | 11,434 | Ground truth (reprojected 3D) | High |
| **HumanRig (posed, NEW)** | **~45,000** | **Ground truth (Blender raycast)** | **High** |
| Meshy CC0 (flat + textured) | ~15,900 | Pipeline joints | Medium |
| FBAnimeHQ | ~101,630 | RTMPose | Medium (noisy on anime) |
| anime-seg v1+v2 | 14,579 | RTMPose | Medium |
| Gemini diverse | ~698 | Pseudo-labeled | Medium |
| **GT ratio** | **~56K / ~189K = 30%** | | **Up from 19%** |

The posed dataset adds diverse dynamic poses (kicks, crawling, dancing, planking, grenade throws) with perfect ground-truth joint positions via Blender raycast — no RTMPose noise. This should improve accuracy especially for non-standing poses where RTMPose struggles on illustrated characters.

### Run 5 Gemini Data

| Split | Count | Notes |
|-------|------:|-------|
| Training | 698 | Ingested, pseudo-labeling on A100 |
| Validation (held-out) | ~200 | TODO — separate characters, never trained on |
| **Total** | **~900** | Up from 221 in run 4 |

### Key Files
- `ingest/humanrig_posed_renderer.py` — production batch renderer (render_posed_sample, render_directory)
- `run_humanrig_posed.py` — Blender CLI entry point
- `scripts/ingest_gemini.py` — Gemini ingest + pseudo-label pipeline
- `pipeline/pose_applicator.py` — FBX/BVH pose retargeting (with action-clearing fix for FBX)
- `training/run_fifth.sh` — A100 orchestration (pre-flight → joints → seg → weights → ONNX → upload)
- `training/configs/joints_a100_run5.yaml` — joints config with humanrig_posed
- `training/configs/segmentation_a100_run5.yaml` — seg config with expanded Gemini
- Mixamo FBX poses: `/Volumes/TAMWoolff/data/poses/` (13 clips: Hurricane Kick, Crawling, Hip Hop Dancing, etc.)

### Production Render Command
```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python run_humanrig_posed.py -- \
    --input_dir /Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final \
    --pose_dir /Volumes/TAMWoolff/data/poses \
    --output_dir /Volumes/TAMWoolff/data/preprocessed/humanrig_posed \
    --max_samples 1000 \
    --angles front,three_quarter,side \
    --only_new
```

## Sixth Training Run — OUTLINE

**Goal: Train models 5 (texture inpainting) and 6 (novel view synthesis).**

### Data Strategy — Self-Bootstrapping Loop
Models 1-4 (trained in runs 4-5) generate the labeled data that models 5-6 need:

1. **Pretrain on 3D renders**: HumanRig + Meshy CC0 multi-angle renders provide perfect GT pairs (front→back, front→side). Style augmentation (cel-shading, flat, sketch) bridges the 3D→illustrated domain gap.
2. **Synthesize illustrated pairs**: Run models 1-4 on Gemini front-view characters to get seg+depth+normals+joints. Model 6 then generates back/side views conditioned on these features. Model 5 generates complete UV textures from partial front-view unwraps.
3. **Filter + retrain**: Confidence thresholds and cross-view consistency checks filter bad synthetic outputs. Retrain on combined 3D GT + filtered synthetic pairs.

### Model 6 (Novel View) Training Pairs
- Input: front-view image + seg + depth + normals + joints
- Target: back/side/3/4 view of same character
- 3D source: HumanRig multi-angle (built in run 5 prep) + Meshy CC0 multi-angle
- Illustrated source: Gemini front views → model-6-synthesized other views (iterative)

### Model 5 (Texture Inpainting) Training Pairs
- Input: partial UV texture (from front-view unwrap onto 3D mesh)
- Target: complete UV texture (from full 3D model)
- Source: HumanRig + Meshy CC0 GLBs (have complete textures to create partial→complete pairs)

## Seventh Run and Beyond — OUTLINE

**The flywheel**: Each run's models become data generators for the next.

- Run 7+: Models 5-6 generate multi-view illustrated pairs from Gemini characters → retrain models 5-6 on better synthetic data → improved models generate even better pairs
- **Gemini characters**: 698 training images ingested (target was 600). ~200 held-out validation characters still needed (never train on these). Quality and diversity > quantity.
- **Multi-view Gemini prompts**: Generate matched front/back/side prompts only when needed as a validation benchmark for model 6 — not for training (too inconsistent across views).
- **Scale**: 600 training front views × 4 synthesized angles = 2,400 illustrated multi-view pairs per iteration, growing with each improvement cycle.
