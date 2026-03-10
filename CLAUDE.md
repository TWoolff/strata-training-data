# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 6 ONNX AI models. Also houses animation intelligence scripts and curated metadata for animation training. Runs independently of the Strata codebase.

## Strata's 6 ONNX Models

Strata (Tauri/Rust/React desktop app at `../strata/`) uses 6 ONNX models defined in `src-tauri/src/ai/runtime.rs`:

| # | Model | ONNX File | Architecture | Training Pipeline | Status |
|---|-------|-----------|-------------|-------------------|--------|
| 1 | **Segmentation** | `segmentation.onnx` | DeepLabV3+ MobileNetV3 (multi-head: seg + depth + normals) | `training/train_segmentation.py` | Has pipeline + data |
| 2 | **Joint Refinement** | `joint_refinement.onnx` | MobileNetV3 + regression heads | `training/train_joints.py` | Has pipeline + data |
| 3 | **Weight Prediction** | `weight_prediction.onnx` | Per-vertex MLP with optional encoder features (31→128→256→128→20, +encoder branch) | `training/train_weights.py` | Has pipeline + data |
| 4 | **Inpainting** | `inpainting.onnx` | U-Net for occluded body regions | `training/train_inpainting.py` | Has pipeline, broken data loader |
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

Models 1-3 have complete training pipelines with configs for local (4070 Ti), lean A100, and full A100 runs. Models 4-6 still need training pipelines built in this repo. All models are bundled in `../strata/src-tauri/models/` (~55MB total) and loaded lazily via the `ort` ONNX runtime crate.

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

### Bucket Contents (as of March 4 2026)

Bucket: `strata-training-data` at `fsn1.your-objectstorage.com`. ~790K+ files, ~160+ GiB total.

| Prefix | Files | Size | Source |
|--------|------:|-----:|--------|
| `animation/` | 18,628 | 66.7 GiB | 100STYLE retargeted mocap (100 styles × 8-10 contents) |
| `anime_instance_seg/` | ~135K | ~15 GiB | CartoonSegmentation instance masks (partially uploaded ~45K) |
| `anime_seg/` | ~65K | ~3.5 GiB | SkyTNT anime-segmentation v1+v2, RTMPose joints enriched |
| `animerun/` | 11,276 | 663 MiB | AnimeRun v2.2 segmentation frames |
| `animerun_correspondence/` | 19,493 | 930 MiB | AnimeRun cross-frame correspondence |
| `animerun_flow/` | 16,704 | 11.6 GiB | AnimeRun optical flow pairs |
| `animerun_linearea/` | 4,236 | 119 MiB | AnimeRun line area maps |
| `animerun_segment/` | 11,276 | 628 MiB | AnimeRun segment labels |
| `conr/` | ~7,269 | ~580 MiB | CoNR multi-view anime character sheets |
| `fbanimehq/` | 304,889 | 11.4 GiB | FBAnimeHQ face/body crops |
| `humanrig/` | 148,643+ | 5.6+ GiB | HumanRig rendered chars + joints + weights (incl. 11,434 weight.json) |
| `ingest/vroid_lite/` | 9,302 | 771 MiB | VRoid Lite CC0 characters |
| `instaorder/` | ~11,868 | ~1.5 GiB | InstaOrder draw order maps (val split) |
| `live2d/` | 3,587 | 212 MiB | Live2D .moc3 rendered models |
| `nova_human/` | ~40K | ~2.5 GiB | NOVA-Human ortho views + RTMPose joints |
| `segmentation/` | 12,216 | 599 MiB | Mixamo pipeline segmentation output |
| `unirig/` | 66,030+ | 42.6+ GiB | UniRig rigged meshes (+ 14,950 weight.json pending upload) |


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

Score: **0.5453 mIoU** (run 1, epoch 94/100, March 6 2026). Run 2 regressed to 0.38 (killed early).

### Run 3 changes (code ready, not yet trained):
- Depth head: retrained with Marigold depth labels (replaces draw_order)
- Normals head: new 3-channel output (tanh), L1 loss against Marigold normals labels
- ~14K+ examples with depth.png + normals.png (segmentation, live2d, humanrig, unirig)
- ONNX outputs: segmentation, depth, normals, confidence, encoder_features (5 heads)
- PRD for Strata Rust runtime changes: `docs/prd-segmentation-model-v2.md`
- Goal: mIoU 0.55+ (recover from run 2), depth + normals quality approaching Marigold

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

Score: **BROKEN** — run 1 failed. Root cause: `discover_images()` used `rglob("*.png")` which grabbed segmentation masks and depth maps as source images, producing almost no valid pairs. Fixed in commit `011c3ba` — now uses `glob("*/image.png")`. Also capped at 15K source images (~45K pairs). Not retrained in run 3 — scheduled for run 4.

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

## Fourth Training Run — PLAN

**Goal: Ship-ready models 1-4.** Script: `training/run_fourth.sh`

### Pre-Run 4 Checklist
1. [x] **Save run 1 seg checkpoint** — saved to bucket as `checkpoints_run1/segmentation/best.pt`. Run 3's seg (0.3728) is worse — run 4 resumes from run 1.
2. [x] **Finish run 3** — encoder features (11,434) + diffusion weights (0.0216 MAE) trained. All checkpoints uploaded as `checkpoints_run3/`.
3. [ ] **Ingest Gemini images** — run `scripts/ingest_gemini.py` on all ~300 Gemini illustrations, using run 1's seg checkpoint for pseudo-labeling 22-class masks. Upload to bucket as `gemini_diverse/`.
4. [ ] **Run quality filter locally** — run `scripts/filter_seg_quality.py` on meshy_cc0, meshy_cc0_textured, humanrig, unirig to generate `quality_filter.json` per dataset. Upload each to bucket. The seg dataset loader auto-reads this file to skip rejected examples.
5. [ ] **Push all code fixes** — ensure split_loader, weight_dataset, precompute, filter script, and seg dataset quality filter support are in main.

### Pseudo-Labeling Strategy (self-training loop)
Run 1's seg model (0.545 mIoU) pseudo-labels Gemini images → run 4 trains on them → run 4's improved model re-labels Gemini data at the end of the run (step 7 of `run_fourth.sh`), bootstrapping better labels for future runs.

### Run 4 Strategy

| Step | Task | Est. Time |
|------|------|-----------|
| 0 | Download run 1 seg + run 3 joints/weights checkpoints | ~2 min |
| 1 | Download Gemini diverse dataset | ~2 min |
| 2 | Run quality filter on seg masks | ~10 min |
| 3 | Marigold normals/depth enrichment on new data | ~30 min |
| 4 | **Train segmentation** — resume from run 1 (0.545 mIoU), fine-tune 50 epochs at 5e-5 LR, label smoothing 0.05 | ~2-3 hrs |
| 5 | Generate inpainting pairs + train inpainting | ~1 hr |
| 6 | ONNX export (all 4 models) | ~5 min |
| 7 | Re-enrich Gemini data with new seg model (bootstrap for run 5) | ~5 min |
| 8 | Upload to bucket | ~5 min |

**NOT retraining in run 4**: Joints (0.001206 is good enough), Weights (0.023 MAE is 3.6x better than run 1).

### Run 4 Seg Data Strategy
- **Quality filter** per-example masks: reject <4 regions, >70% single region, missing head/torso
- **Keep**: humanrig (11K, ground-truth), unirig (~10K), anime_seg (14K binary), gemini_diverse (~300 pseudo-labeled)
- **Filter**: meshy_cc0 + meshy_cc0_textured — keep only examples that pass quality filter
- **Remove**: live2d (prohibited license)
- **Label smoothing**: 0.05 to reduce impact of remaining noisy labels
- **Expected**: mIoU 0.55-0.65 (recover run 1 quality + gain from cleaner data + Gemini domain diversity)
