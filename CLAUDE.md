# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 6 ONNX AI models. Runs independently of the Strata codebase.

## Strata's 6 ONNX Models

Strata (Tauri/Rust/React desktop app at `../strata/`) uses 6 ONNX models defined in `src-tauri/src/ai/runtime.rs`:

| # | Model | Architecture | Training Pipeline | Status |
|---|-------|-------------|-------------------|--------|
| 1 | **Segmentation** | DeepLabV3+ MobileNetV3 (multi-head: seg + depth + normals) | `training/train_segmentation.py` | Has pipeline + data |
| 2 | **Joint Refinement** | MobileNetV3 + regression heads | `training/train_joints.py` | Has pipeline + data |
| 3 | **Weight Prediction** | Per-vertex MLP with optional encoder features | `training/train_weights.py` | Has pipeline + data |
| 4 | **Inpainting** | U-Net for occluded body regions | `training/train_inpainting.py` | Has pipeline + data |
| 5 | **Texture Inpainting** | Diffusion-based 3D texture fill | **Not yet built** | Needs pipeline + data |
| 6 | **Novel View Synthesis** | Multi-view conditioned diffusion | **Not yet built** | Needs pipeline + data |

### Model I/O

- **Segmentation** — Input: [1,3,512,512]. Outputs: 22-class logits + depth (Marigold-distilled) + normals [3ch] + confidence + encoder_features (→ model 3).
- **Joint Refinement** — Input: [1,3,512,512]. Outputs: [1,2,20] offsets + [1,20] confidence + [1,20] presence.
- **Weight Prediction** — Input A: [1,31,2048,1] vertex features. Input B (optional): encoder features from model 1. Outputs: [1,20,2048,1] weights + [1,1,2048,1] confidence. Encoder branch uses dropout so model works with or without visual context.
- **Inpainting** — U-Net fills occluded body regions. Fallback: EdgeExtend.
- **Texture Inpainting** — Fills unobserved UV texture regions. **No pipeline yet.**
- **Novel View Synthesis** — Generates unseen views from reference views. Fallback: PaletteFill. **No pipeline yet.**

Models 1-4 have complete training pipelines. All bundled in `../strata/src-tauri/models/` (~55MB total), loaded via `ort` ONNX runtime.

## Current Model Scores (best per model)

| Model | Best Score | Run | Notes |
|-------|-----------|-----|-------|
| Segmentation | 0.3573 mIoU (MobileNetV3) | Run 7 (Li + CVAT) | Up from 0.3491 baseline. ResNet-50 ref: 0.3657. Bootstrapping more diverse data for run 8 |
| Joints | 0.001206 mean_offset_error | Run 3 | 110K examples, early stopped epoch 36 |
| Weights | 0.023137 MAE | Run 3 | 12K examples (3.6x better than run 1) |
| Inpainting | 0.0028 val/l1 | Run 6 | Fully converged (50/50 epochs). No further improvement expected |

**Key insight:** Run 7 confirmed data diversity is the bottleneck. Li + CVAT annotations improved MobileNetV3 from 0.3491 → 0.3573 mIoU but plateaued at epoch 64. Next step: bootstrap hundreds of corrected pseudo-labels from the model itself.

## Project Layout

Key dirs: `pipeline/` (Blender rendering), `ingest/` (external dataset adapters), `training/` (model training + configs + run scripts), `scripts/` (utilities), `animation/` (BVH/mocap), `tests/`, `docs/`.

Entry points: `run_pipeline.py`, `run_validation.py`. Config: `pipeline/config.py`.

**Tracked:** `pipeline/`, `ingest/`, `training/`, `scripts/`, `animation/`, `tests/`, `docs/`. **Ignored:** `data/` (large binaries), `output/` (generated).

## Strata Standard Skeleton (22 classes)

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

Requires Blender 4.0+. Multi-angle rendering via `--angles` flag (front, three_quarter, side, three_quarter_back, back).

## Rendering Details

- Orthographic camera, auto-framed to bbox + 10% padding, 512×512, transparent bg
- 22 Emission-only materials for seg masks (no AA, nearest-neighbor = exact region IDs)
- Masks are per-pose, NOT per-style
- Split by **character** (not image) — 80/10/10 train/val/test

## Asset Sources & Licensing

Only CC0, CC-BY, CC-BY-SA. **Never CC-NC.**

**Prohibited (removed from training):**
- Mixamo — Adobe ToS bans AI/ML training
- Live2D — Proprietary terms
- ArtStation/curated_diverse — No AI training permission
- CartoonSegmentation — No license, Bandai Namco IP

## Storage Policy

- **Hetzner bucket** — training-ready output only (images, masks, joints, metadata)
- **External HD (`/Volumes/TAMWoolff/data/`)** — all raw source data, never delete
- **Local SSD** — avoid storing large datasets, process on external HD

### Bucket Operations — Always Use rclone

```bash
rclone copy ./output/ hetzner:strata-training-data/output/ \
  --transfers 32 --checkers 64 --fast-list --size-only -P
```

- Config: `~/.config/rclone/rclone.conf`, remote: `hetzner`
- **Never use `rclone sync`** (deletes remote files) or `aws s3 sync` (too slow)

### Bucket Contents (March 16, 2026)

Bucket: `strata-training-data` at `fsn1.your-objectstorage.com`. ~248 GiB total.

**Tars (for A100 setup):**
| Tar | Size | Contents |
|-----|-----:|---------|
| `humanrig.tar` | 16.8 GiB | 11,434 GT 22-class T-pose renders |
| `fbanimehq.tar` | 14.2 GiB | FBAnimeHQ face/body crops |
| `meshy_cc0_textured.tar` | 4.3 GiB | Meshy CC0 textured (flat structure) |
| `meshy_cc0_textured_restructured.tar` | 2.8 GiB | Meshy CC0 textured (per-example subdirs, 15,281 examples) |
| `anime_seg.tar` | 3.0 GiB | anime-segmentation v1+v2 (~14K) — 32% rejection rate, candidate for removal |
| `vroid_cc0.tar` | 203 MiB | VRoid CC0 GT 22-class (11 chars, 1,386 examples) |
| `gemini_li_converted.tar` | 223 MiB | Dr. Li's 694 expert-labeled diverse illustrated chars |
| `gemini_diverse.tar` | 131 MiB | Gemini diverse (698 images, SAM2 pseudo-labels) — needs re-upload with 885 corrected |
| `cvat_annotated.tar` | 9.0 MiB | 49 hand-annotated diverse illustrated chars |

**Other prefixes:** `animation/` (67 GiB, 100STYLE mocap), `encoder_features/` (42 GiB), `unirig/` (53 GiB), `checkpoints*/`, `models/`, `logs/`.

## A100 Training Run Workflow

1. **Prep locally** — enrich data, upload to bucket, push code
2. **Spin up A100** — `cloud_setup.sh lean` (installs deps + configures rclone, no data download)
3. **Run script** — e.g. `./training/run_seg_li.sh` (downloads only needed data, trains, exports ONNX, uploads)
4. **Destroy instance** — everything safe in bucket
5. **Download to Mac** — `rclone copy` checkpoints + ONNX
6. **Benchmark** — `python3 run_benchmark.py` (7 Gemini test characters)

### Key Scripts

| Script | Purpose |
|--------|---------|
| `training/cloud_setup.sh` | A100 setup (deps + rclone, no data) |
| `training/run_seg_li.sh` | Run 7: seg with Dr. Li + CVAT annotations |
| `training/run_seg_bootstrap.sh` | **Run 8**: bootstrapped pseudo-labels + drop anime_seg |
| `training/run_seg_backbone.sh` | Backbone comparison (EfficientNet-B3, ResNet-50) |
| `run_normals_enrich.py` | Marigold normals + depth enrichment |
| `run_benchmark.py` | Benchmark on 7 Gemini test characters |
| `scripts/convert_li_labels.py` | Convert Dr. Li's 19-class → Strata 22-class |
| `scripts/convert_cvat_export.py` | Convert CVAT export → Strata format |
| `scripts/batch_pseudo_label.py` | Batch seg inference + review manifest |
| `scripts/review_masks.py` | Tkinter UI for correcting pseudo-labeled masks |
| `scripts/filter_reviewed.py` | Extract reviewed-only examples for training |
| `scripts/ingest_gemini.py` | Preprocess Gemini images (rembg + resize + pseudo-label) |

## Run 7 Results (March 16, 2026)

**0.3573 mIoU** (epoch 64/80, manually stopped — plateau). Up from 0.3491 MobileNetV3 baseline.

Config: `training/configs/segmentation_a100_run7_li.yaml`. Script: `training/run_seg_li.sh`.

| Dataset | Examples | Weight | Source |
|---------|----------|--------|--------|
| cvat_annotated | 49 | 10.0 | Hand-annotated diverse illustrated |
| gemini_li_converted | 694 | 3.0 | Dr. Li's expert labels (19-class → 22-class converted) |
| vroid_cc0 | 1,386 | 2.5 | GT 22-class VRoid characters |
| humanrig | 11,434 | 2.0 | GT 22-class 3D renders |
| meshy_cc0_textured | 15,281 | 1.5 | GT 22-class diverse 3D |
| anime_seg | ~14K | 1.0 | Existing masks (32% rejection rate) |

**Learnings:**
- CVAT weight bumped 4.0 → 10.0 mid-run (49 examples were only 0.4% effective volume at 4.0)
- Li + CVAT annotations helped but insufficient alone to break 0.37 ceiling
- anime_seg has 32% rejection rate — noisy labels may be hurting

## Mask Correction Workflow (Bootstrapping Loop)

Use the trained model to pseudo-label new images, then correct mistakes manually. Turns 10-min CVAT annotation into ~1-2 min correction.

```bash
# 1. Preprocess new Gemini images (rembg + resize)
python scripts/ingest_gemini.py --no-seg --only-new

# 2. Pseudo-label with current best checkpoint
python scripts/batch_pseudo_label.py \
    --input-dir /Volumes/TAMWoolff/data/preprocessed/gemini/ \
    --output-dir ./output/gemini_corrected \
    --checkpoint checkpoints/segmentation/run7_best.pt

# 3. Review & correct masks in Tkinter UI
python scripts/review_masks.py --data-dir ./output/gemini_corrected

# 4. Filter reviewed-only examples
python scripts/filter_reviewed.py \
    --input-dir ./output/gemini_corrected \
    --output-dir ./output/gemini_corrected_clean

# 5. Tar and upload
tar cf gemini_corrected.tar -C ./output gemini_corrected_clean
rclone copy gemini_corrected.tar hetzner:strata-training-data/tars/ -P
```

885 images pseudo-labeled with run 7 checkpoint. Correction in progress.

## Current Plan: Run 8 Seg (Corrected Gemini + Cleanup)

**Goal: Break 0.45 mIoU with bootstrapped diverse data.**

| Dataset | Examples | Weight | Strategy |
|---------|----------|--------|----------|
| gemini_corrected | ~500-800 (TBD) | 4.0 | Human-corrected pseudo-labels from 885 Gemini images |
| cvat_annotated | 49 | 10.0 | Hand-annotated diverse illustrated |
| gemini_li_converted | 694 | 3.0 | Dr. Li's expert labels |
| vroid_cc0 | 1,386 | 2.5 | GT 22-class VRoid characters |
| humanrig | 11,434 | 2.0 | GT 22-class 3D renders |
| meshy_cc0_textured | 15,281 | 1.5 | GT 22-class diverse 3D |
| anime_seg | ~14K or DROP | 0.5 or 0 | 32% rejection rate — test with/without |

**Key changes from run 7:**
- Add corrected Gemini diverse masks at high weight (4.0) — largest illustrated dataset
- Consider dropping or downweighting anime_seg (noisy labels, 32% rejected)
- Resume from run 7 checkpoint (0.3573 mIoU)
- If plateau persists: try curriculum learning (illustrated-only warmup → full mix)

## After Seg Converges: Ship Run

Once seg hits target (>0.45 mIoU), one combined run to ship all 4 models:
- Retrain joints with 45K humanrig_posed GT examples (rendered, not yet uploaded)
- Retrain weights with new seg encoder features
- Inpainting already converged — keep run 6 checkpoint
- ONNX export all 4 → ship to `../strata/src-tauri/models/`

## Future: Models 5-6 (Self-Bootstrapping)

Models 1-4 generate labeled data for models 5-6:
- **Model 6 (Novel View)**: 3D multi-angle GT pairs (HumanRig, Meshy) → pretrain, then synthesize illustrated pairs
- **Model 5 (Texture Inpainting)**: Partial→complete UV texture pairs from 3D models
- Each run's models become data generators for the next

## Dr. Li's Label Schema Conversion

Dr. Li provided 694 segmentation labels using a 19-class clothing-oriented schema (indices ×10: hair=0, face=20, topwear=110, bottomwear=130, etc.). Converted to Strata's 22-class skeleton schema using joint-based L/R splitting:
- topwear → chest/spine/shoulder/upper_arm (split L/R by joint positions + midline)
- handwear → forearm/hand, bottomwear → hips/upper_leg, legwear → lower_leg, footwear → foot
- Dresses labeled as leg regions (not accessories) — correct for rigging convention

Script: `scripts/convert_li_labels.py`. Pipeline: align raw images (rembg + crop) → joints inference → convert labels with joint-based splitting.
