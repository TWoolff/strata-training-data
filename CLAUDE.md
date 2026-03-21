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
| 5 | **Texture Inpainting** | U-Net partial→complete RGBA fill | `training/train_texture_inpainting.py` | Needs pipeline + data (depends on model 6) |
| 6 | **Back View Generation** | U-Net (front+3/4 → back) | `training/train_back_view.py` | Has pipeline + data (~1,805 pairs) |

### Model I/O

- **Segmentation** — Input: [1,3,512,512]. Outputs: 22-class logits + depth (Marigold-distilled) + normals [3ch] + confidence + encoder_features (→ model 3).
- **Joint Refinement** — Input: [1,3,512,512]. Outputs: [1,2,20] offsets + [1,20] confidence + [1,20] presence.
- **Weight Prediction** — Input A: [1,31,2048,1] vertex features. Input B (optional): encoder features from model 1. Outputs: [1,20,2048,1] weights + [1,1,2048,1] confidence.
- **Inpainting** — U-Net fills occluded body regions. Fallback: EdgeExtend.
- **Texture Inpainting** — Input: [B,5,512,512] (RGBA + observation mask). Output: [B,4,512,512] completed RGBA. Depends on model 6.
- **Back View Generation** — Input: [B,8,512,512] (front RGBA + 3/4 RGBA concatenated). Output: [B,4,512,512] back view RGBA.

Models 1-4 have complete training pipelines. All bundled in `../strata/src-tauri/models/` (~55MB total), loaded via `ort` ONNX runtime.

## Model Targets & Current Scores

Goal: uploaded 2D character illustrations look natural from all generated angles when rigged and posed.

| # | Model | Current Best | Target | What moves the needle |
|---|-------|-------------|--------|----------------------|
| 1 | **Segmentation** | 0.5561 mIoU (run 14) | **>0.65 mIoU** | More illustrated data (Sora/Gemini/ChatGPT). GT humanrig_posed masks. |
| 2 | **Joints** | 0.00121 offset (run 3) | **<0.0008** | Retrain with humanrig_posed GT joints (diverse poses). |
| 3 | **Weights** | 0.0231 MAE (run 3) | **<0.015** | Retrain with better seg encoder features. Tied to seg quality. |
| 4 | **Inpainting** | 0.0028 val/l1 (run 6) | **<0.002** | Converged — may need architecture change or illustrated training data. |
| 5 | **Texture Inpaint** | No model yet | **<0.005 val/l1** | Blocked on model 6. |
| 6 | **Back View** | 0.2368 val/l1 (run 13a) | **<0.15 val/l1** | More training pairs from new Meshy characters. |

**Priority order:**
1. Seg to 0.65 — keep adding illustrated data + bootstrapping loop
2. Back view to 0.15 — render more pairs from 328 new Meshy characters
3. Ship run — retrain joints + weights with better seg encoder
4. Texture inpainting — depends on back view
5. Inpainting — may need architecture work

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

### Bucket Contents (March 20, 2026)

Bucket: `strata-training-data` at `fsn1.your-objectstorage.com`. ~142 GiB total.

**Tars (for A100 setup):**
| Tar | Size | Contents |
|-----|-----:|---------|
| `humanrig.tar` | 16.8 GiB | 45,738 GT 22-class T-pose renders |
| `humanrig_posed.tar` | ~20 GiB | 81,864 posed renders (images+seg+joints, no depth/normals) |
| `toon_pseudo.tar` | ~5 GiB | 13,740 toon-style renders (pseudo-labeled, 6,317 pass quality filter) |
| `meshy_cc0_textured_restructured.tar` | 2.8 GiB | Meshy CC0 textured (15,281 examples) |
| `vroid_cc0.tar` | 203 MiB | VRoid CC0 GT 22-class (11 chars, 1,386 examples) |
| `gemini_li_converted.tar` | 223 MiB | Dr. Li's 694 expert-labeled diverse illustrated chars |
| `cvat_annotated.tar` | 9.0 MiB | 49 hand-annotated diverse illustrated chars |
| `flux_diverse_clean.tar` | ~300 MiB | 1,569 cleaned FLUX chars (no-torso removed) |
| `sora_diverse.tar` | ~200 MiB | 1,279 Sora/Gemini chars (run 10 labels) |
| `back_view_pairs.tar` | 652 MiB | 1,085 back view triplets (Meshy FBX+GLB+VRoid) |
| `back_view_pairs_unrigged.tar` | 319 MiB | ~720 additional back view triplets (unrigged Meshy GLB) |

**Enriched tars** (include depth+normals, skip Marigold on A100):
`{name}_enriched.tar` uploaded after run 13a for: sora_diverse, flux_diverse_clean, gemini_li_converted, toon_pseudo, humanrig_posed. The run script tries enriched tars first, falls back to regular. Saves ~10 hrs Marigold per A100 session.

**Other prefixes:** `animation/` (67 GiB, 100STYLE mocap), `checkpoints_run*/` (runs 10, 12, 13), `models/` (ONNX exports), `logs/`.

## A100 Training Run Workflow

1. **Prep locally** — ingest new images, upload to bucket, push code
2. **Spin up A100** — `cloud_setup.sh lean` (installs deps + configures rclone)
3. **Run script** — e.g. `./training/run_seg_run13.sh` (downloads data, trains, exports ONNX, uploads)
4. **Destroy instance** — everything safe in bucket
5. **Download to Mac** — `rclone copy` checkpoints + ONNX
6. **Benchmark** — `python3 run_benchmark.py` (7 Gemini test characters)

### Key Scripts

| Script | Purpose |
|--------|---------|
| `training/cloud_setup.sh` | A100 setup (deps + rclone) |
| `training/run_seg_run13.sh` | Run 13a: run 12 mix + toon_pseudo, enriched tar upload, + back view |
| `training/run_ship.sh` | Ship run: retrain joints+weights, export all 4 ONNX |
| `run_normals_enrich.py` | Marigold normals + depth enrichment |
| `run_benchmark.py` | Benchmark on 7 Gemini test characters |
| `scripts/ingest_gemini.py` | Preprocess Gemini/Sora/ChatGPT images (rembg + resize) |
| `scripts/batch_pseudo_label.py` | Batch seg inference for pseudo-labeling |
| `scripts/auto_triage.py` | Auto-accept/reject pseudo-labels by mask heuristics |
| `scripts/filter_seg_quality.py` | Quality filter: min-regions, max-single-region, min-foreground |
| `scripts/render_back_view_data.py` | Render front+3/4+back triplets from FBX/GLB |
| `scripts/render_toon_styles.py` | Render FBX/GLB in toon styles (multi-angle) |
| `scripts/convert_li_labels.py` | Convert Dr. Li's 19-class → Strata 22-class |

## Segmentation Run History

| Run | mIoU | Key change | Learnings |
|-----|------|-----------|-----------|
| 7 | 0.3573 | +Dr. Li labels, +CVAT | Li + CVAT insufficient alone to break 0.37 |
| 8 | 0.4721 | Drop anime_seg, +gemini_diverse pseudo-labels | Bootstrapping loop validated. 32% jump |
| 10 | 0.5038 | Bootstrap round 3 | Incremental gains |
| 12 | 0.5068 | +sora_diverse, +flux_diverse_clean, lr=1e-5 flat | Val plateau — needs more data diversity |
| 13a | 0.5425 | +toon_pseudo (wt 1.0), run 12 mix unchanged | +7% from toon data alone. Change one thing at a time. |
| **14** | **0.5561** | +295 illustrated chars, no humanrig_posed | **+2.5% from illustrated data. Pseudo-labeled humanrig_posed confirmed unusable.** |

### Run 13/14 Learnings

- **humanrig_posed pseudo-labels are unusable**: Tried with run 12 labels (66% rejected), run 13a labels (94% rejected), at weights 2.0/0.5/0.3 — always causes mIoU regression. Even quality-filtered examples are too noisy.
- **GT masks needed**: humanrig_posed needs GT seg masks rendered in Blender (seg-only mode), not pseudo-labels. Rendering in progress on Mac (~81K examples).
- **Key principle**: Change one dataset at a time. Isolate what helps vs hurts.
- **Enriched tars**: Upload depth+normals-enriched datasets after training. Saves ~10 hrs Marigold per future run.
- **More illustrated data = best lever**: 295 new chars gave +2.5% mIoU (run 14). Keep generating.

## Bootstrapping Loop

Model → pseudo-label new data → quality filter → retrain. Each cycle improves both the model and the pseudo-labels.

```bash
# 1. Ingest new images (rembg + resize, no seg)
python scripts/ingest_gemini.py --input-dir /Volumes/TAMWoolff/data/raw/gemini_diverse \
    --output-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse --no-seg --only-new

# 2. Pseudo-label on A100 with best checkpoint (handled by run script)

# 3. Quality filter auto-rejects bad masks (handled by run script)

# 4. Train with filtered pseudo-labels
```

## Run 13a Results (March 20, 2026)

**0.5425 mIoU** (epoch 19/20). Up from 0.5068 (run 12) — **+7% improvement**.

Config: `training/configs/segmentation_a100_run13.yaml`. Script: `training/run_seg_run13.sh`.

| Dataset | Examples (passed) | Weight |
|---------|----------|--------|
| cvat_annotated | 49 | 10.0 |
| sora_diverse | 1,279 | 4.0 |
| gemini_li_converted | 694 | 3.0 |
| flux_diverse_clean | 1,569 | 2.5 |
| vroid_cc0 | 1,386 | 2.5 |
| toon_pseudo | 6,317 | 1.0 |
| meshy_cc0_textured | 15,281 | 1.5 |
| humanrig | 11,434 | 0.5 |

## Next Steps

### Immediate (before next A100 run)
- [ ] GT seg masks for humanrig_posed — rendering on Mac via Blender seg-only mode (~81K renders, in progress)
- [ ] Keep generating illustrated chars (Sora/Gemini/ChatGPT) — 200 new prompts ready (1501-1700)
- [ ] Ingest + tar new illustrated images → upload to bucket
- [ ] 328 new Meshy CC0 characters extracted — render back view triplets
- [ ] Contact Layered Temporal Dataset authors via LinkedIn (both at Meta Reality Labs)

### Next A100 Run (Run 15)
- Resume from run 14 (0.5561 mIoU)
- Add GT humanrig_posed masks (from Blender, not pseudo-labels)
- Add more illustrated chars (ongoing generation)
- Target: >0.60 mIoU

### Ship Run (after seg >0.65)
- Retrain joints with humanrig_posed GT (diverse poses)
- Retrain weights with new seg encoder features
- Inpainting: keep run 6 checkpoint (already converged)
- ONNX export all 4 → `../strata/src-tauri/models/`

## Model 6: Back View Generation

**Status:** Run 3 complete. 0.2408 val/l1 — improving but still overfitting.

| Run | val/l1 | Pairs | Notes |
|-----|--------|-------|-------|
| 1 | 0.2982 | 1,085 | First model, clear overfitting |
| 2 | 0.2354 | 1,085 | Same data, longer training |
| 3 | 0.2408 | 1,805 | Fixed unrigged merge, early stopped 127/200 |

**Data:** `scripts/render_back_view_data.py` renders front+3/4+back triplets from 3D characters.
**Architecture:** U-Net, input [B,8,512,512], output [B,4,512,512].
**Loss:** L1 (alpha-weighted) + perceptual (VGG) + palette consistency.
**Next:** 328 new Meshy CC0 characters available for rendering more pairs.

## Quality Filter

`scripts/filter_seg_quality.py` — rejects pseudo-labeled examples with implausible masks:
- `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`
- Also checks for `missing_head` and `missing_torso`
- GT datasets: <1% rejected. Pseudo-labeled: varies (humanrig_posed had 66% rejection with run 12 model).

## Dr. Li's Label Schema Conversion

Dr. Li provided 694 segmentation labels using a 19-class clothing-oriented schema. Converted to Strata's 22-class skeleton schema using joint-based L/R splitting via `scripts/convert_li_labels.py`.
