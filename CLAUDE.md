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
| 1 | **Segmentation** | 0.6485 test mIoU (run 20) | **>0.65 mIoU** | Boundary softening breakthrough. Next: PatchGAN (run 21), more GT labels. |
| 2 | **Joints** | 0.00121 offset (run 3) | **<0.0008** | Retrain with humanrig_posed GT joints (diverse poses). |
| 3 | **Weights** | 0.0231 MAE (run 3) | **<0.015** | Retrain with better seg encoder features. Tied to seg quality. |
| 4 | **Inpainting** | 0.0028 val/l1 (run 6) | **<0.002** | Converged — may need architecture change or illustrated training data. |
| 5 | **Texture Inpaint** | No model yet | **<0.005 val/l1** | Blocked on model 6. |
| 6 | **Back View** | 0.2152 val/l1 (run 4) | **<0.15 val/l1** | +1,244 new pairs helped. Run 5: PatchGAN discriminator. |

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

### Bucket Contents (March 24, 2026)

Bucket: `strata-training-data` at `fsn1.your-objectstorage.com`.

**Tars (for A100 setup):**
| Tar | Size | Contents |
|-----|-----:|---------|
| `humanrig.tar` | 16.8 GiB | 45,738 GT 22-class T-pose renders |
| `humanrig_posed.tar` | 12 GiB | 81,864 GT posed renders (Blender seg masks, no depth/normals) |
| `toon_pseudo.tar` | ~5 GiB | 13,740 toon-style renders (pseudo-labeled, 6,317 pass quality filter) |
| `meshy_cc0_textured_restructured.tar` | 2.8 GiB | Meshy CC0 textured (15,281 examples) |
| `vroid_cc0.tar` | 203 MiB | VRoid CC0 GT 22-class (11 chars, 1,386 examples) |
| `gemini_li_converted.tar` | 223 MiB | Dr. Li's 694 expert-labeled diverse illustrated chars |
| `cvat_annotated.tar` | 9.0 MiB | 49 hand-annotated diverse illustrated chars |
| `flux_diverse_clean.tar` | ~300 MiB | 1,569 cleaned FLUX chars (no-torso removed) |
| `sora_diverse.tar` | ~380 MiB | ~2,467 Sora/Gemini/ChatGPT chars (incl Pixabay, pseudo-labeled in run 17) |
| `back_view_pairs.tar` | 652 MiB | 1,085 back view triplets (Meshy FBX+GLB+VRoid) |
| `back_view_pairs_unrigged.tar` | 319 MiB | ~720 additional back view triplets (unrigged Meshy GLB) |

**Enriched tars** (include depth+normals, skip Marigold on A100):
`{name}_enriched.tar` for: flux_diverse_clean, gemini_li_converted, toon_pseudo. Note: sora_diverse_enriched and humanrig_posed_enriched were deleted (stale data).

**Frozen splits:** `data_cloud/frozen_val_test.json` — 3,016 val + 3,015 test characters (from 30,154 total, seed=42). Generated during run 17, persisted in bucket.

**Other prefixes:** `animation/` (67 GiB, 100STYLE mocap), `checkpoints_run*/` (runs 10-17), `models/` (ONNX exports), `logs/`.

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
| `training/run_seg_run18.sh` | Run 18: same data as run 17, frozen val/test splits |
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
| 14 | 0.5561 | +295 illustrated chars, no humanrig_posed | +2.5% from illustrated data. Pseudo-labeled humanrig_posed confirmed unusable. |
| 15 | 0.5695 | +99 illustrated chars, no humanrig_posed | +2.4%. GT humanrig_posed causes split change → mIoU regression. |
| 16 | 0.5808 | +200 illustrated + relaxed filter + frozen val | +2.0%. Frozen val set deployed. humanrig_posed confirmed harmful even with frozen val. |
| 17 | pending | +617 illustrated (2,467 total incl Pixabay), same mix | Frozen val/test generated (3,016 val + 3,015 test chars). mIoU not directly comparable (pre-freeze). |
| 18 | 0.5750 | Same data as 17, frozen val/test splits | True baseline with frozen val. Plateaued epoch 9. Per-class eval: forearm_r 0.42, feet 0.45-0.48, class 20 "unused" 69.8% acc dragging others down. |
| 19 | 0.5287 | Class 20 remap to background | Val set different (1,658 chars). mIoU not comparable to run 18 directly. |
| **20** | **0.6171 val / 0.6485 test** | **Boundary label softening (radius=2)** | **+8.8% over run 19. Biggest single improvement ever. forearm_r 0.42→0.57, feet 0.45→0.61. Neck regressed 0.62→0.45.** |

### Run 13/14 Learnings

- **humanrig_posed pseudo-labels are unusable**: Tried with run 12 labels (66% rejected), run 13a labels (94% rejected), at weights 2.0/0.5/0.3 — always causes mIoU regression. Even quality-filtered examples are too noisy.
- **GT masks needed**: humanrig_posed has GT seg masks rendered in Blender (81K examples, 0.2% rejection), but was not included in runs 16/17 configs (missing from dataset_dirs). Available for future runs.
- **Key principle**: Change one dataset at a time. Isolate what helps vs hurts.
- **Enriched tars**: Upload depth+normals-enriched datasets after training. Saves ~10 hrs Marigold per future run.
- **More illustrated data = best lever**: each batch of ~100-300 chars gives +2-3% mIoU. Keep generating.
- **Frozen val/test splits**: Generated during run 17 (3,016 val + 3,015 test chars from 30,154 total). Persisted in bucket at `data_cloud/frozen_val_test.json`. All future runs must use this file.
- **Class 20 "unused" is dead**: Not used by Strata's rigging pipeline. At 69.8% accuracy it dragged down mIoU and confused adjacent classes. Remapped to background in dataset loader for run 19+.
- **Pseudo-label ceiling**: 617 new illustrated chars in run 17/18 didn't improve mIoU. More pseudo-labeled data alone won't break through — need more GT expert labels (Dr. Li).
- **sora_diverse 52% rejection**: many illustrated images fail quality filter. May need to relax filter for illustrated data or review rejected examples.

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

## Run 18 Results (March 24, 2026)

**0.5750 mIoU** (epoch 9/20, early stopped). Frozen val/test splits (3,016 val + 3,015 test chars). Test set mIoU: 0.5995.

Config: `training/configs/segmentation_a100_run18.yaml`. Script: `training/run_seg_run18.sh`.

| Dataset | Examples (approx) | Weight |
|---------|----------|--------|
| cvat_annotated | 49 | 10.0 |
| sora_diverse | ~2,467 | 4.0 |
| gemini_li_converted | 694 | 3.0 |
| flux_diverse_clean | 1,569 | 2.5 |
| vroid_cc0 | 1,386 | 2.5 |
| meshy_cc0_textured | 15,281 | 1.5 |
| humanrig | ~45,738 | 0.5 |

### Per-Class Analysis (test set)

**Weakest classes:** forearm_r (0.42), accessory (0.45), foot_r (0.45), hand_r (0.48), foot_l (0.48).
**Strongest:** background (0.94), head (0.74), hips (0.69), chest (0.69).
**Key finding:** Class 20 "unused" at 69.8% accuracy — confusion bleeds into lower_legs and accessory. Remapped to background for run 19.
**Adjacent-region confusion** is the main error source: chest↔spine (5%), upper_arm↔shoulder (4%), lower_leg↔foot (4%), upper_arm↔forearm (3-5%).

## Next Steps

### Next Run (Run 21)
- PatchGAN discriminator on top of boundary softening
- Resume from run 20 checkpoint (0.6171 mIoU)
- Investigate neck regression (0.62→0.45) — may need to exclude neck from boundary softening
- Config: `training/configs/segmentation_a100_run21.yaml`

### Other Tasks
- [ ] Keep generating illustrated chars (Sora/Gemini/ChatGPT) — prompts 1501-2100 ready
- [ ] Waiting on Dr. Li for GT illustrated labels (emailed March 24)
- [ ] Add humanrig_posed GT (81K examples) as train-only — never tested with boundary softening
- [ ] Manually correct 100-200 pseudo-labeled illustrated examples
- [ ] 328 new Meshy CC0 characters extracted — render back view triplets
- [ ] Contact Layered Temporal Dataset authors via LinkedIn (both at Meta Reality Labs)

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
**Next:** `back_view_pairs_new.tar` in bucket (1,244 new pairs from FBX chars). Combined with existing pairs = ~3,049 total. Ready for next back view training run.

## Quality Filter

`scripts/filter_seg_quality.py` — rejects pseudo-labeled examples with implausible masks:
- `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`
- Also checks for `missing_head` and `missing_torso`
- GT datasets: <1% rejected. Pseudo-labeled: varies (humanrig_posed had 66% rejection with run 12 model).

## Dr. Li's Label Schema Conversion

Dr. Li provided 694 segmentation labels using a 19-class clothing-oriented schema. Converted to Strata's 22-class skeleton schema using joint-based L/R splitting via `scripts/convert_li_labels.py`.
