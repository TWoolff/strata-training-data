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
| 6 | **Back View Generation** | U-Net (front+3/4 → back) | `training/train_back_view.py` | Has pipeline + data (~3,049 pairs) |

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
| 1 | **Segmentation** | 0.6485 test mIoU (run 20) | **>0.70 mIoU** | Blocked: SAM labels don't help (clothing≠anatomy). Wait for Dr. Li's training code (April 12). |
| 2 | **Joints** | 0.00121 offset (run 3) | **<0.0008** | Retrain with humanrig_posed GT joints (diverse poses). |
| 3 | **Weights** | 0.0215 MAE (retrained) | **<0.015** | Retrained with run 20 seg encoder. 7% better than old 0.0231. ONNX in bucket. |
| 4 | **Inpainting** | 0.0028 val/l1 (run 6) | **<0.002** | Converged — low priority. |
| 5 | **Texture Inpaint** | No model yet | — | May not need if 3D reconstruction works. |
| 6 | **View Synthesis** | 0.2047 val/l1 (run 2) | — | U-Net too weak for sharp views. Replacing with SHARP 3D Gaussian approach. |

**Priority order:**
1. **SHARP 3D fine-tune** — single image → 3D Gaussian splats, fine-tuned on turnaround sheets (next A100 run)
2. Seg blocked until April 12 (Dr. Li's multi-decoder training code)
3. Retrain joints with humanrig_posed GT
4. Demo video once SHARP produces usable 3D from bear chef

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

### Bucket Contents (March 27, 2026)

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
| `back_view_pairs_new.tar` | ~500 MiB | 1,244 new pairs from Meshy FBX chars |
| `sora_diverse_new.tar` | 74 MiB | 291 new illustrated chars (March 27 batch) |
| `soft_targets_precomputed.tar` | 1.9 GiB | Precomputed boundary-softened seg targets for all datasets |
| `demo_back_view_pairs.tar` | 3.4 GiB | 6,210 illustrated pairs (407 chars from 199 turnaround sheets + bear chef) |
| `sam_labels.tar` | 28 MiB | Raw SAM Body Parsing npz files for 2,467 sora_diverse images |
| `sam_seg_converted.tar` | small | SAM labels converted to Strata 22-class (old conversion, superseded) |

**Enriched tars** (include depth+normals, skip Marigold on A100):
`{name}_enriched.tar` for: flux_diverse_clean, gemini_li_converted, toon_pseudo. Note: sora_diverse_enriched and humanrig_posed_enriched were deleted (stale data).

**Frozen splits:** `data_cloud/frozen_val_test.json` — frozen val/test characters. Persisted in bucket.

**Other prefixes:** `animation/` (67 GiB, 100STYLE mocap), `checkpoints_run*/` (runs 10-21), `checkpoints_back_view*/` (runs 1-5), `models/` (ONNX exports), `evaluation_run*/`, `logs/`.

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
| `training/run_seg_run19.sh` | Run 19: class 20 remap, frozen splits |
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
| `scripts/convert_sam_labels.py` | Convert SAM Body Parsing 19-class → Strata 22-class (anatomy-aware) |
| `training/train_sharp.py` | Fine-tune SHARP for illustrated character 3D reconstruction |
| `training/run_sharp_finetune.sh` | A100 run script for SHARP fine-tuning |

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
| 20 | 0.6171 val / 0.6485 test | Boundary label softening (radius=2) | +8.8% over run 19. Biggest single improvement ever. forearm_r 0.42→0.57, feet 0.45→0.61. Neck regressed 0.62→0.45. |
| 21 | 0.6060 val / 0.6361 test | Re-pseudo-label with run 20 model, no softening | sora_diverse 854→2,122 through filter. +7.5% from better pseudo-labels alone. Neck 0.5558 (better than run 20's 0.4515). |
| 22 | 0.5597 val (epoch 10) | SAM labels + improved conversion + no softening | SAM labels don't help — clothing→anatomy mismatch. Peaked at 0.5597, well below run 20. |
| 22b | 0.5891 val (epoch 4) | SAM labels + boundary softening | Softening + SAM still worse than run 20. SAM labels are fundamentally wrong domain. |

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
- **Boundary label softening is the biggest lever found**: Gaussian blur at body-part boundaries converts hard one-hot targets to soft distributions. Interior pixels keep hard labels. Reduced mIoU penalty for ambiguous boundary pixels, letting the model focus on interior classification. +8.8% val mIoU in a single run.
- **Boundary softening hurts small/thin regions**: Neck (0.62→0.45) and accessory (0.45→0.29) regressed because softening blurs them into neighbors. Future runs should exclude these from softening or reduce radius for small regions.
- **PatchGAN doesn't help at 3K pairs**: Back view run 5 with gan_weight=0.01 was worse than run 4. Discriminator overfits quickly and destabilizes generator. GAN needs 5K+ pairs minimum.
- **A100 is 40GB, not 80GB**: Can't run seg + back view in parallel. Boundary softening adds ~700MB/batch for soft targets [B,22,512,512]. Batch 32 OOMs on 40GB — use batch 16.
- **SAM2 pseudo-labels are poor for body parts**: SAM2 segments by visual features (clothing, hair), not anatomy. Joint-based region assignment gave only 13.3% match rate. SAM2 is not a viable pseudo-labeler for our task.
- **Bootstrapping loop works again with stronger model**: Re-pseudo-labeling sora_diverse with run 20 model (0.6485) pushed pass rate from 854 to 2,122 (+149%). Each bootstrapping round now has much more impact.
- **Boundary softening must be precomputed**: scipy gaussian_filter + max_filter per sample is too slow for training (killed A100 process). Precompute as `.npz` files (~17KB each, 1.9GB total for all datasets). Loader checks for `.npz` before falling back to on-the-fly computation.
- **Meshy tar extracts as `meshy_cc0_restructured`** (not `meshy_cc0_textured` or `meshy_cc0_textured_restructured`). Must use this exact name in configs.
- **Gemini turnaround sheets are the best data source**: Single-image prompt generates 5 consistent views (front, 3/4, side, back 3/4, back). 81 sheets → 171 characters → 2,640 training pairs. Can generate ~120 sheets/day. Script `split_turnaround.py` auto-splits with rembg.
- **Unified view synthesis beats back-view-only**: Same U-Net, 9ch input (vs 8ch), handles any target angle. Run 1: 0.2139 val/l1 on multi-angle task ≈ old back-view-only model (0.2152) but far more capable.
- **Rectified flow failed**: Run 7 with flow matching loss severely overfit (train 0.06, val 0.37). L1 + perceptual loss remains the reliable choice.
- **Foot rendering is the biggest visual artifact**: Model shows front-facing toes instead of heels in back views. Doesn't understand 3D rotation — needs more training data with visible foot rotation across views.
- **Soapbox VC opportunity**: Connected with CEO Jesse Heasman. Preparing video demo of 3D character rotation from 2D illustrations.
- **SAM Body Parsing labels don't help segmentation**: SAM segments by clothing (topwear, legwear, handwear), not anatomy (forearm, shoulder, upper_arm). The 19→22 class conversion (`convert_sam_labels.py`) can't bridge this gap. Rewrote with anatomy-aware logic (connected components for L/R, torso core detection for arm splitting, legwear→hips+upper_leg+lower_leg when no bottomwear), but still plateaued at 0.56-0.59. Run 20 (0.6485) remains best. All illustrated datasets (sora_diverse, flux_diverse_clean, gemini_li_converted) have clothing-based labels, not anatomy.
- **gemini_li_converted labels are also clothing-based**: The 694 labels from Dr. Li were from her model, not hand-annotated. Same clothing→anatomy mismatch as SAM.
- **View synthesis U-Net is fundamentally limited**: 29M param U-Net + L1/perceptual loss produces blurry novel views. Bear chef fine-tune at lr=1e-4 (val/l1=0.0548) had tile artifacts on unseen angles. At lr=1e-5 (val/l1=0.0853) preserved general ability but too blurry. Mixed training (full dataset + bear chef at 10-20x weight) barely improves over general model. L1 loss learns pixel averages → inherently blurry.
- **SHARP (Apple) is the replacement for view synthesis**: Single image → 3D Gaussian splats in <1s on Mac MPS. Open-source (Apache code, research-only model weights). Runs inference on MPS but training needs CUDA (gsplat renderer). Fine-tuning pipeline built: `training/train_sharp.py` + `training/data/sharp_dataset.py` + `training/run_sharp_finetune.sh`.
- **Python 3.14 multiprocessing breaking change**: Default start method changed to forkserver (requires pickling). Our dataset stores unpicklable module ref. Fixed with `multiprocessing.set_start_method("fork")`.
- **see-through repo setup**: Clone to ../see-through, `sys.path.insert(0, '../see-through/common')`, install `einops pycocotools segment-anything-hq timm`.
- **Never `pkill -f python3` on cloud instances**: Kills system processes and crashes the instance.

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

## Run 20 Results (March 26, 2026)

**0.6171 val mIoU / 0.6485 test mIoU** (epoch 17/20). Boundary label softening (radius=2, sigma=1.0).

Config: `training/configs/segmentation_a100_run20.yaml`. Batch size 16 (soft targets need extra memory).

| Dataset | Examples (approx) | Weight |
|---------|----------|--------|
| cvat_annotated | 49 | 10.0 |
| sora_diverse | 854 | 4.0 |
| gemini_li_converted | 694 | 3.0 |
| flux_diverse_clean | 1,569 | 2.5 |
| vroid_cc0 | 1,386 | 2.5 |
| meshy_cc0_textured | 15,281 | 1.5 |
| humanrig | ~45,738 | 0.5 |

### Per-Class Analysis (run 20 vs run 18 test set)

| Class | Run 18 | Run 20 | Change |
|-------|--------|--------|--------|
| background | 0.9354 | 0.9585 | +2.3% |
| head | 0.7413 | 0.7911 | +5.0% |
| neck | 0.6243 | **0.4515** | **-17.3% regression** |
| chest | 0.6863 | 0.7103 | +2.4% |
| spine | 0.6306 | 0.6800 | +4.9% |
| hips | 0.6933 | 0.7382 | +4.5% |
| shoulder_l | 0.6826 | 0.7473 | +6.5% |
| forearm_r | 0.4216 | 0.5685 | **+14.7%** |
| foot_r | 0.4526 | 0.6084 | **+15.6%** |
| foot_l | 0.4811 | 0.6255 | **+14.4%** |
| accessory | 0.4526 | **0.2901** | **-16.3% regression** |

**Key findings:**
- Boundary softening gave biggest improvement ever (+8.8% val mIoU over run 19)
- Massive gains on weak classes: forearms, feet, hands all improved 7-16%
- **Neck regressed badly** (0.62→0.45) — small thin region gets softened into neighbors. May need to exclude neck/accessory from softening.
- **Accessory regressed** (0.45→0.29) — same issue, small scattered regions.
- Class 20 "unused" correctly at 0.00 (remapped to background)

## Next Steps

### Next A100 Run (April 2) — SHARP Fine-tune
**Script:** `./training/run_sharp_finetune.sh` | **Storage:** 30 GB | **Time:** ~4-5 hrs
```
git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
./training/cloud_setup.sh lean
./training/run_sharp_finetune.sh
```

**SHARP 3D Gaussian fine-tune:**
- Apple's SHARP model: single image → 3D Gaussian splats in <1s
- Fine-tune on ~6,200 turnaround sheet pairs (demo_back_view_pairs.tar)
- Frozen DINOv2 encoder, train decoder + heads only
- gsplat differentiable renderer for multi-view supervision loss
- lr=1e-5, 300 samples/epoch, 20 epochs, internal_resolution=768
- Goal: teach SHARP that illustrated characters are 3D beings, not flat cards
- **License: research-only weights** — for production need own model or MIT alternative

### Completed April 1
- [x] **SAM labels tested — don't help seg** (clothing≠anatomy, peaked at 0.56-0.59 vs run 20's 0.6485)
- [x] Rewrote `convert_sam_labels.py` with anatomy-aware logic (all 22 classes present)
- [x] SAM inference pipeline working: see-through repo + 19-class SAM on A100
- [x] Bear chef view synthesis fine-tune (lr=1e-4: val/l1=0.0548, lr=1e-5: val/l1=0.0853)
- [x] Both bear chef ONNX models exported and in bucket
- [x] **View synthesis U-Net confirmed too weak** for sharp novel views (blurry output from L1 loss)
- [x] SHARP installed and tested on Mac MPS (7s inference, produces .ply Gaussian splats)
- [x] SHARP fine-tuning pipeline built: `train_sharp.py` + `sharp_dataset.py` + `run_sharp_finetune.sh`
- [x] Fixed Python 3.14 multiprocessing breaking change in train_segmentation.py

### Blocked — Waiting
- **Seg >0.70 mIoU** — blocked until Dr. Li's training code (April 12). SAM/pseudo-labels can't bridge clothing→anatomy gap. Need anatomy-native model or hand-annotated illustrated data.
- **Demo video** — waiting on SHARP fine-tune results for usable 3D bear chef rotation

### After SHARP
1. **Retrain joints** with humanrig_posed GT (diverse poses)
2. **Retrain weights** with improved seg encoder (once seg improves)
3. **Dr. Li's training code** (April 12) — multi-decoder approach for anatomy-native seg
4. **More turnaround sheets** — 150 prompts ready (TS-001 to TS-150)
5. Try Dr. Li's anime-tuned Marigold depth (`24yearsold/seethroughv0.0.1_marigold`)

### Later — Post Launch
- Production 3D reconstruction (train own model with MIT license, or use cloud API)
- Interactive view correction paint tool in Strata
- Blueprint marketplace
- InnoFounder application (August 2026 start, 430K DKK grant)
- Approach PreSeed Ventures / Accelerace / byFounders for angel investment

## Model 6: View Synthesis → 3D Reconstruction

### U-Net View Synthesis (deprecated)

U-Net approach (29M params, L1+perceptual loss) produces inherently blurry novel views. L1 loss learns pixel averages. GAN loss failed at 3K pairs. **Replaced by SHARP 3D Gaussian approach.**

| Run | val/l1 | Notes |
|-----|--------|-------|
| run 2 (general) | 0.2047 | Best general model. Works OK for humanoid chars, fails on bear chef. |
| bear chef lr=1e-4 | 0.0548 | Sharp on training pairs, tile artifacts on unseen angles. Destroyed general ability. |
| bear chef lr=1e-5 | 0.0853 | Smoother but too blurry. Barely moved from general model. |
| mixed (20x bear chef) | 0.1936 | Full dataset + bear chef at high weight. Barely better than general. |

**Conclusion:** U-Net + L1 loss cannot produce sharp novel views. Need fundamentally different approach.

### SHARP 3D Gaussian Approach (new)

**Apple SHARP** — single image → 3D Gaussian splats in <1s. Feedforward neural network, no diffusion.
- **Repo:** `../ml-sharp/` (cloned from `github.com/apple/ml-sharp`)
- **Architecture:** DINOv2-L16 encoder → monodepth → Gaussian decoder → 1.18M Gaussians
- **Inference:** Works on Mac MPS (~7s) and CUDA (<1s)
- **Training:** Requires CUDA (gsplat differentiable renderer for backprop)
- **License:** Code is Apache-2.0. **Model weights are research-only** (no commercial use). For production: train own model from scratch.
- **Current issue:** Interprets illustrated characters as flat cards (photos of paintings). Fine-tuning should teach it that characters are 3D beings.

**Fine-tuning pipeline:**
- `training/train_sharp.py` — training loop with gsplat rendering loss
- `training/data/sharp_dataset.py` — turnaround sheet data loader with camera pose computation
- `training/run_sharp_finetune.sh` — A100 run script
- Training: predict Gaussians from one view → render from target camera → compare to GT view
- Frozen DINOv2 encoder, train decoder + heads only

**Data:** `demo_back_view_pairs.tar` (3.4 GB, 6,210 pairs, 3 views per pair: front, three_quarter, back)

**ONNX contract (old U-Net):** Input 9ch (was 8ch). Strata Rust code updated. Will need new contract for SHARP integration (Gaussian splats → render on device).

## Quality Filter

`scripts/filter_seg_quality.py` — rejects pseudo-labeled examples with implausible masks:
- `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`
- Also checks for `missing_head` and `missing_torso`
- GT datasets: <1% rejected. Pseudo-labeled: varies (humanrig_posed had 66% rejection with run 12 model).

## Dr. Li's Label Schema Conversion

Dr. Li's SAM Body Parsing model and earlier See-Through model both use a 19-class **clothing-oriented** schema (topwear, legwear, handwear, etc.). Converted to Strata's 22-class **skeleton** schema via `scripts/convert_li_labels.py` (694 images) and `scripts/convert_sam_labels.py` (2,467+ images).

**Critical finding (April 1):** Clothing-based labels fundamentally disagree with anatomy-based labels. SAM doesn't distinguish forearm from upper_arm (both are "topwear"), can't detect bare hands/legs, and the L/R split by image center fails on non-frontal views. Even with anatomy-aware conversion (connected components, torso core detection, legwear splitting), the labels still hurt training because the val set has GT bone-based labels. **Seg improvement requires anatomy-native labels** — either hand-annotated or from Dr. Li's upcoming training code (April 12).
