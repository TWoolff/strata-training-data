# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 6 ONNX AI models. Runs independently of the Strata codebase.

## Strata's 6 ONNX Models

Strata (Tauri/Rust/React desktop app at `../strata/`) uses 6 ONNX models defined in `src-tauri/src/ai/runtime.rs`:

### Current Models (being replaced — see New Model Strategy below)

| # | Model | Architecture | Status |
|---|-------|-------------|--------|
| 1 | **Segmentation** | DeepLabV3+ MobileNetV3 | 0.6485 mIoU. Replacing with SAM 2.1 fine-tune. |
| 2 | **Joint Refinement** | MobileNetV3 + regression | 0.00121 offset. Replacing with ViTPose++. |
| 3 | **Weight Prediction** | Per-vertex MLP | 0.0215 MAE. Study Puppeteer architecture. |
| 4 | **Inpainting** | U-Net | 0.0028 val/l1. Low priority. |
| 5 | **Texture Inpainting** | None | May not need if 3D reconstruction works. |
| 6 | **View Synthesis / 3D** | U-Net (blurry) | Replacing with TRELLIS.2 (MIT). |

All current models bundled in `../strata/src-tauri/models/` (~55MB total), loaded via `ort` ONNX runtime.

### Model Strategy (April 3, 2026) — Combined Approach

Strata's unique value: combining 3D reconstruction (Meta SAM 3D) with illustrated character understanding (Dr. Li's See-Through) for animation-ready rigging. Nobody else does this.

| # | Task | Model | License | Status |
|---|------|-------|---------|--------|
| 1 | **Segmentation** | **Dr. Li's SAM-HQ** (encoder) + **our 22-class decoder** | Apache 2.0 | Training code releases April 12. Interim: SAM 3 fine-tune running now. |
| 2 | **3D Mesh** | **SAM 3D Objects** | SAM License (commercial OK) | **Validated** — bear chef produces good 3D mesh from single image. |
| 3 | **Skeleton/Rigging** | **SAM 3D Body** (70 keypoints, MHR rig) | SAM License (commercial OK) | **Tested** — great on humanoid, poor on chibi/non-human. |
| 4 | **Anatomy Labels** | **SAM 3D Body → 2D projection** | Our pipeline | Built but labels hurt training (body mesh ≠ clothing silhouette). |
| 5 | **Texture** | Multi-view projection + inpainting | Our code | Front + side views projected onto SAM 3D mesh, inpaint gaps. |
| 6 | **Backup 3D** | **TRELLIS.2** (Microsoft, 4B params) | **MIT** | Needs HF access. MIT license = can ship. |

**Why our 22-class anatomy schema (not Dr. Li's 19-class):**
Dr. Li's 19 classes are clothing-based (topwear, legwear, handwear). They CANNOT drive animation because:
- No left/right distinction (no shoulder_l vs shoulder_r)
- No joint separation (topwear = chest + shoulders + arms all in one)
- No forearm vs upper_arm, no upper_leg vs lower_leg
- No hips region

Our 22-class schema maps directly to skeleton bones — each class = one bone's influence zone. Required for rigging.

**Why Dr. Li's encoder is still valuable (April 12):**
- SAM-HQ with 19 independent decoders — each body part gets a specialist
- Pretrained on 9K illustrated characters (anime/Live2D) — understands illustrated character boundaries
- SIGGRAPH 2026 quality (8x H200, 129 hrs training)
- **Plan:** Take her pretrained encoder, replace 19 clothing class heads with our 22 anatomy class heads, fine-tune on our data

**What Dr. Li's model doesn't do (we solve):**
- Her data is anime-only — we have diverse styles (3D-rendered, Gemini, Flux, hand-drawn)
- She works in 2.5D (Live2D layers) — we do full 3D with SAM 3D Objects
- She segments by clothing — we segment by anatomy (SAM 3D Body generates GT labels)

**New Strata Pipeline (vision):**
1. User imports front view illustration
2. **SAM 3D Objects** → 3D mesh **geometry only** (vertices, faces, normals) — discard SAM's blurry texture
3. **Texture projection** — ray-cast the artist's original illustration onto front-facing mesh faces → pixel-perfect texture where visible (standard CG, no AI)
4. User optionally adds side/back views → project those onto side/back-facing faces → more pixel-perfect coverage
5. **Inpaint** remaining texture gaps (~10% of surface, mostly back/occluded areas) → our inpainting model fills using surrounding colors
6. **Result:** 90%+ of texture is the artist's actual drawing. Style-preserving. No hallucinated details.
7. **Segmentation** (Dr. Li encoder + our 22-class decoder) → anatomy regions on the 2D views
8. **SAM 3D Body** adds skeleton for humanoid / our joint model for non-human
9. Weight prediction on the real 3D mesh
10. Export rigged, textured, animatable 3D character

**Why this texture approach is groundbreaking:**
- SAM 3D Objects generates good geometry but blurry/hallucinated textures (guesses what back looks like)
- Our approach: use SAM 3D for geometry only, then project the artist's actual illustration onto it
- The artist stays in control of the look — their drawing IS the texture
- Only gaps (back, occluded) get AI inpainting, using surrounding color context
- Character looks exactly like the artist's drawing from the front, plausibly consistent from other angles

**Texture inpainting approach (TEXTure trimap method):**

Use a **trimap** state per UV texel: `generated` (projected from artist view), `partial` (seam edge), `empty` (unseen).

1. Project front view → mark visible texels as "generated" (pixel-perfect, never overwritten)
2. Project side/back views if available → more "generated" texels
3. **Depth-conditioned inpainting** fills "empty" regions — AI follows 3D surface shape (normals/depth)
4. **Blend at "partial" seams** — smooth transition between projected and inpainted
5. Artist's pixels ALWAYS win — AI only fills what the drawing doesn't cover

**Phase 1 (ship first):** Simple palette fill + edge extension for gaps. Works for 80% of characters.
**Phase 2 (post-launch):** Depth-conditioned diffusion inpainting fine-tuned on illustrated character textures. Generate training data from 3D-rendered characters (project front only → mask rest → train to reconstruct).

**Reference implementations (open source):**
- TEXTure (MIT) — trimap approach, `github.com/TEXTurePaper/TEXTurePaper`
- Text2Tex (Apache 2.0) — depth-aware progressive inpainting, `github.com/daveredrum/Text2Tex`
- Paint3D — UV inpainting with surface position maps
- Meta 3D TextureGen — normal+position conditioned, best quality
- SeqTex (SIGGRAPH Asia 2025) — video diffusion for texture consistency

**Also available:**
- **SAM 3** (Meta, SAM License) — text-prompted seg, training now as interim solution
- **ViTPose++** (Apache 2.0) — joint estimation for illustrated characters
- **Puppeteer** (NeurIPS 2025) — attention-based skinning weights
- **TRELLIS.2** (MIT) — backup 3D reconstruction with PBR materials
- **Sapiens** (CC-BY-NC) — pseudo-labeling teacher only

## Model Targets & Current Scores

Goal: uploaded 2D character illustrations look natural from all generated angles when rigged and posed.

| # | Task | Current | New Model | Status |
|---|------|---------|-----------|--------|
| 1 | **Segmentation** | 0.6485 mIoU (DeepLabV3+) | **SAM 3** fine-tune | Training overnight (loss 787→399) |
| 2 | **3D Mesh** | None (U-Net was blurry) | **SAM 3D Objects** | Validated on bear chef — works! |
| 3 | **Skeleton** | 0.00121 offset (MobileNetV3) | **SAM 3D Body** / ViTPose++ | Tested — good on humanoid |
| 4 | **Texture** | N/A | Multi-view projection + inpaint | New pipeline concept |
| 5 | **Weights** | 0.0215 MAE (MLP) | Puppeteer / current | Study Puppeteer architecture |

**Priority order (April 5, 2026):**
1. **Wait for Dr. Li's training code (April 12)** — THE endgame for seg. Take her SAM-HQ encoder, replace with our 22-class anatomy heads, fine-tune on our data.
2. **SAM 3D Objects for 3D mesh** — validated on bear chef. Need to fix kaolin dep on A100 and test multi-view input for better depth.
3. **Multi-view texture pipeline** — project front+side onto SAM 3D mesh, inpaint gaps.
4. **TRELLIS.2** — MIT licensed backup 3D. Needs DINOv3 + RMBG HF access.
5. **SAM 3D Body anatomy labels** — pipeline built but labels don't help training (see learnings below).

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
- **SAM 3D Body labels don't help seg training**: SAM 3D Body reconstructs the nude body mesh underneath, not the clothing silhouette. When projected back to 2D, the mesh doesn't cover clothing pixels (jacket sleeves, pants, boots). mIoU dropped from 0.5952 → 0.5369 when these labels were added. Same fundamental problem as SAM Body Parsing labels (April 1). Anatomy labels from 3D body models don't match 2D pixel boundaries.
- **SAM 3 seg fine-tune results**: 3 epochs on 24K GT data reached mIoU 0.5952. Adding 3,394 SAM 3D Body illustrated labels in epoch 4 hurt to 0.5369. Best checkpoint: epoch 3 (GT-only), uploaded as `best_epoch3_miou0595.pt`.

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

### SAM 3 Seg Fine-tune Results (April 3-5)
**Best model: epoch 3 checkpoint, mIoU 0.5952** (GT-only data, 24K images). Uploaded as `best_epoch3_miou0595.pt`.

| Epoch | Data | Loss | mIoU | Notes |
|-------|------|------|------|-------|
| 0→1 | GT only (24K) | 381 | 0.5633 | First epoch |
| 1→2 | GT only | 369 | 0.5868 | +0.0235 |
| 2→3 | GT only | 365 | 0.5952 | +0.0084, diminishing returns |
| 3→4 | GT + SAM 3D Body (28K) | 380 | 0.5369 | **Dropped** — body mesh labels don't match clothing silhouette |

**Key settings:** 840M params, batch 1, lr_scale 0.05, 1.6s/step, ~12 hrs/epoch on A100. 100 GB storage needed.

**Run commands:**
```bash
git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
./training/cloud_setup.sh lean

# Install SAM 3
cd /workspace && git clone --depth 1 https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e ".[train]"
pip install huggingface_hub submitit hydra-submitit-launcher hydra-colorlog fvcore

# Login to HuggingFace (SAM 3 is gated)
python3 -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"

# Patch fused.py for training mode
cat > sam3/perflib/fused.py << 'PYEOF'
import torch
def addmm_act(act_cls, linear, x):
    if torch.is_grad_enabled():
        out = torch.nn.functional.linear(x, linear.weight, linear.bias)
        return act_cls()(out)
    try:
        from sam3.perflib._C import addmm_gelu, addmm_relu
        if act_cls == torch.nn.GELU:
            return addmm_gelu(x, linear.weight, linear.bias)
        elif act_cls == torch.nn.ReLU:
            return addmm_relu(x, linear.weight, linear.bias)
    except ImportError:
        pass
    out = torch.nn.functional.linear(x, linear.weight, linear.bias)
    return act_cls()(out)
PYEOF

# Download GT data and convert to COCO
cd /workspace/strata-training-data
pip install scipy pycocotools
# (download tars + convert — see run_model_tests.sh Part 2 for full commands)
# Then split train/val:
python3 -c "..." # (split script from April 2)

# Fix config save frequency
sed -i 's/save_freq: 5/save_freq: 1/' /workspace/sam3/sam3/train/configs/sam3_seg_finetune.yaml

# Train (let run for 2-3 epochs = ~27 hrs)
cd /workspace/sam3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 sam3/train/train.py \
    -c configs/sam3_seg_finetune.yaml \
    --use-cluster 0 --num-gpus 1
```

**Config:** `training/configs/sam3_seg_finetune.yaml`
**Key settings:** 840M params, 22K train images, 21 anatomy classes, batch 1, lr_scale 0.05, segmentation enabled
**Patch needed:** `sam3/perflib/fused.py` — replace fused ops with standard ops when grad enabled

### Completed April 1-2
- [x] **SAM labels tested — don't help seg** (clothing≠anatomy, peaked at 0.56-0.59)
- [x] **View synthesis U-Net confirmed too weak** (blurry from L1 loss)
- [x] SHARP tested on Mac MPS — research-only license, abandoned
- [x] **Strata post-processing improvements committed** (6 changes in segmentation.rs, joints.rs, weights.rs)
- [x] **Comprehensive model research** — identified SAM 3, SAM 3D, TRELLIS.2, ViTPose++, Puppeteer
- [x] **SAM 3D Body tested on A100** — great on humanoid, poor on chibi/non-human
- [x] **SAM 3D Objects validated** (online demo) — bear chef produces good 3D mesh from single image
- [x] **SAM 3 seg fine-tune started** — 22K train, 2.5K val, training overnight
- [x] GT data → COCO format converter: `scripts/convert_gt_to_coco.py` (24,779 images, 1M+ annotations)
- [x] TRELLIS.2 installed (needs DINOv3 + RMBG HF access)

### Next Steps (April 3+)
1. **Evaluate SAM 3 seg checkpoint** — download epoch 1, test on illustrated characters
2. **SAM 3D Objects on A100** — fix kaolin dep, test multi-view (front+side) for better depth
3. **Multi-view texture pipeline** — project front+side textures onto SAM 3D mesh → inpaint gaps
4. **TRELLIS.2 inference test** — once DINOv3 + RMBG access approved
5. **Dr. Li's training code (April 12)** — could further improve SAM 3 seg
6. **Integrate SAM 3D into Strata** — replace view synthesis with real 3D mesh pipeline

### New Pipeline Vision
```
User imports front view illustration
    → SAM 3D Objects generates 3D mesh (~3 sec)
    → User optionally adds side/back views → project textures onto mesh
    → Inpaint remaining texture gaps
    → SAM 3 segments body parts (text-prompted anatomy)
    → SAM 3D Body adds skeleton (humanoid) or our joint model (non-human)
    → Weight prediction on the real 3D mesh
    → Export rigged, textured 3D character
```

### Model Repos (cloned)
| Model | Repo | License | Status |
|-------|------|---------|--------|
| SAM 3 | `github.com/facebookresearch/sam3` | SAM License (commercial OK) | **Training now** |
| SAM 3D Objects | `github.com/facebookresearch/sam-3d-objects` | SAM License | Validated. Kaolin dep issue on A100. |
| SAM 3D Body | `github.com/facebookresearch/sam-3d-body` | SAM License | **Tested** — works on humanoid |
| TRELLIS.2 | `github.com/microsoft/TRELLIS.2` | MIT | Installed. Needs DINOv3 + RMBG HF access. |
| ViTPose++ | `github.com/ViTAE-Transformer/ViTPose` | Apache 2.0 | For joints (future) |
| Puppeteer | `github.com/Seed3D/Puppeteer` | TBD | For weights (study architecture) |

### Later — Post Launch
- Interactive view correction paint tool in Strata
- Blueprint marketplace
- InnoFounder application (August 2026 start, 430K DKK grant)
- Approach PreSeed Ventures / Accelerace / byFounders for angel investment

## Model 6: View Synthesis → 3D Reconstruction

### U-Net View Synthesis (deprecated)

U-Net approach (29M params, L1+perceptual loss) produces inherently blurry novel views. L1 loss learns pixel averages. GAN loss failed at 3K pairs. **Replaced by TRELLIS.2 3D reconstruction.**

| Run | val/l1 | Notes |
|-----|--------|-------|
| run 2 (general) | 0.2047 | Best general model. Works OK for humanoid chars, fails on bear chef. |
| bear chef lr=1e-4 | 0.0548 | Sharp on training pairs, tile artifacts on unseen angles. Destroyed general ability. |
| bear chef lr=1e-5 | 0.0853 | Smoother but too blurry. Barely moved from general model. |
| mixed (20x bear chef) | 0.1936 | Full dataset + bear chef at high weight. Barely better than general. |

**Conclusion:** U-Net + L1 loss cannot produce sharp novel views. Need fundamentally different approach.

### SHARP (Apple) — Research Only, Deprecated
- Tested April 1-2. Runs on Mac MPS (~7s). Research-only license — **cannot ship in Strata**.
- Interprets illustrated characters as flat cards. Fine-tuning attempted but ~1 min/step at 1536 resolution — too slow.
- Fine-tuning pipeline exists: `training/train_sharp.py` + `training/data/sharp_dataset.py` + `training/run_sharp_finetune.sh`
- **Replaced by TRELLIS.2 (MIT license).**

### TRELLIS.2 (Microsoft) — New 3D Reconstruction Model
**Single image → 3D mesh + PBR materials (base color, roughness, metallic, opacity). MIT license.**
- **Repo:** `github.com/microsoft/TRELLIS.2`
- **License:** MIT (code + weights). Note: nvdiffrast dependency has separate NVIDIA license.
- **Architecture:** 4B params. Generates 512³ resolution 3D in ~3 seconds on A100.
- **Fine-tuning:** Full training codebase provided. Can fine-tune on turnaround sheet data.
- **Key advantage over SHARP:** MIT license (can ship), higher quality, PBR output, full training code.
- **Integration plan:** Fine-tune on turnaround sheets → export → run in Strata for character 3D reconstruction at import time.

**Data:** Same turnaround sheet data — `demo_back_view_pairs.tar` (3.4 GB, 6,210 pairs, 3 views per pair).

**ONNX contract (old U-Net):** Input 9ch. Strata Rust code exists. Will need new integration for TRELLIS.2 mesh output.

## Quality Filter

`scripts/filter_seg_quality.py` — rejects pseudo-labeled examples with implausible masks:
- `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`
- Also checks for `missing_head` and `missing_torso`
- GT datasets: <1% rejected. Pseudo-labeled: varies (humanrig_posed had 66% rejection with run 12 model).

## Dr. Li's Label Schema Conversion

Dr. Li's SAM Body Parsing model and earlier See-Through model both use a 19-class **clothing-oriented** schema (topwear, legwear, handwear, etc.). Converted to Strata's 22-class **skeleton** schema via `scripts/convert_li_labels.py` (694 images) and `scripts/convert_sam_labels.py` (2,467+ images).

**Critical finding (April 1):** Clothing-based labels fundamentally disagree with anatomy-based labels. SAM doesn't distinguish forearm from upper_arm (both are "topwear"), can't detect bare hands/legs, and the L/R split by image center fails on non-frontal views. Even with anatomy-aware conversion (connected components, torso core detection, legwear splitting), the labels still hurt training because the val set has GT bone-based labels. **Seg improvement requires anatomy-native labels** — either hand-annotated or from Dr. Li's upcoming training code (April 12).

**Key insight (April 3):** Dr. Li's 19-class schema is clothing-oriented (designed for Live2D layer decomposition), NOT anatomy-oriented. It cannot drive animation — no L/R distinction, no forearm/upper_arm split, no hips. Our 22-class anatomy schema is correct for rigging. But her **encoder** (SAM-HQ pretrained on 9K illustrated chars) is extremely valuable — we'll take her encoder and replace the 19 clothing class heads with our 22 anatomy class heads.

**New approach to anatomy labels:** SAM 3D Body → 3D mesh → project back to 2D → perfect anatomy segmentation. Built in `scripts/sam3d_body_to_seg.py` and `scripts/batch_sam3d_body_labels.py`. This generates clothing-independent anatomy labels because it reconstructs the actual 3D body underneath.

**Dr. Li's paper details (arxiv 2602.03749):**
- 9,102 Live2D models (7,404 train), 19 body part classes, 1024×1024 resolution
- 8x NVIDIA H200 for 129 hours (Stage 2 diffusion training)
- SAM-HQ with independent decoders per body part
- Diffusion-based Body Part Consistency Module
- Training code releases no later than April 12, 2026

## Strata Post-Processing Improvements (April 2, 2026)

6 changes committed to Strata (`../strata/`) that improve effective model quality without retraining:

1. **Alpha-aware preprocessing** (`segmentation.rs`, `joints.rs`) — composite RGBA onto black before normalization. Matches Blender training data.
2. **Confidence-gated seg cleanup** (`segmentation.rs`) — replace low-confidence pixels (<0.4) with 5×5 neighborhood majority. Remove connected components <16 pixels.
3. **Bilinear logit upscaling** (`segmentation.rs`) — interpolate raw class logits then argmax, instead of nearest-neighbor on labels. Sub-pixel smooth region boundaries.
4. **Laplacian weight smoothing** (`weights.rs`) — 2 iterations of confidence-adaptive smoothing on mesh adjacency graph. Reduces discontinuities.
5. **Adaptive joint offset cap** (`joints.rs`) — 15% of character bounding extent, clamped [0.02, 0.10]. Replaces hardcoded 5%.
6. **Confidence-weighted MLP/heat blending** (`weights.rs`) — low-confidence MLP predictions blend toward heat diffusion fallback.
