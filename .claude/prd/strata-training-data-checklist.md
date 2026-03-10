# Strata Training Data — Checklist

**Last updated:** March 10, 2026 (v20)

---

## A100 Training Results

### Run 1 (March 5-6, 2026) — COMPLETE

| Model | Metric | Score | Training Data | Notes |
|-------|--------|-------|--------------|-------|
| **1. Segmentation** (multi-head) | mIoU | **0.5453** | ~6K examples (Mixamo + Live2D with 22-class masks) | Best at epoch 94/100. |
| **2. Joint Refinement** | Mean offset error | **0.001287** | ~142K examples with joints | Excellent. Early stopped epoch 28. |
| **3. Weight Prediction** (merged) | MAE | **0.0840** (geom) / 0.0894 (w/ encoder) | 54 examples (Mixamo only!) | Models 3+4 merged into single model. |
| **4. Inpainting** | L1 | **BROKEN** (0.0000) | 338K pairs generated, loader found 3 | `rglob("*.png")` grabbed seg masks as source images. |

### Run 2 (March 7, 2026) — ABANDONED

Seg regressed to 0.38 mIoU at epoch 44 (vs run 1's 0.545). Killed early. Other models skipped. Enriched datasets re-tarred and uploaded to bucket. Instance destroyed.

**Fixes deployed during run 2:**
- Inpainting image discovery: `glob("*/image.png")` not `rglob("*.png")` (commit `011c3ba`)
- Pair generation capped at 15K source images (~45K pairs) to avoid filling disk
- rclone `region=fsn1` added to cloud_setup.sh
- All datasets tar-packed in bucket (~30min setup vs 5h loose files)

### Run 3 (March 9-10, 2026) — COMPLETE

| Model | Metric | Score | Training Data | Notes |
|-------|--------|-------|--------------|-------|
| **1. Segmentation** (multi-head) | mIoU | **0.3728** | ~30K examples (+ noisy Meshy CC0 auto-rig) | Regressed from 0.545. Noisy auto-rigged data hurt. |
| **2. Joint Refinement** | Mean offset error | **0.001206** | ~110K+ examples | Slight improvement over run 1 (0.001287). Early stopped epoch 36. |
| **3. Weight Prediction** (geometry) | MAE | **0.023137** | 12,027 examples (HumanRig via split_loader fix) | **3.6x better** than run 1 (0.084). Early stopped epoch 33. |
| **3b. Weight Prediction** (w/ encoder) | MAE | **0.021646** | 9,129 examples with encoder features | Better than geometry-only. Encoder features help with more data. |
| **4. Inpainting** | — | Not retrained | — | Scheduled for run 4. |

**Key fixes deployed in run 3:**
- `split_loader.py`: Added nested view + weight-only dataset discovery (UniRig `{id}/front/weights.json`)
- `weight_dataset.py`: Fixed example_id mismatch — used `child.name` instead of `{child}_{view}`
- `precompute_encoder_features.py`: float16 saving + MAX_VERTICES=2048 cap (200GB → 42GB)
- Segmentation model v2: depth + normals heads (Marigold-distilled), replacing draw_order
- ONNX outputs: segmentation, depth, normals, confidence, encoder_features (5 heads)

**Lesson learned:** Noisy auto-rigged Meshy CC0 data caused seg regression. Run 4 will quality-filter these examples.

### Run 4 (March 10, 2026) — IN PROGRESS

**Goal: Ship-ready models 1-4.** Script: `training/run_fourth.sh`

**Strategy:**
- Resume seg from run 1 checkpoint (0.545 mIoU), not run 3 (0.3728)
- Quality filter Meshy CC0 data (reject <4 regions, >70% single region, missing head/torso)
- Add ~221 Gemini diverse pseudo-labeled examples for domain diversity
- Label smoothing 0.05 to reduce impact of remaining noisy labels
- Fine-tune 50 epochs at 5e-5 LR
- Train inpainting (first real attempt with fixed data loader)
- NOT retraining: joints (0.001206 good enough), weights (0.023 MAE is 3.6x better)

**Data:** 30,021 train / 3,711 val (with quality filtering + meshy_cc0 + meshy_cc0_textured + meshy_cc0_unrigged)

**Expected:** mIoU 0.55-0.65 (recover + cleaner data + Gemini domain diversity)

---

## What's in the Hetzner Bucket

> **Verified March 7, 2026.** Core training datasets now stored as tar archives in `tars/` prefix. Loose files purged for tarred datasets. `cloud_setup.sh` downloads + extracts tars (~30min vs 5h for loose files).

| Prefix | Files | Size | Notes |
|--------|------:|-----:|-------|
| `tars/` | 7 | ~36 GiB | segmentation, live2d, humanrig, anime_seg, fbanimehq, unirig (curated_diverse removed — ArtStation) |
| `animation/` (incl. 100style) | 18,628 | 66.7 GiB | |
| `anime_instance_seg/` | ~135K | ~15 GiB | Partially uploaded (~45K of 98K) |
| `animerun/` | 11,276 | 663 MiB | |
| `animerun_correspondence/` | 19,493 | 930 MiB | |
| `animerun_flow/` | 16,704 | 11.6 GiB | |
| `animerun_linearea/` | 4,236 | 119 MiB | |
| `animerun_segment/` | 11,276 | 628 MiB | |
| `conr/` | ~7,269 | ~580 MiB | |
| `ingest/vroid_lite/` | 9,302 | 771 MiB | |
| `instaorder/` | ~11,868 | ~1.5 GiB | |
| `nova_human/` | ~40K | ~2.5 GiB | |
| `gemini_diverse/` | ~1,300 | ~55 MiB | 221 Gemini pseudo-labeled examples |
| `meshy_cc0/` | ~20K | ~3 GiB | Meshy CC0 rigged multi-view renders |
| `meshy_cc0_textured/` | ~20K | ~3 GiB | Meshy CC0 textured multi-view renders |
| `meshy_cc0_unrigged/` | ~20K | ~4 GiB | Meshy CC0 unrigged multi-view (image + depth + normals only) |
| `checkpoints/` | varies | varies | Run 1 + run 2 + run 3 checkpoints |
| **Total** | | **~176+ GiB** | |

> Core training datasets (segmentation, live2d, humanrig, anime_seg, fbanimehq, unirig) are tar-packed. curated_diverse removed (ArtStation — no AI training permission). Loose files purged from bucket after tar verified. Includes Marigold-enriched normals.png + depth.png from run 2.

---

## What's on External HD (TAMWoolff)

> **Storage policy:** All raw source data lives on the external HD permanently — never delete.

| Dataset | Size | Ingested | Notes |
|---------|-----:|:--------:|-------|
| `fbanimehq/` | 31 GB | Yes | 12 zip files |
| `anime_segmentation/` (v1) | 29 GB | Yes | 11,802 fg images |
| `anime_seg_v2/` | 12 GB | Yes | 13,000 fg images |
| `animerun/` | 21 GB | Yes | All 5 data types |
| `vroid_lite/` | 9.4 GB | Yes | 4,651 images |
| `anime_instance_seg/` | 98 GB | Yes | 98,428 examples |
| `humanrig/` | 234 GB | Yes | 11,434 meshes + images + weights |
| `unirig/` | 128 GB | Yes | 16,641 meshes, weight conversion running |
| `100style/` | 53 GB | Yes | 810 retargeted sequences |
| `nova_human/` | 391 GB | Yes | ~40K ortho views uploaded |
| `mixamo_line240/` | 97 GB | Ref only | CC-BY-NC-SA, vertex correspondence studied |
| `stdgen/` | 232 MB | Blocked | VRoid Hub 404'd |
| `conr/` | 8.5 GB | Yes | 2,423 examples |
| `fbx/` | 3.7 GB | Yes | 106 Mixamo chars |
| `poses/` | 2.1 GB | Yes | 2,022 Mixamo animation clips |
| `mocap/cmu/` | 5.7 GB | Yes | 2,548 BVH clips |
| `live2d/` | 15 GB | Yes | 280 .moc3 models |

---

## Training Data by Model

### Segmentation (22 regions)

| Dataset | Examples | 22-Region Masks | Joints | Style |
|---------|--------:|:---:|:---:|-------|
| ~~Mixamo renders~~ | ~~5,250~~ | ~~Yes~~ | ~~Yes~~ | ~~3D rendered~~ — **PROHIBITED (Adobe ToS)** |
| ~~Live2D composites~~ | ~~844~~ | ~~Yes~~ | ~~Yes~~ | ~~2D illustrated~~ — **PROHIBITED (Live2D ToS)** |
| Meshy CC0 (rigged) | ~4,700 | Yes (auto-rig) | No | 3D rendered (quality-filtered) |
| Meshy CC0 (textured) | ~4,700 | Yes (auto-rig) | No | 3D rendered (quality-filtered) |
| Meshy CC0 (unrigged) | ~20K | No (image+depth+normals only) | No | 3D rendered |
| Gemini diverse | 221 | Yes (pseudo-labeled) | No | 2D illustrated (AI-generated) |
| HumanRig | 34,302 | Yes (auto-rig) | Ground truth | 3D rendered |
| anime-seg v1+v2 | ~14,586 | No (fg/bg) | RTMPose | 2D illustrated |
| anime_instance_seg | 98,428 | No (instance) | Partial | 2D illustrated |
| FBAnimeHQ | ~101,630 | No | RTMPose | 2D illustrated |
| NOVA-Human | ~40,000 | No (fg/bg) | RTMPose | 3D rendered (VRoid) |
| **With 22-region masks** | **~44K** | | | **Improved from ~6K (Meshy CC0 + HumanRig + Gemini)** |

### Joint Prediction (19 bones)

| Dataset | Examples | Source |
|---------|--------:|-------|
| FBAnimeHQ | ~101,630 | RTMPose |
| NOVA-Human | ~40,000 | RTMPose |
| HumanRig | 34,302 | Ground truth |
| anime-seg v1+v2 | 14,579 | RTMPose |
| anime_instance_seg | 10,072 | RTMPose (partial) |
| Mixamo renders | ~5,250 | Ground truth |
| Live2D composites | 844 | Ground truth |
| **Total** | **~206,677** | |

### Weight Prediction MLP (20 bones)

| Dataset | Examples | Pipeline Format | Notes |
|---------|--------:|:---:|-------|
| HumanRig | 11,434 | Yes | `humanrig_weights_converter.py` |
| UniRig Rig-XL | 14,950 | Yes (pending upload) | `unirig_weights_converter.py` — 1,691 failed (non-humanoid) |
| Mixamo renders | 105 | Yes | First run used only these 54 |
| **Total** | **~26,489** | | ~490x improvement over first run |

### Depth + Surface Normals (Marigold-distilled)

The segmentation model's depth and normals heads are trained against Marigold LCM labels. Any dataset with `depth.png` + `normals.png` provides supervision.

| Dataset | Examples | depth.png | normals.png | Notes |
|---------|--------:|:---------:|:-----------:|-------|
| segmentation/ | 1,598 | Yes | Yes | Enriched in run 2 |
| live2d/ | 844 | Yes | Yes | Enriched in run 2 |
| ~~curated_diverse/~~ | ~~748~~ | | | **REMOVED — ArtStation, no AI training permission** |
| humanrig/ | ~11,434 | Partial | Partial | ~4,800 done in run 2, rest in run 3 |
| unirig/ | ~10,000 | Pending | Pending | Front views, enriched in run 3 |
| **Total with labels** | **~14K+** | | | Conditional loss skips missing labels |

Legacy `draw_order.png` still exists in some datasets but is no longer used for training.

---

## Critical Gaps & Next Actions

| Gap | Severity | Fix |
|-----|----------|-----|
| **Seg quality regression** | Critical | Run 4 in progress: quality filter + resume from run 1 checkpoint + label smoothing |
| **22-region masks on illustrated images** | High | Only Gemini (221) has pseudo-labeled 2D masks. See-Through (late March): 9,102 models with 19-region masks |
| **Depth + normals on all datasets** | Medium | Conditional loss handles missing. Meshy CC0 unrigged has depth+normals. |
| **Weight prediction data** | Fixed (run 3) | 54 → 12K examples, 0.084 → 0.023 MAE (3.6x improvement) |
| **Inpainting model** | In progress | Run 4 will train inpainting for first time with fixed data loader |
| **Joints on anime_instance_seg** | Medium | Run RTMPose on remaining ~88K on A100 |
| **Mixamo/Live2D prohibited** | Legal | Replaced by Meshy CC0 + Gemini. Need more illustrated 22-class data (See-Through). |

---

## Datasets — Active / Blocked

### See-Through (PP-14) — HIGHEST PRIORITY, pending release

9,102 annotated Live2D models with 19-class body part segmentation + draw order. Expected late March 2026. In contact with Dr. Li (issue #183). Would be the single biggest unlock for segmentation + draw order models.

### ChildlikeSHAPES (PP-15) — Blocked on paper acceptance

16,075 hand-drawn figures with 25-class semantic segmentation. Pending paper acceptance + release.

### Layered Temporal PSD (NEW-14) — Not started

20,000 anime PSD files with layer structure = free draw order ground truth. ~1.6 TB raw. Contact authors for access/license (issue #191).

### NOVA-Human (PP-1) — Done (partial depth)

~40K ortho views ingested + uploaded. RTMPose joints enriched. Depth enrichment partial (~9.6K of ~40K — Mac OOM). Remaining depth can run on A100.

### UniRig (PP-5) — Weight conversion complete, pending upload

66,030 files already in bucket. Weight conversion finished: 14,950 converted, 1,691 failed (non-humanoid meshes with <5 mapped bones). Upload with:
```
rclone copy ./output/unirig_weights/ hetzner:strata-training-data/unirig/ --transfers 32 --fast-list --size-only -P
```

### Datasets not pursued

| Dataset | Reason |
|---------|--------|
| StdGEN / PAniC-3D / VRoid Hub | VRoid Hub 404'd (March 2026) |
| LinkTo-Anime | CC-BY-NC license |
| MagicAnime | Restricted institutional access |
| AnimeDrawingsDataset | No license, only 2K images, RTMPose better |
| AMASS | Overlap with CMU BVH already ingested |
| RigAnything | Already have 28K weight examples |
| Manga109 | Requires access request, lineart-only |
| Bizarre Pose | Danbooru copyright |

---

## Datasets — Low Priority (Open Issues)

| Dataset | Issue | What | License | Status |
|---------|-------|------|---------|--------|
| CartoonSegmentation (LI-2) | #165 | 100K anime silhouettes | Public | Not started |
| RigNet | #166 | 2.7K rigged meshes | Research | Not started |
| Sakuga-42M annotations | #167 | Timing labels (on-ones/twos/threes) | CC-BY-NC-SA | Not started |
| ATD-12K | #168 | 12K animation triplets | Research | Not started |
| Anita Dataset | #170 | Sketch-color pairs | CC-BY-NC-SA | Not started |
| OCHuman | #172 | Occlusion-aware joints | Research | Not started |
| Bandai Namco Motion | #173 | Content x style motion labels | Research | Not started |
| AIST++ | #174 | Dance motion (10 genres) | Research | Not started |
| InstaOrder train split | — | 97K images (Flickr mixed-license) | CC-BY-SA | Deferred |

---

## Completed Datasets (in bucket, no further action)

| Dataset | Bucket Prefix | Files | Size |
|---------|--------------|------:|-----:|
| Mixamo renders | `segmentation/` | 12,216 | 599 MiB |
| Live2D composites | `live2d/` | 3,587 | 212 MiB |
| FBAnimeHQ | `fbanimehq/` | 304,889 | 11.4 GiB |
| anime-seg v1+v2 | `anime_seg/` | ~65K | ~3.5 GiB |
| anime_instance_seg | `anime_instance_seg/` | ~135K | ~15 GiB |
| AnimeRun (all types) | `animerun*/` | 63K+ | ~15 GiB |
| CMU + 100STYLE | `animation/` | 18,628 | 66.7 GiB |
| HumanRig | `humanrig/` | 148K+ | 5.6+ GiB |
| UniRig | `unirig/` | 66K+ | 42.6+ GiB |
| VRoid Lite | `ingest/vroid_lite/` | 9,302 | 771 MiB |
| CoNR | `conr/` | ~7,269 | ~580 MiB |
| InstaOrder (val) | `instaorder/` | ~11,868 | ~1.5 GiB |
| NOVA-Human | `nova_human/` | ~40K | ~2.5 GiB |

---

## Infrastructure

### Ingest Adapters (14 + 2 weight converters)

| Adapter | Status |
|---------|--------|
| `fbanimehq_adapter.py` | Working |
| `nova_human_adapter.py` | Working |
| `anime_seg_adapter.py` | Working |
| `animerun_contour_adapter.py` | Working |
| `animerun_flow_adapter.py` | Working |
| `animerun_segment_adapter.py` | Working |
| `animerun_correspondence_adapter.py` | Working |
| `animerun_linearea_adapter.py` | Working |
| `vroid_lite_adapter.py` | Working |
| `anime_instance_seg_adapter.py` | Working |
| `instaorder_adapter.py` | Working |
| `conr_adapter.py` | Working |
| `unirig_adapter.py` + `unirig_skeleton_mapper.py` | Working |
| `humanrig_adapter.py` + `humanrig_blender_renderer.py` | Working |
| `humanrig_weights_converter.py` | Working (11,434 extracted) |
| `unirig_weights_converter.py` | Working (14,950 converted, 1,691 failed) |

### Training Infrastructure

All implemented: dataset loaders, DeepLabV3+ multi-head model (seg + depth + normals + confidence + encoder_features), training metrics (mIoU), segmentation/joints/weights/inpainting training scripts, ONNX export + validation, evaluation/visualization. Configs for local, lean A100, full A100. Models 3+4 merged into single weight prediction model. 6 models total (was 7). Orchestration scripts: `run_second.sh`, `run_third.sh`. Tar-based bucket storage for fast A100 setup.

### Cloud Storage

Hetzner Object Storage: `s3://strata-training-data` (Falkenstein). ~870K+ files, ~166+ GiB. Use **rclone** (remote: `hetzner`), never `aws s3 sync`.

---

## Open Issues (23 remaining)

### P0 — Do now
- **#201** Guide: train 4 core ONNX models with existing bucket data (documentation)

### P1 — High priority
- **#162** Complete Mixamo render pipeline: additional styles and multi-angle passes
- **#164** Build See-Through adapter (blocked — pending late March release)
- **#179** License audit: confirm commercial training use for all datasets
- **#181** Run final validation: segmentation consistency across views
- **#183** Contact Dr. Li: See-Through taxonomy, release, collaboration

### P2 — Medium priority
- **#163** Extract humanoid subset from UniRig for weight validation
- **#165** Download CartoonSegmentation (100K silhouettes)
- **#166** Download RigNet (2.7K rigged meshes)
- **#167** Sakuga-42M timing annotations
- **#168** ATD-12K animation triplets
- **#170** Anita Dataset sketch-color pairs
- **#172** OCHuman occlusion-aware joints
- **#177** Expand Live2D collection via Booth.pm
- **#178** Generate contour line augmentation pairs
- **#185** Monitor ChildlikeSHAPES release (blocked)
- **#190** Improve Live2D fragment-region mapping
- **#191** Layered Temporal PSD dataset (blocked on access)

### P2 — Future models (5 & 6)
- **#206** Novel view synthesis: architecture + data (generates all unseen views, not just back)
- **#207** Novel view synthesis: training + ONNX
- **#210** Texture inpainting: architecture + data
- **#211** Texture inpainting: training + ONNX

### P3 — Low priority
- **#173** Bandai Namco Motion Dataset
- **#174** AIST++ dance motion

---

## Dr. Li Contact (See-Through)

Dr. Chengze Li, Saint Francis University HK. In contact re: See-Through dataset.

**Key asks:**
1. Release timeline for See-Through dataset + annotation engine
2. Exact 19-region taxonomy labels (verify overlap with Strata's 22 regions)
3. Will pretrained segmentation model weights be released?
4. Body Part Segmentation paper (LI-3): dataset/code available?
5. CartoonSegmentation (LI-2): license for commercial use?
6. Collaboration opportunity?

**Other relevant papers from her lab:**
- LI-2: CartoonSegmentation — 100K instance masks, publicly available on GitHub
- LI-3: Body Part Seg of Anime Characters (CAVW 2024) — ask for code/data
- LI-4: Manga109 CVPR 2025 segmentation — on HuggingFace
- LI-5: Shading/Reflectance decomposition — potential seg preprocessing

---

## Legal Checklist

Full audit: `docs/license-audit.md` (v2, March 7 2026)

**Safe (commercial training OK):**
- [x] CMU mocap: Custom permissive — attribute CMU Graphics Lab
- [x] 100STYLE: CC-BY 4.0 — attribute authors
- [x] VRoid Lite: CC0 — no restrictions
- [x] InstaOrder: CC-BY-SA 4.0 — SA unlikely to apply to model weights

**Likely safe (low-medium risk):**
- [x] anime-seg v1+v2: Apache 2.0 / CC0 — masks safe, source images from Danbooru
- [x] CoNR: MIT — some hand-drawn sheets may be copyrighted

**Ambiguous (high risk — needs legal counsel):**
- [ ] UniRig / Rig-XL: ODC-BY but Sketchfab ToS prohibits AI training on user content
- [ ] FBAnimeHQ: CC0 dataset label but source images are copyrighted Danbooru art

**PROHIBITED (must exclude from production training):**
- [x] Mixamo renders: Adobe Additional Terms explicitly ban AI/ML training — **CRITICAL: only source of 22-class seg masks**
- [x] Live2D models: Proprietary terms, no ML training permission
- [x] CartoonSegmentation: No license, Bandai Namco IP, research fair use only
- [x] AnimeRun: CC-BY-NC 4.0 — not used for core models
- [x] HumanRig: CC-BY-NC 4.0 — contact authors for commercial permission
- [x] NOVA-Human: VRoid Hub — Pixiv prohibits AI training data collection
- [x] LinkTo-Anime: CC-BY-NC 4.0 — permanently excluded

**Action items:**
1. [ ] URGENT: Render VRoid Lite CC0 characters to replace Mixamo as 22-class seg source
2. [ ] Contact HumanRig authors for commercial training permission
3. [ ] Contact Live2D Inc. for ML training permission
4. [ ] Legal counsel on FBAnimeHQ (discriminative model on Danbooru data)
5. [ ] Legal counsel on UniRig/Objaverse-XL (Sketchfab ToS vs ODC-BY)
6. [ ] Create production training configs excluding all prohibited datasets
7. [ ] Create `ATTRIBUTIONS.md` for CC-BY/CC-BY-SA datasets
