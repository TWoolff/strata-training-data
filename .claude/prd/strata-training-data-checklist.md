# Strata Training Data — Checklist

**Last updated:** March 7, 2026 (v18)

---

## First A100 Training Results (March 5, 2026)

| Model | Metric | Score | Training Data | Notes |
|-------|--------|-------|--------------|-------|
| **1. Segmentation** (multi-head: seg + depth + normals) | mIoU | 0.5453 | ~6K examples (Mixamo + Live2D with 22-class masks) | Best at epoch 94/100. Depth + normals heads planned for run 3 (distilled from Marigold). |
| **2. Joint Refinement** | Mean offset error | 0.001287 | ~142K examples with joints | Excellent |
| **3. Weight Prediction** (merged, optional encoder features) | MAE | 0.0840 (geom) / 0.0894 (w/ encoder) | 54 examples (Mixamo only!) | Now have ~26.5K from HumanRig + UniRig. Models 3+4 merged into single model. |
| **4. Inpainting** | L1 | BROKEN (0.0000) | 338K occlusion pairs generated, but loader found only 3 | Data path mismatch — needs fix |

**Key findings:**
- Weight model trained on only 54 Mixamo examples. HumanRig (11,434) + UniRig (14,950) weight.json now extracted — ~490x more data for next run. Models 3+4 merged into single model with optional encoder features (branch dropout during training).
- Inpainting pipeline broken: 338K occlusion pairs generated but dataset loader only found 3 (data path mismatch). All metrics 0.0000. Needs fix.
- Checkpoints + ONNX + logs uploaded to bucket and downloaded locally. A100 instance destroyed March 7.

---

## What's in the Hetzner Bucket

> **Verified March 6, 2026.**

| Prefix | Files | Size |
|--------|------:|-----:|
| `animation/` (incl. 100style) | 18,628 | 66.7 GiB |
| `anime_instance_seg/` | ~135K | ~15 GiB |
| `anime_seg/` | ~65K | ~3.5 GiB |
| `animerun/` | 11,276 | 663 MiB |
| `animerun_correspondence/` | 19,493 | 930 MiB |
| `animerun_flow/` | 16,704 | 11.6 GiB |
| `animerun_linearea/` | 4,236 | 119 MiB |
| `animerun_segment/` | 11,276 | 628 MiB |
| `conr/` | ~7,269 | ~580 MiB |
| `fbanimehq/` | 304,889 | 11.4 GiB |
| `humanrig/` | 148,643+ | 5.6+ GiB |
| `ingest/vroid_lite/` | 9,302 | 771 MiB |
| `instaorder/` | ~11,868 | ~1.5 GiB |
| `live2d/` | 3,587 | 212 MiB |
| `nova_human/` | ~40K | ~2.5 GiB |
| `segmentation/` | 12,216 | 599 MiB |
| `unirig/` | 66,030+ | 42.6+ GiB |
| **Total** | **~870K+** | **~166+ GiB** |

> `anime_instance_seg/` partially uploaded (~45K of 98K). `humanrig/` includes 11,434 weight.json files. UniRig weight.json conversion in progress (14,950).

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
| Mixamo renders | ~5,250 | Yes | Yes | 3D rendered |
| Live2D composites | 844 | Yes | Yes | 2D illustrated |
| NOVA-Human | ~40,000 | No (fg/bg) | RTMPose | 3D rendered (VRoid) |
| HumanRig | 34,302 | No | Ground truth | 3D rendered |
| anime-seg v1+v2 | ~14,586 | No (fg/bg) | RTMPose | 2D illustrated |
| anime_instance_seg | 98,428 | No (instance) | Partial | 2D illustrated |
| FBAnimeHQ | ~101,630 | No | RTMPose | 2D illustrated |
| **With 22-region masks** | **~6,094** | | | **Critical gap** |

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

### Draw Order

| Dataset | Examples | Notes |
|---------|--------:|-------|
| Mixamo renders | ~5,250 | Per-pixel Z-depth |
| Live2D composites | 844 | Fragment stacking |
| InstaOrder (val) | 3,956 | Pairwise depth orderings |
| **Total** | **~10,050** | Major gap — See-Through will add 9,102 |

---

## Critical Gaps & Next Actions

| Gap | Severity | Fix |
|-----|----------|-----|
| **22-region masks on illustrated images** | Critical | See-Through (late March 2026): 9,102 models with 19-region masks |
| **Draw order on illustrations** | Critical | See-Through + Layered Temporal PSD |
| **Weight prediction data** | Fixed | 54 -> ~26.5K examples (HumanRig + UniRig converters) |
| **Joints on anime_instance_seg** | Medium | Run RTMPose on remaining ~88K on A100 |
| **NOVA-Human depth enrichment** | Medium | Run remaining ~30K depth on A100 (OOM on Mac) |
| **Multi-angle Mixamo renders** | Medium | Re-render 105 chars with more styles + angles |

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

All implemented: dataset loaders, DeepLabV3+ multi-head model (seg + depth + normals), training metrics (mIoU), segmentation/joints/weights/inpainting training scripts, ONNX export + validation, evaluation/visualization. Configs for local, lean A100, full A100. Models 3+4 merged into single weight prediction model. 6 models total (was 7).

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

- [ ] Mixamo: Adobe terms for ML training
- [ ] UniRig Rig-XL: Objaverse-XL derived, check terms
- [ ] AnimeRun: CC-BY-NC 4.0 — does Strata's use qualify?
- [ ] HumanRig: CC-BY-NC-4.0 — non-commercial training use
- [ ] FBAnimeHQ: Derived from Danbooru — check terms
- [ ] NOVA-Human: Research use (VRoid Hub derived)
- [ ] Live2D models: Per-model license in manifest
- [ ] CMU mocap: Historically permissive, confirm
- [x] LinkTo-Anime: CC-BY-NC-4.0 — permanently excluded
