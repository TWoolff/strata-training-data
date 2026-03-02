# Strata Training Data — Complete Gathering Checklist

**Date:** February 27, 2026 (v2 — updated with pre-processed dataset research)
**Last updated:** March 2, 2026 (v11 — new datasets researched: HumanRig, Anymate, MagicAnime, AnimeDrawingsDataset, Body Part Seg Anime (Ou 2024), ChildlikeSHAPES, See-through)
**Sources:** strata-training-data-research-prd.md (v1.1), strata-3d-mesh-research-prd.md, web research on available datasets

---

## Progress Summary

### What's Done

| Category | Status | Details |
|----------|--------|---------|
| **Blender 3D pipeline** | Working | 105 Mixamo chars downloaded; re-rendering with 22-class IDs + mixamorig5: bone fix |
| **Hetzner Object Storage** | Set up | S3-compatible bucket at `s3://strata-training-data` (Falkenstein, €4.99/mo) |
| **Ingest framework** | Complete | `run_ingest.py` with 9 registered adapters, CLI with `--enrich` pose estimation |
| **FBAnimeHQ** | Ingested + uploaded | 304,889 files (11.5 GB) in bucket — all shards 00-11 processed |
| **anime-segmentation v1** | Ingested + uploaded | 35,406 files (1.8 GB) in bucket — 11,802 images |
| **anime-segmentation v2** | Ingested + uploaded | ~39,000 files in bucket — 13,000 images from fg-01/02/03 |
| **AnimeRun contour pairs** | Ingested + uploaded | 11,276 files (689 MB) in bucket — 2,819 frames |
| **AnimeRun LineArea** | Ingested + uploaded | 4,236 files (119 MB) in bucket — 1,059 frames |
| **Blender segmentation** | Uploaded | 12,216 files (619 MB) in bucket — 105 chars × 50 poses, 22-class IDs, mixamorig5: fix |
| **CMU animation** | Uploaded | 17,823 files (58 GB) in bucket — 2,548 clips × 7 degradations |
| **22-class region ID fix** | Complete | All pipeline code, config, tests updated to match Strata's skeleton.ts (1,389 tests pass) |
| **mixamorig5: bone fix** | Complete | 12 PARTIAL characters now fully mappable — `MIXAMO_BONE_MAP` + `COMMON_PREFIXES` updated |
| **Live2D models** | Rendered + uploaded | 4,764 files (327 MB) in bucket — 1,112 images from 279 models |
| **AnimeRun flow** | Ingested + uploaded | 16,704 files in bucket — 2,789 flow pairs from 30 scenes |
| **AnimeRun segment** | Ingested + uploaded | 11,276 files in bucket — 2,819 segmentation frames from 30 scenes |
| **AnimeRun correspondence** | Ingested + uploaded | 19,493 files (975 MB) in bucket — 2,789 pairs from 30 scenes |
| **AnimeRun linearea adapter** | Implemented | `animerun_linearea_adapter.py` + 32 tests passing |
| **Training infrastructure** | Implemented | configs, data loaders, utils, model architectures (issues #122-124) |
| **VRoid Lite** | Ingested + uploaded | 9,302 files (788 MB) in bucket — 4,651 images from 16 CC0 VRoid chars |
| **Live2D GitHub scraper** | Implemented | `run_live2d_scrape.py` — searches GitHub for .moc3 repos, sparse checkout, CSV manifest |
| **.moc3 parser + atlas extractor** | Implemented | `pipeline/moc3_parser.py` + atlas fragment extraction in `live2d_renderer.py` (issue #146) |
| **Label Studio integration** | Ready | import/export pipeline + config XML |
| **CMU action labels** | Tracked | `animation/labels/cmu_action_labels.csv` (80 clips labeled) |
| **Test suite** | 39 test files | Covering all pipeline modules, adapters, and utilities |
| **Documentation** | Complete | data-sources, preprocessed-datasets, labeling-guide, annotation-guide, taxonomy-comparison |

### What's in the Hetzner Bucket

| Prefix | Files | Size |
|--------|------:|-----:|
| `animation/` | 17,823 | 58 GB |
| `anime_seg/` | 50,406 | 1.9 GB |
| `animerun/` | 11,276 | 689 MB |
| `animerun_correspondence/` | 19,493 | 975 MB |
| `animerun_flow/` | 16,704 | 12 GB |
| `animerun_linearea/` | 4,236 | 119 MB |
| `animerun_segment/` | 11,276 | 651 MB |
| `fbanimehq/` | 304,889 | 11.5 GB |
| `ingest/vroid_lite/` | 9,302 | 788 MB |
| `live2d/` | 4,764 | 327 MB |
| `segmentation/` | 12,216 | 619 MB |
| **Total** | **~462,385** | **~87.5 GB** |

### What's Downloaded Locally (data/preprocessed/)

| Dataset | Size | Downloaded | Status |
|---------|------|-----------|--------|
| anime_segmentation (v1) | 17 GB | Deleted | Ingested + uploaded, local copy deleted (batch 4) |
| anime_seg_v2 | 12 GB | Deleted | Ingested + uploaded, zips deleted |
| animerun | 21 GB (zip) | Deleted | All 5 data types ingested + uploaded, zip deleted |
| fbanimehq | 17 GB | Yes | All shards ingested, leftovers cleaned |
| cmu_degraded | 58 GB | Deleted | Uploaded to bucket, local copy deleted (`--delete-local`) |
| vroid_lite | 7.3 GB | Deleted | Ingested + uploaded, local copy deleted |
| humanrig | 36 GB (zip) | Yes | Downloaded, extracting — `humanrig.zip` on external HD |
| nova_human | — | Structure only |
| linkto_anime | — | Structure only |
| stdgen | — | Structure only |
| charactergen | — | Structure only |
| unirig | — | Structure only |
| anime_instance_seg | — | Structure only |

---

## What's Next (Immediate)

### Completed Since Last Update

- [x] Fix 22-class region ID alignment — all pipeline code, config, bone maps, and 14 test files updated to match Strata's skeleton.ts RegionId enum (1,389 tests pass, 0 fail)
- [x] Upload CMU animation data — 17,823 files (58 GB) uploaded to `animation/` bucket prefix
- [x] Re-render segmentation with 22-class IDs + mixamorig5: bone fix — complete (105 chars × 50 poses, 12,216 files, 619 MB uploaded)
- [x] Live2D GitHub scraper ran — 279 .moc3 models downloaded (saturated GitHub source)
- [x] Process 279 Live2D models — 1,112 images rendered, 4,764 files (327 MB) uploaded to `live2d/` bucket prefix
- [x] Fix mixamorig5: bone prefix — 12 previously PARTIAL characters now fully mappable
- [x] Download all 105 available Mixamo characters (108 total on site, 105 downloaded)
- [x] UniRig Rig-XL downloaded + extracted — 16,641 meshes (66 GB) in `data/preprocessed/unirig/rigxl/`
- [x] Build .moc3 parser + atlas extractor (issue #146)
- [x] Build vroid_lite adapter + ingest 4,651 images + upload to bucket (9,302 files, 788 MB)
- [x] Delete vroid_lite local data (7.3 GB source + 788 MB output reclaimed)
- [x] Build Live2D GitHub scraper (`run_live2d_scrape.py`) — issue #141, PR #142 merged
- [x] Run overnight batch 4 — AnimeRun flow (2,789 pairs) + segment (2,819 frames) uploaded successfully
- [x] Fix correspondence adapter bug — SegMatching path missing `forward/` subdir + files are `.json` not `.png`
- [x] Run overnight batch 5 — AnimeRun correspondence (2,789 pairs, 19,493 files) uploaded successfully
- [x] Delete v1 anime_segmentation local copy (17 GB reclaimed, batch 4 phase 4)
- [x] Delete AnimeRun zip (all data types fully ingested)
- [x] Finish AnimeRun segment adapter (#136) — implemented + tests passing
- [x] Finish AnimeRun correspondence adapter (#137) — implemented + 43 tests passing
- [x] Build AnimeRun LineArea adapter — implemented + 32 tests passing
- [x] Run overnight batch 3 — FBAnimeHQ shards 08-11 succeeded
- [x] Ingest anime_seg_v2 (13,000 images from fg-01/02/03)
- [x] Clean FBAnimeHQ leftovers (~17 GB reclaimed)
- [x] Delete anime_seg_v2 fg zips after ingestion

### This Week

- [x] ~~Fix the 13 Mixamo characters that failed rendering~~ — 12 PARTIAL (2/5 poses) due to `mixamorig5:` prefix mismatch, pre-existing
- [x] Wait for segmentation re-render to complete + upload to bucket
- [x] Build vroid_lite adapter + ingest + upload (4,651 images, 9,302 files, 788 MB in bucket)

### Near-Term

- [x] Process 279 Live2D models through renderer pipeline — 1,112 images, 4,764 files (327 MB) uploaded to bucket
- [ ] Download NOVA-Human dataset (Alipan link — needs China-based help, see-through team contacted)
- [x] ~~Download StdGEN~~ — BLOCKED: VRoid Hub models all 404'd, pre-renders not distributed. Repo cloned for rendering scripts only.
- [x] Download UniRig Rig-XL dataset
- [x] Start training pipeline (issues #125-133 — dataset loader, model, training script, ONNX export)
- [x] Download more Mixamo characters (currently 62, target 150-250)
- [x] Download Mixamo animation clips to `data/poses/` (2,022 FBX clips)
- [x] ~~Run Live2D GitHub scraper~~ — 279 models scraped, GitHub saturated

---

## PRE-PROCESSED DATASETS (Download First — Highest ROI)

These are already rendered, annotated, or both. Downloading and converting them is dramatically cheaper than rendering from scratch.

### PP-1: NOVA-Human Dataset ⭐ HIGHEST PRIORITY

**What:** 10,200 VRoid characters pre-rendered from multiple views
**Source:** NOVA-3D GitHub repo (https://github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D)
**License:** Research use (derived from VRoid Hub models)

**What's included per character:**
- 4 fixed orthogonal views (front 0°, right 90°, back 180°, left 270°) — orthographic projection
- 16 randomly sampled perspective views with camera parameters
- Binary foreground masks per view
- XYZA normal maps per view
- Per-model metadata JSON with camera parameters
- All rendered at 512px, A-pose, diffuse-only lighting

**What this gives you:**
- ~163,200 pre-rendered multi-view anime character images
- Front, side, and back views already done for 10.2K characters
- Camera parameters included — can compute draw order, measurements

**What's missing (you need to add):**
- 45° (3/4) view — their fixed views are 0°, 90°, 180°, 270° only
- Segmentation masks (they have foreground masks, not body-part labels)
- Joint positions
- Contour line pairs

**Action items:**
- [ ] Download NOVA-Human dataset (follow repo instructions)
- [ ] Verify folder structure: ortho/, ortho_mask/, ortho_xyza/, rgb/, rgb_mask/, xyza/
- [ ] Estimate storage (~50–80 GB for full dataset)
- [ ] Plan supplementary 45° render pass using their source VRoid models

**Status:** BLOCKED on download. Dataset hosted exclusively on Alipan (`https://www.alipan.com/s/FqiHyraNCZd`) — inaccessible from outside China. GitHub issue #2 requesting alternative mirror went unanswered. Contacted see-through paper team in China for download assistance (March 2026). Adapter implemented (`nova_human_adapter.py`).

---

### PP-2: StdGEN Anime3D++ Semantic Annotations ⭐ HIGHEST PRIORITY

**What:** Train/test splits + Blender rendering scripts for 10,811 VRoid characters WITH semantic part maps
**Source:** StdGEN GitHub repo (https://github.com/hyz317/StdGEN)
**License:** Research use

**What's included:**
- `data/train_list.json` and `data/test_list.json` — curated character splits
- Blender rendering script (`blender/distributed_uniform.py`) for multi-view + semantic maps
- Semantic annotations: body, clothes, hair, face as separate layers
- Quality-filtered from ~14,000 to 10,811 characters (bad models already removed)

**What this gives you:**
- The curation work is done — which VRoid models are good quality, which aren't
- Semantic part maps (4-class) that partially map to Strata's 22-label taxonomy
- A proven Blender rendering pipeline you can extend rather than build from scratch

**What's missing (you need to add):**
- Their 4 semantic classes (body/clothes/hair/face) need splitting into Strata's 22 labels
- Their rendering script needs your additional outputs (joints, draw order, measurements, contours)
- The 45° camera angle

**Action items:**
- [ ] Clone StdGEN repo, download train/test lists
- [ ] Download pretrained weights from HuggingFace (hyz317/StdGEN)
- [ ] Install Blender + VRM addon per their instructions
- [ ] Map their 4-class semantic labels → Strata 22-label taxonomy
- [ ] Extend their `distributed_uniform.py` with Strata's additional render outputs
- [ ] Run extended rendering pipeline on the 10,811 curated characters

**Status:** BLOCKED. VRoid Hub models are gone — 0/15 sampled IDs returned 404 (March 2026). StdGEN team confirmed they cannot release pre-rendered images ("policy restrictions"). Repo cloned to `data/preprocessed/stdgen/StdGEN/` — rendering scripts and 4-class semantic mapping logic salvaged for use with any VRM models we obtain elsewhere. Train/test lists (10,702 + 109 IDs) archived but useless without source VRMs.

---

### PP-3: PAniC-3D VRoid Downloader (Master Collection Tool)

**What:** The VRoid Hub download tool + metadata that every other team builds on
**Source:** https://github.com/ShuhongChen/vroid-dataset
**License:** Tool is open source; downloaded models follow VRoid Hub per-model terms

**What's included:**
- `metadata.json` with all VRoid model attributions and IDs
- Download scripts using VRoid Hub cookie authentication
- ~14,500 model IDs curated by the PAniC-3D team

**What this gives you:**
- The actual .vrm source files that NOVA-Human and StdGEN rendered from
- Ability to render your own custom views (45° angle, new poses)
- Access to mesh data for measurement extraction ground truth

**Action items:**
- [ ] Clone vroid-dataset repo
- [ ] Download `panic_data_models_merged.zip` from their Google Drive
- [ ] Set up VRoid Hub cookie (login → devtools → copy `_vroid_session`)
- [ ] Run downloader to get the source .vrm files
- [ ] Cross-reference with StdGEN's train/test lists to know which 10,811 are quality-filtered

**Status:** BLOCKED. VRoid Hub models all return 404 as of March 2026. The scraper tool and metadata are archived in `data/preprocessed/stdgen/vroid-dataset/` but source VRM files cannot be downloaded. Same blocker as PP-2 and DS-2.

---

### PP-4: CharacterGen Rendering Scripts

**What:** Additional Blender + Three.js rendering scripts for VRoid → multi-view images
**Source:** https://github.com/zjp-shadow/CharacterGen
**License:** Research use

**What's included:**
- Blender render script: `render_script/blender/render.py`
- Three.js render script alternative
- Support for rendering in A-pose or with Mixamo FBX animation applied
- Model weights on HuggingFace (`zjpshadow/CharacterGen`) for the multi-view diffusion model

**What this gives you:**
- A second rendering pipeline option (useful for cross-validation)
- Ability to render VRoid characters with Mixamo animations applied (animated poses, not just A-pose)

**Action items:**
- [ ] Clone CharacterGen repo
- [ ] Review render scripts for features StdGEN's scripts might lack
- [ ] Consider using their animation-applied rendering for pose-diverse training data

**Status:** Directory structure created but not downloaded.

---

### PP-5: UniRig Rig-XL Dataset

**What:** 14,000+ rigged 3D models with skeleton + skinning weights
**Source:** https://github.com/VAST-AI-Research/UniRig
**License:** Released dataset (derived from Objaverse-XL, cleaned)

**What's included:**
- 14,000+ diverse rigged 3D models (humanoids, quadrupeds, fantasy creatures)
- Skeleton hierarchies
- Skinning weight assignments
- VRoid subset with anime-specific spring bones
- Model checkpoint on HuggingFace (VAST-AI/UniRig)

**What this gives you:**
- Ground truth skeleton + weight data for validating Strata's rigging pipeline
- Diverse non-humanoid models for future extension (wings, tails, etc.)
- A pre-trained auto-rigging model as potential fallback for unusual character shapes

**Action items:**
- [ ] Download Rig-XL dataset from UniRig releases
- [ ] Extract humanoid subset for skeleton/weight ground truth
- [ ] Map their skeleton conventions to Strata's 20-bone hierarchy
- [ ] Use as validation set for weight painting prediction

**Status:** Adapter planned (`unirig_skeleton_mapper.py` exists). Not yet downloaded.

---

### PP-6: AnimeRun Dataset ✅ FULLY INGESTED + UPLOADED

**What:** 2D animation frames with contours, optical flow, segmentation, and correspondence data
**Source:** https://lisiyao21.github.io/projects/AnimeRun
**License:** CC-BY-NC 4.0

**What's included:**
- Colored frames + corresponding contour-only (line art) frames
- Pixel-level optical flow between frames (forward + backward)
- Per-object instance segmentation maps
- Segment matching correspondence + occlusion masks
- Derived from open-source Blender movies (Agent 327, Caminandes 3, Sprite Fright)

**What this gives you:**
- ~2,819 contour/color paired frames — contour detection training
- Optical flow for motion estimation / frame interpolation training
- Instance segmentation for object tracking
- Temporal correspondence for draw order and occlusion understanding

**Action items:**
- [x] Download AnimeRun dataset (18 GB zip in `data/preprocessed/animerun/`)
- [x] Build contour adapter (`animerun_contour_adapter.py`) — 2,819 frames ingested
- [x] Upload contour pairs to Hetzner bucket (11,276 files, 689 MB)
- [x] Build optical flow adapter (`animerun_flow_adapter.py`) — issue #135 complete, 39 tests passing
- [x] Build instance segmentation adapter (`animerun_segment_adapter.py`) — issue #136 complete
- [x] Build temporal correspondence adapter (`animerun_correspondence_adapter.py`) — issue #137 complete, 43 tests
- [x] Build line area adapter (`animerun_linearea_adapter.py`) — 32 tests passing, 1,059 frames ingested
- [x] Run flow extraction + upload (batch 4) — 2,789 pairs, 16,704 files, 12 GB
- [x] Run segment extraction + upload (batch 4) — 2,819 frames, 11,276 files, 651 MB
- [x] Fix correspondence adapter (SegMatching `forward/` subdir + JSON format)
- [x] Run correspondence extraction + upload (batch 5) — 2,789 pairs, 19,493 files, 975 MB
- [x] Delete AnimeRun zip (all data types ingested, 21 GB reclaimed)
- [ ] Evaluate whether contour style matches anime/game character needs
- [ ] Supplement with Blender Freestyle renders for anime-specific contours

---

### PP-7: LinkTo-Anime VRoid+Mixamo Animation Dataset

**What:** 80 VRoid models rigged with Mixamo skeletons, animated, rendered from multiple camera angles
**Source:** arXiv 2506.02733 (check for GitHub/download link)
**License:** Check paper

**What's included:**
- 29,270 frames from 395 video clips
- 80 VRoid characters × diverse Mixamo motions (boxing, dancing, etc.)
- Forward/backward optical flow ground truth
- Occlusion masks
- Mixamo skeleton data per frame
- Multiple camera angles: full body, upper body, lower body, back view
- Both colored frames and line-art versions

**What this gives you:**
- Pre-rigged VRoid characters with Mixamo animations (exactly the VRoid→Mixamo pipeline you need)
- Optical flow and occlusion masks (bonus data for animation understanding)
- Multi-view animated character renders
- Line-art + color pairs (more contour training data)

**Action items:**
- [ ] Check if dataset is publicly released (paper is June 2025)
- [ ] If available, download full dataset
- [ ] Extract skeleton + optical flow data for animation intelligence training
- [ ] Use line-art pairs for contour detection supplementary data

**Status:** Adapter implemented (`linkto_adapter.py`). Directory structure created but not downloaded.

---

### PP-8: Supplementary Datasets

**skytnt/anime-segmentation** (HuggingFace) ✅ DOWNLOADED + FULLY INGESTED
- Figure/background segmentation for anime characters
- v1: 17 GB downloaded, 11,802 images ingested and uploaded to bucket (1.8 GB)
- v2: 12 GB downloaded, 13,000 images ingested and uploaded to bucket
- Combined: ~50,406 files in bucket under `anime_seg/`
- [x] Download v1
- [x] Ingest v1 (11,802 images → 35,406 files in bucket)
- [x] Download v2
- [x] Ingest v2 (13,000 images from fg-01/02/03 → ~15,000 files in bucket)

**dreMaz/AnimeInstanceSegmentationDataset** (HuggingFace)
- Instance segmentation (which pixels belong to which character in multi-character scenes)
- Useful for multi-character handling, not body-part segmentation
- [ ] Download: `git clone https://huggingface.co/datasets/dreMaz/AnimeInstanceSegmentationDataset`

**skytnt/fbanimehq** (HuggingFace) ✅ DOWNLOADED + FULLY INGESTED
- 112,806 full-body anime character images at 1024×512, background-removed
- All shards 00-11 ingested (304,889 files, 11.5 GB in bucket)
- Local leftovers cleaned (extracted dirs + zips deleted)
- [x] Download all shards
- [x] Ingest shards 00-07
- [x] Ingest shards 08-11 (batch 3)

**SMPL/SMPL-X** (smpl.is.tue.mpg.de)
- Parametric human body model with body-part segmentation definitions
- 6,890 vertices with semantic part labels (pelvis, left_hip, right_hip, spine, etc.)
- Useful for taxonomy validation and as additional realistic body mesh source
- [ ] Register and download SMPL model files
- [ ] Extract part segmentation mapping for taxonomy comparison

**MakeHuman** (makehuman.org)
- Open-source parametric human generator (AGPL license)
- Generate infinite diverse body types programmatically
- Already planned for template meshes T02–T05, but also a training data source
- [ ] Install MakeHuman
- [ ] Generate 200–500 diverse body shapes for rendering pipeline

---

### PP-9: HumanRig Dataset ⭐ HIGH PRIORITY — READY TO DOWNLOAD

**What:** 11,434 T-posed humanoid meshes with uniform Mixamo skeleton topology, skinning weights, and joint positions
**Source:** https://github.com/c8241998/HumanRig | Dataset: https://huggingface.co/datasets/jellyczd/HumanRig
**License:** MIT (code repo) + CC-BY-NC-4.0 (dataset) — non-commercial training use OK
**Paper:** "HumanRig: Learning Automatic Rigging for Humanoid Character in a Large Scale Dataset" CVPR 2025

**What's included:**
- 11,434 AI-generated T-posed meshes in Parquet format (diverse styles: realistic, cartoon, anthropomorphic)
- Uniform Mixamo skeleton topology — maps directly to Strata's 19-bone standard
- Varied head-to-body ratios (good diversity for non-standard proportions)
- Skinning weights + 3D joint positions + **2D joint positions** + camera parameters + front-view image per sample
- Train/val/test splits: 80/10/10

**What this gives you:**
- Ground truth skeleton + weight data for weight prediction MLP training
- Humanoid-only (unlike UniRig which is mixed categories) — no filtering needed
- Mixamo-compatible skeleton means adapter is trivial
- Includes pre-rendered **2D front-view images + 2D joint positions** — directly usable for joint CNN training without rendering step

**What's missing:**
- CC-BY-NC-4.0 — non-commercial only, same as AnimeRun
- No segmentation masks or draw order annotations
- Parquet format requires pandas/pyarrow to load

**Skeleton joint names (22 joints, Mixamo naming — maps directly to Strata):**
`Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand, LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase, RightUpLeg, RightLeg, RightFoot, RightToeBase`

**Action items:**
- [x] Download: `humanrig.zip` (36 GB) downloaded to `data/preprocessed/humanrig/` on external HD
- [ ] Wait for extraction to complete (running in background)
- [ ] Inspect extracted structure + verify content
- [ ] Build `humanrig_skeleton_mapper.py` adapter (maps to Strata 22-bone hierarchy)
- [ ] Ingest 2D images + joint positions as joint refinement training data (no rendering needed)
- [ ] Optionally render 3D meshes with Blender for segmentation training images

**Status:** DOWNLOADING/EXTRACTING. 36 GB zip downloaded, extraction in progress. On external HD at `data/preprocessed/humanrig/`.

---

### PP-10: Anymate Dataset ⭐ HIGH PRIORITY — READY TO DOWNLOAD

**What:** 230K rigged 3D assets with expert-crafted skeleton + skinning weights, curated from Objaverse-XL
**Source:** https://github.com/yfde/Anymate | Dataset: https://huggingface.co/datasets/yfdeng/Anymate
**License:** Not specified in repo — check HuggingFace dataset card before use
**Paper:** "Anymate: A Dataset and Baselines for Learning 3D Object Rigging" SIGGRAPH 2025

**What's included:**
- 230K diverse 3D assets (humanoids, animals, vehicles, furniture, etc.)
- Artist-created rigging: joint positions, bone connectivity, skinning weights per vertex
- 70× larger than any prior public rigging dataset
- PyTorch `.pt` format: 9 files (Anymate_test.pt + Anymate_train_0.pt through Anymate_train_7.pt)
- Downloaded via `bash Anymate/get_datasets.sh` (fetches from HuggingFace)

**What this gives you:**
- Massive scale for weight prediction MLP training (230K > UniRig's 14K)
- Ground truth artist-crafted weights (higher quality than auto-computed)
- Humanoid subset (filter needed) could give 20K+ high-quality humanoid rigs

**What's missing:**
- License not stated — verify before training use
- Mixed categories — need to filter to humanoid subset
- No 2D renders or segmentation annotations included
- Skeleton topology varies per asset (not uniform Mixamo — harder to map than HumanRig)
- Size unknown — likely very large (230K meshes with skinning data)

**Action items:**
- [ ] Check HuggingFace dataset card for license: https://huggingface.co/datasets/yfdeng/Anymate
- [ ] If license OK: run `bash Anymate/get_datasets.sh` to download
- [ ] Filter to humanoid subset (estimate ~20–40K of 230K)
- [ ] Map diverse skeleton topologies → Strata 19-bone standard
- [ ] Build `anymate_skeleton_mapper.py` adapter

**Status:** READY (pending license verification). Direct HuggingFace download via get_datasets.sh script.

---

### PP-11: MagicAnime Dataset — Keypoint Subset

**What:** 400K cartoon video clips with a 50K-clip subset with 133 whole-body keypoint annotations (incl. 68 facial)
**Source:** https://arxiv.org/abs/2507.20368 — "MagicAnime: A Hierarchically Annotated, Multimodal and Multitasking Dataset" (July 2025)
**License:** Research-only, restricted access — requires institutional affiliation + application + signed agreement

**What's included:**
- 50K video clip + whole-body keypoint annotation pairs (133 keypoints per frame incl. 68 facial)
- Cartoon animation style (not 3D render style)
- Temporal sequences (useful for animation intelligence)

**What this gives you:**
- Large-scale whole-body joint annotation data for cartoon characters
- Joint refinement CNN training on non-3D-rendered style (fills domain gap)
- Temporal sequences for animation intelligence data

**What's missing:**
- **BLOCKED: Restricted access** — requires institutional affiliation + formal application process
- No body part segmentation masks
- Video-based — need frame extraction
- No GitHub or HuggingFace link provided in paper

**Action items:**
- [ ] Contact authors via arXiv email if institutional access is available
- [ ] Otherwise, deprioritise — restricted access makes this impractical

**Status:** BLOCKED on access. Restricted to research institutions via application. Not downloadable without approval.

---

### PP-12: AnimeDrawingsDataset — Anime/Manga Joint Keypoints

**What:** 2,000 annotated anime/manga images with 9 skeleton joint positions
**Source:** https://github.com/dragonmeteor/AnimeDrawingsDataset
**License:** Not specified in repository

**What's included:**
- 2,000 images (1,400 train / 100 val / 500 test)
- 9 joint keypoints per image (head, neck, shoulders, hips, elbows, wrists, knees, ankles, feet)
- JSON format annotations
- Actual hand-drawn anime/manga style images (not 3D renders)

**What this gives you:**
- Joint training data in authentic illustrated style — fills domain gap vs 3D render data
- Small but high-quality; good for fine-tuning joint refinement CNN on 2D art style

**What's missing:**
- Only 9 joints (Strata needs 19) — spine, chest, forearms not covered
- Very small dataset (2K images)
- No body part segmentation masks
- No license specified

**Action items:**
- [ ] Clone repo: `git clone https://github.com/dragonmeteor/AnimeDrawingsDataset`
- [ ] Install Ruby + bundler, run `bundle install && rake build` to download images
- [ ] Or use Docker: `docker run dragonmeteor/animedrawingsdataset`
- [ ] Check license (contact author if unclear)
- [ ] Build adapter to map 9 joints → Strata's 19-bone subset (9-joint subset usable)
- [ ] Ingest as supplementary joint training data

**Status:** READY. Small dataset (~2K images) but directly addresses illustrated-style joint domain gap. Download via Ruby/rake build or Docker.

---

### PP-13: Body Part Segmentation of Anime Characters (Ou et al. 2024)

**What:** Body part segmentation dataset specifically for anime characters — multiple semantic regions
**Source:** https://onlinelibrary.wiley.com/doi/10.1002/cav.2295 (CGI 2024 / Computer Animation and Virtual Worlds)
**License:** Unknown — academic paper, dataset availability unclear

**What this gives you:**
- Anime-specific body part segmentation ground truth (most relevant to Strata's segmentation model)
- 2D illustration style — directly fills the domain gap vs 3D renders
- Part labels align with Strata's 22-region taxonomy (details need verification)

**Action items:**
- [ ] Access paper (may need institutional access or email authors)
- [ ] Check if dataset is publicly released
- [ ] If released, download and assess annotation quality + label taxonomy
- [ ] Map their label taxonomy → Strata's 22-region taxonomy

**Status:** Not started. Paper is paywalled — email authors for dataset access.

---

### PP-14: See-through Live2D Layer Decomposition Dataset ⏳ AVAILABLE LATE MARCH 2026

**What:** 9,102 annotated 2.5D Live2D models with pixel-perfect body part segmentation + draw order
**Source:** https://arxiv.org/abs/2602.03749 (SIGGRAPH Asia 2025)
**License:** Not yet specified — authors committed to releasing dataset on paper acceptance

**What's included:**
- 9,102 fully annotated 2.5D Live2D models (7,404 train / 851 val / 847 test)
- 19 semantic body part classes with pixel-perfect boundaries
- Occluded region labels (hidden anatomy behind overlapping parts)
- Fragment-level draw order (pseudo-depth values) — exactly what Strata's draw_order.png requires
- Sourced from ArtStation, Booth, DeviantArt

**What this gives you:**
- Draw order ground truth from real Live2D models — directly trains Strata's draw order prediction head
- 19-class body part segmentation aligned with Strata's taxonomy
- Occluded region data: unique annotation type not available in any other dataset
- 2D illustration style characters — fills domain gap vs 3D renders

**Action items:**
- [ ] Monitor https://arxiv.org/abs/2602.03749 and author GitHub for dataset release (expected late March 2026)
- [ ] Request early access from authors (contact via paper email)
- [ ] When released: download, verify label taxonomy matches Strata's 22 regions
- [ ] Build `see_through_adapter.py` to ingest annotations

**Status:** PENDING RELEASE. Highest-value dataset found — directly provides draw order + 19-class segmentation for Live2D illustrated characters. Check back late March 2026.

---

### PP-15: ChildlikeSHAPES — Hand-drawn Figure Segmentation

**What:** 16,075 annotated hand-drawn childlike figure drawings with 25-class semantic segmentation
**Source:** https://arxiv.org/abs/2504.08022
**License:** Not yet specified — authors plan to release on paper acceptance

**What's included:**
- 16,075 manually annotated drawings (14,075 train / 2,000 test)
- 25 semantic classes: body parts + facial features (eyes, nose, mouth, ears, eyebrows)
- Built from the Amateur Drawings Dataset
- Pixel-level masks per class

**What this gives you:**
- Segmentation training data for sketch/hand-drawn art style inputs
- Bridges the gap to Strata's sketch style output (one of 6 render styles)
- 25-class taxonomy overlaps substantially with Strata's 22 regions

**What's missing:**
- Childlike/schematic drawing style — more abstract than typical anime/game characters
- No joint positions or draw order annotations
- Dataset pending release (paper under review)

**Action items:**
- [ ] Monitor arXiv paper for acceptance + dataset release
- [ ] When released: download, map 25-class taxonomy → Strata's 22 regions
- [ ] Use primarily for sketch-style generalization in segmentation model

**Status:** PENDING RELEASE. Watch paper for acceptance.

---

## PRIMARY DATA SOURCES (Gather + Render)

These require downloading raw assets and running your own rendering pipeline.

### DS-1: Mixamo FBX Characters ✅ PARTIALLY COMPLETE

**What:** Rigged, animated 3D humanoid characters from Adobe Mixamo (mixamo.com)
**Format:** FBX files with skeleton + mesh + textures
**Why:** Primary source of realistic/western-style segmentation ground truth with perfect bone-to-label mapping. No pre-processed dataset covers this style.
**License:** Free for use (Adobe account required)

| Item | Target Volume | Current | Status |
|------|--------------|---------|--------|
| Character FBX models | 150–250 | 105 | 100% of available (108 total on Mixamo) |
| Animation FBX clips | 50–100 | 2,022 | ✅ In `data/poses/` |
| Diverse body types | Cover all 8 archetypes | Partial | Need more variety |
| Male/female split | ~50/50 | Unknown | Check distribution |

**Camera angles rendered:** 0° (front), 45° (3/4), 90° (side), 135° (3/4 back), 180° (back)

**Rendering status:**
- [x] Pipeline working end-to-end (flat style + seg masks + joints + draw order + layers)
- [x] 22-class region IDs aligned with Strata's skeleton.ts
- [x] Re-render in progress: 62 chars × all poses × flat style × front angle
- [x] Per-region RGBA layer extraction working (Blended transparency + Emission shader)
- [x] 49 chars OK, 12 PARTIAL (2/5 poses due to `mixamorig5:` bone prefix mismatch)
- [ ] Download more characters (target: 150-250 total)
- [ ] Render additional styles (cel, pixel, painterly, sketch, unlit)
- [ ] Render multi-angle (3/4, side, back) passes

---

### DS-2: VRoid Characters (Via PAniC-3D + StdGEN Pipeline)

**What:** Anime-style 3D characters — now sourced primarily through pre-processed datasets above
**Format:** VRM files from PAniC-3D downloader; pre-rendered images from NOVA-Human + StdGEN

**Revised strategy:** Instead of scraping VRoid Hub from scratch, the workflow is:

1. **Download NOVA-Human pre-renders** (PP-1) — gives you 10.2K characters × 20 views = 204K images already rendered
2. **Download StdGEN train/test lists** (PP-2) — gives you the quality-filtered character IDs + semantic annotations
3. **Download source VRM files** via PAniC-3D downloader (PP-3) — gives you the raw models for custom rendering
4. **Extend StdGEN's Blender script** — add 45° camera angle, Strata's 22-label segmentation, joints, draw order, measurements, contours
5. **Run extended pipeline on 10,811 curated characters** — fills the gaps that pre-processed data doesn't cover

**What you get from pre-processed data (free):**
- 10,200 characters × 4 ortho views = ~40,800 images (NOVA-Human)
- 10,200 characters × 16 random views = ~163,200 images (NOVA-Human)
- Foreground masks for all views (NOVA-Human)
- Semantic part maps for 10,811 characters (StdGEN, 4-class)
- Quality-filtered model list (StdGEN)

**What you need to render yourself:**
- 45° (3/4) view for all characters
- Strata 22-label segmentation masks (extend StdGEN's 4-class)
- Joint positions per view
- Draw order maps
- Body part measurements
- Contour line pairs
- Additional poses beyond A-pose

**Expected new rendering work:** ~10,000 characters × 1 new angle (45°) × 5 poses = ~50,000 images to render. Much less than the original plan of rendering everything from scratch.

**Status:** MOSTLY BLOCKED. VRoid Hub models are gone (all 404'd as of March 2026). PAniC-3D scraper and StdGEN pipeline both depend on VRoid Hub. Only VRoid data we have: 16 CC0 characters ingested + uploaded (9,302 files, 788 MB). VRoid importer + mapper implemented. Unless VRM files can be obtained from another source, this pipeline is limited to the 16 CC0 chars.

---

### DS-3: Live2D Community Models

**What:** 2D illustrated character models in Live2D format (.moc3 + texture atlas)
**Format:** .moc3 model files with PNG texture atlases
**Why:** Bridges 3D render → 2D illustration domain gap. Front-facing only but with pre-decomposed body part layers.
**License:** Per-model (check carefully)

| Item | Target Volume | Where to Get | Priority |
|------|--------------|--------------|----------|
| Live2D models (free/CC-licensed) | 300–500 | Booth.pm (free section) | P1 |
| Additional from DeviantArt | 50–100 | DeviantArt Live2D tag | P2 |
| Open-source VTuber models | 30–50 | GitHub repositories | P1 |
| Live2D official samples | 10–20 | Live2D Inc. sample downloads | P0 |

**Expected output:** ~400 models × 4 augmentations = **~1,600 training images**

**Note:** The See-through paper team collected ~9,100 Live2D models. If they release their dataset or data engine code, that would supersede manual collection. Monitor their repo.

**Status:** Pipeline implemented (renderer, review UI, Live2D mapper, .moc3 parser, atlas fragment extractor). GitHub scraper built and run — **279 models downloaded** (GitHub source saturated). .moc3 binary parser + atlas fragment extraction working (issue #146). Chinese body-part regex patterns added to config. Next: process 279 models through renderer to generate training images, then expand collection via Booth.pm.

---

### DS-4: CMU Motion Capture (BVH)

**What:** Human motion capture data in BVH format
**Source:** cgspeed.com (BVH conversions) or mocap.cs.cmu.edu
**License:** Free for research and commercial use

| Item | Target Volume | Current | Status |
|------|--------------|---------|--------|
| CMU BVH motion clips | 2,548 (full dataset) | 2,548 | ✅ Downloaded to `data/mocap/cmu/` |
| Action labels per clip | All clips labeled | 80 | `animation/labels/cmu_action_labels.csv` |
| Strata-compatible flag per clip | All clips flagged | 80 | In labels CSV |

**Expected output:** ~~~500 Strata-compatible clips × 7 degradations = ~3,500 training pairs~~ **ACTUAL: 2,548 clips × 7 degradations = 17,822 training pairs** (all clips compatible after bone mapping fixes)

**Note:** LinkTo-Anime (PP-7) provides 80 VRoid characters pre-rigged with Mixamo skeletons — this may partially overlap with CMU data needs for animation training.

**Status:** COMPLETE + UPLOADED. All 2,548 clips retargeted + degraded → 17,823 JSON training pairs (58 GB). Uploaded to `animation/` in Hetzner bucket (17,823 files). Local copy deleted. BVH parser, retargeting, and degradation scripts implemented. Action labels started (80 clips).

---

### DS-5: PSD Files (Opportunistic)

**What:** Layered Photoshop documents with body-part-separated layers
**Format:** .psd files
**Why:** Real artist work with natural layer structure as segmentation ground truth
**License:** Varies — only collect files with explicit permission for derivative/ML use

| Item | Target Volume | Where to Get | Priority |
|------|--------------|--------------|----------|
| Character PSDs with body-part layers | 50–100 (aspirational) | OpenGameArt, itch.io asset packs, Patreon art packs | P3 |

**Expected output:** **~50–100 training images** (small but high-value for style diversity)

**Status:** PSD extractor implemented (`pipeline/psd_extractor.py`). No PSD files acquired yet.

---

### DS-6: Contour Line Augmentation (Generated)

**Revised strategy:** Start with pre-processed contour data from AnimeRun (PP-6) and LinkTo-Anime (PP-7), then supplement with Blender Freestyle renders.

**Pre-processed contour data available:**
- AnimeRun: 2,819 contour/color paired frames ingested and in bucket ✅
- LinkTo-Anime: 29,270 frames with both colored and line-art versions (not yet downloaded)

**Still need to generate:**
- Anime-specific contour styles (AnimeRun is from 3D movies, not anime characters)
- Contour pairs at all 5 camera angles for Mixamo + VRoid characters

| Style | Width | Color | Effect |
|-------|-------|-------|--------|
| Thin black | 1px | (0,0,0) | Clean manga/anime line |
| Medium black | 2px | (0,0,0) | Standard cartoon outline |
| Thick brown | 3px | (50,25,12) | Painterly/warm line art |
| Colored per-region | 1px | Varies by body part | "Colored line" anime style |
| Hand-drawn wobbly | 2px + jitter | (0,0,0) | Simulates hand-drawn inconsistency |

**Expected output:** ~10,000 base pairs × 5 styles = **~50,000 contour training pairs** (supplemented by ~37,000 pre-processed pairs)

**Status:** Contour renderer and augmenter implemented. AnimeRun contour pairs ingested. Freestyle rendering pipeline ready.

---

## OUTPUTS GENERATED (What the Pipeline Produces)

### Per training image

| Output | Format | Used By | Priority |
|--------|--------|---------|----------|
| Composite image | PNG 512×512 | All models | P0 |
| Segmentation mask | PNG 512×512 (22 colors) | Segmentation model | P0 |
| 2D joint positions | JSON (x,y per joint) | Joint prediction model | P0 |
| Draw order map | PNG grayscale (0=back, 255=front) | Layer ordering model | P1 |
| Per-region RGBA layers | PNG 512×512 (transparent bg) | Layer decomposition model | P1 |
| Camera angle | Float in metadata.json | Multi-view consistency | P0 |
| Character ID + Pose ID | String in metadata.json | Cross-view linking | P0 |
| Body part measurements | JSON (width, height, depth per label) | Measurement extraction validation | P1 |
| True 3D dimensions | JSON (from mesh bounding boxes) | Template deformation ground truth | P1 |
| Contour mask | PNG binary | Contour removal model | P1 |
| Source type | String in metadata.json | Domain balance during training | P0 |

### Per character (aggregated)

| Output | Format | Used By | Priority |
|--------|--------|---------|----------|
| Measurement profile | JSON (all body part dimensions) | Template archetype clustering | P1 |
| Proportion archetype | Enum (8 archetypes) | Template selection model | P2 |

### Animation outputs

| Output | Format | Used By | Priority |
|--------|--------|---------|----------|
| Optical flow pairs | NPY (H×W×2 float32) + frames | Frame interpolation / in-betweening | P1 |
| Instance segmentation | PNG (pixel = instance ID) + frames | Object tracking | P2 |
| Temporal correspondence | Occlusion masks + segment matching | Draw order / animation coherence | P2 |
| Retargeted BVH → Strata blueprint | JSON | Animation Intelligence | P1 |
| Degraded animation pairs | JSON pairs (sparse, full) | In-betweening model | P1 |
| Timing norms per action | JSON statistics | Animation Intelligence | P2 |

---

## TOTAL VOLUME SUMMARY (Revised with Actuals)

| Source | Target | Actual | Status |
|--------|--------|--------|--------|
| NOVA-Human (PP-1) | ~204,000 images | 0 | BLOCKED — Alipan only, seeking China-based help |
| StdGEN semantic maps (PP-2) | 10,811 chars | 0 | BLOCKED — VRoid Hub models all 404'd |
| HumanRig (PP-9) | 11,434 meshes + 2D images | 0 | DOWNLOADING — 36 GB zip on external HD, extracting |
| Anymate (PP-10) | 230K assets (humanoid subset ~20-40K) | 0 | READY (pending license check) — HuggingFace via get_datasets.sh |
| MagicAnime keypoints (PP-11) | 50K clips | 0 | BLOCKED — restricted access, institutional affiliation required |
| AnimeDrawingsDataset (PP-12) | 2,000 images | 0 | READY — clone + rake build or Docker |
| Body Part Seg Anime Ou 2024 (PP-13) | Unknown | 0 | Not started — email authors |
| See-through Live2D (PP-14) | 9,102 annotated chars | 0 | PENDING — expected late March 2026 |
| ChildlikeSHAPES (PP-15) | 16,075 drawings | 0 | PENDING — awaiting paper acceptance |
| AnimeRun contour pairs (PP-6) | ~8,000 | 2,819 ingested | ✅ In bucket |
| AnimeRun linearea (PP-6) | ~1,000 | 1,059 ingested | ✅ In bucket |
| AnimeRun flow | ~2,800 | 2,789 ingested | ✅ In bucket (16,704 files) |
| AnimeRun segment | ~2,800 | 2,819 ingested | ✅ In bucket (11,276 files) |
| AnimeRun correspondence | ~2,800 | 2,789 ingested | ✅ In bucket (19,493 files) |
| LinkTo-Anime (PP-7) | ~29,270 | 0 | SKIPPED — CC-BY-NC license forbidden |
| UniRig Rig-XL (PP-5) | 14,000 meshes | 16,641 meshes | ✅ Downloaded + extracted (66 GB in `data/preprocessed/unirig/rigxl/`) |
| CMU animation pairs | 17,823 | 17,823 uploaded | ✅ In bucket (58 GB) |
| Mixamo renders (DS-1) | ~10,000 | 5,250 rendered | ✅ In bucket (12,216 files, 619 MB) |
| VRoid Lite (DS-2) | 4,651 | 4,651 ingested | ✅ In bucket (9,302 files) |
| VRoid supplementary renders | ~50,000 | 0 | BLOCKED — VRoid Hub models gone |
| Live2D composites (DS-3) | ~1,600 | 1,112 rendered | ✅ In bucket (4,764 files, 327 MB) |
| FBAnimeHQ (PP-8) | 112,806 | ~101,630 ingested | ✅ All shards in bucket |
| anime-segmentation (PP-8) | ~25,000 | ~24,800 | ✅ v1 + v2 in bucket |
| PSD extractions (DS-5) | ~50–100 | 0 | Extractor ready |
| Generated contour pairs | ~50,000 | 0 | Pipeline ready |
| **TOTAL** | **~470,000+** | **~164,000+** | **~35%** |

---

## GATHERING TIMELINE (Revised with Progress)

### ~~Week 1 (Downloads — Immediate)~~ PARTIALLY COMPLETE

- [ ] **Download NOVA-Human dataset** (PP-1) — highest priority, ~50-80 GB
- [ ] **Clone StdGEN repo** (PP-2) — get train/test lists, rendering scripts, semantic annotation code
- [ ] **Clone PAniC-3D vroid-dataset** (PP-3) — get downloader + metadata.json
- [ ] **Clone CharacterGen repo** (PP-4) — get alternative rendering scripts
- [ ] **Download UniRig Rig-XL dataset** (PP-5) — rigged mesh ground truth
- [x] **Download AnimeRun dataset** (PP-6) — contour line pairs ✅
- [ ] **Check LinkTo-Anime availability** (PP-7) — if released, download
- [x] Download Mixamo FBX characters (61/250 downloaded) — need more
- [x] Download CMU BVH dataset (full 2,548 clips) ✅
- [x] Download anime-segmentation (v1 + v2) ✅
- [x] Download FBAnimeHQ (all shards) ✅
- [ ] Set up license tracking CSV for all sources

### ~~Week 2–3 (Setup + First Renders)~~ PARTIALLY COMPLETE

- [ ] Set up VRoid Hub cookie and run PAniC-3D downloader for source VRM files
- [ ] Cross-reference NOVA-Human models with StdGEN quality-filtered list
- [ ] Map StdGEN 4-class semantics → Strata 22-label taxonomy
- [ ] Extend StdGEN Blender rendering script with Strata outputs (joints, draw order, measurements, contours, 45° camera)
- [x] Mixamo Blender rendering pipeline working (49 chars done) ✅
- [ ] Download Live2D official sample models (~10–20)
- [x] Start CMU action labeling (80 clips labeled) ✅

### Week 3–6 (Main Rendering Phase)

- [ ] Run extended StdGEN pipeline on 10,811 VRoid characters (45° angle + Strata annotations)
- [ ] Complete Mixamo render pipeline (all 250 characters × poses × angles × styles)
- [ ] Begin Booth.pm Live2D collection (target 200 models)
- [ ] Begin Live2D fragment→label mapping + review
- [ ] Extract measurement ground truth from VRoid + Mixamo meshes
- [ ] Run proportion clustering on measurement profiles → determine template archetypes
- [ ] Generate contour line augmentation pairs from Mixamo renders

### Week 6–10 (Completion + Validation)

- [ ] Complete Live2D collection to 400+ models, finish manual review
- [ ] Complete contour line augmentation for VRoid renders
- [ ] Complete CMU action labeling
- [ ] Build BVH retargeting pipeline + synthetic degradation pairs
- [ ] Run validation: check segmentation model on held-out multi-view test set
- [ ] Fill gaps in body type / art style coverage based on clustering results
- [ ] Opportunistically collect PSD files

### Week 10–12 (Quality Pass)

- [ ] Validate measurement extraction accuracy against mesh ground truth
- [ ] Validate segmentation consistency across views for same character
- [ ] Identify and fix systematic failures (specific body types, styles, or angles)
- [ ] Generate final training/validation/test splits

---

## INFRASTRUCTURE COMPLETED

### Pipeline Modules (32 implemented + 1 scraper)

**Core rendering:** generate_dataset, renderer, layer_extractor, draw_order_extractor, exporter, importer, joint_extractor
**Data processing:** bone_mapper, live2d_mapper, vroid_mapper, vroid_importer, weight_extractor, measurement_ground_truth
**Post-processing:** style_augmentor, contour_augmenter, contour_renderer, psd_extractor, pose_estimator, pose_applicator
**2D character support:** spine_parser, live2d_renderer, live2d_review_ui, moc3_parser
**Dataset management:** validator, manifest, splitter, multiview_validator, dataset_merger, measurement_extractor
**Data acquisition:** run_live2d_scrape (GitHub .moc3 repo search + sparse checkout download)
**Configuration:** config, accessory_detector

### Ingest Adapters (9 registered + supporting modules)

| Adapter | CLI name | Status |
|---------|----------|--------|
| `fbanimehq_adapter.py` | `--adapter fbanimehq` | ✅ Working |
| `nova_human_adapter.py` | `--adapter nova_human` | ✅ Working (no data) |
| `anime_seg_adapter.py` | `--adapter anime_seg` | ✅ Working |
| `animerun_contour_adapter.py` | `--adapter animerun` | ✅ Working |
| `animerun_flow_adapter.py` | `--adapter animerun_flow` | ✅ Working |
| `animerun_segment_adapter.py` | `--adapter animerun_segment` | ✅ Working (#136 complete) |
| `animerun_correspondence_adapter.py` | `--adapter animerun_correspondence` | ✅ Working (#137 complete) |
| `animerun_linearea_adapter.py` | `--adapter animerun_linearea` | ✅ Working (32 tests) |
| `vroid_lite_adapter.py` | `--adapter vroid_lite` | ✅ Working (4,651 images ingested) |
| `linkto_adapter.py` | — | ✅ Implemented (not registered) |
| `stdgen_semantic_mapper.py` | — | 📋 Planned |
| `unirig_skeleton_mapper.py` | — | 📋 Planned |
| `humanrig_skeleton_mapper.py` | — | 📋 Planned (PP-9) |
| `anymate_skeleton_mapper.py` | — | 📋 Planned (PP-10) |
| `anime_drawings_adapter.py` | — | 📋 Planned (PP-12) |
| `see_through_adapter.py` | — | 📋 Planned (PP-14, pending release) |

### Training Infrastructure (issues #122-133)

| Component | Issue | Status |
|-----------|-------|--------|
| Dataset loader | #125 | ✅ Implemented |
| Multi-head DeepLabV3+ model | #126 | ✅ Implemented |
| Training metrics (mIoU) | #127 | ✅ Implemented |
| Segmentation training script | #128 | ✅ Implemented |
| ONNX export pipeline | #129 | ✅ Implemented |
| ONNX validation script | #130 | ✅ Implemented |
| Joint refinement training | #131 | ✅ Implemented |
| Weight prediction training | #132 | ✅ Implemented |
| Evaluation/visualization | #133 | ✅ Implemented |

### Cloud Storage (Hetzner Object Storage)

- Bucket: `s3://strata-training-data` (Falkenstein datacenter)
- Cost: €4.99/month (1 TB storage + 1 TB egress)
- Total uploaded: ~87 GB across 10 dataset prefixes
- Access: AWS CLI compatible, credentials in `.env`

---

## STORAGE ESTIMATE (Revised)

| Source | Size | Notes |
|--------|------|-------|
| NOVA-Human download | ~50–80 GB | 10.2K chars × 20 views × images + masks + normals |
| PAniC-3D VRM source files | ~30 GB | ~14K .vrm files |
| UniRig Rig-XL | ~20 GB | 14K rigged meshes |
| AnimeRun | ~18 GB (deleted, in bucket) | All 5 data types ingested + uploaded |
| LinkTo-Anime | ~10 GB (est.) | 29K frames + annotations |
| FBAnimeHQ | ~17 GB (local) | 112K images |
| anime-segmentation | ~29 GB (local, v1+v2) | fg/bg segmentation |
| **Pre-processed total** | **~175–205 GB** | |
| Mixamo FBX files | ~5 GB | Raw models |
| Mixamo renders | ~8 GB | 10K images + all annotations |
| VRoid supplementary renders | ~40 GB | 50K images (45° + poses + Strata annotations) |
| Live2D models + renders | ~3.5 GB | 400 models + 1,600 rendered images |
| CMU BVH + processed | ~58 GB | 2,548 clips + retargeted + degraded (uploaded, local deleted) |
| PSD files | ~2 GB | Opportunistic |
| Generated contour pairs | ~40 GB | 50K pairs |
| **Self-rendered total** | **~100 GB** | |
| **GRAND TOTAL** | **~275–305 GB** | |

**Hetzner bucket capacity:** 1 TB (plenty of headroom)
**Local disk:** 460 GB total, ~113 GB free — extract→ingest→upload→delete workflow freed ~55 GB

---

## LEGAL CHECKLIST

- [ ] Mixamo: Confirm Adobe terms allow ML training use
- [ ] VRoid Hub / PAniC-3D collection: Document per-model license in manifest CSV
- [ ] NOVA-Human dataset: Check NOVA-3D release terms (research use)
- [ ] StdGEN Anime3D++: Check release terms (likely research use, CC-BY-NC for paper)
- [ ] UniRig Rig-XL: Check release terms (Objaverse-XL derived, likely permissive)
- [ ] AnimeRun: CC-BY-NC 4.0 — non-commercial only, check if Strata's use qualifies
- [ ] LinkTo-Anime: Check paper's license terms
- [ ] FBAnimeHQ: Derived from Danbooru — check terms
- [ ] Live2D models: Document per-model license in manifest CSV
- [ ] Live2D SDK: If using Cubism SDK for rendering, check license terms
- [ ] CMU mocap: Confirm free use terms (historically permissive)
- [ ] PSD files: Only collect with explicit derivative/ML training permission
- [ ] SMPL: Non-commercial research license — check if applicable
- [ ] MakeHuman: AGPL — check implications for generated training data

---

## QUICK REFERENCE: What's Free vs What Costs Time

| Task | Effort | Impact | Status |
|------|--------|--------|--------|
| Download NOVA-Human | 1 day (bandwidth) | 204K multi-view anime images | Not started |
| Download StdGEN lists + scripts | 1 hour | Quality-filtered model IDs + render pipeline | Not started |
| Download UniRig Rig-XL | 1 day (bandwidth) | 14K rigged meshes with weights | Not started |
| Download AnimeRun | 2 hours | 8K contour/color pairs | ✅ Done |
| Download FBAnimeHQ | 1 day (bandwidth) | 112K full-body anime images (unlabeled) | ✅ Done |
| Download anime-segmentation | 4 hours | 22K+ fg/bg segmentation pairs | ✅ Done |
| Ingest AnimeRun (flow/seg/corr) | Done (batch 4+5) | All data types in bucket | ✅ Done |
| Ingest FBAnimeHQ shards 08-11 | Done | All shards in bucket | ✅ Done |
| Ingest anime_seg_v2 | Done | 13K images in bucket | ✅ Done |
| Ingest vroid_lite | Done | 4,651 images in bucket | ✅ Done |
| Build Live2D GitHub scraper | Done | `run_live2d_scrape.py` for .moc3 repos | ✅ Done |
| Build .moc3 parser + extractor | Done | Parses binary mesh data, extracts fragments from atlas | ✅ Done |
| Run Live2D GitHub scraper | Done | 279 models downloaded, GitHub saturated | ✅ Done |
| Upload CMU animation | Done | 17,823 files (58 GB) in bucket | ✅ Done |
| Fix 22-class region IDs | Done | Pipeline, config, tests all aligned with Strata skeleton.ts | ✅ Done |
| Download HumanRig (PP-9) | 1 hour | 11.4K humanoid meshes + 2D images + joints, CC-BY-NC-4.0 | READY — `huggingface-cli download jellyczd/HumanRig` |
| Check Anymate license + download (PP-10) | 30 min + 1 day | 230K rigged meshes, humanoid subset ~20-40K | READY (check license first) |
| Contact MagicAnime authors (PP-11) | 30 min | 50K cartoon clips with 133-keypoint annotations | BLOCKED — institutional access only |
| Download AnimeDrawingsDataset (PP-12) | 30 min | 2K anime/manga joint annotations (illustrated style) | READY — clone + rake build |
| Email Ou 2024 authors for dataset (PP-13) | 30 min | Anime body part seg ground truth (2D style) | Not started |
| Monitor See-through release (PP-14) | — | 9.1K Live2D chars with draw order + 19-class seg | Late March 2026 |
| Monitor ChildlikeSHAPES release (PP-15) | — | 16K hand-drawn figures with 25-class seg masks | Pending paper accept |
| Run PAniC-3D downloader | 2–3 days | Source VRM files for custom rendering | Not started |
| Extend StdGEN Blender script | 3–5 days (coding) | Adds all Strata-specific outputs | Not started |
| Render 45° + Strata annotations for 10K VRoid | 1–2 weeks (compute) | Core multi-view training data | Not started |
| Render more Mixamo chars | 2–3 days (compute) | Western-style training data | 49/250 done |
| Live2D collection + mapping | 2–3 weeks | 2D illustration style coverage | Not started |
| CMU labeling + retargeting | 2–3 weeks | Animation intelligence data | ✅ Retargeted + uploaded |
| Start model training | 1–2 weeks (coding) | Segmentation model MVP | Issues #125-128 open |
