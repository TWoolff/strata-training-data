# Pre-Processed External Datasets

Catalog of external pre-processed datasets that can be downloaded and converted into Strata training data via `ingest/` adapter scripts. Each dataset is downloaded to `data/preprocessed/{dataset_name}/`.

See also: `docs/data-sources.md` for primary pipeline sources (Mixamo, Sketchfab, etc.).

## Summary

| Dataset | Est. Images | Download Size | License | Adapter Status |
|---------|-------------|---------------|---------|----------------|
| [NOVA-Human](#nova-human) | ~204K | ~50–80 GB | Research (per-model VRoid terms) | `nova_human_adapter.py` |
| [StdGEN / Anime3D++](#stdgen--anime3d) | ~10.8K chars | Varies | Apache-2.0 (code); research (data) | Planned: `stdgen_semantic_mapper.py` |
| [UniRig / Rig-XL](#unirig--rig-xl) | 14K meshes | ~20 GB | MIT | Planned: `unirig_skeleton_mapper.py` |
| [AnimeRun](#animerun) | ~8K pairs | ~5 GB | Not specified | Planned: `animerun_contour_adapter.py` |
| [LinkTo-Anime](#linkto-anime) | ~29K frames | ~10 GB | Research | Planned: `linkto_adapter.py` |
| [FBAnimeHQ](#fbanimehq) | ~113K | ~25 GB | Unclear (Danbooru-sourced) | `fbanimehq_adapter.py` |
| [anime-segmentation](#anime-segmentation) | Varies | ~18 GB | Apache-2.0 | Not started |
| [AnimeInstanceSeg](#anime-instance-segmentation) | ~100K+ | Varies | Not specified | Not started |
| [CharacterGen](#charactergen) | ~13.7K chars | Varies | Apache-2.0 (code); research (data) | Not started |

**Total available:** ~465K+ images across all sources, ~355K already rendered and downloadable.

## Download

All datasets can be downloaded via the master script:

```bash
# Download everything
./ingest/download_datasets.sh all

# Download specific datasets
./ingest/download_datasets.sh nova_human stdgen

# See available datasets
./ingest/download_datasets.sh
```

Requires `git` for most datasets and `huggingface-cli` (`pip install huggingface_hub[cli]`) for HuggingFace-hosted data. Downloads are idempotent — already-downloaded datasets are skipped.

---

## NOVA-Human

**Paper:** "NOVA-3D: Non-overlapped Views for 3D Anime Character Reconstruction"
Wang, H., Yao, N., Zhou, X., Zhang, S., Xu, H., Wu, F., & Lin, F. (2024). arXiv:2405.12505. ACM MMAsia 2024 Workshops.

**Download:** https://github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D — clone repo, then follow README for dataset links (hosted on Alipan).

**License:** Code is MIT. Dataset contains renders of 10.2K VRoid Hub models — individual model licenses vary (check per-character). Research use is explicitly supported; commercial use depends on per-model VRoid Hub terms.

**Size:** ~50–80 GB. 10,200 anime characters × 20 views = ~204,000 images.

**Format:**
```
nova_human/{character_id}/
├── ortho/              # Front/back orthographic renders (PNG)
├── ortho_mask/         # Binary foreground masks (PNG)
├── ortho_xyza/         # Position + alpha maps
├── rgb/                # 16 random-view perspective renders (PNG)
├── rgb_mask/           # Foreground masks for perspective views (PNG)
├── xyza/               # Position + alpha for perspective views
└── {character_id}_meta.json
```

**Content:** Color renders (orthographic + perspective), binary foreground/background masks, position+alpha maps (XYZA), per-character metadata JSON.

**Strata adapter:** `ingest/nova_human_adapter.py` — converts ortho views (optionally RGB views) to Strata format. Resizes to 512×512. Masks stored as binary fg/bg (pixel 1 = foreground).

**Known limitations:**
- Does NOT provide Strata 19-region segmentation, joint positions, or draw order
- Adapter flags `missing_annotations: ["strata_segmentation", "joints", "draw_order"]`
- Useful as image+mask inputs; must supplement with extended rendering for full annotations

**Licensing risk:** Medium. Per-model VRoid Hub terms apply. Many models allow derivative works, but must verify per character. Use for research/validation is safe; commercial training requires per-model license audit. See PRD §12.

---

## StdGEN / Anime3D++

**Paper:** "StdGEN: Semantic-Decomposed 3D Character Generation from Single Images"
He, Y., Zhou, Y., Zhao, W., Wu, Z., Xiao, K., Yang, W., Liu, Y.-J., & Han, X. CVPR 2025.

**Download:** https://github.com/hyz317/StdGEN — code and rendering scripts. Weights and data via HuggingFace: `huggingface-cli download hyz317/StdGEN`.

**License:** Apache-2.0 (code). Cannot redistribute raw VRM models due to VRoid Hub policy. Dataset access requires following PAniC-3D project instructions to obtain source VRM files, then rendering with provided scripts.

**Size:** 10,811 annotated characters. Download size varies depending on components.

**Format:** Train/test split lists with Blender rendering scripts. Source data is VRM format 3D characters. Rendered outputs include multi-view images and semantic maps.

**Content:** Semantic decomposition into 4 classes (body, clothes, hair, face). Hair mask predictions. Alpha channel data. Multi-view rendered images.

**Strata adapter:** Planned (`stdgen_semantic_mapper.py`). Also planned: `stdgen_pipeline_ext.py` to extend StdGEN's Blender rendering with Strata outputs (45° angle, joints, draw order, measurements, contours).

**Known limitations:**
- Only 4 semantic classes — cannot distinguish upper_arm from forearm within "body"
- Must re-render from VRM source files using bone weights for full 19-region Strata labels
- StdGEN 4-class annotations still valuable for coarse training and as validation signal
- Source VRM files not directly redistributable

**Licensing risk:** Medium. Code is Apache-2.0, but underlying VRM models have per-model VRoid Hub terms. Same considerations as NOVA-Human.

---

## UniRig / Rig-XL

**Paper:** "One Model to Rig Them All: Diverse Skeleton Rigging with UniRig"
Zhang, J.-P., Pu, C.-F., Guo, M.-H., Cao, Y.-P., & Hu, S.-M. ACM Transactions on Graphics (SIGGRAPH 2025).

**Download:** https://github.com/VAST-AI-Research/UniRig — code, pre-trained checkpoint, dataset processing utilities. Rig-XL data hosted on HuggingFace with a `mapping.json` for asset lookup.

**License:** MIT.

**Size:** ~20 GB. 14,000+ rigged 3D models spanning diverse categories.

**Format:** Processed data stored in float16 with keys: vertices, vertex normals, faces, face normals, joint positions, skinning weights, parent indices, joint names, local bone matrices. Source files in .glb, .fbx, .obj, .vrm formats.

**Content:** Rigged 3D models from Objaverse and VRoid assets. Includes skeleton hierarchies, skinning weights, and joint positions — directly relevant to Strata's weight prediction and joint extraction tasks.

**Strata adapter:** Planned (`unirig_skeleton_mapper.py`). Will map UniRig skeleton data to Strata's 19-bone standard skeleton.

**Known limitations:**
- Models span diverse categories (not all humanoid) — filtering needed
- Not image data — these are 3D meshes that would need rendering for segmentation training
- Primarily valuable for skeleton/weight validation and mesh pipeline training

**Licensing risk:** Low. MIT license is permissive. However, underlying Objaverse/VRoid assets may have individual license terms.

---

## AnimeRun

**Paper:** "AnimeRun: 2D Animation Visual Correspondence from Open Source 3D Movies"
NeurIPS 2022.

**Download:** https://github.com/lisiyao21/AnimeRun — clone repo, then follow README for dataset links (hosted on Google Drive / Baidu).

**License:** Not specified in repository (noted as TODO). Likely research use.

**Size:** ~5 GB. Approximately 8,000 contour/color pairs.

**Format:**
```
animerun/
├── test/
│   ├── contour/
│   ├── flow_fwd/          # Forward optical flow
│   ├── flow_bwd/          # Backward optical flow
│   ├── anime/             # Anime frames
│   ├── line_area/
│   ├── segmat/            # Segment matching
│   └── unmatched/
└── train/
    └── (same structure)
```

**Content:** Anime frames with corresponding contour maps, forward/backward optical flow, line area information, and segment matching data. Derived from open-source 3D animated movies.

**Strata adapter:** Planned (`animerun_contour_adapter.py`). Will extract contour pairs for contour line detection training.

**Known limitations:**
- Repository appears incomplete ("Complete and clean code is on the way!")
- Download requires manual steps via Google Drive / Baidu links
- Movie-style animation, not illustration-style — different visual characteristics than typical Strata training data
- Contour data is the primary value; no body part segmentation

**Licensing risk:** Medium. No explicit license. Derived from open-source 3D movies — check source movie licenses for training use.

---

## LinkTo-Anime

**Paper:** "LinkTo-Anime: A 2D Animation Optical Flow Dataset from 3D Model Rendering"
Feng, X., Zou, K., Cen, C., Huang, T., Guo, H., Huang, Z., Zhao, Y., Zhang, M., Zheng, Z., Wang, D., Zou, Y., & Li, D. (2025). arXiv:2506.02733.

**Download:** See arXiv paper project page for download links. Fully manual — no git repo with automated download.

**License:** Research use. Check paper for specific terms.

**Size:** ~10 GB. 395 video sequences: 24,230 training frames, 720 validation frames, 4,320 test frames (~29K total).

**Format:** Video frames with paired annotations. Includes forward/backward optical flow, occlusion masks, and Mixamo skeleton data.

**Content:** Cel anime character motion generated from 3D model rendering. Characters animated using Mixamo skeletons and rendered from multiple viewpoints in two cel animation styles. Rich annotations: optical flow, occlusion masks, skeleton joint positions.

**Strata adapter:** Planned (`linkto_adapter.py`). Skeleton data uses Mixamo naming — should map cleanly to Strata's 19-region skeleton.

**Known limitations:**
- Manual download required (no automated script)
- Focused on motion/flow estimation, not static character segmentation
- Skeleton annotations use Mixamo convention (good for Strata compatibility)
- Relatively small compared to NOVA-Human/FBAnimeHQ

**Licensing risk:** Medium. Research dataset — check paper terms for commercial training use.

---

## FBAnimeHQ

**Source:** skytnt (HuggingFace community dataset).

**Download:** https://huggingface.co/datasets/skytnt/fbanimehq — direct download via `huggingface-cli download skytnt/fbanimehq`.

**License:** Unclear. Images collected from Danbooru (anime image board) using YOLOv5 for detection and cropping. Original images have varied artist copyrights. No explicit dataset license specified.

**Size:** ~25 GB. 112,806 full-body anime character images at 1024×512 resolution.

**Format:** High-resolution PNG/JPG images. Single directory of cropped full-body character images.

**Content:** Full-body anime girl images, cropped and aligned. No annotations — images only. High visual quality and style diversity.

**Strata adapter:** `ingest/fbanimehq_adapter.py` — discovers images recursively across the shard/bucket hierarchy, resizes longest edge to 512 preserving aspect ratio, centers on transparent 512×512 canvas, and generates metadata. Supports `--max_images` and `--random_sample` for phased runs. Run via `python run_ingest.py --adapter fbanimehq`.

**Known limitations:**
- No segmentation masks, joints, or any annotations — images only
- All images are female characters (bias concern for training)
- Danbooru-sourced — mixed art quality, some NSFW content may need filtering
- Resolution is 512×1024 (portrait, non-square) — adapter pads to 512×512

**Licensing risk:** High. Danbooru images are typically copyrighted by original artists. No clear license for ML training. Safest to use for validation/testing only, or as style reference — not as primary training data.

---

## anime-segmentation

**Source:** SkyTNT. GitHub: https://github.com/SkyTNT/anime-segmentation

**Download:** https://huggingface.co/datasets/skytnt/anime-segmentation — direct download via `huggingface-cli download skytnt/anime-segmentation`.

**License:** Apache-2.0 (code and model). Dataset combines AniSeg and Danbooru sources with manual cleaning.

**Size:** ~18 GB.

**Format:** Foreground images paired with binary segmentation masks. Dataset cleaned using DeepDanbooru, then manually verified.

**Content:** Anime character images with binary foreground/background masks. Pre-trained models available (ISNet, U2Net, MODNet, InSPyReNet architectures). Background images restored using Real-ESRGAN.

**Strata adapter:** Not started. Binary masks are fg/bg only — same granularity as NOVA-Human masks, not Strata 19-region segmentation. Could be useful for training a foreground extraction pre-processing step.

**Known limitations:**
- Binary masks only (foreground vs. background) — no body part segmentation
- Mixed Danbooru + AniSeg sources — licensing varies per image
- Primarily a model repo with training data, not a standalone dataset
- Mask quality depends on automated cleaning pipeline

**Licensing risk:** Medium. Code is Apache-2.0. Underlying image data from Danbooru/AniSeg has mixed licensing. Pre-trained model weights may be safe to use; training data for commercial models needs careful review.

---

## Anime Instance Segmentation

**Paper:** "Instance-guided Cartoon Editing with a Large-scale Dataset"
Lin, J., Li, C., Liu, X., & Ge, Z. (2023). arXiv:2312.01943. Published in The Visual Computer (2025).

**Download:** https://huggingface.co/datasets/dreMaz/AnimeInstanceSegmentationDataset — dataset. Pre-trained model: https://huggingface.co/dreMaz/AnimeInstanceSegmentation. Code: https://github.com/CartoonSegmentation/CartoonSegmentation.

**License:** Not explicitly specified. Research dataset.

**Size:** 100K+ paired high-resolution cartoon images with instance labeling masks.

**Format:** Cartoon images with per-instance segmentation masks. Two-stage model: low-resolution instance detection, then high-resolution mask refinement.

**Content:** Instance-level segmentation masks for characters in cartoon/anime images. Supports multiple characters per image with occlusion-aware instance separation. Enables cartoon editing applications (Ken Burns effect, style editing, puppet animation).

**Strata adapter:** Not started. Instance masks identify whole characters, not body parts — different task than Strata's 19-region semantic segmentation. Could be useful as a pre-processing step to isolate individual characters from multi-character scenes.

**Known limitations:**
- Instance segmentation (whole character), not semantic part segmentation
- Focused on cartoon editing, not body part decomposition
- Would need significant adapter work to extract Strata-compatible annotations
- Dataset quality varies (collected from diverse cartoon sources)

**Licensing risk:** Medium. No explicit license. Research dataset — assume research use only unless clarified.

---

## CharacterGen

**Paper:** "CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Canonicalization"
Peng, H.-Y., Zhang, J.-P., Guo, M.-H., Cao, Y.-P., & Hu, S.-M. ACM Transactions on Graphics (2024).

**Download:** https://github.com/zjp-shadow/CharacterGen — code and rendering scripts. Weights via HuggingFace: `huggingface-cli download zjpshadow/CharacterGen`.

**License:** Apache-2.0 (code). Cannot redistribute raw VRM models due to VRoid Hub policy. Users must obtain source data via PAniC-3D project instructions.

**Size:** ~13,746 VRoid characters. Download size varies.

**Format:** VRM/VRoid 3D models (glTF-based). Rendering scripts for Blender and three.js. Pre-trained model weights for 2D and 3D generation stages.

**Content:** 3D anime character generation pipeline. Source dataset of VRoid characters with standardized humanoid skeletons. Multi-view rendering capabilities. Related to StdGEN (same research group).

**Strata adapter:** Not started. VRM models have standardized humanoid skeletons that should map to Strata's 19 regions via bone weights. Rendering scripts could be extended for Strata-format output.

**Known limitations:**
- Raw VRM data not redistributable — must follow PAniC-3D instructions
- Primarily a generation model, not a pre-rendered dataset
- Requires Blender rendering to produce training images
- Overlaps significantly with StdGEN (same underlying VRoid sources)

**Licensing risk:** Medium. Code is Apache-2.0. Same VRoid Hub per-model licensing considerations as NOVA-Human and StdGEN. The PAniC-3D team navigated this for 14,500 models — follow their methodology.

---

## Licensing Risk Summary

Most pre-processed datasets are released for **research use only**. This is the highest-impact risk for commercial Strata training (PRD §12: Medium likelihood, High impact).

**Mitigation strategies:**
1. Train only on permissively-licensed subsets (MIT, Apache-2.0, CC-BY)
2. Use research-only datasets for validation/testing only, not training
3. Re-render from source VRM files under VRoid Hub's per-model terms
4. Audit per-model licenses for VRoid-sourced datasets (NOVA-Human, StdGEN, CharacterGen)

| Risk Level | Datasets |
|------------|----------|
| Low | UniRig (MIT) |
| Medium | StdGEN, CharacterGen (Apache code, per-model data), NOVA-Human, AnimeRun, LinkTo-Anime, anime-segmentation, AnimeInstanceSeg |
| High | FBAnimeHQ (Danbooru-sourced, no license) |

---

## Cross-Reference: Ingest Adapters

| Dataset | Adapter Script | Status |
|---------|---------------|--------|
| NOVA-Human | `ingest/nova_human_adapter.py` | Implemented |
| StdGEN | `ingest/stdgen_semantic_mapper.py` | Planned |
| StdGEN (extended) | `ingest/stdgen_pipeline_ext.py` | Planned |
| UniRig | `ingest/unirig_skeleton_mapper.py` | Planned |
| AnimeRun | `ingest/animerun_contour_adapter.py` | Planned |
| LinkTo-Anime | `ingest/linkto_adapter.py` | Planned |
| FBAnimeHQ | `ingest/fbanimehq_adapter.py` | Implemented |
| anime-segmentation | — | Not started |
| AnimeInstanceSeg | — | Not started |
| CharacterGen | — | Not started |

All downloads handled by `ingest/download_datasets.sh`.
