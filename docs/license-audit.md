# License Audit: Training Datasets

**Date:** March 7, 2026 (v2 — updated with detailed web research)
**Purpose:** Confirm commercial ML training use is permitted for all datasets in the Hetzner bucket. Strata is a commercial desktop app — models trained on these datasets ship in the product.

## Key Legal Question

> Does using CC-BY-NC data to train a commercial ML model constitute "commercial use"?

**Conservative view:** Yes. The trained model is a commercial product, so training is commercial use.
**Creative Commons official guidance (2024):** "If AI training data includes the NonCommercial restriction, following the NC restriction would require that all stages, from copying the data during training to sharing the trained model, must not be for commercial gain."
**Legal reality:** CC-BY-NC clearly prohibits commercial ML training per CC's own guidance.

**Recommendation:** Exclude all CC-BY-NC and research-only datasets from production training. Use them only for research/development.

---

## Audit Summary

| Dataset | License | Commercial Training | Risk | Action |
|---------|---------|:-------------------:|:----:|--------|
| CMU mocap | Custom permissive | **OK** | Low | Safe |
| 100STYLE | CC-BY 4.0 | **OK** | Low | Safe, attribute |
| VRoid Lite | CC0 | **OK** | Low | Safe |
| curated_diverse | ArtStation artwork | **PROHIBITED** | High | No AI training permission (NoAI opt-out ≠ consent) |
| InstaOrder | CC-BY-SA 4.0 | **OK** | Low | SA unlikely to apply to weights |
| anime-seg v1+v2 | Apache 2.0 / CC0 | **OK*** | Low-Med | Masks OK; source images from Danbooru |
| CoNR | MIT | **Likely OK** | Low-Med | Some hand-drawn sheets may be copyrighted |
| UniRig / Rig-XL | ODC-BY (Objaverse-XL) | **Ambiguous** | High | Sketchfab ToS prohibits AI training |
| FBAnimeHQ | CC0 (dataset) | **Ambiguous** | High | Source images are copyrighted Danbooru art |
| Mixamo renders | Adobe Additional Terms | **PROHIBITED** | High | Explicit AI/ML training ban |
| Live2D models | Proprietary | **PROHIBITED** | High | Restrictive terms, no ML permission |
| CartoonSegmentation | None (research only) | **PROHIBITED** | High | Bandai Namco IP, no license |
| AnimeRun | CC-BY-NC 4.0 | **PROHIBITED** | High | Non-commercial only |
| HumanRig | CC-BY-NC 4.0 | **PROHIBITED** | High | Non-commercial only |
| NOVA-Human | Per-model VRoid Hub | **PROHIBITED** | High | Pixiv prohibits AI training collection |

---

## Detailed Analysis

### SAFE: Clear Commercial Use Permitted

#### CMU Motion Capture — Custom permissive
- License: "You may include this data in commercially-sold products, but you may not resell this data directly."
- Training a model (not reselling data) falls within permitted use.
- Source: [mocap.cs.cmu.edu](https://mocap.cs.cmu.edu/)
- **Verdict:** Safe. Attribute CMU Graphics Lab.

#### 100STYLE — CC-BY 4.0
- CC-BY 4.0 explicitly permits commercial use with attribution.
- Source: [100STYLE on Zenodo](https://zenodo.org/records/8127870)
- **Verdict:** Safe. Attribute authors.

#### VRoid Lite — CC0
- CC0 is a full copyright waiver. No restrictions.
- Curated to include only CC0-marked VRoid Hub models.
- Note: Pixiv platform ToS may prohibit crawler-based collection, but CC0 intent is clear.
- **Verdict:** Safe.

#### InstaOrder — CC-BY-SA 4.0
- SA requires derivatives to use the same license.
- Model weights are widely considered NOT a derivative work of training data.
- Source: [InstaOrder GitHub](https://github.com/SNU-VGILab/InstaOrder)
- **Verdict:** Safe. Model weights are not SA-encumbered.

#### anime-segmentation (SkyTNT) — Apache 2.0 / CC0
- Code: Apache 2.0. Dataset on HuggingFace: CC0.
- Caveat: Source images likely from Danbooru (copyrighted fan art). Segmentation masks are SkyTNT's original work.
- We train on the images + masks. The masks are CC0, the images' copyright status is uncertain.
- Source: [GitHub (Apache 2.0)](https://github.com/SkyTNT/anime-segmentation), [HuggingFace (CC0)](https://huggingface.co/datasets/skytnt/anime-segmentation)
- **Verdict:** Low-medium risk. Masks are safe; source image provenance is uncertain but we train a discriminative model (not generative).

#### CoNR — MIT
- MIT is maximally permissive.
- Contains hand-drawn and synthesized character sheets. Synthesized are original; hand-drawn may include copyrighted anime IP.
- Source: [CoNR GitHub](https://github.com/megvii-research/IJCAI2023-CoNR)
- **Verdict:** Likely safe. MIT license is clear.

---

### AMBIGUOUS: High Risk

#### UniRig / Rig-XL — ODC-BY (Objaverse-XL)
- UniRig code: MIT. Rig-XL data: derived from Objaverse-XL.
- Objaverse-XL scraped models from Sketchfab (CC-BY 4.0), GitHub, Thingiverse, Polycam.
- **Problem:** Sketchfab's ToS explicitly prohibit using user-generated content for training generative AI. Many artists were not informed. Polycam data is restricted to non-commercial academic research.
- Source: [Objaverse-XL GitHub](https://github.com/allenai/objaverse-xl), [FlippedNormals ethics article](https://blog.flippednormals.com/objaverse-raises-concerns-about-ethics-of-scraping-3d-content/), [The Decoder: Sketchfab dispute](https://the-decoder.com/sketchfab-objaverse-ai-copyright-dispute-enters-third-dimension/)
- **Verdict:** HIGH RISK. Despite ODC-BY license on the collection, Sketchfab ToS and artist consent issues create significant legal and reputational risk.

#### FBAnimeHQ — CC0 (dataset), copyrighted (images)
- Dataset released as CC0 by SkyTNT on HuggingFace.
- Underlying images are from Danbooru — copyrighted fan art by various artists.
- CC0 label cannot override original artists' copyrights. The dataset creator cannot waive rights they don't hold.
- We train segmentation/joint models (not generative image models), which is more defensible.
- Source: [FBAnimeHQ on HuggingFace](https://huggingface.co/datasets/skytnt/fbanimehq), [Danbooru2021 (Gwern)](https://gwern.net/danbooru2021)
- **Verdict:** MEDIUM-HIGH RISK. Sole source for inpainting pairs and 101K joint examples.

---

### PROHIBITED: Cannot Use for Commercial Training

#### curated_diverse — ArtStation artwork
- 748 diverse character images sourced from ArtStation.
- ArtStation's ToS does not grant AI training rights. The NoAI tag is an opt-out mechanism; its absence does not constitute consent for AI training.
- All artwork is copyrighted by default. No explicit license or permission was obtained from the artists.
- **Impact:** 748 examples with fg/bg masks + draw_order. Used in segmentation + joints training.
- **Action:** MUST EXCLUDE. Removed from all training configs.

#### Mixamo — Adobe Additional Terms (CRITICAL)
- Adobe's Mixamo Additional Terms (June 23, 2021) contain an explicit **"Restriction on AI/ML"** section:
  > *"You will not, and will not instruct or allow third parties to, use the Services or Software (or any content, data, output, or other information received or derived from the Services or Software) to directly or indirectly create, train, test, or otherwise improve any machine learning algorithms or artificial intelligence systems, including, but not limited to, any architectures, models, or weights."*
- This covers rendered images, segmentation masks, joint positions — everything derived from Mixamo characters.
- Source: [Mixamo Additional Terms PDF](https://wwwimages2.adobe.com/content/dam/cc/en/legal/servicetou/Mixamo-Addl-Terms-en_US-20210623.pdf), [Adobe Community discussion](https://community.adobe.com/t5/mixamo-discussions/mixamo-license-for-machine-learning-datasets/td-p/14734915)
- **Impact:** 1,598 segmentation examples (`segmentation/` dataset) + 105 weight examples. This is the foundation dataset.
- **Action:** MUST EXCLUDE from production training. Replace with VRoid Lite renders, HumanRig (if NC permission granted), or new renders from non-Mixamo characters.

#### Live2D — Proprietary Terms
- Live2D Free Material License Agreement restricts usage to defined purposes.
- Sample models: "may not be used for any purposes other than Internal evaluation of the Software and Training" (user training, not ML training).
- No explicit ML training permission exists in any Live2D license tier.
- Source: [Live2D Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html), [Live2D Sample Model Terms](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)
- **Impact:** 844 examples (`live2d/` dataset). Good 22-class segmentation data.
- **Action:** MUST EXCLUDE from production training unless Live2D grants explicit ML permission.

#### CartoonSegmentation — No License (Research Only)
- No formal open-source license in the repository.
- Project page states: "Copyright BANDAI NAMCO Entertainment Inc., We believe this is a fair use for research and educational purpose only."
- Contains frames from Bandai Namco IP (anime shows).
- Source: [CartoonSegmentation GitHub](https://github.com/CartoonSegmentation/CartoonSegmentation)
- **Impact:** `anime_instance_seg/` (~135K files in bucket). NOT used in lean training configs.
- **Action:** MUST EXCLUDE. Already not in lean configs, so no immediate impact.

#### AnimeRun — CC-BY-NC 4.0
- CC-BY-NC explicitly prohibits commercial use. CC's own guidance confirms this applies to ML training.
- Source: [AnimeRun Project Page](https://lisiyao21.github.io/projects/AnimeRun), [CC on AI Training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
- **Impact:** `animerun*/` prefixes (~15 GiB). Used for optical flow/correspondence, NOT core model training.
- **Action:** Exclude from production. Already not used for core models.

#### HumanRig — CC-BY-NC 4.0
- Same CC-BY-NC prohibition as AnimeRun.
- Source: [HumanRig Project Page](https://c8241998.github.io/HumanRig/), [HumanRig GitHub](https://github.com/c8241998/HumanRig)
- **Impact:** 11,434 examples. Major dataset for weights + segmentation.
- **Action:** Contact authors for commercial training permission. If denied, train weights on UniRig+Mixamo only (but Mixamo is also prohibited — see above).

#### NOVA-Human — VRoid Hub (Research Only)
- Derived from 10.2K VRoid Hub models with heterogeneous per-model licenses.
- Pixiv's September 2024 guidelines update prohibits AI training data collection from VRoid Hub.
- Source: [NOVA-3D GitHub](https://github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D), [VRoid Hub Guidelines Update](https://vroid.pixiv.help/hc/en-us/articles/37342372606873)
- **Impact:** ~40K images with RTMPose joints.
- **Action:** Exclude from production. Joint model still has 101K FBAnimeHQ examples.

---

## Production Training Set (Conservative)

Excluding ALL prohibited + ambiguous datasets:

| Model | Safe Datasets | Examples | Viable? |
|-------|--------------|----------|---------|
| **Segmentation** | anime_seg, InstaOrder | ~14K (fg/bg only, no 22-class!) | **CRITICAL GAP** — no 22-class mask source |
| **Joints** | FBAnimeHQ*, anime_seg | ~116K | Yes, but FBAnimeHQ is ambiguous |
| **Weights** | UniRig* | ~15K | Yes, but UniRig is ambiguous (Sketchfab) |
| **Inpainting** | FBAnimeHQ* (pairs) | ~45K pairs | Yes, but FBAnimeHQ is ambiguous |

*\* Ambiguous datasets included to show non-zero training set. Without them, joints and inpainting have zero safe sources.*

### The Mixamo Problem

Mixamo's AI/ML ban is the most damaging finding. Our `segmentation/` dataset (1,598 examples) is the **only source of high-quality 22-class body region masks**. Without it:
- No 22-class segmentation training data from safe sources
- The entire segmentation model depends on data we cannot legally use

### Paths Forward

1. **Generate new safe training data:** Render characters from CC0/CC-BY sources (VRoid Lite's 4,651 CC0 characters, Objaverse CC0 meshes) through our Blender pipeline. This replaces Mixamo as the primary segmentation data source.

2. **Seek explicit permissions:**
   - Contact HumanRig authors for commercial training permission
   - Contact Live2D Inc. for ML training permission
   - Consider Adobe's Mixamo team (unlikely to grant exception)

3. **See-Through dataset (late March 2026):** 9,102 Live2D models with 19-class segmentation — if the license permits commercial training, this solves the 22-class gap.

4. **Accept FBAnimeHQ risk:** For discriminative models (segmentation, joints), training on copyrighted images is more defensible than for generative models. The model cannot reproduce the training images. Many commercial ML products train on similar data.

---

## Action Items

1. [ ] **URGENT: Render VRoid Lite characters** — Generate 22-class seg masks from CC0 VRoid characters to replace Mixamo data
2. [ ] **Contact HumanRig authors** — Request commercial training permission (CC-BY-NC → explicit grant)
3. [ ] **Contact Live2D Inc.** — Request ML training permission for rendered outputs
4. [ ] **Legal counsel on FBAnimeHQ** — Discriminative model training on Danbooru-derived data
5. [ ] **Legal counsel on UniRig/Objaverse-XL** — Sketchfab ToS vs ODC-BY license
6. [ ] **See-Through license check** — When released, verify license before ingesting
7. [ ] **Create production training configs** — Configs that exclude all prohibited datasets
8. [ ] **Create `ATTRIBUTIONS.md`** — For all CC-BY/CC-BY-SA datasets used in production
