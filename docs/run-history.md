# Segmentation & Texture Run History

Detailed historical record of training experiments. Active work is in `CLAUDE.md`.
Use this when revisiting an old experiment, debugging a recurring failure mode, or
planning future ML work.

---

## Segmentation runs

| Run | mIoU | Key change | Verdict |
|-----|------|-----------|---------|
| 8 | 0.4721 | Bootstrapping loop validated | first big jump |
| 13a | 0.5425 | +toon_pseudo | data lever working |
| 16 | 0.5808 | Frozen val deployed | comparable baselines from here on |
| 18 | 0.5750 | True baseline with frozen val | reference |
| **20** | **0.6485** | Boundary softening (radius=2) | **canonical baseline + shipped checkpoint** |
| 21 | 0.6361 | Re-pseudo-label, no softening | softening confirmed as biggest lever |
| 22/22b | 0.56-0.59 | SAM labels — don't help | clothing-vs-anatomy mismatch |
| SAM-HQ (Apr 15) | ~0.29 (killed step 5500) | Li's encoder frozen + fresh 22-class decoder | plateaued well below baseline |
| 23 (Apr 21) | **0.3819 killed @ e13** | ResNet-50 + Pascal-Person-Part pretrain | adjacent-domain priors don't transfer |
| 24 (Apr 21) | broken | +gemini_diverse, softening OFF (bug), meshy tar name wrong | pipeline bugs |
| 25 (Apr 21) | ~0.60-0.61 (killed) | Same data, softening back on, lower LR on resume | confounded experiment |
| 26 (Apr 21) | 0.5471 @ e19 | See-Through SAM + naive heuristic converter | worse than Run 20 |
| 27 (Apr 21) | 0.5664 @ e19 | See-Through SAM → Dr. Li converter + joints | still worse |
| 28 (Apr 22) | 0.6030 val / ~0.634 test | Run 20 hyperparams + gemini_diverse wt 2.0 | proved Run 25 confound; didn't beat Run 20 |
| 29 (Apr 22) | **0.6095 val / ~0.640 test** | Run 28 + label cleaning + full sora corpus | +0.007 val over Run 28; still under Run 20 |
| 30 (Apr 23) | 0.5876 | Ensemble pseudo-labels (Run 20 TTA + SAM 2.1 + joints) | label-quality lever didn't help |
| 31 (Apr 30) | 0.4168 @ e5 (killed) | DINOv2-base backbone | overfitting started early; trajectory ~0.50 by e20 |

**Conclusion (April 30, 2026):** Run 20 (0.6485) is the seg ceiling. Architecture and data-expansion levers both exhausted. Ship Run 20.

---

## April 15 SAM-HQ Seg Run — Detailed Trajectory

Ran Dr. Li's V3 training pipeline with her SAM-HQ ViT-B encoder frozen, fresh 22-class decoder, 8000 steps.

| Step | val/mIoU |
|------|----------|
| 1000 | 0.198 |
| 2000 | 0.223 |
| 3000 | 0.239 |
| 4000 | 0.269 |
| 5000 | 0.288 |

Killed at step ~5500. Linear extrapolation: ~0.35 at step 8000.

**Lessons:**
- Her encoder was pretrained for clothing/style boundaries, not anatomy — would have needed unfreezing to adapt.
- Decoder from scratch at 8K steps is cold-start; needs 20K+ for meaningful learning.
- Multiple bugs in upstream V3 code required patches (gradient_checkpointing, checkpoint_total_limit, dataset_seg exceptions, resume_from_checkpoint unimplemented).

---

## April 20-21 Experiments — Pseudo-label + Architectural Failures

Five runs, no wins. Total cost ~$25-30.

| Run | Experiment | Val mIoU | Root cause |
|-----|------------|----------|------------|
| 24 | +gemini_diverse wt 3.5, softening OFF (bug), meshy tar missing (bug) | broken | pipeline bugs |
| 25 | +gemini_diverse wt 2.0, softening ON, **LR 5e-6 (half of Run 20), 12 epochs (60% of Run 20)** | 0.5965 @ e19 (killed) | confounded — hyperparams changed alongside data |
| 26 | See-Through SAM + naive L/R+vertical heuristic converter | 0.5471 @ e19 | naive converter drops hair_back, bad splits |
| 27 | See-Through SAM + Dr. Li converter + joints | 0.5664 @ e19 (killed) | inherited Run 25's wrong hyperparams; mislocalized labels |
| 23 | ResNet-50 + Pascal-Person-Part pretrain (architectural) | 0.3819 @ e13 (killed) | adjacent-domain priors don't transfer; severe overfitting |

**Conclusions:**
- **Confounded experiments are worse than no experiment.** Run 25 changed LR + epochs + data simultaneously. All later runs inherited its bad hyperparams.
- **Adjacent-domain pretraining doesn't transfer to illustrated chars.** Confirmed three times: SAM-HQ encoder (Apr 15, plateaued 0.29), ResNet-50 + Pascal-Person-Part (Apr 21, plateaued 0.38), DINOv2-base (Apr 30, plateaued 0.42).
- **Quality filter checks shape, not pixel correctness.** See-Through+Li labels passed quality filter at 4× the rate of Run 20 self-distill but trained worse. Noisy-but-localized > well-formed-but-mislocalized.
- **Boundary softening (radius=2) remains the single biggest lever** (+8.8% mIoU over Run 19). Any run that disabled it lost ~0.05 mIoU.

---

## April 21 Pipeline Hygiene Rules

Lessons accumulated from $25-30 of debugging on the Apr 20-21 runs. Apply to any future training work.

1. **One variable per experiment.** If changing data, keep hyperparams. If changing architecture, keep data and hyperparams.
2. **Reproduction check before any new experiment.** Shortest possible run (~1 hr) reproducing a known baseline on current code/data/deps. If that doesn't land, debug before spending more.
3. **Pre-flight tar integrity.** A silent tar-not-found (e.g., Run 24's `meshy_cc0_restructured.tar` vs actual `meshy_cc0_textured_restructured.tar`) costs a full run. Verify sizes and names.
4. **Ctrl+C, never Ctrl+Z.** Suspended pipe-with-tee jobs leave zombie CUDA contexts that take 30 min to untangle on a live A100.
5. **Archive `.pt` with `.onnx` every time.** Joints `.pt` was missing from bucket for months. Recovered from HD April 21 and uploaded to `checkpoints_joints/best.pt`.
6. **Never `pkill -f python3`** on cloud instances — kills system processes and crashes the instance. Use `pkill -f train_segmentation` or specific module names.
7. **Tar names matter.** Pre-flight should verify tar download succeeded before extracting.
8. **`ingest_gemini.py` should run joints inference by default** when given a joints checkpoint, so new gemini_diverse never ships without `joints.json`. Currently it only does seg pseudo-labels.

---

## Texture Inpainting (deferred to v2)

| Run | val/l1 | Key change |
|-----|--------|-----------|
| 1 | 0.1509 | 500 pairs, baseline |
| 2 | 0.1520 | 2,891 pairs |
| 3 | 0.1497 | 2,891 pairs, 100 epochs |
| **4 (v3)** | **0.1282** | 1,244 pairs **with real geometry maps**, fine-tuned from v3. SSIM 0.418. Geometry maps are the biggest single lever. |

**Architecture:** ControlNet on SD 1.5 Inpainting, 9-channel input (noisy latent + mask + partial).

**Pipeline experiment (lichtung cat, April 14):**
- SAM 3D Objects mesh + 1 watercolor cat illustration → silhouette-IoU search found best camera angle (72°, IoU 0.77).
- Hero view projects pixel-perfect for ~30% of UV. Rest needs inpainting.
- Multi-view (front/back) approach failed — Gemini-generated views don't match mesh from "true" front/back angles.
- TPS landmark-based warping helps but needs 20+ landmarks per view to be useful.
- Single-view + strong inpainting is the best UX.
- **v3 ControlNet failed on illustrated style:** Meshy-trained model produced dark solid fill, not watercolor. Empty-prompt training means text prompts + CFG have no effect.

**Notes:**
- Manual denoising loop needed for validation (diffusers pipeline incompatible with 9-ch ControlNet).
- Use `torch.amp.autocast` for mixed fp16/fp32 validation.

**Next attempt (when v2 starts):** test StyleTex (SIGGRAPH 2024, Apache 2.0) — takes illustration as style reference, decouples style from content via CLIP manipulation. Run script ready: `training/run_styletex_test.sh`.

---

## 3D Reconstruction (deferred to v2)

- U-Net view synthesis deprecated (blurry).
- SHARP abandoned (research-only license).
- **SAM 3D Objects = primary 3D mesh source.** Validated.
- TRELLIS.2 (MIT) = backup if SAM 3D quality plateaus.
- Texture from artist's illustration projected onto mesh. AI only fills gaps.
- Single-view "hero" pipeline: 1 illustration → SAM 3D mesh → auto-detect camera angle → project illustration → ControlNet inpaint gaps.

---

## Bucket Inventory

**Tars in `tars/` prefix:** humanrig (16.8G), humanrig_posed (12G), meshy_cc0_textured_restructured (2.8G), flux_diverse_clean (300M), sora_diverse (380M), gemini_li_converted (223M), vroid_cc0 (203M), cvat_annotated (9M), soft_targets_precomputed (1.9G), demo_back_view_pairs (3.4G).

**Texture inpainting tars in root:** texture_pairs_front (4.9G, with geometry maps), texture_pairs_side (2.8G), texture_pairs_back (2.8G), texture_pairs_100avatars (113M). Total ~4,135 pairs.

**Frozen splits:** `data_cloud/frozen_val_test.json`. All training runs must use this file.

**Shipped models:**
- `models/onnx_run20_seg/segmentation_run20.onnx` (67.8 MB, 0.6485 mIoU) — bundled in Strata as of April 30, 2026.
- `checkpoints_run20_seg/segmentation/run20_best.pt` (203 MB) — source `.pt` for the export.

---

## Key Learnings (Segmentation)

These are the load-bearing facts that shape future ML work. Read these before proposing a new run.

- **Boundary label softening = biggest lever** (+8.8% mIoU, Run 20). Precompute as `.npz` files. Auto-excludes neck (class 2) and hair_back (class 21) via `SOFTENING_EXCLUDE_CLASSES`.
- **Clothing-based labels don't help** (SAM, Dr. Li's 19-class). Anatomy ≠ clothing.
- **SAM 3D Body labels don't help.** Body mesh ≠ clothing silhouette.
- **Adjacent-domain pretrained backbones don't help.** Confirmed three times: SAM-HQ, Pascal-Person-Part ResNet-50, DINOv2-base. Real-photo or anime-only priors don't bridge the stylized-character distribution gap.
- **Pseudo-label data expansion is exhausted.** Six experiments (Runs 24-30) all underperformed Run 20. Quality filter checks shape, not pixel correctness.
- **Bootstrapping works only when teacher matches student domain.** GT-on-rendered → bootstrap rendered = ✓. Run-20-on-rendered → bootstrap illustrated = ✗.
- **Class 20 (accessory) remapped to background** — unused by rigging pipeline.
- **Label cleaning is a small but real lever.** Filter flags `--drop-head-below-torso --max-bg-bleed 0.10 --min-silhouette-coverage 0.50` lifted Run 29 by +0.007 over Run 28. Ported into `ingest_gemini.py` for clean ingestion.
- **`gemini_li_converted` is real GT, not pseudo-labels.** Dr. Li's 694 hand-labeled examples through `convert_li_labels.py` with joints. Had weight 3.0 in Run 20 for a reason.
- **A100 is 40GB.** Batch 16 for soft targets. Use frozen val/test splits.

---

## Untried levers (may revisit post-funding)

If we ever return to seg training with budget:

- **CVAT hand-labels** — 300-500 illustrated chars annotated by hand, convert via `convert_li_labels.py` (how cvat_annotated wt 10.0 and gemini_li_converted wt 3.0 were built). ~2 days human work, real GT, +0.02-0.04 expected.
- **Per-class boundary softening** — radius=1 for thin classes, radius=3 for large. +0.01-0.02 expected.
- **Different/smaller ViT backbones** — ViT-S, Swin, etc. Lower expected gain than the failed attempts above.

The advisor's plan deprioritizes all of these until v1 ships and users tell us seg quality is actually blocking.
