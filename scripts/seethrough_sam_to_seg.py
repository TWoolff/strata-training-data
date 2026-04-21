#!/usr/bin/env python3
"""Pseudo-label illustrated characters with See-Through SAM body parsing model.

Runs Dr. Chengze Li's See-Through SAM-HQ body-parsing model (19 clothing
classes, Apache-2.0, HuggingFace ``24yearsold/l2d_sam_iter2``) on a directory
of Strata-format example dirs and converts the 19-class output to our 22-class
anatomy schema. Writes ``segmentation.png`` in each example dir.

Why this exists
---------------
The original Run A plan was to run See-Through's full LayerDiff pipeline to
produce layer PNGs, then convert via ``scripts/convert_seethrough_to_seg.py``.
LayerDiff is diffusion-based (~60-90s/image) and does not fit the A100 budget
for 3K+ images. The SAM body-parsing model is ~1s/image and outputs 19 semantic
classes directly — this script maps those to our 22 anatomy classes using the
same split heuristics as ``convert_seethrough_to_seg.py``'s unsplit-layer
fallbacks.

Usage
-----
::

    export SEETHROUGH_ROOT=/workspace/see-through
    python3 scripts/seethrough_sam_to_seg.py \\
        --input-dir /workspace/data_cloud/gemini_diverse \\
        --checkpoint /workspace/weights/li_sam_iter2.pt \\
        --device cuda

Input layout: ``<input-dir>/<char>/image.png`` (+ optional fg_mask.png for alpha).
Output: writes ``<char>/segmentation.png`` (uint8, 0-21) and updates
``metadata.json`` with ``segmentation_source: "seethrough_sam_v2"``.

19 → 22 class mapping
---------------------
SAM outputs per-pixel argmax over:
    0=hair, 1=headwear, 2=face, 3=eyes, 4=eyewear, 5=ears, 6=earwear,
    7=nose, 8=mouth, 9=neck, 10=neckwear, 11=topwear, 12=handwear,
    13=bottomwear, 14=legwear, 15=footwear, 16=tail, 17=wings, 18=objects

Our 22 anatomy classes:
    0=bg, 1=head, 2=neck, 3=chest, 4=spine, 5=hips,
    6-9=shoulder/upper_arm/forearm/hand_l, 10-13=same_r,
    14-16=upper_leg/lower_leg/foot_l, 17-19=same_r,
    20=accessory, 21=hair_back

Mapping:
- hair, face, eyes, eyewear, ears, earwear, nose, mouth → head (1)
  (See-Through doesn't distinguish front/back hair; all hair → head)
- headwear, neckwear, tail, wings, objects → accessory (20)
- neck → neck (2)
- topwear → split vertically → chest (3) / spine (4)
- bottomwear → hips (5)
- handwear → split L/R by center of mass → each side split into 3
  vertical bands (40%/35%/25%) → upper_arm, forearm, hand
- legwear → split L/R → each side split 50/50 → upper_leg, lower_leg
- footwear → split L/R → foot_l, foot_r

Shoulder classes (6, 10) are not generated explicitly — upper_arm spans the
region a shoulder would occupy. Quality filter's min-regions=4 is safely met.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 22-class anatomy IDs
# -------------------------------------------------------------------------
BG, HEAD, NECK, CHEST, SPINE, HIPS = 0, 1, 2, 3, 4, 5
SHOULDER_L, UPPER_ARM_L, FOREARM_L, HAND_L = 6, 7, 8, 9
SHOULDER_R, UPPER_ARM_R, FOREARM_R, HAND_R = 10, 11, 12, 13
UPPER_LEG_L, LOWER_LEG_L, FOOT_L = 14, 15, 16
UPPER_LEG_R, LOWER_LEG_R, FOOT_R = 17, 18, 19
ACCESSORY, HAIR_BACK = 20, 21

REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]

# See-Through SAM output channels (VALID_BODY_PARTS_V2)
SEETHROUGH_IDX = {
    "hair": 0, "headwear": 1, "face": 2, "eyes": 3, "eyewear": 4,
    "ears": 5, "earwear": 6, "nose": 7, "mouth": 8,
    "neck": 9, "neckwear": 10, "topwear": 11, "handwear": 12,
    "bottomwear": 13, "legwear": 14, "footwear": 15,
    "tail": 16, "wings": 17, "objects": 18,
}


# -------------------------------------------------------------------------
# Mask splitting helpers
# -------------------------------------------------------------------------
def _split_lr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a binary mask into left and right halves by horizontal center-of-mass."""
    if not mask.any():
        return mask.copy(), mask.copy()
    ys, xs = np.where(mask)
    cx = int(xs.mean())
    left = mask.copy()
    right = mask.copy()
    left[:, cx:] = False
    right[:, :cx] = False
    return left, right


def _split_arm(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a single arm mask top-down: upper_arm (40%) / forearm (35%) / hand (25%)."""
    if not mask.any():
        return mask.copy(), mask.copy(), mask.copy()
    ys, _ = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    h = max(y_max - y_min, 1)
    t_u = y_min + int(h * 0.40)
    t_f = y_min + int(h * 0.75)
    upper = mask.copy(); upper[t_u:, :] = False
    fore = mask.copy(); fore[:t_u, :] = False; fore[t_f:, :] = False
    hand = mask.copy(); hand[:t_f, :] = False
    return upper, fore, hand


def _split_leg(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a single leg mask top-down: upper_leg (50%) / lower_leg (50%)."""
    if not mask.any():
        return mask.copy(), mask.copy()
    ys, _ = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    h = max(y_max - y_min, 1)
    t_mid = y_min + h // 2
    upper = mask.copy(); upper[t_mid:, :] = False
    lower = mask.copy(); lower[:t_mid, :] = False
    return upper, lower


def _split_topwear(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split torso top 50% → chest, bottom 50% → spine."""
    if not mask.any():
        return mask.copy(), mask.copy()
    ys, _ = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    h = max(y_max - y_min, 1)
    t_mid = y_min + h // 2
    chest = mask.copy(); chest[t_mid:, :] = False
    spine = mask.copy(); spine[:t_mid, :] = False
    return chest, spine


# -------------------------------------------------------------------------
# 19-class See-Through output → 22-class anatomy seg
# -------------------------------------------------------------------------
def convert_seethrough_sam_to_22class(
    class_masks: np.ndarray, alpha: np.ndarray, threshold: float = 0.0,
) -> np.ndarray:
    """Convert 19-channel See-Through SAM logits to a single 22-class mask.

    Args:
        class_masks: ``[19, H, W]`` float (logits) or bool (thresholded).
        alpha: ``[H, W]`` uint8 — non-zero where character pixels live.
        threshold: logit threshold for binarization (ignored if class_masks is bool).

    Returns:
        ``[H, W]`` uint8 segmentation with IDs 0-21.
    """
    if class_masks.dtype != np.bool_:
        binary = class_masks > threshold
    else:
        binary = class_masks
    H, W = binary.shape[1:]
    seg = np.zeros((H, W), dtype=np.uint8)
    fg = alpha > 10

    def write(mask: np.ndarray, class_id: int) -> None:
        """Paint ``class_id`` where mask is True, seg is still 0, and pixel is foreground."""
        if not mask.any():
            return
        idx = mask & (seg == 0) & fg
        seg[idx] = class_id

    # Depth order: head components first (highest), then accessories, clothing, body.
    # Inside each tier, prioritize smaller/more specific regions.

    # Head (face/eyes/nose/mouth/ears paint over hair in overlap regions)
    for tag in ("face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth", "hair"):
        write(binary[SEETHROUGH_IDX[tag]], HEAD)

    # Neck
    write(binary[SEETHROUGH_IDX["neck"]], NECK)

    # Accessories (headwear, neckwear, tail, wings, objects)
    for tag in ("headwear", "neckwear", "tail", "wings", "objects"):
        write(binary[SEETHROUGH_IDX[tag]], ACCESSORY)

    # Topwear → chest + spine (vertical split)
    chest, spine = _split_topwear(binary[SEETHROUGH_IDX["topwear"]])
    write(chest, CHEST)
    write(spine, SPINE)

    # Bottomwear → hips
    write(binary[SEETHROUGH_IDX["bottomwear"]], HIPS)

    # Handwear → L/R + upper_arm/forearm/hand per side
    hw = binary[SEETHROUGH_IDX["handwear"]]
    hw_l, hw_r = _split_lr(hw)
    ul, fl, hl = _split_arm(hw_l)
    ur, fr, hr = _split_arm(hw_r)
    write(ul, UPPER_ARM_L); write(fl, FOREARM_L); write(hl, HAND_L)
    write(ur, UPPER_ARM_R); write(fr, FOREARM_R); write(hr, HAND_R)

    # Legwear → L/R + upper_leg/lower_leg per side
    lw = binary[SEETHROUGH_IDX["legwear"]]
    lw_l, lw_r = _split_lr(lw)
    ul_l, ll_l = _split_leg(lw_l)
    ul_r, ll_r = _split_leg(lw_r)
    write(ul_l, UPPER_LEG_L); write(ll_l, LOWER_LEG_L)
    write(ul_r, UPPER_LEG_R); write(ll_r, LOWER_LEG_R)

    # Footwear → L/R
    fw = binary[SEETHROUGH_IDX["footwear"]]
    fw_l, fw_r = _split_lr(fw)
    write(fw_l, FOOT_L)
    write(fw_r, FOOT_R)

    return seg


# -------------------------------------------------------------------------
# Model setup — inlined so this script can run standalone with see-through on PYTHONPATH
# -------------------------------------------------------------------------
def _build_seethrough_model(checkpoint_path: Path, device: str = "cuda"):
    """Load See-Through SAM body-parsing model.

    Expects ``see-through`` repo cloned and ``see-through/common`` on ``sys.path``.
    See README for setup: ``git clone https://github.com/shitagaki-lab/see-through``.
    """
    seethrough_root = Path(os.environ.get("SEETHROUGH_ROOT", "/workspace/see-through"))
    if not seethrough_root.exists():
        raise FileNotFoundError(
            f"See-Through repo not found at {seethrough_root}. "
            "Set SEETHROUGH_ROOT or clone: "
            "git clone https://github.com/shitagaki-lab/see-through /workspace/see-through",
        )
    common = str(seethrough_root / "common")
    if common not in sys.path:
        sys.path.insert(0, common)

    from utils.torch_utils import init_model_from_pretrained  # type: ignore
    from modules.semanticsam import SemanticSam  # type: ignore

    logger.info("Loading See-Through SAM from %s ...", checkpoint_path)
    model = init_model_from_pretrained(
        pretrained_model_name_or_path=str(checkpoint_path),
        module_cls=SemanticSam,
        download_from_hf=False,
        model_args=dict(
            class_num=19,
            model_type="b_hq",  # ViT-B HQ — fits 40GB A100 with batch 1 at 1024
            fix_img_en=True,
            fix_prompt_en=True,
            fix_mask_de=False,
        ),
    )
    model = model.to(device=device)
    model.eval()
    return model


def _maybe_download_checkpoint(local_path: Path) -> Path:
    """Download Li's checkpoint-18000.pt from HF if not present locally."""
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    logger.info("Downloading 24yearsold/l2d_sam_iter2/checkpoint-18000.pt from HuggingFace ...")
    path = hf_hub_download(
        repo_id="24yearsold/l2d_sam_iter2",
        filename="checkpoint-18000.pt",
    )
    import shutil
    shutil.copy(path, local_path)
    return local_path


# -------------------------------------------------------------------------
# Main ingestion loop
# -------------------------------------------------------------------------
def process_example(
    example_dir: Path, model, device: str,
) -> tuple[bool, int, float]:
    """Run See-Through SAM on one example and write segmentation.png.

    Returns (success, num_regions, mean_confidence).
    """
    import torch  # local import so the converter unit-tests don't need torch

    image_path = example_dir / "image.png"
    if not image_path.exists():
        return False, 0, 0.0

    img_rgba = Image.open(image_path).convert("RGBA")
    img_np = np.array(img_rgba)
    alpha = img_np[:, :, 3]
    # See-Through expects RGB (composited onto white or black — we use black to preserve edges)
    rgb = img_np[:, :, :3].copy()
    rgb[alpha < 10] = 0

    with torch.inference_mode():
        preds = model.inference(rgb)[0]  # [19, H, W] logits (torch tensor)
        class_masks = (preds > 0).to(device="cpu", dtype=torch.bool).numpy()
        # Confidence per pixel = max logit across classes (sigmoid would be nicer but this is fast)
        max_logit = preds.max(dim=0).values.to(device="cpu", dtype=torch.float32).numpy()
        # Squash to [0,1] for reporting
        confidence = 1.0 / (1.0 + np.exp(-max_logit))

    seg = convert_seethrough_sam_to_22class(class_masks, alpha)

    # Zero out non-foreground pixels (defensive)
    fg_mask = alpha >= 10
    seg[~fg_mask] = 0

    # Mean confidence over foreground pixels
    mean_conf = float(confidence[fg_mask].mean()) if fg_mask.any() else 0.0

    unique_regions = sorted(int(v) for v in np.unique(seg) if v > 0)
    region_names = [REGION_NAMES[r] for r in unique_regions]

    Image.fromarray(seg).save(example_dir / "segmentation.png")

    # Save confidence map for downstream review
    conf_uint8 = (confidence * 255).clip(0, 255).astype(np.uint8)
    conf_uint8[~fg_mask] = 0
    Image.fromarray(conf_uint8).save(example_dir / "confidence.png")

    # Update metadata
    meta_path = example_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"id": example_dir.name}
    meta["has_segmentation_mask"] = True
    meta["segmentation_source"] = "seethrough_sam_v2"
    meta["seg_mean_confidence"] = round(mean_conf, 4)
    meta["seg_num_regions"] = len(unique_regions)
    meta["seg_regions"] = region_names
    if "missing_annotations" in meta:
        meta["missing_annotations"] = [
            a for a in meta["missing_annotations"] if a != "strata_segmentation"
        ]
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True, len(unique_regions), mean_conf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pseudo-label with See-Through SAM body parsing model.",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Dir of Strata-format examples: <input>/<char>/image.png")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/workspace/weights/li_sam_iter2.pt"),
                        help="Path to checkpoint-18000.pt. Auto-downloads from HF if missing.")
    parser.add_argument("--device", default="cuda",
                        help="torch device (cuda / cpu / mps)")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip examples whose segmentation_source is already seethrough_sam_v2")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N examples (debug)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = args.input_dir
    if not input_dir.is_dir():
        logger.error("input-dir not found: %s", input_dir)
        return 1

    ckpt = _maybe_download_checkpoint(args.checkpoint)
    model = _build_seethrough_model(ckpt, device=args.device)

    examples = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "image.png").exists()
    )
    logger.info("Found %d examples in %s", len(examples), input_dir)

    skipped = 0
    if args.only_missing:
        pre = len(examples)
        filtered = []
        for d in examples:
            meta = d / "metadata.json"
            if meta.exists():
                try:
                    m = json.loads(meta.read_text(encoding="utf-8"))
                    if m.get("segmentation_source") == "seethrough_sam_v2":
                        skipped += 1
                        continue
                except Exception:
                    pass
            filtered.append(d)
        examples = filtered
        logger.info("--only-missing: %d to process, %d skipped", len(examples), skipped)

    if args.limit is not None:
        examples = examples[: args.limit]

    ok = 0
    fail = 0
    total_conf = 0.0
    start = time.monotonic()
    for i, ex in enumerate(examples):
        try:
            success, n_regions, mean_conf = process_example(ex, model, args.device)
            if success:
                ok += 1
                total_conf += mean_conf
            else:
                fail += 1
        except Exception as e:
            logger.warning("fail %s: %s", ex.name, e)
            fail += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(examples):
            elapsed = time.monotonic() - start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            avg_conf = total_conf / max(ok, 1)
            logger.info(
                "Progress: %d/%d (%.2f img/s, %d ok, %d fail, mean conf %.3f)",
                i + 1, len(examples), speed, ok, fail, avg_conf,
            )

    logger.info("Done. %d succeeded, %d failed, %d skipped.", ok, fail, skipped)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
