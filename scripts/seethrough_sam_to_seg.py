#!/usr/bin/env python3
"""Pseudo-label illustrated characters with See-Through SAM, convert via Dr. Li pipeline.

Runs Dr. Chengze Li's See-Through SAM body-parsing model (19 clothing classes,
Apache-2.0, HuggingFace ``24yearsold/l2d_sam_iter2``) on a directory of
Strata-format example dirs, then delegates the 19→22 class conversion to
``scripts.convert_li_labels`` — the same code that produced
``gemini_li_converted`` (Run 20's most heavily-weighted illustrated dataset).

Why this layout
---------------
Run 24/25 regressed because pseudo-labels came from Run 20 itself (classic
self-distillation ceiling). Run 26 v1 tried naive L/R + vertical splits to
convert See-Through's 19 clothing classes to our 22 anatomy classes and
regressed even harder — the heuristic splits introduced too much systematic
error (e.g., legwear 50/50 puts the knee boundary at mid-leg when it actually
sits ~60-70% down).

Dr. Li's ``convert_li_labels.py`` already does this conversion properly:
- uses joint positions when joints.json is present (anatomically correct)
- falls back to body-proportion heuristics (~70% knee, face-center midline,
  bbox-derived shoulder width) when joints are unavailable
- recovers the hair_back class by splitting hair at face-bottom

This script reuses those code paths directly, so our pseudo-labels get the
same conversion quality as Dr. Li's hand-labels.

Usage
-----
::

    # On A100:
    export SEETHROUGH_ROOT=/workspace/see-through

    # (Optional) run joints inference first for anatomically-correct splits:
    python3 scripts/run_joints_inference.py \\
        --input-dir ./data_cloud/gemini_diverse \\
        --checkpoint checkpoints/joints/run20_best.pt \\
        --device cuda

    # Pseudo-label with See-Through SAM + joint-based conversion:
    python3 scripts/seethrough_sam_to_seg.py \\
        --input-dir ./data_cloud/gemini_diverse \\
        --checkpoint /workspace/weights/li_sam_iter2.pt \\
        --device cuda

Writes ``segmentation.png`` (uint8 0-21), ``li_label.png`` (Dr. Li's 19-class
format, saved for auditing), ``confidence.png``, and updates ``metadata.json``.
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

# Project-relative imports so convert_li_labels is reachable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_li_labels import (  # noqa: E402
    convert_with_heuristics,
    convert_with_joints,
    load_joints,
)

logger = logging.getLogger(__name__)

REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]

# See-Through SAM output channels (VALID_BODY_PARTS_V2) mapped to Dr. Li's pixel values.
# Order matters: earlier classes win when multiple channels fire at the same pixel
# (we write highest-confidence-first, then skip pixels already claimed).
# Values match Dr. Li's PNG encoding: 0, 10, 20, ..., 180; 255 = background.
SEETHROUGH_CHANNELS = [
    ("hair",       0,    0),   # (name, channel_idx, li_pixel_value)
    ("headwear",   1,   10),
    ("face",       2,   20),
    ("eyes",       3,   30),
    ("eyewear",    4,   40),
    ("ears",       5,   50),
    ("earwear",    6,   60),
    ("nose",       7,   70),
    ("mouth",      8,   80),
    ("neck",       9,   90),
    ("neckwear",  10,  100),
    ("topwear",   11,  110),
    ("handwear",  12,  120),
    ("bottomwear",13,  130),
    ("legwear",   14,  140),
    ("footwear",  15,  150),
    ("tail",      16,  160),
    ("wings",     17,  170),
    ("objects",   18,  180),
]

# Channels with higher semantic priority are written first so that, e.g., face
# (class 20) overrides the overlapping hair (class 0) region when both fire.
# Priority order: small/specific → large/covering.
PRIORITY_ORDER = [
    "nose", "mouth", "eyes", "eyewear", "earwear", "ears",  # face details
    "face",
    "neckwear", "neck",
    "headwear",
    "handwear", "footwear",
    "topwear", "bottomwear", "legwear",
    "hair",                                                   # large background of head
    "tail", "wings", "objects",
]
_NAME_TO_CHANNEL = {name: (idx, val) for name, idx, val in SEETHROUGH_CHANNELS}


def build_li_mask(class_logits: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Build a Dr-Li-format 19-class mask from See-Through SAM logits.

    Args:
        class_logits: ``[19, H, W]`` float32 logits.
        alpha: ``[H, W]`` uint8 — foreground mask from the image alpha channel.

    Returns:
        ``[H, W]`` uint8 with Li pixel values (0, 10, ..., 180; 255 = background).
    """
    H, W = class_logits.shape[1:]
    li = np.full((H, W), 255, dtype=np.uint8)  # start as background
    fg = alpha > 10
    binary = class_logits > 0.0  # [19, H, W] bool
    for name in PRIORITY_ORDER:
        ch_idx, li_val = _NAME_TO_CHANNEL[name]
        mask = binary[ch_idx] & (li == 255) & fg
        if mask.any():
            li[mask] = li_val
    return li


# -------------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------------
def _build_seethrough_model(checkpoint_path: Path, device: str = "cuda"):
    """Load See-Through SAM body-parsing model (Dr. Li's checkpoint-18000.pt)."""
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
            # Li's checkpoint-18000.pt is ViT-H HQ (hidden=1280); b_hq would
            # size-mismatch on every encoder block.
            model_type="h_hq",
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
# Per-example loop
# -------------------------------------------------------------------------
def process_example(example_dir: Path, model, device: str) -> tuple[bool, int, float, bool]:
    """Run See-Through SAM on one example, then Dr. Li conversion. Returns
    ``(success, num_regions, mean_confidence, used_joints)``.
    """
    import torch

    image_path = example_dir / "image.png"
    if not image_path.exists():
        return False, 0, 0.0, False

    img_rgba = Image.open(image_path).convert("RGBA")
    img_np = np.array(img_rgba)
    alpha = img_np[:, :, 3]
    rgb = img_np[:, :, :3].copy()
    rgb[alpha < 10] = 0  # composite onto black to preserve edges

    with torch.inference_mode():
        preds = model.inference(rgb)[0]  # [19, H, W] torch tensor logits
        class_logits_np = preds.to(device="cpu", dtype=torch.float32).numpy()
        max_logit = preds.max(dim=0).values.to(device="cpu", dtype=torch.float32).numpy()

    # Step 1: build Li-format 19-class mask (same encoding as Dr. Li's hand labels)
    li_mask = build_li_mask(class_logits_np, alpha)

    # Step 2: Dr. Li's 19→22 conversion (joints if available, else body-proportion heuristics)
    joints = load_joints(example_dir / "joints.json")
    if joints:
        # Scale joints if label resolution differs from image (same as convert_li_labels.main())
        src_w, src_h = img_rgba.size
        lbl_h, lbl_w = li_mask.shape
        if (src_w, src_h) != (lbl_w, lbl_h):
            joints = {
                name: (int(pos[0] * lbl_w / src_w), int(pos[1] * lbl_h / src_h))
                for name, pos in joints.items()
            }
        seg = convert_with_joints(li_mask, joints)
        used_joints = True
    else:
        seg = convert_with_heuristics(li_mask)
        used_joints = False

    fg_mask = alpha >= 10
    seg[~fg_mask] = 0

    # Confidence map: sigmoid of max logit (per-pixel) for downstream filtering/review
    confidence = 1.0 / (1.0 + np.exp(-max_logit))
    confidence[~fg_mask] = 0.0
    mean_conf = float(confidence[fg_mask].mean()) if fg_mask.any() else 0.0

    unique_regions = sorted(int(v) for v in np.unique(seg) if v > 0)
    region_names = [REGION_NAMES[r] for r in unique_regions]

    Image.fromarray(seg).save(example_dir / "segmentation.png")
    # Save the intermediate Li-format label for auditing / re-conversion
    Image.fromarray(li_mask, mode="L").save(example_dir / "li_label.png")
    conf_uint8 = (confidence * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(conf_uint8).save(example_dir / "confidence.png")

    meta_path = example_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"id": example_dir.name}
    meta["has_segmentation_mask"] = True
    meta["segmentation_source"] = "seethrough_sam_li_converted"
    meta["segmentation_conversion"] = "joint_based" if used_joints else "heuristic"
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

    return True, len(unique_regions), mean_conf, used_joints


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pseudo-label with See-Through SAM and Dr. Li's converter.",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Strata-format examples: <input>/<char>/image.png")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/workspace/weights/li_sam_iter2.pt"),
                        help="Path to checkpoint-18000.pt. Auto-downloads from HF if missing.")
    parser.add_argument("--device", default="cuda",
                        help="torch device (cuda / cpu / mps)")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip examples whose segmentation_source is already seethrough_sam_li_converted")
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
        filtered = []
        for d in examples:
            meta = d / "metadata.json"
            if meta.exists():
                try:
                    m = json.loads(meta.read_text(encoding="utf-8"))
                    if m.get("segmentation_source") == "seethrough_sam_li_converted":
                        skipped += 1
                        continue
                except Exception:
                    pass
            filtered.append(d)
        examples = filtered
        logger.info("--only-missing: %d to process, %d skipped", len(examples), skipped)

    if args.limit is not None:
        examples = examples[: args.limit]

    ok = fail = joint_count = heuristic_count = 0
    total_conf = 0.0
    start = time.monotonic()
    for i, ex in enumerate(examples):
        try:
            success, n_regions, mean_conf, used_joints = process_example(ex, model, args.device)
            if success:
                ok += 1
                total_conf += mean_conf
                if used_joints:
                    joint_count += 1
                else:
                    heuristic_count += 1
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
                "Progress: %d/%d (%.2f img/s, %d ok [%d joint, %d heuristic], %d fail, mean conf %.3f)",
                i + 1, len(examples), speed, ok, joint_count, heuristic_count, fail, avg_conf,
            )

    logger.info(
        "Done. %d succeeded (%d joint-based, %d heuristic), %d failed, %d skipped.",
        ok, joint_count, heuristic_count, fail, skipped,
    )
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
