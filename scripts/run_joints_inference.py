"""Run joints inference on a dataset and save joints.json per example.

Takes a trained joints model checkpoint and runs inference on all examples
in a dataset directory, saving joints.json files compatible with the SAM2
pseudo-labeling pipeline.

Usage::

    python scripts/run_joints_inference.py \
        --input-dir ./data_cloud/gemini_diverse \
        --checkpoint checkpoints/joints/best.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BONE_ORDER: list[str] = [
    "hips", "spine", "chest", "neck", "head",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "hair_back",
]

RESOLUTION = 512


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Load the joints model from checkpoint."""
    from training.models.joint_model import JointModel

    model = JointModel(num_joints=20, pretrained_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: str,
    resolution: int = RESOLUTION,
) -> dict:
    """Run joints inference on a single image.

    Returns:
        Dict with "joints" key mapping bone names to {position: [x, y], confidence: float}.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        offsets, confidence, presence = model(tensor)

    # offsets: [1, 2, 20] — normalized (dx, dy) offsets from center
    # confidence: [1, 20]
    # presence: [1, 20]
    offsets = offsets.squeeze(0).cpu().numpy()  # [2, 20]
    confidence = torch.sigmoid(confidence).squeeze(0).cpu().numpy()  # [20]
    presence = torch.sigmoid(presence).squeeze(0).cpu().numpy()  # [20]

    joints = {}
    for i, name in enumerate(BONE_ORDER):
        # Convert normalized offsets to pixel coordinates
        dx, dy = offsets[0, i], offsets[1, i]
        x = (0.5 + dx) * resolution
        y = (0.5 + dy) * resolution
        conf = float(confidence[i] * presence[i])

        joints[name] = {
            "position": [int(round(x)), int(round(y))],
            "confidence": round(conf, 4),
        }

    return {"joints": joints}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run joints inference on a dataset")
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Joints model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, mps, cpu)")
    parser.add_argument("--only-missing", action="store_true", help="Skip examples with existing joints.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    device = args.device

    model = load_model(Path(args.checkpoint), device)
    logger.info("Loaded joints model from %s on %s", args.checkpoint, device)

    # Discover examples
    examples = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        image_path = child / "image.png"
        if not image_path.exists():
            continue
        if args.only_missing and (child / "joints.json").exists():
            continue
        examples.append(child)

    logger.info("Found %d examples to process", len(examples))

    processed = 0
    for i, example_dir in enumerate(examples):
        try:
            result = run_inference(model, example_dir / "image.png", device)
            joints_path = example_dir / "joints.json"
            joints_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            processed += 1
        except Exception:
            logger.exception("Failed to process %s", example_dir.name)

        if (i + 1) % 100 == 0 or i + 1 == len(examples):
            logger.info("  %d/%d done (%d saved)", i + 1, len(examples), processed)

    logger.info("Done: %d joints.json files saved", processed)


if __name__ == "__main__":
    main()
