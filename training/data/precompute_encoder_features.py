"""Precompute segmentation encoder features for diffusion weight training.

Runs the segmentation model on each training image and extracts the backbone
encoder features (before the ASPP head). For each vertex in the corresponding
weight data, bilinearly samples the feature map at the vertex position and
saves the result as a ``.npy`` file.

Usage::

    python -m training.data.precompute_encoder_features \\
        --segmentation-checkpoint checkpoints/segmentation/best.pt \\
        --data-dirs ./data_cloud/humanrig ./data_cloud/segmentation \\
        --output-dir ./data_cloud/encoder_features \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

from training.models.segmentation_model import SegmentationModel

logger = logging.getLogger(__name__)

# ImageNet normalization (matches segmentation training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Cap vertices to match weight model's max — reduces .npy from ~30MB to ~3.7MB
MAX_VERTICES = 2048


def load_segmentation_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load the segmentation model backbone for feature extraction."""
    model = SegmentationModel(num_classes=22, pretrained_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, resolution: int = 512) -> torch.Tensor:
    """Load and preprocess an image for the segmentation model."""
    img = Image.open(image_path).convert("RGBA")

    # Composite onto white background (matches training)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("RGB")

    img = img.resize((resolution, resolution), Image.BILINEAR)

    # To tensor [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    # ImageNet normalize
    tensor = normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    return tensor.unsqueeze(0)  # [1, 3, H, W]


def extract_backbone_features(
    model: SegmentationModel,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Extract backbone features [1, C, H/16, W/16] from the segmentation model."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.backbone(image_tensor)
    return features["out"]  # [1, 960, 32, 32]


def sample_features_at_vertices(
    feature_map: torch.Tensor,
    vertex_positions: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Bilinearly sample feature map at vertex positions.

    Args:
        feature_map: ``[1, C, fH, fW]`` backbone features.
        vertex_positions: List of ``(x_pixel, y_pixel)`` in image space.
        image_width: Original image width (for normalizing coords).
        image_height: Original image height (for normalizing coords).

    Returns:
        ``[C, N]`` numpy array of sampled features.
    """
    if not vertex_positions:
        return np.zeros((feature_map.shape[1], 0), dtype=np.float32)

    n = len(vertex_positions)

    # Build grid for F.grid_sample: [-1, 1] range
    # grid_sample expects (x, y) in [-1, 1] where (-1, -1) = top-left
    grid = torch.zeros(1, 1, n, 2, device=feature_map.device)
    for i, (vx, vy) in enumerate(vertex_positions):
        # Normalize to [-1, 1]
        grid[0, 0, i, 0] = 2.0 * vx / max(image_width - 1, 1) - 1.0
        grid[0, 0, i, 1] = 2.0 * vy / max(image_height - 1, 1) - 1.0

    # Sample: [1, C, 1, N]
    sampled = F.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    return sampled[0, :, 0, :].cpu().numpy()  # [C, N]


def discover_examples(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find weight+image examples. Returns [(example_id, image_path, weights_path)]."""
    examples = []

    # Per-example layout
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        weights_path = child / "weights.json"
        image_path = child / "image.png"
        if weights_path.exists() and image_path.exists():
            examples.append((child.name, image_path, weights_path))

    # Flat layout
    if not examples:
        weights_dir = data_dir / "weights"
        images_dir = data_dir / "images"
        if weights_dir.is_dir() and images_dir.is_dir():
            for wp in sorted(weights_dir.glob("*.json")):
                stem = wp.stem
                # Try exact match first, then prefix match with style suffix
                # (e.g., weight stem "Aj_pose_00" → image "Aj_pose_00_flat.png")
                ip = images_dir / f"{stem}.png"
                if not ip.exists():
                    candidates = sorted(images_dir.glob(f"{stem}_*.png"))
                    if candidates:
                        ip = candidates[0]  # Use first style variant
                    else:
                        continue
                examples.append((stem, ip, wp))

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute segmentation encoder features for diffusion weight training"
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        required=True,
        help="Path to segmentation model checkpoint",
    )
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Dataset directories containing weight data + images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .npy encoder feature files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input resolution for segmentation model",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip examples that already have precomputed features",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation model
    logger.info("Loading segmentation model from %s", args.segmentation_checkpoint)
    model = load_segmentation_backbone(args.segmentation_checkpoint, device)

    # Discover examples across all data dirs
    all_examples = []
    for data_dir in args.data_dirs:
        examples = discover_examples(data_dir)
        logger.info("Found %d examples in %s", len(examples), data_dir)
        all_examples.extend(examples)

    logger.info("Total: %d examples to process", len(all_examples))

    processed = 0
    skipped = 0
    errors = 0

    for example_id, image_path, weights_path in all_examples:
        out_path = output_dir / f"{example_id}.npy"

        if args.only_missing and out_path.exists():
            skipped += 1
            continue

        try:
            # Load weight data to get vertex positions
            weight_data = json.loads(weights_path.read_text(encoding="utf-8"))
            vertices = weight_data.get("vertices", [])
            image_size = weight_data.get("image_size", [512, 512])

            if not vertices:
                skipped += 1
                continue

            # Extract vertex positions (cap at MAX_VERTICES to bound file size)
            if len(vertices) > MAX_VERTICES:
                # Uniformly subsample to keep spatial coverage
                indices = np.linspace(0, len(vertices) - 1, MAX_VERTICES, dtype=int)
                vertices = [vertices[i] for i in indices]
            vertex_positions = [
                (float(v["position"][0]), float(v["position"][1])) for v in vertices
            ]

            # Run segmentation backbone
            img_tensor = preprocess_image(image_path, args.resolution)
            feature_map = extract_backbone_features(model, img_tensor, device)

            # Sample at vertex positions → [C, N]
            sampled = sample_features_at_vertices(
                feature_map, vertex_positions, image_size[0], image_size[1]
            )

            # Save as float16 to halve disk usage (~3.5MB vs ~7.5MB per file)
            np.save(out_path, sampled.astype(np.float16))
            processed += 1

            if processed % 500 == 0:
                logger.info("Processed %d / %d examples", processed, len(all_examples))

        except Exception as e:
            logger.warning("Error processing %s: %s", example_id, e)
            errors += 1

    logger.info(
        "Done. Processed: %d, Skipped: %d, Errors: %d",
        processed,
        skipped,
        errors,
    )


if __name__ == "__main__":
    main()
