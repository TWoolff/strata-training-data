"""Evaluate trained models on the test set and produce reports.

CLI entry point::

    python training/evaluate.py --model segmentation \
        --checkpoint checkpoints/segmentation/best.pt \
        --dataset-dir ../output/segmentation/

    python training/evaluate.py --model joints \
        --checkpoint checkpoints/joints/best.pt \
        --dataset-dir ../output/segmentation/

    python training/evaluate.py --model weights \
        --checkpoint checkpoints/weights/best.pt \
        --dataset-dir ../output/segmentation/

    python training/evaluate.py --all --output-dir evaluation_results/

Produces per-class metric tables, confusion matrix heatmaps, prediction
overlay grids, worst-N example images, and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.config import REGION_NAMES
from training.data.transforms import BONE_ORDER
from training.utils.checkpoint import load_checkpoint
from training.utils.metrics import JointMetrics, SegmentationMetrics, WeightMetrics
from training.utils.visualization import (
    colorize_mask,
    overlay_segmentation,
    plot_confusion_matrix,
    plot_per_class_iou,
    plot_per_joint_error,
    save_prediction_grid,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    """Auto-detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Denormalize image for visualization
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize_image(tensor: np.ndarray) -> np.ndarray:
    """Convert a normalized [C, H, W] float image to [H, W, 3] uint8.

    Args:
        tensor: Normalized float image with shape ``[3, H, W]``.

    Returns:
        RGB uint8 image with shape ``[H, W, 3]``.
    """
    img = tensor.transpose(1, 2, 0)  # [H, W, 3]
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Segmentation evaluation
# ---------------------------------------------------------------------------


def _seg_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for segmentation eval (subset of train collate)."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "segmentation": torch.stack([b["segmentation"] for b in batch]),
        "draw_order": torch.stack([b["draw_order"] for b in batch]),
        "has_draw_order": torch.tensor([b["has_draw_order"] for b in batch], dtype=torch.bool),
        "confidence_target": torch.stack([b["confidence_target"] for b in batch]),
    }


def evaluate_segmentation(
    checkpoint_path: Path,
    dataset_dirs: list[Path],
    output_dir: Path,
    worst_n: int = 16,
    grid_n: int = 8,
) -> dict:
    """Run segmentation model evaluation on the test set.

    Args:
        checkpoint_path: Path to model checkpoint.
        dataset_dirs: Directories containing the dataset.
        output_dir: Directory to write evaluation outputs.
        worst_n: Number of worst examples to save.
        grid_n: Number of rows in the prediction grid.

    Returns:
        Summary metrics dict.
    """
    from training.data.segmentation_dataset import DatasetConfig, SegmentationDataset
    from training.models.segmentation_model import SegmentationModel

    device = _select_device()
    logger.info("Evaluating segmentation model on %s", device)

    # Load model
    model = SegmentationModel(num_classes=22, pretrained_backbone=False)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()

    # Load test dataset
    ds_config = DatasetConfig()
    test_dataset = SegmentationDataset(dataset_dirs, split="test", augment=False, config=ds_config)
    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty — skipping segmentation evaluation")
        return {"model": "segmentation", "error": "empty_test_set"}

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_seg_collate_fn,
    )

    # Run inference
    metrics = SegmentationMetrics(num_classes=22)

    # Track per-example IoU for worst-N
    example_ious: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]] = []
    example_idx = 0

    # Collect grid examples from first batch
    grid_images: list[np.ndarray] = []
    grid_gt: list[np.ndarray] = []
    grid_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            seg_pred = outputs["segmentation"].argmax(dim=1).cpu().numpy()
            seg_target = batch["segmentation"].numpy()
            batch_images = batch["image"].numpy()

            metrics.update(seg_pred, seg_target)

            # Per-example IoU for worst-N
            for i in range(seg_pred.shape[0]):
                per_ex_metrics = SegmentationMetrics(num_classes=22)
                per_ex_metrics.update(seg_pred[i : i + 1], seg_target[i : i + 1])
                ex_iou = per_ex_metrics.miou()

                if len(example_ious) < worst_n or ex_iou < example_ious[-1][1]:
                    example_ious.append(
                        (
                            example_idx,
                            ex_iou,
                            _denormalize_image(batch_images[i]),
                            seg_target[i].astype(np.uint8),
                            seg_pred[i].astype(np.uint8),
                        )
                    )
                    example_ious.sort(key=lambda x: x[1])
                    example_ious = example_ious[:worst_n]

                # Collect grid examples
                if len(grid_images) < grid_n:
                    grid_images.append(_denormalize_image(batch_images[i]))
                    grid_gt.append(seg_target[i].astype(np.uint8))
                    grid_pred.append(seg_pred[i].astype(np.uint8))

                example_idx += 1

    # Compute metrics
    miou = metrics.miou()
    per_class_iou = metrics.per_class_iou()
    per_class_acc = metrics.per_class_accuracy()
    overall_acc = metrics.overall_accuracy()

    # Print results
    print("\n=== Segmentation Evaluation ===")
    print(f"mIoU:             {miou:.4f}")
    print(f"Pixel Accuracy:   {overall_acc:.4f}")
    print(f"Test Examples:    {example_idx}")
    print("\nPer-Class IoU:")
    for name, iou_val in per_class_iou.items():
        print(f"  {name:20s} {iou_val:.4f}")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class names for confusion matrix
    class_names = [REGION_NAMES.get(i, f"class_{i}") for i in range(22)]

    # Confusion matrix
    plot_confusion_matrix(metrics.confusion, class_names, output_dir / "confusion_matrix.png")

    # Per-class IoU chart
    plot_per_class_iou(per_class_iou, output_dir / "per_class_iou.png")

    # Prediction grid
    if grid_images:
        save_prediction_grid(
            grid_images, grid_gt, grid_pred, output_dir / "prediction_grid.png", n=grid_n
        )

    # Worst-N examples
    worst_dir = output_dir / "worst_examples"
    worst_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for rank, (idx, ex_iou, img, gt, pred) in enumerate(example_ious):
        overlay_img = overlay_segmentation(img, pred, alpha=0.5)
        out = np.concatenate([img, colorize_mask(gt), colorize_mask(pred), overlay_img], axis=1)
        Image.fromarray(out).save(worst_dir / f"worst_{rank:02d}_iou{ex_iou:.3f}_idx{idx}.png")

    # Summary JSON
    summary = {
        "model": "segmentation",
        "num_examples": example_idx,
        "miou": miou,
        "pixel_accuracy": overall_acc,
        "per_class_iou": per_class_iou,
        "per_class_accuracy": per_class_acc,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved evaluation summary to %s", output_dir / "summary.json")

    return summary


# ---------------------------------------------------------------------------
# Joint evaluation
# ---------------------------------------------------------------------------


def _joint_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for joint eval."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "gt_positions": torch.stack([b["gt_positions"] for b in batch]),
        "gt_visible": torch.stack([b["gt_visible"] for b in batch]),
        "geo_positions": torch.stack([b["geo_positions"] for b in batch]),
        "gt_offsets": torch.stack([b["gt_offsets"] for b in batch]),
    }


def evaluate_joints(
    checkpoint_path: Path,
    dataset_dirs: list[Path],
    output_dir: Path,
    worst_n: int = 16,
) -> dict:
    """Run joint model evaluation on the test set.

    Args:
        checkpoint_path: Path to model checkpoint.
        dataset_dirs: Directories containing the dataset.
        output_dir: Directory to write evaluation outputs.
        worst_n: Number of worst examples to save.

    Returns:
        Summary metrics dict.
    """
    from training.data.joint_dataset import JointDataset, JointDatasetConfig
    from training.models.joint_model import JointModel

    device = _select_device()
    logger.info("Evaluating joint model on %s", device)

    # Load model
    model = JointModel(num_joints=20, pretrained_backbone=False)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()

    # Load test dataset
    ds_config = JointDatasetConfig()
    test_dataset = JointDataset(dataset_dirs, split="test", augment=False, config=ds_config)
    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty — skipping joint evaluation")
        return {"model": "joints", "error": "empty_test_set"}

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_joint_collate_fn,
    )

    # Run inference
    metrics = JointMetrics(num_joints=20)

    # Collect GT/pred positions for scatter plot
    all_gt_positions: list[np.ndarray] = []
    all_pred_positions: list[np.ndarray] = []

    # Track per-example error for worst-N
    example_errors: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]] = []
    example_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            # Model outputs [B, 2, 20]; metrics expects [B, J, 2]
            pred_offsets = outputs["offsets"].cpu().numpy()  # [B, 2, 20]
            gt_offsets = batch["gt_offsets"].numpy()  # [B, 2, 20]
            pred_present = (torch.sigmoid(outputs["present"]) > 0.5).cpu().numpy()
            gt_visible = batch["gt_visible"].numpy()  # [B, 20]

            pred_offsets_jxy = np.transpose(pred_offsets, (0, 2, 1))  # [B, 20, 2]
            gt_offsets_jxy = np.transpose(gt_offsets, (0, 2, 1))  # [B, 20, 2]
            metrics.update(pred_offsets_jxy, gt_offsets_jxy, pred_present, gt_visible)

            # Collect positions for scatter plot
            gt_pos = batch["gt_positions"].numpy()  # [B, 20, 2]
            geo_pos = batch["geo_positions"].numpy()  # [B, 20, 2]
            # Predicted absolute position = geo + pred_offset
            pred_pos = geo_pos + np.transpose(pred_offsets, (0, 2, 1))  # [B, 20, 2]

            for i in range(gt_pos.shape[0]):
                vis = gt_visible[i].astype(bool)
                if vis.any():
                    all_gt_positions.append(gt_pos[i][vis])
                    all_pred_positions.append(pred_pos[i][vis])

                    # Per-example total error
                    err = np.sqrt(((gt_pos[i][vis] - pred_pos[i][vis]) ** 2).sum(axis=-1)).mean()
                    example_errors.append(
                        (
                            example_idx,
                            float(err),
                            batch["image"][i].numpy(),
                            gt_pos[i],
                            pred_pos[i],
                        )
                    )
                    example_errors.sort(key=lambda x: -x[1])
                    example_errors = example_errors[:worst_n]

                example_idx += 1

    # Compute metrics
    mean_err = metrics.mean_offset_error()
    pres_acc = metrics.presence_accuracy()
    per_joint = metrics.per_joint_error()

    # Print results
    print("\n=== Joint Evaluation ===")
    print(f"Mean Offset Error: {mean_err:.6f}")
    print(f"Presence Accuracy: {pres_acc:.4f}")
    print(f"Test Examples:     {example_idx}")
    print("\nPer-Joint MSE:")
    for name, err_val in per_joint.items():
        print(f"  {name:20s} {err_val:.6f}")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-joint error chart
    plot_per_joint_error(per_joint, output_dir / "per_joint_error.png")

    # Scatter plot
    if all_gt_positions:
        from training.utils.visualization import plot_joint_scatter

        gt_all = np.concatenate(all_gt_positions, axis=0)
        pred_all = np.concatenate(all_pred_positions, axis=0)
        plot_joint_scatter(gt_all, pred_all, output_dir / "joint_scatter.png")

    # Worst-N examples
    worst_dir = output_dir / "worst_examples"
    worst_dir.mkdir(parents=True, exist_ok=True)
    from training.utils.visualization import save_joint_comparison

    resolution = 512
    for rank, (idx, err, img_tensor, gt_pos, pred_pos) in enumerate(example_errors):
        img = _denormalize_image(img_tensor)
        gt_joints = {
            BONE_ORDER[j]: (float(gt_pos[j, 0]) * resolution, float(gt_pos[j, 1]) * resolution)
            for j in range(20)
        }
        pred_joints = {
            BONE_ORDER[j]: (
                float(pred_pos[j, 0]) * resolution,
                float(pred_pos[j, 1]) * resolution,
            )
            for j in range(20)
        }
        save_joint_comparison(
            img,
            gt_joints,
            pred_joints,
            worst_dir / f"worst_{rank:02d}_err{err:.4f}_idx{idx}.png",
        )

    # Summary JSON
    summary = {
        "model": "joints",
        "num_examples": example_idx,
        "mean_offset_error": mean_err,
        "presence_accuracy": pres_acc,
        "per_joint_error": per_joint,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved evaluation summary to %s", output_dir / "summary.json")

    return summary


# ---------------------------------------------------------------------------
# Weight evaluation
# ---------------------------------------------------------------------------


def _weight_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for weight eval."""
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "weights_target": torch.stack([b["weights_target"] for b in batch]),
        "confidence_target": torch.stack([b["confidence_target"] for b in batch]),
        "num_vertices": torch.tensor([b["num_vertices"] for b in batch], dtype=torch.long),
    }


def evaluate_weights(
    checkpoint_path: Path,
    dataset_dirs: list[Path],
    output_dir: Path,
) -> dict:
    """Run weight prediction model evaluation on the test set.

    Args:
        checkpoint_path: Path to model checkpoint.
        dataset_dirs: Directories containing the dataset.
        output_dir: Directory to write evaluation outputs.

    Returns:
        Summary metrics dict.
    """
    from training.data.weight_dataset import WeightDataset, WeightDatasetConfig
    from training.models.weight_prediction_model import WeightPredictionModel

    device = _select_device()
    logger.info("Evaluating weight prediction model on %s", device)

    # Load model
    model = WeightPredictionModel(num_features=31, num_bones=20)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    model.eval()

    # Load test dataset
    ds_config = WeightDatasetConfig()
    test_dataset = WeightDataset(dataset_dirs, split="test", config=ds_config)
    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty — skipping weight evaluation")
        return {"model": "weights", "error": "empty_test_set"}

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_weight_collate_fn,
    )

    # Run inference
    metrics = WeightMetrics(num_bones=20)

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            outputs = model(features)

            pred_weights_logits = outputs["weights"].squeeze(-1).cpu()  # [B, 20, N]
            pred_weights = F.softmax(pred_weights_logits, dim=1).numpy()
            gt_weights = batch["weights_target"].numpy()  # [B, 20, N]

            pred_conf = (
                (torch.sigmoid(outputs["confidence"].squeeze(-1).squeeze(1)) > 0.5).cpu().numpy()
            )  # [B, N]
            gt_conf = batch["confidence_target"].numpy()  # [B, N]
            num_verts = batch["num_vertices"].numpy()  # [B]

            metrics.update(pred_weights, gt_weights, pred_conf, gt_conf, num_verts)

    # Compute metrics
    mae = metrics.mae()
    conf_acc = metrics.confidence_accuracy()
    per_bone = metrics.per_bone_mae()

    # Print results
    print("\n=== Weight Evaluation ===")
    print(f"MAE:                   {mae:.6f}")
    print(f"Confidence Accuracy:   {conf_acc:.4f}")
    print(f"Test Examples:         {len(test_dataset)}")
    print("\nPer-Bone MAE:")
    for name, err_val in per_bone.items():
        print(f"  {name:20s} {err_val:.6f}")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-bone MAE chart
    from training.utils.visualization import plot_per_joint_error

    plot_per_joint_error(per_bone, output_dir / "per_bone_mae.png")

    # Summary JSON
    summary = {
        "model": "weights",
        "num_examples": len(test_dataset),
        "mae": mae,
        "confidence_accuracy": conf_acc,
        "per_bone_mae": per_bone,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved evaluation summary to %s", output_dir / "summary.json")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained Strata models on the test set")
    parser.add_argument(
        "--model",
        type=str,
        choices=["segmentation", "joints", "weights"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        action="append",
        help="Dataset directory (can be specified multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory for evaluation outputs (default: evaluation_results/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all model types (looks for best.pt in standard checkpoint dirs)",
    )
    parser.add_argument(
        "--min-miou",
        type=float,
        default=None,
        help="Minimum mIoU threshold — exit code 1 if segmentation mIoU is below this",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=16,
        help="Number of worst examples to save (default: 16)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.all and not args.model:
        parser.error("Either --model or --all is required")

    if not args.all and not args.checkpoint:
        parser.error("--checkpoint is required when using --model")

    dataset_dirs = [Path(d) for d in (args.dataset_dir or ["../output/segmentation"])]
    output_base = Path(args.output_dir)

    summaries = {}

    if args.all:
        # Evaluate all models with standard checkpoint paths
        seg_ckpt = Path("checkpoints/segmentation/best.pt")
        if seg_ckpt.exists():
            summaries["segmentation"] = evaluate_segmentation(
                seg_ckpt,
                dataset_dirs,
                output_base / "segmentation",
                worst_n=args.worst_n,
            )
        else:
            logger.warning("Segmentation checkpoint not found: %s", seg_ckpt)

        joint_ckpt = Path("checkpoints/joints/best.pt")
        if joint_ckpt.exists():
            summaries["joints"] = evaluate_joints(
                joint_ckpt,
                dataset_dirs,
                output_base / "joints",
                worst_n=args.worst_n,
            )
        else:
            logger.warning("Joint checkpoint not found: %s", joint_ckpt)

        weight_ckpt = Path("checkpoints/weights/best.pt")
        if weight_ckpt.exists():
            summaries["weights"] = evaluate_weights(
                weight_ckpt,
                dataset_dirs,
                output_base / "weights",
            )
        else:
            logger.warning("Weight checkpoint not found: %s", weight_ckpt)

    elif args.model == "segmentation":
        summaries["segmentation"] = evaluate_segmentation(
            Path(args.checkpoint),
            dataset_dirs,
            output_base / "segmentation",
            worst_n=args.worst_n,
        )
    elif args.model == "joints":
        summaries["joints"] = evaluate_joints(
            Path(args.checkpoint),
            dataset_dirs,
            output_base / "joints",
            worst_n=args.worst_n,
        )
    elif args.model == "weights":
        summaries["weights"] = evaluate_weights(
            Path(args.checkpoint),
            dataset_dirs,
            output_base / "weights",
        )

    # Combined summary
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "all_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("All summaries saved to %s", output_base / "all_summaries.json")

    # Check mIoU threshold
    if args.min_miou is not None and "segmentation" in summaries:
        miou = summaries["segmentation"].get("miou", 0.0)
        if miou < args.min_miou:
            logger.error("mIoU %.4f is below threshold %.4f — failing", miou, args.min_miou)
            sys.exit(1)
        else:
            logger.info("mIoU %.4f meets threshold %.4f", miou, args.min_miou)


if __name__ == "__main__":
    main()
