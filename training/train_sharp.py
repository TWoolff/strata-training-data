"""Fine-tune Apple SHARP for illustrated character view synthesis.

Takes a single character illustration and predicts 3D Gaussian splats.
Training uses multi-view supervision from turnaround sheet data:
predict Gaussians from one view, render from other known camera angles,
compare to ground truth views.

Requires CUDA (gsplat differentiable renderer).

Usage::

    python -m training.train_sharp \
        --data-dir ./data/training/demo_pairs \
        --checkpoint ~/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt \
        --output-dir ./checkpoints/sharp \
        --epochs 50 --lr 1e-5 --batch-size 1
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Append ml-sharp to path (try multiple locations)
for candidate in [
    Path(__file__).parent.parent.parent / "ml-sharp",  # sibling to strata-training-data
    Path("/workspace/ml-sharp"),  # A100 cloud
    Path("../ml-sharp"),  # relative
]:
    if candidate.exists():
        sys.path.insert(0, str(candidate / "src"))
        break


def create_model(checkpoint_path: str | None, device: torch.device):
    """Load SHARP predictor model."""
    from sharp.models import PredictorParams, create_predictor

    model = create_predictor(PredictorParams())

    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.info("Downloading default SHARP checkpoint...")
        url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        model.load_state_dict(state_dict)

    model.to(device)
    return model


def create_renderer(device: torch.device):
    """Create gsplat differentiable renderer."""
    from sharp.utils.gsplat import GSplatRenderer

    renderer = GSplatRenderer(
        color_space="linearRGB",
        background_color="white",
    )
    renderer.to(device)
    return renderer


def render_from_gaussians(
    renderer,
    gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
):
    """Render Gaussians from given camera pose.

    Args:
        renderer: GSplatRenderer instance.
        gaussians: Gaussians3D from model prediction.
        extrinsics: [4, 4] camera extrinsics.
        intrinsics: [4, 4] camera intrinsics.
        image_size: Output resolution.

    Returns:
        rendered_rgb: [1, 3, H, W] rendered image.
        rendered_alpha: [1, 1, H, W] alpha mask.
    """
    from sharp.utils.gaussians import Gaussians3D

    # Ensure batch dimension
    if gaussians.mean_vectors.dim() == 2:
        gaussians = Gaussians3D(
            mean_vectors=gaussians.mean_vectors.unsqueeze(0),
            singular_values=gaussians.singular_values.unsqueeze(0),
            quaternions=gaussians.quaternions.unsqueeze(0),
            colors=gaussians.colors.unsqueeze(0),
            opacities=gaussians.opacities.unsqueeze(0),
        )

    output = renderer(
        gaussians,
        extrinsics.unsqueeze(0),
        intrinsics.unsqueeze(0),
        image_width=image_size,
        image_height=image_size,
    )
    return output.color, output.alpha


def compute_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    rendered_alpha: torch.Tensor,
):
    """Compute training loss.

    Args:
        rendered: [1, 3, H, W] rendered image.
        target: [3, H, W] ground truth image.
        rendered_alpha: [1, 1, H, W] alpha mask.

    Returns:
        loss: scalar tensor.
        metrics: dict of individual loss components.
    """
    target = target.unsqueeze(0)  # [1, 3, H, W]

    # L1 loss
    l1_loss = F.l1_loss(rendered, target)

    # Alpha regularization — encourage non-zero alpha where target has content
    target_has_content = (target.mean(dim=1, keepdim=True) < 0.98).float()
    alpha_loss = F.binary_cross_entropy(
        rendered_alpha.clamp(1e-6, 1 - 1e-6),
        target_has_content,
    )

    loss = l1_loss + 0.1 * alpha_loss

    return loss, {
        "l1": l1_loss.item(),
        "alpha": alpha_loss.item(),
        "total": loss.item(),
    }


def train_one_epoch(
    model,
    renderer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    image_size: int,
    internal_size: int,
):
    """Train for one epoch."""
    from sharp.utils.gaussians import unproject_gaussians

    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    n_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_image = batch["input_image"].to(device)  # [B, 3, H, W]
        target_image = batch["target_image"].to(device)  # [B, 3, H, W]
        input_extrinsics = batch["input_extrinsics"].to(device)  # [B, 4, 4]
        target_extrinsics = batch["target_extrinsics"].to(device)  # [B, 4, 4]
        intrinsics = batch["intrinsics"].to(device)  # [B, 4, 4]
        disparity_factor = batch["disparity_factor"].to(device)  # [B]

        batch_size = input_image.shape[0]
        batch_loss = 0.0

        for b in range(batch_size):
            # Resize input to SHARP's internal resolution
            img_resized = F.interpolate(
                input_image[b:b+1],
                size=(internal_size, internal_size),
                mode="bilinear",
                align_corners=True,
            )

            # Predict Gaussians
            gaussians_ndc = model(img_resized, disparity_factor[b:b+1])

            # Unproject to world space using input camera
            intrinsics_resized = intrinsics[b].clone()
            intrinsics_resized[0] *= internal_size / image_size
            intrinsics_resized[1] *= internal_size / image_size

            gaussians = unproject_gaussians(
                gaussians_ndc,
                input_extrinsics[b],
                intrinsics_resized,
                (internal_size, internal_size),
            )

            # Render from target camera and compute loss
            rendered_rgb, rendered_alpha = render_from_gaussians(
                renderer, gaussians,
                target_extrinsics[b],
                intrinsics[b],
                image_size,
            )

            loss, metrics = compute_loss(
                rendered_rgb, target_image[b], rendered_alpha,
            )
            batch_loss += loss

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += metrics["total"]
        total_l1 += metrics["l1"]
        n_batches += 1

    return {
        "train/loss": total_loss / max(n_batches, 1),
        "train/l1": total_l1 / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model,
    renderer,
    val_loader: DataLoader,
    device: torch.device,
    image_size: int,
    internal_size: int,
):
    """Validate on held-out characters."""
    from sharp.utils.gaussians import unproject_gaussians

    model.eval()
    total_l1 = 0.0
    n_samples = 0

    for batch in val_loader:
        input_image = batch["input_image"].to(device)
        target_image = batch["target_image"].to(device)
        input_extrinsics = batch["input_extrinsics"].to(device)
        target_extrinsics = batch["target_extrinsics"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        disparity_factor = batch["disparity_factor"].to(device)

        batch_size = input_image.shape[0]

        for b in range(batch_size):
            img_resized = F.interpolate(
                input_image[b:b+1],
                size=(internal_size, internal_size),
                mode="bilinear",
                align_corners=True,
            )

            gaussians_ndc = model(img_resized, disparity_factor[b:b+1])

            intrinsics_resized = intrinsics[b].clone()
            intrinsics_resized[0] *= internal_size / image_size
            intrinsics_resized[1] *= internal_size / image_size

            gaussians = unproject_gaussians(
                gaussians_ndc,
                input_extrinsics[b],
                intrinsics_resized,
                (internal_size, internal_size),
            )

            rendered_rgb, _ = render_from_gaussians(
                renderer, gaussians,
                target_extrinsics[b],
                intrinsics[b],
                image_size,
            )

            l1 = F.l1_loss(rendered_rgb, target_image[b].unsqueeze(0))
            total_l1 += l1.item()
            n_samples += 1

    return {"val/l1": total_l1 / max(n_samples, 1)}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SHARP on turnaround sheets")
    parser.add_argument("--data-dir", type=str, nargs="+", required=True,
                        help="Directories with turnaround sheet characters")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SHARP .pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/sharp",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 recommended, model is large)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--internal-resolution", type=int, default=768,
                        help="SHARP internal resolution (1536 is default but OOMs, use 768)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze the DINOv2 encoder (train only decoder + heads)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.error("CUDA required for training (gsplat renderer). Use A100.")
        sys.exit(1)

    # Create model
    model = create_model(args.checkpoint, device)

    # Optionally freeze encoder
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "monodepth_model" in name and "decoder" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Frozen encoder. Trainable: %d / %d params (%.1f%%)",
                    trainable, total, 100 * trainable / total)

    # Create renderer
    renderer = create_renderer(device)

    # Create datasets
    from training.data.sharp_dataset import SharpDataset, SharpDatasetConfig

    config = SharpDatasetConfig(
        dataset_dirs=[Path(d) for d in args.data_dir],
        resolution=args.resolution,
        internal_resolution=args.internal_resolution,
        min_views=2,
    )

    train_ds = SharpDataset(config, split="train")
    val_ds = SharpDataset(config, split="val")

    logger.info("Train: %d characters, Val: %d characters", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
    )

    # Optimizer — lower lr for encoder, higher for decoder
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Checkpointing
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tb_logs")

    best_val_l1 = float("inf")
    patience_counter = 0

    logger.info("Starting training: %d epochs, lr=%.1e, batch_size=%d, internal_res=%d",
                args.epochs, args.lr, args.batch_size, args.internal_resolution)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, renderer, train_loader, optimizer, device,
            args.resolution, args.internal_resolution,
        )
        val_metrics = validate(
            model, renderer, val_loader, device,
            args.resolution, args.internal_resolution,
        )

        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d — train/l1=%.4f | val/l1=%.4f | lr=%.2e | %.0fs",
            epoch + 1, args.epochs,
            train_metrics["train/l1"], val_metrics["val/l1"],
            lr, elapsed,
        )

        # TensorBoard
        for k, v in {**train_metrics, **val_metrics}.items():
            writer.add_scalar(k, v, epoch)

        # Save checkpoint
        torch.save(model.state_dict(), output_dir / "latest.pt")

        if val_metrics["val/l1"] < best_val_l1:
            best_val_l1 = val_metrics["val/l1"]
            torch.save(model.state_dict(), output_dir / "best.pt")
            logger.info("  New best val/l1: %.4f", best_val_l1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Training complete. Best val/l1: %.4f", best_val_l1)
    writer.close()


if __name__ == "__main__":
    main()
