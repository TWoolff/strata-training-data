"""Stage A of Run 23 — pretrain DeepLabV3-ResNet50 on Pascal-Person-Part.

Produces an anatomy-aware backbone checkpoint that can be loaded into
``SegmentationModel(backbone='resnet50', backbone_weights_path=...)`` for
the Strata segmentation fine-tune (Stage B). The backbone priors come from
real humans annotated with 6 body parts — a signal our synthetic-only Run 20
init (ImageNet) does not have.

Pascal-Person-Part schema (7 classes including background):
    0: background
    1: head       (hair, head, leye, reye, lebrow, rebrow, lear, rear, nose, mouth)
    2: torso      (neck, torso)
    3: upper_arm  (luarm, ruarm)
    4: lower_arm  (llarm, rlarm, lhand, rhand)
    5: upper_leg  (luleg, ruleg)
    6: lower_leg  (llleg, rlleg, lfoot, rfoot)

Outputs ``./checkpoints/anatomy_init/backbone.pth`` containing ONLY the
backbone ``state_dict`` (no head, no aux heads) — safe to load with
``strict=False`` into the Strata seg model.

Usage::

    python3 -m training.pretrain_pascal_person_part \\
        --output ./checkpoints/anatomy_init/backbone.pth \\
        --epochs 40 \\
        --batch-size 16

Data source (both free, no login):
    - VOC 2010 images: http://pjreddie.com/media/files/VOCtrainval_03-May-2010.tar
    - Pascal-Part annotations: http://roozbehm.info/pascal-parts/trainval.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset URLs and part → class mapping
# ---------------------------------------------------------------------------

# Use final redirect targets directly — urllib.request.urlretrieve's redirect
# handling is fine in recent Python, but going straight to the CDN avoids a
# chain of 301s (host.robots.ox.ac.uk → www.robots.ox.ac.uk → thor) that can
# flake on flaky network conditions.
VOC_IMAGES_URL = "https://thor.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
PASCAL_PART_URL = "https://roozbehm.info/pascal-parts/trainval.tar.gz"
PASCAL_PART_MD5 = "2fa0a19ee9b5e43b2bee520166111120"

# Pascal-Part subparts → 6 anatomy parts (1..6; 0 = background)
PART_TO_CLASS = {
    # head (1)
    "head": 1, "hair": 1, "leye": 1, "reye": 1, "lebrow": 1, "rebrow": 1,
    "lear": 1, "rear": 1, "nose": 1, "mouth": 1,
    # torso (2)
    "neck": 2, "torso": 2,
    # upper arm (3)
    "luarm": 3, "ruarm": 3,
    # lower arm + hand (4)
    "llarm": 4, "rlarm": 4, "lhand": 4, "rhand": 4,
    # upper leg (5)
    "luleg": 5, "ruleg": 5,
    # lower leg + foot (6)
    "llleg": 6, "rlleg": 6, "lfoot": 6, "rfoot": 6,
}
NUM_CLASSES = 7  # background + 6 anatomy parts


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------


def _download(url: str, dst: Path, expected_md5: str | None = None) -> None:
    """Download ``url`` to ``dst`` (skipping if already present).

    Validates MD5 when ``expected_md5`` is provided. On mismatch the file is
    removed so the next run redownloads it.
    """
    if dst.exists() and dst.stat().st_size > 0:
        logger.info("  Already present: %s (%.1f MB)", dst.name, dst.stat().st_size / 1e6)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Downloading %s → %s", url, dst)
    urllib.request.urlretrieve(url, dst)
    logger.info("  Downloaded %.1f MB", dst.stat().st_size / 1e6)
    if expected_md5:
        h = hashlib.md5()
        with open(dst, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != expected_md5:
            dst.unlink()
            raise RuntimeError(f"MD5 mismatch for {dst.name}: expected {expected_md5}, got {got}")


def _extract(archive: Path, out_dir: Path) -> None:
    """Extract ``archive`` into ``out_dir`` if marker file does not exist."""
    marker = out_dir / f".extracted_{archive.name}"
    if marker.exists():
        logger.info("  Already extracted: %s", archive.name)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Extracting %s → %s", archive, out_dir)
    with tarfile.open(archive) as tf:
        tf.extractall(out_dir)
    marker.touch()


# ---------------------------------------------------------------------------
# Pascal-Part .mat parser
# ---------------------------------------------------------------------------


def _parse_pascal_part_mat(mat_path: Path) -> np.ndarray | None:
    """Parse a single Pascal-Part .mat → HxW uint8 mask with 7-class IDs.

    Returns None if no person annotations are found (the file contains only
    non-person objects — airplane, dog, etc.). This skips the ~7K non-person
    images in the archive.
    """
    from scipy.io import loadmat  # type: ignore

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    anno = mat.get("anno")
    if anno is None:
        return None

    # anno.objects is a list of struct per object in the image
    objects = anno.objects if hasattr(anno, "objects") else []
    # When there is a single object, scipy returns it un-listed.
    if not isinstance(objects, (list, np.ndarray)):
        objects = [objects]

    person_objs = [o for o in objects if getattr(o, "class", "") == "person"]
    if not person_objs:
        return None

    # Initialize mask from the first person's object mask shape
    first_mask = person_objs[0].mask
    h, w = first_mask.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for obj in person_objs:
        parts = getattr(obj, "parts", [])
        # Unpack single-object scipy edge case
        if not isinstance(parts, (list, np.ndarray)):
            parts = [parts]
        # Empty-object edge case: parts may be an empty ndarray
        if isinstance(parts, np.ndarray) and parts.size == 0:
            continue
        for p in parts:
            part_name = getattr(p, "part_name", None)
            if part_name is None:
                continue
            cls = PART_TO_CLASS.get(str(part_name))
            if cls is None:
                continue  # unknown subpart (e.g. non-person parts)
            pm = p.mask.astype(bool)
            # Overwrite with this class ID (order doesn't really matter since
            # subparts don't overlap much within a single person).
            out[pm] = cls

    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PascalPersonPartDataset(Dataset):
    """Pascal-Person-Part with on-the-fly .mat parsing and PNG caching.

    Caches parsed masks as PNG on first access so subsequent epochs are
    fast. Pairs each .mat with its VOC2010 image by filename (e.g.
    2008_000008.mat ↔ 2008_000008.jpg).
    """

    def __init__(
        self,
        voc_root: Path,
        part_root: Path,
        cache_dir: Path,
        split: str = "train",
        crop: int = 513,
    ) -> None:
        self.voc_images = voc_root / "VOCdevkit" / "VOC2010" / "JPEGImages"
        self.part_anno = part_root / "Annotations_Part"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.crop = crop
        self.split = split

        # Walk .mat files, keep only those with a person (filter at parse time
        # on first read; cache the result).
        if not self.part_anno.is_dir():
            raise FileNotFoundError(f"Pascal-Part annotations dir not found: {self.part_anno}")
        all_mats = sorted(p.stem for p in self.part_anno.glob("*.mat"))

        # Pass 1: parse + cache (one-time cost)
        kept: list[str] = []
        marker = cache_dir / ".parse_complete"
        if marker.exists():
            # Fast path: read list from marker
            kept = [line.strip() for line in marker.read_text().splitlines() if line.strip()]
            logger.info("Loaded cached index: %d person images", len(kept))
        else:
            logger.info("Parsing %d .mat files (one-time cost, ~60s)...", len(all_mats))
            for stem in all_mats:
                cache_png = cache_dir / f"{stem}.png"
                if cache_png.exists():
                    kept.append(stem)
                    continue
                mat_path = self.part_anno / f"{stem}.mat"
                mask = _parse_pascal_part_mat(mat_path)
                if mask is None:
                    continue
                Image.fromarray(mask, mode="L").save(cache_png)
                kept.append(stem)
            marker.write_text("\n".join(kept))
            logger.info("Cached %d person images with 7-class masks", len(kept))

        # Split 90/10 train/val (deterministic)
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(kept))
        n_val = max(1, len(kept) // 10)
        val_idx = set(perm[:n_val].tolist())
        self.samples = [
            kept[i] for i in range(len(kept))
            if (i in val_idx) == (split == "val")
        ]
        logger.info("Pascal-Person-Part %s split: %d images", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.samples[idx]
        img = Image.open(self.voc_images / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.cache_dir / f"{stem}.png")

        # Random crop (train) or center crop (val). Keep mask aligned.
        c = self.crop
        w, h = img.size
        if self.split == "train":
            # Random scale 0.5..2.0 then crop
            scale = float(np.random.uniform(0.5, 2.0))
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)
            # Pad if smaller than crop
            if nw < c or nh < c:
                pad_w = max(0, c - nw)
                pad_h = max(0, c - nh)
                img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
                mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)
                nw, nh = img.size
            # Random crop
            x = int(np.random.randint(0, nw - c + 1))
            y = int(np.random.randint(0, nh - c + 1))
            img = img.crop((x, y, x + c, y + c))
            mask = mask.crop((x, y, x + c, y + c))
            # Horizontal flip
            if np.random.rand() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            # Center-crop/pad to exactly c×c for deterministic val batching
            img = TF.center_crop(TF.resize(img, c, antialias=True), [c, c])
            mask = TF.center_crop(TF.resize(mask, c, interpolation=TF.InterpolationMode.NEAREST), [c, c])

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)
        return {"image": img_t, "mask": mask_t}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_model() -> torch.nn.Module:
    """Build DeepLabV3-ResNet50, COCO-pretrained, re-head to 7 classes."""
    # COCO-pretrained head gives a reasonable starting point; replacing the
    # final conv with a 7-class layer is a standard transfer-learning pattern.
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # Swap classifier last layer (256 → 7)
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1)
    # Swap aux classifier too to avoid shape mismatch
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = torch.nn.Conv2d(aux_in, NUM_CLASSES, kernel_size=1)
    return model


def _compute_miou(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Compute mean IoU ignoring classes with no support."""
    ious: list[float] = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        inter = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Data prep ---
    data_root = Path(args.data_root)
    voc_tar = data_root / "VOCtrainval_03-May-2010.tar"
    part_tar = data_root / "pascal_part_trainval.tar.gz"
    voc_dir = data_root / "voc2010"
    part_dir = data_root / "pascal_part"
    cache_dir = data_root / "cache_7class_masks"

    logger.info("[1/4] Downloading data...")
    _download(VOC_IMAGES_URL, voc_tar)
    _download(PASCAL_PART_URL, part_tar, expected_md5=PASCAL_PART_MD5)

    logger.info("[2/4] Extracting...")
    _extract(voc_tar, voc_dir)
    _extract(part_tar, part_dir)

    logger.info("[3/4] Building dataset...")
    train_ds = PascalPersonPartDataset(voc_dir, part_dir, cache_dir, split="train")
    val_ds = PascalPersonPartDataset(voc_dir, part_dir, cache_dir, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model, optimizer, scheduler ---
    logger.info("[4/4] Training...")
    model = _build_model().to(device)

    # Separate LR: backbone gets lower LR than head (standard for transfer)
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], momentum=0.9, weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)

    def poly_lr(step: int) -> float:
        return (1.0 - step / max(total_steps, 1)) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    best_miou = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            out = model(imgs)  # dict with 'out' and 'aux'
            loss = F.cross_entropy(out["out"], masks, ignore_index=255)
            if "aux" in out:
                loss = loss + 0.4 * F.cross_entropy(out["aux"], masks, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss)
            n_batches += 1
            step += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        total_inter = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                out = model(imgs)["out"]
                preds = out.argmax(dim=1)
                for c in range(NUM_CLASSES):
                    p_c = preds == c
                    m_c = masks == c
                    total_inter[c] += (p_c & m_c).sum().item()
                    total_union[c] += (p_c | m_c).sum().item()
        per_class_iou = (total_inter / total_union.clamp(min=1)).numpy()
        miou = float(per_class_iou[total_union > 0].mean()) if (total_union > 0).any() else 0.0

        logger.info(
            "Epoch %d/%d — loss=%.4f, val mIoU=%.4f (per-class: %s)",
            epoch + 1, args.epochs, avg_loss, miou,
            " ".join(f"{x:.2f}" for x in per_class_iou),
        )

        if miou > best_miou:
            best_miou = miou
            # Save backbone state_dict only — this is the artifact Stage B loads.
            torch.save(model.backbone.state_dict(), output_path)
            logger.info("  New best mIoU %.4f — saved backbone to %s", best_miou, output_path)

    logger.info("Pretrain complete. Best val mIoU: %.4f", best_miou)
    logger.info("Backbone state_dict: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain DeepLabV3-ResNet50 on Pascal-Person-Part")
    parser.add_argument("--output", type=str, default="./checkpoints/anatomy_init/backbone.pth",
                        help="Path to save backbone state_dict")
    parser.add_argument("--data-root", type=str, default="./data_cloud/pascal_person_part",
                        help="Directory to store VOC and Pascal-Part archives")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.007)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — pretraining on CPU will be very slow")

    try:
        train(args)
    except Exception:
        logger.exception("Pretrain failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
