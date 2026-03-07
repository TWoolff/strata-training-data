"""Benchmark Strata models on the Gemini test set.

Runs the 7 curated Gemini character images through all available models
(segmentation, joints, draw order, surface normals, depth) and produces a
labeled overview grid.

Usage::

    # Auto-increments: training01_overview.png, training02_overview.png, ...
    python run_benchmark.py

    # Explicit name
    python run_benchmark.py --name training02

    # Skip normals/depth (faster)
    python run_benchmark.py --no-normals --no-depth

    # Custom test images (default: external HD gemini folder)
    python run_benchmark.py --input-dir ./my_test_images
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import RENDER_RESOLUTION
from run_seg_enrich import load_seg_model, predict_segmentation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path("/Volumes/TAMWoolff/data/preprocessed/gemini")
OUTPUT_DIR = Path("./output")
BENCHMARK_DIR = OUTPUT_DIR / "benchmark"

REGION_COLORS = [
    (0, 0, 0), (255, 80, 80), (255, 160, 80), (255, 200, 40),
    (200, 255, 40), (80, 255, 80), (40, 200, 255), (80, 160, 255),
    (80, 80, 255), (160, 80, 255), (255, 40, 200), (255, 80, 160),
    (255, 40, 120), (255, 80, 200), (40, 255, 160), (40, 255, 255),
    (40, 200, 200), (200, 40, 255), (255, 40, 255), (200, 40, 200),
    (180, 180, 180), (120, 80, 40),
]

REGION_NAMES = [
    "bg", "head", "neck", "chest", "spine", "hips",
    "shldr_l", "arm_l", "farm_l", "hand_l",
    "shldr_r", "arm_r", "farm_r", "hand_r",
    "uleg_l", "lleg_l", "foot_l",
    "uleg_r", "lleg_r", "foot_r",
    "accsry", "hair",
]

SKELETON_BONES = [
    ("head", "neck"), ("neck", "chest"), ("chest", "spine"), ("spine", "hips"),
    ("shoulder_l", "upper_arm_l"), ("upper_arm_l", "forearm_l"), ("forearm_l", "hand_l"),
    ("shoulder_r", "upper_arm_r"), ("upper_arm_r", "forearm_r"), ("forearm_r", "hand_r"),
    ("chest", "shoulder_l"), ("chest", "shoulder_r"),
    ("hips", "upper_leg_l"), ("upper_leg_l", "lower_leg_l"), ("lower_leg_l", "foot_l"),
    ("hips", "upper_leg_r"), ("upper_leg_r", "lower_leg_r"), ("lower_leg_r", "foot_r"),
]

RTMPOSE_DET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
RTMPOSE_POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"

MARIGOLD_NORMALS_MODEL = "prs-eth/marigold-normals-lcm-v0-1"
MARIGOLD_DEPTH_MODEL = "prs-eth/marigold-depth-lcm-v1-0"


def conf_color(conf: float) -> tuple[int, int, int]:
    """Green (high) -> Yellow (mid) -> Red (low)."""
    if conf >= 0.6:
        return (0, 230, 0)
    elif conf >= 0.3:
        t = (conf - 0.3) / 0.3
        return (int(255 * (1 - t)), int(230 * t), 0)
    else:
        t = conf / 0.3
        return (255, int(100 * t), 0)


def outlined_text(draw: ImageDraw.Draw, xy, text, fill=(255, 255, 255), outline=(0, 0, 0)):
    x, y = xy
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx or dy:
                draw.text((x + dx, y + dy), text, fill=outline)
    draw.text((x, y), text, fill=fill)


def next_training_name(output_dir: Path) -> str:
    """Find next trainingNN name by scanning existing files."""
    existing = list(output_dir.glob("training*_overview.png"))
    nums = []
    for f in existing:
        m = re.match(r"training(\d+)_overview\.png", f.name)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums, default=0) + 1
    return f"training{next_num:02d}"


def run_segmentation(input_images: list[Path], checkpoint: Path, device, resolution: int) -> dict[str, dict]:
    """Run segmentation + rembg on all images. Returns {name: {img, seg, draw_order}}."""
    import torch

    model = load_seg_model(checkpoint, device)

    rembg_session = None
    try:
        from rembg import new_session
        rembg_session = new_session("u2net")
        logger.info("rembg loaded")
    except ImportError:
        logger.warning("rembg not installed — skipping background removal")

    results = {}
    for img_path in input_images:
        name = img_path.stem
        img = Image.open(img_path).convert("RGBA")

        if rembg_session is not None:
            from rembg import remove
            img = remove(img, session=rembg_session)

        img_resized = img.resize((resolution, resolution), Image.BILINEAR)

        # Save temp for predict_segmentation
        tmp_dir = BENCHMARK_DIR / f"_tmp_{name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "image.png"
        img_resized.save(tmp_path)

        seg_mask, draw_order, confidence = predict_segmentation(model, tmp_path, device, resolution)

        # Zero out bg from alpha
        alpha = np.array(img_resized)[:, :, 3]
        seg_mask[alpha < 10] = 0
        draw_order[alpha < 10] = 0

        results[name] = {
            "img": img_resized,
            "seg": seg_mask,
            "draw_order": draw_order,
        }

        # Save outputs for joint estimation
        img_resized.save(tmp_dir / "image.png")
        Image.fromarray(seg_mask).save(tmp_dir / "segmentation.png")
        Image.fromarray(draw_order).save(tmp_dir / "draw_order.png")

    return results


def run_joints(names: list[str]) -> dict[str, dict]:
    """Run RTMPose on benchmark images. Returns {name: joints_dict}."""
    from pipeline.pose_estimator import enrich_example, load_model

    model = load_model(
        RTMPOSE_DET_URL, RTMPOSE_POSE_URL,
        device="cpu", backend="onnxruntime",
        det_input_size=(640, 640), pose_input_size=(192, 256),
    )

    results = {}
    for name in names:
        example_dir = BENCHMARK_DIR / f"_tmp_{name}"
        enrich_example(model, example_dir, (512, 512), confidence_threshold=0.3)
        joints_path = example_dir / "joints.json"
        if joints_path.exists():
            results[name] = json.loads(joints_path.read_text())
        else:
            results[name] = {"joints": {}}

    return results


def run_normals(seg_results: dict[str, dict], device) -> dict[str, Image.Image]:
    """Run Marigold surface normal estimation. Returns {name: normal_pil_image}."""
    import torch
    from diffusers import MarigoldNormalsPipeline

    pipe = MarigoldNormalsPipeline.from_pretrained(
        MARIGOLD_NORMALS_MODEL,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)

    results = {}
    for name, data in seg_results.items():
        img = data["img"]
        # Marigold needs RGB
        rgb = Image.new("RGB", img.size, (128, 128, 255))
        rgb.paste(img, mask=img.split()[3])

        output = pipe(rgb, num_inference_steps=4)
        normal_np = output.prediction[0]  # [H, W, 3] float32 in [-1, 1]

        # Map [-1,1] to [0,255]
        normal_uint8 = ((normal_np + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        if normal_uint8.ndim == 3 and normal_uint8.shape[0] == 3:
            normal_uint8 = normal_uint8.transpose(1, 2, 0)

        # Mask out background
        alpha = np.array(img)[:, :, 3]
        normal_uint8[alpha < 10] = [20, 20, 25]

        results[name] = Image.fromarray(normal_uint8)

    return results


def run_depth(seg_results: dict[str, dict], device) -> dict[str, Image.Image]:
    """Run Marigold depth estimation. Returns {name: depth_pil_image}."""
    import torch
    from diffusers import MarigoldDepthPipeline

    pipe = MarigoldDepthPipeline.from_pretrained(
        MARIGOLD_DEPTH_MODEL,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)

    results = {}
    for name, data in seg_results.items():
        img = data["img"]
        rgb = Image.new("RGB", img.size, (128, 128, 128))
        rgb.paste(img, mask=img.split()[3])

        output = pipe(rgb, num_inference_steps=4)
        depth_np = output.prediction[0].squeeze()  # [H, W] float32 in [0, 1]
        depth_uint8 = (depth_np * 255).clip(0, 255).astype(np.uint8)

        # Mask out background
        alpha = np.array(img)[:, :, 3]
        depth_uint8[alpha < 10] = 0

        results[name] = Image.fromarray(depth_uint8, "L")

    return results


def build_overview(
    names: list[str],
    seg_results: dict[str, dict],
    joint_results: dict[str, dict],
    normal_results: dict[str, Image.Image] | None,
    depth_results: dict[str, Image.Image] | None,
    run_name: str,
) -> Image.Image:
    """Build the overview grid."""
    cell = 512
    has_normals = normal_results is not None and len(normal_results) > 0
    has_depth = depth_results is not None and len(depth_results) > 0
    cols = 4 + (1 if has_normals else 0) + (1 if has_depth else 0)
    rows = len(names)
    header_h = 36
    footer_h = 24
    total_h = header_h + cell * rows + footer_h

    grid = Image.new("RGBA", (cell * cols, total_h), (20, 20, 25, 255))
    gd = ImageDraw.Draw(grid)

    # Header
    labels = ["Original", "Segmentation (22-class)", "Joints (RTMPose)", "Draw Order"]
    if has_normals:
        labels.append("Surface Normals")
    if has_depth:
        labels.append("Depth (Marigold)")
    for i, label in enumerate(labels):
        x0 = i * cell
        gd.rectangle([x0, 0, x0 + cell - 1, header_h - 1], fill=(40, 40, 50))
        outlined_text(gd, (x0 + cell // 2 - len(label) * 3, 10), label, fill=(220, 220, 220))

    for row, name in enumerate(names):
        y_off = header_h + row * cell
        data = seg_results[name]
        img = data["img"]
        seg = data["seg"]
        draw_order_arr = data["draw_order"]
        joints = joint_results.get(name, {}).get("joints", {})

        # Col 0: Original
        col0 = Image.new("RGBA", (cell, cell), (20, 20, 25, 255))
        col0 = Image.alpha_composite(col0, img)
        c0d = ImageDraw.Draw(col0)
        outlined_text(c0d, (8, 8), name.upper(), fill=(255, 255, 255))
        grid.paste(col0, (0, y_off))

        # Col 1: Segmentation with labels
        seg_color = np.zeros((cell, cell, 4), dtype=np.uint8)
        for rid in range(1, 22):
            mask = seg == rid
            if mask.sum() == 0:
                continue
            r, g, b = REGION_COLORS[rid]
            seg_color[mask] = [r, g, b, 180]

        col1 = Image.new("RGBA", (cell, cell), (20, 20, 25, 255))
        col1 = Image.alpha_composite(col1, img)
        col1 = Image.alpha_composite(col1, Image.fromarray(seg_color, "RGBA"))
        s_draw = ImageDraw.Draw(col1)

        for rid in range(1, 22):
            mask = seg == rid
            if mask.sum() < 100:
                continue
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            rname = REGION_NAMES[rid]
            r, g, b = REGION_COLORS[rid]
            outlined_text(s_draw, (cx - len(rname) * 3, cy - 5), rname, fill=(r, g, b))

        grid.paste(col1, (cell, y_off))

        # Col 2: Joints (all shown, color = confidence)
        col2 = Image.new("RGBA", (cell, cell), (20, 20, 25, 255))
        col2 = Image.alpha_composite(col2, img)
        col2 = Image.alpha_composite(col2, Image.new("RGBA", (cell, cell), (0, 0, 0, 140)))
        jd = ImageDraw.Draw(col2)

        for j1n, j2n in SKELETON_BONES:
            j1, j2 = joints.get(j1n), joints.get(j2n)
            if not j1 or not j2:
                continue
            x1, y1 = j1["position"]
            x2, y2 = j2["position"]
            avg_c = (j1["confidence"] + j2["confidence"]) / 2
            color = conf_color(avg_c)
            jd.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=9)
            jd.line([(x1, y1), (x2, y2)], fill=color, width=5)

        for jname, jdata in joints.items():
            x, y = jdata["position"]
            conf = jdata["confidence"]
            color = conf_color(conf)
            jd.ellipse([x - 9, y - 9, x + 9, y + 9], fill=color, outline=(255, 255, 255), width=2)

        # Legend
        jd.rectangle([8, 480, 200, 508], fill=(0, 0, 0, 180))
        jd.ellipse([12, 490, 20, 498], fill=(0, 230, 0))
        jd.text((24, 488), ">0.6", fill=(200, 200, 200))
        jd.ellipse([72, 490, 80, 498], fill=(255, 200, 0))
        jd.text((84, 488), ">0.3", fill=(200, 200, 200))
        jd.ellipse([132, 490, 140, 498], fill=(255, 50, 0))
        jd.text((144, 488), "<0.3", fill=(200, 200, 200))

        grid.paste(col2, (cell * 2, y_off))

        # Col 3: Draw order (warm colormap, bg masked)
        t = draw_order_arr.astype(np.float32) / 255.0
        do_color = np.zeros((cell, cell, 4), dtype=np.uint8)
        do_color[:, :, 0] = (t * 255).astype(np.uint8)
        do_color[:, :, 1] = (t * 220).astype(np.uint8)
        do_color[:, :, 2] = ((1 - t) * 100).astype(np.uint8)
        alpha = np.array(img)[:, :, 3]
        do_color[alpha < 10] = [20, 20, 25, 255]
        do_color[:, :, 3] = 255
        grid.paste(Image.fromarray(do_color, "RGBA"), (cell * 3, y_off))

        # Col 4+: Surface normals and depth
        extra_col = 4
        if has_normals and name in normal_results:
            normal_img = normal_results[name].convert("RGBA")
            grid.paste(normal_img, (cell * extra_col, y_off))
            extra_col += 1
        if has_depth and name in depth_results:
            depth_img = depth_results[name].convert("L")
            # Apply a blue-white colormap for depth visualization
            d_arr = np.array(depth_img).astype(np.float32) / 255.0
            d_color = np.zeros((cell, cell, 4), dtype=np.uint8)
            d_color[:, :, 0] = (d_arr * 200).astype(np.uint8)
            d_color[:, :, 1] = (d_arr * 220).astype(np.uint8)
            d_color[:, :, 2] = (80 + d_arr * 175).astype(np.uint8)
            img_alpha = np.array(img)[:, :, 3]
            d_color[img_alpha < 10] = [20, 20, 25, 255]
            d_color[:, :, 3] = 255
            grid.paste(Image.fromarray(d_color, "RGBA"), (cell * extra_col, y_off))
            extra_col += 1

    # Footer
    gd.rectangle([0, total_h - footer_h, cell * cols, total_h], fill=(40, 40, 50))
    outlined_text(gd, (8, total_h - footer_h + 6), f"{run_name} | Strata Model Benchmark", fill=(160, 160, 170))

    return grid


def cleanup_tmp():
    """Remove temporary benchmark directories."""
    import shutil
    for d in BENCHMARK_DIR.glob("_tmp_*"):
        if d.is_dir():
            shutil.rmtree(d)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Strata models on Gemini test set.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help=f"Directory of test images (default: {DEFAULT_INPUT})")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/segmentation/best.pt"),
                        help="Segmentation model checkpoint.")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: auto-increment trainingNN)")
    parser.add_argument("--resolution", type=int, default=RENDER_RESOLUTION)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-normals", action="store_true",
                        help="Skip surface normal estimation (faster).")
    parser.add_argument("--no-depth", action="store_true",
                        help="Skip depth estimation (faster).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    run_name = args.name or next_training_name(OUTPUT_DIR)

    images = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and not p.name.startswith("._")
    )
    names = [p.stem for p in images]

    print(f"Benchmark: {run_name}")
    print(f"  Images: {len(images)} from {args.input_dir}")
    print(f"  Device: {device}")
    print(f"  Normals: {'skip' if args.no_normals else 'Marigold LCM'}")
    print(f"  Depth:   {'skip' if args.no_depth else 'Marigold LCM'}")
    print()

    start = time.monotonic()

    print("Running segmentation + rembg...")
    seg_results = run_segmentation(images, args.checkpoint, device, args.resolution)

    print("Running RTMPose joints...")
    joint_results = run_joints(names)

    normal_results = None
    if not args.no_normals:
        print("Running Marigold surface normals...")
        normal_results = run_normals(seg_results, device)

    depth_results = None
    if not args.no_depth:
        print("Running Marigold depth estimation...")
        depth_results = run_depth(seg_results, device)

    print("Building overview grid...")
    grid = build_overview(names, seg_results, joint_results, normal_results, depth_results, run_name)

    out_path = OUTPUT_DIR / f"{run_name}_overview.png"
    grid.save(out_path)

    cleanup_tmp()

    elapsed = time.monotonic() - start
    print(f"\nBenchmark complete:")
    print(f"  Output: {out_path}")
    print(f"  Size:   {out_path.stat().st_size / 1024:.0f} KB")
    print(f"  Time:   {elapsed:.1f}s")


if __name__ == "__main__":
    main()
