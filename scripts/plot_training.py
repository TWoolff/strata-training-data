#!/usr/bin/env python3
"""
Plot training curves from log files.

Usage:
    # Plot all models from a log directory
    python scripts/plot_training.py logs/train_20260306_083940/

    # Compare multiple runs
    python scripts/plot_training.py logs/train_20260306_083940/ logs/run3_*/train_all.log

    # Plot from pasted terminal output (run 3 style)
    python scripts/plot_training.py --paste run3_terminal.txt

    # Save to file instead of showing
    python scripts/plot_training.py logs/train_20260306_083940/ -o output/training_curves.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# --- Log parsers -----------------------------------------------------------

SEG_RE = re.compile(
    r"Epoch (\d+)/(\d+) .+ train_loss=([\d.]+), val_loss=([\d.]+), mIoU=([\d.]+), lr=([\d.e+-]+)"
)

JOINT_RE = re.compile(
    r"Epoch (\d+)/(\d+) .+ train_loss=([\d.]+), val_loss=([\d.]+), mean_err=([\d.]+), pres_acc=([\d.]+), lr=([\d.e+-]+)"
)

WEIGHT_RE = re.compile(
    r"Epoch (\d+)/(\d+) .+ train_loss=([\d.]+), val_loss=([\d.]+), mae=([\d.]+), conf_acc=([\d.]+), lr=([\d.e+-]+)"
)

INPAINT_RE = re.compile(
    r"Epoch (\d+)/(\d+) .+ lr=([\d.e+-]+) \| train/l1=([\d.]+) \| val/l1=([\d.]+) \| val/loss=([\d.]+)"
)


def parse_log(filepath: Path) -> dict:
    """Parse a training log file and return structured data."""
    text = filepath.read_text()
    name = filepath.stem  # e.g. "segmentation", "joints"

    # Try each pattern
    seg_matches = SEG_RE.findall(text)
    if seg_matches:
        epochs = [int(m[0]) for m in seg_matches]
        return {
            "name": name,
            "type": "segmentation",
            "epochs": epochs,
            "train_loss": [float(m[2]) for m in seg_matches],
            "val_loss": [float(m[3]) for m in seg_matches],
            "mIoU": [float(m[4]) for m in seg_matches],
            "lr": [float(m[5]) for m in seg_matches],
        }

    joint_matches = JOINT_RE.findall(text)
    if joint_matches:
        epochs = [int(m[0]) for m in joint_matches]
        return {
            "name": name,
            "type": "joints",
            "epochs": epochs,
            "train_loss": [float(m[2]) for m in joint_matches],
            "val_loss": [float(m[3]) for m in joint_matches],
            "mean_err": [float(m[4]) for m in joint_matches],
            "pres_acc": [float(m[5]) for m in joint_matches],
            "lr": [float(m[6]) for m in joint_matches],
        }

    weight_matches = WEIGHT_RE.findall(text)
    if weight_matches:
        epochs = [int(m[0]) for m in weight_matches]
        return {
            "name": name,
            "type": "weights",
            "epochs": epochs,
            "train_loss": [float(m[2]) for m in weight_matches],
            "val_loss": [float(m[3]) for m in weight_matches],
            "mae": [float(m[4]) for m in weight_matches],
            "conf_acc": [float(m[5]) for m in weight_matches],
            "lr": [float(m[6]) for m in weight_matches],
        }

    inpaint_matches = INPAINT_RE.findall(text)
    if inpaint_matches:
        epochs = [int(m[0]) for m in inpaint_matches]
        return {
            "name": name,
            "type": "inpainting",
            "epochs": epochs,
            "train_l1": [float(m[3]) for m in inpaint_matches],
            "val_l1": [float(m[4]) for m in inpaint_matches],
            "val_loss": [float(m[5]) for m in inpaint_matches],
            "lr": [float(m[2]) for m in inpaint_matches],
        }

    return None


def parse_directory(dirpath: Path) -> list[dict]:
    """Parse all log files in a directory."""
    results = []
    for log_file in sorted(dirpath.glob("*.log")):
        data = parse_log(log_file)
        if data:
            results.append(data)
    return results


# --- Plotting ---------------------------------------------------------------

COLORS = {
    "run1": "#2196F3",
    "run2": "#FF9800",
    "run3": "#4CAF50",
    "run4": "#E91E63",
    "run5": "#9C27B0",
}


def plot_segmentation(ax_loss, ax_metric, runs: list[tuple[str, dict]]):
    """Plot segmentation curves."""
    for label, data in runs:
        color = COLORS.get(label, "#666")
        ax_loss.plot(data["epochs"], data["train_loss"], "-", color=color, alpha=0.5, label=f"{label} train")
        ax_loss.plot(data["epochs"], data["val_loss"], "--", color=color, alpha=0.8, label=f"{label} val")
        ax_metric.plot(data["epochs"], data["mIoU"], "-", color=color, linewidth=2, label=label)
        best_idx = int(np.argmax(data["mIoU"]))
        ax_metric.annotate(
            f'{data["mIoU"][best_idx]:.4f}',
            (data["epochs"][best_idx], data["mIoU"][best_idx]),
            textcoords="offset points", xytext=(5, 5), fontsize=8, color=color,
        )
    ax_loss.set_title("Segmentation — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_metric.set_title("Segmentation — mIoU")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("mIoU")
    ax_metric.legend(fontsize=7)
    ax_metric.grid(True, alpha=0.3)


def plot_joints(ax_loss, ax_metric, runs: list[tuple[str, dict]]):
    """Plot joint refinement curves."""
    for label, data in runs:
        color = COLORS.get(label, "#666")
        ax_loss.plot(data["epochs"], data["train_loss"], "-", color=color, alpha=0.5, label=f"{label} train")
        ax_loss.plot(data["epochs"], data["val_loss"], "--", color=color, alpha=0.8, label=f"{label} val")
        ax_metric.plot(data["epochs"], data["mean_err"], "-", color=color, linewidth=2, label=label)
        best_idx = int(np.argmin(data["mean_err"]))
        ax_metric.annotate(
            f'{data["mean_err"][best_idx]:.6f}',
            (data["epochs"][best_idx], data["mean_err"][best_idx]),
            textcoords="offset points", xytext=(5, 5), fontsize=8, color=color,
        )
    ax_loss.set_title("Joint Refinement — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_metric.set_title("Joint Refinement — Mean Offset Error")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Mean Error")
    ax_metric.legend(fontsize=7)
    ax_metric.grid(True, alpha=0.3)
    ax_metric.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))


def plot_weights(ax_loss, ax_metric, runs: list[tuple[str, dict]]):
    """Plot weight prediction curves."""
    for label, data in runs:
        color = COLORS.get(label, "#666")
        ax_loss.plot(data["epochs"], data["train_loss"], "-", color=color, alpha=0.5, label=f"{label} train")
        ax_loss.plot(data["epochs"], data["val_loss"], "--", color=color, alpha=0.8, label=f"{label} val")
        ax_metric.plot(data["epochs"], data["mae"], "-", color=color, linewidth=2, label=label)
        best_idx = int(np.argmin(data["mae"]))
        ax_metric.annotate(
            f'{data["mae"][best_idx]:.6f}',
            (data["epochs"][best_idx], data["mae"][best_idx]),
            textcoords="offset points", xytext=(5, 5), fontsize=8, color=color,
        )
    ax_loss.set_title("Weight Prediction — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_metric.set_title("Weight Prediction — MAE")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("MAE")
    ax_metric.legend(fontsize=7)
    ax_metric.grid(True, alpha=0.3)


def plot_inpainting(ax_loss, ax_metric, runs: list[tuple[str, dict]]):
    """Plot inpainting curves."""
    for label, data in runs:
        color = COLORS.get(label, "#666")
        ax_loss.plot(data["epochs"], data["val_loss"], "-", color=color, linewidth=2, label=f"{label} val_loss")
        ax_metric.plot(data["epochs"], data["val_l1"], "-", color=color, linewidth=2, label=f"{label} val_l1")
    ax_loss.set_title("Inpainting — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_metric.set_title("Inpainting — L1")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("L1")
    ax_metric.legend(fontsize=7)
    ax_metric.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log files")
    parser.add_argument("paths", nargs="+", help="Log directories or files to plot")
    parser.add_argument("-o", "--output", help="Save plot to file instead of showing")
    parser.add_argument("--labels", nargs="*", help="Labels for each path (default: run1, run2, ...)")
    args = parser.parse_args()

    # Collect all parsed data grouped by model type
    by_type: dict[str, list[tuple[str, dict]]] = {}

    for i, path_str in enumerate(args.paths):
        path = Path(path_str)
        label = args.labels[i] if args.labels and i < len(args.labels) else f"run{i + 1}"

        if path.is_dir():
            parsed = parse_directory(path)
        elif path.is_file():
            parsed = [parse_log(path)]
            parsed = [p for p in parsed if p]
        else:
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue

        for data in parsed:
            model_type = data["type"]
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append((label, data))

    if not by_type:
        print("No valid training data found in the provided paths.", file=sys.stderr)
        sys.exit(1)

    # Filter out broken inpainting (all zeros)
    if "inpainting" in by_type:
        by_type["inpainting"] = [
            (label, data) for label, data in by_type["inpainting"]
            if any(v > 0 for v in data.get("val_l1", [0]))
        ]
        if not by_type["inpainting"]:
            del by_type["inpainting"]

    # Layout: one row per model type, 2 columns (loss + metric)
    plot_funcs = {
        "segmentation": plot_segmentation,
        "joints": plot_joints,
        "weights": plot_weights,
        "inpainting": plot_inpainting,
    }

    # Determine which model types to plot (in order)
    type_order = ["segmentation", "joints", "weights", "inpainting"]
    types_to_plot = [t for t in type_order if t in by_type]

    n_rows = len(types_to_plot)
    if n_rows == 0:
        print("No plottable data found.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, model_type in enumerate(types_to_plot):
        plot_func = plot_funcs[model_type]
        plot_func(axes[row, 0], axes[row, 1], by_type[model_type])

    fig.suptitle("Strata Training Curves", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
