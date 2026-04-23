#!/usr/bin/env python3
"""Fast SAM-assisted annotation tool for segmentation pseudo-labels.

Click anywhere on the image → SAM 2 predicts a clean mask for that region →
the mask is painted with your currently-selected class. 22-class anatomy
keyboard shortcuts mean most corrections are one-click-plus-one-keypress.

Workflow:
    1. Select class with keyboard (1-9, 0, a, h, b, or Shift+digit for
       classes 10-19).
    2. Click the image to get a SAM mask for that point.
    3. The mask is painted with the selected class. Immediately.
    4. Right-click to add a "negative point" (exclude area from mask) before
       committing — useful when SAM over-segments.
    5. Enter / Space to save + advance. R to reject. U to undo last stroke.

This is ~5-10× faster than brush-painting (``scripts/review_masks.py``) on
the same corrections, because SAM does the boundary work for you.

Keyboard shortcuts (mirror ``review_masks.py``):
    1–9        classes 1..9   (head..hand_l)
    0          class 10       (shoulder_r)
    Shift+1..9 classes 11..19 (upper_arm_r..foot_r)
    a          class 20       (accessory)
    h          class 21       (hair_back)
    b          class 0        (background, eraser)
    [ / ]      decrease / increase SAM threshold (sensitivity)
    Space      commit pending mask + next image
    Enter      same as Space
    r          reject example (mark in metadata) + next
    u / Ctrl+Z undo last stroke
    Left/Right previous / next image
    Escape     save + quit

Usage::

    python3 scripts/annotate_with_sam.py \\
        --data-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse \\
        --sam-checkpoint /Volumes/TAMWoolff/data/models/sam2.1_hiera_small.pt

Or on A100 with the existing SAM install::

    python3 scripts/annotate_with_sam.py \\
        --data-dir ./data_cloud/gemini_diverse \\
        --sam-checkpoint /workspace/weights/sam2.1_hiera_large.pt \\
        --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --device cuda

Launch order-by-badness (highest-priority examples first)::

    # First generate badness ranking
    python3 scripts/audit_labels.py --data-dir ... --csv-out /tmp/audit.csv
    # Then launch the annotator with that order
    python3 scripts/annotate_with_sam.py --data-dir ... --order-csv /tmp/audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Project imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# 22-class anatomy
REGION_NAMES = [
    "background", "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
    "accessory", "hair_back",
]
REGION_COLORS = [
    (40, 40, 40),    (255, 0, 0),   (255, 128, 0), (0, 255, 0),
    (0, 200, 0),     (0, 150, 0),   (0, 0, 255),   (0, 100, 255),
    (0, 200, 255),   (100, 255, 255), (128, 0, 255), (180, 0, 255),
    (220, 0, 255),   (255, 0, 255), (255, 255, 0), (200, 200, 0),
    (150, 150, 0),   (255, 200, 0), (200, 150, 0), (150, 100, 0),
    (180, 180, 180), (180, 100, 60),
]

CANVAS_SIZE = 768
OVERLAY_ALPHA = 0.5
MAX_UNDO = 30

# Keyboard mapping → class index
KEY_TO_CLASS = {
    **{str(i): i for i in range(1, 10)},
    "0": 10,
    "a": 20, "A": 20,
    "h": 21, "H": 21,
    "b": 0,  "B": 0,
    # Shift+digit → 11..19
    "!": 11, "@": 12, "#": 13, "$": 14, "%": 15,
    "^": 16, "&": 17, "*": 18, "(": 19,
}


@dataclass
class Stroke:
    """Record of a painted mask for undo."""
    prev_mask: np.ndarray  # the full previous segmentation.png state
    class_id: int
    description: str


@dataclass
class AppState:
    example_dirs: list[Path]
    index: int = 0
    current_class: int = 1              # head
    pending_points: list[tuple[int, int]] = field(default_factory=list)
    pending_labels: list[int] = field(default_factory=list)
    pending_mask: np.ndarray | None = None  # current SAM preview
    undo_stack: list[Stroke] = field(default_factory=list)
    sam_threshold: float = 0.5


class SAMAnnotator:
    def __init__(self, args):
        self.args = args
        self.state = AppState(example_dirs=self._discover_examples(args))
        if not self.state.example_dirs:
            raise RuntimeError(f"No examples with image.png found in {args.data_dir}")

        logger.info("Loading SAM 2 predictor ...")
        self.sam_predictor = self._load_sam(args.sam_checkpoint, args.sam_config, args.device)

        # Tkinter UI
        self.root = tk.Tk()
        self.root.title("Strata — SAM annotation tool")
        self._build_ui()
        self._load_example(0)

    # ------------------------------------------------------------------
    # Example discovery
    # ------------------------------------------------------------------
    def _discover_examples(self, args) -> list[Path]:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(data_dir)

        if args.order_csv:
            # Order by badness descending (worst first)
            ordered: list[tuple[float, Path]] = []
            with open(args.order_csv) as f:
                for row in csv.DictReader(f):
                    name = row["name"]
                    badness = float(row.get("badness", 0))
                    ex_dir = data_dir / name
                    if (ex_dir / "image.png").exists():
                        ordered.append((badness, ex_dir))
            ordered.sort(key=lambda t: -t[0])
            dirs = [d for _, d in ordered]
        else:
            dirs = sorted(
                d for d in data_dir.iterdir()
                if d.is_dir() and (d / "image.png").exists()
            )

        if args.limit:
            dirs = dirs[: args.limit]

        # Skip already-reviewed if requested
        if args.only_unreviewed:
            dirs = [
                d for d in dirs
                if not self._is_reviewed(d)
            ]
        return dirs

    def _is_reviewed(self, example_dir: Path) -> bool:
        meta = example_dir / "metadata.json"
        if not meta.exists():
            return False
        try:
            return json.loads(meta.read_text()).get("manually_reviewed", False)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # SAM loading
    # ------------------------------------------------------------------
    def _load_sam(self, checkpoint: str, config: str, device: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(config, checkpoint, device=device)
        return SAM2ImagePredictor(model)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Top status bar
        self.status_var = tk.StringVar()
        self.class_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var).pack(fill="x", padx=6, pady=3)
        ttk.Label(self.root, textvariable=self.class_var,
                  font=("Helvetica", 14, "bold")).pack(fill="x", padx=6)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#222", highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)  # macOS Ctrl+click
        self.canvas.bind("<Button-2>", self._on_right_click)

        # Key bindings
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<space>", lambda e: self._commit_and_next())
        self.root.bind("<Return>", lambda e: self._commit_and_next())
        self.root.bind("<Left>", lambda e: self._goto(self.state.index - 1))
        self.root.bind("<Right>", lambda e: self._goto(self.state.index + 1))
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Command-z>", lambda e: self._undo())  # macOS
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ------------------------------------------------------------------
    # Example loading / rendering
    # ------------------------------------------------------------------
    def _load_example(self, idx: int):
        if not (0 <= idx < len(self.state.example_dirs)):
            return
        self.state.index = idx
        self.state.pending_points.clear()
        self.state.pending_labels.clear()
        self.state.pending_mask = None
        self.state.undo_stack.clear()

        ex = self.state.example_dirs[idx]
        img_path = ex / "image.png"
        seg_path = ex / "segmentation.png"

        self.src_image = Image.open(img_path).convert("RGBA")
        self.src_size = self.src_image.size  # (W, H)

        if seg_path.exists():
            self.seg = np.array(Image.open(seg_path).convert("L"))
        else:
            self.seg = np.zeros(
                (self.src_image.height, self.src_image.width), dtype=np.uint8,
            )

        # Tell SAM to embed this image (slow: ~1-2 sec first time)
        rgb = np.array(self.src_image.convert("RGB"))
        logger.info("Setting SAM image for %s (%dx%d)", ex.name, self.src_size[0], self.src_size[1])
        self.sam_predictor.set_image(rgb)

        self._refresh()

    def _refresh(self):
        """Redraw canvas with image + seg overlay + pending mask preview."""
        W, H = self.src_size
        # Scale for display
        scale = CANVAS_SIZE / max(W, H)
        dw, dh = int(W * scale), int(H * scale)
        self.display_scale = scale

        # Base image (RGB)
        disp = self.src_image.convert("RGB").resize((dw, dh), Image.BILINEAR)

        # Seg overlay
        seg_img = self._colorize_seg(self.seg).resize((dw, dh), Image.NEAREST)
        overlay = Image.blend(disp, seg_img, OVERLAY_ALPHA)

        # Pending mask preview (bright yellow outline)
        if self.state.pending_mask is not None:
            pend_vis = self._preview_mask_vis(self.state.pending_mask).resize(
                (dw, dh), Image.NEAREST,
            )
            overlay = Image.alpha_composite(
                overlay.convert("RGBA"), pend_vis.convert("RGBA"),
            )

        # Pending points
        if self.state.pending_points:
            draw = ImageDraw.Draw(overlay.convert("RGBA"))
            overlay = overlay.convert("RGBA")
            draw = ImageDraw.Draw(overlay)
            for (x, y), lbl in zip(self.state.pending_points, self.state.pending_labels):
                cx, cy = int(x * scale), int(y * scale)
                color = (0, 255, 0, 255) if lbl == 1 else (255, 0, 0, 255)
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=color)

        self.tk_img = ImageTk.PhotoImage(overlay.convert("RGB"))
        self.canvas.delete("all")
        self.canvas.create_image(CANVAS_SIZE // 2, CANVAS_SIZE // 2, image=self.tk_img)

        # Status
        ex = self.state.example_dirs[self.state.index]
        self.status_var.set(
            f"[{self.state.index + 1}/{len(self.state.example_dirs)}] {ex.name} — "
            f"{W}x{H} — undo={len(self.state.undo_stack)}"
        )
        cls = self.state.current_class
        self.class_var.set(
            f"Class {cls} — {REGION_NAMES[cls]} "
            f"(color {REGION_COLORS[cls]})"
        )

    def _colorize_seg(self, mask: np.ndarray) -> Image.Image:
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cid, color in enumerate(REGION_COLORS):
            out[mask == cid] = color
        return Image.fromarray(out)

    def _preview_mask_vis(self, mask: np.ndarray) -> Image.Image:
        """Yellow semi-transparent highlight of pending mask."""
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask] = (255, 255, 0, 140)
        return Image.fromarray(rgba)

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------
    def _canvas_to_image_xy(self, cx: int, cy: int) -> tuple[int, int]:
        W, H = self.src_size
        # The displayed image is centered in the canvas
        scale = self.display_scale
        dw, dh = int(W * scale), int(H * scale)
        ox, oy = (CANVAS_SIZE - dw) // 2, (CANVAS_SIZE - dh) // 2
        x = int((cx - ox) / scale)
        y = int((cy - oy) / scale)
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        return x, y

    def _on_left_click(self, event):
        x, y = self._canvas_to_image_xy(event.x, event.y)
        self.state.pending_points.append((x, y))
        self.state.pending_labels.append(1)  # positive point
        self._predict_pending()
        self._refresh()

    def _on_right_click(self, event):
        x, y = self._canvas_to_image_xy(event.x, event.y)
        self.state.pending_points.append((x, y))
        self.state.pending_labels.append(0)  # negative point
        self._predict_pending()
        self._refresh()

    def _predict_pending(self):
        if not self.state.pending_points:
            self.state.pending_mask = None
            return
        pts = np.array(self.state.pending_points, dtype=np.float32)
        lbls = np.array(self.state.pending_labels, dtype=np.int32)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            multimask_output=True,
        )
        # Pick the mask with highest IOU-predicted score
        best = int(np.argmax(scores))
        self.state.pending_mask = masks[best].astype(bool)

    # ------------------------------------------------------------------
    # Key handler + commit
    # ------------------------------------------------------------------
    def _on_key(self, event):
        key = event.keysym
        char = event.char
        # Class selection
        if char in KEY_TO_CLASS:
            self.state.current_class = KEY_TO_CLASS[char]
            self._refresh()
            # If pending mask exists, commit it immediately with new class
            if self.state.pending_mask is not None:
                self._commit_pending()
            return
        if key == "u" or key == "U":
            self._undo()
        elif key == "r" or key == "R":
            self._reject_and_next()
        elif key == "bracketleft":
            self.state.sam_threshold = max(0.1, self.state.sam_threshold - 0.1)
            logger.info("SAM threshold %.1f", self.state.sam_threshold)
        elif key == "bracketright":
            self.state.sam_threshold = min(1.0, self.state.sam_threshold + 0.1)
            logger.info("SAM threshold %.1f", self.state.sam_threshold)

    def _commit_pending(self):
        """Apply current pending mask to segmentation with current class."""
        if self.state.pending_mask is None:
            return
        prev = self.seg.copy()
        self.seg[self.state.pending_mask] = self.state.current_class
        self.state.undo_stack.append(Stroke(
            prev_mask=prev,
            class_id=self.state.current_class,
            description=f"paint class {self.state.current_class} ({REGION_NAMES[self.state.current_class]})",
        ))
        if len(self.state.undo_stack) > MAX_UNDO:
            self.state.undo_stack.pop(0)
        # Clear pending
        self.state.pending_points.clear()
        self.state.pending_labels.clear()
        self.state.pending_mask = None
        self._refresh()

    def _undo(self):
        if not self.state.undo_stack:
            return
        stroke = self.state.undo_stack.pop()
        self.seg = stroke.prev_mask
        self._refresh()

    def _save_current(self, reviewed: bool = True, rejected: bool = False):
        ex = self.state.example_dirs[self.state.index]
        seg_path = ex / "segmentation.png"
        Image.fromarray(self.seg).save(seg_path)
        # Metadata
        meta_path = ex / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"id": ex.name}
        meta["manually_reviewed"] = reviewed
        meta["manually_rejected"] = rejected
        meta["segmentation_source"] = "manual_sam_review"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    def _commit_and_next(self):
        if self.state.pending_mask is not None:
            self._commit_pending()
        self._save_current(reviewed=True, rejected=False)
        self._goto(self.state.index + 1)

    def _reject_and_next(self):
        self._save_current(reviewed=True, rejected=True)
        self._goto(self.state.index + 1)

    def _goto(self, idx: int):
        if 0 <= idx < len(self.state.example_dirs):
            self._load_example(idx)
        elif idx >= len(self.state.example_dirs):
            logger.info("Reached end of examples. Exiting.")
            self._quit()

    def _quit(self):
        if self.state.pending_mask is not None:
            self._commit_pending()
        self._save_current(reviewed=True, rejected=False)
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main() -> int:
    p = argparse.ArgumentParser(description="SAM-assisted segmentation annotation tool.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--sam-checkpoint",
                   default=os.environ.get("SAM2_CHECKPOINT",
                                          "/Volumes/TAMWoolff/data/models/sam2.1_hiera_small.pt"))
    p.add_argument("--sam-config",
                   default=os.environ.get("SAM2_CONFIG",
                                          "configs/sam2.1/sam2.1_hiera_s.yaml"))
    p.add_argument("--device",
                   default="mps" if sys.platform == "darwin" else "cuda",
                   help="torch device (cuda / mps / cpu)")
    p.add_argument("--order-csv", type=Path, default=None,
                   help="Optional CSV from audit_labels.py; examples will be "
                        "shown worst-first.")
    p.add_argument("--only-unreviewed", action="store_true",
                   help="Skip examples whose metadata already has "
                        "manually_reviewed=true.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    app = SAMAnnotator(args)
    logger.info(
        "Loaded %d examples. Start annotating. Esc to save+quit.",
        len(app.state.example_dirs),
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
