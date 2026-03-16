#!/usr/bin/env python3
"""Tkinter-based mask correction UI for reviewing pseudo-labeled segmentation masks.

Displays each image with a colored mask overlay. Paint corrections with a
circular brush, accept/reject examples, and navigate between images. Saves
corrected masks back as 8-bit grayscale segmentation.png (region IDs 0-21).

Usage::

    python scripts/review_masks.py --data-dir ./output/gemini_corrected

    # Jump to first unreviewed example
    python scripts/review_masks.py --data-dir ./output/gemini_corrected --only-needs-review

    # Start from a specific example index
    python scripts/review_masks.py --data-dir ./output/gemini_corrected --start-from 50

Keyboard shortcuts:
    1-9, 0      Select region 1-10 (head..shoulder_r)
    Shift+1-9   Select region 11-19 (upper_arm_r..foot_r)
    a           Select accessory (20)
    h           Select hair_back (21)
    b           Select background (0)
    [ / ]       Decrease / increase brush size
    Space       Toggle display mode (overlay / mask / image)
    Ctrl+Z      Undo last stroke
    Enter       Accept and go to next
    r           Reject and go to next
    Left/Right  Navigate prev/next
    Escape      Save and quit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.config import REGION_COLORS, REGION_NAMES

logger = logging.getLogger(__name__)

# Display settings
CANVAS_SIZE = 768
BRUSH_SIZES = [3, 5, 10, 20, 40, 80]
DEFAULT_BRUSH_IDX = 3  # 20px
OVERLAY_ALPHA = 0.5
MAX_UNDO = 20

# Build color LUT for fast mask colorization
COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for _rid, _rgb in REGION_COLORS.items():
    if 0 <= _rid < 256:
        COLOR_LUT[_rid] = _rgb


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Region-ID mask → RGB color image."""
    return COLOR_LUT[mask]


def overlay_image_mask(image_rgba: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend colored mask overlay onto RGBA image, return RGB."""
    img_rgb = image_rgba[:, :, :3].astype(np.float32)
    colored = colorize_mask(mask).astype(np.float32)
    # Only blend where mask > 0 (foreground)
    fg = mask > 0
    blended = img_rgb.copy()
    blended[fg] = (1 - alpha) * img_rgb[fg] + alpha * colored[fg]
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Review state
# ---------------------------------------------------------------------------


@dataclass
class ReviewState:
    data_dir: Path
    examples: list[str] = field(default_factory=list)
    current_idx: int = 0
    manifest: dict = field(default_factory=dict)

    def load(self, only_needs_review: bool = False) -> None:
        """Discover examples and load manifest."""
        # Load manifest
        manifest_path = self.data_dir / "review_manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"total": 0, "reviewed": 0, "rejected": 0, "needs_review": 0, "examples": {}}

        # Discover example directories
        all_examples = sorted(
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "image.png").exists()
            and not d.name.startswith(".")
        )

        if only_needs_review:
            self.examples = [
                e for e in all_examples
                if self.manifest.get("examples", {}).get(e, {}).get("status", "needs_review") == "needs_review"
            ]
        else:
            self.examples = all_examples

        # Ensure all examples are in manifest
        for name in all_examples:
            if name not in self.manifest.get("examples", {}):
                self.manifest.setdefault("examples", {})[name] = {"status": "needs_review"}

    def save_manifest(self) -> None:
        examples = self.manifest.get("examples", {})
        self.manifest["total"] = len(examples)
        self.manifest["reviewed"] = sum(1 for e in examples.values() if e["status"] == "reviewed")
        self.manifest["rejected"] = sum(1 for e in examples.values() if e["status"] == "rejected")
        self.manifest["needs_review"] = sum(1 for e in examples.values() if e["status"] == "needs_review")
        path = self.data_dir / "review_manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")

    def get_status(self, name: str) -> str:
        return self.manifest.get("examples", {}).get(name, {}).get("status", "needs_review")

    def set_status(self, name: str, status: str) -> None:
        self.manifest.setdefault("examples", {})[name] = {"status": status}

    @property
    def current_name(self) -> str:
        if not self.examples:
            return ""
        return self.examples[self.current_idx]

    @property
    def current_dir(self) -> Path:
        return self.data_dir / self.current_name


# ---------------------------------------------------------------------------
# Mask editor
# ---------------------------------------------------------------------------


class MaskEditor:
    """Manages the numpy mask array with undo support."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask.copy()
        self.undo_stack: list[np.ndarray] = []
        self.modified = False

    def push_undo(self) -> None:
        if len(self.undo_stack) >= MAX_UNDO:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.mask.copy())

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.mask = self.undo_stack.pop()
        self.modified = True
        return True

    def paint(self, x: int, y: int, region_id: int, radius: int) -> None:
        """Paint a circle at (x, y) with the given region ID."""
        h, w = self.mask.shape
        yy, xx = np.ogrid[-y:h - y, -x:w - x]
        circle = xx * xx + yy * yy <= radius * radius
        self.mask[circle] = region_id
        self.modified = True

    def pick(self, x: int, y: int) -> int:
        """Return region ID at pixel (x, y)."""
        h, w = self.mask.shape
        if 0 <= x < w and 0 <= y < h:
            return int(self.mask[y, x])
        return 0


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class ReviewApp:
    def __init__(self, state: ReviewState, start_from: int = 0):
        self.state = state
        self.state.current_idx = min(start_from, len(state.examples) - 1) if state.examples else 0

        self.selected_region: int = 1  # head
        self.brush_idx: int = DEFAULT_BRUSH_IDX
        self.display_mode: int = 0  # 0=overlay, 1=mask, 2=image
        self.painting: bool = False

        # Current image/mask data
        self.image_rgba: np.ndarray | None = None
        self.editor: MaskEditor | None = None

        # Build UI
        self.root = tk.Tk()
        self.root.title("Strata Mask Review")
        self.root.configure(bg="#2b2b2b")
        self.root.resizable(False, False)

        self._build_ui()
        self._bind_keys()

        if state.examples:
            self._load_current()

    def _build_ui(self) -> None:
        # Top bar
        top = tk.Frame(self.root, bg="#2b2b2b")
        top.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.progress_label = tk.Label(top, text="", fg="#aaa", bg="#2b2b2b", font=("Menlo", 12))
        self.progress_label.pack(side=tk.LEFT)

        self.name_label = tk.Label(top, text="", fg="#fff", bg="#2b2b2b", font=("Menlo", 12, "bold"))
        self.name_label.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(top, text="", fg="#ff0", bg="#2b2b2b", font=("Menlo", 12))
        self.status_label.pack(side=tk.RIGHT)

        # Middle: canvas + region panel
        mid = tk.Frame(self.root, bg="#2b2b2b")
        mid.pack(fill=tk.BOTH, padx=8)

        # Canvas
        self.canvas = tk.Canvas(mid, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#1a1a1a",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Region selector panel
        panel = tk.Frame(mid, bg="#2b2b2b", width=200)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

        tk.Label(panel, text="Regions", fg="#aaa", bg="#2b2b2b", font=("Menlo", 11, "bold")).pack(anchor=tk.W)

        self.region_buttons: list[tk.Button] = []
        for rid in range(22):
            name = REGION_NAMES.get(rid, f"region_{rid}")
            r, g, b = REGION_COLORS.get(rid, (128, 128, 128))
            # Brighten dark colors for button visibility (min 100 per channel if any > 0)
            if rid > 0:
                max_c = max(r, g, b)
                if max_c > 0 and max_c < 128:
                    scale = 160 / max_c
                    r, g, b = min(255, int(r * scale)), min(255, int(g * scale)), min(255, int(b * scale))
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            fg_color = "#000" if brightness > 128 else "#fff"

            btn = tk.Button(
                panel, text=f"{rid:2d} {name}", anchor=tk.W, width=20,
                bg=hex_color, fg=fg_color, activebackground=hex_color,
                font=("Menlo", 10), relief=tk.FLAT, padx=4, pady=1,
                command=lambda r=rid: self._select_region(r),
            )
            btn.pack(fill=tk.X, pady=1)
            self.region_buttons.append(btn)

        tk.Label(panel, text="", bg="#2b2b2b").pack()  # spacer

        # Brush size
        self.brush_label = tk.Label(panel, text="", fg="#aaa", bg="#2b2b2b", font=("Menlo", 10))
        self.brush_label.pack(anchor=tk.W)

        # Display mode
        self.mode_label = tk.Label(panel, text="", fg="#aaa", bg="#2b2b2b", font=("Menlo", 10))
        self.mode_label.pack(anchor=tk.W)

        # Bottom bar
        bottom = tk.Frame(self.root, bg="#2b2b2b")
        bottom.pack(fill=tk.X, padx=8, pady=(4, 8))

        for text, cmd in [
            ("< Prev", self._prev),
            ("Next >", self._next),
            ("Accept (Enter)", self._accept),
            ("Reject (R)", self._reject),
            ("Undo (Ctrl+Z)", self._undo),
            ("Save & Quit (Esc)", self._quit),
        ]:
            tk.Button(
                bottom, text=text, command=cmd,
                bg="#444", fg="#fff", activebackground="#555",
                font=("Menlo", 10), relief=tk.FLAT, padx=8, pady=4,
            ).pack(side=tk.LEFT, padx=4)

        # Shortcuts help
        help_text = "Shortcuts: 1-0=regions | Shift+1-9=11-19 | a=accessory | h=hair | b=bg | [/]=brush | Space=mode"
        tk.Label(bottom, text=help_text, fg="#666", bg="#2b2b2b", font=("Menlo", 9)).pack(side=tk.RIGHT)

        self._update_brush_label()
        self._update_mode_label()
        self._highlight_selected_region()

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("<Return>", lambda e: self._accept())
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<space>", lambda e: self._toggle_mode())
        self.root.bind("<bracketleft>", lambda e: self._brush_smaller())
        self.root.bind("<bracketright>", lambda e: self._brush_larger())
        self.root.bind("[", lambda e: self._brush_smaller())
        self.root.bind("]", lambda e: self._brush_larger())
        self.root.bind("-", lambda e: self._brush_smaller())
        self.root.bind("=", lambda e: self._brush_larger())
        self.root.bind("<Control-z>", lambda e: self._undo())

        # Region shortcuts: 1-9, 0 → regions 1-10
        for i in range(10):
            region_id = i + 1 if i < 9 else 10  # 1→1, 2→2, ..., 9→9, 0→10
            self.root.bind(str((i + 1) % 10), lambda e, r=region_id: self._select_region(r))

        # Shift+1-9 → regions 11-19
        shift_keys = ["!", "@", "#", "$", "%", "^", "&", "*", "("]
        for i, key in enumerate(shift_keys):
            region_id = 11 + i
            self.root.bind(key, lambda e, r=region_id: self._select_region(r))

        # Special keys
        self.root.bind("r", lambda e: self._reject())
        self.root.bind("a", lambda e: self._select_region(20))  # accessory
        self.root.bind("h", lambda e: self._select_region(21))  # hair_back
        self.root.bind("b", lambda e: self._select_region(0))   # background (eraser)

        # Canvas mouse events
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<ButtonPress-2>", self._on_right_click)  # Middle click on Mac
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)  # Right click
        self.canvas.bind("<Control-ButtonPress-1>", self._on_right_click)  # Ctrl+click on Mac

    # --- Loading / saving ---

    def _load_current(self) -> None:
        if not self.state.examples:
            return

        example_dir = self.state.current_dir
        image_path = example_dir / "image.png"
        mask_path = example_dir / "segmentation.png"

        # Load image
        img = Image.open(image_path).convert("RGBA")
        if img.size != (CANVAS_SIZE, CANVAS_SIZE):
            img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
        self.image_rgba = np.array(img)

        # Load mask
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.shape != (CANVAS_SIZE, CANVAS_SIZE):
                mask = np.array(
                    Image.fromarray(mask).resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)
                )
        else:
            mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

        self.editor = MaskEditor(mask)
        self._refresh_display()
        self._update_labels()

    def _save_current(self) -> None:
        """Save mask if modified."""
        if self.editor is None or not self.editor.modified:
            return

        example_dir = self.state.current_dir
        Image.fromarray(self.editor.mask).save(example_dir / "segmentation.png")

        # Update metadata
        meta_path = example_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {}
        meta["segmentation_source"] = "pseudo_label_corrected"
        meta["annotation_quality"] = "manual_corrected"
        meta["review_status"] = self.state.get_status(self.state.current_name)
        meta["regions"] = sorted(int(r) for r in np.unique(self.editor.mask) if r > 0)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        self.editor.modified = False

    # --- Display ---

    def _refresh_display(self) -> None:
        if self.image_rgba is None or self.editor is None:
            return

        if self.display_mode == 0:  # overlay
            rgb = overlay_image_mask(self.image_rgba, self.editor.mask, OVERLAY_ALPHA)
        elif self.display_mode == 1:  # mask only
            rgb = colorize_mask(self.editor.mask)
        else:  # image only
            rgb = self.image_rgba[:, :, :3]

        pil_img = Image.fromarray(rgb, "RGB")
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfig(self.canvas_image_id, image=self._tk_image)

    def _update_labels(self) -> None:
        if not self.state.examples:
            self.progress_label.config(text="No examples")
            return

        total_all = len(self.state.manifest.get("examples", {}))
        reviewed = self.state.manifest.get("reviewed", 0)
        rejected = self.state.manifest.get("rejected", 0)
        remaining = total_all - reviewed - rejected

        self.progress_label.config(
            text=f"{self.state.current_idx + 1}/{len(self.state.examples)}  "
                 f"({reviewed} accepted, {rejected} rejected, {remaining} remaining)"
        )
        self.name_label.config(text=self.state.current_name)

        status = self.state.get_status(self.state.current_name)
        colors = {"reviewed": "#0f0", "rejected": "#f00", "needs_review": "#ff0"}
        self.status_label.config(text=status, fg=colors.get(status, "#fff"))

    def _update_brush_label(self) -> None:
        size = BRUSH_SIZES[self.brush_idx]
        self.brush_label.config(text=f"Brush: {size}px  [ / ]")

    def _update_mode_label(self) -> None:
        modes = ["overlay", "mask only", "image only"]
        self.mode_label.config(text=f"View: {modes[self.display_mode]}  (Space)")

    def _highlight_selected_region(self) -> None:
        for i, btn in enumerate(self.region_buttons):
            r, g, b = REGION_COLORS.get(i, (128, 128, 128))
            if i == self.selected_region:
                btn.config(relief=tk.SUNKEN, bd=3)
            else:
                btn.config(relief=tk.FLAT, bd=1)

    # --- Actions ---

    def _select_region(self, region_id: int) -> None:
        self.selected_region = region_id
        self._highlight_selected_region()

    def _brush_smaller(self) -> None:
        self.brush_idx = max(0, self.brush_idx - 1)
        self._update_brush_label()

    def _brush_larger(self) -> None:
        self.brush_idx = min(len(BRUSH_SIZES) - 1, self.brush_idx + 1)
        self._update_brush_label()

    def _toggle_mode(self) -> None:
        self.display_mode = (self.display_mode + 1) % 3
        self._update_mode_label()
        self._refresh_display()

    def _undo(self) -> None:
        if self.editor and self.editor.undo():
            self._refresh_display()

    def _navigate(self, delta: int) -> None:
        if not self.state.examples:
            return
        self._save_current()
        self.state.save_manifest()
        new_idx = self.state.current_idx + delta
        if 0 <= new_idx < len(self.state.examples):
            self.state.current_idx = new_idx
            self._load_current()

    def _prev(self) -> None:
        self._navigate(-1)

    def _next(self) -> None:
        self._navigate(1)

    def _accept(self) -> None:
        if not self.state.examples:
            return
        self.state.set_status(self.state.current_name, "reviewed")
        self._update_labels()
        self._navigate(1)

    def _reject(self) -> None:
        if not self.state.examples:
            return
        self.state.set_status(self.state.current_name, "rejected")
        self._update_labels()
        self._navigate(1)

    def _quit(self) -> None:
        self._save_current()
        self.state.save_manifest()
        self.root.destroy()

    # --- Mouse painting ---

    def _on_mouse_down(self, event: tk.Event) -> None:
        if self.editor is None:
            return
        self.painting = True
        self.editor.push_undo()
        self._paint_at(event.x, event.y)

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if not self.painting or self.editor is None:
            return
        self._paint_at(event.x, event.y)

    def _on_mouse_up(self, event: tk.Event) -> None:
        self.painting = False

    def _on_right_click(self, event: tk.Event) -> None:
        """Eyedropper — pick region from mask."""
        if self.editor is None:
            return
        region_id = self.editor.pick(event.x, event.y)
        self._select_region(region_id)

    def _paint_at(self, x: int, y: int) -> None:
        if self.editor is None:
            return
        radius = BRUSH_SIZES[self.brush_idx]
        self.editor.paint(x, y, self.selected_region, radius)
        self._refresh_display()

    # --- Run ---

    def run(self) -> None:
        if not self.state.examples:
            logger.warning("No examples found in %s", self.state.data_dir)
            return
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and correct pseudo-labeled segmentation masks.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with per-example subdirs (image.png + segmentation.png)")
    parser.add_argument("--only-needs-review", action="store_true",
                        help="Only show examples that need review")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from this example index")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    state = ReviewState(data_dir=args.data_dir)
    state.load(only_needs_review=args.only_needs_review)

    logger.info("Loaded %d examples from %s", len(state.examples), args.data_dir)

    app = ReviewApp(state, start_from=args.start_from)
    app.run()


if __name__ == "__main__":
    main()
