"""Tkinter-based review UI for verifying Live2D fragment-to-Strata label mappings.

Loads a Live2D model's fragment images, highlights each fragment one at a time,
shows the auto-assigned Strata label from the mapper, and allows a human reviewer
to confirm or correct the label with a keypress. Updates the CSV with
``confirmed=manual`` status.

Designed for ~5 seconds per fragment review time. Supports resume — fragments
already marked ``manual`` in the CSV are skipped.

This module is pure Python + Tkinter (no Blender dependency).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import REGION_NAME_TO_ID, REGION_NAMES, RegionId
from .live2d_mapper import (
    FragmentMapping,
    ModelMapping,
    export_csv,
    load_csv,
    map_fragment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = Path("data/live2d/labels/live2d_mappings.csv")

# Highlight overlay color (RGBA) for the active fragment
HIGHLIGHT_COLOR = (0, 255, 0, 100)

# Display resolution for the canvas
DISPLAY_SIZE = 512

# Region labels sorted by ID for the label selector (skip background)
SELECTABLE_REGIONS: list[tuple[RegionId, str]] = [
    (rid, name) for rid, name in sorted(REGION_NAMES.items()) if rid > 0
]

# Keyboard shortcut mapping: digit keys select region groups
# 1=head, 2=neck, 3=chest, 4=spine, 5=hips, 6-8=arms, 9=legs, 0=shoulders
SHORTCUT_GROUPS: dict[str, list[str]] = {
    "1": ["head"],
    "2": ["neck"],
    "3": ["chest"],
    "4": ["spine"],
    "5": ["hips"],
    "6": ["upper_arm_l", "upper_arm_r"],
    "7": ["forearm_l", "forearm_r", "hand_l", "hand_r"],
    "8": ["upper_leg_l", "upper_leg_r"],
    "9": ["lower_leg_l", "lower_leg_r", "foot_l", "foot_r"],
    "0": ["shoulder_l", "shoulder_r"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReviewState:
    """Tracks the review session state across models and fragments."""

    csv_path: Path
    models: list[ModelMapping] = field(default_factory=list)
    current_model_idx: int = 0
    current_fragment_idx: int = 0

    @property
    def current_model(self) -> ModelMapping | None:
        """Return the current model being reviewed, or None if done."""
        if 0 <= self.current_model_idx < len(self.models):
            return self.models[self.current_model_idx]
        return None

    @property
    def current_fragment(self) -> FragmentMapping | None:
        """Return the current fragment being reviewed, or None if done."""
        model = self.current_model
        if model is None:
            return None
        pending = self.pending_fragments(model)
        if 0 <= self.current_fragment_idx < len(pending):
            return pending[self.current_fragment_idx]
        return None

    @property
    def total_fragments(self) -> int:
        """Total fragments across all models."""
        return sum(m.total_count for m in self.models)

    @property
    def reviewed_fragments(self) -> int:
        """Fragments already confirmed (auto or manual)."""
        return sum(1 for m in self.models for f in m.mappings if f.confirmed == "manual")

    @property
    def models_completed(self) -> int:
        """Number of models fully reviewed."""
        return sum(1 for m in self.models if all(f.confirmed == "manual" for f in m.mappings))

    def pending_fragments(self, model: ModelMapping) -> list[FragmentMapping]:
        """Return fragments that still need review in a model."""
        return [f for f in model.mappings if f.confirmed != "manual"]

    def advance(self) -> bool:
        """Move to the next pending fragment. Returns False if all done."""
        model = self.current_model
        if model is None:
            return False

        pending = self.pending_fragments(model)
        self.current_fragment_idx += 1

        if self.current_fragment_idx >= len(pending):
            # Move to next model with pending fragments
            self.current_fragment_idx = 0
            self.current_model_idx += 1
            while self.current_model_idx < len(self.models):
                model = self.models[self.current_model_idx]
                if self.pending_fragments(model):
                    return True
                self.current_model_idx += 1
            return False

        return True

    def go_back(self) -> bool:
        """Move to the previous fragment. Returns False if at the start."""
        if self.current_fragment_idx > 0:
            self.current_fragment_idx -= 1
            return True
        # Try previous model
        if self.current_model_idx > 0:
            self.current_model_idx -= 1
            model = self.current_model
            if model:
                pending = self.pending_fragments(model)
                self.current_fragment_idx = max(0, len(pending) - 1)
                return True
        return False


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------


def save_review_csv(state: ReviewState) -> None:
    """Write the current review state back to the CSV file.

    Overwrites the entire CSV to reflect any confirmed/corrected labels.

    Args:
        state: Current review state with all models and their mappings.
    """
    export_csv(state.models, state.csv_path)
    logger.info("Saved review progress to %s", state.csv_path)


def load_or_create_csv(
    csv_path: Path,
    model_dirs: list[Path],
) -> list[ModelMapping]:
    """Load existing CSV or create initial mappings from model directories.

    If the CSV exists, loads it. Otherwise, discovers fragments from model
    directories and runs the auto-mapper to create initial mappings.

    Args:
        csv_path: Path to the mappings CSV.
        model_dirs: List of Live2D model directories to scan.

    Returns:
        List of ModelMapping objects.
    """
    if csv_path.exists() and csv_path.stat().st_size > 0:
        logger.info("Loading existing mappings from %s", csv_path)
        return load_csv(csv_path)

    logger.info("No existing CSV found, running auto-mapper on %d models", len(model_dirs))
    models: list[ModelMapping] = []

    for model_dir in model_dirs:
        fragment_names = _discover_fragment_names(model_dir)
        if not fragment_names:
            continue
        model_id = model_dir.name
        mapping = _auto_map_model(model_id, fragment_names)
        models.append(mapping)

    if models:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_csv(models, csv_path)
        logger.info("Created initial CSV with %d models", len(models))

    return models


def _discover_fragment_names(model_dir: Path) -> list[str]:
    """Discover fragment image names from a model directory.

    Checks common subdirectory names for part images, same logic as the
    renderer's ``_discover_fragment_images``.

    Args:
        model_dir: Directory containing the Live2D model files.

    Returns:
        Sorted list of fragment names (file stems).
    """
    names: list[str] = []
    seen: set[Path] = set()

    candidate_dirs = [
        model_dir / "parts",
        model_dir / "textures",
        model_dir / "images",
        model_dir,
    ]

    for search_dir in candidate_dirs:
        if not search_dir.is_dir():
            continue
        for png_path in sorted(search_dir.glob("*.png")):
            if png_path in seen:
                continue
            seen.add(png_path)
            names.append(png_path.stem)

    return names


def _auto_map_model(model_id: str, fragment_names: list[str]) -> ModelMapping:
    """Run the auto-mapper on a list of fragment names.

    Args:
        model_id: Unique identifier for the model.
        fragment_names: List of fragment names to map.

    Returns:
        ModelMapping with auto or pending confirmed status.
    """
    result = ModelMapping(model_id=model_id)
    for name in fragment_names:
        region_name, region_id = map_fragment(name)
        confirmed = "auto" if region_id >= 0 else "pending"
        result.mappings.append(
            FragmentMapping(
                fragment_name=name,
                strata_label=region_name,
                strata_region_id=region_id,
                confirmed=confirmed,
            )
        )
    return result


def update_fragment_label(
    fragment: FragmentMapping,
    new_label: str,
) -> None:
    """Update a fragment's label and mark it as manually confirmed.

    Args:
        fragment: The fragment mapping to update.
        new_label: The corrected Strata region name.
    """
    region_id = REGION_NAME_TO_ID.get(new_label, -1)
    fragment.strata_label = new_label
    fragment.strata_region_id = region_id
    fragment.confirmed = "manual"


def confirm_fragment(fragment: FragmentMapping) -> None:
    """Mark a fragment's current label as manually confirmed.

    Args:
        fragment: The fragment mapping to confirm.
    """
    fragment.confirmed = "manual"


# ---------------------------------------------------------------------------
# Fragment image loading (for display)
# ---------------------------------------------------------------------------


def load_fragment_image(model_dir: Path, fragment_name: str) -> Any | None:
    """Load a fragment's PNG image for display.

    Args:
        model_dir: Directory containing the model files.
        fragment_name: Name of the fragment (file stem).

    Returns:
        PIL Image in RGBA mode, or None if not found.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow is required for image loading")
        return None

    candidate_dirs = [
        model_dir / "parts",
        model_dir / "textures",
        model_dir / "images",
        model_dir,
    ]

    for search_dir in candidate_dirs:
        png_path = search_dir / f"{fragment_name}.png"
        if png_path.exists():
            try:
                return Image.open(png_path).convert("RGBA")
            except Exception as exc:
                logger.warning("Failed to load %s: %s", png_path, exc)

    return None


def build_composite_image(
    model_dir: Path,
    fragment_names: list[str],
    resolution: int = DISPLAY_SIZE,
) -> Any | None:
    """Build a composite image from all fragment images.

    Args:
        model_dir: Directory containing the model files.
        fragment_names: Ordered list of fragment names.
        resolution: Output resolution (square).

    Returns:
        PIL Image (RGBA), or None if no fragments could be loaded.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    loaded: list[Any] = []
    for name in fragment_names:
        img = load_fragment_image(model_dir, name)
        if img is not None:
            loaded.append(img)

    if not loaded:
        return None

    # Find bounding dimensions
    max_w = max(img.width for img in loaded)
    max_h = max(img.height for img in loaded)

    # Scale to fit
    scale = min(resolution / max_w, resolution / max_h) if max(max_w, max_h) > 0 else 1.0
    scale = min(scale, 1.0)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    for img in loaded:
        new_w = max(1, round(img.width * scale))
        new_h = max(1, round(img.height * scale))
        scaled = img.resize((new_w, new_h), Image.BILINEAR)
        paste_x = (resolution - new_w) // 2
        paste_y = (resolution - new_h) // 2
        canvas.paste(scaled, (paste_x, paste_y), scaled)

    return canvas


def build_highlight_image(
    model_dir: Path,
    fragment_name: str,
    resolution: int = DISPLAY_SIZE,
    highlight_color: tuple[int, int, int, int] = HIGHLIGHT_COLOR,
) -> Any | None:
    """Build a highlight overlay for a single fragment.

    Creates a semi-transparent colored overlay where the fragment has
    opaque pixels.

    Args:
        model_dir: Directory containing the model files.
        fragment_name: Name of the fragment to highlight.
        resolution: Output resolution (square).
        highlight_color: RGBA color for the highlight overlay.

    Returns:
        PIL Image (RGBA) with highlight overlay, or None if not found.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    frag_img = load_fragment_image(model_dir, fragment_name)
    if frag_img is None:
        return None

    # Scale to match composite
    scale = min(resolution / frag_img.width, resolution / frag_img.height)
    scale = min(scale, 1.0)
    new_w = max(1, round(frag_img.width * scale))
    new_h = max(1, round(frag_img.height * scale))
    scaled = frag_img.resize((new_w, new_h), Image.BILINEAR)

    paste_x = (resolution - new_w) // 2
    paste_y = (resolution - new_h) // 2

    # Create highlight overlay from alpha channel
    import numpy as np

    arr = np.array(scaled)
    alpha = arr[:, :, 3]

    highlight = np.zeros((resolution, resolution, 4), dtype=np.uint8)

    # Only highlight where the fragment has opaque pixels
    y_end = min(paste_y + new_h, resolution)
    x_end = min(paste_x + new_w, resolution)
    y_start = max(0, paste_y)
    x_start = max(0, paste_x)

    src_y0 = max(0, -paste_y)
    src_x0 = max(0, -paste_x)
    src_y1 = src_y0 + (y_end - y_start)
    src_x1 = src_x0 + (x_end - x_start)

    if y_end > y_start and x_end > x_start:
        alpha_slice = alpha[src_y0:src_y1, src_x0:src_x1]
        opaque = alpha_slice > 0
        for c in range(4):
            highlight[y_start:y_end, x_start:x_end, c][opaque] = highlight_color[c]

    overlay = Image.fromarray(highlight, "RGBA")
    return overlay


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------


def _check_tkinter() -> bool:
    """Check if tkinter is available."""
    try:
        import tkinter  # noqa: F401

        return True
    except ImportError:
        return False


def launch_review_ui(
    model_dir: Path,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> None:
    """Launch the Tkinter review UI for a Live2D model directory.

    Args:
        model_dir: Root directory containing Live2D model subdirectories.
        csv_path: Path to the mappings CSV file.
    """
    if not _check_tkinter():
        logger.error(
            "tkinter is not available. Install it with your system package manager "
            "(e.g., 'brew install python-tk' on macOS, 'apt install python3-tk' on Ubuntu)."
        )
        sys.exit(1)

    try:
        from PIL import ImageTk  # validates Pillow is installed
    except ImportError:
        logger.error("Pillow is required for the review UI. Install with: pip install Pillow")
        sys.exit(1)

    import tkinter as tk
    from tkinter import messagebox, ttk

    # Discover model directories
    model_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
    if not model_dirs:
        logger.error("No model directories found in %s", model_dir)
        sys.exit(1)

    # Load or create CSV
    models = load_or_create_csv(csv_path, model_dirs)
    if not models:
        logger.error("No models to review")
        sys.exit(1)

    # Build model_dir lookup
    model_dir_map: dict[str, Path] = {d.name: d for d in model_dirs}

    # Initialize review state
    state = ReviewState(csv_path=csv_path, models=models)

    # Skip to first model with pending fragments
    while state.current_model_idx < len(state.models):
        model = state.current_model
        if model and state.pending_fragments(model):
            break
        state.current_model_idx += 1

    if state.current_model is None:
        logger.info("All fragments have been reviewed!")
        print("All fragments have been reviewed!")
        return

    # --- Build UI ---
    root = tk.Tk()
    root.title("Live2D Fragment Review")
    root.geometry("900x700")
    root.configure(bg="#2b2b2b")

    # Top info bar
    info_frame = tk.Frame(root, bg="#2b2b2b")
    info_frame.pack(fill=tk.X, padx=10, pady=5)

    progress_label = tk.Label(
        info_frame,
        text="",
        fg="white",
        bg="#2b2b2b",
        font=("Helvetica", 12),
    )
    progress_label.pack(side=tk.LEFT)

    model_label = tk.Label(
        info_frame,
        text="",
        fg="#aaaaaa",
        bg="#2b2b2b",
        font=("Helvetica", 10),
    )
    model_label.pack(side=tk.RIGHT)

    # Main content area
    content_frame = tk.Frame(root, bg="#2b2b2b")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Canvas for image display
    canvas = tk.Canvas(
        content_frame,
        width=DISPLAY_SIZE,
        height=DISPLAY_SIZE,
        bg="#1a1a1a",
        highlightthickness=0,
    )
    canvas.pack(side=tk.LEFT, padx=(0, 10))

    # Right panel: fragment info + label selector
    right_panel = tk.Frame(content_frame, bg="#2b2b2b")
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    fragment_label = tk.Label(
        right_panel,
        text="Fragment:",
        fg="white",
        bg="#2b2b2b",
        font=("Helvetica", 14, "bold"),
        anchor="w",
    )
    fragment_label.pack(fill=tk.X, pady=(0, 5))

    current_label = tk.Label(
        right_panel,
        text="Current label:",
        fg="#66ff66",
        bg="#2b2b2b",
        font=("Helvetica", 12),
        anchor="w",
    )
    current_label.pack(fill=tk.X, pady=(0, 10))

    # Label selector listbox
    selector_frame = tk.Frame(right_panel, bg="#2b2b2b")
    selector_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        selector_frame,
        text="Select correct label:",
        fg="white",
        bg="#2b2b2b",
        font=("Helvetica", 10),
        anchor="w",
    ).pack(fill=tk.X)

    listbox = tk.Listbox(
        selector_frame,
        font=("Helvetica", 11),
        bg="#333333",
        fg="white",
        selectbackground="#0078d4",
        selectforeground="white",
        height=20,
    )
    listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    for rid, name in SELECTABLE_REGIONS:
        listbox.insert(tk.END, f"{rid:2d}  {name}")

    # Bottom controls
    control_frame = tk.Frame(root, bg="#2b2b2b")
    control_frame.pack(fill=tk.X, padx=10, pady=10)

    shortcuts_label = tk.Label(
        control_frame,
        text="Enter=Confirm | Space=Select from list | Left/Right=Navigate | Esc=Save & Quit",
        fg="#888888",
        bg="#2b2b2b",
        font=("Helvetica", 9),
    )
    shortcuts_label.pack(side=tk.BOTTOM, pady=(5, 0))

    btn_frame = tk.Frame(control_frame, bg="#2b2b2b")
    btn_frame.pack()

    # Keep references to PhotoImage objects to prevent garbage collection
    _photo_refs: list[Any] = []

    def update_display() -> None:
        """Update the canvas and labels for the current fragment."""
        _photo_refs.clear()
        canvas.delete("all")

        model = state.current_model
        if model is None:
            fragment_label.config(text="All done!")
            current_label.config(text="All fragments have been reviewed.")
            progress_label.config(
                text=f"Reviewed: {state.reviewed_fragments}/{state.total_fragments}"
            )
            save_review_csv(state)
            messagebox.showinfo("Complete", "All fragments have been reviewed!")
            return

        fragment = state.current_fragment
        if fragment is None:
            # Current model is done, advance
            if not state.advance():
                update_display()
                return
            fragment = state.current_fragment
            if fragment is None:
                update_display()
                return

        mdir = model_dir_map.get(model.model_id)
        pending = state.pending_fragments(model)

        # Update labels
        progress_label.config(
            text=(
                f"Reviewed: {state.reviewed_fragments}/{state.total_fragments}  |  "
                f"Models: {state.models_completed}/{len(state.models)}"
            )
        )
        model_label.config(
            text=f"Model: {model.model_id}  |  Fragment {state.current_fragment_idx + 1}/{len(pending)}"
        )
        fragment_label.config(text=f"Fragment: {fragment.fragment_name}")

        status_color = "#ff6666" if fragment.confirmed == "pending" else "#66ff66"
        current_label.config(
            text=f"Auto-label: {fragment.strata_label} (ID: {fragment.strata_region_id})",
            fg=status_color,
        )

        # Select current label in listbox
        for i, (_rid, name) in enumerate(SELECTABLE_REGIONS):
            if name == fragment.strata_label:
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(i)
                listbox.see(i)
                break

        # Draw composite + highlight
        if mdir is not None:
            frag_names = [f.fragment_name for f in model.mappings]
            composite = build_composite_image(mdir, frag_names)
            if composite is not None:
                photo = ImageTk.PhotoImage(composite)
                _photo_refs.append(photo)
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            highlight = build_highlight_image(mdir, fragment.fragment_name)
            if highlight is not None:
                h_photo = ImageTk.PhotoImage(highlight)
                _photo_refs.append(h_photo)
                canvas.create_image(0, 0, anchor=tk.NW, image=h_photo)

    def on_confirm(_event: Any = None) -> None:
        """Confirm the current fragment's label and advance."""
        fragment = state.current_fragment
        if fragment is None:
            return
        confirm_fragment(fragment)
        save_review_csv(state)
        if not state.advance():
            update_display()
            return
        update_display()

    def on_select_and_confirm(_event: Any = None) -> None:
        """Apply the listbox selection as the label and advance."""
        fragment = state.current_fragment
        if fragment is None:
            return
        selection = listbox.curselection()
        if selection:
            idx = selection[0]
            _rid, name = SELECTABLE_REGIONS[idx]
            update_fragment_label(fragment, name)
        else:
            confirm_fragment(fragment)
        save_review_csv(state)
        if not state.advance():
            update_display()
            return
        update_display()

    def on_prev(_event: Any = None) -> None:
        """Go to the previous fragment."""
        state.go_back()
        update_display()

    def on_next(_event: Any = None) -> None:
        """Skip to the next fragment without confirming."""
        state.advance()
        update_display()

    def on_quit(_event: Any = None) -> None:
        """Save and quit."""
        save_review_csv(state)
        root.destroy()

    def on_set_background(_event: Any = None) -> None:
        """Quick-assign background label."""
        fragment = state.current_fragment
        if fragment is None:
            return
        update_fragment_label(fragment, "background")
        save_review_csv(state)
        if not state.advance():
            update_display()
            return
        update_display()

    def on_key(event: Any) -> None:
        """Handle digit key shortcuts for quick region group selection."""
        key = event.char
        if key in SHORTCUT_GROUPS:
            regions = SHORTCUT_GROUPS[key]
            # If only one region in group, apply directly
            if len(regions) == 1:
                fragment = state.current_fragment
                if fragment:
                    update_fragment_label(fragment, regions[0])
                    save_review_csv(state)
                    if not state.advance():
                        update_display()
                        return
                    update_display()
            else:
                # Select the first matching region in the listbox
                for i, (_rid, name) in enumerate(SELECTABLE_REGIONS):
                    if name in regions:
                        listbox.selection_clear(0, tk.END)
                        listbox.selection_set(i)
                        listbox.see(i)
                        break

    # Bind keys
    root.bind("<Return>", on_confirm)
    root.bind("<space>", on_select_and_confirm)
    root.bind("<Left>", on_prev)
    root.bind("<Right>", on_next)
    root.bind("<Escape>", on_quit)
    root.bind("b", on_set_background)
    root.bind("<Key>", on_key)

    # Buttons
    ttk.Button(btn_frame, text="Confirm (Enter)", command=on_confirm).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Apply Selection (Space)", command=on_select_and_confirm).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(btn_frame, text="Prev (<-)", command=on_prev).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Next (->)", command=on_next).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Background (b)", command=on_set_background).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(btn_frame, text="Save & Quit (Esc)", command=on_quit).pack(side=tk.LEFT, padx=5)

    # Initial display
    update_display()

    root.mainloop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the Live2D review UI."""
    parser = argparse.ArgumentParser(
        description="Review and correct Live2D fragment-to-Strata label mappings."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("data/live2d"),
        help="Root directory containing Live2D model subdirectories (default: data/live2d)",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the mappings CSV file (default: {DEFAULT_CSV_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    launch_review_ui(args.model_dir, args.csv_path)


if __name__ == "__main__":
    main()
