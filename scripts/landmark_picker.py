#!/usr/bin/env python3
"""Interactive landmark picker for illustration-to-mesh alignment.

Displays illustration and mesh preview side-by-side. Click on matching
feature pairs (eye on illustration → eye on mesh). Press keys to name
each landmark. Saves to JSON.

Usage::

    python3 scripts/landmark_picker.py --output_dir output/lichtung_test/

Expects these files in output_dir:
    - landmarks_template.json (from --prepare_landmarks)
    - _mesh_preview_<view>.png
    - The illustration path from the template

Controls:
    - Click on left image (illustration) → left point
    - Click on right image (mesh) → right point
    - Both clicked → enter landmark name, press Enter to save
    - d → delete last pair
    - n → next view
    - s → save and quit
    - q → quit without saving
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog

from PIL import Image, ImageTk

# Suggested landmark names per view angle
LANDMARK_SUGGESTIONS = {
    "front": [
        # Head
        "eye_l", "eye_r", "nose_tip", "mouth_center", "chin",
        "ear_tip_l", "ear_tip_r", "ear_base_l", "ear_base_r",
        "forehead", "cheek_l", "cheek_r",
        # Neck/chest
        "neck_front", "chest_center", "collar_l", "collar_r",
        # Body silhouette (left/right belly outline)
        "shoulder_l", "shoulder_r", "armpit_l", "armpit_r",
        "belly_l", "belly_r", "belly_center", "waist_l", "waist_r",
        "hip_l", "hip_r",
        # Limbs
        "elbow_l", "elbow_r", "wrist_l", "wrist_r",
        "paw_front_l", "paw_front_r",
        "knee_l", "knee_r", "ankle_l", "ankle_r",
        "paw_back_l", "paw_back_r",
        # Tail
        "tail_base", "tail_mid", "tail_tip",
    ],
    "three_quarter": [
        # Same as front but maybe asymmetric visibility
        "eye_l", "eye_r", "nose_tip", "mouth_center", "chin",
        "ear_tip_l", "ear_tip_r", "ear_base_l", "ear_base_r",
        "forehead", "cheek_l", "cheek_r",
        "neck_front", "neck_back", "chest_center",
        "shoulder_l", "shoulder_r",
        "belly_l", "belly_r", "belly_center",
        "hip_l", "hip_r", "back_center",
        "elbow_l", "elbow_r", "wrist_l", "wrist_r",
        "paw_front_l", "paw_front_r",
        "knee_l", "knee_r", "paw_back_l", "paw_back_r",
        "tail_base", "tail_mid", "tail_tip",
    ],
    "side": [
        "eye", "nose_tip", "mouth_center", "chin",
        "ear_tip", "ear_base", "forehead", "cheek",
        "neck_front", "neck_back",
        "chest_front", "chest_bottom", "belly_front", "belly_bottom",
        "back_top", "back_mid", "hip_top", "hip_bottom",
        "shoulder", "elbow", "wrist", "paw_front",
        "knee", "ankle", "paw_back",
        "tail_base", "tail_mid", "tail_tip",
    ],
    "back": [
        # Head (back)
        "ear_tip_l", "ear_tip_r", "ear_base_l", "ear_base_r",
        "skull_top", "skull_back", "neck_back",
        # Back/body
        "shoulder_blade_l", "shoulder_blade_r",
        "back_top", "back_mid", "back_lower", "spine_mid",
        "hip_l", "hip_r", "butt_l", "butt_r",
        # Limbs from behind
        "upper_arm_l", "upper_arm_r", "elbow_l", "elbow_r",
        "paw_front_l", "paw_front_r",
        "upper_leg_l", "upper_leg_r", "knee_l", "knee_r",
        "paw_back_l", "paw_back_r",
        # Tail
        "tail_base", "tail_mid", "tail_tip",
    ],
}

# Default for unknown view names
DEFAULT_LANDMARKS = LANDMARK_SUGGESTIONS["front"]


class LandmarkPicker:
    def __init__(self, output_dir: Path, display_size: int = 600):
        self.output_dir = output_dir
        self.display_size = display_size
        self.template_path = output_dir / "landmarks_template.json"
        self.output_path = output_dir / "landmarks.json"

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

        self.template = json.loads(self.template_path.read_text())
        # Load existing landmarks if available
        if self.output_path.exists():
            self.landmarks = json.loads(self.output_path.read_text())
        else:
            self.landmarks = {}

        self.view_names = list(self.template.keys())
        self.current_view = 0

        self.root = tk.Tk()
        self.root.title("Landmark Picker")
        self.root.geometry(f"{display_size * 2 + 40}x{display_size + 150}")

        self.status = tk.Label(self.root, text="", font=("Arial", 14), height=2)
        self.status.pack(fill=tk.X)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        self.canvas_illus = tk.Canvas(
            self.canvas_frame, width=display_size, height=display_size, bg="black",
        )
        self.canvas_illus.pack(side=tk.LEFT, padx=5)
        self.canvas_illus.bind("<Button-1>", self.on_click_illus)

        self.canvas_mesh = tk.Canvas(
            self.canvas_frame, width=display_size, height=display_size, bg="black",
        )
        self.canvas_mesh.pack(side=tk.LEFT, padx=5)
        self.canvas_mesh.bind("<Button-1>", self.on_click_mesh)

        # Bottom panel: suggestions + placed list
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: suggestion picker
        self.suggest_frame = tk.Frame(self.bottom_frame)
        self.suggest_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(self.suggest_frame, text="Suggested landmarks (click to select):",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W)
        suggest_scroll = tk.Scrollbar(self.suggest_frame)
        suggest_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.suggest_listbox = tk.Listbox(
            self.suggest_frame, width=30, height=8, font=("Courier", 10),
            yscrollcommand=suggest_scroll.set, exportselection=False,
        )
        self.suggest_listbox.pack(side=tk.LEFT, fill=tk.Y)
        suggest_scroll.config(command=self.suggest_listbox.yview)
        self.suggest_listbox.bind("<<ListboxSelect>>", self.on_suggestion_select)
        self.selected_suggestion = None

        # Right: placed landmarks list
        self.placed_frame = tk.Frame(self.bottom_frame)
        self.placed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        tk.Label(self.placed_frame, text="Placed landmarks:",
                 font=("Arial", 11, "bold")).pack(anchor=tk.W)
        self.list_text = tk.Text(self.placed_frame, height=8, font=("Courier", 10))
        self.list_text.pack(fill=tk.BOTH, expand=True)

        # Instructions
        inst = tk.Label(
            self.root,
            text=("Select landmark from list → CLICK illustration (LEFT) → CLICK mesh (RIGHT) | "
                  "d=delete last | n=next view | s=save+quit | q=quit | t=type custom name"),
            font=("Arial", 10),
            fg="gray",
        )
        inst.pack(fill=tk.X, pady=5)

        self.root.bind("d", lambda e: self.delete_last())
        self.root.bind("n", lambda e: self.next_view())
        self.root.bind("s", lambda e: self.save_and_quit())
        self.root.bind("q", lambda e: self.root.quit())
        self.root.bind("t", lambda e: self.type_custom_name())

        self.pending_illus = None
        self.pending_mesh = None

        self.load_view()

    def load_view(self):
        view_name = self.view_names[self.current_view]
        view_data = self.template[view_name]

        # Load illustration
        illus_path = Path(view_data["illustration"])
        if not illus_path.is_absolute():
            illus_path = self.output_dir.parent.parent / illus_path
            if not illus_path.exists():
                # Try relative to cwd
                illus_path = Path(view_data["illustration"])
        self.illus_img = Image.open(illus_path).convert("RGBA")
        self.illus_w, self.illus_h = self.illus_img.size
        self.illus_display = self.fit_to_canvas(self.illus_img, self.display_size)
        self.illus_photo = ImageTk.PhotoImage(self.illus_display)
        self.canvas_illus.delete("all")
        self.canvas_illus.create_image(
            self.display_size // 2, self.display_size // 2,
            image=self.illus_photo, anchor=tk.CENTER,
        )

        # Load mesh preview
        mesh_path = self.output_dir / view_data["mesh_preview"]
        self.mesh_img = Image.open(mesh_path).convert("RGBA")
        self.mesh_w, self.mesh_h = self.mesh_img.size
        self.mesh_display = self.fit_to_canvas(self.mesh_img, self.display_size)
        self.mesh_photo = ImageTk.PhotoImage(self.mesh_display)
        self.canvas_mesh.delete("all")
        self.canvas_mesh.create_image(
            self.display_size // 2, self.display_size // 2,
            image=self.mesh_photo, anchor=tk.CENTER,
        )

        # Compute image display offsets (for pixel mapping)
        self.illus_offset, self.illus_scale = self.compute_offset(self.illus_img)
        self.mesh_offset, self.mesh_scale = self.compute_offset(self.mesh_img)

        # Init landmarks for this view
        if view_name not in self.landmarks:
            self.landmarks[view_name] = {
                "illustration": view_data["illustration"],
                "mesh_preview": view_data["mesh_preview"],
                "angle": view_data["angle"],
                "points": [],
            }

        self.pending_illus = None
        self.pending_mesh = None
        self.selected_suggestion = None

        # Populate suggestions list (pick best match by view name)
        suggestions = LANDMARK_SUGGESTIONS.get(view_name)
        if suggestions is None:
            # Try to match partial names: front, side, back, three_quarter
            lname = view_name.lower()
            if "back" in lname:
                suggestions = LANDMARK_SUGGESTIONS["back"]
            elif "side" in lname or "left" in lname or "right" in lname:
                suggestions = LANDMARK_SUGGESTIONS["side"]
            elif "three" in lname or "quarter" in lname or "34" in lname:
                suggestions = LANDMARK_SUGGESTIONS["three_quarter"]
            else:
                suggestions = DEFAULT_LANDMARKS

        # Filter out already-placed landmarks
        placed_names = {p["name"] for p in self.landmarks[view_name]["points"]}
        self.current_suggestions = [s for s in suggestions if s not in placed_names]

        self.suggest_listbox.delete(0, tk.END)
        for s in self.current_suggestions:
            self.suggest_listbox.insert(tk.END, s)

        self.redraw_markers()
        self.update_status()

    def fit_to_canvas(self, img, size):
        w, h = img.size
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def compute_offset(self, img):
        w, h = img.size
        scale = min(self.display_size / w, self.display_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        off_x = (self.display_size - new_w) // 2
        off_y = (self.display_size - new_h) // 2
        return (off_x, off_y), scale

    def canvas_to_image(self, cx, cy, offset, scale):
        ix = (cx - offset[0]) / scale
        iy = (cy - offset[1]) / scale
        return int(ix), int(iy)

    def image_to_canvas(self, ix, iy, offset, scale):
        cx = int(ix * scale + offset[0])
        cy = int(iy * scale + offset[1])
        return cx, cy

    def on_click_illus(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y, self.illus_offset, self.illus_scale)
        if 0 <= ix < self.illus_w and 0 <= iy < self.illus_h:
            self.pending_illus = (ix, iy)
            self.redraw_markers()
            self.try_commit()

    def on_click_mesh(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y, self.mesh_offset, self.mesh_scale)
        if 0 <= ix < self.mesh_w and 0 <= iy < self.mesh_h:
            self.pending_mesh = (ix, iy)
            self.redraw_markers()
            self.try_commit()

    def on_suggestion_select(self, event):
        sel = self.suggest_listbox.curselection()
        if sel:
            self.selected_suggestion = self.current_suggestions[sel[0]]
            self.update_status()

    def type_custom_name(self):
        name = simpledialog.askstring(
            "Custom landmark name",
            "Enter landmark name:",
            parent=self.root,
        )
        if name:
            self.selected_suggestion = name
            self.update_status()

    def try_commit(self):
        if self.pending_illus and self.pending_mesh:
            # Use selected suggestion, or prompt
            if self.selected_suggestion:
                name = self.selected_suggestion
            else:
                name = simpledialog.askstring(
                    "Landmark name",
                    "No suggestion selected. Enter landmark name:",
                    parent=self.root,
                )
            if name:
                view_name = self.view_names[self.current_view]
                self.landmarks[view_name]["points"].append({
                    "name": name,
                    "illus": list(self.pending_illus),
                    "mesh": list(self.pending_mesh),
                })
                # Remove from suggestions
                if name in self.current_suggestions:
                    idx = self.current_suggestions.index(name)
                    self.current_suggestions.pop(idx)
                    self.suggest_listbox.delete(idx)
                # Auto-select next suggestion
                if self.current_suggestions:
                    self.suggest_listbox.selection_clear(0, tk.END)
                    self.suggest_listbox.selection_set(0)
                    self.suggest_listbox.see(0)
                    self.selected_suggestion = self.current_suggestions[0]
                else:
                    self.selected_suggestion = None
            self.pending_illus = None
            self.pending_mesh = None
            self.redraw_markers()
            self.update_status()

    def delete_last(self):
        view_name = self.view_names[self.current_view]
        pts = self.landmarks[view_name]["points"]
        if pts:
            removed = pts.pop()
            # Add back to suggestions if it was a known one
            # (Reload the view to re-populate the list correctly)
            self.load_view()
            return
        self.redraw_markers()
        self.update_status()

    def next_view(self):
        self.save()
        self.current_view = (self.current_view + 1) % len(self.view_names)
        self.load_view()

    def save_and_quit(self):
        self.save()
        self.root.quit()

    def save(self):
        self.output_path.write_text(json.dumps(self.landmarks, indent=2) + "\n")
        print(f"Saved landmarks to {self.output_path}")

    def redraw_markers(self):
        view_name = self.view_names[self.current_view]
        pts = self.landmarks[view_name]["points"]

        # Reload canvases
        self.canvas_illus.delete("marker")
        self.canvas_mesh.delete("marker")

        for i, p in enumerate(pts):
            ix, iy = p["illus"]
            mx, my = p["mesh"]
            cx1, cy1 = self.image_to_canvas(ix, iy, self.illus_offset, self.illus_scale)
            cx2, cy2 = self.image_to_canvas(mx, my, self.mesh_offset, self.mesh_scale)

            color = "#ff4080"
            self.canvas_illus.create_oval(cx1-6, cy1-6, cx1+6, cy1+6, outline=color, width=2, tags="marker")
            self.canvas_illus.create_text(cx1+10, cy1, text=p["name"], fill=color, font=("Arial", 10, "bold"), anchor=tk.W, tags="marker")
            self.canvas_mesh.create_oval(cx2-6, cy2-6, cx2+6, cy2+6, outline=color, width=2, tags="marker")
            self.canvas_mesh.create_text(cx2+10, cy2, text=p["name"], fill=color, font=("Arial", 10, "bold"), anchor=tk.W, tags="marker")

        # Draw pending
        if self.pending_illus:
            ix, iy = self.pending_illus
            cx, cy = self.image_to_canvas(ix, iy, self.illus_offset, self.illus_scale)
            self.canvas_illus.create_oval(cx-8, cy-8, cx+8, cy+8, outline="yellow", width=3, tags="marker")

        if self.pending_mesh:
            mx, my = self.pending_mesh
            cx, cy = self.image_to_canvas(mx, my, self.mesh_offset, self.mesh_scale)
            self.canvas_mesh.create_oval(cx-8, cy-8, cx+8, cy+8, outline="yellow", width=3, tags="marker")

    def update_status(self):
        view_name = self.view_names[self.current_view]
        view_data = self.landmarks[view_name]
        n_points = len(view_data["points"])
        angle = view_data["angle"]
        selected = self.selected_suggestion or "(none)"
        self.status.config(
            text=f"View: {view_name} ({angle}°) — {n_points} placed | "
                 f"Selected: {selected} | "
                 f"{self.current_view + 1}/{len(self.view_names)} views"
        )

        # Update list
        self.list_text.delete("1.0", tk.END)
        for p in view_data["points"]:
            self.list_text.insert(
                tk.END,
                f"  {p['name']:15s}  illus={tuple(p['illus'])}  mesh={tuple(p['mesh'])}\n",
            )

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--display_size", type=int, default=600)
    args = parser.parse_args()

    picker = LandmarkPicker(args.output_dir, args.display_size)
    picker.run()


if __name__ == "__main__":
    main()
