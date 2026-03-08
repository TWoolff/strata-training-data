"""Match rigged Meshy FBX models to their original unrigged GLBs by vertex count.

The Meshy rigging API strips textures, so rigged FBX/GLB have no materials.
The original unrigged GLBs have 2048x2048 textures. This script matches them
by comparing vertex counts (identical geometry = same model).

Outputs a JSON mapping: { rigged_folder_name: original_glb_path }

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/match_rigged_to_originals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

RIGGED_DIR = Path("/Volumes/TAMWoolff/data/raw/meshy_cc0_rigged")
ORIGINAL_DIR = Path("/Volumes/TAMWoolff/data/raw/meshy_cc0_glb")
OUTPUT_PATH = RIGGED_DIR / "rigged_to_original.json"


def get_vertex_count(filepath: Path) -> int | None:
    """Import a model and return total vertex count across all meshes."""
    bpy.ops.wm.read_homefile(use_empty=True)

    suffix = filepath.suffix.lower()
    try:
        if suffix == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(filepath))
        elif suffix in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(filepath))
        else:
            return None
    except Exception as exc:
        print(f"  Import error: {exc}")
        return None

    total = sum(
        len(obj.data.vertices)
        for obj in bpy.data.objects
        if obj.type == "MESH"
    )
    return total if total > 0 else None


def main() -> None:
    # Index original GLBs by vertex count
    original_glbs = sorted(ORIGINAL_DIR.rglob("*.glb"))
    print(f"Indexing {len(original_glbs)} original GLBs...")

    originals_by_verts: dict[int, list[Path]] = {}
    for i, glb in enumerate(original_glbs):
        if glb.name.startswith("._"):
            continue
        vcount = get_vertex_count(glb)
        if vcount is not None:
            originals_by_verts.setdefault(vcount, []).append(glb)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(original_glbs)} indexed")

    print(f"Indexed {len(originals_by_verts)} unique vertex counts")

    # Match rigged FBX files
    rigged_dirs = sorted(
        d for d in RIGGED_DIR.iterdir()
        if d.is_dir() and d.name.startswith("rigged_")
    )
    print(f"\nMatching {len(rigged_dirs)} rigged models...")

    mapping: dict[str, str] = {}
    matched = 0
    ambiguous = 0
    unmatched = 0

    for i, rdir in enumerate(rigged_dirs):
        fbx = rdir / f"{rdir.name}.fbx"
        if not fbx.exists():
            # Try any FBX in the directory
            fbxs = list(rdir.glob("*.fbx"))
            if not fbxs:
                print(f"  {rdir.name}: no FBX found")
                unmatched += 1
                continue
            fbx = fbxs[0]

        vcount = get_vertex_count(fbx)
        if vcount is None:
            print(f"  {rdir.name}: failed to get vertex count")
            unmatched += 1
            continue

        candidates = originals_by_verts.get(vcount, [])
        if len(candidates) == 1:
            mapping[rdir.name] = str(candidates[0])
            matched += 1
        elif len(candidates) > 1:
            # Multiple originals with same vertex count — pick first
            mapping[rdir.name] = str(candidates[0])
            ambiguous += 1
            print(f"  {rdir.name}: {len(candidates)} candidates with {vcount} verts (using first)")
        else:
            print(f"  {rdir.name}: no match for {vcount} verts")
            unmatched += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rigged_dirs)} matched")

    # Save mapping
    OUTPUT_PATH.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"\nResults: {matched} matched, {ambiguous} ambiguous, {unmatched} unmatched")
    print(f"Mapping saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
