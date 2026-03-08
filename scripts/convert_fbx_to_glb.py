"""Batch convert FBX files to GLB using Blender.

Also decimates high-poly models to stay under Meshy's 300K face limit.

Usage:
    blender --background --python scripts/convert_fbx_to_glb.py -- \
        --input-dir /Volumes/TAMWoolff/data/raw/meshy_cc0 \
        --output-dir /Volumes/TAMWoolff/data/raw/meshy_cc0_glb \
        --max-faces 280000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy  # type: ignore[import-untyped]


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def import_fbx(fbx_path: Path) -> None:
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))


def get_face_count() -> int:
    total = 0
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            total += len(obj.data.polygons)
    return total


def decimate_to_target(max_faces: int) -> None:
    current = get_face_count()
    if current <= max_faces:
        return

    ratio = max_faces / current
    for obj in bpy.data.objects:
        if obj.type == "MESH" and len(obj.data.polygons) > 100:
            mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
            mod.ratio = ratio
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod.name)

    final = get_face_count()
    print(f"  Decimated {current:,} -> {final:,} faces (ratio={ratio:.3f})")


def export_glb(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Blender 5.0+ changed gltf export params
    try:
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            export_texcoords=True,
            export_normals=True,
            export_materials="EXPORT",
            export_image_format="AUTO",
        )
    except TypeError:
        # Fallback: minimal params for Blender 5.0+
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
        )


def process_one(fbx_path: Path, output_dir: Path, max_faces: int) -> bool:
    name = fbx_path.parent.name
    output_path = output_dir / name / f"{name}.glb"

    if output_path.exists():
        print(f"  SKIP (exists): {name}")
        return True

    clear_scene()

    try:
        import_fbx(fbx_path)
    except Exception as exc:
        print(f"  FAIL import: {name}: {exc}")
        return False

    faces = get_face_count()
    print(f"  {name}: {faces:,} faces", end="")

    if faces > max_faces:
        decimate_to_target(max_faces)

    try:
        export_glb(output_path)
        print(f" -> {output_path.name}")
        return True
    except Exception as exc:
        print(f"  FAIL export: {name}: {exc}")
        return False


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-faces", type=int, default=280000,
                        help="Decimate above this (Meshy limit is 300K)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N models (0=all)")
    args = parser.parse_args(argv)

    fbx_files = sorted(args.input_dir.rglob("*.fbx"))
    fbx_files = [f for f in fbx_files if not f.name.startswith("._")]

    if args.limit > 0:
        fbx_files = fbx_files[:args.limit]

    print(f"Found {len(fbx_files)} FBX files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    fail = 0
    for i, fbx in enumerate(fbx_files):
        print(f"[{i+1}/{len(fbx_files)}] {fbx.parent.name}")
        if process_one(fbx, args.output_dir, args.max_faces):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} converted, {fail} failed")


if __name__ == "__main__":
    main()
