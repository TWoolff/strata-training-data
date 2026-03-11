"""Debug script: compare Mixamo and VRM bone orientations.

Run in Blender:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/debug_retarget.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bpy
from mathutils import Matrix, Quaternion

# --- Import one VRM character ---
vrm_path = Path("/Volumes/TAMWoolff/data/raw/vroid_cc0/1481154051300941882.glb")
bpy.ops.import_scene.gltf(filepath=str(vrm_path))
vrm_arm = None
for obj in bpy.context.scene.objects:
    if obj.type == "ARMATURE":
        vrm_arm = obj
        break

print("\n" + "=" * 80)
print("VRM ARMATURE:", vrm_arm.name if vrm_arm else "NOT FOUND")
print("=" * 80)

if vrm_arm:
    # Print key bones
    key_bones = ["J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest",
                 "J_Bip_L_UpperArm", "J_Bip_L_LowerArm", "J_Bip_L_Hand",
                 "J_Bip_R_UpperArm", "J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg"]
    print(f"\n{'Bone':<30} {'Y-axis (rest)':<30} {'matrix_local rot':<40}")
    print("-" * 100)
    for name in key_bones:
        bone = vrm_arm.data.bones.get(name)
        if bone:
            y = bone.matrix_local.to_3x3() @ bpy.mathutils.Vector((0, 1, 0)) if hasattr(bpy, 'mathutils') else None
            rot = bone.matrix_local.to_3x3().to_quaternion()
            y_axis = bone.y_axis
            print(f"{name:<30} {str(tuple(round(v,3) for v in y_axis)):<30} {str(tuple(round(v,4) for v in rot)):<40}")
        else:
            print(f"{name:<30} NOT FOUND")

    # List ALL bone names
    print(f"\nAll VRM bones ({len(vrm_arm.data.bones)}):")
    for b in vrm_arm.data.bones:
        parent = b.parent.name if b.parent else "ROOT"
        print(f"  {b.name:<40} parent={parent}")

# --- Import one Mixamo animation ---
print("\n" + "=" * 80)
print("IMPORTING MIXAMO ANIMATION")
print("=" * 80)

fbx_path = Path("/Volumes/TAMWoolff/data/poses/Catwalk Walk Forward HighKnees.fbx")
before = set(o.name for o in bpy.data.objects)
bpy.ops.import_scene.fbx(filepath=str(fbx_path))
after = set(o.name for o in bpy.data.objects)
new_objs = after - before

mix_arm = None
for name in new_objs:
    obj = bpy.data.objects.get(name)
    if obj and obj.type == "ARMATURE":
        mix_arm = obj
        break

print("MIXAMO ARMATURE:", mix_arm.name if mix_arm else "NOT FOUND")

if mix_arm:
    key_bones_mix = ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
                     "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
                     "mixamorig:RightArm", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg"]
    print(f"\n{'Bone':<35} {'Y-axis (rest)':<30} {'matrix_local rot':<40}")
    print("-" * 105)
    for name in key_bones_mix:
        bone = mix_arm.data.bones.get(name)
        if bone:
            rot = bone.matrix_local.to_3x3().to_quaternion()
            y_axis = bone.y_axis
            print(f"{name:<35} {str(tuple(round(v,3) for v in y_axis)):<30} {str(tuple(round(v,4) for v in rot)):<40}")
        else:
            print(f"{name:<35} NOT FOUND")

    # List ALL bone names
    print(f"\nAll Mixamo bones ({len(mix_arm.data.bones)}):")
    for b in mix_arm.data.bones:
        parent = b.parent.name if b.parent else "ROOT"
        print(f"  {b.name:<40} parent={parent}")

    # --- Test: evaluate animation at frame 10 and show posed matrices ---
    action = None
    if mix_arm.animation_data and mix_arm.animation_data.action:
        action = mix_arm.animation_data.action
    if action is None and bpy.data.actions:
        action = bpy.data.actions[0]

    if action:
        if mix_arm.animation_data is None:
            mix_arm.animation_data_create()
        mix_arm.animation_data.action = action
        bpy.context.scene.frame_set(10)
        bpy.context.view_layer.update()

        print(f"\n{'Bone':<35} {'Posed Y-axis':<30} {'pose rotation_quat':<40}")
        print("-" * 105)
        for name in key_bones_mix:
            pbone = mix_arm.pose.bones.get(name)
            if pbone:
                pbone.rotation_mode = "QUATERNION"
                y_axis = pbone.matrix.to_3x3() @ bpy.mathutils.Vector((0,1,0)) if hasattr(bpy, 'mathutils') else pbone.y_axis
                q = pbone.rotation_quaternion
                posed_y = (pbone.matrix.to_3x3() @ type(pbone.matrix.to_3x3().col[0])((0,1,0)))
                print(f"{name:<35} {str(tuple(round(v,3) for v in posed_y)):<30} {str(tuple(round(v,4) for v in q)):<40}")

# --- Test name mapping ---
print("\n" + "=" * 80)
print("TESTING NAME MAPPING")
print("=" * 80)

from pipeline.pose_applicator import _build_name_map

if mix_arm and vrm_arm:
    mix_names = [b.name for b in mix_arm.pose.bones]
    vrm_names = [b.name for b in vrm_arm.pose.bones]
    name_map = _build_name_map(mix_names, vrm_names)

    print(f"\nMatched {len(name_map)} bones:")
    for src, tgt in sorted(name_map.items()):
        print(f"  {src:<40} → {tgt}")

    unmatched_mix = [n for n in mix_names if n not in name_map]
    print(f"\nUnmatched Mixamo bones ({len(unmatched_mix)}):")
    for n in unmatched_mix:
        print(f"  {n}")
