"""Debug BVH pose transfer: dump raw rotations from BVH armature vs what's applied."""
import bpy, sys, math
from pathlib import Path
from mathutils import Quaternion, Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import _import_animation_bvh, _build_name_map, _apply_tpose

bvh_path = Path("/Volumes/TAMWoolff/data/poses_bvh/Kick_ID.bvh")
glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import character
bpy.ops.import_scene.gltf(filepath=glb)
for obj in list(bpy.data.objects):
    if obj.name == "Icosphere":
        bpy.data.objects.remove(obj, do_unlink=True)

char_arm = None
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        char_arm = obj
        break

# Import BVH
anim_arm = _import_animation_bvh(bvh_path)
print(f"BVH armature: {anim_arm.name}")

# Check action
action = None
if anim_arm.animation_data and anim_arm.animation_data.action:
    action = anim_arm.animation_data.action
if action is None and bpy.data.actions:
    action = bpy.data.actions[0]

print(f"Action: {action.name if action else 'NONE'}")
if action:
    print(f"Frame range: {action.frame_range[0]} - {action.frame_range[1]}")

# Bind action
if anim_arm.animation_data is None:
    anim_arm.animation_data_create()
anim_arm.animation_data.action = action

# Check at rest frame (frame 0/1)
scene = bpy.context.scene
scene.frame_set(1)
bpy.context.view_layer.update()

print("\n=== BVH BONES AT REST (frame 1) ===")
check_bones = ["LeftHip", "LeftKnee", "LeftAnkle", "RightHip", "Hips", "Head"]
for bn in check_bones:
    pb = anim_arm.pose.bones.get(bn)
    if pb:
        rest_3x3 = pb.bone.matrix_local.to_3x3()
        posed_3x3 = pb.matrix.to_3x3()
        delta = rest_3x3.inverted() @ posed_3x3
        q = delta.to_quaternion()
        angle = math.degrees(q.angle)
        print(f"  {bn}: angle_from_rest={angle:.1f}°, q={[round(v, 4) for v in q]}")

# Now check mid-frame (where the kick should be happening)
mid_frame = int((action.frame_range[0] + action.frame_range[1]) / 2)
print(f"\n=== BVH BONES AT MID-FRAME ({mid_frame}) ===")
scene.frame_set(mid_frame)
bpy.context.view_layer.update()

for bn in check_bones:
    pb = anim_arm.pose.bones.get(bn)
    if pb:
        rest_3x3 = pb.bone.matrix_local.to_3x3()
        posed_3x3 = pb.matrix.to_3x3()
        delta = rest_3x3.inverted() @ posed_3x3
        q = delta.to_quaternion()
        angle = math.degrees(q.angle)
        print(f"  {bn}: angle_from_rest={angle:.1f}°, q={[round(v, 4) for v in q]}")

# Check at frame 426 (the one we tried before)
print(f"\n=== BVH BONES AT FRAME 426 ===")
scene.frame_set(426)
bpy.context.view_layer.update()

for bn in check_bones:
    pb = anim_arm.pose.bones.get(bn)
    if pb:
        rest_3x3 = pb.bone.matrix_local.to_3x3()
        posed_3x3 = pb.matrix.to_3x3()
        delta = rest_3x3.inverted() @ posed_3x3
        q = delta.to_quaternion()
        angle = math.degrees(q.angle)
        print(f"  {bn}: angle_from_rest={angle:.1f}°, q={[round(v, 4) for v in q]}")

# Now do the transfer manually and show each step
print(f"\n=== TRANSFER DEBUG AT FRAME 426 ===")
scene.frame_set(426)
bpy.context.view_layer.update()

anim_bone_names = [b.name for b in anim_arm.pose.bones]
char_bone_names = [b.name for b in char_arm.pose.bones]
name_map = _build_name_map(anim_bone_names, char_bone_names)
print(f"Name map ({len(name_map)} matches):")
for a, c in sorted(name_map.items()):
    print(f"  {a} -> {c}")

_apply_tpose(char_arm)
bpy.context.view_layer.update()

# Transfer LeftHip specifically
pairs_to_debug = [("LeftHip", "LeftUpLeg"), ("LeftKnee", "LeftLeg"), ("Hips", "Hips")]
for anim_name, expected_char in pairs_to_debug:
    char_name = name_map.get(anim_name)
    if char_name is None:
        print(f"\n  {anim_name}: NOT MAPPED")
        continue

    anim_pbone = anim_arm.pose.bones.get(anim_name)
    char_pbone = char_arm.pose.bones.get(char_name)
    if not anim_pbone or not char_pbone:
        print(f"\n  {anim_name} -> {char_name}: bone not found")
        continue

    print(f"\n  {anim_name} -> {char_name}:")

    anim_rest = anim_pbone.bone.matrix_local.to_3x3()
    anim_posed = anim_pbone.matrix.to_3x3()
    char_rest = char_pbone.bone.matrix_local.to_3x3()

    # World-space delta (how much did the anim bone actually rotate?)
    world_delta = anim_rest.inverted() @ anim_posed
    world_q = world_delta.to_quaternion()
    print(f"    Anim world delta: {math.degrees(world_q.angle):.1f}°")

    if anim_pbone.parent and char_pbone.parent:
        anim_parent_rest = anim_pbone.parent.bone.matrix_local.to_3x3()
        char_parent_rest = char_pbone.parent.bone.matrix_local.to_3x3()
        anim_parent_posed = anim_pbone.parent.matrix.to_3x3()

        anim_rest_local = anim_parent_rest.inverted() @ anim_rest
        char_rest_local = char_parent_rest.inverted() @ char_rest
        anim_posed_local = anim_parent_posed.inverted() @ anim_posed

        local_delta = anim_rest_local.inverted() @ anim_posed_local
        R = anim_rest_local.inverted() @ char_rest_local
        pose_basis = R.inverted() @ local_delta @ R
        q = pose_basis.to_quaternion()

        local_delta_q = local_delta.to_quaternion()
        R_q = R.to_quaternion()
        print(f"    Local delta: {math.degrees(local_delta_q.angle):.1f}°")
        print(f"    R correction: {math.degrees(R_q.angle):.1f}°")
        print(f"    Final pose_basis: {math.degrees(q.angle):.1f}°, q={[round(v, 4) for v in q]}")
        print(f"    Anim parent: {anim_pbone.parent.name}, char parent: {char_pbone.parent.name}")
    else:
        local_delta = anim_rest.inverted() @ anim_posed
        R = anim_rest.inverted() @ char_rest
        pose_basis = R.inverted() @ local_delta @ R
        q = pose_basis.to_quaternion()
        print(f"    Root: delta={math.degrees(q.angle):.1f}°")

print("\nDone.")
