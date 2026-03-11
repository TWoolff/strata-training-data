"""Debug the full rotation chain from BVH to character."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector, Quaternion, Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import _import_animation_bvh, _build_name_map, _apply_tpose

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

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
anim_arm = _import_animation_bvh(pose_dir / "Kick_ID.bvh")
action = bpy.data.actions[0]
if anim_arm.animation_data is None:
    anim_arm.animation_data_create()
anim_arm.animation_data.action = action

# Evaluate at frame 41
scene.frame_set(41)
bpy.context.view_layer.update()

# Build name map
anim_names = [b.name for b in anim_arm.pose.bones]
char_names = [b.name for b in char_arm.pose.bones]
name_map = _build_name_map(anim_names, char_names)

_apply_tpose(char_arm)
bpy.context.view_layer.update()

# Trace the left leg chain: Hips -> LeftHip -> LeftKnee
chain = [("Hips", "Hips"), ("LeftHip", "LeftUpLeg"), ("LeftKnee", "LeftLeg")]

print("=== FRAME 41: LEFT LEG CHAIN ===\n")

for anim_name, char_name in chain:
    anim_pb = anim_arm.pose.bones.get(anim_name)
    char_pb = char_arm.pose.bones.get(char_name)

    print(f"--- {anim_name} -> {char_name} ---")

    # Raw matrices
    anim_rest = anim_pb.bone.matrix_local
    anim_posed = anim_pb.matrix
    char_rest = char_pb.bone.matrix_local

    print(f"  Anim rest (armature space):")
    for row in anim_rest.to_3x3():
        print(f"    [{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f}]")

    print(f"  Anim posed (armature space):")
    for row in anim_posed.to_3x3():
        print(f"    [{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f}]")

    print(f"  Char rest (armature space):")
    for row in char_rest.to_3x3():
        print(f"    [{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f}]")

    # World-space delta
    world_delta = anim_rest.to_3x3().inverted() @ anim_posed.to_3x3()
    world_q = world_delta.to_quaternion()
    print(f"  World delta: {math.degrees(world_q.angle):.1f}°")

    if anim_pb.parent and char_pb.parent:
        anim_pr = anim_pb.parent.bone.matrix_local.to_3x3()
        anim_pp = anim_pb.parent.matrix.to_3x3()
        char_pr = char_pb.parent.bone.matrix_local.to_3x3()

        anim_lr = anim_pr.inverted() @ anim_rest.to_3x3()
        anim_lp = anim_pp.inverted() @ anim_posed.to_3x3()
        char_lr = char_pr.inverted() @ char_rest.to_3x3()

        local_delta = anim_lr.inverted() @ anim_lp
        local_q = local_delta.to_quaternion()
        print(f"  Local delta (parent-relative): {math.degrees(local_q.angle):.1f}°")

        # The correction matrix
        R = anim_lr.inverted() @ char_lr
        R_q = R.to_quaternion()
        print(f"  R correction: {math.degrees(R_q.angle):.1f}°")

        # Original method: R⁻¹ @ delta @ R
        pose1 = R.inverted() @ local_delta @ R
        q1 = pose1.to_quaternion()
        print(f"  Method 1 (R⁻¹ @ Δ @ R): {math.degrees(q1.angle):.1f}°, q={[round(v,3) for v in q1]}")

        # Alternative: R @ delta @ R⁻¹
        pose2 = R @ local_delta @ R.inverted()
        q2 = pose2.to_quaternion()
        print(f"  Method 2 (R @ Δ @ R⁻¹): {math.degrees(q2.angle):.1f}°, q={[round(v,3) for v in q2]}")

        # Direct: just apply local_delta as-is
        q3 = local_delta.to_quaternion()
        print(f"  Method 3 (raw local_delta): {math.degrees(q3.angle):.1f}°, q={[round(v,3) for v in q3]}")

        # Check: are anim and char rest local frames similar?
        anim_lr_q = anim_lr.to_quaternion()
        char_lr_q = char_lr.to_quaternion()
        diff = anim_lr_q.rotation_difference(char_lr_q)
        print(f"  Rest frame diff (anim vs char local): {math.degrees(diff.angle):.1f}°")
    else:
        print(f"  (root bone)")

    print()

# Now let's try just applying raw local deltas and see what happens
print("\n=== APPLYING RAW LOCAL DELTAS ===\n")
_apply_tpose(char_arm)
bpy.context.view_layer.update()

for anim_name, char_name in name_map.items():
    anim_pb = anim_arm.pose.bones.get(anim_name)
    char_pb = char_arm.pose.bones.get(char_name)
    if not anim_pb or not char_pb:
        continue

    name_lower = char_pb.name.lower()
    if any(kw in name_lower for kw in ("thumb", "index", "middle", "ring", "little", "pinky", "finger", "j_sec_", "_sec_")):
        continue

    if anim_pb.parent and char_pb.parent:
        anim_pr = anim_pb.parent.bone.matrix_local.to_3x3()
        anim_pp = anim_pb.parent.matrix.to_3x3()
        anim_lr = anim_pr.inverted() @ anim_pb.bone.matrix_local.to_3x3()
        anim_lp = anim_pp.inverted() @ anim_pb.matrix.to_3x3()
        local_delta = anim_lr.inverted() @ anim_lp
        q = local_delta.to_quaternion()
    else:
        delta = anim_pb.bone.matrix_local.to_3x3().inverted() @ anim_pb.matrix.to_3x3()
        q = delta.to_quaternion()
        e = q.to_euler("YXZ")
        e.y = 0.0
        q = e.to_quaternion()

    char_pb.rotation_mode = "QUATERNION"
    char_pb.rotation_quaternion = q

bpy.context.view_layer.update()

# Check results
for bn in ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "Head", "LeftArm"]:
    pb = char_arm.pose.bones.get(bn)
    if pb:
        q = pb.rotation_quaternion
        angle = math.degrees(2 * math.acos(max(-1, min(1, q[0]))))
        print(f"  {bn}: {angle:.1f}°")
