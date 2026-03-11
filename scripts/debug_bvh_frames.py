"""Scan BVH frames to find where the most motion is."""
import bpy, sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import _import_animation_bvh

bvh_path = Path("/Volumes/TAMWoolff/data/poses_bvh/Kick_ID.bvh")

bpy.ops.wm.read_factory_settings(use_empty=True)

anim_arm = _import_animation_bvh(bvh_path)
action = bpy.data.actions[0]
print(f"Action: {action.name}, range: {action.frame_range[0]}-{action.frame_range[1]}")

if anim_arm.animation_data is None:
    anim_arm.animation_data_create()
anim_arm.animation_data.action = action

scene = bpy.context.scene
check_bone = "LeftHip"

# Sample every 50 frames to find where the action is
start = int(action.frame_range[0])
end = int(action.frame_range[1])
print(f"\n=== {check_bone} rotation from rest across frames ===")
max_angle = 0
max_frame = 0
for frame in range(start, end + 1, 20):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    pb = anim_arm.pose.bones.get(check_bone)
    if pb:
        rest_3x3 = pb.bone.matrix_local.to_3x3()
        posed_3x3 = pb.matrix.to_3x3()
        delta = rest_3x3.inverted() @ posed_3x3
        q = delta.to_quaternion()
        angle = math.degrees(q.angle)
        bar = "#" * int(angle / 2)
        print(f"  frame {frame:4d}: {angle:5.1f}° {bar}")
        if angle > max_angle:
            max_angle = angle
            max_frame = frame

print(f"\nMax rotation: {max_angle:.1f}° at frame {max_frame}")

# Also check another BVH
print("\n\n=== Checking Lunge_ID.bvh ===")
for obj in list(bpy.data.objects):
    if obj.type == "ARMATURE":
        bpy.data.objects.remove(obj, do_unlink=True)
for a in list(bpy.data.actions):
    bpy.data.actions.remove(a)
for ad in list(bpy.data.armatures):
    bpy.data.armatures.remove(ad)

bvh_path2 = Path("/Volumes/TAMWoolff/data/poses_bvh/Lunge_ID.bvh")
anim_arm2 = _import_animation_bvh(bvh_path2)
action2 = bpy.data.actions[0]
print(f"Action: {action2.name}, range: {action2.frame_range[0]}-{action2.frame_range[1]}")

if anim_arm2.animation_data is None:
    anim_arm2.animation_data_create()
anim_arm2.animation_data.action = action2

max_angle = 0
max_frame = 0
start2 = int(action2.frame_range[0])
end2 = int(action2.frame_range[1])
for frame in range(start2, end2 + 1, 20):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    pb = anim_arm2.pose.bones.get("LeftHip")
    if pb:
        rest_3x3 = pb.bone.matrix_local.to_3x3()
        posed_3x3 = pb.matrix.to_3x3()
        delta = rest_3x3.inverted() @ posed_3x3
        q = delta.to_quaternion()
        angle = math.degrees(q.angle)
        bar = "#" * int(angle / 2)
        print(f"  frame {frame:4d}: {angle:5.1f}° {bar}")
        if angle > max_angle:
            max_angle = angle
            max_frame = frame

print(f"\nMax rotation: {max_angle:.1f}° at frame {max_frame}")
