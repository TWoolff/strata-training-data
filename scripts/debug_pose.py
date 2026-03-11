"""Debug pose application on HumanRig GLB."""
import bpy, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import PoseInfo, apply_pose

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)

for obj in list(bpy.data.objects):
    if obj.name == "Icosphere":
        bpy.data.objects.remove(obj, do_unlink=True)

armature = None
meshes = []
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        armature = obj
    elif obj.type == "MESH":
        meshes.append(obj)

print("=== ARMATURE ===")
print(f"Name: {armature.name}")
bone_names = [b.name for b in armature.data.bones]
print(f"Bones ({len(bone_names)}): {bone_names}")

print("\n=== MESHES ===")
for m in meshes:
    print(f"  {m.name}: parent={m.parent.name if m.parent else None}, "
          f"mods={[(mod.name, mod.type) for mod in m.modifiers]}, "
          f"vgroups={len(m.vertex_groups)}")

# Pre-pose
bpy.context.view_layer.update()
check_bones = ["Head", "LeftHand", "LeftArm", "Hips", "LeftUpLeg"]
pre = {}
print("\n=== PRE-POSE ===")
for bn in check_bones:
    pb = armature.pose.bones.get(bn)
    if pb:
        pre[bn] = pb.head.copy()
        print(f"  {bn}: head={[round(v, 4) for v in pb.head]}, "
              f"rot_q={[round(v, 4) for v in pb.rotation_quaternion]}, "
              f"mode={pb.rotation_mode}")
    else:
        print(f"  {bn}: NOT FOUND")

# Apply akimbo
print("\n=== APPLYING AKIMBO ===")
pose = PoseInfo(name="akimbo_id_frame_01", source="Akimbo_ID.bvh", frame=1)
ok = apply_pose(armature, pose, pose_dir)
print(f"apply_pose result: {ok}")

bpy.context.view_layer.update()

print("\n=== POST-POSE ===")
for bn in check_bones:
    pb = armature.pose.bones.get(bn)
    if pb:
        delta = (pb.head - pre[bn]).length if bn in pre else 0
        print(f"  {bn}: head={[round(v, 4) for v in pb.head]}, "
              f"rot_q={[round(v, 4) for v in pb.rotation_quaternion]}, "
              f"moved={delta:.6f}")
