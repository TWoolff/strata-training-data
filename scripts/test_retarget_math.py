"""Minimal retarget test: import Mixamo FBX + VRM, transfer one pose, render.

Run in Blender:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/test_retarget_math.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bpy
from mathutils import Matrix, Quaternion, Vector

# --- Clean scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- Import VRM character ---
vrm_path = Path("/Volumes/TAMWoolff/data/raw/vroid_cc0/1481154051300941882.glb")
bpy.ops.import_scene.gltf(filepath=str(vrm_path))
vrm_arm = None
for obj in bpy.context.scene.objects:
    if obj.type == "ARMATURE":
        vrm_arm = obj
        break
print(f"VRM armature: {vrm_arm.name}")

# --- Import Mixamo animation FBX ---
fbx_path = Path("/Volumes/TAMWoolff/data/poses/Hurricane Kick.fbx")
before = set(o.name for o in bpy.data.objects)
bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=True, ignore_leaf_bones=True)
after = set(o.name for o in bpy.data.objects)
mix_arm = None
for name in (after - before):
    obj = bpy.data.objects.get(name)
    if obj and obj.type == "ARMATURE":
        mix_arm = obj
        break
print(f"Mixamo armature: {mix_arm.name}")

# --- Bind action ---
action = None
if mix_arm.animation_data and mix_arm.animation_data.action:
    action = mix_arm.animation_data.action
if action is None and bpy.data.actions:
    action = bpy.data.actions[0]
if mix_arm.animation_data is None:
    mix_arm.animation_data_create()
mix_arm.animation_data.action = action
if hasattr(mix_arm.animation_data, "action_slot"):
    slots = list(action.slots) if hasattr(action, "slots") else []
    if slots:
        try:
            mix_arm.animation_data.action_slot = slots[0]
        except Exception:
            pass

# --- Capture rest matrices ---
mix_rest = {}
for bone in mix_arm.data.bones:
    mix_rest[bone.name] = bone.matrix_local.copy()

vrm_rest = {}
for bone in vrm_arm.data.bones:
    vrm_rest[bone.name] = bone.matrix_local.copy()

# --- Evaluate at frame 20 (mid-kick) ---
bpy.context.scene.frame_set(20)
bpy.context.view_layer.update()

mix_posed = {}
for pbone in mix_arm.pose.bones:
    mix_posed[pbone.name] = pbone.matrix.copy()

# --- Build name map ---
from pipeline.pose_applicator import _build_name_map
mix_names = [b.name for b in mix_arm.pose.bones]
vrm_names = [b.name for b in vrm_arm.pose.bones]
name_map = _build_name_map(mix_names, vrm_names)
print(f"\nMatched {len(name_map)} bones")

# --- Method: world-space delta, but ONLY rotation (3x3), ignore translation ---
# Reset VRM to T-pose
for pbone in vrm_arm.pose.bones:
    pbone.location = (0, 0, 0)
    pbone.rotation_quaternion = (1, 0, 0, 0)
    pbone.rotation_euler = (0, 0, 0)
    pbone.scale = (1, 1, 1)
bpy.context.view_layer.update()

# Sort by depth
def bone_depth(bname):
    b = vrm_arm.data.bones.get(bname)
    d = 0
    while b and b.parent:
        d += 1
        b = b.parent
    return d

pairs = []
for mix_name, vrm_name in name_map.items():
    lower = vrm_name.lower()
    if "j_sec_" in lower or "_sec_" in lower:
        continue
    if any(kw in lower for kw in ("thumb", "index", "middle", "ring", "little", "pinky")):
        continue
    if mix_name in mix_rest and mix_name in mix_posed and vrm_name in vrm_rest:
        pairs.append((mix_name, vrm_name))

pairs.sort(key=lambda p: bone_depth(p[1]))

print(f"\nProcessing {len(pairs)} bone pairs (sorted by depth):")
print(f"{'Mixamo':<35} {'VRM':<35} {'Delta angle':<15}")
print("-" * 85)

for mix_name, vrm_name in pairs:
    mr = mix_rest[mix_name].to_3x3()
    mp = mix_posed[mix_name].to_3x3()
    cr = vrm_rest[vrm_name].to_3x3()

    # World-space delta
    world_delta = mp @ mr.inverted()

    # Angle of the delta
    dq = world_delta.to_quaternion()
    import math
    angle_deg = math.degrees(2 * math.acos(max(-1, min(1, dq.w))))

    # Target for VRM bone in armature space
    char_target = world_delta @ cr

    vrm_pbone = vrm_arm.pose.bones.get(vrm_name)

    if vrm_pbone.parent:
        bpy.context.view_layer.update()
        parent_posed = vrm_pbone.parent.matrix.to_3x3()
        parent_rest_mat = vrm_rest.get(vrm_pbone.parent.bone.name)
        if parent_rest_mat is not None:
            rest_offset = parent_rest_mat.to_3x3().inverted() @ cr
        else:
            rest_offset = cr
        in_parent = parent_posed.inverted() @ char_target
        q = (rest_offset.inverted() @ in_parent).to_quaternion()
    else:
        q = (cr.inverted() @ char_target).to_quaternion()

    vrm_pbone.rotation_mode = "QUATERNION"
    vrm_pbone.rotation_quaternion = q

    print(f"{mix_name:<35} {vrm_name:<35} {angle_deg:>8.1f}°")

bpy.context.view_layer.update()

# --- Render to check ---
# Set up camera
cam = bpy.data.cameras.new("TestCam")
cam_obj = bpy.data.objects.new("TestCam", cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam.type = 'ORTHO'
cam.ortho_scale = 3.0
cam_obj.location = (0, -5, 1.0)
cam_obj.rotation_euler = (1.5708, 0, 0)  # point at origin

# Hide mixamo armature
mix_arm.hide_render = True
mix_arm.hide_viewport = True
for child in mix_arm.children:
    child.hide_render = True
    child.hide_viewport = True

# Setup render
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

# Add light
light = bpy.data.lights.new("Sun", 'SUN')
light_obj = bpy.data.objects.new("Sun", light)
bpy.context.scene.collection.objects.link(light_obj)
light.energy = 2.0

out_path = "/tmp/retarget_test_worlddelta.png"
bpy.context.scene.render.filepath = out_path
bpy.ops.render.render(write_still=True)
print(f"\nRendered to {out_path}")
