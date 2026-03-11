"""Debug joint extraction on HumanRig GLB."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector, Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.bone_mapper import map_bones
from pipeline.joint_extractor import extract_joints, _select_primary_bone, _project_bone_to_2d, _check_occlusion, REGION_NAMES, NUM_JOINT_REGIONS

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"

bpy.ops.wm.read_factory_settings(use_empty=True)

# Setup scene
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True

# Import GLB
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

# Create orthographic camera (front view)
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 1.3
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Position camera at front (same as humanrig_blender_renderer)
cam_obj.location = Vector((0, -2.0, 0))
cam_obj.rotation_euler = (math.radians(90), 0, 0)

bpy.context.view_layer.update()

# Map bones
mapping = map_bones(armature, meshes, "humanrig_0")
bone_to_region = mapping.bone_to_region

# Manually check a few joints
print("\n=== MANUAL JOINT CHECK ===")
region_to_bones = {}
for bone_name, region_id in bone_to_region.items():
    if region_id == 0:
        continue
    region_to_bones.setdefault(region_id, []).append(bone_name)

for region_id in [1, 5, 7, 14]:  # head, hips, upper_arm_l, upper_leg_l
    region_name = REGION_NAMES[region_id]
    bones = region_to_bones.get(region_id, [])
    primary = _select_primary_bone(region_id, bones)

    if primary is None:
        print(f"  {region_name}: no primary bone")
        continue

    pb = armature.pose.bones.get(primary)
    if pb is None:
        print(f"  {region_name}: pose bone '{primary}' not found")
        continue

    bone_world = armature.matrix_world @ pb.head
    print(f"\n  {region_name} (bone: {primary}):")
    print(f"    World pos: {[round(v, 4) for v in bone_world]}")

    # Project
    proj = _project_bone_to_2d(scene, cam_obj, armature, primary)
    if proj:
        (px, py), depth = proj
        print(f"    Projected: ({px:.1f}, {py:.1f}), depth={depth:.4f}")

        # Check occlusion
        visible = _check_occlusion(scene, cam_obj, armature, primary, meshes, depth)
        print(f"    Visible: {visible}")

        # Manual raycast debug
        cam_forward = cam_obj.matrix_world.to_3x3() @ Vector((0, 0, -1))
        cam_forward.normalize()
        ray_origin = bone_world - cam_forward * 100.0
        depsgraph = bpy.context.evaluated_depsgraph_get()
        hit, location, normal, index, hit_obj, matrix = scene.ray_cast(depsgraph, ray_origin, cam_forward)
        if hit:
            hit_dist = (location - ray_origin).length
            bone_dist = (bone_world - ray_origin).length
            print(f"    Ray hit: obj={hit_obj.name}, hit_dist={hit_dist:.4f}, bone_dist={bone_dist:.4f}")
            print(f"    hit_obj in meshes: {hit_obj in meshes}")
            print(f"    hit_obj.original in meshes: {hit_obj.original in meshes}")
        else:
            print(f"    Ray: NO HIT")
    else:
        print(f"    Projected: FAILED (behind camera?)")

# Run full extraction
print("\n=== FULL EXTRACTION ===")
joints = extract_joints(scene, cam_obj, armature, meshes, bone_to_region)
for name, data in joints.get("joints", {}).items():
    vis = data.get("visible", False)
    pos = data.get("position", [0, 0])
    conf = data.get("confidence", 0)
    print(f"  {name:15s}: pos=({pos[0]:6.1f},{pos[1]:6.1f}) visible={vis} conf={conf:.2f}")
