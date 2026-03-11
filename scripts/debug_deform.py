"""Debug why mesh doesn't follow pose on HumanRig GLB."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import PoseInfo, apply_pose, reset_pose

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")

bpy.ops.wm.read_factory_settings(use_empty=True)

# Setup
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 16
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"

# Import
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
        for mat in obj.data.materials:
            if mat:
                mat.blend_method = "OPAQUE"

mesh = meshes[0]
print(f"Armature: {armature.name}")
print(f"Mesh: {mesh.name}")
print(f"Mesh parent: {mesh.parent.name if mesh.parent else 'None'}")
print(f"Mesh parent type: {mesh.parent_type if mesh.parent else 'N/A'}")
print(f"Mesh modifiers: {[(m.name, m.type, m.object.name if hasattr(m, 'object') and m.object else 'N/A') for m in mesh.modifiers]}")

# Check armature modifier details
for mod in mesh.modifiers:
    if mod.type == "ARMATURE":
        print(f"\nArmature modifier '{mod.name}':")
        print(f"  object: {mod.object.name if mod.object else 'None'}")
        print(f"  use_vertex_groups: {mod.use_vertex_groups}")
        print(f"  use_bone_envelopes: {mod.use_bone_envelopes}")

# Check vertex groups match bone names
vg_names = [vg.name for vg in mesh.vertex_groups]
bone_names = [b.name for b in armature.data.bones]
print(f"\nVertex groups ({len(vg_names)}): {vg_names[:10]}...")
print(f"Bone names ({len(bone_names)}): {bone_names[:10]}...")
matching = set(vg_names) & set(bone_names)
print(f"Matching: {len(matching)}/{len(bone_names)} bones have vertex groups")

# Camera
world = bpy.data.worlds.new("World")
scene.world = world
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 1.3
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = Vector((0, -2.0, 0))
cam_obj.rotation_euler = (math.radians(90), 0, 0)

# Lights
for energy, rx, rz in [(5.0, 45, 30), (5.0, 45, 210)]:
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = energy
    obj = bpy.data.objects.new("Sun", light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(rx), 0, math.radians(rz))

# Render T-pose
bpy.context.view_layer.update()
scene.render.filepath = "output/posed_humanrig_test/debug_tpose.png"
bpy.ops.render.render(write_still=True)
print("\nRendered T-pose")

# Track a vertex position before pose
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_mesh = mesh.evaluated_get(depsgraph)
v0_pre = eval_mesh.data.vertices[0].co.copy()
print(f"Vertex 0 pre-pose: {[round(v, 4) for v in v0_pre]}")

# Apply a dramatic pose - use Kick mid-frame
pose = PoseInfo(name="kick_id_frame_426", source="Kick_ID.bvh", frame=426)
ok = apply_pose(armature, pose, pose_dir)
print(f"\nApplied kick pose: {ok}")

# Check bone moved
pb = armature.pose.bones["LeftUpLeg"]
print(f"LeftUpLeg rotation: {[round(v, 4) for v in pb.rotation_quaternion]}")

# Force full update
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()

# Check vertex moved
eval_mesh = mesh.evaluated_get(depsgraph)
v0_post = eval_mesh.data.vertices[0].co.copy()
print(f"Vertex 0 post-pose: {[round(v, 4) for v in v0_post]}")
print(f"Vertex 0 moved: {(v0_post - v0_pre).length:.6f}")

# Render posed
scene.render.filepath = "output/posed_humanrig_test/debug_kick.png"
bpy.ops.render.render(write_still=True)
print("Rendered kick pose")
