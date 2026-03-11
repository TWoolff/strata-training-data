"""Debug: why mesh doesn't follow skeleton with Mixamo FBX poses."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector, Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Import character
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

mesh = meshes[0]

# Check armature modifier
print("=== ARMATURE MODIFIER CHECK ===")
for mod in mesh.modifiers:
    print(f"  {mod.name}: type={mod.type}, object={mod.object.name if hasattr(mod, 'object') and mod.object else 'None'}")
    if mod.type == "ARMATURE":
        print(f"    use_vertex_groups={mod.use_vertex_groups}")
        print(f"    use_bone_envelopes={mod.use_bone_envelopes}")
        print(f"    show_viewport={mod.show_viewport}")
        print(f"    show_render={mod.show_render}")

# Get vertex position before any pose
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_mesh = mesh.evaluated_get(depsgraph)
v0_rest = eval_mesh.data.vertices[0].co.copy()
print(f"\nVertex 0 at rest: {[round(v, 4) for v in v0_rest]}")

# Test 1: Manually set a large rotation on LeftUpLeg directly
print("\n=== TEST 1: MANUAL ROTATION ===")
pb = armature.pose.bones["LeftUpLeg"]
print(f"LeftUpLeg rotation_mode: {pb.rotation_mode}")
pb.rotation_mode = "QUATERNION"
# Rotate 90 degrees around X (forward kick)
pb.rotation_quaternion = Quaternion((1, 0, 0), math.radians(90))
print(f"Set rotation: {[round(v, 4) for v in pb.rotation_quaternion]}")

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_mesh = mesh.evaluated_get(depsgraph)
v0_posed = eval_mesh.data.vertices[0].co.copy()
print(f"Vertex 0 after manual pose: {[round(v, 4) for v in v0_posed]}")
print(f"Vertex 0 moved: {(v0_posed - v0_rest).length:.6f}")

# Check a vertex that's definitely in the left leg area
# Find vertices weighted to LeftUpLeg
left_upleg_vg = mesh.vertex_groups.get("LeftUpLeg")
if left_upleg_vg:
    vg_idx = left_upleg_vg.index
    weighted_verts = []
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == vg_idx and g.weight > 0.5:
                weighted_verts.append(v.index)
                break
    print(f"\nVertices heavily weighted to LeftUpLeg: {len(weighted_verts)}")
    if weighted_verts:
        # Check first weighted vertex
        vi = weighted_verts[0]
        # Need to get rest position first
        # Reset pose
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = mesh.evaluated_get(depsgraph)
        v_rest = eval_mesh.data.vertices[vi].co.copy()

        # Apply pose again
        pb.rotation_quaternion = Quaternion((1, 0, 0), math.radians(90))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = mesh.evaluated_get(depsgraph)
        v_posed = eval_mesh.data.vertices[vi].co.copy()

        print(f"  Vertex {vi} rest: {[round(v, 4) for v in v_rest]}")
        print(f"  Vertex {vi} posed: {[round(v, 4) for v in v_posed]}")
        print(f"  Vertex {vi} moved: {(v_posed - v_rest).length:.6f}")

# Test 2: Now import a Mixamo FBX and apply pose via _apply_animation_pose
print("\n=== TEST 2: VIA _apply_animation_pose ===")
from pipeline.pose_applicator import (
    _import_animation_fbx, _apply_animation_pose, _cleanup_imported_armature, _apply_tpose
)

# Reset first
_apply_tpose(armature)
bpy.context.view_layer.update()

# Import Mixamo FBX
fbx_path = Path("/Volumes/TAMWoolff/data/poses/Hurricane Kick.fbx")
anim_arm = _import_animation_fbx(fbx_path)
print(f"Imported animation armature: {anim_arm.name}")
print(f"Anim bones: {[b.name for b in anim_arm.pose.bones][:10]}...")

# Apply
transferred = _apply_animation_pose(armature, anim_arm, 29)
print(f"Transferred: {transferred} bones")

# Check LeftUpLeg rotation
pb = armature.pose.bones["LeftUpLeg"]
q = pb.rotation_quaternion
angle = math.degrees(2 * math.acos(max(-1, min(1, q[0]))))
print(f"LeftUpLeg: {angle:.1f}°, q={[round(v, 3) for v in q]}")

# Check vertex deformation
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_mesh = mesh.evaluated_get(depsgraph)

if weighted_verts:
    vi = weighted_verts[0]
    v_after = eval_mesh.data.vertices[vi].co.copy()
    print(f"Vertex {vi} after _apply_animation_pose: {[round(v, 4) for v in v_after]}")
    print(f"Vertex {vi} moved from rest: {(v_after - v_rest).length:.6f}")

# Check if there's an action bound that might be overriding
print(f"\nChar armature animation_data: {armature.animation_data}")
if armature.animation_data:
    print(f"  action: {armature.animation_data.action}")
    print(f"  action_influence: {armature.animation_data.action_influence if hasattr(armature.animation_data, 'action_influence') else 'N/A'}")

_cleanup_imported_armature(anim_arm, keep_actions=False)

# Check if any actions remain
print(f"\nRemaining actions: {[a.name for a in bpy.data.actions]}")

print("\nDone!")
