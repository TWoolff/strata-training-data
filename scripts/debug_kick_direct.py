"""Test direct world-space rotation transfer (bypass correction matrix)."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector, Quaternion, Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import (
    _import_animation_bvh, _build_name_map, _apply_tpose, _cleanup_imported_armature
)
from pipeline.bone_mapper import map_bones
from pipeline.joint_extractor import extract_joints

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")
output_dir = Path("output/posed_humanrig_test4")
output_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.cycles.device = "CPU"
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Strength"].default_value = 0.0

for energy, rx, rz in [(5.0, 45, 30), (5.0, 45, 210)]:
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = energy
    obj = bpy.data.objects.new("Sun", light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(rx), 0, math.radians(rz))

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb)
for obj in list(bpy.data.objects):
    if obj.name == "Icosphere":
        bpy.data.objects.remove(obj, do_unlink=True)

char_arm = None
meshes = []
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        char_arm = obj
    elif obj.type == "MESH":
        meshes.append(obj)
        for mat in obj.data.materials:
            if mat:
                mat.blend_method = "OPAQUE"

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 1.3
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = Vector((0, -2.0, 0))
cam_obj.rotation_euler = (math.radians(90), 0, 0)

mapping = map_bones(char_arm, meshes, "humanrig_0")
bone_to_region = mapping.bone_to_region


def apply_bvh_pose_direct(char_arm, bvh_path, frame):
    """Apply BVH pose using direct world-space delta transfer."""
    anim_arm = _import_animation_bvh(bvh_path)
    if not anim_arm:
        return 0

    action = None
    if anim_arm.animation_data and anim_arm.animation_data.action:
        action = anim_arm.animation_data.action
    if action is None and bpy.data.actions:
        action = bpy.data.actions[0]
    if not action:
        _cleanup_imported_armature(anim_arm)
        return 0

    if anim_arm.animation_data is None:
        anim_arm.animation_data_create()
    anim_arm.animation_data.action = action

    # Evaluate at target frame
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    # Build name map
    anim_names = [b.name for b in anim_arm.pose.bones]
    char_names = [b.name for b in char_arm.pose.bones]
    name_map = _build_name_map(anim_names, char_names)

    # Reset character
    _apply_tpose(char_arm)
    bpy.context.view_layer.update()

    # Sort by depth
    def _depth(bn):
        b = char_arm.data.bones.get(bn)
        d = 0
        while b and b.parent:
            d += 1
            b = b.parent
        return d

    sorted_pairs = sorted(name_map.items(), key=lambda p: _depth(p[1]))

    transferred = 0
    for anim_name, char_name in sorted_pairs:
        anim_pb = anim_arm.pose.bones.get(anim_name)
        char_pb = char_arm.pose.bones.get(char_name)
        if not anim_pb or not char_pb:
            continue

        name_lower = char_pb.name.lower()
        if any(kw in name_lower for kw in ("thumb", "index", "middle", "ring", "little", "pinky", "finger", "j_sec_", "_sec_")):
            continue

        # Get the animation bone's LOCAL rotation delta.
        # In Blender, pose bone's .matrix is the final world-space (armature-space) matrix.
        # bone.matrix_local is the rest pose in armature space.
        # The actual pose rotation is: rest_local_inv @ posed_matrix
        # But we need to account for parent transformations properly.

        # Method: Extract the pose-space rotation directly.
        # For the animation bone, the "pose delta" from rest is:
        #   anim_rest_inv @ anim_posed (in armature space)
        # This is equivalent to the accumulated rotation from rest.
        #
        # For proper retargeting, we want the LOCAL rotation delta
        # (how much this bone rotated relative to its parent's posed frame,
        # compared to how it sits in rest relative to parent's rest frame).

        anim_rest = anim_pb.bone.matrix_local
        anim_posed = anim_pb.matrix
        char_rest = char_pb.bone.matrix_local

        if anim_pb.parent and char_pb.parent:
            anim_parent_rest = anim_pb.parent.bone.matrix_local
            anim_parent_posed = anim_pb.parent.matrix
            char_parent_rest = char_pb.parent.bone.matrix_local

            # Local rest: how bone sits relative to parent in rest
            anim_local_rest = anim_parent_rest.inverted() @ anim_rest
            char_local_rest = char_parent_rest.inverted() @ char_rest

            # Local posed: how bone sits relative to parent when posed
            anim_local_posed = anim_parent_posed.inverted() @ anim_posed

            # The local rotation delta: from local rest to local posed
            local_delta = anim_local_rest.inverted() @ anim_local_posed

            # Now we need to re-express this in the character bone's frame.
            # The key insight: local_delta is expressed in anim bone's local axes.
            # We need to convert to char bone's local axes.
            #
            # If we define:
            #   R = char_local_rest.inverted() @ anim_local_rest
            # Then R maps from anim local frame to char local frame.
            # The delta in char frame = R @ local_delta @ R.inverted()

            R = char_local_rest.to_3x3().inverted() @ anim_local_rest.to_3x3()
            delta_3x3 = local_delta.to_3x3()
            corrected = R @ delta_3x3 @ R.inverted()
            q = corrected.to_quaternion()
        elif not anim_pb.parent and not char_pb.parent:
            # Root bone
            anim_rest_3 = anim_rest.to_3x3()
            anim_posed_3 = anim_posed.to_3x3()
            char_rest_3 = char_rest.to_3x3()

            delta = anim_rest_3.inverted() @ anim_posed_3
            R = char_rest_3.inverted() @ anim_rest_3
            corrected = R @ delta @ R.inverted()
            q = corrected.to_quaternion()

            # Zero out facing direction
            e = q.to_euler("YXZ")
            e.y = 0.0
            q = e.to_quaternion()
        else:
            # Mismatched hierarchy
            continue

        char_pb.rotation_mode = "QUATERNION"
        char_pb.rotation_quaternion = q
        transferred += 1

    bpy.context.view_layer.update()
    _cleanup_imported_armature(anim_arm, keep_actions=False)
    return transferred


# Test frames
from PIL import Image, ImageDraw

JOINT_COLORS = [
    (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0),
    (0, 255, 0), (0, 255, 128), (0, 255, 255), (0, 128, 255),
    (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128),
    (128, 0, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (255, 128, 255), (128, 255, 255), (255, 255, 128),
]
connections = [
    ("head", "neck"), ("neck", "chest"), ("chest", "spine"), ("spine", "hips"),
    ("neck", "shoulder_l"), ("shoulder_l", "upper_arm_l"),
    ("upper_arm_l", "forearm_l"), ("forearm_l", "hand_l"),
    ("neck", "shoulder_r"), ("shoulder_r", "upper_arm_r"),
    ("upper_arm_r", "forearm_r"), ("forearm_r", "hand_r"),
    ("hips", "upper_leg_l"), ("upper_leg_l", "lower_leg_l"), ("lower_leg_l", "foot_l"),
    ("hips", "upper_leg_r"), ("upper_leg_r", "lower_leg_r"), ("lower_leg_r", "foot_r"),
]
region_names = [
    "head", "neck", "chest", "spine", "hips",
    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
]

for frame in [41, 61]:
    _apply_tpose(char_arm)
    bpy.context.view_layer.update()

    n = apply_bvh_pose_direct(char_arm, pose_dir / "Kick_ID.bvh", frame)
    print(f"\nFrame {frame}: transferred {n} bones")

    for bn in ["LeftUpLeg", "LeftLeg", "Hips"]:
        pb = char_arm.pose.bones.get(bn)
        if pb:
            q = pb.rotation_quaternion
            angle = math.degrees(2 * math.acos(max(-1, min(1, q[0]))))
            print(f"  {bn}: angle={angle:.1f}°, q={[round(v, 3) for v in q]}")

    joints_data = extract_joints(scene, cam_obj, char_arm, meshes, bone_to_region)

    render_path = output_dir / f"kick_direct_f{frame}_render.png"
    scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)

    # Overlay
    img = Image.open(str(render_path)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    joints = joints_data.get("joints", {})
    for a, b in connections:
        ja, jb = joints.get(a, {}), joints.get(b, {})
        if ja.get("visible") and jb.get("visible"):
            pa, pb_ = ja["position"], jb["position"]
            draw.line([(pa[0], pa[1]), (pb_[0], pb_[1])], fill=(255, 255, 255, 180), width=2)
    for idx, name in enumerate(region_names):
        j = joints.get(name, {})
        if not j.get("visible"):
            continue
        x, y = j["position"]
        c = JOINT_COLORS[idx] + (255,)
        draw.ellipse([x-6, y-6, x+6, y+6], fill=c, outline=(255, 255, 255, 255))
    overlay_path = output_dir / f"kick_direct_f{frame}_overlay.png"
    img.save(str(overlay_path))
    print(f"  Saved: {overlay_path}")

print("\nDone!")
