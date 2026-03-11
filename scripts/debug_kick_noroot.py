"""Test with root rotation preserved (not zeroed)."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector, Quaternion, Euler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import (
    _import_animation_bvh, _build_name_map, _apply_tpose, _cleanup_imported_armature
)
from pipeline.bone_mapper import map_bones
from pipeline.joint_extractor import extract_joints

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")
output_dir = Path("output/posed_humanrig_test5")
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

for energy, rx, rz in [(5.0, 45, 30), (5.0, 45, 210)]:
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = energy
    obj = bpy.data.objects.new("Sun", light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(rx), 0, math.radians(rz))

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


def apply_bvh_full_rotation(char_arm, bvh_path, frame, zero_root_y=False):
    """Apply BVH with full root rotation (not zeroed)."""
    anim_arm = _import_animation_bvh(bvh_path)
    if not anim_arm:
        return 0

    action = bpy.data.actions[0]
    if anim_arm.animation_data is None:
        anim_arm.animation_data_create()
    anim_arm.animation_data.action = action

    scene.frame_set(frame)
    bpy.context.view_layer.update()

    anim_names = [b.name for b in anim_arm.pose.bones]
    char_names = [b.name for b in char_arm.pose.bones]
    name_map = _build_name_map(anim_names, char_names)

    _apply_tpose(char_arm)
    bpy.context.view_layer.update()

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

        anim_rest = anim_pb.bone.matrix_local.to_3x3()
        anim_posed = anim_pb.matrix.to_3x3()
        char_rest = char_pb.bone.matrix_local.to_3x3()

        if anim_pb.parent and char_pb.parent:
            anim_pr = anim_pb.parent.bone.matrix_local.to_3x3()
            anim_pp = anim_pb.parent.matrix.to_3x3()
            char_pr = char_pb.parent.bone.matrix_local.to_3x3()

            anim_lr = anim_pr.inverted() @ anim_rest
            anim_lp = anim_pp.inverted() @ anim_posed
            char_lr = char_pr.inverted() @ char_rest

            local_delta = anim_lr.inverted() @ anim_lp
            R = anim_lr.inverted() @ char_lr
            pose_basis = R.inverted() @ local_delta @ R
            q = pose_basis.to_quaternion()
        else:
            # Root bone
            local_delta = anim_rest.inverted() @ anim_posed
            R = anim_rest.inverted() @ char_rest
            pose_basis = R.inverted() @ local_delta @ R
            q = pose_basis.to_quaternion()

            if zero_root_y:
                e = q.to_euler("YXZ")
                e.y = 0.0
                q = e.to_quaternion()

        char_pb.rotation_mode = "QUATERNION"
        char_pb.rotation_quaternion = q
        transferred += 1

    bpy.context.view_layer.update()
    _cleanup_imported_armature(anim_arm, keep_actions=False)
    return transferred


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

# Compare: with vs without root Y zeroing at frame 41
for label, zero_y in [("full_root", False), ("zero_root_y", True)]:
    _apply_tpose(char_arm)
    bpy.context.view_layer.update()

    n = apply_bvh_full_rotation(char_arm, pose_dir / "Kick_ID.bvh", 41, zero_root_y=zero_y)
    print(f"\n{label}: transferred {n} bones")

    for bn in ["Hips", "LeftUpLeg", "LeftLeg"]:
        pb = char_arm.pose.bones.get(bn)
        if pb:
            q = pb.rotation_quaternion
            angle = math.degrees(2 * math.acos(max(-1, min(1, q[0]))))
            print(f"  {bn}: {angle:.1f}°")

    joints_data = extract_joints(scene, cam_obj, char_arm, meshes, bone_to_region)

    render_path = output_dir / f"kick_{label}_render.png"
    scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)

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
    overlay_path = output_dir / f"kick_{label}_overlay.png"
    img.save(str(overlay_path))

print("\nDone!")
