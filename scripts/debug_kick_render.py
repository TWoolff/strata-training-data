"""Render Kick BVH at peak action frame (61) to verify mesh deformation."""
import bpy, sys, math, json
from pathlib import Path
from mathutils import Vector, Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import PoseInfo, apply_pose, reset_pose
from pipeline.bone_mapper import map_bones
from pipeline.joint_extractor import extract_joints

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")
output_dir = Path("output/posed_humanrig_test3")
output_dir.mkdir(parents=True, exist_ok=True)

# Setup
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

# World
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Strength"].default_value = 0.0

# Lights
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

# Camera
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 1.3
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = Vector((0, -2.0, 0))
cam_obj.rotation_euler = (math.radians(90), 0, 0)

# Map bones
mapping = map_bones(armature, meshes, "humanrig_0")
bone_to_region = mapping.bone_to_region

# Test specific frames: 1 (rest-ish), 41 (action), 61 (peak), 81 (action)
test_frames = [1, 41, 61, 81]
for frame in test_frames:
    pose = PoseInfo(name=f"kick_frame_{frame}", source="Kick_ID.bvh", frame=frame)
    ok = apply_pose(armature, pose, pose_dir)
    print(f"\nFrame {frame}: apply_pose={ok}")

    if ok:
        bpy.context.view_layer.update()

        # Check bone rotations
        for bn in ["LeftUpLeg", "LeftLeg", "Hips"]:
            pb = armature.pose.bones.get(bn)
            if pb:
                q = pb.rotation_quaternion
                angle = math.degrees(2 * math.acos(max(-1, min(1, q[0]))))
                print(f"  {bn}: angle={angle:.1f}°, q={[round(v, 3) for v in q]}")

        # Extract joints
        joints_data = extract_joints(scene, cam_obj, armature, meshes, bone_to_region)

        # Render
        render_path = output_dir / f"kick_frame_{frame}_render.png"
        scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)

        # Save overlay
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
        overlay_path = output_dir / f"kick_frame_{frame}_overlay.png"
        img.save(str(overlay_path))
        print(f"  Saved: {overlay_path}")

    reset_pose(armature)

print("\nDone!")
