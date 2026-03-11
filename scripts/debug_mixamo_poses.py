"""Test Mixamo FBX poses on HumanRig character — these should be much more dynamic."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import (
    PoseInfo, apply_pose, reset_pose, list_poses,
)
from pipeline.bone_mapper import map_bones
from pipeline.joint_extractor import extract_joints

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses")
output_dir = Path("output/posed_humanrig_test7")
output_dir.mkdir(parents=True, exist_ok=True)

# Discover all poses (with keyframe sampling)
all_poses = list_poses(pose_dir, keyframes_per_clip=3, include_builtins=False)
print(f"Found {len(all_poses)} pose keyframes from Mixamo FBX files:")
for p in all_poses:
    print(f"  {p.name} (source={p.source}, frame={p.frame})")

# Pick mid-frame from each clip (most dynamic moment)
from collections import OrderedDict
by_source: dict[str, list] = OrderedDict()
for p in all_poses:
    by_source.setdefault(p.source, []).append(p)

chosen = []
for source, frames in by_source.items():
    mid_idx = len(frames) // 2
    chosen.append(frames[mid_idx])

print(f"\nChosen {len(chosen)} mid-frame poses:")
for p in chosen:
    print(f"  {p.name} (frame={p.frame})")

# Render helpers
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

overlays = []
labels = []

for pose in chosen:
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

    ok = apply_pose(char_arm, pose, pose_dir)
    print(f"\n{pose.name}: apply_pose={ok}")

    if not ok:
        continue

    # Print bone rotations for key bones
    for bn in ["Hips", "LeftUpLeg", "LeftLeg", "LeftArm", "LeftForeArm"]:
        pb = char_arm.pose.bones.get(bn)
        if pb:
            q = pb.rotation_quaternion
            angle = math.degrees(2 * math.acos(max(-1, min(1, q[0]))))
            print(f"  {bn}: {angle:.1f}°")

    bpy.context.view_layer.update()
    joints_data = extract_joints(scene, cam_obj, char_arm, meshes, bone_to_region)

    safe_name = pose.name.replace(" ", "_").replace("(", "").replace(")", "")
    render_path = output_dir / f"{safe_name}_render.png"
    scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)

    # Create overlay
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

    overlay_path = output_dir / f"{safe_name}_overlay.png"
    img.save(str(overlay_path))
    overlays.append(str(overlay_path))
    labels.append(pose.name.replace("_", " ")[:30])

# Create grid
if overlays:
    cols = min(4, len(overlays))
    rows = math.ceil(len(overlays) / cols)
    cell_w, cell_h = 512, 552
    grid = Image.new("RGBA", (cols * cell_w, rows * cell_h), (40, 40, 40, 255))
    draw_grid = ImageDraw.Draw(grid)

    for i, (op, label) in enumerate(zip(overlays, labels)):
        r, c = divmod(i, cols)
        img = Image.open(op)
        grid.paste(img, (c * cell_w, r * cell_h))
        draw_grid.text((c * cell_w + 10, r * cell_h + 517), label, fill=(255, 255, 255, 255))

    grid_path = output_dir / "mixamo_pose_grid.png"
    grid.save(str(grid_path))
    print(f"\nSaved grid: {grid_path}")

print("\nDone!")
