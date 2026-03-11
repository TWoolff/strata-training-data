"""Find the best (most dynamic) frame from each BVH and render a grid."""
import bpy, sys, math
from pathlib import Path
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pose_applicator import (
    _import_animation_bvh, _build_name_map, _apply_tpose, _cleanup_imported_armature,
    PoseInfo, apply_pose, reset_pose,
)
from pipeline.bone_mapper import map_bones
from pipeline.joint_extractor import extract_joints

glb = "/Volumes/TAMWoolff/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final/0/rigged.glb"
pose_dir = Path("/Volumes/TAMWoolff/data/poses_bvh")
output_dir = Path("output/posed_humanrig_test6")
output_dir.mkdir(parents=True, exist_ok=True)

# Find best frames from each BVH
bvh_files = sorted(f for f in pose_dir.glob("*.bvh") if not f.name.startswith("._"))[:8]

print("=== FINDING PEAK ACTION FRAMES ===\n")
best_frames = {}

for bvh_path in bvh_files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    anim_arm = _import_animation_bvh(bvh_path)
    if not anim_arm:
        continue

    action = bpy.data.actions[0]
    if anim_arm.animation_data is None:
        anim_arm.animation_data_create()
    anim_arm.animation_data.action = action

    scene = bpy.context.scene
    start = int(action.frame_range[0])
    end = int(action.frame_range[1])

    # Find the frame with maximum total rotation across all bones
    max_total = 0
    max_frame = start

    for frame in range(start, end + 1, 10):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        total_angle = 0
        for pb in anim_arm.pose.bones:
            rest = pb.bone.matrix_local.to_3x3()
            posed = pb.matrix.to_3x3()
            delta = rest.inverted() @ posed
            q = delta.to_quaternion()
            total_angle += abs(q.angle)

        if total_angle > max_total:
            max_total = total_angle
            max_frame = frame

    # Refine around the best frame
    for frame in range(max(start, max_frame - 10), min(end + 1, max_frame + 11)):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        total_angle = 0
        for pb in anim_arm.pose.bones:
            rest = pb.bone.matrix_local.to_3x3()
            posed = pb.matrix.to_3x3()
            delta = rest.inverted() @ posed
            q = delta.to_quaternion()
            total_angle += abs(q.angle)

        if total_angle > max_total:
            max_total = total_angle
            max_frame = frame

    best_frames[bvh_path.name] = max_frame
    print(f"  {bvh_path.stem}: best frame = {max_frame} (total rotation = {math.degrees(max_total):.0f}°)")

    _cleanup_imported_armature(anim_arm)

# Now render each best frame
print(f"\n=== RENDERING {len(best_frames)} POSES ===\n")

from PIL import Image, ImageDraw, ImageFont

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

for bvh_name, frame in best_frames.items():
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

    clip_name = Path(bvh_name).stem.lower()
    pose = PoseInfo(name=f"{clip_name}_f{frame}", source=bvh_name, frame=frame)
    ok = apply_pose(char_arm, pose, pose_dir)
    print(f"  {clip_name} frame {frame}: apply_pose={ok}")

    if not ok:
        continue

    bpy.context.view_layer.update()
    joints_data = extract_joints(scene, cam_obj, char_arm, meshes, bone_to_region)

    render_path = output_dir / f"{clip_name}_f{frame}_render.png"
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

    overlay_path = output_dir / f"{clip_name}_f{frame}_overlay.png"
    img.save(str(overlay_path))
    overlays.append(str(overlay_path))
    labels.append(f"{clip_name} f{frame}")

# Create a grid of all poses
if overlays:
    cols = min(4, len(overlays))
    rows = math.ceil(len(overlays) / cols)
    cell_w, cell_h = 512, 552  # Extra space for label
    grid = Image.new("RGBA", (cols * cell_w, rows * cell_h), (40, 40, 40, 255))
    draw_grid = ImageDraw.Draw(grid)

    for i, (overlay_path, label) in enumerate(zip(overlays, labels)):
        r, c = divmod(i, cols)
        img = Image.open(overlay_path)
        grid.paste(img, (c * cell_w, r * cell_h))
        draw_grid.text((c * cell_w + 10, r * cell_h + 512 + 5), label, fill=(255, 255, 255, 255))

    grid_path = output_dir / "pose_grid.png"
    grid.save(str(grid_path))
    print(f"\nSaved grid: {grid_path}")

print("\nDone!")
