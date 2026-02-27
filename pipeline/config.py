"""Pipeline constants: region definitions, bone mappings, render settings.

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RegionId = int
RGB = tuple[int, int, int]

# ---------------------------------------------------------------------------
# Strata Standard Skeleton — 19 body regions + background (20 total)
# ---------------------------------------------------------------------------

REGION_NAMES: dict[RegionId, str] = {
    0: "background",
    1: "head",
    2: "neck",
    3: "chest",
    4: "spine",
    5: "hips",
    6: "upper_arm_l",
    7: "lower_arm_l",
    8: "hand_l",
    9: "upper_arm_r",
    10: "lower_arm_r",
    11: "hand_r",
    12: "upper_leg_l",
    13: "lower_leg_l",
    14: "foot_l",
    15: "upper_leg_r",
    16: "lower_leg_r",
    17: "foot_r",
    18: "shoulder_l",
    19: "shoulder_r",
}

REGION_COLORS: dict[RegionId, RGB] = {
    0: (0, 0, 0),  # background (transparent in final output)
    1: (255, 0, 0),  # head
    2: (0, 255, 0),  # neck
    3: (0, 0, 255),  # chest
    4: (255, 255, 0),  # spine
    5: (255, 0, 255),  # hips
    6: (128, 0, 0),  # upper_arm_l
    7: (0, 128, 0),  # lower_arm_l
    8: (0, 0, 128),  # hand_l
    9: (128, 128, 0),  # upper_arm_r
    10: (128, 0, 128),  # lower_arm_r
    11: (0, 128, 128),  # hand_r
    12: (64, 0, 0),  # upper_leg_l
    13: (0, 64, 0),  # lower_leg_l
    14: (0, 0, 64),  # foot_l
    15: (64, 64, 0),  # upper_leg_r
    16: (64, 0, 64),  # lower_leg_r
    17: (0, 64, 64),  # foot_r
    18: (192, 64, 0),  # shoulder_l
    19: (0, 64, 192),  # shoulder_r
}

NUM_REGIONS: int = 20  # 0–19 inclusive

# ---------------------------------------------------------------------------
# Mixamo bone name → region ID mapping
# ---------------------------------------------------------------------------
# Covers all standard Mixamo humanoid bones (prefix: "mixamorig:").
# Finger bones map to the corresponding hand region.

MIXAMO_BONE_MAP: dict[str, RegionId] = {
    # Head / neck
    "mixamorig:Head": 1,
    "mixamorig:HeadTop_End": 1,
    "mixamorig:Neck": 2,
    # Torso
    "mixamorig:Spine2": 3,  # chest
    "mixamorig:Spine1": 4,  # spine
    "mixamorig:Spine": 4,  # spine
    "mixamorig:Hips": 5,
    # Left shoulder / arm
    "mixamorig:LeftShoulder": 18,
    "mixamorig:LeftArm": 6,
    "mixamorig:LeftForeArm": 7,
    "mixamorig:LeftHand": 8,
    # Left fingers → hand_l
    "mixamorig:LeftHandThumb1": 8,
    "mixamorig:LeftHandThumb2": 8,
    "mixamorig:LeftHandThumb3": 8,
    "mixamorig:LeftHandThumb4": 8,
    "mixamorig:LeftHandIndex1": 8,
    "mixamorig:LeftHandIndex2": 8,
    "mixamorig:LeftHandIndex3": 8,
    "mixamorig:LeftHandIndex4": 8,
    "mixamorig:LeftHandMiddle1": 8,
    "mixamorig:LeftHandMiddle2": 8,
    "mixamorig:LeftHandMiddle3": 8,
    "mixamorig:LeftHandMiddle4": 8,
    "mixamorig:LeftHandRing1": 8,
    "mixamorig:LeftHandRing2": 8,
    "mixamorig:LeftHandRing3": 8,
    "mixamorig:LeftHandRing4": 8,
    "mixamorig:LeftHandPinky1": 8,
    "mixamorig:LeftHandPinky2": 8,
    "mixamorig:LeftHandPinky3": 8,
    "mixamorig:LeftHandPinky4": 8,
    # Right shoulder / arm
    "mixamorig:RightShoulder": 19,
    "mixamorig:RightArm": 9,
    "mixamorig:RightForeArm": 10,
    "mixamorig:RightHand": 11,
    # Right fingers → hand_r
    "mixamorig:RightHandThumb1": 11,
    "mixamorig:RightHandThumb2": 11,
    "mixamorig:RightHandThumb3": 11,
    "mixamorig:RightHandThumb4": 11,
    "mixamorig:RightHandIndex1": 11,
    "mixamorig:RightHandIndex2": 11,
    "mixamorig:RightHandIndex3": 11,
    "mixamorig:RightHandIndex4": 11,
    "mixamorig:RightHandMiddle1": 11,
    "mixamorig:RightHandMiddle2": 11,
    "mixamorig:RightHandMiddle3": 11,
    "mixamorig:RightHandMiddle4": 11,
    "mixamorig:RightHandRing1": 11,
    "mixamorig:RightHandRing2": 11,
    "mixamorig:RightHandRing3": 11,
    "mixamorig:RightHandRing4": 11,
    "mixamorig:RightHandPinky1": 11,
    "mixamorig:RightHandPinky2": 11,
    "mixamorig:RightHandPinky3": 11,
    "mixamorig:RightHandPinky4": 11,
    # Left leg
    "mixamorig:LeftUpLeg": 12,
    "mixamorig:LeftLeg": 13,
    "mixamorig:LeftFoot": 14,
    "mixamorig:LeftToeBase": 14,
    "mixamorig:LeftToe_End": 14,
    # Right leg
    "mixamorig:RightUpLeg": 15,
    "mixamorig:RightLeg": 16,
    "mixamorig:RightFoot": 17,
    "mixamorig:RightToeBase": 17,
    "mixamorig:RightToe_End": 17,
}

# ---------------------------------------------------------------------------
# Common bone name aliases for non-Mixamo rigs
# ---------------------------------------------------------------------------
# Covers Blender-style (bone.L / bone.R), generic uppercase, and generic
# lowercase variants. Used as fallback when exact Mixamo names don't match.

COMMON_BONE_ALIASES: dict[str, RegionId] = {
    # --- Head / neck ---
    "head": 1,
    "Head": 1,
    "HEAD": 1,
    "skull": 1,
    "Skull": 1,
    "neck": 2,
    "Neck": 2,
    "NECK": 2,
    # --- Torso ---
    "chest": 3,
    "Chest": 3,
    "upper_body": 3,
    "torso_upper": 3,
    "Spine2": 3,
    "spine": 4,
    "Spine": 4,
    "Spine1": 4,
    "torso": 4,
    "torso_lower": 4,
    "abdomen": 4,
    "hips": 5,
    "Hips": 5,
    "pelvis": 5,
    "Pelvis": 5,
    "root": 5,
    "Root": 5,
    # --- Left shoulder ---
    "LeftShoulder": 18,
    "shoulder.L": 18,
    "Shoulder.L": 18,
    "clavicle.L": 18,
    "Clavicle.L": 18,
    "L_clavicle": 18,
    "l_clavicle": 18,
    "L_shoulder": 18,
    "l_shoulder": 18,
    # --- Left arm ---
    "upper_arm.L": 6,
    "upperarm.L": 6,
    "UpperArm.L": 6,
    "L_upperarm": 6,
    "l_upperarm": 6,
    "LeftArm": 6,
    "forearm.L": 7,
    "Forearm.L": 7,
    "lower_arm.L": 7,
    "L_forearm": 7,
    "l_forearm": 7,
    "LeftForeArm": 7,
    "hand.L": 8,
    "Hand.L": 8,
    "L_hand": 8,
    "l_hand": 8,
    "LeftHand": 8,
    # --- Right shoulder ---
    "RightShoulder": 19,
    "shoulder.R": 19,
    "Shoulder.R": 19,
    "clavicle.R": 19,
    "Clavicle.R": 19,
    "R_clavicle": 19,
    "r_clavicle": 19,
    "R_shoulder": 19,
    "r_shoulder": 19,
    # --- Right arm ---
    "upper_arm.R": 9,
    "upperarm.R": 9,
    "UpperArm.R": 9,
    "R_upperarm": 9,
    "r_upperarm": 9,
    "RightArm": 9,
    "forearm.R": 10,
    "Forearm.R": 10,
    "lower_arm.R": 10,
    "R_forearm": 10,
    "r_forearm": 10,
    "RightForeArm": 10,
    "hand.R": 11,
    "Hand.R": 11,
    "R_hand": 11,
    "r_hand": 11,
    "RightHand": 11,
    # --- Left leg ---
    "thigh.L": 12,
    "Thigh.L": 12,
    "upper_leg.L": 12,
    "L_thigh": 12,
    "l_thigh": 12,
    "LeftUpLeg": 12,
    "shin.L": 13,
    "Shin.L": 13,
    "lower_leg.L": 13,
    "calf.L": 13,
    "L_calf": 13,
    "l_calf": 13,
    "LeftLeg": 13,
    "foot.L": 14,
    "Foot.L": 14,
    "L_foot": 14,
    "l_foot": 14,
    "LeftFoot": 14,
    "toe.L": 14,
    "LeftToeBase": 14,
    # --- Right leg ---
    "thigh.R": 15,
    "Thigh.R": 15,
    "upper_leg.R": 15,
    "R_thigh": 15,
    "r_thigh": 15,
    "RightUpLeg": 15,
    "shin.R": 16,
    "Shin.R": 16,
    "lower_leg.R": 16,
    "calf.R": 16,
    "R_calf": 16,
    "r_calf": 16,
    "RightLeg": 16,
    "foot.R": 17,
    "Foot.R": 17,
    "R_foot": 17,
    "r_foot": 17,
    "RightFoot": 17,
    "toe.R": 17,
    "RightToeBase": 17,
}

# ---------------------------------------------------------------------------
# Bone mapping: prefix stripping
# ---------------------------------------------------------------------------
# Common bone name prefixes to strip when attempting prefix-based matching.

COMMON_PREFIXES: list[str] = [
    "mixamorig:",
    "Bip01_",
    "Bip001_",
    "Bip01 ",
    "Bip001 ",
    "DEF-",
    "def_",
    "ORG-",
    "MCH-",
    "CC_Base_",
    "Bone_",
    "bone_",
    "RIG_",
    "rig_",
]

# ---------------------------------------------------------------------------
# Bone mapping: substring keywords
# ---------------------------------------------------------------------------
# Case-insensitive keyword tuples → region ID. Checked in order; first match
# wins. More specific patterns (e.g., "forearm") must come before general
# ones (e.g., "arm") to avoid false matches.

# ---------------------------------------------------------------------------
# Primary bone per region (for joint extraction)
# ---------------------------------------------------------------------------
# When multiple bones map to the same region, the joint extractor picks the
# bone whose name contains one of these substrings (checked in order, case-
# insensitive). The first matching bone wins.  If none match, the first bone
# encountered for that region is used.

PRIMARY_BONE_KEYWORDS: dict[RegionId, list[str]] = {
    1: ["Head"],  # head (not HeadTop_End)
    2: ["Neck"],
    3: ["Spine2"],  # chest — Spine2 is upper torso
    4: ["Spine1"],  # spine — Spine1 is mid-torso
    5: ["Hips"],
    6: ["LeftArm", "upper_arm.L", "L_upperarm"],
    7: ["LeftForeArm", "forearm.L", "L_forearm"],
    8: ["LeftHand", "hand.L", "L_hand"],
    9: ["RightArm", "upper_arm.R", "R_upperarm"],
    10: ["RightForeArm", "forearm.R", "R_forearm"],
    11: ["RightHand", "hand.R", "R_hand"],
    12: ["LeftUpLeg", "thigh.L", "L_thigh"],
    13: ["LeftLeg", "shin.L", "L_calf"],
    14: ["LeftFoot", "foot.L", "L_foot"],
    15: ["RightUpLeg", "thigh.R", "R_thigh"],
    16: ["RightLeg", "shin.R", "R_calf"],
    17: ["RightFoot", "foot.R", "R_foot"],
    18: ["LeftShoulder", "shoulder.L", "clavicle.L"],
    19: ["RightShoulder", "shoulder.R", "clavicle.R"],
}

# Number of joint regions (body regions only, excluding background)
NUM_JOINT_REGIONS: int = 19  # regions 1–19

# Bounding box padding for joint extraction (fraction of bbox dimension)
JOINT_BBOX_PADDING: float = 0.05

SUBSTRING_KEYWORDS: list[tuple[list[str], RegionId]] = [
    # Head / neck
    (["head"], 1),
    (["skull"], 1),
    (["neck"], 2),
    # Torso
    (["chest"], 3),
    (["spine2"], 3),
    (["spine1"], 4),
    (["spine"], 4),
    (["hip"], 5),
    (["pelvis"], 5),
    # Shoulders (before arm to avoid "shoulder" matching "arm")
    (["shoulder", "left"], 18),
    (["shoulder", "l"], 18),
    (["clavicle", "left"], 18),
    (["clavicle", "l"], 18),
    (["shoulder", "right"], 19),
    (["shoulder", "r"], 19),
    (["clavicle", "right"], 19),
    (["clavicle", "r"], 19),
    # Left arm (forearm before arm to avoid false match)
    (["forearm", "left"], 7),
    (["forearm", "l"], 7),
    (["lower", "arm", "left"], 7),
    (["lower", "arm", "l"], 7),
    (["arm", "left", "up"], 6),
    (["arm", "left"], 6),
    (["arm", "l", "up"], 6),
    (["upper", "arm", "l"], 6),
    # Left hand / fingers
    (["hand", "left"], 8),
    (["hand", "l"], 8),
    (["finger", "left"], 8),
    (["finger", "l"], 8),
    (["thumb", "left"], 8),
    (["thumb", "l"], 8),
    # Right arm
    (["forearm", "right"], 10),
    (["forearm", "r"], 10),
    (["lower", "arm", "right"], 10),
    (["lower", "arm", "r"], 10),
    (["arm", "right", "up"], 9),
    (["arm", "right"], 9),
    (["arm", "r", "up"], 9),
    (["upper", "arm", "r"], 9),
    # Right hand / fingers
    (["hand", "right"], 11),
    (["hand", "r"], 11),
    (["finger", "right"], 11),
    (["finger", "r"], 11),
    (["thumb", "right"], 11),
    (["thumb", "r"], 11),
    # Left leg (shin/calf before generic leg)
    (["shin", "left"], 13),
    (["shin", "l"], 13),
    (["calf", "left"], 13),
    (["calf", "l"], 13),
    (["lower", "leg", "left"], 13),
    (["lower", "leg", "l"], 13),
    (["thigh", "left"], 12),
    (["thigh", "l"], 12),
    (["upper", "leg", "left"], 12),
    (["upper", "leg", "l"], 12),
    (["leg", "left", "up"], 12),
    (["leg", "l", "up"], 12),
    # Left foot / toes
    (["foot", "left"], 14),
    (["foot", "l"], 14),
    (["toe", "left"], 14),
    (["toe", "l"], 14),
    # Right leg
    (["shin", "right"], 16),
    (["shin", "r"], 16),
    (["calf", "right"], 16),
    (["calf", "r"], 16),
    (["lower", "leg", "right"], 16),
    (["lower", "leg", "r"], 16),
    (["thigh", "right"], 15),
    (["thigh", "r"], 15),
    (["upper", "leg", "right"], 15),
    (["upper", "leg", "r"], 15),
    (["leg", "right", "up"], 15),
    (["leg", "r", "up"], 15),
    # Right foot / toes
    (["foot", "right"], 17),
    (["foot", "r"], 17),
    (["toe", "right"], 17),
    (["toe", "r"], 17),
]

# ---------------------------------------------------------------------------
# Bone mapping: fuzzy keyword matching
# ---------------------------------------------------------------------------
# Used after substring matching fails. Bone names are normalized (prefix-stripped,
# camelCase-split, tokenized) before matching. Score = matched keywords / total
# keywords in pattern; minimum FUZZY_MIN_SCORE required.
#
# Each entry: (keyword_tuple, region_id). Keywords are matched against normalized
# tokens. Laterality keywords ("left"/"right"/"l"/"r") are handled separately
# via token-boundary detection to avoid "leg" matching "l".

FUZZY_MIN_SCORE: float = 0.6

FUZZY_KEYWORD_PATTERNS: list[tuple[tuple[str, ...], RegionId]] = [
    # Head / neck
    (("head",), 1),
    (("skull",), 1),
    (("cranium",), 1),
    (("neck",), 2),
    # Torso
    (("chest",), 3),
    (("upper", "body"), 3),
    (("upper", "torso"), 3),
    (("spine",), 4),
    (("torso",), 4),
    (("abdomen",), 4),
    (("hip",), 5),
    (("hips",), 5),
    (("pelvis",), 5),
    (("root",), 5),
    # Left shoulder
    (("shoulder", "left"), 18),
    (("clavicle", "left"), 18),
    # Right shoulder
    (("shoulder", "right"), 19),
    (("clavicle", "right"), 19),
    # Left arm — forearm/lower before upper to avoid false match
    (("forearm", "left"), 7),
    (("lower", "arm", "left"), 7),
    (("upper", "arm", "left"), 6),
    (("arm", "upper", "left"), 6),
    (("bicep", "left"), 6),
    # Left hand
    (("hand", "left"), 8),
    (("finger", "left"), 8),
    (("thumb", "left"), 8),
    (("wrist", "left"), 8),
    # Right arm
    (("forearm", "right"), 10),
    (("lower", "arm", "right"), 10),
    (("upper", "arm", "right"), 9),
    (("arm", "upper", "right"), 9),
    (("bicep", "right"), 9),
    # Right hand
    (("hand", "right"), 11),
    (("finger", "right"), 11),
    (("thumb", "right"), 11),
    (("wrist", "right"), 11),
    # Left leg — lower before upper
    (("shin", "left"), 13),
    (("calf", "left"), 13),
    (("lower", "leg", "left"), 13),
    (("thigh", "left"), 12),
    (("upper", "leg", "left"), 12),
    # Left foot
    (("foot", "left"), 14),
    (("toe", "left"), 14),
    (("ankle", "left"), 14),
    # Right leg
    (("shin", "right"), 16),
    (("calf", "right"), 16),
    (("lower", "leg", "right"), 16),
    (("thigh", "right"), 15),
    (("upper", "leg", "right"), 15),
    # Right foot
    (("foot", "right"), 17),
    (("toe", "right"), 17),
    (("ankle", "right"), 17),
]

# Laterality aliases: map short/long forms to canonical "left" / "right".
# Only matched at token boundaries (whole tokens) to avoid "leg" → "l".
LATERALITY_ALIASES: dict[str, str] = {
    "l": "left",
    "r": "right",
}

# ---------------------------------------------------------------------------
# Character normalization
# ---------------------------------------------------------------------------

TARGET_CHARACTER_HEIGHT: float = 2.0  # Blender units (bounding box height)

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

RENDER_RESOLUTION: int = 512
CAMERA_TYPE: str = "ORTHO"
CAMERA_PADDING: float = 0.1
CAMERA_DISTANCE: float = 10.0  # distance along -Y axis (avoids near-plane clipping)
CAMERA_CLIP_START: float = 0.1
CAMERA_CLIP_END: float = 100.0
BACKGROUND_TRANSPARENT: bool = True

# Lighting (color render only — segmentation uses Emission materials that ignore lighting)
SUN_POSITION: tuple[float, float, float] = (0.0, -5.0, 10.0)
SUN_ENERGY: float = 1.0
AMBIENT_COLOR: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)

# ---------------------------------------------------------------------------
# Pose library
# ---------------------------------------------------------------------------

KEYFRAMES_PER_CLIP: int = 4  # number of evenly-spaced keyframes to sample per animation
A_POSE_SHOULDER_ANGLE: float = 45.0  # degrees downward rotation for A-pose upper arms

# ---------------------------------------------------------------------------
# Pose augmentation
# ---------------------------------------------------------------------------

ENABLE_FLIP: bool = False  # Y-axis (horizontal) flip augmentation
ENABLE_SCALE: bool = False  # uniform scale variation augmentation
SCALE_FACTORS: list[float] = [0.85, 1.0, 1.15]  # scale factors to apply

# Left↔right region ID swap pairs for flip augmentation
FLIP_REGION_SWAP: dict[RegionId, RegionId] = {
    6: 9,  # upper_arm_l ↔ upper_arm_r
    7: 10,  # lower_arm_l ↔ lower_arm_r
    8: 11,  # hand_l ↔ hand_r
    9: 6,
    10: 7,
    11: 8,
    12: 15,  # upper_leg_l ↔ upper_leg_r
    13: 16,  # lower_leg_l ↔ lower_leg_r
    14: 17,  # foot_l ↔ foot_r
    15: 12,
    16: 13,
    17: 14,
    18: 19,  # shoulder_l ↔ shoulder_r
    19: 18,
}

# Left↔right joint name swap pairs for flip augmentation
FLIP_JOINT_SWAP: dict[str, str] = {
    "upper_arm_l": "upper_arm_r",
    "lower_arm_l": "lower_arm_r",
    "hand_l": "hand_r",
    "upper_arm_r": "upper_arm_l",
    "lower_arm_r": "lower_arm_l",
    "hand_r": "hand_l",
    "upper_leg_l": "upper_leg_r",
    "lower_leg_l": "lower_leg_r",
    "foot_l": "foot_r",
    "upper_leg_r": "upper_leg_l",
    "lower_leg_r": "lower_leg_l",
    "foot_r": "foot_l",
    "shoulder_l": "shoulder_r",
    "shoulder_r": "shoulder_l",
}

# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

WEIGHT_THRESHOLD: float = 0.01  # Minimum bone weight to include (noise reduction)

# ---------------------------------------------------------------------------
# Art styles
# ---------------------------------------------------------------------------
# Render-time (Blender shaders): flat, cel, unlit
# Post-render (Python/PIL/OpenCV): pixel, painterly, sketch

ART_STYLES: list[str] = ["flat", "cel", "pixel", "painterly", "sketch", "unlit"]
RENDER_TIME_STYLES: set[str] = {"flat", "cel", "unlit"}
POST_RENDER_STYLES: set[str] = {"pixel", "painterly", "sketch"}

# Style routing registry: maps style name → type ("render" or "post")
# Render-time styles modify Blender materials before bpy.ops.render.render()
# Post-render styles take a PIL Image and return a PIL Image
STYLE_REGISTRY: dict[str, str] = {
    "flat": "render",
    "cel": "render",
    "unlit": "render",
    "pixel": "post",
    "painterly": "post",
    "sketch": "post",
}

# The base style used as input for all post-render transforms
POST_RENDER_BASE_STYLE: str = "flat"

# Pixel art style parameters
PIXEL_ART_DOWNSCALE_SIZE: int = 64  # downscale target (64 or 128)
PIXEL_ART_PALETTE_SIZE: int = 16  # number of colors after quantization (16–32)

# Painterly style parameters
PAINTERLY_BILATERAL_D: int = 9  # bilateral filter diameter
PAINTERLY_SIGMA_COLOR: int = 75  # bilateral filter sigma in color space
PAINTERLY_SIGMA_SPACE: int = 75  # bilateral filter sigma in coordinate space
PAINTERLY_PASSES: dict[str, int] = {  # bilateral filter pass count by strength
    "light": 1,
    "medium": 2,
    "heavy": 3,
}
PAINTERLY_DEFAULT_STRENGTH: str = "medium"
PAINTERLY_HUE_JITTER: int = 5  # max hue shift in degrees (OpenCV hue is 0–180)
PAINTERLY_SAT_JITTER: int = 25  # max saturation shift (0–255 scale)
PAINTERLY_VAL_JITTER: int = 25  # max value/brightness shift (0–255 scale)
PAINTERLY_NOISE_SIGMA: float = 0.02  # Gaussian noise σ (fraction of 255)

# Sketch/lineart style parameters
SKETCH_BLUR_KSIZE: int = 5  # Gaussian blur kernel size (must be odd)
SKETCH_CANNY_THRESHOLD1: int = 50  # Canny lower threshold
SKETCH_CANNY_THRESHOLD2: int = 150  # Canny upper threshold
SKETCH_LINE_THICKNESS: int = 3  # Dilation kernel size (controls line width ~2-4px)
SKETCH_BG_COLOR: RGB = (252, 248, 240)  # Cream background
SKETCH_WOBBLE_RANGE: int = 1  # Max pixel displacement for hand-drawn wobble
SKETCH_ENABLE_WOBBLE: bool = True  # Whether to apply wobble effect

# Cel/toon shading parameters
CEL_RAMP_STOPS: list[tuple[float, float]] = [
    (0.0, 0.3),  # shadow tone (position, brightness factor)
    (0.4, 0.7),  # mid tone
    (0.7, 1.0),  # highlight tone
]
CEL_OUTLINE_THICKNESS: float = 2.0  # Freestyle line thickness in pixels

# Fallback base color when material has no Principled BSDF or texture
DEFAULT_BASE_COLOR: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

SPLIT_RATIOS: dict[str, float] = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}
