"""Pipeline constants: region definitions, bone mappings, render settings.

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pipeline version
# ---------------------------------------------------------------------------

PIPELINE_VERSION: str = "0.1.0"

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
# VRM humanoid bone name → region ID mapping
# ---------------------------------------------------------------------------
# VRM/VRoid models use standardized camelCase humanoid bone names.
# These map directly to Strata regions with near-100% coverage.
# Finger bones map to the corresponding hand region.
# Reference: https://vrm.dev/en/univrm/humanoid/humanoid_overview

VRM_BONE_ALIASES: dict[str, RegionId] = {
    # Head / neck
    "head": 1,
    "neck": 2,
    # Torso
    "upperChest": 3,
    "chest": 3,
    "spine": 4,
    "hips": 5,
    # Left shoulder / arm
    "leftShoulder": 18,
    "leftUpperArm": 6,
    "leftLowerArm": 7,
    "leftHand": 8,
    # Left fingers → hand_l
    "leftThumbMetacarpal": 8,
    "leftThumbProximal": 8,
    "leftThumbDistal": 8,
    "leftIndexProximal": 8,
    "leftIndexIntermediate": 8,
    "leftIndexDistal": 8,
    "leftMiddleProximal": 8,
    "leftMiddleIntermediate": 8,
    "leftMiddleDistal": 8,
    "leftRingProximal": 8,
    "leftRingIntermediate": 8,
    "leftRingDistal": 8,
    "leftLittleProximal": 8,
    "leftLittleIntermediate": 8,
    "leftLittleDistal": 8,
    # Right shoulder / arm
    "rightShoulder": 19,
    "rightUpperArm": 9,
    "rightLowerArm": 10,
    "rightHand": 11,
    # Right fingers → hand_r
    "rightThumbMetacarpal": 11,
    "rightThumbProximal": 11,
    "rightThumbDistal": 11,
    "rightIndexProximal": 11,
    "rightIndexIntermediate": 11,
    "rightIndexDistal": 11,
    "rightMiddleProximal": 11,
    "rightMiddleIntermediate": 11,
    "rightMiddleDistal": 11,
    "rightRingProximal": 11,
    "rightRingIntermediate": 11,
    "rightRingDistal": 11,
    "rightLittleProximal": 11,
    "rightLittleIntermediate": 11,
    "rightLittleDistal": 11,
    # Left leg
    "leftUpperLeg": 12,
    "leftLowerLeg": 13,
    "leftFoot": 14,
    "leftToes": 14,
    # Right leg
    "rightUpperLeg": 15,
    "rightLowerLeg": 16,
    "rightFoot": 17,
    "rightToes": 17,
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
# Accessory detection
# ---------------------------------------------------------------------------
# Keywords matched case-insensitively against mesh object names.
# A mesh matching any pattern is a candidate for hiding (combined with
# skinning heuristics in accessory_detector.py).

ACCESSORY_NAME_PATTERNS: list[str] = [
    "weapon",
    "sword",
    "shield",
    "bow",
    "staff",
    "spear",
    "axe",
    "mace",
    "dagger",
    "cape",
    "cloak",
    "wings",
    "wing",
    "tail",
    "armor",
    "pauldron",
    "gauntlet",
    "hat",
    "helmet",
    "crown",
    "headpiece",
    "headband",
    "ribbon",
    "ornament",
    "jewelry",
    "necklace",
    "earring",
    "accessory",
    "belt",
    "scarf",
    "glasses",
    "mask",
    "quiver",
    "backpack",
    "bag",
    "pouch",
]

# Maximum number of vertex groups for a mesh to be considered an accessory
# via weak-skinning heuristic. Meshes with fewer vertex groups than this
# are likely accessories parented to a single bone.
ACCESSORY_MAX_VERTEX_GROUPS: int = 5

# Minimum fraction of vertices that must be weighted to a single bone
# for the mesh to be considered weakly-skinned (likely an accessory).
ACCESSORY_WEAK_SKIN_THRESHOLD: float = 0.8

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
CAMERA_DISTANCE: float = 10.0  # distance from character center (avoids near-plane clipping)
CAMERA_CLIP_START: float = 0.1
CAMERA_CLIP_END: float = 100.0
BACKGROUND_TRANSPARENT: bool = True

# ---------------------------------------------------------------------------
# Multi-angle camera positions
# ---------------------------------------------------------------------------
# Azimuth = rotation around the vertical (Z) axis in degrees.
# 0° = front, 90° = side (left), 180° = back.
# Elevation is always 0 (eye-level).

CameraAngle = dict[str, int]

CAMERA_ANGLES: dict[str, CameraAngle] = {
    "front": {"azimuth": 0, "elevation": 0},
    "three_quarter": {"azimuth": 45, "elevation": 0},
    "side": {"azimuth": 90, "elevation": 0},
    "three_quarter_back": {"azimuth": 135, "elevation": 0},
    "back": {"azimuth": 180, "elevation": 0},
}

DEFAULT_CAMERA_ANGLES: list[str] = ["front"]
ALL_CAMERA_ANGLES: list[str] = list(CAMERA_ANGLES.keys())

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

# ---------------------------------------------------------------------------
# Reverse lookup: region name → region ID
# ---------------------------------------------------------------------------

REGION_NAME_TO_ID: dict[str, RegionId] = {name: rid for rid, name in REGION_NAMES.items()}

# ---------------------------------------------------------------------------
# Live2D fragment-to-Strata label mapping
# ---------------------------------------------------------------------------
# Ordered list of (regex_pattern, strata_region_name) tuples.
# Patterns are matched case-insensitively against Live2D ArtMesh fragment names.
# First match wins — place specific patterns before general ones.
# Covers English, Japanese romaji, and common Live2D naming conventions.

LIVE2D_FRAGMENT_PATTERNS: list[tuple[str, str]] = [
    # --- Head (facial features, hair) ---
    (r"eye|me_[lr]|hitomi", "head"),
    (r"brow|mayu", "head"),
    (r"mouth|kuchi|lip", "head"),
    (r"nose|hana", "head"),
    (r"\bear\b|mimi", "head"),
    (r"hair|bangs|maegami|ushirogami|kami", "head"),
    (r"face|kao", "head"),
    (r"head|atama", "head"),
    # --- Neck ---
    (r"neck|kubi", "neck"),
    # --- Shoulders (before arms to avoid false matches) ---
    (r"shoulder.*[lL]|kata.*[lL]|shoulder.*left|kata.*hidari", "shoulder_l"),
    (r"shoulder.*[rR]|kata.*[rR]|shoulder.*right|kata.*migi", "shoulder_r"),
    # --- Left arm (forearm/lower before upper to avoid substring clash) ---
    (
        r"arm.*(?:lower|fore).*[lL]|arm.*(?:lower|fore).*left|forearm.*[lL]|forearm.*left",
        "lower_arm_l",
    ),
    (
        r"arm.*upper.*[lL]|arm.*upper.*left|ude.*ue.*[lL]|ude.*ue.*left|upper.*arm.*[lL]",
        "upper_arm_l",
    ),
    (r"hand.*[lL]|hand.*left|te_[lL]|te.*hidari", "hand_l"),
    # --- Right arm ---
    (
        r"arm.*(?:lower|fore).*[rR]|arm.*(?:lower|fore).*right|forearm.*[rR]|forearm.*right",
        "lower_arm_r",
    ),
    (
        r"arm.*upper.*[rR]|arm.*upper.*right|ude.*ue.*[rR]|ude.*ue.*right|upper.*arm.*[rR]",
        "upper_arm_r",
    ),
    (r"hand.*[rR]|hand.*right|te_[rR]|te.*migi", "hand_r"),
    # --- Left leg (lower/shin before upper to avoid substring clash) ---
    (
        r"leg.*(?:lower|shin).*[lL]|leg.*(?:lower|shin).*left|shin.*[lL]|shin.*left|sune.*[lL]",
        "lower_leg_l",
    ),
    (
        r"leg.*upper.*[lL]|leg.*upper.*left|thigh.*[lL]|thigh.*left|momo.*[lL]|momo.*left|upper.*leg.*[lL]",
        "upper_leg_l",
    ),
    (r"foot.*[lL]|foot.*left|ashi.*[lL]|ashi.*hidari", "foot_l"),
    # --- Right leg ---
    (
        r"leg.*(?:lower|shin).*[rR]|leg.*(?:lower|shin).*right|shin.*[rR]|shin.*right|sune.*[rR]",
        "lower_leg_r",
    ),
    (
        r"leg.*upper.*[rR]|leg.*upper.*right|thigh.*[rR]|thigh.*right|momo.*[rR]|momo.*right|upper.*leg.*[rR]",
        "upper_leg_r",
    ),
    (r"foot.*[rR]|foot.*right|ashi.*[rR]|ashi.*migi", "foot_r"),
    # --- Hips ---
    (r"hip|pelvis|koshi|waist", "hips"),
    # --- Accessories → background (region 0) — before torso to avoid "armor_chest" matching "chest" ---
    (
        r"cloth|dress|skirt|hat|ribbon|accessory|cape|armor|weapon|shield|bow|jewel|ornament|belt|glove|boot|scarf",
        "background",
    ),
    # --- Torso / body (general — after specific regions and accessories) ---
    (r"body|torso|karada|chest|mune|bust", "chest"),
    (r"spine|senaka|back(?!ground)", "spine"),
]

# ---------------------------------------------------------------------------
# Spine bone/slot-to-Strata label mapping
# ---------------------------------------------------------------------------
# Ordered list of (regex_pattern, strata_region_name) tuples.
# Patterns are matched case-insensitively against Spine bone or slot names.
# First match wins — place specific patterns before general ones.
# Covers common Spine naming conventions (hyphenated and underscore variants).

SPINE_BONE_PATTERNS: list[tuple[str, str]] = [
    # --- Head (facial features, hair) ---
    (r"eye|brow|mouth|lip|nose|jaw|hair|bangs|goggles", "head"),
    (r"\bhead\b", "head"),
    # --- Neck ---
    (r"\bneck\b", "neck"),
    # --- Shoulders (before arms to avoid false matches) ---
    (
        r"shoulder.*(?:left|[-_]l\b)|(?:left|l[-_]).*shoulder|clavicle.*(?:left|[-_]l\b)",
        "shoulder_l",
    ),
    (
        r"shoulder.*(?:right|[-_]r\b)|(?:right|r[-_]).*shoulder|clavicle.*(?:right|[-_]r\b)",
        "shoulder_r",
    ),
    # --- Left arm (forearm/lower before upper) ---
    (
        r"(?:front|rear)?[-_]?(?:fore[-_]?arm|bracer|lower[-_]?arm).*(?:left|[-_]l\b)|(?:left|l[-_]).*(?:fore[-_]?arm|bracer|lower[-_]?arm)",
        "lower_arm_l",
    ),
    (
        r"(?:front|rear)?[-_]?upper[-_]?arm.*(?:left|[-_]l\b)|(?:left|l[-_]).*upper[-_]?arm",
        "upper_arm_l",
    ),
    (
        r"(?:front|rear)?[-_]?(?:fist|hand|finger|thumb).*(?:left|[-_]l\b)|(?:left|l[-_]).*(?:fist|hand|finger|thumb)",
        "hand_l",
    ),
    # --- Right arm ---
    (
        r"(?:front|rear)?[-_]?(?:fore[-_]?arm|bracer|lower[-_]?arm).*(?:right|[-_]r\b)|(?:right|r[-_]).*(?:fore[-_]?arm|bracer|lower[-_]?arm)",
        "lower_arm_r",
    ),
    (
        r"(?:front|rear)?[-_]?upper[-_]?arm.*(?:right|[-_]r\b)|(?:right|r[-_]).*upper[-_]?arm",
        "upper_arm_r",
    ),
    (
        r"(?:front|rear)?[-_]?(?:fist|hand|finger|thumb).*(?:right|[-_]r\b)|(?:right|r[-_]).*(?:fist|hand|finger|thumb)",
        "hand_r",
    ),
    # --- Left leg (lower/shin before upper) ---
    (
        r"(?:front|rear)?[-_]?(?:shin|lower[-_]?leg|calf).*(?:left|[-_]l\b)|(?:left|l[-_]).*(?:shin|lower[-_]?leg|calf)",
        "lower_leg_l",
    ),
    (
        r"(?:front|rear)?[-_]?(?:thigh|upper[-_]?leg).*(?:left|[-_]l\b)|(?:left|l[-_]).*(?:thigh|upper[-_]?leg)",
        "upper_leg_l",
    ),
    (r"(?:front|rear)?[-_]?(?:foot|toe).*(?:left|[-_]l\b)|(?:left|l[-_]).*(?:foot|toe)", "foot_l"),
    # --- Right leg ---
    (
        r"(?:front|rear)?[-_]?(?:shin|lower[-_]?leg|calf).*(?:right|[-_]r\b)|(?:right|r[-_]).*(?:shin|lower[-_]?leg|calf)",
        "lower_leg_r",
    ),
    (
        r"(?:front|rear)?[-_]?(?:thigh|upper[-_]?leg).*(?:right|[-_]r\b)|(?:right|r[-_]).*(?:thigh|upper[-_]?leg)",
        "upper_leg_r",
    ),
    (
        r"(?:front|rear)?[-_]?(?:foot|toe).*(?:right|[-_]r\b)|(?:right|r[-_]).*(?:foot|toe)",
        "foot_r",
    ),
    # --- Front/rear without explicit side (Spine convention: front=left, rear=right) ---
    (r"\bfront[-_]?(?:fore[-_]?arm|bracer|lower[-_]?arm)", "lower_arm_l"),
    (r"\bfront[-_]?upper[-_]?arm", "upper_arm_l"),
    (r"\bfront[-_]?(?:fist|hand|finger|thumb)", "hand_l"),
    (r"\brear[-_]?(?:fore[-_]?arm|bracer|lower[-_]?arm)", "lower_arm_r"),
    (r"\brear[-_]?upper[-_]?arm", "upper_arm_r"),
    (r"\brear[-_]?(?:fist|hand|finger|thumb)", "hand_r"),
    (r"\bfront[-_]?(?:shin|lower[-_]?leg|calf)", "lower_leg_l"),
    (r"\bfront[-_]?(?:thigh|upper[-_]?leg)", "upper_leg_l"),
    (r"\bfront[-_]?(?:foot|toe)", "foot_l"),
    (r"\brear[-_]?(?:shin|lower[-_]?leg|calf)", "lower_leg_r"),
    (r"\brear[-_]?(?:thigh|upper[-_]?leg)", "upper_leg_r"),
    (r"\brear[-_]?(?:foot|toe)", "foot_r"),
    # --- Hips / pelvis ---
    (r"\bhip|pelvis|waist", "hips"),
    # --- Accessories → background ---
    (
        r"weapon|sword|shield|bow|staff|gun|crosshair|muzzle|hover|portal|dust|cloak|cape|armor|helmet|hat|crown",
        "background",
    ),
    # --- Torso / body (general — after specific regions) ---
    (r"\btorso\b|\bchest\b|\bbody\b", "chest"),
    (r"\bspine\b|\bback\b", "spine"),
]

# ---------------------------------------------------------------------------
# VRoid material slot → Strata label mapping
# ---------------------------------------------------------------------------
# Ordered list of (regex_pattern, strata_region_name) tuples.
# Patterns are matched case-insensitively against VRoid material slot names.
# First match wins — place specific patterns before general ones.
# Covers standard VRoid Studio material slots and common custom naming.
#
# VRoid material-level mapping is COARSE — e.g., "Body" covers the entire
# body mesh.  Fine-grained per-vertex labeling uses bone weights via
# VRM_BONE_ALIASES in bone_mapper.py (issue #83).  Material patterns serve
# as a fallback when bone data is incomplete.

VROID_MATERIAL_PATTERNS: list[tuple[str, str]] = [
    # --- Head (facial features, hair) ---
    (r"\bface\b|\bFace_", "head"),
    (r"\beye\b|\bEyeWhite\b|\bEyeIris\b|\bEyeHighlight\b|\bEyeExtra\b", "head"),
    (r"\bbrow\b|\bEyebrow\b", "head"),
    (r"\bhair\b|\bHair(?:[-_]\w+)*\b|\bbangs\b|\bponytail\b|\bahoge\b", "head"),
    (r"\bmouth\b|\bMouth\b|\btooth\b|\btongue\b", "head"),
    (r"\bear\b|\bEar\b", "head"),
    (r"\bnose\b|\bNose\b", "head"),
    (r"\bhead\b|\bHead\b", "head"),
    # --- Neck ---
    (r"\bneck\b|\bNeck\b", "neck"),
    # --- Shoulders (before arms to avoid false matches) ---
    (r"shoulder.*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*shoulder", "shoulder_l"),
    (r"shoulder.*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*shoulder", "shoulder_r"),
    # --- Left arm (lower/forearm before upper to avoid substring clash) ---
    (
        r"(?:fore[-_]?arm|lower[-_]?arm).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:fore[-_]?arm|lower[-_]?arm)",
        "lower_arm_l",
    ),
    (
        r"(?:upper[-_]?arm|arm).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:upper[-_]?arm|arm)",
        "upper_arm_l",
    ),
    (
        r"(?:glove|hand|finger).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:glove|hand|finger)",
        "hand_l",
    ),
    # --- Right arm ---
    (
        r"(?:fore[-_]?arm|lower[-_]?arm).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:fore[-_]?arm|lower[-_]?arm)",
        "lower_arm_r",
    ),
    (
        r"(?:upper[-_]?arm|arm).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:upper[-_]?arm|arm)",
        "upper_arm_r",
    ),
    (
        r"(?:glove|hand|finger).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:glove|hand|finger)",
        "hand_r",
    ),
    # --- Left leg (lower/shin before upper to avoid substring clash) ---
    (
        r"(?:shin|lower[-_]?leg|calf).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:shin|lower[-_]?leg|calf)",
        "lower_leg_l",
    ),
    (
        r"(?:thigh|upper[-_]?leg|leg).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:thigh|upper[-_]?leg|leg)",
        "upper_leg_l",
    ),
    (
        r"(?:shoe|foot|toe|sock|boot).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:shoe|foot|toe|sock|boot)",
        "foot_l",
    ),
    # --- Right leg ---
    (
        r"(?:shin|lower[-_]?leg|calf).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:shin|lower[-_]?leg|calf)",
        "lower_leg_r",
    ),
    (
        r"(?:thigh|upper[-_]?leg|leg).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:thigh|upper[-_]?leg|leg)",
        "upper_leg_r",
    ),
    (
        r"(?:shoe|foot|toe|sock|boot).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:shoe|foot|toe|sock|boot)",
        "foot_r",
    ),
    # --- Hips / lower body ---
    (r"\bhip|pelvis|waist|Outfit_Lower|pants|skirt|shorts", "hips"),
    # --- Accessories → background (before torso to prevent false matches) ---
    (
        r"accessory|ribbon|wing|tail|cape|cloak|weapon|shield|bag|mask|ornament|jewelry|crown|hat|glasses|goggle|belt|scarf|collar",
        "background",
    ),
    # --- Torso / body (general — after specific regions and accessories) ---
    (r"Outfit_Upper|jacket|shirt|vest|coat|blazer|uniform", "chest"),
    (r"\bbody\b|Body|torso|chest|mune|bust", "chest"),
    (r"\bspine\b|\bback\b", "spine"),
]

# ---------------------------------------------------------------------------
# PSD layer name → Strata label mapping
# ---------------------------------------------------------------------------
# Ordered list of (regex_pattern, strata_region_name) tuples.
# Patterns are matched case-insensitively against PSD layer names.
# First match wins — place specific patterns before general ones.
# Covers common artist naming conventions for body-part-separated PSDs
# (game sprites, paper dolls, character rigs).

PSD_LAYER_PATTERNS: list[tuple[str, str]] = [
    # --- Head (facial features, hair) ---
    (r"eye|iris|pupil|sclera|eyelid|eyelash", "head"),
    (r"brow|eyebrow", "head"),
    (r"mouth|lip|teeth|tongue|jaw", "head"),
    (r"nose", "head"),
    (r"\bear\b", "head"),
    (r"hair|bangs|fringe|ponytail|ahoge|braid|pigtail", "head"),
    (r"face|cheek|chin|forehead", "head"),
    (r"\bhead\b", "head"),
    # --- Neck ---
    (r"\bneck\b", "neck"),
    # --- Shoulders (before arms to avoid false matches) ---
    (r"shoulder.*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*shoulder", "shoulder_l"),
    (r"shoulder.*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*shoulder", "shoulder_r"),
    # --- Left arm (forearm/lower before upper to avoid substring clash) ---
    (
        r"(?:fore[-_]?arm|lower[-_]?arm).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:fore[-_]?arm|lower[-_]?arm)",
        "lower_arm_l",
    ),
    (
        r"(?:upper[-_]?arm|arm).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:upper[-_]?arm|arm)",
        "upper_arm_l",
    ),
    (
        r"(?:hand|finger|thumb|palm).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:hand|finger|thumb|palm)",
        "hand_l",
    ),
    # --- Right arm ---
    (
        r"(?:fore[-_]?arm|lower[-_]?arm).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:fore[-_]?arm|lower[-_]?arm)",
        "lower_arm_r",
    ),
    (
        r"(?:upper[-_]?arm|arm).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:upper[-_]?arm|arm)",
        "upper_arm_r",
    ),
    (
        r"(?:hand|finger|thumb|palm).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:hand|finger|thumb|palm)",
        "hand_r",
    ),
    # --- Left leg (lower/shin before upper to avoid substring clash) ---
    (
        r"(?:shin|lower[-_]?leg|calf).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:shin|lower[-_]?leg|calf)",
        "lower_leg_l",
    ),
    (
        r"(?:thigh|upper[-_]?leg|leg).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:thigh|upper[-_]?leg|leg)",
        "upper_leg_l",
    ),
    (
        r"(?:foot|toe|shoe|boot).*(?:left|[-_.]?[lL]\b)|(?:left|[lL][-_.]).*(?:foot|toe|shoe|boot)",
        "foot_l",
    ),
    # --- Right leg ---
    (
        r"(?:shin|lower[-_]?leg|calf).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:shin|lower[-_]?leg|calf)",
        "lower_leg_r",
    ),
    (
        r"(?:thigh|upper[-_]?leg|leg).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:thigh|upper[-_]?leg|leg)",
        "upper_leg_r",
    ),
    (
        r"(?:foot|toe|shoe|boot).*(?:right|[-_.]?[rR]\b)|(?:right|[rR][-_.]).*(?:foot|toe|shoe|boot)",
        "foot_r",
    ),
    # --- Hips / lower body ---
    (r"\bhip|pelvis|waist", "hips"),
    # --- Accessories → background (before torso to prevent false matches) ---
    (
        r"accessory|ribbon|wing|tail|cape|cloak|weapon|shield|bag|mask|ornament|jewelry|crown|hat|glasses|belt|scarf",
        "background",
    ),
    # --- Skip layers (rendering concerns, not body parts) → background ---
    (r"\blineart\b|\bline[-_]?art\b|\boutline\b|\bink\b", "background"),
    (r"\bshad(?:ow|ing)\b|\bhighlight\b|\brim[-_]?light\b", "background"),
    (r"\bflat[-_]?color\b|\bbase[-_]?color\b|\bcolor\b", "background"),
    (r"\bbackground\b|\bbg\b|\bsky\b|\bground\b|\bfloor\b", "background"),
    (r"\beffect\b|\bfx\b|\bparticle\b|\bglow\b|\bblur\b", "background"),
    # --- Torso / body (general — after specific regions and accessories) ---
    (r"\btorso\b|\bchest\b|\bupper[-_]?body\b|\bbreast\b|\bbust\b", "chest"),
    (r"\bspine\b|\bback\b", "spine"),
    (r"\bbody\b|\btorso\b", "chest"),
]

# ---------------------------------------------------------------------------
# StdGEN semantic class → Strata mapping
# ---------------------------------------------------------------------------
# StdGEN (CVPR 2025) annotates VRoid characters with 4 semantic classes.
# "hair" and "face" map directly to head (region 1).
# "body" and "clothes" require per-vertex bone-weight refinement using
# VRM_BONE_ALIASES to resolve into Strata's 16 body sub-regions.

STDGEN_SEMANTIC_CLASSES: dict[str, RegionId | None] = {
    "hair": 1,  # → head
    "face": 1,  # → head
    "body": None,  # → per-vertex bone-weight refinement
    "clothes": None,  # → underlying body region via bone weights
}

# ---------------------------------------------------------------------------
# Live2D renderer augmentation settings
# ---------------------------------------------------------------------------
# Rotation angles (degrees) applied to Live2D composites for data augmentation.
LIVE2D_AUGMENTATION_ROTATIONS: list[float] = [-5.0, 0.0, 5.0]

# Uniform scale factors applied to Live2D composites for data augmentation.
LIVE2D_AUGMENTATION_SCALES: list[float] = [0.9, 1.0, 1.1]

# Color jitter ranges: (hue_shift_degrees, saturation_factor, brightness_factor).
# Each element is (min, max) for random uniform sampling.
LIVE2D_AUGMENTATION_COLOR_JITTER: dict[str, tuple[float, float]] = {
    "hue": (-15.0, 15.0),  # degrees on hue wheel
    "saturation": (0.85, 1.15),  # multiplicative factor
    "brightness": (0.85, 1.15),  # multiplicative factor
}

# ---------------------------------------------------------------------------
# Contour rendering (Freestyle line pairs for contour-removal training)
# ---------------------------------------------------------------------------

# Freestyle edge detection flags — which edge types to render as contour lines.
CONTOUR_EDGE_SILHOUETTE: bool = True
CONTOUR_EDGE_CREASE: bool = True
CONTOUR_EDGE_MATERIAL_BOUNDARY: bool = True
CONTOUR_EDGE_BORDER: bool = False  # Open mesh edges — usually not wanted
CONTOUR_EDGE_MARK: bool = False  # Manually marked edges

# Freestyle line thickness for the base contour render (in pixels).
CONTOUR_FREESTYLE_THICKNESS: float = 2.0

# Pixel-difference threshold for computing the binary contour mask.
# Pixels where abs(with_contours - without_contours) > threshold are contour pixels.
CONTOUR_DIFF_THRESHOLD: int = 30

# ---------------------------------------------------------------------------
# Contour augmentation styles
# ---------------------------------------------------------------------------
# Each style defines: (name, line_width_px, color_rgb, opacity, wobble).
# color_rgb can be a single RGB tuple or "per_region" for region-based coloring.

CONTOUR_STYLE_THIN_BLACK: dict[str, object] = {
    "name": "thin_black",
    "line_width": 1,
    "color": (0, 0, 0),
    "opacity": 1.0,
    "wobble": False,
}

CONTOUR_STYLE_MEDIUM_BLACK: dict[str, object] = {
    "name": "medium_black",
    "line_width": 2,
    "color": (0, 0, 0),
    "opacity": 1.0,
    "wobble": False,
}

CONTOUR_STYLE_THICK_BROWN: dict[str, object] = {
    "name": "thick_brown",
    "line_width": 3,
    "color": (80, 50, 30),
    "opacity": 0.9,
    "wobble": False,
}

CONTOUR_STYLE_PER_REGION: dict[str, object] = {
    "name": "per_region",
    "line_width": 1,
    "color": "per_region",
    "opacity": 0.8,
    "wobble": False,
}

CONTOUR_STYLE_WOBBLY: dict[str, object] = {
    "name": "wobbly",
    "line_width": 2,
    "color": (0, 0, 0),
    "opacity": 1.0,
    "wobble": True,
}

CONTOUR_STYLES: list[dict[str, object]] = [
    CONTOUR_STYLE_THIN_BLACK,
    CONTOUR_STYLE_MEDIUM_BLACK,
    CONTOUR_STYLE_THICK_BROWN,
    CONTOUR_STYLE_PER_REGION,
    CONTOUR_STYLE_WOBBLY,
]

# Wobble displacement range (pixels) for hand-drawn contour effect.
CONTOUR_WOBBLE_RANGE: int = 2

# Per-region contour colors — maps region ID to RGB.
# Uses warm/cool alternating scheme so adjacent regions contrast.
CONTOUR_REGION_COLORS: dict[RegionId, RGB] = {
    0: (0, 0, 0),  # background — black (fallback)
    1: (200, 50, 50),  # head — red
    2: (180, 100, 60),  # neck — brown
    3: (50, 100, 200),  # chest — blue
    4: (80, 150, 80),  # spine — green
    5: (150, 80, 150),  # hips — purple
    6: (200, 130, 50),  # upper_arm_l — orange
    7: (50, 170, 170),  # lower_arm_l — teal
    8: (200, 50, 120),  # hand_l — magenta
    9: (200, 130, 50),  # upper_arm_r — orange
    10: (50, 170, 170),  # lower_arm_r — teal
    11: (200, 50, 120),  # hand_r — magenta
    12: (80, 80, 200),  # upper_leg_l — indigo
    13: (50, 150, 50),  # lower_leg_l — green
    14: (180, 50, 50),  # foot_l — crimson
    15: (80, 80, 200),  # upper_leg_r — indigo
    16: (50, 150, 50),  # lower_leg_r — green
    17: (180, 50, 50),  # foot_r — crimson
    18: (160, 120, 80),  # shoulder_l — tan
    19: (160, 120, 80),  # shoulder_r — tan
}
