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
# Strata Standard Skeleton — 17 body regions + background (18 total)
# ---------------------------------------------------------------------------

REGION_NAMES: dict[RegionId, str] = {
    0:  "background",
    1:  "head",
    2:  "neck",
    3:  "chest",
    4:  "spine",
    5:  "hips",
    6:  "upper_arm_l",
    7:  "lower_arm_l",
    8:  "hand_l",
    9:  "upper_arm_r",
    10: "lower_arm_r",
    11: "hand_r",
    12: "upper_leg_l",
    13: "lower_leg_l",
    14: "foot_l",
    15: "upper_leg_r",
    16: "lower_leg_r",
    17: "foot_r",
}

REGION_COLORS: dict[RegionId, RGB] = {
    0:  (0, 0, 0),        # background (transparent in final output)
    1:  (255, 0, 0),      # head
    2:  (0, 255, 0),      # neck
    3:  (0, 0, 255),      # chest
    4:  (255, 255, 0),    # spine
    5:  (255, 0, 255),    # hips
    6:  (128, 0, 0),      # upper_arm_l
    7:  (0, 128, 0),      # lower_arm_l
    8:  (0, 0, 128),      # hand_l
    9:  (128, 128, 0),    # upper_arm_r
    10: (128, 0, 128),    # lower_arm_r
    11: (0, 128, 128),    # hand_r
    12: (64, 0, 0),       # upper_leg_l
    13: (0, 64, 0),       # lower_leg_l
    14: (0, 0, 64),       # foot_l
    15: (64, 64, 0),      # upper_leg_r
    16: (64, 0, 64),      # lower_leg_r
    17: (0, 64, 64),      # foot_r
}

NUM_REGIONS: int = 18  # 0–17 inclusive

# ---------------------------------------------------------------------------
# Mixamo bone name → region ID mapping
# ---------------------------------------------------------------------------
# Covers all standard Mixamo humanoid bones (prefix: "mixamorig:").
# Finger bones map to the corresponding hand region.

MIXAMO_BONE_MAP: dict[str, RegionId] = {
    # Head / neck
    "mixamorig:Head":              1,
    "mixamorig:HeadTop_End":       1,
    "mixamorig:Neck":              2,
    # Torso
    "mixamorig:Spine2":            3,   # chest
    "mixamorig:Spine1":            4,   # spine
    "mixamorig:Spine":             4,   # spine
    "mixamorig:Hips":              5,
    # Left arm
    "mixamorig:LeftShoulder":      6,
    "mixamorig:LeftArm":           6,
    "mixamorig:LeftForeArm":       7,
    "mixamorig:LeftHand":          8,
    # Left fingers → hand_l
    "mixamorig:LeftHandThumb1":    8,
    "mixamorig:LeftHandThumb2":    8,
    "mixamorig:LeftHandThumb3":    8,
    "mixamorig:LeftHandThumb4":    8,
    "mixamorig:LeftHandIndex1":    8,
    "mixamorig:LeftHandIndex2":    8,
    "mixamorig:LeftHandIndex3":    8,
    "mixamorig:LeftHandIndex4":    8,
    "mixamorig:LeftHandMiddle1":   8,
    "mixamorig:LeftHandMiddle2":   8,
    "mixamorig:LeftHandMiddle3":   8,
    "mixamorig:LeftHandMiddle4":   8,
    "mixamorig:LeftHandRing1":     8,
    "mixamorig:LeftHandRing2":     8,
    "mixamorig:LeftHandRing3":     8,
    "mixamorig:LeftHandRing4":     8,
    "mixamorig:LeftHandPinky1":    8,
    "mixamorig:LeftHandPinky2":    8,
    "mixamorig:LeftHandPinky3":    8,
    "mixamorig:LeftHandPinky4":    8,
    # Right arm
    "mixamorig:RightShoulder":     9,
    "mixamorig:RightArm":          9,
    "mixamorig:RightForeArm":      10,
    "mixamorig:RightHand":         11,
    # Right fingers → hand_r
    "mixamorig:RightHandThumb1":   11,
    "mixamorig:RightHandThumb2":   11,
    "mixamorig:RightHandThumb3":   11,
    "mixamorig:RightHandThumb4":   11,
    "mixamorig:RightHandIndex1":   11,
    "mixamorig:RightHandIndex2":   11,
    "mixamorig:RightHandIndex3":   11,
    "mixamorig:RightHandIndex4":   11,
    "mixamorig:RightHandMiddle1":  11,
    "mixamorig:RightHandMiddle2":  11,
    "mixamorig:RightHandMiddle3":  11,
    "mixamorig:RightHandMiddle4":  11,
    "mixamorig:RightHandRing1":    11,
    "mixamorig:RightHandRing2":    11,
    "mixamorig:RightHandRing3":    11,
    "mixamorig:RightHandRing4":    11,
    "mixamorig:RightHandPinky1":   11,
    "mixamorig:RightHandPinky2":   11,
    "mixamorig:RightHandPinky3":   11,
    "mixamorig:RightHandPinky4":   11,
    # Left leg
    "mixamorig:LeftUpLeg":         12,
    "mixamorig:LeftLeg":           13,
    "mixamorig:LeftFoot":          14,
    "mixamorig:LeftToeBase":       14,
    "mixamorig:LeftToe_End":       14,
    # Right leg
    "mixamorig:RightUpLeg":        15,
    "mixamorig:RightLeg":          16,
    "mixamorig:RightFoot":         17,
    "mixamorig:RightToeBase":      17,
    "mixamorig:RightToe_End":      17,
}

# ---------------------------------------------------------------------------
# Common bone name aliases for non-Mixamo rigs
# ---------------------------------------------------------------------------
# Covers Blender-style (bone.L / bone.R), generic uppercase, and generic
# lowercase variants. Used as fallback when exact Mixamo names don't match.

COMMON_BONE_ALIASES: dict[str, RegionId] = {
    # --- Head / neck ---
    "head":             1,
    "Head":             1,
    "HEAD":             1,
    "skull":            1,
    "Skull":            1,
    "neck":             2,
    "Neck":             2,
    "NECK":             2,
    # --- Torso ---
    "chest":            3,
    "Chest":            3,
    "upper_body":       3,
    "torso_upper":      3,
    "Spine2":           3,
    "spine":            4,
    "Spine":            4,
    "Spine1":           4,
    "torso":            4,
    "torso_lower":      4,
    "abdomen":          4,
    "hips":             5,
    "Hips":             5,
    "pelvis":           5,
    "Pelvis":           5,
    "root":             5,
    "Root":             5,
    # --- Left arm ---
    "upper_arm.L":      6,
    "upperarm.L":       6,
    "UpperArm.L":       6,
    "L_upperarm":       6,
    "l_upperarm":       6,
    "LeftArm":          6,
    "LeftShoulder":     6,
    "shoulder.L":       6,
    "forearm.L":        7,
    "Forearm.L":        7,
    "lower_arm.L":      7,
    "L_forearm":        7,
    "l_forearm":        7,
    "LeftForeArm":      7,
    "hand.L":           8,
    "Hand.L":           8,
    "L_hand":           8,
    "l_hand":           8,
    "LeftHand":         8,
    # --- Right arm ---
    "upper_arm.R":      9,
    "upperarm.R":       9,
    "UpperArm.R":       9,
    "R_upperarm":       9,
    "r_upperarm":       9,
    "RightArm":         9,
    "RightShoulder":    9,
    "shoulder.R":       9,
    "forearm.R":        10,
    "Forearm.R":        10,
    "lower_arm.R":      10,
    "R_forearm":        10,
    "r_forearm":        10,
    "RightForeArm":     10,
    "hand.R":           11,
    "Hand.R":           11,
    "R_hand":           11,
    "r_hand":           11,
    "RightHand":        11,
    # --- Left leg ---
    "thigh.L":          12,
    "Thigh.L":          12,
    "upper_leg.L":      12,
    "L_thigh":          12,
    "l_thigh":          12,
    "LeftUpLeg":        12,
    "shin.L":           13,
    "Shin.L":           13,
    "lower_leg.L":      13,
    "calf.L":           13,
    "L_calf":           13,
    "l_calf":           13,
    "LeftLeg":          13,
    "foot.L":           14,
    "Foot.L":           14,
    "L_foot":           14,
    "l_foot":           14,
    "LeftFoot":         14,
    "toe.L":            14,
    "LeftToeBase":      14,
    # --- Right leg ---
    "thigh.R":          15,
    "Thigh.R":          15,
    "upper_leg.R":      15,
    "R_thigh":          15,
    "r_thigh":          15,
    "RightUpLeg":       15,
    "shin.R":           16,
    "Shin.R":           16,
    "lower_leg.R":      16,
    "calf.R":           16,
    "R_calf":           16,
    "r_calf":           16,
    "RightLeg":         16,
    "foot.R":           17,
    "Foot.R":           17,
    "R_foot":           17,
    "r_foot":           17,
    "RightFoot":        17,
    "toe.R":            17,
    "RightToeBase":     17,
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
BACKGROUND_TRANSPARENT: bool = True

# ---------------------------------------------------------------------------
# Art styles
# ---------------------------------------------------------------------------
# Render-time (Blender shaders): flat, cel, unlit
# Post-render (Python/PIL/OpenCV): pixel, painterly, sketch

ART_STYLES: list[str] = ["flat", "cel", "pixel", "painterly", "sketch", "unlit"]

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

SPLIT_RATIOS: dict[str, float] = {
    "train": 0.8,
    "val":   0.1,
    "test":  0.1,
}
