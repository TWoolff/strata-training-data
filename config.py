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
    18: "shoulder_l",
    19: "shoulder_r",
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
    18: (192, 64, 0),     # shoulder_l
    19: (0, 64, 192),     # shoulder_r
}

NUM_REGIONS: int = 20  # 0–19 inclusive

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
    # Left shoulder / arm
    "mixamorig:LeftShoulder":      18,
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
    # Right shoulder / arm
    "mixamorig:RightShoulder":     19,
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
    # --- Left shoulder ---
    "LeftShoulder":     18,
    "shoulder.L":       18,
    "Shoulder.L":       18,
    "clavicle.L":       18,
    "Clavicle.L":       18,
    "L_clavicle":       18,
    "l_clavicle":       18,
    "L_shoulder":       18,
    "l_shoulder":       18,
    # --- Left arm ---
    "upper_arm.L":      6,
    "upperarm.L":       6,
    "UpperArm.L":       6,
    "L_upperarm":       6,
    "l_upperarm":       6,
    "LeftArm":          6,
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
    # --- Right shoulder ---
    "RightShoulder":    19,
    "shoulder.R":       19,
    "Shoulder.R":       19,
    "clavicle.R":       19,
    "Clavicle.R":       19,
    "R_clavicle":       19,
    "r_clavicle":       19,
    "R_shoulder":       19,
    "r_shoulder":       19,
    # --- Right arm ---
    "upper_arm.R":      9,
    "upperarm.R":       9,
    "UpperArm.R":       9,
    "R_upperarm":       9,
    "r_upperarm":       9,
    "RightArm":         9,
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
]

# ---------------------------------------------------------------------------
# Bone mapping: substring keywords
# ---------------------------------------------------------------------------
# Case-insensitive keyword tuples → region ID. Checked in order; first match
# wins. More specific patterns (e.g., "forearm") must come before general
# ones (e.g., "arm") to avoid false matches.

SUBSTRING_KEYWORDS: list[tuple[list[str], RegionId]] = [
    # Head / neck
    (["head"],                              1),
    (["skull"],                             1),
    (["neck"],                              2),
    # Torso
    (["chest"],                             3),
    (["spine2"],                            3),
    (["spine1"],                            4),
    (["spine"],                             4),
    (["hip"],                               5),
    (["pelvis"],                            5),
    # Shoulders (before arm to avoid "shoulder" matching "arm")
    (["shoulder", "left"],                  18),
    (["shoulder", "l"],                     18),
    (["clavicle", "left"],                  18),
    (["clavicle", "l"],                     18),
    (["shoulder", "right"],                 19),
    (["shoulder", "r"],                     19),
    (["clavicle", "right"],                 19),
    (["clavicle", "r"],                     19),
    # Left arm (forearm before arm to avoid false match)
    (["forearm", "left"],                   7),
    (["forearm", "l"],                      7),
    (["lower", "arm", "left"],              7),
    (["lower", "arm", "l"],                 7),
    (["arm", "left", "up"],                 6),
    (["arm", "left"],                       6),
    (["arm", "l", "up"],                    6),
    (["upper", "arm", "l"],                 6),
    # Left hand / fingers
    (["hand", "left"],                      8),
    (["hand", "l"],                         8),
    (["finger", "left"],                    8),
    (["finger", "l"],                       8),
    (["thumb", "left"],                     8),
    (["thumb", "l"],                        8),
    # Right arm
    (["forearm", "right"],                  10),
    (["forearm", "r"],                      10),
    (["lower", "arm", "right"],             10),
    (["lower", "arm", "r"],                 10),
    (["arm", "right", "up"],                9),
    (["arm", "right"],                      9),
    (["arm", "r", "up"],                    9),
    (["upper", "arm", "r"],                 9),
    # Right hand / fingers
    (["hand", "right"],                     11),
    (["hand", "r"],                         11),
    (["finger", "right"],                   11),
    (["finger", "r"],                       11),
    (["thumb", "right"],                    11),
    (["thumb", "r"],                        11),
    # Left leg (shin/calf before generic leg)
    (["shin", "left"],                      13),
    (["shin", "l"],                         13),
    (["calf", "left"],                      13),
    (["calf", "l"],                         13),
    (["lower", "leg", "left"],              13),
    (["lower", "leg", "l"],                 13),
    (["thigh", "left"],                     12),
    (["thigh", "l"],                        12),
    (["upper", "leg", "left"],              12),
    (["upper", "leg", "l"],                 12),
    (["leg", "left", "up"],                 12),
    (["leg", "l", "up"],                    12),
    # Left foot / toes
    (["foot", "left"],                      14),
    (["foot", "l"],                         14),
    (["toe", "left"],                       14),
    (["toe", "l"],                          14),
    # Right leg
    (["shin", "right"],                     16),
    (["shin", "r"],                         16),
    (["calf", "right"],                     16),
    (["calf", "r"],                         16),
    (["lower", "leg", "right"],             16),
    (["lower", "leg", "r"],                 16),
    (["thigh", "right"],                    15),
    (["thigh", "r"],                        15),
    (["upper", "leg", "right"],             15),
    (["upper", "leg", "r"],                 15),
    (["leg", "right", "up"],                15),
    (["leg", "r", "up"],                    15),
    # Right foot / toes
    (["foot", "right"],                     17),
    (["foot", "r"],                         17),
    (["toe", "right"],                      17),
    (["toe", "r"],                          17),
]

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
