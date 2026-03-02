"""Training data transforms with region-aware L/R flipping.

Provides augmentation utilities that correctly swap left/right body region
IDs when horizontally flipping segmentation masks and joint positions.
Also defines bone ordering constants that match the Strata Rust runtime.

Pure Python + NumPy/Torch (no Blender dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# ImageNet normalization (must match Rust runtime preprocessing exactly)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Bone ordering — matches Rust's joints.rs BONE_NAMES (20-slot list)
# ---------------------------------------------------------------------------

BONE_ORDER: list[str] = [
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "shoulder_l",
    "upper_arm_l",
    "forearm_l",
    "hand_l",
    "shoulder_r",
    "upper_arm_r",
    "forearm_r",
    "hand_r",
    "upper_leg_l",
    "lower_leg_l",
    "foot_l",
    "upper_leg_r",
    "lower_leg_r",
    "foot_r",
    "hair_back",
]

BONE_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(BONE_ORDER)}

# ---------------------------------------------------------------------------
# Pipeline name → Rust bone name mapping
# ---------------------------------------------------------------------------
# Pipeline and Rust now both use "forearm" naming.

PIPELINE_TO_BONE: dict[str, str] = {
    "hips": "hips",
    "spine": "spine",
    "chest": "chest",
    "neck": "neck",
    "head": "head",
    "shoulder_l": "shoulder_l",
    "upper_arm_l": "upper_arm_l",
    "forearm_l": "forearm_l",
    "hand_l": "hand_l",
    "shoulder_r": "shoulder_r",
    "upper_arm_r": "upper_arm_r",
    "forearm_r": "forearm_r",
    "hand_r": "hand_r",
    "upper_leg_l": "upper_leg_l",
    "lower_leg_l": "lower_leg_l",
    "foot_l": "foot_l",
    "upper_leg_r": "upper_leg_r",
    "lower_leg_r": "lower_leg_r",
    "foot_r": "foot_r",
    "accessory": "accessory",
    "hair_back": "hair_back",
}

# ---------------------------------------------------------------------------
# L/R region swap pairs (region IDs from pipeline/config.py)
# ---------------------------------------------------------------------------
# When a segmentation mask is horizontally flipped, left regions become right
# and vice versa.  These pairs define the bidirectional swaps.

FLIP_REGION_SWAP: dict[int, int] = {
    6: 10,  # shoulder_l ↔ shoulder_r
    10: 6,
    7: 11,  # upper_arm_l ↔ upper_arm_r
    11: 7,
    8: 12,  # forearm_l ↔ forearm_r
    12: 8,
    9: 13,  # hand_l ↔ hand_r
    13: 9,
    14: 17,  # upper_leg_l ↔ upper_leg_r
    17: 14,
    15: 18,  # lower_leg_l ↔ lower_leg_r
    18: 15,
    16: 19,  # foot_l ↔ foot_r
    19: 16,
}

# L/R joint name swap pairs for joint position flipping.
FLIP_JOINT_SWAP: dict[str, str] = {
    "shoulder_l": "shoulder_r",
    "shoulder_r": "shoulder_l",
    "upper_arm_l": "upper_arm_r",
    "upper_arm_r": "upper_arm_l",
    "forearm_l": "forearm_r",
    "forearm_r": "forearm_l",
    "hand_l": "hand_r",
    "hand_r": "hand_l",
    "upper_leg_l": "upper_leg_r",
    "upper_leg_r": "upper_leg_l",
    "lower_leg_l": "lower_leg_r",
    "lower_leg_r": "lower_leg_l",
    "foot_l": "foot_r",
    "foot_r": "foot_l",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flip_mask(mask: np.ndarray) -> np.ndarray:
    """Horizontally flip a segmentation mask and swap L/R region IDs.

    Args:
        mask: 2D uint8 array where pixel values are region IDs (0-21).

    Returns:
        New array with horizontal flip applied and L/R regions swapped.
    """
    # Flip horizontally
    flipped = np.flip(mask, axis=1).copy()

    # Build a lookup table for region swapping (avoids overwrite issues)
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in FLIP_REGION_SWAP.items():
        lut[src] = dst

    return lut[flipped]


def flip_joints(joints_dict: dict, image_width: int) -> dict:
    """Flip joint positions horizontally and swap L/R joint names.

    Args:
        joints_dict: Dict mapping joint names to position dicts with
            ``"x"``, ``"y"`` keys (and optionally ``"visible"``).
        image_width: Width of the image in pixels (for mirroring x coords).

    Returns:
        New dict with mirrored x coordinates and swapped L/R joint names.
    """
    result: dict = {}
    for name, pos in joints_dict.items():
        new_name = FLIP_JOINT_SWAP.get(name, name)
        result[new_name] = {
            "x": image_width - 1 - pos["x"],
            "y": pos["y"],
            **({"visible": pos["visible"]} if "visible" in pos else {}),
        }
    return result


def normalize_imagenet(img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a float image tensor.

    Args:
        img_tensor: ``[C, H, W]`` float tensor with values in ``[0, 1]``.

    Returns:
        Normalized tensor with ImageNet mean subtracted and divided by std.
    """
    import torch

    mean = torch.tensor(IMAGENET_MEAN, dtype=img_tensor.dtype, device=img_tensor.device)
    std = torch.tensor(IMAGENET_STD, dtype=img_tensor.dtype, device=img_tensor.device)
    return (img_tensor - mean[:, None, None]) / std[:, None, None]
