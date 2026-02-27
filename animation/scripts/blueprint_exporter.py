"""Export retargeted animation as Strata blueprint JSON.

Produces a JSON file consumable by the Strata animation system, containing
per-frame bone rotations and root positions for all 19 Strata skeleton bones.

No Blender dependency — pure Python.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from animation.scripts.bvh_to_strata import STRATA_BONES, RetargetedAnimation

logger = logging.getLogger(__name__)

SKELETON_ID: str = "strata_19"


# ---------------------------------------------------------------------------
# Blueprint construction
# ---------------------------------------------------------------------------

def _round_vec(vec: tuple[float, float, float], decimals: int = 4) -> list[float]:
    """Round a 3-component vector for compact JSON output."""
    return [round(v, decimals) for v in vec]


def build_blueprint(animation: RetargetedAnimation) -> dict[str, Any]:
    """Build a Strata blueprint dict from a retargeted animation.

    The blueprint contains all 19 Strata bones per frame.  Each bone has
    a ``rotation`` [x, y, z] in degrees.  The root bone (``hips``) also
    has a ``position`` [x, y, z].

    Args:
        animation: Retargeted (and optionally proportion-normalized) animation.

    Returns:
        Blueprint dict matching the Strata animation schema.
    """
    frames_out: list[dict[str, Any]] = []

    for frame in animation.frames:
        frame_dict: dict[str, Any] = {}
        for bone in STRATA_BONES:
            rotation = frame.rotations.get(bone, (0.0, 0.0, 0.0))
            bone_data: dict[str, Any] = {"rotation": _round_vec(rotation)}

            # Only the root bone carries position data
            if bone == "hips":
                bone_data["position"] = _round_vec(frame.root_position)

            frame_dict[bone] = bone_data
        frames_out.append(frame_dict)

    return {
        "skeleton": SKELETON_ID,
        "frame_count": animation.frame_count,
        "frame_rate": animation.frame_rate,
        "rotation_order": animation.rotation_order,
        "frames": frames_out,
    }


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------

def export_blueprint(
    animation: RetargetedAnimation,
    output_path: Path | str,
) -> Path:
    """Export a retargeted animation as a Strata blueprint JSON file.

    Args:
        animation: Retargeted animation data.
        output_path: Destination file path (.json).

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = build_blueprint(animation)

    output_path.write_text(
        json.dumps(blueprint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Exported blueprint: %s (%d frames, %.1f fps)",
        output_path,
        animation.frame_count,
        animation.frame_rate,
    )

    return output_path
