"""Filter flux_diverse dataset — remove anatomically implausible examples.

Removes examples where the pseudo-labeled segmentation mask is missing
torso regions (chest/spine/hips), which indicates either a broken FLUX
image or a bad pseudo-label.

Also writes a review manifest of all remaining examples sorted by
anatomical completeness score, for manual review of borderline cases.

Usage:
    python scripts/filter_flux_diverse.py \
        --input-dir /Volumes/TAMWoolff/data/output/flux_diverse \
        --output-dir /Volumes/TAMWoolff/data/output/flux_diverse_clean \
        --manifest ./output/flux_diverse_review_manifest.json
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Strata region IDs
TORSO_REGIONS = {3, 4, 5}       # chest, spine, hips
HEAD_REGIONS = {1, 2}           # head, neck
ARM_REGIONS_L = {6, 7, 8, 9}   # shoulder_l, upper_arm_l, forearm_l, hand_l
ARM_REGIONS_R = {10, 11, 12, 13}
LEG_REGIONS_L = {14, 15, 16}
LEG_REGIONS_R = {17, 18, 19}


def score_example(seg: np.ndarray) -> tuple[float, dict]:
    """Return anatomical completeness score and details."""
    unique = set(np.unique(seg).tolist()) - {0}  # exclude background

    has_torso = bool(unique & TORSO_REGIONS)
    has_head = bool(unique & HEAD_REGIONS)
    has_arms = bool(unique & ARM_REGIONS_L) or bool(unique & ARM_REGIONS_R)
    has_legs = bool(unique & LEG_REGIONS_L) or bool(unique & LEG_REGIONS_R)
    fg_ratio = float((seg > 0).sum() / seg.size)
    num_regions = len(unique)

    score = (
        (1.0 if has_torso else 0.0) +
        (0.5 if has_head else 0.0) +
        (0.3 if has_arms else 0.0) +
        (0.3 if has_legs else 0.0) +
        min(num_regions / 20.0, 0.5) +
        min(fg_ratio / 0.3, 0.4)
    )

    return score, {
        "has_torso": has_torso,
        "has_head": has_head,
        "has_arms": has_arms,
        "has_legs": has_legs,
        "num_regions": num_regions,
        "fg_ratio": round(fg_ratio, 3),
        "score": round(score, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="./output/flux_diverse_review_manifest.json")
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="Minimum anatomical score to keep (default: 1.0 = must have torso)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't copy files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    examples = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(examples)} examples in {input_dir}")

    kept = []
    rejected = []

    for ex in examples:
        seg_path = ex / "segmentation.png"
        if not seg_path.exists():
            rejected.append({"name": ex.name, "reason": "no segmentation.png", "score": 0})
            continue

        seg = np.array(Image.open(seg_path))
        score, details = score_example(seg)
        details["name"] = ex.name

        if not details["has_torso"]:
            details["reason"] = "no_torso"
            rejected.append(details)
        elif score < args.min_score:
            details["reason"] = f"low_score ({score:.2f} < {args.min_score})"
            rejected.append(details)
        else:
            kept.append(details)

    # Sort kept by score ascending (worst first) for manual review
    kept.sort(key=lambda x: x["score"])

    print(f"\nResults:")
    print(f"  Kept:     {len(kept)} ({len(kept)/len(examples):.1%})")
    print(f"  Rejected: {len(rejected)} ({len(rejected)/len(examples):.1%})")
    print(f"    - no torso:  {sum(1 for r in rejected if r.get('reason') == 'no_torso')}")
    print(f"    - low score: {sum(1 for r in rejected if 'low_score' in r.get('reason',''))}")
    print(f"    - no seg:    {sum(1 for r in rejected if r.get('reason') == 'no segmentation.png')}")

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total": len(examples),
        "kept": len(kept),
        "rejected": len(rejected),
        "kept_examples": kept,       # sorted worst-first for review
        "rejected_examples": rejected,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print(f"  Review the {min(50, len(kept))} lowest-scoring kept examples for manual culling")

    if args.dry_run:
        print("\nDry run — no files copied.")
        return

    # Copy kept examples to output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for details in kept:
        src = input_dir / details["name"]
        dst = output_dir / details["name"]
        if not dst.exists():
            shutil.copytree(src, dst)

    print(f"\nCopied {len(kept)} examples to {output_dir}")


if __name__ == "__main__":
    main()
