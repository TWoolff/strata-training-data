"""Cluster character measurement profiles into proportion archetypes.

Reads ``mesh/measurements/measurement_profiles.json`` (produced by
``aggregate_measurements.py``), computes proportion ratio features, and
clusters characters into natural archetype groups using K-Means.

Auto-selects the optimal number of clusters via silhouette score and
assigns interpretable labels (e.g. "chibi", "realistic") based on
centroid proportions.

No Blender dependency — pure Python + scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLUSTERS_VERSION = "1.0"

# Minimum measured regions required to include a character in clustering.
MIN_MEASURED_REGIONS = 10

# K-Means search range for automatic cluster count selection.
MIN_K = 2
MAX_K = 10

# Minimum characters required for meaningful clustering.
MIN_CHARACTERS_FOR_CLUSTERING = 5

# Region groups for proportion ratio computation.
TORSO_REGIONS = ("chest", "spine", "hips")

# Regions used for arm length (upper_arm + lower_arm + hand), averaged L+R.
ARM_UPPER_REGIONS = ("upper_arm_l", "upper_arm_r")
ARM_LOWER_REGIONS = ("lower_arm_l", "lower_arm_r")
HAND_REGIONS = ("hand_l", "hand_r")

# Regions used for leg length (upper_leg + lower_leg + foot), averaged L+R.
LEG_UPPER_REGIONS = ("upper_leg_l", "upper_leg_r")
LEG_LOWER_REGIONS = ("lower_leg_l", "lower_leg_r")
FOOT_REGIONS = ("foot_l", "foot_r")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _safe_avg(values: list[float]) -> float | None:
    """Return the average of non-None values, or None if empty."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _get_dim(
    measurements: dict[str, dict[str, float]],
    region: str,
    dim: str,
) -> float | None:
    """Get a single dimension from a region, or None if missing/zero."""
    region_data = measurements.get(region)
    if region_data is None:
        return None
    val = region_data.get(dim, 0.0)
    return val if val > 0 else None


def _sum_dim(
    measurements: dict[str, dict[str, float]],
    regions: tuple[str, ...],
    dim: str,
) -> float | None:
    """Sum a dimension across multiple regions. None if any is missing."""
    total = 0.0
    for region in regions:
        val = _get_dim(measurements, region, dim)
        if val is None:
            return None
        total += val
    return total


def _avg_pair_dim(
    measurements: dict[str, dict[str, float]],
    left_right_regions: tuple[str, str],
    dim: str,
) -> float | None:
    """Average a dimension across a left/right pair."""
    vals = [_get_dim(measurements, r, dim) for r in left_right_regions]
    return _safe_avg(vals)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Compute numerator / denominator, or None if either is None or denom is zero."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def compute_proportion_features(
    measurements: dict[str, dict[str, float]],
) -> dict[str, float | None]:
    """Compute proportion ratio features from a character's measurements.

    Args:
        measurements: Per-region measurement dict with ``width``, ``depth``,
            ``height`` for each region.

    Returns:
        Dict of feature name → value (or None if not computable).
    """
    torso_h = _sum_dim(measurements, TORSO_REGIONS, "height")
    head_h = _get_dim(measurements, "head", "height")
    head_w = _get_dim(measurements, "head", "width")
    chest_w = _get_dim(measurements, "chest", "width")
    hips_w = _get_dim(measurements, "hips", "width")

    # Arm length: average of left and right (upper + lower + hand heights)
    arm_upper_h = _avg_pair_dim(measurements, ARM_UPPER_REGIONS, "height")
    arm_lower_h = _avg_pair_dim(measurements, ARM_LOWER_REGIONS, "height")
    hand_h = _avg_pair_dim(measurements, HAND_REGIONS, "height")
    arm_length = None
    if arm_upper_h is not None and arm_lower_h is not None and hand_h is not None:
        arm_length = arm_upper_h + arm_lower_h + hand_h

    # Leg length: average of left and right (upper + lower + foot heights)
    leg_upper_h = _avg_pair_dim(measurements, LEG_UPPER_REGIONS, "height")
    leg_lower_h = _avg_pair_dim(measurements, LEG_LOWER_REGIONS, "height")
    foot_h = _avg_pair_dim(measurements, FOOT_REGIONS, "height")
    leg_length = None
    if leg_upper_h is not None and leg_lower_h is not None and foot_h is not None:
        leg_length = leg_upper_h + leg_lower_h + foot_h

    # Shoulder width: shoulders + chest
    shoulder_l_w = _get_dim(measurements, "shoulder_l", "width")
    shoulder_r_w = _get_dim(measurements, "shoulder_r", "width")
    shoulder_width = None
    if shoulder_l_w is not None and shoulder_r_w is not None and chest_w is not None:
        shoulder_width = shoulder_l_w + shoulder_r_w + chest_w

    # Arm thickness: average width of upper and lower arm segments
    arm_upper_w = _avg_pair_dim(measurements, ARM_UPPER_REGIONS, "width")
    arm_lower_w = _avg_pair_dim(measurements, ARM_LOWER_REGIONS, "width")
    arm_thickness = _safe_avg([arm_upper_w, arm_lower_w])

    # Leg thickness: average width of upper and lower leg segments
    leg_upper_w = _avg_pair_dim(measurements, LEG_UPPER_REGIONS, "width")
    leg_lower_w = _avg_pair_dim(measurements, LEG_LOWER_REGIONS, "width")
    leg_thickness = _safe_avg([leg_upper_w, leg_lower_w])

    # Arm length without hands (for hand-to-arm ratio)
    arm_without_hand = None
    if arm_upper_h is not None and arm_lower_h is not None:
        arm_without_hand = arm_upper_h + arm_lower_h

    # Leg length without feet (for foot-to-leg ratio)
    leg_without_foot = None
    if leg_upper_h is not None and leg_lower_h is not None:
        leg_without_foot = leg_upper_h + leg_lower_h

    features: dict[str, float | None] = {}

    # 1. Head-to-body height ratio
    features["head_to_body_height"] = _safe_ratio(head_h, torso_h)

    # 2. Arm length relative to torso
    features["arm_length_to_torso"] = _safe_ratio(arm_length, torso_h)

    # 3. Leg length relative to torso
    features["leg_length_to_torso"] = _safe_ratio(leg_length, torso_h)

    # 4. Shoulder width relative to hip width
    features["shoulder_to_hip_width"] = _safe_ratio(shoulder_width, hips_w)

    # 5. Arm thickness relative to chest width
    features["arm_thickness_ratio"] = _safe_ratio(arm_thickness, chest_w)

    # 6. Leg thickness relative to hip width
    features["leg_thickness_ratio"] = _safe_ratio(leg_thickness, hips_w)

    # 7. Head width relative to shoulder width
    features["head_width_to_shoulder"] = _safe_ratio(head_w, shoulder_width)

    # 8. Hand-to-arm length ratio
    features["hand_to_arm_ratio"] = _safe_ratio(hand_h, arm_without_hand)

    # 9. Foot-to-leg length ratio
    features["foot_to_leg_ratio"] = _safe_ratio(foot_h, leg_without_foot)

    return features


# ---------------------------------------------------------------------------
# Feature names (canonical order)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "head_to_body_height",
    "arm_length_to_torso",
    "leg_length_to_torso",
    "shoulder_to_hip_width",
    "arm_thickness_ratio",
    "leg_thickness_ratio",
    "head_width_to_shoulder",
    "hand_to_arm_ratio",
    "foot_to_leg_ratio",
]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def build_feature_matrix(
    profiles: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a feature matrix from measurement profiles.

    Args:
        profiles: The parsed ``measurement_profiles.json`` data.

    Returns:
        Tuple of:
        - feature_matrix: (n_characters, n_features) array with NaN for missing
        - character_ids: list of character IDs (same order as rows)
        - feature_names: list of feature names (same order as columns)
    """
    characters = profiles.get("characters", [])
    character_ids: list[str] = []
    rows: list[list[float]] = []

    for char in characters:
        if char.get("measured_regions", 0) < MIN_MEASURED_REGIONS:
            logger.debug(
                "Skipping %s: only %d measured regions (min %d)",
                char.get("character_id", "unknown"),
                char.get("measured_regions", 0),
                MIN_MEASURED_REGIONS,
            )
            continue

        features = compute_proportion_features(char.get("measurements", {}))
        row = [
            val if (val := features.get(name)) is not None else float("nan")
            for name in FEATURE_NAMES
        ]
        rows.append(row)
        character_ids.append(char["character_id"])

    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), [], list(FEATURE_NAMES)

    return np.array(rows), character_ids, list(FEATURE_NAMES)


def _impute_nans(matrix: np.ndarray) -> np.ndarray:
    """Replace NaN values in each column with the column median."""
    imputed = matrix.copy()
    medians = np.nanmedian(matrix, axis=0)
    for col in range(matrix.shape[1]):
        mask = np.isnan(imputed[:, col])
        imputed[mask, col] = medians[col]
    return imputed


def impute_and_scale(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impute NaN values with column medians and standard-scale features.

    Args:
        matrix: (n, d) feature matrix, may contain NaN values.

    Returns:
        Tuple of:
        - scaled: (n, d) scaled feature matrix with no NaN
        - medians: (d,) median values used for imputation
        - std: (d,) standard deviations used for scaling (0 replaced with 1)
    """
    medians = np.nanmedian(matrix, axis=0)
    imputed = _impute_nans(matrix)

    # Standard scaling
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant features
    scaled = (imputed - means) / stds

    return scaled, medians, stds


def find_optimal_k(
    scaled: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
) -> tuple[int, dict[int, float]]:
    """Find the optimal number of clusters using silhouette score.

    Args:
        scaled: (n, d) scaled feature matrix.
        min_k: Minimum number of clusters to test.
        max_k: Maximum number of clusters to test.

    Returns:
        Tuple of (optimal_k, scores_dict mapping k -> silhouette_score).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = scaled.shape[0]
    actual_max_k = min(max_k, n_samples - 1)

    if actual_max_k < min_k:
        return min_k, {}

    scores: dict[int, float] = {}
    for k in range(min_k, actual_max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(scaled)
        n_unique_labels = len(set(labels))
        if n_unique_labels < 2:
            logger.debug("k=%d produced only %d distinct cluster(s), skipping", k, n_unique_labels)
            continue
        score = silhouette_score(scaled, labels)
        scores[k] = round(float(score), 4)
        logger.debug("k=%d silhouette=%.4f", k, score)

    if not scores:
        logger.warning("No valid k found — all points may be identical. Using k=%d.", min_k)
        return min_k, {}

    optimal_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info("Optimal k=%d (silhouette=%.4f)", optimal_k, scores[optimal_k])
    return optimal_k, scores


def assign_cluster_label(centroid: dict[str, float]) -> str:
    """Assign an interpretable label to a cluster based on centroid proportions.

    Args:
        centroid: Dict of feature name → centroid value (proportion ratios).

    Returns:
        A human-readable label like "chibi", "realistic", etc.
    """
    head_ratio = centroid.get("head_to_body_height", 0)
    leg_ratio = centroid.get("leg_length_to_torso", 0)
    shoulder_ratio = centroid.get("shoulder_to_hip_width", 0)
    arm_thick = centroid.get("arm_thickness_ratio", 0)
    leg_thick = centroid.get("leg_thickness_ratio", 0)

    # Chibi: very large head relative to body
    if head_ratio > 0.5:
        return "chibi"

    # Muscular/stocky: wide shoulders + thick limbs
    if shoulder_ratio > 3.5 and (arm_thick > 0.35 or leg_thick > 0.35):
        return "muscular"

    # Tall/thin: long legs + thin limbs
    if leg_ratio > 1.6 and arm_thick < 0.25 and leg_thick < 0.25:
        return "tall_thin"

    # Stylized: large head but not chibi-level
    if head_ratio > 0.35:
        return "stylized"

    # Default: realistic proportions
    return "realistic"


def cluster_profiles(
    profiles: dict[str, Any],
    *,
    k_override: int | None = None,
) -> dict[str, Any]:
    """Cluster character measurement profiles into proportion archetypes.

    Args:
        profiles: The parsed ``measurement_profiles.json`` data.
        k_override: Force a specific number of clusters instead of
            auto-selecting via silhouette score.

    Returns:
        Archetype cluster results dict ready for JSON serialization.
    """
    from sklearn.cluster import KMeans

    matrix, character_ids, feature_names = build_feature_matrix(profiles)
    n_characters = matrix.shape[0]
    n_total = len(profiles.get("characters", []))
    n_skipped = n_total - n_characters

    if n_characters == 0:
        logger.warning("No characters with sufficient measurements for clustering.")
        return _empty_result(feature_names, characters_skipped=n_skipped)

    if n_characters < MIN_CHARACTERS_FOR_CLUSTERING:
        logger.warning(
            "Only %d characters — too few for meaningful clustering. "
            "Assigning all to a single cluster.",
            n_characters,
        )
        return _single_cluster_result(
            matrix, character_ids, feature_names, characters_skipped=n_skipped
        )

    scaled, _medians, _stds = impute_and_scale(matrix)

    # Determine k
    silhouette_scores: dict[int, float] = {}
    if k_override is not None:
        optimal_k = k_override
    else:
        optimal_k, silhouette_scores = find_optimal_k(scaled)

    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(scaled)

    # Compute silhouette score for the final result
    final_silhouette: float | None = None
    n_unique_labels = len(set(labels))
    if n_unique_labels > 1 and n_characters > n_unique_labels:
        from sklearn.metrics import silhouette_score

        final_silhouette = round(float(silhouette_score(scaled, labels)), 4)

    imputed = _impute_nans(matrix)

    # Build cluster results
    clusters: list[dict[str, Any]] = []
    used_labels: set[str] = set()

    for cluster_id in range(optimal_k):
        mask = labels == cluster_id
        cluster_char_ids = [
            cid for cid, in_cluster in zip(character_ids, mask, strict=True) if in_cluster
        ]
        cluster_features = imputed[mask]

        # Centroid in original (unscaled) feature space
        centroid_values = cluster_features.mean(axis=0)
        centroid = {
            name: round(float(val), 4)
            for name, val in zip(feature_names, centroid_values, strict=True)
        }

        # Auto-label
        label = assign_cluster_label(centroid)
        if label in used_labels:
            label = f"{label}_{cluster_id}"
        used_labels.add(label)

        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "character_count": len(cluster_char_ids),
                "centroid": centroid,
                "characters": sorted(cluster_char_ids),
            }
        )

    # Sort clusters by character_count descending for readability
    clusters.sort(key=lambda c: c["character_count"], reverse=True)
    # Re-assign cluster_ids after sorting
    for i, cluster in enumerate(clusters):
        cluster["cluster_id"] = i

    result: dict[str, Any] = {
        "version": CLUSTERS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "kmeans",
        "optimal_k": optimal_k,
        "silhouette_score": final_silhouette,
        "silhouette_scores_by_k": silhouette_scores if silhouette_scores else None,
        "characters_analyzed": n_characters,
        "characters_skipped": n_skipped,
        "features_used": feature_names,
        "clusters": clusters,
    }

    return result


def _empty_result(
    feature_names: list[str],
    *,
    characters_skipped: int = 0,
) -> dict[str, Any]:
    """Return an empty cluster result when no characters are available."""
    return {
        "version": CLUSTERS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "kmeans",
        "optimal_k": 0,
        "silhouette_score": None,
        "silhouette_scores_by_k": None,
        "characters_analyzed": 0,
        "characters_skipped": characters_skipped,
        "features_used": feature_names,
        "clusters": [],
    }


def _single_cluster_result(
    matrix: np.ndarray,
    character_ids: list[str],
    feature_names: list[str],
    *,
    characters_skipped: int = 0,
) -> dict[str, Any]:
    """Return a single-cluster result for when there are too few characters."""
    imputed = _impute_nans(matrix)
    centroid_values = imputed.mean(axis=0)
    centroid = {
        name: round(float(val), 4) for name, val in zip(feature_names, centroid_values, strict=True)
    }

    return {
        "version": CLUSTERS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "single_cluster",
        "optimal_k": 1,
        "silhouette_score": None,
        "silhouette_scores_by_k": None,
        "characters_analyzed": len(character_ids),
        "characters_skipped": characters_skipped,
        "features_used": feature_names,
        "clusters": [
            {
                "cluster_id": 0,
                "label": assign_cluster_label(centroid),
                "character_count": len(character_ids),
                "centroid": centroid,
                "characters": sorted(character_ids),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_visualization(
    profiles: dict[str, Any],
    clusters_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a 2D PCA scatter plot of clusters.

    Args:
        profiles: The parsed ``measurement_profiles.json`` data.
        clusters_result: The clustering result from ``cluster_profiles``.
        output_path: Path to save the PNG visualization.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    matrix, character_ids, _feature_names = build_feature_matrix(profiles)
    if matrix.shape[0] < 2:
        logger.warning("Too few characters for visualization.")
        return

    scaled, _medians, _stds = impute_and_scale(matrix)

    # PCA to 2D
    n_components = min(2, scaled.shape[1], scaled.shape[0])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)

    # Build character → cluster_id mapping
    char_to_cluster: dict[str, int] = {}
    for cluster in clusters_result.get("clusters", []):
        for cid in cluster.get("characters", []):
            char_to_cluster[cid] = cluster["cluster_id"]

    # Assign cluster labels to each point
    cluster_labels = [char_to_cluster.get(cid, -1) for cid in character_ids]
    unique_clusters = sorted(set(cluster_labels))

    # Cluster label names
    cluster_names: dict[int, str] = {}
    for cluster in clusters_result.get("clusters", []):
        cluster_names[cluster["cluster_id"]] = cluster["label"]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_clusters), 1)))

    for i, cid in enumerate(unique_clusters):
        mask = [cl == cid for cl in cluster_labels]
        points = coords[mask]
        name = cluster_names.get(cid, f"cluster_{cid}")
        ax.scatter(
            points[:, 0],
            points[:, 1] if n_components > 1 else np.zeros(points.shape[0]),
            c=[colors[i]],
            label=f"{name} (n={points.shape[0]})",
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    if n_components > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Character Proportion Archetypes (PCA projection)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved visualization to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path("mesh/measurements/measurement_profiles.json")
DEFAULT_OUTPUT_PATH = Path("mesh/measurements/archetype_clusters.json")
DEFAULT_VIZ_PATH = Path("mesh/measurements/archetype_clusters.png")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for proportion clustering."""
    parser = argparse.ArgumentParser(
        description="Cluster character measurement profiles into proportion archetypes",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input measurement profiles JSON (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output archetype clusters JSON (default: %(default)s)",
    )
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=None,
        help="Force a specific number of clusters (default: auto-select)",
    )
    parser.add_argument(
        "--viz",
        type=Path,
        default=DEFAULT_VIZ_PATH,
        help="Output visualization PNG path (default: %(default)s)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generating the visualization plot",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.is_file():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    # Load profiles
    profiles = json.loads(args.input.read_text(encoding="utf-8"))
    logger.info(
        "Loaded %d character profiles from %s",
        len(profiles.get("characters", [])),
        args.input,
    )

    # Run clustering
    result = cluster_profiles(profiles, k_override=args.clusters)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote archetype clusters to %s", args.output)

    # Summary
    print(f"Analyzed {result['characters_analyzed']} characters.")
    print(f"Optimal k={result['optimal_k']} clusters:")
    for cluster in result["clusters"]:
        print(
            f"  [{cluster['cluster_id']}] {cluster['label']}: "
            f"{cluster['character_count']} characters"
        )

    if result.get("silhouette_score") is not None:
        print(f"Silhouette score: {result['silhouette_score']}")

    # Visualization
    if not args.no_viz and result["optimal_k"] > 0:
        save_visualization(profiles, result, args.viz)

    print("Done.")


if __name__ == "__main__":
    main()
