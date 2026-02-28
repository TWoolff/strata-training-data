# Issue #81: Cluster body measurement profiles into template archetypes

## Understanding
- Read `mesh/measurements/measurement_profiles.json` (produced by `aggregate_measurements.py`)
- Cluster characters by **proportion ratios** (not absolute size) — e.g., head-to-body height ratio, arm length relative to torso, leg length relative to torso, shoulder/hip width ratio, limb thickness ratios
- Auto-determine optimal cluster count (k) using silhouette score or elbow method
- Output cluster centroids as template archetype definitions
- Optional: visualization scatter plot (matplotlib)
- Pure Python — no Blender dependency
- Type: new feature

## Approach

### Feature Engineering (Proportion Ratios)
The key insight is clustering on **proportions** not raw measurements. Characters are already height-normalized (`TARGET_CHARACTER_HEIGHT = 2.0`), but proportions capture style better than absolute dimensions.

**Derived ratio features:**
1. `head_to_body_height` — head.height / total_body_height (chest + spine + hips)
2. `arm_length_to_torso` — (upper_arm + lower_arm + hand).height / (chest + spine + hips).height
3. `leg_length_to_torso` — (upper_leg + lower_leg + foot).height / (chest + spine + hips).height
4. `shoulder_to_hip_width` — (shoulder_l.width + shoulder_r.width + chest.width) / hips.width
5. `arm_thickness` — avg(upper_arm.width, lower_arm.width) / chest.width
6. `leg_thickness` — avg(upper_leg.width, lower_leg.width) / hips.width
7. `head_width_to_shoulder` — head.width / (shoulder_l.width + shoulder_r.width + chest.width)
8. `hand_to_arm` — hand.height / (upper_arm + lower_arm).height
9. `foot_to_leg` — foot.height / (upper_leg + lower_leg).height

Use height as the primary dimension for lengths (Z-axis), width for breadths (X-axis).

### Clustering Strategy
- Use scikit-learn KMeans
- Auto-select k: test k=2..10, pick by silhouette score (highest)
- Fallback: if fewer than 10 characters, use k=min(n_characters // 2, 3)
- StandardScaler for feature normalization before clustering
- PCA for visualization (2D scatter plot)

### Handling Missing Regions
Some characters may not have all 19 regions measured. Strategy:
- Require minimum measured regions (e.g., 10) to include a character
- For ratio computation, skip ratios where denominator regions are missing
- Impute missing ratio features with the median of available characters

### Output Schema
`mesh/measurements/archetype_clusters.json`:
```json
{
  "version": "1.0",
  "generated_at": "...",
  "method": "kmeans",
  "optimal_k": 5,
  "silhouette_score": 0.45,
  "features_used": ["head_to_body_height", "arm_length_to_torso", ...],
  "clusters": [
    {
      "cluster_id": 0,
      "label": "chibi",  // auto-assigned from proportion rules
      "character_count": 42,
      "centroid": {"head_to_body_height": 0.4, ...},
      "characters": ["mixamo_ybot", ...]
    }
  ]
}
```

### Auto-labeling Clusters
Based on centroid proportions, assign interpretable labels:
- "chibi" — head_to_body_height > 0.35
- "realistic" — proportions close to human average
- "stylized" — significant deviation from realistic but not chibi
- "tall_thin" — high leg_length_to_torso, low thickness ratios
- "muscular" — high shoulder_to_hip_width, high thickness ratios
- Fallback: "archetype_N"

## Files to Modify
- **NEW** `mesh/scripts/proportion_clusterer.py` — main clustering script
- **NEW** `tests/test_proportion_clusterer.py` — unit tests
- **EDIT** `requirements.txt` — add scikit-learn, matplotlib

## Risks & Edge Cases
- Very few characters (< 5): clustering is meaningless → return all as one cluster with warning
- All characters identical proportions: k=1 is optimal → handle gracefully
- Missing regions producing NaN ratios → need imputation or filtering
- Characters with 0-height regions (division by zero) → guard against in ratio computation
- Left/right symmetry: average L+R values to avoid handedness affecting clusters

## Open Questions
- None — requirements are clear from the issue and PRD

## Implementation Notes

### What was implemented
- `mesh/scripts/proportion_clusterer.py` (757 lines) — full clustering pipeline
- `tests/test_proportion_clusterer.py` (465 lines) — 29 tests covering all components
- `requirements.txt` — added `scikit-learn>=1.3` and `matplotlib>=3.7`

### Design decisions
- **9 proportion ratio features** derived from 19 body regions, all computed as ratios to avoid sensitivity to absolute scale
- **Left/right averaging** — all paired regions (arms, legs, shoulders) are averaged to prevent handedness artifacts
- **Silhouette score** for optimal k selection (k=2..10 range, capped at n_samples-1)
- **Auto-labeling** uses threshold-based rules on centroid proportions (chibi > 0.5 head ratio, muscular > 3.5 shoulder ratio + thick limbs, etc.)
- **Clusters sorted by size** (descending) with sequential IDs re-assigned after sorting
- **Duplicate label disambiguation** — if two clusters get the same auto-label, suffix with cluster_id

### Edge cases handled
- Fewer than 5 characters → single-cluster fallback with `method: "single_cluster"`
- Zero characters passing filter → empty result
- All identical points → KMeans may produce fewer clusters than requested; silhouette_score is skipped when only 1 distinct label exists
- Missing regions → NaN in feature matrix, imputed with column median before clustering
- Zero-value denominators → `_safe_ratio` returns None, converted to NaN

### Follow-up
- No follow-up needed — script is self-contained and ready for use once `measurement_profiles.json` is populated
