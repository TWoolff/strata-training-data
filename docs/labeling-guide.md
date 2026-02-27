# Labeling Guide

How to annotate action types, quality, and animation principles for mocap data.

## Action Type Labels

Categorize each mocap clip by its primary action:

| Category | Examples |
|----------|----------|
| locomotion | walk, run, jog, sprint, crawl |
| jump | jump, hop, leap, vault |
| gesture | wave, point, nod, shrug |
| combat | punch, kick, block, dodge |
| dance | ballet, hip-hop, freestyle |
| sport | throw, catch, swing, kick |
| daily | sit, stand, reach, pick up |

## Quality Annotations

Rate each clip on a 1–5 scale:
- **5**: Professional motion capture, clean data
- **4**: Good quality, minor noise
- **3**: Usable, some artifacts
- **2**: Noisy, needs cleanup
- **1**: Poor quality, reference only

## Strata Compatibility

The `strata_compatible` column in `animation/labels/cmu_action_labels.csv` indicates whether a BVH clip maps cleanly to Strata's 19-bone skeleton.

### Criteria

- **yes** — Full-body clip where all significant motion occurs on bones that map to Strata's 19-bone skeleton (hips through feet, spine through head, arms through hands). Finger, toe, and facial bones may be present in the skeleton but must have negligible rotation data.
- **no** — Clip that depends heavily on bones outside the Strata skeleton: finger articulation (piano playing, sign language, detailed hand gestures), facial animation, or other non-standard bones with significant rotation data.

### What counts as "significant motion"

A bone has significant motion if any rotation axis changes by more than 1 degree across frames (compared to the first frame). This threshold is configurable via the `--threshold` flag in the compatibility checker.

### Automated checking

Use the compatibility checker to auto-evaluate BVH files:

```bash
python -m animation.scripts.bvh_to_strata check-compat path/to/*.bvh
python -m animation.scripts.bvh_to_strata check-compat --threshold 2.0 clip.bvh
```

### Common compatible categories

Locomotion (walk, run, jog), jumping, dance, martial arts, balance, turning, crouching, idle poses, falling — these typically use only Strata-mapped bones.

### Common incompatible categories

Sign language, instrument playing (piano, guitar), detailed hand gestures (wave, point, beckon), typing, facial animation — these depend on finger or facial bones not in the Strata skeleton.
