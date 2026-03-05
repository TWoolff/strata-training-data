# Issue #214: Diversify training data — western, male, body-type-diverse sources

## Understanding
Training data is 75% anime-style female characters. Need to add diverse sources across art styles, genders, and body types. This is a large issue with 7 steps spanning adapter code, Blender rendering, data download, and config updates.

## What Can Be Done in Code Now

### Step 1: UniRig Blender rendering pass
- Adapter exists (`ingest/unirig_adapter.py`) and data is on external HD (16,641 NPZ files)
- The adapter already has `_import_npz_to_blender()`, `_render_segmentation()`, `_render_color()`
- Currently renders front + back only (`RENDER_ANGLES = [0, 180]`)
- Extend to all 5 standard angles: 0, 45, 90, 135, 180
- The adapter is code-complete — just needs to be run against the data

### Step 2: HumanRig multi-angle rendering
- `humanrig_blender_renderer.py` already exists and renders three_quarter, side, back angles
- `humanrig_adapter.py` processes front views from pre-rendered `front.png`
- **Blocker**: HumanRig data was deleted from external HD (per MEMORY.md) — needs re-download
- Code changes: update `ANGLE_CONFIGS` to include `three_quarter_back` (135°)
- Currently missing: `("three_quarter_back", 135)` in humanrig_adapter.py's ANGLE_CONFIGS

### Step 3: Danbooru diverse adapter
- New adapter `ingest/danbooru_diverse_adapter.py` mirroring `fbanimehq_adapter.py`
- Tag-filtered download targeting male, dark_skin, muscular, western fantasy chars
- RTMPose joint enrichment (same pipeline as fbanimehq)
- This is the highest-impact new code: ~50-100K diverse examples

### Step 4: Training config updates
- Add UniRig dataset entries to segmentation and joints configs
- Add Danbooru diverse entries to joints configs
- Update example counts and estimated training times

## Approach

Focus on implementable code changes:
1. **Extend UniRig adapter** — add all 5 angles, verify it runs
2. **Extend HumanRig adapter** — add three_quarter_back angle
3. **Create Danbooru diverse adapter** — new file mirroring fbanimehq pattern
4. **Update training configs** — add new dataset entries

Skip for now (requires manual action or external downloads):
- Downloading more Mixamo characters (manual browser action)
- Sketchfab API scraping (needs API key + significant new code)
- Anita Dataset download (needs investigation)
- OpenGameArt filtering (needs investigation)
- Actual data processing runs (user will trigger these)

## Files to Modify

1. `ingest/unirig_adapter.py` — extend RENDER_ANGLES to all 5 standard angles
2. `ingest/humanrig_adapter.py` — add three_quarter_back (135°) to ANGLE_CONFIGS
3. `ingest/humanrig_blender_renderer.py` — add three_quarter_back to default angles
4. `ingest/danbooru_diverse_adapter.py` — NEW FILE, tag-filtered Danbooru download + conversion
5. `training/configs/segmentation_a100_lean.yaml` — add unirig dataset
6. `training/configs/segmentation_a100.yaml` — add unirig dataset
7. `training/configs/joints_a100_lean.yaml` — add unirig + danbooru_diverse
8. `training/configs/joints_a100.yaml` — add unirig + danbooru_diverse
9. `training/configs/segmentation.yaml` — add unirig dataset
10. `training/configs/joints.yaml` — add unirig + danbooru_diverse

## Risks & Edge Cases

- UniRig humanoid filtering may be too aggressive (60% coverage threshold) — keep as-is for now
- Danbooru tag filtering quality varies — need minimum score threshold
- HumanRig CC-BY-NC license needs verification for commercial training
- RTMPose joint quality on non-anime characters (western art) untested

## Open Questions

- Should Danbooru adapter include RTMPose enrichment inline or as separate step?
  Decision: separate step (matches fbanimehq pattern — adapter downloads, enrichment is post-processing)
- What Danbooru minimum score threshold? Decision: score >= 10 (filters low-quality posts)
