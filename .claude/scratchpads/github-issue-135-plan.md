# Issue #135: AnimeRun optical flow adapter for motion/interpolation training

## Understanding
- AnimeRun provides forward/backward optical flow data (~20 GB) alongside anime frames
- Flow data is in `AnimeRun_v2/{train,test}/Flow/{scene}/forward/` and `backward/`
- Corresponding anime frames are in `Frame_Anime/{scene}/original/`
- Need an adapter that pairs flow files with frames and outputs training examples
- This is a new feature — a new adapter module alongside the existing contour adapter

## Approach
- Follow the exact same patterns as `animerun_contour_adapter.py` (same dataset, different data type)
- Reuse the `_find_root()` logic and scene discovery pattern from the contour adapter
- Borrow flow I/O utilities (read_flo_file, load_flow) from `linkto_adapter.py`
- Add HSV color wheel flow visualization for QA (standard optical flow viz technique)
- Output format: frame_t + frame_t1 + flow_forward + flow_backward + flow_viz + metadata
- Register as `animerun_flow` adapter in run_ingest.py (distinct from existing `animerun` contour adapter)

## Files to Modify
1. **NEW: `ingest/animerun_flow_adapter.py`** — Main adapter module
   - FlowPair dataclass (frame_t, frame_t1, flow_fwd, flow_bwd paths)
   - Scene/pair discovery matching flow files to consecutive frame pairs
   - Flow visualization (HSV color wheel encoding)
   - Resolution normalization for flow vectors when resizing frames
   - convert_pair → convert_scene → convert_directory pattern
2. **MODIFY: `run_ingest.py`** — Register `animerun_flow` adapter
   - Add `_run_animerun_flow()` dispatch function
   - Add to `_ADAPTERS` dict and `choices` list
3. **NEW: `tests/test_animerun_flow_adapter.py`** — Tests
   - Test flow visualization, discovery, conversion, metadata
   - Create fake directory structures with synthetic flow arrays

## Key Design Decisions
- Flow pairs are between consecutive frames (t, t+1), so discovery must match flow files to frame pairs
- Flow vectors must be scaled proportionally when frames are resized (multiply by scale factor)
- HSV flow visualization: angle → hue, magnitude → saturation/value (standard Middlebury convention)
- Output uses `flow_forward.npy` and `flow_backward.npy` (not `flow_fwd`/`flow_bwd` to match issue spec)
- Each flow file produces one training example pairing frame_t and frame_t+1

## Risks & Edge Cases
- Flow files may be .npy or .flo — handle both formats (reuse linkto logic)
- Some frames may lack corresponding flow (first/last frame in sequence) — skip gracefully
- Frames may have different resolution than flow arrays — need proportional scaling
- Nested extraction (AnimeRun_v2 inside the target dir) — reuse _find_root pattern
- Scene directories may not have both forward AND backward flow — handle partial data

## Open Questions
- None — the issue spec and existing adapters provide clear patterns to follow
