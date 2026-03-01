# Issue #137: AnimeRun temporal correspondence adapter (SegMatching + occlusion)

## Understanding
- AnimeRun provides SegMatching, UnmatchedForward, and UnmatchedBackward data
- SegMatching: segment correspondence across consecutive frames (which segment in frame_t maps to which in frame_t+1)
- UnmatchedForward: pixels visible in frame_t but occluded in frame_t+1 (disocclusion masks)
- UnmatchedBackward: pixels visible in frame_t+1 but not in frame_t (occlusion masks)
- This is a new feature: create `ingest/animerun_correspondence_adapter.py`
- Data is frame-pair oriented (like the flow adapter), not single-frame (like the segment adapter)

## Approach
- Model after the flow adapter (frame-pair pattern) since SegMatching and occlusion data pairs consecutive frames
- Use a `CorrespondencePair` dataclass with paths for frame_t, frame_t1, seg_matching, unmatched_forward, unmatched_backward
- Discovery scans SegMatching directory as primary, checks for UnmatchedForward/Backward and Frame_Anime
- SegMatching data format: likely .npy or .png — support both like other adapters
- Occlusion masks: likely .png binary masks — load and resize with nearest-neighbor (like instance masks)
- Generate occlusion overlay visualization for QA
- Output format matches issue spec exactly

## Files to Modify
1. **NEW: `ingest/animerun_correspondence_adapter.py`** — Main adapter module
   - Constants, dataclasses, discovery, loading, conversion, save, entry points
2. **EDIT: `run_ingest.py`** — Register `animerun_correspondence` adapter
   - Add choice, add `_run_animerun_correspondence()`, add to `_ADAPTERS` dict
3. **NEW: `tests/test_animerun_correspondence_adapter.py`** — Test suite

## Risks & Edge Cases
- Unknown SegMatching data format — support both .npy and .png, log warnings for unexpected shapes
- Some scenes may have SegMatching but no occlusion data (or vice versa) — handle gracefully with has_* flags
- Unmatched masks may be binary (0/255) or multi-valued — normalize to binary uint8 for output
- Last frame in a sequence has no t+1 pair — skip like the flow adapter does
- Some scenes may not have Frame_Anime data — skip those scenes

## Open Questions
- Exact SegMatching format (npy shape, dtype) — will handle generically and log shape info
- Whether occlusion masks are binary or have instance IDs — treat as binary masks for now

## Implementation Notes
- Modeled closely after `animerun_flow_adapter.py` (frame-pair pattern)
- `CorrespondencePair` dataclass with all three data paths optional (seg_matching, unmatched_fwd, unmatched_bwd)
- Discovery uses SegMatching as primary directory; occlusion dirs are optional
- `load_correspondence_map()` returns raw numpy array (shape/dtype preserved for flexibility)
- `load_occlusion_mask()` normalizes to binary 0/255 uint8, handles 3D→2D reduction
- Occlusion overlay: red=forward occlusion, blue=backward occlusion, alpha-blended over frame_t
- `_DATA_EXTENSIONS` expressed as `_IMAGE_EXTENSIONS | {".npy"}` for clarity
- 40 tests covering: data loading (png/npy), resizing, overlay generation, metadata, discovery, pair conversion, scene conversion, directory conversion, edge cases
- Registered as `--adapter animerun_correspondence` in `run_ingest.py`
