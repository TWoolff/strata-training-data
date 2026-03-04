# Issue #188: Check LinkTo-Anime public availability (80 VRoid+Mixamo chars, 29K frames)

## Understanding
- Research issue to confirm whether LinkTo-Anime dataset is publicly available and its license
- The checklist already marked it as SKIPPED due to CC-BY-NC license
- Need to confirm that status and close the issue

## Findings

### Public Availability
- **Dataset IS publicly released** on HuggingFace: https://huggingface.co/datasets/LecterF/LinkTo-Anime
- Gated access (requires HuggingFace login + agreement to share contact info)
- Paper: arXiv 2506.02733, revised July 29, 2025
- Size: 10K–100K items (29,270 frames per paper)
- Monthly downloads: ~12

### License Confirmation
- **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0)
- Confirmed on HuggingFace dataset page
- Paper itself doesn't explicitly state license, but HuggingFace metadata is authoritative
- **This confirms the dataset is EXCLUDED from commercial training per Strata licensing policy**

### Dataset Contents (for reference)
- 80 VRoid characters with Mixamo animations
- 395 video clips, 29,270 total frames
- Forward/backward optical flow, occlusion masks, Mixamo skeleton data
- Line-art + colored render pairs
- Curators: XIAOYIFENG, KAIFENG ZOU; Funder: LINKTO

## Approach
1. Update checklist to mark investigation items as done, confirm SKIPPED status
2. Update `docs/preprocessed-datasets.md` to reflect confirmed license and HuggingFace URL
3. No adapter registration or download needed

## Files to Modify
- `.claude/prd/strata-training-data-checklist.md` — mark PP-7 items as checked, confirm SKIPPED
- `docs/preprocessed-datasets.md` — update LinkTo-Anime entry with confirmed license + URL

## Risks & Edge Cases
- None — straightforward documentation update

## Open Questions
- None — license confirmed as CC-BY-NC-4.0, permanently skipped

## Implementation Notes
- License confirmed as CC-BY-NC-4.0 on HuggingFace dataset page
- Dataset IS publicly available (gated) at https://huggingface.co/datasets/LecterF/LinkTo-Anime
- Updated checklist: PP-7 action items marked done, status changed to PERMANENTLY SKIPPED
- Updated checklist: gathering timeline item marked done, legal checklist item marked done
- Updated checklist: adapter table entry changed to SKIPPED
- Updated docs/preprocessed-datasets.md: summary table, full entry, risk table, adapter cross-reference
- No code changes needed — this was purely a research + documentation task
