# Issue #198: Run RTMPose enrichment on anime_seg and anime_instance_seg (123K joints)

## Understanding
- Two large anime datasets in the bucket lack joint annotations (joints.json)
- anime_instance_seg: 98,428 examples (90,944 train + 7,484 val) — locally available
- anime_seg: ~24,800 examples — output dir empty locally, source data on external HD
- All infrastructure exists: run_enrich.py, pose_estimator.py, ONNX models
- This is an execution task — no code changes needed

## Approach
1. Run `run_enrich.py --only_missing` on anime_instance_seg (local data ready)
2. Upload anime_instance_seg joints.json files to Hetzner bucket
3. For anime_seg: re-run ingest with `--enrich` from external HD source data (Option B from issue)
4. Upload anime_seg joints.json files to bucket
5. Update checklist

CPU enrichment at ~2-3 img/sec → ~98K images ≈ 9-14 hours for anime_instance_seg.

## Files to Modify
- No code changes needed
- Checklist update: `docs/preprocessed-datasets.md` or equivalent with joint coverage numbers
- Scratchpad: this file (implementation notes at end)

## Risks & Edge Cases
- Long runtime (~10+ hours for anime_instance_seg on CPU)
- External HD may not be mounted for anime_seg
- Some images may fail pose estimation (no person detected) — tracked by failed count
- anime_seg adapter may need --enrich flag support verification

## Open Questions
- Is GPU (CUDA) available? → No, using CPU. Speed is ~8 img/sec (~500/min)
- anime_seg: is re-ingest with --enrich the preferred approach, or download from bucket? → Both: download from bucket + re-ingest missing from HD

## Implementation Notes

### Actual Dataset Numbers
- anime_instance_seg: 98,428 examples (90,944 train + 7,484 val), all local
- anime_seg: 15,081 examples in bucket (10,081 v1 + 5,000 v2), output dir was empty locally
- anime_seg source on external HD: ~11,802 fg images across 6 dirs

### Execution Steps
1. Started anime_instance_seg enrichment via nohup (PID 95947)
   - First run processed 4,980 before getting killed (10-min tool timeout)
   - Restarted with nohup, using --only_missing to skip completed
   - Speed: ~500 images/min, 0 failures
   - Estimated total time: ~3 hours

2. Downloaded all anime_seg from Hetzner bucket to output/anime_seg/
   - PID 96045, ~15K examples, download speed ~250 KiB/s

3. Created `scripts/enrich_and_upload.sh` auto-chain script (PID 98462):
   - Waits for anime_seg download to complete
   - Re-ingests missing v1 images from external HD (--only_new)
   - Enriches all anime_seg with RTMPose
   - Waits for anime_instance_seg enrichment
   - Auto-restarts if enrichment was incomplete
   - Uploads joints.json + metadata.json for both datasets

### Monitoring
- Chain script log: `/tmp/issue198/chain.log`
- anime_instance_seg enrichment log: `/tmp/enrich_anime_instance_seg.log`
- anime_seg download log: `/tmp/download_anime_seg.log`
- Check progress: `tail -5 /tmp/issue198/chain.log`
