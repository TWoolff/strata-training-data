# Issue #235: Annotator image upload + export approved annotations

## Understanding
Two Python CLI scripts that bridge the annotation web app with the training pipeline:
1. **upload-images.py** — scan preprocessed dataset dirs, upload image+seg to Hetzner bucket, insert rows into Turso DB
2. **export-approved.py** — query Turso for approved annotations, download images, save corrected masks in Strata training format

Type: new feature

## Approach
- Both scripts are standalone Python CLIs in `annotator/scripts/`
- Use `rclone` for bucket uploads/downloads (already configured)
- Use Turso HTTP API (`libsql` Python package or raw HTTP) for DB operations
- Use `Pillow` for image inspection (dimensions, region ID analysis)
- Both scripts support `--dry-run` for safe testing

### Upload script
- Scan `--source-dir` for `{example_id}/image.png` + `{example_id}/segmentation.png` pairs
- Upload to `annotations/{dataset}/{example_id}/` prefix in bucket
- Insert into Turso `images` table with public URLs
- Priority scoring: boost images containing weak classes (neck, forearm, accessory, feet)
- Skip already-uploaded (query Turso for existing example_id)
- Use subprocess rclone calls for upload (parallel transfers)

### Export script
- Query Turso: JOIN annotations + reviews + images WHERE approved=1
- Decode mask_data (base64 PNG) → grayscale PNG
- Download original image from Hetzner
- Save as `{example_id}/image.png` + `{example_id}/segmentation.png` + `{example_id}/metadata.json`
- metadata includes `segmentation_source: "human_corrected"`, reviewer info

## Files to Modify
- NEW: `annotator/scripts/upload_images.py`
- NEW: `annotator/scripts/export_approved.py`
- NEW: `annotator/scripts/requirements.txt` (Pillow, libsql-experimental, requests)

## Risks & Edge Cases
- External HD not always mounted — script should error clearly
- Turso rate limits on bulk inserts — batch in chunks
- Large datasets (2,925 gemini_diverse) — need progress reporting
- Bucket URL format must match what the web app expects
- mask_data in DB is base64 — need to handle both raw PNG bytes and base64-encoded

## Open Questions
- None blocking — DB schema from #230 is already deployed, bucket is configured
