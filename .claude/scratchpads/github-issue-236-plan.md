# Issue #236: Annotator — Deploy to Vercel + Hetzner public read

## Understanding
- Deploy the Next.js annotator app to Vercel for public access
- Configure Hetzner bucket for public read on `annotations/` prefix so images load in browser
- Set up CORS so Vercel domain can fetch images from bucket
- This is a devops/deployment task, not a code change task

## Approach
1. **Hetzner bucket policy** — set public read on `annotations/*` prefix using s3cmd or rclone
2. **CORS on Hetzner** — allow GET from any origin (images are AI-generated, no sensitivity)
3. **Next.js config** — add `remotePatterns` for Hetzner domain so `<img>` tags work (though Canvas uses raw fetch, not Next Image)
4. **Vercel config** — `vercel.json` not strictly needed, but add `.env.example` with REVIEW_SECRET
5. **Update .env.example** — add REVIEW_SECRET

## What's Already Done
- App is fully functional locally (issues 230-232 closed)
- Images uploaded to `annotations/` prefix in Hetzner bucket via `upload_images.py`
- Image URLs in Turso DB point to `https://fsn1.your-objectstorage.com/strata-training-data/annotations/...`
- Canvas loads images via `<img>` element (standard browser fetch, needs CORS)

## Files to Modify
- `annotator/next.config.ts` — add `images.remotePatterns` for Hetzner domain
- `annotator/.env.example` — add REVIEW_SECRET
- `annotator/vercel.json` — optional, for build settings (root dir = annotator/)

## What Needs Manual Steps (not code)
- Vercel: connect repo, set root dir to `annotator/`, add env vars
- Hetzner: set bucket policy for public read + CORS
- These will be documented as instructions, not automated

## Risks & Edge Cases
- Hetzner S3-compatible API may not support all bucket policy features
- CORS preflight may cache — test in incognito
- Canvas uses crossOrigin on img elements? Need to check — if not, tainted canvas won't export
