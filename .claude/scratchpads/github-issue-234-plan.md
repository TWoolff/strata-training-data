# Issue #234: QC Review Page for Owner

## Understanding
- Build a review page where the owner (Taw) can approve/reject annotations
- Approved annotations become high-value training data (10x weight)
- Depends on #232 (canvas + annotations) which is already merged
- Type: new feature

## Approach
- **Access control**: Simple `REVIEW_SECRET` env var, checked via query param `?key=...` on the review page. Store in localStorage once verified so the URL doesn't need the key on every page load.
- **Grid view**: Fetch all images with status='annotated' that don't have a review yet. Show thumbnails with colored seg overlay. Support filtering by dataset/annotator.
- **Detail view**: Three-panel comparison (original, pseudo-label, annotator correction). Diff highlighting by XOR-ing the two masks. Approve/reject/skip buttons + keyboard shortcuts.
- **API**: Single `/api/review` route with GET (list reviewable annotations) and POST (save decision). GET joins images + annotations + reviews to find unreviewed annotations.
- **Batch actions**: Checkbox selection in grid view, batch approve/reject.
- **Navigation**: Add "Review" link in annotate page header (only visible if review key is set).

## Files to Create
- `annotator/src/app/review/page.tsx` — Review page (grid + detail views)
- `annotator/src/app/api/review/route.ts` — Review API (GET list, POST decision)

## Files to Modify
- `annotator/src/app/annotate/page.tsx` — Add "Review" nav link (conditional on localStorage flag)

## Risks & Edge Cases
- Large mask_data blobs in annotations table — don't fetch mask_data in grid view, only in detail view
- Multiple annotations per image — show most recent annotation per image
- Diff computation client-side — need to decode two grayscale PNGs and compare pixel-by-pixel
- Image URLs may be external (S3) — CORS for canvas operations on diff computation

## Open Questions
- None blocking — the schema already has a `reviews` table with the right shape

## Implementation Notes
- Used `Suspense` wrapper for `useSearchParams` (Next.js 16 requires this)
- Auth flow: URL `?key=` param stores to localStorage; subsequent visits use stored key
- API route handles both single and batch reviews in one POST endpoint
- Diff computed client-side by decoding two grayscale PNGs to pixel arrays
- Grid view doesn't fetch mask_data (too large); detail view fetches via `?annotation_id=`
- POST updates image status to 'approved' or 'rejected' (extends the existing 'pending'/'annotated' states)
- Batch selection uses a Set<number> of annotation_ids
