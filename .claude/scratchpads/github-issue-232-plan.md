# Issue #232: Annotator Canvas painting + region palette + toolbar

## Understanding
- Build the core annotation canvas where users paint body region corrections onto illustrated character images
- The model gets ~65% right, annotators correct mistakes by painting the right region color over wrong areas
- Character image as background, semi-transparent colored segmentation overlay on top
- Type: new feature (the heart of the annotator app)

## Approach
- **Canvas**: Two stacked HTML canvases — background (character image, read-only) + mask canvas (editable seg overlay)
- **Painting**: Pure Canvas API, no React re-renders on mouse move. Use requestAnimationFrame for smooth painting.
- **Undo/Redo**: Store ImageData snapshots of mask canvas (512x512x4 = 1MB each, 20 max = 20MB)
- **Zoom/Pan**: CSS transform on a container div. Track scale + offset.
- **Region Palette**: 22 colored buttons grouped by body area, with keyboard shortcuts from regions.ts
- **Toolbar**: Brush size, overlay opacity, undo/redo, zoom, skip/submit
- **API Routes**: GET /api/images (next pending image), POST /api/annotations (save corrected mask)
- **Export**: On submit, read mask canvas → convert colored pixels back to grayscale region IDs → PNG → base64

## Files to Create
- `src/components/Canvas.tsx` — dual-canvas painting component
- `src/components/RegionPalette.tsx` — 22 region buttons with shortcuts
- `src/components/Toolbar.tsx` — brush size, opacity, undo/redo, zoom, skip/submit
- `src/app/api/images/route.ts` — GET next pending image
- `src/app/api/annotations/route.ts` — POST save annotation

## Files to Modify
- `src/app/annotate/page.tsx` — replace placeholder with canvas + palette + toolbar

## Risks & Edge Cases
- Cross-origin image loading from Hetzner bucket — need to handle CORS or proxy
- Empty image queue (no pending images) — show "all done" message
- Large brush at canvas edges — clip to canvas bounds
- Background color [0,0,0] same as transparent — need alpha channel to distinguish painted vs unpainted
- Keyboard shortcuts must not fire when not on annotate page

## Open Questions
- None — issue spec is detailed enough to implement directly

## Implementation Notes
- All files created as specified in the issue
- Canvas uses 3 stacked canvases: bg (character image), mask (seg overlay), cursor (brush preview)
- Zoom/pan via CSS transform on container div, not canvas redraw — performant
- Eyedropper dispatches a custom `regionpick` event that bubbles up to the page
- Grayscale→color conversion uses pre-built LUT for fast rendering
- Color→ID conversion uses Map with (R<<16 | G<<8 | B) key for fast export
- Fixed ESLint error: can't read ref during render (removed spaceDown.current from JSX)
- Fixed TypeScript null check on cursor element in contextmenu handler
- Build + ESLint pass clean
- Pre-existing ruff errors (241) in Python files are unrelated
