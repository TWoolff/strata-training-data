# Issue #231: Annotator Landing Page + Onboarding + User Registration

## Understanding
- Build the entry point for the Strata Label annotation app
- Landing page with name input → creates/reuses user in Turso DB → redirects to /annotate
- Onboarding modal on first visit showing the 22-class body region reference
- Type: new feature, depends on #230 scaffold (already merged)

## Approach
- Create `/api/users/route.ts` API route for user upsert
- Replace placeholder `page.tsx` with landing page (name input + CTA button)
- Build `OnboardingGuide.tsx` as a client component with localStorage persistence
- Create `/annotate/page.tsx` as placeholder for next issue
- Generate a body map SVG diagram showing all 22 regions color-coded (no screenshot needed — SVG is cleaner and always accurate)
- All client-side state (user ID, onboarding flag) via localStorage

## Files to Modify
- `src/app/page.tsx` — replace with landing page
- `src/app/api/users/route.ts` — NEW: POST (upsert) + GET (lookup)
- `src/components/OnboardingGuide.tsx` — NEW: modal with region reference
- `src/components/BodyMap.tsx` — NEW: SVG body silhouette with colored regions
- `src/app/annotate/page.tsx` — NEW: placeholder annotate page
- `src/app/globals.css` — minor additions if needed

## Risks & Edge Cases
- Name collisions: handled by UNIQUE constraint + upsert logic
- Empty name submission: validate client-side + server-side
- localStorage not available: graceful fallback (re-prompt for name)
- Mobile layout: keep it simple, single column

## Open Questions
- None — issue is well-specified
