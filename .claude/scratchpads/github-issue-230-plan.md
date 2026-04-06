# Issue #230: Annotator — Project scaffold + Turso DB setup

## Understanding
- Set up a Next.js annotation web app ("Strata Label") in `annotator/` subdirectory
- Foundation for crowdsourced segmentation mask correction tool
- Friends help correct pseudo-labeled body part masks through a painting UI
- The seg model is bottlenecked on label quality — hand-corrected labels at 10x weight are highest impact

## Approach
- Use `create-next-app` with App Router, Tailwind, TypeScript
- Install `@libsql/client` for Turso (SQLite-compatible edge DB)
- Create DB schema with 4 tables: users, images, annotations, reviews
- Port region definitions from `pipeline/config.py` to TypeScript
- Dark theme root layout

## Files to Create
- `annotator/` — Next.js project via create-next-app
- `annotator/src/lib/db.ts` — Turso client singleton
- `annotator/src/lib/schema.ts` — DB schema + init function
- `annotator/src/lib/regions.ts` — 22-class region defs from config.py
- `annotator/.env.example` — Placeholder env vars
- Modify `annotator/src/app/layout.tsx` — Dark theme

## Risks & Edge Cases
- Turso requires actual DB URL + token for real connection — dev can use local SQLite
- Region colors/names must exactly match pipeline/config.py
- `.env.local` must be in .gitignore (create-next-app handles this by default)

## Open Questions
- None — issue is well-specified
