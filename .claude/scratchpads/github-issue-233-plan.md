# Issue #233: Annotator Gamification — progress bar, leaderboard, streaks

## Understanding
- Add gamification elements to the annotation tool: progress bar, leaderboard, streak toasts, and stats header
- Type: new feature
- Depends on #232 (canvas + annotations) which is already merged

## Approach
- Create a lightweight Toast component (no library needed — CSS animation + setTimeout)
- Create a ProgressBar component showing pending vs total images + personal count
- Add a leaderboard API route aggregating per-user stats from annotations + reviews tables
- Add a stats API route for total counts (used by progress bar)
- Create a leaderboard page at `/leaderboard`
- Integrate streak tracking into annotation page state (client-side counter, reset on page load)
- Stats header integrated into the existing annotation page header

## Files to Create
- `src/components/Toast.tsx` — reusable toast notification with auto-dismiss
- `src/components/ProgressBar.tsx` — progress bar + personal counter
- `src/app/leaderboard/page.tsx` — leaderboard table page
- `src/app/api/leaderboard/route.ts` — aggregated user stats endpoint
- `src/app/api/stats/route.ts` — total/pending image counts

## Files to Modify
- `src/app/annotate/page.tsx` — add streak logic, toast triggers, stats header, progress bar, leaderboard link

## Risks & Edge Cases
- LibSQL doesn't support `FILTER (WHERE ...)` — use `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` instead
- Streak is session-local (resets on refresh) — acceptable for MVP
- Leaderboard auto-refresh every 30s as specified
- Toast must not block annotation workflow

## Open Questions
- None — issue is well-specified

## Implementation Notes
- Toast system uses module-level state with listener pattern (no context provider needed)
- CSS `animate-fade-in` keyframe added to `globals.css`
- Streak is session-local (resets on page refresh) — simple `useState` counter
- Speed toast fires with 300ms delay to avoid overlapping the streak toast
- Leaderboard uses `HAVING COUNT(a.id) > 0` to hide users with no annotations
- Used `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` for approved count (SQLite compat)
- Stats API is separate from leaderboard to keep concerns clean
- Total annotations in header derived from leaderboard API (sum of all users)
- Build passes, TypeScript clean, all routes registered
