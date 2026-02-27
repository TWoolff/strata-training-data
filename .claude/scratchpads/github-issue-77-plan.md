# Issue #77: Create data/psd/ directory and README

## Understanding
- Create `data/psd/` directory and README documenting PSD file sources
- Opportunistic collection — low priority, but directory and docs should exist
- `.gitignore` already has `data/psd/**` and `!data/psd/README.md` entries
- The directory itself does not exist yet — need to create it

## Approach
- Create `data/psd/` directory with `.gitkeep`
- Create `data/psd/README.md` following the same pattern as `data/vroid/README.md` and `data/live2d/README.md`
- Cover sources: OpenGameArt, itch.io, Patreon art packs
- Emphasize the body-part-separation requirement vs. rendering-concern layers

## Files to Modify
- `data/psd/.gitkeep` — New (placeholder to track directory)
- `data/psd/README.md` — New (documentation)

## Risks & Edge Cases
- None — documentation task

## Open Questions
- None
