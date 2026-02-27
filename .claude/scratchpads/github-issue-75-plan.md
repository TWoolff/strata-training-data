# Issue #75: Create VRoid model acquisition strategy and data directory

## Understanding
- Documentation task: set up `data/vroid/` directory and write a comprehensive README documenting VRoid Hub model acquisition strategy
- Type: documentation / research setup
- No code changes needed — just directory structure, README, and .gitignore updates

## Approach
- Follow existing patterns from `data/fbx/README.md` and `data/live2d/` for directory structure
- README should cover: sources, PAniC-3D methodology, license requirements, filtering criteria, download instructions, volume targets, and paper references
- Mirror the `.gitignore` pattern used for `data/live2d/` (ignore model files, track README and labels/)

## Files to Modify
- `data/vroid/.gitkeep` — new, placeholder to track empty directory
- `data/vroid/README.md` — new, acquisition strategy documentation
- `data/vroid/labels/` — new directory (with .gitkeep) for future annotation metadata
- `.gitignore` — add VRoid ignore rules + exceptions for README and labels/

## Risks & Edge Cases
- None — purely a documentation task
- Ensure .gitignore rules are ordered correctly so exceptions work (negation must come after the ignore rule)

## Open Questions
- None — requirements are clear from the issue

## Implementation Notes
- Created `data/vroid/` with `labels/.gitkeep` subdirectory (mirrors `data/live2d/labels/` pattern)
- README covers all acceptance criteria: sources (VRoid Hub, PAniC-3D, VRoid Studio samples), methodology (5-step PAniC-3D pipeline), license requirements, filtering criteria (5 rules), download instructions (API + Blender import), tooling suggestions, VRM→Strata bone mapping table, volume targets (2,000–5,000), and paper references (PAniC-3D, CharacterGen, StdGEN, NOVA-3D)
- `.gitignore` updated following exact pattern from `data/live2d/`: ignore `data/vroid/**`, except README.md and `labels/` directory
- No code changes — purely documentation and directory setup
