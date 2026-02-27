# Issue #79: Build label_actions.py CLI

## Understanding
- Interactive CLI for reviewing/tagging BVH mocap clips in cmu_action_labels.csv
- All 78 existing clips are already labeled — tool supports review and updates
- Uses bvh_parser.py to show BVH metadata when BVH files are available
- Pure Python, no Blender dependency

## Approach
- argparse-based CLI with filter options (--unlabeled, --action-type)
- Load CSV, present each clip for review, accept typed input
- Save incrementally after each update
- Graceful Ctrl+C handling (save on interrupt)
- Show BVH metadata if the file exists in data/mocap/

## Files to Modify
- `animation/scripts/label_actions.py` — New file

## Risks & Edge Cases
- BVH files may not be present locally (in .gitignore) — handle gracefully
- User may Ctrl+C mid-session — catch KeyboardInterrupt and save
- CSV may have trailing whitespace or encoding issues — use csv module

## Open Questions
- None
