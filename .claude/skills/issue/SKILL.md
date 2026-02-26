---
name: issue
description: Analyze and implement a GitHub issue for the Strata synthetic data pipeline. Use when working on GitHub issues, implementing features from tickets, or when user mentions issue numbers.
user-invokable: true
argument-hint: "<issue number>"
---

# GitHub Issue Implementation

Analyze and implement the GitHub issue: $ARGUMENTS

## PHASE 1: UNDERSTAND

1. **Create Feature Branch FIRST**
   ```bash
   git checkout -b feature/issue-$ARGUMENTS
   ```
   If already on a feature branch for this issue, skip this step.

2. **Fetch Issue Details**
   - Use `gh issue view $ARGUMENTS` to get the full issue description
   - Identify the type: bug fix, new feature, optimization, or refactor

3. **Check Existing Context**
   - Reference PRD in `.claude/prd/strata-synthetic-data-pipeline.md`
   - Look for related issues or discussions

4. **Understand the Codebase**
   - Identify relevant pipeline modules:
     - `generate_dataset.py` — Main orchestrator
     - `importer.py` — FBX loading and normalization
     - `bone_mapper.py` — Bone name → Strata region ID mapping
     - `pose_applicator.py` — Animation keyframe application
     - `renderer.py` — Color + segmentation render passes
     - `style_augmentor.py` — Post-render style transforms (pixel art, painterly, sketch)
     - `joint_extractor.py` — 3D bone positions → 2D pixel coordinates
     - `weight_extractor.py` — Per-vertex bone weight extraction
     - `exporter.py` — Image, mask, JSON output
     - `config.py` — Region colors, bone mappings, constants

## PHASE 2: PLAN (Write scratchpad BEFORE any code changes)

**CRITICAL: You MUST write the scratchpad BEFORE writing any code. The scratchpad is a planning tool and local diary — not a retrospective log. Do NOT skip ahead to implementation.**

1. **Break Down Requirements**
   - Split into small, testable tasks
   - Identify files that need modification
   - Note any new constants needed in `config.py`

2. **Write the Scratchpad** (BLOCKING — no code changes until this is written)
   - Create: `.claude/scratchpads/github-issue-$ARGUMENTS-plan.md`
   - Use this structure:
     ```markdown
     # Issue #N: <title>

     ## Understanding
     - What the issue is asking for
     - What type: bug fix / new feature / optimization / refactor

     ## Approach
     - High-level strategy and design decisions
     - Why this approach over alternatives (if applicable)

     ## Files to Modify
     - List each file and what changes it needs

     ## Risks & Edge Cases
     - What could go wrong
     - Edge cases to watch for (non-Mixamo skeletons, accessories, thin geometry)

     ## Open Questions
     - Anything uncertain before starting
     ```
   - This is your chance to think through the problem before touching code
   - Future sessions can reference this to understand the intent and context

3. **Follow Pipeline Patterns**
   - Constants in `config.py` (REGION_COLORS, bone mapping tables, RENDER_RESOLUTION)
   - `snake_case` for functions/variables, `ALL_CAPS` for constants
   - Type hints on function signatures
   - `pathlib.Path` for file paths
   - Google-style docstrings for public functions
   - Each module has single responsibility
   - Pipeline flow: Import → Map → Pose → Render → Style → Export

## PHASE 3: IMPLEMENT (only after scratchpad is written)

1. **Make Changes**
   - Follow existing code patterns
   - All constants in `config.py` or as `ALL_CAPS` at top of file
   - Type hints for all function signatures
   - Use `pathlib.Path` for file operations
   - Ensure all bpy code works in `--background` mode (no GUI assumptions)

2. **Key Implementation Notes**
   - **Rendering**: EEVEE with Emission shaders for segmentation, orthographic camera
   - **Bone Mapping**: exact → prefix → substring → manual override priority
   - **Segmentation**: Per-face material assignment via vertex group majority vote
   - **Masks**: 8-bit grayscale PNG, pixel value = region ID, no anti-aliasing
   - **Joints**: `bpy_extras.object_utils.world_to_camera_view` for 3D→2D projection
   - **Styles**: Render-time (Blender shaders) + post-render (PIL/OpenCV)
   - **Cleanup**: Always clear scene between characters to prevent data leaks

3. **Commit Incrementally**
   - Small, focused commits
   - Clear commit messages

## PHASE 4: VERIFY

1. **Run Quality Checks**
   ```bash
   # Run linter
   ruff check .

   # Run type checker (if configured)
   mypy *.py

   # Run pipeline on a test character
   blender --background --python generate_dataset.py -- \
     --input_dir ./source_characters/ \
     --output_dir ./dataset/ \
     --styles flat \
     --resolution 512
   ```

2. **Validate Dataset Output**
   - Check output images are 512×512 RGBA PNG
   - Verify masks are single-channel grayscale with valid region IDs (0–17)
   - Confirm joint JSON has all 17 joints with positions within image bounds
   - Overlay mask on image to visually verify alignment
   - Check file naming follows `{source}_{id}_pose_{nn}_{style}.png` convention

3. **Test Edge Cases**
   - Characters with accessories (should be hidden or mapped)
   - Bones that don't map to any region (should be logged as warnings)
   - Extreme poses with self-occlusion (joints should have `visible: false`)
   - Non-Mixamo bone naming conventions

## PHASE 5: FINALIZE

1. **Run Code Simplifier**
   - Get the list of modified files using `git diff --name-only HEAD~1` or similar
   - Invoke the `code-simplifier` skill using the Skill tool, passing the file paths as arguments
   - This ensures code clarity, consistency, and maintainability
   - Apply the simplifications before committing final changes

2. **Update Scratchpad with Implementation Notes**
   - Add a `## Implementation Notes` section to the existing scratchpad
   - Document what was actually implemented (may differ from plan)
   - Note any design decisions made during implementation
   - Note any follow-up work needed or discovered issues

3. **Create Pull Request** (if requested)
   - Use `gh pr create` with clear description
   - Reference the issue number

## Code Standards

- **Constants**: In `config.py` or `ALL_CAPS` at top of file
- **Naming**: `snake_case` for functions/variables
- **Types**: Type hints on all function signatures
- **Paths**: `pathlib.Path`, not string concatenation
- **Docstrings**: Google style for public functions
- **Imports**: stdlib → third-party (bpy, cv2, numpy, PIL) → local modules
- **Blender**: All code must work in `--background` mode
- **Pipeline**: Import → Map → Pose → Render → Style → Export
