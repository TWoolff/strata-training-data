---
name: issue
description: This skill should be used when the user asks to "implement an issue", "work on issue #N", "fix issue N", mentions a GitHub issue number, or says "/issue". It analyzes and implements GitHub issues for the Strata synthetic data pipeline.
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
   - Reference PRDs in `.claude/prd/` for specifications
   - Look for related issues or discussions

4. **Understand the Codebase**
   - Identify relevant pipeline modules:

   **Core pipeline (`pipeline/`):**
   - `generate_dataset.py` — Main orchestrator
   - `config.py` — Region colors, bone mappings, render settings, all constants (~1000 lines)
   - `importer.py` — FBX loading and normalization
   - `bone_mapper.py` — Bone name -> Strata region ID mapping (Mixamo, VRM, generic, fuzzy)
   - `pose_applicator.py` — Animation keyframe application
   - `renderer.py` — Color + segmentation + multi-angle render passes
   - `style_augmentor.py` — Post-render style transforms (pixel art, painterly, sketch)
   - `joint_extractor.py` — 3D bone positions -> 2D pixel coordinates
   - `weight_extractor.py` — Per-vertex bone weight extraction
   - `draw_order_extractor.py` — Per-pixel depth from Z-buffer
   - `exporter.py` — Image, mask, JSON output
   - `manifest.py` — Dataset statistics + quality report
   - `splitter.py` — Train/val/test split by character
   - `validator.py` — Automated post-generation validation

   **Source-specific modules (`pipeline/`):**
   - `vroid_importer.py` — VRM/VRoid character import + A-pose normalization
   - `vroid_mapper.py` — VRoid material slot -> Strata region mapping
   - `live2d_mapper.py` — Live2D ArtMesh fragment -> Strata label mapping
   - `spine_parser.py` — Spine 2D JSON project parsing (pure Python, no Blender)
   - `accessory_detector.py` — Detect and hide accessories for clean training data
   - `measurement_ground_truth.py` — Body measurements from 3D mesh vertices

   **External dataset adapters (`ingest/`):**
   - `nova_human_adapter.py` — NOVA-Human dataset -> Strata format conversion
   - `stdgen_semantic_mapper.py` — StdGEN 4-class -> Strata 20-class mapping
   - `stdgen_pipeline_ext.py` — StdGEN Blender rendering extension

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
   - Constants in `config.py` (REGION_COLORS, REGION_NAMES, bone mapping tables, RENDER_RESOLUTION, CAMERA_ANGLES, VRM_BONE_ALIASES, VROID_MATERIAL_PATTERNS, SPINE_BONE_PATTERNS, LIVE2D_FRAGMENT_PATTERNS, STDGEN_SEMANTIC_CLASSES, style parameters, accessory patterns)
   - `snake_case` for functions/variables, `ALL_CAPS` for constants
   - Type hints on function signatures
   - `pathlib.Path` for file paths
   - Google-style docstrings for public functions
   - `from __future__ import annotations` at top of each module
   - Each module has single responsibility
   - Pipeline flow: Import -> Map -> Pose -> Render -> Style -> Export -> Validate -> Manifest

## PHASE 3: IMPLEMENT (only after scratchpad is written)

1. **Make Changes**
   - Follow existing code patterns
   - All constants in `config.py` or as `ALL_CAPS` at top of file
   - Type hints for all function signatures
   - Use `pathlib.Path` for file operations
   - Ensure all bpy code works in `--background` mode (no GUI assumptions)

2. **Key Implementation Notes**
   - **Rendering**: EEVEE with Emission shaders for segmentation, orthographic camera
   - **Bone Mapping**: exact -> prefix-stripped -> substring -> fuzzy keyword priority (see `bone_mapper.py`)
   - **Segmentation**: Per-face material assignment via vertex group majority vote
   - **Masks**: 8-bit grayscale PNG, pixel value = region ID (0-19, 20 total), no anti-aliasing
   - **Joints**: `bpy_extras.object_utils.world_to_camera_view` for 3D->2D projection, 19 joints (regions 1-19)
   - **Styles**: Render-time (Blender shaders: flat, cel, unlit) + post-render (PIL/OpenCV: pixel, painterly, sketch)
   - **Multi-angle**: 5 camera angles (front, three-quarter, side, three-quarter-back, back) via `CAMERA_ANGLES`
   - **Cleanup**: Always clear scene between characters to prevent data leaks

3. **Commit Incrementally**
   - Small, focused commits
   - Clear commit messages

## PHASE 4: VERIFY

1. **Run Quality Checks**
   - Invoke `/run-tests` to run the test suite, linting, and format checks
   - If the pipeline was modified and test data is available, run a single-character pipeline test:
     ```bash
     blender --background --python run_pipeline.py -- \
       --input_dir ./data/fbx/ \
       --pose_dir ./data/poses/ \
       --output_dir ./output/segmentation/ \
       --styles flat \
       --resolution 512
     ```

2. **Validate Dataset Output** (if pipeline produced output)
   - Invoke `/validate-dataset` to run automated validation on generated output

3. **Test Edge Cases**
   - Characters with accessories (should be hidden or mapped)
   - Bones that don't map to any region (should be logged as warnings)
   - Extreme poses with self-occlusion (joints should have `visible: false`)
   - Non-Mixamo bone naming conventions (VRM, generic, Blender-style)

## PHASE 5: FINALIZE

1. **Run Code Simplifier**
   - Get the list of modified files using `git diff --name-only HEAD~1` or similar
   - Invoke `/code-simplifier` with the file paths as arguments
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
- **Imports**: `from __future__ import annotations` first, then stdlib -> third-party (bpy, cv2, numpy, PIL) -> local modules
- **Blender**: All code must work in `--background` mode
- **Pipeline**: Import -> Map -> Pose -> Render -> Style -> Export -> Validate -> Manifest
- **Linting**: `ruff check .` and `ruff format .` (line-length=100, py310)
