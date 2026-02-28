# Issue #52: Diff3F Semantic Feature Exploration Notebook

## Understanding
- Create a Jupyter notebook exploring whether Diffusion 3D Features (Diff3F) could improve Strata's weight painting prediction
- The ASMR paper (arxiv 2503.13579) uses Diff3F semantic descriptors from diffusion model activations for improved rigging/skinning
- Key insight: pixels at joints carry features that say "this is a bend region" regardless of art style
- **Type:** Research exploration (notebook only, no production code changes)

## Approach
- Create `docs/research/diff3f_exploration.ipynb` as a self-contained research notebook
- Structure the notebook as a 4-step exploration:
  1. **Setup & Installation** — Install Diff3F, load models, verify environment
  2. **Feature Extraction** — Extract features from 20-30 Strata training images
  3. **Visualization** — Cluster analysis, body part grouping, joint proximity analysis
  4. **Evaluation** — Compare features vs. current geometric approach, recommendation

### Key Technical Details
- Diff3F works on 3D meshes (not 2D images directly) — it renders from multiple views, extracts diffusion+DINO features, and unprojects to 3D vertices
- Output: per-vertex feature vector of dimension 2048 (1280 diffusion + 768 DINO)
- Original repo: https://github.com/niladridutt/Diffusion-3D-Features (MIT license)
- Requires: PyTorch, PyTorch3D, diffusers (ControlNet), DINO ViT

### Adaptation for Strata
- Strata already has 3D meshes (FBX characters) — can extract Diff3F features per vertex
- Compare Diff3F feature clustering against Strata's 19-region segmentation
- Evaluate whether features capture joint/deformation semantics beyond what bone weights provide
- If working with 2D images only (no mesh access at inference), explore 2D feature extraction via DINO/diffusion features on rendered images as a fallback

## Files to Create
- `docs/research/diff3f_exploration.ipynb` — The main exploration notebook

## Files to Reference (read-only)
- `pipeline/weight_extractor.py` — Current weight extraction approach
- `pipeline/config.py` — REGION_NAMES, bone mappings
- `pipeline/bone_mapper.py` — How bones map to regions

## Risks & Edge Cases
- Diff3F requires 3D meshes + PyTorch3D — heavy GPU dependency
- Feature extraction is slow (~100 views per mesh rendered)
- Training images may not be pre-rendered (output/ is empty) — notebook should handle generating or loading sample data
- The PRD says "20-30 Strata training images" but Diff3F works on meshes, not images — need to clarify this in the notebook (extract from FBX meshes, visualize on rendered images)

## Open Questions
- Should the notebook include a 2D-only feature exploration path (DINO features on rendered images) as a lightweight alternative?
  → Yes, include as a secondary exploration since Strata's inference is 2D
- How to handle the case where no GPU is available?
  → Document requirements clearly, provide cached/pre-computed results cells

## Implementation Notes

### What was built
- Created `docs/research/diff3f_exploration.ipynb` with 30 cells (15 code + 15 markdown)
- Notebook covers the full exploration pipeline:
  1. **Setup** — dependency checking, FBX mesh loading via trimesh
  2. **Diff3F extraction** — clones repo, extracts 2048-dim per-vertex features with caching
  3. **DINO-only fallback** — DINOv2 patch-level features from 2D images (more practical for inference)
  4. **Visualization** — t-SNE clustering by region, PCA feature maps, cross-character similarity
  5. **Joint proximity analysis** — boundary vs. interior feature distribution comparison
  6. **Region classification** — k-NN comparison of geometric-only vs. Diff3F vs. combined features
  7. **Weight correlation** — intra/inter region similarity ratios
  8. **Conclusion** — checklist template for findings + follow-up issue template

### Design decisions
- Used trimesh for FBX loading (no Blender dependency in notebook)
- Heuristic Y-axis region assignment as proxy for ground truth (since bone weights need Blender)
- Included DINOv2 as a practical 2D alternative since Strata's inference is 2D
- Feature caching to `output/research/diff3f_features/` avoids re-extraction
- Reduced default views from 100 to 50 for faster iteration

### No production code changes
- This is notebook-only per the issue scope
- No changes to `pipeline/`, `ingest/`, or any other production modules
