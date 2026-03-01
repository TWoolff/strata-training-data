# Strata: 3D Mesh Reconstruction — Research-Informed PRD

**Product:** Strata — From Painting to Animation  
**Module:** 3D Mesh Pipeline (v2.0 Feature)  
**Date:** February 27, 2026  
**Status:** Research Complete → Architecture Decision  

---

## 1. Executive Summary

This PRD documents a comprehensive research survey of 3D character reconstruction from 2D artwork, covering papers and systems from 2023 through February 2026. The goal: determine the optimal technical approach for Strata's 3D mesh pipeline, where an artist paints character views and Strata produces a fully textured, rigged, animatable 3D mesh.

**Key Decision: Front + 3/4 view (required) + back view (optional)**

The 3/4 view replaces the side view as the secondary required input. Research and industry practice converge on this: the 3/4 angle provides both width and depth information in a single natural image, directly covers the hardest texture-seam transition zone, and is the most natural angle for character artists to draw.

**Key Decision: Hybrid architecture — template mesh deformation + neural texture completion**

Rather than pure neural reconstruction (unpredictable, artist loses control) or pure template deformation (limited expressiveness), Strata uses template meshes for geometry and neural methods for texture gap-filling. This gives artists deterministic control over proportions while leveraging AI for the tedious parts.

---

## 2. Research Landscape (2023–2026)

### 2.1 The Tsinghua/Tripo Pipeline (State of the Art)

The most important development is a coherent pipeline emerging from Tsinghua University and Tripo (VAST AI), published across three papers that together solve the full problem:

**CharacterGen** (SIGGRAPH 2024, TOG)  
Single image → multi-view A-pose images → 3D mesh. Introduced the Anime3D dataset of 13,746 VRoid characters rendered in multiple poses and views. Key innovation: a multi-view diffusion model that simultaneously canonicalizes body pose (converting any pose to A-pose) and generates consistent views. Uses a transformer-based sparse-view reconstruction model for mesh creation and texture-back-projection for texture maps. Limitation: produces monolithic watertight meshes with no semantic decomposition — you can't separate hair from body from clothes.

**StdGEN** (CVPR 2025, accepted February 27, 2026)  
Direct successor to CharacterGen that solves the decomposition problem. From a single image, generates semantically decomposed 3D characters — body, clothes, and hair as separate mesh layers — in three minutes. Core component is S-LRM (Semantic-aware Large Reconstruction Model), a transformer that jointly reconstructs geometry, color, and semantic labels from multi-view images. Uses the Anime3D++ dataset with finely annotated multi-view, multi-pose semantic parts. Significantly outperforms CharacterGen in geometry quality, texture detail, and decomposability. This is the current state-of-the-art for single-image anime character generation.

**UniRig** (SIGGRAPH 2025, TOG)  
Automatic skeletal rigging for any 3D mesh. Uses a GPT-like autoregressive transformer with a novel Skeleton Tree Tokenization to predict topologically valid skeleton hierarchies, plus a Bone-Point Cross Attention mechanism for skinning weight prediction. Trained on Rig-XL, a dataset of 14,000+ rigged 3D models spanning humanoids, animals, and fantasy creatures. Achieves 215% improvement in rigging accuracy and 194% in motion accuracy over previous methods. Processes most models in 1–5 seconds. Open source.

**Together, these three papers represent a complete single-image-to-animated-character pipeline:** CharacterGen/StdGEN generates the mesh → UniRig rigs it → standard animation retargeting drives it. This pipeline is Strata's primary competitive reference.

### 2.2 Other Neural Reconstruction Approaches

**NOVA-3D** (2024)  
Reconstructs full-body anime characters from front + back non-overlapping views. Trained on 10,200 VRoid models. Demonstrates that views with zero visual overlap can still produce coherent 3D reconstruction through view-aware feature fusion. However, side views (90° rotation) remain soft and blurry — the network hallucinates geometry it has never seen. Occasional "ghost face" problem where back view looks like front.

**DrawingSpinUp** (SIGGRAPH Asia 2024)  
The most relevant paper for hand-drawn input. Takes a single character drawing and produces 3D animation. Key insight: contour lines in 2D drawings are view-dependent — they don't exist in 3D — and they confuse reconstruction networks. Solution: removal-then-restoration strategy strips contour lines before reconstruction, then re-renders them per frame using a geometry-aware stylization network. Uses skeleton-based thinning deformation to fix overly thick limbs. User study: 4.55/5 motion consistency, 4.53/5 style preservation. This contour-handling approach is directly applicable to Strata.

**PAniC-3D** (CVPR 2023)  
Single-view 3D reconstruction for anime heads only. Uses EG3D (3D-aware GAN) to generate neural radiance fields. Built the VRoid dataset that subsequent papers all use. Head-only scope keeps the problem tractable. Full-body remains much harder.

**DeformSplat** (SIGGRAPH Asia 2025)  
Gaussian-to-Pixel Matching bridges 3D Gaussian representations with 2D pixel observations, enabling deformation guidance from single images. Rigid Part Segmentation identifies rigid regions (limbs, torso) to maintain geometric coherence. Demonstrated distortion-free animation of 3D characters from single photographs across all viewing angles.

### 2.3 Foundation Models for 3D Generation

**TripoSG** (February 2025, VAST/Tripo)  
High-fidelity 3D shape synthesis from single images using a 1.5B parameter rectified flow transformer. Trained on 2 million curated Image-SDF pairs. Handles diverse input styles including cartoons and sketches. Open source (MIT license). Represents a major quality jump in general image-to-3D — previously a rough approximation, now detailed enough for game assets. The scribble variant (TripoSG-scribble) can generate 3D from rough sketches with text prompts.

**Hunyuan3D** (Tencent, 2024–2025, now at v3.0)  
Two-stage pipeline: Hunyuan3D-DiT generates bare mesh geometry, Hunyuan3D-Paint synthesizes PBR textures (albedo, normal, roughness, metallic). Supports text-to-3D, image-to-3D, and multi-view-to-3D input. Solves the Janus problem (multi-faced objects) through view-aware diffusion. Now at 10B parameters. Open source. Notably supports up to 4 side-view photos as input — relevant for Strata's multi-view approach.

**DiffSplat** (ICLR 2025)  
Repurposes image diffusion models to directly generate 3D Gaussian Splatting representations. Maintains 3D consistency in a unified model. Demonstrates that 2D generative priors can be adapted for native 3D output without two-stage pipelines.

### 2.4 Multi-View Texture Projection

**MVPaint** (2024)  
The most comprehensive texture projection system. Three-stage pipeline: (1) Synchronized Multi-view Generation for coarse texturing; (2) Spatial-aware 3D Inpainting for unobserved areas — resolves inpainting in 3D space by considering spatial relations among uniformly sampled surface points; (3) UV Refinement with super-resolution upscaling to 2K and a Spatial-aware Seam-Smoothing Algorithm (SSA) that repairs texture discontinuities at UV seams by smoothing color vectors using their 3D neighbors. This SSA approach is directly applicable to Strata's texture blending problem.

**TexPainter** (2024)  
Joint optimization approach: during each denoising step, decodes all view latents into color space, blends them into a texture image, then optimizes latents across all views so they generate the same rendering as the blended texture. Eliminates sequential view dependence. Improves consistency over iterative per-view painting approaches.

### 2.5 Character Turnaround Industry Practice

Professional character turnaround sheets standardly include front, 3/4, side (profile), 3/4 back, and back views. The 3/4 view is described across multiple industry sources as providing "a more natural sense of depth and form" and being "especially valuable for animators and 3D modelers who need to visualize how the character will move and occupy space." Many professional workflows start with the 3/4 view as the primary design angle.

For 3D modeling reference, industry practice is front + side + back at minimum, with the 3/4 view as the most commonly added optional angle. Sources note that the 3/4 view is the hardest to derive from other views — front and side can be more mechanically translated, but the 3/4 requires understanding of 3D volume.

---

## 3. The 3/4 View Decision

### 3.1 Why 3/4 View Replaces Side View

**Information density:** A 3/4 view (approximately 45° between front and side) simultaneously provides partial width AND partial depth information. A strict side view provides depth only. Since the front view already provides full width, the side view is partially redundant — both measure height, but only the side adds depth. The 3/4 view adds depth WHILE also providing visual information for the critical transition zone.

**Texture coverage:** With front + side + back views, the worst texture-seam zones are at 45° and 135° — exactly where no view provides direct information. With front + 3/4 + back, the worst gap is at 90° (pure side), which is a much simpler zone to interpolate because character silhouettes tend to be smoother in strict profile.

**Artist ergonomics:** Character artists universally report that the 3/4 angle is the most natural to draw. It's the default "hero" angle in concept art. Requiring a strict orthographic side view is unnatural and requires careful discipline to avoid perspective cues. The 3/4 view lets artists draw more naturally.

**Measurement extraction still works:** While a strict orthographic side view gives cleaner depth measurements, the 3/4 view's depth information can be extracted through known-angle projection decomposition. At 45°, apparent width = true_width × cos(45°) + true_depth × sin(45°), which combined with the front view's true_width measurement, solves for true_depth. At any known angle, the math is straightforward.

**Angular coverage:** Front (0°) + 3/4 (45°) + back (180°) covers 0–45° with direct paint data and 45–180° with interpolation over 135°. Front (0°) + side (90°) + back (180°) covers each 90° gap equally but the 0–90° zone is the most visually important (characters are primarily seen from the front). The 3/4 configuration front-loads coverage where it matters most.

### 3.2 Recommended View Configuration

**Primary (required):**
- Front view (0°) — Full width measurements, primary face/body texture
- 3/4 view (45°) — Depth extraction, transition-zone texture, volume verification

**Secondary (optional but recommended):**
- Back view (180°) — Back texture, hair/cape details, design verification

**Optional (for maximum quality):**
- 3/4 back view (135°) — If artist wants to control the back-to-side transition
- Side view (90°) — For characters with important strict-profile features

**Minimum viable input:** Front + 3/4 only. This provides enough information for full geometry reconstruction and covers 0–45° with direct texture. The 45–180° range is filled by: mirroring the front (for symmetric characters), AI-assisted texture generation (DrawingSpinUp/MVPaint approach), or simple color extrapolation from the 3/4 view's edge colors.

---

## 4. Architecture: Hybrid Template + Neural

### 4.1 Why Not Pure Neural Reconstruction?

The StdGEN/CharacterGen pipeline is impressive but fundamentally wrong for Strata's use case:

1. **Artist control:** Neural methods hallucinate unseen geometry. The artist paints a front view; the AI invents what the side looks like. If the AI makes a bad choice, the artist has no recourse. Strata's value proposition is that the artist's painting IS the character — not an AI interpretation of it.

2. **Consistency:** Neural methods produce variable quality depending on input style, complexity, and training distribution. StdGEN was trained on VRoid anime characters. A painterly/watercolor style, a chibi character, or a realistic style may produce degraded results.

3. **Determinism:** Same input can produce different outputs due to stochastic sampling. Professional tools must be deterministic — same painting should always produce same mesh.

4. **Compute requirements:** StdGEN requires significant GPU resources. Strata needs to run on consumer hardware or with minimal cloud dependency.

### 4.2 Why Not Pure Template Deformation?

Template deformation alone is too rigid:

1. **Limited expressiveness:** 5–10 base templates can't capture the full diversity of character designs. A character with wings, a tail, oversized weapons, or non-humanoid proportions breaks template assumptions.

2. **Texture projection artifacts:** Simple angle-weighted blending between views produces visible seams, especially where face normals transition between view-dominant zones.

3. **No intelligence:** Template deformation doesn't understand the artwork — it just stretches mesh to match measurements. It can't infer that a painted cape should flow around the back, or that asymmetric armor requires different geometry on each side.

### 4.3 The Hybrid Architecture

**Geometry: Template mesh + measurement-driven deformation**
- Artist paints front + 3/4 views
- Strata's existing 22-label segmentation runs on each view independently
- Measurement extraction derives width, depth, and height for each body part
- Nearest template mesh is selected and deformed via cage/lattice to match
- Result: predictable, artist-controlled geometry with clean topology

**Texture: Direct projection + neural gap-filling**
- Front view texture → projected onto front-facing mesh faces
- 3/4 view texture → projected onto 45°-facing mesh faces
- Back view texture (if provided) → projected onto back-facing mesh faces
- Transition zones: angle-weighted blending with MVPaint-style Spatial-aware Seam-Smoothing
- Unobserved regions (pure side, under-chin, top-of-head): neural inpainting using diffusion model conditioned on surrounding projected texture
- Contour handling: DrawingSpinUp-style removal before projection, optional re-rendering after

**Rigging: UniRig-style auto-rigging OR Strata's existing skeleton**
- For template-deformed meshes: pre-rigged skeleton deforms with template (fastest)
- For custom/unusual meshes: UniRig-style auto-rigging as fallback
- Weight painting: Strata's existing system, initialized from template weights

### 4.4 Pipeline Stages

```
STAGE 1: Multi-View Painting
├── Artist paints front view on guided canvas
├── Artist paints 3/4 view on guided canvas (with front overlay)
├── [Optional] Artist paints back view
└── View consistency validation (height matching, proportion checks)

STAGE 2: Segmentation & Measurement
├── Run 22-label segmentation on each view
├── Extract bounding boxes per body part per view
├── Derive measurements: width (front), depth (from 3/4 decomposition)
├── Height cross-validation between views
└── Generate measurement profile

STAGE 3: Template Selection & Deformation
├── Select nearest base template from library
├── Apply cage/lattice deformation to match measurements
├── Verify silhouette match against painted views
├── Artist adjustment pass (optional manual tweaks)
└── Output: deformed mesh with clean UV layout

STAGE 4: Texture Projection
├── Project front view onto front-facing faces
├── Project 3/4 view onto 45°-facing faces
├── Project back view onto back-facing faces (if provided)
├── Angle-weighted blending in transition zones
├── Spatial-aware seam smoothing (MVPaint SSA approach)
├── Neural inpainting for unobserved regions
└── Output: complete UV texture map

STAGE 5: Rigging & Animation
├── Apply pre-deformed skeleton from template
├── Transfer skinning weights from template
├── Validate with test animation
└── Connect to Strata animation timeline
```

---

## 5. Template Mesh Library

### 5.1 Base Templates (8 archetypes)

| ID | Archetype | Proportions | Use Case |
|----|-----------|-------------|----------|
| T01 | Standard Adult | 7–8 heads tall | Default humanoid |
| T02 | Heroic Adult | 8–9 heads tall | Action characters, superheroes |
| T03 | Slim/Elegant | 7–8 heads, narrow build | Fashion, elves, elegant characters |
| T04 | Stocky/Muscular | 6–7 heads, wide build | Dwarves, brawlers, heavy characters |
| T05 | Child/Teen | 5–6 heads tall | Young characters, smaller stature |
| T06 | Chibi/Deformed | 2–4 heads tall | SD/chibi style, mascots |
| T07 | Tall Exaggerated | 9–10 heads tall | Stylized fashion, CLAMP-style |
| T08 | Non-standard | Modular base | Characters with wings, tails, extra limbs |

### 5.2 Template Requirements

Each template must include:
- Clean quad-dominant topology (3,000–8,000 polygons)
- Pre-defined UV layout optimized for front-heavy texture projection
- Pre-rigged with Strata's 20-bone skeleton hierarchy
- Proper edge loops at joints for clean deformation
- Semantic vertex groups matching Strata's 22 labels
- Cage/lattice control structure for deformation
- LOD variants (high-poly for rendering, low-poly for real-time)

### 5.3 Template Sourcing Options

1. **Build custom** — Highest quality, perfect Strata integration, most expensive (~$2–5K per template for professional modeler)
2. **MakeHuman base** — Open source, good topology, needs adaptation for stylized proportions
3. **VRoid/VRM format** — Anime-optimized, large community, but VRoid-specific topology
4. **Commission from community** — Moderate cost, can specify exact requirements

**Recommendation:** Build T01 and T06 custom (the two most common archetypes), adapt MakeHuman for T02–T05, build T07–T08 as extensions of T01.

---

## 6. Measurement Extraction from 3/4 View

### 6.1 The Geometry

For a 3/4 view at angle θ from front (typically θ = 45°):

```
apparent_width_in_3/4 = true_width × cos(θ) + true_depth × sin(θ)
apparent_height_in_3/4 = true_height  (unchanged by horizontal rotation)
```

Since the front view gives us `true_width` directly:

```
true_depth = (apparent_width_in_3/4 - true_width × cos(θ)) / sin(θ)
```

At θ = 45°: `true_depth = (apparent_width_3/4 - true_width × 0.707) / 0.707`

### 6.2 Practical Extraction

For each segmented body part:
1. Compute bounding box in front view → `front_width`, `front_height`
2. Compute bounding box in 3/4 view → `three_quarter_width`, `three_quarter_height`
3. Validate: `|front_height - three_quarter_height| < tolerance` (should match)
4. Solve: `depth = (three_quarter_width - front_width × cos(45°)) / sin(45°)`

### 6.3 Angle Determination

The 3/4 view angle doesn't need to be exactly 45°. Strata can:
- **Enforce 45°** by providing a rotated template guide on the canvas
- **Detect angle** by comparing the head width ratio between front and 3/4 views (the head is the most reliably segmented body part, and its width changes predictably with rotation)
- **Allow artist specification** with an angle slider (35°–55° range)

**Recommendation:** Enforce 45° with a template guide for v2.0. Add angle detection as v2.1 improvement.

---

## 7. Texture Projection & Seam Resolution

### 7.1 Projection Strategy

For each mesh face, compute the angle between the face normal and each camera view direction. Assign texture from the view with the smallest angle (most direct viewing).

```
weight_front(face) = max(0, dot(face_normal, front_camera_dir))
weight_3/4(face)   = max(0, dot(face_normal, three_quarter_camera_dir))
weight_back(face)   = max(0, dot(face_normal, back_camera_dir))

// Normalize
total = weight_front + weight_3/4 + weight_back
color(face) = (weight_front × front_color + weight_3/4 × 3/4_color + weight_back × back_color) / total
```

### 7.2 Seam Smoothing (from MVPaint)

At UV seam boundaries where adjacent faces source from different views:
1. Detect seam mask via connectivity analysis of the UV map
2. For each seam pixel, find its 3D position on the mesh surface
3. Sample neighboring non-seam pixels in 3D space (not UV space — critical for correct blending across UV cuts)
4. Smooth color vectors using 3D-neighbor weighted average
5. Result: spatially continuous textures even across UV boundaries

### 7.3 Contour Line Handling (from DrawingSpinUp)

2D artwork typically has contour/outline strokes that are view-dependent. These cause artifacts when projected onto 3D mesh faces that face away from the original camera:

1. **Detection:** Identify contour lines via edge detection on the segmented painting
2. **Removal:** Inpaint contour regions with surrounding fill colors before texture projection
3. **Re-rendering (optional):** After 3D mesh is textured, render new view-dependent contours using mesh silhouette detection (faces where `dot(face_normal, camera_dir) ≈ 0`)

For Strata's hand-painted aesthetic, contour re-rendering is likely unnecessary — the painted style already has intentional outlines that are part of the character's look. Contour removal before projection is the critical step.

### 7.4 Unobserved Region Filling

With front (0°) + 3/4 (45°) + back (180°), the unobserved zone is approximately 45°–180° from one side. Strategies, in order of preference:

1. **Mirror (symmetric characters):** For symmetric designs, mirror the 0°–45° texture to fill 315°–360°, and mirror the 180°–225° to fill 135°–180°. Remaining gap: 45°–135° on the unseen side.

2. **Color extrapolation:** For each unobserved face, sample the nearest observed face's color and extrapolate. Works well for uniform-colored regions (clothing, skin).

3. **Neural inpainting:** For complex unobserved regions, use a diffusion model conditioned on the surrounding projected texture to generate plausible fill. This is where DrawingSpinUp and MVPaint's approaches are most useful — inpainting in 3D-aware UV space.

4. **Artist override:** Let the artist paint directly onto the UV map in a dedicated texture editing mode for full control.

---

## 8. Integration with Existing Strata Systems

### 8.1 What Carries Forward Unchanged

| System | Status | Notes |
|--------|--------|-------|
| 22-label segmentation model | ✅ Reused | Runs on each view independently |
| 20-bone skeleton hierarchy | ✅ Reused | Pre-embedded in templates |
| Weight painting system | ✅ Reused | Initialized from template weights |
| Animation timeline | ✅ Reused | Drives skeleton same as 2.5D |
| Blueprint system | ✅ Reused | Same layered approach |
| Export pipeline | ⚠️ Extended | Add 3D mesh export (glTF, FBX) |
| Animation Intelligence | ⚠️ Extended | Retarget to 3D skeleton |

### 8.2 New Components Required

| Component | Complexity | Priority |
|-----------|------------|----------|
| Multi-view painting canvas with overlay guides | Medium | P0 |
| Measurement extraction from segmented views | Low | P0 |
| Template mesh library (8 base meshes) | High (asset creation) | P0 |
| Cage/lattice deformation system | Medium | P0 |
| Texture projection with angle-weighted blending | Medium | P0 |
| Spatial-aware seam smoothing | Medium | P1 |
| Contour line detection and removal | Low | P1 |
| Neural inpainting for unobserved regions | High | P1 |
| View consistency validation | Low | P1 |
| 3D mesh preview renderer | Medium | P0 |
| UV texture editing mode | Medium | P2 |
| Auto-rigging fallback (UniRig-style) | High | P2 |

### 8.3 2.5D → 3D Upgrade Path

The 2.5D pipeline (v1.0) and 3D mesh pipeline (v2.0) share a common foundation:

```
v1.0 (2.5D):  Front painting → Segmentation → Layer separation → Skeleton → Animation
v2.0 (3D):    Multi-view painting → Segmentation → Mesh deformation → Skeleton → Animation
                     ↑                      ↑                ↑
                  New input            Same model        New geometry step
```

An artist's v1.0 front painting can be imported as the front view for v2.0. They only need to add the 3/4 view (and optionally back). This makes the upgrade path frictionless.

---

## 9. Competitive Landscape & Positioning

### 9.1 AI-Powered Competitors (as of Feb 2026)

| System | Input | Output | Time | Quality | Artist Control |
|--------|-------|--------|------|---------|----------------|
| StdGEN | 1 image | Decomposed mesh | 3 min | High (anime) | None |
| TripoSG | 1 image | Mesh | <1 min | High (general) | None |
| Hunyuan3D 3.0 | 1 image/text | PBR mesh | Minutes | High (general) | Minimal |
| DrawingSpinUp | 1 drawing | Animated 3D | Minutes | Medium | None |
| **Strata v2.0** | **2–3 paintings** | **Textured rigged mesh** | **Seconds** | **High (any style)** | **Full** |

### 9.2 Strata's Differentiators

1. **Artist authority.** Every pixel of the 3D character comes from the artist's painting. AI fills gaps; it never invents the design. This is the opposite of StdGEN/TripoSG where AI generates the character.

2. **Style agnosticism.** Template deformation works identically for anime, western animation, pixel art, painterly, chibi, or realistic styles. Neural methods are trained on specific distributions and degrade outside them.

3. **Determinism.** Same paintings always produce the same mesh. No stochastic variation, no "generate and pray."

4. **Integrated pipeline.** Painting → mesh → rig → animation in one tool. Competitors produce a mesh that must be manually imported into Maya/Blender for rigging and animation.

5. **Speed.** Template deformation + projection is computationally trivial compared to neural reconstruction. Mesh generation in seconds, not minutes.

---

## 10. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Measurement extraction errors from 3/4 view | Medium | Height cross-validation between views; artist approval step before mesh generation |
| Template library too limited for diverse designs | High | Start with 8 templates; add community template submission; fallback to neural reconstruction for edge cases |
| Texture seams visible in transition zones | Medium | MVPaint SSA seam smoothing; artist UV editing mode as escape hatch |
| Contour lines causing projection artifacts | Medium | DrawingSpinUp removal strategy; detection built into segmentation pipeline |
| Neural inpainting quality for unobserved regions | Medium | Keep as optional enhancement; symmetric mirror as reliable fallback |
| Artist resistance to drawing multiple views | Low | 3/4 view is natural; guided canvas with overlay makes it straightforward; v1.0 single-view path always available |

---

## 11. Open Questions for Implementation

1. **Template mesh format:** glTF, FBX, or custom binary? Needs to support embedded cage/lattice control structure.
2. **Deformation algorithm:** Cage-based (Mean Value Coordinates, Harmonic Coordinates) vs. lattice-based (FFD)? Cage gives more artist-intuitive control; lattice is simpler to implement.
3. **Texture resolution:** 1K, 2K, or 4K UV maps? 2K is the sweet spot for quality vs. performance.
4. **Neural inpainting model:** Train custom on Strata's art styles or use pre-trained diffusion model? Pre-trained is faster to ship; custom gives better results.
5. **3D preview renderer:** WebGL (runs everywhere), or native GPU (faster)? WebGL for cross-platform consistency.
6. **A-pose vs T-pose:** Templates should be in A-pose (arms at 45°) per industry standard and CharacterGen/StdGEN convention. A-pose produces better shoulder deformation than T-pose.

---

## 12. Research References

### Primary Papers (Directly Applicable)

- **StdGEN** — He et al., "Semantic-Decomposed 3D Character Generation from Single Images," CVPR 2025. arXiv:2411.05738
- **CharacterGen** — Peng et al., "Efficient 3D Character Generation from Single Images with Multi-View Pose Canonicalization," SIGGRAPH/TOG 2024. arXiv:2402.17214
- **UniRig** — Zhang et al., "One Model to Rig Them All: Diverse Skeleton Rigging with UniRig," SIGGRAPH/TOG 2025. arXiv:2504.12451
- **DrawingSpinUp** — "3D Animation from Single Character Drawings," SIGGRAPH Asia 2024
- **NOVA-3D** — Wang et al., "Non-overlapped Views for 3D Anime Character Reconstruction," 2024
- **MVPaint** — "Synchronized Multi-View Diffusion for Painting Anything 3D," 2024. arXiv:2411.02336
- **PAniC-3D** — Chen et al., "Stylized Single-view 3D Reconstruction from Portraits of Anime Characters," CVPR 2023. arXiv:2303.14587

### Foundation Models

- **TripoSG** — Li et al., "High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models," 2025. arXiv:2502.06608
- **Hunyuan3D 2.0** — Tencent, "Scaling Diffusion Models for High Resolution Textured 3D Assets Generation," 2025. arXiv:2501.12202
- **DiffSplat** — Lin et al., "Repurposing Image Diffusion Models for Scalable 3D Gaussian Splat Generation," ICLR 2025
- **DeformSplat** — Kim et al., "Rigidity-Aware 3D Gaussian Deformation from a Single Image," SIGGRAPH Asia 2025

### Texture & Style

- **TexPainter** — Zhang et al., "Generative Mesh Texturing with Multi-view Consistency," 2024. arXiv:2406.18539
- **SSMVtex** — Pagés et al., "Seamless, Static Multi-Texturing of 3D Meshes," Computer Graphics Forum 2015

---

## 13. Implementation Roadmap

### Phase 1: Foundation (v2.0-alpha)
- Build 3 core templates (T01 Standard, T05 Child, T06 Chibi)
- Multi-view painting canvas with front + 3/4 guide overlays
- Measurement extraction pipeline
- Basic cage deformation
- Simple angle-weighted texture projection
- 3D mesh preview renderer

### Phase 2: Quality (v2.0-beta)
- Complete template library (all 8 archetypes)
- Spatial-aware seam smoothing
- Contour line detection and removal
- View consistency validation
- Back view support
- 3D mesh export (glTF, FBX)

### Phase 3: Intelligence (v2.0)
- Neural inpainting for unobserved texture regions
- Automatic template selection from measurements
- UV texture editing mode
- Animation retargeting from 2.5D to 3D

### Phase 4: Advanced (v2.1+)
- Auto-rigging fallback for non-standard meshes
- 3/4 view angle auto-detection
- Community template marketplace
- AI-assisted back view generation from front + 3/4
- Multi-view consistency enforcement during painting (real-time preview)
