# Issue #87: Build partial-to-complete texture projection training data generator

## Understanding
- Generate (partial_texture, complete_texture) training pairs for a neural inpainting model
- The inpainting model fills unobserved texture regions during 3D mesh texturing
- "Complete" texture = UV map baked from all 24 viewing angles (every 15°)
- "Partial" texture = UV map baked from only 3 views (front, three-quarter, back)
- Inpainting mask = binary mask showing where texture is missing (partial vs complete difference)
- Training pair: (partial_texture, inpainting_mask) → complete_texture
- This is a Blender bpy module — uses Blender's texture baking to project rendered views onto UV maps
- Type: new feature (Phase 4 mesh pipeline)

## Approach

### High-level strategy
Use Blender's built-in texture baking system to project camera views onto UV texture maps:

1. **Dense angle sampling**: Define 24 camera angles (every 15°, azimuth 0-345°) for complete coverage
2. **Partial view subset**: Use 3 views (front 0°, three-quarter 45°, back 180°) for partial texture
3. **UV bake process per view**:
   - Position camera at angle
   - Create a temporary image texture node on all materials
   - Use `bpy.ops.object.bake(type='DIFFUSE')` or render-and-project approach
4. **Accumulate coverage**: Track which UV texels have been filled from each view
5. **Inpainting mask**: Binary difference between partial and complete coverage

### Key design decision: Render-then-project vs Direct bake
- **Direct bake** (`bpy.ops.object.bake`): Blender bakes lighting/color from scene directly to UV. Requires Cycles engine. Simpler but slower.
- **Render-then-project**: Render from camera, then project rendered pixels back onto UV space using ray casting. More control but complex.

**Chosen approach: Direct bake via Cycles** — Blender's bake system handles UV projection natively. We switch to Cycles temporarily for baking (EEVEE doesn't support baking), bake diffuse color from each camera angle, and composite the results.

Actually, Blender's bake requires lighting setup and bakes the actual shading, which isn't what we want. We want to project camera-view renders onto UV space. The better approach:

**Revised approach: UV-space projection via vertex mapping**
For each camera view:
1. Render the color image from that angle (already supported)
2. For each mesh face visible from the camera:
   - Project face vertices to screen space (2D pixel coords)
   - Sample the rendered image at those pixel coords
   - Write those colors to the UV texture map at the face's UV coords
3. Use a z-buffer / visibility check to handle occlusion

This is essentially what Blender's "Bake from Active" or texture painting projection does. We can leverage `bpy.ops.object.bake(type='DIFFUSE')` with a carefully set up scene using Emission materials to capture pure color.

**Final approach: Cycles bake with Emission materials**
1. Ensure meshes have UV maps (auto-unwrap if needed via Smart UV Project)
2. Set up Emission-only materials (pure color, no lighting influence)
3. For each camera angle, use `bpy.ops.object.bake(type='EMIT')` with a margin to handle seams
4. Composite results: accumulate baked textures, tracking coverage per texel
5. Partial = composite of 3 views; Complete = composite of 24 views

Wait — `bpy.ops.object.bake` bakes the material as-is from all directions, not from a specific camera view. To bake from a specific view direction, we need to use Blender's "bake from active" or set up a hemispherical light from the camera direction.

**Actual final approach: Camera-projection texture painting**

The cleanest method:
1. Render color from each camera angle (already in pipeline)
2. Use `bpy.ops.paint.project_image()` or manual UV projection:
   - For each face, check visibility from camera via ray cast
   - Project visible face vertices to image space
   - Sample rendered image pixels
   - Write to UV texture at the face's UV coordinates
3. This is pure geometry — no baking system needed

For implementation without relying on Blender's paint operators (which need UI context), we'll do manual projection:

1. For each mesh and each polygon:
   - Check if face normal points toward camera (backface test)
   - Cast rays to verify no occlusion
   - Project vertices to camera image coords
   - Rasterize the triangle in UV space, sampling from the camera image
2. Build UV texture incrementally across views

This is the most reliable approach for `--background` mode.

## Files to Modify

### New files
- `mesh/scripts/texture_projection_trainer.py` — Main module
  - Dense angle constants (24 angles, partial subset)
  - UV unwrap helper (Smart UV Project fallback)
  - Visibility computation (backface + occlusion)
  - Camera-to-UV projection per view
  - Multi-view texture compositing
  - Inpainting mask computation
  - CLI entry point
- `tests/test_texture_projection_trainer.py` — Unit tests for pure-Python logic

### Modified files
- `pipeline/config.py` — Add texture projection constants:
  - `TEXTURE_DENSE_ANGLES` (24 angles, every 15°)
  - `TEXTURE_PARTIAL_ANGLES` (front, three-quarter, back)
  - `TEXTURE_RESOLUTION` (1024×1024 default)
  - `TEXTURE_BAKE_MARGIN` (pixel margin for UV seam bleeding)

## Risks & Edge Cases
- **UV seam artifacts**: Texels at UV seam boundaries may have discontinuities. Mitigate with margin/bleed pass.
- **No UV maps**: Some FBX characters may lack UV maps. Need Smart UV Project fallback.
- **Self-occlusion**: Fingers, hair, etc. may occlude body parts from certain angles. Visibility check handles this.
- **Thin geometry**: Hair planes, clothing ribbons may not project cleanly. Accept some noise.
- **Performance**: 24 renders + UV projection per character could be slow. Keep resolution configurable.
- **Backface culling**: Need consistent normal direction. Some imported meshes have flipped normals.
- **Background mode**: All ops must work without GUI. `bpy.ops.paint.project_image()` likely requires active viewport — avoid it.

## Open Questions
- Should we support elevation variation for the 24 views (e.g., slight downward tilt) or keep all at elevation=0? → Issue says every 15° azimuth only, so elevation=0.
- Should partial views be configurable or always front/3-4/back? → Start with hardcoded 3 views per issue spec, make it a config constant for future flexibility.
