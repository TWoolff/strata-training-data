# Issue #14: Implement render-time Blender shader styles: flat, cel/toon, and unlit

## Understanding
- Add three render-time art styles to `renderer.py` that modify Blender materials before the color render pass
- **Flat**: Diffuse BSDF only, flat shading (no smooth gradients), single color per face
- **Cel/toon**: Shader-to-RGB → ColorRamp with 2–3 hard stops, plus Freestyle black outlines
- **Unlit**: Emission shader with character's base color, zero lighting influence
- Styles must be **non-destructive** — original materials restored after rendering
- Type: **new feature**

## Approach
1. Add cel-style constants (outline thickness, toon color stops) to `config.py`
2. Add three style-application functions to `renderer.py`:
   - `apply_flat_style(meshes)` → Override materials with Diffuse BSDF, set flat shading
   - `apply_cel_style(scene, meshes)` → Shader-to-RGB + ColorRamp node tree, enable Freestyle
   - `apply_unlit_style(scene, meshes)` → Emission shader, disable all lights
3. Add `restore_original_style(scene, meshes, backup)` to clean up after rendering
4. Add a helper `_extract_base_color(material)` to pull albedo from existing Principled BSDF or use fallback
5. Modify `generate_dataset.py` to call the appropriate style function before each color render, then restore

### Why this approach
- Each style is a self-contained function — easy to test independently
- Material backup/restore already exists in `generate_dataset.py` (`_backup_materials` / `_restore_materials`) — we extend that pattern
- The `_extract_base_color` helper centralizes color extraction logic used by all three styles

### Style details
- **Flat**: For each mesh, set `mesh.data.use_auto_smooth = False`, override material with Diffuse BSDF using extracted base color, no specular
- **Cel**: Build node tree: Principled BSDF → Shader to RGB → ColorRamp (3 stops, constant interpolation: shadow, mid, highlight) → Output. Enable Freestyle with ~2px line thickness
- **Unlit**: Replace all materials with Emission BSDF using base color, energy=1.0. Disable sun lamp in scene

## Files to Modify
- `pipeline/config.py` — Add cel/toon constants (outline thickness, color ramp factors)
- `pipeline/renderer.py` — Add style application/restoration functions
- `pipeline/generate_dataset.py` — Integrate style functions into the per-style color render loop

## Risks & Edge Cases
- Characters without Principled BSDF (e.g., custom node trees) — fallback to a default gray
- Characters with image textures — extract dominant color or use texture node in shader
- Freestyle outlines may behave differently in headless mode — need to test
- `use_auto_smooth` was removed in Blender 4.1+ — need to handle flat shading via mesh attribute instead
- Cel style modifies scene-level settings (Freestyle) — must disable after rendering

## Open Questions
- Should flat style use the character's original colors or a single uniform color? → Issue says "accept character's existing texture/color information as base", so extract existing colors
- How to handle textured materials in cel/unlit? → Use the texture as input to the shader chain if available, otherwise use base color

## Implementation Notes

### What was implemented
- **config.py**: Added `RENDER_TIME_STYLES` (set), `CEL_RAMP_STOPS`, `CEL_OUTLINE_THICKNESS`, `DEFAULT_BASE_COLOR`
- **renderer.py**: Added 8 new functions:
  - `_extract_base_color()` — pulls albedo from Principled BSDF, fallback to gray
  - `_get_image_texture_node()` — finds image texture connected to Base Color input
  - `_wire_color_source()` — shared helper that wires texture or solid color into any shader input (deduplicates pattern used by all 3 styles)
  - `apply_flat_style()` — Diffuse BSDF, flat face shading, full roughness
  - `apply_cel_style()` — Diffuse→ShaderToRGB→ColorRamp(CONSTANT)→MixRGB(MULTIPLY with base color)→Emission→Output, plus Freestyle outlines
  - `apply_unlit_style()` — Emission shader, hides all lights
  - `restore_style()` — cleans up temp materials + scene-level state (Freestyle, lights)
  - `apply_style()` — dispatcher that routes to the correct style function
- **generate_dataset.py**: Color render loop now restores original materials + applies/restores style per iteration

### Design decisions
- Used `polygon.use_smooth = False` for flat shading instead of deprecated `use_auto_smooth` (Blender 4.1+)
- Cel shader uses white Diffuse BSDF for pure lighting capture, then multiplies quantized result with base color via MixRGB → Emission output (avoids double-lighting)
- `restore_style()` takes no `meshes` param — material slots are restored by `_restore_materials` in the orchestrator; `restore_style` only handles scene-level cleanup and temp material deletion
- Textured materials are supported: texture images are re-referenced in new node trees rather than extracting a dominant color

### Follow-up work
- Post-render styles (pixel art, painterly, sketch) are tracked as separate issues
- Freestyle outline behavior in headless EEVEE should be validated with actual renders
- CEL_RAMP_STOPS values may need tuning after visual review of renders
