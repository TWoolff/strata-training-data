# PRD: Segmentation Model v2 — Depth + Normals Heads

**Date**: March 7, 2026
**Status**: Ready for implementation
**Scope**: Strata Rust runtime changes to accept the new segmentation ONNX model

## Summary

The segmentation model is being upgraded from 3 output heads to 5 output heads. The `draw_order` output is replaced by Marigold-distilled `depth` and `normals` heads, providing higher-quality 3D surface information from a single forward pass. This PRD describes the required changes in the Strata desktop app (`src-tauri/src/ai/`).

## ONNX Contract Change

### Old Model (v1)
```
Input:  "input"             [1, 3, 512, 512]   float32 (ImageNet-normalized RGB)

Output: "segmentation"      [1, 22, 512, 512]  float32 (raw logits, argmax → region ID)
Output: "draw_order"        [1, 1, 512, 512]   float32 (sigmoid, 0=back 1=front)
Output: "confidence"        [1, 1, 512, 512]   float32 (sigmoid, 0-1)
```

### New Model (v2)
```
Input:  "input"             [1, 3, 512, 512]   float32 (ImageNet-normalized RGB)

Output: "segmentation"      [1, 22, 512, 512]  float32 (raw logits, argmax → region ID)
Output: "depth"             [1, 1, 512, 512]   float32 (sigmoid, 0=near 1=far)
Output: "normals"           [1, 3, 512, 512]   float32 (tanh, xyz in [-1, 1])
Output: "confidence"        [1, 1, 512, 512]   float32 (sigmoid, 0-1)
Output: "encoder_features"  [1, 960, 64, 64]   float32 (raw backbone activations)
```

### What Changed
| Aspect | Old | New |
|--------|-----|-----|
| `draw_order` | 1-channel sigmoid depth proxy from vertex Z-order | **Removed** |
| `depth` | — | 1-channel sigmoid, Marigold-quality monocular depth |
| `normals` | — | 3-channel tanh, Marigold-quality surface normals |
| `encoder_features` | Optional, sometimes missing | Always present, 960 channels at 64x64 |
| Total outputs | 3 (+ optional encoder_features) | 5 |

## Required Changes

### 1. `segmentation.rs` — Output Extraction

**File**: `src-tauri/src/ai/segmentation.rs`

#### 1a. Replace `draw_order` with `depth`

The current code extracts `draw_order` from the model outputs:
```rust
outputs.get("draw_order")
```

Replace with:
```rust
outputs.get("depth")
```

Processing is identical — both are single-channel sigmoid outputs clamped to [0, 1]. The semantic meaning changes (draw_order was a crude Z-ordering, depth is Marigold-quality monocular depth), but the data format is the same.

#### 1b. Add `normals` extraction

After depth extraction, add normals extraction:
```rust
let normals_output = outputs.get("normals");
```

Processing:
- Shape: `[1, 3, 512, 512]` — 3 channels (X, Y, Z) in NCHW layout
- Values in `[-1, 1]` (tanh activation)
- X: left/right, Y: up/down, Z: toward/away from camera
- Upscale to original image resolution using **bilinear** interpolation (not nearest-neighbor — normals are continuous)
- Store as `Vec<[f32; 3]>` or `Vec<f32>` (3 values per pixel)

#### 1c. Make `encoder_features` non-optional

Currently `encoder_features` returns `Option<EncoderFeatures>`. With v2 the model always outputs them. Change to always expect the output (but keep graceful fallback if missing for backward compat during transition).

The encoder features shape changes from variable to fixed: `[1, 960, 64, 64]`.

### 2. `SegmentationResult` Struct

**Current struct:**
```rust
pub struct SegmentationResult {
    pub region_map: Vec<u8>,
    pub draw_order_map: Vec<f32>,
    pub confidence_map: Vec<f32>,
    pub encoder_features: Option<EncoderFeatures>,
    pub width: u32,
    pub height: u32,
}
```

**New struct:**
```rust
pub struct SegmentationResult {
    pub region_map: Vec<u8>,           // [H*W] region IDs (0-21)
    pub depth_map: Vec<f32>,           // [H*W] depth values (0=near, 1=far)
    pub normals_map: Vec<f32>,         // [H*W*3] surface normals (x,y,z in [-1,1])
    pub confidence_map: Vec<f32>,      // [H*W] confidence (0-1)
    pub encoder_features: Option<EncoderFeatures>,  // [960, 64, 64] backbone activations
    pub width: u32,
    pub height: u32,
}
```

Changes:
- `draw_order_map` renamed to `depth_map` (same type, same range)
- `normals_map` added (3 floats per pixel: x, y, z)

### 3. Downstream Consumers

#### 3a. `inpainting.rs` — Occlusion Detection

**Current usage** (lines ~66-71): Uses `draw_order_map` to detect which pixels are in front/behind for occlusion pair generation.

**Required change**: Replace `result.draw_order_map` references with `result.depth_map`. The semantics are improved — Marigold depth is more accurate than the old vertex Z-ordering, so occlusion detection should improve with no logic changes.

#### 3b. Draw Order Rendering / Layer Sorting

If any UI code uses `draw_order_map` for rendering layer order or z-sorting of body parts:
- Replace with `depth_map`
- The depth values may be inverted compared to old draw_order (0=near vs 0=back). Verify the convention and add `1.0 - depth` if needed.

**Depth convention**: The new model uses sigmoid activation, trained against Marigold depth where 0=near and 1=far. The old `draw_order` used 0=back and 1=front. If your layer sorting assumes higher values = closer to camera, you'll need to invert: `depth_map.iter().map(|d| 1.0 - d)`.

#### 3c. New: Surface Normals for Shading/Lighting

The `normals_map` enables new features:
- **Relighting**: Apply directional/point lights to the character based on surface orientation
- **Rim lighting**: Detect silhouette edges (normals perpendicular to camera)
- **Texture inpainting guidance**: Normal direction helps predict texture continuity
- **3D mesh normal mapping**: Apply normals to the generated 3D mesh for better shading

Normal map convention:
- X: positive = right, negative = left
- Y: positive = up, negative = down
- Z: positive = toward camera, negative = away
- Values in [-1, 1] (tanh output)
- To convert to uint8 for display: `((n + 1.0) * 0.5 * 255.0) as u8`

#### 3d. `weights.rs` — Encoder Feature Sampling

No changes needed. The existing `build_diffusion_feature_tensor()` already handles arbitrary channel counts from encoder features. The channel count changes from variable to fixed 960, but this is handled dynamically.

### 4. Model File Replacement

**File**: `src-tauri/models/segmentation.onnx`

Replace with the new v2 model from training run 3. The file size will increase slightly due to the additional normals head (~3 conv layers). Expected size: ~12-14 MB (up from ~11 MB).

### 5. Backward Compatibility

During transition, the runtime should handle both v1 and v2 models gracefully:

```rust
// Try new output names first, fall back to old
let depth = outputs.get("depth")
    .or_else(|| outputs.get("draw_order"));

let normals = outputs.get("normals");  // None for v1 model
```

Once the v2 model is verified and deployed, remove the v1 fallback code.

### 6. Frontend/UI Changes (if applicable)

If the Strata React frontend displays draw order visualization:
- Rename "Draw Order" to "Depth" in any UI labels
- Add "Surface Normals" visualization option (render as RGB image)
- The depth map is higher quality — visual output should improve noticeably

### 7. Testing Checklist

- [ ] Load new `segmentation.onnx` without errors
- [ ] Verify 5 output tensors with correct names and shapes
- [ ] Verify depth map values in [0, 1]
- [ ] Verify normals map values in [-1, 1]
- [ ] Verify encoder_features shape is [1, 960, 64, 64]
- [ ] Run weight prediction with new encoder features — verify no regression
- [ ] Run inpainting with depth_map instead of draw_order_map
- [ ] Visual comparison: new depth should look sharper/more accurate than old draw_order
- [ ] Visual comparison: normals should show smooth surface orientation
- [ ] Benchmark: inference time should be similar (~5-10% increase acceptable)
- [ ] Test on all 7 Gemini benchmark characters

## Timeline

1. **Training run 3** produces the new model (this repo, ~4-5h on A100)
2. **Strata runtime changes** (~2-4h, mostly renaming + adding normals extraction)
3. **Integration testing** with benchmark characters (~1h)
4. **Ship** — update bundled model in `src-tauri/models/`
