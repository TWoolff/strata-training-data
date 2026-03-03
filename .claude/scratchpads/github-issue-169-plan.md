# Issue #169: Download MixamoLine240 and study geometric correspondence

## Understanding
- Research + download task — study a dataset's format for potential use in Strata's joint interpolation
- MixamoLine240: 240 Mixamo sequences rendered as cel-shaded line art with vertex correspondence
- Built on same Mixamo + Blender pipeline as Strata
- ICCV 2023, AnimeInbet project (lisiyao21/AnimeInbet)

## Dataset Details
- 100 train + 44 val + 100 test sequences = 244 total (listed as "240" in paper)
- 19,930 training frames
- Format: PNG line art images + JSON vertex correspondence labels
- License: CC BY-NC-SA 4.0 (non-commercial — cannot use for training, reference only)
- Download: Google Drive link from GitHub repo

## Key Format: Vertex Correspondence JSON
Each frame has a JSON label with:
1. **Vertex locations** — 2D positions of line art vertices
2. **Connection** — adjacency table (graph structure)
3. **Original index** — links vertices back to original 3D mesh

This is a graph-based representation of rasterized drawings, enabling geometric
inbetweening by matching vertices across frames.

## Relevance to Strata
- Shared Mixamo skeleton means direct bone name compatibility
- The vertex→graph→correspondence approach could inform joint interpolation
- Cel-shaded rendering similar to Strata's cel style

## Files to Modify
- `.claude/prd/strata-training-data-checklist.md` — lines 1160-1163

## Risks & Edge Cases
- CC BY-NC-SA license means reference only, not training data
- Google Drive download may require manual intervention

## Open Questions
- None — straightforward download + study task

## Implementation Notes

### Vertex Correspondence Format (from code study)

Each JSON label file contains three fields:
```json
{
  "vertex location": [[x1, y1], [x2, y2], ...],  // 2D pixel positions of line art vertices
  "connection": [[nb1, nb2], [nb3], ...],          // adjacency list (graph topology)
  "original index": [idx1, idx2, ...]              // 3D mesh vertex index per 2D vertex
}
```

**How correspondence works:**
- The `original index` field maps each 2D line art vertex back to its source vertex in the original 3D Mixamo mesh
- Two frames share correspondence when they have vertices with the same `original index` value
- `ids_to_mat(id1, id2)` builds a boolean match matrix: if frame1 vertex i has the same 3D index as frame2 vertex j, they correspond
- Unmatched vertices (appearing/disappearing due to occlusion) get motion estimated from graph neighbors

**Key insight for Strata:** This is vertex-level correspondence (thousands of points per frame), not joint-level (19 bones). The approach geometrizes raster line art into graphs and matches via shared 3D mesh ancestry. Strata could use this principle at the joint level:
- Instead of vertex→vertex matching, use bone→bone matching across frames
- The motion propagation from matched→unmatched nodes (via graph adjacency) is directly applicable to occluded joint prediction

### Dataset Structure
```
ml144_norm_100_44_split/
├── train/
│   ├── frames/{character}_{action}/  ← 720×720 PNG line art
│   └── labels/{character}_{action}/  ← JSON vertex correspondence
├── test/
│   ├── frames/...
│   └── labels/...
└── validation/
    ├── frames/...
    └── labels/...
```

Characters include: ganfaul, firlscout, jolleen, kachujin, knight, maria, michelle, peasant, timmy, uriel
Actions include: hip_hop, slash, breakdance, capoeira, fist_fight, flying, climb, running, reaction, magic, tripping

### Verified Dataset Statistics (from extracted data)
```
ml144_norm_100_44_split/     (61 GB)
├── train/   100 sequences, 19,930 frames
└── test/     44 sequences, 11,102 frames

ml100_norm/                  (36 GB)
└── all/     101 sequences, 18,230 frames
```
- Total: ~49,262 frames across both subsets
- Image format: 720×720 RGB PNG (cel-shaded line art)
- Labels: JSON with ~1,500–1,600 vertices per frame
- Vertex correspondence between consecutive frames: ~92% (remaining ~8% from occlusion changes)
- Characters: ganfaul, girlscout, jolleen, kachujin, knight, maria, michelle, peasant_girl, timmy, uriel, police, warrok, xbot, etc.
- Actions: breakdance, capoeira, chapa-giratoria, fist_fight, flying_knee_punch, slash, chip, corkscrew_evade, etc.
- Naming: `{action}_{character}` for ml144, `{action}_{character}` for ml100
- Storage: `/Volumes/TAMWoolff/data/preprocessed/mixamo_line240/` (external HD, 4.5GB zip + 97GB extracted)

### Decision: Reference Only (No Full Adapter)
Reasons:
1. **License: CC BY-NC-SA 4.0** — non-commercial, cannot use for Strata training
2. **Format mismatch** — vertex-level graphs aren't directly compatible with Strata's joint-based format
3. **Value is conceptual** — the correspondence approach and motion propagation algorithms are the useful part, not the raw data
4. The cel-shaded line art images could serve as visual references for Strata's sketch/cel style augmentation
