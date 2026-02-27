# Issue #97: Create docs/preprocessed-datasets.md cataloging all external datasets

## Understanding
- Create a central catalog document for all 9 pre-processed external datasets
- Must document: citation, download URL, license, size, format, content, adapter status, limitations, licensing risk
- PRD explicitly designates `docs/preprocessed-datasets.md` as the canonical location
- Cross-reference with existing `ingest/` adapter scripts

## Approach
- Create a comprehensive markdown document with a summary table + detailed per-dataset sections
- Use consistent structure for each dataset entry
- Pull data from: web research, PRD sections 9/12, `ingest/download_datasets.sh`, `ingest/nova_human_adapter.py`
- Include licensing risk assessment aligned with PRD §12 risk table

## Files to Modify
- `docs/preprocessed-datasets.md` — NEW FILE (the deliverable)

## Research Summary

| Dataset | Paper | License | Size | Adapter |
|---------|-------|---------|------|---------|
| NOVA-Human | Wang et al. 2024, arXiv 2405.12505 | Research (per-model VRoid terms) | ~50-80 GB, 10.2K chars | `nova_human_adapter.py` EXISTS |
| StdGEN / Anime3D++ | He et al. CVPR 2025 | Apache-2.0 (code), research (data) | Varies, 10.8K chars | Planned |
| UniRig / Rig-XL | Zhang et al. SIGGRAPH 2025 | MIT | ~20 GB, 14K meshes | Planned |
| AnimeRun | NeurIPS 2022 | Not specified | ~5 GB, ~8K pairs | Planned |
| LinkTo-Anime | Feng et al. 2025, arXiv 2506.02733 | Research | ~10 GB, ~29K frames | Planned |
| FBAnimeHQ | skytnt (HuggingFace) | Unclear (Danbooru-sourced) | ~25 GB, 112K images | Not started |
| anime-segmentation | SkyTNT (GitHub/HF) | Apache-2.0 | ~18 GB | Not started |
| AnimeInstanceSeg | Lin et al. 2023, arXiv 2312.01943 | Not specified | Varies | Not started |
| CharacterGen | Peng et al. 2024, ACM TOG | Apache-2.0 (code) | Varies | Not started |

## Risks & Edge Cases
- Licensing is the main concern — most datasets are research-only
- Some datasets may go offline; pin versions/commits
- Download sizes are estimates; actual may vary
- FBAnimeHQ sourced from Danbooru — unclear licensing for commercial use

## Open Questions
- None — issue scope is clear documentation work
