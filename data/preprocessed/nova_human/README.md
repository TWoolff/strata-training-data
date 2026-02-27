# NOVA-Human

10.2K 3D anime characters rendered from VRoid Hub models. Each character includes 4 orthographic views (front/back/left/right) and 16 random-perspective views.

## Download

- **URL:** https://github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D
- **Size:** ~50–80 GB
- **Method:** Clone repo, then follow download instructions in README

```bash
./ingest/download_datasets.sh nova_human
```

## License

VRoid Hub models — check individual model licenses. NOVA-3D code is MIT licensed.

## Format

```
nova_human/
├── human_rutileE/
│   ├── ortho/              ← front/back orthographic renders
│   ├── ortho_mask/         ← foreground masks for ortho views
│   ├── ortho_xyza/         ← position + alpha data
│   ├── rgb/                ← 16 random-view perspective renders
│   ├── rgb_mask/           ← foreground masks for random views
│   ├── xyza/               ← position + alpha for random views
│   └── human_rutileE_meta.json
├── human_example2/
│   └── ...
└── ...
```

## Strata Adapter

**Status:** Implemented — `ingest/nova_human_adapter.py`

Converts ortho and RGB views to Strata format. NOVA-Human does NOT provide 19-region segmentation, joint positions, or draw order — these are flagged as missing in output metadata.
