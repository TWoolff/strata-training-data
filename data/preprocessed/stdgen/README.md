# StdGEN (Anime3D++)

10.8K annotated 3D anime characters with semantic part annotations. CVPR 2025.

## Download

- **URL:** https://github.com/hyz317/StdGEN
- **HuggingFace:** https://huggingface.co/hyz317/StdGEN
- **Size:** Varies (model weights + data)
- **Method:** Clone repo + HuggingFace download

```bash
./ingest/download_datasets.sh stdgen
```

## License

Check repository license. Academic use.

## Format

Rendering scripts and train/test splits for 3D anime character generation. Includes semantic part annotations that may map to Strata regions.

## Strata Adapter

**Status:** Not yet implemented. Needs adapter to convert semantic part annotations to Strata 19-region format.
