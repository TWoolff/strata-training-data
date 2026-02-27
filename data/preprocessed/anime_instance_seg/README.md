# anime-instance-segmentation

Instance segmentation dataset for anime characters.

## Download

- **URL:** https://github.com/dreMaz/AnimeInstanceSegmentationDataset
- **Size:** Varies
- **Method:** Clone repo, follow download instructions

```bash
./ingest/download_datasets.sh anime_instance_seg
```

## License

Check repository license.

## Format

Anime images with per-instance segmentation annotations. Multiple characters per image with instance-level masks.

## Strata Adapter

**Status:** Not yet implemented. Needs adapter to extract single-character crops and convert instance masks. Instance-level (not part-level) annotations.
