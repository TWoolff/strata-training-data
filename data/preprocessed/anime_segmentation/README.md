# anime-segmentation

Foreground/background segmentation masks for anime characters.

## Download

- **URL:** https://huggingface.co/skytnt/anime-segmentation
- **Size:** Varies
- **Method:** HuggingFace CLI

```bash
./ingest/download_datasets.sh anime_segmentation
```

## License

Check HuggingFace model card for license terms.

## Format

Pre-trained model and/or dataset for anime character foreground/background segmentation. Binary masks (not semantic part segmentation).

## Strata Adapter

**Status:** Not yet implemented. Binary fg/bg masks could supplement training data but lack 19-region part annotations.
