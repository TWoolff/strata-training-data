# FBAnimeHQ

~112K high-quality full-body anime character images.

## Download

- **URL:** https://huggingface.co/datasets/skytnt/fbanimehq
- **Size:** ~25 GB
- **Method:** HuggingFace CLI

```bash
./ingest/download_datasets.sh fbanimehq
```

## License

Check HuggingFace dataset card for license terms.

## Format

Full-body anime character images at high resolution. No segmentation or joint annotations included — useful for style diversity and augmentation reference.

## Strata Adapter

**Status:** Not yet implemented. Needs adapter to resize images and generate metadata. Segmentation and joints would require model inference.
