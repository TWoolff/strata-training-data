# LinkTo-Anime

~29K frames with skeleton annotations and optical flow from anime videos.

## Download

- **Reference:** arXiv 2506.02733
- **Size:** ~10 GB
- **Method:** Follow download instructions from the paper's project page

```bash
./ingest/download_datasets.sh linkto_anime
```

## License

Check paper and dataset terms. Academic use.

## Format

Anime frames with skeleton joint annotations and optical flow maps. Useful for drawn pose estimation training.

## Strata Adapter

**Status:** Not yet implemented. Needs adapter to convert skeleton annotations to Strata joint format and extract segmentation from skeleton data.
