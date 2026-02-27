"""Entry point for running the pipeline via Blender.

Usage::

    blender --background --python run_pipeline.py -- \\
      --input_dir ./data/fbx/ \\
      --output_dir ./output/segmentation/ \\
      --styles flat \\
      --resolution 512
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``pipeline`` is importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.generate_dataset import main

if __name__ == "__main__":
    main()
