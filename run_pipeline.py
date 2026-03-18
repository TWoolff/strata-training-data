"""Launcher for Blender — avoids relative import issues.

Usage::

    blender --background --python run_pipeline.py -- \
      --input_dir ./data/fbx/ \
      --output_dir ./output/ \
      --angles "front,three_quarter,back" \
      --styles textured \
      --resolution 512
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so pipeline/ is importable as a package
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.generate_dataset import main

main()
