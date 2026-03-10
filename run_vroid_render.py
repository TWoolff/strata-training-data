"""Blender entry point: render VRoid CC0 characters with 22-class seg masks.

Run inside Blender's Python environment::

    /Applications/Blender.app/Contents/MacOS/Blender --background --python run_vroid_render.py -- \\
        --vroid_dir /Volumes/TAMWoolff/data/raw/vroid_cc0 \\
        --output_dir /Volumes/TAMWoolff/data/output/vroid_cc0 \\
        --styles flat,unlit,textured \\
        --angles all

Arguments after ``--`` are parsed by this script; everything before is
consumed by Blender itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so pipeline package is importable inside Blender.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pipeline.generate_dataset import main  # noqa: E402

main()
