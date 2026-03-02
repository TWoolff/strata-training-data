"""Blender entry point for the UniRig / Rig-XL ingest adapter.

Must be run inside Blender's Python runtime:

    blender --background --python run_unirig.py -- \\
        --input_dir ./data/preprocessed/unirig/rigxl \\
        --output_dir ./output/unirig \\
        --max_images 100

Or via run_ingest.py (which handles the Blender subprocess automatically):

    python run_ingest.py \\
        --adapter unirig \\
        --input_dir ./data/preprocessed/unirig/rigxl \\
        --output_dir ./output/unirig \\
        --max_images 100
"""

import argparse
import logging
import site
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``ingest`` and ``pipeline`` are importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Blender's bundled Python doesn't include user site-packages by default.
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments after Blender's own -- separator."""
    # Blender passes its own args before --; ours come after.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="UniRig ingest adapter (Blender runtime)")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--only_new", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    from ingest.unirig_adapter import convert_directory

    max_ex = args.max_images if args.max_images > 0 else None

    logger.info(
        "UniRig adapter starting: input=%s output=%s max=%s",
        args.input_dir,
        args.output_dir,
        max_ex,
    )

    stats = convert_directory(
        args.input_dir,
        args.output_dir,
        max_examples=max_ex,
        only_new=args.only_new,
    )

    print("\nUniRig ingestion complete:")
    print(f"  {stats.summary()}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
