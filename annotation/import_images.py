"""Batch-import 2D character images into a Label Studio project.

Resizes images to 512×512 (letterboxed with transparency), then uploads
them via the Label Studio API.  Each image becomes one annotation task.

Usage::

    python -m annotation.import_images \
        --image_dir ./data/sprites/ \
        --output_dir ./data/sprites_resized/ \
        --ls_url http://localhost:8080 \
        --ls_token <YOUR_API_TOKEN> \
        --project_id 1

If ``--ls_url`` is omitted, only the resize step runs (useful for
preparing images before importing them manually through the UI).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import RENDER_RESOLUTION

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------


def resize_and_pad(
    img: Image.Image,
    target_size: int = RENDER_RESOLUTION,
) -> Image.Image:
    """Resize an image to fit within a square, padding with transparency.

    Maintains aspect ratio.  The image is centered on a transparent
    background of ``target_size × target_size``.

    Args:
        img: Source image (any mode — converted to RGBA).
        target_size: Output width and height in pixels.

    Returns:
        RGBA image of exactly ``target_size × target_size``.
    """
    img = img.convert("RGBA")
    w, h = img.size

    scale = target_size / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def prepare_images(
    image_dir: Path,
    output_dir: Path,
    target_size: int = RENDER_RESOLUTION,
) -> list[Path]:
    """Resize all images in a directory and save to output_dir.

    Args:
        image_dir: Directory containing source images.
        output_dir: Directory for resized 512×512 PNGs.
        target_size: Output resolution.

    Returns:
        List of output paths for successfully processed images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not source_files:
        logger.warning("No image files found in %s", image_dir)
        return []

    logger.info("Processing %d images from %s", len(source_files), image_dir)

    output_paths: list[Path] = []
    for src in source_files:
        try:
            img = Image.open(src)
            resized = resize_and_pad(img, target_size)

            out_path = output_dir / f"{src.stem}.png"
            resized.save(out_path, format="PNG", compress_level=9)
            output_paths.append(out_path)
            logger.debug("Resized %s → %s", src.name, out_path)
        except Exception:
            logger.exception("Failed to process %s", src)

    logger.info("Prepared %d / %d images → %s", len(output_paths), len(source_files), output_dir)
    return output_paths


# ---------------------------------------------------------------------------
# Label Studio upload
# ---------------------------------------------------------------------------


def upload_to_label_studio(
    image_paths: list[Path],
    ls_url: str,
    api_token: str,
    project_id: int,
) -> int:
    """Upload prepared images to a Label Studio project.

    Uses the Label Studio REST API directly (no SDK dependency).

    Args:
        image_paths: List of 512×512 PNG paths to upload.
        ls_url: Label Studio server URL (e.g. ``http://localhost:8080``).
        api_token: API access token.
        project_id: Target project ID.

    Returns:
        Number of successfully uploaded images.
    """
    import urllib.request

    ls_url = ls_url.rstrip("/")
    uploaded = 0

    for img_path in image_paths:
        try:
            # Upload the file to Label Studio storage
            with open(img_path, "rb") as f:
                file_data = f.read()

            boundary = "----StrataPipelineBoundary"
            body = (
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'
                    f"Content-Type: image/png\r\n\r\n"
                ).encode()
                + file_data
                + f"\r\n--{boundary}--\r\n".encode()
            )

            req = urllib.request.Request(
                f"{ls_url}/api/projects/{project_id}/import",
                data=body,
                headers={
                    "Authorization": f"Token {api_token}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req) as resp:
                if resp.status in (200, 201):
                    uploaded += 1
                    logger.debug("Uploaded %s", img_path.name)
                else:
                    logger.warning("Upload %s returned status %d", img_path.name, resp.status)

        except Exception:
            logger.exception("Failed to upload %s", img_path.name)

    logger.info("Uploaded %d / %d images to project %d", uploaded, len(image_paths), project_id)
    return uploaded


# ---------------------------------------------------------------------------
# Task JSON generation (alternative to API upload)
# ---------------------------------------------------------------------------


def generate_task_json(
    image_paths: list[Path],
    output_path: Path,
    image_url_prefix: str = "/data/local-files/?d=",
) -> Path:
    """Generate a Label Studio import JSON file for local file serving.

    This is an alternative to API upload — useful when images are served
    from Label Studio's local file storage.

    Args:
        image_paths: List of image file paths.
        output_path: Path to write the JSON task file.
        image_url_prefix: URL prefix for local file references.

    Returns:
        The output path.
    """
    tasks = [
        {"data": {"image": f"{image_url_prefix}{img_path.resolve()}"}} for img_path in image_paths
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Generated %d tasks → %s", len(tasks), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize 2D character images and import into Label Studio.",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing source character images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for resized 512x512 PNGs.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RENDER_RESOLUTION,
        help=f"Target resolution (default: {RENDER_RESOLUTION}).",
    )
    parser.add_argument(
        "--ls_url",
        type=str,
        default="",
        help="Label Studio server URL (omit to skip upload).",
    )
    parser.add_argument(
        "--ls_token",
        type=str,
        default="",
        help="Label Studio API token.",
    )
    parser.add_argument(
        "--project_id",
        type=int,
        default=1,
        help="Label Studio project ID (default: 1).",
    )
    parser.add_argument(
        "--generate_tasks",
        type=Path,
        default=None,
        help="Path to write Label Studio import JSON (alternative to API upload).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    # Step 1: Resize images
    image_paths = prepare_images(args.image_dir, args.output_dir, args.resolution)

    if not image_paths:
        logger.error("No images processed — exiting.")
        sys.exit(1)

    # Step 2: Upload or generate task JSON
    if args.ls_url and args.ls_token:
        upload_to_label_studio(
            image_paths,
            args.ls_url,
            args.ls_token,
            args.project_id,
        )
    elif args.generate_tasks:
        generate_task_json(image_paths, args.generate_tasks)
    else:
        logger.info(
            "No --ls_url or --generate_tasks specified. "
            "Images resized to %s — import them manually into Label Studio.",
            args.output_dir,
        )


if __name__ == "__main__":
    main()
