"""Batch-generate diverse 2D character images via the Gemini API.

Uses the google-genai SDK with gemini-2.5-flash-image (Nano Banana 2) to
generate full-body character illustrations across multiple art styles, genres,
body types, and poses.  Output: raw PNG images ready for the
gemini_diverse_adapter ingestion pipeline.

Usage::

    export GOOGLE_API_KEY='...'
    python scripts/gemini_batch_generate.py \
        --output-dir /Volumes/TAMWoolff/data/raw/gemini_diverse \
        --count 1000

    # Resume after interruption (skips existing images):
    python scripts/gemini_batch_generate.py \
        --output-dir /Volumes/TAMWoolff/data/raw/gemini_diverse \
        --count 1000

    # Generate a small test batch:
    python scripts/gemini_batch_generate.py \
        --output-dir ./output/gemini_test \
        --count 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diversity axes — combinatorial sampling produces unique prompts
# ---------------------------------------------------------------------------

ART_STYLES = [
    "flat vector illustration",
    "cel-shaded cartoon",
    "painterly digital painting",
    "watercolor illustration",
    "pixel art",
    "ink sketch with clean lineart",
    "realistic digital art",
    "chibi / super-deformed",
]

GENRES = [
    "fantasy RPG",
    "science fiction",
    "steampunk",
    "modern urban",
    "historical medieval",
    "superhero comic book",
    "dark gothic horror",
    "ancient mythology",
]

BODY_TYPES = [
    "slim and athletic",
    "muscular and broad-shouldered",
    "stocky and sturdy",
    "tall and lanky",
    "short and compact",
]

GENDER_PRESENTATIONS = [
    "masculine",
    "feminine",
    "androgynous",
]

SKIN_TONES = [
    "light skin",
    "medium skin",
    "tan skin",
    "brown skin",
    "dark skin",
]

POSES = [
    "standing facing forward in a neutral pose",
    "standing in a three-quarter view",
    "in a dynamic action pose",
    "in a relaxed casual stance",
    "standing with arms slightly away from body (A-pose)",
]


def _build_prompt(
    style: str,
    genre: str,
    body_type: str,
    gender: str,
    skin_tone: str,
    pose: str,
) -> str:
    """Build a structured character generation prompt."""
    return (
        f"Full-body {style} of a {genre} character. "
        f"The character is {gender}-presenting, {body_type}, with {skin_tone}. "
        f"The character is {pose}. "
        f"Single character only, centered in the frame, full body visible from head to feet. "
        f"Solid plain white background. No text, no watermark, no UI elements, no props on the ground."
    )


def _build_prompt_list(count: int, seed: int = 42) -> list[dict]:
    """Build a list of diverse prompts by sampling from all axes.

    Returns list of dicts with 'prompt' and 'tags' keys.
    """
    rng = random.Random(seed)

    # Build all combinations
    axes = list(
        itertools.product(
            ART_STYLES,
            GENRES,
            BODY_TYPES,
            GENDER_PRESENTATIONS,
            SKIN_TONES,
            POSES,
        )
    )
    # 8 * 8 * 5 * 3 * 5 * 5 = 24,000 combinations — plenty for 1000
    rng.shuffle(axes)

    prompts = []
    for style, genre, body_type, gender, skin_tone, pose in axes[:count]:
        prompt = _build_prompt(style, genre, body_type, gender, skin_tone, pose)
        tags = {
            "style": style,
            "genre": genre,
            "body_type": body_type,
            "gender": gender,
            "skin_tone": skin_tone,
            "pose": pose,
        }
        prompts.append({"prompt": prompt, "tags": tags})

    return prompts


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Summary of a batch generation run."""

    generated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _generate_single(
    client,
    prompt: str,
    output_path: Path,
    model: str,
) -> bool:
    """Generate a single image. Returns True on success."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    if not response.parts:
        return False

    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save(str(output_path))
            return True

    return False


def run_batch(
    output_dir: Path,
    count: int = 1000,
    seed: int = 42,
    model: str = "gemini-2.5-flash-image",
    delay: float = 1.1,
) -> GenerationResult:
    """Generate a batch of character images via Gemini API.

    Args:
        output_dir: Directory to save raw PNG images + manifest.
        count: Number of images to generate.
        seed: Random seed for prompt diversity.
        model: Gemini model ID for image generation.
        delay: Seconds between API calls (rate limiting).

    Returns:
        GenerationResult with counts of generated/skipped/failed.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai SDK not installed. Run: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get(
        "GEMINI_KEY", ""
    )
    if not api_key:
        # Try .env file
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY=") or line.startswith(
                    "GEMINI_KEY="
                ):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_KEY not set. Export it or add to .env file."
        )

    client = genai.Client(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build prompts
    prompts = _build_prompt_list(count, seed)
    manifest_path = output_dir / "manifest.json"

    # Load existing manifest for resume support
    existing_manifest: dict[str, dict] = {}
    if manifest_path.exists():
        existing_manifest = {
            item["filename"]: item
            for item in json.loads(manifest_path.read_text())
        }

    result = GenerationResult()
    manifest_items: list[dict] = list(existing_manifest.values())

    for i, entry in enumerate(prompts):
        filename = f"gemini_{i:04d}.png"
        image_path = output_dir / filename

        # Skip if already generated
        if image_path.exists() and filename in existing_manifest:
            result.skipped += 1
            continue

        prompt = entry["prompt"]
        tags = entry["tags"]

        logger.info(
            "[%d/%d] Generating %s — %s / %s",
            i + 1,
            count,
            filename,
            tags["style"],
            tags["genre"],
        )

        try:
            success = _generate_single(client, prompt, image_path, model)
            if success:
                result.generated += 1
                manifest_entry = {
                    "filename": filename,
                    "index": i,
                    "prompt": prompt,
                    **tags,
                }
                manifest_items.append(manifest_entry)

                # Save manifest after each successful generation (crash-safe)
                manifest_path.write_text(
                    json.dumps(manifest_items, indent=2, ensure_ascii=False)
                )
            else:
                result.failed += 1
                result.errors.append(f"{filename}: no image in response")
                logger.warning("  No image returned for %s", filename)

        except Exception as e:
            result.failed += 1
            error_msg = f"{filename}: {e}"
            result.errors.append(error_msg)
            logger.error("  Error generating %s: %s", filename, e)

            # Back off on errors (might be rate limit)
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning("  Rate limited — waiting 60s...")
                time.sleep(60)

        # Rate limiting
        if result.generated > 0 or result.failed > 0:
            time.sleep(delay)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate diverse character images via Gemini API.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save raw PNG images.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of images to generate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt diversity (default: 42).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-image",
        help="Gemini model ID (default: gemini-2.5-flash-image).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.1,
        help="Seconds between API calls for rate limiting (default: 1.1).",
    )
    args = parser.parse_args()

    result = run_batch(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        model=args.model,
        delay=args.delay,
    )

    print("\nBatch generation complete:")
    print(f"  Generated: {result.generated}")
    print(f"  Skipped:   {result.skipped}")
    print(f"  Failed:    {result.failed}")
    print(f"  Total:     {result.generated + result.skipped + result.failed}")
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for err in result.errors[:10]:
            print(f"  - {err}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more")


if __name__ == "__main__":
    main()
