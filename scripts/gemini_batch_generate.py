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
    "anime illustration",
    "game concept art",
    "comic book art with bold outlines",
    "soft pastel illustration",
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
    "post-apocalyptic survivor",
    "pirate adventure",
    "martial arts wuxia",
    "fairy tale",
]

BODY_TYPES = [
    "slim and athletic",
    "muscular and broad-shouldered",
    "stocky and sturdy",
    "tall and lanky",
    "short and compact",
    "heavyset and powerful",
    "petite and nimble",
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
    "standing with one hand on hip",
    "walking forward mid-stride",
    "standing with arms crossed",
]

# Outfit/silhouette modifiers — these create varied shapes for segmentation training
OUTFITS = [
    "",  # no specific outfit (let genre determine it)
    "wearing a long flowing cape",
    "wearing heavy plate armor with pauldrons",
    "wearing a hooded cloak",
    "wearing a long dress or robe",
    "wearing minimal clothing showing skin",
    "wearing bulky winter gear with a fur collar",
    "wearing a backpack and utility belt",
    "carrying a large weapon on their back",
    "wearing a hat or helmet with distinctive silhouette",
]

# Hair styles — important for hair_back segmentation class
HAIR_STYLES = [
    "",  # no specific hair
    "with long flowing hair past their shoulders",
    "with short cropped hair",
    "with a large ponytail or braid",
    "with big curly or afro hair",
    "bald or shaved head",
    "with twin tails or pigtails",
    "with hair tied up in a bun",
]


def _build_prompt(
    style: str,
    genre: str,
    body_type: str,
    gender: str,
    skin_tone: str,
    pose: str,
    outfit: str = "",
    hair: str = "",
) -> str:
    """Build a structured character generation prompt."""
    base = (
        f"Full-body {style} of a {genre} character. "
        f"The character is {gender}-presenting, {body_type}, with {skin_tone}. "
    )
    if hair:
        base += f"The character is {hair}. "
    if outfit:
        base += f"The character is {outfit}. "
    base += (
        f"The character is {pose}. "
        f"Single character only, centered in the frame, full body visible from head to feet. "
        f"Solid plain white background. No text, no watermark, no UI elements, no props on the ground."
    )
    return base


def _build_prompt_list(count: int, seed: int = 42) -> list[dict]:
    """Build a list of diverse prompts by sampling from all axes.

    Samples randomly from all axes rather than full cartesian product
    (which would be 12*12*7*3*5*8*10*8 = 9.7M combinations). This ensures
    every image is a unique combination while keeping diversity high.

    Returns list of dicts with 'prompt' and 'tags' keys.
    """
    rng = random.Random(seed)

    prompts = []
    seen: set[tuple] = set()

    while len(prompts) < count:
        combo = (
            rng.choice(ART_STYLES),
            rng.choice(GENRES),
            rng.choice(BODY_TYPES),
            rng.choice(GENDER_PRESENTATIONS),
            rng.choice(SKIN_TONES),
            rng.choice(POSES),
            rng.choice(OUTFITS),
            rng.choice(HAIR_STYLES),
        )
        if combo in seen:
            continue
        seen.add(combo)

        style, genre, body_type, gender, skin_tone, pose, outfit, hair = combo
        prompt = _build_prompt(style, genre, body_type, gender, skin_tone, pose, outfit, hair)
        tags = {
            "style": style,
            "genre": genre,
            "body_type": body_type,
            "gender": gender,
            "skin_tone": skin_tone,
            "pose": pose,
            "outfit": outfit,
            "hair": hair,
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


def _load_env_key(key_names: list[str]) -> str:
    """Load an API key from environment or .env file."""
    for name in key_names:
        val = os.environ.get(name, "")
        if val:
            return val
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            for name in key_names:
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _generate_single_gemini(
    client,
    prompt: str,
    output_path: Path,
    model: str,
) -> bool:
    """Generate a single image via Gemini API. Returns True on success."""
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


def _generate_single_hf(
    hf_token: str,
    prompt: str,
    output_path: Path,
    model: str,
) -> bool:
    """Generate a single image via HuggingFace Inference API. Returns True on success."""
    import requests

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}

    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{model}",
        headers=headers,
        json=payload,
        timeout=180,
    )

    if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
        output_path.write_bytes(response.content)
        return True

    if response.status_code == 429:
        raise Exception(f"429 Rate limited: {response.text[:200]}")

    logger.warning("  HF API returned %d: %s", response.status_code, response.text[:200])
    return False


def run_batch(
    output_dir: Path,
    count: int = 1000,
    seed: int = 42,
    model: str = "gemini-2.5-flash-image",
    delay: float = 1.1,
    backend: str = "auto",
) -> GenerationResult:
    """Generate a batch of character images.

    Args:
        output_dir: Directory to save raw PNG images + manifest.
        count: Number of images to generate.
        seed: Random seed for prompt diversity.
        model: Model ID (Gemini or HuggingFace model name).
        delay: Seconds between API calls (rate limiting).
        backend: "gemini", "huggingface", or "auto" (try gemini, fall back to HF).

    Returns:
        GenerationResult with counts of generated/skipped/failed.
    """
    # Determine backend
    use_hf = False
    hf_token = ""
    gemini_client = None

    if backend in ("auto", "gemini"):
        api_key = _load_env_key(["GOOGLE_API_KEY", "GEMINI_KEY"])
        if api_key:
            try:
                from google import genai
                gemini_client = genai.Client(api_key=api_key)
                # Quick test
                if backend == "auto":
                    try:
                        gemini_client.models.generate_content(
                            model=model, contents="test",
                        )
                    except Exception as e:
                        if "429" in str(e) or "not available" in str(e).lower():
                            logger.info("Gemini unavailable (%s), falling back to HuggingFace", str(e)[:80])
                            gemini_client = None
            except ImportError:
                gemini_client = None

    if gemini_client is None or backend == "huggingface":
        hf_token = _load_env_key(["HF_TOKEN"])
        if not hf_token:
            raise ValueError(
                "No working backend. Set GEMINI_KEY or HF_TOKEN in .env"
            )
        use_hf = True
        if model == "gemini-2.5-flash-image":
            model = "black-forest-labs/FLUX.1-schnell"
        logger.info("Using HuggingFace backend with model: %s", model)

    if not use_hf:
        logger.info("Using Gemini backend with model: %s", model)

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
            if use_hf:
                success = _generate_single_hf(hf_token, prompt, image_path, model)
            else:
                success = _generate_single_gemini(gemini_client, prompt, image_path, model)
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
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "gemini", "huggingface"],
        default="auto",
        help="Backend: 'gemini', 'huggingface' (FLUX.1-schnell), or 'auto' (default: auto).",
    )
    args = parser.parse_args()

    result = run_batch(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        model=args.model,
        delay=args.delay,
        backend=args.backend,
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
