"""Generate diverse 2D character illustrations locally via Stable Diffusion XL.

Uses DreamShaper XL on Apple Silicon (MPS) to generate full-body character
art across multiple styles, genres, body types, and poses.  Output: raw PNG
images ready for the gemini_diverse_adapter ingestion pipeline.

All generated images are AI-generated with no copyright holder.

Usage::

    # Generate 1000 images (~3-6 hours on M4):
    python scripts/local_generate.py \
        --output-dir /Volumes/TAMWoolff/data/raw/sd_characters \
        --count 1000

    # Quick test (5 images):
    python scripts/local_generate.py \
        --output-dir ./output/sd_test \
        --count 5

    # Resume after interruption (skips existing images):
    python scripts/local_generate.py \
        --output-dir /Volumes/TAMWoolff/data/raw/sd_characters \
        --count 1000

Requirements::

    pip install diffusers transformers accelerate safetensors
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diversity axes — combinatorial sampling
# ---------------------------------------------------------------------------

ART_STYLES = [
    "flat vector illustration",
    "cel-shaded cartoon",
    "painterly digital painting",
    "watercolor illustration",
    "chibi super-deformed cartoon",
    "ink sketch with clean lineart",
    "comic book illustration",
    "pixel art",
    "gouache painting",
    "colored pencil drawing",
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
    "western frontier",
    "post-apocalyptic",
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

CHARACTER_CLASSES = [
    "warrior with sword and shield",
    "mage with a staff",
    "rogue with dual daggers",
    "ranger with a bow",
    "knight in plate armor",
    "paladin with a glowing weapon",
    "druid in nature-themed clothing",
    "monk in martial arts robes",
    "pirate with a cutlass",
    "samurai in traditional armor",
    "ninja in dark clothing",
    "barbarian with a great axe",
    "merchant in fine clothing",
    "explorer in travel gear",
    "witch with a pointed hat",
    "assassin in hooded cloak",
    "bard with a lute",
    "blacksmith with a hammer",
    "noble in royal attire",
    "soldier in military uniform",
]


def _build_prompt(
    style: str,
    genre: str,
    body_type: str,
    gender: str,
    skin_tone: str,
    pose: str,
    character_class: str,
) -> str:
    """Build a structured character generation prompt for SDXL.

    Kept under 77 CLIP tokens to avoid truncation.
    """
    return (
        f"a single {genre} {character_class}, {style}, "
        f"{gender}-presenting, {body_type}, {skin_tone}, {pose}, "
        f"solo, only one person, correct face, white background"
    )


NEGATIVE_PROMPT = (
    "background, scenery, landscape, room, floor, ground, shadow on ground, "
    "gradient background, colored background, dark background, "
    "3d render, 3d model, CGI, photorealistic, hyperrealistic, photo, "
    "anime, manga, deformed, ugly, blurry, low quality, bad anatomy, "
    "bad eyes, cross-eyed, asymmetric eyes, wonky eyes, misaligned eyes, "
    "bad face, distorted face, extra fingers, bad hands, "
    "bad proportions, extra limbs, missing limbs, disfigured, mutated, "
    "watermark, text, signature, logo, frame, border, cropped, out of frame, "
    "worst quality, low resolution, multiple characters, multiple people, "
    "two people, group, crowd, couple, duo, pair, "
    "character sheet, turnaround, multiple views, multiple angles, reference sheet, "
    "model sheet, concept sheet, side view and front view, "
    "NSFW, nude, naked, explicit"
)


def _build_prompt_list(count: int, seed: int = 42) -> list[dict]:
    """Build diverse prompts by sampling from all axes."""
    rng = random.Random(seed)

    axes = list(
        itertools.product(
            ART_STYLES,
            GENRES,
            BODY_TYPES,
            GENDER_PRESENTATIONS,
            SKIN_TONES,
            POSES,
            CHARACTER_CLASSES,
        )
    )
    # 8*10*6*3*5*6*20 = 864,000 combinations
    rng.shuffle(axes)

    prompts = []
    for style, genre, body_type, gender, skin_tone, pose, char_class in axes[:count]:
        prompt = _build_prompt(style, genre, body_type, gender, skin_tone, pose, char_class)
        tags = {
            "style": style,
            "genre": genre,
            "body_type": body_type,
            "gender": gender,
            "skin_tone": skin_tone,
            "pose": pose,
            "character_class": char_class,
        }
        prompts.append({"prompt": prompt, "tags": tags})

    return prompts


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def run_batch(
    output_dir: Path,
    count: int = 1000,
    seed: int = 42,
    model_id: str = "Lykon/dreamshaper-xl-v2-turbo",
    steps: int = 8,
    guidance_scale: float = 2.0,
    width: int = 768,
    height: int = 1024,
) -> dict:
    """Generate character images locally via SDXL.

    Args:
        output_dir: Directory to save PNG images + manifest.
        count: Number of images to generate.
        seed: Random seed for prompt diversity.
        model_id: HuggingFace model ID.
        steps: Number of inference steps (turbo models use fewer).
        guidance_scale: CFG scale.
        width: Output image width.
        height: Output image height.

    Returns:
        Dict with generation statistics.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        logger.info("Using Apple Silicon MPS backend")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        logger.info("Using CUDA backend")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("Using CPU backend (will be slow)")

    logger.info("Loading model: %s", model_id)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    # Memory optimization for Mac
    pipe.enable_attention_slicing()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    # Load existing manifest for resume
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        existing = {
            item["filename"]: item
            for item in json.loads(manifest_path.read_text())
        }
        logger.info("Resuming — %d images already exist", len(existing))

    manifest_items: list[dict] = list(existing.values())
    prompts = _build_prompt_list(count, seed)

    stats = {"generated": 0, "skipped": 0, "failed": 0}

    for i, entry in enumerate(prompts):
        filename = f"sd_{i:04d}.png"
        image_path = output_dir / filename

        if image_path.exists() and filename in existing:
            stats["skipped"] += 1
            continue

        prompt = entry["prompt"]
        tags = entry["tags"]

        logger.info(
            "[%d/%d] %s — %s / %s / %s",
            i + 1, count, filename,
            tags["style"], tags["genre"], tags["character_class"],
        )

        try:
            gen = torch.Generator(device="cpu").manual_seed(seed + i)

            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=gen,
            )

            image = result.images[0]
            image.save(str(image_path), format="PNG")

            stats["generated"] += 1
            manifest_entry = {
                "filename": filename,
                "index": i,
                "prompt": prompt,
                **tags,
            }
            manifest_items.append(manifest_entry)

            # Save manifest periodically
            if stats["generated"] % 10 == 0:
                manifest_path.write_text(
                    json.dumps(manifest_items, indent=2, ensure_ascii=False)
                )

        except Exception as e:
            stats["failed"] += 1
            logger.error("Error generating %s: %s", filename, e)

    # Final manifest save
    manifest_path.write_text(
        json.dumps(manifest_items, indent=2, ensure_ascii=False)
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diverse character illustrations locally via SDXL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save PNG images.",
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
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Lykon/dreamshaper-xl-v2-turbo",
        help="HuggingFace model ID (default: dreamshaper-xl-v2-turbo).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Inference steps (default: 8 for turbo models).",
    )
    args = parser.parse_args()

    result = run_batch(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        model_id=args.model,
        steps=args.steps,
    )

    print("\nGeneration complete:")
    print(f"  Generated: {result['generated']}")
    print(f"  Skipped:   {result['skipped']}")
    print(f"  Failed:    {result['failed']}")


if __name__ == "__main__":
    main()
