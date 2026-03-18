"""Generate character images with FLUX.1-schnell on a GPU instance.

Standalone script — only needs diffusers, torch, and Pillow.
Generates diverse 2D character illustrations for segmentation training.

Usage::

    pip install diffusers torch accelerate sentencepiece protobuf Pillow
    python generate_flux_gpu.py --output-dir ./generated --count 2000 --seed 43

    # Resume after interruption (skips existing images):
    python generate_flux_gpu.py --output-dir ./generated --count 2000 --seed 43
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diversity axes
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

OUTFITS = [
    "",
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

HAIR_STYLES = [
    "",
    "with long flowing hair past their shoulders",
    "with short cropped hair",
    "with a large ponytail or braid",
    "with big curly or afro hair",
    "bald or shaved head",
    "with twin tails or pigtails",
    "with hair tied up in a bun",
]


def _build_prompt(
    style: str, genre: str, body_type: str, gender: str,
    skin_tone: str, pose: str, outfit: str = "", hair: str = "",
) -> str:
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
    rng = random.Random(seed)
    prompts = []
    seen: set[tuple] = set()

    while len(prompts) < count:
        combo = (
            rng.choice(ART_STYLES), rng.choice(GENRES), rng.choice(BODY_TYPES),
            rng.choice(GENDER_PRESENTATIONS), rng.choice(SKIN_TONES),
            rng.choice(POSES), rng.choice(OUTFITS), rng.choice(HAIR_STYLES),
        )
        if combo in seen:
            continue
        seen.add(combo)
        style, genre, body_type, gender, skin_tone, pose, outfit, hair = combo
        prompt = _build_prompt(style, genre, body_type, gender, skin_tone, pose, outfit, hair)
        tags = {
            "style": style, "genre": genre, "body_type": body_type,
            "gender": gender, "skin_tone": skin_tone, "pose": pose,
            "outfit": outfit, "hair": hair,
        }
        prompts.append({"prompt": prompt, "tags": tags})
    return prompts


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate character images with FLUX.1-schnell")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=1, help="Images per batch (increase if VRAM allows)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (1-4 for schnell)")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    logger.info("Loading FLUX.1-schnell pipeline...")
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    # Use CPU offload to keep only the active component on GPU — essential for ≤24GB VRAM
    pipe.enable_model_cpu_offload()
    logger.info("Pipeline loaded with CPU offload.")

    # Build prompts
    prompts = _build_prompt_list(args.count, args.seed)

    # Load existing manifest for resume
    manifest_path = output_dir / "manifest.json"
    existing_manifest: dict[str, dict] = {}
    if manifest_path.exists():
        existing_manifest = {
            item["filename"]: item
            for item in json.loads(manifest_path.read_text())
        }

    manifest_items: list[dict] = list(existing_manifest.values())
    generated = 0
    skipped = 0
    failed = 0

    logger.info("Generating %d images (batch_size=%d, steps=%d, res=%d)",
                args.count, args.batch_size, args.steps, args.resolution)

    t_start = time.time()

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        batch_filenames = [f"gemini_{i + j:04d}.png" for j in range(len(batch))]

        # Skip if all already exist
        all_exist = all(
            (output_dir / fn).exists() and fn in existing_manifest
            for fn in batch_filenames
        )
        if all_exist:
            skipped += len(batch)
            continue

        batch_prompts = [entry["prompt"] for entry in batch]

        try:
            with torch.inference_mode():
                results = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=args.steps,
                    height=args.resolution,
                    width=args.resolution,
                    guidance_scale=0.0,  # schnell doesn't use guidance
                )

            for j, image in enumerate(results.images):
                idx = i + j
                filename = batch_filenames[j]
                image_path = output_dir / filename

                # Skip if already exists
                if image_path.exists() and filename in existing_manifest:
                    skipped += 1
                    continue

                image.save(image_path, "PNG")
                generated += 1

                manifest_entry = {
                    "filename": filename,
                    "index": idx,
                    "prompt": batch[j]["prompt"],
                    **batch[j]["tags"],
                }
                manifest_items.append(manifest_entry)

            # Save manifest periodically (every batch)
            manifest_path.write_text(
                json.dumps(manifest_items, indent=2, ensure_ascii=False)
            )

            elapsed = time.time() - t_start
            rate = generated / elapsed if elapsed > 0 else 0
            eta = (args.count - generated - skipped) / rate if rate > 0 else 0
            logger.info(
                "[%d/%d] Generated %d, skipped %d — %.1f img/s, ETA %.0fm",
                min(i + args.batch_size, len(prompts)), len(prompts),
                generated, skipped, rate, eta / 60,
            )

        except Exception as e:
            failed += len(batch)
            logger.error("Batch %d failed: %s", i, e)

    elapsed = time.time() - t_start
    logger.info(
        "\nDone! Generated: %d, Skipped: %d, Failed: %d (%.1f minutes)",
        generated, skipped, failed, elapsed / 60,
    )


if __name__ == "__main__":
    main()
