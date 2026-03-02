"""GitHub Live2D model scraper for Strata training data collection.

Searches GitHub for repositories containing Live2D model files (.moc3, .model3.json),
filters by permissive license, downloads qualifying models via sparse git checkout,
and organizes them into data/live2d/NNN/ directories.

The directory naming uses plain numeric IDs (001, 002, ...) because the downstream
Live2D renderer (pipeline/live2d_renderer.py) prepends "live2d_" to the directory
name when constructing char_id. So directory "001" → char_id "live2d_001".

Usage:
    # Dry run — search and report without downloading
    python run_live2d_scrape.py --dry_run

    # Download up to 50 models
    python run_live2d_scrape.py --max_models 50

    # Resume after interruption (skips already-downloaded models)
    python run_live2d_scrape.py

    # Include repos with no license for manual review
    python run_live2d_scrape.py --include_unverified --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ACCEPTED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-3.0",
    "CC-BY-SA-3.0",
    "0BSD",
    "Unlicense",
    "ISC",
}

SEARCH_DELAY_S = 2.5
CORE_API_DELAY_S = 0.5
RATE_LIMIT_BUFFER_S = 5.0
MAX_RETRIES = 3

MANIFEST_COLUMNS = [
    "model_id",
    "source",
    "url",
    "license",
    "license_verified",
    "fragment_count",
    "notes",
]

MODEL_FILE_EXTENSIONS = {".moc3", ".model3.json"}
TEXTURE_SUBDIRS = {"textures", "parts", "images"}


@dataclass
class RepoInfo:
    """Metadata for a GitHub repository containing Live2D models."""

    full_name: str
    html_url: str
    license_key: str
    default_branch: str
    size_kb: int
    model_dirs: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """A single Live2D model found within a repository."""

    model_id: str
    repo_full_name: str
    repo_url: str
    license_key: str
    license_verified: bool
    repo_path: str  # path within repo to the model directory
    fragment_count: int
    notes: str = ""


def _run_gh(args: list[str], *, retries: int = MAX_RETRIES) -> subprocess.CompletedProcess[str]:
    """Run a gh CLI command with retry logic for rate limits.

    Args:
        args: Arguments to pass to gh (e.g. ["api", "/repos/..."]).
        retries: Maximum number of retries on rate limit errors.

    Returns:
        Completed process result.

    Raises:
        subprocess.CalledProcessError: If the command fails after all retries.
    """
    for attempt in range(retries):
        result = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result

        stderr = result.stderr.lower()
        if "rate limit" in stderr or "403" in stderr or "429" in stderr:
            wait = SEARCH_DELAY_S * (2**attempt) + RATE_LIMIT_BUFFER_S
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %.0fs", attempt + 1, retries, wait
            )
            time.sleep(wait)
            continue

        # Non-rate-limit error — fail immediately
        result.check_returncode()

    # All retries exhausted
    logger.error("gh command failed after %d retries: gh %s", retries, " ".join(args))
    result.check_returncode()
    return result  # unreachable, but satisfies type checker


def _check_prerequisites() -> bool:
    """Verify that gh and git CLIs are available."""
    for cmd in ("gh", "git"):
        if shutil.which(cmd) is None:
            logger.error("%s CLI not found. Install it first.", cmd)
            return False

    # Check gh auth status
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gh is not authenticated. Run 'gh auth login' first.")
        return False

    return True


def _search_code(query: str, max_pages: int = 5) -> list[dict]:
    """Search GitHub code with pagination.

    Args:
        query: GitHub code search query string.
        max_pages: Maximum number of result pages to fetch.

    Returns:
        List of search result items (raw JSON dicts).
    """
    all_items: list[dict] = []
    for page in range(1, max_pages + 1):
        logger.info("Code search page %d: %s", page, query)
        result = _run_gh(
            [
                "api",
                "search/code",
                "-X",
                "GET",
                "-f",
                f"q={query}",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
            ]
        )
        data = json.loads(result.stdout)
        items = data.get("items", [])
        if not items:
            break
        all_items.extend(items)
        if len(items) < 100:
            break
        time.sleep(SEARCH_DELAY_S)

    return all_items


def _search_repos(query: str, max_pages: int = 3) -> list[dict]:
    """Search GitHub repositories with pagination.

    Args:
        query: GitHub repository search query string.
        max_pages: Maximum number of result pages to fetch.

    Returns:
        List of repository result items (raw JSON dicts).
    """
    all_items: list[dict] = []
    for page in range(1, max_pages + 1):
        logger.info("Repo search page %d: %s", page, query)
        result = _run_gh(
            [
                "api",
                "search/repositories",
                "-X",
                "GET",
                "-f",
                f"q={query}",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
            ]
        )
        data = json.loads(result.stdout)
        items = data.get("items", [])
        if not items:
            break
        all_items.extend(items)
        if len(items) < 100:
            break
        time.sleep(SEARCH_DELAY_S)

    return all_items


def search_github() -> dict[str, RepoInfo]:
    """Execute all three search strategies and deduplicate by repo full_name.

    Returns:
        Dict mapping repo full_name to RepoInfo (without model_dirs populated yet).
    """
    repos: dict[str, RepoInfo] = {}

    # Strategy 1: extension:moc3
    logger.info("=== Search strategy 1: extension:moc3 ===")
    for item in _search_code("extension:moc3"):
        repo = item.get("repository", {})
        full_name = repo.get("full_name", "")
        if full_name and full_name not in repos:
            repos[full_name] = RepoInfo(
                full_name=full_name,
                html_url=repo.get("html_url", f"https://github.com/{full_name}"),
                license_key="",
                default_branch="",
                size_kb=0,
            )
    time.sleep(SEARCH_DELAY_S)

    # Strategy 2: filename:model3.json live2d
    logger.info("=== Search strategy 2: filename:model3.json live2d ===")
    for item in _search_code("filename:model3.json live2d"):
        repo = item.get("repository", {})
        full_name = repo.get("full_name", "")
        if full_name and full_name not in repos:
            repos[full_name] = RepoInfo(
                full_name=full_name,
                html_url=repo.get("html_url", f"https://github.com/{full_name}"),
                license_key="",
                default_branch="",
                size_kb=0,
            )
    time.sleep(SEARCH_DELAY_S)

    # Strategy 3: repo search for "live2d model"
    logger.info("=== Search strategy 3: repo search for 'live2d model' ===")
    for item in _search_repos("live2d model"):
        full_name = item.get("full_name", "")
        if full_name and full_name not in repos:
            license_info = item.get("license") or {}
            repos[full_name] = RepoInfo(
                full_name=full_name,
                html_url=item.get("html_url", f"https://github.com/{full_name}"),
                license_key=license_info.get("spdx_id", "") or "",
                default_branch=item.get("default_branch", "main"),
                size_kb=item.get("size", 0),
            )

    logger.info("Found %d unique repos across all searches", len(repos))
    return repos


def fetch_repo_info(repo: RepoInfo) -> bool:
    """Fetch license and default branch for a repo (if not already populated).

    Args:
        repo: RepoInfo to update in-place.

    Returns:
        True if successful, False on error.
    """
    if repo.license_key and repo.default_branch:
        return True

    try:
        result = _run_gh(["api", f"repos/{repo.full_name}"])
        data = json.loads(result.stdout)
        license_info = data.get("license") or {}
        repo.license_key = license_info.get("spdx_id", "") or ""
        repo.default_branch = data.get("default_branch", "main")
        repo.size_kb = data.get("size", 0)
        time.sleep(CORE_API_DELAY_S)
        return True
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch repo info for %s: %s", repo.full_name, e)
        return False


def check_license(repo: RepoInfo, *, include_unverified: bool = False) -> bool:
    """Check if a repo's license is in the accepted list.

    Args:
        repo: Repository to check.
        include_unverified: If True, accept repos with no detected license.

    Returns:
        True if the license is acceptable.
    """
    if not repo.license_key or repo.license_key == "NOASSERTION":
        if include_unverified:
            logger.info("  %s: no license detected (including as unverified)", repo.full_name)
            return True
        logger.info("  %s: skipped (no license detected)", repo.full_name)
        return False

    if repo.license_key in ACCEPTED_LICENSES:
        return True

    logger.info("  %s: skipped (license: %s)", repo.full_name, repo.license_key)
    return False


def find_model_dirs(repo: RepoInfo) -> list[str]:
    """Fetch the repo's file tree and identify directories containing Live2D models.

    A model directory is one that contains:
    - A .moc3 or .model3.json file, AND
    - .png files in the same directory or in textures/parts/images/ subdirectories.

    Args:
        repo: Repository to inspect.

    Returns:
        List of directory paths (relative to repo root) containing models.
    """
    try:
        result = _run_gh(
            [
                "api",
                f"repos/{repo.full_name}/git/trees/{repo.default_branch}",
                "-f",
                "recursive=1",
            ]
        )
        data = json.loads(result.stdout)
        tree = data.get("tree", [])
        time.sleep(CORE_API_DELAY_S)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch tree for %s: %s", repo.full_name, e)
        return []

    if data.get("truncated"):
        logger.warning(
            "  %s: tree truncated (repo too large), may miss some models", repo.full_name
        )

    # Build sets of paths by type
    model_file_dirs: set[str] = set()
    png_dirs: set[str] = set()

    for entry in tree:
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue

        lower_path = path.lower()

        # Check for model files
        if lower_path.endswith(".moc3") or lower_path.endswith(".model3.json"):
            parent = str(Path(path).parent)
            model_file_dirs.add(parent)

        # Check for PNG textures
        if lower_path.endswith(".png"):
            parent = str(Path(path).parent)
            png_dirs.add(parent)

    # A model directory has model files AND nearby PNGs
    model_dirs: list[str] = []
    for model_dir in sorted(model_file_dirs):
        # Check if PNGs exist in the same dir or in texture subdirectories
        has_textures = model_dir in png_dirs
        if not has_textures:
            for subdir_name in TEXTURE_SUBDIRS:
                candidate = f"{model_dir}/{subdir_name}" if model_dir != "." else subdir_name
                if candidate in png_dirs:
                    has_textures = True
                    break
        # Also check parent dir for PNGs (some repos put model + textures in parent)
        if not has_textures and "/" in model_dir:
            parent = str(Path(model_dir).parent)
            if parent in png_dirs:
                has_textures = True

        if has_textures:
            model_dirs.append(model_dir)
        else:
            logger.debug(
                "  %s: model files in %s but no textures found nearby", repo.full_name, model_dir
            )

    return model_dirs


def _count_pngs(directory: Path) -> int:
    """Count PNG files in a directory and its texture subdirectories."""
    count = 0
    for _png in directory.glob("*.png"):
        count += 1
    for subdir_name in TEXTURE_SUBDIRS:
        subdir = directory / subdir_name
        if subdir.is_dir():
            count += sum(1 for _ in subdir.glob("*.png"))
    return count


def sparse_checkout_model(
    repo: RepoInfo,
    model_dir_path: str,
    dest_dir: Path,
) -> bool:
    """Download a model directory from a repo using sparse git checkout.

    Args:
        repo: Repository containing the model.
        model_dir_path: Path within the repo to the model directory.
        dest_dir: Local destination directory for the model files.

    Returns:
        True if download succeeded.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clone_dir = tmp_path / "repo"

        try:
            # Initialize sparse clone
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--depth=1",
                    f"https://github.com/{repo.full_name}.git",
                    str(clone_dir),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )

            # Set up sparse checkout
            subprocess.run(
                ["git", "-C", str(clone_dir), "sparse-checkout", "init", "--cone"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Set the model directory (and parent paths)
            subprocess.run(
                ["git", "-C", str(clone_dir), "sparse-checkout", "set", model_dir_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Checkout the files
            subprocess.run(
                ["git", "-C", str(clone_dir), "checkout"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )

            # Verify the model directory exists
            source = clone_dir / model_dir_path
            if not source.is_dir():
                logger.warning("Sparse checkout succeeded but %s not found", model_dir_path)
                return False

            # Move model files to destination
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                dest = dest_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            return True

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(
                "Sparse checkout failed for %s/%s: %s", repo.full_name, model_dir_path, e
            )
            return False


def load_manifest(manifest_path: Path) -> dict[str, ModelInfo]:
    """Load existing manifest CSV.

    Args:
        manifest_path: Path to the CSV file.

    Returns:
        Dict mapping model_id to ModelInfo.
    """
    models: dict[str, ModelInfo] = {}
    if not manifest_path.is_file():
        return models

    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row.get("model_id", "")
            if not model_id:
                continue
            models[model_id] = ModelInfo(
                model_id=model_id,
                repo_full_name=row.get("source", ""),
                repo_url=row.get("url", ""),
                license_key=row.get("license", ""),
                license_verified=row.get("license_verified", "").lower() == "true",
                repo_path="",
                fragment_count=int(row.get("fragment_count", 0)),
                notes=row.get("notes", ""),
            )

    logger.info("Loaded %d existing models from manifest", len(models))
    return models


def save_manifest(manifest_path: Path, models: dict[str, ModelInfo]) -> None:
    """Save manifest CSV.

    Args:
        manifest_path: Path to write the CSV file.
        models: Dict mapping model_id to ModelInfo.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for model in sorted(models.values(), key=lambda m: m.model_id):
            writer.writerow(
                {
                    "model_id": model.model_id,
                    "source": model.repo_full_name,
                    "url": model.repo_url,
                    "license": model.license_key,
                    "license_verified": str(model.license_verified),
                    "fragment_count": model.fragment_count,
                    "notes": model.notes,
                }
            )

    logger.info("Saved manifest with %d models to %s", len(models), manifest_path)


def _next_model_id(output_dir: Path, existing_models: dict[str, ModelInfo]) -> int:
    """Determine the next available model ID number.

    Scans both existing directories (named NNN) and manifest entries to find
    the highest used ID, then returns the next one.

    Args:
        output_dir: Base directory containing NNN subdirectories.
        existing_models: Already-loaded manifest models.

    Returns:
        Next available integer ID.
    """
    max_id = 0

    # Check existing directories (named as plain numbers: 001, 002, ...)
    if output_dir.is_dir():
        for d in output_dir.iterdir():
            if d.is_dir():
                try:
                    num = int(d.name)
                    max_id = max(max_id, num)
                except ValueError:
                    pass

    # Check manifest entries (model_id format: "live2d_NNN")
    for model_id in existing_models:
        if model_id.startswith("live2d_"):
            try:
                num = int(model_id.split("_", 1)[1])
                max_id = max(max_id, num)
            except (ValueError, IndexError):
                pass

    return max_id + 1


def _already_downloaded(repo_full_name: str, repo_path: str, models: dict[str, ModelInfo]) -> bool:
    """Check if a model from this repo+path is already in the manifest."""
    source_key = f"{repo_full_name}:{repo_path}"
    for model in models.values():
        existing_key = f"{model.repo_full_name}:{model.notes}"
        if source_key == existing_key:
            return True
        # Also check by repo name if notes don't contain path
        if model.repo_full_name == repo_full_name and repo_path in model.notes:
            return True
    return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search GitHub for Live2D models and download them for Strata training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/live2d"),
        help="Base output directory for downloaded models (default: data/live2d)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Search and report results without downloading",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=0,
        help="Maximum number of models to download (0 = unlimited)",
    )
    parser.add_argument(
        "--include_unverified",
        action="store_true",
        default=False,
        help="Include repos with no detected license (for manual review)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Live2D GitHub scraper."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    output_dir: Path = args.output_dir
    manifest_path = output_dir / "labels" / "live2d_manifest.csv"

    # Check prerequisites
    if not _check_prerequisites():
        sys.exit(1)

    # Load existing manifest for resumability
    existing_models = load_manifest(manifest_path)
    next_id = _next_model_id(output_dir, existing_models)

    # Search GitHub
    repos = search_github()
    logger.info("Found %d unique repos", len(repos))

    # Process repos
    downloaded = 0
    skipped_license = 0
    skipped_no_models = 0
    skipped_existing = 0
    failed = 0

    for repo in sorted(repos.values(), key=lambda r: r.full_name):
        if args.max_models > 0 and downloaded >= args.max_models:
            logger.info("Reached max_models limit (%d)", args.max_models)
            break

        logger.info("Processing: %s", repo.full_name)

        # Fetch repo info (license, default branch)
        if not fetch_repo_info(repo):
            failed += 1
            continue

        # Check license
        if not check_license(repo, include_unverified=args.include_unverified):
            skipped_license += 1
            continue

        # Find model directories in repo
        model_dirs = find_model_dirs(repo)
        if not model_dirs:
            logger.info("  %s: no qualifying model directories found", repo.full_name)
            skipped_no_models += 1
            continue

        logger.info("  %s: found %d model dir(s): %s", repo.full_name, len(model_dirs), model_dirs)

        # Download each model directory
        for model_dir_path in model_dirs:
            if args.max_models > 0 and downloaded >= args.max_models:
                break

            # Check if already downloaded
            if _already_downloaded(repo.full_name, model_dir_path, existing_models):
                logger.info("  Skipping %s/%s (already downloaded)", repo.full_name, model_dir_path)
                skipped_existing += 1
                continue

            model_id = f"live2d_{next_id:03d}"
            dir_name = f"{next_id:03d}"
            dest_dir = output_dir / dir_name

            if args.dry_run:
                logger.info(
                    "  [DRY RUN] Would download %s/%s -> %s",
                    repo.full_name,
                    model_dir_path,
                    dest_dir,
                )
                downloaded += 1
                next_id += 1
                continue

            # Download via sparse checkout
            logger.info("  Downloading %s/%s -> %s", repo.full_name, model_dir_path, dest_dir)
            if not sparse_checkout_model(repo, model_dir_path, dest_dir):
                failed += 1
                continue

            # Count downloaded fragments
            fragment_count = _count_pngs(dest_dir)

            # Record in manifest
            license_verified = repo.license_key in ACCEPTED_LICENSES
            model_info = ModelInfo(
                model_id=model_id,
                repo_full_name=repo.full_name,
                repo_url=repo.html_url,
                license_key=repo.license_key,
                license_verified=license_verified,
                repo_path=model_dir_path,
                fragment_count=fragment_count,
                notes=model_dir_path,
            )
            existing_models[model_id] = model_info
            downloaded += 1
            next_id += 1

            logger.info(
                "  Downloaded %s: %d PNG files, license=%s",
                model_id,
                fragment_count,
                repo.license_key,
            )

            # Save manifest after each download (for resumability)
            save_manifest(manifest_path, existing_models)

    # Final manifest save
    if not args.dry_run:
        save_manifest(manifest_path, existing_models)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Repos found:         %d", len(repos))
    logger.info("  Models downloaded:   %d%s", downloaded, " (dry run)" if args.dry_run else "")
    logger.info("  Skipped (license):   %d", skipped_license)
    logger.info("  Skipped (no models): %d", skipped_no_models)
    logger.info("  Skipped (existing):  %d", skipped_existing)
    logger.info("  Failed:              %d", failed)
    logger.info("  Total in manifest:   %d", len(existing_models))


if __name__ == "__main__":
    main()
