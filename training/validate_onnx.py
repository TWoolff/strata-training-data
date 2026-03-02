"""Validate exported ONNX models against Strata Rust runtime contracts.

Standalone script that checks tensor names, shapes, value ranges, and file
sizes for each model type. Catches integration issues before models reach the
app.

Usage::

    python training/validate_onnx.py --model segmentation --path models/segmentation.onnx
    python training/validate_onnx.py --all --models-dir ./exported/
    python training/validate_onnx.py --all --models-dir ./exported/ --dataset-dir ../output/segmentation/

Exit code 0 on all checks pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching Rust runtime
# ---------------------------------------------------------------------------

RESOLUTION: int = 512
NUM_CLASSES: int = 22
NUM_JOINTS: int = 20
NUM_BONES: int = 20

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Model filenames (must match runtime.rs ModelType::filename())
MODEL_FILENAMES: dict[str, str] = {
    "segmentation": "segmentation.onnx",
    "joints": "joint_refinement.onnx",
    "weights": "weight_prediction.onnx",
}

# File size bounds in MB (min, max)
FILE_SIZE_BOUNDS: dict[str, tuple[float, float]] = {
    "segmentation": (20.0, 80.0),
    "joints": (1.0, 15.0),
    "weights": (3.0, 30.0),
}


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------


@dataclass
class OutputSpec:
    """Expected specification for a single output tensor."""

    name: str
    shape: tuple[int, ...] | None = None
    """Expected shape, or None for dynamic/variable shapes."""
    ndim: int | None = None
    """Expected number of dimensions (when shape has dynamic axes)."""
    dim_checks: dict[int, int] = field(default_factory=dict)
    """Specific dimension index → expected size (for partial shape checks)."""
    value_range: tuple[float, float] | None = None
    """Expected (min, max) range for values, or None to skip."""
    non_negative: bool = False
    """Whether all values must be >= 0."""


@dataclass
class ModelSpec:
    """Full specification for a model's ONNX contract."""

    name: str
    input_name: str
    input_shape: tuple[int, ...]
    outputs: list[OutputSpec]
    file_size_mb: tuple[float, float]
    optional_outputs: list[str] = field(default_factory=list)


SEGMENTATION_SPEC = ModelSpec(
    name="segmentation",
    input_name="input",
    input_shape=(1, 3, RESOLUTION, RESOLUTION),
    outputs=[
        OutputSpec(
            name="segmentation",
            shape=(1, NUM_CLASSES, RESOLUTION, RESOLUTION),
        ),
        OutputSpec(
            name="draw_order",
            shape=(1, 1, RESOLUTION, RESOLUTION),
            value_range=(-0.01, 1.01),
        ),
        OutputSpec(
            name="confidence",
            shape=(1, 1, RESOLUTION, RESOLUTION),
            value_range=(-0.01, 1.01),
        ),
    ],
    file_size_mb=FILE_SIZE_BOUNDS["segmentation"],
    optional_outputs=["encoder_features"],
)

JOINT_SPEC = ModelSpec(
    name="joints",
    input_name="input",
    input_shape=(1, 3, RESOLUTION, RESOLUTION),
    outputs=[
        OutputSpec(name="offsets", shape=(NUM_JOINTS * 2,)),
        OutputSpec(
            name="confidence",
            shape=(NUM_JOINTS,),
            value_range=(-0.01, 1.01),
        ),
        OutputSpec(name="present", shape=(NUM_JOINTS,)),
    ],
    file_size_mb=FILE_SIZE_BOUNDS["joints"],
)

WEIGHT_SPEC = ModelSpec(
    name="weights",
    input_name="input",
    input_shape=(1, 3, RESOLUTION, RESOLUTION),
    outputs=[
        OutputSpec(
            name="weights",
            ndim=4,
            dim_checks={0: 1, 1: NUM_BONES, 3: 1},
            non_negative=True,
        ),
        OutputSpec(
            name="confidence",
            ndim=4,
            dim_checks={0: 1, 1: 1, 3: 1},
            value_range=(-0.01, 1.01),
        ),
    ],
    file_size_mb=FILE_SIZE_BOUNDS["weights"],
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "segmentation": SEGMENTATION_SPEC,
    "joints": JOINT_SPEC,
    "weights": WEIGHT_SPEC,
}


# ---------------------------------------------------------------------------
# Validation result tracking
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single validation check."""

    passed: bool
    message: str


class ValidationReport:
    """Collects check results and prints a summary."""

    def __init__(self, model_name: str, path: Path) -> None:
        self.model_name = model_name
        self.path = path
        self.checks: list[CheckResult] = []

    def add(self, passed: bool, message: str) -> None:
        self.checks.append(CheckResult(passed=passed, message=message))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print_report(self) -> None:
        status = "PASS" if self.all_passed else "FAIL"
        print(f"\n{self.path.name}: {status}")
        for check in self.checks:
            icon = "+" if check.passed else "x"
            print(f"  {icon} {check.message}")


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_model(
    model_name: str,
    onnx_path: Path,
    *,
    dataset_dir: Path | None = None,
    check_file_size: bool = True,
) -> ValidationReport:
    """Validate a single ONNX model against its Rust runtime contract.

    Args:
        model_name: One of ``"segmentation"``, ``"joints"``, ``"weights"``.
        onnx_path: Path to the ``.onnx`` file.
        dataset_dir: Optional path to dataset for real-image validation.
        check_file_size: Whether to enforce file size bounds.

    Returns:
        ValidationReport with all check results.
    """
    spec = MODEL_SPECS[model_name]
    report = ValidationReport(model_name, onnx_path)

    # 1. File exists
    if not onnx_path.exists():
        report.add(False, f"File not found: {onnx_path}")
        return report

    # 2. File size
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    if check_file_size:
        min_mb, max_mb = spec.file_size_mb
        size_ok = min_mb <= file_size_mb <= max_mb
        report.add(size_ok, f"File exists ({file_size_mb:.1f} MB, expected {min_mb}-{max_mb} MB)")
    else:
        report.add(True, f"File exists ({file_size_mb:.1f} MB)")

    # 3. Load with onnxruntime
    try:
        session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        report.add(False, f"Failed to load ONNX: {e}")
        return report

    # 4. Input validation
    inputs = session.get_inputs()
    input_ok = len(inputs) == 1 and inputs[0].name == spec.input_name
    if input_ok:
        # Check shape (may have dynamic dims represented as strings)
        actual_shape = inputs[0].shape
        shape_parts = []
        for actual, expected in zip(actual_shape, spec.input_shape, strict=False):
            if isinstance(actual, int):
                shape_parts.append(str(actual))
                if actual != expected:
                    input_ok = False
            else:
                # Dynamic dim (string like "batch") — acceptable for dim 0
                shape_parts.append(str(actual))
        shape_str = f"[{', '.join(shape_parts)}]"
        report.add(input_ok, f'Input: "{spec.input_name}" {shape_str}')
    else:
        actual = [(i.name, i.shape) for i in inputs]
        report.add(False, f"Input mismatch: expected single '{spec.input_name}', got {actual}")
        return report

    # 5. Output names
    actual_outputs = session.get_outputs()
    actual_names = [o.name for o in actual_outputs]
    expected_names = [o.name for o in spec.outputs]
    # Allow optional outputs to be present
    required_present = all(name in actual_names for name in expected_names)
    extra_names = [n for n in actual_names if n not in expected_names + spec.optional_outputs]
    names_ok = required_present and not extra_names
    if not names_ok:
        report.add(False, f"Output names: expected {expected_names}, got {actual_names}")
    # Continue even if names don't match, to report other issues

    # 6. Run dummy inference
    dummy = np.random.randn(*spec.input_shape).astype(np.float32)
    try:
        t0 = time.monotonic()
        results = session.run(None, {spec.input_name: dummy})
        elapsed_ms = (time.monotonic() - t0) * 1000
        report.add(True, f"Inference: {elapsed_ms:.0f}ms")
    except Exception as e:
        report.add(False, f"Inference failed: {e}")
        return report

    # Build name → result mapping
    result_map: dict[str, np.ndarray] = {}
    for output_meta, result_arr in zip(actual_outputs, results, strict=False):
        result_map[output_meta.name] = result_arr

    # 7. Per-output validation
    for output_spec in spec.outputs:
        arr = result_map.get(output_spec.name)
        if arr is None:
            report.add(False, f'Output "{output_spec.name}": missing')
            continue

        # Shape check (exact)
        if output_spec.shape is not None:
            shape_ok = arr.shape == output_spec.shape
            report.add(
                shape_ok,
                f'Output: "{output_spec.name}" {list(arr.shape)}'
                + ("" if shape_ok else f" (expected {list(output_spec.shape)})"),
            )
        # Ndim + partial dimension checks
        elif output_spec.ndim is not None:
            ndim_ok = arr.ndim == output_spec.ndim
            dim_ok = all(arr.shape[dim] == size for dim, size in output_spec.dim_checks.items())
            ok = ndim_ok and dim_ok
            report.add(
                ok,
                f'Output: "{output_spec.name}" {list(arr.shape)}'
                + (
                    ""
                    if ok
                    else f" (expected ndim={output_spec.ndim}, dims={output_spec.dim_checks})"
                ),
            )

        # Value range check
        if output_spec.value_range is not None:
            vmin, vmax = output_spec.value_range
            range_ok = bool(arr.min() >= vmin and arr.max() <= vmax)
            if not range_ok:
                report.add(
                    False,
                    f'"{output_spec.name}" range [{arr.min():.4f}, {arr.max():.4f}]'
                    f" outside [{vmin}, {vmax}]",
                )

        # Non-negative check
        if output_spec.non_negative:
            nn_ok = bool(arr.min() >= -0.01)
            if not nn_ok:
                report.add(False, f'"{output_spec.name}" has negative values: min={arr.min():.4f}')

    # 8. Model-specific semantic checks
    if model_name == "segmentation" and "segmentation" in result_map:
        seg = result_map["segmentation"]
        argmax = seg.argmax(axis=1)  # [1, H, W]
        vmin, vmax = int(argmax.min()), int(argmax.max())
        argmax_ok = vmin >= 0 and vmax <= NUM_CLASSES - 1
        report.add(argmax_ok, f"Argmax range: {vmin}-{vmax} (valid 0-{NUM_CLASSES - 1})")

        # Draw order should be finite
        if "draw_order" in result_map:
            do = result_map["draw_order"]
            finite_ok = bool(np.isfinite(do).all())
            report.add(
                finite_ok,
                "Draw order: all finite" if finite_ok else "Draw order: non-finite values",
            )

    if names_ok:
        # Add the names check here so it appears in the right order
        report.add(True, f"Output names: {expected_names}")

    # 9. Real-image cross-validation
    if dataset_dir is not None and model_name == "segmentation":
        _validate_with_real_image(session, spec, dataset_dir, report)

    return report


def _validate_with_real_image(
    session: ort.InferenceSession,
    spec: ModelSpec,
    dataset_dir: Path,
    report: ValidationReport,
) -> None:
    """Run inference on a real image and check for reasonable output."""
    from PIL import Image

    # Find a real image
    image_files = list(dataset_dir.rglob("image.png"))
    if not image_files:
        # Try any PNG that isn't a mask
        image_files = [
            p
            for p in dataset_dir.rglob("*.png")
            if "segmentation" not in p.name and "draw_order" not in p.name
        ]
    if not image_files:
        report.add(False, "Cross-validation: no images found in dataset dir")
        return

    img_path = image_files[0]
    try:
        img = Image.open(img_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    except Exception as e:
        report.add(False, f"Cross-validation: failed to load {img_path}: {e}")
        return

    # Preprocess with ImageNet normalization (matching Rust runtime)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    # HWC -> NCHW
    tensor = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]  # [1, 3, 512, 512]

    try:
        results = session.run(None, {spec.input_name: tensor})
    except Exception as e:
        report.add(False, f"Cross-validation: inference failed: {e}")
        return

    # Check segmentation produces reasonable region distribution
    seg = results[0]  # [1, 22, 512, 512]
    argmax = seg.argmax(axis=1).flatten()  # [H*W]
    unique_regions = len(np.unique(argmax))
    # A real image should produce at least 2 distinct regions (background + something)
    region_ok = unique_regions >= 2
    report.add(
        region_ok,
        f"Cross-validation: {unique_regions} distinct regions"
        + ("" if region_ok else " (expected >= 2)"),
    )

    # Check no single region dominates >95% (would indicate degenerate model)
    total_pixels = argmax.size
    for region_id in np.unique(argmax):
        count = int((argmax == region_id).sum())
        frac = count / total_pixels
        if frac > 0.95:
            report.add(
                False,
                f"Cross-validation: region {region_id} dominates ({frac:.1%} of pixels)",
            )
            return

    report.add(True, "Cross-validation: no degenerate region dominance")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on pass, 1 on failure."""
    parser = argparse.ArgumentParser(
        description="Validate ONNX models against Strata Rust runtime contracts.",
    )
    parser.add_argument(
        "--model",
        choices=["segmentation", "joints", "weights"],
        help="Model type to validate.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to .onnx file (for single model validation).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="validate_all",
        help="Validate all three models from --models-dir.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Directory containing model .onnx files (for --all mode).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Optional dataset directory for real-image cross-validation.",
    )
    parser.add_argument(
        "--no-check-file-size",
        action="store_true",
        help="Skip file size bounds check (useful for test/mock models).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reports: list[ValidationReport] = []

    check_size = not args.no_check_file_size

    if args.validate_all:
        if args.models_dir is None:
            parser.error("--models-dir is required when using --all")
        for model_name, filename in MODEL_FILENAMES.items():
            path = args.models_dir / filename
            report = validate_model(
                model_name, path, dataset_dir=args.dataset_dir, check_file_size=check_size
            )
            reports.append(report)
    else:
        if args.model is None:
            parser.error("--model is required (or use --all)")
        if args.path is None:
            parser.error("--path is required for single model validation")
        report = validate_model(
            args.model, args.path, dataset_dir=args.dataset_dir, check_file_size=check_size
        )
        reports.append(report)

    # Print reports
    for report in reports:
        report.print_report()

    # Summary
    all_passed = all(r.all_passed for r in reports)
    total = sum(len(r.checks) for r in reports)
    passed = sum(sum(1 for c in r.checks if c.passed) for r in reports)
    failed = total - passed
    print(f"\n{'=' * 40}")
    print(f"Total: {passed}/{total} checks passed" + (f", {failed} failed" if failed else ""))
    print(f"Result: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
