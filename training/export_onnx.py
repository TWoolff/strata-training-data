"""Export trained PyTorch models to ONNX format for the Strata Rust runtime.

Wrapper classes convert dict-returning ``forward()`` methods into tuple outputs
with explicit ``output_names`` required by ``torch.onnx.export``. Post-export
validation with onnxruntime verifies tensor names and shapes.

Usage::

    python training/export_onnx.py --model segmentation --checkpoint best.pt --output seg.onnx
    python training/export_onnx.py --model joints --checkpoint best.pt --output joints.onnx
    python training/export_onnx.py --model weights --checkpoint best.pt --output weights.onnx
    python training/export_onnx.py --model weights_vertex --checkpoint best.pt --output wv.onnx
    python training/export_onnx.py --all --output-dir ./models/

ONNX contracts (must match Rust runtime exactly):

- ``segmentation.onnx``: input ``[1,3,512,512]`` →
  ``segmentation[1,22,512,512]``, ``draw_order[1,1,512,512]``,
  ``confidence[1,1,512,512]``
- ``joint_refinement.onnx``: input ``[1,3,512,512]`` →
  ``offsets[40]``, ``confidence[20]``, ``present[20]``
- ``weight_prediction.onnx``: input ``[1,3,512,512]`` →
  ``weights[1,20,N,1]``, ``confidence[1,1,N,1]``
- ``weight_prediction_vertex.onnx``: input ``[1,31,2048,1]`` →
  ``weights[1,20,2048,1]``, ``confidence[1,1,2048,1]``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import numpy as np

from training.models.joint_model import JointModel
from training.models.segmentation_model import SegmentationModel
from training.models.weight_model import WeightModel
from training.models.weight_prediction_model import WeightPredictionModel
from training.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

RESOLUTION: int = 512


# ---------------------------------------------------------------------------
# ONNX wrapper classes (dict → tuple)
# ---------------------------------------------------------------------------


class SegmentationWrapper(nn.Module):
    """Wraps SegmentationModel for ONNX export (dict → tuple)."""

    def __init__(self, model: SegmentationModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["segmentation"], out["draw_order"], out["confidence"]


class JointWrapper(nn.Module):
    """Wraps JointModel for ONNX export (dict → tuple, flatten + squeeze batch dim)."""

    def __init__(self, model: JointModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        # Flatten [B, 2, 20] → [B, 40] then squeeze batch → [40]
        # Layout: first 20 = dx, next 20 = dy (matches Rust runtime)
        offsets = out["offsets"].flatten(1).squeeze(0)
        return offsets, out["confidence"].squeeze(0), out["present"].squeeze(0)


class WeightWrapper(nn.Module):
    """Wraps WeightModel for ONNX export (dict → tuple, reshape to [1,C,N,1])."""

    def __init__(self, model: WeightModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        # Reshape from [B, C, H, W] → [B, C, H*W, 1] for Rust vertex-indexed access
        w = out["weights"]  # [1, 20, 512, 512]
        c = out["confidence"]  # [1, 1, 512, 512]
        b, num_bones, h, w_dim = w.shape
        w = w.reshape(b, num_bones, h * w_dim, 1)
        c = c.reshape(b, 1, h * w_dim, 1)
        return w, c


class WeightPredictionWrapper(nn.Module):
    """Wraps WeightPredictionModel for ONNX export (dict → tuple).

    The per-vertex model already outputs [B, C, N, 1] tensors, so no
    reshaping is needed. Outputs are raw logits — softmax/sigmoid applied
    at inference by the Rust runtime.
    """

    def __init__(self, model: WeightPredictionModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["weights"], out["confidence"]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "segmentation": {
        "model_class": SegmentationModel,
        "wrapper_class": SegmentationWrapper,
        "model_kwargs": {"num_classes": 22, "pretrained_backbone": False},
        "output_names": ["segmentation", "draw_order", "confidence"],
        "dynamic_axes": {
            "input": {0: "batch"},
            "segmentation": {0: "batch"},
            "draw_order": {0: "batch"},
            "confidence": {0: "batch"},
        },
        "default_filename": "segmentation.onnx",
    },
    "joints": {
        "model_class": JointModel,
        "wrapper_class": JointWrapper,
        "model_kwargs": {"num_joints": 20, "pretrained_backbone": False},
        "output_names": ["offsets", "confidence", "present"],
        "dynamic_axes": {
            "input": {0: "batch"},
        },
        "default_filename": "joint_refinement.onnx",
    },
    "weights": {
        "model_class": WeightModel,
        "wrapper_class": WeightWrapper,
        "model_kwargs": {"num_bones": 20, "pretrained_backbone": False},
        "output_names": ["weights", "confidence"],
        "dynamic_axes": {
            "input": {0: "batch"},
            "weights": {0: "batch", 2: "vertices"},
            "confidence": {0: "batch", 2: "vertices"},
        },
        "default_filename": "weight_prediction.onnx",
    },
    "weights_vertex": {
        "model_class": WeightPredictionModel,
        "wrapper_class": WeightPredictionWrapper,
        "model_kwargs": {"num_features": 31, "num_bones": 20},
        "output_names": ["weights", "confidence"],
        "dynamic_axes": {
            "input": {0: "batch"},
            "weights": {0: "batch", 2: "vertices"},
            "confidence": {0: "batch", 2: "vertices"},
        },
        "default_filename": "weight_prediction_vertex.onnx",
        "input_shape": (1, 31, 2048, 1),
    },
}


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------


def export_model(
    model_name: str,
    checkpoint_path: Path | None,
    output_path: Path,
    *,
    validate: bool = True,
) -> Path:
    """Export a single model to ONNX format.

    Args:
        model_name: One of ``"segmentation"``, ``"joints"``, ``"weights"``.
        checkpoint_path: Path to ``.pt`` checkpoint, or ``None`` for random weights.
        output_path: Path for the output ``.onnx`` file.
        validate: Run onnxruntime validation after export.

    Returns:
        Path to the exported ONNX file.
    """
    config = MODEL_CONFIGS[model_name]

    # Build model
    model = config["model_class"](**config["model_kwargs"])

    # Load checkpoint if provided
    if checkpoint_path is not None:
        info = load_checkpoint(checkpoint_path, model)
        logger.info("Loaded checkpoint epoch %d, metrics: %s", info["epoch"], info["metrics"])

    model.eval()

    # Wrap for ONNX export
    wrapper = config["wrapper_class"](model)
    wrapper.eval()

    # Dummy input — use custom shape if specified (e.g. per-vertex models)
    input_shape = config.get("input_shape", (1, 3, RESOLUTION, RESOLUTION))
    dummy_input = torch.randn(*input_shape)

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting %s to %s ...", model_name, output_path)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=config["output_names"],
        dynamic_axes=config["dynamic_axes"],
        opset_version=17,
        do_constant_folding=True,
    )

    # Check the ONNX model is well-formed
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model check passed for %s", model_name)

    # Post-export validation
    if validate:
        validate_onnx(model_name, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported %s (%.1f MB)", output_path.name, file_size_mb)
    return output_path


def validate_onnx(model_name: str, onnx_path: Path) -> None:
    """Validate an exported ONNX model with onnxruntime.

    Loads the model, runs dummy inference, and checks that output tensor
    names and shapes match the expected ONNX contract.

    Args:
        model_name: One of ``"segmentation"``, ``"joints"``, ``"weights"``.
        onnx_path: Path to the ``.onnx`` file.

    Raises:
        ValueError: If output names or shapes don't match expectations.
    """
    import numpy as np
    import onnxruntime as ort

    config = MODEL_CONFIGS[model_name]
    expected_names = config["output_names"]

    session = ort.InferenceSession(str(onnx_path))

    # Verify input
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "input":
        raise ValueError(
            f"Expected single input named 'input', got: {[(i.name, i.shape) for i in inputs]}"
        )

    # Verify output names
    outputs = session.get_outputs()
    actual_names = [o.name for o in outputs]
    if actual_names != expected_names:
        raise ValueError(
            f"Output name mismatch for {model_name}. Expected {expected_names}, got {actual_names}"
        )

    # Run dummy inference — use custom input shape if specified
    input_shape = config.get("input_shape", (1, 3, RESOLUTION, RESOLUTION))
    dummy = np.random.randn(*input_shape).astype(np.float32)
    results = session.run(None, {"input": dummy})

    # Shape validation per model
    if model_name == "segmentation":
        _validate_segmentation_outputs(results, expected_names)
    elif model_name == "joints":
        _validate_joint_outputs(results, expected_names)
    elif model_name == "weights":
        _validate_weight_outputs(results, expected_names)
    elif model_name == "weights_vertex":
        _validate_weight_vertex_outputs(results, expected_names)

    logger.info("Validation passed for %s", model_name)


def _validate_segmentation_outputs(results: list[np.ndarray], names: list[str]) -> None:
    seg, draw_order, confidence = results
    if seg.shape != (1, 22, RESOLUTION, RESOLUTION):
        raise ValueError(f"segmentation shape: expected (1,22,512,512), got {seg.shape}")
    if draw_order.shape != (1, 1, RESOLUTION, RESOLUTION):
        raise ValueError(f"draw_order shape: expected (1,1,512,512), got {draw_order.shape}")
    if confidence.shape != (1, 1, RESOLUTION, RESOLUTION):
        raise ValueError(f"confidence shape: expected (1,1,512,512), got {confidence.shape}")
    # draw_order and confidence should be in [0, 1] (sigmoid)
    if draw_order.min() < -0.01 or draw_order.max() > 1.01:
        raise ValueError(f"draw_order out of [0,1] range: [{draw_order.min()}, {draw_order.max()}]")
    if confidence.min() < -0.01 or confidence.max() > 1.01:
        raise ValueError(f"confidence out of [0,1] range: [{confidence.min()}, {confidence.max()}]")


def _validate_joint_outputs(results: list[np.ndarray], names: list[str]) -> None:
    offsets, confidence, present = results
    if offsets.shape != (40,):
        raise ValueError(f"offsets shape: expected (40,), got {offsets.shape}")
    if confidence.shape != (20,):
        raise ValueError(f"confidence shape: expected (20,), got {confidence.shape}")
    if present.shape != (20,):
        raise ValueError(f"present shape: expected (20,), got {present.shape}")
    # confidence should be in [0, 1] (sigmoid)
    if confidence.min() < -0.01 or confidence.max() > 1.01:
        raise ValueError(f"confidence out of [0,1] range: [{confidence.min()}, {confidence.max()}]")


def _validate_weight_outputs(results: list[np.ndarray], names: list[str]) -> None:
    weights, confidence = results
    n_vertices = RESOLUTION * RESOLUTION
    if weights.shape != (1, 20, n_vertices, 1):
        raise ValueError(f"weights shape: expected (1,20,{n_vertices},1), got {weights.shape}")
    if confidence.shape != (1, 1, n_vertices, 1):
        raise ValueError(f"confidence shape: expected (1,1,{n_vertices},1), got {confidence.shape}")
    # weights should be softmax (>=0, sum to ~1 over bone dim)
    if weights.min() < -0.01:
        raise ValueError(f"weights has negative values: min={weights.min()}")
    # confidence should be in [0, 1] (sigmoid)
    if confidence.min() < -0.01 or confidence.max() > 1.01:
        raise ValueError(f"confidence out of [0,1] range: [{confidence.min()}, {confidence.max()}]")


def _validate_weight_vertex_outputs(results: list[np.ndarray], names: list[str]) -> None:
    weights, confidence = results
    max_vertices = 2048
    if weights.shape != (1, 20, max_vertices, 1):
        raise ValueError(f"weights shape: expected (1,20,{max_vertices},1), got {weights.shape}")
    if confidence.shape != (1, 1, max_vertices, 1):
        raise ValueError(
            f"confidence shape: expected (1,1,{max_vertices},1), got {confidence.shape}"
        )
    # Per-vertex model outputs raw logits — no range constraint on weights
    # (softmax applied at inference by Rust runtime)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ONNX model export."""
    parser = argparse.ArgumentParser(
        description="Export Strata PyTorch models to ONNX format.",
    )
    parser.add_argument(
        "--model",
        choices=["segmentation", "joints", "weights", "weights_vertex"],
        help="Model to export.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .onnx file path (for single model export).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="export_all",
        help="Export all three models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for --all mode.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-export onnxruntime validation.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.export_all:
        if args.output_dir is None:
            parser.error("--output-dir is required when using --all")
        output_dir = args.output_dir
        for name, config in MODEL_CONFIGS.items():
            checkpoint = _find_checkpoint(name) if args.checkpoint is None else args.checkpoint
            out_path = output_dir / config["default_filename"]
            export_model(name, checkpoint, out_path, validate=not args.no_validate)
    else:
        if args.model is None:
            parser.error("--model is required (or use --all)")
        if args.output is None:
            args.output = Path(MODEL_CONFIGS[args.model]["default_filename"])
        export_model(args.model, args.checkpoint, args.output, validate=not args.no_validate)


def _find_checkpoint(model_name: str) -> Path | None:
    """Try to find a default checkpoint for the given model."""
    default_dirs = {
        "segmentation": Path("checkpoints/segmentation/best.pt"),
        "joints": Path("checkpoints/joints/best.pt"),
        "weights": Path("checkpoints/weights/best.pt"),
        "weights_vertex": Path("checkpoints/weights/best.pt"),
    }
    path = default_dirs.get(model_name)
    if path is not None and path.exists():
        return path
    logger.warning("No checkpoint found for %s, exporting with random weights", model_name)
    return None


if __name__ == "__main__":
    main()
