"""Tests for training/export_onnx.py — ONNX export pipeline."""

from __future__ import annotations

import pytest

try:
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401

    _HAS_ONNX = _HAS_TORCH  # ONNX export also needs torch
except ImportError:
    _HAS_ONNX = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch/torchvision not installed")

_skip_no_onnx = pytest.mark.skipif(
    not _HAS_ONNX, reason="onnx/onnxruntime/onnxscript not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export(model_name: str, tmp_path):
    """Export a model with random weights and return the ONNX path."""
    from training.export_onnx import export_model

    out_path = tmp_path / f"{model_name}.onnx"
    return export_model(model_name, checkpoint_path=None, output_path=out_path, validate=True)


# ---------------------------------------------------------------------------
# Model wrapper tests
# ---------------------------------------------------------------------------


class TestSegmentationWrapper:
    def test_wrapper_returns_tuple(self):
        from training.export_onnx import SegmentationWrapper
        from training.models.segmentation_model import SegmentationModel

        model = SegmentationModel(num_classes=22, pretrained_backbone=False)
        wrapper = SegmentationWrapper(model)
        wrapper.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            result = wrapper(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wrapper_output_shapes(self):
        from training.export_onnx import SegmentationWrapper
        from training.models.segmentation_model import SegmentationModel

        model = SegmentationModel(num_classes=22, pretrained_backbone=False)
        wrapper = SegmentationWrapper(model)
        wrapper.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            seg, draw_order, confidence = wrapper(x)
        assert seg.shape == (1, 22, 128, 128)
        assert draw_order.shape == (1, 1, 128, 128)
        assert confidence.shape == (1, 1, 128, 128)


class TestJointWrapper:
    def test_wrapper_returns_tuple(self):
        from training.export_onnx import JointWrapper
        from training.models.joint_model import JointModel

        model = JointModel(num_joints=20, pretrained_backbone=False)
        wrapper = JointWrapper(model)
        wrapper.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            result = wrapper(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wrapper_output_shapes(self):
        from training.export_onnx import JointWrapper
        from training.models.joint_model import JointModel

        model = JointModel(num_joints=20, pretrained_backbone=False)
        wrapper = JointWrapper(model)
        wrapper.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            offsets, confidence, present = wrapper(x)
        # Batch dim squeezed
        assert offsets.shape == (40,)
        assert confidence.shape == (20,)
        assert present.shape == (20,)


class TestWeightWrapper:
    def test_wrapper_returns_tuple(self):
        from training.export_onnx import WeightWrapper
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        wrapper = WeightWrapper(model)
        wrapper.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            result = wrapper(x)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_wrapper_output_shapes(self):
        from training.export_onnx import WeightWrapper
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        wrapper = WeightWrapper(model)
        wrapper.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            weights, confidence = wrapper(x)
        # 64*64 = 4096 pixels reshaped to [1, 20, 4096, 1]
        assert weights.shape == (1, 20, 4096, 1)
        assert confidence.shape == (1, 1, 4096, 1)


# ---------------------------------------------------------------------------
# Joint model tests
# ---------------------------------------------------------------------------


class TestJointModel:
    def test_instantiates(self):
        from training.models.joint_model import JointModel

        model = JointModel(num_joints=20, pretrained_backbone=False)
        assert model.num_joints == 20

    def test_forward_output_keys(self):
        from training.models.joint_model import JointModel

        model = JointModel(num_joints=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert set(out.keys()) == {"offsets", "confidence", "present"}

    def test_forward_output_shapes(self):
        from training.models.joint_model import JointModel

        model = JointModel(num_joints=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["offsets"].shape == (2, 40)
        assert out["confidence"].shape == (2, 20)
        assert out["present"].shape == (2, 20)

    def test_confidence_sigmoid_range(self):
        from training.models.joint_model import JointModel

        model = JointModel(num_joints=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["confidence"].min() >= 0.0
        assert out["confidence"].max() <= 1.0


# ---------------------------------------------------------------------------
# Weight model tests
# ---------------------------------------------------------------------------


class TestWeightModel:
    def test_instantiates(self):
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        assert model.num_bones == 20

    def test_forward_output_keys(self):
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert set(out.keys()) == {"weights", "confidence"}

    def test_forward_output_shapes(self):
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["weights"].shape == (1, 20, 128, 128)
        assert out["confidence"].shape == (1, 1, 128, 128)

    def test_weights_softmax(self):
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        # Weights should sum to ~1.0 over bone dimension
        sums = out["weights"].sum(dim=1)  # [1, 64, 64]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_confidence_sigmoid_range(self):
        from training.models.weight_model import WeightModel

        model = WeightModel(num_bones=20, pretrained_backbone=False)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["confidence"].min() >= 0.0
        assert out["confidence"].max() <= 1.0


# ---------------------------------------------------------------------------
# End-to-end ONNX export tests
# ---------------------------------------------------------------------------


@_skip_no_onnx
class TestOnnxExport:
    def test_export_segmentation(self, tmp_path):
        """Segmentation model exports and validates successfully."""
        path = _export("segmentation", tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_joints(self, tmp_path):
        """Joint model exports and validates successfully."""
        path = _export("joints", tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_weights(self, tmp_path):
        """Weight model exports and validates successfully."""
        path = _export("weights", tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_segmentation_output_names(self, tmp_path):
        """Segmentation ONNX has correct output tensor names."""
        import onnxruntime as ort

        path = _export("segmentation", tmp_path)
        session = ort.InferenceSession(str(path))
        names = [o.name for o in session.get_outputs()]
        assert names == ["segmentation", "draw_order", "confidence"]

    def test_joints_output_names(self, tmp_path):
        """Joint ONNX has correct output tensor names."""
        import onnxruntime as ort

        path = _export("joints", tmp_path)
        session = ort.InferenceSession(str(path))
        names = [o.name for o in session.get_outputs()]
        assert names == ["offsets", "confidence", "present"]

    def test_weights_output_names(self, tmp_path):
        """Weight ONNX has correct output tensor names."""
        import onnxruntime as ort

        path = _export("weights", tmp_path)
        session = ort.InferenceSession(str(path))
        names = [o.name for o in session.get_outputs()]
        assert names == ["weights", "confidence"]

    def test_input_name(self, tmp_path):
        """All models use 'input' as the input tensor name."""
        import onnxruntime as ort

        for model_name in ["segmentation", "joints", "weights"]:
            path = _export(model_name, tmp_path)
            session = ort.InferenceSession(str(path))
            inputs = session.get_inputs()
            assert len(inputs) == 1
            assert inputs[0].name == "input"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


@_skip_no_onnx
class TestCli:
    def test_single_model_export(self, tmp_path):
        """CLI --model exports a single model."""
        from training.export_onnx import main

        out = tmp_path / "test_seg.onnx"
        main(["--model", "segmentation", "--output", str(out)])
        assert out.exists()

    def test_all_model_export(self, tmp_path):
        """CLI --all exports all three models."""
        from training.export_onnx import main

        main(["--all", "--output-dir", str(tmp_path)])
        assert (tmp_path / "segmentation.onnx").exists()
        assert (tmp_path / "joint_refinement.onnx").exists()
        assert (tmp_path / "weight_prediction.onnx").exists()

    def test_no_validate_flag(self, tmp_path):
        """CLI --no-validate skips onnxruntime validation."""
        from training.export_onnx import main

        out = tmp_path / "test_joints.onnx"
        main(["--model", "joints", "--output", str(out), "--no-validate"])
        assert out.exists()
