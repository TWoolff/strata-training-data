"""Tests for training/validate_onnx.py — ONNX validation against Rust runtime contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import onnx
    import onnxruntime  # noqa: F401
    from onnx import TensorProto, helper

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

pytestmark = pytest.mark.skipif(not _HAS_ONNX, reason="onnx/onnxruntime not installed")


# ---------------------------------------------------------------------------
# Helpers — build minimal ONNX models for testing
# ---------------------------------------------------------------------------


def _make_segmentation_onnx(path: Path, *, wrong_names: bool = False) -> Path:
    """Create a minimal ONNX model mimicking segmentation output contract.

    Uses Identity ops to pass a [1,3,512,512] input through with shape transforms
    to produce the expected output tensors.
    """
    # We'll build a graph that takes input [1,3,512,512] and produces:
    # - segmentation [1,22,512,512] via Expand
    # - draw_order [1,1,512,512] via ReduceMean + Sigmoid
    # - confidence [1,1,512,512] via ReduceMean + Sigmoid

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 512, 512])

    seg_name = "wrong_seg" if wrong_names else "segmentation"
    do_name = "wrong_do" if wrong_names else "draw_order"
    conf_name = "wrong_conf" if wrong_names else "confidence"

    # Output value infos
    seg_output = helper.make_tensor_value_info(seg_name, TensorProto.FLOAT, [1, 22, 512, 512])
    do_output = helper.make_tensor_value_info(do_name, TensorProto.FLOAT, [1, 1, 512, 512])
    conf_output = helper.make_tensor_value_info(conf_name, TensorProto.FLOAT, [1, 1, 512, 512])

    # Shape constant for segmentation output: [1, 22, 512, 512]
    seg_shape = helper.make_tensor("seg_shape", TensorProto.INT64, [4], [1, 22, 512, 512])
    # Shape constant for single-channel output: [1, 1, 512, 512]
    single_shape = helper.make_tensor("single_shape", TensorProto.INT64, [4], [1, 1, 512, 512])
    # Zero constant for ConstantOfShape
    zero_val = helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0])
    half_val = helper.make_tensor("half_val", TensorProto.FLOAT, [1], [0.5])

    nodes = [
        # Segmentation: constant zeros [1, 22, 512, 512]
        helper.make_node("ConstantOfShape", ["seg_shape"], ["seg_raw"], value=zero_val),
        helper.make_node("Identity", ["seg_raw"], [seg_name]),
        # Draw order: constant 0.5 [1, 1, 512, 512] (in sigmoid range)
        helper.make_node("ConstantOfShape", ["single_shape"], ["do_raw"], value=half_val),
        helper.make_node("Identity", ["do_raw"], [do_name]),
        # Confidence: constant 0.5 [1, 1, 512, 512]
        helper.make_node("ConstantOfShape", ["single_shape"], ["conf_raw"], value=half_val),
        helper.make_node("Identity", ["conf_raw"], [conf_name]),
    ]

    graph = helper.make_graph(
        nodes,
        "test_segmentation",
        [input_tensor],
        [seg_output, do_output, conf_output],
        initializer=[seg_shape, single_shape, zero_val, half_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _make_joint_onnx(path: Path, *, wrong_names: bool = False) -> Path:
    """Create a minimal ONNX model mimicking joint refinement output contract."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 512, 512])

    off_name = "wrong_offsets" if wrong_names else "offsets"
    conf_name = "wrong_conf" if wrong_names else "confidence"
    pres_name = "wrong_present" if wrong_names else "present"

    off_output = helper.make_tensor_value_info(off_name, TensorProto.FLOAT, [40])
    conf_output = helper.make_tensor_value_info(conf_name, TensorProto.FLOAT, [20])
    pres_output = helper.make_tensor_value_info(pres_name, TensorProto.FLOAT, [20])

    off_shape = helper.make_tensor("off_shape", TensorProto.INT64, [1], [40])
    conf_shape = helper.make_tensor("conf_shape", TensorProto.INT64, [1], [20])
    zero_val = helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0])
    half_val = helper.make_tensor("half_val", TensorProto.FLOAT, [1], [0.5])

    nodes = [
        helper.make_node("ConstantOfShape", ["off_shape"], ["off_raw"], value=zero_val),
        helper.make_node("Identity", ["off_raw"], [off_name]),
        helper.make_node("ConstantOfShape", ["conf_shape"], ["conf_raw"], value=half_val),
        helper.make_node("Identity", ["conf_raw"], [conf_name]),
        helper.make_node("ConstantOfShape", ["conf_shape"], ["pres_raw"], value=zero_val),
        helper.make_node("Identity", ["pres_raw"], [pres_name]),
    ]

    graph = helper.make_graph(
        nodes,
        "test_joints",
        [input_tensor],
        [off_output, conf_output, pres_output],
        initializer=[off_shape, conf_shape, zero_val, half_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _make_weight_onnx(path: Path, *, wrong_names: bool = False) -> Path:
    """Create a minimal ONNX model mimicking weight prediction output contract."""
    n_vertices = 512 * 512
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 512, 512])

    w_name = "wrong_weights" if wrong_names else "weights"
    conf_name = "wrong_conf" if wrong_names else "confidence"

    w_output = helper.make_tensor_value_info(w_name, TensorProto.FLOAT, [1, 20, n_vertices, 1])
    conf_output = helper.make_tensor_value_info(conf_name, TensorProto.FLOAT, [1, 1, n_vertices, 1])

    w_shape = helper.make_tensor("w_shape", TensorProto.INT64, [4], [1, 20, n_vertices, 1])
    conf_shape = helper.make_tensor("conf_shape", TensorProto.INT64, [4], [1, 1, n_vertices, 1])
    # Use 0.05 for weights (will be ~uniform across 20 bones, sums to 1.0)
    weight_val = helper.make_tensor("weight_val", TensorProto.FLOAT, [1], [0.05])
    half_val = helper.make_tensor("half_val", TensorProto.FLOAT, [1], [0.5])

    nodes = [
        helper.make_node("ConstantOfShape", ["w_shape"], ["w_raw"], value=weight_val),
        helper.make_node("Identity", ["w_raw"], [w_name]),
        helper.make_node("ConstantOfShape", ["conf_shape"], ["conf_raw"], value=half_val),
        helper.make_node("Identity", ["conf_raw"], [conf_name]),
    ]

    graph = helper.make_graph(
        nodes,
        "test_weights",
        [input_tensor],
        [w_output, conf_output],
        initializer=[w_shape, conf_shape, weight_val, half_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------------------
# Tests — correct models pass validation
# ---------------------------------------------------------------------------


class TestSegmentationValidation:
    def test_valid_model_passes(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_segmentation_onnx(tmp_path / "segmentation.onnx")
        report = validate_model("segmentation", onnx_path, check_file_size=False)
        report.print_report()
        assert report.all_passed

    def test_wrong_names_fails(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_segmentation_onnx(tmp_path / "segmentation.onnx", wrong_names=True)
        report = validate_model("segmentation", onnx_path, check_file_size=False)
        report.print_report()
        assert not report.all_passed

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        report = validate_model("segmentation", tmp_path / "nonexistent.onnx")
        assert not report.all_passed
        assert any("not found" in c.message.lower() for c in report.checks)


class TestJointValidation:
    def test_valid_model_passes(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_joint_onnx(tmp_path / "joint_refinement.onnx")
        report = validate_model("joints", onnx_path, check_file_size=False)
        report.print_report()
        assert report.all_passed

    def test_wrong_names_fails(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_joint_onnx(tmp_path / "joint_refinement.onnx", wrong_names=True)
        report = validate_model("joints", onnx_path, check_file_size=False)
        report.print_report()
        assert not report.all_passed


class TestWeightValidation:
    def test_valid_model_passes(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_weight_onnx(tmp_path / "weight_prediction.onnx")
        report = validate_model("weights", onnx_path, check_file_size=False)
        report.print_report()
        assert report.all_passed

    def test_wrong_names_fails(self, tmp_path: Path) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_weight_onnx(tmp_path / "weight_prediction.onnx", wrong_names=True)
        report = validate_model("weights", onnx_path, check_file_size=False)
        report.print_report()
        assert not report.all_passed


# ---------------------------------------------------------------------------
# Tests — CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_single_model_pass(self, tmp_path: Path) -> None:
        from training.validate_onnx import main

        onnx_path = _make_joint_onnx(tmp_path / "joints.onnx")
        exit_code = main(["--model", "joints", "--path", str(onnx_path), "--no-check-file-size"])
        assert exit_code == 0

    def test_single_model_fail(self, tmp_path: Path) -> None:
        from training.validate_onnx import main

        onnx_path = _make_joint_onnx(tmp_path / "joints.onnx", wrong_names=True)
        exit_code = main(["--model", "joints", "--path", str(onnx_path), "--no-check-file-size"])
        assert exit_code == 1

    def test_all_models_pass(self, tmp_path: Path) -> None:
        from training.validate_onnx import main

        _make_segmentation_onnx(tmp_path / "segmentation.onnx")
        _make_joint_onnx(tmp_path / "joint_refinement.onnx")
        _make_weight_onnx(tmp_path / "weight_prediction.onnx")
        exit_code = main(["--all", "--models-dir", str(tmp_path), "--no-check-file-size"])
        assert exit_code == 0

    def test_all_models_one_missing_fails(self, tmp_path: Path) -> None:
        from training.validate_onnx import main

        _make_segmentation_onnx(tmp_path / "segmentation.onnx")
        _make_joint_onnx(tmp_path / "joint_refinement.onnx")
        # weight_prediction.onnx is missing
        exit_code = main(["--all", "--models-dir", str(tmp_path)])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Tests — cross-validation with real image
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_cross_validation_with_synthetic_image(self, tmp_path: Path) -> None:
        """Cross-validation runs when dataset_dir has an image."""
        from PIL import Image

        from training.validate_onnx import validate_model

        # Create a synthetic "dataset" with a test image
        dataset_dir = tmp_path / "dataset" / "example_001"
        dataset_dir.mkdir(parents=True)
        img = Image.new("RGB", (512, 512), color=(128, 64, 200))
        img.save(dataset_dir / "image.png")

        onnx_path = _make_segmentation_onnx(tmp_path / "segmentation.onnx")
        report = validate_model(
            "segmentation", onnx_path, dataset_dir=tmp_path / "dataset", check_file_size=False
        )
        report.print_report()
        # The model is trivial (all zeros), so cross-validation should note
        # degenerate output — but the validation itself should run
        assert any("cross-validation" in c.message.lower() for c in report.checks)

    def test_cross_validation_no_images(self, tmp_path: Path) -> None:
        """Cross-validation reports failure when no images found."""
        from training.validate_onnx import validate_model

        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir()

        onnx_path = _make_segmentation_onnx(tmp_path / "segmentation.onnx")
        report = validate_model(
            "segmentation", onnx_path, dataset_dir=empty_dir, check_file_size=False
        )
        report.print_report()
        assert any("no images found" in c.message.lower() for c in report.checks)


# ---------------------------------------------------------------------------
# Tests — report output format
# ---------------------------------------------------------------------------


class TestReportFormat:
    def test_report_shows_pass(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_joint_onnx(tmp_path / "joints.onnx")
        report = validate_model("joints", onnx_path, check_file_size=False)
        report.print_report()
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_report_shows_fail(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from training.validate_onnx import validate_model

        onnx_path = _make_joint_onnx(tmp_path / "joints.onnx", wrong_names=True)
        report = validate_model("joints", onnx_path, check_file_size=False)
        report.print_report()
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
