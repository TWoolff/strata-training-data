"""Tests for the 2D pose estimator module.

All tests exercise pure-Python mapping/builder logic — no ONNX model
weights required.  Integration tests that require model weights are
marked ``@pytest.mark.slow`` and skip automatically if the model is
not available.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.config import NUM_JOINT_REGIONS, REGION_NAMES
from pipeline.pose_estimator import (
    COCO_LEFT_ANKLE,
    COCO_LEFT_ELBOW,
    COCO_LEFT_HIP,
    COCO_LEFT_KNEE,
    COCO_LEFT_SHOULDER,
    COCO_LEFT_WRIST,
    COCO_NOSE,
    COCO_RIGHT_ANKLE,
    COCO_RIGHT_ELBOW,
    COCO_RIGHT_HIP,
    COCO_RIGHT_KNEE,
    COCO_RIGHT_SHOULDER,
    COCO_RIGHT_WRIST,
    INTERPOLATION_CONFIDENCE_FACTOR,
    _compute_bbox,
    _make_joint,
    build_joint_data,
    coco_to_strata,
    estimate_pose,
)

# ---------------------------------------------------------------------------
# Expected region names (regions 1–19, no background)
# ---------------------------------------------------------------------------

EXPECTED_JOINT_NAMES = {REGION_NAMES[rid] for rid in range(1, NUM_JOINT_REGIONS + 1)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_coco_keypoints(
    *,
    center_x: float = 256.0,
    center_y: float = 256.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create plausible COCO 17-point keypoints for a standing figure.

    Returns:
        (keypoints, confidences) — shapes (17, 2) and (17,).
    """
    kp = np.zeros((17, 2), dtype=np.float32)
    cf = np.ones(17, dtype=np.float32) * 0.9

    # Approximate a standing human centered at (center_x, center_y)
    cx, cy = center_x, center_y
    kp[COCO_NOSE] = [cx, cy - 100]  # nose (top of head area)
    kp[1] = [cx - 10, cy - 105]  # left eye
    kp[2] = [cx + 10, cy - 105]  # right eye
    kp[3] = [cx - 20, cy - 95]  # left ear
    kp[4] = [cx + 20, cy - 95]  # right ear
    kp[COCO_LEFT_SHOULDER] = [cx - 40, cy - 60]  # left shoulder
    kp[COCO_RIGHT_SHOULDER] = [cx + 40, cy - 60]  # right shoulder
    kp[COCO_LEFT_ELBOW] = [cx - 55, cy - 20]  # left elbow
    kp[COCO_RIGHT_ELBOW] = [cx + 55, cy - 20]  # right elbow
    kp[COCO_LEFT_WRIST] = [cx - 60, cy + 10]  # left wrist
    kp[COCO_RIGHT_WRIST] = [cx + 60, cy + 10]  # right wrist
    kp[COCO_LEFT_HIP] = [cx - 20, cy + 30]  # left hip
    kp[COCO_RIGHT_HIP] = [cx + 20, cy + 30]  # right hip
    kp[COCO_LEFT_KNEE] = [cx - 22, cy + 80]  # left knee
    kp[COCO_RIGHT_KNEE] = [cx + 22, cy + 80]  # right knee
    kp[COCO_LEFT_ANKLE] = [cx - 25, cy + 130]  # left ankle
    kp[COCO_RIGHT_ANKLE] = [cx + 25, cy + 130]  # right ankle

    return kp, cf


# ---------------------------------------------------------------------------
# coco_to_strata
# ---------------------------------------------------------------------------


class TestCocoToStrata:
    """Test COCO 17-point → Strata 19-region mapping."""

    def test_produces_19_joints(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        assert len(joints) == 19

    def test_all_region_names_present(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        assert set(joints.keys()) == EXPECTED_JOINT_NAMES

    def test_direct_mapping_head(self) -> None:
        """Head joint should map from COCO nose."""
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        # Head position should be near the nose position
        head = joints["head"]
        assert head["position"][0] == round(kp[COCO_NOSE][0])
        assert head["position"][1] == round(kp[COCO_NOSE][1])
        assert head["confidence"] == pytest.approx(cf[COCO_NOSE], abs=0.01)

    def test_direct_mapping_shoulders(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        assert joints["shoulder_l"]["position"][0] == round(kp[COCO_LEFT_SHOULDER][0])
        assert joints["shoulder_r"]["position"][0] == round(kp[COCO_RIGHT_SHOULDER][0])

    def test_hips_is_midpoint(self) -> None:
        """Hips should be the midpoint of both COCO hips."""
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        expected_x = round((kp[COCO_LEFT_HIP][0] + kp[COCO_RIGHT_HIP][0]) / 2)
        expected_y = round((kp[COCO_LEFT_HIP][1] + kp[COCO_RIGHT_HIP][1]) / 2)
        assert joints["hips"]["position"] == [expected_x, expected_y]

    def test_interpolated_joints_have_reduced_confidence(self) -> None:
        """Interpolated joints should have confidence × 0.8."""
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))

        interpolated = [
            "neck",
            "chest",
            "spine",
            "hand_l",
            "hand_r",
            "foot_l",
            "foot_r",
        ]
        for name in interpolated:
            assert joints[name]["confidence"] < cf[0], f"{name} confidence should be reduced"
            # All test confidences are 0.9 → interpolated = 0.9 * 0.8 = 0.72
            assert joints[name]["confidence"] == pytest.approx(
                0.9 * INTERPOLATION_CONFIDENCE_FACTOR,
                abs=0.01,
            )

    def test_neck_position_between_nose_and_shoulders(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        neck = joints["neck"]["position"]
        nose_y = kp[COCO_NOSE][1]
        shoulder_mid_y = (kp[COCO_LEFT_SHOULDER][1] + kp[COCO_RIGHT_SHOULDER][1]) / 2
        # Neck Y should be between nose and shoulder midpoint
        assert min(nose_y, shoulder_mid_y) <= neck[1] <= max(nose_y, shoulder_mid_y)

    def test_chest_closer_to_shoulders_than_hips(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        chest_y = joints["chest"]["position"][1]
        shoulder_mid_y = (kp[COCO_LEFT_SHOULDER][1] + kp[COCO_RIGHT_SHOULDER][1]) / 2
        hip_mid_y = (kp[COCO_LEFT_HIP][1] + kp[COCO_RIGHT_HIP][1]) / 2
        # Chest is 1/3 from shoulders → closer to shoulders
        assert abs(chest_y - shoulder_mid_y) < abs(chest_y - hip_mid_y)

    def test_spine_closer_to_hips_than_shoulders(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        spine_y = joints["spine"]["position"][1]
        shoulder_mid_y = (kp[COCO_LEFT_SHOULDER][1] + kp[COCO_RIGHT_SHOULDER][1]) / 2
        hip_mid_y = (kp[COCO_LEFT_HIP][1] + kp[COCO_RIGHT_HIP][1]) / 2
        # Spine is 2/3 from shoulders → closer to hips
        assert abs(spine_y - hip_mid_y) < abs(spine_y - shoulder_mid_y)

    def test_upper_arm_at_elbow(self) -> None:
        """upper_arm should be placed at the elbow (COCO elbow keypoint)."""
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        expected_x = round(float(kp[COCO_LEFT_ELBOW][0]))
        expected_y = round(float(kp[COCO_LEFT_ELBOW][1]))
        assert joints["upper_arm_l"]["position"] == [expected_x, expected_y]

    def test_upper_leg_at_knee(self) -> None:
        """upper_leg should be placed at the knee (COCO knee keypoint)."""
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        expected_x = round(float(kp[COCO_LEFT_KNEE][0]))
        expected_y = round(float(kp[COCO_LEFT_KNEE][1]))
        assert joints["upper_leg_l"]["position"] == [expected_x, expected_y]

    def test_positions_clamped_to_image_bounds(self) -> None:
        """Keypoints outside image bounds should be clamped."""
        kp, cf = _dummy_coco_keypoints()
        kp[COCO_NOSE] = [-10, -20]  # out of bounds
        joints = coco_to_strata(kp, cf, (512, 512))
        head = joints["head"]
        assert head["position"][0] >= 0
        assert head["position"][1] >= 0

    def test_low_confidence_marks_invisible(self) -> None:
        """Joints with confidence below threshold should be invisible."""
        kp, cf = _dummy_coco_keypoints()
        cf[COCO_NOSE] = 0.1  # below default threshold of 0.3
        joints = coco_to_strata(kp, cf, (512, 512))
        assert joints["head"]["visible"] is False

    def test_high_confidence_marks_visible(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        joints = coco_to_strata(kp, cf, (512, 512))
        assert joints["head"]["visible"] is True

    def test_zero_confidence_all_invisible(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        cf[:] = 0.0
        joints = coco_to_strata(kp, cf, (512, 512))
        for name, joint in joints.items():
            assert joint["visible"] is False, f"{name} should be invisible"


# ---------------------------------------------------------------------------
# _make_joint
# ---------------------------------------------------------------------------


class TestMakeJoint:
    """Test single joint dict construction."""

    def test_basic_output(self) -> None:
        pos = np.array([100.0, 200.0])
        joint = _make_joint(pos, 0.95, 512, 512)
        assert joint["position"] == [100, 200]
        assert joint["confidence"] == 0.95
        assert joint["visible"] is True

    def test_clamps_to_bounds(self) -> None:
        pos = np.array([600.0, -10.0])
        joint = _make_joint(pos, 0.9, 512, 512)
        assert joint["position"] == [511, 0]

    def test_below_threshold_invisible(self) -> None:
        pos = np.array([100.0, 200.0])
        joint = _make_joint(pos, 0.1, 512, 512)
        assert joint["visible"] is False

    def test_rounds_position(self) -> None:
        pos = np.array([100.7, 200.3])
        joint = _make_joint(pos, 0.9, 512, 512)
        assert joint["position"] == [101, 200]

    def test_custom_threshold(self) -> None:
        pos = np.array([100.0, 200.0])
        joint = _make_joint(pos, 0.5, 512, 512, confidence_threshold=0.6)
        assert joint["visible"] is False
        joint = _make_joint(pos, 0.7, 512, 512, confidence_threshold=0.6)
        assert joint["visible"] is True


# ---------------------------------------------------------------------------
# _compute_bbox
# ---------------------------------------------------------------------------


class TestComputeBbox:
    """Test bounding box computation from joints."""

    def test_basic_bbox(self) -> None:
        joints = {
            "head": {"position": [256, 50], "confidence": 0.9, "visible": True},
            "foot_l": {"position": [200, 400], "confidence": 0.9, "visible": True},
            "foot_r": {"position": [300, 400], "confidence": 0.9, "visible": True},
        }
        bbox = _compute_bbox(joints, 512, 512)
        assert len(bbox) == 4
        x_min, y_min, x_max, y_max = bbox
        assert x_min <= 200
        assert y_min <= 50
        assert x_max >= 300
        assert y_max >= 400

    def test_no_visible_joints_returns_full_image(self) -> None:
        joints = {
            "head": {"position": [256, 50], "confidence": 0.1, "visible": False},
        }
        bbox = _compute_bbox(joints, 512, 512)
        assert bbox == [0, 0, 512, 512]

    def test_bbox_clamped_to_image(self) -> None:
        joints = {
            "head": {"position": [5, 5], "confidence": 0.9, "visible": True},
        }
        bbox = _compute_bbox(joints, 512, 512)
        x_min, y_min, x_max, y_max = bbox
        assert x_min >= 0
        assert y_min >= 0
        assert x_max <= 512
        assert y_max <= 512


# ---------------------------------------------------------------------------
# build_joint_data
# ---------------------------------------------------------------------------


class TestBuildJointData:
    """Test full joint data dict construction."""

    def test_output_schema(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        strata_joints = coco_to_strata(kp, cf, (512, 512))
        data = build_joint_data("test_char_001", strata_joints, (512, 512))

        assert data["character_id"] == "test_char_001"
        assert data["pose_name"] == "default"
        assert data["source_animation"] == ""
        assert data["source_frame"] == 0
        assert data["image_size"] == [512, 512]
        assert len(data["joints"]) == 19
        assert len(data["bbox"]) == 4

    def test_all_region_names_in_output(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        strata_joints = coco_to_strata(kp, cf, (512, 512))
        data = build_joint_data("test_char", strata_joints, (512, 512))
        assert set(data["joints"].keys()) == EXPECTED_JOINT_NAMES

    def test_json_serializable(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        strata_joints = coco_to_strata(kp, cf, (512, 512))
        data = build_joint_data("test_char", strata_joints, (512, 512))
        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["character_id"] == "test_char"

    def test_joint_positions_are_int_lists(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        strata_joints = coco_to_strata(kp, cf, (512, 512))
        data = build_joint_data("test_char", strata_joints, (512, 512))
        for name, joint in data["joints"].items():
            assert isinstance(joint["position"], list), f"{name} position not a list"
            assert len(joint["position"]) == 2
            assert isinstance(joint["position"][0], int), f"{name} x not int"
            assert isinstance(joint["position"][1], int), f"{name} y not int"


# ---------------------------------------------------------------------------
# estimate_pose (mocked model)
# ---------------------------------------------------------------------------


class TestEstimatePose:
    """Test pose estimation with mocked model."""

    def test_single_detection(self) -> None:
        kp, cf = _dummy_coco_keypoints()
        # Shape: (1, 17, 2) and (1, 17)
        mock_model = MagicMock()
        mock_model.return_value = (kp[np.newaxis], cf[np.newaxis])

        result = estimate_pose(mock_model, np.zeros((512, 512, 3), dtype=np.uint8))
        assert result is not None
        keypoints, scores = result
        assert keypoints.shape == (17, 2)
        assert scores.shape == (17,)

    def test_no_detection_returns_none(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = (np.array([]), np.array([]))

        result = estimate_pose(mock_model, np.zeros((512, 512, 3), dtype=np.uint8))
        assert result is None

    def test_none_detection_returns_none(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = (None, None)

        result = estimate_pose(mock_model, np.zeros((512, 512, 3), dtype=np.uint8))
        assert result is None

    def test_multiple_detections_picks_largest(self) -> None:
        """Should pick the person with the largest bounding box."""
        # Person 1: small (100x100 area)
        kp1, cf1 = _dummy_coco_keypoints(center_x=50, center_y=50)
        kp1 *= 0.2  # compress into small area

        # Person 2: large (full body)
        kp2, cf2 = _dummy_coco_keypoints(center_x=256, center_y=256)

        keypoints = np.stack([kp1, kp2])  # (2, 17, 2)
        scores = np.stack([cf1, cf2])  # (2, 17)

        mock_model = MagicMock()
        mock_model.return_value = (keypoints, scores)

        result = estimate_pose(mock_model, np.zeros((512, 512, 3), dtype=np.uint8))
        assert result is not None
        kp_result, _ = result

        # Should have picked person 2 (larger bbox)
        np.testing.assert_array_almost_equal(kp_result, kp2)


# ---------------------------------------------------------------------------
# enrich_example (filesystem interaction)
# ---------------------------------------------------------------------------


class TestEnrichExample:
    """Test the high-level enrich_example function with mocked model."""

    def _setup_example(self, tmp_path: Path) -> Path:
        """Create a minimal example directory with image.png and metadata.json."""
        from PIL import Image

        example_dir = tmp_path / "fbanimehq_0000_000001"
        example_dir.mkdir()

        # Create a simple test image
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(example_dir / "image.png")

        # Create metadata
        meta = {
            "id": "fbanimehq_0000_000001",
            "source": "fbanimehq",
            "has_joints": False,
            "missing_annotations": ["strata_segmentation", "joints", "draw_order", "fg_mask"],
        }
        (example_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

        return example_dir

    def test_writes_joints_json(self, tmp_path: Path) -> None:
        example_dir = self._setup_example(tmp_path)
        kp, cf = _dummy_coco_keypoints()

        mock_model = MagicMock()
        mock_model.return_value = (kp[np.newaxis], cf[np.newaxis])

        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            from pipeline.pose_estimator import enrich_example

            success = enrich_example(mock_model, example_dir, (512, 512))

        assert success is True
        assert (example_dir / "joints.json").is_file()

        joint_data = json.loads((example_dir / "joints.json").read_text())
        assert len(joint_data["joints"]) == 19
        assert joint_data["character_id"] == "fbanimehq_0000_000001"

    def test_updates_metadata(self, tmp_path: Path) -> None:
        example_dir = self._setup_example(tmp_path)
        kp, cf = _dummy_coco_keypoints()

        mock_model = MagicMock()
        mock_model.return_value = (kp[np.newaxis], cf[np.newaxis])

        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            from pipeline.pose_estimator import enrich_example

            enrich_example(mock_model, example_dir, (512, 512))

        meta = json.loads((example_dir / "metadata.json").read_text())
        assert meta["has_joints"] is True
        assert "joints" not in meta["missing_annotations"]
        # Other missing annotations should be preserved
        assert "strata_segmentation" in meta["missing_annotations"]

    def test_no_image_returns_false(self, tmp_path: Path) -> None:
        example_dir = tmp_path / "empty_example"
        example_dir.mkdir()

        mock_model = MagicMock()

        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            from pipeline.pose_estimator import enrich_example

            success = enrich_example(mock_model, example_dir, (512, 512))

        assert success is False

    def test_no_detection_returns_false(self, tmp_path: Path) -> None:
        example_dir = self._setup_example(tmp_path)

        mock_model = MagicMock()
        mock_model.return_value = (np.array([]), np.array([]))

        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            from pipeline.pose_estimator import enrich_example

            success = enrich_example(mock_model, example_dir, (512, 512))

        assert success is False
        assert not (example_dir / "joints.json").exists()


# ---------------------------------------------------------------------------
# Validator compatibility
# ---------------------------------------------------------------------------


class TestValidatorCompatibility:
    """Verify output is compatible with pipeline/validator.py checks."""

    def test_joint_count_check(self) -> None:
        """Output should have exactly NUM_JOINT_REGIONS joints."""
        from pipeline.validator import check_joint_count

        kp, cf = _dummy_coco_keypoints()
        strata_joints = coco_to_strata(kp, cf, (512, 512))
        data = build_joint_data("test", strata_joints, (512, 512))

        passed, detail = check_joint_count(data)
        assert passed, f"Joint count check failed: {detail}"

    def test_joint_bounds_check(self) -> None:
        """All visible joints should be within image bounds."""
        from pipeline.validator import check_joint_bounds

        kp, cf = _dummy_coco_keypoints()
        strata_joints = coco_to_strata(kp, cf, (512, 512))
        data = build_joint_data("test", strata_joints, (512, 512))

        passed, detail = check_joint_bounds(data)
        assert passed, f"Joint bounds check failed: {detail}"
