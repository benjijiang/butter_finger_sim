"""Validate the YAML configuration files and the config loader."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from butter_finger.config import (
    JOINT_NAMES,
    JointLimits,
    load_arm_config,
    load_camera_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="module")
def joints_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "joints.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def poses_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "poses.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def geometry_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "geometry.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def camera_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "camera.yaml").read_text(encoding="utf-8"))


def test_confirmed_pwm_port_mapping(joints_cfg: dict) -> None:
    assert joints_cfg["physical"]["pwm_ports"] == {
        "base": 1,
        "shoulder": 3,
        "elbow": 4,
        "wrist": 5,
    }
    assert joints_cfg["physical"]["unused_ports"] == [2, 6]


def test_recorded_home_pwm(joints_cfg: dict) -> None:
    assert joints_cfg["physical"]["home_pwm_us"] == {
        "base": 1500,
        "shoulder": 2200,
        "elbow": 2490,
        "wrist": 1400,
    }


def test_verified_pwm_limits(joints_cfg: dict) -> None:
    limits = joints_cfg["physical"]["tested_pwm_limits_us"]
    assert limits["shoulder"] == {"min": 1200, "max": 2220}
    for joint in ("base", "elbow", "wrist"):
        assert limits[joint] == {"min": 505, "max": 2495}, (
            f"{joint} PWM limits were verified as 505-2495 us on 2026-07-13"
        )


def test_home_pwm_within_tested_limits(joints_cfg: dict) -> None:
    home = joints_cfg["physical"]["home_pwm_us"]
    limits = joints_cfg["physical"]["tested_pwm_limits_us"]
    for joint, pulse in home.items():
        assert limits[joint]["min"] <= pulse <= limits[joint]["max"], (
            f"{joint} home pulse {pulse} us is outside its tested range"
        )


def test_shoulder_home_note_present(joints_cfg: dict) -> None:
    note = joints_cfg["physical"]["notes"]["shoulder_home"]
    assert "2200" in note
    assert "2220" in note
    assert "revalidated" in note.lower()


def test_simulation_limits(joints_cfg: dict) -> None:
    limits = joints_cfg["simulation"]["limits_rad"]
    assert set(limits) == set(JOINT_NAMES)
    assert limits == {
        "base": {"lower": -1.5708, "upper": 1.5708},
        "shoulder": {"lower": -1.571, "upper": 0.050},
        "elbow": {"lower": -3.075, "upper": 0.065},
        "wrist": {"lower": -2.071, "upper": 1.0708},
    }


def test_simulation_limits_clamp_slider_rounding() -> None:
    limits = JointLimits(lower_rad=-3.075, upper_rad=0.065)
    assert limits.clamp(-3.075000047683716) == -3.075
    assert limits.clamp(0.0650000050663948) == 0.065
    assert limits.clamp(-0.9) == -0.9


def test_simulation_zero_offsets(joints_cfg: dict) -> None:
    offsets = joints_cfg["simulation"]["zero_offset_rad"]
    assert offsets == {
        "base": 0.0,
        "shoulder": 0.0,
        "elbow": 0.0,
        "wrist": 0.0,
    }
    assert all(math.isfinite(float(offset)) for offset in offsets.values())


def test_sim_home_pose(poses_cfg: dict) -> None:
    home = poses_cfg["poses"]["sim_home"]
    assert home == {
        "base": 0.0,
        "shoulder": 0.0,
        "elbow": 0.0,
        "wrist": -1.571,
    }


def test_wrist_range_extends_below_home_by_point_five_rad(
    joints_cfg: dict, poses_cfg: dict
) -> None:
    lower = joints_cfg["simulation"]["limits_rad"]["wrist"]["lower"]
    home = poses_cfg["poses"]["sim_home"]["wrist"]
    assert home - lower == pytest.approx(0.5)


def test_cad_link_properties_are_positive(geometry_cfg: dict) -> None:
    assert set(geometry_cfg["links"]) == {
        "base_link",
        "turret_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_link",
    }
    for name, link in geometry_cfg["links"].items():
        assert link["mass_kg"] > 0, f"mass of {name} must be positive"
        assert link["mesh"].endswith(".stl")
        for diagonal in ("ixx", "iyy", "izz"):
            assert link["inertia"][diagonal] > 0


def test_reference_drawing_dimensions(geometry_cfg: dict) -> None:
    dimensions = geometry_cfg["reference_dimensions_m"]
    assert dimensions["base_height"] == pytest.approx(0.03253)
    assert dimensions["shoulder_joint_height"] == pytest.approx(0.07390)
    assert dimensions["upper_arm_length"] == pytest.approx(0.1452)
    assert dimensions["forearm_length"] == pytest.approx(0.100)
    assert dimensions["wrist_length"] == pytest.approx(0.049)
    assert dimensions["shoulder_joint_height"] > dimensions["base_height"]


def test_cad_joint_names_are_stable(geometry_cfg: dict) -> None:
    assert tuple(geometry_cfg["joints"]) == (
        "base_joint",
        "shoulder_joint",
        "elbow_joint",
        "wrist_joint",
    )


def test_recorded_camera_mount_and_optical_axes(
    geometry_cfg: dict, camera_cfg: dict
) -> None:
    mount = geometry_cfg["camera_mount"]
    optical = camera_cfg["optical_frame"]
    assert mount["parent"] == "wrist_link"
    assert mount["child"] == "camera_link"
    assert mount["origin_xyz"] == pytest.approx([0.011, 0.044, 0.013])
    assert mount["origin_rpy"] == pytest.approx([0.0, 0.0, 0.0])
    assert optical["link_name"] == "camera_link"
    assert optical["forward_xyz"] == pytest.approx([0.0, 1.0, 0.0])
    assert optical["up_xyz"] == pytest.approx([-1.0, 0.0, 0.0])


def test_camera_stream_and_projection_config(camera_cfg: dict) -> None:
    assert camera_cfg["native"] == {
        "width": 640,
        "height": 480,
        "fps": 30,
        "pixel_format": "YUYV 4:2:2",
    }
    assert camera_cfg["projection"] == {
        "model": "pinhole",
        "vertical_fov_deg": 120.0,
        "near_plane_m": 0.02,
        "far_plane_m": 2.0,
        "calibrated": False,
    }
    assert camera_cfg["output"] == {
        "pixel_format": "RGB",
        "rotate_clockwise_deg": 90,
        "width": 480,
        "height": 640,
    }


def test_load_camera_config() -> None:
    config = load_camera_config()
    assert config.native_aspect_ratio == pytest.approx(4.0 / 3.0)
    assert config.output_shape == (640, 480, 3)
    assert config.intrinsics_calibrated is False


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("projection", "vertical_fov_deg", 180.0, "between 0 and 180"),
        ("projection", "near_plane_m", 2.0, "near_plane_m < far_plane_m"),
        ("output", "width", 640, "must swap native"),
        ("optical_frame", "up_xyz", [0.0, 1.0, 0.0], "must be orthogonal"),
    ],
)
def test_rejects_invalid_camera_config(
    tmp_path: Path,
    camera_cfg: dict,
    section: str,
    key: str,
    value,
    message: str,
) -> None:
    invalid = copy.deepcopy(camera_cfg)
    invalid[section][key] = value
    (tmp_path / "camera.yaml").write_text(
        yaml.safe_dump(invalid),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_camera_config(tmp_path)


def test_load_arm_config() -> None:
    config = load_arm_config()
    assert config.joint_order == JOINT_NAMES
    assert config.control_rate_hz == 240
    assert config.time_step_s == pytest.approx(1.0 / 240.0)
    assert set(config.home_pose) == set(JOINT_NAMES)
    for joint, angle in config.home_pose.items():
        assert config.sim_limits[joint].contains(angle)
