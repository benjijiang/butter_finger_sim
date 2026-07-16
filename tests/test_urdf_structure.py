"""Validate the generated URDF structure without PyBullet.

Uses only the standard library (xml.etree.ElementTree), so it runs on any
machine, including one without PyBullet installed.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "models" / "butter_finger_simple.urdf"
GEOMETRY_PATH = REPO_ROOT / "config" / "geometry.yaml"

EXPECTED_REVOLUTE_JOINTS = ["base_joint", "shoulder_joint", "elbow_joint", "wrist_joint"]


@pytest.fixture(scope="module")
def robot() -> ET.Element:
    assert URDF_PATH.is_file(), (
        f"{URDF_PATH} missing; run 'python scripts/generate_urdf.py'"
    )
    return ET.parse(URDF_PATH).getroot()


@pytest.fixture(scope="module")
def geometry() -> dict:
    return yaml.safe_load(GEOMETRY_PATH.read_text(encoding="utf-8"))


def test_root_is_robot(robot: ET.Element) -> None:
    assert robot.tag == "robot"
    assert robot.get("name") == "butter_finger"


def test_exact_four_revolute_joints(robot: ET.Element) -> None:
    revolute = [j.get("name") for j in robot.findall("joint") if j.get("type") == "revolute"]
    assert revolute == EXPECTED_REVOLUTE_JOINTS


def test_camera_link_exists_and_is_fixed(robot: ET.Element) -> None:
    link_names = {link.get("name") for link in robot.findall("link")}
    assert "camera_link" in link_names
    camera_joints = [
        j for j in robot.findall("joint")
        if j.find("child") is not None and j.find("child").get("link") == "camera_link"
    ]
    assert len(camera_joints) == 1
    assert camera_joints[0].get("type") == "fixed"


def test_base_link_is_the_root(robot: ET.Element) -> None:
    """base_link must be the URDF root so useFixedBase=True fixes the base."""
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    link_names = {link.get("name") for link in robot.findall("link")}
    roots = link_names - children
    assert roots == {"base_link"}


def test_kinematic_chain_is_connected(robot: ET.Element) -> None:
    link_names = {link.get("name") for link in robot.findall("link")}
    for joint in robot.findall("joint"):
        assert joint.find("parent").get("link") in link_names
        assert joint.find("child").get("link") in link_names


def test_cad_links_have_visual_collision_inertial(robot: ET.Element) -> None:
    for link in robot.findall("link"):
        name = link.get("name")
        if name == "camera_link":
            continue
        assert link.find("visual") is not None, f"{name} has no <visual>"
        assert link.find("collision") is not None, f"{name} has no <collision>"
        assert link.find("inertial") is not None, f"{name} has no <inertial>"


def test_masses_and_inertias_are_positive(robot: ET.Element) -> None:
    for link in robot.findall("link"):
        name = link.get("name")
        if name == "camera_link":
            continue
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        assert mass > 0, f"{name} mass must be positive"
        inertia = inertial.find("inertia")
        for axis in ("ixx", "iyy", "izz"):
            value = float(inertia.get(axis))
            assert value > 0, f"{name} {axis} must be positive"
            assert math.isfinite(value)


def test_revolute_joints_have_valid_limits(robot: ET.Element) -> None:
    for joint in robot.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        assert limit is not None, f"{joint.get('name')} has no <limit>"
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        assert lower < upper
        assert float(limit.get("effort")) > 0
        assert float(limit.get("velocity")) > 0


def test_joint_transforms_match_cad_geometry(robot: ET.Element, geometry: dict) -> None:
    for joint_name, expected in geometry["joints"].items():
        origin = robot.find(f"./joint[@name='{joint_name}']/origin")
        assert origin is not None
        actual_xyz = tuple(float(value) for value in origin.get("xyz").split())
        actual_rpy = tuple(float(value) for value in origin.get("rpy").split())
        actual_axis = tuple(
            float(value)
            for value in robot.find(f"./joint[@name='{joint_name}']/axis")
            .get("xyz")
            .split()
        )
        assert actual_xyz == pytest.approx(expected["origin_xyz"])
        assert actual_rpy == pytest.approx(expected["origin_rpy"])
        assert actual_axis == pytest.approx(expected["axis_xyz"])


def test_camera_transform_matches_geometry(robot: ET.Element, geometry: dict) -> None:
    origin = robot.find("./joint[@name='camera_joint']/origin")
    assert origin is not None
    actual_xyz = tuple(float(value) for value in origin.get("xyz").split())
    actual_rpy = tuple(float(value) for value in origin.get("rpy").split())
    assert actual_xyz == pytest.approx((0.011, 0.044, 0.013))
    assert actual_rpy == pytest.approx((0.0, 0.0, 0.0))
    assert actual_xyz == pytest.approx(geometry["camera_mount"]["origin_xyz"])
    assert actual_rpy == pytest.approx(geometry["camera_mount"]["origin_rpy"])


def test_cad_meshes_exist(robot: ET.Element) -> None:
    for mesh in robot.findall("./link/visual/geometry/mesh"):
        path = URDF_PATH.parent / mesh.get("filename")
        assert path.is_file(), f"missing CAD mesh: {path}"
