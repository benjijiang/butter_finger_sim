#!/usr/bin/env python3
"""Generate models/butter_finger_simple.urdf from the config files.

Run this on the sim machine (or any machine with PyYAML installed) after editing
config/geometry.yaml or the simulation settings in config/joints.yaml:

    python scripts/generate_urdf.py

Primary joint-to-joint dimensions come from a user-provided reference drawing.
Primitive shapes, cross-sections, masses, and inertias remain placeholders until
the real CAD model is recovered. When CAD meshes become available, only the
<visual>/<collision> geometry should change; joint names and the control API
must stay the same.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = REPO_ROOT / "config" / "geometry.yaml"
JOINTS_PATH = REPO_ROOT / "config" / "joints.yaml"
OUTPUT_PATH = REPO_ROOT / "models" / "butter_finger_simple.urdf"


def fmt(value: float) -> str:
    """Format a number for URDF output, stable across regenerations."""
    return f"{value:.8g}"


def axis_offset_rpy(axis: str, offset_rad: float) -> str:
    """Express a zero-configuration offset as URDF roll/pitch/yaw."""
    rpy_by_axis = {
        "1 0 0": (offset_rad, 0.0, 0.0),
        "0 1 0": (0.0, offset_rad, 0.0),
        "0 0 1": (0.0, 0.0, offset_rad),
    }
    try:
        rpy = rpy_by_axis[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported joint axis for zero offset: {axis!r}") from exc
    return " ".join(fmt(value) for value in rpy)


def box_inertia(mass: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Diagonal inertia of a solid box about its center of mass."""
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def cylinder_inertia(mass: float, radius: float, height: float) -> tuple[float, float, float]:
    """Diagonal inertia of a solid cylinder (axis along Z) about its center of mass."""
    ixx = mass * (3.0 * radius * radius + height * height) / 12.0
    return (ixx, ixx, mass * radius * radius / 2.0)


def cylinder_link(
    name: str,
    radius: float,
    length: float,
    mass: float,
    material: str,
    rgba: str,
    comment: str,
) -> str:
    """Cylinder link extending +Z from the link origin (its parent joint)."""
    z = length / 2.0
    ixx, iyy, izz = cylinder_inertia(mass, radius, length)
    return f"""
  <!-- {comment} -->
  <link name="{name}">
    <visual>
      <origin xyz="0 0 {fmt(z)}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{fmt(radius)}" length="{fmt(length)}"/>
      </geometry>
      <material name="{material}">
        <color rgba="{rgba}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {fmt(z)}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{fmt(radius)}" length="{fmt(length)}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {fmt(z)}" rpy="0 0 0"/>
      <mass value="{fmt(mass)}"/>
      <inertia ixx="{fmt(ixx)}" ixy="0" ixz="0" iyy="{fmt(iyy)}" iyz="0" izz="{fmt(izz)}"/>
    </inertial>
  </link>
"""


def box_link(
    name: str,
    width: float,
    length: float,
    mass: float,
    material: str,
    rgba: str,
    comment: str,
) -> str:
    """Box link extending +Z from the link origin (its parent joint)."""
    z = length / 2.0
    ixx, iyy, izz = box_inertia(mass, width, width, length)
    return f"""
  <!-- {comment} -->
  <link name="{name}">
    <visual>
      <origin xyz="0 0 {fmt(z)}" rpy="0 0 0"/>
      <geometry>
        <box size="{fmt(width)} {fmt(width)} {fmt(length)}"/>
      </geometry>
      <material name="{material}">
        <color rgba="{rgba}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {fmt(z)}" rpy="0 0 0"/>
      <geometry>
        <box size="{fmt(width)} {fmt(width)} {fmt(length)}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {fmt(z)}" rpy="0 0 0"/>
      <mass value="{fmt(mass)}"/>
      <inertia ixx="{fmt(ixx)}" ixy="0" ixz="0" iyy="{fmt(iyy)}" iyz="0" izz="{fmt(izz)}"/>
    </inertial>
  </link>
"""


def revolute_joint(
    name: str,
    parent: str,
    child: str,
    origin_z: float,
    zero_offset_rad: float,
    axis: str,
    lower: float,
    upper: float,
    effort: float,
    velocity: float,
    comment: str,
) -> str:
    origin_rpy = axis_offset_rpy(axis, zero_offset_rad)
    return f"""
  <!-- {comment} -->
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="0 0 {fmt(origin_z)}" rpy="{origin_rpy}"/>
    <axis xyz="{axis}"/>
    <limit lower="{fmt(lower)}" upper="{fmt(upper)}" effort="{fmt(effort)}" velocity="{fmt(velocity)}"/>
  </joint>
"""


def fixed_joint(name: str, parent: str, child: str, origin_z: float, comment: str) -> str:
    return f"""
  <!-- {comment} -->
  <joint name="{name}" type="fixed">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="0 0 {fmt(origin_z)}" rpy="0 0 0"/>
  </joint>
"""


def main() -> None:
    geo = yaml.safe_load(GEOMETRY_PATH.read_text(encoding="utf-8"))
    joints_cfg = yaml.safe_load(JOINTS_PATH.read_text(encoding="utf-8"))

    masses = geo["masses_kg"]
    zero_offsets = joints_cfg["simulation"]["zero_offset_rad"]
    limits = joints_cfg["simulation"]["limits_rad"]
    control = joints_cfg["simulation"]["control"]
    effort = control["max_force_nm"]
    velocity = control["max_velocity_rad_s"]

    width = geo["link_width"]
    turret_height = geo["shoulder_joint_height"] - geo["base_height"]
    if turret_height <= 0:
        raise ValueError("shoulder_joint_height must be greater than base_height")

    def lim(joint: str) -> tuple[float, float]:
        return limits[joint]["lower"], limits[joint]["upper"]

    parts: list[str] = []
    parts.append(
        """<?xml version="1.0"?>
<!--
  Butter Finger: simplified placeholder model of the four-joint desktop arm.

  GENERATED FILE. Do not edit by hand.
  Regenerate with: python scripts/generate_urdf.py
  Dimension source: config/geometry.yaml
  Joint source:     config/joints.yaml (simulation limits and zero offsets)

  MODEL ASSUMPTIONS:
  - Primary link lengths come from a user-provided dimensioned drawing dated
    2026-07-15; they have not been independently verified against CAD.
  - Primitive shapes, cross-sections, masses, and inertias remain placeholders
    and are NOT suitable for payload, structural, or servo-torque conclusions.
  - The wrist joint axis (Y) is an UNVERIFIED assumption; verify against
    the physical arm.
  - Joint limits of +/- 1.5708 rad are simulation-development limits, not
    physical calibration.
  - The camera transform is a guess; measure it on the real arm.
  - Load with useFixedBase=True so the base stays fixed in PyBullet.
-->
<robot name="butter_finger">
"""
    )

    parts.append(
        cylinder_link(
            "base_link",
            geo["base_radius"],
            geo["base_height"],
            masses["base"],
            "base_grey",
            "0.25 0.25 0.25 1",
            "base_link: fixed base cylinder (placeholder dimensions)",
        )
    )
    parts.append(
        revolute_joint(
            "base_joint",
            "base_link",
            "turret_link",
            geo["base_height"],
            zero_offsets["base"],
            "0 0 1",
            *lim("base"),
            effort,
            velocity,
            "base_joint: yaw about the vertical Z axis",
        )
    )
    parts.append(
        cylinder_link(
            "turret_link",
            geo["turret_radius"],
            turret_height,
            masses["turret"],
            "turret_grey",
            "0.45 0.45 0.45 1",
            "turret_link: small yaw platform between base and shoulder (placeholder)",
        )
    )
    parts.append(
        revolute_joint(
            "shoulder_joint",
            "turret_link",
            "upper_arm_link",
            turret_height,
            zero_offsets["shoulder"],
            "0 1 0",
            *lim("shoulder"),
            effort,
            velocity,
            "shoulder_joint: pitch about the horizontal Y axis",
        )
    )
    parts.append(
        box_link(
            "upper_arm_link",
            width,
            geo["upper_arm_length"],
            masses["upper_arm"],
            "arm_yellow",
            "0.95 0.8 0.2 1",
            "upper_arm_link: extends +Z from the shoulder joint",
        )
    )
    parts.append(
        revolute_joint(
            "elbow_joint",
            "upper_arm_link",
            "forearm_link",
            geo["upper_arm_length"],
            zero_offsets["elbow"],
            "0 1 0",
            *lim("elbow"),
            effort,
            velocity,
            "elbow_joint: pitch about a horizontal Y axis parallel to the shoulder",
        )
    )
    parts.append(
        box_link(
            "forearm_link",
            width,
            geo["forearm_length"],
            masses["forearm"],
            "forearm_orange",
            "0.9 0.6 0.15 1",
            "forearm_link: extends +Z from the elbow joint",
        )
    )
    parts.append(
        revolute_joint(
            "wrist_joint",
            "forearm_link",
            "wrist_link",
            geo["forearm_length"],
            zero_offsets["wrist"],
            "0 1 0",
            *lim("wrist"),
            effort,
            velocity,
            "wrist_joint: TEMPORARY Y-axis assumption; verify on the physical arm",
        )
    )
    parts.append(
        box_link(
            "wrist_link",
            width,
            geo["wrist_length"],
            masses["wrist"],
            "wrist_grey",
            "0.45 0.45 0.45 1",
            "wrist_link: extends +Z from the wrist joint",
        )
    )
    parts.append(
        fixed_joint(
            "camera_joint",
            "wrist_link",
            "camera_link",
            geo["wrist_length"],
            "camera_joint: fixed mount at the wrist tip; transform is a PLACEHOLDER",
        )
    )
    parts.append(
        box_link(
            "camera_link",
            geo["camera_size"],
            geo["camera_size"],
            masses["camera"],
            "camera_blue",
            "0.2 0.4 0.9 1",
            "camera_link: camera modeled as a small cube (rendering not implemented yet)",
        )
    )
    parts.append("</robot>\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("".join(parts), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
