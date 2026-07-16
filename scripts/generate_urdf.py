#!/usr/bin/env python3
"""Generate the Butter Finger URDF from CAD geometry and simulation config.

Run after editing config/geometry.yaml or the simulation section of
config/joints.yaml:

    python3 scripts/generate_urdf.py

The CAD mesh frames, joint transforms, masses, and inertias came from the
2026-07-16 SolidWorks SW2URDF export. Joint names remain the stable names used
by the ArmBackend API.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = REPO_ROOT / "config" / "geometry.yaml"
JOINTS_PATH = REPO_ROOT / "config" / "joints.yaml"
OUTPUT_PATH = REPO_ROOT / "models" / "butter_finger_simple.urdf"

LINK_ORDER = (
    "base_link",
    "turret_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_link",
)
JOINT_ORDER = (
    "base_joint",
    "shoulder_joint",
    "elbow_joint",
    "wrist_joint",
)


def fmt(value: float) -> str:
    """Format a finite number stably across regenerations."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"URDF values must be finite, got {value!r}")
    return f"{number:.15g}"


def vector(values: list[float], *, label: str) -> str:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return " ".join(fmt(value) for value in values)


def mesh_link(name: str, config: dict[str, Any]) -> str:
    """Render one CAD mesh link with its SolidWorks inertial properties."""
    mesh_path = config["mesh"]
    absolute_mesh_path = OUTPUT_PATH.parent / mesh_path
    if not absolute_mesh_path.is_file():
        raise FileNotFoundError(f"{name}: mesh not found at {absolute_mesh_path}")

    center_of_mass = vector(
        config["center_of_mass_xyz"], label=f"{name}.center_of_mass_xyz"
    )
    inertia = config["inertia"]
    inertia_attributes = " ".join(
        f'{component}="{fmt(inertia[component])}"'
        for component in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    )

    return f"""
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
      <material name="cad_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="{center_of_mass}" rpy="0 0 0"/>
      <mass value="{fmt(config['mass_kg'])}"/>
      <inertia {inertia_attributes}/>
    </inertial>
  </link>
"""


def revolute_joint(
    name: str,
    geometry: dict[str, Any],
    limits: dict[str, float],
    effort: float,
    velocity: float,
) -> str:
    origin_xyz = vector(geometry["origin_xyz"], label=f"{name}.origin_xyz")
    origin_rpy = vector(geometry["origin_rpy"], label=f"{name}.origin_rpy")
    axis_xyz = vector(geometry["axis_xyz"], label=f"{name}.axis_xyz")
    axis_norm = math.sqrt(sum(float(value) ** 2 for value in geometry["axis_xyz"]))
    if axis_norm <= 0:
        raise ValueError(f"{name}.axis_xyz must be non-zero")

    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{geometry['parent']}"/>
    <child link="{geometry['child']}"/>
    <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>
    <axis xyz="{axis_xyz}"/>
    <limit lower="{fmt(limits['lower'])}" upper="{fmt(limits['upper'])}" effort="{fmt(effort)}" velocity="{fmt(velocity)}"/>
  </joint>
"""


def fixed_camera_joint(config: dict[str, Any]) -> str:
    origin_xyz = vector(config["origin_xyz"], label="camera_mount.origin_xyz")
    origin_rpy = vector(config["origin_rpy"], label="camera_mount.origin_rpy")
    return f"""
  <joint name="camera_joint" type="fixed">
    <parent link="{config['parent']}"/>
    <child link="{config['child']}"/>
    <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>
  </joint>
"""


def main() -> None:
    geo = yaml.safe_load(GEOMETRY_PATH.read_text(encoding="utf-8"))
    joints_cfg = yaml.safe_load(JOINTS_PATH.read_text(encoding="utf-8"))

    links = geo["links"]
    joint_geometry = geo["joints"]
    if tuple(links) != LINK_ORDER:
        raise ValueError(f"CAD links must be ordered as {LINK_ORDER}")
    if tuple(joint_geometry) != JOINT_ORDER:
        raise ValueError(f"CAD joints must be ordered as {JOINT_ORDER}")

    simulation = joints_cfg["simulation"]
    limits = simulation["limits_rad"]
    control = simulation["control"]

    parts = [
        """<?xml version="1.0"?>
<!--
  Butter Finger CAD model.

  GENERATED FILE. Do not edit by hand.
  Regenerate with: python3 scripts/generate_urdf.py

  Mesh frames, joint transforms, masses, and inertia tensors were exported
  from SolidWorks with SW2URDF on 2026-07-16. The SW2URDF zero-width limits
  were intentionally replaced with development-only simulation limits from
  config/joints.yaml. The camera frame uses the recorded optical-center
  transform from config/geometry.yaml; its intrinsics remain provisional.
  Load with useFixedBase=True.
-->
<robot name="butter_finger">
  <material name="cad_grey">
    <color rgba="0.75 0.75 0.75 1"/>
  </material>
"""
    ]

    for link_name in LINK_ORDER:
        parts.append(mesh_link(link_name, links[link_name]))

    for joint_name in JOINT_ORDER:
        logical_name = joint_name.removesuffix("_joint")
        parts.append(
            revolute_joint(
                joint_name,
                joint_geometry[joint_name],
                limits[logical_name],
                control["max_force_nm"],
                control["max_velocity_rad_s"],
            )
        )

    # The camera is already part of the wrist CAD mesh. Keep only a numerically
    # negligible inertial frame, avoiding duplicate visible/collision geometry
    # and PyBullet's 1 kg fallback for links with no inertial element.
    parts.append(
        """
  <link name="camera_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.000001"/>
      <inertia ixx="0.000000000001" ixy="0" ixz="0" iyy="0.000000000001" iyz="0" izz="0.000000000001"/>
    </inertial>
  </link>
"""
    )
    parts.append(fixed_camera_joint(geo["camera_mount"]))
    parts.append("</robot>\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("".join(parts), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
