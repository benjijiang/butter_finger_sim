"""Load and validate the YAML configuration files.

Keeps the configuration concepts separate (see config/joints.yaml):
shared calibrated radians, physical PWM microseconds, verified hardware
limits, named poses, named actions, simulation-only idle behavior, and
simulation-only camera rendering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
URDF_PATH = REPO_ROOT / "models" / "butter_finger_simple.urdf"

# Logical joint names used throughout the API. The URDF joint names are
# "<name>_joint" (see URDF_JOINT_NAMES).
JOINT_NAMES: tuple[str, ...] = ("base", "shoulder", "elbow", "wrist")
URDF_JOINT_NAMES: dict[str, str] = {name: f"{name}_joint" for name in JOINT_NAMES}


@dataclass(frozen=True)
class JointLimits:
    """Inclusive joint command limits in radians."""

    lower_rad: float
    upper_rad: float

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower_rad, "lower_rad")
        upper = _finite_float(self.upper_rad, "upper_rad")
        if lower >= upper:
            raise ValueError("joint limits must satisfy lower_rad < upper_rad")

    def contains(self, position_rad: float) -> bool:
        return self.lower_rad <= position_rad <= self.upper_rad

    def clamp(self, position_rad: float) -> float:
        """Restrict a simulator value to this inclusive range."""
        return min(self.upper_rad, max(self.lower_rad, position_rad))


@dataclass(frozen=True)
class PulseLimits:
    """Tested hardware PWM pulse limits in microseconds."""

    min_us: int
    max_us: int

    def contains(self, pulse_us: float) -> bool:
        return self.min_us <= pulse_us <= self.max_us


@dataclass(frozen=True)
class JointCalibration:
    """Validated two-point linear mapping between radians and PWM."""

    lower_rad: float
    upper_rad: float
    lower_pwm_us: float
    upper_pwm_us: float

    def __post_init__(self) -> None:
        lower_rad = _finite_float(self.lower_rad, "calibration.lower_rad")
        upper_rad = _finite_float(self.upper_rad, "calibration.upper_rad")
        lower_pwm = _finite_float(self.lower_pwm_us, "calibration.lower_pwm_us")
        upper_pwm = _finite_float(self.upper_pwm_us, "calibration.upper_pwm_us")
        if lower_rad >= upper_rad:
            raise ValueError("calibration radians must satisfy lower_rad < upper_rad")
        if lower_pwm == upper_pwm:
            raise ValueError("calibration PWM endpoints must be different")

    @property
    def min_pwm_us(self) -> float:
        return min(self.lower_pwm_us, self.upper_pwm_us)

    @property
    def max_pwm_us(self) -> float:
        return max(self.lower_pwm_us, self.upper_pwm_us)

    def contains_rad(self, position_rad: float) -> bool:
        return self.lower_rad <= position_rad <= self.upper_rad

    def contains_pwm(self, pulse_us: float) -> bool:
        return self.min_pwm_us <= pulse_us <= self.max_pwm_us

    def rad_to_pwm(self, position_rad: float) -> float:
        """Interpolate a radian target without clamping or extrapolating."""
        position = _finite_float(position_rad, "position_rad")
        if not self.contains_rad(position):
            raise ValueError(
                f"{position} rad is outside the calibrated range "
                f"[{self.lower_rad}, {self.upper_rad}] rad"
            )
        fraction = (position - self.lower_rad) / (self.upper_rad - self.lower_rad)
        return self.lower_pwm_us + fraction * (
            self.upper_pwm_us - self.lower_pwm_us
        )

    def pwm_to_rad(self, pulse_us: float) -> float:
        """Invert an in-range PWM value without clamping or extrapolating."""
        pulse = _finite_float(pulse_us, "pulse_us")
        if not self.contains_pwm(pulse):
            raise ValueError(
                f"{pulse} us is outside the calibrated range "
                f"[{self.min_pwm_us}, {self.max_pwm_us}] us"
            )
        fraction = (pulse - self.lower_pwm_us) / (
            self.upper_pwm_us - self.lower_pwm_us
        )
        return self.lower_rad + fraction * (self.upper_rad - self.lower_rad)


@dataclass(frozen=True)
class PhysicalConfig:
    """Validated hardware data and radians/PWM calibration.

    The PWM layer uses ports and pulse limits. RaspberryPiArm additionally
    uses the per-joint linear calibrations.
    """

    joint_order: tuple[str, ...]
    pwm_ports: dict[str, int]
    home_pwm_us: dict[str, int]
    pulse_limits_us: dict[str, PulseLimits]
    calibrations: dict[str, JointCalibration]


@dataclass(frozen=True)
class ArmConfig:
    joint_order: tuple[str, ...]
    sim_limits: dict[str, JointLimits]
    poses: dict[str, dict[str, float]]
    max_force_nm: float
    max_velocity_rad_s: float
    control_rate_hz: float

    @property
    def time_step_s(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def home_pose(self) -> dict[str, float]:
        return dict(self.poses["sim_home"])


@dataclass(frozen=True)
class CameraConfig:
    """Validated simulation camera parameters and physical stream metadata."""

    native_width: int
    native_height: int
    fps: int
    hardware_pixel_format: str
    link_name: str
    forward_xyz: tuple[float, float, float]
    up_xyz: tuple[float, float, float]
    projection_model: str
    vertical_fov_deg: float
    near_plane_m: float
    far_plane_m: float
    intrinsics_calibrated: bool
    output_pixel_format: str
    rotate_clockwise_deg: int
    output_width: int
    output_height: int

    @property
    def native_aspect_ratio(self) -> float:
        return self.native_width / self.native_height

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """NumPy HWC shape of a captured RGB frame."""
        return (self.output_height, self.output_width, 3)


@dataclass(frozen=True)
class ActionStep:
    """One timed action step expressed in calibrated radians."""

    targets_rad: Mapping[str, float]
    duration_s: float


@dataclass(frozen=True)
class ArmAction:
    """A named sequence of one or more timed joint targets."""

    name: str
    description: str
    steps: tuple[ActionStep, ...]


@dataclass(frozen=True)
class ActionConfig:
    """Validated named actions in the shared radians command domain."""

    actions: Mapping[str, ArmAction]

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(self.actions)


@dataclass(frozen=True)
class IdleConfig:
    """Validated simulation-only fallback motion when nobody is tracked."""

    pose_name: str
    pose_rad: Mapping[str, float]
    scan_joint: str
    lower_rad: float
    upper_rad: float
    half_cycle_s: float

    @property
    def scan_speed_rad_s(self) -> float:
        return (self.upper_rad - self.lower_rad) / self.half_cycle_s


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    result = tuple(_finite_float(component, label) for component in value)
    if math.sqrt(sum(component**2 for component in result)) <= 0:
        raise ValueError(f"{label} must be non-zero")
    return result


def load_arm_config(config_dir: Path = CONFIG_DIR) -> ArmConfig:
    """Load shared radians limits, poses, and simulation control parameters."""
    joints_cfg = _load_yaml(config_dir / "joints.yaml")
    poses_cfg = _load_yaml(config_dir / "poses.yaml")

    joint_order = tuple(joints_cfg["joint_order"])
    if set(joint_order) != set(JOINT_NAMES):
        raise ValueError(
            f"joint_order in joints.yaml {joint_order} does not match the "
            f"expected joints {JOINT_NAMES}"
        )

    limits_raw = joints_cfg["simulation"]["limits_rad"]
    sim_limits = {
        name: JointLimits(
            _finite_float(limits_raw[name]["lower"], f"{name}.limits_rad.lower"),
            _finite_float(limits_raw[name]["upper"], f"{name}.limits_rad.upper"),
        )
        for name in joint_order
    }
    physical_config = load_physical_config(config_dir)
    for name in joint_order:
        limits = sim_limits[name]
        calibration = physical_config.calibrations[name]
        if (
            limits.lower_rad != calibration.lower_rad
            or limits.upper_rad != calibration.upper_rad
        ):
            raise ValueError(
                f"{name}: simulation limits must match the physical "
                "calibration radians exactly"
            )

    poses: dict[str, dict[str, float]] = {}
    for pose_name, pose in poses_cfg["poses"].items():
        poses[pose_name] = {
            joint: _finite_float(angle, f"Pose {pose_name!r} target for {joint!r}")
            for joint, angle in pose.items()
        }
        for joint, angle in poses[pose_name].items():
            if joint not in sim_limits:
                raise ValueError(f"Pose {pose_name!r} references unknown joint {joint!r}")
            if not sim_limits[joint].contains(angle):
                raise ValueError(
                    f"Pose {pose_name!r} puts {joint!r} at {angle} rad, outside "
                    f"the simulation limits"
                )

    control = joints_cfg["simulation"]["control"]
    max_force_nm = _finite_float(control["max_force_nm"], "control.max_force_nm")
    max_velocity_rad_s = _finite_float(
        control["max_velocity_rad_s"], "control.max_velocity_rad_s"
    )
    control_rate_hz = _finite_float(control["control_rate_hz"], "control.control_rate_hz")
    if max_force_nm <= 0 or max_velocity_rad_s <= 0 or control_rate_hz <= 0:
        raise ValueError("simulation control values must be greater than zero")

    return ArmConfig(
        joint_order=joint_order,
        sim_limits=sim_limits,
        poses=poses,
        max_force_nm=max_force_nm,
        max_velocity_rad_s=max_velocity_rad_s,
        control_rate_hz=control_rate_hz,
    )


def load_camera_config(config_dir: Path = CONFIG_DIR) -> CameraConfig:
    """Load the simulation-only RGB camera configuration."""
    camera_cfg = _load_yaml(config_dir / "camera.yaml")
    if not isinstance(camera_cfg, dict):
        raise ValueError("camera.yaml must contain a mapping")

    try:
        native = camera_cfg["native"]
        optical = camera_cfg["optical_frame"]
        projection = camera_cfg["projection"]
        output = camera_cfg["output"]
    except KeyError as exc:
        raise ValueError(f"camera.yaml is missing section {exc.args[0]!r}") from exc
    if not all(isinstance(section, dict) for section in (native, optical, projection, output)):
        raise ValueError("camera.yaml sections must be mappings")

    native_width = _positive_int(native.get("width"), "native.width")
    native_height = _positive_int(native.get("height"), "native.height")
    fps = _positive_int(native.get("fps"), "native.fps")
    hardware_pixel_format = _nonempty_string(
        native.get("pixel_format"), "native.pixel_format"
    )

    link_name = _nonempty_string(optical.get("link_name"), "optical_frame.link_name")
    forward_xyz = _vector3(optical.get("forward_xyz"), "optical_frame.forward_xyz")
    up_xyz = _vector3(optical.get("up_xyz"), "optical_frame.up_xyz")
    forward_norm = math.sqrt(sum(component**2 for component in forward_xyz))
    up_norm = math.sqrt(sum(component**2 for component in up_xyz))
    cosine = sum(a * b for a, b in zip(forward_xyz, up_xyz)) / (
        forward_norm * up_norm
    )
    if not math.isclose(cosine, 0.0, abs_tol=1e-6):
        raise ValueError("optical_frame forward_xyz and up_xyz must be orthogonal")

    projection_model = _nonempty_string(projection.get("model"), "projection.model")
    if projection_model != "pinhole":
        raise ValueError("projection.model must be 'pinhole'")
    vertical_fov_deg = _finite_float(
        projection.get("vertical_fov_deg"), "projection.vertical_fov_deg"
    )
    if not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("projection.vertical_fov_deg must be between 0 and 180")
    near_plane_m = _finite_float(
        projection.get("near_plane_m"), "projection.near_plane_m"
    )
    far_plane_m = _finite_float(
        projection.get("far_plane_m"), "projection.far_plane_m"
    )
    if near_plane_m <= 0 or far_plane_m <= near_plane_m:
        raise ValueError(
            "projection planes must satisfy 0 < near_plane_m < far_plane_m"
        )
    intrinsics_calibrated = projection.get("calibrated")
    if not isinstance(intrinsics_calibrated, bool):
        raise ValueError("projection.calibrated must be a boolean")

    output_pixel_format = _nonempty_string(
        output.get("pixel_format"), "output.pixel_format"
    )
    if output_pixel_format != "RGB":
        raise ValueError("output.pixel_format must be 'RGB'")
    rotate_clockwise_deg = _positive_int(
        output.get("rotate_clockwise_deg"), "output.rotate_clockwise_deg"
    )
    if rotate_clockwise_deg != 90:
        raise ValueError("output.rotate_clockwise_deg must be 90")
    output_width = _positive_int(output.get("width"), "output.width")
    output_height = _positive_int(output.get("height"), "output.height")
    if (output_width, output_height) != (native_height, native_width):
        raise ValueError(
            "90-degree output dimensions must swap native width and height"
        )

    return CameraConfig(
        native_width=native_width,
        native_height=native_height,
        fps=fps,
        hardware_pixel_format=hardware_pixel_format,
        link_name=link_name,
        forward_xyz=forward_xyz,
        up_xyz=up_xyz,
        projection_model=projection_model,
        vertical_fov_deg=vertical_fov_deg,
        near_plane_m=near_plane_m,
        far_plane_m=far_plane_m,
        intrinsics_calibrated=intrinsics_calibrated,
        output_pixel_format=output_pixel_format,
        rotate_clockwise_deg=rotate_clockwise_deg,
        output_width=output_width,
        output_height=output_height,
    )


def load_action_config(
    config_dir: Path = CONFIG_DIR,
    arm_config: ArmConfig | None = None,
) -> ActionConfig:
    """Load named actions and resolve pose references in calibrated radians."""
    arm_config = arm_config if arm_config is not None else load_arm_config(config_dir)
    actions_cfg = _load_yaml(config_dir / "actions.yaml")
    if not isinstance(actions_cfg, dict) or not isinstance(actions_cfg.get("actions"), dict):
        raise ValueError("actions.yaml must contain an 'actions' mapping")

    actions_raw = actions_cfg["actions"]
    if not actions_raw:
        raise ValueError("actions.yaml must define at least one action")

    actions: dict[str, ArmAction] = {}
    for action_name, action_raw in actions_raw.items():
        if not isinstance(action_name, str) or not action_name:
            raise ValueError("Action names must be non-empty strings")
        if not isinstance(action_raw, dict):
            raise ValueError(f"Action {action_name!r} must be a mapping")

        unknown_action_keys = set(action_raw) - {"description", "steps"}
        if unknown_action_keys:
            raise ValueError(
                f"Action {action_name!r} has unknown keys "
                f"{sorted(unknown_action_keys)}"
            )

        description = action_raw.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"Action {action_name!r} description must be a string")

        steps_raw = action_raw.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError(f"Action {action_name!r} must contain at least one step")

        steps: list[ActionStep] = []
        for index, step_raw in enumerate(steps_raw, start=1):
            label = f"Action {action_name!r} step {index}"
            if not isinstance(step_raw, dict):
                raise ValueError(f"{label} must be a mapping")

            unknown_step_keys = set(step_raw) - {"pose", "targets_rad", "duration_s"}
            if unknown_step_keys:
                raise ValueError(f"{label} has unknown keys {sorted(unknown_step_keys)}")

            has_pose = "pose" in step_raw
            has_targets = "targets_rad" in step_raw
            if has_pose == has_targets:
                raise ValueError(
                    f"{label} must define exactly one of 'pose' or 'targets_rad'"
                )

            duration_s = _finite_float(step_raw.get("duration_s"), f"{label} duration_s")
            if duration_s <= 0:
                raise ValueError(f"{label} duration_s must be greater than zero")

            if has_pose:
                pose_name = step_raw["pose"]
                if not isinstance(pose_name, str) or pose_name not in arm_config.poses:
                    raise ValueError(f"{label} references unknown pose {pose_name!r}")
                targets = dict(arm_config.poses[pose_name])
            else:
                targets_raw = step_raw["targets_rad"]
                if not isinstance(targets_raw, dict) or not targets_raw:
                    raise ValueError(f"{label} targets_rad must be a non-empty mapping")
                targets = {}
                for joint, raw_angle in targets_raw.items():
                    if joint not in arm_config.sim_limits:
                        raise ValueError(f"{label} references unknown joint {joint!r}")
                    angle = _finite_float(raw_angle, f"{label} target for {joint!r}")
                    if not arm_config.sim_limits[joint].contains(angle):
                        raise ValueError(
                            f"{label} puts {joint!r} at {angle} rad, outside "
                            "the simulation limits"
                        )
                    targets[joint] = angle

            steps.append(
                ActionStep(
                    targets_rad=MappingProxyType(targets),
                    duration_s=duration_s,
                )
            )

        actions[action_name] = ArmAction(
            name=action_name,
            description=description,
            steps=tuple(steps),
        )

    return ActionConfig(actions=MappingProxyType(actions))


def load_idle_config(
    config_dir: Path = CONFIG_DIR,
    arm_config: ArmConfig | None = None,
) -> IdleConfig:
    """Load the simulation-only fallback idle scan configuration."""
    arm_config = arm_config if arm_config is not None else load_arm_config(config_dir)
    idle_cfg = _load_yaml(config_dir / "idle.yaml")
    if not isinstance(idle_cfg, dict) or not isinstance(idle_cfg.get("idle"), dict):
        raise ValueError("idle.yaml must contain an 'idle' mapping")
    if set(idle_cfg) != {"idle"}:
        raise ValueError("idle.yaml must contain only the 'idle' section")

    idle_raw = idle_cfg["idle"]
    expected_keys = {
        "pose",
        "scan_joint",
        "lower_rad",
        "upper_rad",
        "half_cycle_s",
    }
    unknown_keys = set(idle_raw) - expected_keys
    missing_keys = expected_keys - set(idle_raw)
    if unknown_keys:
        raise ValueError(f"idle configuration has unknown keys {sorted(unknown_keys)}")
    if missing_keys:
        raise ValueError(f"idle configuration is missing keys {sorted(missing_keys)}")

    pose_name = _nonempty_string(idle_raw["pose"], "idle.pose")
    if pose_name not in arm_config.poses:
        raise ValueError(f"idle.pose references unknown pose {pose_name!r}")
    pose = dict(arm_config.poses[pose_name])
    if set(pose) != set(arm_config.joint_order):
        raise ValueError(
            f"idle pose {pose_name!r} must define every joint "
            f"{list(arm_config.joint_order)}"
        )

    scan_joint = _nonempty_string(idle_raw["scan_joint"], "idle.scan_joint")
    if scan_joint not in arm_config.sim_limits:
        raise ValueError(f"idle.scan_joint references unknown joint {scan_joint!r}")

    lower_rad = _finite_float(idle_raw["lower_rad"], "idle.lower_rad")
    upper_rad = _finite_float(idle_raw["upper_rad"], "idle.upper_rad")
    if lower_rad >= upper_rad:
        raise ValueError("idle scan range must satisfy lower_rad < upper_rad")
    limits = arm_config.sim_limits[scan_joint]
    if not limits.contains(lower_rad) or not limits.contains(upper_rad):
        raise ValueError(
            f"idle scan range [{lower_rad}, {upper_rad}] rad is outside the "
            f"simulation limits for {scan_joint!r}"
        )
    if not lower_rad <= pose[scan_joint] <= upper_rad:
        raise ValueError(
            f"idle pose {pose_name!r} puts scan joint {scan_joint!r} outside "
            "the configured idle scan range"
        )

    half_cycle_s = _finite_float(idle_raw["half_cycle_s"], "idle.half_cycle_s")
    if half_cycle_s <= 0:
        raise ValueError("idle.half_cycle_s must be greater than zero")

    return IdleConfig(
        pose_name=pose_name,
        pose_rad=MappingProxyType(pose),
        scan_joint=scan_joint,
        lower_rad=lower_rad,
        upper_rad=upper_rad,
        half_cycle_s=half_cycle_s,
    )


def load_physical_config(config_dir: Path = CONFIG_DIR) -> PhysicalConfig:
    """Load physical hardware data and measured two-point calibration.

    Home and calibration pulses must remain inside tested hardware limits.
    """
    joints_cfg = _load_yaml(config_dir / "joints.yaml")
    physical = joints_cfg["physical"]

    joint_order = tuple(joints_cfg["joint_order"])
    if set(joint_order) != set(JOINT_NAMES):
        raise ValueError(
            f"joint_order in joints.yaml {joint_order} does not match the "
            f"expected joints {JOINT_NAMES}"
        )

    pwm_ports = {
        name: _positive_int(physical["pwm_ports"][name], f"{name}.pwm_port")
        for name in joint_order
    }
    home_pwm_us = {
        name: _positive_int(physical["home_pwm_us"][name], f"{name}.home_pwm_us")
        for name in joint_order
    }
    pulse_limits_us = {
        name: PulseLimits(
            _positive_int(
                physical["tested_pwm_limits_us"][name]["min"],
                f"{name}.tested_pwm_limits_us.min",
            ),
            _positive_int(
                physical["tested_pwm_limits_us"][name]["max"],
                f"{name}.tested_pwm_limits_us.max",
            ),
        )
        for name in joint_order
    }
    calibration_raw = physical["calibration"]
    calibrations = {
        name: JointCalibration(
            lower_rad=_finite_float(
                calibration_raw[name]["lower"]["rad"],
                f"{name}.calibration.lower.rad",
            ),
            upper_rad=_finite_float(
                calibration_raw[name]["upper"]["rad"],
                f"{name}.calibration.upper.rad",
            ),
            lower_pwm_us=_finite_float(
                calibration_raw[name]["lower"]["pwm_us"],
                f"{name}.calibration.lower.pwm_us",
            ),
            upper_pwm_us=_finite_float(
                calibration_raw[name]["upper"]["pwm_us"],
                f"{name}.calibration.upper.pwm_us",
            ),
        )
        for name in joint_order
    }

    for name in joint_order:
        limits = pulse_limits_us[name]
        if limits.min_us >= limits.max_us:
            raise ValueError(f"{name}: invalid pulse limits {limits}")
        if not limits.contains(home_pwm_us[name]):
            raise ValueError(
                f"{name}: home pulse {home_pwm_us[name]} us is outside the "
                f"tested range [{limits.min_us}, {limits.max_us}] us"
            )
        calibration = calibrations[name]
        for endpoint, pulse_us in (
            ("lower", calibration.lower_pwm_us),
            ("upper", calibration.upper_pwm_us),
        ):
            if not limits.contains(pulse_us):
                raise ValueError(
                    f"{name}: calibration {endpoint} pulse {pulse_us} us is "
                    f"outside the tested range [{limits.min_us}, "
                    f"{limits.max_us}] us"
                )

    return PhysicalConfig(
        joint_order=joint_order,
        pwm_ports=pwm_ports,
        home_pwm_us=home_pwm_us,
        pulse_limits_us=pulse_limits_us,
        calibrations=calibrations,
    )
