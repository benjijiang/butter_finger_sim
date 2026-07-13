"""PyBullet implementation of the ArmBackend interface.

Runtime environment: the Mac (PyBullet is intentionally not installed on
the Windows authoring machine). PyBullet is imported lazily so that the
rest of the package works without it.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from butter_finger.arm import (
    ArmBackend,
    BackendUnavailableError,
    JointLimitError,
    UnknownJointError,
)
from butter_finger.config import (
    URDF_JOINT_NAMES,
    URDF_PATH,
    ArmConfig,
    load_arm_config,
)

_MISSING_PYBULLET_MSG = (
    "PyBullet is not installed in this Python environment. This simulator is "
    "meant to run on the Mac inside its own virtual environment:\n"
    "    python3 -m venv .venv\n"
    "    source .venv/bin/activate\n"
    "    python -m pip install -r requirements-sim.txt\n"
    "See MAC_SETUP.md for the full instructions."
)

_MISSING_URDF_MSG = (
    "URDF not found at {path}. Generate it first with:\n"
    "    python scripts/generate_urdf.py"
)


def _import_pybullet():
    try:
        import pybullet
    except ImportError as exc:
        raise BackendUnavailableError(_MISSING_PYBULLET_MSG) from exc
    return pybullet


class PyBulletArm(ArmBackend):
    """Simulated Butter Finger arm in PyBullet.

    Parameters
    ----------
    gui:
        True opens the interactive PyBullet GUI (``p.GUI``); False runs
        headless (``p.DIRECT``), which is what the tests use.
    urdf_path:
        Path to the arm URDF. Defaults to models/butter_finger_simple.urdf,
        resolved relative to the repository, independent of the terminal's
        current directory.
    config:
        Pre-loaded ArmConfig; loaded from config/ when omitted.
    """

    def __init__(
        self,
        gui: bool = True,
        urdf_path: Path | str | None = None,
        config: ArmConfig | None = None,
    ) -> None:
        self._config = config if config is not None else load_arm_config()
        self._urdf_path = Path(urdf_path) if urdf_path is not None else URDF_PATH
        self._pb = _import_pybullet()
        self._client: int | None = None

        if not self._urdf_path.is_file():
            raise FileNotFoundError(_MISSING_URDF_MSG.format(path=self._urdf_path))

        # Each PyBulletArm owns its own physics client and passes its id to
        # every call, so multiple instances cannot conflict. Note PyBullet
        # allows only one GUI connection per process.
        mode = self._pb.GUI if gui else self._pb.DIRECT
        client = self._pb.connect(mode)
        if client < 0:
            raise BackendUnavailableError(
                "Could not create a PyBullet physics client. If a GUI "
                "connection already exists in this process, disconnect it "
                "first or use gui=False."
            )
        self._client = client

        import pybullet_data

        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        self._pb.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        self._pb.setTimeStep(self.time_step_s, physicsClientId=client)
        # Deterministic stepping: never let the engine free-run.
        self._pb.setRealTimeSimulation(0, physicsClientId=client)

        self._plane_id = self._pb.loadURDF("plane.urdf", physicsClientId=client)
        self._robot_id = self._pb.loadURDF(
            str(self._urdf_path), useFixedBase=True, physicsClientId=client
        )
        self._joint_indices = self._discover_joints()

    # ------------------------------------------------------------------
    # Introspection helpers (used by the examples)
    # ------------------------------------------------------------------

    @property
    def pb(self):
        """The pybullet module, for example scripts (debug sliders etc.)."""
        return self._pb

    @property
    def client_id(self) -> int:
        if self._client is None:
            raise BackendUnavailableError("PyBullet client is disconnected.")
        return self._client

    @property
    def robot_id(self) -> int:
        return self._robot_id

    @property
    def config(self) -> ArmConfig:
        return self._config

    @property
    def time_step_s(self) -> float:
        return self._config.time_step_s

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._config.joint_order

    def is_connected(self) -> bool:
        return self._client is not None and bool(self._pb.isConnected(self._client))

    # ------------------------------------------------------------------
    # Joint discovery and validation
    # ------------------------------------------------------------------

    def _discover_joints(self) -> dict[str, int]:
        """Map logical joint names to PyBullet joint indices by URDF name."""
        found: dict[str, int] = {}
        for index in range(self._pb.getNumJoints(self._robot_id, physicsClientId=self._client)):
            info = self._pb.getJointInfo(self._robot_id, index, physicsClientId=self._client)
            found[info[1].decode("utf-8")] = index

        indices: dict[str, int] = {}
        missing: list[str] = []
        for logical, urdf_name in URDF_JOINT_NAMES.items():
            if urdf_name in found:
                indices[logical] = found[urdf_name]
            else:
                missing.append(urdf_name)
        if missing:
            raise RuntimeError(
                f"URDF {self._urdf_path} is missing expected joints {missing}; "
                f"found {sorted(found)}. Regenerate it with "
                "'python scripts/generate_urdf.py'."
            )
        return indices

    def _validate(self, joint: str, position_rad: float) -> None:
        if joint not in self._joint_indices:
            raise UnknownJointError(
                f"Unknown joint {joint!r}; expected one of {list(self._joint_indices)}"
            )
        limits = self._config.sim_limits[joint]
        if not limits.contains(position_rad):
            raise JointLimitError(
                f"Target {position_rad} rad for joint {joint!r} is outside the "
                f"configured simulation limits "
                f"[{limits.lower_rad}, {limits.upper_rad}] rad"
            )

    # ------------------------------------------------------------------
    # ArmBackend interface
    # ------------------------------------------------------------------

    def move_joint(self, joint: str, position_rad: float) -> None:
        """Set the position-control target for one joint (radians)."""
        self._validate(joint, position_rad)
        self._pb.setJointMotorControl2(
            bodyUniqueId=self._robot_id,
            jointIndex=self._joint_indices[joint],
            controlMode=self._pb.POSITION_CONTROL,
            targetPosition=position_rad,
            force=self._config.max_force_nm,
            maxVelocity=self._config.max_velocity_rad_s,
            physicsClientId=self._client,
        )

    def move_joints(self, targets_rad: Mapping[str, float]) -> None:
        """Set position-control targets for multiple joints (radians).

        All targets are validated before any joint is commanded, so an
        invalid request leaves the arm untouched.
        """
        for joint, position in targets_rad.items():
            self._validate(joint, position)
        for joint, position in targets_rad.items():
            self.move_joint(joint, position)

    def get_joint_positions(self) -> dict[str, float]:
        return {
            joint: self._pb.getJointState(
                self._robot_id, index, physicsClientId=self._client
            )[0]
            for joint, index in self._joint_indices.items()
        }

    def go_home(self, settle_time_s: float = 2.0) -> None:
        """Move to the configured simulated reference pose ('sim_home')."""
        self.move_joints(self._config.home_pose)
        self.run_for(settle_time_s)

    def disconnect(self) -> None:
        """Disconnect from the physics server. Safe to call repeatedly."""
        if self._client is not None:
            if self._pb.isConnected(self._client):
                self._pb.disconnect(self._client)
            self._client = None

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self, steps: int = 1, realtime: bool = False) -> None:
        """Advance the simulation deterministically by fixed time steps.

        With realtime=True, sleeps each step so GUI motion runs at roughly
        wall-clock speed; physics stays identical either way.
        """
        for _ in range(steps):
            self._pb.stepSimulation(physicsClientId=self._client)
            if realtime:
                time.sleep(self.time_step_s)

    def run_for(self, seconds: float, realtime: bool | None = None) -> None:
        """Step the simulation for the given simulated duration."""
        if realtime is None:
            realtime = self._is_gui()
        self.step(max(1, round(seconds / self.time_step_s)), realtime=realtime)

    def reset_joints(self, targets_rad: Mapping[str, float]) -> None:
        """Instantly set joint states (no dynamics). Useful for viewing."""
        for joint, position in targets_rad.items():
            self._validate(joint, position)
        for joint, position in targets_rad.items():
            self._pb.resetJointState(
                self._robot_id,
                self._joint_indices[joint],
                targetValue=position,
                physicsClientId=self._client,
            )

    def _is_gui(self) -> bool:
        info = self._pb.getConnectionInfo(self._client)
        return info.get("connectionMethod") == self._pb.GUI
