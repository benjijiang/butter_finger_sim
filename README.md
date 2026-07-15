# Butter Finger Simulator

A simplified PyBullet digital clone of **Butter Finger**, a four-joint desktop
robot arm (base, shoulder, elbow, wrist) with a camera near the end of the
arm. The physical arm is driven by a Raspberry Pi 5 through a Hiwonder
RasAdapter5A V1.0 servo controller over UART.

> **This is a functional approximate model, not a validated physical digital
> twin.** Nearly every physical parameter is a placeholder (see below). Its
> purpose is to let high-level motion and perception code be developed
> without the physical arm nearby.

## What it currently simulates

- A four-joint revolute arm built from primitive boxes and cylinders.
- Interactive GUI control with one slider per joint.
- Smooth scripted motion using smoothstep interpolation.
- Config-driven named actions with validated targets and durations.
- Position control with gravity, a fixed 1/240 s time step, and deterministic
  stepping.
- A fixed `camera_link` at the wrist tip (camera rendering not implemented
  yet).

## Why primitive geometry?

The CAD model of the real arm is currently unavailable. Simple boxes and
cylinders use joint-to-joint dimensions from a user-provided reference
drawing, while their cross-sections and exact shapes remain placeholders.
When the CAD is recovered, mesh files will replace the primitive `<visual>`
(and eventually `<collision>`) geometry in the URDF **without changing joint
names or the control API**, so no application code will need to change.

## Placeholder vs. physically verified

| Recorded / user-provided reference data | Unknown / placeholder |
|---|---|
| Servo models: base/elbow/wrist = SG90, shoulder = LD-1501MG | Servo datasheet PWM/angle specs (deliberately not recorded) |
| PWM port mapping: base=1, shoulder=3, elbow=4, wrist=5 (2, 6 unused) | Exact CAD shapes, link cross-sections, masses, inertias |
| Recorded home PWM: base 1500, shoulder 2200, elbow 2490, wrist 1400 µs | Joint angular ranges (±1.5708 rad is a development limit) |
| Tested pulse ranges: base/elbow/wrist 505–2495 µs, shoulder 1200–2220 µs | Wrist joint axis (Y is an unverified assumption) |
| Shoulder home (2200 µs) revalidated on the real machine (2026-07-13) | Camera transform, servo force/velocity |
| Pi → UART → RasAdapter5A → servo PWM signal chain | PWM-to-angle calibration (none exists yet) |
| Drawing dimensions: base 32.53 mm; shoulder axis height 73.90 mm; shoulder–elbow 145.2 mm; elbow–wrist 100 mm; wrist–tip 49 mm | Independent CAD/physical verification of those drawing dimensions |

## Repository layout

```text
butter-finger-sim/
├── config/
│   ├── geometry.yaml        # temporary link dimensions and masses (source of truth)
│   ├── joints.yaml          # PWM ports, recorded PWM data, simulation limits
│   ├── poses.yaml           # named simulation poses in radians
│   └── actions.yaml         # named simulation action sequences in radians
├── models/
│   └── butter_finger_simple.urdf   # GENERATED from config/ — do not hand-edit
├── scripts/
│   └── generate_urdf.py     # regenerates the URDF from the config files
├── src/butter_finger/
│   ├── arm.py               # ArmBackend abstract interface (radians only)
│   ├── actions.py           # backend-neutral named-action scheduler
│   ├── config.py            # YAML config loader (sim + physical sections)
│   └── backends/
│       ├── pybullet_arm.py      # simulation backend (runs on the sim machine)
│       ├── pwm_robot_arm.py     # REAL hardware, PWM microseconds (Raspberry Pi)
│       └── raspberry_pi_arm.py  # radians hardware stub — awaits calibration
├── examples/
│   ├── view_robot.py        # display the arm in the PyBullet GUI
│   ├── joint_sliders.py     # one GUI slider per joint
│   ├── go_home.py           # move to the simulated reference pose
│   ├── scripted_motion.py   # smoothstep motion sequence
│   ├── run_action.py        # list/run configured simulation actions
│   ├── pi_test_pose.py      # REAL HARDWARE: joint-by-joint home-pose test
│   └── pi_sweep_base.py     # REAL HARDWARE: base sweep around home
├── tests/                   # dependency-light; none require PyBullet or hardware
├── SIM_SETUP.md             # simulation machine setup (Linux primary, Mac fallback)
└── CLAUDE.md                # persistent context for future Claude Code sessions
```

## Shared backend architecture

Simulation-facing application code commands the arm **only in radians**
through the `ArmBackend` interface. Calls may omit `duration_s` for
non-blocking target updates (such as sliders), or provide it for a blocking
timed move. Backends translate radians and duration into whatever their
target needs; high-level code never sends PWM directly.

```text
Application command in radians          Pi-only scripts (PWM microseconds)
            |                                        |
       ArmBackend API                           PWMRobotArm
        /          \                                 |
PyBulletArm     RaspberryPiArm            ros_robot_controller_sdk (board_demo)
joint target    future calibrated PWM                |
                                          UART -> RasAdapter5A -> servos
```

```python
from butter_finger import PyBulletArm

with PyBulletArm(gui=True) as arm:
    arm.go_home()
    arm.move_joint("base", 0.5)          # non-blocking target, radians
    arm.move_joints(
        {"shoulder": 0.4, "elbow": -0.6},
        duration_s=2.0,                  # blocking smoothstep move
    )
    print(arm.get_joint_positions())
```

`RaspberryPiArm` (radians) is a stub that raises `NotImplementedError`. It
will only be implemented after measured PWM-to-angle calibration and verified
safe limits exist for every joint — commands without verified limits will be
rejected, never clamped to guesses.

## Named simulation actions

`config/actions.yaml` defines development-only actions as timed steps. A step
either references a full pose from `config/poses.yaml` or gives absolute
targets for selected joints. List and run them with:

```bash
python examples/run_action.py --list
python examples/run_action.py base_scan
python examples/run_action.py reach_and_return
```

The configured actions are `home`, `demo_reach`, `base_scan`,
`reach_and_return`, `wrist_up`, and `wrist_down`. `ActionRunner` schedules
their steps through `ArmBackend`; `PyBulletArm` currently supplies the only
working radians backend. These action values are temporary simulation data,
not hardware-approved commands, and cannot be sent to `PWMRobotArm`.

## Real hardware: the PWM layer (Raspberry Pi only)

Until the PWM-to-angle calibration exists, the real arm is driven directly
in PWM pulse widths (microseconds) through `PWMRobotArm`, the verified port
of the original `board_demo/butter_finger.py` control code. Ports, tested
pulse limits, and the recorded home pose come from the `physical` section of
`config/joints.yaml`; out-of-range pulses raise `JointLimitError` and are
never clamped.

```python
from butter_finger import PWMRobotArm

arm = PWMRobotArm()                      # opens the serial Board via the SDK
arm.move_joint("base", 1500, duration=1.0)   # PWM microseconds
arm.move_joints({"shoulder": 2200, "wrist": 1400}, duration=2.0)
arm.home()                               # recorded physical home pose
```

The Hiwonder SDK (`ros_robot_controller_sdk.py`, serial `/dev/ttyAMA0` at
1,000,000 baud) is **not vendored** in this repository. `PWMRobotArm`
imports it lazily at instantiation, looking in `sys.path`, then
`$BUTTER_FINGER_SDK_PATH`, then the directory containing this repository
checkout — on the Pi the repo lives inside `board_demo/`, right next to the
SDK file, so it is found automatically. On the simulation machine
(Linux/Mac) the SDK is absent and everything else still works.

Timed `ArmBackend` calls now carry an optional `duration_s`. `PyBulletArm`
implements it by generating smoothstep targets at the configured control
rate. `PWMRobotArm` already passes its required `duration` to the controller,
which performs its own interpolation. After measured calibration exists,
`RaspberryPiArm` will convert each radians target to PWM and forward the same
duration; it remains deliberately unimplemented today.

## Getting started

See [SIM_SETUP.md](SIM_SETUP.md). Short version, on the simulation machine
(the Linux GPU machine; the Mac also works as a fallback):

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-sim.txt
python scripts/generate_urdf.py
python -m pytest
python examples/joint_sliders.py
python examples/run_action.py --list
python examples/run_action.py base_scan
```

## Limitations of simulating cheap open-loop hobby servos

The real arm uses inexpensive hobby servos driven open-loop by PWM pulse
width. The simulation cannot faithfully reproduce them because:

- **No verified feedback exists on the real arm.** The SDK does expose
  `pwm_servo_read_position()` / `pwm_servo_read_offset()`, but the servos
  themselves are open-loop, so the board most likely reports the last
  commanded value rather than the true shaft angle (unverified). Until this
  is tested, treat the simulator's `get_joint_positions()` as having no
  trustworthy physical counterpart.
- **PWM-to-angle mapping is unknown and nonlinear-ish.** Until it is measured
  per joint, simulated radians and real pulse widths are unrelated scales.
- **Torque, speed, deadband, and backlash are guesses.** The simulator uses
  a placeholder force/velocity cap; real servos stall, buzz, overshoot, and
  sag under load.
- **Compliance and slop are absent.** The simulated arm is rigid; the real
  arm's brackets and gear trains flex.

Treat simulated trajectories as *kinematic intent*, not as predictions of
real dynamic behavior.

## Roadmap

1. ~~Recover the original Pi source code~~ — recovered: it lives on the Pi
   as `board_demo/butter_finger.py` (class `RobotArm`) and is now ported
   into this repository as `PWMRobotArm`. The CAD model is still missing.
2. Verify the reference-drawing dimensions against CAD or the physical arm
   and update `config/geometry.yaml` if needed.
3. Replace primitive visuals with CAD meshes (same joint names, same API).
4. Measure PWM-to-angle calibration per joint on the physical arm.
5. Implement `RaspberryPiArm` (radians) on top of the PWM layer with
   verified calibration only.
6. Add camera rendering from `camera_link`.
