# Butter Finger Simulator

A PyBullet digital clone of **Butter Finger**, a four-joint desktop
robot arm (base, shoulder, elbow, wrist) with a camera near the end of the
arm. The physical arm is driven by a Raspberry Pi 5 through a Hiwonder
RasAdapter5A V1.0 servo controller over UART.

> **This is not yet a validated physical digital twin.** Its geometry and
> inertial properties come from a SolidWorks SW2URDF export and each joint
> has a measured two-point radians/PWM mapping. The arm remains open-loop;
> servo dynamics, load behavior, CAD alignment, and camera intrinsics are
> not independently validated.

## What it currently simulates

- A four-joint revolute arm rendered from SolidWorks CAD meshes.
- Interactive GUI control with one slider per joint.
- Smooth scripted motion using smoothstep interpolation.
- Config-driven utility and emotional actions with deliberately expressive timing.
- A non-blocking no-person idle scan, ready for future tracking hand-off.
- Position control with gravity, a fixed 1/240 s time step, and deterministic
  stepping.
- RGB rendering from a wrist-mounted `camera_link`.

## CAD geometry

The link meshes, joint transforms, masses, centers of mass, and inertia tensors
were exported from the recovered SolidWorks assembly with SW2URDF on
2026-07-16. The exporter used different names, which are mapped onto the
project's stable four-joint API by `config/geometry.yaml`. The same meshes are
temporarily used for visual and collision geometry; simplified collision meshes
can be added later without changing joint names or application code.

## Placeholder vs. physically verified

| Recorded / CAD-derived data | Unknown / placeholder |
|---|---|
| Servo models: base/elbow/wrist = SG90, shoulder = LD-1501MG | Servo datasheet PWM/angle specs (deliberately not recorded) |
| SolidWorks link meshes, joint transforms, masses, and inertia tensors | Independent physical validation of CAD mass properties and transforms |
| Recorded home PWM: base 1500, shoulder 2200, elbow 2490, wrist 1400 µs | Closed-loop joint-angle feedback |
| Tested pulse ranges: base/elbow/wrist 505–2495 µs, shoulder 1200–2220 µs | Servo force, speed, backlash, compliance |
| Two-point radians/PWM calibration for all four joints | Calibration linearity between measured endpoints and behavior under load |
| Recorded camera optical center `[0.011, 0.044, 0.013]` m from `wrist_link`; optical axis +Y, native image top -X | Exact camera FOV, intrinsic matrix, and lens distortion |
| Pi → UART → RasAdapter5A → servo PWM signal chain | Whether SDK readback reflects shaft angle rather than command state |
| Drawing dimensions: base 32.53 mm; shoulder axis height 73.90 mm; shoulder–elbow 145.2 mm; elbow–wrist 100 mm; wrist–tip 49 mm | Independent CAD/physical verification of those drawing dimensions |

## Repository layout

```text
butter-finger-sim/
├── config/
│   ├── geometry.yaml        # CAD mesh, joint-transform, and inertial source of truth
│   ├── joints.yaml          # PWM limits, two-point calibration, shared radian limits
│   ├── poses.yaml           # named poses in calibrated radians
│   ├── actions.yaml         # named action sequences in calibrated radians
│   ├── idle.yaml            # simulation-only no-person scan behavior
│   └── camera.yaml          # camera stream metadata and simulation projection
├── models/
│   ├── meshes/                      # SolidWorks-exported link meshes
│   └── butter_finger_simple.urdf   # GENERATED from config/ — do not hand-edit
├── scripts/
│   └── generate_urdf.py     # regenerates the URDF from the config files
├── src/butter_finger/
│   ├── arm.py               # ArmBackend abstract interface (radians only)
│   ├── actions.py           # backend-neutral named-action scheduler
│   ├── idle.py              # non-blocking fallback idle scan
│   ├── camera.py            # optical-frame math and RGB rotation
│   ├── config.py            # YAML config loader (sim + physical sections)
│   └── backends/
│       ├── pybullet_arm.py      # simulation backend (runs on the sim machine)
│       ├── pwm_robot_arm.py     # REAL hardware, PWM microseconds (Raspberry Pi)
│       └── raspberry_pi_arm.py  # REAL calibrated radians adapter
├── examples/
│   ├── view_robot.py        # display the arm in the PyBullet GUI
│   ├── joint_sliders.py     # one GUI slider per joint
│   ├── go_home.py           # move to the simulated reference pose
│   ├── scripted_motion.py   # smoothstep motion sequence
│   ├── run_action.py        # list/run actions with sim or real backend
│   ├── idle_motion.py       # continuous slow no-person scan
│   ├── emotion_showcase.py  # play emotions with sim or real backend
│   ├── camera_snapshot.py   # render one simulated RGB frame
│   ├── pi_test_pose.py      # REAL HARDWARE: joint-by-joint home-pose test
│   └── pi_sweep_base.py     # REAL HARDWARE: base sweep around home
├── tests/                   # dependency-light; none require PyBullet or hardware
├── SIM_SETUP.md             # simulation machine setup (Linux primary, Mac fallback)
└── CLAUDE.md                # persistent context for future Claude Code sessions
```

## Shared backend architecture

Application code commands the arm **only in radians**
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
joint target      calibrated PWM                     |
                                          UART -> RasAdapter5A -> servos
```

```python
from butter_finger import PyBulletArm

with PyBulletArm(gui=True) as arm:
    arm.go_home()
    arm.move_joint("base", 0.5)          # non-blocking target, radians
    arm.move_joints(
        {"shoulder": -0.4, "elbow": -0.6},
        duration_s=2.0,                  # blocking smoothstep move
    )
    print(arm.get_joint_positions())
```

On the Pi, the same API wraps the PWM layer:

```python
from butter_finger import RaspberryPiArm

with RaspberryPiArm() as arm:
    arm.go_home()  # exact recorded home PWM; waits for completion
    arm.move_joints(
        {"base": 0.3, "shoulder": -0.5},
        duration_s=2.0,
    )
```

`RaspberryPiArm` interpolates only inside each measured two-point range.
Unknown joints, non-finite values, and out-of-range radians are rejected
before any PWM command. Its `get_joint_positions()` value is only the last
complete command estimate; it is not physical angle feedback.

## Named actions

`config/actions.yaml` defines calibrated-radian actions as timed steps. A step
either references a full pose from `config/poses.yaml` or gives absolute
targets for selected joints. List and run them with:

```bash
python examples/run_action.py --list
python examples/run_action.py base_scan
python examples/run_action.py happy
python examples/run_action.py happy --backend real --confirm-hardware
python examples/emotion_showcase.py happy curious sad surprised
python examples/emotion_showcase.py --all
python examples/emotion_showcase.py happy curious --backend real --confirm-hardware
```

The original utility actions remain: `home`, `demo_reach`, `base_scan`,
`reach_and_return`, `wrist_up`, and `wrist_down`. Twenty conversational
actions treat the wrist camera as the robot's face:

- Social: `greet`, `nod_yes`, `shake_no`, `attentive`
- Positive: `happy`, `excited`, `proud`, `playful`, `affectionate`, `shy`
- Reflective: `curious`, `thinking`, `confused`
- Low-energy: `sad`, `disappointed`, `bored`, `sleepy`
- Reactive: `surprised`, `scared`, `angry`

The emotional actions begin and end at `idle_ready`, not `sim_home`.
`duration_s` is deliberately designed per step: fast motion conveys surprise,
fear, excitement, or force; slow motion conveys affection, thought, sadness,
or fatigue. Repeated identical targets create expressive holds while still
advancing simulation. These timings are simulator choreography, not measured
servo performance; test speed and load behavior cautiously on hardware.

`ActionRunner` schedules each finite action through `ArmBackend`.
`emotion_showcase.py` plays either a supplied list or all twenty actions. In
simulation it runs a slow idle scan between actions. On real hardware it only
waits between gestures and does not stream `IdleController` targets.

## Simulation idle behavior

`IdleController` reads `config/idle.yaml` and sends non-blocking targets for a
slow triangle-wave base scan around the full `idle_ready` posture:

```bash
python examples/idle_motion.py
```

This is only the no-person fallback. A future attention layer will stop
calling `IdleController.update()` while a person is detected, command tracking
targets from camera perception, then call `resume()` when the person is lost.
Perception remains outside `ArmBackend`.

Named actions are inside the measured calibration domain and may be sent
through `RaspberryPiArm`. The continuous `IdleController` scan remains
simulation-only because its high-frequency real-hardware behavior is untested.

## Simulated RGB camera

`PyBulletArm.capture_rgb()` renders the current `camera_link` view at the
sensor's native `640×480`, converts PyBullet RGBA to RGB, then rotates the
frame 90° clockwise. The returned NumPy array has HWC shape `(640, 480, 3)`,
corresponding to a portrait image width×height of `480×640`.

```python
with PyBulletArm(gui=False) as arm:
    arm.reset_joints(arm.config.home_pose)
    arm.step()
    rgb = arm.capture_rgb()
```

Create a PNG snapshot:

```bash
python examples/camera_snapshot.py
python examples/camera_snapshot.py --output camera.png --gui
```

The configured `120°` native vertical FOV is provisional. PyBullet uses a
pinhole projection and does not reproduce the physical camera's strong
wide-angle/fisheye distortion. Replace the projection parameters only after
checkerboard calibration provides measured intrinsics and distortion.

## Real hardware (Raspberry Pi only)

Use `RaspberryPiArm` for normal application code in radians. It wraps
`PWMRobotArm`, the verified port of the original
`board_demo/butter_finger.py` control code. The `physical` section of
`config/joints.yaml` is the source of truth for ports, tested pulse limits,
home PWM, and these measured endpoint mappings:

- base: `-1.5708 rad → 510 µs`, `1.5708 rad → 2490 µs`
- shoulder: `-1.530 rad → 1200 µs`, `0 rad → 2200 µs`
- elbow: `-3.075 rad → 510 µs`, `0.065 rad → 2490 µs`
- wrist: `-2.9824 rad → 510 µs`, `0.1576 rad → 2490 µs`

The duplicated URDF/simulation limits must match these radian endpoints;
configuration loading fails if they diverge. Conversion is interpolation
only—there is no clamping or extrapolation.

```python
from butter_finger import RaspberryPiArm

with RaspberryPiArm() as arm:
    arm.go_home()
    arm.move_joint("base", 0.4, duration_s=1.0)
```

`PWMRobotArm` remains available for low-level diagnostics that intentionally
work in PWM microseconds.

The Hiwonder SDK (`ros_robot_controller_sdk.py`, serial `/dev/ttyAMA0` at
1,000,000 baud) is **not vendored** in this repository. `PWMRobotArm`
imports it lazily at instantiation, looking in `sys.path`, then
`$BUTTER_FINGER_SDK_PATH`, then the directory containing this repository
checkout — on the Pi the repo lives inside `board_demo/`, right next to the
SDK file, so it is found automatically. On the simulation machine
(Linux/Mac) the SDK is absent and everything else still works.

Timed `ArmBackend` calls carry an optional `duration_s`. `PyBulletArm`
implements it by generating smoothstep targets at the configured control
rate. `PWMRobotArm` already passes its required `duration` to the controller,
which performs its own interpolation. `RaspberryPiArm` forwards the duration
and waits before returning so sequential action keyframes cannot overwrite
one another.

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
python examples/run_action.py happy
python examples/idle_motion.py
python examples/emotion_showcase.py happy curious sad surprised
python examples/camera_snapshot.py
```

## Limitations of simulating cheap open-loop hobby servos

The real arm uses inexpensive hobby servos driven open-loop by PWM pulse
width. The simulation cannot faithfully reproduce them because:

- **No verified angle feedback exists on the real arm.** The SDK does expose
  `pwm_servo_read_position()` / `pwm_servo_read_offset()`, but the servos
  themselves are open-loop, so the board most likely reports the last
  commanded value rather than the true shaft angle (unverified). Until this
  is tested, `RaspberryPiArm.get_joint_positions()` reports only the last
  complete command estimate.
- **Calibration is only a two-point linear model.** Its endpoints are
  measured, but intermediate linearity, repeatability, and loaded behavior
  have not been independently characterized.
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
   into this repository as `PWMRobotArm`.
2. ~~Replace primitive geometry with recovered CAD meshes~~ — imported from
   the SolidWorks SW2URDF export on 2026-07-16.
3. Verify the CAD joint frames, recorded camera transform, and mass properties
   against the physical arm; create simplified collision meshes.
4. ~~Measure two-point PWM-to-angle calibration per joint~~ — recorded in
   `config/joints.yaml`; intermediate linearity still needs characterization.
5. ~~Implement `RaspberryPiArm` (radians) on top of the PWM layer~~ —
   calibrated interpolation, atomic validation, exact home, and CLI switching
   are implemented.
6. ~~Add camera rendering from `camera_link`~~ — RGB capture implemented with
   provisional pinhole intrinsics; physical camera calibration remains.
