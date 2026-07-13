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
- Position control with gravity, a fixed 1/240 s time step, and deterministic
  stepping.
- A fixed `camera_link` at the wrist tip (camera rendering not implemented
  yet).

## Why primitive geometry?

The CAD model of the real arm is currently unavailable. Simple boxes and
cylinders with plausible dimensions keep the kinematic structure (joint
names, axes, chain order) correct while the exact shapes are unknown. When
the CAD is recovered, mesh files will replace the primitive `<visual>` (and
eventually `<collision>`) geometry in the URDF **without changing joint
names or the control API**, so no application code will need to change.

## Placeholder vs. physically verified

| Physically verified (recorded from the real arm) | Placeholder / assumed |
|---|---|
| PWM port mapping: base=1, shoulder=3, elbow=4, wrist=5 (2, 6 unused) | All link dimensions, masses, inertias |
| Recorded home PWM: base 1500, shoulder 2200, elbow 2490, wrist 1400 µs | Joint angular ranges (±1.5708 rad is a development limit) |
| Tested shoulder pulse range: 1200–2220 µs | Wrist joint axis (Y is an unverified assumption) |
| Pi → UART → RasAdapter5A → servo PWM signal chain | Camera transform, servo force/velocity, PWM-to-angle calibration (none exists) |

⚠️ The recorded shoulder home value (2200 µs) is only 20 µs below the tested
maximum (2220 µs). It is preserved as historical data in
[config/joints.yaml](config/joints.yaml) and **must be revalidated before the
physical arm is operated again**.

## Repository layout

```text
butter-finger-sim/
├── config/
│   ├── geometry.yaml        # temporary link dimensions and masses (source of truth)
│   ├── joints.yaml          # PWM ports, recorded PWM data, simulation limits
│   └── poses.yaml           # named simulation poses in radians
├── models/
│   └── butter_finger_simple.urdf   # GENERATED from config/ — do not hand-edit
├── scripts/
│   └── generate_urdf.py     # regenerates the URDF from the config files
├── src/butter_finger/
│   ├── arm.py               # ArmBackend abstract interface (radians only)
│   ├── config.py            # YAML config loader
│   └── backends/
│       ├── pybullet_arm.py      # simulation backend (runs on the Mac)
│       └── raspberry_pi_arm.py  # hardware stub — intentionally NotImplemented
├── examples/
│   ├── view_robot.py        # display the arm in the PyBullet GUI
│   ├── joint_sliders.py     # one GUI slider per joint
│   ├── go_home.py           # move to the simulated reference pose
│   └── scripted_motion.py   # smoothstep motion sequence
├── tests/                   # dependency-light; none require PyBullet
├── MAC_SETUP.md             # exact Mac Terminal setup and run commands
└── CLAUDE.md                # persistent context for future Claude Code sessions
```

## Shared backend architecture

Application code commands the arm **only in radians** through the
`ArmBackend` interface. Backends translate radians into whatever their
target needs; high-level code never sends PWM directly.

```text
Application command in radians
            |
       ArmBackend API
        /          \
PyBulletArm     RaspberryPiArm
joint target    future calibrated PWM
```

```python
from butter_finger import PyBulletArm

with PyBulletArm(gui=True) as arm:
    arm.go_home()
    arm.move_joint("base", 0.5)          # radians
    arm.move_joints({"shoulder": 0.4, "elbow": -0.6})
    print(arm.get_joint_positions())
```

`RaspberryPiArm` is a stub that raises `NotImplementedError`. It will only be
implemented after measured PWM-to-angle calibration and verified safe limits
exist for every joint — commands without verified limits will be rejected,
never clamped to guesses.

## Getting started

See [MAC_SETUP.md](MAC_SETUP.md). Short version, on the Mac:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-sim.txt
python scripts/generate_urdf.py
python -m pytest
python examples/joint_sliders.py
```

## Limitations of simulating cheap open-loop hobby servos

The real arm uses inexpensive hobby servos driven open-loop by PWM pulse
width. The simulation cannot faithfully reproduce them because:

- **No feedback exists on the real arm.** The servos report nothing back;
  the simulator's `get_joint_positions()` has no physical counterpart yet.
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

1. Recover the CAD model and original Pi source code.
2. Measure real link dimensions; update `config/geometry.yaml` and
   regenerate the URDF.
3. Replace primitive visuals with CAD meshes (same joint names, same API).
4. Measure PWM-to-angle calibration per joint on the physical arm.
5. Implement `RaspberryPiArm` on the Pi with verified limits only.
6. Add camera rendering from `camera_link`.
