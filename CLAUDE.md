# CLAUDE.md — Butter Finger Simulator

Persistent context for Claude Code sessions working on this repository.

## Project objective

A simplified PyBullet digital clone of **Butter Finger**, a four-joint
desktop robot arm (base, shoulder, elbow, wrist) with a camera near the end
of the arm. Immediate goal: an interactive four-joint PyBullet arm with GUI
sliders. Long-term goal: develop high-level motion and perception code
without the physical arm nearby. This is a functional approximate model,
**not** a validated digital twin.

## Device split (strict)

| Device | Role |
|---|---|
| Windows computer | Claude Code authors the project, validates without PyBullet, commits, pushes to GitHub |
| Mac | Clones the repo, creates its own `.venv`, installs PyBullet, runs the GUI and tests |
| Raspberry Pi 5 | Will later run only the physical-arm backend; currently at home and inaccessible |

- **PyBullet runs on the Mac only** — never install or run it on Windows or the Pi.
- Never commit a virtual environment; each machine creates its own.
- The Windows machine has **no Python runtime installed** (only Microsoft
  Store alias stubs), so Windows-side validation is limited to XML parsing
  via PowerShell/.NET and text checks. `models/butter_finger_simple.urdf`
  was hand-transcribed to match `scripts/generate_urdf.py` output; the Mac
  regenerates it, and any tiny formatting diff after regeneration should
  simply be committed.

## Confirmed physical hardware (verified facts)

- Raspberry Pi 5 → UART → Hiwonder RasAdapter5A V1.0 → servo PWM.
- Original physical control class/file was called `Butter_Finger`; that
  source code is currently unavailable, as is the CAD model.
- PWM port mapping: base=1, shoulder=3, elbow=4, wrist=5; ports 2 and 6 unused.
- Recorded home PWM (µs): base 1500, shoulder 2200, elbow 2490, wrist 1400.
- Tested shoulder pulse range: 1200–2220 µs. **Only the shoulder was tested.**

⚠️ **Shoulder warning:** the recorded shoulder home (2200 µs) is 20 µs below
the tested maximum (2220 µs). It is historical calibration data only and
must be revalidated before the physical arm is operated again. Never label
it safe merely because it was previously called "home".

## Known unknowns (do not invent values for these)

- CAD model and exact link geometry.
- Real joint angular ranges and servo rotation directions.
- PWM limits for base, elbow, wrist (kept `null` in config/joints.yaml).
- PWM-to-angle calibration (none exists — no conversion code anywhere yet).
- Real masses, inertias, servo torque/speed.
- Exact camera transform.
- Wrist joint axis (Y in the URDF is a temporary assumption).

## Current temporary geometry (meters, placeholders)

base_height 0.050 · upper_arm_length 0.120 · forearm_length 0.120 ·
wrist_length 0.050 · link_width 0.035. Source of truth is
`config/geometry.yaml`; `scripts/generate_urdf.py` regenerates
`models/butter_finger_simple.urdf` from it. Simulation joint limits are
±1.5708 rad (development-only). Zero radians is a neutral mathematical
reference pose, not a servo electrical midpoint.

## Architectural rules

1. Application code uses **radians only**, through the `ArmBackend`
   interface in `src/butter_finger/arm.py`. Only the future Pi backend will
   know about PWM.
2. `RaspberryPiArm` stays a `NotImplementedError` stub (no hardware imports)
   until measured calibration and verified limits exist for every joint.
3. Keep the five config concepts separate (see `config/joints.yaml` header):
   sim radians, recorded PWM, verified hardware limits, temporary sim
   limits, named poses.
4. URDF joint names are fixed: `base_joint`, `shoulder_joint`,
   `elbow_joint`, `wrist_joint`, plus fixed `camera_link`. Never rename.

## Safety rules

- **Never command real hardware using placeholder limits.**
- Never run or implement a real servo command from this repository.
- The future real backend must reject (not clamp) commands without verified
  safe limits.

## CAD migration plan

When the CAD model is recovered: export meshes, replace the primitive
`<visual>` geometry (and later `<collision>`) in the URDF generator, update
`config/geometry.yaml` with measured dimensions, regenerate. Joint names and
the control API must not change, so application code is unaffected.

## Milestones

- [x] Repository authored on Windows with configs, URDF generator,
      `ArmBackend`, `PyBulletArm`, examples, tests, docs.
- [ ] First run on the Mac: `pytest` green, sliders working (see MAC_SETUP.md).
- [ ] Recover CAD + original Pi code; measure real geometry; update configs.
- [ ] Replace primitives with CAD meshes.
- [ ] Measure PWM-to-angle calibration per joint; record verified limits.
- [ ] Implement `RaspberryPiArm` on the Pi (UART → RasAdapter5A).
- [ ] Camera rendering from `camera_link`.
