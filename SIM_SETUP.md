# Simulation Machine Setup — Butter Finger Simulator

The simulator **runs on the simulation machine**: primarily the **Linux
machine with the GPU**; the **Mac works as a fallback** (see the Mac section
at the end). The repository is authored on Windows and pushed to GitHub; the
simulation machine clones it, creates its own virtual environment, installs
PyBullet, and runs the GUI and tests.

Run everything below in a **local terminal session on the simulation
machine's own display**. Do **not** run the GUI examples over a plain SSH
connection (to this machine or from it to the Raspberry Pi) — the PyBullet
window must open on the simulation machine's own screen. `ssh -X` forwarding
technically works but is slow; prefer running locally.

## 1. Clone and inspect

```bash
git clone <GITHUB_REPOSITORY_URL>
cd butter-finger-sim

python3 --version
```

- `python3 --version` should print Python 3.10 or newer.
- If `python3` is missing or too old, install it from your distribution's
  package manager (e.g. Debian/Ubuntu: `sudo apt install python3 python3-venv`).

## 2. Linux system packages (if needed)

On a typical desktop install nothing extra is required. On a minimal
install, the PyBullet GUI needs OpenGL libraries and venv support:

```bash
# Debian/Ubuntu
sudo apt install python3-venv python3-dev libgl1 mesa-utils
```

If `pip` ends up compiling PyBullet from source (no matching wheel), it also
needs a C++ toolchain: `sudo apt install build-essential`.

With an NVIDIA GPU and the proprietary driver installed, the PyBullet GUI
uses hardware OpenGL automatically — no configuration needed. RGB snapshots
currently use PyBullet's portable Tiny Renderer in both GUI and DIRECT mode;
an EGL renderer can be added later if high-rate headless capture is needed.

## 3. Create the virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sim.txt
```

Notes:

- `.venv` is created **on the simulation machine** and is never committed
  (it is in `.gitignore`).
- **Never copy a virtual environment between machines** (Windows, Linux,
  Mac, Pi). They contain machine-specific binaries; each machine builds its
  own.
- If PyBullet fails to compile: first run
  `python -m pip install --upgrade pip setuptools wheel` and retry inside
  `.venv`. On Linux also install `build-essential` and `python3-dev`.

## 4. Generate the URDF and run the tests

```bash
python scripts/generate_urdf.py
python -m pytest
```

All tests are dependency-light (no PyBullet needed) and should pass.

## 5. Run the simulator

```bash
python examples/view_robot.py
python examples/joint_sliders.py
python examples/scripted_motion.py
python examples/run_action.py --list
python examples/run_action.py happy
python examples/idle_motion.py
python examples/emotion_showcase.py happy curious sad surprised
python examples/emotion_showcase.py --all
python examples/camera_snapshot.py
python examples/face_tracking.py
python examples/face_tracking.py --source sim --detector scripted
```

- `view_robot.py` — displays the arm; close the PyBullet window to exit.
- `joint_sliders.py` — one slider per joint (base, shoulder, elbow, wrist),
  in radians, using the shared calibrated limits.
- `scripted_motion.py` — smoothstep motion sequence: home → base → shoulder
  and elbow → wrist → home.
- `run_action.py --list` — lists the validated, config-driven actions without
  opening a backend.
- `run_action.py <name>` — runs one action from `config/actions.yaml` in the
  GUI. In addition to the utility actions, twenty camera-as-face gestures cover
  greeting, agreement, positive, reflective, low-energy, and reactive moods.
- `idle_motion.py` — continuously runs the slow no-person base scan around the
  `idle_ready` posture (now the full base ±90° range). `face_tracking.py` is
  the attention layer that switches this out for camera tracking when a face is
  detected.
- `emotion_showcase.py` — plays selected emotional actions in order, or all
  twenty with `--all`, with a short idle scan between actions.
- `camera_snapshot.py` — renders the wrist camera in DIRECT mode and writes
  `camera_snapshot.png`; add `--gui` to open PyBullet during capture.
- `face_tracking.py` — simulation-only camera face tracking: the base pans, the
  wrist tilts, and the shoulder adjusts stand-off to keep a face centered, then
  falls back to the idle base scan when the face is lost. Defaults to
  `--source webcam --detector haar` (a real camera via OpenCV/V4L2); use
  `--source sim --detector scripted` to run the whole loop with no camera, and
  `--show` to open the camera preview window. On Linux the webcam needs V4L2
  access (`/dev/video0`; your user in the `video` group). If the arm drives the
  face away from center, flip the matching `sign_*` in `config/tracking.yaml`.
- There is also `python examples/go_home.py`, which moves the arm to the
  simulated reference pose.

Named actions use the shared calibrated radian domain. Emotional `duration_s`
values are intentional: short steps communicate surprise, fear, excitement,
or force, while long steps communicate affection, thought, sadness, or
fatigue. These speeds and load behavior remain unverified on the real arm. A
timed simulator move uses smoothstep interpolation at the configured 240 Hz
control rate; idle and slider target updates omit the duration and remain
non-blocking. Continuous `IdleController` motion remains simulation-only.

The camera renders natively at `640×480`, then rotates the RGB frame 90°
clockwise to a `480×640` portrait image. Its `120°` vertical FOV is a temporary
pinhole approximation; exact intrinsics and wide-angle distortion still need
physical checkerboard calibration.

## 6. Run calibrated actions on the Raspberry Pi

The same action CLIs can select the real radians backend. Run them only on the
Pi connected to the arm, keep the mechanism clear, and include the explicit
hardware acknowledgement:

```bash
python examples/run_action.py happy --backend real --confirm-hardware
python examples/emotion_showcase.py happy curious --backend real --confirm-hardware
```

Both commands first send the exact recorded physical home PWM. Every action is
fully prechecked before its first movement. Out-of-calibration targets are
rejected rather than clamped or extrapolated. The emotional showcase waits
between real actions; it does not run the high-frequency idle scan.

Python API:

```python
from butter_finger import RaspberryPiArm

with RaspberryPiArm() as arm:
    arm.go_home()
    arm.move_joints({"base": 0.3, "shoulder": -0.5}, duration_s=2.0)
```

The real arm has no verified shaft-angle feedback. Returned positions are
last-command estimates, and the two-point mapping does not validate speed,
torque, backlash, or behavior under load.

## 7. Deactivate / reactivate the environment

```bash
# leave the virtual environment
deactivate

# come back later (from the butter-finger-sim directory)
source .venv/bin/activate
```

## Fallback: running on the Mac

The same steps work on macOS (Apple Silicon or Intel) from a normal local
Mac Terminal — clone, `python3 -m venv .venv`, install
`requirements-sim.txt`, generate the URDF, run tests and examples. Skip the
Linux system-packages step. If Python 3 is missing, install it from
<https://www.python.org/downloads/> or Homebrew (`brew install python`). If
PyBullet fails to compile, upgrade pip/setuptools/wheel, install Apple's
command-line developer tools with `xcode-select --install`, and retry inside
`.venv`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: PyBullet is not installed...` | You are outside `.venv`; run `source .venv/bin/activate` first |
| `URDF not found...` | Run `python scripts/generate_urdf.py` |
| GUI window never appears | You are in an SSH session; use a local terminal on the machine's own display |
| GUI fails with OpenGL/GLX errors (Linux) | Install `libgl1` and `mesa-utils`; on NVIDIA, check `nvidia-smi` shows the driver |
| Camera snapshot is empty or points the wrong way | Confirm `config/geometry.yaml` camera offset and the `+Y` forward / `-X` native-up convention |
| `pip install pybullet` compile error | Upgrade pip/setuptools/wheel; Linux: install `build-essential python3-dev`; Mac: `xcode-select --install`; retry inside `.venv` |
