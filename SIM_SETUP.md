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
python examples/run_action.py base_scan
python examples/camera_snapshot.py
```

- `view_robot.py` — displays the arm; close the PyBullet window to exit.
- `joint_sliders.py` — one slider per joint (base, shoulder, elbow, wrist),
  in radians, using the temporary simulation limits.
- `scripted_motion.py` — smoothstep motion sequence: home → base → shoulder
  and elbow → wrist → home.
- `run_action.py --list` — lists the validated, config-driven simulation
  actions without opening PyBullet.
- `run_action.py <name>` — runs one action from `config/actions.yaml` in the
  GUI; available names include `home`, `demo_reach`, `base_scan`,
  `reach_and_return`, `wrist_up`, and `wrist_down`.
- `camera_snapshot.py` — renders the wrist camera in DIRECT mode and writes
  `camera_snapshot.png`; add `--gui` to open PyBullet during capture.
- There is also `python examples/go_home.py`, which moves the arm to the
  simulated reference pose.

The named actions use temporary radians and are not approved for the physical
PWM-controlled arm. A timed simulator move uses smoothstep interpolation at
the configured 240 Hz control rate; slider updates omit the duration and
remain non-blocking.

The camera renders natively at `640×480`, then rotates the RGB frame 90°
clockwise to a `480×640` portrait image. Its `120°` vertical FOV is a temporary
pinhole approximation; exact intrinsics and wide-angle distortion still need
physical checkerboard calibration.

## 6. Deactivate / reactivate the environment

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
