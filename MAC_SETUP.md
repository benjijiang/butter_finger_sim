# Mac Setup — Butter Finger Simulator

The simulator **runs on the Mac**. The repository is authored on Windows and
pushed to GitHub; the Mac clones it, creates its own virtual environment,
installs PyBullet, and runs the GUI and tests.

Run everything below in a **normal local Mac Terminal** (or a local VS Code
terminal). Do **not** run the GUI examples from an SSH terminal connected to
the Raspberry Pi — the PyBullet window must open on the Mac's own display.

## 1. Clone and inspect

```bash
git clone <GITHUB_REPOSITORY_URL>
cd butter-finger-sim

python3 --version
uname -m
```

- `python3 --version` should print Python 3.10 or newer.
- `uname -m` prints `arm64` (Apple Silicon) or `x86_64` (Intel); both work.
- If `python3` is missing, install current Python 3 with the official
  installer from <https://www.python.org/downloads/> or with Homebrew
  (`brew install python`).

## 2. Create the virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sim.txt
```

Notes:

- `.venv` is created **on the Mac** and is never committed (it is in
  `.gitignore`).
- **Never copy a virtual environment between Windows, Mac, and Pi.** They
  contain machine-specific binaries; each machine builds its own.
- If PyBullet fails to compile: first run
  `python -m pip install --upgrade pip setuptools wheel` and retry. If it
  still fails, install Apple's command-line developer tools with
  `xcode-select --install`, then retry the install **inside `.venv`**.

## 3. Generate the URDF and run the tests

```bash
python scripts/generate_urdf.py
python -m pytest
```

All tests are dependency-light (no PyBullet needed) and should pass.

## 4. Run the simulator

```bash
python examples/view_robot.py
python examples/joint_sliders.py
python examples/scripted_motion.py
```

- `view_robot.py` — displays the arm; close the PyBullet window to exit.
- `joint_sliders.py` — one slider per joint (base, shoulder, elbow, wrist),
  in radians, using the temporary simulation limits.
- `scripted_motion.py` — smoothstep motion sequence: home → base → shoulder
  and elbow → wrist → home.
- There is also `python examples/go_home.py`, which moves the arm to the
  simulated reference pose.

## 5. Deactivate / reactivate the environment

```bash
# leave the virtual environment
deactivate

# come back later (from the butter-finger-sim directory)
source .venv/bin/activate
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: PyBullet is not installed...` | You are outside `.venv`; run `source .venv/bin/activate` first |
| `URDF not found...` | Run `python scripts/generate_urdf.py` |
| GUI window never appears | You are in an SSH session; use a local Mac Terminal |
| `pip install pybullet` compile error | Upgrade pip/setuptools/wheel, then `xcode-select --install`, retry inside `.venv` |
