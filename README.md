# Vinea — greenhouse harvesting robot, in simulation

A MuJoCo simulation of a robot arm harvesting tomatoes in a Dutch high-wire
greenhouse. A Fairino FR5 arm with a Robotiq 2F85 gripper finds fruit with a
camera, plans a route that keeps clear of every tomato it is *not* picking,
picks, and drops the fruit in a crate. It grows from one arm at one row
(Weeks 1–4) to a whole house worked from a trolley on the heating-pipe rails
with two arms (Week 5), and to tomato-on-the-vine trusses that are cut rather
than pulled (Week 5b).

It was built **simulation-first**, before any hardware, to find out whether
the idea holds up and where it breaks. The answers, including the ones that
came out badly, are written up in [docs/results.md](docs/results.md).

> **Status: complete research sim, not under active development.** The work
> here ran as a four-week sprint in August 2026. Vinea's module work has since
> moved to separate repos:
> [B.L.U.S.H](https://github.com/krishaanth5831/B.L.U.S.H) (harvest, on the SO-101 arm),
> [S.H.E.A.R](https://github.com/krishaanth5831/S.H.E.A.R) (deleafing, on the SO-101 arm) and
> [H.O.I.S.T](https://github.com/krishaanth5831/H.O.I.S.T) (lowering and rehanging the crop).
> Nothing in this repo has run on a real robot.

---

## What it does

- **A real arm model.** The FR5 is loaded from Fairino's official URDF (real
  masses and inertias), with the Robotiq 2F85 from MuJoCo Menagerie on the
  flange. Inverse kinematics is done with [`mink`](https://github.com/kevinzakka/mink).
- **A greenhouse row with breakable stems.** Venlo-style high-wire rows at real
  dimensions. Each tomato hangs on a stem that lets go above a force threshold,
  so a careless arm knocks fruit off the plant.
- **Collision-aware picking.** Every pick is planned before the arm moves and
  checked against every fruit it is not picking. If no route keeps 40 mm of
  clearance, the pick is refused instead of improvised. A runtime guard measures
  the same clearance every control cycle and abandons the pick under 15 mm.
- **Failure attribution.** A contact "black box" records why a fruit fell, and
  turns repeat failures into constraints the planner reads next time
  (`simulation/lessons.json`).
- **Two cameras.** An eye-in-hand camera on the wrist and a survey camera on a
  pan-tilt mast that scans the whole row and decides the pick order. Fruit is
  found by a colour (HSV) detector, with a stock YOLO model as a comparison.
- **Click-to-place demos.** Double-click anywhere on a green board to hang a
  tomato, press SPACE, and watch the arm work them out. The board is exactly the
  arm's measured reachable zone.
- **A whole house.** Four rows on a trolley that rides the heating pipes: scout
  the aisle, map the ripe fruit, plan the stops, drive and harvest, with two arms
  stepped in one physics loop.
- **Trusses.** A second crop: six tomatoes on one stem, judged as a cluster, cut
  with a blade and crated whole.

## Requirements

- **Linux.** Developed on Ubuntu 26.04. Other platforms are untested.
- **Python 3.14** (what the repo was developed and tested on; nearby versions
  may work but are untested).
- **A display** for the windowed demos. Everything also runs headless through
  MuJoCo's EGL renderer.
- **Optional: an NVIDIA GPU.** Only the YOLO comparison detector uses torch, and
  it falls back to CPU.
- **Not needed: ROS 2.** The simulation is plain Python. ROS 2 is only checked
  by the smoke test (see [Troubleshooting](#troubleshooting)).

## Setup

Run all of this from the repository root.

**1. Clone the repo and make a virtual environment.**

```bash
git clone https://github.com/krishaanth5831/vinea.git
cd vinea
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs MuJoCo (3.10 or newer), `mink`, torch, Ultralytics
and OpenCV, plus a few small packages the ROS 2 Python layer needs if you have
ROS installed.

**2. Fetch the robot models.** The arm and gripper come from two upstream
repos. They are cloned into `third_party/` (which git ignores), never copied
into this repo. Only the folders the sim needs are checked out:

```bash
git clone --filter=blob:none --sparse \
    https://github.com/FAIR-INNOVATION/frcobot_ros2.git third_party/frcobot_ros2
git -C third_party/frcobot_ros2 sparse-checkout set fairino_description

git clone --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set robotiq_2f85
```

Developed against `frcobot_ros2` at `867cb32` and `mujoco_menagerie` at
`71f066a`; a fresh clone at `eb6d7ce` and `370c439` (November 2026) passed the
same tests. If a newer upstream breaks something,
`git -C third_party/<repo> checkout <hash>` pins it back.

**3. (Optional) Fetch the YOLO weights.** Only needed for `--detector yolo` and
for `detect.py`, which scores both detectors. This is the stock 5.6 MB
Ultralytics COCO checkpoint, not a trained tomato model:

```bash
./.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
mv yolo11n.pt models/weights/
```

**4. Check the toolchain.**

```bash
./.venv/bin/python scripts/phase0_smoketest.py
```

It runs six checks: MuJoCo physics, offscreen rendering, URDF import, an IK
solve, the CV stack, and ROS 2. **5 of 6 with only the ROS 2 check failing is
fine** for everything in this repo.

Always run scripts with `./.venv/bin/python` explicitly. Your system Python
does not have these packages.

## Quick start

Open a MuJoCo window with a green board, hang tomatoes on it, and watch the arm
harvest them:

```bash
./.venv/bin/python simulation/mujoco/week4_place.py --windowed
```

| key | does |
|---|---|
| double-click the board | hang a tomato there |
| `A` | auto-fill the board |
| `C` | clear it |
| `SPACE` | harvest |
| `Q` | quit |

Up to 15 fruit, as close together as you like. You can add more while it is
picking, and it re-plans around them.

## Usage

Every script runs standalone from the repo root and prints the numbers it
stands on. [COMMANDS.md](COMMANDS.md) lists every script and flag, and what you
actually see when you run it (window, clicks, text only, image, or video).

**Demos that open a window:**

```bash
# click a tomato onto the board; the arm picks it and crates it (Week 1)
./.venv/bin/python simulation/mujoco/week1_mousereach.py

# plan and harvest a whole greenhouse row (Week 2)
./.venv/bin/python simulation/mujoco/week2_pick.py

# four panels: scene, deck camera with the plan drawn on it, wrist camera, stats (Week 3)
./.venv/bin/python simulation/mujoco/week3_watch.py --view all

# click fruit onto the board, then watch the four panels pick them (Week 4)
./.venv/bin/python simulation/mujoco/week4_watch.py

# a whole house: scout, plan, drive and harvest, in six panels (Week 5)
./.venv/bin/python simulation/mujoco/farm/watch.py

# two arms on one trolley, two live windows (Week 5)
./.venv/bin/python simulation/mujoco/two_arm_farm.py

# the two-arm trolley, but you place the tomatoes along each row first
./.venv/bin/python simulation/mujoco/two_arm_farm.py mousereach

# the truss crop: a shift in six panels, and the two-arm version (Week 5b)
./.venv/bin/python simulation/mujoco/farm/watch_truss.py
./.venv/bin/python simulation/mujoco/two_arm_farm_truss.py
```

`python 2armfarm.py` and `python 2armfarmtruss.py` at the repo root are
shortcuts for the two `two_arm_farm*.py` scripts and take the same arguments
(for example `./.venv/bin/python 2armfarm.py mousereach`).

**Headless runs and measurements** (text only, faster than real time):

```bash
./.venv/bin/python simulation/mujoco/week4_place.py --grid 15              # place 15 fruit, harvest, print results
./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless --no-lessons   # ten jittered picks
./.venv/bin/python simulation/mujoco/farm/run.py --truth --seed 7          # a whole-house shift
./.venv/bin/python simulation/mujoco/farm/trussrun.py --truth              # a truss shift
./.venv/bin/python simulation/mujoco/farm/throughput.py                    # kg/hr from measured numbers
```

**Recording a video.** Most demos take `--out file.mp4` and render offscreen
instead of opening a window. For example:

```bash
./.venv/bin/python simulation/mujoco/week2_pick.py --headless --out week2_pick.mp4 --camera row
```

Name captures after the script that made them, so a video always says which
code produced it. Videos are git-ignored.

### Things to know before you run anything

- **A window and `--out` cannot be combined.** Recording uses the offscreen
  renderer and a window needs GLFW, so you watch or record, not both.
- **Two default speeds, on purpose.** The interactive Week 4 tools run at 0.4
  of the arm's rated joint speed so you are not watching a pick for 30 seconds.
  Every measurement tool stays at 0.15, the speed every recorded number was
  taken at. Pass `--speed 0.15` when comparing against a recorded number.
- **`week2_pick.py` writes to `simulation/lessons.json`** (the planner's lesson
  store) unless you pass `--no-lessons`. Use it for any run that is not
  deliberately about learning, or the next comparison runs against a different
  baseline.
- **Measured constants carry their measurement in a comment.** Many numbers in
  the code were read off a sweep. Changing one without re-running its sweep
  quietly invalidates a result.

## Configuration

There are no config files or environment variables to set. Behaviour is chosen
per run with command-line flags (`--help` on any script lists them). The only
environment variable involved is `MUJOCO_GL=egl` (MuJoCo's offscreen renderer),
which the scripts that render offscreen set for themselves.

## Project layout

```
simulation/mujoco/        the simulation: every module runs standalone
  fr5.py                  the arm, the gripper, the tool frame
  reach.py                IK, the speed limit, "has it arrived"
  plant_row.py            breakable stems: the weld and the force threshold
  greenhouse.py           the house, and the one place the scene is assembled
  mission.py              plan a pick, check it clears the crop, refuse if not
  pickcycle.py            the one pick state machine Week 1's demos share
  incident.py             why a tomato nobody was picking ended up on the floor
  lessons.py              turn that into a constraint the next plan must satisfy
  camera.py, detect.py    the wrist camera, deprojection, and the detectors
  deck_cam.py             the survey camera on its pan-tilt mast; picks the order
  outcomes.py, picklog.py one attempt -> one named result, appended to a log
  carrytrace.py           why a fruit that was gripped never reached the crate
  ripeness.py             why uniform red spheres carry no ripeness signal
  legacy_cycle.py         the old unplanned cycle, kept as the baseline to beat
  week1_*.py ... week4_*.py   the weekly demos and measurements
  two_arm_farm.py         both arms on one trolley, two live windows
  two_arm_farm_truss.py   the same machine, on trusses
  farm/                   Week 5: the whole house (house, trolley, crop, scout,
                          route, run, watch, duo, decks, truss, throughput, ...)
  parked/                 an early cradle-and-blade gripper, parked on purpose
simulation/lessons.json   what the planner has learned so far
simulation/urdf/          a placeholder URDF the smoke test loads
2armfarm.py, 2armfarmtruss.py   root shortcuts for the two-arm sims
tests/                    correctness checks and regression baselines
layouts/                  saved fruit layouts for week4_place.py --layout
models/weights/           YOLO weights go here (not committed)
scripts/                  phase0_smoketest.py; setup.sh/run_sim.sh are the old ROS plan
hardware/so101_greenhouse/  a fully 3D-printed tabletop test greenhouse for the
                          SO-101 arms (Onshape FeatureScript source + STLs)
docs/results.md           the full write-up: what was measured, week by week
docs/concept.md           the concept the repo started from
docs/architecture.md, docs/decisions/   the original ROS 2 plan, superseded
docs/bugsweep/            an August 2026 bug-sweep report and its bug-log entries
ros2/                     empty ROS 2 package skeletons for a future hardware phase
third_party/              Fairino URDF and MuJoCo Menagerie (cloned, git-ignored)
COMMANDS.md               every command, and what you see when you run it
```

## Testing

There is no test framework on purpose; each check is a plain script that prints
PASS/FAIL and exits non-zero on failure.

```bash
./.venv/bin/python scripts/phase0_smoketest.py   # the machine: 6 toolchain checks
./.venv/bin/python tests/test_sim.py             # the repo: picks, the weld, the tool frame, trusses (seconds)
./.venv/bin/python tests/board_walk.py           # every point on the click board still crates (~40 s, expect 42/42)
./.venv/bin/python tests/baseline.py             # Week 1's two pick cycles, diffed line by line against tests/baselines/
```

None of them asserts a headline number. They check that the mechanisms the
numbers are built on still work.

## Troubleshooting

- **The smoke test fails the ROS 2 check.** Expected if ROS 2 is not installed.
  Nothing in the simulation uses it.
- **An error about a missing `fairino5_v6.urdf` or `2f85.xml`.** `third_party/`
  is missing or its sparse checkout does not include the right folder. Redo
  Setup step 2.
- **YOLO fails to load `models/weights/yolo11n.pt`.** Do Setup step 3, or stick
  to the default HSV detector.
- **No window appears, or a GLFW error over SSH.** The windowed demos need a
  local display. Use the headless form of the command (`--headless`, or no
  `--windowed` for `week4_place.py`) or record with `--out`.
- **`git status` shows `simulation/lessons.json` changed.** A `week2_pick.py`
  run without `--no-lessons` added a lesson. `git checkout simulation/lessons.json`
  puts it back.
- **Windowed runs are slow.** Watching runs at wall-clock speed. Headless runs
  go faster than real time, but only when nothing else is loading the machine.
- **The `farm/` scripts take a while before anything moves.** The survey runs
  first; on the two-arm mousereach demo that is roughly a minute or more.

## Contributing

Issues and pull requests are welcome. This repo is no longer under active
development, so expect slow replies. If you change a measured constant (one
whose comment cites a sweep, a force or a failure rate), re-run the sweep it
came from and quote the new numbers in the pull request, and run the four
checks under [Testing](#testing) before sending it.

## License

[MIT](LICENSE). Built by Krishaanth Ramaraj for Vinea ([getvinea.nl](https://getvinea.nl)).
