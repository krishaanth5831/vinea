# Vinea

Modular autonomous robot for Dutch commercial greenhouses. Harvesting first, scouting as a second module on the same base. Runs in existing row infrastructure — no greenhouse rebuild.

> **Status: simulation-first.** No hardware budget yet, on purpose — the technical bet gets proven in software before a cent is spent on parts. A Fairino FR5 picks tomatoes off a plant row in MuJoCo today. Week 1 of a 4-week sprint ending 2026-08-27, whose deliverable is a recorded autonomous pick cycle with a success rate and a defensible kg/hr.

---

## What runs today

Everything below is in `simulation/mujoco/` and runs on one laptop. No ROS, no hardware.

```bash
./.venv/bin/python scripts/phase0_smoketest.py          # 6 checks, toolchain health
```

**Pick a tomato off a plant row, by mouse.** Double-click anywhere on the green board and the arm approaches square-on, grips, pulls until the stem gives, carries the fruit to a crate and drops it in.

```bash
./.venv/bin/python simulation/mujoco/week1_mousereach.py
./.venv/bin/python simulation/mujoco/week1_mousereach.py --headless \
    --click -0.50 0.20 --click 0.0 0.60 --click 0.50 0.92 --out demo.mp4
```

**Pick from a table**, same state machine without the plant row:

```bash
./.venv/bin/python simulation/mujoco/week1_gripper.py
./.venv/bin/python simulation/mujoco/week1_gripper.py --wave   # gripper only
```

**Reach a point you name**, no gripper — for reading the IK loop on its own:

```bash
./.venv/bin/python simulation/mujoco/week1_targetreach.py       # type xyz
./.venv/bin/python simulation/mujoco/week1_reach.py             # follow a circle
```

### What's underneath

| | |
|---|---|
| **Arm** | Fairino FR5, loaded from Fairino's official URDF — real SolidWorks masses and inertias, so MuJoCo imports it directly. No CAD conversion. |
| **Gripper** | Robotiq 2F85 from MuJoCo Menagerie, 1.05 kg, mounted on the flange. A **stand-in** — the MVP calls for a cradle-from-below gripper with a peduncle blade, built in Week 2. |
| **Physics** | MuJoCo 3.10. Gazebo dropped (weak deformables and rendering); Isaac Sim ruled out — needs 8 GB VRAM, the dev GPU has 6. |
| **Motion** | `mink` differential IK. MoveIt 2 is not installable on ROS 2 Lyrical, and an open greenhouse row is not a cluttered cell. |
| **Detachment** | The stem is a weld equality constraint switched off above a force threshold. Not FEM fracture. |
| **Reach** | 0.967 m to the flange, **1.100 m to the fingertips** with the gripper on. Both measured by sampling 20k joint configurations, not derived. |
| **Accuracy** | ~3.2 mm steady-state bare, ~5.2 mm with the gripper hanging off it. The difference is droop — a P+D position servo cannot null a constant gravity load. |

The plant row is sized to exactly the band this arm can work: a 31×24 grid over the board was driven point by point, and the panel is the largest rectangle that fits inside the result with margin. Every click on it is pickable.

## What this does not prove

Worth saying before anyone else says it:

- **Sim contact is not real contact.** A MuJoCo grasp rate is not a field grasp rate. A real tomato is soft, wet, and bruises.
- **The stem's break force is invented.** No force gauge has been near a real peduncle for this project. It decides whether picks succeed and it flows straight into any throughput figure.
- **No leaf occlusion, no real lighting, no cluttered truss.** The renderer is basic, which is why the detector will train on real imagery rather than rendered frames.
- **Pick counts here are deterministic replays**, not trials. Repeatability under jitter is Week 2's milestone, and a defensible kg/hr is Week 4's.

## Layout

```
simulation/mujoco/    fr5.py (arm + gripper + scene), reach.py (control loop),
                      week1_*.py (the demos)
third_party/          Fairino URDF, MuJoCo Menagerie — never edited
scripts/              phase0_smoketest.py, setup
docs/                 architecture, concept, decision records
vinea_*/              empty ROS 2 package skeleton, unbuilt this sprint
```

ROS 2 is deliberately out of the demo loop. It is a distributed-systems layer whose value shows up with real hardware and multiple nodes; the packages stay in the repo for the hardware phase.

## Setup

Python 3.14 + venv. MuJoCo 3.10, `mink`, torch (CUDA), Ultralytics, OpenCV.

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/phase0_smoketest.py      # expect 6/6
```

Use `./.venv/bin/python` explicitly — the system Python has none of this.

## The problem

Dutch greenhouse harvesting labour costs about **€5/m² per year** — €50k/ha/year, so roughly €250k/year for a 5-hectare grower, before scouting. Seasonal labour supply is structurally shrinking. The only funded competitor requires a full greenhouse rebuild, which leaves most of the market with nothing to buy.

## The solution

One chassis, swappable modules, working in the rows that already exist.

- **Harvest** — autonomous picking, CV-guided, crop-specific gripper (tomato first)
- **Scout** — plant health inspection, as a passive byproduct of the harvest cameras rather than a separate pass

Only the gripper, row-width adapter and module set change between customers.

**Business model: Robot-as-a-Service.** Monthly subscription covering hardware, software, maintenance and data, 12-month minimum, three tiers in the €4,500–6,500/robot/month range.

> **Pricing is provisional and under active rebuild** — not final until after a paid pilot. The unit economics are blocked on a validated kg/hr from this repo. Design target for that number: 24,000 kg/week/robot at 24/7 operation, roughly 3 robots per 5-hectare grower.

## Roadmap

| | |
|---|---|
| **Phase 0** | Toolchain — ✅ closed, smoke test 6/6 |
| **Week 1** | MuJoCo fluency, IK reach, gripper mounted and picking — ✅ build items done |
| **Week 2** | Greenhouse row, breakable stems, contact tuning, **ten picks in a row** |
| **Week 3** | Detection: camera in sim, tomato detector, 2D→3D against ground truth |
| **Week 4** | Closed loop with no hardcoded positions, 50+ logged picks, success rate and **kg/hr** |

Then: hardware. Nothing built here gets thrown away — the sim becomes the spec for the physical prototype.

## Status

- [x] Idea anatomy — 9/9 pillars
- [x] Toolchain and simulation environment
- [x] Arm in sim, reaching arbitrary targets
- [x] Gripper mounted, full pick-and-crate cycle
- [ ] Grower validation — 4 interviews done, more booked
- [ ] Repeatable grasp under position jitter
- [ ] Perception loop closed
- [ ] Validated kg/hr
- [ ] Technical cofounder
- [ ] Pre-seed raise

---

*Solo project. Pre-team, pre-funding, pre-hardware.* · [getvinea.nl](https://getvinea.nl)
