# Vinea

Modular autonomous robot for Dutch commercial greenhouses. Harvesting first, scouting as a second module on the same base. Runs in existing row infrastructure — no greenhouse rebuild.

> **Status: simulation-first.** No hardware budget yet, on purpose — the technical bet gets proven in software before a cent is spent on parts. A Fairino FR5 finds tomatoes with a camera, plans a route that clears the crop, and harvests them in a MuJoCo greenhouse — **without disturbing the fruit it is not picking**. Week 4 of a 4-week sprint ending 2026-08-27, whose deliverable is a recorded autonomous pick cycle with a success rate and a defensible kg/hr.

---

## What runs today

Everything below is in `simulation/mujoco/` and runs on one laptop. No ROS, no hardware.

📖 **[COMMANDS.md](COMMANDS.md) lists every command with what you actually see** — which open a window, which you can click in, which only print numbers, which record silently.

```bash
./.venv/bin/python scripts/phase0_smoketest.py          # 6 checks, toolchain health
```

**Put tomatoes wherever you like and watch it work them out.** Up to 15 fruit, anywhere in the arm's measured envelope, in an arrangement the robot has never seen — and more can be added while it is picking, which it notices.

```bash
# place them yourself — a window opens with a green board that is exactly
# the arm's reachable zone. Double-click it to hang a tomato there.
#   double-click  place      A  auto-fill      C  clear
#   SPACE         harvest    Q  quit
./.venv/bin/python simulation/mujoco/week4_place.py --windowed

# watch it work an auto-placed row
./.venv/bin/python simulation/mujoco/week4_place.py --grid 6 --windowed

# four fruit, then three more appear mid-run and it re-plans around them
./.venv/bin/python simulation/mujoco/week4_place.py --grid 4 --add-at 2 --windowed

./.venv/bin/python simulation/mujoco/week4_place.py --grid 15         # headless
./.venv/bin/python simulation/mujoco/week4_place.py --grid 12 --save layouts/a.json
```

The green board is not decoration — it is the guaranteed reachable rectangle measured cell by cell on a 31×24 grid, so **anywhere you can click is somewhere the arm can work**. Placements that break the 200 mm spacing rule are refused with the reason printed.

The placement rules are the interesting part, and both are measured rather than chosen. Fruit may go anywhere inside the reach envelope that was mapped cell by cell on a 31×24 grid — a guaranteed rectangle, then a dome that closes overhead — and **no closer than 200 mm to each other**, because below that the stems load one another past the detach threshold and a truss snaps itself before the arm has moved.

Adding fruit mid-run **voids every checked plan** rather than weakening it: the planner's guarantee is that the route clears every fruit it is *not* picking, and a fruit that appears after the check was never in it. The crop carries a version, and a plan built against an older one is thrown away.

### The Week 4 result — a throughput number, and it is bad

57 logged attempts across four crop densities, arbitrary layouts the robot had never seen:

| | |
|---|---|
| cycle time | **31.3 s** mean (7.7 – 60.6) |
| clean picks | **26/57 — 46%** |
| collateral damage | **0** |
| kg/hr, single arm | **6.3** |
| kg/week/robot at 24/7 | **1,058** |

**The design target is 24,000 kg/week/robot. This is 22.7× short.** That is the most useful thing in this repo. It was found in a simulator in August, before a cent was spent on an arm, which is the entire argument for building it this way.

Two things it is not. It is not a forecast — it is an upper bound with the chassis bolted in place, no travel between trusses, no headland turn, no ripeness selection and no crate swap, any of which only makes it worse. And it is not a verdict on the arm: cycle time is fine, the losses are all in the tool.

**Where the picks go:**

| | |
|---|---|
| clean | 26 |
| **grasp failed** (mostly: knocked the fruit off instead of picking it) | **20** |
| guard aborted the mission on purpose | 5 |
| dropped in carry | 3 |
| **ejected** — left the gripper at speed | 3 |

⚠️ **The clean rate is substantially luck.** The grasp goes unstable on **34 of 57 attempts**, and **17 of those crate anyway** — the fruit is flung and happens to fly toward the crate. Peak speed of a fruit that is supposed to be held: **4.22 m/s**. So "ejected: 3" undercounts the problem by an order of magnitude, and 46% clean is not 46% of grasps under control.

That instability was measured directly (`carrytrace.py`) and **is not a tunable**: it survives hold force 255→80, pad rolling friction 1e-4→0.3, `CARRY_SPEED` 0.25→0.05, and a 4× smaller physics timestep. The 2F85 pinches a smooth 66 mm sphere at ~113 N when holding it needs 1.18 N, and a sphere between converging pads is unstable — 3.9 mm of position error is enough to start it. **The fix is the tool, not a constant**, which is the sharpest argument this project has for the cradle-from-below gripper the MVP specifies.

**Harvest a row in a greenhouse.** The arm plans each pick before it moves, checks the route clears every fruit it is *not* picking, then flies it.

```bash
./.venv/bin/python simulation/mujoco/week2_pick.py                          # watch it, in a window
./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless   # the milestone
./.venv/bin/python simulation/mujoco/week2_pick.py --headless --out week2_pick.mp4 --camera row
```

### The Week 2 result

The first version of this cycle scored **10 picks out of 10**. It was also knocking a neighbouring tomato off the plant on **8 of those 10**, and reporting none of it, because the scorer only ever looked at the fruit it meant to take.

| | old cycle | planned cycle |
|---|---|---|
| fruit in the crate | 10/10 | 10/10 |
| **neighbours knocked off the plant** | **8/10** | **0/10** |
| **clean picks** (crated, nothing else touched) | **2/10** | **10/10** |
| closest approach to a fruit it was not picking | contact | 32 mm |
| cycle time, mean | 23.8 s | 28.1 s |

Same grasp, same stem model, same 20 mm of position jitter. The 4.3 s is what routing around the crop costs, and it buys back four tomatoes in every ten.

The whole row picked in sequence, with no reset between fruit, is **5/5 clean**.

### How fast can it go?

`--speed` is a fraction of the FR5's rated 180 deg/s. Ten jittered trials at each:

| speed | clean | cycle time | collateral |
|---|---|---|---|
| 0.15 (default, ~27 deg/s) | **10/10** | 28.1 s | 0 |
| 0.25 | 7/10 | 20.7 s | 0 |
| 0.50 | 7/10 | 18.1 s | 0 |
| 0.75 | 6/10 | 17.6 s | 0 |
| 1.00 (full rated speed) | 6/10 | 18.1 s | 0 |

Two things worth reading off that table.

**Collateral damage is zero at every speed.** The planner and the guard are geometric, so they do not care how fast the arm moves — nothing the arm does at 180 deg/s puts it closer to a neighbouring tomato than 32 mm. Speed costs grasps, not plants.

**Cycle time stops improving past ~0.5.** Beyond that the fixed-duration phases dominate — the grip ramp, the settles, and the pull, which is deliberately held at 0.05 whatever the cycle speed is because a 12 N detach threshold cannot be resolved inside the ~48 N setpoint step a fast joint move produces. So above half speed you buy no throughput and lose a fruit in ten.

**What actually breaks is the gripper, not the arm.** At full speed the approach and grasp are fine — insert arrives 2-5 mm out, the stem releases at 12-16 N — and then the arm accelerates out of the row and throws the tomato clean out of the pads, 185 mm and 376 mm from the tool by the end of a 290 mm move. The legs that carry fruit are therefore capped separately (`mission.CARRY_SPEED`), and sweeping that cap shows a sharp cliff between 0.35 and 0.25 rather than a graceful rolloff: the fruit is either held or launched.

That is a result, not a tuning detail. **Payload retention is what caps kg/hr, not joint speed** — and it is the strongest argument yet for the cradle-from-below gripper the MVP calls for, which supports a truss instead of pinching it.

```bash
./.venv/bin/python simulation/mujoco/week2_pick.py --speed 1.0    # watch it run flat out
```

**Why it was failing.** Not the grasp and not the pull — the trip *home*. The arm's home posture puts `tool0` at x=0.671 and the fruit hang at x=0.60, so home is **inside the canopy**, and driving there from the crate cut the corner through the plants at 20.9 N against a 12 N stem threshold. `STAGE_X` existed to stop exactly this on the way in; the return leg never got the same treatment.

### Pick a tomato by mouse (Week 1)

Double-click anywhere on the green board and the arm approaches square-on, grips, pulls until the stem gives, carries the fruit to a crate and drops it in.

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

### How the picking works now

Four pieces, each of which does one job:

| | |
|---|---|
| **`mission.py`** | Plans the whole route as a waypoint list, replays it kinematically, and measures the distance from every collision geom on the arm to every fruit it is not picking, at every point along the way. Under 40 mm and the route is replaced — a different pull direction, a rolled wrist, a lane under or over the canopy, a deeper staging plane. If none of them clears, **the pick is refused** rather than improvised. Planning costs ~0.15 s against a 28 s pick. |
| **`Guard`** | The net under the plan. The same clearance is measured *for real* every control cycle; under 15 mm and the mission is abandoned on purpose. The plan is kinematic and cannot see servo lag, so it is not a proof. |
| **`incident.py`** | A black box on the physics loop. Every contact involving a fruit is logged as it happens, so when a peduncle lets go there is a record of what was touching it in the milliseconds before, and the cause is a lookup rather than a guess: *tool strike*, *pinned against the row*, *knocked by another fruit*, *stem overload*. |
| **`lessons.py`** | Turns an attributed incident into a constraint the planner reads next time, and widens it if the same thing happens again. |

The arm no longer parks at its home posture — it parks on the staging plane, outside the crop, and every mission begins and ends there.

### What's underneath

| | |
|---|---|
| **Arm** | Fairino FR5, loaded from Fairino's official URDF — real SolidWorks masses and inertias, so MuJoCo imports it directly. No CAD conversion. |
| **Gripper** | Robotiq 2F85 from MuJoCo Menagerie, 1.05 kg, mounted on the flange. A **stand-in** — the MVP calls for a cradle-from-below gripper with a peduncle blade. That design is [parked](simulation/mujoco/parked/), deliberately: Week 2's real problem was motion, not the tool. |
| **Greenhouse** | Venlo-type high-wire house — substrate gutter at 0.32 m, plants on strings to a 2.60 m high wire, 1.60 m row pitch, heating pipes doubling as the trolley rail. All of it `contype=0` scenery: it is drawn and cannot be hit, so none of the tuned contact numbers move. |
| **Collision checking** | The arm is covered by enclosing spheres and the crop kept as exact primitives. ⚠️ Not `mj_geomDistance` — the arm's geoms are meshes, MuJoCo 3.10 has no native CCD, and the MPR fallback returned **0.0 mm for two bodies 276 mm apart**. A planner built on that refuses routes for collisions that are not there. |
| **Physics** | MuJoCo 3.10. Gazebo dropped (weak deformables and rendering); Isaac Sim ruled out — needs 8 GB VRAM, the dev GPU has 6. |
| **Motion** | `mink` differential IK. MoveIt 2 is not installable on ROS 2 Lyrical, and an open greenhouse row is not a cluttered cell. |
| **Detachment** | The stem is a weld equality constraint switched off above a force threshold. Not FEM fracture. |
| **Reach** | 0.967 m to the flange, **1.100 m to the fingertips** with the gripper on. Both measured by sampling 20k joint configurations, not derived. |
| **Accuracy** | ~3.2 mm steady-state bare, ~5.2 mm with the gripper hanging off it. The difference is droop — a P+D position servo cannot null a constant gravity load. |

The plant row is sized to exactly the band this arm can work: a 31×24 grid over the board was driven point by point, and the panel is the largest rectangle that fits inside the result with margin. Every click on it is pickable.

## What this does not prove

Worth saying before anyone else says it:

- **Sim contact is not real contact.** A MuJoCo grasp rate is not a field grasp rate. A real tomato is soft, wet, and bruises.
- **The stem's break force is invented.** No force gauge has been near a real peduncle for this project. It is set to 12 N by the *simulator* — gripping an attached fruit loads the stem 6-8.6 N by itself, so anything under ~9 N detaches on contact.
  **It does not, however, decide whether picks succeed.** Swept 9 / 12 / 16 / 20 N over one 8-fruit layout, every fruit returned the *identical* outcome at every value, and kg/hr moved 11% on cycle-time noise alone. `SNAP_N` sets when the stem parts and nothing downstream depends on it: the failures happen either before detachment (the arm never arrives, or never grips) or after it (the fruit is flung during carry). Still an invented number that has to be declared — just not the one the throughput figure is fragile to.
  ```bash
  ./.venv/bin/python simulation/mujoco/week4_snap.py     # the sweep
  ```
- **The planner is no longer given perfect positions, but the camera is perfect.** Week 3 closed the loop — an eye-in-hand camera, a detector and a deprojection verified to 0.39 mm against ground truth. What remains ideal is the *sensor*: square pixels, no distortion, principal point exactly centred, no noise, and extrinsics known to machine precision. Real hand-eye calibration is a millimetre-level problem on its own and is not in this repo.
- **The chassis never moves.** Every number here is measured with the arm bolted in one place, working fruit inside its own envelope. A real harvester spends a large fraction of its cycle travelling down the row and turning at the headland, and none of that is simulated or counted in any cycle time. It is the single largest reason a kg/hr figure from this repo is an upper bound.
- **It picks every fruit it sees.** No ripeness selection — that is deferred by design and blocked on a dataset licence — so the throughput assumes every tomato is a target, which no real pass ever is.
- **A hand-placed crop is not a crop.** `week4_place.py` will put fruit anywhere, which is what makes it useful for stress-testing the planner, and it means the layouts are arrangements *chosen* rather than ones a plant produced. Real trusses sit where the plant puts them.
- **The greenhouse is scenery, not obstacles.** Every leaf and post is `contype=0` — drawn, uncollidable. The arm cannot be occluded by a leaf or blocked by a wire, and in a real row both happen. It looks like a greenhouse; it does not yet behave like one.
- **Nothing in the crop can move except the fruit.** Real trusses swing, stems bend, and a plant pushed at 09:00 is somewhere else at 09:01.
- **A clean run is not a proof of no collateral damage** — it is ten samples at one jitter level on a five-fruit row where the fruit are 130 mm apart. The one refusal the planner produced before the wrist-roll search was added is the honest signal here: this row has configurations that are genuinely tight.

### Can it learn from its mistakes?

Yes, with a caveat worth being precise about, because the word covers two very different things.

What is built is **case-based constraint learning**. An incident is attributed to a cause, the cause selects a remedy, and the remedy is a constraint the planner reads before the next pick — widening if the same failure recurs. Fed the four real losses from the old cycle, it produces one rule, not four: *"keep clear of t3 in transit"*, seen 4×, because the lesson is keyed on the victim and the kind of move, not on which tomato was being picked at the time. That generalisation is the part that makes it useful.

What it is **not** is reinforcement learning. No policy, no reward, no gradient; it cannot invent a manoeuvre nobody programmed. That trade is deliberate: it learns from *one* example rather than thousands (a pick is 28 s), `simulation/lessons.json` is a readable file of sentences so a grower can be told why the robot avoids one truss, and a constraint can only ever refuse more routes — so the worst case is a refused pick, never a knocked truss.

The honest limit: a lesson about `t3` applies to `t3`. Making it a lesson about *geometry* — "fruit hanging within 150 mm below the pull line get struck by a straight-down pull" — is the Week 4 version, and the incident record already carries the relative offsets needed to do it.

## Layout

```
simulation/mujoco/
  fr5.py              the arm, the gripper, the tool frame
  reach.py            IK, the speed limit, "has it arrived"
  plant_row.py        breakable stems — the weld and the force threshold
  greenhouse.py       the house, and the one place the scene is assembled
  mission.py          plan a pick, verify it clears the crop, refuse if it cannot
  incident.py         why a tomato that nobody was picking ended up on the floor
  lessons.py          turn that into a constraint the next plan has to satisfy
  week2_pick.py       plan → fly → explain → learn, and the scoring
  camera.py           the eye-in-hand sensor, intrinsics, deprojection
  detect.py           the detectors, scored on recall and false positives
  week3_perceive.py   see the fruit, estimate where it is, pick it from that
  outcomes.py         one attempt → exactly one named failure bucket
  picklog.py          one row per attempt, appended, with the cycle time
  carrytrace.py       why a fruit that was gripped never reached the crate
  week4_place.py      put fruit anywhere, harvest them, add more mid-run
  week4_watch.py      the same, in a four-panel window you click to place in
  week4_run.py        the throughput campaign across crop densities
  legacy_cycle.py     the unplanned cycle, kept as the baseline to beat
  week1_*.py          the Week 1 demos
  parked/             the cradle-and-blade gripper — early, not abandoned
simulation/lessons.json   what it has learned so far
third_party/          Fairino URDF, MuJoCo Menagerie — never edited
scripts/              phase0_smoketest.py, setup
docs/                 architecture, concept, decision records
vinea_*/              empty ROS 2 package skeleton, unbuilt this sprint
```

Every module runs standalone and prints the numbers it stands on:

```bash
./.venv/bin/python simulation/mujoco/plant_row.py    # does the weld read 1.18 N?
./.venv/bin/python simulation/mujoco/mission.py      # plan every fruit, show the routes
./.venv/bin/python simulation/mujoco/incident.py     # the old cycle, with the black box on
./.venv/bin/python simulation/mujoco/lessons.py      # real failures -> constraints
./.venv/bin/python simulation/mujoco/greenhouse.py   # render the house from three cameras
```

A bare run harvests the whole row and leaves the window open when it finishes; `--trials N` is the jittered repeatability measurement instead. Captures are named after the script that produced them — `week2_pick.py` writes `week2_pick.mp4` — so a video in the repo root always says which code made it.

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
| **Week 2** | Greenhouse row, breakable stems, mission planning, **ten clean picks in a row** — ✅ 10/10, zero collateral |
| **Week 3** | Detection: camera in sim, tomato detector, 2D→3D against ground truth — ✅ deprojection PASS at 0.39 mm, 4/5 clean on estimated positions vs 5/5 on truth |
| **Week 4** | Closed loop with no hardcoded positions, 50+ logged picks, success rate and **kg/hr** — in progress |

Then: hardware. Nothing built here gets thrown away — the sim becomes the spec for the physical prototype.

## Status

- [x] Idea anatomy — 9/9 pillars
- [x] Toolchain and simulation environment
- [x] Arm in sim, reaching arbitrary targets
- [x] Gripper mounted, full pick-and-crate cycle
- [x] Repeatable grasp under position jitter — 10/10 at ±20 mm
- [x] Collision-aware mission planning, scored on collateral damage
- [x] Failure attribution and a lesson store the planner reads
- [ ] Grower validation — 4 interviews done, more booked
- [ ] Perception loop closed
- [ ] Validated kg/hr
- [ ] Technical cofounder
- [ ] Pre-seed raise

---

*Solo project. Pre-team, pre-funding, pre-hardware.* · [getvinea.nl](https://getvinea.nl)
