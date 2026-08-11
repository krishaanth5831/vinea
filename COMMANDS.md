# Commands — what each one does, and what you actually see

Every command in the repo, Weeks 1–4, with what happens when you run it.

**Run everything from the repo root** (`~/Desktop/personal_projects/vinea`) and use `./.venv/bin/python` explicitly — the system Python has none of this installed.

---

## Legend

| | what you get |
|---|---|
| 🪟 | **opens a window** you can orbit, zoom and watch in real time |
| 🖱 | **you can click** in the window to place or pick things |
| ⌨️ | **asks you to type** something (coordinates, a target) |
| 📝 | **text only** — prints numbers to the terminal, no window |
| 🖼 | **writes an image file** and exits |
| 🎬 | **records an mp4** — no window while it runs |

⚠️ **`--out` and a window are mutually exclusive everywhere.** Recording forces MuJoCo's offscreen renderer (EGL) and a live window needs GLFW. You cannot watch and record in the same run.

⚠️ **Two different default speeds, on purpose.** The Week 4 interactive tools (`week4_place.py`, `week4_watch.py`) run at **0.4** of the FR5's rated joint speed, because watching a pick at 0.15 is 30 seconds of staring. Every *measurement* tool — `week4_run.py`, `week4_snap.py`, `week4_envelope.py`, and all of Weeks 1–3 — stays at **0.15**, because every number in this repo was taken there and changing it would silently break the comparisons. Pass `--speed 0.15` to an interactive tool to match.

---

## Start here

If you want to see the robot do something interesting, in order:

```bash
# 🪟🖱 place tomatoes wherever you like, then watch it harvest them
./.venv/bin/python simulation/mujoco/week4_place.py --windowed

# 🪟🖱 click anywhere on a board, arm picks that tomato and crates it
./.venv/bin/python simulation/mujoco/week1_mousereach.py

# 🪟 the full planned harvest of a greenhouse row
./.venv/bin/python simulation/mujoco/week2_pick.py

# 🪟 what the camera sees, the plan drawn before it moves, four panels at once
./.venv/bin/python simulation/mujoco/week3_watch.py --view all

# 🪟🖱 both at once: click fruit onto the board, then watch four panels pick them
./.venv/bin/python simulation/mujoco/week4_watch.py
```

---

## Health check

```bash
# 📝 six checks: MuJoCo, EGL render, URDF import, IK solve, CUDA, ROS 2
./.venv/bin/python scripts/phase0_smoketest.py          # expect 6/6
```

Run this after any machine rebuild. If it is not 6/6, nothing below is trustworthy.

---

## Week 1 — the arm, the gripper, reaching a point

### 🪟🖱 `week1_mousereach.py` — click a tomato onto the row and it picks it

```bash
# 🪟🖱 double-click anywhere on the green board — a tomato hangs there and the
#      arm approaches, grips, snaps the stem, carries it and drops it in a crate
./.venv/bin/python simulation/mujoco/week1_mousereach.py

# 📝 same pipeline without a mouse — name points on the board directly
./.venv/bin/python simulation/mujoco/week1_mousereach.py \
    --click 0.0 0.60 --click -0.30 0.40

# 🎬 record it
./.venv/bin/python simulation/mujoco/week1_mousereach.py --headless \
    --click -0.50 0.20 --click 0.0 0.60 --out week1_mousereach.mp4
```

**The board is the arm's working area** — it was measured cell by cell on a 31×24 grid, so every point on it is pickable and a click never lands somewhere the arm has to refuse.

### 🪟 `week1_gripper.py` — pick from a table

```bash
# 🪟 the whole cycle: approach, grip, lift, carry, drop in the crate
./.venv/bin/python simulation/mujoco/week1_gripper.py

# 🪟 gripper only — watch the fingers open and close, no arm motion
./.venv/bin/python simulation/mujoco/week1_gripper.py --wave

# 🎬
./.venv/bin/python simulation/mujoco/week1_gripper.py --headless --out week1_gripper.mp4
```

### 🪟⌨️ `week1_targetreach.py` — type a coordinate, watch it go

```bash
# 🪟⌨️ opens a window and asks you for "x y z" in the terminal
./.venv/bin/python simulation/mujoco/week1_targetreach.py

# 🪟 skip the typing — give targets up front
./.venv/bin/python simulation/mujoco/week1_targetreach.py --target 0.6 -0.2 0.7
./.venv/bin/python simulation/mujoco/week1_targetreach.py --random 5 --seed 1
```

Good for finding the edges of the workspace by hand. Try a point the arm cannot reach and watch what failing looks like.

### 🪟 `week1_reach.py` — IK following a circle, no gripper

```bash
# 🪟 the solver on its own, nothing else in the way
./.venv/bin/python simulation/mujoco/week1_reach.py

# 🎬
./.venv/bin/python simulation/mujoco/week1_reach.py --headless --out week1_reach.mp4
```

⚠️ Its `--out` used to default to `week1.mp4`, which is `week1_gripper.py`'s capture — a headless run here silently overwrote it. Now defaults to `week1_reach.mp4`.

### 🖼 `fr5.py` — the arm on its own

```bash
# 🖼 writes fr5_home.png and exits
./.venv/bin/python simulation/mujoco/fr5.py

# 🖼 writes fr5_gripper.png, with the 2F85 mounted and tool0 at the fingertips
./.venv/bin/python simulation/mujoco/fr5.py --gripper
```

---

## Week 2 — the greenhouse row, planning, the pick

### 🪟 `week2_pick.py` — the main harvest demo

```bash
# 🪟 harvest the row, window stays open at the end
./.venv/bin/python simulation/mujoco/week2_pick.py

# 🪟 flat out, at the FR5's full rated joint speed
./.venv/bin/python simulation/mujoco/week2_pick.py --speed 1.0

# 📝 the milestone: ten jittered trials, scored on collateral damage
./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless

# 📝 the whole row in sequence, nothing reset between fruit
./.venv/bin/python simulation/mujoco/week2_pick.py --sequence --headless

# 📝 what camera error will cost: plan on nominal, execute on jittered
./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless --blind

# 🎬
./.venv/bin/python simulation/mujoco/week2_pick.py --headless --out week2_pick.mp4 --camera row
```

⚠️ **Add `--no-lessons` to any run that is not deliberately about learning.** Without it the run writes to `simulation/lessons.json`, which is shared with the Week 2 milestone — a comparison run quietly changes the baseline it is being compared against.

### 🪟 `legacy_cycle.py` — the unplanned cycle, kept as the baseline to beat

```bash
# 🪟 the old cycle: crates fruit, and strips neighbouring trusses doing it
./.venv/bin/python simulation/mujoco/legacy_cycle.py

# 📝
./.venv/bin/python simulation/mujoco/legacy_cycle.py --trials 10 --headless
```

Run this next to `week2_pick.py` to see what collision-aware planning bought: 2/10 clean → 10/10.

### 📝 The library modules — each prints the numbers it stands on

```bash
# 📝 does the weld read 1.18 N hanging? what force snaps a stem?
./.venv/bin/python simulation/mujoco/plant_row.py

# 📝 plan every fruit in the row and print each route leg by leg
./.venv/bin/python simulation/mujoco/mission.py

# 📝 the old cycle with the black box on — why each tomato was lost
./.venv/bin/python simulation/mujoco/incident.py

# 📝 turn real failures into constraints the planner reads next time
./.venv/bin/python simulation/mujoco/lessons.py

# 🖼 renders greenhouse_row.png / _aisle.png / _wide.png
./.venv/bin/python simulation/mujoco/greenhouse.py
```

---

## Week 3 — the camera, the detector, 3D positions

### 🪟 `week3_watch.py` — the best "what is it thinking" view

```bash
# 🪟 four panels at once: the scene, the deck camera with the plan drawn into
#    it, the eye-in-hand camera with live detections, and a stats panel
./.venv/bin/python simulation/mujoco/week3_watch.py --view all

# 🪟 just what the detector sees — boxes, 3D estimates, error vs ground truth
./.venv/bin/python simulation/mujoco/week3_watch.py --view wrist

# 🪟 a clean shot with no annotations, for showing someone
./.venv/bin/python simulation/mujoco/week3_watch.py --view scene

# 🪟 one fruit, brisk — about 90 seconds
./.venv/bin/python simulation/mujoco/week3_watch.py --view all --fruit t2 --speed 0.4

# 🎬
./.venv/bin/python simulation/mujoco/week3_watch.py --view all --no-window --out week3_watch.mp4
```

Three phases are named on every frame: **SCAN** (the arm sweeps the row looking), **PLAN** (the checked route drawn into the scene, held 2.5 s so you can read it *before* the arm commits), **PICK** (flying it with the waypoints still drawn).

Quit with the **QUIT** button, **q**, **Esc**, or the window's X — all four work and all report whatever picks finished.

### 🪟 `week3_perceive.py` — the instrument behind that view

```bash
# 📝 the geometry gate: project a known fruit into the image, deproject it
#    back, demand sub-millimetre agreement. Run this first.
./.venv/bin/python simulation/mujoco/week3_perceive.py --calib

# 📝 detector recall and false positives per frame
./.venv/bin/python simulation/mujoco/week3_perceive.py --score

# 📝 per-axis position error, mean and p95, against the 40 mm clearance
./.venv/bin/python simulation/mujoco/week3_perceive.py --budget

# 🪟 watch one cycle picked from a camera estimate rather than ground truth
./.venv/bin/python simulation/mujoco/week3_perceive.py --pick --windowed --fruit t2

# 📝 the whole row from perception — the headline number
./.venv/bin/python simulation/mujoco/week3_perceive.py --pick --headless
```

With no flags it runs `--calib --score`.

⚠️ The `--windowed` view is the *scene*, not the sensor. The detection overlay only exists in `week3_watch.py`, in `--out` recordings, and in `--save-frames`.

### 📝 `camera.py` / `detect.py` — the pieces on their own

```bash
# 📝 the deprojection gate from three arm poses, PASS/FAIL in millimetres
./.venv/bin/python simulation/mujoco/camera.py

# 📝 both detectors scored across eight arm poses
./.venv/bin/python simulation/mujoco/detect.py
./.venv/bin/python simulation/mujoco/detect.py --only hsv
./.venv/bin/python simulation/mujoco/detect.py --save /tmp/frames   # 🖼 overlaid stills
```

---

## Week 4 — put fruit anywhere, and the throughput number

### 🪟🖱 `week4_place.py` — **the one to run**

```bash
# 🪟🖱 a window opens with a GREEN BOARD showing exactly where the arm can
#      work. Double-click it to hang a tomato there.
./.venv/bin/python simulation/mujoco/week4_place.py --windowed
```

While the window is open:

| | |
|---|---|
| **double-click the board** | place a tomato there |
| **A** | auto-fill the rest (15 max) — **a different arrangement every press** |
| **C** | clear them all and start over |
| **SPACE** | start harvesting what you placed |
| **Q** or close the window | quit |

Placing and picking happen in **one continuous window** — it does not close and reopen.

**The board shows two zones, and both were measured on this scene** by `week4_envelope.py` — 49 cells, one full pick each:

| | |
|---|---|
| **bright green core** — y ±0.37, z 0.50–0.70 | every probe picked clean (10/10) |
| **dim amber surround** — y ±0.55, z 0.42–0.72 | most did (18/21 = 86%) |
| outside | refused — the arm was measured to fail there |

⚠️ **These numbers were corrected on 2026-08-04 and used to be wrong.** The board previously advertised y ±0.55, z 0.15–0.95, inherited from `week1_mousereach.py` — which measured that against a *different scene*, with the crate out at y=−0.80. Week 2 moved the crate to y=−0.52 and nobody re-measured. Only **21 of 49 cells** in that old region actually picked cleanly, which is why fruit placed high or low kept getting dropped.

Both failing bands have a physical cause in the scene:
- **above z 0.83** the fruit's stem anchor is at or above the support bar the trusses hang from (`SUPPORT_Z` 0.88 − `STEM_LEN` 0.05), so the arm reaches into the bar
- **below z 0.32** the fruit is under the substrate gutter (`GUTTER_Z`), where no real truss ever is

The five fixed trusses this repo has always scored 5/5 on sit at z 0.54–0.72 — inside the good band, which is why they always worked.

⚠️ **There is no minimum spacing any more.** It used to refuse anything within 200 mm of another fruit, and the cap of ten fruit was arithmetic on that lattice rather than anything about the arm. Put them as close as you like now, down to touching (70 mm centre to centre — closer than that and the two spheres interpenetrate, which MuJoCo resolves by firing them apart). Fifteen fit.

A tight placement is **accepted and annotated** rather than refused, because how tight it is decides how much the pick order matters:

| nearest neighbour | what you are told |
|---|---|
| under 120 mm | *"square-on this pair cannot both be planned; the order is what solves it"* |
| 120 – 170 mm | *"tight, expect a fallback route"* |
| over 170 mm | nothing — a neighbour that far away costs the planner nothing |

Those thresholds are swept, not chosen: `deck_cam.py --pairs`.

### 📷 The two cameras

Both are now **visible in the 3D scene** — a D435-shaped housing on a pan-tilt head on a mast behind the arm, and a smaller D405-shaped one on the wrist, built from primitives at the real products' dimensions. Both are `contype=0` with zero mass, so neither changes a single clearance or inertia number (`deck_cam.py` asserts it).

The deck camera surveys the whole row **with the arm parked** and does two jobs the script used to do: it tells the wrist camera where to look, and it decides the pick order.

**The head is articulated**, and the reason is not the obvious one. Rotating a camera about its own optical centre cannot see round anything — every occlusion stays exactly where it was. The lens sits on a **100 mm yoke offset from the pan axis**, so panning *translates* it up to 90 mm, and that is what separates fruit a fixed frame merges: 31/48 → 40/48 on rows packed to 72–100 mm centres, for 2.9 s of head slew and **no arm motion at all**.

⚠️ The head is a **mocap body, not two hinge joints**. Hinges would add two DOFs to `mjModel`, and `mink.Configuration` in `reach.Reacher` is built over the whole model — so the IK solver would see two free joints it could park anywhere, and "where the arm is reaching" would start quietly coupling to "where the camera is pointing". A mocap body has no DOFs: it is commanded, not simulated, which is what a pan-tilt unit with position servos actually is.

```bash
# 📝 the gate: does the mast find the row, and what does the head add?
./.venv/bin/python simulation/mujoco/deck_cam.py
./.venv/bin/python simulation/mujoco/deck_cam.py -n 12   # verifies the head + inert-geometry checks
./.venv/bin/python simulation/mujoco/deck_cam.py -n 15

# 📝 what looking around is worth — sets SCAN_POSES
./.venv/bin/python simulation/mujoco/deck_cam.py --scan
./.venv/bin/python simulation/mujoco/deck_cam.py --eclipse    # can it see round a staged arm?

# 📝 the exact pick-order solver against the hill-climb it replaced
./.venv/bin/python simulation/mujoco/deck_cam.py --optimal

# 📝 the sweeps every constant in the cost model is read off
./.venv/bin/python simulation/mujoco/deck_cam.py --pairs      # what a neighbour costs
./.venv/bin/python simulation/mujoco/deck_cam.py --corridor   # the pull-down wedge
./.venv/bin/python simulation/mujoco/deck_cam.py --mounts     # why the mast is where it is
./.venv/bin/python simulation/mujoco/deck_cam.py --sweep      # the arm's swept volume

# 📝 the deck survey vs sweeping the row with the wrist — what the mast buys
./.venv/bin/python simulation/mujoco/deck_cam.py --vs-sweep --speed 0.15

# 🖼 stills: both cameras, and what each one sees
./.venv/bin/python simulation/mujoco/deck_cam.py --shot
```

⚠️ `--eclipse` answers a question honestly in the negative. The scan recovers a fruit the arm eclipses when staged mid-row (11/12 → 12/12) but **not** when staged high (11/12 either way), so `deck_cam.parked()` stays a hard precondition rather than becoming a preference.

### 🔧 `week4_grip.py` — why the tomato fell out, and the number that stopped it

The gripper used to pluck the tomato and then drop it. The cause was **pad compliance**: a soft contact does not hold a tangential load indefinitely, so the fruit crept out of a closed gripper under its own weight, ~30 mm during the carry, and fell off the edge of the pad. `greenhouse.PAD_SOLREF` went `0.02` → `0.006`, which takes 5/8 crated to 8/8.

⚠️ The old value was chosen to protect the peduncle at the close, back when the grip was a step input. `--close` shows it never did that job either (`0.02` still snapped the stem at 15.86 N) and that `reach.Gripper.ramp` does it at every stiffness. The softening had been buying nothing and costing three picks in eight.

```bash
# 📝 the gate: does the shipped value hold the fruit AND spare the stem?
./.venv/bin/python simulation/mujoco/week4_grip.py

# 📝 both sides of the trade, one full pick per cell
./.venv/bin/python simulation/mujoco/week4_grip.py --solref
./.venv/bin/python simulation/mujoco/week4_grip.py --close    # ramped vs step input

# 📝 the substep trace that found it — watch `gap mm` during `extract`
./.venv/bin/python simulation/mujoco/week4_grip.py --trace
./.venv/bin/python simulation/mujoco/week4_grip.py --trace --seed 71   # the residual

# 🪟 watch one pick at wall-clock speed
./.venv/bin/python simulation/mujoco/week4_grip.py --windowed
```

⚠️ The gate prints **two verdicts on purpose**. The grasp (which `PAD_SOLREF` decides) passes 8/8. The carry does not: 1 of 8 still throws the fruit over 1 m/s, starting in the `turn` leg, and that one is older and separate — the same layout at the old value never reached the crate at all.

### 📝 `week4_order.py` — does the order actually help?

Flies the same clustered layouts twice, once in the deck camera's order and once in placement order, and reports what came out of the crate. **Both arms get the deck camera's positions** — only the order differs.

```bash
./.venv/bin/python simulation/mujoco/week4_order.py                     # contested band
./.venv/bin/python simulation/mujoco/week4_order.py --band blocked      # ties by construction
./.venv/bin/python simulation/mujoco/week4_order.py --band loose        # ties by construction
./.venv/bin/python simulation/mujoco/week4_order.py --spread            # the other control
```

⚠️ **`--band` matters more than any other flag here, and getting it wrong produces a confident null.** Ordering can only earn anything between the distance where a neighbour starts refusing picks and the distance where it stops mattering at all. Outside that window every order ties, for opposite reasons. Rows in any of these bands were **impossible to construct** until the 200 mm rule came out.

⚠️ **It currently returns a tie, and that is the honest state of the feature** — 19 crated vs 18, **12 refused vs 12**, on 32 attempts per arm. The corrected cost model forecast that tie exactly (`0.00` fruit of predicted gain on all four layouts); the model it replaced forecast 3.33 fruit that never arrived. `deck_cam._pair_risk` is symmetric, so it scores both fruit of a close pair as blocked while the sweep it was fitted to says one of them plans fine; the model therefore cannot express the asymmetry the ordering exists to exploit. See the note by `deck_cam.BLOCKED_M`.

```bash
# 🪟 skip the clicking — auto-place 6 and watch it work them
./.venv/bin/python simulation/mujoco/week4_place.py --grid 6 --windowed

# 🪟 THE DEMO: 4 fruit, then 3 more appear mid-run and it re-plans around them
./.venv/bin/python simulation/mujoco/week4_place.py --grid 4 --add-at 2 --windowed

# 📝 headless is the default — no flag needed
./.venv/bin/python simulation/mujoco/week4_place.py --grid 10

# 📝 save an arrangement so it can be replayed exactly
./.venv/bin/python simulation/mujoco/week4_place.py --grid 10 --save layouts/dense10.json
./.venv/bin/python simulation/mujoco/week4_place.py --layout layouts/dense10.json

# 📝 perception in the loop instead of telling the planner where fruit are
./.venv/bin/python simulation/mujoco/week4_place.py --layout layouts/dense10.json --seen

# 📝 log every attempt for later analysis
./.venv/bin/python simulation/mujoco/week4_place.py --grid 8 --log runs/mine.jsonl

# 📝 the old behaviour: no chassis camera, fruit picked in placement order
./.venv/bin/python simulation/mujoco/week4_place.py --grid 10 --no-deck

# 🎬
./.venv/bin/python simulation/mujoco/week4_place.py --grid 10 --seed 3 --out week4_place.mp4

# 📝 match the measurement runs instead of the faster interactive default
./.venv/bin/python simulation/mujoco/week4_place.py --grid 10 --speed 0.15
```

### 🪟🖱 `week4_watch.py` — the four-panel view, with clicking

Week 3's OpenCV window with Week 4's free placement in front of it. **This is the one that shows the most at once.**

```bash
# 🪟🖱 click the green board in the deck panel to place fruit, SPACE to harvest
./.venv/bin/python simulation/mujoco/week4_watch.py

# 🪟 pre-place 12 and skip straight to watching
./.venv/bin/python simulation/mujoco/week4_watch.py --grid 10

# 🪟 plan from the camera instead of being told where fruit are
./.venv/bin/python simulation/mujoco/week4_watch.py --grid 8 --seen

# 🪟 one big panel instead of four
./.venv/bin/python simulation/mujoco/week4_watch.py --view deck
./.venv/bin/python simulation/mujoco/week4_watch.py --view wrist

# 🎬 record the whole session, placement included
./.venv/bin/python simulation/mujoco/week4_watch.py --grid 10 --out week4_watch.mp4
```

```
+----------------------+----------------------+
| deck cam + the plan  | wrist cam, live      |
| CLICK HERE TO PLACE  | detections + error   |
+----------------------+----------------------+
| clean scene shot     | stats / placements   |
+----------------------+----------------------+
```

| | |
|---|---|
| **click the green board** (top-left panel) | place a tomato exactly where the cursor is |
| **A** | auto-fill · **C** clear · **SPACE** harvest · **Q** quit |

**The bottom-right panel is a live readout of what the arm is doing**, not a static summary:

```
PHASE   PICK
  fruit 0/8 attempted
  target  p00
  head  pan    +0  tilt   +0  ----------|----------   <- live
  then    p03 p05 p01
  deck saw 8/8 in 5 head poses
  plan expects 1.24 refusals  (proved optimal)
  this one: clear  (worst neighbour 0.00)

  nearest crop   243 mm (p01)      <- live, every frame
  tool [+0.32 -0.54 +0.52]         <- live
  GRIP  fruit   2.8 mm off the pinch                  <- live
        [......................]
  fruit peak 0.52 m/s while held   <- live
  pads  L  11.4 N  R  11.3 N       <- live

PLAN lane 'direct' clears 71 mm
  checked in 0.82 s
  leg 10/17
   grasp
 > extract
     backing out with the fruit held
   turn
   carry
```

It moves through **SURVEY → SELECT → SCAN → PLAN → PICK → DONE**, and the phase is also burned across the top of every panel. During PLAN it shows the lane the planner chose and the clearance it verified; during PICK it names the current leg, explains in words what that leg is for, and reports the closest fruit the arm is *not* picking — the number the guard is watching.

**Three things to watch for, one per change made in Week 4:**

- **`head pan/tilt` sweeps during SURVEY.** The deck camera is on a pan-tilt head and the survey visits five poses, so the top-left panel *swings* and the mast in the scene panel rotates. The angles are read back out of `mjData`, not from the commanded value. The head walks at its rated slew rather than teleporting, so this takes the ~2.9 s it claims to.
- **`GRIP  fruit N mm off the pinch`, with a gauge, while carrying.** This is the gripper bug made watchable: the tomato was never flung at the pluck, it *crept* out of a closed gripper during the carry and fell off the edge of the pad. Green under 15 mm, amber while slipping, red past 40 mm — where it is gone. It should now sit at 3–20 mm the whole way to the crate.
- **`this one: clear / crowded / BLOCKED`.** Why this fruit is next, and what the cost model expects of it. When it says BLOCKED and the pick is then refused, that is the model being right rather than the robot failing at something it thought it could do.

⚠️ **The pick order is the deck camera's, and the panel says what it expects to lose.** `--no-deck` puts back the old as-placed order, and the panel labels that too rather than implying an optimisation happened.

**Clicking is ray-cast, not guessed.** The pixel is turned into a ray through that camera's pinhole — the same `camera.pixel_ray` the Week 3 deprojection gate is built on — and intersected with the board plane. Verified exact to **0.000 mm** at four points from two different camera angles, so the tomato lands under the cursor regardless of which camera you are looking through.

Refusals appear on the stats panel with the reason — off the board, or two tomatoes in the same space. A *tight but legal* placement is accepted and annotated in blue instead, because that is a pick-order problem rather than a placement error.

⚠️ The top-left panel is now the **real** deck camera, not the `row` cinematic framing wearing that label. It is the sensor on the mast you can see in the scene panel, so the survey shown is the survey that was used. `--deck-camera row` puts the old framing back.

### 🪟 `carrytrace.py` — watch a tomato get thrown

```bash
# 🪟 the bug 40 ejection, live, at real speed — it leaves the pads sideways
#    mid-carry at 2.3 m/s rather than being dropped
./.venv/bin/python simulation/mujoco/carrytrace.py --windowed

# 📝 the same run with the force trace printed
./.venv/bin/python simulation/mujoco/carrytrace.py

# 📝 the control — the same fruit from ground truth, which crates fine
./.venv/bin/python simulation/mujoco/carrytrace.py --truth

# 📝 the four hypothesis levers, all of which turned out not to fix it
./.venv/bin/python simulation/mujoco/carrytrace.py --hold 80
./.venv/bin/python simulation/mujoco/carrytrace.py --rolling 0.1
./.venv/bin/python simulation/mujoco/carrytrace.py --carry 0.05
./.venv/bin/python simulation/mujoco/carrytrace.py --timestep 0.001
```

### 📝 `week4_envelope.py` — where can the arm actually pick?

```bash
# 📝 49 cells, one full pick each, prints a map. ~26 min
./.venv/bin/python simulation/mujoco/week4_envelope.py

# 📝 finer grid
./.venv/bin/python simulation/mujoco/week4_envelope.py --ny 9 --nz 9
```

Re-run this **whenever the crate, the gripper, the standoff gaps or the row position move** — all four change the answer, and the last time one did (the crate moving from y=−0.80 to −0.52) nobody re-measured and the placement board was wrong for two weeks. Output looks like:

```
        -0.55  -0.37  -0.18  +0.00  +0.18  +0.37  +0.55
z 0.68      O      O      O      O      O      O      g
z 0.55      O      O      O      O      O      O      O
z 0.42      O      O      d      O      O      g      O
z 0.28      X      g      g      u      u      X      g
```

`O` clean · `d` dropped · `X` ejected · `g` grasp failed · `u` unreachable · `a` guard abort · `r` refused.

### 📝 The measurement runs — long, no window

```bash
# 📝 the throughput campaign: ~58 picks across four crop densities, ~30 min
./.venv/bin/python simulation/mujoco/week4_run.py --out runs/campaign.jsonl

# 📝 the same campaign with the chassis camera surveying and choosing the order
./.venv/bin/python simulation/mujoco/week4_run.py --deck --out runs/campaign_deck.jsonl

# 📝 how much kg/hr moves when SNAP_N moves — it barely does
./.venv/bin/python simulation/mujoco/week4_snap.py --n 8

# 📝 re-read any log later without re-running the physics
./.venv/bin/python simulation/mujoco/picklog.py runs/campaign.jsonl

# 📝 the failure taxonomy, explained, with a worked example
./.venv/bin/python simulation/mujoco/outcomes.py
```

⚠️ **`week4_run.py` appends.** Point `--out` at a fresh file or you mix two campaigns in one log. This has already happened once.

⚠️ **`--deck` is a different campaign, not more samples of the same one.** It changes the pick order, which is a methodology change — the 57-attempt figure in the README was taken without it, on rows generated under the old 200 mm spacing minimum. Log it separately and compare the two, rather than pooling them.

---

## Week 5 — the whole house: scout it, plan it, harvest it

Everything above works one row from a base bolted to the floor. `simulation/mujoco/farm/` is a **separate package** that asks the next question: given four rows and a machine that has to drive, what does a shift look like?

⚠️ **It imports freely from Weeks 1–4 and changes none of it.** Those numbers — 46% clean, 31.3 s cycle, 40 mm clearance — were all measured in a one-row scene with a fixed base. `farm.house` is a new scene *next to* `greenhouse.py`, not a replacement for it.

### 🪟🪟 `two_arm_farm.py` — **the one to run.** One full row, two arms, two windows

```bash
# 🪟🪟 the whole cycle for ONE FULL ROW: map -> plan -> travel -> pick -> crate.
#      TWO windows open. A new random house each launch; the seed is printed.
./.venv/bin/python simulation/mujoco/two_arm_farm.py

# 🪟🪟 the same house again
./.venv/bin/python simulation/mujoco/two_arm_farm.py --seed 7

# 🪟🪟 skip the mapping pass and route the real crop — isolates everything
#      downstream of perception
./.venv/bin/python simulation/mujoco/two_arm_farm.py --truth

# 🪟🪟 short: stop after three trolley stops
./.venv/bin/python simulation/mujoco/two_arm_farm.py --truth --stops 3

# 🖼 render every panel once and exit, and print where each camera is aimed
./.venv/bin/python simulation/mujoco/two_arm_farm.py --shot

# 📝 no windows at all
./.venv/bin/python simulation/mujoco/two_arm_farm.py --headless

# 🎬 record both windows: <out>_sensors.mp4 and <out>_mission.mp4
./.venv/bin/python simulation/mujoco/two_arm_farm.py --out twoarm

# 🪟🪟 cheaper live view: render the panels at half size and scale up.
#     The window layout and the mp4 dimensions do not change, and the HSV
#     detector keeps its own resolution — this is a display cost knob only.
./.venv/bin/python simulation/mujoco/two_arm_farm.py --panel-scale 0.5
```

`python 2armfarm.py` runs the same thing — it is a shim that execs the real module. ⚠️ A module name starting with a digit is not a legal Python identifier, so `2armfarm.py` can be *run* but never *imported*; all the code lives in `two_arm_farm.py` and nothing but the shim refers to the digit name.

**What you actually see: two OpenCV windows.**

```
WINDOW 1 — "vinea — SENSORS"        WINDOW 2 — "vinea — MISSION"
+--------------+--------------+     +----------------+----------------+
| arm1 wrist   | arm2 wrist   |     | THE MAP        | DOWN THE AISLE |
| HSV ripeness | HSV ripeness |     | what it found  | trolley + both |
| boxes, live  | boxes, live  |     | and what it    | arms, slow-    |
+--------------+--------------+     | did about it   | tracking       |
| arm1 deck    | arm2 deck    |     +----------------+----------------+
| own row, own | own row, own |     | PIPELINE       | PER-ARM STATS  |
| pan/tilt     | pan/tilt     |     | what each arm  | pick times and |
+--------------+--------------+     | is doing NOW   | running mean   |
                                    +----------------+----------------+
```

**Window 1 (SENSORS)** — four live camera panels, each captioned with which arm and which camera. The two wrist cams carry the HSV ripeness overlay: a box round every fruit found, coloured by stage, labelled **RIPE** or **unripe**, with the counts and the classifier's name in the footer. The two deck cams are **one per arm, independently articulated** — each reports its own live pan angle, and they hold different angles at the same time because each scans only its own row.

**Window 2 (MISSION)** — the map top-left (all four rows, every mapped fruit as a dot coloured by believed ripeness, ripe ones ringed, **x** picked, **X** refused, **◇** lost, dim ring skipped, a green circle on whatever an arm is currently targeting, ground truth as a small grey dot above each so a wrong dot reads as wrong). Top-right the aisle shot, tracking the trolley with a lag so the machine stays in frame while visibly travelling. Bottom-left the live pipeline text, one block per arm — phase, current executor leg, deck pan, live guard clearance. Bottom-right per-arm stats: crated / refused / missed, last pick time, running mean, and the last nine pick times.

⚠️ **Both arms work at the same time, and the window says so with a number.** They were serialised, and it was forced rather than chosen: `farm/armframe.py` made Weeks 1–4's world-frame planner work for a moving arm by *rebinding `mission`'s module globals* (`PARK`, `STAGE_X`, `BIN_POS`, `ROW_X`, `INTO_ROW`) to the current arm's frame, so two arms mid-mission needed two conflicting sets of them in one interpreter; and `week2_pick.execute` owned its own loop, so two of them could not interleave. Both are gone — `mission.ArmFrame` carries the five constants as a value on the plan, and `week2_pick.MissionRun` is a generator that stops each control cycle with its setpoints written and physics pending. `farm.duo.Machine` commands every arm and then steps the plant **once**.

⚠️ **Arm-vs-arm clearance is mandatory now, and it is doing real work.** While the arms were serialised the idle one was stationary, so checking against it was checking against a fact. With both flying into a working volume that overlaps by 1.44 m it is the only thing between them: `work()` refuses to fly an arm whose `others` set is empty, and both the planner's preview and the runtime `Guard` carry it. The guard is the one that holds the line — `ArmObstacles` reports where the other arm is *now*, which is a measurement taken every control cycle for the guard and a one-snapshot prediction for the planner.

⚠️ **The two arms interlock on one token, for the returning half of the cycle only.** Both `PARK` postures fold the elbow across the aisle, so an arm sitting at PARK is in the shared middle of the deck. Measured: both parked +110 mm, both reaching into their own rows +188 mm, either stowed +318 mm, one moving while the other sits at PARK **−40 mm**. So a non-flying arm is folded (`duo.STOW`), and `duo.DeckCentre` lets only one arm at a time run the legs that swing it through the middle — unfolding and setting out, and the swing round to the crate and back to park. Approach, insert, grip, pull and extract run with both arms moving.

⚠️ **The pipeline text is read out of the running mission, not scripted.** `farm.duo.ArmState` is written where the work happens, and the current leg is read live off `week2_pick.MissionRun.leg`, which the executor sets as it flies. The panel also shows, per arm: the target with its ripeness class, the map's estimated position and the error against ground truth; whether the planner accepted or refused and how many candidates it tried; the refusal's breach in mm against the 40 mm budget; the guard's live minimum clearance and the leg it is on; the distance any abort happened at; what the arm is waiting on, including waiting for the other arm; and the running picks / mean / refusals / misses. Everything the panel wanted that the mission objects did not already expose was **added to `ArmState`**, not reconstructed in the viewer.

#### Where the time actually goes

⚠️ **The docstring used to say the camera cadences were the performance story. They are not.** Profiled headless, with the renderer switched off entirely — one two-armed pick, 123.5 s of work:

| cost | | |
|---|---:|---|
| `plant_row.weld_force` | 48.3 s | 39% — one `efc` scan per fruit per physics substep |
| `daqp` IK solve | 44.1 s | 36% — a QP over **317 DOF** to move a 6-DOF arm |
| `mink` task objective | 7.4 s | 6% |
| `mj_step` | 5.7 s | 5% |
| guard clearance | 2.5 s | 2% |
| rendering | 0.0 s | 0% — there was none |

The panels are ~5% of a harvest control cycle. So the win was taken where the time was: `plant_row.weld_forces` reads the whole row's weld forces in **one** `efc` pass instead of one per fruit — 3.690 ms → 0.125 ms per call, **29.5×**, verified bit-identical over 300 steps × 48 fruit. Same command, same seed, per-phase panel rate:

| phase | before | after |
|---|---:|---:|
| pick | 2.2 fps | **3.5 fps** |
| map | 8.7 fps | 8.5 fps |
| travel | 19.8 fps | 20.1 fps |

The pick phase is where the run spends its time and it is 59% faster. The *overall* mean is not comparable between runs that harvested different numbers of fruit; read the per-phase rates, which is why they are printed.

⚠️ **288 of those 317 IK DOF are tomato free joints**, and a crop-free model solves the same IK 45× faster (14.14 ms → 0.31 ms). That is recorded and **not** fixed — it needs a reduced model for the solver, which changes IK answers, so it wants its own measurement pass rather than a drive-by. Bug Log 67.

⚠️ **EGL was already on the NVIDIA card.** `GL_RENDERER` reports `NVIDIA RTX 1000 Ada Generation Laptop GPU/PCIe/SSE2`, driver 595.84 — `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` sorts ahead of `50_mesa.json`, so nothing was landing on the AMD iGPU and there was no offload to force.

### 🪟 `farm/watch.py` — six panels, one arm, including the map

```bash
# 🪟 the whole shift: scout the house, plan the route, harvest into the crate
./.venv/bin/python simulation/mujoco/farm/watch.py

# 🪟 skip the scouting pass and route the real crop — isolates routing from perception
./.venv/bin/python simulation/mujoco/farm/watch.py --truth

# 🪟 short: stop after two trolley stops
./.venv/bin/python simulation/mujoco/farm/watch.py --truth --stops 2

# 🎬 record it
./.venv/bin/python simulation/mujoco/farm/watch.py --out farm_watch.mp4
```

```
+-------------------+-------------------+-------------------+
| scout cam         | down the aisle    | the house         |
| what it maps      |                   |                   |
+-------------------+-------------------+-------------------+
| THE MAP           | the robot's shift | wrist cam         |
| top-down, live    | what it thinks    | what it picks     |
+-------------------+-------------------+-------------------+
```

**The map panel is the new thing.** All four rows top-down, every fruit as a dot coloured by the ripeness the robot believes it has, ripe ones ringed, route stops numbered, the trolley drawn to scale with a line to whatever it is currently reaching for, and picked fruit crossed out as the row empties.

⚠️ **It is drawn from `HouseMap`, not from the simulator.** Where the robot is wrong, the map is wrong — which is the only way to watch a perception failure happen rather than read about it afterwards. Ground truth is drawn faintly underneath so the gap is visible.

### 📝 The pieces, each runnable on its own

```bash
# 🪟 the four-row house: 1.60 m pitch, 51 mm heating pipes as the rails,
#    glazing bars every 1.10 m, side walls on a concrete upstand, screen wires
./.venv/bin/python simulation/mujoco/farm/house.py
./.venv/bin/python simulation/mujoco/farm/house.py --plan     # 📝 the layout arithmetic
./.venv/bin/python simulation/mujoco/farm/house.py --shot     # 🖼

# 🪟 the trolley: drives the length of the house on the pipe rail
./.venv/bin/python simulation/mujoco/farm/trolley.py --drive
./.venv/bin/python simulation/mujoco/farm/trolley.py --reach  # 📝 the gate, arm A
./.venv/bin/python simulation/mujoco/farm/trolley.py --shot --arms 2

# 📝 BOTH arms, each against its own row. Prints two tables and one verdict.
#    This is the check that says a second arm is really mounted rather than
#    merely drawn: arm B is bolted round 180° and works the row on the other
#    side of the aisle at the same 600 mm standoff.
./.venv/bin/python simulation/mujoco/farm/trolley.py --reach --arms 2

# 🪟 a random crop — a different house every time you open it, and it now
#    prints the seed it drew so you can open the same one again
./.venv/bin/python simulation/mujoco/farm/crop.py
./.venv/bin/python simulation/mujoco/farm/crop.py --stats     # 📝
./.venv/bin/python simulation/mujoco/farm/crop.py --stats --seed 355527551

# 🪟 the mapping pass — the head turns to face each row in turn as it drives
./.venv/bin/python simulation/mujoco/farm/scout.py --windowed
./.venv/bin/python simulation/mujoco/farm/scout.py --recall   # 📝 per-stage recall
./.venv/bin/python simulation/mujoco/farm/scout.py --shot     # 🖼 boxes + bands

# 📝 does turning the head buy anything? The same house scouted twice —
#    once with the head locked at arm A's row, once turning. Prints the A/B.
./.venv/bin/python simulation/mujoco/farm/scout.py --articulate

# 📝 the route over a map
./.venv/bin/python simulation/mujoco/farm/route.py --truth --compare

# 📝 a whole shift, headless, with the full report
./.venv/bin/python simulation/mujoco/farm/run.py
./.venv/bin/python simulation/mujoco/farm/run.py --truth --stops 3
```

### 🪟 `farm/eyes.py` — both arms' wrist cams, with the ripeness call drawn on

```bash
# 🪟 two wrist cams side by side with live ripe/unripe boxes, deck cam below
./.venv/bin/python simulation/mujoco/farm/eyes.py

# 🪟 arm A only — arm B's panel says "not fitted" rather than going blank
./.venv/bin/python simulation/mujoco/farm/eyes.py --arms 1

# 🎬 record it
./.venv/bin/python simulation/mujoco/farm/eyes.py --out farm_eyes.mp4
```

### 🖼 `farm/decks.py` — one deck camera per arm, and the proof they are independent

```bash
# 🖼 aim the two heads apart, render both, print the angle between them.
#    Writes twoarm_deck_a.png and twoarm_deck_b.png. PASS if > 30 deg.
./.venv/bin/python simulation/mujoco/farm/decks.py --split

# 📝 drive the whole aisle with both heads sweeping, and log the
#    camera-to-trolley offset every control cycle. PASS if it never moves.
./.venv/bin/python simulation/mujoco/farm/decks.py --offset
```

**`--offset` is the check that the two deck-camera bugs are gone**, and it is a measurement rather than a look at the render. Both bugs were one mechanism: the head was a worldbody **mocap body** driven from Python while its own mast was a geom on the **trolley body**, so one rigid assembly moved by two transports. `Drive.drive_to` steps physics for a whole traverse and never called `follow`, so the camera sat frozen for the drive and then jumped the full distance (the *teleport*); and nothing calls `follow` at all during the harvest, so the mast rode away and left the camera behind (the *desync*).

The head is now a child body of the trolley — `deck_yaw_<tag>` → `deck_head_<tag>`, pan and tilt hinges, position-servoed — so position is inherited through the model tree and only articulation is driven from Python. There is no second mechanism and so nothing to desync. On seed 7:

```
  driving -3.60 m to +3.60 m, both heads articulating
  2084 control cycles, trolley odometer 7.20 m

  arm            pan swept   pivot offset dev    lens offset dev
  A       -30.0.. +30.0 deg           0.000 mm            50.0 mm
  B       -30.0.. +30.0 deg           0.000 mm            50.2 mm
```

⚠️ **The pivot offset is the invariant; the lens offset is not.** The head's own axis must never move relative to the trolley — that is what "bolted to it" means, and it is exactly 0.000 mm. The lens *should* move, by up to 50 mm, because that is the pan turning it on a 90 mm yoke. Both are printed so the two can never be mistaken for each other.

**Prints two lines and writes two stills.** `farm/scout.py`'s head is *one* pan-tilt unit on the aisle centreline that turns 180° at every stop to serve both rows. These are **two heads, one per arm**, each over its own arm's mount plate and each scanning only its own row — so they can look different ways at once, which is what `--split` measures: **103.4°** between the two lines of sight on seed 7, arm A on r1 and arm B on r0.

⚠️ Two heads that always mirror each other are one head with extra geometry. That is what the gate exists to rule out.

Each head also sits **0.60 m** from its own row against the shared head's 0.70 m — the Week 1–4 standoff exactly — and never crosses the aisle, so the shared head's 22.5 s of cross-aisle slew per pass disappears.

```
+------------------------+------------------------+
| arm A wrist cam        | arm B wrist cam        |
| RIPE / unripe boxes    | RIPE / unripe boxes    |
+------------------------+------------------------+
| deck cam — the mapping pass, head pan live      |
+-------------------------------------------------+
```

**What you actually see:** a box round every fruit the colour classifier finds, coloured by stage (green / breaker / turning / red) with the word **RIPE** or **unripe** on it — which is `crop.STAGES`'s own `pick` flag, so it is exactly the decision the harvest acts on. A turning fruit is boxed amber and labelled unripe, which is what a grower does with it. The footer counts both and names the classifier. The deck panel reports the head's live pan angle and which row it is facing.

⚠️ **It is the Week 3 colour control, not a model.** Two OpenCV `inRange` bands over hue plus a circularity filter — the same code the mapping pass scores itself with. Nothing was trained. An overlay running a *better* detector than the robot uses would show a machine that sees clearly and picks badly, and the gap would read as a manipulation bug.

### 📝 `farm/misses.py` — why is a ripe tomato not in the crate?

```bash
# 📝 five shifts, every ripe fruit attributed to one bucket, dominant one named
./.venv/bin/python simulation/mujoco/farm/misses.py

# 📝 more shifts, tighter numbers
./.venv/bin/python simulation/mujoco/farm/misses.py --shifts 10

# 📝 with a perfect map — isolates everything downstream of perception
./.venv/bin/python simulation/mujoco/farm/misses.py --truth
```

**Prints a table, no window.** It starts from every ripe fruit that was really in the house and gives each one a reason it is or is not crated, then names the dominant bucket and tells you not to tune the others.

⚠️ **Counting attempts measures the robot's aim and calls it its yield.** `run.py` logs one row per fruit it *tried* to pick, so a shift can report 5/5 clean while ripe fruit stand untouched two metres away. This adds the two buckets an attempt log cannot see — `not_mapped` (the scout never saw it) and `not_routed` (mapped, but no stop could reach it) — ahead of the six `outcomes.classify` already names.

### What it does, and what it does not

**Five shifts, 56-fruit houses, aisle a0, one arm — 22 of the 25 ripe fruit on the worked row crated, 88%.** Measured with `farm/misses.py`, which starts from every ripe fruit that was really there rather than from the attempts the robot chose to make:

| bucket | n | share | |
|---|---|---|---|
| clean | 22 | 88.0% | in the crate |
| misbanded | 2 | 8.0% | seen, but the colour classifier called it unripe |
| not_mapped | 1 | 4.0% | the scouting pass never saw it |

**Nothing fails at the arm.** Zero refusals, zero guard aborts, zero grasp failures, zero drops and zero ejections across 22 attempts. Every remaining miss is perception, upstream of any motion — which is a different machine from the one the Week 5 notes described, where a refusal and two missed detections were the story.

⚠️ **Three misses is a thin basis for ranking two buckets**, and 2-against-1 is not a ranking. What five shifts support is the shape, not the order.

⚠️ **A shift is not bit-reproducible.** The same seed gave 2/4 and then 3/4 on the same shift with unchanged harvest code. `--seed` reproduces the *crop layout* — verified directly — but MuJoCo running multi-threaded does not reproduce contact ordering, so a single shift's count is a sample and not a fixed result.

**Ripeness is measured per stage, and the total is the wrong number to quote.** Hue separation was measured before the detector was designed:

| | hue | separated from the canopy? |
|---|---|---|
| red | 2 | yes — **98.7% correctly banded** (75/78 → 77/78, see below) |
| turning | 14 | yes — **98.2%** (52/57 → 56/57) |
| breaker | 29 | yes — 100% |
| green | 49 | **no** — a stem is 55, a leaf is 62. Banded right when found, but **most are never found at all** |

⚠️ **`stage_of` used to average the whole bounding box while its docstring claimed it averaged "its own pixels".** A tomato's projection is round and its box is square, so ~21% of what it sampled was corner, and the saturation floor does not reject a *breaker* tomato one truss behind — which is exactly what pulls a red fruit's mean hue up into `turning`, so the map calls it unripe and the harvest walks past it. Sampling the **inscribed disc** fixes it. No band moved; the function now reads the pixels it always claimed to. Eight houses, 156 matched fruit: red 96.2% → **98.7%**, turning 91.2% → **98.2%**, green and breaker 100% either way.

⚠️ That costs the harvest nothing, because only red is picked. It would cost a **scouting yield forecast** a great deal, and that is Vinea's second module — so green recall is the number to quote there, not the average.

⚠️ ~~The scout camera looks one way, so arm B's row maps 0/14.~~ **Fixed — the head articulates.** It is a pan-tilt unit now and faces each arm's row in turn at every stop. Measured on one house, twice, with `scout.py --articulate`:

| head | r1 (arm A) | r0 (arm B) | phantom | slew |
|---|---|---|---|---|
| locked at r1 (what shipped) | 12/14 | **0/14** | 1 | 0.0 s |
| turning | 12/14 | **8/14** | 1 | 22.5 s |

It costs 22.5 s a pass and is the difference between a second arm having a map and having nothing to pick. The pan axis sits on the aisle centreline so both rows are 0.70 m away and neither arm gets the better camera.

⚠️ ~~**Nothing checks arm against arm.**~~ **Fixed — `mission.ArmObstacles`.** Each `Guard`, `ClearanceModel` and `Planner` now takes an `others=` tuple of arm prefixes and puts those arms in the obstacle set, at `ARM_CLEARANCE` = 40 mm (the crop's budget, not structure's 15 mm — another 22 kg arm is not a thing you may scuff). Empty by default, so Weeks 1–4 are untouched.

The check found a real clash the moment it existed: **the two arms were parked 83 mm inside each other**, forearm through forearm, on every two-armed scene ever built. `farm/duo.py` also serialises the arms and stows the idle one. See the Bug Log, entries 43, 54, 55, 56 and 57.

⚠️ **Every launch is a different house, and it tells you which one.** `crop.spawn` always randomised; what is new is that the seed is drawn outside and printed before anything is built, so a layout that breaks the planner is reproducible with `--seed`. There is deliberately **no 200 mm minimum fruit spacing** — Week 4 removed it and `week4_place.py` records why: a close pair is a pick-order problem to be solved, not a layout to be avoided. What *is* enforced is `Z_LOCAL` = the measured `MARGINAL_Z` band and a 75 mm minimum separation against `TOUCHING` at 70 mm, so nothing spawns unpickable or interpenetrating.

⚠️ **Lifting the shadows made detection worse, and it was measured rather than assumed.** The house is lit by six lamps on a grid now instead of four in a line down the middle, which is why the outer rows are no longer noticeably darker. The obvious companion change — raising the ambient term, since a Venlo roof genuinely does scatter light — was tried and **reverted**: the ripeness bands have a minimum *saturation* of 95–100 as well as a minimum value, and ambient light washes saturation out everywhere at once. Three houses, both rows, 168 fruit:

| lighting | found | phantoms |
|---|---|---|
| line of 4 + 2 side fills (what shipped) | 101/168 | 7 |
| grid of 6 + ambient 0.32 | **91/168** | **14** |
| grid of 6, ambient untouched | **104/168** | **5** |
| grid of 6 + 2 side fills | 102/168 | 6 |
| grid of 6 + the glazing bars and side walls | **108/168** | **3** |

The last row is the one that ships, and the extra gain is not from the lighting: the **side walls** put a solid background behind the outer rows where open sky used to show through the gaps, and bright sky is where phantom detections came from. Enclosing the house for looks turned out to be worth 4 fruit and 2 phantoms.

⚠️ **`farm/armframe.py` rebinds other modules' globals.** `mission.py` is written in absolute world coordinates (`PARK`, `STAGE_X`, `BIN_POS`, `ROW_X`) which breaks the moment the arm rides a moving base. The adapter rebinds 8 bindings across 4 modules inside a context manager and asserts it restores them (`armframe.check`). It is the smaller, more reversible change than threading a frame through six classes — but it is a real cost, and the file says so.

---

## Things that will catch you

**No window appears.** Check for `--out` in the command — recording and a live window cannot coexist. `week4_place.py` and `carrytrace.py` also need `--windowed` explicitly; every Week 1 and Week 2 demo opens a window by default and needs `--headless` to suppress it.

**A window opens but the arm never moves.** It is probably planning. `week4_place.py` with a full pool checks the route against 24 fruit and takes ~1.5 s per pick before anything moves.

**Wayland warnings** — `libdecor`, `GLFWError: window position`. Cosmetic, about decorations and placement. The window works.

**`EGLError` on exit, or a segfault after the run finishes.** Both are upstream MuJoCo teardown issues on this machine, both happen after the work is done, and neither loses anything. Do not go debugging them.

**The run is slower than real time.** Two MuJoCo processes on one machine halve each other's speed — check nothing else is running. In `week4_watch.py` the four-panel composite is the expensive part; it renders at `--fps` (default 30) rather than at every control cycle, but dropping to `--fps 15` buys real speed if the machine is loaded.

**Fruit appear as odd shapes floating behind the arm.** Fixed — the 24-truss reserve pool used to be drawn while parked. If you see it again, `Crop._show` is not being called.

**A capture got overwritten.** Captures are named after the script that made them, and `*.mp4` is gitignored — so there is no copy. Check what `--out` defaults to before running anything headless.

**A `farm/` run says the arm cannot reach, on a pick the planner just verified.** Almost certainly the base is not pinned. `mink` builds its configuration over the *whole* model, and the trolley's slide joint looks like a free DOF to it — so it plans base motion the chassis never makes, and every leg lands short by roughly the same amount in the direction the base would have moved. `armframe.pin_base(reacher)` fixes it, and has to be re-applied after any `set_speed`, which rebuilds the velocity limits from the arm's joints alone.

**A `farm/` pick carries the fruit to the crate, drops it in, and still reports `in_bin: False`.** Something is reading a stale `BIN_POS`. `from mission import BIN_POS` copies the reference at import time, so rebinding `mission.BIN_POS` does not reach it — `week2_pick` holds its own. Add the module to `armframe._HOLDERS`.

**The `farm/` house renders dark, or the aisle ends in open sky.** `farm.house` has its own `_lighting` and `_gables`; `greenhouse._lighting` puts two lamps over a 3 m scene and a 9×9 m floor, which leaves an 8 m bay black at the far end. Cameras must also sit *inside* the gables — one placed beyond the end wall shoots through a glazing bar.

---

## What a cycle costs

Useful when deciding whether to sit and watch:

| | |
|---|---|
| one pick at 0.15 (measurement tools) | ~28–31 s of simulated time |
| one pick at 0.4 (interactive tools) | ~19–21 s |
| a 5-fruit row, interactive | ~2 min |
| 10 fruit, interactive | ~3.5 min |
| the full campaign | ~30 min |
| `week4_snap.py --n 8` | ~20 min |
| `week4_envelope.py` (49 cells) | ~26 min |
| `deck_cam.py` (the gate) | ~2 min — renders only, no picks |
| `deck_cam.py --scan` | ~12 min — 6 patterns × 6 layouts × up to 9 renders |
| `deck_cam.py --optimal` | ~2 min — no physics at all, just the solver |
| `deck_cam.py --pairs` / `--corridor` | ~15 / ~25 min — planning only, no flying |
| `deck_cam.py --vs-sweep --speed 0.15` | ~4 min |
| `week4_grip.py` (the gate) | ~15 min — 8 full picks |
| `week4_grip.py --solref` | ~90 min — 6 settings × 8 picks |
| `week4_grip.py --close` | ~6 min — no pull, no carry |
| `week4_order.py` (4 layouts, both orders) | **~2 h** — 64 picks, and every survey is a 5-pose scan |

⚠️ `week4_order.py` got slower when the deck head arrived: each survey is now five renders instead of one, and a re-survey fires on every crop change. Budget accordingly, or drop `--layouts`.

**Week 5, the whole house** — measured, not estimated:

| | |
|---|---|
| `farm/house.py --plan` | instant — arithmetic only |
| `farm/crop.py --stats` | ~1 s |
| `farm/house.py --shot` | ~6 s |
| `farm/route.py --truth --compare` | instant — no physics, no rendering |
| `farm/trolley.py --reach` | ~12 s — 10 routes planned, none flown |
| `farm/scout.py --recall` | ~13 s — 15 frames and 7 m of driving |
| `farm/run.py --truth --stops 3` | minutes — 5 picks is ~90 s of *simulated* time and this scene runs well under real time |
| `farm/watch.py` | the same plus six panels at `--fps` |

⚠️ **The house is a much bigger scene than the Week 1–4 row** — ~5,300 geoms and 407 DOF against ~800 and 44 — so physics runs a good deal slower than real time, and a full shift is dominated by the picking rather than by the driving or the scouting. Use `--stops` to cut a run short while you are iterating.

Watching runs at wall-clock speed. Headless runs faster than real time, but only when nothing else is competing for the machine.

---

[README](README.md) · [Week 4 instructions](../k7_ideaverse_2.0/03%20Projects/Vinea/08%20Technical/) · captures are named after the script that produced them
