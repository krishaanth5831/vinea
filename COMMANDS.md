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

### 🪟 `farm/watch.py` — **the one to run.** Six panels, including the map

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
# 🪟 the four-row house: 1.60 m pitch, 51 mm heating pipes as the rails
./.venv/bin/python simulation/mujoco/farm/house.py
./.venv/bin/python simulation/mujoco/farm/house.py --plan     # 📝 the layout arithmetic
./.venv/bin/python simulation/mujoco/farm/house.py --shot     # 🖼

# 🪟 the trolley: drives the length of the house on the pipe rail
./.venv/bin/python simulation/mujoco/farm/trolley.py --drive
./.venv/bin/python simulation/mujoco/farm/trolley.py --reach  # 📝 the gate
./.venv/bin/python simulation/mujoco/farm/trolley.py --shot --arms 2

# 🪟 a random crop — different every time you open it
./.venv/bin/python simulation/mujoco/farm/crop.py
./.venv/bin/python simulation/mujoco/farm/crop.py --stats     # 📝

# 🪟 the mapping pass
./.venv/bin/python simulation/mujoco/farm/scout.py --windowed
./.venv/bin/python simulation/mujoco/farm/scout.py --recall   # 📝 per-stage recall
./.venv/bin/python simulation/mujoco/farm/scout.py --shot     # 🖼 boxes + bands

# 📝 the route over a map
./.venv/bin/python simulation/mujoco/farm/route.py --truth --compare

# 📝 a whole shift, headless, with the full report
./.venv/bin/python simulation/mujoco/farm/run.py
./.venv/bin/python simulation/mujoco/farm/run.py --truth --stops 3
```

### What it does, and what it does not

**A shift on a 48-fruit house**, aisle a0, one arm: 4 crated of the 5 ripe fruit on the worked row, 1 refused (a real neighbour 30 mm from the pad), 2 never detected. 20 s scouting, 33 s driving, 76 s picking.

**Ripeness is measured per stage, and the total is the wrong number to quote.** Hue separation was measured before the detector was designed:

| | hue | separated from the canopy? |
|---|---|---|
| red | 2 | yes — **100% recall, 100% correctly banded** |
| turning | 14 | yes |
| breaker | 29 | yes |
| green | 49 | **no** — a stem is 55, a leaf is 62. **33% recall** |

⚠️ That costs the harvest nothing, because only red is picked. It would cost a **scouting yield forecast** a great deal, and that is Vinea's second module — so green recall is the number to quote there, not the average.

⚠️ **The scout camera looks one way, so arm B's row maps 0/14.** A two-armed trolley needs a second head. This is a known gap, not an oversight; `--arms 2` fits the second arm and its plate is drawn either way.

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
