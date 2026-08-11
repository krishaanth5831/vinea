# Vinea

Modular autonomous robot for Dutch commercial greenhouses. Harvesting first, scouting as a second module on the same base. Runs in existing row infrastructure — no greenhouse rebuild.

> **Status: simulation-first.** No hardware budget yet, on purpose — the technical bet gets proven in software before a cent is spent on parts. A Fairino FR5 finds tomatoes with a camera, plans a route that clears the crop, and harvests them in a MuJoCo greenhouse — **without disturbing the fruit it is not picking**. Weeks 1–4 are built and Week 5 (a whole house on a pipe-rail trolley, two arms) is running ahead of a 4-week sprint that ends 2026-08-27. The sprint's deliverable is a recorded autonomous pick cycle with a success rate and a defensible kg/hr; **the kg/hr is the piece still outstanding.**

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

The green board is not decoration — it is the reachable band re-measured cell by cell in `week4_envelope.py`, one full pick per cell, so **anywhere you can click is somewhere the arm can work**.

**There is no minimum spacing.** Put them as close together as you like, down to touching.

The rule used to refuse anything within 200 mm, on the grounds that below it *"the stems load one another past the detach threshold and a truss snaps itself before the arm has moved"*. **That turned out not to be true.** Two hanging fruit, settled three seconds, peduncle force at the peak:

| centres | 70 mm | 85 mm | 100 mm | 140 mm | 200 mm |
|---|---|---|---|---|---|
| peak force | 1.18 N | 1.18 N | 1.18 N | 1.18 N | 1.18 N |

1.18 N is 0.12 kg × 9.81 — each fruit carrying its own weight and nothing else, at every spacing down to touching. The stems are `contype=0` and the welds are independent, so neighbouring trusses cannot load each other at all. The rule had misread its own evidence: the 127 mm incident it cited was a fruit spawned **inside the open gripper at the arm's home posture**, which is a scene-layout bug and has nothing to do with fruit-to-fruit distance.

What close spacing *does* cost is planning, not physics — and that is a pick-order problem, which is what the deck camera is for.

Adding fruit mid-run **voids every checked plan** rather than weakening it: the planner's guarantee is that the route clears every fruit it is *not* picking, and a fruit that appears after the check was never in it. The crop carries a version, and a plan built against an older one is thrown away — and the deck camera re-surveys and re-orders the whole row.

### The deck camera: it decides where to look and what to pick first

There are now **two** cameras, and you can see both in the scene — an RGBD sensor on a pan-tilt head on a mast behind the arm, and a second on the wrist. They do different jobs, and neither can do the other's:

| | deck (chassis mast) | wrist (eye in hand) |
|---|---|---|
| range | 1.26 – 1.48 m | 0.28 m |
| sees | the whole row, arm stationary | one fruit |
| accuracy | ~2 mm on an isolated fruit | sub-millimetre (Week 3's gate) |
| blind to | the difference between two touching fruit | everything it has not been driven to |

**The deck camera is articulated, and the head had to earn its place.** The obvious argument — "it can look around" — is worth nothing here, twice over: one fixed frame already covers the whole placement band, and a camera rotating about its own optical centre cannot see round anything, because pure rotation about the pinhole leaves every occlusion in the scene exactly where it was. What pays is that the lens sits on a **100 mm yoke offset from the pan axis**, so panning *translates* it by up to 90 mm — and translation is the only thing that changes which fruit is hidden behind which. Measured over 48 fruit packed to 72–100 mm centres, the band the old 200 mm spacing rule forbade:

| | fruit found | head slew | arm motion |
|---|---|---|---|
| bolted down, one frame | 31/48 (65%) | — | none |
| five head poses | **40/48 (83%)** | 2.9 s | none |

Nine poses buy one more fruit for 3.6 s more slew; going wider is worse on every column. Five is where it stops paying. The mast also moved *back* 100 mm to make room for the yoke, so the lens stays exactly where the optics were measured and the head is **further** from the arm when panned than when parked — articulating it spends none of the swept-volume clearance that sited it.

This closes the last cheat in the Week 4 loop. The arm used to know where to point the wrist camera because **the script told it** — the staging pose came out of the operator's ground truth. So what the mast replaces is not slow code, it is a cheat, and the fair comparison is against the honest alternative: with one eye-in-hand sensor you have to *sweep* the row. Both measured on the same 8-fruit row, counting simulated arm-seconds:

| | fruit found | arm travel | arm time |
|---|---|---|---|
| the deck survey | 8/8 | 0.000 m | **0.00 s** |
| a wrist sweep (10 poses) | 8/8 | 2.557 m | **24.14 s** |

Both find everything, so this is not an accuracy argument. 24 seconds is most of a whole pick — the campaign's mean cycle is 31.3 s — spent before the first tomato is touched, and paid again on every row. The sweep's cost scales with the length of the row; the deck survey's does not scale at all.

⚠️ Both rows are at **`--speed 0.15`**, the speed the campaign and the 31.3 s cycle were measured at. That matters and it is a flag you have to pass: the sub-command's own default is 0.5, where the same ten poses cost 11.56 s. Quoting the sweep at one speed and the cycle at another would flatter the mast by more than 2×. The deck survey costs 0.00 s of arm time at every speed, because the arm does not move — the head slews for 2.9 s instead.

It also decides the **order**, which is not cosmetic. Swept in `deck_cam.py --pairs`, two fruit at a given centre-to-centre distance:

| centres | outcome |
|---|---|
| ≤ 100 mm | **one of the two is refused outright, the other plans fine** |
| 120 – 140 mm | pickable, but only via a fallback route — a wrist roll, a leaned-back pull, or a deeper staging plane |
| ≥ 170 mm | the direct route works; the neighbour costs nothing |

Read the first row again: on a close pair, **which one you pick first is the difference between harvesting one fruit and harvesting two** — pick the plannable one and the other becomes a lone fruit with a clear route. So the survey is fed to a cost model (crowding, plus the wedge-shaped corridor the 140 mm pull sweeps *below* each fruit). The obstacle set shrinks as the row empties, which is exactly why the order cannot be a sort.

**What the order minimises is tomatoes lost, and it is proved rather than approximated.** The objective used to be "minimise total risk", which is the wrong quantity: summing a risk score treats one pick at risk 2.0 as equal to two picks at risk 1.0, but a pick whose worst neighbour is inside the refusal distance *is refused* — so the second case loses two fruit and the first loses one. An optimiser on that sum will trade a whole tomato to shave a fraction off a score. The rewrite splits what a neighbour does into a **saturating max** (does this pick get refused — the loss) and a **sum** (route length — a tie-break), and counts the first in fruit.

Given that objective the problem is *almost* small enough to solve exactly, and the "almost" is one line of physics. Under the relaxation "every attempt removes its fruit", the cost of a pick depends only on the **set** still standing, which makes the state space 2ⁿ rather than n! — Held-Karp over (set already picked, fruit picked last) settles a 15-fruit row in 451 ms against a 31 s cycle. But a **refused** pick leaves its fruit standing, so which fruit are up after k attempts is not a function of which k were attempted, and the set stops being a sufficient state. The shipped planner therefore solves the relaxation exactly and then hill-climbs that answer against the true cost. Against brute force over all permutations at n = 5–8, the pair reaches the true optimum on 21–27 of 30 layouts and averages 0.0006 of cost above it when it misses, against 0.077 for the exact-but-relaxed answer alone.

| n=15, 30 layouts | cost | expected fruit lost | tour | solve |
|---|---|---|---|---|
| placement order | 18.758 | 14.588 | 3.73 m | — |
| greedy | 18.758 | 14.588 | 3.59 m | 7 ms |
| greedy + hill-climb | 18.703 | 14.588 | 1.59 m | 1067 ms |
| exact on relaxation | 18.730 | 14.588 | 2.25 m | 630 ms |
| **exact + refined** (ships) | **18.702** | **14.588** | **1.58 m** | 1473 ms |

⚠️ **Read the `expected fruit lost` column. It is identical for every method.** That is not a formatting error — it is the finding, and it is explained two sections down: the cost model is symmetric, so it believes no order can save a fruit, and the optimiser correctly reports that there is nothing to win. What ordering *does* buy under this model is tour length, 3.73 m → 1.58 m, and `mission.park_arm` teleports between picks so that cannot show up in a measured cycle time either.

**Does it actually help? ⚠️ This README previously answered "yes, 2 refusals against 6". That result has been withdrawn.** Two separate faults, both in the test rather than the robot, and both found by re-running it:

1. **The control arm was handed ground truth.** `week4_order.py` passed `deck=None` for the placement-order arm, which does not only drop the ordering — it also switches that arm's staging poses over to `crop.placed`, the operator's exact positions. So the baseline was running with *better* position knowledge than the arm being tested. The control is now `deck_order=False`, which keeps the survey and throws away only the sequence.
2. **The "clustered" rows were not in the band they claimed.** `cluster_layout` aimed pairs at 90–150 mm but never enforced a floor on the *incidental* pairs eight fruit make in a small envelope, so rows labelled clustered had closest pairs of **76–97 mm**. That is the blocked band — where by construction no order can help, because a fruit inside 120 mm of a neighbour is refused whichever one you take first. Re-running with the ground-truth fault corrected gave exactly what that band predicts: **19 refusals against 19, no difference at all.**

Both are now fixed — the floor is checked pair by pair, and `--band` names the three regimes explicitly:

| band | centres | what ordering can do |
|---|---|---|
| `blocked` | 75 – 95 mm | nothing. Everything is inside the refusal distance of something. |
| `contested` | 100 – 150 mm | this is the band the pair sweep says order decides |
| `loose` | 175 – 260 mm | nothing. No neighbour costs anything past 170 mm. |

With both faults fixed, the `contested`-band run — 32 attempts per arm, both arms on the deck survey, only the sequence differing:

| order | clean | crated | **refused** | disturbed | predicted refusals |
|---|---|---|---|---|---|
| deck-planned | 19/32 (59%) | 19 | **12** | 0 | 6.95 / 6.37 / 8.00 / 6.05 |
| placement order | 18/32 (56%) | 18 | **12** | 0 | 6.95 / 6.37 / 8.00 / 6.05 |

**That is a tie, and — this is the part that matters — the corrected cost model predicted the tie exactly.** Its forecast gain was `0.00` fruit on every one of the four layouts, and the measured refusal difference was 0. The *old* model, on the same layouts, forecast a 3.33-fruit gain that never appeared. So the rewrite did not make the robot harvest more; it made the planner stop claiming it would. That is a smaller result and a real one.

The reason it forecasts zero is a flaw in the optimiser that only showed up by flying it:

**`_pair_risk` is symmetric, and the effect the whole thing exists to exploit is not.** The pair sweep's finding is that at 100 mm *one* fruit of a pair is refused and *the other plans fine* — that asymmetry is the entire ordering argument. But the cost model is a function of the separation vector, so at 100 mm side by side it returns 1.000 in both directions: it says both fruit are blocked, and cannot say which to take first because it does not believe there is a difference. Run `deck_cam.py --optimal` and the consequence is stark — **every method, from placement order to the exact solver, produces an identical expected loss** (14.588 fruit at n=15). Only tour length moves, from 3.73 m to 1.58 m.

So the machinery is right and it is pointed at a cost function with no ordering signal in it. **The fix is a measurement, not a weight:** `--pairs` records *that* one of a close pair was refused but not *which*, and the crowd term needs a directional part fitted to that. Until then the honest claim is narrow — the deck camera's ordering is optimal under a model which says order does not matter, and it demonstrably does not harm anything.

⚠️ **One loose end, reported because it is consistent and not because it is understood.** The deck-ordered arm ran a **16.3 s mean cycle against 21.0 s**, and the direction held on three of the four layouts. That should not happen: `mission.park_arm` teleports between picks, so tour length cannot reach the clock. The plausible mechanism is route complexity — a fruit picked after its neighbours are gone gets a direct route, one picked while crowded gets a fallback with a deeper staging plane — but the per-layout magnitudes are too large for the number of picks involved, so something else is in there. It has not been traced and is not claimed as a result.

One thing did come out of the corrected model, and it came from watching a real failure. Layout 1 opened with `p06`, which was refused (`gr_right_pad within 25 mm of p00`), and then `p00` was refused too — `within 32 mm of p06`, by the fruit the model had already crossed off. **A refused pick does not remove the fruit.** The old cost function removed every attempted fruit whether the attempt worked or not; it now removes them only on success, which is what makes the exact solver only exact on a relaxation and is why the shipped planner refines its answer afterwards.

```bash
./.venv/bin/python simulation/mujoco/week4_order.py --band contested   # the real test
./.venv/bin/python simulation/mujoco/week4_order.py --band blocked     # ties, by construction
./.venv/bin/python simulation/mujoco/week4_order.py --band loose       # ties, by construction
```

```bash
./.venv/bin/python simulation/mujoco/deck_cam.py             # the survey gate
./.venv/bin/python simulation/mujoco/deck_cam.py --scan      # what the head buys
./.venv/bin/python simulation/mujoco/deck_cam.py --optimal   # exact vs the hill-climb
./.venv/bin/python simulation/mujoco/deck_cam.py --pairs     # what a neighbour costs
./.venv/bin/python simulation/mujoco/deck_cam.py --shot      # stills of both cameras
./.venv/bin/python simulation/mujoco/week4_order.py          # does the order actually help?
./.venv/bin/python simulation/mujoco/week4_place.py --grid 10 --no-deck   # the old behaviour
```

Three things it is honest about. The deck camera **cannot separate touching fruit at 1.3 m** — four in a 70 mm cluster come back as one blob from *every* head pose, because 90 mm of parallax does not undo a 70 mm separation at 1300 mm. The head widens the band it copes with; it does not remove the floor, and that is why the wrist camera still goes in. The cost model's travel term **cannot show up in the measured cycle time**, because `mission.park_arm` teleports the arm back to park between picks; it is weighted low, reported separately, and it is there because a real machine pays it. And **"optimal" means optimal under the proxy** — a geometric model that runs in milliseconds, not the kinematic replay that decides whether a route actually exists. The only thing that settles whether a better order harvests more tomatoes is flying both, which is what `week4_order.py` does.

### The Week 4 result — a throughput number, and it is bad

57 logged attempts across four crop densities, arbitrary layouts the robot had never seen.

⚠️ **These numbers predate the deck camera and the pick ordering**, and they were taken on rows generated under the old 200 mm spacing minimum — so every layout in them is comfortably spread, which is the regime where the ordering makes no difference. They stand as the Week 4 milestone figure and re-running the campaign is its own measurement, not a footnote to this one.

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

Worse, the **`grasp failed: 20` row above was hiding a second failure**. Fourteen of those twenty have `broke: true` and a resting position **3 to 39 metres from the row** — those are not failed grasps, they are fruit leaving the gripper and travelling. The bucket said "the tool failed once the arm arrived" and stopped there, and the distinction was sitting in the log the whole time.

#### ⚠️ Correction: most of this *was* a tunable, and this section said it was not

This section previously claimed the instability "is not a tunable", listing hold force, pad rolling friction, `CARRY_SPEED` and timestep as things it survived. Every one of those tests was real and every one of those conclusions holds. The list was simply missing the parameter that mattered: **pad compliance**.

Traced substep by substep (`week4_grip.py --trace`), the fruit is not flung when the peduncle parts — at that instant it is still centred, 2.3 mm off the tool site, with the pads loaded at ~99 N. It is lost a second and a half later, during the carry: it **creeps out of a closed gripper under its own weight**, because a soft MuJoCo contact does not hold a tangential load indefinitely, it drifts. Once the centre has walked ~30 mm the sphere is off the edge of a 37 mm pad and there is nothing left to hold. Eight layouts, one full pick each:

| pad `solref` | crated | peak grip force | fruit creep |
|---|---|---|---|
| `[0.004, 1]` | 8/8 | 10.1 N | 280 mm |
| **`[0.006, 1]`** — now | **8/8** | **7.0 N** | **34 mm** |
| `[0.008, 1]` | 6/8 | 6.2 N | 1519 mm |
| `[0.020, 1]` — was | 5/8 | 6.0 N | 1486 mm |

**The old value's justification had expired.** The pads were softened to `0.02` to keep the *closing* transient off the peduncle, back when the grip was a step input. Closing on a hanging fruit and reading the stem shows softening never actually fixed that — `0.02` still snapped it, at 16.97 N against a `SNAP_N` of 12.0 — and that `reach.Gripper.ramp`, which arrived later, fixes it at **every** stiffness tested (9.54 / 6.47 / 5.16 N). So the softening had been buying nothing and costing three picks in eight. That is the lesson worth keeping: not that a number was wrong, but that it was *right when it was chosen* and nobody re-took the measurement when the thing it compensated for was fixed properly.

What survives the fix: **1 of 8 picks still shows a fruit over 1 m/s**, and it starts in the `turn` leg, where the pad force spikes to 761 N and collapses to 15 N while the wrist rotates. That one is older and genuinely separate — the same layout at the old `0.02` was *worse* (never reached the crate at all), and it does not scale with cycle speed. `mission.TURN_DONE_DEG` carries a note about a 2F85 rolling a sphere out of its pads past ~22°, measured on the soft pads; that measurement is now due to be re-taken.

So the argument for the cradle-from-below gripper the MVP specifies is **weaker than this README claimed, and still standing**. A 2F85 pinching a smooth 66 mm sphere at ~100 N when holding it needs 1.18 N is not a good way to hold a tomato, and the residual proves it. But "not a tunable" was an overclaim, and it was reached by sweeping four parameters and concluding about all of them.

⚠️ **The 57-attempt campaign above predates this fix and has not been re-run.** Re-running it is its own measurement.

```bash
./.venv/bin/python simulation/mujoco/week4_grip.py            # the gate
./.venv/bin/python simulation/mujoco/week4_grip.py --solref   # the trade, both sides
./.venv/bin/python simulation/mujoco/week4_grip.py --close    # ramp vs step input
./.venv/bin/python simulation/mujoco/week4_grip.py --trace    # where the fruit goes
./.venv/bin/python simulation/mujoco/week4_grip.py --windowed # watch one pick
```

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

⚠️ **This speed sweep predates the `PAD_SOLREF` fix and has not been re-run.** It was taken at pad compliance `0.02`, where the fruit crept out of the pads under its own weight on 3 picks in 8 *at any speed at all* — so some of what this table attributes to speed is that. The direction is unlikely to reverse: the fix does not touch what happens when the arm accelerates out of the row. But the numbers are due to be re-taken, and `mission.CARRY_SPEED`'s 0.25 cap was chosen against a gripper that is no longer the one in the model.

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

---

## Week 5 — a whole house: scout it, plan it, harvest it

Everything above works **one row, from a base bolted to the floor**. That answers a manipulation question. `simulation/mujoco/farm/` asks the logistics one: four rows, eight metres, and a machine that has to drive to reach any of it.

```bash
./.venv/bin/python simulation/mujoco/farm/watch.py     # the whole shift, six panels
```

    +-------------------+-------------------+-------------------+
    | scout cam         | down the aisle    | the house         |
    +-------------------+-------------------+-------------------+
    | THE MAP           | the robot's shift | wrist cam         |
    +-------------------+-------------------+-------------------+

**A shift, with nothing handed to it.** The robot is told the house exists and which aisle it is in. It is not told where a single tomato is. It drives the aisle mapping as it goes, decides which fruit are ripe, plans where to stop, and takes them: on a 48-fruit house, **4 crated of the 5 ripe fruit on its row**, 1 refused (a real neighbour 30 mm from the pad), 2 never detected. 20 s scouting, 33 s driving, 76 s picking.

**The pipe rail is the finding.** There is no off-the-shelf mobile base for the FR5 and no public URDF for one, and the open-source bases that do ship URDFs are laptop-shelf research platforms. But a free-roaming AMR is the wrong machine anyway: **Dutch glasshouses already have a rail network in every aisle** — the heating pipes, 51 mm OD at 550 mm centres, which every trolley in the industry runs on. That makes the base a **1-DOF robot**. No localisation, no drift, no navigation margin, and it physically cannot wander into a crop row. An AMR would be solving SLAM to reproduce a constraint the building gives away free.

**The 1.60 m row pitch is what forces two arms**, and the arithmetic is worth following. A trolley in the aisle sits 800 mm from the crop either side — reachable, and *not* where any Week 1–4 number was measured, which was a **600 mm** standoff. Mount the arms ±200 mm off the centreline and they are back at 600 mm exactly, every measured clearance and cycle time carried over unchanged. The 200 mm on the other side is then the second arm's mount. A one-armed trolley drives every aisle twice.

**Ripeness works where it matters and fails where it is interesting.** The hue separation was measured before the detector was designed, not after:

| stage | hue | vs the canopy | recall |
|---|---|---|---|
| red | 2 | clear | **100%**, and 100% correctly banded |
| turning | 14 | clear | 100% |
| breaker | 29 | clear | 100% |
| green | 49 | **a stem is 55, a leaf is 62** | **33%** |

⚠️ **That costs the harvest nothing and would cost a scouting product a great deal.** Only red is picked, and red is the band that works — a green fruit the map misses is one nobody was going to touch. But Vinea's second module is *scouting*, whose output is a yield forecast, and a forecast built on "we counted the fruit we could see" is wrong in the direction that flatters it. Green recall is the number to quote there, not the average.

**Two gaps, stated rather than papered over.** ~~The scout camera looks one way, so the second arm's row maps **0/14**~~ — **fixed twice over**: the shared head was made to articulate and turn 180° (0/14 → 8/14), and then given up in favour of **one head per arm** (`farm/decks.py`), each over its own arm's plate at the Week 1–4 standoff of 0.60 m, scanning only its own row and never crossing the aisle. ~~And `farm/armframe.py` rebinds other modules' globals, because `mission.py` is written in absolute world coordinates~~ — **also fixed**, and it had to be, because that rebinding was what made two arms mid-mission inexpressible. `mission.ArmFrame` carries the five world constants as a value on the `Planner` and stamps them onto the `Mission`, so a plan is self-contained and nothing global moves. The re-taking of every clearance number was done rather than deferred: the two-arm reach gate is 20/20 with every clearance and chosen route byte-identical across the refactor.

### Both arms, one row, two windows

```bash
./.venv/bin/python simulation/mujoco/two_arm_farm.py   # or: python 2armfarm.py
```

The whole cycle for one full row — map, plan, travel, pick, crate — with two arms and two live windows: a **SENSORS** window (both wrist cams with the HSV ripeness overlay, both deck cams with their live pan angles) and a **MISSION** window (the map, a tracking aisle shot, the live pipeline state per arm, and per-arm pick times with a running mean). See [COMMANDS.md](COMMANDS.md).

**Both arms are stepped inside one physics loop**, each with its own mission state machine — one clock, one plant, two control laws. They were *architecturally* serialised until the two things that made concurrency inexpressible were removed: `armframe` rebound `mission`'s module globals per arm, so two arms mid-mission needed two conflicting sets of them in one interpreter (now `mission.ArmFrame`, a value carried on the plan), and `execute` owned its own loop, so two of them could not interleave (now `week2_pick.MissionRun`, a generator that stops each control cycle with its setpoints written and physics pending). Neither was load-bearing for collision safety.

That removed the blocker. It did not, on this machine, buy overlapping manipulation — see immediately below, which is the more interesting result.

⚠️ **The architecture is concurrent; this deck geometry is not, and the shipped default has 0% manipulation overlap.** Both arms' `PARK` postures fold the elbow back across the aisle, so the arms interleave in x and are held apart only by the 500 mm stagger along the row. Swept, worst arm-vs-arm gap: both parked **+110 mm**, both reaching into their own rows **+188 mm**, either one stowed **+318 mm**, one arm *moving* while the other *sits at PARK* **−40 mm**.

So an arm that is not flying is folded rather than parked, and the two interlock on a single token for the shared middle of the deck. **That interlock ended up covering a whole mission, and it was narrowed four times first — each time guided by a guard abort rather than by reasoning about the geometry, and each narrowing found another contact at 12–15 mm.** The pattern underneath: the hazard is the arm that is *waiting*, not the one moving. A waiting arm must hold some posture, every posture except the stow is within the other arm's reach somewhere in its cycle, and an arm holding a fruit cannot stow — `park_arm` is a teleport. There is no safe point to hand the deck over mid-pick.

More overlap costs fruit, monotonically, and every abort is the guard being right:

| interlock | crated | guard aborts | cycles with both arms moving |
|---|---:|---:|---:|
| none, both arms free | — | many | ~47% |
| the returning half | 5/7 | 2 | 19% |
| + `extract` | 4/7 | 2 | 15% |
| **whole mission (ships)** | **6/7** | **0** | **0%** |

**This is a measured mechanical limit, and that is a different kind of thing from the one it replaced.** The old serialisation was `armframe` rebinding module globals — no measurement could have moved it. This one shrinks the moment `trolley.ARM_STAGGER`, the deck width, or a `PARK` that does not fold the elbow across the aisle changes, with no change to the code; `duo.CROSSING_LEGS` is the knob. Everything other than the manipulation runs concurrently — mapping, planning, waiting, both deck heads scanning, travel — and the run prints both readings rather than the flattering one: cycles with both arms mid-mission, and cycles with both arms actually moving.

Arm-vs-arm clearance is mandatory in both the planner's preview and the runtime guard, and the guard is the one that holds the line: `ArmObstacles` reports where the other arm is *now*, which is a measurement every control cycle for the guard and a one-snapshot prediction for the planner.

**Building it found four bugs that had been shipping since the second arm was fitted**, all recorded in the Bug Log (entries 43, 54, 55, 56): the two arms were **parked 83 mm inside each other** (forearm through forearm, on every two-armed scene ever built); nothing checked arm against arm, so the clash was in nobody's obstacle set; `week2_pick.execute` scored arm B's grasps against **arm A's gripper**, returning `grasped: False` on picks that had worked; and the planner previewed routes with 13 free DOF that the executor flew with 6. The first one is the instructive one — it had been visible in every render since the arm was added, and nothing was looking.

**Making the arms concurrent found six more** (entries 63–68), and the pattern in them is worth as much as the fixes. Two were the deck camera moving by one mechanism while its own mast moved by another — one rigid assembly, two transports — which presented as a *teleport* and as a *pole leaving its camera behind* and was a single defect. Two were the planner's preview disagreeing with the executor about things that live in the **null space**: the posture it prefers and the joint speed it may use. Those had been invisible for as long as there was one arm, because the tool goes to the same place either way and the crop only cares where the tool is — it took a second arm, whose elbow is what you collide with, to make the null space observable. One was a `numpy` mask that was **39% of the entire simulation** and had never been profiled because a plausible story about camera cadences had been written down as a comment. And one was the third instance of a bare unprefixed joint name, found the same way as the first two: by making something run that had never run.

## What this does not prove

Worth saying before anyone else says it:

- **Sim contact is not real contact.** A MuJoCo grasp rate is not a field grasp rate. A real tomato is soft, wet, and bruises.
- **The Week 5 ripeness signal is a colour the scene file sets.** `farm/crop.py` hangs fruit across the real horticultural stages so there *is* something to classify — which answers `ripeness.py`'s objection that uniform red spheres carry no signal — but the hue comes straight out of an `rgba`, with no calyx, no shoulder, no ribbing, no bloom and no lighting variation. A classifier that reads it perfectly has demonstrated the **pipeline**, not the perception. The honest claim is "the robot can carry a ripeness decision through mapping, routing and picking"; the number that means anything is how many ripe fruit reached the crate, not the classifier's accuracy.
- **One aisle, one arm, one pass.** The Week 5 shift works a single aisle with arm A. The second arm is mounted and its geometry is proven, but nothing drives it; the house is four rows and a shift covers one of them properly.
- **The stem's break force is invented.** No force gauge has been near a real peduncle for this project. It is set to 12 N by the *simulator* — gripping an attached fruit loads the stem 6-8.6 N by itself, so anything under ~9 N detaches on contact.
  **It does not, however, decide whether picks succeed.** Swept 9 / 12 / 16 / 20 N over one 8-fruit layout, every fruit returned the *identical* outcome at every value, and kg/hr moved 11% on cycle-time noise alone. `SNAP_N` sets when the stem parts and nothing downstream depends on it: the failures happen either before detachment (the arm never arrives, or never grips) or after it (the fruit is flung during carry). Still an invented number that has to be declared — just not the one the throughput figure is fragile to.
  ```bash
  ./.venv/bin/python simulation/mujoco/week4_snap.py     # the sweep
  ```
- **The planner is no longer given perfect positions, but the cameras are perfect.** Week 3 closed the loop — an eye-in-hand camera, a detector and a deprojection verified to 0.39 mm against ground truth — and the deck camera closed the last hole in it, which was that the arm still learned *where to look* from the script. What remains ideal is the *sensors*: square pixels, no distortion, principal point exactly centred, no noise, and extrinsics known to machine precision. Real hand-eye calibration is a millimetre-level problem on its own and is not in this repo, and a second camera means two of them plus the transform between.
- **The deck camera's recall is flattered by the scene.** Red spheres against green foliage, noiseless depth, and every leaf `contype=0` and above the fruit. It finds 21/21 across the whole band, and that number says more about the renderer than about the sensor. The one place it fails is real and is not tuned away: four fruit in a 70 mm cluster fuse into a single blob and come back as *one* detection with a position 48 mm off, between them — **and the pan-tilt head does not fix it**, returning the same 3 of 6 from all five poses.
- **The pick order is scored by a proxy, and the proxy currently has no ordering signal in it.** `deck_cam.plan_order` runs a geometric cost model in milliseconds; whether a route actually exists is `mission.Planner`'s kinematic replay at ~150 ms. The proxy's thresholds were swept against the planner on *two-fruit* rows, and they do not survive the trip to fifteen: `_pair_risk` is symmetric, so it scores both fruit of a close pair as blocked while the sweep it was fitted to says one of them plans fine. The search over that model is exact-then-refined and near-optimal; the model it is searching believes no order can save a fruit. `week4_order.py` flies both orders and returns a tie, which is the model's own prediction. The fix is a directional term fitted to a sweep that records *which* fruit of a pair gets refused.
- ~~**The chassis never moves.**~~ **Out of date for Weeks 1–4's numbers only.** Every Week 1–4 figure quoted above *was* measured with the arm bolted in one place, and that is still what those numbers mean. But Week 5 built the trolley: `farm.trolley` is a pipe-rail chassis with a real prismatic drive joint and a position servo, `farm/trolley.py --drive` drives the length of the house, and `farm/duo.py` counts the travel time as its own line in the shift report. What is still true is the narrower claim — **no Week 1–4 cycle time includes travel**, and there is no headland turn anywhere, because a pipe-rail trolley in a single bay never makes one.
- **It picks every fruit it sees.** ⚠️ **Out of date.** Weeks 1–4 had no ripeness selection because every fruit was one `rgba`. `farm.crop` hangs fruit across four horticultural colour stages and only `red` is taken, so the Week 5 throughput is over *targets* rather than over every tomato. The honest remaining caveat is that the classifier reads a hue the scene sets directly — no calyx, no shoulder, no bloom, no lighting variation — so it demonstrates the pipeline, not the perception.
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
  camera.py           the eye-in-hand sensor, intrinsics, deprojection,
                      and the housings that make both cameras visible
  deck_cam.py         the chassis survey camera on its pan-tilt head: scans
                      the whole row and decides what to pick first
  detect.py           the detectors, scored on recall and false positives
  week3_perceive.py   see the fruit, estimate where it is, pick it from that
  outcomes.py         one attempt → exactly one named failure bucket
  picklog.py          one row per attempt, appended, with the cycle time
  carrytrace.py       why a fruit that was gripped never reached the crate
  week4_place.py      put fruit anywhere, harvest them, add more mid-run
  week4_watch.py      the same, in a four-panel window you click to place in
  week4_order.py      does the deck camera's pick order beat placement order?
  week4_grip.py       why the tomato fell out, and the number that stopped it
  week4_run.py        the throughput campaign across crop densities
  legacy_cycle.py     the unplanned cycle, kept as the baseline to beat
  week1_*.py          the Week 1 demos
  farm/               Week 5: the whole house, kept separate from weeks 1-4
    house.py          four rows at real Venlo dimensions, pipe rails and all
    trolley.py        the pipe-rail trolley the arm rides, room for a 2nd arm
    crop.py           a random crop, random ripeness, different every open
    scout.py          drive the aisle and come back with a map
    route.py          which stops to make, and what to take at each
    run.py            a whole shift: scout, plan, harvest into the crate
    watch.py          all of it in six panels, including the live map
    armframe.py       makes the world-frame planner work for a moving arm
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
| **Week 4** | Closed loop with no hardcoded positions, 50+ logged picks, success rate and **kg/hr** — ✅ loop closed and placement is free-form; **kg/hr still outstanding** |
| **Week 5** | A whole house on a pipe-rail trolley: scout it, route it, drive it, harvest it, two arms — ✅ built, both arms stepped in one physics loop |

Then: hardware. Nothing built here gets thrown away — the sim becomes the spec for the physical prototype.

## Status

- [x] Idea anatomy — 9/9 pillars
- [x] Toolchain and simulation environment
- [x] Arm in sim, reaching arbitrary targets
- [x] Gripper mounted, full pick-and-crate cycle
- [x] Repeatable grasp under position jitter — 10/10 at ±20 mm
- [x] Collision-aware mission planning, scored on collateral damage
- [x] Failure attribution and a lesson store the planner reads
- [x] Perception loop closed — no hardcoded fruit positions anywhere in the cycle
- [x] A whole house: scout, route, drive, harvest — pipe-rail trolley, two arms, one physics loop
- [ ] Grower validation — 4 interviews done, more booked
- [ ] **Validated kg/hr** — the sprint deliverable still outstanding
- [ ] Two arms manipulating *concurrently* — architecture done, blocked on deck geometry (see Week 5)
- [ ] Technical cofounder
- [ ] Pre-seed raise

---

*Solo project. Pre-team, pre-funding, pre-hardware.* · [getvinea.nl](https://getvinea.nl)
