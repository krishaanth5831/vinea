#!/usr/bin/env python3
"""Week 3: see the fruit, say where it is, and pick it from the estimate.

Week 2 handed the planner exact coordinates out of the simulator. This closes
the half of the loop that removes that luxury: the arm still executes against
the real world, but the plan is now built on what a camera thinks it saw, which
is precisely the asymmetry a real robot lives with.

Four things this file does, and the order is the same as the instructions:

    --calib    Step 3.  Deproject known pixels, score against ground truth in
                        millimetres. No detector. **The gate.**
    --score    Step 4.  Run the detectors on rendered frames, report recall and
                        false positives per frame.
    --budget   Step 5.  Per-axis position error, mean and p95, against the
                        planner's 40 mm clearance.
    --pick     Step 6.  Plan on estimates, execute on truth. The headline.

⚠️ **Every 3D estimate in here is compared to `data.body("tN").xpos`, every
time.** MuJoCo gives ground truth away for nothing, and a perception pipeline
that is never scored against it will be wrong silently — then poison Week 4's
kg/hr figure without ever throwing an error.

The number the week exists to produce is not a detection rate. It is this
sentence: *this pipeline's p95 position error is X mm, the planner needs Y mm
of clearance and currently spends all but 8 mm of it, therefore the clean-pick
rate on estimated positions is Z.* Everything here builds that sentence.

    ./.venv/bin/python simulation/mujoco/week3_perceive.py --calib
    ./.venv/bin/python simulation/mujoco/week3_perceive.py --score
    ./.venv/bin/python simulation/mujoco/week3_perceive.py --budget
    ./.venv/bin/python simulation/mujoco/week3_perceive.py --pick --headless
    ./.venv/bin/python simulation/mujoco/week3_perceive.py --pick \\
        --headless --out week3_perceive.mp4
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ⚠️ Association gate. A detection is only accepted as truss `tN` if its
# estimate lands within this of where the row map says tN hangs.
#
# The trusses on this row are 140 mm apart in y, so half that is the widest
# gate that can never confuse two neighbours. 80 mm is comfortably inside it
# and still four times the p95 error the pipeline actually produces.
#
# **Why a map and not nearest-neighbour-to-the-last-estimate.** The naive answer
# is to sort detections and hand them to the planner in order. That works right
# up until a fruit is missed, at which point every index after it shifts by one
# and the arm reaches for t2 while calling it t1 — silently, with a valid plan
# and a confident log line. Matching against a fixed row map cannot do that: a
# missed fruit produces a missing entry, not a renumbering.
ASSOC_GATE_M = 0.080


class Sighting:
    """One fruit, as perceived. Carries the truth alongside, always."""

    def __init__(self, name, est, truth, det=None):
        self.name = name
        self.est = est
        self.truth = truth
        self.det = det

    @property
    def err(self):
        return self.est - self.truth

    @property
    def err_mm(self):
        return float(np.linalg.norm(self.err) * 1000)

    def __repr__(self):
        return f"<{self.name} err {self.err_mm:.1f} mm>"


class Perception:
    """Camera + detector + deprojection, as one thing the planner can ask.

    Deliberately stateless between looks. A tracker that carried estimates
    forward would score better and would hide exactly the failure Step 6 is
    supposed to expose: what happens when this frame's detector misses.
    """

    def __init__(self, model, sensor, detector, row_map, gate=ASSOC_GATE_M):
        self.model = model
        self.sensor = sensor
        self.detector = detector
        self.row_map = row_map      # name -> nominal hanging position
        self.gate = gate

    def look(self, data):
        """Render, detect, deproject, associate. Returns (sightings, report).

        `report` is the association log, and it exists because Week 4's failure
        diagnosis needs to know which detection was matched to which truss. A
        pick that goes to the wrong fruit and a pick that goes to the right
        fruit badly look identical in a success rate and completely different
        here.
        """
        from detect import estimate

        rgb, depth = self.sensor.both(data)
        R, C = self.sensor.pose(data)
        intr = self.sensor.intr

        dets = [estimate(d, depth, intr, R, C) for d in self.detector(rgb)]
        usable = [d for d in dets if d.est is not None]

        # Greedy over (distance, detection, truss), closest pair first. Each
        # detection and each truss gets used at most once, so two boxes on one
        # tomato cannot become two tomatoes.
        pairs = []
        for i, d in enumerate(usable):
            for name, nominal in self.row_map.items():
                dist = float(np.linalg.norm(d.est - nominal))
                if dist <= self.gate:
                    pairs.append((dist, i, name))
        pairs.sort()

        taken_d, taken_n, sightings = set(), set(), {}
        log = []
        for dist, i, name in pairs:
            if i in taken_d or name in taken_n:
                continue
            taken_d.add(i)
            taken_n.add(name)
            d = usable[i]
            d.matched = name
            s = Sighting(name, d.est, data.body(name).xpos.copy(), d)
            sightings[name] = s
            log.append(f"{name} <- box at ({d.centre[0]:.0f},"
                       f"{d.centre[1]:.0f}) {dist * 1000:.0f} mm from map, "
                       f"err {s.err_mm:.1f} mm")

        for i, d in enumerate(usable):
            if i not in taken_d:
                log.append(f"UNASSOCIATED box at ({d.centre[0]:.0f},"
                           f"{d.centre[1]:.0f}) -> {d.est.round(3)} "
                           f"(no truss within {self.gate * 1000:.0f} mm)")

        return sightings, {"log": log, "dets": dets, "rgb": rgb,
                           "depth": depth, "R": R, "C": C}


# --- Step 5: the error budget ------------------------------------------------
def budget(model, data, sensor, detectors, park_q, row, jitters=(0.0, 20.0),
           seed=0):
    """Per-axis error, mean and p95, across poses / jitter / detectors.

    ⚠️ **Produce the distribution, not a single number.** `mission.CLEARANCE` is
    40 mm — the planner refuses any route that brings the arm nearer than that
    to a fruit it is not picking — and on the milestone run the closest approach
    was 32 mm. So the planner is already spending 8 mm of its own margin against
    fruit positions that are *exactly correct*. It is the tail of this
    distribution that eats the remaining 8 mm, not the mean, which is why p95 is
    the number quoted and the mean is nearly decoration.
    """
    import mujoco

    from camera import stage
    from detect import pose_set, truth_boxes
    from mission import CLEARANCE, reset_park

    rng = np.random.default_rng(seed)
    rows = []

    print(f"\n{'=' * 78}\n  STEP 5 — the error budget")
    print(f"  planner clearance {CLEARANCE * 1000:.0f} mm · "
          f"measured closest approach on the Week 2 milestone 32 mm · "
          f"so 8 mm of margin is already spent")

    for det in detectors:
        for jit in jitters:
            errs = []
            for label, target in pose_set(model, park_q):
                # ⚠️ The jitter has to be applied *after* the reset, not before.
                # `stage` resets the scene, so jittering first and staging
                # second throws it away — see the warning on `camera.stage`.
                def place():
                    row.reset()
                    if jit:
                        row.jitter(rng, jit)

                if target is None:
                    reset_park(model, data, park_q)
                    place()
                    mujoco.mj_forward(model, data)
                else:
                    stage(model, data, park_q, target, row, after_reset=place)

                rgb, depth = sensor.both(data)
                R, C = sensor.pose(data)
                truth = truth_boxes(model, data, sensor.intr, R, C, row.names)
                if not truth:
                    continue

                from detect import estimate, match
                dets = [estimate(d, depth, sensor.intr, R, C)
                        for d in det(rgb)]
                match(dets, truth)
                for d in dets:
                    if d.matched and d.est is not None:
                        errs.append(d.est - truth[d.matched]["pos"])

            if not errs:
                continue
            e = np.array(errs)
            n = np.linalg.norm(e, axis=1)
            rows.append({
                "detector": det.name, "jitter": jit, "n": len(n),
                "mean": n.mean(), "p95": np.percentile(n, 95),
                "max": n.max(),
                "axis_mean": np.abs(e).mean(axis=0),
                "axis_p95": np.percentile(np.abs(e), 95, axis=0),
            })

    print(f"\n  {'detector':>9} {'jitter':>7} {'n':>4} "
          f"{'mean':>7} {'p95':>7} {'max':>7}   "
          f"{'p95 x':>7} {'p95 y':>7} {'p95 z':>7}   (mm)")
    for r in rows:
        print(f"  {r['detector']:>9} {r['jitter']:6.0f}mm {r['n']:4d} "
              f"{1000 * r['mean']:7.1f} {1000 * r['p95']:7.1f} "
              f"{1000 * r['max']:7.1f}   "
              + " ".join(f"{1000 * v:7.1f}" for v in r["axis_p95"]))

    if rows:
        from mission import GUARD_STOP

        worst = max(rows, key=lambda r: r["p95"])
        p95 = worst["p95"] * 1000
        approach = 32.0                    # measured, Week 2 milestone run
        guard = GUARD_STOP * 1000
        headroom = approach - guard

        # The margin chain, written out, because this is the argument the whole
        # week exists to produce and it is easy to state wrongly.
        #
        #   The planner verifies every route clears by CLEARANCE against the
        #   positions it was *given*. On the milestone run, against exact
        #   positions, the arm's closest actual approach was 32 mm. Guard
        #   abandons the mission at GUARD_STOP. So the distance a position error
        #   can eat before a pick turns into a refusal is 32 - 15 = 17 mm,
        #   *not* the 8 mm of planner margin — the 8 mm is what the planner
        #   gives away to IK and dynamics, and it is already spent.
        print(f"\n  {'-' * 66}\n  the margin chain, and it is the point of the "
              f"week:")
        print(f"    planner verifies every route clears by     "
              f"{CLEARANCE * 1000:6.1f} mm")
        print(f"    measured closest approach, exact positions {approach:6.1f} "
              f"mm   (so {CLEARANCE * 1000 - approach:.0f} mm is spent on IK "
              f"and dynamics)")
        print(f"    Guard abandons the mission at              {guard:6.1f} mm")
        print(f"    => a position error may eat up to          "
              f"{headroom:6.1f} mm before a pick becomes a refusal")
        print(f"    worst measured p95 position error          {p95:6.1f} mm   "
              f"({worst['detector']}, ±{worst['jitter']:.0f} mm jitter)")
        verdict = ("INSIDE" if p95 <= headroom else "OVER")
        print(f"\n  p95 is {verdict} the budget by "
              f"{abs(headroom - p95):.1f} mm.")
        if p95 > headroom:
            print("  So the tail of this distribution is expected to produce "
                  "refusals — which is")
            print("  the designed-in safe failure, not a crash. A pipeline that "
                  "refuses constantly")
            print("  is still a pipeline that does not harvest, and that is the "
                  "Week 4 problem.")
        print("\n  ⚠️ Two things this table is not:")
        print("     - It is not a field error. The depth sensor here is exact "
              "to machine")
        print("       precision; on real stereo or ToF, depth noise is the "
              "dominant term and")
        print("       the z column would lead by a wide margin instead of "
              "tracking the others.")
        print("     - n is small (13-20 per row). The p95 is read off ~20 "
              "samples, so it")
        print("       moves by millimetres between runs; treat it as an order "
              "of magnitude,")
        print("       not a figure to quote to two decimal places.")
    return rows


MAX_NUDGE_M = 0.12


def _recentre(look, det, rep, intr):
    """A staging pose that puts `det` nearer the middle of the frame.

    Translating the camera by `d` along its own x moves image content by
    -f*d/range in u, so inverting that gives the world translation which brings
    a box at (u, v) to the principal point. It is exact only for a pure
    translation of the camera — the wrist rotates a little as the arm moves, so
    this lands the fruit *near* the centre rather than on it, which is all that
    is needed to stop the box touching a border.

    Clamped, because an unreachable staging pose is a worse outcome than a
    badly framed one: the arm would stall against its own limits and the pick
    would be lost to something that has nothing to do with perception.
    """
    u, v = det.centre
    rng = det.depth_m if np.isfinite(det.depth_m) else 0.4
    # In camera axes: +x right, +y up, and the image row counts downward.
    delta_cam = np.array([(u - intr.cx) / intr.f * rng,
                          -(v - intr.cy) / intr.f * rng,
                          0.0])
    delta = rep["R"] @ delta_cam
    n = np.linalg.norm(delta)
    if n > MAX_NUDGE_M:
        delta *= MAX_NUDGE_M / n
    # Only y and z. Moving in x would leave the staging plane, which is the one
    # coordinate the planner's whole clearance argument is built on.
    return np.array([look[0], look[1] + delta[1], look[2] + delta[2]])


# --- Step 6: plan on estimates, execute on truth -----------------------------
def _plan_perceived(planner, row, name, sightings, fallback=None):
    """Plan against where the fruit are *believed* to be, execute on truth.

    Same trick as `week2_pick._plan_blind`, and deliberately so — that function
    was Week 3 arriving early, and this is Week 3. The fruit are moved to their
    estimated positions, the plan is built and verified there, and the truth is
    restored before a single leg is flown.

    ⚠️ It has to be *every* fruit, not just the target. `CropObstacles` reads
    the other fruits' positions to decide whether a route clears them, so
    planning with a perceived target against ground-truth obstacles would be a
    robot that can see the fruit it wants and has perfect knowledge of every
    fruit it doesn't. That is not a camera, it is a cheat, and it would make the
    clearance figure meaningless.

    A fruit that was **not** seen falls back to `fallback[n]` if a fallback map
    is given, and to its ground-truth position if not.

    ⚠️ **The ground-truth fallback is a cheat and it was flagged as one here
    from the start.** Its own note used to read: "the alternative — dropping
    unseen fruit from the obstacle set — would let the planner fly straight
    through a tomato it failed to detect. Neither is right. A real machine would
    use the row map." `fallback` is that row map. Week 4's deck camera surveys
    the whole row in one frame, so a fruit this frame's wrist detection missed
    still has a *measured* position to be routed around rather than a perfect
    one. Passing it closes the last place in the perceive-plan-execute chain
    where the planner was reading the simulator.

    It is still not free: the deck estimate is ~2 mm on an isolated fruit and
    tens of millimetres on one whose blob fused with a neighbour, so the
    obstacle it routes around may be in slightly the wrong place. That is a
    measurement error, which `mission.CLEARANCE` has margin for. Knowing the
    exact answer was not an error at all, which is why it was worse.
    """
    import mujoco

    truth = {n: row.pos(n).copy() for n in row.names}
    if fallback:
        for n, p in fallback.items():
            if n not in sightings and n in row.home:
                row.place(n, np.asarray(p, float))
    for n, s in sightings.items():
        row.place(n, s.est)
    mujoco.mj_forward(planner.model, planner.data)
    try:
        return planner.plan(name)
    finally:
        for n in row.names:
            row.place(n, truth[n])
        mujoco.mj_forward(planner.model, planner.data)


def harvest(model, data, row, park_q, sensor, detector, speed=None,
            on_tick=None, verbose=False, clearance=None, look_pose=None,
            on_look=None, only=None, log=None):
    """Harvest the row from perception. The week's headline number.

    Per fruit: stage in front of where the row map says it hangs, take one
    frame, detect, associate, plan on the estimate, fly it against the truth.
    """
    import mujoco

    from camera import stage
    from detect import draw
    from incident import Blackbox
    from mission import (CLEARANCE, DEFAULT_SPEED, GUARD_STOP, Guard, Planner,
                         STAGE_X, reset_park)
    from plant_row import fruit_home
    from week2_pick import anchor_posture, execute, make_reacher

    speed = DEFAULT_SPEED if speed is None else speed
    clearance = CLEARANCE if clearance is None else clearance

    row_map = {n: fruit_home(n) for n in row.names}
    per = Perception(model, sensor, detector, row_map)

    reset_park(model, data, park_q)
    row.reset()
    mujoco.mj_forward(model, data)

    # `only` narrows the run to one truss. Not a shortcut — it is the mode for
    # *watching* a cycle, where five picks at 28 s each is four minutes of
    # staring. The pipeline is identical either way; the other four fruit stay
    # on the plant and remain obstacles the planner has to clear.
    targets = list(row.names) if only is None else list(only)
    unknown = [n for n in targets if n not in row.names]
    if unknown:
        raise SystemExit(f"no such fruit: {', '.join(unknown)} "
                         f"(row has {', '.join(row.names)})")

    # An accumulator the caller can pass in, so closing the window on pick four
    # still reports picks one to three instead of throwing the run away.
    log = [] if log is None else log
    print(f"\n{'=' * 78}\n  STEP 6 — plan on estimates, execute on truth")
    print(f"  detector '{detector.name}' · camera '{sensor.name}' · "
          f"clearance {clearance * 1000:.0f} mm")
    print(f"  association: row map, {ASSOC_GATE_M * 1000:.0f} mm gate")
    print(f"  picking: {', '.join(targets)}\n")

    for name in targets:
        if not row.attached(name):
            print(f"  {name}: already off the plant, skipping")
            continue

        # Look from a staging pose in front of where the map says it hangs. A
        # real machine has this map from the plant spacing and the previous
        # pass; it is not the same thing as knowing where the fruit *is*.
        nominal = row_map[name]
        look = (np.array([STAGE_X, nominal[1], nominal[2]])
                if look_pose is None else look_pose)
        # ⚠️ `reset="arm"`, and the default was wrong here. `stage` used to do a
        # full `mj_resetData` before every look, which restores every fruit to
        # the plant and re-welds every stem — so a tomato picked and crated on
        # pick one was back on the truss for pick two. That made this loop five
        # independent single-pick trials wearing the costume of a row harvest,
        # and it made the headline non-comparable with the `--sequence`
        # baseline it was being quoted against. Park the arm, leave the crop.
        stage(model, data, park_q, look, row, reset="arm")

        sightings, rep = per.look(data)
        for line in rep["log"]:
            print(f"      {line}")
        if on_look is not None:
            on_look(name, rep, sightings)

        if name not in sightings:
            # ⚠️ A fruit the detector never found is a **miss**, and it goes in
            # the log as one. Silently harvesting the four it did find and
            # reporting a successful row is how a perception failure becomes
            # invisible in a kg/hr figure. This is a Week 4 line item arriving
            # a week early.
            print(f"  {name}: MISS — not detected from this pose\n")
            log.append({"fruit": name, "seen": False, "err_mm": float("nan"),
                        "in_bin": False, "grasped": False, "clean": False,
                        "refused": None, "note": "not detected"})
            continue

        s = sightings[name]

        # ⚠️ **Re-frame a target whose box touches the border.** Measured on
        # this row: t0 framed at the bottom edge fitted a radius of 20 mm
        # against a true 33 mm, because the box width it was fitted from was
        # the *clipped* width. The surface-to-centre step then fell 13 mm short
        # and the estimate landed 19.5 mm out — enough to miss the grasp, while
        # every other fruit in the same run fitted 29-30 mm and came in under
        # 4 mm. One bad frame, one bad radius, one lost pick.
        #
        # This is the failure mode that does *not* announce itself: a clipped
        # box is still a confident detection with a plausible position. Looking
        # again from a pose that centres the fruit is the cheap fix and it is
        # what the eye-in-hand mount is for — the camera can move.
        if s.det.extra.get("edge"):
            nudged = _recentre(look, s.det, rep, sensor.intr)
            print(f"      {name}: box on the frame edge (fitted r "
                  f"{s.det.est_r * 1000:.0f} mm) — re-framing from "
                  f"{nudged.round(3)}")
            stage(model, data, park_q, nudged, row, reset="arm")
            again, rep2 = per.look(data)
            for line in rep2["log"]:
                print(f"      {line}")
            if on_look is not None:
                on_look(name, rep2, again)
            if name in again and not again[name].det.extra.get("edge"):
                s = again[name]
                sightings[name] = s
                rep = rep2
            else:
                print(f"      {name}: re-frame did not help, using the first "
                      f"estimate")

        edge = bool(s.det.extra.get("edge"))
        print(f"  {name}: seen, estimate {s.est.round(4)} vs truth "
              f"{s.truth.round(4)} -> {s.err_mm:.1f} mm  "
              f"(fitted r {s.det.est_r * 1000:.0f} mm"
              f"{', EDGE-CLIPPED' if edge else ''})")

        reacher = make_reacher(model, data, speed=speed)
        anchor_posture(reacher, model, data, park_q)
        from reach import Gripper
        gripper = Gripper(model, data)
        planner = Planner(model, data, row, lessons=None, clearance=clearance,
                          park_q=park_q, speed=speed)
        mission = _plan_perceived(planner, row, name, sightings)

        if not mission.ok:
            why = mission.breaches[0] if mission.breaches else None
            print(f"  {name}: REFUSED — {why}\n")
            log.append({"fruit": name, "seen": True, "err_mm": s.err_mm,
                        "in_bin": False, "grasped": False, "clean": False,
                        "refused": str(why), "note": "refused"})
            continue

        box = Blackbox(model, data, row, name)
        guard = Guard(model, data, row, name,
                      stop=min(GUARD_STOP, 0.4 * clearance))
        guard.armed = False
        r = execute(mission, reacher, gripper, row, box=box, guard=guard,
                    on_tick=on_tick, verbose=verbose)
        clean = bool(r["in_bin"] and not r["lost"] and not r["disturbed"])
        print(f"  {name}: grasped={r['grasped']} stem={r['broke']} "
              f"crate={r['in_bin']} gap={r['clearance'] * 1000:.0f}mm "
              f"{'CLEAN' if clean else 'not clean'}\n")
        log.append({"fruit": name, "seen": True, "err_mm": s.err_mm,
                    "in_bin": r["in_bin"], "grasped": r["grasped"],
                    "clean": clean, "refused": None,
                    "clearance": r["clearance"], "lost": r["lost"],
                    "disturbed": r["disturbed"],
                    "note": "" if clean else "collateral or not crated"})
    return log


def report_pick(log):
    """The comparison the week is for: estimates against ground truth."""
    n = len(log)
    seen = sum(1 for r in log if r["seen"])
    clean = sum(1 for r in log if r["clean"])
    crated = sum(1 for r in log if r["in_bin"])
    refused = sum(1 for r in log if r["refused"])
    missed = [r["fruit"] for r in log if not r["seen"]]
    errs = [r["err_mm"] for r in log if r["seen"] and np.isfinite(r["err_mm"])]

    print(f"\n  {'-' * 66}")
    print(f"  fruit in the row      {n}")
    print(f"  detected              {seen}/{n}"
          + (f"   MISSED: {', '.join(missed)}" if missed else ""))
    print(f"  planned (not refused) {seen - refused}/{seen}")
    print(f"  in the crate          {crated}/{n}")
    print(f"  CLEAN PICKS           {clean}/{n}"
          f"   ({100 * clean / n if n else 0:.0f}%)")
    if errs:
        print(f"\n  position error on the fruit it acted on: "
              f"mean {np.mean(errs):.1f} mm · p95 "
              f"{np.percentile(errs, 95):.1f} mm · max {max(errs):.1f} mm")
    # The baseline this is compared against, measured rather than remembered:
    #   ./.venv/bin/python simulation/mujoco/week2_pick.py \
    #       --headless --sequence --no-lessons
    # gives 5/5 clean on this row, closest approach 32 mm, on 2026-08-03.
    print(f"\n  Ground truth on this same row, same sequence, is 5/5 "
          f"(measured, not assumed:")
    print(f"  week2_pick.py --sequence --no-lessons). The gap between that and "
          f"{clean}/{n}")
    print(f"  is what perception costs, and it is the first honest input to "
          f"kg/hr.")
    if missed:
        print(f"\n  ⚠️ {len(missed)} fruit were never detected and are logged as "
              f"misses, not")
        print(f"     as successes. A row that reports 4/4 while leaving a "
              f"tomato on the")
        print(f"     plant is the failure mode Week 4 has to be able to see.")
    return clean


def _watch(model, data, run, args, collected=None):
    """Run the harvest with a live viewer open, at wall-clock speed.

    Same shape as `week2_pick.run_windowed`, and it inherits the same two
    hard-won details:

    ⚠️ **The tick has to pace itself.** Physics runs far faster than real time
    headless, so syncing the viewer without sleeping produces a blur. The clock
    is advanced by one control period per tick and the loop sleeps the
    difference.

    ⚠️ **Hold the window open at the end.** `exit_without_teardown` is
    `os._exit`, so returning from here kills the process mid-frame and the
    window vanishes the instant the last pick lands — which reads as a crash,
    not a finish. Physics is deliberately *not* stepped while it is held: the
    run is over, and stepping on with nothing commanding the servos would let
    the arm sag while you watch.
    """
    import time
    import warnings

    import mujoco.viewer

    from fr5 import exit_without_teardown
    from reach import CTRL_DT

    warnings.filterwarnings("ignore", module="glfw")
    log = [] if collected is None else collected
    with mujoco.viewer.launch_passive(model, data) as viewer:
        clock = [time.perf_counter()]

        def tick(_t=None):
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            clock[0] += CTRL_DT
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

        try:
            run(tick)
            print("\n  run finished — the window stays open, close it to quit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            # Closing the window mid-run is a legitimate way to stop. Whatever
            # picks completed are still in `log`, because `harvest` appends to
            # the list it was handed rather than building its own.
            print("\n  stopped early — reporting the picks that finished")

    report_pick(log)
    # Never returns; the print above is the last thing that happens.
    exit_without_teardown()


# --- wiring ------------------------------------------------------------------
def build_detector(which, conf=0.10):
    from detect import HSVDetector, YoloDetector

    if which == "hsv":
        return HSVDetector()
    return YoloDetector(conf=conf)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_argument_group("what to run (default: --calib --score)")
    mode.add_argument("--calib", action="store_true",
                      help="Step 3 — deproject known pixels, score in mm")
    mode.add_argument("--score", action="store_true",
                      help="Step 4 — recall and false positives per frame")
    mode.add_argument("--budget", action="store_true",
                      help="Step 5 — per-axis error, mean and p95, vs 40 mm")
    mode.add_argument("--pick", action="store_true",
                      help="Step 6 — plan on estimates, execute on truth")

    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"],
                    help="hsv is the control and never fails on this scene; "
                         "yolo is the honest attempt (default: hsv)")
    ap.add_argument("--camera", default="wrist",
                    help="sensor camera (default: the eye-in-hand one)")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--clearance", type=float, default=None, help="mm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--windowed", action="store_true",
                    help="watch it live in a viewer window (--pick only)")
    ap.add_argument("--fruit", nargs="+", default=None, metavar="tN",
                    help="pick only these trusses, e.g. --fruit t2. "
                         "The rest stay on the plant as obstacles.")
    ap.add_argument("--out", default=None,
                    help="write an mp4 of --pick (implies headless)")
    ap.add_argument("--record-camera", default="row",
                    help="camera to record the mp4 from")
    ap.add_argument("--save-frames", default=None,
                    help="directory for the overlaid detection stills")
    args = ap.parse_args()

    if not any([args.calib, args.score, args.budget, args.pick]):
        args.calib = args.score = True

    # ⚠️ Only force EGL when nothing is being watched. Setting it
    # unconditionally is what made this file headless-only for its first
    # version: EGL has no window to open, so --windowed silently did nothing.
    # A live viewer needs GLFW, and `mujoco.Renderer` is happy under either, so
    # the sensor camera keeps working while the window is up.
    if not args.windowed:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    from camera import SensorCamera
    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from plant_row import Row

    model = build_scene(wrist_cam=True)
    data = mujoco.MjData(model)
    park = park_posture(model)
    reset_park(model, data, park)
    row = Row(model, data)
    mujoco.mj_forward(model, data)

    clearance = None if args.clearance is None else args.clearance / 1000.0

    if args.calib:
        from camera import Intrinsics, calibration_error, _report
        from camera import stage as _stage
        from fr5 import tool_pos
        from mission import STAGE_X

        # ⚠️ Not named in bug log entry 58, which lists five modules. This is a
        # sixth, and the entry's own repro grep does not find it — the grep
        # matches double quotes and this site uses single ones. Same shape,
        # same failure: the 0.39 mm gate's pose captions would read arm A's
        # tool under arm B's label. "" is the unprefixed Week 1-4 arm.
        prefix = getattr(args, "arm", "") or ""

        sensor = SensorCamera(model, args.camera)
        print(f"\n{'=' * 78}\n  STEP 3 — deprojection against ground truth, "
              f"no detector")
        print(f"  camera '{args.camera}' · {sensor.intr}")
        ok = True
        try:
            for label, target in [
                    ("parked", None),
                    ("staged mid", np.array([STAGE_X, 0.10, 0.66])),
                    ("staged low", np.array([STAGE_X, -0.18, 0.58]))]:
                if target is None:
                    reset_park(model, data, park)
                    row.reset()
                    mujoco.mj_forward(model, data)
                else:
                    _stage(model, data, park, target, row)
                ok &= _report(f"{label} — {prefix}tool0 at "
                              f"{tool_pos(data, prefix).round(3)}",
                              calibration_error(model, data, args.camera,
                                                sensor.intr, sensor=sensor))
        finally:
            sensor.close()
        print(f"\n  GATE: {'PASS' if ok else 'FAIL'} — every visible fruit "
              f"within 1 mm")
        if not ok:
            print("  Stop. Everything downstream inherits this error.")
            return 1

    if args.score:
        from detect import HSVDetector, YoloDetector, score_detector, summarise

        sensor = SensorCamera(model, args.camera)
        print(f"\n{'=' * 78}\n  STEP 4 — detection on rendered frames")
        print("  recall and false positives per frame. Not mAP.")
        results = []
        try:
            dets = [HSVDetector()]
            try:
                dets.append(YoloDetector(conf=args.conf))
            except Exception as exc:
                print(f"  yolo unavailable: {exc}")
            for d in dets:
                results.append(score_detector(model, data, sensor, d, park,
                                              row, save=args.save_frames))
        finally:
            sensor.close()
        for r in results:
            summarise(r)

    if args.budget:
        from detect import HSVDetector, YoloDetector

        sensor = SensorCamera(model, args.camera)
        try:
            dets = [HSVDetector()]
            try:
                dets.append(YoloDetector(conf=args.conf))
            except Exception as exc:
                print(f"  yolo unavailable: {exc}")
            budget(model, data, sensor, dets, park, row, seed=args.seed)
        finally:
            sensor.close()

    if args.pick:
        sensor = SensorCamera(model, args.camera)
        detector = build_detector(args.detector, args.conf)
        writer = renderer = None
        on_tick = None

        on_look = None
        if args.out:
            import cv2

            from detect import draw, truth_boxes
            from reach import CTRL_DT

            fps, w, h = 30, 1280, 960
            writer = cv2.VideoWriter(args.out,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"could not open video writer for {args.out}")
            renderer = mujoco.Renderer(model, height=h, width=w)
            every = max(1, int(round((1 / fps) / CTRL_DT)))
            cycles = [0]

            def on_tick(_t=None):
                cycles[0] += 1
                if cycles[0] % every == 0:
                    renderer.update_scene(data, camera=args.record_camera)
                    writer.write(cv2.cvtColor(renderer.render(),
                                              cv2.COLOR_RGB2BGR))

            # ⚠️ The frame worth having this week is **not the arm**. It is the
            # detection overlaid on the sensor's own view with the 3D estimate
            # and the ground-truth error printed on it. A grower does not read
            # a success rate; they read "the robot can see the tomato and knows
            # where it is, to two millimetres". So every time the pipeline
            # looks, the annotated sensor frame is held on screen for a beat
            # before the arm moves.
            HOLD_S = 2.0

            def on_look(target, rep, sightings):
                truth = truth_boxes(model, data, sensor.intr, rep["R"],
                                    rep["C"], row.names)
                errors = {n: np.linalg.norm(s.err)
                          for n, s in sightings.items()}
                frame = draw(rep["rgb"], rep["dets"], truth, errors)
                banner = np.full((70, frame.shape[1], 3), 22, np.uint8)
                cv2.putText(banner, f"sensor view - target {target}",
                            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (235, 235, 235), 2, cv2.LINE_AA)
                if target in sightings:
                    s = sightings[target]
                    cv2.putText(banner,
                                f"estimate {s.est.round(3)} m   "
                                f"ground-truth error {s.err_mm:.1f} mm",
                                (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                                (90, 235, 250), 2, cv2.LINE_AA)
                else:
                    cv2.putText(banner, "NOT DETECTED - logged as a miss",
                                (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                                (80, 80, 245), 2, cv2.LINE_AA)
                stacked = np.vstack([banner, frame])
                stacked = cv2.resize(stacked, (w, h))
                for _ in range(int(HOLD_S * fps)):
                    writer.write(stacked)

        collected = []

        def run(tick):
            return harvest(model, data, row, park, sensor, detector,
                           speed=args.speed, on_tick=tick,
                           verbose=args.verbose, clearance=clearance,
                           on_look=on_look, only=args.fruit, log=collected)

        try:
            if args.windowed:
                # Never returns — it reports and then `os._exit`s past the
                # GLFW teardown segfault, so the cleanup below is unreachable
                # on this path. Same trade `week2_pick` makes.
                _watch(model, data, run, args, collected)
            report_pick(run(on_tick))
        finally:
            sensor.close()
            if renderer is not None:
                renderer.close()
            if writer is not None:
                writer.release()
                print(f"\n  wrote {args.out}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
