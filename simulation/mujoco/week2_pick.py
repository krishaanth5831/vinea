#!/usr/bin/env python3
"""Week 2 — picking tomatoes off a plant row without wrecking the plant.

The first cut of this file drove the tool at each waypoint and let IK find the
route. It scored ten picks out of ten. It also knocked a neighbouring tomato
off the plant on eight of those ten, and reported none of it, because the
scorer only ever looked at the fruit it was picking. `legacy_cycle.py` is that
version, kept so the difference is measurable rather than asserted.

This one plans first and counts everything:

    plan       `mission.py` builds the whole route and checks it clears every
               fruit it is not picking, before a joint moves. A route that does
               not clear is replaced; if none clears, the pick is refused.
    guard      the same clearance is measured for real every control cycle. Too
               close and the mission is abandoned — an aborted pick is cheap and
               a stripped truss is not.
    explain    `incident.py` records why anything that fell, fell.
    learn      `lessons.py` turns that into a constraint the planner reads next
               time, so the same mistake is not available twice.

A pick counts as good only if the fruit reaches the crate **and** nothing else
was disturbed. That is a stricter bar than the old 10/10 and it is the one that
means anything to a grower — a harvester that drops one tomato for every one it
picks has a negative yield.

Usage:

    # watch it harvest the row, in a window
    ./.venv/bin/python simulation/mujoco/week2_pick.py

    # the milestone: ten picks against jittered fruit, scored on collateral
    ./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless

    # the same harvest, headless
    ./.venv/bin/python simulation/mujoco/week2_pick.py --sequence --headless

    # show the learning loop: run it blind, then again with what it learned
    ./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless --forget --no-lessons
    ./.venv/bin/python simulation/mujoco/week2_pick.py --trials 10 --headless

    # record the weekly capture — name it after this file, see the README
    ./.venv/bin/python simulation/mujoco/week2_pick.py --headless --out week2_pick.mp4 --camera row
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fr5 import (  # noqa: E402
    MAX_REACH_GRIPPER,
    TOOL_SITE,
    crate_contains,
    exit_without_teardown,
)
from greenhouse import build_scene  # noqa: E402
from incident import Blackbox  # noqa: E402
from lessons import Lessons  # noqa: E402
from mission import (  # noqa: E402
    BIN_HALF,
    BIN_POS,
    BIN_WALL,
    CLEARANCE,
    GUARD_STOP,
    Guard,
    ORIENTATION_COST,
    Planner,
    ROLL_COST,
    INTO_ROW,
    park_posture,
    reset_park,
)
from plant_row import FRUIT_R, ROW_X, SNAP_N, Row  # noqa: E402
from reach import CTRL_DT, DEFAULT_SPEED, Gripper, Reacher, hold  # noqa: E402

# Arrival tolerance for the loaded arm — the 1.05 kg gripper droops ~5.2 mm and
# a P+D position servo cannot null out a constant gravity load. It still grasps
# a 66 mm fruit with room to spare.
REACHED_MM_LOADED = 8.0

# Give up pulling rather than tear the plant apart.
PULL_MAX_S = 6.0


class Aborted(Exception):
    """The guard tripped. Raised out of the tick callback, caught by execute()."""

    def __init__(self, why):
        super().__init__(str(why))
        self.why = why


def anchor_posture(reacher, model, data, park_q):
    """Make the solver prefer the canonical park posture in its null space.

    ⚠️ Without this a sequence harvest quietly winds the arm up. A `PostureTask`
    seeded from "wherever we happen to be" gives the solver no reason to come
    back to a standard configuration, so every pick starts from the posture the
    last one ended in and the drift compounds. Measured across five picks the
    tool returned to PARK to within 2 mm every time — and the joints went from
    [0, -39, -144, 2, 90, -90]° to [67, -7, -141, -39, 24, -173]°: the same
    point in space, reached with the shoulder swung 67° round and the wrist
    flipped. From there, reaching into the row is a different problem, and IK
    took the local gradient the wrong way — the approach to t3 ended 243 mm
    short, in the wrong direction, and grasped nothing.

    A Cartesian target does not pin a posture. This does, weakly (cost 1e-2), so
    it only acts where nothing else is competing for the redundant DOF.
    """
    from fr5 import JOINTS

    q = data.qpos[: model.nq].copy()
    for jname, value in zip(JOINTS, park_q):
        q[model.joint(jname).qposadr[0]] = value
    reacher.posture.set_target(q)
    return reacher


def make_reacher(model, data, speed=DEFAULT_SPEED, prefix=""):
    """A Reacher configured for reaching into a row.

    ⚠️ mocap=None matters. The row's stems *are* mocap bodies, and the Reacher
    parks a marker on mocap 0 by default — which would drag stem t0 around the
    scene every control tick, welded tomato and all.

    ⚠️ reset=False matters just as much. The caller has already parked the arm
    and jittered the fruit; a Reacher built with the default would call
    `reset_home` and undo both, putting the arm back inside the canopy with the
    jitter silently switched off.
    """
    return Reacher(model, data, speed=speed, standoff=0.0,
                   max_reach=MAX_REACH_GRIPPER, reached_mm=REACHED_MM_LOADED,
                   orientation_cost=ORIENTATION_COST, roll_cost=ROLL_COST,
                   approach=INTO_ROW, mocap=None, reset=False, prefix=prefix)


def _run_leg(leg, reacher, gripper, row, target, tick, say):
    """Do what one leg says. This is the whole executor — the thinking is in the
    plan, and that separation is the point: the route is decided and checked
    while the arm is stationary, then replayed by something with no opinions."""
    if leg.approach is not None:
        reacher.approach = leg.approach
    reacher.roll = leg.roll
    reacher.set_orientation_cost(leg.orientation_cost, roll_cost=leg.roll_cost)
    if abs(reacher.speed - leg.speed) > 1e-9:
        reacher.set_speed(leg.speed)

    goal = (reacher.data.site(TOOL_SITE).xpos.copy() if leg.goal is None
            else np.asarray(leg.goal, dtype=float))

    if leg.kind == "move":
        r = reacher.drive_to(goal, on_tick=tick)
        say(leg.name, f"arm {r['arm_mm']:.1f} mm · {r['axis_deg']:.1f}°"
                      if r["reached"] else
                      f"DID NOT ARRIVE — {r['arm_mm']:.0f} mm short")
        return r

    if leg.kind == "settle":
        hold(reacher, goal, leg.seconds, tick)
    elif leg.kind == "grip":
        # Ramped, never a step input. Commanded shut in one cycle the pads
        # arrive at speed and the peduncle went 1.18 -> 91 -> 164 N inside
        # 10 ms — twenty-seven times SNAP_N — so every stem snapped on contact
        # and the pull that followed lifted an already-loose tomato.
        gripper.ramp(reacher, goal, leg.seconds, on_tick=tick)
    elif leg.kind == "release":
        gripper.open()
        hold(reacher, goal, leg.seconds, tick)
    elif leg.kind == "pull":
        # Stop the moment it lets go. Driving the full stroke regardless keeps
        # yanking after the peduncle has parted, and the compliant weld has
        # stretched by then — the release fires that stored energy into a fruit
        # held by two pads and knocks it 22 mm off the pinch centre.
        t = 0.0
        while row.attached(target) and t < PULL_MAX_S:
            reacher.step(goal)
            tick()
            t += CTRL_DT
        say(leg.name, f"peak {row.peak[target]:.2f} N vs SNAP_N "
                      f"{row.snap_n:.1f} N — "
                      f"{'released' if not row.attached(target) else 'STILL ON'}")
    elif leg.kind == "home":
        # Joint space, not tool space. Ramped over `seconds` so the servos are
        # given a trajectory rather than a step, and stepped through the same
        # substep hook as everything else so a stem that breaks during it is
        # still seen and still recorded.
        import mujoco

        from fr5 import JOINTS
        from reach import TICKS_PER_CTRL

        data = reacher.data
        start = np.array([data.joint(j).qpos[0] for j in JOINTS])
        want = np.asarray(leg.posture, dtype=float)
        steps = max(1, int(leg.seconds / CTRL_DT))
        for i in range(steps):
            data.ctrl[reacher.arm_ctrl] = start + (want - start) * (i + 1) / steps
            for _ in range(TICKS_PER_CTRL):
                mujoco.mj_step(reacher.model, data)
                if reacher.on_substep is not None:
                    reacher.on_substep()
            tick()
        # The IK solver has been bypassed for this leg, so tell it where the arm
        # actually ended up. Without this the next mission plans from a stale
        # configuration and the whole exercise repeats one level up.
        reacher.sync()
        say(leg.name, f"joints to park, {np.degrees(np.abs(want - start)).max():.0f}° "
                      f"largest change")
    elif leg.kind == "turn":
        t = 0.0
        while (reacher.tool_axis_error_deg(goal) > leg.tilt_stop
               and t < leg.seconds):
            reacher.step(goal)
            tick()
            t += CTRL_DT
        say(leg.name, f"tool {reacher.tool_axis_error_deg(goal):.1f}° "
                      f"off target axis in {t:.1f}s")
    else:
        raise ValueError(f"unknown leg kind {leg.kind!r}")
    return None


def execute(mission, reacher, gripper, row, box=None, guard=None, on_tick=None,
            verbose=False):
    """Fly a planned mission. Returns what happened, including what it cost."""
    data = reacher.data
    target = mission.target
    fruit = data.body(target)
    clock = [0.0]
    grasped = [False]

    # ⚠️ Per *physics* step, not per control cycle, and the difference is not
    # academic: polled at 100 Hz a stiff arm pulling on a weld drove the force
    # to 77 N before a 12 N threshold was noticed — an effective detach force
    # six times the one written down, feeding straight into the kg/hr figure.
    # The black box wraps `row.update` rather than replacing it, so the break
    # and the evidence for it are read on the same step.
    reacher.on_substep = row.update if box is None else box.substep

    def tick(_t=None):
        clock[0] += CTRL_DT
        if guard is not None and not guard.check():
            raise Aborted(guard.tripped)
        if on_tick is not None:
            on_tick(_t)

    def say(state, note=""):
        if verbose:
            p = data.site(TOOL_SITE).xpos
            print(f"    {state:<9} tool [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]"
                  f"   {note}")

    gripper.open()
    # Taken before the settle leg so it can be put back afterwards. See
    # Row.settle: on any pick after the first in a sequence, a blanket reset
    # re-welds already-harvested fruit to their stems and drags them out of the
    # crate.
    attached_before = row.attachment()
    aborted = None
    try:
        for i, leg in enumerate(mission.legs):
            if box is not None:
                box.leg = leg.name
            if guard is not None:
                guard.leg = leg.name
            _run_leg(leg, reacher, gripper, row, target, tick, say)

            if i == 0:
                # ⚠️ Nothing before this point counts. The weld overshoots for
                # ~50 ms on startup and the peak is above SNAP_N, so a recorder
                # armed any earlier reports the whole row falling off the plant
                # before the arm has moved. The settle leg exists to let it find
                # its resting force; the books open afterwards.
                row.settle(attached_before)
                if box is not None:
                    box.rebase(target)
                if guard is not None:
                    guard.armed = True
            elif leg.name == "extract":
                held = float(np.linalg.norm(fruit.xpos
                                            - data.site(TOOL_SITE).xpos))
                grasped[0] = held < FRUIT_R * 2.0
                say("check", f"fruit {held * 1000:.0f} mm from the tool — "
                             f"{'holding' if grasped[0] else 'DROPPED'}")
    except Aborted as stop:
        aborted = stop.why
        say("ABORT", f"{stop.why[3]} within {stop.why[1] * 1000:.0f} mm "
                     f"during `{stop.why[0]}`")
        # Back out along -x, which is the one direction that always moves away
        # from the crop, then park. The guard stays off for it: it has already
        # fired, and re-firing during the escape would strand the arm in the row.
        guard.armed = False
        here = data.site(TOOL_SITE).xpos.copy()
        reacher.set_orientation_cost(ORIENTATION_COST)
        reacher.approach = INTO_ROW
        reacher.drive_to(np.array([mission.stage_x, here[1], here[2]]),
                         on_tick=tick)

    incidents = box.sweep() if box is not None else []
    return {
        "fruit": target,
        "target": mission.target,
        "in_bin": bool(crate_contains(fruit.xpos, BIN_POS, BIN_HALF, BIN_WALL)),
        "grasped": bool(grasped[0]),
        "broke": bool(not row.attached(target)),
        "peak_n": float(row.peak[target]),
        "seconds": clock[0],
        "aborted": aborted,
        "incidents": incidents,
        "lost": sum(1 for h in incidents if h.detached),
        "disturbed": sum(1 for h in incidents if not h.detached),
        "clearance": (float(guard.min_seen) if guard is not None
                      and np.isfinite(guard.min_seen) else float("nan")),
        "planned": mission.clearance,
        "lane": mission.lane,
        "applied": list(mission.applied),
    }


def _plan_blind(planner, row, name):
    """Plan against where the fruit are *believed* to be, not where they are.

    This is Week 3 arriving early. Today the planner reads exact positions out
    of the simulator, which is a luxury it loses the moment a camera is in the
    loop: a detector gives a centroid with real error in it, and the plan gets
    built against that. A route verified to clear by 40 mm against a fruit whose
    true position is 30 mm off is not a route that clears by 40 mm.

    So `--blind` plans against the nominal row and executes against the jittered
    one. It is the honest stress test of the whole stack, and it is what makes
    the incident-and-lesson loop do visible work: with exact positions the
    planner is simply right and there is nothing to learn from.
    """
    import mujoco

    truth = {n: row.pos(n).copy() for n in row.names}
    for n in row.names:
        row.place(n, row.home[n])
    mujoco.mj_forward(planner.model, planner.data)
    try:
        return planner.plan(name)
    finally:
        for n in row.names:
            row.place(n, truth[n])
        mujoco.mj_forward(planner.model, planner.data)


def _attempt(model, data, row, name, lessons, speed, on_tick, verbose,
             blind=False, clearance=CLEARANCE, park_q=None):
    """Plan one pick and, if the plan holds up, fly it."""
    reacher = make_reacher(model, data, speed=speed)
    if park_q is not None:
        anchor_posture(reacher, model, data, park_q)
    gripper = Gripper(model, data)

    planner = Planner(model, data, row, lessons=lessons, clearance=clearance,
                      park_q=park_q, speed=speed)
    mission = _plan_blind(planner, row, name) if blind else planner.plan(name)
    if verbose:
        print("    " + mission.summary().replace("\n", "\n    "))

    if not mission.ok:
        # Refusing is a result, not an error. It is also the only safe answer
        # when every route the planner knows comes too close — improvising one
        # anyway is exactly the behaviour this whole file exists to remove.
        return mission, {
            "fruit": name, "target": name, "in_bin": False, "grasped": False,
            "broke": False, "peak_n": 0.0, "seconds": 0.0, "aborted": None,
            "refused": mission.breaches[0] if mission.breaches else None,
            "incidents": [], "lost": 0, "disturbed": 0,
            "clearance": float("nan"), "planned": mission.clearance,
            "lane": mission.lane, "applied": list(mission.applied),
        }

    box = Blackbox(model, data, row, name)
    # The guard's floor tracks the planned margin. Left at its default it would
    # be *above* the margin whenever the planner is run permissively, and every
    # deliberately-close route would abort before it could teach anything.
    guard = Guard(model, data, row, name,
                  stop=min(GUARD_STOP, 0.4 * clearance))
    guard.armed = False
    result = execute(mission, reacher, gripper, row, box=box, guard=guard,
                     on_tick=on_tick, verbose=verbose)
    result["refused"] = None

    if lessons is not None:
        lessons.learn(result["incidents"], mission)
        lessons.confirm(mission, result["incidents"])
    return mission, result


def _row_line(i, r):
    def flag(ok, yes="yes", no="NO"):
        return yes if ok else no

    note = ""
    if r["refused"] is not None:
        note = "refused"
    elif r["aborted"] is not None:
        note = f"abort:{r['aborted'][0]}"
    elif r["lost"] or r["disturbed"]:
        note = f"{r['lost']} lost, {r['disturbed']} moved"

    gap = ("   -  " if not np.isfinite(r["clearance"])
           else f"{r['clearance'] * 1000:6.0f}")
    return (f"  {i:3d} {r['fruit']:>6} {flag(r['grasped']):>6} "
            f"{flag(r['broke']):>6} {flag(r['in_bin']):>6} "
            f"{r['peak_n']:7.2f} {gap} {r['seconds']:6.1f}  {note}")


def _header(title):
    print(f"\n  {title}\n")
    print(f"  {'#':>3} {'fruit':>6} {'grasp':>6} {'stem':>6} {'crate':>6} "
          f"{'peak N':>7} {'gap mm':>6} {'secs':>6}  notes")


def run_trials(model, data, row, park, n, jitter_mm, seed=0,
               speed=DEFAULT_SPEED, lessons=None, on_tick=None, verbose=False,
               blind=False, clearance=CLEARANCE):
    """Run the cycle N times against jittered fruit, resetting the row between.

    This is the repeatability measurement: the same cycle, a different set of
    fruit positions each time. Picking one coordinate ten times mostly proves
    the simulator is deterministic.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    log = []
    _header(f"{n} attempts · jitter ±{jitter_mm:.0f} mm · SNAP_N {SNAP_N} N "
            f"(assumed) · clearance {clearance * 1000:.0f} mm · seed {seed}"
            f"{'' if lessons and lessons.enabled else ' · lessons OFF'}")

    for i in range(n):
        name = row.names[i % len(row.names)]
        # ⚠️ Order matters and getting it wrong silently disables the jitter:
        # reset first, then move the fruit, then sync the solver to what is
        # actually there. A Reacher built before the reset throws the jitter
        # away, and the run still prints a jitter figure and still looks fine.
        reset_park(model, data, park)
        row.reset()
        if jitter_mm:
            row.jitter(rng, jitter_mm)
        mujoco.mj_forward(model, data)

        _, r = _attempt(model, data, row, name, lessons, speed, on_tick,
                        verbose, blind=blind, clearance=clearance,
                        park_q=park)
        log.append(r)
        print(_row_line(i + 1, r))
    return log


def run_sequence(model, data, row, park, seed=0, jitter_mm=0.0,
                 speed=DEFAULT_SPEED, lessons=None, on_tick=None,
                 verbose=False, blind=False, clearance=CLEARANCE):
    """Harvest every fruit in the row, one after another, without resetting.

    Closer to what the machine actually has to do, and a strictly harder test:
    the arm never returns to a clean scene, so a fruit knocked loose on pick two
    is still on the floor for pick five, and the plan has to account for a row
    that empties as it goes.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    reset_park(model, data, park)
    row.reset()
    if jitter_mm:
        row.jitter(rng, jitter_mm)
    mujoco.mj_forward(model, data)

    log = []
    _header(f"sequence · {len(row.names)} fruit, no reset between picks · "
            f"jitter ±{jitter_mm:.0f} mm")
    for i, name in enumerate(row.names):
        if not row.attached(name):
            print(f"  {i + 1:3d} {name:>6} {'-':>6} {'-':>6} {'-':>6} "
                  f"{0.0:7.2f} {'   -  '} {0.0:6.1f}  already off the plant")
            continue
        _, r = _attempt(model, data, row, name, lessons, speed, on_tick,
                        verbose, blind=blind, clearance=clearance,
                        park_q=park)
        log.append(r)
        print(_row_line(i + 1, r))
    return log


def report(log, lessons=None, clearance=CLEARANCE):
    """Score it. A pick is clean only if nothing else moved."""
    n = len(log)
    if not n:
        return 0

    flown = [r for r in log if r["refused"] is None]
    clean = [r for r in flown if r["in_bin"] and not r["lost"]
             and not r["disturbed"] and r["aborted"] is None]
    binned = sum(r["in_bin"] for r in flown)
    lost = sum(r["lost"] for r in log)
    moved = sum(r["disturbed"] for r in log)
    refused = sum(1 for r in log if r["refused"] is not None)
    aborted = sum(1 for r in log if r["aborted"] is not None)

    best = run = 0
    for r in log:
        ok = (r["refused"] is None and r["in_bin"] and not r["lost"]
              and not r["disturbed"] and r["aborted"] is None)
        run = run + 1 if ok else 0
        best = max(best, run)

    print(f"\n  {'-' * 66}")
    print(f"  planned         {len(flown)}/{n}"
          f"{f'   ({refused} refused — no route cleared the crop)' if refused else ''}")
    print(f"  grasped         {sum(r['grasped'] for r in flown)}/{len(flown)}")
    print(f"  stem broke      {sum(r['broke'] for r in flown)}/{len(flown)}")
    print(f"  in crate        {binned}/{len(flown)}")
    if aborted:
        print(f"  aborted         {aborted}   (guard tripped — pick dropped "
              f"on purpose)")
    print(f"\n  collateral      {lost} knocked off the plant, {moved} disturbed")
    print(f"  CLEAN PICKS     {len(clean)}/{n}   "
          f"({100 * len(clean) / n:.0f}%)   fruit in the crate, nothing else touched")
    print(f"  longest clean run: {best}")

    gaps = [r["clearance"] for r in flown if np.isfinite(r["clearance"])]
    if gaps:
        print(f"\n  closest approach to a fruit it was not picking: "
              f"{min(gaps) * 1000:.0f} mm   (planned for {clearance * 1000:.0f})")
    secs = [r["seconds"] for r in flown if r["seconds"] > 0]
    if secs:
        print(f"  cycle time      mean {np.mean(secs):.1f} s · "
              f"min {np.min(secs):.1f} · max {np.max(secs):.1f}")
    peaks = [r["peak_n"] for r in flown if r["peak_n"] > 0]
    if peaks:
        print(f"  peak stem force mean {np.mean(peaks):.2f} N · "
              f"max {np.max(peaks):.2f} N   (SNAP_N {SNAP_N} N, assumed)")

    hits = [h for r in log for h in r["incidents"]]
    if hits:
        print(f"\n  what went wrong ({len(hits)}):")
        for h in hits:
            print(f"    {h.explain()}")

    if lessons is not None and lessons.lessons:
        print(f"\n  lessons on file:")
        print(lessons.report())

    if best >= 10:
        print("\n  MILESTONE: ten consecutive clean picks.")
    else:
        print(f"\n  Milestone not met — needs 10 consecutive clean, "
              f"best was {best}.")
    return best


# --- running it --------------------------------------------------------------

def _go(model, data, row, park, args, lessons, on_tick=None):
    # ⚠️ A bare `week2_pick.py` used to run exactly one pick and quit, because
    # --trials defaulted to 1. That is a fine default for a scripted milestone
    # and a bad one for "run it and show me", which is what someone typing the
    # filename with no arguments means. Harvest the row instead; ask for
    # --trials explicitly when you want the jittered repeatability measurement.
    if args.sequence or args.trials is None:
        return run_sequence(model, data, row, park, seed=args.seed,
                            jitter_mm=args.jitter, speed=args.speed,
                            lessons=lessons, on_tick=on_tick,
                            verbose=args.verbose, blind=args.blind,
                            clearance=args.clearance / 1000.0)
    return run_trials(model, data, row, park, args.trials, args.jitter,
                      seed=args.seed, speed=args.speed, lessons=lessons,
                      on_tick=on_tick, verbose=args.verbose, blind=args.blind,
                      clearance=args.clearance / 1000.0)


def run_windowed(model, data, row, park, args, lessons):
    import mujoco.viewer

    warnings.filterwarnings("ignore", module="glfw")
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
            report(_go(model, data, row, park, args, lessons, tick), lessons,
                   args.clearance / 1000.0)
            # ⚠️ Hold the window open. `exit_without_teardown` is `os._exit`, so
            # returning from here kills the process mid-frame and the viewer
            # vanishes the instant the last pick lands — which reads as the sim
            # crashing, not finishing. Physics is deliberately not stepped: the
            # run is over, and stepping on with nothing commanding the servos
            # would just let the arm sag while you look at it.
            print("\n  run finished — the window stays open, close it to quit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\nwindow closed early")
    exit_without_teardown()


def run_recorded(model, data, row, park, args, lessons, out_path):
    import cv2
    import mujoco

    fps, width, height = 30, 1280, 960
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_every = max(1, int(round((1 / fps) / CTRL_DT)))
    cycles = [0]

    def tick(_t=None):
        cycles[0] += 1
        if cycles[0] % frame_every == 0:
            renderer.update_scene(data, camera=args.camera)
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    try:
        report(_go(model, data, row, park, args, lessons, tick), lessons,
                   args.clearance / 1000.0)
    finally:
        renderer.close()
        writer.release()
        print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=None,
                    help="how many picks to run against jittered fruit, "
                         "resetting the row between each (the milestone is 10). "
                         "With neither this nor --sequence, the whole row is "
                         "harvested once.")
    ap.add_argument("--sequence", action="store_true",
                    help="harvest the whole row without resetting between fruit")
    ap.add_argument("--jitter", type=float, default=20.0,
                    help="fruit position jitter in mm, per axis (default 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help="fraction of the FR5's rated joint speed, 0-1. "
                         "Default 0.15 (~27 deg/s) is watchable; 1.0 is the "
                         "real arm's 180 deg/s. The pull always stays slow — "
                         "see mission.PULL_SPEED.")
    ap.add_argument("--verbose", action="store_true",
                    help="print the plan and every leg of every pick")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--out", default=None, help="write an mp4 (implies headless)")
    ap.add_argument("--camera", default="row",
                    choices=["row", "aisle", "wide"], help="camera to record")
    ap.add_argument("--bare", action="store_true",
                    help="the old grey-floor rig instead of the greenhouse")
    ap.add_argument("--no-lessons", dest="lessons", action="store_false",
                    help="plan without anything learned so far — the baseline")
    ap.add_argument("--clearance", type=float, default=CLEARANCE * 1000,
                    help="how close the arm may come to a fruit it is not "
                         "picking, in mm (default 40). Drop it to watch the "
                         "planner take risks and the lesson store learn the "
                         "margins back from actual damage.")
    ap.add_argument("--blind", action="store_true",
                    help="plan against nominal fruit positions and execute "
                         "against the jittered ones — a preview of what "
                         "Week 3's camera error does to a verified plan")
    ap.add_argument("--forget", action="store_true",
                    help="clear the lesson store before running")
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene(greenhouse=not args.bare)
    data = mujoco.MjData(model)
    row = Row(model, data)
    park = park_posture(model)
    reset_park(model, data, park)

    lessons = Lessons.load(enabled=args.lessons)
    if args.forget:
        lessons.forget()

    print(f"\nFR5 + Robotiq 2F85 + plant row: {model.nq} DoF, {model.nu} "
          f"actuators, {model.neq} equality constraints")
    print(f"row at x={ROW_X:.2f}, crate at [{BIN_POS[0]:.2f} {BIN_POS[1]:.2f}], "
          f"arm parks at {data.site(TOOL_SITE).xpos.round(2)}")
    print(f"lessons: {len(lessons.lessons)} on file, "
          f"{'applied' if lessons.enabled else 'IGNORED (--no-lessons)'}")

    try:
        if args.out:
            run_recorded(model, data, row, park, args, lessons, args.out)
        elif args.headless:
            report(_go(model, data, row, park, args, lessons), lessons,
                   args.clearance / 1000.0)
        else:
            run_windowed(model, data, row, park, args, lessons)
    finally:
        if args.lessons or args.forget:
            path = lessons.save()
            print(f"  lessons written to {path}")


if __name__ == "__main__":
    main()
