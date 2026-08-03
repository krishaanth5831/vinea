#!/usr/bin/env python3
"""The unplanned Week 2 cycle, kept so the planner has something to beat.

This is the original flat script: drive the tool at each waypoint in turn and
let IK choose the route. Its behaviour is preserved exactly, because it is the
baseline every claim in `week2_pick.py` is measured against, and a baseline you
keep editing is not one.

Measured over ten picks, jitter +/-20 mm, seed 0:

    in crate            10/10   (100%)
    neighbours lost      8/10   <- never counted, because nothing looked

That second line is why `mission.py`, `incident.py` and `lessons.py` exist. The
failure is the return leg: the arm's home posture puts tool0 at x=0.671 and the
fruit hang at x=0.60, so driving home from the crate cuts the corner through
the canopy and strips whatever is in it.

Run it to reproduce the baseline:

    ./.venv/bin/python simulation/mujoco/legacy_cycle.py --trials 10 --headless
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
    reset_home,
)
from plant_row import FRUIT_R, ROW_X, SNAP_N, Row  # noqa: E402
from greenhouse import build_scene  # noqa: E402
from reach import CTRL_DT, DEFAULT_SPEED, Gripper, Reacher, hold  # noqa: E402

BIN_POS = np.array([0.30, -0.52, 0.0])
BIN_HALF = 0.16
BIN_WALL = 0.12

# --- the cycle ---------------------------------------------------------------
# The arm comes at the row along +x, so every waypoint is expressed as an offset
# from the fruit in that frame. tool0 sits between the fingertips, so "go to the
# fruit" means exactly that — the pads end up either side of it.
APPROACH_GAP = 0.10     # start this far back along -x, fingers open

# How far in front of the row the arm stages before moving sideways.
#
# ⚠️ This is not tidiness, it is the difference between 4/10 and a working
# cycle. `drive_to` has no path planning — it drives the tool at the target and
# lets IK choose the route — and the arm's home posture puts tool0 at x=0.672,
# which is *past* the fruit at x=0.60 and therefore inside the row. So every
# move from home swept the gripper through the plants: measured per phase, the
# peduncles of t1 and t3 saw 75 N and 83 N during *approach*, tearing the fruit
# off before the gripper had closed on anything. The pick then reported "stem
# broke: yes" and dropped a tomato it never held.
#
# Retracting to this plane first, traversing sideways out here where there is
# nothing to hit, and only then driving straight in along +x keeps the gripper
# out of the canopy except on the one axis it has to enter along.
STAGE_X = ROW_X - 0.28
PULL_DOWN = 0.14        # how far to pull down to load the peduncle
RETREAT = 0.18          # back out of the row before turning for the crate
BIN_DROP_UP = 0.28      # release height above the crate

GRIP_S = 0.8            # hold still after the fingers arrive, to settle contacts
# Reorienting the wrist in place before the carry. See step 6.
TURN_COST = 0.6         # strong, because nothing is competing with it here
TURN_MAX_S = 3.5        # give up rather than stall the cycle

# How far to turn, and why it is not "all the way to vertical".
#
# ⚠️ A 2F85 holding a sphere between two flat pads loses it if you rotate far
# enough. Menagerie gives the pads rolling friction 0.0, so as the wrist turns,
# gravity's component along the pad face rolls the fruit toward the fingertips
# — and the tendon, still commanded shut, closes in behind it and squeezes it
# out. Watched live, the pad gap ran 76 -> 68 -> 40 -> 15 mm and the tomato was
# gone by 20° of tilt. Grip force is not the problem: the pads carry 99.8 N of
# normal force against 1.18 N of fruit weight, a 59x margin.
#
# The fruit survives down to about 22°. Stopping at 30° keeps it and still
# gives IK enough freedom to reach the crate; going to 45° or 60° keeps the
# fruit but leaves the wrist too level for the arm to get across (measured,
# the carry ended 421 mm short from the +y end of the row).
TURN_DONE_DEG = 30.0
SETTLE_S = 0.3          # let the weld find its resting force before picking
PULL_SPEED = 0.05       # slower than the rest of the cycle; see step 4
PULL_MAX_S = 6.0        # give up pulling rather than tear the plant apart
SETTLE_GRASP_S = 0.5    # let the fruit settle in the pads after the stem parts

# How long the fingers take to shut, and how soft they are when they land.
#
# Both exist to stop the *grip* breaking the stem. Commanded shut in one step
# against Menagerie's stiff pads, closing on a hanging fruit spiked the peduncle
# to 164 N — twenty-seven times SNAP_N — so every stem snapped on contact and
# the "pull" that followed was picking up an already-loose tomato. Ramping the
# command (which is what a real Robotiq's speed setting does) and softening the
# pads take that peak to about 6 N, near the 3.8 N the grip settles at.
GRIP_RAMP_S = 1.5

# Arrival tolerance for the loaded arm — the 1.05 kg gripper droops ~5.2 mm and
# a P+D position servo cannot null out a constant gravity load. Same 8 mm as
# Week 1; it still grasps a 66 mm fruit with room to spare.
REACHED_MM_LOADED = 8.0

# The approach direction, and the two weights behind it.
#
# Fingers point +x: straight into the row, along the row normal. Coming in
# level means the gripper reaches *past* the peduncle rather than through it,
# and — measured — the pads then close along world y, around the fruit's sides,
# leaving the stem above untouched. A top-down approach closes across the stem.
#
# orientation_cost 0.1 against position_cost 1.0 is not timidity. FrameTask
# sums position error in metres against orientation error in radians, so at
# equal weights a 50° error (0.87 rad) outranks half a metre of position error
# and the solver straightens the wrist first — swinging the arm 685 mm off
# course, which in this scene means through the plants. At 0.1 position leads
# and the angle arrives on the way: 0.3-3.2° at the grasp, against 47-52° in
# Week 1, while position error actually *improved* from ~4.6 mm to ~3.9 mm.
INTO_ROW = np.array([1.0, 0.0, 0.0])
DOWNWARD = np.array([0.0, 0.0, -1.0])   # for carrying to the crate; see step 6
ORIENTATION_COST = 0.1
ROLL_COST = 0.01        # just enough to stop the wrist spinning on the way in


def make_reacher(model, data, speed=DEFAULT_SPEED):
    """A Reacher configured for reaching into a row.

    ⚠️ mocap=None matters. The row's stems *are* mocap bodies, and the Reacher
    parks a marker on mocap 0 by default — which would drag stem t0 around the
    scene every control tick, welded tomato and all.
    """
    return Reacher(model, data, speed=speed, standoff=0.0,
                   max_reach=MAX_REACH_GRIPPER,
                   reached_mm=REACHED_MM_LOADED,
                   orientation_cost=ORIENTATION_COST,
                   roll_cost=ROLL_COST,
                   approach=INTO_ROW,
                   mocap=None)


def pick_one(reacher, gripper, row, name, on_tick=None, verbose=True,
             box=None):
    """One full pick of one hanging fruit. Returns what happened.

    The state machine is Week 1's with a pull added:

        approach -> insert -> close -> pull (stem snaps) -> retreat
                 -> carry -> release -> home

    `pull` is the new state and the only one that touches the plant. Everything
    else is the Week 1 sequence, which is the point — Week 4 needs this to be a
    state machine it can re-enter after a failure, not a longer flat script.
    """
    data = reacher.data
    fruit = data.body(name)
    t0 = [0.0]

    # Snapping is checked after every physics step, not every control cycle.
    # See Reacher.on_substep — at 100 Hz the force overshot 6 N to 77 N before
    # anything noticed.
    # `box`, when given, wraps row.update and records why anything fell.
    reacher.on_substep = row.update if box is None else box.substep

    def tick(_t=None):
        """Every control cycle: age the clock and drive the display."""
        t0[0] += CTRL_DT
        if on_tick is not None:
            on_tick(_t)

    def say(state, note=""):
        if verbose:
            p = data.site(TOOL_SITE).xpos
            print(f"  {state:<9} tool [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]"
                  f"   {note}")

    def phase(name):
        """Tell the black box which move is running, so it can name the culprit.

        Without this every incident is attributed to leg "?" — the recorder
        knows a fruit was struck and by what, but not during which move, which
        is the one field a fix actually keys on.
        """
        if box is not None:
            box.leg = name

    def go(state, target, note=""):
        phase(state)
        r = reacher.drive_to(target, on_tick=tick)
        arrived = (f"arm {r['arm_mm']:.1f} mm · {r['axis_deg']:.1f}°"
                   if r["reached"] else
                   f"DID NOT ARRIVE — {r['arm_mm']:.0f} mm short")
        say(state, f"{arrived}   {note}".rstrip())
        return r

    home_tool = data.site(TOOL_SITE).xpos.copy()
    target = row.pos(name)

    # 0. Let the weld settle. It overshoots for ~50 ms on startup and the peak
    #    is above SNAP_N — snap during that and the crop falls off the plant
    #    before the arm has moved.
    #
    #    Hold the *current* pose while it does. Stepping toward the fruit here
    #    starts the arm moving before the row has settled, and since home is
    #    inside the row that lurch alone loaded a peduncle to 52 N.
    gripper.open()
    phase("settle")
    park = data.site(TOOL_SITE).xpos.copy()
    for _ in range(int(SETTLE_S / CTRL_DT)):
        reacher.step(park)
        tick()
    row.reset()

    # 1. Retract clear of the row, then traverse to the fruit's line, then
    #    drive straight in. Three moves instead of one, because the arm has no
    #    path planner and the row is in the way — see STAGE_X.
    go("retract", np.array([STAGE_X, park[1], park[2]]))
    go("traverse", np.array([STAGE_X, target[1], target[2]]))
    go("approach", target + np.array([-APPROACH_GAP, 0, 0]))

    # 2. Insert — tool0 is between the fingertips, so drive it to the fruit.
    go("insert", target)

    # 3. Close on the fruit — ramped, not commanded shut in one step. See
    #    GRIP_RAMP_S: a step input here is an impact, and the impact broke the
    #    peduncle before the arm ever pulled on it.
    phase("close")
    gripper.ramp(reacher, target, GRIP_RAMP_S, on_tick=tick)
    hold(reacher, target, GRIP_S, tick)
    say("close", f"fingers commanded shut · stem carrying "
                 f"{row.force(name):.2f} N")

    # 4. Pull. This is the harvest: load the peduncle until it lets go.
    #
    #    Slower than the rest of the cycle, because this is the one move whose
    #    *force* matters. At full speed one control cycle moves the setpoint
    #    0.0047 rad, which against kp=4000 is ~48 N at the tool — a 6 N
    #    threshold cannot be resolved inside a 48 N step, and the fruit tears
    #    off at whatever force the first sample happens to read.
    #    Stop the moment it lets go. Driving the full PULL_DOWN regardless
    #    means the arm keeps yanking after the peduncle has already parted,
    #    and because the weld is compliant it has stretched by then — so the
    #    release fires that stored energy into a fruit held only by two pads.
    #    Measured, it knocked the tomato 22 mm off the pinch centre, and a
    #    fruit sitting on the edge of the pads is one that falls out during the
    #    carry: those attempts ended up on the floor, one of them 5 m away.
    #    A picker stops pulling when the fruit comes free, so this one does.
    phase("pull")
    cycle_speed = reacher.speed
    reacher.set_speed(PULL_SPEED)
    pull_to = target + np.array([0, 0, -PULL_DOWN])
    pulled = 0.0
    while row.attached(name) and pulled < PULL_MAX_S:
        reacher.step(pull_to)
        tick()
        pulled += CTRL_DT
    broke = not row.attached(name)

    # Let the fruit settle into the pads before moving anywhere.
    hold(reacher, data.site(TOOL_SITE).xpos.copy(), SETTLE_GRASP_S, tick)
    reacher.set_speed(cycle_speed)
    say("pull", f"peak {row.peak[name]:.2f} N vs SNAP_N {row.snap_n:.1f} N — "
                f"{'stem released' if broke else 'STILL ATTACHED'}")

    # 5. Straight back out to the staging plane before turning, so the fruit
    #    in the gripper does not sweep the plants on its way to the crate.
    go("retreat", np.array([STAGE_X, target[1], target[2] - PULL_DOWN]))
    held = float(np.linalg.norm(fruit.xpos - data.site(TOOL_SITE).xpos))
    grasped = held < FRUIT_R * 2.0
    say("check", f"fruit {held * 1000:.0f} mm from the tool — "
                 f"{'holding' if grasped else 'DROPPED'}")

    # 6. Carry to the crate and let go. Point the fingers down from here on:
    #    holding them level into the row is right for reaching a plant and
    #    wrong for everything after, and pinned that way the arm cannot get
    #    across to the crate at all — measured, it stopped 199 mm short.
    #    Releasing downwards also drops the fruit into the crate rather than
    #    flicking it sideways.
    #    Turn the wrist *in place* first, out here where there is nothing to
    #    hit, then translate. Asking for both at once makes the solver trade
    #    one against the other over a half-metre move and it takes a bad
    #    route — measured, the tool ended 325 mm past the crate at [-0.02,
    #    -0.60, 0.30] whenever the fruit came from the +y end of the row.
    #    Turn it *hard*: 0.1 is deliberately too weak to lead a move, and at
    #    that weight the wrist rotated only 7° in 1.2 s and the carry set off
    #    still horizontal. There is no position to trade against while holding
    #    station, so the objection to a high weight does not apply here. Put it
    #    back before translating.
    phase("turn")
    reacher.approach = DOWNWARD
    reacher.set_orientation_cost(TURN_COST)
    out = np.array([STAGE_X, target[1], target[2] - PULL_DOWN])
    turned = 0.0
    while reacher.tool_tilt_deg() > TURN_DONE_DEG and turned < TURN_MAX_S:
        reacher.step(out)
        tick()
        turned += CTRL_DT
    reacher.set_orientation_cost(ORIENTATION_COST)
    say("turn", f"wrist to {reacher.tool_tilt_deg():.1f}° off vertical "
                f"in {turned:.1f}s")

    over_bin = BIN_POS + np.array([0, 0, BIN_DROP_UP])
    go("carry", over_bin)
    phase("release")
    gripper.open()
    hold(reacher, over_bin, GRIP_S, tick)
    say("release", "fingers opened")

    # 7. Back to a known posture, so the next cycle starts where this one did.
    #    Orientation off: the gripper is empty, nothing about the wrist angle
    #    matters any more, and insisting on one makes the home pose — which
    #    the arm sits in naturally at ~50° — unreachable, leaving it stranded
    #    150 mm short every cycle.
    reacher.set_orientation_cost(0.0)
    go("home", home_tool)
    hold(reacher, home_tool, 0.4, tick)

    in_bin = crate_contains(fruit.xpos, BIN_POS, BIN_HALF, BIN_WALL)
    return {
        "fruit": name,
        "target": target.copy(),
        "grasped": bool(grasped),
        "broke": bool(broke),
        "in_bin": bool(in_bin),
        "peak_n": float(row.peak[name]),
        "seconds": t0[0],
        "final": fruit.xpos.copy(),
    }


def run_trials(model, data, n, jitter_mm, seed=0, speed=DEFAULT_SPEED,
               on_tick=None, verbose=False):
    """Run the cycle N times against jittered fruit and score it.

    The per-attempt log is the direct input to Week 4's kg/hr figure, which is
    what the sprint is actually for, so the columns are the ones that number
    needs: where the fruit was, whether each stage worked, and how long it took.
    """
    rng = np.random.default_rng(seed)
    row = Row(model, data)
    log = []

    print(f"  {n} attempts · jitter ±{jitter_mm:.0f} mm · SNAP_N {SNAP_N} N "
          f"(assumed) · seed {seed}\n")
    print(f"  {'#':>3} {'fruit':>6} {'target xyz':>24} {'grasp':>6} "
          f"{'stem':>6} {'crate':>6} {'peak N':>7} {'secs':>6}")

    import mujoco

    for i in range(n):
        name = row.names[i % len(row.names)]

        # ⚠️ Order matters, and getting it wrong silently disables the jitter.
        # Reacher.__init__ calls reset(), which calls reset_home(), which
        # calls mj_resetData — so a Reacher built *after* jittering throws the
        # jitter away and every attempt runs against the same coordinates. It
        # still prints a jitter figure, and the run still looks like a pass.
        # Build the arm first, then move the fruit, then sync the solver to
        # what is actually there.
        reacher = make_reacher(model, data, speed=speed)
        gripper = Gripper(model, data)

        reset_home(model, data)
        row.reset()
        if jitter_mm:
            row.jitter(rng, jitter_mm)
        mujoco.mj_forward(model, data)

        reacher.config.update(data.qpos[: model.nq].copy())
        reacher.posture.set_target_from_configuration(reacher.config)

        r = pick_one(reacher, gripper, row, name, on_tick=on_tick,
                     verbose=verbose)
        log.append(r)
        t = r["target"]
        print(f"  {i + 1:3d} {name:>6} [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] "
              f"{'yes' if r['grasped'] else 'NO':>6} "
              f"{'yes' if r['broke'] else 'NO':>6} "
              f"{'yes' if r['in_bin'] else 'NO':>6} "
              f"{r['peak_n']:7.2f} {r['seconds']:6.1f}")

    return log


def report(log):
    """Success rate, cycle time, and the longest consecutive run."""
    n = len(log)
    if not n:
        return
    binned = [r["in_bin"] for r in log]
    best = run = 0
    for ok in binned:
        run = run + 1 if ok else 0
        best = max(best, run)

    print(f"\n  {'-' * 62}")
    print(f"  grasped      {sum(r['grasped'] for r in log)}/{n}")
    print(f"  stem broke   {sum(r['broke'] for r in log)}/{n}")
    print(f"  in crate     {sum(binned)}/{n}   "
          f"({100 * sum(binned) / n:.0f}%)")
    print(f"  longest consecutive run: {best}")
    secs = [r["seconds"] for r in log]
    print(f"  cycle time   mean {np.mean(secs):.1f} s · "
          f"min {np.min(secs):.1f} · max {np.max(secs):.1f}")
    peaks = [r["peak_n"] for r in log]
    print(f"  peak stem force  mean {np.mean(peaks):.2f} N · "
          f"max {np.max(peaks):.2f} N   (SNAP_N {SNAP_N} N, assumed)")
    if best >= 10:
        print("\n  MILESTONE: ten consecutive picks.")
    else:
        print(f"\n  Milestone not met — needs 10 consecutive, best was {best}.")
    return best


# --- running it --------------------------------------------------------------

def run_windowed(model, data, args):
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
            log = run_trials(model, data, args.trials, args.jitter,
                             seed=args.seed, speed=args.speed, on_tick=tick,
                             verbose=args.verbose)
            report(log)
        except KeyboardInterrupt:
            print("\nwindow closed early")
    exit_without_teardown()


def run_recorded(model, data, args, out_path):
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
            renderer.update_scene(data, camera="row")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    try:
        log = run_trials(model, data, args.trials, args.jitter, seed=args.seed,
                         speed=args.speed, on_tick=tick, verbose=args.verbose)
        report(log)
    finally:
        renderer.close()
        writer.release()
        print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1,
                    help="how many picks to run (the milestone is 10)")
    ap.add_argument("--jitter", type=float, default=20.0,
                    help="fruit position jitter in mm, per axis (default 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    ap.add_argument("--verbose", action="store_true",
                    help="print every state of every pick")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--out", default=None, help="write an mp4 (implies headless)")
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene(greenhouse=False)
    data = mujoco.MjData(model)

    print(f"\nFR5 + Robotiq 2F85 + plant row: {model.nq} DoF, "
          f"{model.nu} actuators, {model.neq} equality constraints")
    print(f"row at x={ROW_X:.2f}, crate at "
          f"[{BIN_POS[0]:.2f} {BIN_POS[1]:.2f}]\n")

    if args.out:
        run_recorded(model, data, args, args.out)
    elif args.headless:
        report(run_trials(model, data, args.trials, args.jitter,
                          seed=args.seed, speed=args.speed,
                          verbose=args.verbose))
    else:
        run_windowed(model, data, args)


if __name__ == "__main__":
    main()
