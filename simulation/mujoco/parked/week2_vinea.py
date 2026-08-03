#!/usr/bin/env python3
"""Week 2, Step 6 — the same row, picked with the Vinea gripper instead.

`week2_pick.py` is the milestone and it uses the Robotiq. This is the tool the
MVP spec actually calls for: a passive cradle that supports the truss from
below, and a blade that severs the peduncle. One moving part against the 2F85's
eight.

The cycle changes shape because the tool does:

    Robotiq   approach -> insert -> CLOSE -> PULL until the stem tears -> carry
    Vinea     approach -> insert -> CUT -> lift away -> carry -> tip out

There is no grip and no pull. The fruit is never squeezed, so the three worst
problems of the week simply do not arise — the close is not an impact, nothing
gets squeezed out under rotation, and `SNAP_N` stops mattering because the
peduncle is cut rather than torn. That last one is the real argument for this
design: with the Robotiq, `SNAP_N` had a floor imposed by the gripper itself
(6.0-8.6 N just to hold the fruit), so the threshold was never freely
choosable. A blade removes the coupling.

Usage:

    ./.venv/bin/python simulation/mujoco/week2_vinea.py                 # window
    ./.venv/bin/python simulation/mujoco/week2_vinea.py --headless --trials 5
    ./.venv/bin/python simulation/mujoco/week2_vinea.py --headless --out week2_vinea.mp4
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
    TOOL_SITE,
    add_crate,
    build_fr5_spec,
    crate_contains,
    exit_without_teardown,
    reset_home,
)
from plant_row import FRUIT_R, ROW_X, Row, add_row  # noqa: E402
from reach import CTRL_DT, DEFAULT_SPEED, Reacher, hold  # noqa: E402
from vinea_gripper import Blade, add_vinea_gripper  # noqa: E402
from week2_pick import (  # noqa: E402
    BIN_DROP_UP,
    BIN_HALF,
    BIN_POS,
    BIN_WALL,
    DOWNWARD,
    INTO_ROW,
    ORIENTATION_COST,
    ROLL_COST,
    SETTLE_S,
    STAGE_X,
    TURN_COST,
    TURN_MAX_S,
)

# Reach to the cradling point, measured the same way as MAX_REACH_GRIPPER:
# 20k random joint configurations, largest distance from the shoulder. It is
# 50 mm shorter than the 2F85's because the cradle point sits closer in than
# the fingertips. Not derivable from the mount offset — see vinea_gripper.py.
MAX_REACH_VINEA = 1.050

APPROACH_GAP = 0.12     # start this far back along -x, blade retracted
# How far below the fruit the cradle slides in before lifting. The channel rim
# sits 20 mm above the tool point, so this has to clear the fruit's underside
# (33 mm) plus the rim plus room for the arm's own droop.
UNDER_DROP = 0.10
LIFT_OFF = 0.06         # lift clear of the cut stem before backing out
CUT_S = 1.0             # blade stroke time
SETTLE_CRADLE_S = 0.6   # let the fruit settle into the channel after the cut

# Arrival tolerance. ⚠️ Re-measured, not inherited: droop scales with what is
# hanging off the flange, and this tool is 1.20 kg against the Robotiq's 1.05.
# See the printout in main().
REACHED_MM_LOADED = 9.0

# Tipping the fruit out. The channel is open along the approach axis, so
# pointing the tool down empties it — no release actuator, because there is
# nothing to release. This is what "passive, zero DOF" buys.
TIP_DONE_DEG = 20.0
VINEA_DROP_UP = 0.45    # release height above the crate; see the carry step
CARRY_SETTLE_S = 0.8    # keep converging on the crate before tipping

# The roll matters for this tool. See make_reacher.
ROLL_COST_CRADLE = 0.10


def build_scene():
    """FR5 + the Vinea gripper + the plant row + a crate."""
    spec = build_fr5_spec(gripper=False)
    add_vinea_gripper(spec)
    add_row(spec)
    add_crate(spec, name="bin", pos=BIN_POS, half=BIN_HALF, wall=BIN_WALL)
    spec.worldbody.add_camera(
        name="row", pos=[0.28, -1.85, 1.20],
        xyaxes=[0.9949, -0.1013, 0.0, 0.0399, 0.3922, 0.9190])
    return spec.compile()


def make_reacher(model, data, speed=DEFAULT_SPEED):
    """⚠️ Unlike the Robotiq cycle, the roll is *constrained* here.

    `week2_pick.py` leaves it nearly free (`roll_cost=0.01`) because pinning an
    arbitrary roll traps the solver, and two parallel pads closing on a sphere
    do not care which way up they are. A cradle cares completely: the roll is
    what decides which way the open side of the channel faces.

    Left free, the solver picked its own — measured, the blade came out 50 mm
    along world -y instead of 50 mm up, i.e. the whole tool rotated 90°, so the
    channel floor was acting as a side wall and a wall clipped the fruit on the
    way in (17.4 N through the peduncle, over SNAP_N, stem gone before the
    blade moved). With roll=0 and the approach along +x, the tool's local +y
    maps to world +z, which puts the floor under the fruit where it belongs.
    """
    return Reacher(model, data, speed=speed, standoff=0.0,
                   max_reach=MAX_REACH_VINEA, reached_mm=REACHED_MM_LOADED,
                   orientation_cost=ORIENTATION_COST,
                   roll_cost=ROLL_COST_CRADLE,
                   approach=INTO_ROW, mocap=None)


def pick_one(reacher, blade, row, name, on_tick=None, verbose=True):
    """One cradle-and-cut pick."""
    data = reacher.data
    fruit = data.body(name)
    t0 = [0.0]
    reacher.on_substep = row.update      # stems can still be overloaded

    def tick(_t=None):
        t0[0] += CTRL_DT
        if on_tick is not None:
            on_tick(_t)

    def say(state, note=""):
        if verbose:
            p = data.site(TOOL_SITE).xpos
            print(f"  {state:<9} tool [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]"
                  f"   {note}")

    def go(state, target, note=""):
        r = reacher.drive_to(target, on_tick=tick)
        arrived = (f"arm {r['arm_mm']:.1f} mm · {r['axis_deg']:.1f}°"
                   if r["reached"] else
                   f"DID NOT ARRIVE — {r['arm_mm']:.0f} mm short")
        say(state, f"{arrived}   {note}".rstrip())
        return r

    blade.retract()
    park = data.site(TOOL_SITE).xpos.copy()
    for _ in range(int(SETTLE_S / CTRL_DT)):
        reacher.step(park)
        tick()
    row.reset()
    target = row.pos(name)

    # Same three-move entry as the Robotiq cycle: the row is still in the way
    # and drive_to still has no path planner.
    go("retract", np.array([STAGE_X, park[1], park[2]]))
    go("traverse", np.array([STAGE_X, target[1], target[2] - UNDER_DROP]))

    # Slide in *underneath* the truss, with the open channel below the fruit
    # and the rim passing clear of it, then lift into the fruit. This is the
    # entire difference between a cradle and a gripper: nothing closes, the
    # tool just arrives under the load and takes its weight.
    go("under", np.array([ROW_X, target[1], target[2] - UNDER_DROP]))
    go("cradle", target)
    say("cradle", f"stem carrying {row.force(name):.2f} N")

    # The cut. Geometry moves; the actual detachment is one line.
    blade.cut()
    cut = False
    for _ in range(int(CUT_S / CTRL_DT)):
        reacher.step(target)
        tick()
        if not cut and blade.severed(row.stem_pos(name)):
            data.eq_active[row.eq_id[name]] = 0
            cut = True
    say("cut", f"blade at {blade.travel * 1000:.0f} mm — "
               f"{'peduncle severed' if cut else 'NOT SEVERED'}")

    hold(reacher, target, SETTLE_CRADLE_S, tick)

    # Lift a little, so the fruit clears the cut stem, then back out level.
    go("lift", target + np.array([0, 0, LIFT_OFF]))
    go("retreat", np.array([STAGE_X, target[1], target[2] + LIFT_OFF]))
    held = float(np.linalg.norm(fruit.xpos - data.site(TOOL_SITE).xpos))
    carried = held < FRUIT_R * 2.5
    say("check", f"fruit {held * 1000:.0f} mm from the cradle point — "
                 f"{'carried' if carried else 'LOST'}")

    # Carry level — no wrist rotation while the fruit is only resting in a
    # channel — then tip it out over the crate.
    #
    # ⚠️ Higher than the Robotiq's release point, and not for tidiness. The
    # 2F85 cycle turns the wrist down *before* translating, which frees the IK
    # to drop to 0.28 m. A cradle cannot do that: turning is what empties it.
    # Carrying level, the arm tops out around 0.46 m above the crate and the
    # move stopped 208 mm short. Release from where it can actually reach.
    over_bin = BIN_POS + np.array([0, 0, VINEA_DROP_UP])
    go("carry", over_bin)

    # Keep converging before tipping. The stall check gives up on this long
    # swing while the arm is still ~200 mm out, and the tip loop that follows
    # translates *and* rotates — so a cradle that has not arrived yet empties
    # itself somewhere between the row and the crate.
    hold(reacher, over_bin, CARRY_SETTLE_S, tick)
    say("settle", f"{np.linalg.norm(data.site(TOOL_SITE).xpos - over_bin) * 1000:.0f} mm "
                  f"from the release point")

    reacher.approach = DOWNWARD
    reacher.set_orientation_cost(TURN_COST)
    tipped = 0.0
    while reacher.tool_tilt_deg() > TIP_DONE_DEG and tipped < TURN_MAX_S:
        reacher.step(over_bin)
        tick()
        tipped += CTRL_DT
    reacher.set_orientation_cost(ORIENTATION_COST)
    say("tip", f"cradle to {reacher.tool_tilt_deg():.1f}° — fruit tipped out")
    hold(reacher, over_bin, 1.0, tick)

    reacher.set_orientation_cost(0.0)
    go("home", park)
    hold(reacher, park, 0.4, tick)

    in_bin = crate_contains(fruit.xpos, BIN_POS, BIN_HALF, BIN_WALL)
    return {"fruit": name, "target": target.copy(), "cut": bool(cut),
            "carried": bool(carried), "in_bin": bool(in_bin),
            "seconds": t0[0], "final": fruit.xpos.copy()}


def run_trials(model, data, n, jitter_mm, seed=0, speed=DEFAULT_SPEED,
               on_tick=None, verbose=False):
    import mujoco

    rng = np.random.default_rng(seed)
    row = Row(model, data)
    log = []
    print(f"  {n} attempts · jitter ±{jitter_mm:.0f} mm · Vinea cradle+blade\n")
    print(f"  {'#':>3} {'fruit':>6} {'target xyz':>24} {'cut':>5} "
          f"{'carried':>8} {'crate':>6} {'secs':>6}")

    for i in range(n):
        name = row.names[i % len(row.names)]
        reacher = make_reacher(model, data, speed=speed)
        blade = Blade(model, data)
        reset_home(model, data)
        row.reset()
        if jitter_mm:
            row.jitter(rng, jitter_mm)
        mujoco.mj_forward(model, data)
        reacher.config.update(data.qpos[: model.nq].copy())
        reacher.posture.set_target_from_configuration(reacher.config)

        r = pick_one(reacher, blade, row, name, on_tick=on_tick,
                     verbose=verbose)
        log.append(r)
        t = r["target"]
        print(f"  {i + 1:3d} {name:>6} [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] "
              f"{'yes' if r['cut'] else 'NO':>5} "
              f"{'yes' if r['carried'] else 'NO':>8} "
              f"{'yes' if r['in_bin'] else 'NO':>6} {r['seconds']:6.1f}")
    return log


def report(log):
    n = len(log)
    if not n:
        return 0
    binned = [r["in_bin"] for r in log]
    best = run = 0
    for ok in binned:
        run = run + 1 if ok else 0
        best = max(best, run)
    secs = [r["seconds"] for r in log]
    print(f"\n  {'-' * 58}")
    print(f"  peduncle cut  {sum(r['cut'] for r in log)}/{n}")
    print(f"  carried       {sum(r['carried'] for r in log)}/{n}")
    print(f"  in crate      {sum(binned)}/{n}   ({100 * sum(binned) / n:.0f}%)")
    print(f"  longest consecutive run: {best}")
    print(f"  cycle time    mean {np.mean(secs):.1f} s · "
          f"min {np.min(secs):.1f} · max {np.max(secs):.1f}")
    return best


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
            report(run_trials(model, data, args.trials, args.jitter,
                              seed=args.seed, speed=args.speed, on_tick=tick,
                              verbose=args.verbose))
        except KeyboardInterrupt:
            print("\nwindow closed early")
    exit_without_teardown()


def run_recorded(model, data, args, out_path):
    import cv2
    import mujoco

    fps, width, height = 30, 1280, 960
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_every = max(1, int(round((1 / fps) / CTRL_DT)))
    cycles = [0]

    def tick(_t=None):
        cycles[0] += 1
        if cycles[0] % frame_every == 0:
            renderer.update_scene(data, camera="row")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    try:
        report(run_trials(model, data, args.trials, args.jitter,
                          seed=args.seed, speed=args.speed, on_tick=tick,
                          verbose=args.verbose))
    finally:
        renderer.close()
        writer.release()
        print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene()
    data = mujoco.MjData(model)

    mass = sum(model.body(i).mass[0] for i in range(model.nbody)
               if model.body(i).name.startswith("vg_"))
    print(f"\nFR5 + Vinea cradle/blade: {model.nq} DoF, {model.nu} actuators, "
          f"tool {mass:.2f} kg")
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
