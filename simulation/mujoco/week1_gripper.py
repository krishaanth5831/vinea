#!/usr/bin/env python3
"""Week 1, Step 6 — the FR5 with a gripper on it, picking something up.

`week1_targetreach.py` proves the arm can put its tool where you ask. This adds
the last piece of Week 1: a gripper that opens and closes, and a state machine
that sequences a whole pick.

    approach → descend → close → lift → carry → release → home

That sequence is the point of the file. It is the same shape as the harvest
cycle in Week 4, which is why it is worth writing while it is eight lines long
and every position is hardcoded. Week 3 replaces the hardcoded tomato position
with one from the camera; Week 4 wraps it in retries and logging. The states do
not change.

**The gripper is a stand-in.** A Robotiq 2F85 is a two-finger industrial gripper
that pinches. The Vinea gripper cradles a truss from below and cuts the peduncle
with a blade, which is a different machine. Mounting this one first gets the
attachment code, the tool frame and the state machine working against a model
Menagerie already tuned, so that when the homemade geometry arrives in Week 2
there is only one new thing that can be wrong.

**What this does not prove.** A MuJoCo grasp is not a real grasp. Contact
parameters here are Menagerie's defaults against a smooth sphere; a real tomato
is soft, wet and bruises. Do not quote a grasp success rate from this file.

Usage:

    # watch it in a window
    ./.venv/bin/python simulation/mujoco/week1_gripper.py

    # record the weekly capture
    ./.venv/bin/python simulation/mujoco/week1_gripper.py --headless \
        --out week1.mp4

    # just open and close the gripper, no pick — for checking the actuator
    ./.venv/bin/python simulation/mujoco/week1_gripper.py --wave
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
    GRIPPER_CLOSED,
    MAX_REACH_GRIPPER,
    TOOL_SITE,
    add_crate,
    build_fr5_spec,
    crate_contains,
    exit_without_teardown,
    reach_fraction,
)
from reach import (  # noqa: E402
    CTRL_DT,
    DEFAULT_SPEED,
    Gripper,
    Reacher,
    hold,
)

# --- the scene ---------------------------------------------------------------
# Deliberately crude. Week 2 builds the real greenhouse row with hanging trusses
# and breakable stems; this is the least scenery that lets a pick happen at all.
TOMATO_R = 0.033           # a truss tomato is roughly 65 mm across
TOMATO_MASS = 0.12         # ~120 g, in the middle of the range for a vine tomato
TABLE_TOP = 0.45
TOMATO_POS = np.array([0.55, 0.15, TABLE_TOP + TOMATO_R])

BIN_POS = np.array([0.30, -0.50, 0.0])
BIN_HALF = 0.16            # half-width of the crate opening
BIN_WALL = 0.12            # how tall the walls stand

# Where the tool goes at each stage, relative to the fruit or the crate.
PREGRASP_UP = 0.14         # start this far above the tomato, gripper open
LIFT_UP = 0.20             # how far to raise it before carrying
BIN_DROP_UP = 0.30         # hold the fruit this far above the crate to release

# How long to hold still while the fingers travel. The 2f85 closes in well
# under a second; the extra time is so the video reads as a deliberate action
# rather than a glitch, and so contacts settle before the arm lifts.
GRIP_S = 1.0

# Arrival tolerance, loosened from reach.REACHED_MM's 5 mm. Not slop: the 2F85
# weighs 1.05 kg, and hanging that off the flange takes the arm's steady-state
# error from ~3.2 mm to ~5.2 mm, because a P+D position servo cannot null out a
# constant gravity load. The droop is systematic — the arm sags the same way
# every time — which is why an 8 mm tolerance still grasps a 66 mm tomato with
# room to spare. Week 2 can shrink it back with gravity compensation or by
# leaning on the gains; leaving it visible is more useful than hiding it.
REACHED_MM_LOADED = 8.0


def add_table(spec):
    """Something for the tomato to sit on, at a height the arm works at."""
    import mujoco

    body = spec.worldbody.add_body(
        name="table", pos=[TOMATO_POS[0], TOMATO_POS[1], TABLE_TOP / 2])
    body.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.10, 0.10, TABLE_TOP / 2], rgba=[0.35, 0.30, 0.26, 1.0],
    )
    return body


def add_tomato(spec):
    """A free-floating sphere with mass. The thing being picked.

    A *free joint* is what makes it pickable. Without one the body is welded to
    the world and the gripper would squeeze something immovable; with one, it
    falls, rolls, and can be carried. It is also why this file adds 7 numbers to
    qpos — a free joint is 3 for position and 4 for the orientation quaternion.

    Week 2 replaces the free joint with a breakable weld to a stem, so the fruit
    hangs until the pull exceeds a threshold. That is the detachment model in the
    roadmap, and it is the one real difference between this and a harvest.
    """
    import mujoco

    body = spec.worldbody.add_body(name="tomato", pos=TOMATO_POS.tolist())
    body.add_freejoint()
    body.add_geom(
        name="tomato_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[TOMATO_R, 0, 0], rgba=[0.85, 0.16, 0.12, 1.0],
        mass=TOMATO_MASS,
        # Menagerie's finger pads set priority=1, so their friction wins on any
        # contact with them and this value only governs tomato-table and
        # tomato-crate. Left at something plausible for wet fruit on plastic.
        friction=[0.6, 0.01, 0.001],
    )
    return body


def build_scene():
    """FR5 + 2F85 + a table, a tomato and a crate."""
    spec = build_fr5_spec(gripper=True)
    add_table(spec)
    add_tomato(spec)
    add_crate(spec, name="bin", pos=BIN_POS, half=BIN_HALF, wall=BIN_WALL)
    return spec.compile()


# --- the pick cycle ----------------------------------------------------------

def run_pick(reacher, gripper, on_tick=None, verbose=True):
    """One full pick, start to finish. Returns a dict of what happened.

    Written as a flat sequence on purpose. Every step is "put the tool here" or
    "move the fingers", and reading it top to bottom should tell you exactly
    what the arm is about to do. When Week 4 turns this into something that
    retries and skips, this sequence becomes the body of the loop.
    """
    data = reacher.data
    tomato = data.body("tomato")

    def say(state, note=""):
        if verbose:
            pos = data.site(TOOL_SITE).xpos
            print(f"  {state:<9} tool [{pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}]"
                  f"   {note}")

    def go(state, target, note=""):
        result = reacher.drive_to(target, on_tick=on_tick)
        arrived = f"arm {result['arm_mm']:.1f} mm" if result["reached"] \
            else f"DID NOT ARRIVE — {result['arm_mm']:.0f} mm short"
        say(state, f"{arrived}   {note}".rstrip())
        return result

    home_tool = data.site(TOOL_SITE).xpos.copy()
    start_height = float(tomato.xpos[2])

    # 1. Approach — above the fruit, fingers open, touching nothing.
    gripper.open()
    above = TOMATO_POS + np.array([0, 0, PREGRASP_UP])
    go("approach", above, f"{reach_fraction(above, MAX_REACH_GRIPPER) * 100:.0f}% of reach")

    # 2. Descend — tool0 sits between the fingertips, so driving it to the
    #    centre of the tomato puts the fruit between the open pads.
    go("descend", TOMATO_POS)

    # 3. Close. The arm holds position while the fingers travel.
    gripper.close()
    hold(reacher, TOMATO_POS, GRIP_S, on_tick)
    say("close", f"fingers commanded to {GRIPPER_CLOSED:.0f}")

    # 4. Lift. This is the honest test of the grasp: if the fingers did not
    #    hold, the tomato stays on the table and only the arm goes up.
    lifted = TOMATO_POS + np.array([0, 0, LIFT_UP])
    go("lift", lifted)
    holding = float(tomato.xpos[2]) - start_height > LIFT_UP * 0.5
    say("check", "holding the tomato" if holding else "GRASP FAILED — fruit left behind")

    # 5. Carry to the crate and drop it in.
    over_bin = BIN_POS + np.array([0, 0, BIN_DROP_UP])
    go("carry", over_bin)
    gripper.open()
    hold(reacher, over_bin, GRIP_S, on_tick)
    say("release", "fingers opened")

    # 6. Out of the way, so the next cycle starts from a known posture.
    go("home", home_tool)
    hold(reacher, home_tool, 0.5, on_tick)

    in_bin = crate_contains(tomato.xpos, BIN_POS, BIN_HALF, BIN_WALL)
    return {"grasped": holding, "in_bin": in_bin,
            "tomato": tomato.xpos.copy()}


def run_wave(reacher, gripper, cycles=3, on_tick=None):
    """Just open and close the gripper, arm parked. The minimum Step 6 check."""
    target = reacher.data.site(TOOL_SITE).xpos.copy()
    for i in range(cycles):
        gripper.close()
        hold(reacher, target, 0.8, on_tick)
        print(f"  cycle {i + 1}: closed")
        gripper.open()
        hold(reacher, target, 0.8, on_tick)
        print(f"  cycle {i + 1}: open")


# --- running it --------------------------------------------------------------

def run_windowed(reacher, gripper, wave):
    """Same sequence, in a viewer, paced to wall-clock time."""
    import mujoco.viewer

    warnings.filterwarnings("ignore", module="glfw")
    model, data = reacher.model, reacher.data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        clock = [time.perf_counter()]

        def tick(_t):
            # One sync per control cycle would draw at 100 fps and outrun the
            # sim. Sleep the difference instead, so what you watch runs at the
            # speed the physics says it does.
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            clock[0] += CTRL_DT
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

        try:
            if wave:
                run_wave(reacher, gripper, on_tick=tick)
            else:
                report(run_pick(reacher, gripper, on_tick=tick))
        except KeyboardInterrupt:
            print("\nwindow closed early")

    exit_without_teardown()


def run_recorded(reacher, gripper, wave, out_path):
    """Same sequence, offscreen, written to an mp4."""
    import cv2
    import mujoco

    model, data = reacher.model, reacher.data
    fps, width, height = 30, 1280, 960
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_every = max(1, int(round((1 / fps) / CTRL_DT)))
    cycles = [0]

    def tick(_t):
        cycles[0] += 1
        if cycles[0] % frame_every == 0:
            renderer.update_scene(data, camera="demo")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    try:
        if wave:
            run_wave(reacher, gripper, on_tick=tick)
        else:
            report(run_pick(reacher, gripper, on_tick=tick))
    finally:
        renderer.close()
        writer.release()
        print(f"\nwrote {out_path}")


def report(result):
    print()
    if result["in_bin"]:
        print("  PICKED AND BINNED")
    elif result["grasped"]:
        p = result["tomato"]
        print(f"  grasped and carried, but the tomato ended at "
              f"[{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}] — not in the crate")
    else:
        print("  grasp failed — the fingers closed but the fruit stayed put")
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", action="store_true",
                    help="just open and close the gripper, no pick")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help=f"fraction of rated joint speed (default {DEFAULT_SPEED})")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--out", default=None, help="write an mp4 (implies headless)")
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene()
    data = mujoco.MjData(model)
    # standoff 0: with the gripper on, tool0 is already at the fingertips, so
    # every waypoint in run_pick is where the fingers should actually be. The
    # pre-grasp gap is an explicit step instead of a hidden offset.
    reacher = Reacher(model, data, speed=args.speed, standoff=0.0,
                      max_reach=MAX_REACH_GRIPPER,
                      reached_mm=REACHED_MM_LOADED)
    gripper = Gripper(model, data)
    gripper.open()

    print(f"FR5 + Robotiq 2F85: {model.nq} DoF, {model.nu} actuators, "
          f"reach {MAX_REACH_GRIPPER:.2f} m to the fingertips")
    print(f"tomato at [{TOMATO_POS[0]:.2f} {TOMATO_POS[1]:.2f} "
          f"{TOMATO_POS[2]:.2f}], crate at [{BIN_POS[0]:.2f} {BIN_POS[1]:.2f}]\n")

    if args.out:
        run_recorded(reacher, gripper, args.wave, args.out)
    elif args.headless:
        if args.wave:
            run_wave(reacher, gripper)
        else:
            report(run_pick(reacher, gripper))
    else:
        run_windowed(reacher, gripper, args.wave)


if __name__ == "__main__":
    main()
