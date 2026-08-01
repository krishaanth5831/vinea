#!/usr/bin/env python3
"""Week 1 — click a tomato onto the plant row and the FR5 reaches for it.

`week1_targetreach.py` takes coordinates you type. This one takes a mouse
click. There is a green board standing in front of the arm, a stand-in for a
plant row: **double-click anywhere on it** and the tomato appears there, and
the arm goes for it.

The chain from click to motion is the whole point of this file, and it is four
steps. Follow it once and you will understand how every camera-driven pick in
Week 4 works, because it is the same chain with a detector in place of a mouse:

    1. mouse pixel     the viewer turns the click into "body N, at this point
                       in body N's own local frame"
    2. -> world        local point + the body's pose = a point in world space
    3. -> base frame   re-express it relative to the arm's base. Right now the
                       base sits at the world origin so this is the identity,
                       and the code does it anyway — see to_base_frame()
    4. -> IK           the arm has a position to solve for

Step 3 is the one that looks pointless today and is not. A tomato's position
only means something *relative to something*. The camera will report fruit in
camera coordinates; the arm needs them in its own. Once the base is on a rail
and moving, the identity becomes a real transform and every place you skipped
it becomes a bug.

The arm runs at 15% of its rated joint speed so you can watch it move. Raise
it with --speed.

Usage:

    # click on the board; the arm reaches
    ./.venv/bin/python simulation/mujoco/week1_mousereach.py

    # same thing without a mouse: name points on the board directly
    ./.venv/bin/python simulation/mujoco/week1_mousereach.py \
        --click 0.0 0.8 --click -0.3 0.4

    # record it
    ./.venv/bin/python simulation/mujoco/week1_mousereach.py --headless \
        --click 0.0 0.8 --click -0.3 0.4 --click 0.35 0.9 \
        --out week1_mousereach.mp4
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
    add_mocap_target,
    add_plant_panel,
    build_fr5_spec,
    exit_without_teardown,
    panel_normal,
)
from reach import (  # noqa: E402
    CTRL_DT,
    DEFAULT_SPEED,
    HOLD_S,
    REACHED_MM,
    STANDOFF,
    Reacher,
    attempt,
    describe,
)

# The board: a vertical plane at x = PANEL_X, PANEL_HALF_Y either side of
# centre, running from PANEL_Z_LO to PANEL_Z_HI.
#
# Deliberately taller than the arm can serve. Everything from about z = 1.0
# up is outside the envelope, and clicking there fails. That is not a badly
# sized board, it is the actual problem: Dutch tomato rows run to three metres
# and an FR5 reaches under one. It is the reason the MVP spec has the arm on a
# lift rather than bolted to the floor, and you should see it fail here before
# you read it in a spec.
PANEL_X = 0.55
PANEL_HALF_Y = 0.38
PANEL_Z_LO = 0.15
PANEL_Z_HI = 1.15
PANEL_HALF_THICKNESS = 0.01

TOMATO_R = 0.03


def build_scene():
    """FR5 + the plant row + a tomato marker you can move."""
    import mujoco

    spec = build_fr5_spec()
    add_plant_panel(spec, x=PANEL_X, half_y=PANEL_HALF_Y,
                    z_lo=PANEL_Z_LO, z_hi=PANEL_Z_HI)
    add_mocap_target(spec, name="tomato", radius=TOMATO_R)

    # fr5.py's `demo` camera sits at x=+1.6 — on the far side of the board,
    # filming its back. This scene needs a view from the arm's side. Target
    # mode keeps it aimed at the board however it is moved.
    spec.worldbody.add_camera(
        name="row", pos=[-0.55, -1.75, 1.15],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="plant_panel",
    )
    return spec.compile()


# ---------------------------------------------------------------------------
# click -> world -> base frame
# ---------------------------------------------------------------------------

def selection_to_world(data, body_id, localpos):
    """Step 2: the viewer's "body N, local point P" into a world position.

    A body's pose is a position and a rotation matrix. A point given in the
    body's own frame becomes a world point by rotating it and adding the
    body's position. This is the single most reused piece of maths in
    robotics and it is three lines.
    """
    xpos = np.array(data.xpos[body_id])
    xmat = np.array(data.xmat[body_id]).reshape(3, 3)
    return xpos + xmat @ np.asarray(localpos, dtype=float)


def to_base_frame(data, world_point):
    """Step 3: re-express a world point relative to the arm's base.

    The inverse of what selection_to_world does: subtract the base's position,
    then un-rotate by the base's orientation.

    Today `base_link` sits at the world origin, unrotated, so this returns
    exactly what you gave it. Write it anyway. The moment the arm is mounted on
    a rail — which is the MVP spec — the base moves, and every coordinate that
    skipped this step silently points at the wrong plant.
    """
    base = data.body("base_link")
    rot = np.array(base.xmat).reshape(3, 3)
    return rot.T @ (np.asarray(world_point, dtype=float) - np.array(base.xpos))


def panel_to_world(y, z):
    """A point named in board coordinates, on the front face of the board.

    The --click flag uses this so the whole pipeline can be exercised without
    a mouse: same face, same offsets, same everything downstream.
    """
    return np.array([PANEL_X - PANEL_HALF_THICKNESS, y, z])


def tomato_on_panel(face_point):
    """Sit the tomato proud of the board instead of half-buried in it."""
    return np.asarray(face_point, dtype=float) + TOMATO_R * panel_normal()


def report_placement(data, tomato):
    """Print the conversion, so the chain is visible and not just asserted."""
    rel = to_base_frame(data, tomato)
    print(f"\n  tomato at world  [{tomato[0]:+.3f} {tomato[1]:+.3f} {tomato[2]:+.3f}]"
          f"  ->  base frame [{rel[0]:+.3f} {rel[1]:+.3f} {rel[2]:+.3f}]")


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------

BANNER = """
────────────────────────────────────────────────────────────────────────
 Double-click anywhere on the green board to put a tomato there.
────────────────────────────────────────────────────────────────────────

 The arm reaches for it and stops {standoff:.0f} cm short — that gap is where a
 gripper's fingers go in Week 2.

 Ctrl + right-drag on the tomato moves it off the board, anywhere in 3D.
 Close the window to quit.
"""


def run_interactive(reacher, panel_id):
    import mujoco
    import mujoco.viewer

    model, data = reacher.model, reacher.data

    tomato = tomato_on_panel(panel_to_world(0.0, 0.60))
    data.mocap_pos[0] = tomato
    held = elapsed = 0.0
    reported = False
    ik_err = arm_err = float("nan")
    last_click = None

    ctrl_per_frame = 2
    frame_s = ctrl_per_frame * CTRL_DT

    warnings.filterwarnings("ignore", module="glfw")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(BANNER.format(standoff=STANDOFF * 100))
        print(f"  arm at {reacher.speed * 100:.0f}% of rated joint speed")
        report_placement(data, tomato)

        next_frame = time.perf_counter()
        while viewer.is_running():
            new_tomato = None

            with viewer.lock():
                sel = int(viewer.perturb.select)
                localpos = np.array(viewer.perturb.localpos)
                # Ctrl+drag: the viewer records the drag but does not apply it.
                # In passive mode the physics loop owns `data`, so writing it
                # into mocap_pos is our job.
                mujoco.mjv_applyPerturbPose(model, data, viewer.perturb, 0)

            # Clicked the board? Steps 1 and 2 of the chain.
            if sel == panel_id:
                click = (sel, tuple(np.round(localpos, 5)))
                if click != last_click:
                    last_click = click
                    face = selection_to_world(data, panel_id, localpos)
                    new_tomato = tomato_on_panel(face)

            # Dragged the tomato itself instead.
            if new_tomato is None:
                dragged = data.mocap_pos[0].copy()
                if np.linalg.norm(dragged - tomato) > 1e-3:
                    new_tomato = dragged

            if new_tomato is not None:
                tomato = np.asarray(new_tomato, dtype=float)
                data.mocap_pos[0] = tomato
                held = elapsed = 0.0
                reported = False
                report_placement(data, tomato)

            for _ in range(ctrl_per_frame):
                ik_err, arm_err = reacher.step(tomato, panel_normal())
            elapsed += frame_s
            held = held + frame_s if arm_err * 1000 < REACHED_MM else 0.0

            if not reported and held >= HOLD_S:
                print(describe(attempt(tomato, True, ik_err, arm_err, elapsed,
                                       panel_normal())))
                reported = True

            viewer.sync()
            next_frame += frame_s
            time.sleep(max(0.0, next_frame - time.perf_counter()))
            next_frame = max(next_frame, time.perf_counter() - frame_s)

    print("\nclosed.")
    exit_without_teardown()


# ---------------------------------------------------------------------------
# scripted — same pipeline, board coordinates instead of a mouse
# ---------------------------------------------------------------------------

def run_scripted(reacher, clicks, out_path=None, dwell=1.0):
    import mujoco

    model, data = reacher.model, reacher.data
    writer = renderer = None
    fps = 30
    if out_path:
        import cv2
        width, height = 1280, 960
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {out_path}")
        renderer = mujoco.Renderer(model, height=height, width=width)

    frame_every = max(1, int(round((1 / fps) / CTRL_DT)))
    cycles = [0]

    def tick(_t):
        cycles[0] += 1
        if writer is not None and cycles[0] % frame_every == 0:
            import cv2
            renderer.update_scene(data, camera="row")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    results = []
    try:
        for y, z in clicks:
            tomato = tomato_on_panel(panel_to_world(y, z))
            report_placement(data, tomato)
            r = reacher.drive_to(tomato, panel_normal(), on_tick=tick)
            print(describe(r))
            results.append(r)
            for _ in range(int(dwell / CTRL_DT)):
                reacher.step(tomato, panel_normal())
                tick(0)
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.release()
            print(f"\nwrote {out_path}")

    hit = sum(r["reached"] for r in results)
    print(f"\n{hit}/{len(results)} tomatoes reached")
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--click", action="append", nargs=2, type=float,
                    metavar=("Y", "Z"),
                    help="place a tomato at this point on the board, as if "
                         "clicked. Repeat for a sequence. Skips the window.")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help="fraction of the FR5's rated joint speed "
                         f"(default {DEFAULT_SPEED}; 1.0 is full speed)")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="seconds to hold each tomato (scripted mode)")
    ap.add_argument("--headless", action="store_true",
                    help="no window; use with --out to record")
    ap.add_argument("--out", default=None, help="write an mp4 (implies scripted)")
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene()
    data = mujoco.MjData(model)
    reacher = Reacher(model, data, speed=args.speed)
    panel_id = model.body("plant_panel").id

    print(f"Fairino FR5: {model.nq} DoF, tracking site '{TOOL_SITE}', "
          f"physics on, {args.speed * 100:.0f}% joint speed")
    print(f"Plant row at x={PANEL_X:.2f} m, "
          f"y ±{PANEL_HALF_Y:.2f}, z {PANEL_Z_LO:.2f}–{PANEL_Z_HI:.2f}")

    if args.click or args.headless or args.out:
        if not args.click:
            ap.error("headless mode needs at least one --click Y Z")
        run_scripted(reacher, args.click, args.out, args.dwell)
    else:
        run_interactive(reacher, panel_id)


if __name__ == "__main__":
    main()
