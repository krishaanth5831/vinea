#!/usr/bin/env python3
"""Week 1 — click a tomato onto the plant row and the FR5 picks it and crates it.

`week1_targetreach.py` takes coordinates you type. This one takes a mouse
click. There is a green board standing in front of the arm, a stand-in for a
plant row: **double-click anywhere on it** and a tomato appears there, hanging
off a stem — and the arm goes and gets it.

The board is sized to the part of the row this arm can actually work, measured
cell by cell rather than derived from the reach sphere (see PANEL_HALF_Y). So
every click is pickable, which is what you want when someone else is driving.

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

**What happens after the reach** is the part added on 2026-08-01. The arm no
longer stops short and admires the fruit — it closes on it, pulls until the
stem gives, carries it to a crate and drops it in:

    approach -> engage -> close -> pull until the stem snaps -> carry -> release

The stem is a **weld equality constraint** that gets switched off when the
force through it passes a threshold. That is the detachment model the roadmap
settled on, and this is its first working use — Week 2 builds the real row on
top of exactly this mechanic. Two things about it are worth knowing before you
read the code:

  * `SNAP_N` is a number nobody measured. No force gauge has been near a real
    peduncle for this project. It decides whether picks succeed, so it is an
    assumption to declare, not a fact to quote.
  * The weld's anchor **must** be zeroed in `eq.data`. Leave the compiler's
    default and it puts the anchor a metre off to the side, and the resulting
    lever arm sags the fruit by 160 mm with nothing pulling on it. It throws no
    error and looks exactly like a physics bug.

The arm runs at 15% of its rated joint speed so you can watch it move. Raise
it with --speed.

Usage:

    # click on the board; the arm picks what you clicked
    ./.venv/bin/python simulation/mujoco/week1_mousereach.py

    # same pipeline without a mouse: name points on the board directly
    ./.venv/bin/python simulation/mujoco/week1_mousereach.py \
        --click 0.0 0.60 --click -0.30 0.40

    # record it — three corners of the row, low left to high right
    ./.venv/bin/python simulation/mujoco/week1_mousereach.py --headless \
        --click -0.50 0.20 --click 0.0 0.60 --click 0.50 0.92 \
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
    MAX_REACH_GRIPPER,
    TOOL_SITE,
    add_crate,
    add_plant_panel,
    build_fr5_spec,
    crate_contains,
    exit_without_teardown,
    panel_normal,
    reach_fraction,
    reset_home,
)
from reach import (  # noqa: E402
    CTRL_DT,
    DEFAULT_SPEED,
    Gripper,
    Reacher,
    hold,
)

# The board: a vertical plane at x = PANEL_X, PANEL_HALF_Y either side of
# centre, running from PANEL_Z_LO to PANEL_Z_HI.
#
# **The board is exactly the part of the row this arm can work.** Every point
# on it is pickable, so a click never lands somewhere the arm has to refuse.
#
# These numbers are measured, not derived. The reach sphere in fr5.py is a
# bound, and it lies close in and around the wrist's folding limits, so the
# only honest way to size a working area is to try every point: a 31 x 24 grid
# over this plane, each cell driving the arm to the pre-grasp point and then to
# the fruit, with the tomato actually placed there so it blocks what it would
# really block. What came back:
#
#     full width out to y = +/-0.75 for z from 0.15 up to 0.80
#     above 0.80 it closes into a dome, ending near z = 1.15 at |y| < 0.20
#     below 0.15 the crate is in the way of the approach, not the arm
#
# A rectangle has to fit inside that dome. Several fit about equally well
# (+/-0.75 x 0.65 m tall, +/-0.55 x 0.80 m tall, and points between are all
# within 5% on area), so this one is chosen for having a full grid cell of
# margin everywhere and for reading like a section of picking row rather than
# a wall. Re-measure it if the gripper, the standoff or the crate moves — all
# three change the answer.
#
# What this deliberately no longer shows: a real Dutch row runs to three metres
# and this arm serves under one of it. That is still the argument for putting
# the arm on a lift, and it is now a thing to say rather than a thing to watch.
PANEL_X = 0.55
PANEL_HALF_Y = 0.55
PANEL_Z_LO = 0.15
PANEL_Z_HI = 0.95
PANEL_HALF_THICKNESS = 0.01

TOMATO_R = 0.033          # a truss tomato is roughly 65 mm across
TOMATO_MASS = 0.12        # ~120 g, mid-range for a vine tomato

# How far the fruit hangs proud of the foliage. Under Week 1's mocap marker
# this was one radius — the ball sat against the board. A gripper needs room
# to get its fingers around the back of the fruit, and real trusses do hang
# out into the aisle, which is what makes the crop pickable at all.
TRUSS_OFFSET = 0.12

# Pushed out to y = -0.80 when the board widened to +/-0.55. The arm sweeps
# x 0.12-0.42 across the full width of the row while pulling fruit loose, and
# at y = -0.55 the old crate at -0.55 was directly under that path — the
# gripper hit its wall on the way to the pre-grasp point and the pick aborted
# 179 mm short. The crate now clears the working footprint by 9 cm.
CRATE_POS = np.array([0.15, -0.80, 0.0])
CRATE_HALF = 0.16
CRATE_WALL = 0.12

# --- the pick ---------------------------------------------------------------
APPROACH_GAP = 0.15       # stop here first, fingers open, touching nothing
RETRACT_GAP = 0.30        # pull back to here; the stem gives somewhere on the way
CARRY_UP = 0.32           # hold the fruit this far above the crate before letting go
GRIP_S = 1.0              # time to let the fingers travel

# A neutral point in front of the base, between the row and the crate, that the
# arm passes through on its way to drop the fruit.
#
# It is here because `mink` does **differential** IK: it takes the posture the
# arm is in and walks downhill towards the target, one small step at a time. It
# has no map of the workspace, so a single large move — from high on the row at
# +y all the way down to the crate at -y — can walk into a corner it cannot get
# out of and stall a long way short. Measured: 378 mm short, from a pick at
# [0.42, +0.25, 1.00].
#
# Splitting one big move into two small ones fixes it, and this is what real
# pick-and-place does anyway: retract to a known transit posture, then travel.
# Week 2 will hit this again the moment the row has more than one fruit in it.
TRANSIT = np.array([0.38, -0.12, 0.58])

# Newtons through the peduncle before it lets go. **Made up.** Tuned so a
# deliberate pull detaches and the fruit's own 1.18 N of weight does not.
# See the module docstring, and flag it in any demo before someone else does.
SNAP_N = 3.0

# The 2F85 weighs 1.05 kg, which takes the arm's steady-state error from
# ~3.2 mm to ~5.2 mm — that is droop under a constant load, not slop, and a
# P+D position servo cannot null it out. 5 mm would report failure on arrival.
REACHED_MM_LOADED = 8.0

STEM_MOCAP = 0            # the stem is the only mocap body in this scene


def build_scene():
    """FR5 + gripper + the plant row + a tomato on a stem + a crate."""
    import mujoco

    spec = build_fr5_spec(gripper=True)
    add_plant_panel(spec, x=PANEL_X, half_y=PANEL_HALF_Y,
                    z_lo=PANEL_Z_LO, z_hi=PANEL_Z_HI)
    add_crate(spec, pos=CRATE_POS, half=CRATE_HALF, wall=CRATE_WALL)

    start = tomato_on_panel(panel_to_world(0.0, 0.60))

    # The stem is a *mocap* body: drawn, and teleportable from Python, but the
    # solver never pushes it. That is what lets a click move the attachment
    # point anywhere on the row without anything having to fall or swing.
    stem = spec.worldbody.add_body(name="stem", pos=start.tolist(), mocap=True)
    stem.add_geom(
        name="stem_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, TRUSS_OFFSET * 0.9, 0, 0.02], size=[0.005, 0, 0],
        rgba=[0.24, 0.45, 0.22, 1.0], contype=0, conaffinity=0,
    )

    fruit = spec.worldbody.add_body(name="tomato", pos=start.tolist())
    fruit.add_freejoint()
    fruit.add_geom(
        name="tomato_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[TOMATO_R, 0, 0], rgba=[0.85, 0.16, 0.12, 1.0], mass=TOMATO_MASS,
        friction=[0.6, 0.01, 0.001],
    )

    # The peduncle. objtype=BODY welds the two bodies' frames together.
    eq = spec.add_equality(
        name="peduncle", type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY, name1="stem", name2="tomato")
    # [anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]
    #  ^^^^^^^^^ zeroing this is the whole ballgame — see the docstring.
    eq.data = [0, 0, 0,  0, 0, 0,  1, 0, 0, 0,  1.0]

    # fr5.py's `demo` camera sits at x=+1.6 — on the far side of the board,
    # filming its back. This scene needs a view from the arm's side that holds
    # the row, the arm and the crate in one frame, so it is aimed by hand
    # rather than tracking the board: target mode kept the board centred and
    # pushed the crate out of shot once the board grew to 1.6 m.
    #
    # xyaxes is the camera's own x and y axes in world coordinates; MuJoCo
    # looks down its -z. Computed for the position below looking at
    # [0.28, -0.25, 0.42], the middle of the work — far enough back to hold
    # the row, the arm and the crate, close enough that the board is not a
    # postage stamp. Retune it if the board is resized again.
    spec.worldbody.add_camera(
        name="row", pos=[-0.55, -2.05, 1.25],
        xyaxes=[0.9081, -0.4187, 0.0, 0.1617, 0.3507, 0.9223],
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
    """Hang the fruit out in front of the foliage, where a gripper can get it."""
    return np.asarray(face_point, dtype=float) + TRUSS_OFFSET * panel_normal()


def report_placement(data, tomato):
    """Print the conversion, so the chain is visible and not just asserted."""
    rel = to_base_frame(data, tomato)
    print(f"\n  tomato at world  [{tomato[0]:+.3f} {tomato[1]:+.3f} {tomato[2]:+.3f}]"
          f"  ->  base frame [{rel[0]:+.3f} {rel[1]:+.3f} {rel[2]:+.3f}]"
          f"   ({reach_fraction(tomato, MAX_REACH_GRIPPER) * 100:.0f}% of reach)")


# ---------------------------------------------------------------------------
# the stem
# ---------------------------------------------------------------------------

def weld_force(model, data, eq_id):
    """Newtons the peduncle is currently carrying.

    MuJoCo solves every constraint — contacts, joint limits, equalities — as
    rows in one big system, so the force in a weld is found by picking out the
    rows belonging to that equality and reading `efc_force`.

    **Sanity-check this before trusting it:** a tomato hanging still should
    read its own weight, 0.12 x 9.81 = 1.18 N. If it does not, the reading is
    wrong and everything built on it is wrong.
    """
    import mujoco

    rows = np.flatnonzero(
        (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        & (data.efc_id == eq_id))
    return float(np.linalg.norm(data.efc_force[rows][:3])) if rows.size else 0.0


def place_tomato(model, data, eq_id, world_pos):
    """Hang the tomato at a new spot on the row.

    Three things have to move together, and skipping any one of them looks
    like a physics bug rather than a missing line:

      1. the stem, which is where the weld holds *to*
      2. the fruit itself, teleported rather than dragged — leave it behind and
         the weld yanks it across the scene at whatever speed the solver likes
      3. its velocity, zeroed, so it does not arrive already moving

    Then the weld goes back on, because the last pick switched it off.
    """
    import mujoco

    jnt = model.body("tomato").jntadr[0]
    qadr = model.jnt_qposadr[jnt]
    vadr = model.jnt_dofadr[jnt]

    data.mocap_pos[STEM_MOCAP] = world_pos
    data.qpos[qadr:qadr + 3] = world_pos
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[vadr:vadr + 6] = 0.0
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# the pick
# ---------------------------------------------------------------------------

def park_pose(model):
    """Where `tool0` sits at the arm's home posture.

    Computed on a scratch MjData so asking the question does not move the arm
    that is currently holding a tomato.
    """
    import mujoco

    scratch = mujoco.MjData(model)
    reset_home(model, scratch)
    return scratch.site(TOOL_SITE).xpos.copy()


def pick_cycle(reacher, gripper, eq_id, tomato_pos, park=None, on_tick=None,
               verbose=True):
    """Approach, grip, snap the stem, crate it. Returns what happened.

    Same shape as `week1_gripper.run_pick`, with one real difference: the fruit
    is attached to the plant, so getting it loose is a step of its own rather
    than something that happens for free. That step is the harvest.
    """
    model, data = reacher.model, reacher.data
    tomato = data.body("tomato")
    normal = panel_normal()

    def say(state, note=""):
        if verbose:
            p = data.site(TOOL_SITE).xpos
            print(f"  {state:<9} tool [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]   {note}")

    def go(state, target, tick=None, note=""):
        r = reacher.drive_to(target, on_tick=tick or on_tick)
        arrived = (f"arm {r['arm_mm']:.1f} mm" if r["reached"]
                   else f"DID NOT ARRIVE — {r['arm_mm']:.0f} mm short")
        say(state, f"{arrived}   {note}".rstrip())
        return r

    # Where the arm parks between picks. A *fixed* pose, from the home joint
    # posture — reading the tool's current position here instead would make
    # "home" mean "wherever I happened to start", so one aborted pick would
    # leave the next one parking somewhere new, and the one after that
    # somewhere newer. Cycles have to end where they began or they drift.
    if park is None:
        park = park_pose(model)

    # 1. Approach along the row normal, square-on, fingers open. Coming at the
    #    fruit diagonally is how you knock it off the plant.
    gripper.open()
    reached = go("approach", tomato_pos + APPROACH_GAP * normal)
    if not reached["reached"]:
        say("abort", "cannot get to the pre-grasp point — too high, or out of reach")
        go("park", park)
        return {"reached": False, "snapped": False, "in_crate": False,
                "peak_n": 0.0, "tomato": tomato.xpos.copy()}

    # 2. Close the last stretch with the fingers open around the fruit.
    go("engage", tomato_pos)

    # 3. Grip. The stem holds the fruit still while the fingers travel, which
    #    is why this is more reliable than grabbing a loose ball off a table.
    gripper.close()
    hold(reacher, tomato_pos, GRIP_S, on_tick)
    say("close", "fingers closed on the fruit")

    # 4. Pull. The stem carries the load until it does not. Nothing commands
    #    the break — the arm just backs off, and the constraint gives out.
    snap = {"broken": False, "peak": 0.0, "at": 0.0}

    def pull_tick(t):
        f = weld_force(model, data, eq_id)
        snap["peak"] = max(snap["peak"], f)
        if not snap["broken"] and f > SNAP_N:
            data.eq_active[eq_id] = 0
            snap["broken"] = True
            snap["at"] = f
        if on_tick is not None:
            on_tick(t)

    go("pull", tomato_pos + RETRACT_GAP * normal, tick=pull_tick)
    if snap["broken"]:
        # The load is checked once per control cycle — every 10 ms — and a stiff
        # arm retracting at speed loads the stem far faster than that. So the
        # force it actually breaks at overshoots the threshold, often by a lot.
        # The threshold is a floor, not the number the stem sees. Slow the pull
        # or check every physics step if that gap ever needs to be small.
        say("snap", f"stem gave at {snap['at']:.1f} N "
                    f"(threshold {SNAP_N:.1f} N — one cycle's overshoot)")
    else:
        say("snap", f"stem HELD — peak only {snap['peak']:.1f} N of {SNAP_N:.1f} N")

    # 5. Carry and drop, via the transit point — see TRANSIT for why one move
    #    is not enough.
    go("transit", TRANSIT)
    go("carry", CRATE_POS + np.array([0, 0, CARRY_UP]))
    gripper.open()
    hold(reacher, CRATE_POS + np.array([0, 0, CARRY_UP]), GRIP_S, on_tick)
    say("release", "fingers opened")

    go("park", park)
    hold(reacher, park, 0.4, on_tick)

    in_crate = crate_contains(tomato.xpos, CRATE_POS, CRATE_HALF, CRATE_WALL)
    say("done", "IN THE CRATE" if in_crate else "not in the crate")
    return {"reached": True, "snapped": snap["broken"], "in_crate": in_crate,
            "peak_n": snap["peak"], "tomato": tomato.xpos.copy()}


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------

BANNER = """
────────────────────────────────────────────────────────────────────────
 Double-click anywhere on the green board to hang a tomato there.
────────────────────────────────────────────────────────────────────────

 The arm approaches square-on, closes, pulls until the stem gives at
 {snap:.1f} N, carries the fruit to the crate and drops it in.

 The board is sized to what this arm can actually work — every point on
 it is pickable, so anywhere you click, it goes. A real row is three
 metres tall and this is the band of it one FR5 can serve.

 There is one tomato. Clicking again re-hangs the same one on the row,
 wherever you put it. Close the window to quit.
"""


def run_interactive(reacher, gripper, eq_id, panel_id, park):
    import mujoco
    import mujoco.viewer

    model, data = reacher.model, reacher.data
    tomato_pos = tomato_on_panel(panel_to_world(0.0, 0.60))
    place_tomato(model, data, eq_id, tomato_pos)

    warnings.filterwarnings("ignore", module="glfw")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(BANNER.format(snap=SNAP_N))
        print(f"  arm at {reacher.speed * 100:.0f}% of rated joint speed")
        report_placement(data, tomato_pos)
        print("  click the board to pick it.")

        clock = [time.perf_counter()]

        def tick(_t):
            """One control cycle's worth of wall time, and a redraw."""
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            clock[0] += CTRL_DT
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

        # Prime the click state from whatever the viewer already reports.
        # A fresh viewer can come up with a body already selected, and with
        # `last_click` starting at None that reads as a brand-new click — the
        # arm sets off and picks a tomato nobody asked for.
        with viewer.lock():
            last_click = (int(viewer.perturb.select),
                          tuple(np.round(viewer.perturb.localpos, 5)))

        try:
            while viewer.is_running():
                with viewer.lock():
                    sel = int(viewer.perturb.select)
                    localpos = np.array(viewer.perturb.localpos)

                new_click = None
                if sel == panel_id:
                    click = (sel, tuple(np.round(localpos, 5)))
                    if click != last_click:
                        last_click = click
                        # Steps 1 and 2 of the chain.
                        new_click = tomato_on_panel(
                            selection_to_world(data, panel_id, localpos))

                if new_click is not None:
                    tomato_pos = new_click
                    place_tomato(model, data, eq_id, tomato_pos)
                    report_placement(data, tomato_pos)
                    pick_cycle(reacher, gripper, eq_id, tomato_pos, park, on_tick=tick)
                    print("\n  click the board again for another.")
                    # Whatever was clicked while the arm was busy is stale.
                    with viewer.lock():
                        last_click = (int(viewer.perturb.select),
                                      tuple(np.round(viewer.perturb.localpos, 5)))
                else:
                    # Idle: the arm still has to be told to stay where it is,
                    # or it sags. Doing nothing is an instruction.
                    hold(reacher, park, 4 * CTRL_DT, tick)
        except KeyboardInterrupt:
            pass

    print("\nclosed.")
    exit_without_teardown()


# ---------------------------------------------------------------------------
# scripted — same pipeline, board coordinates instead of a mouse
# ---------------------------------------------------------------------------

def run_scripted(reacher, gripper, eq_id, clicks, park, out_path=None):
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
            tomato_pos = tomato_on_panel(panel_to_world(y, z))
            place_tomato(model, data, eq_id, tomato_pos)
            report_placement(data, tomato_pos)
            results.append(
                pick_cycle(reacher, gripper, eq_id, tomato_pos, park, on_tick=tick))
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.release()
            print(f"\nwrote {out_path}")

    crated = sum(r["in_crate"] for r in results)
    print(f"\n{crated}/{len(results)} tomatoes picked and crated")
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--click", action="append", nargs=2, type=float,
                    metavar=("Y", "Z"),
                    help="hang a tomato at this point on the board, as if "
                         "clicked, and pick it. Repeat for a sequence.")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help="fraction of the FR5's rated joint speed "
                         f"(default {DEFAULT_SPEED}; 1.0 is full speed)")
    ap.add_argument("--headless", action="store_true",
                    help="no window; use with --out to record")
    ap.add_argument("--out", default=None, help="write an mp4 (implies scripted)")
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene()
    data = mujoco.MjData(model)
    # standoff 0: tool0 is at the fingertips now, so every waypoint in
    # pick_cycle is where the fingers should actually be. mocap None: the one
    # mocap body in this scene is the stem, and the reach loop must not drag
    # it around as a target marker.
    reacher = Reacher(model, data, speed=args.speed, standoff=0.0,
                      max_reach=MAX_REACH_GRIPPER,
                      reached_mm=REACHED_MM_LOADED, mocap=None)
    gripper = Gripper(model, data)
    gripper.open()

    panel_id = model.body("plant_panel").id
    eq_id = model.equality("peduncle").id
    park = park_pose(model)

    print(f"FR5 + Robotiq 2F85: {model.nq} DoF, {model.nu} actuators, "
          f"reach {MAX_REACH_GRIPPER:.2f} m to the fingertips")
    print(f"Plant row at x={PANEL_X:.2f} m, y ±{PANEL_HALF_Y:.2f}, "
          f"z {PANEL_Z_LO:.2f}–{PANEL_Z_HI:.2f}   ·   "
          f"fruit hangs {TRUSS_OFFSET * 100:.0f} cm proud, stem snaps at {SNAP_N:.1f} N")
    print(f"Crate at [{CRATE_POS[0]:.2f} {CRATE_POS[1]:.2f}]")

    if args.click or args.headless or args.out:
        if not args.click:
            ap.error("headless mode needs at least one --click Y Z")
        run_scripted(reacher, gripper, eq_id, args.click, park, args.out)
    else:
        run_interactive(reacher, gripper, eq_id, panel_id, park)


if __name__ == "__main__":
    main()
