#!/usr/bin/env python3
"""A camera on the chassis that looks at the whole row, so the arm stops hunting.

Week 3 gave the robot an eye in its hand and it worked, but it left a hole that
`week4_place.py` papered over: **the arm only knew where to point the wrist
camera because the script told it.** The staging pose for every look came out of
`crop.placed`, which is the operator's ground truth.

⚠️ **So the thing the mast replaces is not slow code, it is a cheat**, and the
comparison has to be against the honest alternative rather than against what the
repo used to do. With one eye-in-hand sensor and no ground truth, finding the
row means sweeping it — drive the wrist along the staging plane, render, drive
on. Both measured on the same 8-fruit row by `main --vs-sweep --speed 0.15`,
counting *simulated* arm seconds because a robot's cycle time is made of the
seconds the arm spends moving and not of how fast this laptop renders:

    the deck survey    8/8 found    arm moved 0.000 m     0.00 s
    a wrist sweep      8/8 found    arm moved 2.557 m    24.14 s   (10 poses)

⚠️ **`--speed 0.15` is not decoration, it is the speed every other number in
this repo was taken at** — the campaign's, and the one 31.3 s is a cycle at.
The sweep's cost is arm motion, so it scales: the same ten poses cost 17.06 s at
0.25 and 11.56 s at the sub-command's own default of 0.5. Quoting the sweep at
one speed and the cycle at another would flatter the mast by more than a
factor of two. The deck survey costs 0.00 s at every one of them, because the
arm does not move.

24 seconds is most of a whole pick — the campaign's mean cycle is 31.3 s — spent
before the first tomato is touched, and it is paid again on every row. Both find
everything, so this is not an accuracy argument; the sweep's cost is arm motion
and scales with the length of the row, and the deck frame's does not scale at
all.

So this adds the sensor a real harvester has and this repo did not: an RGBD
camera on the chassis, facing the plane the fruit hang in, that sees the whole
working band **with the arm stationary**. It does two jobs:

    1. **It tells the wrist camera where to look.** The survey is the row map.
       No sweep, no ground truth, and the wrist camera goes straight to a
       staging pose it was aimed at rather than one it found.
    2. **It decides the order.** Given where the fruit actually are, it plans
       which to pick first — see `plan_order`. That is not cosmetic: the sweep
       in `main --pairs` shows two fruit 100 mm apart where one is refused
       outright and the other plans fine, so picking the plannable one first is
       the difference between harvesting one and harvesting both.

**The camera is on a pan-tilt head, and the head had to earn its place.** The
obvious argument for articulation — "it can look around" — is worth nothing
here, twice over: one fixed frame already covers the whole placement band, and
a camera rotating about its own optical centre cannot see round anything at
all, because pure rotation about the pinhole leaves every occlusion in the
scene exactly where it was. What pays is that the lens sits on a 100 mm yoke
offset from the pan axis, so panning **translates** it by up to 90 mm, and
translation is the only thing that changes which fruit is hidden behind which.
Measured over 48 fruit packed to 72-100 mm centres — the band the old 200 mm
placement rule forbade — `main --scan`:

    bolted down, one frame       31/48 found
    five head poses              40/48 found     2.9 s of head slew, arm parked

⚠️ It does not remove the floor, it moves it. At exactly 70 mm — touching —
the scan finds the same 3 of 6 the fixed camera does, from every angle: 90 mm
of parallax does not undo a 70 mm separation at 1300 mm. The wrist camera still
goes in.

The division of labour is the point, and it is the one real greenhouse machines
settle on. The deck camera is **wide, far and coarse**; the wrist camera is
**narrow, near and precise**. Neither can do the other's job:

    deck    1.26-1.48 m range, whole band in frame, ~2 mm on an isolated fruit
            and **blind to the difference between two touching ones** — see the
            cluster measurement in `main`.
    wrist   0.28 m range, one fruit, sub-millimetre geometry (`camera.py`'s
            Step 3 gate) — and it has to be driven there to see anything.

**The order is chosen to lose the fewest tomatoes.** What the row is worth is
fruit, so that is the unit the objective is in — see the `W_LOST` block for why
"minimise total risk" was the wrong quantity and quietly traded whole fruit for
fractions of a score.

Solved in two stages, and the split is forced by one line of physics:

    1. Held-Karp over (set already picked, fruit picked last). Exact, because
       under the relaxation "every attempt removes its fruit" the cost of a pick
       depends only on the *set* still standing — 2^n states rather than n!, and
       fifteen fruit settle in 451 ms against a 31 s cycle.
    2. Then hill-climb that answer against the **true** cost, in which a
       *refused* pick leaves its fruit standing. Which fruit are up after k
       attempts is then no longer a function of which k were attempted, so
       (attempted-set, last) stops being a sufficient state and stage 1 stops
       being exact for the real problem.

⚠️ **That second rule is not a nicety, and it was found by flying the thing.**
In the contested-band A/B, the planner opened layout 1 with p06 and followed
with p00. p06 was refused — `insert: gr_right_pad within 25 mm of p00` — and
then p00 was refused too, `within 32 mm of p06`, by the fruit the cost model had
already crossed off. The old function removed every attempted fruit whether or
not the attempt worked. Measured against brute force over all permutations, the
two stages together reach the true optimum on 21-27 of 30 random layouts at
n=5-8, and average 0.0006 of cost above it when they miss — against 0.077 for
the exact-but-relaxed answer alone.

⚠️ **What none of that has done yet is harvest more tomatoes**, and this file is
not the place that gets to decide otherwise. See `week4_order.py`.

Every constant in here is read off a sweep that can be re-run:

    ./.venv/bin/python simulation/mujoco/deck_cam.py             # the gate
    ./.venv/bin/python simulation/mujoco/deck_cam.py --scan      # what the head buys
    ./.venv/bin/python simulation/mujoco/deck_cam.py --eclipse   # can it see round the arm
    ./.venv/bin/python simulation/mujoco/deck_cam.py --optimal   # exact vs the hill-climb
    ./.venv/bin/python simulation/mujoco/deck_cam.py --pairs     # what a neighbour costs
    ./.venv/bin/python simulation/mujoco/deck_cam.py --corridor  # the pull-down wedge
    ./.venv/bin/python simulation/mujoco/deck_cam.py --mounts    # why the mast is there
    ./.venv/bin/python simulation/mujoco/deck_cam.py --sweep     # the arm's swept volume
    ./.venv/bin/python simulation/mujoco/deck_cam.py --vs-sweep  # what the mast buys
    ./.venv/bin/python simulation/mujoco/deck_cam.py --shot      # render both cameras

and whether the order it produces is worth anything is `week4_order.py`, which
flies both orders rather than asking this file to mark its own homework.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera import (Intrinsics, SensorCamera,  # noqa: E402
                    _xyaxes_towards, add_camera_housing)
from plant_row import FRUIT_R, ROW_X  # noqa: E402

DECK_NAME = "deck"

# --- where the deck camera stands --------------------------------------------
#
# ⚠️ **The mount is measured, not chosen, and the binding constraint is not
# optics — it is the arm.** Ten candidate mounts were scored on one frame each,
# 21 fruit spread across the whole placement band, HSV detector, arm parked
# (`main --mounts` re-runs it). Every one of them found 21/21 at ~2 mm, because
# a simulator's depth buffer is noiseless and the fruit are red spheres against
# green. Accuracy discriminated nothing.
#
# What discriminated was where a post can physically stand. `main --sweep` flies
# a six-fruit harvest and records the arm's swept volume:
#
#     x  -0.457 .. +0.628      y  -0.575 .. +0.394      z  -0.102 .. +0.981
#
# The arm reaches **457 mm behind its own shoulder** on the way to the crate.
# Measured as the closest approach to a vertical post on the y=0 line:
#
#     post at x = -0.10    the arm passes 251 mm *inside* it
#     post at x = -0.30    the arm passes 125 mm *inside* it
#     post at x = -0.45    10 mm
#     post at x = -0.60    154 mm        <- the first one that clears
#     post at x = -0.75    301 mm
#
# So x = -0.60 is not a preference, it is the nearest line to the machine that
# the arm does not sweep through. Every mount nearer than that was a camera the
# robot would have hit, and none of the optical scores would have said so.
#
# ⚠️ The y and z extents move a few centimetres between layouts — they depend on
# which fruit the arm happened to reach for. The x extent does not: it is set by
# the trip to the crate, which every pick makes. Re-run it and expect x to
# repeat and y to wander.
DECK_POST_X = -0.60

# Height on that post. The trade is incidence against occlusion, and both ends
# of it were measured:
#
#   z      angle off the row plane's normal    fruit found, arm parked / staged
#   0.95        17 deg                              21/21   19/21
#   1.10        23 deg                              21/21   20/21
#   1.25        29 deg                              21/21   20/21
#   1.40        34 deg                              21/21   20/21
#   1.60        40 deg                              21/21   21/21
#
# Low is square-on to the row — which is what "facing the plane of the row"
# means and what keeps the deprojection honest, because a grazing look is where
# z-depth and range diverge — and low is also where the parked arm starts
# getting in the way. 1.10 m holds 23 degrees, which still reads as facing the
# row, and finds everything with the arm parked.
#
# ⚠️ **The survey therefore runs with the arm parked, and that is a rule, not a
# coincidence.** Staged mid-row the same camera drops to 20/21 — the arm eclipses
# a fruit and the survey silently returns one fewer. `DeckSurvey.look` refuses
# to run unless the arm is at park; see the check there.
DECK_MOUNT = np.array([DECK_POST_X, 0.0, 1.10])

# --- the head that carries it ------------------------------------------------
#
# The camera is on a pan-tilt head, not bolted down. Two joints: pan about the
# post's own vertical axis, tilt about a trunnion at the top of the post.
#
# ⚠️ **The camera sits on a yoke arm, offset from the pan axis, and that offset
# is the entire reason the head is worth having.** A camera that rotates about
# its own optical centre gains *nothing* but field of view: pure rotation about
# the pinhole leaves every occlusion relationship in the scene exactly as it
# was, so a fruit hidden behind another fruit is still hidden and two fruit
# whose blobs merge still merge. Panning a camera in place cannot see round
# anything. Offset it from the axis and panning *translates* it, and translation
# is what changes who is behind what.
#
# So the post moved back rather than the camera moving forward. The optics were
# measured at x = -0.60 and stay there; the mast now stands at -0.70 with a
# 100 mm arm, which puts the lens back on the measured line at home and buys
# clearance rather than spending it:
#
#     pan      camera x      camera y     closest the arm gets (sweep says
#      0°        -0.600         0.000      the arm reaches x = -0.457)
#    ±30°       -0.613        ∓0.050          156 mm
#    ±60°       -0.650        ∓0.087          193 mm
#    ±90°       -0.700         ∓0.100          243 mm
#
# The camera is *furthest* from the arm when panned, so the swept-volume
# clearance that sited the mast is not spent by articulating it. See `--mounts`.
DECK_PAN_AXIS_X = -0.70
DECK_YOKE_M = 0.10

# How fast the head slews, degrees per second. A pan-tilt unit in this class
# (FLIR PTU-D48 and friends) manages 60-100 deg/s under a camera payload; 60 is
# the conservative end.
#
# ⚠️ It is here so `DeckSurvey.scan` can **charge for the look**. The whole
# argument for the mast is that it costs no arm-seconds, and a scan that took
# an unbilled amount of time would quietly undo that. The arm still does not
# move — but the head does, and the report says how long for.
DECK_SLEW_DEG_S = 60.0

# Aimed at the centre of the band `week4_place` lets fruit be placed in, not at
# the row centre in the abstract. The two are the same today and would stop
# being the same the moment the band moves, which is exactly the sort of drift
# that produces a camera pointing at nothing.
DECK_AIM = np.array([ROW_X, 0.0, 0.58])

# A D435's depth vertical FoV. At 1.3 m this puts the band's extreme corners
# 260 px from the nearest frame edge, so a fruit at the limit of where it can be
# placed is still comfortably inside the image rather than clipped — and a
# clipped box is the silent failure `detect.estimate` flags as `edge`, because
# the radius fit reads the truncated width and the position comes back short.
DECK_FOVY = 58.0

# ⚠️ Association gate, and it does a different job from `week3_perceive`'s.
# That one matches a wrist detection against a *map*; this one has no map — it
# is what makes the map. What it matches against is the pool of body names,
# purely so the executor has a handle to command. See `DeckSurvey.look`: the
# name is bookkeeping, the position is the measurement.
#
# 60 mm, which is under the 70 mm at which two fruit are touching. A wider gate
# would let one blob claim a neighbour's name and report a fruit that was never
# separately seen.
DECK_GATE_M = 0.060

# How close two sightings have to be to be called the same fruit. See `_fuse`.
FUSE_M = 0.030

# The poses the head visits when it scans, as (pan, tilt) in degrees from home.
#
# ⚠️ **Articulation had to earn this and very nearly did not.** Panning a camera
# about its own optical centre is worth precisely nothing for finding fruit: one
# fixed frame already covers the whole placement band — that is what `DECK_FOVY`
# was chosen for — and a pure rotation about the pinhole leaves every occlusion
# in the scene exactly where it was. What pays is the 100 mm yoke, which turns
# pan and tilt into *translation* of up to 90 mm, and translation is the only
# thing that changes who is hidden behind what.
#
# Measured by `main --scan` on 8 fruit x 6 layouts packed to 72-100 mm centres —
# the band the old 200 mm rule forbade, and the band `week4_order`'s 75 mm row
# showed the fixed camera losing three fruit of six in:
#
#   pattern         poses    fruit found    mean err    phantoms   head slew
#   fixed             1        31/48         3.8 mm        0         0.00 s
#   pan +-20          3        35/48         4.1 mm        0         1.00 s
#   pan +-35          3        36/48         4.6 mm        0         1.75 s
#   pan +-35          5        36/48         4.3 mm        0         2.95 s
#   cross             5        38/48         4.2 mm        1         2.07 s
#   box               5        40/48         4.8 mm        0         2.92 s   <-
#   grid              7        40/48         5.3 mm        0         4.53 s
#   box wide          5        39/48         5.4 mm        1         4.67 s
#   box               9        41/48         4.7 mm        0         6.53 s
#
# 65% to 83% of a packed row, for 2.9 s in which **the arm does not move**.
# Compare the alternative in `--vs-sweep`: 24.14 s of arm motion.
#
# ⚠️ Read the two failure columns against each other, because they pull opposite
# ways. `found` rewards more poses; `phantoms` punishes them, because two
# sightings of one fruit that fail to fuse become two orders to pick the same
# tomato. `box wide` and `cross` each produced one, and both are patterns that
# move the viewpoint furthest — which is exactly where fusing gets hardest.
#
# Nine poses buy one more fruit for 3.6 s more slew, and going wider is worse on
# every column. Five is where it stops paying.
SCAN_POSES = ((0.0, 0.0), (-25.0, -10.0), (25.0, -10.0),
              (-25.0, 10.0), (25.0, 10.0))


HEAD_BODY = "deck_head"

# The home aim, in world axes. At home the head's quaternion is the identity, so
# head-local and world coincide and this doubles as the camera's declaration in
# the head's own frame. Everything `DeckHead` does is expressed relative to it.
DECK_HOME_XYAXES = _xyaxes_towards(DECK_MOUNT, DECK_AIM, up_hint=(0.0, 0.0, 1.0))


def add_deck_camera(spec, name=DECK_NAME, mount=DECK_MOUNT, aim=DECK_AIM,
                    fovy=DECK_FOVY, post=True, articulated=True):
    """Put the deck camera, its mast and its pan-tilt head into a spec.

    Call before `spec.compile()`. The post is drawn by default because a camera
    floating 1.1 m above the floor with nothing holding it up invites the
    question "how is that mounted", and the answer — a post on the chassis,
    behind the arm's swept volume — is the interesting part.

    ⚠️ **The head is a mocap body, not a pair of hinge joints, and that is a
    correctness decision rather than a shortcut.** Hinges would add two DOFs to
    `mjModel`, and `mink.Configuration` in `reach.Reacher` is built over the
    *whole* model — so the IK solver would see two free joints, notice they
    lower its cost function not at all, and be free to leave them anywhere. Any
    coupling between "where the arm is reaching" and "where the camera is
    pointing" would be a bug that only showed up as a survey quietly missing
    fruit. A mocap body has no DOFs at all: it is commanded, it is not
    simulated, and a pan-tilt unit with position servos is exactly that.

    ⚠️ Everything here is `contype=0`, like `greenhouse.py`'s scenery and for
    the same reason: the collision set this repo's clearance numbers were
    measured against is the fruit, the support bar, the row panel, the floor and
    the crate. Adding a solid post 700 mm behind the shoulder would change what
    the planner routes around, and it would change it invisibly. The mast is
    sited where the arm does not go, so it *could* be solid — the measurement
    above is what says so — but making it solid is a separate change with its
    own numbers to re-take.

    `articulated=False` bolts the camera to the mast at the home pose, which is
    what shipped before the head existed. Kept so `--scan` can measure the two
    against each other in one process rather than against a memory.
    """
    import mujoco

    mount = np.asarray(mount, float)
    xyaxes = _xyaxes_towards(mount, aim, up_hint=(0.0, 0.0, 1.0))
    mast = spec.worldbody.add_body(name="deck_mast", pos=[0, 0, 0])

    if post:
        # ⚠️ The mast stands on the *pan axis*, which is 100 mm behind the lens.
        # Drawing it under the camera instead would put the post where the yoke
        # swings and make the picture disagree with the kinematics.
        x, y = (DECK_PAN_AXIS_X, 0.0) if articulated else (float(mount[0]), 0.0)
        top = float(mount[2]) - 0.05
        mast.add_geom(
            name="deck_post", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[x, y, 0.02, x, y, top], size=[0.022, 0, 0],
            rgba=[0.42, 0.44, 0.47, 1.0], contype=0, conaffinity=0, mass=0.0)
        # A foot, so it reads as bolted to a deck rather than stuck in the floor.
        mast.add_geom(
            name="deck_post_foot", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, 0.015], size=[0.09, 0.09, 0.015],
            rgba=[0.35, 0.37, 0.40, 1.0], contype=0, conaffinity=0, mass=0.0)

    if not articulated:
        spec.worldbody.add_camera(name=name, pos=list(mount), fovy=fovy,
                                  xyaxes=xyaxes)
        add_camera_housing(mast, f"cam_{name}", mount, xyaxes, kind="d435",
                           stalk=[float(mount[0]), 0.0, float(mount[2]) - 0.05])
        return spec

    # The head. Its origin is the pan axis at the trunnion height, so a pure
    # quaternion on the mocap body *is* the pan-tilt command and nothing has to
    # track a moving centre of rotation.
    origin = np.array([DECK_PAN_AXIS_X, 0.0, float(mount[2])])
    head = spec.worldbody.add_body(name=HEAD_BODY, pos=list(origin),
                                   mocap=True)

    # Camera-in-head coordinates. At home the head is the identity, so the
    # camera's local offset is just the yoke and its local axes are the world
    # ones — which is what makes `DeckHead`'s angles read as world pan and tilt.
    local = (mount - origin).tolist()

    head.add_geom(
        name="deck_trunnion", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, -0.030, 0, 0, 0.030, 0], size=[0.020, 0, 0],
        rgba=[0.35, 0.37, 0.40, 1.0], contype=0, conaffinity=0, mass=0.0)
    head.add_geom(
        name="deck_yoke", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, local[0], local[1], local[2]], size=[0.011, 0, 0],
        rgba=[0.45, 0.47, 0.50, 1.0], contype=0, conaffinity=0, mass=0.0)

    head.add_camera(name=name, pos=local, fovy=fovy, xyaxes=xyaxes)
    add_camera_housing(head, f"cam_{name}", local, xyaxes, kind="d435",
                       stalk=[0.0, 0.0, 0.0])
    return spec


class DeckHead:
    """Where the deck camera is pointing, and how to point it somewhere else.

    Two angles, both in degrees and both measured from the home aim rather than
    from an absolute frame — `pan=0, tilt=0` is the pose every fixed-camera
    number in this module was taken at, so the articulated camera and the bolted
    one are the same measurement at the same place.

    ⚠️ Pan and tilt are *not* independent of where the camera is. The lens is on
    a 100 mm yoke, so panning swings it through an arc; asking "point at that
    fruit" is therefore a fixed point rather than a formula, and `look_at`
    iterates it. Three passes is plenty — the yoke is 100 mm against a 1.3 m
    working range, so the first correction is already sub-degree.
    """

    def __init__(self, model, data=None, name=HEAD_BODY):
        import mujoco

        self.model = model
        self.body = name
        try:
            self.mocap = int(model.body(name).mocapid[0])
        except KeyError:
            self.mocap = -1
        if self.mocap < 0:
            raise RuntimeError(
                f"no articulated deck head in this model — build the scene "
                f"with deck_cam=True, or use add_deck_camera(articulated=False) "
                f"and skip the head entirely")
        self.origin = model.body(name).pos.copy()
        self.fwd0 = np.asarray(DECK_AIM, float) - np.asarray(DECK_MOUNT, float)
        self.fwd0 /= np.linalg.norm(self.fwd0)
        # The camera's right axis at home. Horizontal by construction —
        # `_xyaxes_towards` crosses the forward vector with world up — which is
        # what lets pan and tilt decompose into azimuth and elevation below.
        self.right0 = np.asarray(DECK_HOME_XYAXES[:3], float)
        self.pan = 0.0
        self.tilt = 0.0
        if data is not None:
            self.aim(data, 0.0, 0.0)

    # --- pointing ------------------------------------------------------------

    def _quat(self, pan_deg, tilt_deg):
        import mujoco

        qp, qt, out = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qp, np.array([0.0, 0.0, 1.0]),
                                  np.radians(pan_deg))
        mujoco.mju_axisAngle2Quat(qt, self.right0, np.radians(tilt_deg))
        # Pan is the outer rotation and tilt the inner one, which is the order a
        # real pan-tilt unit is built in: the tilt trunnion rides on the pan
        # table. Composed the other way the head yaws about a tilted axis and
        # the horizon rolls.
        mujoco.mju_mulQuat(out, qp, qt)
        return out

    def slew_seconds(self, pan_deg, tilt_deg):
        """How long a real unit would take to get there from where it is now.

        Separate from `aim` so a caller that wants to *animate* the move can ask
        the cost of the whole thing first and then walk there in steps — asking
        `aim` per step would return the cost of each step instead, and the sum
        of those is the same number only by accident.
        """
        swing = max(abs(pan_deg - self.pan), abs(tilt_deg - self.tilt))
        return swing / DECK_SLEW_DEG_S

    def aim(self, data, pan_deg, tilt_deg):
        """Command the head. Returns the seconds a real unit would take to slew.

        ⚠️ Caller must `mj_forward` before rendering. Writing `mocap_quat` does
        not move `cam_xpos` on its own, and a render taken in between is a frame
        from the *previous* pose with the new pose's label on it — which is the
        one failure here that produces plausible wrong numbers rather than an
        error.
        """
        secs = self.slew_seconds(pan_deg, tilt_deg)
        data.mocap_quat[self.mocap] = self._quat(pan_deg, tilt_deg)
        self.pan, self.tilt = float(pan_deg), float(tilt_deg)
        return secs

    def home(self, data):
        return self.aim(data, 0.0, 0.0)

    def current(self, data):
        """(pan, tilt) read back out of `mjData`, not out of this object.

        `self.pan`/`self.tilt` are what was last *commanded*; this is where the
        head actually is. They agree today because a mocap body goes exactly
        where it is put — but a panel that reports a commanded angle while
        claiming to show the robot is the kind of display that keeps looking
        right after the thing behind it has stopped working.
        """
        import mujoco

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, data.mocap_quat[self.mocap])
        v = mat.reshape(3, 3) @ self.fwd0

        def az_el(u):
            return (np.degrees(np.arctan2(u[1], u[0])),
                    np.degrees(np.arcsin(u[2] / np.linalg.norm(u))))

        az, el = az_el(v)
        az0, el0 = az_el(self.fwd0)
        return float(az - az0), float(el - el0)

    def camera_pos(self, pan_deg=None, tilt_deg=None):
        """Where the lens ends up at that pose, without touching `mjData`."""
        import mujoco

        pan = self.pan if pan_deg is None else pan_deg
        tilt = self.tilt if tilt_deg is None else tilt_deg
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self._quat(pan, tilt))
        offset = np.asarray(DECK_MOUNT, float) - self.origin
        return self.origin + mat.reshape(3, 3) @ offset

    def angles_to(self, point, passes=3):
        """(pan, tilt) in degrees that put `point` on the optical axis.

        Exact but for the fixed point: because pan and tilt are pure azimuth and
        elevation offsets from the home aim (see `right0`), each pass is a
        closed-form solve, and the only thing being iterated is where the yoke
        has carried the lens to.
        """
        def az_el(v):
            v = np.asarray(v, float)
            return (np.degrees(np.arctan2(v[1], v[0])),
                    np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))

        az0, el0 = az_el(self.fwd0)
        pan = tilt = 0.0
        for _ in range(passes):
            d = np.asarray(point, float) - self.camera_pos(pan, tilt)
            az, el = az_el(d)
            pan, tilt = az - az0, el - el0
        return float(pan), float(tilt)

    def look_at(self, data, point):
        """Point the head at a world position. Returns the slew seconds."""
        return self.aim(data, *self.angles_to(point))


# --- the survey --------------------------------------------------------------

@dataclass
class Seen:
    """One fruit as the deck camera has it. Coarse, and honest about it."""

    name: str
    est: np.ndarray
    truth: np.ndarray | None = None
    det: object = None
    edge: bool = False

    @property
    def err_mm(self):
        if self.truth is None:
            return float("nan")
        return float(np.linalg.norm(self.est - self.truth) * 1000)


class DeckSurvey:
    """Render once, detect, deproject, and hand back where the fruit are.

    Deliberately the same three stages as `week3_perceive.Perception`, running
    the same `detect` and `camera` code, because the whole argument for the deck
    camera is that it is *the same pipeline at a different range* — if it needed
    its own detector the comparison between the two would mean nothing.
    """

    def __init__(self, model, detector=None, camera=DECK_NAME,
                 width=None, height=None, gate=DECK_GATE_M):
        from camera import RENDER_H, RENDER_W

        if detector is None:
            from detect import HSVDetector

            detector = HSVDetector()
        self.model = model
        self.detector = detector
        self.gate = gate
        self.sensor = SensorCamera(model, camera,
                                   RENDER_W if width is None else width,
                                   RENDER_H if height is None else height)
        self.intr = self.sensor.intr

    def close(self):
        self.sensor.close()

    def look(self, data, names, truth=True):
        """(map, report). `map` is name -> estimated world position.

        `names` is the pool of trusses that could be out there. It is used for
        one thing only and it is worth being blunt about which:

        ⚠️ **The camera returns blobs, not names.** Something has to turn a blob
        into a handle the executor can be told to pick, because `Planner.plan`
        takes a body name. So each detection is matched to the nearest pool
        member within `gate`. That step reads ground truth and it is
        **bookkeeping, not perception** — the position handed onward is the
        camera's, never the simulator's. The distinction matters because the two
        failure modes look identical from outside and are completely different:
        a fruit the camera missed comes back missing (correct), and a fruit two
        of whose blobs fused comes back as *one* name with a position between
        them (also correct, and the reason the wrist camera still goes in).

        A real machine has no such step and does not need one: it picks at a
        position, and the name only exists here because a simulator's fruit are
        named bodies.
        """
        from detect import estimate

        rgb, depth = self.sensor.both(data)
        R, C = self.sensor.pose(data)
        dets = [estimate(d, depth, self.intr, R, C) for d in self.detector(rgb)]
        usable = [d for d in dets if d.est is not None]

        # Greedy closest-pair, each blob and each name used at most once — the
        # same association `Perception.look` does, for the same reason: sorting
        # by index instead lets one miss renumber everything after it.
        pairs = sorted(
            (float(np.linalg.norm(d.est - data.body(n).xpos)), i, n)
            for i, d in enumerate(usable) for n in names
            if np.linalg.norm(d.est - data.body(n).xpos) <= self.gate)

        taken_d, out, log = set(), {}, []
        for dist, i, n in pairs:
            if i in taken_d or n in out:
                continue
            taken_d.add(i)
            d = usable[i]
            out[n] = Seen(name=n, est=d.est.copy(),
                          truth=data.body(n).xpos.copy() if truth else None,
                          det=d, edge=bool(d.extra.get("edge")))
            log.append(f"{n} <- blob at ({d.centre[0]:.0f},{d.centre[1]:.0f}) "
                       f"est {d.est.round(3)} err {out[n].err_mm:.1f} mm"
                       + ("  [touches the frame edge]" if out[n].edge else ""))
        for i, d in enumerate(usable):
            if i not in taken_d:
                log.append(f"UNASSOCIATED blob -> {d.est.round(3)} "
                           f"(no truss within {self.gate * 1000:.0f} mm)")
        return out, {"log": log, "dets": dets, "usable": usable,
                     "rgb": rgb, "depth": depth, "R": R, "C": C}

    # --- looking around ------------------------------------------------------

    def scan(self, data, names, head=None, poses=None, truth=True,
             on_pose=None):
        """Sweep the head across `poses` and fuse what every pose saw.

        Returns the same `(map, report)` shape as `look`, so a caller can be
        handed either and not care. The report gains `poses`, `slew_s` and a
        per-pose breakdown.

        `on_pose` is a no-argument callback for anything watching — a viewer's
        render pump, typically. Given one, the head is **walked** between poses
        rather than teleported, at `DECK_SLEW_DEG_S`, one control period per
        step, with the callback fired at each. Without one nothing is
        interpolated and the scan runs at full speed, so a measurement run costs
        exactly what it did before.

        ⚠️ The interpolation is cosmetic and deliberately so: renders and
        detections still happen only at the listed poses. Detecting from the
        in-between frames would quietly turn a five-pose scan into a
        fifty-pose one and every number in `SCAN_POSES` would stop meaning what
        it says.

        ⚠️ **Detections are fused in world coordinates, not in the image.** Each
        pose deprojects to metres before anything is compared, which is what
        makes fusing across poses meaningful at all — two pixels from two
        different camera orientations have no relationship to each other, and
        two points in the greenhouse do.

        ⚠️ The head is left where the last pose put it. Callers that render
        anything afterwards want `head.home(data)` first, or they get a frame
        from wherever the scan finished.
        """
        import mujoco

        from detect import estimate

        head = DeckHead(self.model, data) if head is None else head
        poses = SCAN_POSES if poses is None else poses

        found, per_pose, slew = [], [], 0.0
        for k, (pan, tilt) in enumerate(poses):
            move_s = head.slew_seconds(pan, tilt)
            slew += move_s
            if on_pose is None:
                head.aim(data, pan, tilt)
                mujoco.mj_forward(self.model, data)   # see DeckHead.aim
            else:
                from reach import CTRL_DT

                p0, t0 = head.pan, head.tilt
                for i in range(1, max(1, round(move_s / CTRL_DT)) + 1):
                    f = i / max(1, round(move_s / CTRL_DT))
                    head.aim(data, p0 + (pan - p0) * f, t0 + (tilt - t0) * f)
                    mujoco.mj_forward(self.model, data)
                    on_pose()
            rgb, depth = self.sensor.both(data)
            R, C = self.sensor.pose(data)
            dets = [estimate(d, depth, self.intr, R, C)
                    for d in self.detector(rgb)]
            usable = [d for d in dets if d.est is not None]
            per_pose.append({"pan": pan, "tilt": tilt, "blobs": len(dets),
                             "usable": len(usable), "cam": C.copy()})
            # Which pose a sighting came from, carried on the detection so
            # `_fuse` can report how many *distinct viewpoints* backed a fruit.
            # Four sightings from four poses is corroboration; four from one
            # pose would be the detector firing repeatedly on one blob, and the
            # two mean opposite things about how much to trust the position.
            found.extend((k, d) for d in usable)

        clusters = _fuse(found)
        pairs = sorted(
            (float(np.linalg.norm(c["est"] - data.body(n).xpos)), i, n)
            for i, c in enumerate(clusters) for n in names
            if np.linalg.norm(c["est"] - data.body(n).xpos) <= self.gate)

        taken, out, log = set(), {}, []
        for _dist, i, n in pairs:
            if i in taken or n in out:
                continue
            taken.add(i)
            c = clusters[i]
            out[n] = Seen(name=n, est=c["est"].copy(),
                          truth=data.body(n).xpos.copy() if truth else None,
                          det=c["dets"][0],
                          edge=all(bool(d.extra.get("edge")) for d in c["dets"]))
            log.append(f"{n} <- {len(c['dets'])} sighting(s) from "
                       f"{len(c['poses'])} of {len(poses)} poses, est "
                       f"{c['est'].round(3)} err {out[n].err_mm:.1f} mm"
                       + ("  [edge in every pose]" if out[n].edge else ""))
        for i, c in enumerate(clusters):
            if i not in taken:
                log.append(f"UNASSOCIATED cluster -> {c['est'].round(3)} "
                           f"({len(c['dets'])} sightings, no truss within "
                           f"{self.gate * 1000:.0f} mm)")
        return out, {"log": log, "clusters": clusters, "per_pose": per_pose,
                     "poses": list(poses), "slew_s": slew}


def _fuse(sightings, radius=None):
    """Group sightings that are the same fruit seen from different poses.

    `sightings` is `[(pose_index, detection)]`. Single-link clustering on the
    deprojected positions, then the mean of each group. Returns
    `[{est, dets, poses}]`, where `poses` is the set of distinct viewpoints that
    saw it — not the number of detections, which is a different claim.

    ⚠️ **`radius` is what decides whether the scan can tell two fruit apart, so
    it is the one number in here that must not be generous.** Fruit centres
    70 mm apart are touching (`plant_row.FRUIT_R` x 2 plus a little); anything
    at or over half of that would merge a real pair into one phantom fruit
    halfway between them and report it as a confident single detection — which
    is worse than missing one, because the arm would then be sent somewhere
    there is no tomato. 30 mm is under half the touching distance and well over
    the ~2 mm the deck camera resolves an isolated fruit to, so the two failure
    modes are nowhere near each other.
    """
    radius = FUSE_M if radius is None else radius
    groups = []
    for k, d in sightings:
        # ⚠️ Collect **every** group this sighting links to and merge them, not
        # just the first. A first-match-wins loop leaves two groups unmerged
        # when the sighting that would have joined them arrives last, and the
        # result is one fruit reported twice — a phantom, i.e. a second order to
        # pick a tomato that is no longer there. It costs one pass to be right.
        #
        # ⚠️ And merged **by index**, never with `list.remove`. `remove` finds
        # its target with `==`, and a group is a list of tuples holding
        # `detect.Detection` — a dataclass with numpy fields — so comparing two
        # same-length groups evaluates an array in a boolean context and raises
        # "truth value of an array with more than one element is ambiguous".
        # It survived every test here by luck: `remove` short-circuits on
        # identity, so it only bites when an earlier group happens to be the
        # same length as the one being removed.
        hit = [i for i, g in enumerate(groups)
               if any(np.linalg.norm(d.est - e[1].est) <= radius for e in g)]
        merged = [(k, d)]
        for i in hit:
            merged.extend(groups[i])
        groups = [g for i, g in enumerate(groups) if i not in set(hit)]
        groups.append(merged)
    return [{"est": np.mean([d.est for _k, d in g], axis=0),
             "dets": [d for _k, d in g],
             "poses": sorted({k for k, _d in g})} for g in groups]


def parked(model, data, park_q, tol=0.02):
    """Is the arm at the park posture? The survey's precondition.

    ⚠️ Not a style check. Staged mid-row the deck camera loses a fruit behind
    the arm and says nothing about it — the survey comes back with one fewer
    entry and every downstream stage treats that as "there is no fruit there".
    A missing fruit is invisible in a way a wrong position is not, so this is
    checked rather than remembered.
    """
    from fr5 import JOINTS

    q = np.array([data.joint(j).qpos[0] for j in JOINTS])
    return float(np.max(np.abs(q - np.asarray(park_q, float)))) <= tol


# --- what a neighbour costs --------------------------------------------------
#
# ⚠️ **These three numbers are the whole ordering argument and they are swept,
# not assumed.** `main --pairs` puts two fruit `d` apart and plans a pick of
# each; `main --corridor` does the same with one below the other, swept
# sideways. Centre to centre, so 66 mm is touching:
#
#   centres    side by side          stacked               diagonal
#     70 mm    both REFUSED          both REFUSED          both REFUSED
#     85 mm    both REFUSED          one refused, one ok   both REFUSED
#    100 mm    one refused, one ok   one refused, one ok   one refused, one ok
#    120 mm    fallback route        direct / fallback     one refused, one ok
#    140 mm    direct                direct / fallback     fallback route
#    170 mm    direct                direct / fallback     direct
#
# Read the 100 mm row again, because it is the finding this module exists for:
# **the two fruit of a close pair are not interchangeable.** One of them cannot
# be planned at all while the other is standing there; the other can. Pick the
# plannable one first and the refused one becomes a lone fruit with a clear
# direct route. Pick them in the order they were placed and it is a coin flip.
#
# The old 200 mm placement minimum hid all of this by making it impossible to
# arrange. With that rule gone the arrangement is the operator's to make, and
# the ordering is the robot's to solve.
BLOCKED_M = 0.120       # under this, a neighbour refuses the pick outright
CROWDED_M = 0.170       # under this, it costs a fallback route and cycle time

# ⚠️ **KNOWN LIMITATION, and it is the one that matters most in this file:
# the crowd term above is symmetric, and the effect this module exists to
# exploit is not.**
#
# Read the 100 mm row of the sweep again — "one refused, one ok". That
# asymmetry is the entire ordering argument: pick the plannable one, it comes
# off, and the other is left a lone fruit with a clear route. But `_pair_risk`
# is a function of the separation vector, so at 100 mm side by side it returns
# 1.000 in **both** directions. The model says both fruit are blocked. It cannot
# tell you which one to take first, because it does not believe there is a
# difference.
#
# The corridor term is directional and does produce some asymmetry — a
# neighbour 100 mm directly below scores 1.636 against 1.000 the other way —
# but `exposure` clips `block` at 1.0, which is correct for a probability and
# erases exactly that difference at the distances where it would have mattered.
#
# What this predicts, and what was then measured:
#
#   * `main --optimal` — every method, from placement order to the exact
#     solver, produces an **identical** expected loss (7.036 at n=8, 11.405 at
#     n=12, 14.588 at n=15). Only tour length moves. Under a symmetric model
#     with mutual blocking, no order can save a fruit, and the optimiser
#     correctly reports that there is nothing to win.
#   * `week4_order.py --band contested` — 32 attempts per arm, deck order
#     against placement order: 19 crated vs 18, **12 refused vs 12**. A tie —
#     and the model forecast that tie exactly, at 0.00 fruit of predicted gain
#     on all four layouts. The *old* model forecast 3.33 fruit on the same rows
#     and delivered none. So the honest reading of this limitation is that it is
#     now **visible in the forecast** rather than hidden behind a number that
#     looked like a result.
#
# So the ordering machinery is correct, exact, and currently pointed at a cost
# function that has no ordering signal in it. **The fix is a measurement, not a
# weight**: `--pairs` has to record *which* of the two fruit was refused, not
# just that one of them was, and the crowd term has to grow a directional part
# fitted to that. Until then, `plan_order` is honest about producing an order
# that is optimal under a model which says order does not matter.

# The pull is `mission.PULL_DOWN` = 140 mm straight down with the fruit in the
# pads, so the gripper sweeps the space *below* the target and nowhere else.
# That asymmetry is why the order comes out roughly bottom-up without anything
# in here saying "go bottom-up": harvesting the low fruit empties the corridor
# the high one has to be pulled through.
#
# ⚠️ **The corridor is a wedge, not a cylinder,** and assuming the cylinder
# would over-penalise every fruit with a distant neighbour under it. Swept by
# `main --corridor` — one neighbour below the target, moved sideways until the
# planner stops needing a non-default pull. Reading the boundary as halfway
# between the last offset that still cost a fallback and the first that went
# direct:
#
#   neighbour below     last fallback     first direct     boundary
#       100 mm             120 mm           160 mm          ~140 mm
#       150 mm              80 mm           120 mm          ~100 mm
#       200 mm              40 mm            80 mm           ~60 mm
#       260 mm               —                 0 mm          ~0 mm
#
# Four points on a straight line: the half-width closes at 0.8 mm per mm of
# depth and reaches zero at 275 mm down. Both constants are read off that fit.
#
# ⚠️ An earlier fit put them at 250/200, which tracked the 100 mm row and ran
# ~40 mm narrow at 150 and 200 mm down — it scored a neighbour the planner was
# visibly working around as costing nothing. `--corridor` prints the model's
# number beside the planner's route for exactly this reason: a cost model that
# is never checked against the thing it is a proxy for is decoration.
CORRIDOR_DOWN_M = 0.275
CORRIDOR_HALF_M = 0.220

# --- what the order is actually optimising -----------------------------------
#
# ⚠️ **The objective used to be "minimise total risk" and that was the wrong
# quantity.** Summing a risk score over the picks treats one pick at risk 2.0 as
# equal to two picks at risk 1.0 — but a pick whose worst neighbour is inside
# BLOCKED_M is *refused*, so the second case loses two fruit and the first loses
# one. An order optimised on the sum will happily trade a fruit to shave a
# fraction off a score. What the row is worth is fruit, so that is the unit.
#
# The rewrite splits what a neighbour does into two things, because the sweeps
# found two mechanisms with different shapes and only one of them costs yield:
#
#   block   the **worst single** neighbour, clipped at 1. `--pairs` measured a
#           pick being refused outright when one neighbour is inside BLOCKED_M,
#           and refusal does not get worse with a second one. Saturating, and a
#           max rather than a sum: this is a probability that the pick is lost.
#   crowd   the **sum** over neighbours. Not a loss — a longer route.
#
# W_LOST is 1.0 by definition: it is the unit, one refused pick is one fruit.
#
# ⚠️ **W_CROWD is small because a fallback route turns out to cost nothing
# measurable.** The expectation was that crowding buys a longer path and so
# cycle time. Planning a target with one neighbour swept underneath it and
# measuring the committed tool path against the same target alone:
#
#   neighbour 150 mm below, swept across      path length vs alone
#     0 / 40 / 80 mm                          -0.098 / -0.063 / -0.063 m
#     120 mm and beyond                        0.000 m
#   neighbour beside it
#     80 / 100 mm                             REFUSED
#     120 mm                                  -0.098 m
#     140 mm and beyond                        0.000 m
#
# Every route the planner accepted was the same length or **shorter**. So
# crowding does not cost seconds in this cycle at all; the only thing a
# neighbour ever costs is the whole fruit, and crowd is kept purely as a
# tie-break between orders that lose the same number.
#
# W_TRAVEL is smaller still, and is in here knowing it measures nothing today:
# `mission.park_arm` teleports the arm back to park between picks (for posture
# reasons, documented there), so fruit-to-fruit distance costs this simulation
# exactly zero. A real machine pays it. It breaks remaining ties in the
# direction a real machine would want, and is reported in its own column rather
# than folded into a score that would imply the sim had measured it.
W_LOST = 1.0
W_CROWD = 0.05
W_TRAVEL = 0.02

# Rows this many fruit or fewer are solved **exactly**; above it, the local
# search below is used instead.
#
# ⚠️ The exact solver is Held-Karp over (set already picked, fruit picked last),
# which is O(n^2 · 2^n) — 15 fruit is 32768 subsets and runs in about a second,
# 20 would be 32x that. `week4_place.MAX_FRUIT` is 15, so in practice the exact
# path is the only one that ever runs and the local search is a guard against a
# caller that places more.
EXACT_MAX = 15


def _pair_risk(a, b):
    """What fruit `b`, still on the plant, costs a pick of fruit `a`.

    Two terms, because the sweeps found two distinct mechanisms and they have
    different shapes:

      * **crowding**, isotropic in the row plane: the pads have to get around
        the target, and a neighbour inside BLOCKED_M means they cannot.
      * **the pull corridor**, one-sided and downward: the pull drags the fruit
        140 mm down through whatever is under it.

    Scored 0 (irrelevant) to 1 (refuses the pick), and squared inside the crowd
    term so that one neighbour at 90 mm outweighs three at 160 mm — which is the
    right shape, because the near one refuses the pick and the far three only
    lengthen it.
    """
    d = np.asarray(b, float) - np.asarray(a, float)
    plane = float(np.hypot(d[1], d[2]))

    crowd = 0.0
    if plane < CROWDED_M:
        crowd = min(1.0, (CROWDED_M - plane) / (CROWDED_M - BLOCKED_M)) ** 2

    corridor = 0.0
    below, across = -d[2], abs(d[1])
    if 0 < below < CORRIDOR_DOWN_M:
        depth = 1 - below / CORRIDOR_DOWN_M     # 1 just under, 0 at 250 mm
        half = CORRIDOR_HALF_M * depth          # the wedge, measured
        if across < half:
            corridor = depth * (1 - across / half)
    return crowd + corridor


def risk(name, positions, remaining):
    """How exposed a pick of `name` is, given what is still on the plant.

    The summed form, kept because it is what the campaign logs as `deck_risk`
    and changing its meaning would silently break comparison against every row
    already written. `exposure` is what the optimiser reads.
    """
    p = positions[name]
    return sum(_pair_risk(p, positions[n]) for n in remaining if n != name)


def exposure(name, positions, remaining):
    """(block, crowd) — the two things the neighbours do, kept apart.

    `block` saturates and is a max, because one neighbour inside BLOCKED_M
    refuses the pick and a second one cannot refuse it twice. `crowd` sums,
    because route length does accumulate. Collapsing them into one number is
    what the old cost model did and what made it prefer losing a fruit.
    """
    p = positions[name]
    rs = [_pair_risk(p, positions[n]) for n in remaining if n != name]
    if not rs:
        return 0.0, 0.0
    return float(min(1.0, max(rs))), float(sum(rs))


@dataclass
class Step:
    """One pick in an ordered plan, with the reason it sits where it does."""

    fruit: str
    pos: np.ndarray
    risk: float          # summed, as the campaign logs it
    unblocks: float
    travel: float
    remaining: int
    block: float = 0.0   # worst single neighbour, clipped — the loss term

    @property
    def verdict(self):
        # ⚠️ Reads `block`, not `risk`. Three distant neighbours can sum to a
        # risk over 1.0 without any one of them being close enough to refuse the
        # pick, and calling that "crowded" put a warning on picks that plan
        # perfectly well while missing the ones that do not.
        if self.block >= 0.99:
            return "BLOCKED"
        if self.block >= 0.5:
            return "crowded"
        if self.risk >= 0.25:
            return "tight"
        return "clear"


@dataclass
class OrderedPlan:
    steps: list = field(default_factory=list)
    total_risk: float = 0.0
    worst_risk: float = 0.0
    travel_m: float = 0.0
    improved_from: tuple = ()
    lost: float = 0.0        # expected fruit the planner will refuse
    cost: float = 0.0        # the objective actually minimised
    optimal: bool = False    # solved exactly, or hill-climbed
    solve_s: float = 0.0

    @property
    def order(self):
        return [s.fruit for s in self.steps]

    def table(self, indent="  "):
        out = [f"{indent}{'#':>2} {'fruit':<6} {'y':>7} {'z':>7} {'block':>6} "
               f"{'risk':>6} {'unblocks':>9} {'travel m':>9} {'left':>5}  note"]
        for i, s in enumerate(self.steps, 1):
            out.append(f"{indent}{i:2d} {s.fruit:<6} {s.pos[1]:+7.3f} "
                       f"{s.pos[2]:7.3f} {s.block:6.2f} {s.risk:6.2f} "
                       f"{s.unblocks:9.2f} {s.travel:9.3f} {s.remaining:5d}  "
                       f"{s.verdict}")
        out.append(f"{indent}expected refusals {self.lost:.2f} fruit · total "
                   f"risk {self.total_risk:.2f} · worst single pick "
                   f"{self.worst_risk:.2f} · tour {self.travel_m:.2f} m")
        out.append(f"{indent}"
                   + ("exact, and unchanged by refinement — proved optimal"
                      if self.optimal else
                      "exact on the relaxation, then refined against the true "
                      "walk")
                   + f" in {self.solve_s * 1000:.0f} ms")
        # What the search bought over taking them in the order they were placed.
        if self.improved_from and tuple(self.order) != tuple(self.improved_from):
            moved = sum(1 for a, b in zip(self.order, self.improved_from)
                        if a != b)
            out.append(f"{indent}(greedy gave {' '.join(self.improved_from)}; "
                       f"the search moved {moved})")
        return "\n".join(out)


def _sequence_cost(order, positions, start):
    """Score a whole order. **This is the true objective.**

    ⚠️ The obstacle set **shrinks as it goes**, which is the entire reason order
    matters and the reason this cannot be a sum of pairwise scores computed up
    front. A fruit's risk is whatever is still standing when its turn comes, so
    the same fruit is cheap ninth and expensive first.

    ⚠️ And it shrinks **only on success** — see the note on `block < 1.0` below.
    That one line is what makes this function not solvable by `_solve_exact`:
    which fruit are standing after k attempts is no longer a function of *which*
    k were attempted, it depends on how they went, so (attempted-set, last) stops
    being a sufficient DP state. `_solve_exact` optimises the relaxation where
    every attempt removes its fruit; `plan_order` then refines that answer
    against this one. See there.
    """
    standing = set(order)
    here = np.asarray(start, float)
    cost = travel = total = worst = lost = 0.0
    steps = []
    for n in order:
        block, crowd = exposure(n, positions, standing - {n})
        step = float(np.linalg.norm(positions[n] - here))
        cost += W_LOST * block + W_CROWD * crowd + W_TRAVEL * step
        lost += block
        total += crowd
        travel += step
        worst = max(worst, crowd)
        # ⚠️ **The fruit comes off the plant only if the pick succeeds.** A
        # refused pick leaves it exactly where it was, still blocking everything
        # attempted after it — and the harness attempts each fruit once, so a
        # refusal is not retried when its blocker clears.
        #
        # Getting this wrong is not academic. In the `contested` A/B, layout 1,
        # the planner opened with p06 (block 1.00) and followed with p00. p06
        # was refused — `insert: gr_right_pad within 25 mm of p00` — and then
        # p00 was refused too, `within 32 mm of p06`, by the fruit the model had
        # already written off. The old cost function removed every attempted
        # fruit whether or not the attempt worked, so it scored p06 as clearing
        # the way for p00 when it had cleared nothing.
        if block < 1.0:
            standing.discard(n)
        here = positions[n]
        steps.append({"fruit": n, "block": block, "crowd": crowd,
                      "travel": step, "standing": len(standing)})
    return {"cost": cost, "total_risk": total, "worst_risk": worst,
            "travel_m": travel, "lost": lost, "steps": steps}


def _greedy(names, positions, start):
    """A decent order, fast. The exact solver's starting point and its check.

    At each step, score every fruit still on the plant on what it costs to pick
    **now**, minus what picking it is worth to everything left (`unblocks` — the
    risk it is currently imposing on others), plus the travel to reach it. The
    obstacle set then shrinks and every remaining score changes, which is why
    this is re-scored per step instead of sorted once.

    Those two terms genuinely pull against each other and neither alone is
    right. Cost alone says "do the easy ones first" and leaves a knot of
    mutually blocking fruit for the end. Unblocking alone says "go straight for
    the fruit in everyone's way", which is the one most likely to be refused.
    """
    remaining = set(names)
    here = np.asarray(start, float)
    order = []
    while remaining:
        best, best_cost = None, np.inf
        for n in sorted(remaining):
            block, crowd = exposure(n, positions, remaining)
            others = remaining - {n}
            unblocks = sum(_pair_risk(positions[m], positions[n])
                           for m in others)
            cost = (W_LOST * block + W_CROWD * crowd - 0.6 * W_LOST * unblocks
                    + W_TRAVEL * float(np.linalg.norm(positions[n] - here)))
            if cost < best_cost:
                best, best_cost = n, cost
        order.append(best)
        remaining.discard(best)
        here = positions[best]
    return order


def _local_search(order, positions, start):
    """Hill-climb the greedy answer: every swap, every relocation, to a fixpoint.

    Only used above `EXACT_MAX`. Kept because the exact solver is exponential
    and something has to answer when a caller places more fruit than
    `week4_place.MAX_FRUIT` allows.
    """
    order = list(order)
    if len(order) <= 2:
        return order
    best = _sequence_cost(order, positions, start)["cost"]
    changed = True
    while changed:
        changed = False
        for i in range(len(order)):
            for j in range(len(order)):
                if i == j:
                    continue
                trial = list(order)
                trial[i], trial[j] = trial[j], trial[i]          # swap
                c = _sequence_cost(trial, positions, start)["cost"]
                if c < best - 1e-9:
                    order, best, changed = trial, c, True
                    continue
                trial = list(order)
                trial.insert(j, trial.pop(i))                    # relocate
                c = _sequence_cost(trial, positions, start)["cost"]
                if c < best - 1e-9:
                    order, best, changed = trial, c, True
    return order


def _solve_exact(names, positions, start):
    """The provably cheapest order, by Held-Karp over subsets.

    ⚠️ **The reason this is possible at all is that the cost of a pick depends
    only on the *set* still standing, never on the path taken to get there.**
    Given that, "which fruit are already picked" is a complete state, and the
    number of states is 2^n rather than n!. For fifteen fruit that is 32768
    against 1.3 trillion.

    State is (set already picked, fruit picked last) — `last` is carried purely
    for the travel term, which is the one part of the cost that is not a
    function of the set alone.

    Two precomputations make the inner loop a single vector op:

      * `crowd[i][S]` and `block[i][S]` — what fruit `i` would face if exactly
        `S` were still standing. Built by subset recurrence off the lowest set
        bit, so each is one add (or one max) rather than a re-sum: O(n · 2^n)
        instead of O(n^2 · 2^n).
      * `step[S][i]` — the whole non-travel cost of picking `i` as the fruit
        that completes `S`. The set still standing afterwards is the complement
        of `S`, which is why this can be indexed by `S` directly.

    Returns (order, seconds).
    """
    import time

    t0 = time.perf_counter()
    n = len(names)
    P = np.array([positions[x] for x in names], float)

    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                R[i, j] = _pair_risk(P[i], P[j])
    dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    from_start = np.linalg.norm(P - np.asarray(start, float), axis=1)

    full = 1 << n
    crowd = np.zeros((n, full))
    block = np.zeros((n, full))
    for S in range(1, full):
        low = (S & -S).bit_length() - 1
        prev = S & (S - 1)
        crowd[:, S] = crowd[:, prev] + R[:, low]
        block[:, S] = np.maximum(block[:, prev], R[:, low])
    np.clip(block, None, 1.0, out=block)

    # Standing after `i` completes `S` is everything not in `S`.
    comp = (full - 1) ^ np.arange(full)
    step = (W_LOST * block[:, comp] + W_CROWD * crowd[:, comp]).T   # (full, n)

    dp = np.full((full, n), np.inf)
    par = np.zeros((full, n), dtype=np.int16)
    for i in range(n):
        dp[1 << i, i] = step[1 << i, i] + W_TRAVEL * from_start[i]

    for S in range(1, full):
        mem = [i for i in range(n) if S >> i & 1]
        if len(mem) < 2:
            continue
        prevs = [S ^ (1 << i) for i in mem]
        # (k, n): cost of arriving at each `mem[k]` from every possible `last`.
        # Entries where `last` is not in the predecessor set are already inf, so
        # they can never win — no masking needed.
        cand = dp[prevs] + W_TRAVEL * dist[:, mem].T
        j = cand.argmin(axis=1)
        dp[S, mem] = cand[np.arange(len(mem)), j] + step[S, mem]
        par[S, mem] = j

    last = int(dp[full - 1].argmin())
    order, S = [], full - 1
    while S:
        order.append(names[last])
        S, last = S ^ (1 << last), int(par[S, last])
    order.reverse()
    return order, time.perf_counter() - t0


def plan_order(survey, start=None, improve=True, exact=None):
    """Decide what to pick first. Returns an `OrderedPlan`.

    `survey` is `{name: position}` — whatever the deck camera came back with,
    which is emphatically not the order the fruit were declared or placed in.

    **What is minimised is expected fruit lost**, then crowding, then travel —
    see the `W_LOST` block for why that is the right quantity and why summing a
    risk score was not.

    **How it is minimised depends on how many fruit there are.** At or under
    `EXACT_MAX` the answer is *proved* optimal by `_solve_exact`; above it the
    old greedy-plus-hill-climb runs instead and `OrderedPlan.optimal` says so.
    The greedy order is computed either way and reported as `improved_from`,
    because the honest question about any optimiser is what it bought over the
    cheap thing.

    ⚠️ **This orders picks, it does not verify them.** Whether a route exists is
    `mission.Planner`'s job and it stays there — this model is a cheap geometric
    proxy that runs in milliseconds on positions from a camera, and the planner
    is a kinematic replay that runs in ~150 ms on the real scene. Using the
    proxy to decide a pick is safe would be exactly the mistake `mission.py`
    exists to prevent. "Optimal" here means optimal *under the proxy*, and the
    only thing that settles whether that is worth anything is `week4_order.py`,
    which flies both orders.
    """
    import time

    from mission import PARK

    positions = {n: (s.est if isinstance(s, Seen) else np.asarray(s, float))
                 for n, s in survey.items()}
    if not positions:
        return OrderedPlan(optimal=True)
    start = np.asarray(PARK if start is None else start, float)

    names = sorted(positions)
    t0 = time.perf_counter()
    greedy = tuple(_greedy(names, positions, start))

    # Stage 1: the relaxation, solved exactly where that is affordable.
    use_exact = (len(names) <= EXACT_MAX) if exact is None else exact
    seeds = [list(greedy)]
    relaxed = None
    if use_exact:
        relaxed, _s = _solve_exact(names, positions, start)
        seeds.append(list(relaxed))

    # Stage 2: refine against the **true** objective, in which a refused pick
    # leaves its fruit standing. `_solve_exact` cannot see that — its whole
    # speed comes from the state being a set — so the exact answer is a very
    # good starting point rather than the finish line. Both seeds are refined
    # and the better one wins, which costs microseconds and means the result is
    # never worse than either stage alone.
    best, best_cost = None, np.inf
    for seed in seeds:
        cand = _local_search(seed, positions, start) if improve else seed
        c = _sequence_cost(cand, positions, start)["cost"]
        if c < best_cost:
            best, best_cost = cand, c
    order = best
    solve_s = time.perf_counter() - t0

    # ⚠️ "Optimal" is claimed only when the exact stage ran **and** refining it
    # against the true objective changed nothing. That is the honest reading:
    # the relaxation's optimum is also a local optimum of the real thing. If
    # refinement moved it, the exact answer was optimal for a problem this is
    # not, and saying so would be worse than saying nothing.
    proved = bool(use_exact and relaxed is not None
                  and tuple(order) == tuple(relaxed))
    plan = OrderedPlan(improved_from=greedy, optimal=proved, solve_s=solve_s)
    # The per-step table is read straight off the true walk, so what it prints
    # is what the optimiser was scored on — including a refused fruit still
    # counting against everything after it.
    s = _sequence_cost(order, positions, start)
    here = start
    for w in s["steps"]:
        n = w["fruit"]
        others = {m for m in order if m != n}
        unblocks = sum(_pair_risk(positions[m], positions[n]) for m in others)
        plan.steps.append(Step(fruit=n, pos=positions[n], risk=w["crowd"],
                               unblocks=unblocks, travel=w["travel"],
                               remaining=w["standing"], block=w["block"]))
        here = positions[n]
    plan.total_risk = s["total_risk"]
    plan.worst_risk = s["worst_risk"]
    plan.travel_m = s["travel_m"]
    plan.lost = s["lost"]
    plan.cost = s["cost"]
    return plan


def score_order(order, survey, start=None):
    """The same numbers for an order somebody else chose — the comparison.

    Scored by the identical function `plan_order` optimises against, so the two
    figures are comparable. Scoring the baseline any other way would be
    marking your own homework with a different pen.
    """
    from mission import PARK

    positions = {n: (s.est if isinstance(s, Seen) else np.asarray(s, float))
                 for n, s in survey.items()}
    return _sequence_cost(list(order), positions,
                          PARK if start is None else start)


# --- the gates ---------------------------------------------------------------

def _scene(place_board=False):
    from greenhouse import build_scene
    from week4_place import pool_trusses

    pool = pool_trusses()
    model = build_scene(wrist_cam=True, deck_cam=True, trusses=pool,
                        place_board=place_board)
    return model, [n for n, _, _ in pool]


def _fresh(model, names):
    import mujoco

    from mission import park_posture, reset_park
    from plant_row import Row
    from week4_place import park_spot

    data = mujoco.MjData(model)
    q = park_posture(model)
    reset_park(model, data, q)
    row = Row(model, data, names=names,
              homes={n: park_spot(i) for i, n in enumerate(names)})
    row.reset()
    mujoco.mj_forward(model, data)
    return data, q, row


def _assert_inert(args):
    """Both housings are drawn and *nothing else*. Checked, not argued.

    ⚠️ This is the claim `camera.add_camera_housing` makes about itself, and it
    is the one that would be expensive to get wrong: the wrist housing hangs
    90 mm off wrist3_link, exactly where `mission.ClearanceModel` measures
    gripper-to-fruit distance, and the arm's joint torque limits were checked
    against a payload that does not include a camera. Both failure modes are
    silent — every clearance number in the README would simply be a different
    number, with nothing to say it had moved.
    """
    from greenhouse import build_scene

    def probe(**kw):
        m = build_scene(**kw)
        coll = sum(1 for i in range(m.ngeom)
                   if m.geom(i).contype[0] or m.geom(i).conaffinity[0])
        return m, coll

    base, base_coll = probe()
    both, both_coll = probe(wrist_cam=True, deck_cam=True)

    # ⚠️ Compared **by name**, not by index. `deck_cam=True` adds the
    # `deck_mast` body, so the two models' body arrays are different lengths
    # and an elementwise comparison raises — which is a broadcast error rather
    # than a finding, and would be a silly reason to skip the check.
    moved = []
    for i in range(base.nbody):
        name = base.body(i).name
        if not name:
            continue
        j = both.body(name).id
        if (abs(base.body_mass[i] - both.body_mass[j]) > 1e-12
                or not np.allclose(base.body_inertia[i], both.body_inertia[j])):
            moved.append(name)

    # ⚠️ **The DOF count is the load-bearing assertion now that the head moves.**
    # `mink.Configuration` in `reach.Reacher` is built over the whole model, so
    # a pan-tilt built from hinge joints would hand the IK solver two extra
    # joints it has no reason to leave anywhere in particular — and the coupling
    # between "where the arm is reaching" and "where the camera is pointing"
    # would show up only as a survey quietly missing fruit. A mocap body has no
    # DOFs. If this line ever moves, the head has been rebuilt from joints and
    # every survey in the repo is suspect.
    dofs_ok = (base.nq, base.nv) == (both.nq, both.nv)
    ok = base_coll == both_coll and not moved and dofs_ok

    print(f"\n  --- the cameras change nothing but the picture ---")
    print(f"  drawn geoms      {base.ngeom} -> {both.ngeom}  "
          f"(+{both.ngeom - base.ngeom})")
    print(f"  collidable geoms {base_coll} -> {both_coll}"
          f"{'' if base_coll == both_coll else '   <-- CHANGED'}")
    print(f"  total mass       {base.body_mass.sum():.6f} -> "
          f"{both.body_mass.sum():.6f} kg")
    print(f"  nq / nv          {base.nq}/{base.nv} -> {both.nq}/{both.nv}"
          f"{'' if dofs_ok else '   <-- THE HEAD HAS DOFs, SEE ABOVE'}")
    print(f"  mocap bodies     {base.nmocap} -> {both.nmocap}  "
          f"(the head is the new one, appended last so the stems keep "
          f"indices 0..{base.nmocap - 1})")
    i = both.body("wrist3_link").id
    print(f"  wrist3_link      {both.body_mass[i]:.6f} kg, inertia "
          f"{both.body_inertia[i].round(8)}")
    print(f"  deck_head        {both.body('deck_head').mass[0]:.6f} kg")
    print(f"  bodies whose mass or inertia moved: {moved or 'none'}")
    print(f"  {'ok' if ok else 'FAIL — a camera has become part of the physics'}")
    return ok


def gate(args):
    """Does the mast find the row? And where does it fail?"""
    import mujoco

    from week4_place import Crop, auto_layout

    inert = _assert_inert(args)
    model, names = _scene()
    data, q, row = _fresh(model, names)
    crop = Crop(model, data, row, names)

    print(f"\n  deck camera at {DECK_MOUNT.round(3)} looking at "
          f"{DECK_AIM.round(3)}")
    print(f"  fovy {DECK_FOVY:.0f} deg · {Intrinsics.from_model(model, DECK_NAME)}")
    d = DECK_AIM - DECK_MOUNT
    print(f"  {np.degrees(np.arccos(abs(d[0]) / np.linalg.norm(d))):.0f} deg "
          f"off the row plane's normal · {np.linalg.norm(d):.2f} m to the band "
          f"centre")

    survey = DeckSurvey(model)
    ok = inert
    try:
        # --- 1. a spread row, which is what it is for ------------------------
        crop.apply(auto_layout(args.n, seed=args.seed))
        mujoco.mj_forward(model, data)
        seen, rep = survey.look(data, names)
        errs = [s.err_mm for s in seen.values()]
        print(f"\n  --- {len(crop.placed)} fruit, spread (seed {args.seed}) ---")
        print(f"  found {len(seen)}/{len(crop.placed)} in ONE frame, arm "
              f"parked, nothing moved")
        if errs:
            print(f"  position error  mean {np.mean(errs):.1f} mm · "
                  f"p95 {np.percentile(errs, 95):.1f} mm · "
                  f"max {np.max(errs):.1f} mm")
        ok &= len(seen) == len(crop.placed)
        for line in rep["log"]:
            print(f"    {line}")

        plan = plan_order(seen)
        print(f"\n  the order it chose (placement order would be "
              f"{', '.join(sorted(crop.placed))}):")
        print(plan.table())
        # Only fruit the camera actually found — scoring placement order over
        # names the survey never returned would KeyError, and silently dropping
        # them would flatter the baseline.
        told = score_order(sorted(n for n in crop.placed if n in seen), seen)
        print(f"    placement order for comparison: total risk "
              f"{told['total_risk']:.2f} · worst {told['worst_risk']:.2f} · "
              f"tour {told['travel_m']:.2f} m")

        # --- 2. a cluster, which is where it fails ---------------------------
        # ⚠️ This half is not a formality. With the 200 mm placement minimum
        # gone, fruit can be put close enough to fuse into one blob, and a
        # survey that quietly under-counts is worse than one that fails loudly.
        crop.clear()
        cluster = [(0.10, 0.60), (0.17, 0.60), (0.10, 0.67), (0.17, 0.67),
                   (-0.30, 0.55), (0.40, 0.50)]
        for y, z in cluster:
            crop.place(y, z, quiet=True)
        mujoco.mj_forward(model, data)
        print(f"\n  --- {len(crop.placed)} fruit, four of them clustered at "
              f"70 mm pitch ---")

        seen2, rep2 = survey.look(data, names)
        print(f"  bolted down, one frame:   found {len(seen2)}/"
              f"{len(crop.placed)}  ({len(rep2['usable'])} usable blobs)")

        # The same cluster, with the head allowed to move. This is the pair of
        # numbers the articulation exists for, and printing them side by side is
        # the only way the claim stays honest as either half changes.
        head = DeckHead(model, data)
        seen3, rep3 = survey.scan(data, names, head=head)
        head.home(data)
        mujoco.mj_forward(model, data)
        print(f"  head scan, {len(rep3['poses'])} poses:      found "
              f"{len(seen3)}/{len(crop.placed)}  "
              f"({len(rep3['clusters'])} fused clusters, "
              f"{rep3['slew_s']:.1f} s of head slew, arm still parked)")
        gained = sorted(set(seen3) - set(seen2))
        if gained:
            print(f"  the scan recovered: {', '.join(gained)}")
        print(f"  ⚠️ a deck camera still cannot separate *touching* fruit at "
              f"1.3 m, from any")
        print(f"     angle — moving the viewpoint 90 mm does not undo a 70 mm "
              f"separation at")
        print(f"     1300 mm. That is the limit of this sensor and the reason "
              f"the wrist camera")
        print(f"     still goes in; the head widens the band, it does not "
              f"remove the floor.")
        for line in rep3["log"]:
            print(f"    {line}")
    finally:
        survey.close()

    print(f"\n{'=' * 78}")
    print(f"  DECK GATE: {'PASS' if ok else 'FAIL'} — a spread row is found "
          f"whole from the mast")
    return 0 if ok else 1


def pairs(args):
    """What does a neighbour actually cost? The sweep behind BLOCKED_M."""
    import mujoco

    from mission import Planner
    from week4_place import park_spot

    model, names = _scene()
    centre = np.array([ROW_X, 0.0, 0.60])
    dirs = {"side by side (y)": (1.0, 0.0), "stacked (z)": (0.0, 1.0),
            "diagonal": (0.707, 0.707)}
    for label, (dy, dz) in dirs.items():
        print(f"\n  --- neighbour {label} ---")
        print(f"  {'centres mm':>10} {'surface mm':>11}  {'lower/left':<30}"
              f"{'upper/right':<30}")
        for d in (0.070, 0.085, 0.100, 0.120, 0.140, 0.170, 0.200):
            data, q, row = _fresh(model, names)
            for i, n in enumerate(names):
                row.place(n, park_spot(i))
            a = centre - np.array([0, dy * d / 2, dz * d / 2])
            b = centre + np.array([0, dy * d / 2, dz * d / 2])
            for n, p in ((names[0], a), (names[1], b)):
                row.place(n, p)
                row.home[n] = p
            mujoco.mj_forward(model, data)
            planner = Planner(model, data, row, park_q=q)
            out = []
            for n in (names[0], names[1]):
                m = planner.plan(n)
                out.append(m.tried[-1].split("(")[0] if m.ok
                           else f"REFUSED ({m.breaches[0].leg})")
            print(f"  {d * 1000:10.0f} {(d - 2 * FRUIT_R) * 1000:11.0f}  "
                  f"{out[0]:<30}{out[1]:<30}")


def corridor(args):
    """The pull-down corridor, swept. The fit behind CORRIDOR_*."""
    import mujoco

    from mission import Planner
    from week4_place import park_spot

    model, names = _scene()
    target = np.array([ROW_X, 0.0, 0.68])
    print(f"\n  target at y+0.000 z0.680, one neighbour below it, swept "
          f"sideways")
    print(f"  {'below mm':>9} {'across mm':>10}  {'route for the TARGET':<32}"
          f"{'clear mm':>9}")
    for below in (0.10, 0.15, 0.20, 0.26):
        for across in (0.0, 0.04, 0.08, 0.12, 0.16, 0.22):
            data, q, row = _fresh(model, names)
            for i, n in enumerate(names):
                row.place(n, park_spot(i))
            nb = target + np.array([0.0, across, -below])
            for n, p in ((names[0], target), (names[1], nb)):
                row.place(n, p)
                row.home[n] = p
            mujoco.mj_forward(model, data)
            m = Planner(model, data, row, park_q=q).plan(names[0])
            route = (m.tried[-1].split("(")[0] if m.ok
                     else f"REFUSED ({m.breaches[0].leg})")
            model_says = _pair_risk(target, nb)
            print(f"  {below * 1000:9.0f} {across * 1000:10.0f}  {route:<32}"
                  f"{m.clearance * 1000:9.0f}   model {model_says:.2f}")
        print()


def vs_sweep(args):
    """One deck frame against the alternative: sweeping the row with the wrist.

    This is the measurement the whole module rests on. "The deck camera saves
    the arm from scanning" is worth nothing as an assertion, because the code it
    replaced was not scanning either — it was reading `crop.placed`, the
    operator's ground truth. So the honest comparison is not against what the
    repo used to do, it is against **what a machine with only an eye-in-hand
    camera would have to do**: drive the wrist along the row, render, drive on.

    Time is counted in **simulated** seconds — control cycles x `reach.CTRL_DT`
    — not wall clock. Wall clock here measures this laptop; a robot's cycle time
    is made of the seconds the arm spends moving.
    """
    import mujoco

    from camera import SensorCamera, stage
    from detect import HSVDetector, estimate
    from mission import STAGE_X
    from reach import CTRL_DT
    from week4_place import (MARGINAL_HALF_Y, MARGINAL_Z, Crop, auto_layout)

    model, names = _scene()
    data, q, row = _fresh(model, names)
    crop = Crop(model, data, row, names)
    crop.apply(auto_layout(args.n, seed=args.seed))
    mujoco.mj_forward(model, data)
    truth = {n: data.body(n).xpos.copy() for n in crop.placed}
    detector = HSVDetector()

    # --- the deck: a head scan, arm stationary -------------------------------
    #
    # ⚠️ Both halves of this are reported, because the head does cost *something*
    # and folding it into "free" would be the same kind of quiet arithmetic this
    # module exists to undo. The claim is not that the survey is free — it is
    # that it costs no **arm** seconds, which is the quantity a cycle time is
    # made of and the quantity the wrist sweep spends 24 of.
    survey = DeckSurvey(model, detector=HSVDetector())
    try:
        one, _rep1 = survey.look(data, list(crop.placed))
        head = DeckHead(model, data)
        seen, rep = survey.scan(data, list(crop.placed), head=head)
        head.home(data)
        mujoco.mj_forward(model, data)
    finally:
        survey.close()
    print(f"\n  --- from the mast, arm stationary ---")
    print(f"  one frame, head fixed   {len(one)}/{len(crop.placed)} found · "
          f"arm moved 0.000 m · 0.00 s of arm time")
    print(f"  {len(rep['poses'])}-pose head scan     "
          f"{len(seen)}/{len(crop.placed)} found · arm moved 0.000 m · "
          f"0.00 s of arm time, {rep['slew_s']:.2f} s of head slew")

    # --- the wrist: a sweep --------------------------------------------------
    # The wrist camera sees 0.31 x 0.23 m at the 0.28 m staging distance
    # (fovy 45, 4:3), so covering the 1.10 x 0.30 m band takes a lattice with
    # overlap. Poses are visited in a serpentine so the traverse does not double
    # back on itself — the fairest version of the alternative, not a strawman.
    cols = np.linspace(-MARGINAL_HALF_Y + 0.06, MARGINAL_HALF_Y - 0.06, 5)
    rows_z = [MARGINAL_Z[0] + 0.06, MARGINAL_Z[1] - 0.04]
    poses = []
    for i, z in enumerate(rows_z):
        poses += [(float(y), float(z))
                  for y in (cols if i % 2 == 0 else cols[::-1])]

    wrist = SensorCamera(model, camera="wrist")
    found, ticks, path = {}, [0], []
    try:
        for y, z in poses:
            before = data.site("tool0").xpos.copy()
            stage(model, data, q, np.array([STAGE_X, y, z]), row=row,
                  speed=args.speed, reset=None,
                  on_tick=lambda _t=None: ticks.__setitem__(0, ticks[0] + 1))
            path.append(float(np.linalg.norm(
                data.site("tool0").xpos - before)))
            rgb, depth = wrist.both(data)
            R, C = wrist.pose(data)
            for d in (estimate(x, depth, wrist.intr, R, C)
                      for x in detector(rgb)):
                if d.est is None:
                    continue
                for n in crop.placed:
                    if np.linalg.norm(d.est - truth[n]) <= DECK_GATE_M:
                        found.setdefault(n, d.est)
    finally:
        wrist.close()

    secs = ticks[0] * CTRL_DT
    print(f"\n  --- a wrist sweep of the same row ---")
    print(f"  {len(found)}/{len(crop.placed)} found · {len(poses)} poses · "
          f"arm moved {sum(path):.3f} m · {secs:.2f} s of arm time")
    print(f"  ({secs / max(len(crop.placed), 1):.2f} s per fruit of scanning, "
          f"before a single pick has started)")
    print(f"\n  ⚠️ Both are run at --speed {args.speed}. The sweep's cost is "
          f"arm motion, so it")
    print(f"     scales with the speed and with the length of row; the deck "
          f"frame's cost is")
    print(f"     one render, and does not scale with either.")


def mounts(args):
    """Score candidate mounts on one frame each. How DECK_MOUNT was chosen.

    Every candidate goes into one model as its own camera, so they are all
    scored against a byte-identical scene rather than against separately built
    ones — which is the only way the two-millimetre differences between them
    mean anything.
    """
    import mujoco

    from camera import occluder, project
    from detect import HSVDetector, estimate
    from fr5 import add_crate, build_fr5_spec
    from greenhouse import add_greenhouse
    from mission import BIN_HALF, BIN_POS, BIN_WALL
    from plant_row import add_row
    from week4_place import park_spot, pool_trusses

    candidates = {  # (pos, fovy) — the post line comes from `--sweep`
        "x-0.10 z1.45": ([-0.10, 0.0, 1.45], 58.0),
        "x-0.30 z1.35": ([-0.30, 0.0, 1.35], 58.0),
        "x-0.60 z0.95": ([DECK_POST_X, 0.0, 0.95], 58.0),
        "x-0.60 z1.10": ([DECK_POST_X, 0.0, 1.10], 58.0),
        "x-0.60 z1.25": ([DECK_POST_X, 0.0, 1.25], 58.0),
        "x-0.60 z1.40": ([DECK_POST_X, 0.0, 1.40], 58.0),
        "x-0.60 z1.60": ([DECK_POST_X, 0.0, 1.60], 58.0),
    }
    pool = pool_trusses()
    spec = build_fr5_spec(with_scene=False, gripper=True)
    add_row(spec, trusses=pool)
    for geom in spec.geoms:
        if "pad" in geom.name:
            geom.solref = [0.02, 1.0]
    add_crate(spec, name="bin", pos=BIN_POS, half=BIN_HALF, wall=BIN_WALL)
    for label, (pos, fovy) in candidates.items():
        spec.worldbody.add_camera(
            name=f"cand{len(spec.cameras)}", pos=list(pos), fovy=fovy,
            xyaxes=_xyaxes_towards(pos, DECK_AIM, up_hint=(0.0, 0.0, 1.0)))
    add_greenhouse(spec)
    model = spec.compile()
    names = [n for n, _, _ in pool]
    data, q, row = _fresh(model, names)

    # The whole band, corner to corner.
    spots = [(float(y), float(z)) for z in np.linspace(0.42, 0.72, 3)
             for y in np.linspace(-0.55, 0.55, 7)]
    used = names[: len(spots)]
    for n, (y, z) in zip(used, spots):
        row.place(n, np.array([ROW_X, y, z]))
    for i, n in enumerate(names[len(spots):], start=len(spots)):
        row.place(n, park_spot(i))
    mujoco.mj_forward(model, data)
    truth = {n: data.body(n).xpos.copy() for n in used}
    detector = HSVDetector()

    def score_all(label_truth):
        out = {}
        for i, (label, (pos, _f)) in enumerate(candidates.items()):
            cam = f"cand{i}"
            sensor = SensorCamera(model, cam)
            try:
                rgb, depth = sensor.both(data)
                R, C = sensor.pose(data)
                dets = [estimate(x, depth, sensor.intr, R, C)
                        for x in detector(rgb)]
                usable = [x for x in dets if x.est is not None]
                pairs_ = sorted(
                    (float(np.linalg.norm(x.est - label_truth[n])), j, n)
                    for j, x in enumerate(usable) for n in used
                    if np.linalg.norm(x.est - label_truth[n]) < 0.12)
                td, tn, errs = set(), set(), []
                for dist, j, n in pairs_:
                    if j in td or n in tn:
                        continue
                    td.add(j)
                    tn.add(n)
                    errs.append(dist * 1000)
                margins, clear = [], 0
                for n in used:
                    u, v2 = project(sensor.intr, R, C, label_truth[n])
                    margins.append(min(u, sensor.intr.width - u, v2,
                                       sensor.intr.height - v2))
                    if occluder(model, data, C, label_truth[n], n) is None:
                        clear += 1
                rngs = [float(np.linalg.norm(label_truth[n] - C))
                        for n in used]
                out[label] = (len(errs), np.mean(errs) if errs else np.nan,
                              min(margins), clear, min(rngs), max(rngs))
            finally:
                sensor.close()
        return out

    parked_scores = score_all(truth)

    # ⚠️ The second pass is what makes the "survey only at park" rule a
    # measurement rather than a preference. With the arm driven out to the
    # staging plane it stands between the mast and the row, and the low mounts
    # start losing fruit behind it — silently, as a shorter survey.
    from camera import stage
    from mission import STAGE_X

    stage(model, data, q, np.array([STAGE_X, 0.0, 0.62]), row=row, speed=0.6,
          reset=None)
    staged_truth = {n: data.body(n).xpos.copy() for n in used}
    staged_scores = score_all(staged_truth)

    print(f"\n  {len(used)} fruit across the whole band, one frame each\n")
    print(f"  {'mount':<14} {'incid':>6} {'range m':>12} {'parked':>8} "
          f"{'staged':>8} {'mean mm':>8} {'edge px':>8}")
    for label, (pos, _f) in candidates.items():
        v = DECK_AIM - np.asarray(pos, float)
        incid = np.degrees(np.arccos(abs(v[0]) / np.linalg.norm(v)))
        found, mean, margin, _clear, lo, hi = parked_scores[label]
        s_found = staged_scores[label][0]
        print(f"  {label:<14} {incid:5.0f}° {lo:5.2f}-{hi:<5.2f} "
              f"{found:4d}/{len(used):<3} {s_found:4d}/{len(used):<3} "
              f"{mean:8.1f} {margin:8.0f}")
    print(f"\n  chosen: x{DECK_POST_X:+.2f} z{DECK_MOUNT[2]:.2f} — see "
          f"DECK_MOUNT for why the post line is fixed first, and why the "
          f"survey only ever runs from the parked column")


def sweep(args):
    """Where does the arm actually go? The measurement that sites the post."""
    import mujoco

    from mission import robot_geoms
    from week4_place import Crop, auto_layout, harvest_placed

    model, names = _scene()
    data, q, row = _fresh(model, names)
    crop = Crop(model, data, row, names)
    crop.apply(auto_layout(args.n, seed=args.seed))

    gids = robot_geoms(model)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    posts = {x: np.inf for x in (-0.10, -0.30, -0.45, -0.60, -0.75)}

    def tick(_t=None):
        nonlocal lo, hi
        for g in gids:
            p = data.geom_xpos[g]
            r = float(np.max(model.geom_size[g])) or 0.02
            lo = np.minimum(lo, p - r)
            hi = np.maximum(hi, p + r)
            for px in posts:
                posts[px] = min(posts[px],
                                float(np.hypot(p[0] - px, p[1])) - r)

    harvest_placed(model, data, row, crop, q, speed=args.speed, on_tick=tick)
    print(f"\n  arm swept volume over {len(crop.placed)} picks "
          f"({len(gids)} collision geoms)")
    for i, ax in enumerate("xyz"):
        print(f"    {ax}  {lo[i]:+.3f} .. {hi[i]:+.3f}")
    print(f"\n  closest approach to a vertical post on the y=0 line:")
    for px, d in posts.items():
        note = "  <- inside the arm" if d < 0 else ""
        print(f"    x {px:+.2f}   {d * 1000:7.0f} mm{note}")


def shot(args):
    """Render stills that show where both cameras are, and what each one sees."""
    import cv2
    import mujoco

    from week4_place import Crop, auto_layout

    model, names = _scene()
    data, q, row = _fresh(model, names)
    crop = Crop(model, data, row, names)
    crop.apply(auto_layout(args.n, seed=args.seed))
    mujoco.mj_forward(model, data)

    out_dir = Path(__file__).resolve().parents[2]
    written = []
    with mujoco.Renderer(model, height=960, width=1280) as r:
        for cam in ("wide", "aisle", DECK_NAME, "wrist"):
            r.update_scene(data, camera=cam)
            path = out_dir / f"deck_cam_{cam}.png"
            cv2.imwrite(str(path), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
            written.append(path.name)

        # A close look at the mast itself, from a free camera, because the
        # cinematic framings are aimed at the crop and the hardware is the
        # point of this one.
        free = mujoco.MjvCamera()
        free.lookat[:] = [DECK_MOUNT[0] + 0.25, 0.0, 0.85]
        free.distance, free.azimuth, free.elevation = 2.2, -55.0, -12.0
        r.update_scene(data, camera=free)
        path = out_dir / "deck_cam_mast.png"
        cv2.imwrite(str(path), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        written.append(path.name)

        # The head itself, at the two extremes of the scan. Two stills rather
        # than one because articulation is the thing a still cannot show — the
        # only way to make "it looks around" visible in a static image is to
        # take the image twice and let the yoke be somewhere different.
        head = DeckHead(model, data)
        close = mujoco.MjvCamera()
        close.lookat[:] = [DECK_PAN_AXIS_X + 0.05, 0.0, DECK_MOUNT[2]]
        close.distance, close.azimuth, close.elevation = 0.75, -50.0, -8.0
        for tag, (pan, tilt) in (("home", (0.0, 0.0)),
                                 ("panned", SCAN_POSES[-1])):
            head.aim(data, pan, tilt)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=close)
            path = out_dir / f"deck_cam_head_{tag}.png"
            cv2.imwrite(str(path), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
            written.append(path.name)
        head.home(data)
        mujoco.mj_forward(model, data)
    print("  wrote " + ", ".join(written))
    print(f"  deck_cam_head_home.png / _panned.png are the same hardware at "
          f"pan 0 and pan {SCAN_POSES[-1][0]:+.0f}, tilt "
          f"{SCAN_POSES[-1][1]:+.0f} — the yoke is what moves the lens, and "
          f"moving the lens is the whole reason the head is there.")


def _cluster_row(n, seed, tight=0.10):
    """A layout with fruit deliberately packed, for testing what a scan sees.

    The old 200 mm placement minimum made this arrangement impossible, which is
    why the fixed camera was never pushed on it. It is now the normal case.
    """
    from week4_place import GUARANTEED_HALF_Y, GUARANTEED_Z

    rng = np.random.default_rng(seed)
    y_half, (z_lo, z_hi) = GUARANTEED_HALF_Y, GUARANTEED_Z

    out, tries = [], 0
    while len(out) < n and tries < 4000:
        tries += 1
        # Mostly grow the cluster off an existing fruit, so the layout is packed
        # rather than merely random — a uniform scatter over this box would put
        # almost nothing inside the band being tested.
        if out and rng.random() < 0.6:
            base = out[rng.integers(len(out))]
            a = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0.072, tight)
            y, z = base[0] + d * np.cos(a), base[1] + d * np.sin(a)
        else:
            y, z = rng.uniform(-y_half, y_half), rng.uniform(z_lo, z_hi)
        if not (-y_half <= y <= y_half and z_lo <= z <= z_hi):
            continue
        # 72 mm is just past touching; closer than that and they interpenetrate.
        if any(np.hypot(y - a, z - b) < 0.072 for a, b in out):
            continue
        out.append((y, z))
    return out


# Candidate scan patterns, in (pan, tilt) degrees from home. The head is on a
# 100 mm yoke, so each of these is also a viewpoint 0-90 mm from the last one.
SCAN_PATTERNS = {
    "fixed (1 pose)": ((0.0, 0.0),),
    "pan +-20 (3)": ((0.0, 0.0), (-20.0, 0.0), (20.0, 0.0)),
    "pan +-35 (3)": ((0.0, 0.0), (-35.0, 0.0), (35.0, 0.0)),
    "pan +-35 (5)": ((0.0, 0.0), (-18.0, 0.0), (18.0, 0.0),
                     (-35.0, 0.0), (35.0, 0.0)),
    "cross (5)": ((0.0, 0.0), (-25.0, 0.0), (25.0, 0.0),
                  (0.0, -12.0), (0.0, 12.0)),
    "box (5)": ((0.0, 0.0), (-25.0, -10.0), (25.0, -10.0),
                (-25.0, 10.0), (25.0, 10.0)),
}


def scan_cmd(args):
    """Is looking around worth it? Count what each pattern finds.

    ⚠️ **The thing to be sceptical about is whether articulation buys anything
    at all**, and the reason is geometric: a camera rotating about its own
    optical centre sees a different part of the room but has exactly the same
    view of everything it can still see. Occlusions do not move, blobs that
    merge still merge. Panning alone is worth nothing here, because one fixed
    frame already covers the whole placement band with room to spare — that is
    what `DECK_FOVY` was chosen for.

    What can be worth something is the 100 mm yoke, which turns pan into
    *translation*. This measures whether it does, on the layouts that actually
    hurt: clusters at 72-100 mm centres, which the old spacing rule made
    impossible and which `week4_order`'s 75 mm row showed the fixed camera
    losing three fruit of six to.
    """
    import mujoco

    from week4_place import Crop

    model, names = _scene()
    survey = DeckSurvey(model)
    print(f"\n  --- what looking around finds, {args.n} clustered fruit x "
          f"{args.trials} layouts ---")
    print(f"  fruit are placed 72-100 mm apart on purpose: that is the band the "
          f"old\n  200 mm rule forbade, and the band the fixed camera loses "
          f"fruit in.\n")
    print(f"  {'pattern':<16} {'poses':>6} {'found':>9} {'err mm':>8} "
          f"{'worst mm':>9} {'phantom':>8} {'slew s':>8}")

    for label, poses in SCAN_PATTERNS.items():
        tot = seen_n = phantom = 0
        errs = []
        slew = 0.0
        for t in range(args.trials):
            data, q, row = _fresh(model, names)
            crop = Crop(model, data, row, names)
            crop.park_all()
            for y, z in _cluster_row(args.n, seed=args.seed + t):
                crop.place(y, z, quiet=True)
            mujoco.mj_forward(model, data)
            standing = list(crop.placed)
            head = DeckHead(model, data)
            seen, rep = survey.scan(data, standing, head=head, poses=poses)
            head.home(data)
            mujoco.mj_forward(model, data)
            tot += len(standing)
            seen_n += len(seen)
            slew += rep["slew_s"]
            errs += [s.err_mm for s in seen.values()]
            # A cluster that matched no truss is a fruit the survey invented.
            phantom += sum(1 for line in rep["log"] if "UNASSOCIATED" in line)
        print(f"  {label:<16} {len(poses):>6} {seen_n:>4}/{tot:<4} "
              f"{np.mean(errs):>8.1f} {max(errs):>9.1f} {phantom:>8} "
              f"{slew / args.trials:>8.2f}")

    print(f"\n  `found` is the count that matters — a fruit the survey never "
          f"separates\n  is a fruit the arm is never sent to. `phantom` is the "
          f"opposite failure and\n  is the one to watch when adding poses: two "
          f"sightings of one fruit that\n  fail to fuse become two commands to "
          f"pick the same tomato.")
    print(f"\n  ⚠️ The arm does not move for any of this. `slew s` is the head's "
          f"own\n  travel at DECK_SLEW_DEG_S={DECK_SLEW_DEG_S:.0f} deg/s — "
          f"compare it against the\n  24.14 s of *arm* motion a wrist sweep "
          f"costs in `--vs-sweep`.")


def eclipse_cmd(args):
    """Can the head see round the arm? The survey's one hard precondition.

    `parked()` refuses to survey unless the arm is at park, because staged
    mid-row the arm eclipses a fruit and the survey comes back one short with
    nothing to say so. That rule costs a re-park every time the crop changes.
    A head that can look from a different position is the first thing that could
    honestly relax it — but only if the yoke's translation is enough to see past
    an arm that is much closer to the camera than the fruit are.
    """
    import mujoco

    from camera import stage
    from mission import STAGE_X
    from week4_place import Crop, auto_layout

    model, names = _scene()
    survey = DeckSurvey(model)
    print(f"\n  --- can the head see round the arm? ---")
    print(f"  {'arm':<22} {'pattern':<16} {'found':>9} {'err mm':>8}")

    for arm_label, staged_at in (("parked", None), ("staged mid-row", 0.0),
                                 ("staged high", 0.30)):
        for label, poses in (("fixed (1 pose)", SCAN_PATTERNS["fixed (1 pose)"]),
                             ("pan +-35 (5)", SCAN_PATTERNS["pan +-35 (5)"])):
            data, q, row = _fresh(model, names)
            crop = Crop(model, data, row, names)
            crop.apply(auto_layout(args.n, seed=args.seed))
            mujoco.mj_forward(model, data)
            if staged_at is not None:
                stage(model, data, q,
                      np.array([STAGE_X, staged_at, 0.60]), row=row,
                      speed=0.5, reset="arm")
            standing = list(crop.placed)
            head = DeckHead(model, data)
            seen, _rep = survey.scan(data, standing, head=head, poses=poses)
            errs = [s.err_mm for s in seen.values()] or [float("nan")]
            print(f"  {arm_label:<22} {label:<16} {len(seen):>4}/"
                  f"{len(standing):<4} {np.mean(errs):>8.1f}")
    print(f"\n  If the scan recovers what the fixed camera loses to the arm, "
          f"`parked()`\n  has an alternative. If it does not, the rule stands "
          f"and the head's value\n  is elsewhere — say so either way.")


def optimal_cmd(args):
    """How far from optimal was the old hill-climb? And what does exact cost?

    ⚠️ **"Optimal" here means optimal under the proxy and nothing more.** The
    only thing that settles whether a better order harvests more tomatoes is
    `week4_order.py`, which flies both. What this measures is the search, not
    the model: given the cost function, how much was being left on the table by
    stopping at a local minimum.
    """
    import time

    from mission import PARK

    start = np.asarray(PARK, float)
    rng = np.random.default_rng(args.seed)
    print(f"\n  --- exact against the hill-climb, {args.trials} layouts each ---")
    print(f"  cost is expected fruit lost (W_LOST) + crowding + travel; lower "
          f"is better.\n  `lost` is the part that is whole tomatoes.\n")
    print(f"  {'n':>3} {'method':<22} {'cost':>8} {'lost':>7} {'tour m':>8} "
          f"{'ms':>8} {'beaten':>8}")

    for n in args.sizes:
        rows = {"placement order": [], "greedy": [], "greedy + hill-climb": [],
                "exact on relaxation": [], "exact + refined (ships)": []}
        times = {k: 0.0 for k in rows}
        beaten = {k: 0 for k in rows}
        for t in range(args.trials):
            pos = {}
            for i in range(n):
                # Clustered, not uniform: a scattered row has no ordering
                # problem to solve and every method ties.
                if pos and rng.random() < 0.5:
                    base = pos[list(pos)[rng.integers(len(pos))]]
                    a = rng.uniform(0, 2 * np.pi)
                    d = rng.uniform(0.075, 0.16)
                    y, z = base[1] + d * np.cos(a), base[2] + d * np.sin(a)
                    y = float(np.clip(y, -0.37, 0.37))
                    z = float(np.clip(z, 0.50, 0.70))
                else:
                    y, z = rng.uniform(-0.37, 0.37), rng.uniform(0.50, 0.70)
                pos[f"p{i:02d}"] = np.array([ROW_X, y, z])

            names = sorted(pos)
            orders = {}
            t0 = time.perf_counter()
            orders["placement order"] = list(names)
            times["placement order"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            g = _greedy(names, pos, start)
            times["greedy"] += time.perf_counter() - t0
            orders["greedy"] = g

            t0 = time.perf_counter()
            orders["greedy + hill-climb"] = _local_search(g, pos, start)
            times["greedy + hill-climb"] += time.perf_counter() - t0

            ex, secs = _solve_exact(names, pos, start)
            times["exact on relaxation"] += secs
            orders["exact on relaxation"] = ex

            t0 = time.perf_counter()
            orders["exact + refined (ships)"] = _local_search(ex, pos, start)
            times["exact + refined (ships)"] += secs + time.perf_counter() - t0

            best = min(_sequence_cost(o, pos, start)["cost"]
                       for o in orders.values())
            for k, o in orders.items():
                s = _sequence_cost(o, pos, start)
                rows[k].append(s)
                if s["cost"] > best + 1e-9:
                    beaten[k] += 1

        for k in ("placement order", "greedy", "greedy + hill-climb",
                  "exact on relaxation", "exact + refined (ships)"):
            rs = rows[k]
            print(f"  {n:>3} {k:<22} {np.mean([r['cost'] for r in rs]):>8.3f} "
                  f"{np.mean([r['lost'] for r in rs]):>7.3f} "
                  f"{np.mean([r['travel_m'] for r in rs]):>8.2f} "
                  f"{1000 * times[k] / args.trials:>8.1f} "
                  f"{beaten[k]:>4}/{args.trials:<3}")
        print()

    print("  `beaten` counts layouts where that method did not reach the best")
    print("  cost any method found.")
    print()
    print("  ⚠️ **`exact on relaxation` is expected to lose rows here, and that")
    print("  is the finding rather than a bug.** It is optimal for the problem")
    print("  where every attempt removes its fruit; the column it is scored in")
    print("  is the real one, where a refused pick leaves the fruit standing and")
    print("  blocking. The gap between those two rows is the cost of that one")
    print("  assumption — see `_sequence_cost`.")


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimal", action="store_true",
                    help="exact solver against the hill-climb it replaced")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[6, 9, 12, 15],
                    help="row sizes for --optimal")
    ap.add_argument("--scan", action="store_true",
                    help="score scan patterns — sets SCAN_POSES")
    ap.add_argument("--eclipse", action="store_true",
                    help="can the head see round a staged arm?")
    ap.add_argument("--trials", type=int, default=6,
                    help="layouts per pattern for --scan")
    ap.add_argument("--pairs", action="store_true",
                    help="sweep what a neighbour costs the planner")
    ap.add_argument("--corridor", action="store_true",
                    help="sweep the pull-down corridor behind CORRIDOR_*")
    ap.add_argument("--mounts", action="store_true",
                    help="score the candidate mounts behind DECK_MOUNT")
    ap.add_argument("--sweep", action="store_true",
                    help="measure the arm's swept volume — sites the post")
    ap.add_argument("--vs-sweep", action="store_true",
                    help="one deck frame vs sweeping the row with the wrist")
    ap.add_argument("--shot", action="store_true",
                    help="render stills showing both cameras")
    ap.add_argument("-n", type=int, default=8, help="fruit to place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.5,
                    help="joint speed for --sweep (geometry only, so the "
                         "measurement does not need the 0.15 the pick rates "
                         "were taken at)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    for flag, fn in (("pairs", pairs), ("corridor", corridor),
                     ("mounts", mounts), ("sweep", sweep),
                     ("vs_sweep", vs_sweep), ("shot", shot),
                     ("scan", scan_cmd), ("eclipse", eclipse_cmd),
                     ("optimal", optimal_cmd)):
        if getattr(args, flag):
            return fn(args) or 0
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
