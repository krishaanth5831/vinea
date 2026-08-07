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
on. Both measured on the same 8-fruit row by `main --vs-sweep`, counting
*simulated* arm seconds because a robot's cycle time is made of the seconds the
arm spends moving and not of how fast this laptop renders:

    one deck frame     8/8 found    arm moved 0.000 m     0.00 s
    a wrist sweep      8/8 found    arm moved 2.557 m    24.14 s   (10 poses)

24 seconds is most of a whole pick — the campaign's mean cycle is 31.3 s — spent
before the first tomato is touched, and it is paid again on every row. Both find
everything, so this is not an accuracy argument; the sweep's cost is arm motion
and scales with the length of the row, and the deck frame's does not scale at
all.

So this adds the sensor a real harvester has and this repo did not: a fixed
RGBD camera on the chassis, facing the plane the fruit hang in, that sees the
whole working band in **one frame with the arm stationary**. It does two jobs:

    1. **It tells the wrist camera where to look.** The survey is the row map.
       No sweep, no ground truth, and the wrist camera goes straight to a
       staging pose it was aimed at rather than one it found.
    2. **It decides the order.** Given where the fruit actually are, it plans
       which to pick first — see `plan_order`. That is not cosmetic: the sweep
       in `main --pairs` shows two fruit 100 mm apart where one is refused
       outright and the other plans fine, so picking the plannable one first is
       the difference between harvesting one and harvesting both.

The division of labour is the point, and it is the one real greenhouse machines
settle on. The deck camera is **wide, far and coarse**; the wrist camera is
**narrow, near and precise**. Neither can do the other's job:

    deck    1.26-1.48 m range, whole band in frame, ~2 mm on an isolated fruit
            and **blind to the difference between two touching ones** — see the
            cluster measurement in `main`.
    wrist   0.28 m range, one fruit, sub-millimetre geometry (`camera.py`'s
            Step 3 gate) — and it has to be driven there to see anything.

Every constant in here is read off a sweep that can be re-run:

    ./.venv/bin/python simulation/mujoco/deck_cam.py             # the gate
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


def add_deck_camera(spec, name=DECK_NAME, mount=DECK_MOUNT, aim=DECK_AIM,
                    fovy=DECK_FOVY, post=True):
    """Put the deck camera, its post and its housing into a spec.

    Call before `spec.compile()`. The post is drawn by default because a camera
    floating 1.1 m above the floor with nothing holding it up invites the
    question "how is that mounted", and the answer — a post on the chassis,
    behind the arm's swept volume — is the interesting part.

    ⚠️ Everything here is `contype=0`, like `greenhouse.py`'s scenery and for
    the same reason: the collision set this repo's clearance numbers were
    measured against is the fruit, the support bar, the row panel, the floor and
    the crate. Adding a solid post 600 mm behind the shoulder would change what
    the planner routes around, and it would change it invisibly. The post is
    sited where the arm does not go, so it *could* be solid — the measurement
    above is what says so — but making it solid is a separate change with its
    own numbers to re-take.
    """
    import mujoco

    xyaxes = _xyaxes_towards(mount, aim, up_hint=(0.0, 0.0, 1.0))
    body = spec.worldbody.add_body(name="deck_mast", pos=[0, 0, 0])
    spec.worldbody.add_camera(name=name, pos=list(mount), fovy=fovy,
                              xyaxes=xyaxes)

    if post:
        x, y = float(mount[0]), float(mount[1])
        top = float(mount[2]) - 0.05
        body.add_geom(
            name="deck_post", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[x, y, 0.02, x, y, top], size=[0.022, 0, 0],
            rgba=[0.42, 0.44, 0.47, 1.0], contype=0, conaffinity=0, mass=0.0)
        # A foot, so it reads as bolted to a deck rather than stuck in the floor.
        body.add_geom(
            name="deck_post_foot", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, 0.015], size=[0.09, 0.09, 0.015],
            rgba=[0.35, 0.37, 0.40, 1.0], contype=0, conaffinity=0, mass=0.0)

    add_camera_housing(body, f"cam_{name}", mount, xyaxes, kind="d435",
                       stalk=[float(mount[0]), float(mount[1]),
                              float(mount[2]) - 0.05])
    return spec


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

# What the two halves of the cost are worth against each other.
#
# ⚠️ **Only `W_RISK` and `W_UNBLOCK` change anything measurable today, and
# `W_TRAVEL` is in here anyway — on purpose.** `mission.park_arm` teleports the
# arm back to park between picks (for posture reasons, documented there), so the
# distance from one fruit to the next costs this simulation exactly nothing and
# tour length cannot show up in a measured cycle time. A real machine pays it.
# So it is weighted low, it breaks ties between equally safe orders in the
# direction a real machine would want, and `plan_order` reports it as its own
# column rather than folding it into a single score that would imply the sim had
# measured it. If the teleport is ever replaced by a driven return, raise it and
# the numbers become real without the model changing.
W_RISK = 1.0
W_UNBLOCK = 0.6
W_TRAVEL = 0.08


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
    """How exposed a pick of `name` is, given what is still on the plant."""
    p = positions[name]
    return sum(_pair_risk(p, positions[n]) for n in remaining if n != name)


@dataclass
class Step:
    """One pick in an ordered plan, with the reason it sits where it does."""

    fruit: str
    pos: np.ndarray
    risk: float
    unblocks: float
    travel: float
    remaining: int

    @property
    def verdict(self):
        if self.risk >= 1.0:
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

    @property
    def order(self):
        return [s.fruit for s in self.steps]

    def table(self, indent="  "):
        out = [f"{indent}{'#':>2} {'fruit':<6} {'y':>7} {'z':>7} {'risk':>6} "
               f"{'unblocks':>9} {'travel m':>9} {'left':>5}  note"]
        for i, s in enumerate(self.steps, 1):
            out.append(f"{indent}{i:2d} {s.fruit:<6} {s.pos[1]:+7.3f} "
                       f"{s.pos[2]:7.3f} {s.risk:6.2f} {s.unblocks:9.2f} "
                       f"{s.travel:9.3f} {s.remaining:5d}  {s.verdict}")
        out.append(f"{indent}total risk {self.total_risk:.2f} · worst single "
                   f"pick {self.worst_risk:.2f} · tour {self.travel_m:.2f} m")
        # Whether the improvement pass actually moved anything. Worth printing:
        # if it never fires, the greedy answer was already optimal under this
        # cost and the second stage is dead weight that should be deleted.
        if self.improved_from and tuple(self.order) != tuple(self.improved_from):
            moved = sum(1 for a, b in zip(self.order, self.improved_from)
                        if a != b)
            out.append(f"{indent}(greedy gave {' '.join(self.improved_from)}; "
                       f"the improvement pass moved {moved})")
        return "\n".join(out)


def _sequence_cost(order, positions, start):
    """Score a whole order.

    ⚠️ The obstacle set **shrinks as it goes**, which is the entire reason order
    matters and the reason this cannot be a sum of pairwise scores computed up
    front. A fruit's risk is whatever is still standing when its turn comes, so
    the same fruit is cheap ninth and expensive first.
    """
    remaining = set(order)
    here = np.asarray(start, float)
    cost = travel = total = worst = 0.0
    for n in order:
        r = risk(n, positions, remaining)
        step = float(np.linalg.norm(positions[n] - here))
        cost += W_RISK * r + W_TRAVEL * step
        total += r
        travel += step
        worst = max(worst, r)
        remaining.discard(n)
        here = positions[n]
    return {"cost": cost, "total_risk": total, "worst_risk": worst,
            "travel_m": travel}


def plan_order(survey, start=None, improve=True):
    """Decide what to pick first. Returns an `OrderedPlan`.

    `survey` is `{name: position}` — whatever the deck camera came back with,
    which is emphatically not the order the fruit were declared or placed in.

    **Two stages, and the second one is what makes this an optimisation rather
    than a sort.**

    *Greedy.* At each step, score every fruit still on the plant on what it
    costs to pick **now** (`risk`, against the fruit still standing), minus what
    picking it is worth to everything left (`unblocks` — the risk it is
    currently imposing on others), plus the travel to reach it. Take the
    cheapest. The obstacle set then shrinks and every remaining score changes,
    which is why this is re-scored per step instead of sorted once.

    Those two terms genuinely pull against each other and neither one alone is
    right. Risk alone says "do the easy ones first" and leaves a knot of
    mutually blocking fruit for the end. Unblocking alone says "go straight for
    the fruit in everyone's way", which is the one most likely to be refused.

    *Improve.* Greedy commits early and cannot undo it, so the order is then
    walked through every pairwise swap and every single-fruit relocation, taking
    any that lowers the whole-sequence cost, until nothing does. n is at most 15
    here so this is microseconds; it is worth having because the greedy answer
    is regularly one swap off the best one.

    ⚠️ **This orders picks, it does not verify them.** Whether a route exists is
    `mission.Planner`'s job and it stays there — this model is a cheap geometric
    proxy that runs in microseconds on positions from a camera, and the planner
    is a kinematic replay that runs in ~150 ms on the real scene. Using the
    proxy to decide a pick is safe would be exactly the mistake `mission.py`
    exists to prevent.
    """
    from mission import PARK

    positions = {n: (s.est if isinstance(s, Seen) else np.asarray(s, float))
                 for n, s in survey.items()}
    if not positions:
        return OrderedPlan()
    start = np.asarray(PARK if start is None else start, float)

    # --- greedy ---------------------------------------------------------------
    remaining = set(positions)
    here = start
    order = []
    while remaining:
        best, best_cost = None, np.inf
        for n in remaining:
            r = risk(n, positions, remaining)
            others = remaining - {n}
            unblocks = sum(_pair_risk(positions[m], positions[n])
                           for m in others)
            cost = (W_RISK * r - W_UNBLOCK * unblocks
                    + W_TRAVEL * float(np.linalg.norm(positions[n] - here)))
            if cost < best_cost:
                best, best_cost = n, cost
        order.append(best)
        remaining.discard(best)
        here = positions[best]

    greedy = tuple(order)

    # --- improve --------------------------------------------------------------
    if improve and len(order) > 2:
        best_cost = _sequence_cost(order, positions, start)["cost"]
        changed = True
        while changed:
            changed = False
            for i in range(len(order)):
                for j in range(len(order)):
                    if i == j:
                        continue
                    trial = list(order)
                    trial[i], trial[j] = trial[j], trial[i]      # swap
                    c = _sequence_cost(trial, positions, start)["cost"]
                    if c < best_cost - 1e-9:
                        order, best_cost, changed = trial, c, True
                        continue
                    trial = list(order)
                    trial.insert(j, trial.pop(i))                # relocate
                    c = _sequence_cost(trial, positions, start)["cost"]
                    if c < best_cost - 1e-9:
                        order, best_cost, changed = trial, c, True

    # --- write it out ---------------------------------------------------------
    plan = OrderedPlan(improved_from=greedy)
    remaining = set(order)
    here = start
    for n in order:
        r = risk(n, positions, remaining)
        others = remaining - {n}
        unblocks = sum(_pair_risk(positions[m], positions[n]) for m in others)
        step = float(np.linalg.norm(positions[n] - here))
        remaining.discard(n)
        plan.steps.append(Step(fruit=n, pos=positions[n], risk=r,
                               unblocks=unblocks, travel=step,
                               remaining=len(remaining)))
        here = positions[n]
    s = _sequence_cost(order, positions, start)
    plan.total_risk = s["total_risk"]
    plan.worst_risk = s["worst_risk"]
    plan.travel_m = s["travel_m"]
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
    ok = base_coll == both_coll and not moved

    print(f"\n  --- the housings change nothing but the picture ---")
    print(f"  drawn geoms      {base.ngeom} -> {both.ngeom}  "
          f"(+{both.ngeom - base.ngeom})")
    print(f"  collidable geoms {base_coll} -> {both_coll}"
          f"{'' if base_coll == both_coll else '   <-- CHANGED'}")
    print(f"  total mass       {base.body_mass.sum():.6f} -> "
          f"{both.body_mass.sum():.6f} kg")
    i = both.body("wrist3_link").id
    print(f"  wrist3_link      {both.body_mass[i]:.6f} kg, inertia "
          f"{both.body_inertia[i].round(8)}")
    print(f"  bodies whose mass or inertia moved: {moved or 'none'}")
    print(f"  {'ok' if ok else 'FAIL — a camera has become part of the physics'}")
    return ok


def gate(args):
    """Does one frame from the chassis find the row? And where does it fail?"""
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
        seen2, rep2 = survey.look(data, names)
        print(f"\n  --- {len(crop.placed)} fruit, four of them clustered at "
              f"70 mm pitch ---")
        print(f"  found {len(seen2)}/{len(crop.placed)}  "
              f"({len(rep2['usable'])} usable blobs for "
              f"{len(crop.placed)} fruit)")
        print(f"  ⚠️ a deck camera cannot separate touching fruit at 1.3 m. "
              f"That is the")
        print(f"     limit of this sensor and the reason the wrist camera "
              f"still goes in —")
        print(f"     it is not a bug to be tuned out at this range.")
        for line in rep2["log"]:
            print(f"    {line}")
    finally:
        survey.close()

    print(f"\n{'=' * 78}")
    print(f"  DECK GATE: {'PASS' if ok else 'FAIL'} — a spread row is found "
          f"whole in one frame")
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

    # --- the deck: one frame, arm stationary ---------------------------------
    survey = DeckSurvey(model, detector=HSVDetector())
    try:
        seen, _rep = survey.look(data, list(crop.placed))
    finally:
        survey.close()
    print(f"\n  --- one deck frame ---")
    print(f"  {len(seen)}/{len(crop.placed)} found · arm moved 0.000 m · "
          f"0.00 s of arm time")

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
    print("  wrote " + ", ".join(written))


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
                     ("vs_sweep", vs_sweep), ("shot", shot)):
        if getattr(args, flag):
            return fn(args) or 0
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
