#!/usr/bin/env python3
"""Decide the whole route before any joint moves, and check it clears the crop.

Week 2's first cut drove the tool straight at each waypoint and let IK pick the
route. That works right up until something is in the way, and in a plant row
something always is. Measured over ten picks with the old flat script:

    10 picks · 10 reported "in crate" · **8 of them knocked t3 off the plant**

The scorer never looked at the fruit it was not picking, so a cycle that
stripped a neighbouring truss every second attempt read as 100%. The mechanism
was not the grasp and not the pull — it was the trip *home*:

    after carry   t3 at [0.617 -0.227 0.599] attached=True
    after home    t3 at [0.613 -0.277 0.033] attached=False

The arm's home posture puts tool0 at x=0.671, and the fruit hang at x=0.60.
Home is *inside the canopy*. `week2_pick.STAGE_X` exists precisely to stop the
gripper crossing the row off-axis on the way in, and the return leg never got
the same treatment: driving from over the crate back to home cut the corner
through the plants, loaded t3's peduncle to 20.9 N against a 12 N threshold,
and dropped it on the floor.

Patching that one leg would fix that one bug. This module fixes the class of
bug, because Week 3 adds a camera and Week 4 removes the hardcoded positions,
and by then "the arm happened to miss everything" is not a claim anyone can
make. So:

**A mission is planned, verified, and only then executed.**

    planner = Planner(model, data, row)
    mission = planner.plan("t0")     # nothing has moved yet
    if not mission.ok:
        ...                          # refuse the pick, do not improvise
    execute(mission)

`plan()` builds a candidate route, replays it kinematically, and measures the
distance from every collision geom on the arm to every fruit it is not picking,
at every point along the way. If the route does not clear the crop by
`CLEARANCE`, it tries a different one — a deeper staging plane, a lane that
ducks under the canopy, a lane that goes over it — and returns the first that
clears, or reports failure with the offending pair named.

⚠️ **The preview is kinematic, not dynamic.** It replays IK without physics, so
it does not model servo lag or the ~5 mm droop of a loaded arm, and the real
path deviates from the predicted one by roughly that much. That is why
CLEARANCE is 40 mm and not 5, and why `Guard` exists: the plan is the primary
defence and the guard is the net under it. A plan that verifies is not a proof.

Run it directly to plan every fruit in the row and print the routes:

    ./.venv/bin/python simulation/mujoco/mission.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fr5 import GRIPPER_PREFIX, LINKS, TOOL_SITE  # noqa: E402
from plant_row import FRUIT_R, ROW_X  # noqa: E402
from reach import DEFAULT_SPEED, approach_rotation  # noqa: E402

# --- where things are --------------------------------------------------------
BIN_POS = np.array([0.30, -0.52, 0.0])
BIN_HALF = 0.16
BIN_WALL = 0.12
BIN_DROP_UP = 0.28

# --- the route, as knobs the planner is allowed to turn ----------------------
# How far in front of the row the arm stages before it moves sideways. The whole
# route is built around this plane: the arm only ever crosses it along +x, on
# the one axis it has to enter the canopy along.
STAGE_X = ROW_X - 0.28

APPROACH_GAP = 0.10     # hold this far back, fingers open, before inserting
PULL_DOWN = 0.14        # how far down to pull to load the peduncle
PULL_SPEED = 0.05       # the one move whose *force* matters; see reach.set_speed

# Speed cap for the legs that move with a tomato in the gripper.
#
# ⚠️ The arm is not the throughput limit — the grip is. At the full rated
# 180 deg/s the approach and the grasp are fine (insert arrives 2-5 mm out, the
# stem releases at 12-16 N), and then `extract` accelerates back out of the row
# and throws the fruit clean out of the pads: measured, 185 mm and 376 mm from
# the tool by the end of a 290 mm move. A 2F85 holds a sphere by friction
# between two flat pads, and after gravity there is almost no lateral
# acceleration budget left. So free-space travel gets whatever speed it is
# given and the fruit-in-hand legs get capped.
#
# Swept at --speed 1.0 over a five-fruit sequence harvest:
#
#   cap    clean   cycle
#   1.00    1/5    14.7 s     fruit thrown out of the gripper
#   0.50    2/5    15.8 s
#   0.35    2/5    16.7 s
#   0.25    5/5    18.3 s     <- the fastest cap that still holds every fruit
#
# The cliff is sharp and it sits between 0.35 and 0.25, which is worth knowing:
# there is no graceful degradation here, the fruit is either held or it is
# launched. This is a real finding rather than a tuning detail — it says payload
# retention is what caps kg/hr, not joint speed, and it is the strongest
# argument yet for the cradle design, which supports a truss from below instead
# of pinching it.
CARRY_SPEED = 0.25

# Where the arm waits between picks.
#
# ⚠️ Not the arm's home posture, and that is the entire point of this file. HOME
# puts tool0 at [0.671, -0.102, 0.478], which is 71 mm *past* the row plane —
# a parking spot inside the crop. Park out on the staging plane instead, where
# there is nothing to hit, and the return leg stops being a hazard.
PARK = np.array([STAGE_X, -0.10, 0.50])

# How close the arm may come to a fruit it is not picking.
#
# Measured, not chosen: the preview is kinematic and the executed path lags it,
# so the arm always ends up slightly closer to things than the plan predicted.
# Over ten clean picks the plan's worst predicted gap was 40 mm and the worst
# gap actually flown was 32 mm — an 8 mm overshoot, almost all of it servo droop
# under the 1.05 kg gripper. 40 mm covers that with the fruit's own 20 mm of
# jitter on top, and still leaves the guard's 15 mm floor untouched.
CLEARANCE = 0.040

# The same budget for the things that do not fall over: the support bar and the
# back panel. Much smaller, and the difference is the point — a tomato brushed
# is a tomato on the floor, a panel brushed is a scuff. Holding structure to
# the crop's 40 mm would refuse every grasp, because the fruit hang 80 mm off
# the panel and the gripper has to get between them.
STRUCTURE_CLEARANCE = 0.015

# The hard floor the runtime Guard trips on. Below this the arm is close enough
# that the next control cycle could make contact, so the mission is abandoned
# rather than finished — an aborted pick costs one tomato's time, a knocked
# truss costs the tomato.
GUARD_STOP = 0.015

# How much room one arm must leave the *other* arm on a two-armed machine. Only
# consulted when a `ClearanceModel` is built with `others=` — see `ArmObstacles`.
#
# ⚠️ **Deliberately the crop's 40 mm and not the structure's 15 mm**, which is
# the opposite of what "it is a rigid steel object, like the panel" suggests.
# Structure gets 15 mm because a panel is stationary, known to the millimetre,
# and a scuff costs nothing. The other arm is none of those: it is 22 kg, it is
# where the *other* arm's servo lag says it is rather than where its setpoint
# does, and a collision damages two machines and drops whatever both were
# holding. It gets the budget reserved for things that must not be touched.
#
# ⚠️ This is a **new** constant, not a retuning of an old one. Nothing in Weeks
# 1-4 reads it, because nothing in Weeks 1-4 had a second arm to clear.
ARM_CLEARANCE = 0.040

# Orientation weights. See week2_pick for the measurements behind them: 0.1 is
# deliberately too weak to lead a move (position leads, the angle arrives on the
# way), and 0.6 is for turning the wrist in place where there is no position to
# trade against.
ORIENTATION_COST = 0.1
ROLL_COST = 0.01

# What the roll costs once the planner has actually chosen one.
#
# ⚠️ It has to be raised, and it must not be raised everywhere. At 0.01 the roll
# is free by design — `approach_rotation` has to return *some* full frame and
# the roll it picks is arbitrary, so charging the solver for missing an
# arbitrary target wastes a DOF (measured in Week 1: pinned, the arm converged
# 82-90° off approach at two of six positions and the sweep dropped to 2/6).
# But a roll chosen to fit the gripper between two fruit is not arbitrary, and
# at 0.01 the solver simply ignores it. So it is pinned on the legs that enter
# the row and left free everywhere else.
ROLL_COST_PINNED = 0.08

# Rolls to try, in radians. Capped at ±50°: the pads close along this axis, and
# past about 60° they are closing in a near-vertical plane, which swipes the
# peduncle on the way in instead of coming around the fruit's sides.
ROLL_OPTIONS = (np.radians(30), np.radians(-30), np.radians(50), np.radians(-50))
TURN_COST = 0.6
TURN_DONE_DEG = 30.0    # a 2F85 rolls a sphere out of its pads past ~22°
TIP_MAX_S = 3.5

INTO_ROW = np.array([1.0, 0.0, 0.0])
DOWNWARD = np.array([0.0, 0.0, -1.0])


# --- whose world a mission is planned in -------------------------------------

@dataclass(frozen=True)
class ArmFrame:
    """The five world constants a mission is planned against, as a value.

    ⚠️ **This is what used to be five module globals, and moving it here is what
    makes two arms flyable at once.** `PARK`, `STAGE_X`, `BIN_POS`, `ROW_X` and
    `INTO_ROW` describe *one* arm's world — where it rests, the plane it stages
    on, where its crate is, where its crop is, and which way "into the row"
    points. That was fine while there was one arm bolted to the origin.

    On a trolley they are functions of where the machine is standing, and
    `farm.armframe` made them work by **rebinding this module's globals** for
    the duration of a mission. That is why the two arms were serialised: two
    arms mid-mission need two conflicting sets of those globals in one
    interpreter, so concurrency was not merely unimplemented, it was
    inexpressible. See Bug Log 57.

    Carried as a value on the `Mission`, two arms simply hold two frames, and
    nothing global changes at all.

    ⚠️ `WORLD` below is built from the module constants at import, so every
    caller that passes no frame gets exactly the Week 1-4 numbers. Nothing
    measured in Weeks 1-4 can move, because for those callers this is the same
    five values reached by a different name.
    """

    park: np.ndarray
    stage_x: float
    bin_pos: np.ndarray
    row_x: float
    into_row: np.ndarray

    @property
    def back(self):
        """Sign of "out of the canopy" along x. See `Route.pull_vector`."""
        return -float(self.into_row[0])


WORLD = ArmFrame(park=PARK, stage_x=STAGE_X, bin_pos=BIN_POS, row_x=ROW_X,
                 into_row=INTO_ROW)


def _frame(frame):
    """`WORLD` when a caller passes nothing. The Weeks 1-4 default, in one place."""
    return WORLD if frame is None else frame

# The legs that move with fruit in the gripper, between the pull and the
# release. These are the ones CARRY_SPEED applies to.
CARRYING_LEGS = {"extract", "turn", "carry"}

# --- preview resolution ------------------------------------------------------
# How finely the kinematic replay samples each leg, and how many IK iterations
# it spends per sample. Warm-started from the previous sample, three is enough
# to track a 40 mm step; the whole plan costs ~0.3 s, against a 24 s pick.
PREVIEW_STEP = 0.04
PREVIEW_ITERS = 3
PREVIEW_DT = 0.05
DISTMAX = 0.30          # mj_geomDistance early-outs past this; nothing further matters


@dataclass
class Leg:
    """One instruction in a mission. The executor is an interpreter over these.

    `kind` is what the executor does with it:

        move     drive the tool to `goal`
        pull     drive toward `goal` slowly, stop the moment the stem parts
        turn     hold position, rotate the tool to `approach`
        grip     hold position, ramp the fingers shut
        release  hold position, open the fingers
        settle   hold position for `seconds`
        home     drive the *joints* to `posture`, ignoring the tool frame
    """

    name: str
    kind: str
    goal: np.ndarray | None = None
    approach: np.ndarray | None = None
    orientation_cost: float = ORIENTATION_COST
    speed: float = DEFAULT_SPEED
    seconds: float = 0.0
    tilt_stop: float = 0.0
    # Rotation about the finger axis, in radians, and how hard to insist on it.
    # Free by default — see Reacher.roll_cost — and pinned only when the planner
    # has chosen a roll on purpose to fit the gripper between two fruit.
    roll: float = 0.0
    roll_cost: float = ROLL_COST
    # Joint targets, for `home` legs only. A Cartesian goal cannot express
    # "and stand like this", which is the whole reason this field exists.
    posture: np.ndarray | None = None
    # Whether this leg is *allowed* to touch the target fruit. True only for the
    # legs that are supposed to: everything else touching anything is a fault.
    contact_ok: bool = False


@dataclass
class Breach:
    """Where a candidate route came too close, and to what."""

    leg: str
    distance: float
    robot_geom: str
    obstacle: str
    at: np.ndarray

    def __str__(self):
        return (f"{self.leg}: {self.robot_geom} within "
                f"{self.distance * 1000:.0f} mm of {self.obstacle}")


@dataclass
class Mission:
    """A route that has been checked, with the evidence attached."""

    target: str
    legs: list[Leg]
    ok: bool
    clearance: float                       # the worst gap along the whole route
    lane: str = "direct"
    stage_x: float = STAGE_X
    breaches: list[Breach] = field(default_factory=list)
    tried: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)   # lessons that shaped it
    path: np.ndarray | None = None         # the predicted tool path, for drawing
    # ⚠️ Which arm's world this was planned in, carried with the plan rather
    # than left in module state for the executor to look up. `execute` scores
    # the crate from `mission.frame.bin_pos` and escapes an abort along
    # `mission.frame.into_row`, so a plan handed to the executor is complete —
    # there is nothing left to rebind and nothing for a second arm to clobber.
    # `None` means the Week 1-4 world; see `ArmFrame`.
    frame: ArmFrame | None = None

    @property
    def arm_frame(self):
        return _frame(self.frame)

    def summary(self) -> str:
        head = (f"mission {self.target}: {'planned' if self.ok else 'REFUSED'} · "
                f"lane {self.lane} · stage x={self.stage_x:.2f} · "
                f"clearance {self.clearance * 1000:.0f} mm")
        if self.applied:
            head += f" · lessons {', '.join(self.applied)}"
        if not self.ok and self.breaches:
            head += f"\n    closest: {self.breaches[0]}"
        if len(self.tried) > 1:
            head += f"\n    tried {len(self.tried)} routes: {', '.join(self.tried)}"
        return head


def robot_geoms(model, prefix="") -> list[int]:
    """Every collision geom that belongs to the arm or the gripper.

    Visual geoms carry contype=0 and cannot hit anything, so they are excluded —
    checking them would report the arm colliding with its own decoration.

    ⚠️ `prefix` picks *which* arm, and getting it wrong is silent. The filter is
    a name match against `LINKS` and `GRIPPER_PREFIX`, both of which are the
    bare Week 1-4 names — so on a two-armed trolley, a clearance model built for
    arm b without this returns **arm a's** spheres. The planner then checks a
    body that is not the one it is about to fly, reports a clearance in
    millimetres, and is describing the wrong arm.

    ⚠️ This function reports **one arm's own geometry** and has never done
    anything else — it is the "what am I" half, not the "what must I avoid"
    half. The other arm being an obstacle is `ArmObstacles`' job; see the Bug
    Log entry 43 for the period when nothing did it at all.
    """
    ids = []
    links = {prefix + b for b in LINKS}
    grip = prefix + GRIPPER_PREFIX
    for i in range(model.ngeom):
        g = model.geom(i)
        if g.contype[0] == 0 and g.conaffinity[0] == 0:
            continue
        body = model.body(g.bodyid[0]).name
        if body in links or body.startswith(grip):
            ids.append(i)
    return ids


# --- why the robot is spheres and the crop is not ----------------------------
#
# ⚠️ The obvious way to do this is `mj_geomDistance`, and on this MuJoCo it is
# wrong. The arm's collision geoms are meshes straight out of Fairino's URDF,
# and mesh distance queries need native CCD; MuJoCo 3.10 has no
# `mjENBL_NATIVECCD` and falls back to MPR, which computes *penetration* depth
# and returns nonsense for separated bodies. Measured on this scene, with the
# arm at its home posture:
#
#   wrist1_link vs the back panel   mj_geomDistance -> 0.0 mm
#   ...the two are 276 mm apart, and the witness points it hands back
#   ([0.691 -0.150 0.844] and [0.997 -0.279 1.018]) are outside both geoms.
#
# A planner built on that would refuse every route for a collision that is not
# there, which is worse than not checking at all — it looks like it is working.
#
# So the arm is approximated instead, by a set of spheres that *enclose* its
# collision geoms, and the crop is kept exact (a tomato is already a sphere, the
# support is a capsule, the panel is a box). The decomposition is conservative
# by construction: every sphere contains all of its geom's convex hull vertices,
# so it contains the hull, so a reported clearance is never larger than the
# true one. The planner can therefore refuse a route that would in fact have
# been fine — it can never pass one that would not.
#
# It is also about 40x faster than the mj_geomDistance version, which is what
# lets `Guard` run this every control cycle instead of only at plan time.

def _hull_points(model, gid):
    """Points whose convex hull contains the geom, in its *body's* frame.

    Plus a radius to add to any sphere covering them, for the primitives whose
    surface is offset from those points (a capsule is a segment plus a radius).
    """
    import mujoco

    g = model.geom(gid)
    t = g.type[0]
    pad = 0.0

    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mi = int(g.dataid[0])
        adr, n = model.mesh_vertadr[mi], model.mesh_vertnum[mi]
        pts = np.array(model.mesh_vert[adr:adr + n]).reshape(-1, 3)
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        hx, hy, hz = g.size
        pts = np.array([[sx * hx, sy * hy, sz * hz]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        pts, pad = np.zeros((1, 3)), float(g.size[0])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        h = float(g.size[1])
        pts, pad = np.array([[0, 0, -h], [0, 0, h]]), float(g.size[0])
    else:
        # Anything else: fall back to the bounding sphere MuJoCo already keeps.
        pts, pad = np.zeros((1, 3)), float(model.geom_rbound[gid])

    R = np.array(model.geom_quat[gid])
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, R)
    return pts @ mat.reshape(3, 3).T + model.geom_pos[gid], pad


def _cover(pts, pad, k):
    """`k` spheres along the point set's long axis that together contain it.

    Split by position along the first principal axis, then take each slab's
    bounding sphere. Slabs rather than k-means because the shapes here are
    links — long and roughly convex — and slabs give a tight cover with no
    iteration and no random seed.

    ⚠️ The slabs are cut at **equal distances**, not equal vertex counts. Cutting
    on quantiles is the obvious thing and it silently does nothing here: these
    are CAD collision meshes, and their vertices bunch up around bolt holes and
    joint flanges while the shaft between them is nearly bare. The forearm has
    4503 vertices over half a metre, most of them at the two ends — so four
    quantile slabs put three cuts inside one flange and left a single sphere
    213 mm across covering the entire link. Equal distances give 77 mm, which is
    the difference between a planner that refuses good grasps and one that does
    not.
    """
    if len(pts) == 1 or k <= 1:
        c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        return [(c, float(np.linalg.norm(pts - c, axis=1).max()) + pad)]

    centred = pts - pts.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    s = centred @ axis
    edges = np.linspace(s.min(), s.max(), k + 1)
    out = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        sel = (s >= lo - 1e-12) & (s <= hi + 1e-12)
        if not sel.any():
            continue
        p = pts[sel]
        c = 0.5 * (p.min(axis=0) + p.max(axis=0))
        out.append((c, float(np.linalg.norm(p - c, axis=1).max()) + pad))
    return out


# Upper bound on spheres per collision geom; the actual count is chosen per
# geom from its shape by `_sphere_count`.
#
# ⚠️ A fixed count is what makes this approximation useless at the one moment it
# matters. At four-per-mesh the Robotiq's coupler links — thin bars about 60 mm
# long and 10 mm thick — were covered by balls 21 mm across, and the planner
# refused a perfectly good grasp of t0 because that inflated linkage came
# "within 26 mm" of t1. The conservatism is a feature everywhere except inside
# the canopy, which is exactly where the arm has to work. Spending spheres in
# proportion to how elongated a part is puts the resolution where the error is.
MAX_SPHERES_PER_GEOM = 10


def _sphere_count(pts, max_k=MAX_SPHERES_PER_GEOM):
    """How many spheres this shape needs to be covered tightly.

    A sphere cover is only as tight as the shape is round. A ball needs one; a
    60 mm rod 10 mm thick needs six before the cover stops being a sausage of
    fat balls. Sizing by the ratio of the long axis to the thick one spends
    spheres exactly where the error is, and costs nothing on the compact parts.
    """
    if len(pts) < 2:
        return 1
    centred = pts - pts.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    extents = [float(np.ptp(centred @ a)) for a in axes]
    long_axis, thick = extents[0], max(extents[1], extents[2], 1e-6)
    return int(np.clip(np.ceil(long_axis / (0.75 * thick)), 1, max_k))


_SPHERE_CACHE = {}


def arm_spheres(model, prefix=""):
    """`RobotSpheres` for one arm, built once per (model, prefix).

    ⚠️ **The cover depends only on the model, and it was being rebuilt for every
    candidate route.** `_verify` constructs a fresh `ClearanceModel` per
    candidate, which builds a `RobotSpheres` for this arm *and* one for every
    other arm on the machine — and each of those walks all 5433 geoms, unpacks
    the mesh vertices of the ones that belong to the arm, runs an SVD per geom
    to size the cover and another to split it. Identical work, up to 18 times a
    plan, twice over on a two-armed machine.

    Cached on the model's id and the prefix. Nothing in the cover reads
    `mjData` — `world()` takes the data and poses the cached local centres — so
    this cannot go stale against a moving arm. It would go stale against a
    *recompiled model*, which is why the key includes `id(model)` and why this
    repo does not recompile mid-run.
    """
    key = (id(model), prefix)
    got = _SPHERE_CACHE.get(key)
    if got is None:
        got = _SPHERE_CACHE[key] = RobotSpheres(model, prefix=prefix)
    return got


class RobotSpheres:
    """A conservative sphere cover of the arm and gripper, posed from mjData."""

    def __init__(self, model, geom_ids=None, max_k=MAX_SPHERES_PER_GEOM,
                 prefix=""):
        self.model = model
        centres, radii, bodies, labels = [], [], [], []
        for gid in (robot_geoms(model, prefix) if geom_ids is None
                    else geom_ids):
            pts, pad = _hull_points(model, gid)
            k = _sphere_count(pts, max_k)
            body = int(model.geom(gid).bodyid[0])
            for c, r in _cover(pts, pad, k):
                centres.append(c)
                radii.append(r)
                bodies.append(body)
                labels.append(model.body(body).name)
        self.local = np.array(centres)
        self.radius = np.array(radii)
        self.body = np.array(bodies)
        self.label = labels

    def world(self, data):
        """Sphere centres in world coordinates, for the pose in `data`."""
        mats = data.xmat[self.body].reshape(-1, 3, 3)
        return data.xpos[self.body] + np.einsum("nij,nj->ni", mats, self.local)


class CropObstacles:
    """What the arm must not touch, kept as exact primitives.

    The support bar and the back panel sit in here alongside the fruit. They do
    not fall over, but the arm pinning a tomato *against* one is how a truss
    gets stripped without the gripper ever closing — and that reads as a mystery
    if the check only ever watched the fruit.
    """

    def __init__(self, model, row, target=None):
        self.model = model
        # ⚠️ Only fruit still on the plant. A detached one is in the crate or on
        # the floor, and protecting it is actively wrong: the crate is where the
        # arm is *going*, so counting harvested fruit as obstacles makes the
        # planner refuse every pick after the first one in a sequence.
        self.fruit = [(n, model.body(n).id, FRUIT_R)
                      for n in row.names if n != target and row.attached(n)]

        self.capsules, self.boxes = [], []
        for name in ("row_support_bar",):
            try:
                gid = model.geom(name).id
            except KeyError:
                continue
            self.capsules.append((name, gid, float(model.geom(gid).size[0]),
                                  float(model.geom(gid).size[1])))
        for name in ("row_panel_face",):
            try:
                gid = model.geom(name).id
            except KeyError:
                continue
            self.boxes.append((name, gid, np.array(model.geom(gid).size)))

    def distances(self, data, centres, radii):
        """Per-obstacle arrays of (distance to each robot sphere).

        Yields (name, distances) so the caller can apply a different required
        clearance per obstacle — which is exactly what a learned lesson does.
        """
        for name, bid, R in self.fruit:
            yield name, np.linalg.norm(centres - data.xpos[bid], axis=1) - radii - R

        for name, gid, R, half in self.capsules:
            c = data.geom_xpos[gid]
            axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            v = centres - c
            t = np.clip(v @ axis, -half, half)
            yield name, np.linalg.norm(v - t[:, None] * axis, axis=1) - radii - R

        for name, gid, half in self.boxes:
            mat = data.geom_xmat[gid].reshape(3, 3)
            local = (centres - data.geom_xpos[gid]) @ mat
            q = np.abs(local) - half
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            yield name, outside + inside - radii


class ArmObstacles:
    """The *other* arm(s) on the same machine, as spheres the planner must clear.

    ⚠️ **This closes the arm-vs-arm gap, and the gap was real rather than
    theoretical.** Until this existed, `ClearanceModel` held one arm's own
    spheres and the crop, and nothing else — so on a two-armed trolley each arm
    planned as though the other were not there. The two mounts are 400 mm apart
    and each arm reaches 922 mm, so their working volumes overlap by 1.44 m.
    Two arms working one deck could pass straight through each other and every
    clearance number printed would still have read as a pass, because the thing
    they were passing through was in nobody's obstacle set.

    The same conservative sphere cover as `RobotSpheres` is used for both sides,
    so the reported distance is never larger than the true one — this can refuse
    an approach that would have been fine and can never pass one that would not.
    That is the same guarantee the crop check already carries, and it is the
    reason a sphere cover is the right shape here rather than a mesh query.

    ⚠️ It reports the distance to the other arm **where the other arm is now**.
    On a machine that serialises its arms (see `farm.duo`) the idle arm is
    parked and stationary, so "now" is a fact for the whole mission. On a machine
    that ever flew both at once it would be a one-cycle-stale prediction, which
    is exactly why `farm.duo` does not fly both at once.
    """

    def __init__(self, model, prefixes):
        self.arms = [(p.rstrip("_") or "a", arm_spheres(model, p))
                     for p in prefixes]

    def distances(self, data, centres, radii):
        """Same contract as `CropObstacles.distances`: (name, per-sphere array).

        For each of *this* arm's spheres, the distance to the nearest sphere of
        the other arm — so the array lines up with `centres` and the caller's
        `argmin` still names the right part of the robot.
        """
        for name, sph in self.arms:
            other_c = sph.world(data)
            gap = (np.linalg.norm(centres[:, None, :] - other_c[None, :, :],
                                  axis=2)
                   - radii[:, None] - sph.radius[None, :])
            yield f"arm {name}", gap.min(axis=1)


class ClearanceModel:
    """How close the robot is to the crop, in metres, right now."""

    def __init__(self, model, row, target=None, prefix="", others=()):
        # `prefix` picks which arm's geometry is being checked. See
        # `robot_geoms` — the wrong one here is silent and plausible.
        #
        # `others` names the *other* arms on the machine, by prefix — `("b_",)`
        # when this model checks arm a of a two-armed trolley. Empty by default,
        # which is Weeks 1-4's behaviour exactly and leaves every number they
        # measured untouched. See `ArmObstacles`.
        self.prefix = prefix
        self.spheres = arm_spheres(model, prefix)
        self.crop = CropObstacles(model, row, target)
        self.others = ArmObstacles(model, others) if others else None

    def per_obstacle(self, data):
        """{obstacle name: (closest distance, which part of the robot)}."""
        centres = self.spheres.world(data)
        out = {}
        sources = [self.crop.distances(data, centres, self.spheres.radius)]
        if self.others is not None:
            sources.append(self.others.distances(data, centres,
                                                 self.spheres.radius))
        for source in sources:
            for name, d in source:
                i = int(np.argmin(d))
                out[name] = (float(d[i]), self.spheres.label[i])
        return out

    def worst(self, data):
        """(distance, robot part, obstacle) for the closest pair."""
        best, part, obstacle = np.inf, "", ""
        for name, (d, lbl) in self.per_obstacle(data).items():
            if d < best:
                best, part, obstacle = d, lbl, name
        return best, part, obstacle


class Guard:
    """The net under the plan: watch the real clearance while the arm moves.

    The preview cannot see servo lag, and Week 3's camera will hand the planner
    positions that are simply wrong by a centimetre or two. Neither is a reason
    to let the arm push a tomato off a plant, so clearance is also measured for
    real, every control cycle, against the fruit the arm is not picking.

    Tripping is not a failure to hide — it is the mission being abandoned on
    purpose, which is the correct thing to do and is worth strictly more than a
    pick. `incident.py` records why, and `lessons.py` makes sure the same route
    is not planned twice.
    """

    def __init__(self, model, data, row, target, stop=GUARD_STOP, prefix="",
                 others=()):
        self.data = data
        # ⚠️ The guard watches the arm named by `prefix` against the crop, and —
        # when `others` is given — against the other arms on the same machine.
        # Left empty it is Weeks 1-4's behaviour: one arm, crop only. See
        # `ArmObstacles` for why the two-armed case needs it and what it costs.
        self.prefix = prefix
        self.clearance = ClearanceModel(model, row, target, prefix=prefix,
                                        others=others)
        self.stop = stop
        self.min_seen = np.inf
        self.tripped = None      # (leg, distance, robot geom, crop geom)
        self.leg = ""
        self.armed = True

    def check(self):
        """Returns True if the arm is still clear. Call once per control cycle."""
        if not self.armed:
            return True
        d, ra, ob = self.clearance.worst(self.data)
        if d < self.min_seen:
            self.min_seen = d
        if d < self.stop and self.tripped is None:
            self.tripped = (self.leg, d, ra, ob)
            return False
        return True


# --- planning ----------------------------------------------------------------

@dataclass
class Route:
    """The knobs that define one candidate route. What lessons get to change."""

    lane: str = "direct"        # direct | under | over
    pull: str = "down"          # down | out
    roll: float = 0.0           # wrist rotation about the finger axis, radians
    stage_x: float = STAGE_X
    approach_gap: float = APPROACH_GAP
    keepout: dict = field(default_factory=dict)   # fruit name -> extra metres
    # ⚠️ The frame this route's `stage_x` is measured in. It exists so `label`
    # can report a backoff — "+8cm" means eight centimetres behind *this arm's*
    # staging plane — without reading a module global that a second arm may
    # have a different answer for. `farm.armframe._patch_default` used to patch
    # this dataclass's `__init__.__defaults__` for exactly this reason; it no
    # longer has to, because the nominal comes in with the route.
    frame: ArmFrame | None = None

    def label(self):
        back = _frame(self.frame).stage_x - self.stage_x
        bits = [self.lane]
        if self.pull != "down":
            bits.append(f"pull-{self.pull}")
        if abs(self.roll) > 1e-6:
            bits.append(f"roll{np.degrees(self.roll):+.0f}")
        if back > 0.001:
            bits.append(f"+{back * 100:.0f}cm")
        return "/".join(bits)

    def pull_vector(self):
        """Which way the arm loads the peduncle.

        Straight down is the natural read of "pull it off the plant" and it is
        what the row is laid out for. It is also what puts the gripper on top of
        whatever hangs below the target: measured on this row, pulling t0 (at
        y=0.21, z=0.70) down by 140 mm walks the open pad into t4 (y=0.24,
        z=0.55), 30 mm across and 153 mm below. The planner catches it —
        `pull: gr_left_pad within -2 mm of t4` — and these are the ways out.

        The weld does not care about the direction, only the magnitude: SNAP_N
        is a threshold on the norm of the constraint force. So the pull can lean
        backwards out of the canopy instead of descending further into it, and
        `far` leans almost all the way — which is also what a person does,
        pulling the fruit toward themselves rather than toward the floor.
        """
        # ⚠️ The x components are "back out of the canopy", not "toward -x".
        # They are signed by the frame's `into_row` so a mirrored arm leans out
        # of its own row rather than further into it — see
        # `farm.armframe.MIRROR`. For the unmirrored arm `into_row[0]` is +1 and
        # these are the Week 2 numbers unchanged.
        back = _frame(self.frame).back
        return {
            "out": np.array([back * 0.09, 0.0, -PULL_DOWN]),
            "far": np.array([back * 0.17, 0.0, -0.06]),
        }.get(self.pull, np.array([0.0, 0.0, -PULL_DOWN]))


class Planner:
    """Turns a target fruit into a checked route, without moving the arm."""

    def __init__(self, model, data, row, lessons=None, clearance=CLEARANCE,
                 park_q=None, speed=DEFAULT_SPEED, prefix="", others=(),
                 pin=(), frame=None):
        import mink

        self.model = model
        self.data = data
        self.row = row
        self.lessons = lessons
        self.clearance = clearance
        self.park_q = park_q
        # ⚠️ Which arm's world to plan in. `None` is the Week 1-4 world and is
        # what every single-armed caller gets, unchanged. `farm.armframe.frame`
        # builds the trolley-mounted one. See `ArmFrame` — this parameter is
        # what replaced rebinding this module's globals, and it is why two arms
        # can now be mid-mission at the same time.
        self.frame = _frame(frame)
        # ⚠️ The plan carries the speed, not the arm. Every Leg used to default
        # to DEFAULT_SPEED, and the executor sets the arm to `leg.speed` at the
        # top of each leg — so a Reacher built at --speed 1.0 was clobbered back
        # to 0.15 by the first move and the flag did nothing at all. It ran, it
        # printed the speed it was asked for, and the arm crawled.
        self.speed = speed

        # A scratch MjData for the preview. The live `data` is the simulation
        # and must not be touched: writing qpos into it to test a pose would
        # teleport the real arm mid-cycle.
        self._scratch = None
        # Which arm this planner routes. See `reach.Reacher.prefix`.
        self.prefix = prefix
        # The other arms on the same machine, by prefix, which this one must
        # plan around. Empty is Weeks 1-4. See `ArmObstacles`.
        self.others = tuple(others)
        self.tool_site = prefix + TOOL_SITE
        # This arm's six, by name. Needed to seed the preview's posture task on
        # `park_q` the way the executor does — see `_verify`.
        from fr5 import JOINTS as _J

        self.joints = [prefix + j for j in _J]

        # ⚠️ **The preview has to be pinned exactly like the executor, or the
        # planner verifies routes the arm cannot fly.** `mink.Configuration`
        # spans the whole model, so the preview QP below is free to reach by
        # driving the trolley and folding the *other* arm — neither of which the
        # executor will do, because `reach.Reacher.step` writes six actuators and
        # `farm.armframe.pin_base` pins the rest.
        #
        # This is `pin_base`'s bug on the planning side, and it is worse there:
        # `pin_base` produces legs that arrive short, which at least shows up in
        # the leg report. An unpinned *preview* produces a route that was checked
        # in a posture the arm will never adopt, so the clearance it reports is a
        # measurement of a different machine. With one arm it was only the slide
        # joint and the drift was small enough to pass for noise; with two it can
        # fold a whole second arm out of the way and call the route clear.
        #
        # Named explicitly by the caller rather than guessed from the model, the
        # same contract `park_posture` already uses — `farm.duo` passes the drive
        # joint and every other arm's six.
        self.pin = tuple(pin)

        # ⚠️ **The preview gets the executor's joint speed cap, not an
        # unlimited solver, and this is the same argument as `pin` one
        # paragraph up carried to its conclusion.** Pinning stopped the preview
        # using joints the executor will never move. It did nothing about the
        # preview moving the arm's *own* joints faster than the arm can, and
        # `reach.Reacher` caps them at `joint_velocity * speed` because a
        # velocity limit is a constraint inside the QP rather than a scale
        # factor afterwards — so capped and uncapped solvers take **different
        # paths**, not the same path at different speeds.
        #
        # With one arm that difference was invisible, because it lives in the
        # null space: the tool goes to the same place either way, and every
        # clearance that mattered was against the crop, which the tool's own
        # position dominates.
        #
        # With two arms it is not invisible, because the thing the other arm
        # collides with is the **elbow**, which is exactly what the null space
        # decides. Measured on one route, arm a reaching 0.40 m *forward* —
        # away from arm b — while arm b sat parked:
        #
        #     preview, unlimited solver     lane: upperarm_link within -142 mm
        #     same pose solved to converge                        +250 mm
        #
        # The planner was refusing good routes for a posture the arm does not
        # adopt. A preview checked in a posture the arm will never adopt is a
        # measurement of a different machine — the note above says so about
        # pinning, and it is just as true about speed.
        from fr5 import JOINT_VELOCITY as _JV

        caps = {j: 0.0 for j in self.pin}
        for j in self.joints:
            caps[j] = _JV[j[len(prefix):]] * self.speed
        import mink as _mink

        self._limits = [_mink.VelocityLimit(model, caps)]
        self._cfg = mink.Configuration(model)
        self._task = mink.FrameTask(
            frame_name=self.tool_site, frame_type="site", position_cost=1.0,
            orientation_cost=[ORIENTATION_COST, ORIENTATION_COST, ROLL_COST],
            lm_damping=1.0)
        self._posture = mink.PostureTask(model, cost=1e-2)

    # -- the route ------------------------------------------------------------

    def _legs(self, name, target, route: Route) -> list[Leg]:
        """The pick, as a waypoint list.

        Every leg exists for a measured reason; the ones that are not obvious
        are the ones that were added after something broke.
        """
        fr = self.frame
        INTO_ROW, PARK = fr.into_row, fr.park
        sx = route.stage_x
        start = self.data.site(self.tool_site).xpos.copy()
        lane_z = self._lane_z(route, target)
        pull_to = target + route.pull_vector()
        out_z = float(pull_to[2])
        over_bin = fr.bin_pos + np.array([0, 0, BIN_DROP_UP])

        legs = [
            # 0. Settle. The weld overshoots for ~50 ms on startup and the peak
            #    is above SNAP_N; snap during it and the crop falls off the
            #    plant before the arm has moved. Hold the *current* pose — a
            #    lurch toward the fruit here loaded a peduncle to 52 N on its
            #    own.
            Leg("settle", "settle", seconds=0.3, orientation_cost=0.0),

            # 1. Straight back to the staging plane, along -x only. Moving away
            #    from the row is the one direction that is always safe, so this
            #    leg needs no lane and no alternative.
            Leg("clear", "move", np.array([sx, start[1], start[2]]), INTO_ROW),
        ]

        # 2. Traverse to the fruit's line, out at the staging plane. With a
        #    non-direct lane this happens at a different height and a third leg
        #    lifts back onto the fruit's line — see _lane_z.
        legs.append(Leg("lane", "move", np.array([sx, target[1], lane_z]),
                        INTO_ROW))
        if abs(lane_z - target[2]) > 1e-3:
            legs.append(Leg("align", "move",
                            np.array([sx, target[1], target[2]]), INTO_ROW))

        legs += [
            # 3. Drive in along +x. The only leg that crosses the staging plane.
            # Held back along the approach axis, which is -INTO_ROW rather
            # than -x: a mirrored arm stands off on its own side of the fruit.
            Leg("approach", "move",
                target - INTO_ROW * route.approach_gap, INTO_ROW),
            # 4. tool0 sits between the fingertips, so "go to the fruit" puts
            #    the pads either side of it.
            Leg("insert", "move", target.copy(), INTO_ROW, contact_ok=True),
            # 5. Ramped, not commanded shut. A step input is an impact, and on
            #    a hanging fruit that impact spiked the peduncle to 164 N —
            #    twenty-seven times SNAP_N — so every stem snapped on contact
            #    and the "pull" was picking up an already-loose tomato.
            Leg("grip", "grip", target.copy(), INTO_ROW, seconds=1.5,
                contact_ok=True),
            Leg("close", "settle", target.copy(), INTO_ROW, seconds=0.8,
                contact_ok=True),
            # 6. The harvest. Slow, because this is the one move whose force
            #    matters: one control cycle at full speed is a ~48 N step at
            #    the tool, and a 12 N threshold cannot be resolved inside it.
            Leg("pull", "pull", pull_to, INTO_ROW,
                speed=PULL_SPEED, contact_ok=True),
            Leg("grasp", "settle", None, INTO_ROW, seconds=0.5, contact_ok=True),
            # 7. Straight back out along -x before doing anything else, so the
            #    fruit in the gripper does not sweep the plants.
            Leg("extract", "move", np.array([sx, target[1], out_z]), INTO_ROW,
                contact_ok=True),
            # 8. Turn the wrist *in place*, out here where there is nothing to
            #    hit. Asking for the turn and the carry at once makes the solver
            #    trade one against the other over a half-metre move and it takes
            #    a bad route — measured, 325 mm past the crate.
            Leg("turn", "turn", np.array([sx, target[1], out_z]), DOWNWARD,
                orientation_cost=TURN_COST, tilt_stop=TURN_DONE_DEG,
                seconds=TIP_MAX_S, contact_ok=True),
            Leg("carry", "move", over_bin, DOWNWARD, contact_ok=True),
            Leg("release", "release", over_bin, DOWNWARD, seconds=0.8),

            # 9. ⚠️ The leg this whole module exists for. The old cycle drove
            #    from over the crate straight back to the arm's home posture,
            #    and home is 71 mm *past* the row plane — so the return cut the
            #    corner through the canopy and stripped t3 on 8 of 10 picks.
            #    Withdraw to the staging plane first. Wrist stays down: turning
            #    it while translating is the same solver trade as step 8.
            Leg("withdraw", "move", np.array([sx, PARK[1], PARK[2]]), DOWNWARD),
            # 10. Only now turn back to the row-facing pose, ready for the next
            #     fruit. In place, out at the staging plane, hitting nothing.
            Leg("ready", "turn", np.array([sx, PARK[1], PARK[2]]), INTO_ROW,
                orientation_cost=TURN_COST, tilt_stop=12.0, seconds=TIP_MAX_S),
            # 11. ⚠️ Land on PARK properly. A `turn` leg stops the moment the
            #     *angle* is right, and the wrist swinging round drags the tool
            #     with it — measured, `ready` finished with the tool at
            #     [0.28 -0.15 0.71] against a PARK of [0.32 -0.10 0.50], 210 mm
            #     high. Resetting between trials hides it completely. In a
            #     sequence it does not get hidden: the next mission starts from
            #     that pose, its first two legs inherit the error, and the
            #     approach after them missed by 251 mm and grasped nothing.
            #     A mission has to end where it claims to end.
            Leg("park", "move", np.array([sx, PARK[1], PARK[2]]), INTO_ROW),
        ]

        # 12. Unwind the arm in joint space.
        #
        #     ⚠️ Returning the *tool* to PARK does not return the *arm* to park.
        #     Measured over a five-fruit sequence, the tool came back to within
        #     2 mm every time while the joints went from
        #     [0, -39, -144, 2, 90, -90]° to [62, -10, -142, -35, 28, -132]° —
        #     the shoulder swung 60° round and the wrist flipped, because the
        #     swing out to the crate at y = -0.52 is done by j1 and nothing ever
        #     asks for it back. The posture task is a null-space preference at
        #     cost 1e-2; it cannot unwind 60° against a frame task at 1.0.
        #
        #     From that posture, reaching into the row is a different problem
        #     and the local IK step goes the wrong way: the approach to t3
        #     finished 243 mm short, travelling in -y, and grasped nothing. It
        #     looks exactly like a planning failure and it is a posture failure.
        #
        #     Safe by construction, which is why it needs no lane: the whole
        #     move happens at the staging radius, roughly 0.34 m from the base,
        #     while the crop is at 0.60 m and the crate is below z = 0.12.
        if self.park_q is not None:
            legs.append(Leg("unwind", "home",
                            np.array([sx, PARK[1], PARK[2]]), INTO_ROW,
                            seconds=2.0, posture=np.asarray(self.park_q)))

        # The pull keeps its own speed whatever the cycle is set to. It is the
        # one move whose *force* matters rather than its duration: at full rated
        # speed a single control cycle moves the setpoint 0.0047 rad, which
        # against kp=4000 is ~48 N at the tool, and a 12 N detach threshold
        # cannot be resolved inside a 48 N step — the stem tears at whatever
        # force the first sample happens to read. A real harvester decelerates
        # into the detach for the same reason.
        for leg in legs:
            if leg.kind == "pull":
                continue
            leg.speed = (min(self.speed, CARRY_SPEED)
                         if leg.name in CARRYING_LEGS else self.speed)

        # A chosen roll applies to every leg that faces the row, so the wrist
        # arrives already rotated rather than turning inside the canopy — the
        # same argument as turning for the crate out at the staging plane.
        if abs(route.roll) > 1e-6:
            for leg in legs:
                if leg.approach is not None and np.allclose(leg.approach, INTO_ROW):
                    leg.roll = route.roll
                    leg.roll_cost = ROLL_COST_PINNED
        return legs

    def _lane_z(self, route: Route, target) -> float:
        """The height the arm traverses sideways at, out on the staging plane.

        `direct` is the fruit's own height and is what a clear row wants. The
        other two exist for the case the check is actually there to catch: a
        neighbouring truss hanging on the line between where the arm is and
        where it is going. Ducking under the canopy or climbing over it turns a
        blocked route into a longer one instead of into a dropped tomato.
        """
        if route.lane == "direct":
            return float(target[2])
        zs = [self.row.pos(n)[2] for n in self.row.names]
        if route.lane == "under":
            # Not below the crate rim — ducking under the crop and then dragging
            # the gripper through the bin is not an improvement. This arm's
            # crate: on a trolley it rides beside its own arm, so the rim to
            # clear is not the same height for both.
            return float(max(min(zs) - 0.20,
                             self.frame.bin_pos[2] + BIN_WALL + 0.10))
        return float(max(zs) + 0.16)

    # -- checking it ----------------------------------------------------------

    def _verify(self, legs, name, route: Route):
        """Replay the route kinematically and measure how close it comes.

        Returns (worst clearance, [breaches], predicted tool path).

        The replay warm-starts from the arm's actual current configuration, so
        it predicts the route from where the arm really is, not from an assumed
        posture. Legs that translate are sampled every PREVIEW_STEP; `turn` and
        the holds are checked at their single pose.
        """
        import mink
        import mujoco

        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)

        clearance = ClearanceModel(self.model, self.row, name,
                                   prefix=self.prefix, others=self.others)
        # Required gap per obstacle. Lessons raise it for a fruit that has been
        # knocked off before, which is the whole mechanism by which a mistake
        # changes the next route. See lessons.advise.
        required = {n: self.clearance + route.keepout.get(n, 0.0)
                    for n in self.row.names}
        for structure in ("row_support_bar", "row_panel_face"):
            required[structure] = STRUCTURE_CLEARANCE
        if clearance.others is not None:
            for other, _sph in clearance.others.arms:
                required[f"arm {other}"] = ARM_CLEARANCE

        cfg = self._cfg
        cfg.update(self.data.qpos[: self.model.nq].copy())
        # ⚠️ **The preview's null-space bias has to be the executor's, or it
        # previews a posture the arm will never adopt.** `week2_pick.
        # anchor_posture` points the executor's `PostureTask` at `park_q`, so
        # the solver prefers the canonical park configuration wherever nothing
        # else is competing for the redundant DOF. This seeded the same task
        # from *wherever the arm happens to be*, which is a different
        # preference, and over eighteen legs the two diverge.
        #
        # It was invisible with one arm because the divergence is in the null
        # space — the tool goes to the same place either way, so every
        # clearance that mattered was against the crop, which the tool's own
        # position dominates. With two arms it stops being invisible: the
        # carry to the crate swings j1 round, the preview keeps the wound-up
        # posture through `withdraw`/`ready`/`park`, and the elbow it draws
        # sweeps across the aisle into the other arm. Measured, that reported
        #
        #     park: forearm_link within -129 mm of arm b
        #
        # and refused every route for a posture the executor unwinds out of.
        # Same class as the `pin` note above: a preview checked in a posture
        # the arm will never adopt is a measurement of a different machine.
        if self.park_q is not None:
            seed = self.data.qpos[: self.model.nq].copy()
            for jname, value in zip(self.joints, np.asarray(self.park_q)):
                seed[self.model.joint(jname).qposadr[0]] = value
            self._posture.set_target(seed)
        else:
            self._posture.set_target_from_configuration(cfg)
        tasks = [self._task, self._posture]

        crop = set(self.row.names)
        worst = np.inf
        breaches: list[Breach] = []
        path = []

        def measure(leg):
            """Write the previewed posture into scratch and read the gaps."""
            nonlocal worst
            q = cfg.q
            self._scratch.qpos[:] = self.data.qpos
            self._scratch.qpos[: len(q)] = q
            mujoco.mj_forward(self.model, self._scratch)
            tool = self._scratch.site(self.tool_site).xpos.copy()
            path.append(tool)

            for obstacle, (d, part) in clearance.per_obstacle(
                    self._scratch).items():
                is_crop = obstacle in crop
                # The headline clearance is the crop one. The panel sits 80 mm
                # behind the fruit and the gripper has to work in that gap, so
                # folding structure into the same number would report every
                # successful grasp as a near miss.
                if is_crop and d < worst:
                    worst = d
                # The target fruit is already out of the obstacle set, so the
                # grasp legs need no exemption for it. They do need one for the
                # panel behind it — that is where `contact_ok` earns its keep.
                #
                # ⚠️ **The other arm is never exempted, `contact_ok` or not.**
                # That flag exists so the gripper may work in the 80 mm gap
                # between the fruit and the panel behind it; the panel is
                # stationary scenery and touching it is survivable. Another 22 kg
                # arm is neither. Letting the grasp legs off this check would
                # exempt exactly the part of the mission where the tool is
                # furthest from its own base and nearest the other arm's.
                if obstacle.startswith("arm "):
                    if d < required.get(obstacle, ARM_CLEARANCE):
                        breaches.append(Breach(leg.name, d, part, obstacle,
                                               tool))
                    continue
                if not is_crop and leg.contact_ok:
                    continue
                if d < required.get(obstacle, self.clearance):
                    breaches.append(Breach(leg.name, d, part, obstacle, tool))

        for leg in legs:
            if leg.kind in ("move", "pull"):
                here = cfg.get_transform_frame_to_world(
                    self.tool_site, "site").translation()
                goal = np.asarray(leg.goal, dtype=float)
                n_steps = max(2, int(np.linalg.norm(goal - here) / PREVIEW_STEP) + 1)
                self._task.set_orientation_cost(
                    [leg.orientation_cost, leg.orientation_cost, leg.roll_cost])
                for i in range(1, n_steps + 1):
                    p = here + (goal - here) * (i / n_steps)
                    if leg.approach is not None and leg.orientation_cost > 0:
                        R = approach_rotation(leg.approach, leg.roll)
                        self._task.set_target(
                            mink.SE3.from_rotation_and_translation(
                                mink.SO3.from_matrix(R), p))
                    else:
                        self._task.set_target(mink.SE3.from_translation(p))
                    for _ in range(PREVIEW_ITERS):
                        vel = mink.solve_ik(cfg, tasks, PREVIEW_DT, "daqp",
                                            1e-3, limits=self._limits)
                        cfg.integrate_inplace(vel, PREVIEW_DT)
                    measure(leg)
            elif leg.kind == "home" and leg.posture is not None:
                # ⚠️ **The one leg that moves the arm without moving the tool,
                # and the preview used to skip the motion entirely.** `unwind`
                # drives the *joints* to `park_q`; measuring it at whatever
                # posture the previous leg left behind reports the clearance of
                # the posture the arm is unwinding *out of*, which is exactly
                # the wound-up one. Walk the configuration to the posture the
                # executor will ramp to, then measure there.
                q = np.array(cfg.q).copy()
                for jname, value in zip(self.joints,
                                        np.asarray(leg.posture, dtype=float)):
                    q[self.model.joint(jname).qposadr[0]] = value
                cfg.update(q)
                measure(leg)
            elif leg.goal is not None:
                measure(leg)

        breaches.sort(key=lambda b: b.distance)
        return worst, breaches, (np.array(path) if path else None)

    # -- the entry point ------------------------------------------------------

    def plan(self, name) -> Mission:
        """Plan a pick of `name`. Nothing moves; the arm is only read from.

        Candidates are tried cheapest-first — the direct lane at the normal
        staging plane is both the fastest route and the one the row is laid out
        for — and widened only when the check says the direct one is blocked.
        """
        target = self.row.pos(name)
        base = Route(stage_x=self.frame.stage_x, frame=self.frame)
        applied = []
        if self.lessons is not None:
            base, applied = self.lessons.advise(name, self.row, base)

        tried, best = [], None
        for route in self._candidates(base):
            legs = self._legs(name, target, route)
            worst, breaches, path = self._verify(legs, name, route)
            tried.append(f"{route.label()}({worst * 1000:.0f}mm)")
            mission = Mission(target=name, legs=legs, ok=not breaches,
                              clearance=worst, lane=route.lane,
                              stage_x=route.stage_x, breaches=breaches,
                              tried=tried, applied=applied, path=path,
                              frame=self.frame)
            if not breaches:
                return mission
            if best is None or worst > best.clearance:
                best = mission

        best.tried = tried
        return best

    def _candidates(self, base: Route):
        """Routes to try, in order: smallest change to the nominal cycle first.

        The order is not arbitrary. Changing the pull direction is one waypoint
        and costs nothing, so it comes before changing lane, which changes the
        posture the arm enters the row from. Backing the staging plane off is
        last because it lengthens every traverse and so costs cycle time on
        every future pick — and cycle time is what Week 4's kg/hr is made of.
        """
        order = [
            (base.lane, base.pull, base.roll, 0.00),
            (base.lane, "out", base.roll, 0.00),
        ]
        # Rolling the wrist is the next cheapest thing after changing the pull:
        # it moves no waypoint and adds no travel, it just fits the gripper's
        # linkage into a different part of the gap between two fruit. On this
        # row it is the difference between picking t0 and refusing it — t0 and
        # t1 sit 71 mm apart surface to surface and the 2F85 spans 85 mm, so
        # square-on there is no clearance to be had at any staging distance.
        order += [(base.lane, "far", base.roll, 0.00)]
        order += [(base.lane, "out", roll, 0.00) for roll in ROLL_OPTIONS]
        order += [(base.lane, "far", roll, 0.00) for roll in ROLL_OPTIONS]
        order += [
            ("under", base.pull, base.roll, 0.00),
            ("over", base.pull, base.roll, 0.00),
            ("under", "out", base.roll, 0.00),
            ("over", "out", base.roll, 0.00),
            (base.lane, base.pull, base.roll, 0.08),
            (base.lane, "out", base.roll, 0.08),
            (base.lane, "out", base.roll, 0.16),
        ]
        seen = set()
        for lane, pull, roll, backoff in order:
            route = Route(lane=lane, pull=pull, roll=roll,
                          stage_x=base.stage_x - backoff,
                          approach_gap=base.approach_gap,
                          keepout=base.keepout, frame=self.frame)
            if route.label() in seen:
                continue
            seen.add(route.label())
            yield route


def park_posture(model, iters=600, prefix="", pin=(), hold=None, frame=None):
    """Arm joint angles that put the tool at PARK, facing the row.

    Solved once at startup and reused as the pose every attempt begins and ends
    at. It replaces the arm's HOME posture as the resting place for one measured
    reason: at HOME the open gripper pads are **1.7 mm inside the back panel**
    (`gr_right_pad1` against `row_panel_face`, at the posture `reset_home`
    leaves the arm in). Every attempt therefore began in contact with the row,
    and the first move of every cycle started by pushing off it.

    Returns the six arm joint values, not a full qpos vector — writing a whole
    qpos would teleport the fruit's free joints along with the arm.
    """
    import mink
    import mujoco

    from fr5 import JOINTS, reset_home

    # ⚠️ Which arm's world to park in. `None` is Week 1-4's, so the single-armed
    # callers below and in `week2_pick` are unchanged. See `ArmFrame`.
    fr = _frame(frame)
    PARK, INTO_ROW = fr.park, fr.into_row

    # ⚠️ `prefix` names which arm on a machine carrying more than one. Bare
    # names are arm a and every single-armed scene keeps them. Without it the
    # solve runs against arm a's joints while claiming to be arm b's park
    # posture, and the caller gets six plausible numbers for the wrong arm.
    joints = [prefix + j for j in JOINTS]
    tool = prefix + TOOL_SITE

    # ⚠️ `pin` names every joint in the model that this solve must **not**
    # use, and on a multi-armed machine it is not optional. `mink.Configuration`
    # spans the whole model, so with two arms and a trolley the QP has 13 free
    # DOF to put one tool at one point — it will happily fold the other arm and
    # drive the base. Only `joints` is copied back out, so the answer is a
    # posture for six joints solved on the assumption that seven others moved
    # too. It fails loudly here ("park posture missed by 1950 mm") only because
    # the error is checked; see `farm.armframe.pin_base` for the same bug
    # failing quietly.

    # ⚠️ Seed from HOME, not from whatever the caller happens to have in mjData.
    # A fresh MjData is all zeros, and all zeros is the arm standing straight up
    # — a singular configuration where the Jacobian loses rank and IK converges
    # to 21 mm off and stops. It fails as a bad park pose rather than as a bad
    # seed, which is a confusing hour.
    scratch = mujoco.MjData(model)
    reset_home(model, scratch)

    # ⚠️ `hold` puts joints the caller cares about back where the *live* scene
    # has them before the solve. It exists because PARK is a world point: on a
    # trolley it is computed from where the machine is standing now, while this
    # scratch is a fresh MjData with the drive joint at zero. Solve the two
    # together and the arm is asked to reach a park pose several metres down
    # the row — 1668 mm, on the run that found this, and only visible because
    # the error is checked. It was latent with one arm: the single caller
    # happened to run before the trolley had ever moved.
    for jname, value in (hold or {}).items():
        scratch.joint(jname).qpos[0] = value
    if hold:
        mujoco.mj_forward(model, scratch)

    cfg = mink.Configuration(model)
    cfg.update(scratch.qpos[: model.nq].copy())
    task = mink.FrameTask(frame_name=tool, frame_type="site",
                          position_cost=1.0,
                          orientation_cost=[ORIENTATION_COST, ORIENTATION_COST,
                                            ROLL_COST],
                          lm_damping=1.0)
    posture = mink.PostureTask(model, cost=1e-2)
    posture.set_target_from_configuration(cfg)
    task.set_target(mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(approach_rotation(INTO_ROW)), PARK))

    limits = None
    if pin:
        limits = [mink.VelocityLimit(model, {j: 0.0 for j in pin})]
    for _ in range(iters):
        vel = mink.solve_ik(cfg, [task, posture], PREVIEW_DT, "daqp", 1e-3,
                            limits=limits)
        cfg.integrate_inplace(vel, PREVIEW_DT)

    adr = [model.joint(j).qposadr[0] for j in joints]
    q = np.array([cfg.q[a] for a in adr])

    for a, v in zip(adr, q):
        scratch.qpos[a] = v
    mujoco.mj_forward(model, scratch)
    err = float(np.linalg.norm(scratch.site(tool).xpos - PARK))
    if err > 0.02:
        raise RuntimeError(
            f"park posture missed by {err * 1000:.0f} mm — PARK may be "
            f"outside the reach envelope")
    return q


def reset_park(model, data, q, prefix=""):
    """Reset the scene with the arm parked outside the canopy, not inside it.

    ⚠️ **This resets the *whole world*, not just the arm.** `mj_resetData`
    restores every free body to its spawn pose and every `eq_active` flag to its
    model default — so a tomato that was picked and crated reappears on the
    plant with its stem re-welded. That is exactly right when starting a fresh
    independent trial, and completely wrong between the fruit of one row.
    Use `park_arm` for that; see the note there.
    """
    import mujoco

    from fr5 import JOINTS

    mujoco.mj_resetData(model, data)
    for value, jname in zip(q, (prefix + j for j in JOINTS)):
        data.joint(jname).qpos[0] = value
        data.ctrl[model.actuator(f"{jname}_pos").id] = value
    mujoco.mj_forward(model, data)


def park_arm(model, data, q, prefix=""):
    """Put the arm back at the park posture, leaving the world alone.

    The counterpart to `reset_park`, and the one a **sequence harvest** needs.
    Harvesting a row means the scene carries state forward: fruit that were
    picked are in the crate, their stems are broken, and the row empties as the
    arm works it. `reset_park` destroys all of that — it is a fresh trial, not
    the next pick.

    What this touches: the six arm joints (position, velocity, and their
    actuator setpoints) and the gripper command. What it deliberately does not:
    `qpos` of any free body, `eq_active`, or anything else in `mjData`.

    ⚠️ It is a teleport, not a move. That is acceptable *between* picks, where
    the arm is empty and already near park, and it is not acceptable at any
    other time — teleporting with a fruit in the gripper would drag it through
    the scene with the constraint solver fighting it. Call it only when the
    gripper is open and holding nothing.

    Why teleport rather than drive back: posture. Week 2's `anchor_posture`
    found that the same tool point reached from a wound-up posture is a
    different problem for local differential IK, and a long traverse along the
    row winds it. Driving back returns the tool to park without necessarily
    unwinding the joints; setting the joint angles does both.
    """
    import mujoco

    from fr5 import GRIPPER_OPEN, JOINTS, gripper_ctrl

    for value, jname in zip(q, (prefix + j for j in JOINTS)):
        joint = data.joint(jname)
        joint.qpos[0] = value
        joint.qvel[0] = 0.0
        data.ctrl[model.actuator(f"{jname}_pos").id] = value

    grip = gripper_ctrl(model, prefix)
    if grip is not None:
        data.ctrl[grip] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def main():
    """Plan a pick of every fruit in the row and print the routes."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from fr5 import reset_home
    from plant_row import Row
    from greenhouse import build_scene

    model = build_scene()
    data = mujoco.MjData(model)
    reset_home(model, data)
    row = Row(model, data)
    mujoco.mj_forward(model, data)

    park = park_posture(model)
    reset_park(model, data, park)
    print(f"\npark posture {np.degrees(park).round(1)} deg  ->  tool "
          f"{data.site(TOOL_SITE).xpos.round(3)}")

    planner = Planner(model, data, row)
    print(f"planning against {len(row.names)} trusses, "
          f"clearance {CLEARANCE * 1000:.0f} mm crop / "
          f"{STRUCTURE_CLEARANCE * 1000:.0f} mm structure\n")
    for n in row.names:
        m = planner.plan(n)
        print("  " + m.summary().replace("\n", "\n  "))
        for leg in m.legs:
            where = ("      " + np.array2string(leg.goal, precision=2,
                                                suppress_small=True)
                     if leg.goal is not None else "")
            print(f"      {leg.name:<9} {leg.kind:<8}{where}")
        print()


if __name__ == "__main__":
    main()
