#!/usr/bin/env python3
"""The pipe-rail trolley the arm rides on, and why it is a trolley at all.

--- what is actually available, because I went looking ----------------------

There is **no off-the-shelf mobile base for the FR5**, and no public URDF for
one. Fairino sell the arm as a 149 mm-footprint, ~22 kg unit that mounts in any
orientation; the mobile integration is left to the integrator. The open-source
mobile bases that *do* ship URDFs — DiffBot, ROMR, TurtleBot, Remo — are all
small differential-drive research platforms with a laptop shelf on top. None of
them is built to carry 22 kg of arm plus a crate of fruit, and none of them
would be allowed to free-roam a working greenhouse anyway.

⚠️ **And a free-roaming AMR is the wrong machine here regardless, which is the
more interesting finding.** Dutch glasshouses already have a rail network in
every aisle: the heating pipes. Every trolley in the industry — Berg
Hortimotive's BRW and Benomic ranges, Bogaerts, Berkvens — runs on those pipes,
because it means no localisation problem, no drift, no floor space lost to
navigation margin, and a machine that physically cannot wander into a crop row.
A pipe-rail trolley is a **1-DOF robot**: it has a position along the row, and
that is all. An AMR in the same aisle would be solving SLAM to reproduce a
constraint the building gives away for free.

So this is built rather than imported, to the dimensions the real ones use:

    gauge            550 mm c.t.c., matching `house.RAIL_CTC`
    rides on         51 mm OD heating pipe, on grooved rollers
    platform         1.70 x 0.90 m. Berg's BRW170 is 1.70 x 0.40 m for a person;
                     this is wider because it carries two arms and a crate
                     rather than one picker, and the length is theirs.
    deck height      250 mm — see the note on DECK_Z, which is load-bearing
    drive            one slide joint along the row. That is the whole chassis.

--- the two arms ------------------------------------------------------------

The mounts sit at +/-200 mm from the centreline, which `house.py` derives from
the 1.60 m row pitch: it puts each arm 600 mm from the crop it works, which is
the standoff every Week 1-4 number was measured at. Arm A works the row on the
+x side, arm B the -x side. Only A is fitted by default; B's plate is drawn, and
`--arms 2` fits both.

    ./.venv/bin/python simulation/mujoco/farm/trolley.py            # a window
    ./.venv/bin/python simulation/mujoco/farm/trolley.py --drive    # drive it
    ./.venv/bin/python simulation/mujoco/farm/trolley.py --shot     # stills
    ./.venv/bin/python simulation/mujoco/farm/trolley.py --reach    # the gate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import greenhouse as gh  # noqa: E402
from farm import house  # noqa: E402
from greenhouse import _decor  # noqa: E402

# --- the machine -------------------------------------------------------------

# ⚠️ **DECK_Z is load-bearing and it is not a styling choice.** Every reach,
# clearance and cycle-time number in Weeks 1-4 was measured with the arm's base
# on the floor and the fruit at z = 0.42..0.72. What the arm cares about is the
# fruit's height *relative to its own base*, so putting the arm on a 250 mm deck
# and leaving the crop where it was would move every one of those numbers.
#
# So the crop moves up with the arm: `farm.crop` hangs fruit at DECK_Z + the
# Week 1-4 band, i.e. 0.67..0.97 m. That keeps the arm-relative geometry
# identical *and* is the more realistic crop height — a real ripening truss sits
# around 0.7-1.0 m, well above the 0.32 m gutter, which the old absolute numbers
# did not.
DECK_Z = 0.25

DECK_L = 1.70             # along the row. Berg BRW170's platform length.
DECK_W = 1.10             # across it. Wider than theirs: two arms and a crate,
                          # and wide enough that CRATE_LOCAL is on the deck.
DECK_T = 0.04

WHEEL_R = 0.065
WHEEL_BASE = 1.20         # axle to axle along the row

# How fast the trolley drives between stops. Berg's trolleys are speed-adjustable
# and creep in the 0.1-0.5 m/s band under a picker; 0.35 m/s is a working figure
# and, unlike the arm's joint speed, it is *not* a number anything else in the
# repo was measured against — so it is free to change.
DRIVE_SPEED = 0.35

# Position-servo gains for the drive. Sized to the 240 kg chassis: kp has to
# hold it against nothing much (the rail is level and the joint is prismatic
# along it), and kv is the one that matters, because an underdamped 240 kg base
# oscillating under the arm is a moving camera mount and a moving pick target.
DRIVE_KP = 90000.0
DRIVE_KV = 12000.0

TROLLEY = "trolley"
DRIVE_JOINT = "trolley_y"
DRIVE_ACT = "trolley_y_pos"

# Arm mounts, in trolley-local x. See `house.ARM_OFFSET`.
ARM_X = {"a": +house.ARM_OFFSET, "b": -house.ARM_OFFSET}

# The name prefix each arm's joints, actuators, sites, links and cameras carry.
#
# ⚠️ **Arm "a" is deliberately unprefixed**, and this is the whole reason a
# two-armed machine did not require rewriting Weeks 1-4. `mission`, `reach`,
# `week2_pick` and `carrytrace` all look up `j1`, `tool0`, `wrist3_link` and
# `gr_fingers_actuator` by bare name; give arm a a prefix and every one of them
# raises `KeyError` on a name that used to exist. Arm b takes `b_` and every
# module that has to address a specific arm takes a `prefix` argument
# defaulting to "".
ARM_PREFIX = {"a": "", "b": "b_"}

# Which way each arm faces, in degrees of yaw about the deck's z axis.
#
# ⚠️ **Arm b is bolted down turned round, and this is mechanical rather than
# cosmetic.** The two arms work rows on opposite sides of the aisle: arm a
# reaches toward +x, arm b toward -x. An FR5 mounted facing +x and asked to work
# a row behind it has to fold back over its own shoulder — the postures that
# reach at all are wound up, near singular, and nothing Weeks 1-4 measured
# applies to them. Turned 180°, arm b's *own* frame sees exactly what arm a's
# does: a row 600 mm ahead at the same heights. That is the whole reason the
# second arm needed no new manipulation numbers.
#
# `farm.armframe` has to apply the same rotation to the Week 1-4 planning
# constants, which are world-frame. See `_MIRROR` there.
ARM_YAW = {"a": 0.0, "b": 180.0}


def other_arms(tag, arms):
    """Prefixes of the arms on this machine that are *not* `tag`.

    What `farm.armframe.pin_base` needs: one Reacher's IK has every other arm's
    joints inside its own `mink.Configuration`, and they have to be pinned or
    the solver plans with them and the executor never moves them.
    """
    return tuple(ARM_PREFIX[t] for t in arms if t != tag)

CHASSIS = [0.36, 0.38, 0.42, 1.0]
DECKTOP = [0.58, 0.60, 0.63, 1.0]


def y_limits():
    """How far along the row the trolley may drive, as (lo, hi).

    The rails run past the crop at both ends so a trolley can be pushed on at
    the headland; the *working* range is the crop's length.
    """
    return -house.HOUSE_HALF_Y, house.HOUSE_HALF_Y


# --- drawing and building ----------------------------------------------------

def add_trolley(spec, aisle=0, arms=("a",), crate=True, y0=0.0):
    """Put the trolley, its drive joint and its arm(s) into a spec.

    Call before `spec.compile()`. Returns the trolley body.

    ⚠️ Everything structural here is `contype=0`, which is this repo's standing
    rule for anything that is drawn but was not part of the collision set the
    clearance numbers were measured against. A solid deck 250 mm under the arm
    would be a new obstacle in `mission.ClearanceModel`, and it would change the
    planner's answers silently. The crate is the exception and has to be: fruit
    are dropped into it, so it is the one part the physics needs.
    """
    import mujoco

    ax = house.aisle_x(aisle)
    body = spec.worldbody.add_body(name=TROLLEY, pos=[ax, 0.0, 0.0])

    # The whole chassis: one prismatic degree of freedom along the row. A
    # pipe-rail trolley genuinely has no others, which is the point.
    body.add_joint(name=DRIVE_JOINT, type=mujoco.mjtJoint.mjJNT_SLIDE,
                   axis=[0, 1, 0], range=y_limits(), damping=180.0)
    # Mass so the arm swinging on top does not visibly rock the base. A real
    # BRW-class trolley with two arms and a crate is a few hundred kg.
    body.add_geom(name="tr_mass", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, DECK_Z - 0.10], size=[DECK_W / 2, DECK_L / 2, 0.06],
                  rgba=CHASSIS, contype=0, conaffinity=0, mass=240.0)

    # Grooved rollers, one per rail per axle, seated on the 51 mm pipe.
    for sy, ytag in ((-1, "n"), (1, "p")):
        for sx, xtag in ((-1, "l"), (1, "r")):
            # fromto only — a cylinder given both `pos` and `fromto` is refused
            # at compile time, and fromto is the one that carries the axle
            # direction.
            _decor(body, f"tr_wheel_{ytag}{xtag}", mujoco.mjtGeom.mjGEOM_CYLINDER,
                   fromto=[sx * house.RAIL_CTC / 2 - 0.022,
                           sy * WHEEL_BASE / 2, house.RAIL_Z + 0.02,
                           sx * house.RAIL_CTC / 2 + 0.022,
                           sy * WHEEL_BASE / 2, house.RAIL_Z + 0.02],
                   size=[WHEEL_R, 0, 0], rgba=[0.18, 0.18, 0.20, 1.0], mass=0.0)
        # The axle beam between the two rollers.
        _decor(body, f"tr_axle_{ytag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[-house.RAIL_CTC / 2, sy * WHEEL_BASE / 2,
                       house.RAIL_Z + 0.02,
                       house.RAIL_CTC / 2, sy * WHEEL_BASE / 2,
                       house.RAIL_Z + 0.02],
               size=[0.016, 0, 0], rgba=CHASSIS, mass=0.0)

    # Chassis rails running the length of the deck, and the legs up to it.
    for sx, xtag in ((-1, "l"), (1, "r")):
        _decor(body, f"tr_frame_{xtag}", mujoco.mjtGeom.mjGEOM_BOX,
               fromto=[sx * (DECK_W / 2 - 0.05), -DECK_L / 2, DECK_Z - 0.06,
                       sx * (DECK_W / 2 - 0.05), DECK_L / 2, DECK_Z - 0.06],
               size=[0.03, 0.03, 0], rgba=CHASSIS, mass=0.0)
        for sy, ytag in ((-1, "n"), (1, "p")):
            _decor(body, f"tr_leg_{ytag}{xtag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[sx * house.RAIL_CTC / 2, sy * WHEEL_BASE / 2,
                           house.RAIL_Z + 0.02,
                           sx * (DECK_W / 2 - 0.05), sy * WHEEL_BASE / 2,
                           DECK_Z - 0.06],
                   size=[0.014, 0, 0], rgba=CHASSIS, mass=0.0)

    # The deck itself.
    _decor(body, "tr_deck", mujoco.mjtGeom.mjGEOM_BOX,
           pos=[0, 0, DECK_Z - DECK_T / 2], size=[DECK_W / 2, DECK_L / 2,
                                                  DECK_T / 2],
           rgba=DECKTOP, mass=0.0)

    # Mounting plates for both arms, drawn whether or not an arm is fitted —
    # "there is room for a second arm" is a claim about the machine, and a
    # visible empty plate is what makes it checkable rather than asserted.
    for tag, x in ARM_X.items():
        _decor(body, f"tr_plate_{tag}", mujoco.mjtGeom.mjGEOM_CYLINDER,
               pos=[x, 0, DECK_Z + 0.008], size=[0.085, 0.008, 0],
               rgba=[0.30, 0.32, 0.35, 1.0], mass=0.0)

    if crate:
        for tag in arms:
            _add_crate(spec, body, tag)

    for tag in arms:
        _mount_arm(spec, body, tag)

    # A position servo on the drive, so "go to y" is a setpoint the trolley
    # ramps toward rather than a teleport. ⚠️ Added *after* the arms: MuJoCo
    # actuators are ordered as declared, and every Week 1-4 module finds the
    # arm's by name — but `reach.Reacher` also slices `data.ctrl` by the index
    # array it builds, so an actuator inserted ahead of the arm's would be
    # harmless by name and wrong by index if anything ever iterates positionally.
    act = spec.add_actuator(
        name=DRIVE_ACT, target=DRIVE_JOINT,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE)
    act.gainprm[0] = DRIVE_KP
    act.biasprm[1] = -DRIVE_KP
    act.biasprm[2] = -DRIVE_KV
    act.ctrlrange = list(y_limits())
    act.ctrllimited = 1
    return body


# Where the crate sits on the deck, in trolley-local coordinates.
#
# ⚠️ **Placed at exactly `mission.BIN_POS`'s offset from the arm, not wherever
# there was room.** The first version put it behind the arms at local
# (0, -0.65), which is tidy and does not work: the carry leg has to reach 0.2 m
# *behind* the arm's own working direction with the wrist held downward, and the
# tool stalled 923 mm short of the crate on every pick — a fruit successfully
# grasped, detached and then carried nowhere.
#
# BIN_POS is [0.30, -0.52, 0.0] relative to an arm at the origin, and Weeks 1-4
# flew that carry a few hundred times. So the crate goes at the arm's mount plus
# that offset and the whole carry-and-release geometry comes across proven.
# ⚠️ One crate **per arm**, mirrored with the arm that feeds it. Arm b is
# bolted round 180° (see ARM_YAW), so the offset that puts a crate within
# arm a's proven carry puts it *behind* arm b. Sharing arm a's crate would ask
# arm b for a 0.70 m reach across the deck in exactly the direction the carry
# leg is worst at — the 923 mm stall described above, again.
def _crate_local(tag):
    """Where arm `tag`'s crate sits, in trolley-local coordinates."""
    sign = 1.0 if ARM_YAW[tag] == 0.0 else -1.0
    return np.array([ARM_X[tag] + sign * 0.30, sign * -0.52,
                     DECK_Z + 0.016])


CRATE_LOCAL = _crate_local("a")


def _add_crate(spec, body, tag="a"):
    """The container the fruit go into, riding on the deck behind the arms.

    ⚠️ **Collidable, unlike everything else on the trolley, and it has to be.**
    `week2_pick.execute` scores a pick with `fr5.crate_contains`, and a fruit
    released over a crate it falls straight through scores as a miss on an
    attempt that actually worked.

    ⚠️ The geometry is `fr5.add_crate`'s, restated rather than called, and the
    reason is that MjSpec bodies cannot be re-parented after they are made:
    `add_crate` puts its body on `spec.worldbody`, and a crate welded to the
    world is a crate the arm reaches for at the far end of the house. Same
    half-extent, same wall height, same 8 mm panels — driven off the same
    `mission.BIN_*` constants so the two cannot drift.
    """
    import mujoco

    from mission import BIN_HALF, BIN_WALL

    name = "crate" if tag == "a" else f"{ARM_PREFIX[tag]}crate"
    crate = body.add_body(name=name, pos=list(_crate_local(tag)))
    grey = [0.30, 0.33, 0.38, 1.0]
    t = 0.008
    crate.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[BIN_HALF, BIN_HALF, t], pos=[0, 0, t], rgba=grey)
    for tag, p, s in (
            ("x+", [BIN_HALF, 0, BIN_WALL / 2], [t, BIN_HALF, BIN_WALL / 2]),
            ("x-", [-BIN_HALF, 0, BIN_WALL / 2], [t, BIN_HALF, BIN_WALL / 2]),
            ("y+", [0, BIN_HALF, BIN_WALL / 2], [BIN_HALF, t, BIN_WALL / 2]),
            ("y-", [0, -BIN_HALF, BIN_WALL / 2], [BIN_HALF, t, BIN_WALL / 2])):
        crate.add_geom(name=f"{name}_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=s, pos=p, rgba=grey)
    return crate


def crate_pos(model, data, tag="a"):
    """Where arm `tag`'s crate is *now*, in world coordinates.

    ⚠️ The crate rides on the trolley, so it moves. `mission.BIN_POS` is a
    module constant and every Week 1-4 caller reads it as one; a mission planned
    against that constant while the trolley is at y = 3 m would aim the release
    at a patch of floor several metres behind the machine. Anything that drops
    fruit has to ask rather than remember.
    """
    return data.body("crate" if tag == "a"
                     else f"{ARM_PREFIX[tag]}crate").xpos.copy()


def _mount_arm(spec, body, tag):
    """Bolt an FR5 (with gripper) to one of the deck plates."""
    from fr5 import build_fr5_spec

    # `gripper=True` attaches the 2F85 itself and moves `tool0` out to the pinch
    # point; there is no separate attach call to make.
    arm = build_fr5_spec(with_scene=False, gripper=True)

    # ⚠️ Drop the URDF's `home` keyframe before attaching. A keyframe is a full
    # `qpos` vector for the model it belongs to, so the arm's six-number one is
    # meaningless the moment it becomes part of a bigger model — and MuJoCo does
    # not ignore it, it refuses to attach at all: "Keyframe 'home' has invalid
    # qpos size, got 6, should be 14". Nothing in this repo reads it; `park_q`
    # comes from `mission.park_posture`, which solves for the posture rather
    # than looking one up.
    for k in list(arm.keys):
        arm.delete(k)

    yaw = np.radians(ARM_YAW[tag])
    frame = body.add_frame(pos=[ARM_X[tag], 0.0, DECK_Z + 0.016],
                           quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    # See ARM_PREFIX for why arm "a" is the unprefixed one.
    spec.attach(arm, prefix=ARM_PREFIX[tag], frame=frame)
    return frame


def build(aisle=0, arms=("a",), crate=True, wrist_cam=False, deck_cam=False,
          leafy=True, pitch=0.32, seed=0, trusses=None):
    """The whole thing: house, trolley, arm(s), optionally cameras and crop."""
    import mujoco

    spec = mujoco.MjSpec()

    # ⚠️ **Solver options are global and do not survive `attach`.** MuJoCo keeps
    # the *parent* spec's, so the settings `build_fr5_spec` and `attach_gripper`
    # apply to the arm's own spec are silently dropped when it is attached here.
    # Every one of these is load-bearing:
    #   implicitfast — Euler at kv=400 and a 2 ms step is past the stability
    #                  limit and the arm shakes itself apart
    #   elliptic     — the pyramidal cone under-approximates friction
    #   impratio 10  — low values let contacts slide under load, which is
    #                  literally a gripper dropping fruit
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    worked = house.serves(aisle)
    house.add_house(spec, leafy=leafy, pitch=pitch, worked_rows=worked,
                    seed=seed)
    if trusses is not None:
        from farm.crop import add_trusses

        add_trusses(spec, trusses)
    add_trolley(spec, aisle=aisle, arms=arms, crate=crate)

    if wrist_cam:
        from camera import add_wrist_camera

        # One eye per arm, named for the arm. `arms` is the truth about what is
        # fitted, so a scene built with one arm gets one wrist camera and the
        # panel that renders `b_wrist` fails loudly rather than showing arm a's
        # view under arm b's label.
        for tag in arms:
            add_wrist_camera(spec, prefix=ARM_PREFIX[tag])
    if deck_cam:
        from farm.scout import add_deck_camera

        add_deck_camera(spec, aisle=aisle)

    # ⚠️ The pads again — same reason as `greenhouse.build_scene`, and it has to
    # be repeated because that function is not the one building this scene.
    # `greenhouse.PAD_SOLREF` is imported rather than restated so the two scenes
    # cannot drift apart on the number that decides whether a pick holds.
    for geom in spec.geoms:
        if "pad" in geom.name:
            geom.solref = list(gh.PAD_SOLREF)

    model = spec.compile()
    return model


# --- driving it --------------------------------------------------------------

class Drive:
    """The trolley's one degree of freedom, and the clock it costs.

    Position-controlled, like the arm's joints, so "go to y" is a setpoint and
    not a teleport — the deck camera has to see a moving row for the scouting
    pass to mean anything.
    """

    def __init__(self, model, data, speed=DRIVE_SPEED):
        self.model, self.data = model, data
        self.jid = model.joint(DRIVE_JOINT).qposadr[0]
        self.act = model.actuator(DRIVE_ACT).id
        self.speed = speed
        self.travelled = 0.0

    @property
    def y(self):
        return float(self.data.qpos[self.jid])

    def park_at(self, y):
        """Teleport. Only legal before a run starts, or between rows."""
        self.data.qpos[self.jid] = float(y)
        self.data.qvel[self.model.joint(DRIVE_JOINT).dofadr[0]] = 0.0
        self.data.ctrl[self.act] = float(y)

    def drive_to(self, y, on_tick=None, tol=0.004, max_s=60.0):
        """Ramp the setpoint there at `speed`. Returns the seconds it took."""
        import mujoco

        from reach import CTRL_DT, TICKS_PER_CTRL

        lo, hi = y_limits()
        y = float(np.clip(y, lo, hi))
        was = self.y
        t = 0.0
        while abs(self.y - y) > tol and t < max_s:
            step = self.speed * CTRL_DT
            cur = float(self.data.ctrl[self.act])
            self.data.ctrl[self.act] = cur + np.clip(y - cur, -step, step)
            for _ in range(TICKS_PER_CTRL):
                mujoco.mj_step(self.model, self.data)
            if on_tick is not None:
                on_tick()
            t += CTRL_DT
        # Measured off the joint, not off the request: a drive that ran out of
        # `max_s` short of its target has still only travelled how far it got,
        # and the odometer is what the route's distance claim is checked against.
        self.travelled += abs(self.y - was)
        return t


# --- the gate ----------------------------------------------------------------

def reach_gate(seed=1, verbose=True, arms=("a",)):
    """Can the arm on the trolley still reach the crop it is parked beside?

    ⚠️ **This is the check that says whether Weeks 1-4 carry over.** The whole
    argument for the 200 mm arm offset and the 250 mm deck is that together they
    reproduce the geometry those weeks measured — a 600 mm standoff and a crop
    0.42-0.72 m above the arm's base. That is an arithmetic claim, and this
    flies it: park the trolley beside a fruit, ask `mission.Planner` for a
    route, and see whether it gets one at the clearance Week 4 shipped.

    A failure here does not mean the arm is broken. It means the machine has
    been built in a place the arm cannot work from, which is a chassis bug
    wearing a manipulation bug's clothes.

    ⚠️ With `arms=("a", "b")` it flies the same check for **both** arms, each
    against its own row. That is the claim the second arm rests on: arm b is not
    a mirror of arm a by assertion, it is a mirror because `ARM_OFFSET` puts it
    the same 600 mm off a row on the other side of the aisle. Either that
    reproduces Week 1-4's geometry or it does not, and this is where it shows.
    """
    import mujoco

    from farm import armframe
    from farm import crop as fcrop
    from mission import Planner, park_arm, reset_park
    from plant_row import Row

    trusses = fcrop.spawn(n_per_row=10, rows=[0, 1], seed=seed)
    model = build(aisle=0, arms=arms, trusses=trusses, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    names = [t.name for t in trusses]
    total_ok = total_n = 0
    if verbose:
        print("\n  --- can the arms work the crop from the trolley? ---")

    for tag in arms:
        prefix = ARM_PREFIX[tag]
        # ⚠️ Solved in the arm's own frame, not the world's. `mission.park_posture`
        # aims at the constant `PARK`, which is 196 mm outside this arm's reach
        # because this arm is not at the origin. See `farm.armframe`.
        park_q = armframe.park_posture(model, data, tag, arms=arms)
        reset_park(model, data, park_q, prefix=prefix)

        row = Row(model, data, names=names,
                  homes={t.name: t.pos for t in trusses})
        mujoco.mj_forward(model, data)

        drive = Drive(model, data)
        worked = armframe._worked_row(0, tag)
        mine = [t for t in trusses if t.row == worked]

        rows = []
        for t in mine:
            drive.park_at(float(np.clip(t.y, *y_limits())))
            park_arm(model, data, park_q, prefix=prefix)
            mujoco.mj_forward(model, data)
            with armframe.at_trolley(model, data, tag):
                planner = Planner(model, data, row, lessons=None,
                                  clearance=0.040, park_q=park_q, speed=0.4,
                                  prefix=prefix)
                m = planner.plan(t.name)
            rows.append((t, m))

        ok = sum(1 for _t, m in rows if m.ok)
        total_ok += ok
        total_n += len(rows)
        if verbose:
            base = armframe.arm_base(model, data, tag)
            print(f"\n  arm {tag.upper()} at x={base[0]:.2f} works row "
                  f"r{worked} at x={house.row_x(worked):.2f} — standoff "
                  f"{house.WORK_STANDOFF * 1000:.0f} mm, the Week 1-4 figure\n")
            print(f"  {'fruit':<8} {'y':>7} {'z':>7} {'stage':<9} "
                  f"{'planned':>8} {'clearance':>10}  route")
            for t, m in rows:
                print(f"  {t.name:<8} {t.y:>7.2f} {t.z:>7.2f} {t.stage:<9} "
                      f"{str(m.ok):>8} "
                      f"{m.clearance * 1000 if m.ok else float('nan'):>10.0f}  "
                      f"{(m.tried[-1] if m.tried else m.lane) if m.ok else m.breaches[0] if m.breaches else 'no route'}")
            print(f"  planned {ok}/{len(rows)}")

    if verbose:
        print(f"\n  planned {total_ok}/{total_n} across "
              f"{len(arms)} arm{'s' if len(arms) > 1 else ''}")
        print(f"  {'PASS' if total_ok == total_n else 'FAIL'}")
    return total_ok, total_n


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reach", action="store_true", help="the gate, no window")
    ap.add_argument("--drive", action="store_true",
                    help="drive the length of the house in a window")
    ap.add_argument("--shot", action="store_true")
    ap.add_argument("--arms", type=int, default=1, choices=[1, 2])
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print(__doc__)
    if args.reach:
        ok, n = reach_gate(seed=args.seed,
                           arms=("a", "b")[: args.arms])
        return 0 if ok == n else 1

    os.environ.setdefault("MUJOCO_GL", "egl" if args.shot else "glfw")
    import mujoco

    from farm import crop as fcrop

    arms = ("a", "b")[: args.arms]
    seed = fcrop.resolve_seed(args.seed)
    trusses = fcrop.spawn(n_per_row=12, seed=seed)
    model = build(aisle=args.aisle, arms=arms, trusses=trusses, seed=seed)
    data = mujoco.MjData(model)

    from farm import armframe
    from mission import reset_park

    # In the arm's frame — see `farm.armframe`. `mission.park_posture` aims at
    # a world constant that belongs to an arm bolted at the origin.
    mujoco.mj_forward(model, data)
    park_q = armframe.park_posture(model, data)
    reset_park(model, data, park_q)
    mujoco.mj_forward(model, data)
    print(f"\n  compiled: {model.ngeom} geoms, {model.nq} dof, "
          f"{model.nu} actuators · arms {', '.join(arms)}")

    if args.shot:
        import cv2

        out_dir = Path(__file__).resolve().parents[3]
        with mujoco.Renderer(model, height=960, width=1280) as r:
            for cam in ("house", "aisle"):
                r.update_scene(data, camera=cam)
                p = out_dir / f"farm_trolley_{cam}.png"
                cv2.imwrite(str(p), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
                print(f"  wrote {p.name}")
            free = mujoco.MjvCamera()
            free.lookat[:] = [house.aisle_x(args.aisle), 0.0, 0.5]
            free.distance, free.azimuth, free.elevation = 3.6, 135.0, -18.0
            r.update_scene(data, camera=free)
            p = out_dir / "farm_trolley_machine.png"
            cv2.imwrite(str(p), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
            print(f"  wrote {p.name}")
        return 0

    import time

    import mujoco.viewer

    drive = Drive(model, data)
    print("\n  window open — the trolley drives the length of the house")
    with mujoco.viewer.launch_passive(model, data) as v:
        clock = [time.perf_counter()]

        def tick():
            if not v.is_running():
                raise KeyboardInterrupt
            v.sync()
            from reach import CTRL_DT

            clock[0] += CTRL_DT
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

        try:
            if args.drive:
                lo, hi = y_limits()
                for target in (hi - 0.3, lo + 0.3, 0.0):
                    t = drive.drive_to(target, on_tick=tick)
                    print(f"  drove to y={drive.y:+.2f} in {t:.1f} s "
                          f"(odometer {drive.travelled:.1f} m)")
            print("  close the window to quit")
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\n  window closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
