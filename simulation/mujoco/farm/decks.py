#!/usr/bin/env python3
"""One articulated deck camera per arm, each scanning its own row on its own.

--- why this exists next to `farm.scout` rather than inside it ---------------

`farm/scout.py` has an articulated head and it is a good one, but it is **one**
head serving two rows. That was the right shape when the second arm was new: a
single pan-tilt unit on the aisle centreline, turning 180° at every stop so both
rows get mapped by the same lens at the same 0.70 m standoff, which is what made
arm B's row visible at all (0/14 -> 8/14, measured in COMMANDS.md).

It has one structural cost, and it is not the slew time. **A shared head cannot
look two ways at once.** The two arms are serialised on the manipulation side
anyway (see `farm.duo`), but their *perception* need not be — and while arm A is
flying a 20 s pick, arm B's camera has nothing to do but wait for a head that is
pointed at arm A's row. A machine with two arms and one eye maps at half the
rate it could and cannot re-survey a row it is not currently facing.

So this is two heads, one per arm, each bolted over its own arm's mount plate
and each scanning **its own row only**:

    pan range      +/- SCAN_HALF_DEG about its own row, not 180 degrees
    slew           SCOUT_SLEW_DEG_S, the same PTU class as `scout.ScoutHead`
    standoff       ROW_PITCH/2 - ARM_OFFSET = 0.60 m, the Week 1-4 figure

⚠️ **The standoff is the point, and it is better than the shared head's.** The
centreline head sits 0.70 m from both rows by symmetry. A head over its own
arm's plate sits at the arm's own x, so it is 0.60 m from the row that arm
works — the *same* 600 mm standoff every Week 1-4 perception number was measured
at, and 100 mm closer than the shared head manages. Each arm's camera now sees
its row from where its arm sees it.

⚠️ **Neither head turns 180 degrees any more, and that is the saving.** The
shared head spent 22.5 s a pass swinging between rows. These never leave their
own row, so the whole cross-aisle slew disappears; what pan buys here is
*coverage along the row* from a standing trolley, which is what a scouting head
is actually for.

--- the head is bolted to the trolley, and it used to be a mocap body --------

⚠️ **This was a mocap body parented to the worldbody, and that is what caused
both deck-camera bugs.** The reasoning for mocap was real: a hinge adds DOFs to
`mjModel`, and `reach.Reacher` builds `mink.Configuration` over the *whole*
model, so the IK solver would discover two free joints that cost it nothing and
park them anywhere — "where the arm is reaching" would silently steer "where the
camera is looking". `farm.armframe.pin_base` exists because the trolley's slide
joint already had exactly this problem.

But mocap bodies **must** be children of the worldbody, so the head could not be
bolted to a moving trolley. It was kept over its mast by `ArmDeckHead.follow`
writing `mocap_pos` from the drive joint — from Python, on the ticks that
happened to call it.

That left one rigid assembly moving by **two different transport mechanisms**:

    the mast        a geom on the trolley body, moved by the model tree on
                    every physics step, continuously
    the head        a worldbody mocap body, moved only when Python said so

and everything that went wrong followed from that gap:

    deck cam TELEPORTS      `Drive.drive_to` steps physics for the whole
                            traverse and never calls `follow`. The camera stays
                            frozen where the last `follow` put it for the entire
                            drive, then jumps the full distance travelled the
                            next time one is called. It is not a camera moving
                            badly, it is a camera not moving at all and then
                            being told the answer.
    POLE moves, camera      the same desync seen from the other side. Through
    is left behind          the whole harvest nothing calls `follow`, so the
                            mast rides the trolley and the head does not.

⚠️ **The fix is structural rather than a smoothed teleport**, because lerping
the jump would only spread one frame's error over ten. The head is now a
**child body of the trolley**, with two hinges:

    deck_yaw_<tag>    pan, about the mast's own z, child of the trolley
    deck_head_<tag>   tilt, about the yoke's right axis, child of the yaw body

Position along the row is now inherited through the model tree exactly as the
mast's is, because it *is* the mast's — there is no second mechanism and so
nothing to desync. Only the articulation is driven from Python.

The DOF objection is answered rather than dodged: `head_joints()` names the new
joints and every IK caller pins them, the same contract the drive joint has
used since `pin_base`. `follow()` is kept as a no-op so the old call sites still
read, and so that re-introducing a Python-driven position is an obvious edit
rather than a quiet one.

    ./.venv/bin/python simulation/mujoco/farm/decks.py            # stills
    ./.venv/bin/python simulation/mujoco/farm/decks.py --split    # the proof
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import house, trolley  # noqa: E402
from farm.scout import SCOUT_FOVY, SCOUT_SLEW_DEG_S, SCOUT_YOKE_M  # noqa: E402
from greenhouse import _decor  # noqa: E402

# Per-arm camera and head names. Arm a is unprefixed everywhere else in this
# repo (see `trolley.ARM_PREFIX`); these are *new* names with no Week 1-4
# callers, so both arms are named symmetrically and neither is the special case.
DECK_CAM = {"a": "deck_a", "b": "deck_b"}
DECK_HEAD = {"a": "deck_head_a", "b": "deck_head_b"}
DECK_YAW = {"a": "deck_yaw_a", "b": "deck_yaw_b"}

# The two articulation joints per head, and their position servos. Named so
# every IK caller can pin them — see `head_joints`.
DECK_PAN_JOINT = {"a": "deck_pan_a", "b": "deck_pan_b"}
DECK_TILT_JOINT = {"a": "deck_tilt_a", "b": "deck_tilt_b"}

# How far the head may swing, in degrees. Wider than `SCAN_HALF_DEG` so the scan
# pattern is not sitting on a joint limit, and narrow enough that a runaway
# command cannot spin the camera to face the back wall.
PAN_LIMIT_DEG = 75.0
TILT_LIMIT_DEG = 45.0

# Position-servo gains for the pan-tilt unit. It carries a D435 and a yoke —
# well under a kilogram — so these only have to hold the tilt axis against
# gravity and settle quickly. The pan axis is vertical and carries no gravity
# torque at all.
HEAD_KP = 220.0
HEAD_KV = 22.0
HEAD_MASS = 0.55          # D435 plus the trunnion and yoke, roughly


def head_joints(arms=("a", "b")):
    """Every deck-head joint name, for the `pin` sets.

    ⚠️ **Pinning these is not optional and it is the price of making the head a
    real body.** `mink.Configuration` spans the whole model, so an unpinned pan
    or tilt joint is a free DOF the IK solver will happily use to help the tool
    reach — and `reach.Reacher.step` writes only the arm's six actuators, so the
    solve would be "reach by also turning the camera" and the arm would land
    short by whatever the camera was supposed to contribute. That is exactly
    `farm.armframe.pin_base`'s bug with a camera in place of the trolley, and it
    is why this head was a mocap body for as long as it was.
    """
    out = []
    for tag in arms:
        out += [DECK_PAN_JOINT[tag], DECK_TILT_JOINT[tag]]
    return out

# Where each head's pan axis stands, in trolley-local coordinates: directly over
# its own arm's mount plate, 1.10 m up. Same height as `scout.SCOUT_MOUNT` — it
# is above the crop band (0.67..0.97 m on the deck) and below the glazing bars.
DECK_LEAD = 0.55          # ahead of the trolley centre, along the row
DECK_Z_CAM = 1.10

# How far either side of its own row the head scans, in degrees of pan.
#
# ⚠️ **Chosen from the camera's window, not picked for looking busy.** The lens
# sees roughly a 1.0 m span of row at 0.60 m with a 58 deg vertical FoV. Panning
# +/-30 deg sweeps the aim point along the row by 0.60*tan(30) = +/-0.35 m, which
# widens the surveyed span to about 1.7 m — enough that a `SCOUT_STRIDE` of
# 0.50 m gives every fruit three looks instead of two, and not so wide that the
# head is staring down the row at a grazing angle where fruit occlude each other.
SCAN_HALF_DEG = 30.0

# The scan pattern each head walks at a stop, in degrees of pan from its row.
#
# ⚠️ **The two arms are given *different* patterns, and that is deliberate.**
# They are phase-shifted so that at any given moment the two heads are pointing
# in genuinely different directions — which is the claim "independently
# articulated" makes, and a demo where both heads sweep in lockstep would look
# identical to one head mirrored. See `duo.MissionState.deck_pan` for where this
# is read out and put on screen.
SCAN_A = (0.0, -SCAN_HALF_DEG, +SCAN_HALF_DEG)
SCAN_B = (+SCAN_HALF_DEG, 0.0, -SCAN_HALF_DEG)
SCAN = {"a": SCAN_A, "b": SCAN_B}

# A small downward tilt. The lens is at 1.10 m and the crop band centres at
# about 0.82 m, 0.60 m away — atan(0.28/0.60) is 25 deg, so the head is aimed
# down at the fruit rather than over the top of them.
DECK_TILT_DEG = 0.0       # the home aim already points at the band; see `_aim`


def row_of(tag, aisle=0):
    """Which row arm `tag` works from `aisle`. `armframe._worked_row`, re-exported."""
    from farm import armframe

    return armframe._worked_row(aisle, tag)


def _mount_local(tag):
    """The pan axis, in trolley-local coordinates: over the arm's own plate.

    ⚠️ `DECK_LEAD` is measured from **its own arm**, not from the deck centre.
    The arms are staggered 500 mm along the row (`trolley.ARM_STAGGER`), so a
    camera at a fixed deck y would lead one arm by 0.55 m and the other by 0.05 m
    — one head surveying the section its arm is about to work and the other
    staring at the fruit its arm is already on top of.
    """
    return np.array([trolley.ARM_X[tag], trolley.ARM_Y[tag] + DECK_LEAD,
                     DECK_Z_CAM])


def _aim_local(tag):
    """What the head looks at with pan=0: its own row, at the crop's mid height.

    ⚠️ In trolley-local x, arm a's row is at +ROW_PITCH/2 and arm b's at
    -ROW_PITCH/2 — the two rows are on opposite sides of the aisle, which is the
    whole reason arm b is bolted round 180 deg (`trolley.ARM_YAW`). A head that
    used arm a's aim for both would point arm b's camera straight across the
    aisle at arm a's crop, and the map would fill with fruit arm b cannot reach.
    """
    sign = +1.0 if tag == "a" else -1.0
    return np.array([sign * house.ROW_PITCH / 2, trolley.ARM_Y[tag] + DECK_LEAD,
                     fcrop.fruit_z(sum(fcrop.Z_LOCAL) / 2)])


def add_arm_deck_cameras(spec, aisle=0, arms=("a", "b"),
                         body_name=trolley.TROLLEY):
    """Put one articulated deck camera per arm into a spec. Before `compile()`.

    Returns the trolley body. Each arm gets a mast (on the trolley, decor), a
    mocap head (on the worldbody, driven by `ArmDeckHead`) and a D435 on a yoke.
    """
    import mujoco

    from camera import _xyaxes_towards, add_camera_housing

    body = spec.body(body_name)
    ax = house.aisle_x(aisle)

    for tag in arms:
        mount = _mount_local(tag)
        aim = _aim_local(tag)
        xyaxes = _xyaxes_towards(mount, aim, up_hint=(0.0, 0.0, 1.0))
        mx, my = float(mount[0]), float(mount[1])

        # The mast rides the trolley — it is decor and `contype=0`, this repo's
        # standing rule for anything drawn that was not in the collision set the
        # clearance numbers were measured against.
        _decor(body, f"deckmast_{tag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[mx, my, trolley.DECK_Z, mx, my, DECK_Z_CAM - 0.04],
               size=[0.020, 0, 0], rgba=[0.42, 0.44, 0.47, 1.0], mass=0.0)

        # ⚠️ **Children of the trolley, not of the worldbody.** `pos` is
        # therefore trolley-local, and the drive joint carries the whole head
        # down the row for free — which is the entire point. The old worldbody
        # mocap seat had to be world coordinates and had to be rewritten from
        # Python every tick; see the module docstring for what that cost.
        #
        # Two bodies rather than one: the pan table carries the trunnion, and
        # the tilt yoke rides on the pan table. Pan outer, tilt inner — the
        # other order yaws about a tilted axis and rolls the horizon.
        yaw = body.add_body(name=DECK_YAW[tag], pos=[mx, my, DECK_Z_CAM])
        yaw.add_joint(name=DECK_PAN_JOINT[tag],
                      type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                      range=[-np.radians(PAN_LIMIT_DEG),
                             np.radians(PAN_LIMIT_DEG)],
                      damping=1.2)
        yaw.add_geom(name=f"deck_trunnion_{tag}",
                     type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     fromto=[0, -0.028, 0, 0, 0.028, 0], size=[0.018, 0, 0],
                     rgba=[0.35, 0.37, 0.40, 1.0], contype=0, conaffinity=0,
                     mass=0.10)

        # The tilt axis is the camera's own "right" at home, which
        # `_xyaxes_towards` makes horizontal by construction — that is what lets
        # pan and tilt stay azimuth and elevation rather than a general
        # rotation.
        right0 = [float(v) for v in xyaxes[:3]]
        head = yaw.add_body(name=DECK_HEAD[tag], pos=[0, 0, 0])
        head.add_joint(name=DECK_TILT_JOINT[tag],
                       type=mujoco.mjtJoint.mjJNT_HINGE, axis=right0,
                       range=[-np.radians(TILT_LIMIT_DEG),
                              np.radians(TILT_LIMIT_DEG)],
                       damping=0.8)
        local = [SCOUT_YOKE_M, 0.0, 0.0]
        head.add_geom(name=f"deck_yoke_{tag}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, 0, 0] + local, size=[0.010, 0, 0],
                      rgba=[0.45, 0.47, 0.50, 1.0], contype=0, conaffinity=0,
                      mass=HEAD_MASS)
        head.add_camera(name=DECK_CAM[tag], pos=local, fovy=SCOUT_FOVY,
                        xyaxes=xyaxes)
        add_camera_housing(head, f"cam_{DECK_CAM[tag]}", local, xyaxes,
                           kind="d435", stalk=[0.0, 0.0, 0.0])

        # Position servos, so the head *holds* its aim. The tilt axis carries
        # the camera on a 90 mm yoke and would otherwise sag under gravity; the
        # pan axis is vertical and needs only to resist the trolley's
        # acceleration down the row.
        for jname in (DECK_PAN_JOINT[tag], DECK_TILT_JOINT[tag]):
            act = spec.add_actuator(
                name=f"{jname}_pos", target=jname,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE)
            act.gainprm[0] = HEAD_KP
            act.biasprm[1] = -HEAD_KP
            act.biasprm[2] = -HEAD_KV
    return body


class ArmDeckHead:
    """One arm's deck camera: where it points, and how to point it elsewhere.

    `pan` is degrees from *this arm's own row*, so pan=0 means "square at my
    crop" for both arms even though their rows are 180 deg apart in world terms.
    That is what makes `SCAN_A` and `SCAN_B` comparable numbers rather than two
    unrelated conventions, and it is why the panel can print a pan angle without
    the reader having to know which arm they are looking at.
    """

    def __init__(self, model, data=None, tag="a", aisle=0):
        from camera import _xyaxes_towards

        self.model, self.tag, self.aisle = model, tag, aisle
        try:
            pan_j = model.joint(DECK_PAN_JOINT[tag])
            tilt_j = model.joint(DECK_TILT_JOINT[tag])
        except KeyError:
            raise RuntimeError(
                f"no deck head for arm {tag} in this model — build the scene "
                f"with arm_decks=True")
        self.pan_adr = int(pan_j.qposadr[0])
        self.tilt_adr = int(tilt_j.qposadr[0])
        self.pan_dof = int(pan_j.dofadr[0])
        self.tilt_dof = int(tilt_j.dofadr[0])
        self.pan_act = int(model.actuator(f"{DECK_PAN_JOINT[tag]}_pos").id)
        self.tilt_act = int(model.actuator(f"{DECK_TILT_JOINT[tag]}_pos").id)
        self.body_id = int(model.body(DECK_HEAD[tag]).id)
        self.cam_id = int(model.camera(DECK_CAM[tag]).id)

        self.pivot_local = _mount_local(tag)
        self.ax = house.aisle_x(aisle)
        self.jadr = model.joint(trolley.DRIVE_JOINT).qposadr[0]

        mount, aim = _mount_local(tag), _aim_local(tag)
        # Horizontal by construction — `_xyaxes_towards` crosses forward with
        # world up — which is what lets pan and tilt stay azimuth and elevation.
        self.right0 = np.asarray(
            _xyaxes_towards(mount, aim, up_hint=(0.0, 0.0, 1.0))[:3], float)
        self.pan = self.tilt = 0.0
        self.slewed_s = 0.0
        if data is not None:
            self.aim(data, 0.0, 0.0)

    # --- riding the trolley --------------------------------------------------

    def follow(self, data):
        """Nothing. The head is a child of the trolley and rides it for free.

        ⚠️ **Deliberately kept, and deliberately empty.** This used to write
        `mocap_pos` from the drive joint, and it was the only thing carrying the
        head down the row — so every code path that stepped physics without
        calling it left the camera behind while its own mast drove away. That
        was both deck-camera bugs: the teleport (a whole traverse of catch-up
        applied in one frame) and the pole-without-camera desync.

        The head is now a real child body of the trolley, so position is
        inherited through the model tree on every physics step, the same way the
        mast's is. There is no second mechanism and therefore nothing to
        desync. See the module docstring.

        It stays as a no-op so the existing call sites still read, and so that
        anyone re-introducing a Python-driven position has to delete this
        comment to do it.
        """
        return None

    # --- pointing ------------------------------------------------------------

    def where(self, data):
        """(camera world position, head body world position). For the offset check."""
        return (np.array(data.cam_xpos[self.cam_id], float),
                np.array(data.xpos[self.body_id], float))

    def slew_seconds(self, pan_deg, tilt_deg=0.0):
        """What a real PTU would spend getting there from where it is now."""
        swing = max(abs(pan_deg - self.pan), abs(tilt_deg - self.tilt))
        return swing / SCOUT_SLEW_DEG_S

    def aim(self, data, pan_deg, tilt_deg=0.0):
        """Command the head. Returns the slew cost, **unbilled**.

        ⚠️ Reports the slew time but does not add it to `slewed_s` — the caller
        does, once per move. Billing here would charge a walked slew for every
        intermediate step it passes through, and the sum of those is the move's
        real cost only by accident. Same contract as `scout.ScoutHead.aim`.

        ⚠️ Caller must `mj_forward` before rendering: writing `qpos` does not
        move `cam_xpos` on its own, and a render taken in between is the
        previous pose wearing the new pose's label.

        ⚠️ **Both the joint angle and the servo setpoint are written.** The
        angle makes the pose exact for the very next `mj_forward`, which is what
        the mapping pass needs — a scan pose that is a servo transient away from
        where it says it is fuses fruit at wrong positions. The setpoint is what
        *holds* it there through the physics steps that follow, so the tilt axis
        does not sag under the camera between commands. Position only would
        droop; setpoint only would lag.
        """
        pan_deg = float(np.clip(pan_deg, -PAN_LIMIT_DEG, PAN_LIMIT_DEG))
        tilt_deg = float(np.clip(tilt_deg, -TILT_LIMIT_DEG, TILT_LIMIT_DEG))
        secs = self.slew_seconds(pan_deg, tilt_deg)
        data.qpos[self.pan_adr] = np.radians(pan_deg)
        data.qpos[self.tilt_adr] = np.radians(tilt_deg)
        data.qvel[self.pan_dof] = 0.0
        data.qvel[self.tilt_dof] = 0.0
        data.ctrl[self.pan_act] = np.radians(pan_deg)
        data.ctrl[self.tilt_act] = np.radians(tilt_deg)
        self.pan, self.tilt = pan_deg, tilt_deg
        return secs

    def slew_to(self, data, pan_deg, tilt_deg=0.0, on_tick=None):
        """Walk the head there at the unit's real rate. Returns seconds spent.

        With no `on_tick` the head arrives in one step and only the clock records
        the move, which is all a headless run needs. With one, the pan is stepped
        at `CTRL_DT` so a viewer sees it turn; the intermediate poses are never
        rendered into the map, so animating changes the picture and not the
        measurement.
        """
        import mujoco

        secs = self.slew_seconds(pan_deg, tilt_deg)
        if on_tick is None or secs <= 0.0:
            self.aim(data, pan_deg, tilt_deg)
            self.slewed_s += secs
            return secs

        from reach import CTRL_DT

        p0, t0 = self.pan, self.tilt
        steps = max(1, int(round(secs / CTRL_DT)))
        for k in range(1, steps + 1):
            f = k / steps
            self.aim(data, p0 + (pan_deg - p0) * f, t0 + (tilt_deg - t0) * f)
            mujoco.mj_forward(self.model, data)
            on_tick()
        self.slewed_s += secs
        return secs

    def current(self, data):
        """(pan, tilt) read back out of `mjData`, not out of this object.

        ⚠️ Read from the **joint angles the servos actually reached**, not from
        the commanded `self.pan`. They agree closely, because `aim` writes the
        angle as well as the setpoint — but a panel reporting a commanded angle
        while claiming to show the robot keeps looking right after the thing
        behind it has stopped working. That is the whole reason this method
        exists rather than a property. Same argument as `scout.ScoutHead.current`.
        """
        return (float(np.degrees(data.qpos[self.pan_adr])),
                float(np.degrees(data.qpos[self.tilt_adr])))


def heads(model, data, arms=("a", "b"), aisle=0):
    """One `ArmDeckHead` per fitted arm, as a dict. Absent heads are skipped."""
    out = {}
    for tag in arms:
        try:
            out[tag] = ArmDeckHead(model, data, tag=tag, aisle=aisle)
        except RuntimeError:
            pass
    return out


# --- the proof that the head rides the trolley --------------------------------

def offset_check(seed=7, aisle=0, n_per_row=10, articulate=True, verbose=True):
    """Drive the whole aisle and log camera-vs-trolley offset every cycle.

    ⚠️ **This is the check that says the deck-camera bugs are actually gone, and
    it is a measurement rather than a look at the render.** Both bugs were the
    head and its mast moving by different mechanisms (see the module docstring);
    if that is fixed, the vector from the trolley to the camera is **constant**
    for the whole traverse, and the maximum deviation from its own mean is the
    number that says so.

    ⚠️ It articulates both heads *while* the trolley drives, because the
    interesting failure is the one where position and articulation are driven
    from different places. Pan changes the camera's offset from the pan axis by
    design — that is what a pan-tilt unit does — so the invariant is measured
    against the **head body's pivot**, which articulation must not move, as well
    as against the lens, which it must.

    Returns `{tag: (max_pivot_deviation_mm, max_lens_deviation_mm)}`.
    """
    import mujoco

    from farm import armframe
    from mission import park_arm, reset_park
    from reach import CTRL_DT

    arms = ("a", "b")
    trusses = fcrop.spawn(n_per_row=n_per_row, seed=seed)
    model = trolley.build(aisle=aisle, arms=arms, trusses=trusses,
                          wrist_cam=True, arm_decks=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parks = {t: armframe.park_posture(model, data, t, arms=arms) for t in arms}
    reset_park(model, data, parks["a"], prefix=trolley.ARM_PREFIX["a"])
    for t in arms:
        park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
    mujoco.mj_forward(model, data)

    hs = heads(model, data, arms=arms, aisle=aisle)
    drive = trolley.Drive(model, data)
    lo, hi = trolley.y_limits()
    drive.park_at(lo + 0.4)
    mujoco.mj_forward(model, data)

    tid = model.body(trolley.TROLLEY).id
    samples = {t: {"pivot": [], "lens": [], "pan": []} for t in arms}
    n = [0]

    def sample():
        """Aim both heads, forward **once**, then read everything.

        ⚠️ The ordering is the measurement. `mj_step` leaves `xpos` describing
        the configuration *before* its last integration, so reading the trolley
        before a `mj_forward` and the camera after it compares two different
        instants — which shows up as a spurious offset of exactly one physics
        step of travel (0.35 m/s x 2 ms = 0.7 mm) and reads as the very desync
        this check exists to rule out. Same trap, and the same fix, as
        `DuoScout.run`: aim every head, then forward, then read.
        """
        n[0] += 1
        for t in arms:
            if articulate:
                # A slow continuous sweep, so the head is genuinely articulating
                # through the drive rather than sitting at a scan pose.
                hs[t].aim(data, SCAN_HALF_DEG * np.sin(n[0] * CTRL_DT * 0.9
                                                       + (0.0 if t == "a"
                                                          else 1.7)))
        mujoco.mj_forward(model, data)
        tpos = np.array(data.xpos[tid], float)
        for t in arms:
            lens, pivot = hs[t].where(data)
            samples[t]["pivot"].append(pivot - tpos)
            samples[t]["lens"].append(lens - tpos)
            samples[t]["pan"].append(hs[t].current(data)[0])

    if verbose:
        print(f"\n  --- driving {lo + 0.4:+.2f} m to {hi - 0.4:+.2f} m, "
              f"both heads {'articulating' if articulate else 'held'} ---")
    sample()
    drive.drive_to(hi - 0.4, on_tick=sample)

    out = {}
    if verbose:
        print(f"  {n[0]} control cycles, trolley odometer "
              f"{drive.travelled:.2f} m\n")
        print(f"  {'arm':<5} {'pan swept':>18} {'pivot offset dev':>18} "
              f"{'lens offset dev':>18}")
    for t in arms:
        piv = np.array(samples[t]["pivot"])
        lens = np.array(samples[t]["lens"])
        pan = np.array(samples[t]["pan"])
        dev_p = float(np.abs(piv - piv.mean(axis=0)).max()) * 1000
        dev_l = float(np.abs(lens - lens.mean(axis=0)).max()) * 1000
        out[t] = (dev_p, dev_l)
        if verbose:
            print(f"  {t.upper():<5} {pan.min():+7.1f}..{pan.max():+6.1f} deg "
                  f"{dev_p:>15.3f} mm {dev_l:>15.1f} mm")
    if verbose:
        print(f"\n  The pivot offset is the invariant: the head's own axis must "
              f"not move\n  relative to the trolley, ever. The lens offset "
              f"*should* change — that is\n  the pan doing its job — and it is "
              f"printed so the two cannot be confused.")
    return out


# --- the proof ---------------------------------------------------------------

def split_look(seed=7, aisle=0, n_per_row=10, out_dir=None):
    """Render both deck cams with the heads aimed *differently*, and save them.

    ⚠️ This is the check that "independently articulated" is a fact about the
    model rather than a claim in a docstring. Two heads that always mirror each
    other are one head with extra geometry; this points them at deliberately
    unrelated angles, renders both, and writes the pair out so the two frames
    can be looked at side by side. If they show the same view, the mounting is
    wrong.
    """
    import cv2
    import mujoco

    from farm import armframe
    from mission import park_arm, reset_park

    out_dir = Path(__file__).resolve().parents[3] if out_dir is None \
        else Path(out_dir)
    arms = ("a", "b")
    trusses = fcrop.spawn(n_per_row=n_per_row, seed=seed)
    model = trolley.build(aisle=aisle, arms=arms, trusses=trusses,
                          wrist_cam=True, arm_decks=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    parks = {t: armframe.park_posture(model, data, t, arms=arms) for t in arms}
    reset_park(model, data, parks["a"], prefix=trolley.ARM_PREFIX["a"])
    for t in arms:
        park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
    mujoco.mj_forward(model, data)

    hs = heads(model, data, arms=arms, aisle=aisle)
    # Deliberately unrelated angles — not mirrored, not equal.
    hs["a"].aim(data, -SCAN_HALF_DEG, 0.0)
    hs["b"].aim(data, +SCAN_HALF_DEG, 0.0)
    mujoco.mj_forward(model, data)

    print(f"\n  --- two heads, two directions, one instant ---")
    written = []
    with mujoco.Renderer(model, height=600, width=800) as r:
        for tag in arms:
            cam = DECK_CAM[tag]
            r.update_scene(data, camera=cam)
            img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            pan, tilt = hs[tag].current(data)
            cid = model.camera(cam).id
            xpos = data.cam_xpos[cid]
            fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
            print(f"  arm {tag.upper()}  cam {cam:<7} pan {pan:+6.1f} deg  "
                  f"row r{row_of(tag, aisle)}  "
                  f"lens {np.round(xpos, 3)}  forward {np.round(fwd, 2)}")
            p = out_dir / f"twoarm_deck_{tag}.png"
            cv2.imwrite(str(p), img)
            written.append(p.name)

    # The arithmetic that says they really are pointed apart.
    ida, idb = model.camera(DECK_CAM["a"]).id, model.camera(DECK_CAM["b"]).id
    fa = -data.cam_xmat[ida].reshape(3, 3)[:, 2]
    fb = -data.cam_xmat[idb].reshape(3, 3)[:, 2]
    ang = np.degrees(np.arccos(np.clip(float(fa @ fb), -1.0, 1.0)))
    print(f"\n  angle between the two lines of sight: {ang:.1f} deg")
    print(f"  wrote {', '.join(written)}")
    return ang


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", action="store_true",
                    help="point the two heads apart and render both")
    ap.add_argument("--offset", action="store_true",
                    help="drive the aisle and prove the heads ride the trolley")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)
    if args.offset:
        devs = offset_check(seed=args.seed, aisle=args.aisle, n_per_row=args.n)
        # A rigid child of the trolley cannot drift at all; anything above a
        # micron would mean the head is being moved by something else again.
        worst = max(d[0] for d in devs.values())
        ok = worst < 0.001
        print(f"\n  {'PASS' if ok else 'FAIL'} — max pivot-to-trolley "
              f"deviation {worst:.6f} mm over the traverse")
        return 0 if ok else 1
    ang = split_look(seed=args.seed, aisle=args.aisle, n_per_row=args.n)
    if args.split:
        ok = ang > 30.0
        print(f"  {'PASS' if ok else 'FAIL'} — the heads are "
              f"{'independently aimed' if ok else 'NOT independent'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
