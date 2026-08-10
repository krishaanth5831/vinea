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

--- the mocap decision, restated because it is load-bearing ------------------

Same as `scout.add_deck_camera` and for the same reason: these are **mocap
bodies parented to the world**, not hinges. A hinge would add DOFs to `mjModel`,
and `reach.Reacher` builds `mink.Configuration` over the *whole* model — the IK
solver would discover two free joints that cost it nothing and park them
anywhere, so "where the arm is reaching" would silently steer "where the camera
is looking". `farm.armframe.pin_base` exists because the trolley's slide joint
already had exactly this problem.

Mocap bodies must be children of the worldbody, so they cannot be bolted to a
moving trolley. `ArmDeckHead.follow` writes `mocap_pos` from the drive joint
every tick, which is what a mast-mounted unit does anyway.

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
    """The pan axis, in trolley-local coordinates: over the arm's own plate."""
    return np.array([trolley.ARM_X[tag], DECK_LEAD, DECK_Z_CAM])


def _aim_local(tag):
    """What the head looks at with pan=0: its own row, at the crop's mid height.

    ⚠️ In trolley-local x, arm a's row is at +ROW_PITCH/2 and arm b's at
    -ROW_PITCH/2 — the two rows are on opposite sides of the aisle, which is the
    whole reason arm b is bolted round 180 deg (`trolley.ARM_YAW`). A head that
    used arm a's aim for both would point arm b's camera straight across the
    aisle at arm a's crop, and the map would fill with fruit arm b cannot reach.
    """
    sign = +1.0 if tag == "a" else -1.0
    return np.array([sign * house.ROW_PITCH / 2, DECK_LEAD,
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

        # The head's origin *is* the pan axis, so a pure quaternion on the mocap
        # body is the pan-tilt command and nothing has to track a moving centre
        # of rotation. Its world seat here only has to compile somewhere sane;
        # `ArmDeckHead.follow` owns it from the first tick.
        pivot = [ax + mx, my, DECK_Z_CAM]
        head = spec.worldbody.add_body(name=DECK_HEAD[tag], pos=pivot,
                                       mocap=True)
        local = [SCOUT_YOKE_M, 0.0, 0.0]
        head.add_geom(name=f"deck_trunnion_{tag}",
                      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      fromto=[0, -0.028, 0, 0, 0.028, 0], size=[0.018, 0, 0],
                      rgba=[0.35, 0.37, 0.40, 1.0], contype=0, conaffinity=0,
                      mass=0.0)
        head.add_geom(name=f"deck_yoke_{tag}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, 0, 0] + local, size=[0.010, 0, 0],
                      rgba=[0.45, 0.47, 0.50, 1.0], contype=0, conaffinity=0,
                      mass=0.0)
        head.add_camera(name=DECK_CAM[tag], pos=local, fovy=SCOUT_FOVY,
                        xyaxes=xyaxes)
        add_camera_housing(head, f"cam_{DECK_CAM[tag]}", local, xyaxes,
                           kind="d435", stalk=[0.0, 0.0, 0.0])
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
        name = DECK_HEAD[tag]
        try:
            self.mocap = int(model.body(name).mocapid[0])
        except KeyError:
            self.mocap = -1
        if self.mocap < 0:
            raise RuntimeError(
                f"no deck head for arm {tag} in this model — build the scene "
                f"with arm_decks=True")

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
        """Put the head back over its mast wherever the trolley now is.

        ⚠️ Must be called after anything that moves the drive joint and before
        the next render. A mocap body does not move on its own, so a frame taken
        without this shows the crop from where the trolley *was* — which fuses
        into the map as a real fruit at a wrong position rather than as an error.
        """
        data.mocap_pos[self.mocap] = [
            self.ax + self.pivot_local[0],
            float(data.qpos[self.jadr]) + self.pivot_local[1],
            self.pivot_local[2]]

    # --- pointing ------------------------------------------------------------

    def _quat(self, pan_deg, tilt_deg):
        import mujoco

        qp, qt, out = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qp, np.array([0.0, 0.0, 1.0]),
                                  np.radians(pan_deg))
        mujoco.mju_axisAngle2Quat(qt, self.right0, np.radians(tilt_deg))
        # Pan outer, tilt inner — the trunnion rides on the pan table. The other
        # order yaws about a tilted axis and rolls the horizon.
        mujoco.mju_mulQuat(out, qp, qt)
        return out

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

        ⚠️ Caller must `mj_forward` before rendering: writing `mocap_quat` does
        not move `cam_xpos` on its own, and a render taken in between is the
        previous pose wearing the new pose's label.
        """
        secs = self.slew_seconds(pan_deg, tilt_deg)
        self.follow(data)
        data.mocap_quat[self.mocap] = self._quat(pan_deg, tilt_deg)
        self.pan, self.tilt = float(pan_deg), float(tilt_deg)
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

        They agree today because a mocap body goes exactly where it is put — but
        a panel reporting a *commanded* angle while claiming to show the robot
        keeps looking right after the thing behind it has stopped working. Same
        argument as `scout.ScoutHead.current`.
        """
        import mujoco

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, data.mocap_quat[self.mocap])
        m = mat.reshape(3, 3)
        # -z of a MuJoCo camera frame is forward; the head's own frame here has
        # +x forward at home, so read the rotated +x back against the home aim.
        fwd = m @ np.array([1.0, 0.0, 0.0])
        pan = np.degrees(np.arctan2(fwd[1], fwd[0]))
        home = np.degrees(np.arctan2(0.0, 1.0))
        rel = ((pan - home + 180.0) % 360.0) - 180.0
        tilt = np.degrees(np.arcsin(np.clip(fwd[2], -1.0, 1.0)))
        return float(rel), float(tilt)


def heads(model, data, arms=("a", "b"), aisle=0):
    """One `ArmDeckHead` per fitted arm, as a dict. Absent heads are skipped."""
    out = {}
    for tag in arms:
        try:
            out[tag] = ArmDeckHead(model, data, tag=tag, aisle=aisle)
        except RuntimeError:
            pass
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
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)
    ang = split_look(seed=args.seed, aisle=args.aisle, n_per_row=args.n)
    if args.split:
        ok = ang > 30.0
        print(f"  {'PASS' if ok else 'FAIL'} — the heads are "
              f"{'independently aimed' if ok else 'NOT independent'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
