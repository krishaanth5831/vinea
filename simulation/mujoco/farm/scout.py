#!/usr/bin/env python3
"""Drive the house, look at the crop, and come back with a map.

The harvesting pass in Weeks 1-4 was handed a row: `crop.placed` said where the
fruit were, and later the deck camera surveyed a row it was already parked in
front of. Neither answers the question a shift actually starts with — *what is
in this house, and which of it is ready?*

So this is the first pass. The trolley drives the aisle at working speed, the
deck camera looks sideways at the crop as it goes, and every frame is
deprojected, ripeness-banded and fused into one map of the house. Nothing is
picked. The map is the product, and the harvest pass plans against it.

--- what the camera can and cannot separate, measured -----------------------

⚠️ **Green fruit and green foliage are nearly the same colour, and that is not
a tuning problem.** Converting the scene's own materials to OpenCV HSV:

    fruit red         hue   2, sat 220      \\
    fruit turning     hue  14, sat 215       |  cleanly separated from the
    fruit breaker     hue  29, sat 170      /   canopy by hue alone
    ------------------------------------------------------------------
    fruit green       hue  49, sat 153      \\  6 hue from a stem, 13 from
    stem green        hue  55, sat 122       |  a leaf. Colour cannot do
    leaf green        hue  62, sat 141      /   this; shape has to.

So the detector runs two bands with different characters, and `--recall` reports
per stage rather than as one number, because one number would average a
reliable measurement together with an unreliable one and report the mean as if
it meant something.

⚠️ **This costs the harvest nothing and costs a scouting product a great deal.**
Only red is picked, and red is the band that works — a green fruit the map
misses is a fruit nobody was going to touch this pass. But Vinea's second module
is *scouting*, whose whole output is a yield forecast, and a forecast built on
"we counted the fruit we could see" is wrong in the direction that flatters it.
The number to quote there is green recall, not overall recall.

--- the head turns, and that is what makes two arms possible ------------------

⚠️ The camera used to be **bolted to the mast**, facing arm a's row. It framed
that row well — 12/14 found on seed 7 — and arm b's row sat directly behind it,
because a trolley on the aisle centreline has its two rows exactly 180° apart.
Measured on the same house, twice, with `--articulate`:

    head             r1 (arm a)   r0 (arm b)   phantom   slew
    locked at r1        12/14         0/14         1      0.0 s
    turning             12/14         8/14         1     22.5 s

So articulating it costs 22.5 s a pass and is the difference between a second
arm having a map and having nothing to pick. The pan axis is centred on the
aisle so both rows are 0.70 m away and neither arm gets the better camera.

    ./.venv/bin/python simulation/mujoco/farm/scout.py            # the pass
    ./.venv/bin/python simulation/mujoco/farm/scout.py --recall   # per stage
    ./.venv/bin/python simulation/mujoco/farm/scout.py --articulate  # the A/B
    ./.venv/bin/python simulation/mujoco/farm/scout.py --windowed # watch it
    ./.venv/bin/python simulation/mujoco/farm/scout.py --shot     # what it sees
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import house, trolley  # noqa: E402
from greenhouse import _decor  # noqa: E402

# --- the camera on the trolley -----------------------------------------------

SCOUT_CAM = "scout"

# Mast position in trolley-local coordinates, and what it looks at.
#
# ⚠️ **Ahead of the arm along the row, and square to the crop.** The first
# version put it behind the arm at x = -0.15 and 1.35 m up, aimed diagonally
# across the aisle. That render is worth keeping in mind: half the frame was
# concrete, the working row's fruit were squeezed into the top quarter, and the
# rows *behind* it — r2 and r3, metres away and nothing to do with this trolley
# — filled the skyline. One fruit was found in a frame that should have shown
# three.
#
# Two things fix it. Looking **square** at the row instead of diagonally across
# it means the whole frame is crop rather than floor. Sitting **ahead** of the
# arm along the row, rather than behind it across the aisle, keeps the arm out
# of shot entirely and means the camera surveys the section the trolley is about
# to work — which is what a scouting head on a real machine does.
SCOUT_LEAD = 0.75         # ahead of the trolley centre, along the row
SCOUT_MOUNT = np.array([0.0, SCOUT_LEAD, 1.10])
SCOUT_AIM = np.array([house.ROW_PITCH / 2, SCOUT_LEAD,
                      fcrop.fruit_z(sum(fcrop.Z_LOCAL) / 2)])
SCOUT_FOVY = 58.0         # a D435's depth vertical FoV, as in `deck_cam`

# How far the trolley moves between survey frames.
#
# ⚠️ Not free. The camera sees roughly a 1.0 m window of row at this range, so
# a stride wider than that leaves gaps the map never covers; a much narrower one
# buys overlap nobody uses and makes the pass slower for nothing. 0.50 m is
# half the window, so every fruit is seen from at least two positions — which is
# what `_fuse` needs to be able to average a position rather than take one.
SCOUT_STRIDE = 0.50

# Two detection bands. See the module docstring for the measurement.
#
#   ripe band   hue 0-40 and 165-179, the wrap-around for red. Catches red,
#               turning and breaker; the canopy starts at 49 so nothing green
#               is in here at all.
#   green band  hue 41-56. Overlaps stem (55) and nearly touches leaf (62).
#               Only circularity separates a green tomato from a leaf here.
BAND_RIPE = ((0, 100, 60), (40, 255, 255), (165, 100, 60), (179, 255, 255))
BAND_GREEN = ((41, 95, 55), (56, 255, 255))

# A leaf that happens to look round has to look *very* round to pass. The Week 3
# detector uses 0.55; the green band is held to a stricter figure because that
# is the only thing standing between it and the whole canopy.
GREEN_CIRCULARITY = 0.80
RIPE_CIRCULARITY = 0.55
MIN_AREA_PX = 200

# How close two sightings have to be to be the same fruit, in metres. Wider than
# `deck_cam.FUSE_M` because these come from *different trolley positions* rather
# than different head angles, so they carry the drive's own error as well.
FUSE_M = 0.045

# --- the articulated head ----------------------------------------------------

SCOUT_HEAD = "scout_head"

# The pan axis sits behind the lens, as on any pan-tilt unit: the camera is
# carried on a yoke and swings through an arc rather than spinning about its own
# entrance pupil. Same 100 mm as `deck_cam.DECK_YOKE_M`, and for the same reason
# — it is what makes the drawn geometry agree with the kinematics.
SCOUT_YOKE_M = 0.10

# ⚠️ **The pan axis stands on the aisle centreline, and the 100 mm of yoke is
# what the lens sits in front of.** The obvious alternative — pivot 100 mm
# behind the old bolted mount, so pan=0 reproduces the shipped lens position
# exactly — makes the head *asymmetric*: the lens ends up 0.80 m from arm a's
# row and 0.60 m from arm b's, because the yoke swings it 200 mm across the
# aisle. Measured on seed 7, that cost arm b's row 10 mapped fruit against arm
# a's 23. Centred, the lens is 0.70 m from both rows and the two arms get the
# same camera. The price is that the home lens now sits 100 mm nearer arm a's
# row than the Week 5 renders were taken at.
SCOUT_PIVOT_X = 0.0

# ⚠️ Faster than `deck_cam.DECK_SLEW_DEG_S` (60), and it has to be. That head
# nods within one row; this one turns **180°** to swap rows (see `ROW_PAN`), and
# at 60°/s that is 3.0 s of slew per stop against a 1.4 s drive between them —
# the head would become the whole cost of the pass. 120°/s is inside what the
# PTU class this represents actually does (FLIR's PTU-D48 is specified at
# 100°/s, the E46 at 300°/s), so it buys the swap back down to 1.5 s.
SCOUT_SLEW_DEG_S = 120.0

# Which way the head faces to see each arm's row, in degrees of pan from home.
#
# ⚠️ **180° apart, and that is geometry rather than a choice.** The mast stands
# on the aisle centreline; a trolley in aisle `i` has row `i+1` at local
# x = +ROW_PITCH/2 and row `i` at local x = -ROW_PITCH/2. The two rows are
# exactly opposite, so a head that serves both is a head that turns all the way
# round. This is the number that makes the second arm possible: measured on
# seed 7, the bolted camera saw **14/14** of arm a's row and **0/14** of arm
# b's, because arm b's row was directly behind it.
ROW_PAN = {"a": 0.0, "b": 180.0}


def add_deck_camera(spec, aisle=0, body_name=trolley.TROLLEY, articulated=True):
    """Put the scouting camera, its mast and its pan-tilt head into a spec.

    Call before `spec.compile()`.

    ⚠️ **The head is a mocap body parented to the world, not to the trolley,
    and both halves of that are forced.** Mocap first: hinges would add two DOFs
    to `mjModel`, and `reach.Reacher` builds `mink.Configuration` over the
    *whole* model — the solver would see two free joints, find they do nothing
    for its cost, and leave them anywhere. "Where the arm is reaching" would
    silently steer "where the camera is looking". `farm.armframe.pin_base`
    exists because the trolley's own slide joint already had this problem.

    World-parented second: MuJoCo requires mocap bodies to be children of the
    worldbody, so the head cannot simply be bolted to a moving trolley the way
    the mast is. `ScoutHead.follow` closes that gap by writing the head's
    `mocap_pos` from the drive joint every tick, which is what a mast-mounted
    unit does anyway — it rides the machine and is commanded in the machine's
    frame.

    `articulated=False` bolts the camera to the mast at the home pose, which is
    what shipped before the head existed. Kept so `--articulate` can measure the
    two against each other in one process rather than against a memory.
    """
    import mujoco

    from camera import _xyaxes_towards, add_camera_housing

    body = spec.body(body_name)
    xyaxes = _xyaxes_towards(SCOUT_MOUNT, SCOUT_AIM, up_hint=(0.0, 0.0, 1.0))
    x, y = float(SCOUT_MOUNT[0]), float(SCOUT_MOUNT[1])

    # The mast stays on the trolley — it is decor and it rides along. It stands
    # under the *pan axis*, not under the lens, so the post is not where the
    # yoke swings.
    mx = SCOUT_PIVOT_X if articulated else x
    _decor(body, "scout_mast", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[mx, y, trolley.DECK_Z, mx, y, float(SCOUT_MOUNT[2]) - 0.04],
           size=[0.022, 0, 0], rgba=[0.42, 0.44, 0.47, 1.0], mass=0.0)

    if not articulated:
        body.add_camera(name=SCOUT_CAM, pos=list(SCOUT_MOUNT), fovy=SCOUT_FOVY,
                        xyaxes=xyaxes)
        add_camera_housing(body, f"cam_{SCOUT_CAM}", SCOUT_MOUNT, xyaxes,
                           kind="d435",
                           stalk=[x, y, float(SCOUT_MOUNT[2]) - 0.04])
        return body

    # The head's origin is the pan axis at trunnion height, so a pure quaternion
    # on the mocap body *is* the pan-tilt command and nothing has to track a
    # moving centre of rotation. Its world seat is set here only so the model
    # compiles somewhere sensible; `ScoutHead.follow` owns it from then on.
    pivot = np.array([house.aisle_x(aisle) + mx, y, float(SCOUT_MOUNT[2])])
    head = spec.worldbody.add_body(name=SCOUT_HEAD, pos=pivot.tolist(),
                                   mocap=True)
    local = [SCOUT_YOKE_M, 0.0, 0.0]

    head.add_geom(name="scout_trunnion", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  fromto=[0, -0.030, 0, 0, 0.030, 0], size=[0.020, 0, 0],
                  rgba=[0.35, 0.37, 0.40, 1.0], contype=0, conaffinity=0,
                  mass=0.0)
    head.add_geom(name="scout_yoke", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[0, 0, 0] + local, size=[0.011, 0, 0],
                  rgba=[0.45, 0.47, 0.50, 1.0], contype=0, conaffinity=0,
                  mass=0.0)
    head.add_camera(name=SCOUT_CAM, pos=local, fovy=SCOUT_FOVY, xyaxes=xyaxes)
    add_camera_housing(head, f"cam_{SCOUT_CAM}", local, xyaxes, kind="d435",
                       stalk=[0.0, 0.0, 0.0])
    return body


class ScoutHead:
    """Where the scouting camera is pointing, and how to point it elsewhere.

    Pan and tilt are degrees from the home aim, exactly as in `deck_cam.DeckHead`
    — `pan=0, tilt=0` is the pose the bolted camera shipped at, so the
    articulated head and the fixed one are the same measurement at the same
    place. `ROW_PAN` names the two poses that matter.
    """

    def __init__(self, model, data=None, aisle=0, name=SCOUT_HEAD):
        from camera import _xyaxes_towards

        self.model, self.aisle = model, aisle
        try:
            self.mocap = int(model.body(name).mocapid[0])
        except KeyError:
            self.mocap = -1
        if self.mocap < 0:
            raise RuntimeError(
                "no articulated scout head in this model — build the scene "
                "with deck_cam=True, or use add_deck_camera(articulated=False) "
                "and skip the head entirely")

        self.pivot_local = np.array(
            [SCOUT_PIVOT_X, SCOUT_LEAD, float(SCOUT_MOUNT[2])])
        self.ax = house.aisle_x(aisle)
        self.jadr = model.joint(trolley.DRIVE_JOINT).qposadr[0]

        self.fwd0 = np.asarray(SCOUT_AIM, float) - np.asarray(SCOUT_MOUNT, float)
        self.fwd0 /= np.linalg.norm(self.fwd0)
        # Horizontal by construction — `_xyaxes_towards` crosses forward with
        # world up — which is what lets pan and tilt stay azimuth and elevation.
        self.right0 = np.asarray(
            _xyaxes_towards(SCOUT_MOUNT, SCOUT_AIM, up_hint=(0.0, 0.0, 1.0))[:3],
            float)
        self.pan = self.tilt = 0.0
        self.slewed_s = 0.0
        if data is not None:
            self.aim(data, 0.0, 0.0)

    # --- riding the trolley --------------------------------------------------

    def follow(self, data):
        """Put the head back on top of the mast wherever the trolley now is.

        ⚠️ Cheap, and must be called after anything that moves the drive joint
        and before the next render. A mocap body does not move on its own, so a
        frame taken without this is the crop seen from where the trolley *was* —
        which fuses into the map as a real fruit at a wrong position rather than
        as an error.
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

    def slew_seconds(self, pan_deg, tilt_deg):
        """What a real unit would spend getting there from where it is now."""
        swing = max(abs(pan_deg - self.pan), abs(tilt_deg - self.tilt))
        return swing / SCOUT_SLEW_DEG_S

    def aim(self, data, pan_deg, tilt_deg=0.0):
        """Command the head. Returns what the move would cost, unbilled.

        ⚠️ Reports the slew time but does **not** add it to `slewed_s` — the
        caller does, once per move. Billing here would charge a walked slew for
        every intermediate step it passes through, and the sum of those is the
        move's real cost only by accident.

        ⚠️ Caller must `mj_forward` before rendering — writing `mocap_quat` does
        not move `cam_xpos` on its own, and a render taken in between is the
        previous pose wearing the new pose's label.
        """
        secs = self.slew_seconds(pan_deg, tilt_deg)
        self.follow(data)
        data.mocap_quat[self.mocap] = self._quat(pan_deg, tilt_deg)
        self.pan, self.tilt = float(pan_deg), float(tilt_deg)
        return secs

    def look_at_row(self, data, tag):
        """Face the row that arm `tag` works, in one step."""
        secs = self.aim(data, ROW_PAN[tag], 0.0)
        self.slewed_s += secs
        return secs

    def slew_to_row(self, data, tag, on_tick=None):
        """Face arm `tag`'s row, *walking* there at the unit's real rate.

        Returns the seconds spent. With no `on_tick` this is `look_at_row` — the
        head arrives in one step and only the clock records the move, which is
        all a headless run needs. With one, the pan is stepped at `CTRL_DT` so a
        viewer sees the head turn; the intermediate poses are never rendered into
        the map, so animating changes the picture and not the measurement.
        """
        import mujoco

        target = ROW_PAN[tag]
        secs = self.slew_seconds(target, 0.0)
        if on_tick is None or secs <= 0.0:
            return self.look_at_row(data, tag)

        from reach import CTRL_DT

        start = self.pan
        steps = max(1, int(round(secs / CTRL_DT)))
        for k in range(1, steps + 1):
            self.aim(data, start + (target - start) * k / steps, 0.0)
            mujoco.mj_forward(self.model, data)
            on_tick()
        self.slewed_s += secs
        return secs

    def current(self, data):
        """(pan, tilt) read back out of `mjData`, not out of this object.

        They agree today because a mocap body goes exactly where it is put — but
        a panel reporting a *commanded* angle while claiming to show the robot
        keeps looking right after the thing behind it has stopped working.
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
        return float((az - az0 + 180.0) % 360.0 - 180.0), float(el - el0)


# --- the detector ------------------------------------------------------------

class StageDetector:
    """Fruit of any ripeness, with the band it was found in attached.

    Deliberately the same shape as `detect.HSVDetector` — it returns
    `detect.Detection` objects so `detect.estimate` and everything downstream
    works unchanged. What it adds is the second band and the label.
    """

    name = "stage"

    def __init__(self, min_area=MIN_AREA_PX):
        self.min_area = min_area

    def _blobs(self, hsv, mask, circ_min, label):
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        out = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        from detect import Detection

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
            if circ < circ_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection(x, y, x + w, y + h, score=float(circ),
                                 label=label, source=self.name,
                                 extra={"area": float(area),
                                        "circularity": float(circ),
                                        "band": label}))
        return out

    def __call__(self, rgb):
        import cv2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lo1, hi1, lo2, hi2 = BAND_RIPE
        ripe = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(lo1), np.array(hi1)),
            cv2.inRange(hsv, np.array(lo2), np.array(hi2)))
        glo, ghi = BAND_GREEN
        green = cv2.inRange(hsv, np.array(glo), np.array(ghi))
        return (self._blobs(hsv, ripe, RIPE_CIRCULARITY, "ripeband")
                + self._blobs(hsv, green, GREEN_CIRCULARITY, "greenband"))


def stage_of(rgb, det):
    """Which colour stage a detection is, by mean hue over its own pixels.

    ⚠️ Circular mean, not the arithmetic one. Red straddles the hue wrap at
    0/179, so a fruit whose pixels sit either side of it averages to *cyan* if
    you take the plain mean — the same trap `ripeness.classify` documents.
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    u0, v0 = int(max(0, det.u0)), int(max(0, det.v0))
    u1, v1 = int(min(rgb.shape[1], det.u1)), int(min(rgb.shape[0], det.v1))
    if u1 <= u0 or v1 <= v0:
        return None, float("nan")
    box = hsv[v0:v1, u0:u1]
    sat = box[:, :, 1] >= 90
    if not sat.any():
        return None, float("nan")
    hue = box[:, :, 0][sat].astype(float) * (2 * np.pi / 180.0)
    mean = np.arctan2(np.sin(hue).mean(), np.cos(hue).mean())
    h = float((np.degrees(mean) % 360) / 2.0)

    # Nearest stage by hue, on the same wrap-aware distance.
    best, best_d = None, 1e9
    for name, _rgba, ref, _pick in fcrop.STAGES:
        d = abs(h - ref)
        d = min(d, 180 - d)
        if d < best_d:
            best, best_d = name, d
    return best, h


# --- the map -----------------------------------------------------------------

@dataclass
class Sighting:
    """One fruit as the scouting pass has it."""

    pos: np.ndarray
    stage: str
    hue: float
    n_frames: int = 1
    row: int = -1
    truth: object = None      # the Truss, when scoring against ground truth

    @property
    def ripe(self):
        return fcrop.STAGE_BY_NAME.get(self.stage, (None, None, False))[2]

    @property
    def err_mm(self):
        if self.truth is None:
            return float("nan")
        return float(np.linalg.norm(self.pos - self.truth.pos) * 1000)


@dataclass
class HouseMap:
    """Everything the robot believes about the house after a scouting pass."""

    sightings: list = field(default_factory=list)
    aisle: int = 0
    drive_m: float = 0.0
    drive_s: float = 0.0
    frames: int = 0

    @property
    def ripe(self):
        return [s for s in self.sightings if s.ripe]

    def by_row(self, row):
        return [s for s in self.sightings if s.row == row]

    def table(self, indent="  "):
        out = [f"{indent}{len(self.sightings)} fruit mapped in "
               f"{self.frames} frames over {self.drive_m:.1f} m "
               f"({self.drive_s:.0f} s of driving)"]
        by_stage = {}
        for s in self.sightings:
            by_stage[s.stage] = by_stage.get(s.stage, 0) + 1
        out.append(f"{indent}" + " · ".join(
            f"{k} {v}" for k, v in sorted(by_stage.items())))
        out.append(f"{indent}{len(self.ripe)} ripe, which is what the harvest "
                   f"pass will go after")
        return "\n".join(out)


def _fuse(raw, radius=FUSE_M):
    """Group sightings of the same fruit seen from different trolley positions.

    Single-link clustering with full merging — the same shape as
    `deck_cam._fuse`, and merged properly for the same reason: a first-match
    loop leaves two groups unjoined when the sighting that would have linked
    them arrives last, and one fruit then appears twice on the map.
    """
    # ⚠️ Merged **by index**, never with `list.remove`. `remove` finds its target
    # with `==`, and these groups are lists of dataclasses holding numpy arrays,
    # so comparing two same-length groups evaluates an array in a boolean
    # context and raises "truth value of an array with more than one element is
    # ambiguous" — from inside the fuse, after the whole drive has been done.
    groups = []
    for s in raw:
        hit = [i for i, g in enumerate(groups)
               if any(np.linalg.norm(s.pos - e.pos) <= radius for e in g)]
        merged = [s]
        for i in hit:
            merged.extend(groups[i])
        groups = [g for i, g in enumerate(groups) if i not in set(hit)]
        groups.append(merged)

    out = []
    for g in groups:
        pos = np.mean([e.pos for e in g], axis=0)
        # The stage is the *modal* reading across frames, not the last one. A
        # fruit seen six times and banded red five times is red; taking whichever
        # frame happened to be last throws away five sixths of the evidence.
        votes = {}
        for e in g:
            votes[e.stage] = votes.get(e.stage, 0) + 1
        stage = max(votes, key=votes.get)
        out.append(Sighting(pos=pos, stage=stage,
                            hue=float(np.mean([e.hue for e in g])),
                            n_frames=len(g)))
    return out


class Scout:
    """The mapping pass. Drives, looks, fuses."""

    def __init__(self, model, data, aisle=0, detector=None, stride=SCOUT_STRIDE,
                 arms=("a",)):
        from camera import RENDER_H, RENDER_W, SensorCamera

        self.model, self.data = model, data
        self.aisle = aisle
        self.stride = stride
        self.arms = tuple(arms)
        self.detector = StageDetector() if detector is None else detector
        self.sensor = SensorCamera(model, SCOUT_CAM, RENDER_W, RENDER_H)
        self.intr = self.sensor.intr
        self.last_rgb = None
        self.head_s = 0.0
        # The head is optional so a scene built with `articulated=False` still
        # scouts — it just cannot turn round, and maps one row instead of two.
        try:
            self.head = ScoutHead(model, data, aisle=aisle)
        except RuntimeError:
            self.head = None

    def close(self):
        self.sensor.close()

    def look(self):
        """One frame. Returns raw, unfused sightings in world coordinates."""
        import mujoco

        from detect import estimate

        # ⚠️ The head rides a moving trolley and is world-parented, so its pose
        # is only correct after `follow` + `mj_forward`. Skipping this renders
        # the crop from where the trolley was one stop ago.
        if self.head is not None:
            self.head.follow(self.data)
            mujoco.mj_forward(self.model, self.data)
        rgb, depth = self.sensor.both(self.data)
        self.last_rgb = rgb
        R, C = self.sensor.pose(self.data)
        out = []
        for d in self.detector(rgb):
            e = estimate(d, depth, self.intr, R, C)
            if e.est is None:
                continue
            stage, hue = stage_of(rgb, d)
            if stage is None:
                continue
            out.append(Sighting(pos=e.est.copy(), stage=stage, hue=hue))
        return out

    def run(self, drive, on_tick=None, on_frame=None, verbose=True):
        """Drive the length of the aisle, surveying every `stride` metres.

        At every stop the head faces each arm's row in turn, so a two-armed
        trolley comes home with both of its rows mapped. The order **alternates**
        between stops: having just finished on row b, the next stop starts on row
        b rather than swinging 180° back to a and 180° out again. That halves the
        slewing over the pass and costs nothing, because which row is looked at
        first has no effect on what is seen from a stationary trolley.
        """
        lo, hi = trolley.y_limits()
        stops = list(np.arange(lo + 0.4, hi - 0.4 + 1e-9, self.stride))
        raw = []
        secs = 0.0
        head_s = 0.0
        drive.park_at(stops[0])
        import mujoco

        mujoco.mj_forward(self.model, self.data)

        for i, y in enumerate(stops):
            if i:
                secs += drive.drive_to(y, on_tick=on_tick)
            order = self.arms if i % 2 == 0 else tuple(reversed(self.arms))
            seen = []
            for tag in order:
                if self.head is not None:
                    head_s += self.head.slew_to_row(self.data, tag,
                                                    on_tick=on_tick)
                seen.extend(self.look())
                if on_frame is not None:
                    on_frame(i, len(stops), y, seen, self.last_rgb)
            raw.extend(seen)
            if verbose and (i % 4 == 0 or i == len(stops) - 1):
                print(f"    frame {i + 1:>2}/{len(stops)} at y={y:+.2f} — "
                      f"{len(seen)} blobs, {len(raw)} so far")

        self.head_s = head_s
        fused = _fuse(raw)
        # Which row each fruit is on, from its x. The map is about the house,
        # not about the camera, so a sighting that cannot be assigned to a row
        # is a sighting that is probably not a fruit.
        for s in fused:
            s.row = int(round((s.pos[0] - house.ROW_X0) / house.ROW_PITCH))
        keep = [s for s in fused
                if 0 <= s.row < house.N_ROWS
                and abs(s.pos[0] - house.row_x(s.row)) < 0.20]
        return HouseMap(sightings=keep, aisle=self.aisle,
                        drive_m=drive.travelled, drive_s=secs,
                        frames=len(stops))


def score(house_map, trusses, gate=0.12):
    """Match the map against the truth. Returns per-stage recall and errors.

    ⚠️ `gate` is 120 mm, which is generous, and deliberately so: this is asking
    "did the map find this fruit at all", not "how accurate is the position".
    Those are separate questions and the second one is the `err_mm` column. A
    tight gate would fold a position error into a recall miss and report the
    detector as blind when it was merely imprecise.
    """
    mine = [t for t in trusses]
    pairs = sorted((float(np.linalg.norm(s.pos - t.pos)), i, j)
                   for i, s in enumerate(house_map.sightings)
                   for j, t in enumerate(mine)
                   if np.linalg.norm(s.pos - t.pos) <= gate)
    taken_s, taken_t = set(), set()
    for _d, i, j in pairs:
        if i in taken_s or j in taken_t:
            continue
        taken_s.add(i)
        taken_t.add(j)
        house_map.sightings[i].truth = mine[j]

    per = {}
    for name, _rgba, _hue, _pick in fcrop.STAGES:
        want = [t for t in mine if t.stage == name]
        got = [s for s in house_map.sightings
               if s.truth is not None and s.truth.stage == name]
        right = [s for s in got if s.stage == name]
        per[name] = {"truth": len(want), "found": len(got),
                     "staged_right": len(right)}
    matched = [s for s in house_map.sightings if s.truth is not None]
    return {
        "per_stage": per,
        "found": len(matched), "truth": len(mine),
        "phantom": len(house_map.sightings) - len(matched),
        "err_mm": ([s.err_mm for s in matched] or [float("nan")]),
    }


def _articulate_ab(model, trusses, args, rows):
    """Does turning the head actually buy a row? Both arms, same crop, one run.

    The control is not "a different seed" but *the same house*, scouted twice:
    once with the head locked at arm a's row, which is what the bolted camera
    was, and once with it turning. Anything the second finds and the first does
    not is what the articulation bought.
    """
    import mujoco

    from farm import armframe
    from mission import reset_park

    print("  --- what turning the head buys, same house both times ---\n")
    print("   head            r_a found   r_b found   phantom   slew s")
    print("  " + "-" * 58)

    out = {}
    for label, arms in (("locked at r_a", ("a",)), ("turning", ("a", "b"))):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        reset_park(model, data, armframe.park_posture(model, data))
        mujoco.mj_forward(model, data)

        scout = Scout(model, data, aisle=args.aisle, stride=args.stride,
                      arms=arms)
        hm = scout.run(trolley.Drive(model, data), verbose=False)
        s = score(hm, trusses)
        got = {r: sum(1 for g in hm.sightings
                      if g.truth is not None and g.row == r)
               for r in rows.values()}
        n = {r: sum(1 for t in trusses if t.row == r) for r in rows.values()}
        print(f"  {label:<15} {got[rows['a']]:>3}/{n[rows['a']]:<7} "
              f"{got[rows['b']]:>3}/{n[rows['b']]:<8} {s['phantom']:>5}   "
              f"{scout.head_s:>6.1f}")
        out[label] = got
        scout.close()

    gain = (out["turning"][rows["b"]] - out["locked at r_a"][rows["b"]])
    print(f"\n  arm b's row r{rows['b']}: {gain:+d} fruit. A second arm has "
          f"nothing to pick without this.")
    return 0


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recall", action="store_true",
                    help="score the map against the truth, per stage")
    ap.add_argument("--shot", action="store_true",
                    help="write what the scouting camera sees")
    ap.add_argument("--windowed", action="store_true",
                    help="watch the pass in a live viewer")
    ap.add_argument("--articulate", action="store_true",
                    help="A/B the bolted camera against the pan-tilt head")
    ap.add_argument("--arms", type=int, default=2, choices=(1, 2),
                    help="rows the head scans: 1 = arm a's only, 2 = both")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=14, help="fruit per row")
    ap.add_argument("--stride", type=float, default=SCOUT_STRIDE)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    arms = ("a", "b")[: args.arms]

    os.environ.setdefault("MUJOCO_GL", "glfw" if args.windowed else "egl")
    print(__doc__)

    import mujoco

    from farm import armframe
    from mission import reset_park

    trusses = fcrop.spawn(n_per_row=args.n, seed=args.seed)
    model = trolley.build(aisle=args.aisle, arms=("a",), trusses=trusses,
                          deck_cam=True, seed=args.seed or 0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    park_q = armframe.park_posture(model, data)
    reset_park(model, data, park_q)
    mujoco.mj_forward(model, data)

    left, right = house.serves(args.aisle)
    rows = {"a": right, "b": left}
    print(f"\n  --- scouting pass, aisle a{args.aisle} ---")
    print(f"  {len(trusses)} fruit in the house, "
          f"{sum(1 for t in trusses if t.ripe)} of them ripe")
    print(f"  the head scans {', '.join(f'r{rows[t]} (arm {t})' for t in arms)}"
          f"; rows the trolley cannot see from here are not in this map\n")

    if args.articulate:
        return _articulate_ab(model, trusses, args, rows)

    scout = Scout(model, data, aisle=args.aisle, stride=args.stride, arms=arms)
    drive = trolley.Drive(model, data)

    if args.shot:
        import cv2

        drive.park_at(0.0)
        mujoco.mj_forward(model, data)
        seen = scout.look()
        out_dir = Path(__file__).resolve().parents[3]
        rgb = scout.last_rgb.copy()
        for s in seen:
            pass
        for d in scout.detector(scout.last_rgb):
            stage, _hue = stage_of(scout.last_rgb, d)
            colour = {"red": (60, 60, 220), "turning": (60, 160, 240),
                      "breaker": (60, 220, 240),
                      "green": (80, 200, 80)}.get(stage, (200, 200, 200))
            cv2.rectangle(rgb, (int(d.u0), int(d.v0)), (int(d.u1), int(d.v1)),
                          colour, 2)
            cv2.putText(rgb, stage or "?", (int(d.u0), int(d.v0) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        p = out_dir / "farm_scout_view.png"
        cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  wrote {p.name} — {len(seen)} fruit banded in one frame")
        scout.close()
        return 0

    tick = None
    viewer = None
    if args.windowed:
        import time

        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data).__enter__()
        clock = [time.perf_counter()]

        def tick():
            from reach import CTRL_DT

            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            clock[0] += CTRL_DT
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

    try:
        house_map = scout.run(drive, on_tick=tick)
    except KeyboardInterrupt:
        print("\n  window closed mid-pass")
        scout.close()
        return 1

    print(f"\n{house_map.table()}")

    if args.recall:
        # ⚠️ Scored against the **whole house**, not against the worked row.
        # The camera can see r2 and r3 over the top of r1, and it maps them —
        # correctly, they are fruit. Scoring against r1 alone reported fifteen
        # perfectly good sightings as phantoms and made a working detector look
        # like it was hallucinating half a row.
        s = score(house_map, trusses)
        print(f"\n  --- scored against the whole house ---")
        print(f"  {'stage':<10} {'in house':>9} {'mapped':>7} {'recall':>8} "
              f"{'banded right':>13}")
        for name, _rgba, _hue, pick in fcrop.STAGES:
            p = s["per_stage"][name]
            rec = 100 * p["found"] / p["truth"] if p["truth"] else float("nan")
            band = (100 * p["staged_right"] / p["found"]
                    if p["found"] else float("nan"))
            print(f"  {name:<10} {p['truth']:>9} {p['found']:>7} "
                  f"{rec:>7.0f}% {band:>12.0f}%"
                  + ("   <- the one that gets picked" if pick else ""))
        errs = np.array(s["err_mm"])
        print(f"\n  overall {s['found']}/{s['truth']} "
              f"({100 * s['found'] / max(1, s['truth']):.0f}%) · "
              f"{s['phantom']} unmatched · position error "
              f"mean {np.nanmean(errs):.0f} mm, max {np.nanmax(errs):.0f} mm")

        print(f"\n  --- by row: distance is what decides ---")
        print(f"  {'row':<5} {'x':>6} {'in house':>9} {'mapped':>7} "
              f"{'err mm':>8}  ")
        for r in range(house.N_ROWS):
            want = [t for t in trusses if t.row == r]
            got = [x for x in house_map.sightings
                   if x.row == r and x.truth is not None]
            e = [x.err_mm for x in got] or [float("nan")]
            tag = ("   <- worked by arm A" if r == right else
                   "   <- arm B's row" if r == left else
                   "   <- seen, not reachable from this aisle")
            print(f"  r{r:<4} {house.row_x(r):>6.2f} {len(want):>9} "
                  f"{len(got):>7} {np.nanmean(e):>8.0f}{tag}")

        print(f"\n  ⚠️ Read the rows, not the total. Red is what the harvest")
        print(f"  pass acts on; green shares a hue with the canopy and is the")
        print(f"  number a scouting product would live or die on. And a fruit")
        print(f"  mapped on r2 from a trolley in a0 is real but useless — it is")
        print(f"  three metres away through a crop wall, and the position error")
        print(f"  column shows what that costs.")

    scout.close()
    if viewer is not None:
        print("\n  pass finished — close the window to quit")
        try:
            import time

            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
