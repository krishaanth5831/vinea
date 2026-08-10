#!/usr/bin/env python3
"""A camera that exists to be a sensor, and the maths that turns it into metres.

Week 1 and 2 gave the arm exact fruit coordinates because the simulator knows
them. A real machine does not get that, so this file is where the fruit stops
being a number and starts being pixels.

Three separable pieces, deliberately kept apart because mixing them is how a
perception bug becomes undebuggable:

    1. **Where the camera is.**  `add_wrist_camera` bolts one to wrist3_link,
       eye-in-hand, beside the gripper. The mount offset is the only part of the
       extrinsics that is a *choice* rather than something the simulator hands
       over, so it is a named constant with a comment, not a literal buried in a
       call.
    2. **What the camera is.**  `Intrinsics` — f, cx, cy — derived from `fovy`
       and the render size. MuJoCo cameras carry exactly one number and
       everything else follows.
    3. **How a pixel becomes a point.**  `project` / `deproject`, thirty lines
       of sign conventions, and the only part of the week that is genuinely
       first-principles.

⚠️ **The ordering is not an accident.** Piece 3 is validated against fruit whose
positions are already known, with no detector anywhere near it — see `main`. A
detector layered on top of a broken deprojection is wrong in two ways at once
and the two errors are not separable after the fact.

Run it directly for the Step 3 gate:

    ./.venv/bin/python simulation/mujoco/camera.py            # all cameras
    ./.venv/bin/python simulation/mujoco/camera.py --camera wrist
"""

import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fr5 import FLANGE_Z, PINCH_Z  # noqa: E402
from plant_row import FRUIT_R  # noqa: E402

# --- the mount ---------------------------------------------------------------
# Eye-in-hand, on wrist3_link, offset to the side of the gripper.
#
# Why eye-in-hand rather than a fixed mast: the planner already retracts to a
# staging plane before every pick (`mission.STAGE_X`), and a staging pose is
# exactly where a square-on frame of the fruit about to be picked is available
# for free. One render per pick, at a known repeatable pose. A row-facing mast
# is cheaper per row but sees fruit at grazing angles and its occlusion gets
# worse the further down the row it looks.
#
# ⚠️ These three numbers are the only extrinsics that are *ours*. Everything
# else — cam_xpos, cam_xmat — MuJoCo computes from the kinematics to machine
# precision, which is a luxury that ends the moment this runs on hardware. A
# real hand-eye calibration is a millimetre-level problem all by itself.
#
# ⚠️ **Offset in y, not x, and that is the whole trick.** The 2F85's envelope
# measured in this frame, over the wrist3 subtree:
#
#       x  -0.099 .. +0.099     the fingers open and close along x
#       y  -0.079 .. +0.079     bounding *spheres*; the pads themselves are ±15 mm
#       z   0.014 .. +0.266     flange face at 0.100, pinch point at 0.256
#
# So x is the gripper's wide axis and y is its narrow one. The first version of
# this mount sat at x = -0.075 and it failed the Step 3 gate: the sight line to
# t2 ran straight through `gr_base` and the depth buffer returned 0.15 m for a
# fruit 0.55 m away. Deprojected, that is a tomato 340 mm from where it is —
# and the pipeline reports it as a confident detection, because a depth read is
# a depth read. This is exactly the "looking at the back of its own gripper"
# trap, and it is only visible if the estimate is scored against ground truth.
#
# Coordinates are wrist3_link local, where +z is the tool axis:
#   x =  0.000   on the finger axis, so opening and closing the gripper never
#                walks a finger across the frame.
#   y = -0.090   the narrow side, 90 mm out. Measured: at 80 mm the sightlines
#                are still clear, at 90 mm there is margin, and the win is worth
#                nothing on x where 130 mm was still occluded.
#   z =  0.130   30 mm *past* the flange face, level with the drivers. At 0.100
#                the base comes back into the sight line (measured: one occluded
#                fruit returns); at 0.160 the frame starts losing fruit off the
#                edges without buying any more clearance.
CAM_MOUNT = np.array([0.0, -0.090, 0.130])

# Tilted to converge slightly on the tool axis rather than looking dead ahead,
# so the fruit about to be picked lands near the middle of the frame instead of
# 20° off to one side — which matters because off-axis is where z-depth and
# range diverge, and where a real lens has its worst distortion.
#
# It converges to y = -0.02 rather than y = 0. Aiming all the way onto the axis
# costs a fruit off the frame edge and buys nothing measurable (11 sightlines vs
# 12); stopping 20 mm short keeps the sight line leaning away from the gripper.
#
# The aim distance is one approach gap beyond the pinch point: at the staging
# plane the tool is mission.STAGE_X back from the row, i.e. 280 mm from the
# fruit, and that is the pose the detector actually fires at.
CAM_AIM = np.array([0.0, -0.02, PINCH_Z + 0.32])

CAM_NAME = "wrist"

# --- render size -------------------------------------------------------------
# Intrinsics are meaningless without one. 1280x960 is 4:3, which matches the
# fovy convention (vertical FoV) without an aspect surprise, and it is what
# every number quoted in this file and in the Week 3 doc was measured at.
RENDER_W, RENDER_H = 1280, 960

# ⚠️ The depth buffer fills unoccupied pixels with the far plane, and on this
# scene that reads 685.79 m — not `inf`, not NaN, a perfectly plausible float.
# Deproject one and you get a tomato most of a kilometre outside the greenhouse
# and nothing crashes. Anything past this is sky; the working volume is ~2 m.
DEPTH_MAX = 3.0


def _xyaxes_towards(origin, target, up_hint=(1.0, 0.0, 0.0)):
    """`xyaxes` for a camera at `origin` looking at `target`, in the same frame.

    MuJoCo declares a camera by its x (right) and y (up) axes and derives the
    third; the camera looks down its own **-z**. `greenhouse._look_at` does the
    same job in world coordinates with world-up as the hint, which is wrong for
    a camera bolted to a wrist — the wrist rolls, and world-up is not a fixed
    direction in its frame. So this one takes the hint explicitly.
    """
    fwd = np.asarray(target, float) - np.asarray(origin, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up_hint, float))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return [*right, *up]


# --- what the sensor looks like ----------------------------------------------
#
# A MuJoCo camera is a coordinate frame and nothing else. It is invisible, and
# for three weeks that was fine because nothing depended on seeing it. It
# stopped being fine the moment there were two: "where is the deck camera
# looking from" is the first question anyone asks of a survey, and the answer
# was three numbers in a source file. A camera you cannot see is a camera nobody
# can check.
#
# There is no RGBD mesh anywhere in `third_party` — Menagerie ships the arm and
# the gripper, not sensors — so the housings are built from primitives at the
# real products' dimensions. That is the better trade here rather than a
# concession: a mesh would be an opaque blob, and these numbers are the ones
# that decide whether the sensor fits, which is exactly the sort of thing this
# repo writes down. Drop a real .obj in later and only `_housing` changes.
#
# **Two different cameras, because they do two different jobs.**
#
#   wrist   D405-class, 42 x 42 x 23 mm, usable from ~70 mm to ~500 mm.
#           It works at the staging distance, 280 mm (`mission.STAGE_X`), which
#           is inside a D435's blind spot. Short range is why it is the one that
#           inspects.
#   deck    D435-class, 90 x 25 x 25 mm, usable from ~300 mm to ~3 m.
#           It works at 1.3 m, which is what surveying a whole row from the
#           chassis costs. See `deck_cam.DECK_MOUNT`.
#
# The 90 mm long axis is also why the deck camera is the one on a post and the
# wrist one is not: 90 mm bolted to wrist3_link would stand further out than the
# gripper does.
HOUSINGS = {
    #          length   height   depth    baseline   lens r
    "d435":  (0.090,   0.025,   0.025,   0.050,     0.0055),
    "d405":  (0.042,   0.042,   0.023,   0.018,     0.0050),
}

CASE_GREY = [0.16, 0.17, 0.19, 1.0]
LENS_BLACK = [0.03, 0.03, 0.035, 1.0]
LENS_RGB = [0.06, 0.10, 0.20, 1.0]      # the colour imager reads blue-black
MOUNT_STEEL = [0.45, 0.47, 0.50, 1.0]


def _quat_from_xyaxes(xyaxes):
    """`xyaxes` (6 numbers) -> a geom `quat` in the same frame.

    A camera is declared by its x and y axes; a geom is declared by a
    quaternion. The housing has to sit in the camera's frame — front face on
    -z, where the lens looks — so the two conventions have to be bridged here
    rather than by hand-writing a quaternion per mount.

    ⚠️ The matrix is built **columns-first**: R maps camera-local vectors into
    the parent frame, so its columns are the camera's axes expressed in the
    parent. Building it row-wise instead gives R.T, which is a housing rotated
    to a plausible-looking wrong orientation with nothing to flag it.
    """
    import mujoco

    x = np.asarray(xyaxes[:3], float)
    y = np.asarray(xyaxes[3:], float)
    x = x / np.linalg.norm(x)
    y = y - x * float(x @ y)
    y /= np.linalg.norm(y)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, np.cross(x, y)]).flatten())
    return quat.tolist()


def add_camera_housing(body, name, pos, xyaxes, kind="d435", stalk=None):
    """Draw the sensor at `pos`/`xyaxes`, in whatever frame those are given in.

    Call with the *same* pos and xyaxes the camera itself was added with, so the
    drawing and the frame the maths uses cannot drift apart. Returns nothing —
    the geoms go straight onto `body`.

    `stalk` is an optional point in the same frame to run a mounting stub back
    to, which is what stops the deck camera reading as a floating brick.

    ⚠️ **contype=0, conaffinity=0 and mass=0 on every geom, and all three
    matter.**

      * contype/conaffinity: the wrist housing hangs 90 mm off wrist3_link,
        which is precisely where `mission.ClearanceModel` measures gripper-to-
        fruit distance. A collidable box there would move CLEARANCE, the guard
        stop, and every clearance figure in the README — and it would do it
        silently, because they are all measured rather than asserted.
      * mass: the FR5's links carry explicit <inertial> from the URDF, so
        MuJoCo ignores geom mass on those. Bodies created by this repo's own
        spec code do not. Relying on remembering which is which is how a 40 g
        camera quietly joins the payload the joint torque limits were checked
        against. `deck_cam.main` asserts the wrist link's mass and inertia are
        unchanged by this, rather than trusting the argument.
    """
    import mujoco

    length, height, depth, baseline, lens_r = HOUSINGS[kind]
    quat = _quat_from_xyaxes(xyaxes)
    pos = np.asarray(pos, float)

    def local(p):
        """Camera-frame offset -> the frame `pos` is expressed in."""
        R = np.array(_quat_to_mat(quat))
        return (pos + R @ np.asarray(p, float)).tolist()

    def add(gname, gtype, offset, **kw):
        kw.setdefault("contype", 0)
        kw.setdefault("conaffinity", 0)
        kw.setdefault("mass", 0.0)
        return body.add_geom(name=gname, type=gtype, pos=local(offset),
                             quat=quat, **kw)

    # ⚠️ **Every geom here sits strictly behind the pinhole**, i.e. at
    # positive z in camera coordinates, with `BEHIND_APERTURE` of daylight.
    # That is not cosmetic tidiness — it is what keeps `occluder` working.
    #
    # `occluder` casts a ray from the camera's own position. The first version
    # of this centred the lens barrels on z=0 so they straddled the aperture,
    # which put the ray's *origin* inside a geom: `mj_ray` duly returned a hit
    # at ~0 m and every fruit in the Step 3 calibration report came back
    # "occluded by wrist3_link". The gate still passed — the depth render is
    # unaffected, because the near plane clips anything that close — so the only
    # symptom was an occlusion column that had quietly become all-ones. An
    # occlusion test that always says "occluded" is worse than none, because
    # `camera.CAM_MOUNT` was chosen by reading exactly that column.
    #
    # Half a millimetre is invisible at any framing and is also the truthful
    # arrangement: a lens element sits behind its aperture, not in front of it.
    BEHIND_APERTURE = 0.0005
    lens_half = 0.0035

    # The case, its front face level with the aperture.
    add(f"{name}_case", mujoco.mjtGeom.mjGEOM_BOX,
        [0, 0, BEHIND_APERTURE + depth / 2],
        size=[length / 2, height / 2, depth / 2], rgba=CASE_GREY)

    # Imagers just inside the front face, looking down -z with the camera. Left
    # and right IR at the stereo baseline, the projector between them, the
    # colour imager outboard — a D4xx's actual layout, left to right.
    lenses = [(-baseline / 2, lens_r, LENS_BLACK, "ir_l"),
              (0.0, lens_r * 0.7, LENS_BLACK, "proj"),
              (baseline / 2, lens_r, LENS_BLACK, "ir_r"),
              (baseline / 2 + lens_r * 2.4, lens_r * 0.9, LENS_RGB, "rgb")]
    for dx, r, rgba, tag in lenses:
        if abs(dx) + r > length / 2:
            continue        # the colour imager does not fit on the short case
        add(f"{name}_{tag}", mujoco.mjtGeom.mjGEOM_CYLINDER,
            [dx, 0, BEHIND_APERTURE + lens_half],
            size=[r, lens_half, 0], rgba=rgba)

    if stalk is not None:
        end = np.asarray(stalk, float)
        back = np.asarray(local([0, 0, depth]), float)
        body.add_geom(
            name=f"{name}_stub", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*back, *end], size=[0.010, 0, 0], rgba=MOUNT_STEEL,
            contype=0, conaffinity=0, mass=0.0)


def _quat_to_mat(quat):
    import mujoco

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    return mat.reshape(3, 3)


def add_wrist_camera(spec, name=CAM_NAME, mount=CAM_MOUNT, aim=CAM_AIM,
                     fovy=45.0, housing=True, prefix=""):
    """Bolt the sensor camera to wrist3_link. Call before `spec.compile()`.

    `fovy` matches the three cinematic cameras so the intrinsics story is one
    story. A real sensor would be chosen for its FoV and this is where that
    choice would live.

    `housing` draws the sensor as well as declaring it. It is on by default
    because an invisible camera is one nobody can check, and it is a flag at all
    because it is drawing — see `add_camera_housing` for why it changes no
    physics either way.

    ⚠️ `prefix` picks which arm's wrist to bolt it to, and the camera is named
    for that arm too — `wrist` for the first, `b_wrist` for the second. Without
    it, `spec.body("wrist3_link")` resolves to the *first* arm on a two-armed
    machine and the second arm has no eye at all, while a caller asking for the
    `wrist` camera gets a plausible picture from the wrong arm.
    """
    name = prefix + name
    wrist = spec.body(prefix + "wrist3_link")
    xyaxes = _xyaxes_towards(mount, aim)
    wrist.add_camera(name=name, pos=list(mount), fovy=fovy, xyaxes=xyaxes)
    if housing:
        # D405-class: short range, and small enough to sit inside the gripper's
        # own width. See HOUSINGS.
        add_camera_housing(wrist, f"cam_{name}", mount, xyaxes, kind="d405",
                           stalk=[0.0, -0.030, FLANGE_Z + 0.02])
    return spec


# --- what the camera is ------------------------------------------------------
@dataclass(frozen=True)
class Intrinsics:
    """A pinhole, and in this simulator that is not an approximation.

    Square pixels, no distortion, principal point exactly centred. A real camera
    has none of those properties and saying so out loud here is cheaper than
    discovering it in the hardware phase.
    """

    f: float        # focal length in pixels — identical in x and y
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_model(cls, model, camera, width=RENDER_W, height=RENDER_H):
        """The whole derivation. MuJoCo cameras carry one number: fovy.

        fovy is the **vertical** field of view in degrees, so the height is what
        sets the scale and the width only decides how much of the world falls
        off the sides. Getting that backwards gives an f that is right at 4:3
        and quietly wrong at every other aspect ratio.
        """
        cid = camera if isinstance(camera, int) else model.camera(camera).id
        fovy = float(model.cam_fovy[cid])
        f = 0.5 * height / np.tan(np.radians(fovy) / 2)
        return cls(f=f, cx=width / 2, cy=height / 2, width=width, height=height)

    def __str__(self):
        return (f"f={self.f:.2f} px  cx={self.cx:.1f}  cy={self.cy:.1f}  "
                f"@ {self.width}x{self.height}")


def camera_pose(data, model, camera):
    """(R, C): camera->world rotation and the camera's world position.

    `cam_xmat` is row-major and maps **camera axes into world**, so the world->
    camera direction is its transpose. Getting that inverted is silent: the
    result is still a 3-vector of plausible metres.
    """
    cid = camera if isinstance(camera, int) else model.camera(camera).id
    return data.cam_xmat[cid].reshape(3, 3).copy(), data.cam_xpos[cid].copy()


# --- how a pixel becomes a point ---------------------------------------------
# Two conventions govern everything below and both are silent when wrong:
#
#   * A MuJoCo camera looks down its own **-z**, with +x right and +y up. That
#     is why the ray's third component is -1 and why the projection divides by
#     `-p_cam[2]` rather than `p_cam[2]`. A point in front of the lens has a
#     *negative* z in camera coordinates.
#   * Image rows count **downward** from the top, camera +y points **up**. That
#     is the minus sign on every `(v - cy)`.
#
# Get either backwards and you still get coordinates, still in metres, still
# roughly the right magnitude — mirrored about the image centre. There is no
# exception and no NaN to catch it.

def project(intr, R, C, p_world):
    """World point -> (u, v) pixel. The forward direction, used to make truth.

    This is also how Step 4's ground-truth boxes are generated: the simulator
    knows where every fruit is, so labelling a rendered frame is free and exact.
    """
    p_cam = R.T @ (np.asarray(p_world, float) - C)
    u = intr.cx + intr.f * p_cam[0] / -p_cam[2]
    v = intr.cy - intr.f * p_cam[1] / -p_cam[2]
    return np.array([u, v])


def pixel_ray(intr, R, u, v):
    """Unit ray through pixel (u, v), in **world** coordinates."""
    ray = np.array([(u - intr.cx) / intr.f,
                    -(v - intr.cy) / intr.f,
                    -1.0])
    ray /= np.linalg.norm(ray)
    return R @ ray


def deproject(intr, R, C, u, v, z_depth, radius=FRUIT_R):
    """(u, v) + depth -> world point, corrected to the centre of the fruit.

    ⚠️ **`z_depth` is not a range.** MuJoCo's depth buffer holds the distance
    along the camera's *optical axis*, not the distance from the lens to the
    point. On-axis those are the same number and everything appears to work;
    off-axis they diverge by 1/cos(theta) and the error grows with distance from
    the image centre. Measured on the `row` camera: t0 sits 1.89418 m along the
    ray and the buffer reads 1.84560. Treated as a range that is 16 mm of pure
    frame-convention error, in a direction that looks like a bad calibration.
    The division by |ray_z| below is the conversion.

    ⚠️ **The depth buffer returns the front surface, and the target is the
    centre.** A sphere of radius FRUIT_R shows its nearest point to the lens, so
    every deprojected position lands one radius short — 33 mm, identically, on
    every fruit at every pose. A constant offset on every fruit looks exactly
    like a wrong fovy or a frame-sign error, which is three wrong places to go
    looking before arriving here. Stepping along the ray by `radius` after
    deprojecting is the whole fix.

    Pass `radius=0.0` to get the raw surface hit, which is what the calibration
    report needs in order to show the correction doing its work.

    ⚠️ **This correction is a simulator luxury and does not survive contact with
    a real detector.** It assumes a full sphere of known radius seen head-on. A
    real camera sees a partial, occluded, non-spherical surface whose visible
    centroid is not the fruit centre, and the radius is not known in advance —
    it has to be fitted from the box width or taken as a percentile over the
    box's depth pixels. `detect.estimate` does the box-width version and says so.
    """
    ray = np.array([(u - intr.cx) / intr.f,
                    -(v - intr.cy) / intr.f,
                    -1.0])
    ray /= np.linalg.norm(ray)
    # z_depth / |ray_z| converts the axis-aligned depth into a range along the
    # ray; + radius steps from the front surface through to the centre.
    p_cam = ray * (z_depth / abs(ray[2]) + radius)
    return C + R @ p_cam


def occluder(model, data, C, p_world, expect):
    """Name of the body between the lens and `p_world`, or None if it is clear.

    A raycast, not a depth comparison. The depth-buffer version of this test —
    "is the pixel much nearer than the fruit's own front surface" — needs a
    tolerance, and the tolerance is exactly the quantity being measured. The ray
    just says what it hit.

    ⚠️ **The only occluder on this scene is the robot's own gripper**, and that
    is a property of the scene rather than a result. `greenhouse.LEAF_FROM_Z`
    is 0.86: the bottom of a high-wire plant is deleafed to the ripening truss,
    which is horticulturally correct, and the fruit hang at z 0.54-0.74. Every
    leaf in this greenhouse is above every tomato. Occlusion — the single
    hardest problem in greenhouse perception — is currently zero here, and no
    detection rate measured on this scene should be quoted without that caveat.

    Self-occlusion by the gripper is real and it is what forced CAM_MOUNT off
    the x axis. Finding it took a ground-truth comparison, which is the week's
    one rule doing its job.
    """
    import mujoco

    v = np.asarray(p_world, float) - C
    dist = np.linalg.norm(v)
    gid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, C, v / dist, None, 1, -1, gid)
    if gid[0] < 0:
        return None
    name = model.body(model.geom_bodyid[gid[0]]).name
    if name == expect or hit >= dist - 1e-6:
        return None
    return name


# --- the renderer ------------------------------------------------------------
class SensorCamera:
    """RGB and depth from one named camera, with the traps already handled.

    Kept as an object because a `mujoco.Renderer` owns a GL context and building
    one per frame is both slow and a good way to exhaust them.
    """

    def __init__(self, model, camera=CAM_NAME, width=RENDER_W, height=RENDER_H):
        import mujoco

        self.model = model
        self.name = camera
        self.cid = model.camera(camera).id
        self.intr = Intrinsics.from_model(model, self.cid, width, height)
        self._rgb = mujoco.Renderer(model, height=height, width=width)
        self._depth = mujoco.Renderer(model, height=height, width=width)
        self._depth.enable_depth_rendering()

    def rgb(self, data):
        self._rgb.update_scene(data, camera=self.cid)
        return self._rgb.render()

    def depth(self, data):
        """float32, metres, **z-depth along the optical axis**. See `deproject`."""
        self._depth.update_scene(data, camera=self.cid)
        return self._depth.render()

    def both(self, data):
        return self.rgb(data), self.depth(data)

    def pose(self, data):
        return camera_pose(data, self.model, self.cid)

    def valid(self, depth):
        """Mask of pixels that are actually something, not the far plane.

        The working volume is under 2 m and the far plane reads 685.79 m, so the
        threshold is not delicate. What matters is that it exists at all.
        """
        return depth < DEPTH_MAX

    def close(self):
        # ⚠️ Headless depth rendering raises EGLError from Renderer.__del__ at
        # process exit on this machine — after the render has completed, with
        # nothing lost. Same family as the GLFW teardown segfault already logged
        # "upstream, not ours" in the bug log. Closing explicitly here keeps it
        # out of the common path; if it still fires at exit, ignore it.
        for r in (self._rgb, self._depth):
            try:
                r.close()
            except Exception:
                pass


# --- Step 3: the gate --------------------------------------------------------
def calibration_error(model, data, camera, intr=None, radius=FRUIT_R,
                      names=None, sensor=None):
    """Project every fruit into the image, deproject it back, score in mm.

    No detector. The pixel is computed from the fruit's own known position, so
    the only thing under test is the geometry — which is the entire point of
    doing this before Step 4 rather than after.

    Returns a list of per-fruit dicts. `err_mm` is the number the 1 mm gate is
    read off, `raw_mm` is the same measurement with the surface-to-centre
    correction switched off, which should land at ~33 mm on everything.
    """
    from plant_row import TRUSSES

    names = [n for n, _, _ in TRUSSES] if names is None else names
    own = sensor is None
    if own:
        sensor = SensorCamera(model, camera, intr.width if intr else RENDER_W,
                              intr.height if intr else RENDER_H)
    intr = sensor.intr if intr is None else intr

    try:
        depth = sensor.depth(data)
        R, C = sensor.pose(data)
        out = []
        for n in names:
            truth = data.body(n).xpos.copy()
            u, v = project(intr, R, C, truth)
            row = {"fruit": n, "truth": truth, "u": u, "v": v}

            # Off the sensor entirely — a real and reportable outcome, not an
            # error. An eye-in-hand camera at a staging pose does not have the
            # whole row in frame and pretending otherwise inflates the score.
            iu, iv = int(round(u)), int(round(v))
            if not (0 <= iu < intr.width and 0 <= iv < intr.height):
                row.update(visible=False, why="outside the frame")
                out.append(row)
                continue

            z = float(depth[iv, iu])
            if z >= DEPTH_MAX:
                row.update(visible=False, why=f"background pixel ({z:.1f} m)")
                out.append(row)
                continue

            range_true = np.linalg.norm(truth - C)
            blocker = occluder(model, data, C, truth, n)

            est = deproject(intr, R, C, u, v, z, radius=radius)
            raw = deproject(intr, R, C, u, v, z, radius=0.0)
            row.update(
                visible=True, occluded=blocker is not None, blocker=blocker,
                z=z, range_true=float(range_true),
                est=est, err=est - truth,
                err_mm=float(np.linalg.norm(est - truth) * 1000),
                raw_mm=float(np.linalg.norm(raw - truth) * 1000),
            )
            out.append(row)
        return out
    finally:
        if own:
            sensor.close()


def _report(title, rows, gate_mm=1.0):
    """Print one pose's calibration table. Returns True if the gate held."""
    print(f"\n  {title}")
    print(f"  {'fruit':>6} {'u':>7} {'v':>7} {'z m':>7} {'range m':>8} "
          f"{'naive mm':>9} {'corrected mm':>13}   note")
    ok, seen = True, 0
    for r in rows:
        if not r["visible"]:
            print(f"  {r['fruit']:>6} {r['u']:7.1f} {r['v']:7.1f} "
                  f"{'-':>7} {'-':>8} {'-':>9} {'-':>13}   {r['why']}")
            continue
        seen += 1
        flag = "" if r["err_mm"] < gate_mm else "  <-- OVER GATE"
        if r["occluded"]:
            flag += f"  occluded by {r['blocker']}"
        ok = ok and r["err_mm"] < gate_mm
        print(f"  {r['fruit']:>6} {r['u']:7.1f} {r['v']:7.1f} {r['z']:7.4f} "
              f"{r['range_true']:8.4f} {r['raw_mm']:9.1f} {r['err_mm']:13.2f}"
              f"{flag}")
    if not seen:
        print("    nothing visible from this pose — not a pass")
        return False
    return ok


def main():
    """Step 3's gate, from more than one pose, before any detector exists."""
    import argparse

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from greenhouse import build_scene
    from mission import PARK, STAGE_X, park_posture, reset_park
    from plant_row import Row

    ap = argparse.ArgumentParser(
        description="Deproject known pixels and score against ground truth.")
    ap.add_argument("--camera", default=None,
                    help="only this camera (default: wrist + row)")
    ap.add_argument("--gate", type=float, default=1.0, help="mm")
    ap.add_argument("--width", type=int, default=RENDER_W)
    ap.add_argument("--height", type=int, default=RENDER_H)
    args = ap.parse_args()

    model = build_scene(wrist_cam=True)
    data = mujoco.MjData(model)
    q = park_posture(model)
    reset_park(model, data, q)
    row = Row(model, data)
    mujoco.mj_forward(model, data)

    print(f"\n  scene: {model.ngeom} geoms, {model.ncam} cameras "
          f"({', '.join(model.camera(i).name for i in range(model.ncam))})")
    print(f"  fruit radius {FRUIT_R * 1000:.0f} mm · "
          f"far-plane mask at {DEPTH_MAX:.1f} m")

    cams = [args.camera] if args.camera else [CAM_NAME, "row"]
    passed = True

    for cam in cams:
        intr = Intrinsics.from_model(model, cam, args.width, args.height)
        print(f"\n{'=' * 78}\n  camera '{cam}' · fovy "
              f"{model.cam_fovy[model.camera(cam).id]:.1f} deg · {intr}")

        sensor = SensorCamera(model, cam, args.width, args.height)
        try:
            # Pose 1: parked. Where every trial starts.
            reset_park(model, data, q)
            row.reset()
            mujoco.mj_forward(model, data)
            depth = sensor.depth(data)
            print(f"    background pixel reads {depth.max():.2f} m "
                  f"({100 * (depth >= DEPTH_MAX).mean():.0f}% of the frame)")
            passed &= _report(f"pose 1 — parked, tool0 at "
                              f"{data.site('tool0').xpos.round(3)}",
                              calibration_error(model, data, cam, intr,
                                                sensor=sensor), args.gate)

            # Pose 2: at a staging pose in front of the row. A second pose is
            # the whole reason this is a gate and not a coincidence — one pose
            # can be right by accident when two sign errors cancel.
            stage(model, data, q, np.array([STAGE_X, 0.10, 0.66]), row)
            passed &= _report(f"pose 2 — staged, tool0 at "
                              f"{data.site('tool0').xpos.round(3)}",
                              calibration_error(model, data, cam, intr,
                                                sensor=sensor), args.gate)

            # Pose 3: staged lower and to the other end of the row, so the
            # fruit land in a different part of the image. Off-axis is where
            # z-depth-treated-as-range goes wrong, so it has to be sampled.
            stage(model, data, q, np.array([STAGE_X, -0.18, 0.58]), row)
            passed &= _report(f"pose 3 — staged low, tool0 at "
                              f"{data.site('tool0').xpos.round(3)}",
                              calibration_error(model, data, cam, intr,
                                                sensor=sensor), args.gate)
        finally:
            sensor.close()

    print(f"\n{'=' * 78}")
    print(f"  STEP 3 GATE: {'PASS' if passed else 'FAIL'} "
          f"(every visible fruit within {args.gate:.1f} mm, three poses)")
    if not passed:
        print("  Stop here. Everything downstream inherits this error.")
    print(f"  PARK is {PARK.round(3)}\n")
    return 0 if passed else 1


def stage(model, data, park_q, target, row=None, speed=None, after_reset=None,
          on_tick=None, reset="full"):
    """Drive the arm to a tool0 pose and settle, so the wrist camera moves.

    Uses the same Reacher the demos fly — `week2_pick.make_reacher`, posture-
    anchored the same way — so every pose the perception is validated at is a
    pose the machine can actually hold. Building a one-off Reacher here instead
    would validate the camera at poses the picker never visits.

    ⚠️ `row.update()` has to run on every physics substep while the arm moves,
    exactly as it does in the executor. Without it the peduncle force is never
    polled, stems never snap, and a pose reached by shoving a tomato through the
    support bar looks like a clean pose.

    ⚠️ **`after_reset` exists because this function resets the scene, and that
    silently undoes anything done to the fruit beforehand.** `reset_park` calls
    `mj_resetData`, so jittering the row and *then* staging throws the jitter
    away — the run still completes, still prints the jitter figure it was asked
    for, and measures the nominal row. Week 2 logged exactly this trap in
    `run_trials` and Week 3's error budget walked into it anyway on the first
    attempt: the ±0 mm and ±20 mm rows of the table came back bit-identical,
    which is the only reason it was caught. Anything that moves fruit belongs in
    this callback, which fires after the reset and before the arm moves.
    """
    import mujoco

    from mission import DEFAULT_SPEED, INTO_ROW, park_arm, reset_park
    from week2_pick import anchor_posture, make_reacher

    # Three modes, and picking the wrong one is silent:
    #
    #   "full"  mj_resetData — a fresh independent trial. Restores every fruit
    #           to the plant and re-welds every stem, so it must NOT be used
    #           between the fruit of one row.
    #   "arm"   park the arm, leave the crop as it is. What a sequence harvest
    #           wants: fruit already crated stay crated.
    #   None    touch nothing. What a scan traverse wants — one continuous
    #           motion along the row rather than a stutter of teleports.
    if reset is True or reset == "full":
        reset_park(model, data, park_q)
    elif reset == "arm":
        park_arm(model, data, park_q)
    if after_reset is not None:
        after_reset()
        mujoco.mj_forward(model, data)
    reacher = make_reacher(model, data,
                           speed=DEFAULT_SPEED if speed is None else speed)
    anchor_posture(reacher, model, data, park_q)
    if row is not None:
        reacher.on_substep = row.update
    reacher.drive_to(np.asarray(target, float), INTO_ROW, on_tick=on_tick)
    mujoco.mj_forward(model, data)
    return reacher


if __name__ == "__main__":
    raise SystemExit(main())
