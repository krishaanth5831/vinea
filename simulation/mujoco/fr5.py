#!/usr/bin/env python3
"""Fairino FR5 arm, loaded into MuJoCo from the official Fairino URDF.

The FR5 is the arm in the Vinea MVP spec. Fairino publish a real URDF with
SolidWorks-exported masses and inertias, so there is no CAD conversion to do —
MuJoCo reads the URDF directly. This module wraps that import and adds the
things a URDF does not carry: a ground plane, lighting, actuators, a tool
site for IK to aim at, and a sane home posture.

    from fr5 import build_fr5
    model = build_fr5()

Run it directly to render a still and print the model summary:

    ./.venv/bin/python simulation/mujoco/fr5.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAIRINO = REPO / "third_party" / "frcobot_ros2" / "fairino_description"
URDF = FAIRINO / "urdf" / "fairino5_v6.urdf"
MENAGERIE = REPO / "third_party" / "mujoco_menagerie"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"

# The frame IK aims at. Without a gripper it sits on the tool flange at the tip
# of wrist3; with one it moves out between the fingertips. See TOOL_Z.
TOOL_SITE = "tool0"

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]

# The chain, base outwards. Used to switch off collisions between neighbours.
LINKS = ["base_link", "shoulder_link", "upperarm_link", "forearm_link",
         "wrist1_link", "wrist2_link", "wrist3_link"]

# Rotor inertia reflected through the gearbox. The URDF has none — it describes
# the linkage, not the drivetrain — which leaves the wrist joints with almost
# no inertia and makes a stiff position servo numerically explosive. Every real
# arm has this; MuJoCo Menagerie adds ~0.1 to arms of this size for the same
# reason. Purely a dynamics term: it changes nothing about where the arm can go.
ARMATURE = 0.1

# Elbow-up posture with the tool out in front — a good IK seed and roughly how
# the arm would sit facing a plant row. Straight-up (all zeros) is a singular
# configuration and makes IK behave badly, so never start there.
HOME = np.array([0.0, -1.6, -1.6, -0.6, 1.57, 0.0])

# Joint effort limits from the URDF: 150 Nm on the big joints, 28 on the wrist.
JOINT_FORCE = {"j1": 150, "j2": 150, "j3": 150, "j4": 28, "j5": 28, "j6": 28}

# Joint speed limits from the URDF, rad/s. MuJoCo's URDF importer keeps effort
# and drops velocity, so they have to be restated here. 3.15 rad/s is 180 deg/s,
# the FR5's real rated joint speed — fast enough that a full reach takes well
# under a second, which is why the demos run at a fraction of it. See
# reach.DEFAULT_SPEED.
JOINT_VELOCITY = {"j1": 3.15, "j2": 3.15, "j3": 3.15,
                  "j4": 3.2, "j5": 3.2, "j6": 3.2}

# Position servo gains, shared by every script that drives this arm.
# Stiff on purpose: kp holds the arm against gravity, kv damps the overshoot.
KP, KV = 4000.0, 400.0

# --- gripper -----------------------------------------------------------------
# The Robotiq 2F85 from MuJoCo Menagerie: a two-finger parallel gripper with a
# tuned four-bar linkage, working collision pads and a tendon-driven actuator.
# It is a stand-in, not the Vinea gripper — the MVP spec calls for a cradle that
# supports a truss from below plus a blade that severs the peduncle. That one is
# Week 2's job, built from primitive geoms once there is a truss to cradle.
# Mounting this one first means the attachment code, the tool frame and the pick
# state machine are all working before the geometry becomes homemade.
#
# Where things sit along wrist3's local +z, in metres:
#   0.1000  flange face — tool0 without a gripper, and where the gripper mounts
#   0.1070  2f85 base_mount   (+0.007)
#   0.1108  2f85 base         (+0.0038)
#   0.2558  pinch point       (+0.145) — midway between the fingertips
# The fingers move around the pinch point; it does not move when they open or
# close, so a fixed site is the honest place to put the tool frame.
FLANGE_Z = 0.1
PINCH_Z = 0.2558

# Everything the attach brings in is renamed with this prefix, so `base` from
# the gripper cannot collide with `base_link` from the arm.
GRIPPER_PREFIX = "gr_"
GRIPPER_ACTUATOR = GRIPPER_PREFIX + "fingers_actuator"

# The 2f85's actuator is commanded on a 0-255 scale, not in radians — Robotiq's
# own convention, kept by Menagerie.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0

# Bodies at the wrist end whose collision meshes overlap the gripper's mount.
GRIPPER_MOUNT_NEIGHBOURS = ["wrist3_link"]

# --- reach envelope ----------------------------------------------------------
# The arm sweeps its workspace around the j2 (shoulder) axis, not the floor,
# so reach is measured from there. MAX_REACH is empirical: 20k random joint
# configurations sampled inside the URDF's joint limits, largest distance from
# SHOULDER to the tool0 site. It includes the 0.1 m tool offset, so it is the
# reach of the *tool point*, not of the flange.
#
# It is a sphere approximation. The true workspace has a dead zone near the
# base and gaps the wrist cannot fold into, so a target at 0.96 m may still be
# unreachable. Treat it as "definitely too far beyond this", not as a promise.
SHOULDER = np.array([0.0, 0.0, 0.152])
MAX_REACH = 0.967

# Same measurement, re-run with tool0 moved out to the 2f85's pinch point. The
# gain is 134 mm, not the full 156 mm the tool frame moved, because at full
# extension wrist3's axis is not quite radial — the last stretch of the tool
# points slightly off the direction that would buy reach. Mount a different
# gripper and this number has to be measured again; it is not derivable from
# the offset alone.
MAX_REACH_GRIPPER = 1.100


def reach_fraction(pos, max_reach: float = MAX_REACH) -> float:
    """How far out in the workspace a world point sits. 1.0 = at the limit.

    Pass MAX_REACH_GRIPPER when the model was built with a gripper, or every
    percentage you print understates how far the arm can actually go.
    """
    return float(np.linalg.norm(np.asarray(pos, dtype=float) - SHOULDER) / max_reach)


def exit_without_teardown():
    """Quit hard, on purpose. Call this after closing an interactive viewer.

    MuJoCo 3.10's passive viewer segfaults while shutting down GLFW on this
    machine — reproducible with an eight-line script that opens a viewer and
    closes it, so it is not ours, and nothing is lost when it happens. But
    "Segmentation fault (core dumped)" printed after a clean run is exactly
    the kind of message that sends you debugging your own code for an hour.
    Everything is finished by this point, so skip the teardown that crashes.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _urdf_for_mujoco() -> Path:
    """Rewrite the ROS URDF into something MuJoCo's importer accepts.

    Two changes are needed, neither of which alters the robot:
      1. `package://fairino_description/meshes/...` is a ROS-only URI. MuJoCo
         wants a path relative to meshdir.
      2. MuJoCo reads compiler settings from a <mujoco> block inside the URDF.
    """
    src = URDF.read_text()
    src = src.replace("package://fairino_description/meshes/", "")
    compiler = (
        '<mujoco><compiler meshdir="meshes/" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/></mujoco>'
    )
    src = re.sub(r"(<robot[^>]*>)", r"\1" + compiler, src, count=1)

    out = FAIRINO / "_fr5_mujoco.urdf"   # generated; gitignored
    out.write_text(src)
    return out


def build_fr5(with_scene: bool = True, gripper: bool = False):
    """Compile the FR5 into an MjModel ready to simulate.

    with_scene: add floor, light, skybox and a framing camera. Turn off if you
                are embedding the arm in a bigger scene you build yourself.
    gripper:    mount the Robotiq 2F85 and move the tool frame to its fingertips.
    """
    return build_fr5_spec(with_scene, gripper).compile()


def build_fr5_spec(with_scene: bool = True, gripper: bool = False):
    """The FR5 as an *uncompiled* MjSpec, so callers can add things to it.

    build_fr5() just compiles this. Use the spec directly when you need to put
    something else in the scene first — a target marker, a gripper, a plant.
    A compiled MjModel is frozen; an MjSpec is not.

    `gripper` defaults off so the reach demos keep meaning what they meant. Turn
    it on and two things change that every caller has to know about: `tool0`
    moves 156 mm further out, so the same xyz needs a different arm posture, and
    the reach envelope grows to MAX_REACH_GRIPPER.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(str(_urdf_for_mujoco()))

    # Renderable offscreen buffer for recording weekly demos.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # --- physics settings ---------------------------------------------------
    # implicitfast integrates joint damping and actuator velocity feedback
    # implicitly. With Euler, kv=400 at a 2 ms timestep is past the stability
    # limit and the arm shakes itself apart. Costs nothing; MuJoCo recommends
    # it as the default for actuated robots.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    for jname in JOINTS:
        spec.joint(jname).armature = ARMATURE

    # Fairino's collision meshes for neighbouring links overlap slightly —
    # base_link and shoulder_link interpenetrate by 5 mm at every posture. Left
    # alone, the contact solver fights that overlap forever and j1 saturates
    # its 150 Nm limit trying to rotate through it. Neighbours in a serial chain
    # can never usefully collide anyway, so exclude each adjacent pair.
    for a, b in zip(LINKS, LINKS[1:]):
        spec.add_exclude(name=f"{a}__{b}", bodyname1=a, bodyname2=b)

    # --- the tool frame -----------------------------------------------------
    # IK needs a named frame to aim. j6's child is wrist3_link; the flange face
    # sits a little further along that link's local +z. With a gripper mounted,
    # the point that has to reach the tomato is between the fingertips instead,
    # so the site moves out there — same site, same name, different meaning.
    wrist3 = spec.body("wrist3_link")
    wrist3.add_site(name=TOOL_SITE, pos=[0, 0, PINCH_Z if gripper else FLANGE_Z],
                    size=[0.012, 0, 0])

    # --- actuators ----------------------------------------------------------
    # A URDF has no actuators. Position servos are the right choice here: we
    # command joint angles from IK, and MuJoCo holds them.
    for jname in JOINTS:
        act = spec.add_actuator(
            name=f"{jname}_pos",
            target=jname,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        # Stiff position control: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv
        act.gainprm[0] = KP
        act.biasprm[1] = -KP
        act.biasprm[2] = -KV
        f = JOINT_FORCE[jname]
        act.forcerange = [-f, f]

    # --- gripper ------------------------------------------------------------
    # After the arm's actuators, so j1_pos..j6_pos keep control indices 0-5 and
    # the gripper lands at 6. reach.py resolves actuators by name and does not
    # care, but a keyframe's ctrl vector is read by humans.
    if gripper:
        _attach_gripper(spec)

    if with_scene:
        _add_scene(spec)

    # --- home keyframe ------------------------------------------------------
    # Convenience for the viewer's reset button and for arm-only scenes.
    #
    # ⚠️ It is only correct for what exists *right now*. A keyframe is a flat
    # vector, and MuJoCo pads a short one with zeros — not with each body's own
    # spawn pose. So the moment a caller adds a body with a free joint after
    # this line, the keyframe claims that body starts at the world origin, and
    # mj_resetDataKeyframe teleports it there. A tomato on a table drops through
    # the floor and nothing warns you.
    #
    # reset_home() below is the reset that does not have this problem. Use it.
    spec.add_key(name="home", qpos=HOME.tolist(), ctrl=HOME.tolist())

    return spec


def arm_prefixes(model):
    """Every FR5 in this model, by name prefix. `[""]` for the usual one arm.

    Found by looking for a `j1` rather than by being told, so it cannot drift
    from what was actually built. A two-armed trolley returns `["", "b_"]`.
    """
    out = []
    for i in range(model.njnt):
        name = model.joint(i).name
        if name.endswith(JOINTS[0]):
            prefix = name[: -len(JOINTS[0])]
            if all(_has_joint(model, prefix + j) for j in JOINTS):
                out.append(prefix)
    return out


def _has_joint(model, name):
    try:
        model.joint(name)
        return True
    except KeyError:
        return False


def reset_home(model, data):
    """Put every arm at its home posture and everything else at its spawn pose.

    Prefer this over mj_resetDataKeyframe for any scene with objects in it.
    `qpos0` is built by the compiler from where each body was actually declared,
    so it is right for bodies added long after the keyframe was written; the arm
    is then overwritten with HOME on top of it.

    The gripper is left where qpos0 puts it, which is fully open.

    ⚠️ **Every arm, not just the unprefixed one.** `mj_resetData` leaves qpos at
    `qpos0`, which for an FR5 is all zeros — the arm standing straight up, a
    singular configuration where the Jacobian loses rank. Homing only arm a left
    arm b sitting in that singularity in every scene that resets, and the first
    thing to trip over it was `mission.park_posture`, which seeds from here and
    reported arm b's park pose as "missed by 1667 mm". It reads as a park point
    outside the arm's reach and is a solver started somewhere it cannot leave.
    """
    import mujoco

    mujoco.mj_resetData(model, data)
    for prefix in arm_prefixes(model):
        for value, jname in zip(HOME, (prefix + j for j in JOINTS)):
            data.joint(jname).qpos[0] = value
            data.ctrl[model.actuator(f"{jname}_pos").id] = value
    mujoco.mj_forward(model, data)


def _attach_gripper(spec):
    """Graft the Robotiq 2F85 onto the flange.

    MjSpec attachment is why `build_fr5_spec` exists at all. A compiled MjModel
    is frozen — you cannot add a body to it — but a spec is still editable, so
    two independently-authored models can be merged before either is compiled.
    Everything the child carries comes along: bodies, joints, its tendon, its
    equality constraints, its actuator, its defaults.

    Three things do NOT come along, and each one is a bug if you leave it:

    1. **Solver options are global, not per-body.** MuJoCo keeps the parent's
       and warns. Menagerie ships the 2f85 with an elliptic friction cone and
       impratio 10 for a reason: the default pyramidal cone under-approximates
       friction, and a low impratio lets contacts slide under load, which is
       exactly a gripper dropping fruit. Set them before attaching, so parent
       and child agree and there is nothing to warn about.
    2. **Adjacent-link exclusions**, the same problem the arm already had — the
       gripper mount and the wrist it bolts to overlap by design.
    3. **The tool frame.** Handled by the caller; see PINCH_Z.
    """
    import mujoco

    # Point 1. Do this first — the warning MuJoCo prints on a mismatch is
    # accurate and there is no reason to teach yourself to ignore it.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0

    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))

    # A frame is a pure coordinate offset — it adds no body, no mass and no
    # joint, it just says "put the child here". The gripper's own root sits at
    # its mounting face, so bolting it to the flange means attaching at FLANGE_Z.
    frame = spec.body("wrist3_link").add_frame(pos=[0, 0, FLANGE_Z])
    spec.attach(gripper, prefix=GRIPPER_PREFIX, frame=frame)

    # Point 2. The gripper's mount is bolted flush against the wrist, so their
    # collision meshes touch at every posture. Left in, the solver spends every
    # step pushing two things apart that are welded together.
    for link in GRIPPER_MOUNT_NEIGHBOURS:
        for part in ("base_mount", "base"):
            spec.add_exclude(name=f"{link}__{GRIPPER_PREFIX}{part}",
                             bodyname1=link, bodyname2=GRIPPER_PREFIX + part)


def gripper_ctrl(model, prefix=""):
    """Index into `data.ctrl` for the gripper, or None if none is mounted.

    Everything else in this repo drives the arm; this is the one control that
    is not a joint angle. Command GRIPPER_OPEN or GRIPPER_CLOSED, or anything
    between for a partial close.

    ⚠️ `prefix` names *which* gripper on a machine carrying more than one. It
    defaults to "" — the bare Week 1-4 names — so a single-armed scene is
    unaffected. On the two-armed trolley the second arm is attached with a
    `b_` prefix, and without this the second gripper is unreachable while
    `gripper_ctrl` silently hands back the first one's index: both arms would
    open and close together and neither would report an error.
    """
    for i in range(model.nu):
        if model.actuator(i).name == prefix + GRIPPER_ACTUATOR:
            return i
    return None


def add_mocap_target(spec, name: str = "target", radius: float = 0.03,
                     rgba=(0.9, 0.2, 0.15, 0.85)):
    """Add a red marker sphere that you can move but physics cannot.

    A mocap body is drawn and can be teleported (`data.mocap_pos`), but the
    solver never applies forces to it and it never falls. That is exactly what
    an IK target needs to be: a position, not an object.

    contype/conaffinity 0 switch collision off, so the arm passes through the
    marker instead of batting it across the room.
    """
    import mujoco

    body = spec.worldbody.add_body(name=name, mocap=True)
    body.add_geom(
        name=f"{name}_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0], rgba=list(rgba), contype=0, conaffinity=0,
    )
    return body


def add_plant_panel(spec, name: str = "plant_panel", x: float = 0.55,
                    half_y: float = 0.38, z_lo: float = 0.25, z_hi: float = 0.95):
    """A flat vertical board in front of the arm, standing in for a plant row.

    Week 2 replaces this with a real stand and hanging trusses. Its job now is
    to be something you can click on: a surface at a known plane, so a 2D mouse
    position maps to exactly one 3D point instead of a whole ray.

    It is a *body*, not a bare geom in the worldbody, because MuJoCo's viewer
    reports clicks as "you selected body N at this local point" and the world
    body is never reported. No joints, so it is welded in place and free.

    Collision is off. It is scenery this week — an arm quietly clipping the
    board is a much smaller distraction than one fighting it. Week 2 turns
    contact on, because by then contact is the whole point.
    """
    import mujoco

    body = spec.worldbody.add_body(name=name, pos=[x, 0.0, (z_lo + z_hi) / 2])
    body.add_geom(
        name=f"{name}_face", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, half_y, (z_hi - z_lo) / 2],
        rgba=[0.25, 0.42, 0.28, 1.0], contype=0, conaffinity=0,
    )
    return body


def panel_normal():
    """Which way the arm comes at the panel from: straight on, along -x."""
    return np.array([-1.0, 0.0, 0.0])


def add_crate(spec, name: str = "crate", pos=(0.30, -0.50, 0.0),
              half: float = 0.16, wall: float = 0.12, thickness: float = 0.008):
    """An open-topped crate to drop fruit into: a floor and four walls.

    Welded to the world — no joints — so it is scenery the fruit can land in
    and the arm can hit. Everything is a box, because a box is the cheapest
    thing MuJoCo can collide against and this is not the part worth spending
    solver time on.
    """
    import mujoco

    body = spec.worldbody.add_body(name=name, pos=list(pos))
    grey = [0.30, 0.33, 0.38, 1.0]
    t = thickness

    body.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half, half, t], pos=[0, 0, t], rgba=grey)
    for tag, p, s in [
        ("x+", [half, 0, wall / 2], [t, half, wall / 2]),
        ("x-", [-half, 0, wall / 2], [t, half, wall / 2]),
        ("y+", [0, half, wall / 2], [half, t, wall / 2]),
        ("y-", [0, -half, wall / 2], [half, t, wall / 2]),
    ]:
        body.add_geom(name=f"{name}_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=s, pos=p, rgba=grey)
    return body


def crate_contains(pos, crate_pos, half: float, wall: float) -> bool:
    """Did the fruit actually land in the crate, or just near it?"""
    pos = np.asarray(pos, dtype=float)
    crate_pos = np.asarray(crate_pos, dtype=float)
    return bool(abs(pos[0] - crate_pos[0]) < half
                and abs(pos[1] - crate_pos[1]) < half
                and pos[2] < crate_pos[2] + wall)


def add_reach_envelope(spec, name: str = "envelope", group: int = 2,
                       max_reach: float = MAX_REACH):
    """A faint shell at the edge of the arm's reach, as a visual aid.

    Put on geom group 2 so it can be toggled in the viewer by pressing `2`.
    See MAX_REACH above for why this sphere is an approximation.
    """
    import mujoco

    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=SHOULDER.tolist(), size=[max_reach, 0, 0],
        rgba=[0.35, 0.75, 1.0, 0.06], contype=0, conaffinity=0, group=group,
    )

    # Transparency does not exempt a geom from casting shadows, so a 1 m sphere
    # over the arm drops a dark disc on the floor and looks like a bug. MuJoCo
    # has no per-geom shadow switch — only per-light — so shadows come off.
    # The reflective floor still carries the depth cue.
    for light in spec.lights:
        light.castshadow = False


def _add_scene(spec):
    """Floor, sky, light and a camera that frames the whole arm."""
    import mujoco

    sky = spec.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.35, 0.45, 0.6], rgb2=[0.05, 0.07, 0.1], width=512, height=512,
    )
    grid_tex = spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.22, 0.24, 0.26], rgb2=[0.28, 0.30, 0.32],
        width=512, height=512,
    )
    grid_mat = spec.add_material(name="grid", texrepeat=[6, 6], reflectance=0.05)
    grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.05], material="grid",
    )
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    spec.worldbody.add_light(pos=[1.5, -1.5, 2.0], dir=[-0.5, 0.5, -1])
    spec.worldbody.add_camera(
        name="demo", pos=[1.6, -1.5, 1.2],
        xyaxes=[0.68, 0.73, 0.0, -0.31, 0.29, 0.90],
    )
    _ = (sky, grid_tex)


def main():
    import argparse
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import mujoco

    ap = argparse.ArgumentParser(description="Render a still of the FR5.")
    ap.add_argument("--gripper", action="store_true",
                    help="mount the Robotiq 2F85 and put tool0 at its fingertips")
    args = ap.parse_args()

    model = build_fr5(gripper=args.gripper)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"FR5 loaded from {URDF.name}")
    print(f"  {model.nq} DoF, {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators")
    print(f"  tool site '{TOOL_SITE}' at world pos {data.site(TOOL_SITE).xpos.round(3)}")
    reach = float(np.linalg.norm(data.site(TOOL_SITE).xpos[:2]))
    print(f"  horizontal reach at home posture: {reach:.3f} m")

    if args.gripper:
        mass = sum(model.body(i).mass[0] for i in range(model.nbody)
                   if model.body(i).name.startswith(GRIPPER_PREFIX))
        # The pinch site rides along inside the gripper; tool0 is our own site
        # bolted to wrist3. They should sit on top of each other — if they do
        # not, the offset in PINCH_Z is wrong and every grasp will miss by it.
        gap = np.linalg.norm(data.site(GRIPPER_PREFIX + "pinch").xpos
                             - data.site(TOOL_SITE).xpos)
        print(f"  gripper: 2F85, {mass:.3f} kg, ctrl index "
              f"{gripper_ctrl(model)} on 0-255")
        print(f"  tool0 to the gripper's own pinch site: {gap * 1000:.2f} mm")

    out = REPO / ("fr5_gripper.png" if args.gripper else "fr5_home.png")
    with mujoco.Renderer(model, height=960, width=1280) as r:
        r.update_scene(data, camera="demo")
        cv2.imwrite(str(out), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
