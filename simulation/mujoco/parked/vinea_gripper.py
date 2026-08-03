#!/usr/bin/env python3
"""The Vinea gripper: a passive cradle and a blade. One moving part.

The Robotiq 2F85 is a stand-in. The MVP spec calls for a tool that **supports a
truss from below and severs the peduncle**, which is a different machine — and
a much simpler one to simulate, which is the point of building it here.

    Robotiq 2F85     8 joints, a four-bar linkage, a tendon, 2 equalities
    Vinea gripper    1 joint

Everything the 2F85 fought this week comes from squeezing: the close was an
impact that broke stems (bug 28), the pads squeezed the fruit out under
rotation (bug 31), and `priority=1` on the pads quietly overrode the fruit's
own contact parameters (bug 29). A cradle does not squeeze. It holds the fruit
up, and gravity and friction do the rest.

### The blade does not cut anything

MuJoCo cannot sever geometry. There is no mesh cutting and there is not going
to be. The blade is exactly two things:

1. geometry that looks right on video, and
2. a condition in the state machine — `data.eq_active[peduncle] = 0`.

That inverts how it feels. Cutting sounds like the hard problem and cradling
sounds easy; in simulation it is the other way round, and the cut is the
cheapest part of the whole tool.

**Primitive geoms, not Onshape.** Boxes and cylinders collide faster and skip
mesh convexification entirely. Onshape matters when there is a fabrication
drawing and a hardware budget.

Run it directly to check the mount, the mass and the blade travel:

    ./.venv/bin/python simulation/mujoco/vinea_gripper.py
"""

import numpy as np

# --- where things sit along wrist3's local +z, in metres ---------------------
# Same convention as the 2F85 mount: +z runs out of the flange face.
#
# Under `approach_rotation`, driving the tool at a target along +z maps the
# gripper's local x to world y and its local y to world z. So local -y is
# *down* in the world, which is where the cradle floor goes, and local +y is up,
# which is where the peduncle is and therefore where the blade goes.
FLANGE_Z = 0.1
TOOL_Z = 0.20           # the cradling point: where the fruit centre ends up

CRADLE_HALF_X = 0.052   # half-width across the channel
CRADLE_HALF_Z = 0.050   # half-depth along the approach axis
FLOOR_Y = -0.040        # cradle floor, below the fruit centre
# Wall height above the floor is doubled by the box half-size, so the rim ends
# up 20 mm above the fruit centre: deep enough that a 33 mm fruit cannot roll
# out over the side, shallow enough that the blade still has clear air at the
# peduncle 50 mm up.
WALL_H = 0.030

BLADE_Y = 0.050         # blade height above the fruit centre = the stem
BLADE_TRAVEL = 0.030    # 30 mm of stroke
BLADE_CLOSED = 0.0      # retracted
BLADE_CUT = BLADE_TRAVEL

# Mass budget. The Robotiq is 1.05 kg and the arm's droop was characterised
# against that, so keeping this in the same region keeps the FR5 payload margin
# honest when the kg/hr number has to be defended.
TOTAL_MASS = 1.2

ACTUATOR = "vg_blade_pos"
STEEL = [0.62, 0.64, 0.68, 1.0]
DARK = [0.20, 0.22, 0.25, 1.0]
EDGE = [0.85, 0.86, 0.90, 1.0]


def add_vinea_gripper(spec, tool_site="tool0"):
    """Bolt the cradle and blade onto wrist3 and put the tool frame in it.

    Returns nothing; the caller compiles the spec. Unlike the 2F85 this needs
    no `spec.attach()` — there is no second model to merge, just geometry.
    """
    import mujoco

    wrist = spec.body("wrist3_link")

    # The cradle is rigid and passive: no joint, so it is part of wrist3.
    # Zero DOF is the whole design argument — fewer moving contact bodies is
    # the difference between a week that lands and one that does not.
    body = wrist.add_body(name="vg_cradle", pos=[0, 0, FLANGE_Z])

    # Neck, flange out to the cradle.
    body.add_geom(name="vg_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  fromto=[0, 0, 0,
                          0, 0, TOOL_Z - FLANGE_Z - CRADLE_HALF_Z],
                  size=[0.022, 0, 0], rgba=DARK, mass=TOTAL_MASS * 0.28)

    # Floor: what the truss actually rests on.
    body.add_geom(name="vg_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[CRADLE_HALF_X, 0.005, CRADLE_HALF_Z],
                  pos=[0, FLOOR_Y, TOOL_Z - FLANGE_Z],
                  rgba=STEEL, mass=TOTAL_MASS * 0.22,
                  # A cradle works by friction against a fruit resting in it.
                  # Menagerie's pad values do not apply here — nothing has
                  # priority — so this is the number that governs.
                  friction=[1.0, 0.02, 0.01])

    # Two walls, making the U. They stop the fruit rolling out sideways, which
    # is the failure the 2F85 had under rotation.
    for tag, sx in [("l", -1), ("r", +1)]:
        body.add_geom(
            name=f"vg_wall_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.005, WALL_H, CRADLE_HALF_Z],
            pos=[sx * CRADLE_HALF_X, FLOOR_Y + WALL_H, TOOL_Z - FLANGE_Z],
            rgba=STEEL, mass=TOTAL_MASS * 0.09,
            friction=[1.0, 0.02, 0.01])

    # End stops, front and back.
    #
    # ⚠️ Both are needed, and the front one is the whole reason this tool is
    # entered from below. With the channel open along the approach axis, the
    # fruit sits in an open-ended trough: backing out of the row accelerates
    # the cradle in -x and the fruit's own inertia carries it straight out the
    # front. Measured, it was 942 mm away by the end of the retreat — the cut
    # worked perfectly and the harvest still failed.
    #
    # Closed front and back, the channel opens only *upward* (local +y maps to
    # world +z at roll 0), which is what "cradle from below" actually means:
    # the tool goes under the truss and lifts into it.
    for tag, sz in [("back", -1), ("front", +1)]:
        body.add_geom(
            name=f"vg_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[CRADLE_HALF_X, WALL_H, 0.005],
            pos=[0, FLOOR_Y + WALL_H,
                 TOOL_Z - FLANGE_Z + sz * CRADLE_HALF_Z],
            rgba=STEEL, mass=TOTAL_MASS * 0.04,
            friction=[1.0, 0.02, 0.01])

    # The arm that carries the blade up to peduncle height.
    body.add_geom(name="vg_mast", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.006, (BLADE_Y - FLOOR_Y) / 2, 0.006],
                  pos=[0, (BLADE_Y + FLOOR_Y) / 2,
                       TOOL_Z - FLANGE_Z - CRADLE_HALF_Z],
                  rgba=DARK, mass=TOTAL_MASS * 0.14)

    # --- the one moving part -------------------------------------------------
    blade = body.add_body(name="vg_blade",
                          pos=[0, BLADE_Y, TOOL_Z - FLANGE_Z])
    blade.add_joint(name="vg_blade_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[1, 0, 0], range=[0.0, BLADE_TRAVEL],
                    damping=2.0, armature=0.01)
    blade.add_geom(name="vg_blade_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.030, 0.004, 0.0012],
                   pos=[-CRADLE_HALF_X + 0.010, 0, 0],
                   rgba=EDGE, mass=TOTAL_MASS * 0.10,
                   # It must not collide with anything. It is a marker for a
                   # state-machine condition, and a 1.2 mm plate given to the
                   # contact solver is a source of instability and nothing else.
                   contype=0, conaffinity=0)

    # Position servo on the slide, same shape as the arm's joint servos:
    # gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv. Assigned elementwise
    # because MjSpec wants the full-length parameter vectors.
    act = spec.add_actuator(
        name=ACTUATOR, target="vg_blade_slide",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE)
    act.gainprm[0] = 600.0
    act.biasprm[1] = -600.0
    act.biasprm[2] = -30.0
    act.ctrlrange = [0.0, BLADE_TRAVEL]
    act.forcerange = [-60.0, 60.0]

    # The tool frame sits at the cradling point, so "drive tool0 to the fruit"
    # means "put the fruit in the cradle" — same contract the 2F85 mount has.
    #
    # `build_fr5_spec` has already put this site on the flange face, so move it
    # rather than adding a second one; two sites cannot share a name, and a
    # stale tool frame on the flange would silently aim every pick 30 mm short.
    for site in spec.sites:
        if site.name == tool_site:
            site.pos = [0, 0, TOOL_Z]
            break
    else:
        wrist.add_site(name=tool_site, pos=[0, 0, TOOL_Z], size=[0.012, 0, 0])


class Blade:
    """The blade, and the only thing it can do.

    `cut()` extends it. Whether that *detaches* anything is the state machine's
    business — see `severed()`.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.index = next(i for i in range(model.nu)
                          if model.actuator(i).name == ACTUATOR)
        self.qadr = model.joint("vg_blade_slide").qposadr[0]

    def set(self, value):
        self.data.ctrl[self.index] = float(np.clip(value, 0.0, BLADE_TRAVEL))

    def retract(self):
        self.set(BLADE_CLOSED)

    def cut(self):
        self.set(BLADE_CUT)

    @property
    def travel(self):
        return float(self.data.qpos[self.qadr])

    def severed(self, peduncle_pos, tol=0.055):
        """Has the blade actually passed through this peduncle?

        Two conditions, both needed: the blade has travelled far enough, and it
        is physically near the stem. Without the second, extending the blade
        anywhere in the scene would harvest the whole row.
        """
        if self.travel < BLADE_CUT * 0.6:
            return False
        here = self.data.body("vg_blade").xpos
        return bool(np.linalg.norm(np.asarray(peduncle_pos) - here) < tol)


def main():
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from fr5 import build_fr5_spec, reset_home, TOOL_SITE

    spec = build_fr5_spec(gripper=False)
    add_vinea_gripper(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_home(model, data)
    mujoco.mj_forward(model, data)

    mass = sum(model.body(i).mass[0] for i in range(model.nbody)
               if model.body(i).name.startswith("vg_"))
    print("Vinea gripper — cradle and blade\n")
    print(f"  bodies added     {sum(1 for i in range(model.nbody) if model.body(i).name.startswith('vg_'))}")
    print(f"  moving joints    {sum(1 for i in range(model.njnt) if model.joint(i).name.startswith('vg_'))}"
          f"   (the 2F85 has 8)")
    print(f"  total mass       {mass:.3f} kg   (2F85 is 1.05 kg)")
    print(f"  tool0 at         {data.site(TOOL_SITE).xpos.round(3)}")

    blade = Blade(model, data)
    print(f"\n  blade travel, commanded {BLADE_TRAVEL * 1000:.0f} mm:")
    for cmd in [0.0, 0.010, 0.020, 0.030]:
        blade.set(cmd)
        for _ in range(600):
            mujoco.mj_step(model, data)
        print(f"    ctrl {cmd * 1000:5.1f} mm -> travel {blade.travel * 1000:5.1f} mm")

    # Reach has to be re-measured for any new tool; it is not derivable from
    # the offset alone. Same lesson as MAX_REACH_GRIPPER in fr5.py.
    from fr5 import SHOULDER
    rng = np.random.default_rng(0)
    best = 0.0
    from fr5 import JOINTS
    adr = [model.joint(j).qposadr[0] for j in JOINTS]
    lo = np.array([model.joint(j).range[0] for j in JOINTS])
    hi = np.array([model.joint(j).range[1] for j in JOINTS])
    for _ in range(20000):
        data.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)
        best = max(best, float(np.linalg.norm(
            data.site(TOOL_SITE).xpos - SHOULDER)))
    print(f"\n  max reach to the cradle point: {best:.3f} m"
          f"   (2F85 fingertips: 1.100 m)")


if __name__ == "__main__":
    main()
