#!/usr/bin/env python3
"""The reach loop, shared by both Week 1 demos.

`week1_targetreach.py` and `week1_mousereach.py` differ only in how you choose
where the tomato goes — typed coordinates versus a mouse click. Everything
after that is identical and lives here: solve IK, obey the joint speed limits,
hold the answer on the servos, step physics, decide whether the arm made it.

Nothing in this file knows about windows, the mouse, or the terminal.
"""

import numpy as np

from fr5 import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    JOINT_VELOCITY,
    JOINTS,
    MAX_REACH,
    SHOULDER,
    TOOL_SITE,
    gripper_ctrl,
    reach_fraction,
    reset_home,
)

# --- what counts as arriving ------------------------------------------------
# 5 mm suits the bare arm, whose steady-state error is about 3.2 mm. Hang a
# payload on the flange and that error grows: the position servos are P+D with
# no integral term, so a constant load leaves a constant offset they cannot
# remove, and the arm settles slightly below where it was told to go. The 1.05 kg
# Robotiq gripper takes it to about 5.2 mm — just over this line, which would
# make every move report failure while visibly arriving. So the tolerance is a
# property of the loaded arm, not a constant: see Reacher(reached_mm=...).
REACHED_MM = 5.0        # under this and we call it reached
HOLD_S = 0.3            # ...but it has to stay under for this long
STALL_S = 0.8           # no improvement for this long and it is not going to
STALL_MM = 0.5          # "improvement" means at least this much closer
GIVE_UP_S = 25.0        # backstop; the stall check normally fires long before

# --- rates ------------------------------------------------------------------
# Physics runs at 500 Hz; IK runs at 100 Hz and holds its answer in between.
# Real controllers work this way — nothing solves a QP every 2 ms — and the
# solve is ~90% of the cost, so this is also what keeps the demo real-time.
DT = 0.002              # matches the model timestep
TICKS_PER_CTRL = 5
CTRL_DT = DT * TICKS_PER_CTRL

# Fraction of the FR5's rated joint speed to actually use. At 1.0 the arm
# crosses its whole workspace in about 0.7 s, which is correct and useless to
# watch. 0.15 is roughly 27 deg/s at the joints — slow enough to follow the
# motion with your eyes and see which joints move first.
DEFAULT_SPEED = 0.15

# How far short of the tomato the tool stops. `tool0` sits at the flange face,
# so the arm's own surface is only ~4 cm behind it and the marker is 3 cm
# across: measured clearance is 11 mm at 8 cm of standoff, which reads as
# touching, and 51 mm at 12. Retune in Week 2, when the gripper moves `tool0`
# out to the fingertips — the gap is always measured from wherever tool0 is.
STANDOFF = 0.12


def approach_point(ball, direction=None, standoff=STANDOFF):
    """Where the tool aims: `standoff` short of the tomato.

    `direction` is the unit vector the arm comes in along. Default is outward
    from the shoulder, which suits a target floating anywhere in space. The
    plant-row demo passes the panel normal instead, so the arm comes at the
    fruit square-on rather than diagonally — which is how you would actually
    reach into a row.
    """
    ball = np.asarray(ball, dtype=float)

    if direction is None:
        out = ball - SHOULDER
        dist = float(np.linalg.norm(out))
        if dist < 1e-6:
            return ball.copy()
        # For a tomato closer in than the standoff, backing off the full amount
        # would put the aim point behind the shoulder. Never go past halfway.
        return ball - min(standoff, 0.5 * dist) * (out / dist)

    d = np.asarray(direction, dtype=float)
    return ball + standoff * (d / np.linalg.norm(d))


class Reacher:
    """One arm, one IK solver, one control loop.

    Holds the pieces that have to stay consistent with each other: the mink
    configuration, the tasks it optimises, the speed limit it respects, and the
    MuJoCo model and data it drives.
    """

    def __init__(self, model, data, speed=DEFAULT_SPEED, standoff=STANDOFF,
                 max_reach=MAX_REACH, reached_mm=REACHED_MM, mocap=0):
        import mink

        self.model = model
        self.data = data
        self.speed = speed
        self.standoff = standoff
        self.max_reach = max_reach
        self.reached_mm = reached_mm
        # Which mocap body, if any, to park on the goal as a visual marker.
        # Pass None when the scene has its own mocap bodies doing real work —
        # a stem holding a tomato, for instance — and does not want the reach
        # loop dragging one of them around.
        self.mocap = mocap

        # Which slots in ctrl and qpos belong to the arm. Looked up by name
        # rather than assumed to be 0..5, because mounting a gripper adds both
        # an actuator and eight joints. Before this was resolved by name, a
        # gripper made `ctrl[:nu] = q[:nu]` write a finger's *joint angle* into
        # the gripper's 0-255 command — silently, and the arm still moved.
        self.arm_ctrl = np.array(
            [model.actuator(f"{j}_pos").id for j in JOINTS])
        self.arm_qpos = np.array(
            [model.joint(j).qposadr[0] for j in JOINTS])

        self.config = mink.Configuration(model)
        self.task = mink.FrameTask(
            frame_name=TOOL_SITE, frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,   # Week 1: position only. Week 2 adds angle.
            lm_damping=1.0,
        )
        # Keeps the solver from wandering into strange joint configurations.
        self.posture = mink.PostureTask(model, cost=1e-2)
        self.tasks = [self.task, self.posture]

        # The reason the arm moves at a watchable pace. A velocity limit is a
        # constraint inside the QP, not a scale factor bolted on afterwards:
        # the solver finds the best joint motion it can *given* the cap, so the
        # path stays sensible instead of just being replayed in slow motion.
        self.limits = [mink.VelocityLimit(
            model, {j: JOINT_VELOCITY[j] * speed for j in JOINTS})]

        self.reset()

    def reset(self):
        """Put the arm at its home posture and sync the solver to it."""
        reset_home(self.model, self.data)
        self.config.update(self.data.qpos[: self.model.nq].copy())
        self.posture.set_target_from_configuration(self.config)

    def step(self, ball, direction=None):
        """One control cycle. Returns (ik_error_m, arm_error_m).

        Both are measured against the aim point, not the tomato — the aim point
        is what the arm is trying to hit. The gap between the two errors is the
        servos losing to gravity and inertia.
        """
        import mink
        import mujoco

        goal = approach_point(ball, direction, self.standoff)
        self.task.set_target(mink.SE3.from_translation(goal))

        # Keep the red marker on the goal, if the scene has one. The reach
        # demos add it because their target is an abstract point and needs
        # drawing; a scene with a real tomato in it does not, and the Reacher
        # should not insist on scenery it only uses for decoration.
        if self.mocap is not None and self.model.nmocap > self.mocap:
            self.data.mocap_pos[self.mocap] = ball

        vel = mink.solve_ik(self.config, self.tasks, CTRL_DT, "daqp", 1e-3,
                            limits=self.limits)
        self.config.integrate_inplace(vel, CTRL_DT)

        # The IK answer is a setpoint, not a teleport. mj_step makes the arm
        # earn it, and the setpoint stays put for all five ticks, exactly as a
        # real controller's output would between cycles.
        #
        # Only the arm's six slots are written. The gripper is not IK's to
        # command — whatever the caller last asked for stays untouched, so a
        # closed gripper stays closed while the arm carries the fruit.
        self.data.ctrl[self.arm_ctrl] = self.config.q[self.arm_qpos]
        for _ in range(TICKS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)

        ik_pos = self.config.get_transform_frame_to_world(
            TOOL_SITE, "site").translation()
        ik_err = float(np.linalg.norm(ik_pos - goal))
        arm_err = float(np.linalg.norm(self.data.site(TOOL_SITE).xpos - goal))
        return ik_err, arm_err

    def drive_to(self, ball, direction=None, on_tick=None):
        """Run until the arm arrives or stops getting closer.

        Stopping on a stall rather than a fixed timeout matters once the arm
        is running slowly: a target it cannot reach should be called quickly at
        any speed, and the honest signal for that is "it stopped improving",
        not "the clock ran out".
        """
        held = stalled = t = 0.0
        best = float("inf")
        ik_err = arm_err = float("nan")

        while t < GIVE_UP_S:
            ik_err, arm_err = self.step(ball, direction)
            t += CTRL_DT
            if on_tick is not None:
                on_tick(t)

            held = held + CTRL_DT if arm_err * 1000 < self.reached_mm else 0.0
            if held >= HOLD_S:
                break

            if arm_err < best - STALL_MM / 1000:
                best, stalled = arm_err, 0.0
            else:
                stalled += CTRL_DT
                if stalled >= STALL_S:
                    break

        return attempt(ball, held >= HOLD_S, ik_err, arm_err, t, direction,
                       self.standoff, self.max_reach)


class Gripper:
    """The one control in this repo that is not a joint angle.

    The 2F85 is commanded on Robotiq's own 0-255 scale — 0 fully open, 255
    fully closed — through a single actuator pulling a tendon that splits the
    force between both fingers. Two fingers, one number.

    Commanding it is instant; the fingers travelling there is not, so every
    command has to be followed by holding the arm still long enough for them
    to arrive. See hold().
    """

    def __init__(self, model, data):
        self.data = data
        self.index = gripper_ctrl(model)
        if self.index is None:
            raise RuntimeError("no gripper in this model — build with gripper=True")

    def set(self, value):
        self.data.ctrl[self.index] = value

    def open(self):
        self.set(GRIPPER_OPEN)

    def close(self):
        self.set(GRIPPER_CLOSED)


def hold(reacher, target, seconds, on_tick=None, direction=None):
    """Keep the arm where it is for a while, still stepping physics.

    The arm has to keep being commanded to its current target — let go of the
    setpoint and it sags. So "waiting" is an active instruction rather than an
    absence of one, which is true of the real arm too.
    """
    for _ in range(int(seconds / CTRL_DT)):
        reacher.step(target, direction)
        if on_tick is not None:
            on_tick(0)


def attempt(ball, reached, ik_err, arm_err, seconds, direction=None,
            standoff=STANDOFF, max_reach=MAX_REACH):
    """What happened on one attempt, in the shape describe() expects.

    `ball` is where you put the tomato, and what gets printed. `reach_frac` is
    of the aim point, because that is the position the arm actually has to hit
    and therefore the one that decides whether it can.
    """
    ball = np.asarray(ball, dtype=float)
    aim = approach_point(ball, direction, standoff)
    return {
        "goal": ball,
        "aim": aim,
        "reached": reached,
        "ik_mm": ik_err * 1000,
        "arm_mm": arm_err * 1000,
        "seconds": seconds,
        "reach_frac": reach_fraction(aim, max_reach),
    }


def describe(result) -> str:
    """One line of plain English about an attempt."""
    g = result["goal"]
    where = f"[{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}]"
    frac = result["reach_frac"]

    if result["reached"]:
        return (f"  reached {where} in {result['seconds']:.1f}s   "
                f"ik {result['ik_mm']:.1f} mm · arm {result['arm_mm']:.1f} mm   "
                f"({frac * 100:.0f}% of reach)")

    return (f"  NOT reached {where} — {result['arm_mm']:.0f} mm short after "
            f"{result['seconds']:.1f}s\n"
            f"    {why_not(result)}   ({frac * 100:.0f}% of reach)")


def why_not(result) -> str:
    """Which of the three failures this was. They are not the same problem.

    Worth separating, because Week 4 has to make this call automatically: a
    tomato outside the envelope means drive the base, a tomato the arm cannot
    physically get to means approach from another angle.
    """
    if result["reach_frac"] > 1.0:
        return "outside the reach envelope — no posture of the arm gets there"

    # IK is pure geometry: it happily solves for a point underground, then the
    # floor stops the arm from following. A big gap between the two errors is
    # the signature of something physically blocking the way.
    if result["ik_mm"] < 20.0:
        blocked = "the floor" if result["aim"][2] < 0.05 else "something"
        return (f"IK solved it, but the arm could not follow — {blocked} is in "
                f"the way (ik {result['ik_mm']:.1f} mm vs arm "
                f"{result['arm_mm']:.0f} mm)")

    return ("inside the envelope, but IK found no joint solution — joint "
            "limits, or too close in to the base to fold into")


def random_target(rng, max_frac=0.85, min_frac=0.25, min_z=0.15):
    """A random point the arm has a fair chance of reaching.

    Rejection-sampled inside the reach sphere, then pushed above the floor —
    IK has no idea the floor exists, so a target below it just drives the arm
    into the ground.
    """
    from fr5 import MAX_REACH

    while True:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = SHOULDER + v * MAX_REACH * rng.uniform(min_frac, max_frac)
        if p[2] >= min_z:
            return p
