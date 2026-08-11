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

# The same two numbers for the tool's angle, used only when orientation is on.
#
# ⚠️ These exist because Week 2 turned orientation on and every move started
# reporting failure. The cause was not IK: solved to convergence, mink hits
# these targets to 1-2 mm and under 0.2°. It was this stall check. Rotating the
# wrist into the approach angle takes 6-8 s of sim time at DEFAULT_SPEED, and
# through most of it the *position* error barely moves — so a detector watching
# position alone sees 0.8 s of no progress and calls a move that is working
# fine. Progress on either axis now counts as progress.
REACHED_DEG = 5.0
STALL_DEG = 0.5

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


def approach_rotation(approach, roll=0.0):
    """A tool orientation whose fingers point along `approach`.

    `tool0`'s local +z is the finger axis — it runs from the wrist out through
    the pinch point, so pointing it at the fruit is what "coming in straight"
    means. Measured on the Week 1 table pick, position-only IK left it 47-52°
    off vertical at every one of the six sweep positions.

    The axis alone does not fix the frame: anything is still free to spin about
    it, and that spin is the direction the two pads close along. It matters as
    soon as the fruit hangs on something. Pads closing in a vertical plane will
    swipe the stem on the way in; closing horizontally, they come around the
    sides of the fruit and leave the peduncle alone. So the default roll puts
    the closing axis horizontal, and `roll` rotates away from that if you want
    it elsewhere.

    Returns a 3x3 rotation matrix, columns [x, y, z] in world coordinates.
    """
    a = np.asarray(approach, dtype=float)
    a = a / np.linalg.norm(a)

    # Any vector not parallel to the approach works as a seed for the cross
    # product. World up is the natural choice — it makes the first axis
    # horizontal — but it degenerates for a straight-down approach, which is
    # the single most common case here, so fall back to world x.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(a @ ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])

    x = np.cross(ref, a)
    x /= np.linalg.norm(x)
    y = np.cross(a, x)

    R = np.column_stack([x, y, a])

    if roll:
        c, s = np.cos(roll), np.sin(roll)
        R = R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R


class Reacher:
    """One arm, one IK solver, one control loop.

    Holds the pieces that have to stay consistent with each other: the mink
    configuration, the tasks it optimises, the speed limit it respects, and the
    MuJoCo model and data it drives.
    """

    def __init__(self, model, data, speed=DEFAULT_SPEED, standoff=STANDOFF,
                 max_reach=MAX_REACH, reached_mm=REACHED_MM, mocap=0,
                 orientation_cost=0.0, approach=None, roll=0.0,
                 roll_cost=0.0, reached_deg=REACHED_DEG, reset=True,
                 prefix=""):
        import mink

        self.model = model
        self.data = data
        self.speed = speed
        # ⚠️ Which arm this Reacher drives, on a machine carrying more than one.
        # "" is the bare Week 1-4 naming and every single-armed scene keeps it,
        # so nothing measured before this parameter existed can move.
        #
        # Names only. It does **not** make the solver ignore the other arm —
        # `mink.Configuration` is built over the whole model, so a second arm's
        # joints are inside this one's optimisation and the QP will happily use
        # them to help the tool reach, then never command them. That is the
        # `farm.armframe.pin_base` bug with a second arm in place of the
        # trolley, and it is fixed the same way, by `pin_base(..., others=)`.
        self.prefix = prefix
        self.joints = [prefix + j for j in JOINTS]
        self.tool_site = prefix + TOOL_SITE
        # `JOINT_VELOCITY` is keyed by the bare names, because it describes the
        # FR5 rather than any particular one bolted to anything. Re-key it once
        # here so the velocity limits can be built from `self.joints` directly.
        self.joint_velocity = {prefix + j: v for j, v in JOINT_VELOCITY.items()}
        self.standoff = standoff
        self.max_reach = max_reach
        self.reached_mm = reached_mm
        self.reached_deg = reached_deg

        # How hard to insist on the tool's angle, against position_cost=1.0.
        # 0.0 reproduces Week 1 exactly, which is why it is the default: the
        # reach demos are about whether the arm can put its tool somewhere, and
        # constraining the wrist as well would change what they measure.
        #
        # A picker wants this ON but not equal to position, and the reason is
        # units: FrameTask sums position error in *metres* against orientation
        # error in *radians*. The arm starts 52° off, which is 0.91 rad, and at
        # equal weights that outvalues the 0.3 m of position error it also has
        # — so the solver straightens the wrist first and swings the arm 685 mm
        # off course to do it. In this scene that detour goes through the
        # plants. Weight it well below 1.0 and position leads, with orientation
        # arriving along the way. See the table in the Week 2 instructions.
        self.orientation_cost = orientation_cost

        # Cost about the finger axis itself — the roll. Free by default, and
        # that is not a shortcut: `approach_rotation` has to return some full
        # frame, but the roll it picks is arbitrary, and charging the solver for
        # missing an arbitrary target wastes a DOF. Pinned, the QP would rather
        # sit with the fingers horizontal at the right roll than vertical at the
        # wrong one — measured, it converged 82-90° off approach at two of six
        # positions and dropped the sweep to 2/6. Free, the same run is 6/6.
        #
        # Raise it when the roll means something: pads closing in a vertical
        # plane swipe the stem on a hanging truss, which Step 3's row cares
        # about and a tomato on a table does not.
        self.roll_cost = roll_cost

        # World direction the fingers should point, e.g. [0, 0, -1] for a
        # top-down grasp or -panel_normal() for reaching into a row. None means
        # "derive it from the standoff direction", which is right whenever the
        # arm should travel in along the same line it backed off on.
        self.approach = None if approach is None else np.asarray(approach,
                                                                 dtype=float)
        self.roll = roll

        # Called after every *physics* step, not every control cycle. Week 2's
        # breakable stems need it: the check that snaps a peduncle has to run
        # at the rate the force actually changes. Polled at 100 Hz instead, a
        # stiff arm pulling on a weld drove the force to 77 N before the 6 N
        # threshold was noticed — an effective detach force twelve times the
        # one written down, and that number feeds the kg/hr figure.
        self.on_substep = None
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
            [model.actuator(f"{j}_pos").id for j in self.joints])
        self.arm_qpos = np.array(
            [model.joint(j).qposadr[0] for j in self.joints])

        self.config = mink.Configuration(model)
        self.task = mink.FrameTask(
            frame_name=self.tool_site, frame_type="site",
            position_cost=1.0,
            # Per-axis, in the tool's own frame: x and y tilt the finger axis,
            # z spins about it. See roll_cost above.
            orientation_cost=[orientation_cost, orientation_cost, roll_cost],
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
            model, {j: self.joint_velocity[j] * speed for j in self.joints})]

        # ⚠️ Constructing a Reacher *moves the arm*, and that has already cost
        # this repo one silent bug. `reset()` calls `reset_home`, which calls
        # `mj_resetData` — so building one after arranging the scene throws the
        # arrangement away: jittered fruit snap back to their nominal positions
        # and the arm teleports to HOME, which sits inside the plant row. The
        # run still prints the jitter figure it was asked for and still looks
        # like a pass.
        #
        # Callers that have already placed the scene pass `reset=False` and get
        # a solver synced to what is actually there, which is what they meant.
        if reset:
            self.reset()
        else:
            self.sync()

    def sync(self):
        """Point the solver at the arm's current posture, moving nothing."""
        self.config.update(self.data.qpos[: self.model.nq].copy())
        self.posture.set_target_from_configuration(self.config)

    def set_orientation_cost(self, cost, roll_cost=None):
        """Change how hard the tool's angle is insisted on, mid-cycle.

        Used to drop the constraint once it stops meaning anything — going
        home with an empty gripper, for instance, where demanding a particular
        wrist angle only makes a reachable pose unreachable.
        """
        self.orientation_cost = cost
        if roll_cost is not None:
            self.roll_cost = roll_cost
        self.task.set_orientation_cost([cost, cost, self.roll_cost])

    def set_speed(self, speed):
        """Change the joint speed cap, rebuilding the limit the QP sees.

        Needed because the speed is a *constraint inside the solver*, not a
        scale factor applied afterwards, so it cannot be changed by assignment.

        Week 2 changes it mid-cycle for one reason: force resolution. The pull
        that breaks a peduncle is position-controlled, and one control cycle at
        full speed moves the setpoint 0.0047 rad, which against kp=4000 is
        ~19 Nm — about 48 N at the tool. A threshold of 6 N cannot be resolved
        inside a 48 N step: the weld reads 1.18 N, then 48 N, and "snaps at
        6 N" becomes "snaps the instant you pull", with a recorded peak eight
        times the number you wrote down. Pulling slower shrinks the step.
        """
        import mink

        self.speed = speed
        self.limits = [mink.VelocityLimit(
            self.model, {j: self.joint_velocity[j] * speed for j in self.joints})]

    def reset(self):
        """Put the arm at its home posture and sync the solver to it."""
        reset_home(self.model, self.data)
        self.sync()

    def finger_axis(self, ball, direction=None):
        """Which way the fingers should point on the way in.

        An explicit `approach` always wins. Otherwise the arm should travel in
        along the same line it stood off on: `approach_point` puts the aim
        point at `ball + standoff * direction`, so the trip from aim to fruit
        runs along `-direction`. With no direction at all the standoff is
        outward from the shoulder, and the approach is back along it.
        """
        if self.approach is not None:
            return self.approach
        if direction is not None:
            return -np.asarray(direction, dtype=float)

        out = np.asarray(ball, dtype=float) - SHOULDER
        n = float(np.linalg.norm(out))
        return out / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])

    def tool_tilt_deg(self):
        """Angle between the finger axis and straight down, in degrees.

        The honest read on whether orientation is doing anything: 0° is a
        gripper pointing at the floor, 90° is horizontal. Week 1 sat at 47-52°
        at every sweep position without ever being asked to do otherwise.
        """
        axis = self.data.site(self.tool_site).xmat.reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))

    def tool_axis_error_deg(self, ball, direction=None):
        """How far the fingers actually point from where they were told to."""
        want = np.asarray(self.finger_axis(ball, direction), dtype=float)
        want = want / np.linalg.norm(want)
        got = self.data.site(self.tool_site).xmat.reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(float(want @ got), -1.0, 1.0))))

    # ⚠️ **`step` is split into `command` and `advance`, and the split is what
    # lets two arms share one physics loop.** A control cycle is two distinct
    # things: decide what to ask the servos for, and let the world run. Fused,
    # every arm that takes a cycle also takes a *physics* step — so two arms
    # stepping in turn advance the simulation twice per control cycle, at half
    # the timestep the model was tuned for, with arm b seeing a world arm a has
    # already moved through. That is not two arms working together, it is two
    # simulations interleaved.
    #
    # Split, a machine commands every arm and *then* steps once:
    #
    #     for arm in arms: arm.command(...)     # all setpoints written
    #     advance_once()                        # one 5x mj_step for the machine
    #
    # which is what a real dual-arm controller does — one clock, one plant, two
    # control laws. `step` below is the two of them back to back, so every
    # single-armed caller in Weeks 1-4 is untouched.

    def command(self, ball, direction=None):
        """Solve IK and write this arm's setpoints. **Steps no physics.**

        Returns the aim point, which `errors` needs and which the caller
        otherwise has to recompute.
        """
        import mink

        goal = approach_point(ball, direction, self.standoff)

        if self.orientation_cost > 0.0:
            R = approach_rotation(self.finger_axis(ball, direction), self.roll)
            self.task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(R), goal))
        else:
            # Identity rotation, but orientation_cost is 0 so the solver never
            # reads it. Building the SE3 without a rotation says that out loud.
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
        return goal

    def advance(self):
        """Run the plant on for one control cycle. Commands nothing.

        ⚠️ **On a machine with more than one arm, exactly one caller may do
        this per control cycle.** See the note above `command`. `farm.duo`'s
        machine loop owns it there; here it is called by `step`, which is what
        every single-armed caller uses.
        """
        import mujoco

        for _ in range(TICKS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)
            if self.on_substep is not None:
                self.on_substep()

    def errors(self, ball, direction=None):
        """(ik_error_m, arm_error_m) against the aim point, read right now."""
        goal = approach_point(ball, direction, self.standoff)
        ik_pos = self.config.get_transform_frame_to_world(
            self.tool_site, "site").translation()
        return (float(np.linalg.norm(ik_pos - goal)),
                float(np.linalg.norm(self.data.site(self.tool_site).xpos
                                     - goal)))

    def step(self, ball, direction=None):
        """One control cycle. Returns (ik_error_m, arm_error_m).

        Both are measured against the aim point, not the tomato — the aim point
        is what the arm is trying to hit. The gap between the two errors is the
        servos losing to gravity and inertia.
        """
        self.command(ball, direction)
        self.advance()
        return self.errors(ball, direction)

    def drive_to_steps(self, ball, direction=None):
        """`drive_to` as a generator: one `yield` per control cycle.

        ⚠️ **It yields with the setpoints written and physics not yet run**, so
        the driver's contract is `next(gen)` then advance the plant then
        `next(gen)` again. That is the whole protocol two arms share a loop
        through, and it is why the arrival and stall tests below sit *after* the
        yield — they read where the arm actually got to, which is only true once
        the world has run.

        Returns (via `StopIteration.value`) the same `attempt` dict the blocking
        version always did.
        """
        held = stalled = t = 0.0
        best = best_axis = float("inf")
        ik_err = arm_err = float("nan")
        axis_err = 0.0
        oriented = self.orientation_cost > 0.0

        while t < GIVE_UP_S:
            self.command(ball, direction)
            yield
            ik_err, arm_err = self.errors(ball, direction)
            t += CTRL_DT

            # Arriving means both, when orientation is being asked for. A move
            # that is on the spot but pointing the wrong way has not finished:
            # that is exactly the state Week 1 shipped, and it cost a grasp.
            if oriented:
                axis_err = self.tool_axis_error_deg(ball, direction)
            arrived = (arm_err * 1000 < self.reached_mm
                       and (not oriented or axis_err < self.reached_deg))
            held = held + CTRL_DT if arrived else 0.0
            if held >= HOLD_S:
                break

            closer = arm_err < best - STALL_MM / 1000
            straighter = oriented and axis_err < best_axis - STALL_DEG
            if closer or straighter:
                best = min(best, arm_err)
                best_axis = min(best_axis, axis_err)
                stalled = 0.0
            else:
                stalled += CTRL_DT
                if stalled >= STALL_S:
                    break

        return attempt(ball, held >= HOLD_S, ik_err, arm_err, t, direction,
                       self.standoff, self.max_reach, axis_err)

    def drive_to(self, ball, direction=None, on_tick=None):
        """Run until the arm arrives or stops getting closer.

        Stopping on a stall rather than a fixed timeout matters once the arm
        is running slowly: a target it cannot reach should be called quickly at
        any speed, and the honest signal for that is "it stopped improving",
        not "the clock ran out".

        The logic lives in `drive_to_steps`; this drains it, stepping physics
        itself. One arm, one loop — exactly what it always did.
        """
        return drain(self.drive_to_steps(ball, direction), self.advance,
                     on_tick)


def drain(gen, advance, on_tick=None):
    """Run a one-arm control generator to completion, stepping physics for it.

    The bridge between the two worlds: `drive_to_steps` and friends describe a
    motion as a sequence of control cycles and leave the plant to somebody else,
    which is what a two-armed machine needs. Every single-armed caller in Weeks
    1-4 wants a function that just does it, and this is that function.

    `on_tick` is handed the elapsed seconds, exactly as the blocking loops used
    to pass them. Returns whatever the generator returned.
    """
    n = 0
    try:
        while True:
            next(gen)
            advance()
            n += 1
            if on_tick is not None:
                on_tick(n * CTRL_DT)
    except StopIteration as done:
        return done.value


def hold_steps(reacher, target, seconds, direction=None):
    """`hold` as a generator. See `Reacher.drive_to_steps` for the protocol."""
    for _ in range(int(seconds / CTRL_DT)):
        reacher.command(target, direction)
        yield


class Gripper:
    """The one control in this repo that is not a joint angle.

    The 2F85 is commanded on Robotiq's own 0-255 scale — 0 fully open, 255
    fully closed — through a single actuator pulling a tendon that splits the
    force between both fingers. Two fingers, one number.

    Commanding it is instant; the fingers travelling there is not, so every
    command has to be followed by holding the arm still long enough for them
    to arrive. See hold().
    """

    def __init__(self, model, data, prefix=""):
        self.data = data
        self.prefix = prefix
        self.index = gripper_ctrl(model, prefix)
        if self.index is None:
            raise RuntimeError(
                f"no gripper named {prefix}gr_fingers_actuator in this model — "
                f"build with gripper=True")

    def set(self, value):
        self.data.ctrl[self.index] = value

    def open(self):
        self.set(GRIPPER_OPEN)

    def close(self):
        self.set(GRIPPER_CLOSED)

    def ramp(self, reacher, target, seconds, value=GRIPPER_CLOSED,
             on_tick=None, direction=None):
        """Drive the fingers to `value` gradually, holding the arm still.

        ⚠️ `close()` is a step input, and on a fruit that is attached to
        something that step is an impact. Commanding 0 -> 255 in one control
        cycle makes the pads accelerate into the tomato and arrive at speed:
        measured on a hanging fruit, the peduncle went 1.18 N -> 91 N -> 164 N
        within 10 ms, rang down, and settled at 4.6 N. The settled value is
        comfortably under SNAP_N; the *impact* is twenty-seven times over it,
        so every stem snapped on contact and no fruit ever survived to be
        pulled. On video it looks like a successful harvest.

        Ramping the command over a few tenths of a second is what a real
        Robotiq does anyway — it has a speed setting — and it keeps the peak
        near the settled value instead of an order of magnitude above it.
        """
        drain(self.ramp_steps(reacher, target, seconds, value, direction),
              reacher.advance,
              None if on_tick is None else (lambda _t: on_tick(0)))

    def ramp_steps(self, reacher, target, seconds, value=GRIPPER_CLOSED,
                   direction=None):
        """`ramp` as a generator. See `Reacher.drive_to_steps` for the protocol.

        The finger command is written alongside the arm's setpoints, before the
        yield, so a machine stepping two arms at once ramps both grippers
        against the same physics step rather than one per step.
        """
        start = float(self.data.ctrl[self.index])
        steps = max(1, int(seconds / CTRL_DT))
        for i in range(steps):
            self.set(start + (value - start) * (i + 1) / steps)
            reacher.command(target, direction)
            yield


def hold(reacher, target, seconds, on_tick=None, direction=None):
    """Keep the arm where it is for a while, still stepping physics.

    The arm has to keep being commanded to its current target — let go of the
    setpoint and it sags. So "waiting" is an active instruction rather than an
    absence of one, which is true of the real arm too.
    """
    drain(hold_steps(reacher, target, seconds, direction), reacher.advance,
          None if on_tick is None else (lambda _t: on_tick(0)))


def attempt(ball, reached, ik_err, arm_err, seconds, direction=None,
            standoff=STANDOFF, max_reach=MAX_REACH, axis_deg=0.0):
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
        "axis_deg": axis_deg,
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
