#!/usr/bin/env python3
"""Why a tomato that was gripped, pulled and detached does not reach the crate.

`incident.py` explains what happened to fruit the arm was *not* picking. This
explains what happens to the one in its hand, which is a different question and
needed its own instrument.

The case it was built for is bug 40. On the first true sequence harvest from
perception, `t1` is estimated 3.8 mm off, gripped, its stem breaks — and it ends
the run at [6.04, -8.99, 0.03], six metres outside the greenhouse. The same
fruit on the same row from ground truth lands in the crate, as do all five.

`in_bin: False` prints identically for "dropped it beside the crate" and
"launched it across the building", which is why this went unseen from Week 2
until the per-fruit reset was removed and the flung position survived long
enough to be looked at. The difference is not academic: a dropped fruit is lost
revenue, and a fruit leaving the gripper at speed is a projectile in a building
with people and glass in it.

What it records, every control cycle from the grip onward:

    fruit speed              when does it stop moving with the tool
    fruit in the TOOL frame  the direct measurement of "is it rolling out"
    wrist angle              so the turn can be blamed or cleared
    per-pad normal force     an ejection loads one pad as the sphere rolls off

Pad forces are accumulated per *physics substep*, not per control cycle. Polling
contact at 100 Hz is how Week 2 measured an effective detach force six times the
one written down; the same trap applies here and the peak is the whole point.

    ./.venv/bin/python simulation/mujoco/carrytrace.py --windowed  # WATCH it get thrown
    ./.venv/bin/python simulation/mujoco/carrytrace.py            # t1, from perception
    ./.venv/bin/python simulation/mujoco/carrytrace.py --fruit t3
    ./.venv/bin/python simulation/mujoco/carrytrace.py --truth    # the control: ground truth
    ./.venv/bin/python simulation/mujoco/carrytrace.py --csv /tmp/t1.csv
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_row import FRUIT_R  # noqa: E402

# The four pad geoms of the Robotiq 2F85, split by finger. Two geoms per pad:
# Menagerie models each pad as a pair so the contact patch is not a single
# point. Summing within a finger is what makes "one pad loading hard while the
# other unloads" visible, which is the signature an ejection should leave.
LEFT_PADS = ("gr_left_pad1", "gr_left_pad2")
RIGHT_PADS = ("gr_right_pad1", "gr_right_pad2")

# Speed above which a *held* fruit is not being held any more. A pick at
# CARRY_SPEED 0.25 moves the tool at well under 0.5 m/s, so anything past this
# is the fruit leaving under its own energy rather than travelling with the arm.
#
# ⚠️ Measured against the legs where the fruit is supposed to be gripped, and
# `release` is deliberately not one of them. A tomato let go 50 mm above the
# crate is doing 1.0 m/s by the time it lands, purely under gravity — so a
# threshold applied to every leg flags every successful pick as an ejection.
# The first version of this file did exactly that and both control runs "failed".
ESCAPE_MS = 1.0

# The legs where the fruit is supposed to be in the gripper. Matches
# mission.CARRYING_LEGS plus the grip and pull that precede it, because the
# ejection may be set up before the carry ever starts.
WATCHED = ("grip", "pull", "extract", "turn", "carry", "over_bin", "release")

# The subset of WATCHED where the fruit is genuinely held, and therefore the
# only legs where speed means "escaped" rather than "was let go on purpose".
HELD = ("grip", "pull", "extract", "turn", "carry", "over_bin")


@dataclass
class Sample:
    """One control cycle of the carry."""

    t: float
    leg: str
    pos: np.ndarray            # fruit, world
    speed: float               # fruit, m/s
    in_tool: np.ndarray        # fruit centre expressed in the tool frame
    wrist_deg: float           # tool roll about the approach axis
    f_left: float              # peak normal force on the left pad since last sample
    f_right: float
    attached: bool
    ctrl: float                # the gripper command in force at this cycle


class CarryTrace:
    """Ride along on a pick and record what happens to the fruit being carried.

    Wraps a `Blackbox` rather than replacing it, because `execute()` installs
    exactly one substep callback and the incident record is not optional. Pass
    the wrapper where the Blackbox would have gone:

        box = CarryTrace(model, data, row, "t1", Blackbox(model, data, row, "t1"))
        execute(mission, reacher, gripper, row, box=box, on_tick=box.tick, ...)
    """

    def __init__(self, model, data, row, target, inner, hold_ctrl=None):
        import mujoco

        from fr5 import gripper_ctrl

        self.model = model
        self.data = data
        self.row = row
        self.target = target
        self.inner = inner
        self.mj = mujoco

        # Experiment lever, off by default. When set, the gripper command is
        # clamped to this value on every carrying leg after the grip — which
        # tests directly whether the ejection is driven by hold force. The
        # executor otherwise leaves the tendon at GRIPPER_CLOSED (255) from the
        # grip all the way to the release.
        self.hold_ctrl = hold_ctrl
        self.grip_idx = gripper_ctrl(model)

        self.bid = model.body(target).id
        self.dofadr = model.body(target).dofadr[0]
        self.tool = model.site("tool0").id
        self.left = [model.geom(n).id for n in LEFT_PADS]
        self.right = [model.geom(n).id for n in RIGHT_PADS]
        self.fruit_geom = model.geom(f"{target}_geom").id

        self.samples: list[Sample] = []
        self.t = 0.0
        # Peaks accumulated between control cycles, then flushed by `tick`.
        self._peak = [0.0, 0.0]
        self._wrench = np.zeros(6)

    # --- the Blackbox interface, delegated -----------------------------------

    @property
    def leg(self):
        return self.inner.leg

    @leg.setter
    def leg(self, value):
        self.inner.leg = value

    def rebase(self, target):
        return self.inner.rebase(target)

    def sweep(self):
        return self.inner.sweep()

    @property
    def incidents(self):
        return self.inner.incidents

    # --- the recording -------------------------------------------------------

    def substep(self):
        """Per physics step. Runs the real black box, then peaks the pad forces.

        ⚠️ Per substep, not per control cycle. A 2 ms physics step against a
        10 ms control cycle means a peak sampled at tick rate can miss four
        fifths of the contact history, and the peak is the signal here.
        """
        self.inner.substep()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.fruit_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.fruit_geom else g1
            if other in self.left:
                side = 0
            elif other in self.right:
                side = 1
            else:
                continue
            self.mj.mj_contactForce(self.model, self.data, i, self._wrench)
            self._peak[side] = max(self._peak[side], abs(float(self._wrench[0])))

    def tick(self, _t=None):
        """Per control cycle. Pass as `on_tick`."""
        from week2_pick import CTRL_DT

        self.t += CTRL_DT
        # Applied before the sample so the recorded forces are the ones the
        # clamped command actually produced.
        if (self.hold_ctrl is not None and self.grip_idx is not None
                and self.leg in HELD and self.leg != "grip"):
            self.data.ctrl[self.grip_idx] = min(
                float(self.data.ctrl[self.grip_idx]), self.hold_ctrl)

        if self.leg not in WATCHED:
            self._peak = [0.0, 0.0]
            return

        pos = self.data.xpos[self.bid].copy()
        vel = self.data.qvel[self.dofadr:self.dofadr + 3]
        tool_p = self.data.site_xpos[self.tool].copy()
        tool_R = self.data.site_xmat[self.tool].reshape(3, 3)

        self.samples.append(Sample(
            t=self.t,
            leg=self.leg,
            pos=pos,
            speed=float(np.linalg.norm(vel)),
            # The fruit centre in the tool's own frame. If the sphere is rolling
            # toward the fingertips this is the axis it moves along, and it says
            # so directly rather than being inferred from a world position that
            # is also being carried by the arm.
            in_tool=tool_R.T @ (pos - tool_p),
            wrist_deg=float(np.degrees(np.arctan2(tool_R[1, 0], tool_R[0, 0]))),
            f_left=self._peak[0],
            f_right=self._peak[1],
            attached=self.row.attached(self.target),
            ctrl=(float(self.data.ctrl[self.grip_idx])
                  if self.grip_idx is not None else float("nan")),
        ))
        self._peak = [0.0, 0.0]

    # --- reading it back -----------------------------------------------------

    def escape(self):
        """The first sample where a *held* fruit is moving too fast to be held.

        Returns (index, sample) or (None, None). This is the answer to "where
        does the ejection begin", which is the question the bug log asks and the
        reason not to touch TURN_DONE_DEG before running this.
        """
        for i, s in enumerate(self.samples):
            if s.leg in HELD and s.speed > ESCAPE_MS:
                return i, s
        return None, None

    def peak_held(self):
        """Fastest the fruit ever moves while it is supposed to be gripped.

        The cleaner of the two discriminators between a drop and an ejection,
        and the earlier one — it fires mid-flight, before the fruit has gone
        anywhere. Measured on this row: ~0.5 m/s on picks that crate, 2.3 m/s
        on the one that ejects.
        """
        held = [s.speed for s in self.samples if s.leg in HELD]
        return max(held) if held else 0.0

    def by_leg(self):
        """Per-leg summary: duration, peak speed, peak pad forces, drift."""
        out = {}
        for s in self.samples:
            r = out.setdefault(s.leg, {
                "n": 0, "peak_speed": 0.0, "f_left": 0.0, "f_right": 0.0,
                "drift0": None, "drift1": 0.0, "ctrl": 0.0,
            })
            r["n"] += 1
            r["peak_speed"] = max(r["peak_speed"], s.speed)
            r["f_left"] = max(r["f_left"], s.f_left)
            r["f_right"] = max(r["f_right"], s.f_right)
            r["ctrl"] = s.ctrl
            # Drift along the finger-closing axis of the tool frame — the
            # direction a sphere rolls when it leaves the pads.
            d = float(np.linalg.norm(s.in_tool))
            if r["drift0"] is None:
                r["drift0"] = d
            r["drift1"] = d
        return out

    def report(self, ctrl_dt=0.01):
        """Print what happened, in the order it happened."""
        if not self.samples:
            print("  carrytrace: nothing recorded — the pick never reached `grip`")
            return

        print(f"\n  {'-' * 72}")
        print(f"  CARRY TRACE — {self.target}, {len(self.samples)} control cycles")
        print(f"  {'leg':<10} {'cycles':>6} {'peak m/s':>9} "
              f"{'pad L N':>8} {'pad R N':>8} {'grip':>6} {'|tool| mm':>11}")
        for leg, r in self.by_leg().items():
            print(f"  {leg:<10} {r['n']:>6} {r['peak_speed']:>9.2f} "
                  f"{r['f_left']:>8.1f} {r['f_right']:>8.1f} "
                  f"{r['ctrl']:>6.0f} "
                  f"{r['drift0'] * 1000:>5.0f}->{r['drift1'] * 1000:<4.0f}")

        i, s = self.escape()
        if s is None:
            print(f"\n  no escape — peak {self.peak_held():.2f} m/s while held, "
                  f"under the {ESCAPE_MS} m/s threshold.")
        else:
            print(f"\n  ⚠️ ESCAPE at t={s.t:.2f}s during `{s.leg}` — "
                  f"{s.speed:.2f} m/s")
            print(f"     wrist {s.wrist_deg:+.0f}°, pads L {s.f_left:.1f} N / "
                  f"R {s.f_right:.1f} N, fruit "
                  f"{np.linalg.norm(s.in_tool) * 1000:.0f} mm from the tool")
            lo = max(0, i - 12)
            print(f"\n  the 12 cycles before it:")
            print(f"  {'t':>6} {'leg':<10} {'m/s':>6} {'padL':>6} {'padR':>6} "
                  f"{'wrist':>7} {'x':>7} {'y':>7} {'z':>7}  (fruit in tool frame, mm)")
            for x in self.samples[lo:i + 3]:
                mark = " <-" if x is s else ""
                print(f"  {x.t:>6.2f} {x.leg:<10} {x.speed:>6.2f} "
                      f"{x.f_left:>6.1f} {x.f_right:>6.1f} {x.wrist_deg:>+7.0f} "
                      f"{x.in_tool[0] * 1000:>7.1f} {x.in_tool[1] * 1000:>7.1f} "
                      f"{x.in_tool[2] * 1000:>7.1f}{mark}")

        from mission import BIN_POS

        end = self.samples[-1].pos
        away = float(np.linalg.norm(end[:2] - BIN_POS[:2]))
        print(f"\n  peak speed while held: {self.peak_held():.2f} m/s")
        print(f"  fruit ends at [{end[0]:+.2f} {end[1]:+.2f} {end[2]:+.2f}]"
              f" — {away * 1000:.0f} mm from the crate centre")

    def to_csv(self, path):
        import csv

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "leg", "x", "y", "z", "speed", "tool_x", "tool_y",
                        "tool_z", "wrist_deg", "f_left", "f_right", "attached"])
            for s in self.samples:
                w.writerow([f"{s.t:.3f}", s.leg, *[f"{v:.5f}" for v in s.pos],
                            f"{s.speed:.4f}", *[f"{v:.5f}" for v in s.in_tool],
                            f"{s.wrist_deg:.2f}", f"{s.f_left:.3f}",
                            f"{s.f_right:.3f}", int(s.attached)])
        print(f"  wrote {path}")


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fruit", default="t1",
                    help="which truss to trace (default t1, the bug 40 case)")
    ap.add_argument("--after", nargs="*", default=None, metavar="tN",
                    help="pick these first, untraced, so the target is traced "
                         "in SEQUENCE context. The row does not reset between "
                         "them, so the arm arrives with the posture and the "
                         "emptied row a real harvest would give it — which is "
                         "not the same pick as one flown from park.")
    ap.add_argument("--truth", action="store_true",
                    help="plan from ground truth instead of perception — the control")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--windowed", action="store_true",
                    help="open a live viewer and WATCH the ejection happen, at "
                         "real speed. The trace tells you the fruit left at "
                         "2.3 m/s; this shows you it.")
    ex = ap.add_argument_group("hypothesis levers (both default to off)")
    ex.add_argument("--hold", type=float, default=None, metavar="CTRL",
                    help="clamp the gripper command to this (0-255) on every "
                         "carrying leg after the grip. Tests hold force.")
    ex.add_argument("--rolling", type=float, default=None, metavar="MU",
                    help="pad rolling friction (default 1e-4, effectively the "
                         "zero bug #31 names). Tests rolling.")
    ex.add_argument("--carry", type=float, default=None, metavar="FRAC",
                    help="override mission.CARRY_SPEED (default 0.25). Week 2 "
                         "found a cliff between 0.35 and 0.25 — this is where "
                         "that sweep continues downward.")
    ex.add_argument("--timestep", type=float, default=None, metavar="DT",
                    help="physics timestep (default 0.002). The artefact test: "
                         "a contact-solver impulse changes with dt, real "
                         "mechanics does not. TICKS_PER_CTRL is rescaled so "
                         "the control rate stays 100 Hz.")
    args = ap.parse_args()

    # ⚠️ Only force EGL when nothing is being watched — setting it
    # unconditionally makes the file headless-only and --windowed a silent
    # no-op. A live viewer needs GLFW.
    if not args.windowed:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    from camera import SensorCamera, stage
    from detect import HSVDetector
    from greenhouse import build_scene
    from incident import Blackbox
    from mission import (CLEARANCE, GUARD_STOP, Guard, Planner, STAGE_X,
                         park_posture, reset_park)
    from plant_row import Row, fruit_home
    from reach import DEFAULT_SPEED, Gripper
    from week2_pick import anchor_posture, execute, make_reacher
    from week3_perceive import Perception, _plan_perceived

    speed = DEFAULT_SPEED if args.speed is None else args.speed
    if args.carry is not None:
        # Read out of the module namespace inside Planner.plan, so patching it
        # here reaches the legs. Set before the Planner is built.
        import mission as _mission

        _mission.CARRY_SPEED = args.carry
        print(f"  CARRY_SPEED -> {args.carry}")
    model = build_scene(wrist_cam=not args.truth)
    if args.timestep is not None:
        # ⚠️ Both halves, or the test is confounded. Changing the timestep alone
        # changes how much sim time a control cycle advances, so the arm would
        # be flying a different trajectory and the comparison would be measuring
        # that instead. TICKS_PER_CTRL is read from the module namespace inside
        # the step loop, so rescaling it here keeps CTRL_DT at 0.01 s.
        import reach as _reach

        model.opt.timestep = args.timestep
        _reach.TICKS_PER_CTRL = max(1, int(round(_reach.CTRL_DT / args.timestep)))
        print(f"  timestep -> {args.timestep} s, TICKS_PER_CTRL -> "
              f"{_reach.TICKS_PER_CTRL} (control rate held at "
              f"{1 / _reach.CTRL_DT:.0f} Hz)")
    if args.rolling is not None:
        # ⚠️ Setting geom_friction[:, 2] alone does NOTHING, and it fails
        # silently with byte-identical results. Every pad and the fruit ship
        # with **condim = 3** — sliding friction only — and MuJoCo never reads
        # the rolling coefficient for a condim-3 contact whatever its value is.
        #
        # That is worth stating plainly because bug #31 records the pads as
        # having "rolling friction 0.0" and proposes giving them some. The value
        # is 1e-4, and it is irrelevant at any setting until condim goes to 6.
        # The pads carry priority=1, so their condim wins the pair outright and
        # setting it here is sufficient.
        for n in LEFT_PADS + RIGHT_PADS:
            gid = model.geom(n).id
            model.geom_condim[gid] = 6
            model.geom_friction[gid, 2] = args.rolling
        print(f"  pad condim -> 6, rolling friction -> {args.rolling}")
    data = mujoco.MjData(model)
    row = Row(model, data)
    park_q = park_posture(model)
    reset_park(model, data, park_q)
    row.reset()
    mujoco.mj_forward(model, data)

    name = args.fruit
    print(f"\n{'=' * 78}\n  bug 40 — carry trace on {name}, "
          f"{'ground truth' if args.truth else 'perception'}, speed {speed}")

    if args.after:
        # Fly the earlier picks through the real Week 3 harvest so the row
        # empties and the arm ends up where a sequence would leave it. Traced
        # target only; these are context, not measurement.
        from detect import HSVDetector
        from week3_perceive import harvest

        print(f"  sequence context: picking {', '.join(args.after)} first")
        pre_sensor = SensorCamera(model, camera="wrist")
        harvest(model, data, row, park_q, pre_sensor, HSVDetector(),
                speed=speed, only=list(args.after))
        print(f"  context done — {sum(1 for n in row.names if not row.attached(n))}"
              f" stem(s) already detached\n")

    planner = Planner(model, data, row, lessons=None, clearance=CLEARANCE,
                      park_q=park_q, speed=speed)

    if args.truth:
        mission = planner.plan(name)
    else:
        # Same path Week 3's harvest takes: stage in front of where the map says
        # the fruit hangs, take one frame, detect, plan on the estimate. The
        # 3.8 mm error on t1 is the input the bug needs.
        #
        # ⚠️ `_plan_perceived`, not `planner.plan`. The plan has to be built with
        # *every* fruit at its estimated position, not just the target —
        # otherwise the obstacle check runs against perfect knowledge of the
        # fruit the robot cannot see, which is a cheat rather than a camera.
        sensor = SensorCamera(model, camera="wrist")
        per = Perception(model, sensor, HSVDetector(),
                         {n: fruit_home(n) for n in row.names})
        nominal = fruit_home(name)
        stage(model, data, park_q,
              np.array([STAGE_X, nominal[1], nominal[2]]), row=row,
              speed=speed, reset="arm")
        sights, _rep = per.look(data)
        s = sights.get(name)
        if s is None:
            raise SystemExit(f"{name} not detected — nothing to trace")
        print(f"  estimate {np.round(s.est, 4)} vs truth "
              f"{np.round(row.pos(name), 4)} -> {s.err_mm:.1f} mm")
        mission = _plan_perceived(planner, row, name, sights)

    if not mission.ok:
        raise SystemExit(f"  planner refused {name}: {mission.breaches[0]}")

    reacher = make_reacher(model, data, speed=speed)
    anchor_posture(reacher, model, data, park_q)
    gripper = Gripper(model, data)
    trace = CarryTrace(model, data, row, name,
                       Blackbox(model, data, row, name),
                       hold_ctrl=args.hold)
    guard = Guard(model, data, row, name, stop=GUARD_STOP)
    guard.armed = False
    if args.windowed:
        import time
        import warnings

        import mujoco.viewer

        from reach import CTRL_DT

        warnings.filterwarnings("ignore", module="glfw")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            clock = [time.perf_counter()]

            def tick(_t=None):
                trace.tick(_t)
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
                clock[0] += CTRL_DT
                time.sleep(max(0.0, clock[0] - time.perf_counter()))
                clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

            r = execute(mission, reacher, gripper, row, box=trace, guard=guard,
                        on_tick=tick, verbose=args.verbose)
            print("\n  run finished — the window stays open, close it to quit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
    else:
        r = execute(mission, reacher, gripper, row, box=trace, guard=guard,
                    on_tick=trace.tick, verbose=args.verbose)

    print(f"\n  grasped={r['grasped']} stem={r['broke']} crate={r['in_bin']} "
          f"peak stem {r['peak_n']:.1f} N  cycle {r['seconds']:.1f} s")
    trace.report()
    if args.csv:
        trace.to_csv(args.csv)


if __name__ == "__main__":
    main()
