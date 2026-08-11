#!/usr/bin/env python3
"""Bug 40's experiment: does the cradle eject the fruit the way the 2F85 does?

    ./.venv/bin/python simulation/mujoco/parked/cradletrace.py
    ./.venv/bin/python simulation/mujoco/parked/cradletrace.py --err 3.9
    ./.venv/bin/python simulation/mujoco/parked/cradletrace.py --windowed
    ./.venv/bin/python simulation/mujoco/parked/cradletrace.py --sweep

Bug 40: `t1`, estimated a few millimetres off, is gripped, its stem breaks, and
it leaves the gripper at speed. Measured 2026-08-04 with `carrytrace.py`, and
the conclusion there was that **no parameter fixes it** — hold force, pad
rolling friction, `CARRY_SPEED` and the timestep were all swept and the peak
speed barely moved. The reading is that a 2F85 pinching a smooth 66 mm sphere
between two converging pads is a *geometric* instability: any off-centre
displacement is amplified until the sphere squirts out.

If that reading is right, a tool that does not pinch should not have the
failure at all. That is this file. The cradle supports the fruit from below in
an open channel and severs the peduncle with a blade; there is no converging
pair to be unstable between.

⚠️ **This is not `carrytrace.py` re-pointed, and it cannot be.** The two
instruments answer the same question about different machines:

  * `carrytrace` rides `week2_pick.execute`, a `mission.Leg` list. This cycle
    is a flat state machine and never goes near the planner, so the legs are
    not the same legs and cannot be made so — `approach -> insert -> CUT ->
    lift -> carry -> tip out` has no `grip` and no `pull` to compare against.
  * `carrytrace` reads **per-pad** normal force, split left and right, because
    a one-sided unload is the signature it was built to catch. A cradle has no
    pads. The nearest thing is the channel — floor, and the two walls — so
    that is what is recorded, and the *floor* is the load-bearing surface here
    where for the 2F85 there was no load-bearing surface at all, only a squeeze.

So the comparable quantity across the two tools is **peak fruit speed while
held**, plus where the fruit ends up. Those are like for like. The force
columns are not, and the report says so rather than lining them up in one table
and inviting the comparison.

⚠️ Parked code. Nothing here is imported by the shipped cycle and none of it
should be. See parked/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The channel. `floor` takes the weight; the walls are what a fruit rolling
# sideways would load, and are therefore the closest thing this tool has to the
# 2F85's left and right pads.
FLOOR = ("vg_floor",)
LEFT = ("vg_wall_l",)
RIGHT = ("vg_wall_r",)
ENDS = ("vg_back", "vg_front")

# ⚠️ **`carrytrace.ESCAPE_MS` cannot be reused here, and finding that out is
# part of the result.** That threshold is 1.0 m/s of *absolute* fruit speed,
# and it is valid there because the Robotiq cycle carries under
# `mission.CARRY_SPEED = 0.25` and moves the tool at well under 0.5 m/s — so
# anything past 1.0 is the fruit moving on its own. This cycle is pre-planner:
# it drives at DEFAULT_SPEED with no carry throttle, and the tool alone crosses
# 1.0 m/s. Applied unchanged, it flags t0 and t2 as ejections at 1.00 and
# 1.06 m/s — both of which crate, 79-122 mm from the centre, with the fruit
# sitting still in the channel the whole way.
#
# That is carrytrace's own footnote happening again in a new file: "the first
# version of this file did exactly that and both control runs failed."
#
# The discriminator that *is* comparable across two tools with different carry
# speeds is the fruit's speed **relative to the tool**, and its displacement in
# the tool frame. A fruit travelling with the gripper reads near zero however
# fast the arm is going; a fruit leaving one does not.
# ⚠️ And a relative-speed threshold does not work either, which took a second
# run to find out. A fruit rattling in an open channel reaches 0.54 m/s
# relative to the tool without ever leaving it — a cradle does not hold the
# fruit still, it just stops it going anywhere, and those are different things.
#
# So the verdict is on **displacement in the tool frame**, which is what
# "ejected" actually means and is the one measure that needs no calibration
# against either tool's carry speed. Past this, the fruit is outside anything
# either gripper could still be holding it with: the 2F85's pads span ±99 mm
# and the cradle channel ±52 mm, both far inside it.
ESCAPE_MS = 1.0              # absolute, reported for continuity with carrytrace
ESCAPE_REL_MS = 0.5          # relative — reported, not used as the gate
ESCAPE_MM = 100.0            # tool-frame displacement — the gate

# The legs where the fruit is supposed to be in the cradle. `tip` is excluded
# deliberately and for exactly the reason carrytrace excludes `release`: the
# tip is where the tool is *meant* to empty, so a speed threshold applied there
# flags every successful pick as an ejection.
HELD = ("cradle_settle", "lift", "retreat", "carry", "settle")


@dataclass
class Sample:
    t: float
    leg: str
    pos: np.ndarray
    speed: float           # world frame — comparable only at equal carry speed
    rel_speed: float       # relative to the tool — the honest "is it leaving"
    in_tool: np.ndarray
    f_floor: float
    f_left: float
    f_right: float
    attached: bool


class CradleTrace:
    """Ride along on a cradle pick and record what happens to the fruit.

    Contact forces are accumulated per **physics substep** and reported as the
    peak since the last control cycle — not sampled at 100 Hz. Polling contact
    at the control rate is how Week 2 measured an effective detach force six
    times the one written down (entry 24), and the peak is the whole point here
    too.
    """

    def __init__(self, model, data, row, target):
        import mujoco

        self.model, self.data, self.row, self.target = model, data, row, target
        self.mj = mujoco
        self.tool = model.site("tool0").id
        self.body = model.body(target).id
        self.leg = "?"
        self.t = 0.0
        self.samples: list[Sample] = []
        self._peak = [0.0, 0.0, 0.0]
        self._gid = {n: model.geom(n).id for n in FLOOR + LEFT + RIGHT + ENDS}
        self._fruit_geoms = {model.geom(i).id for i in range(model.ngeom)
                             if model.geom(i).name.startswith(target)}

    def set_leg(self, name):
        self.leg = name

    def substep(self):
        """Peak normal force per channel surface, against the traced fruit."""
        d = self.data
        buf = np.zeros(6)
        acc = [0.0, 0.0, 0.0]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._fruit_geoms:
                other = g2
            elif g2 in self._fruit_geoms:
                other = g1
            else:
                continue
            self.mj.mj_contactForce(self.model, d, i, buf)
            n = abs(float(buf[0]))
            if other == self._gid["vg_floor"]:
                acc[0] += n
            elif other == self._gid["vg_wall_l"]:
                acc[1] += n
            elif other == self._gid["vg_wall_r"]:
                acc[2] += n
        for k in range(3):
            self._peak[k] = max(self._peak[k], acc[k])

    def tick(self, _t=None):
        from reach import CTRL_DT

        d = self.data
        self.t += CTRL_DT
        pos = d.xpos[self.body].copy()
        vel = d.cvel[self.body][3:6]
        tool_p = d.site_xpos[self.tool]
        tool_R = d.site_xmat[self.tool].reshape(3, 3)
        tool_v = d.cvel[self.model.site_bodyid[self.tool]][3:6]
        self.samples.append(Sample(
            t=self.t, leg=self.leg, pos=pos,
            speed=float(np.linalg.norm(vel)),
            rel_speed=float(np.linalg.norm(vel - tool_v)),
            in_tool=tool_R.T @ (pos - tool_p),
            f_floor=self._peak[0], f_left=self._peak[1], f_right=self._peak[2],
            attached=self.row.attached(self.target),
        ))
        self._peak = [0.0, 0.0, 0.0]

    # --- reading it back ----------------------------------------------------

    def peak_held(self):
        held = [s.speed for s in self.samples if s.leg in HELD]
        return max(held) if held else 0.0

    def peak_rel_held(self):
        held = [s.rel_speed for s in self.samples if s.leg in HELD]
        return max(held) if held else 0.0

    def walk_mm(self):
        """How far the fruit moves *within the tool frame* while held.

        The direct measurement of "is it rolling out", and the one number that
        means the same thing for a pinching gripper and a cradle. On the 2F85
        ejecting t1 this runs 6 -> 784 mm inside one carry.
        """
        held = [float(np.linalg.norm(s.in_tool)) for s in self.samples
                if s.leg in HELD]
        return (min(held) * 1000, max(held) * 1000) if held else (0.0, 0.0)

    def escape(self):
        """First sample where a held fruit is leaving the tool.

        Tool-frame displacement, not speed — see ESCAPE_MM for why neither
        speed threshold survived contact with this tool.
        """
        for i, s in enumerate(self.samples):
            if (s.leg in HELD
                    and float(np.linalg.norm(s.in_tool)) * 1000 > ESCAPE_MM):
                return i, s
        return None, None

    def by_leg(self):
        out = {}
        for s in self.samples:
            r = out.setdefault(s.leg, {"n": 0, "peak_speed": 0.0, "floor": 0.0,
                                       "left": 0.0, "right": 0.0,
                                       "drift0": None, "drift1": 0.0})
            r["n"] += 1
            r["peak_speed"] = max(r["peak_speed"], s.speed)
            r["floor"] = max(r["floor"], s.f_floor)
            r["left"] = max(r["left"], s.f_left)
            r["right"] = max(r["right"], s.f_right)
            d = float(np.linalg.norm(s.in_tool))
            if r["drift0"] is None:
                r["drift0"] = d
            r["drift1"] = d
        return out

    def report(self):
        from mission import BIN_POS

        if not self.samples:
            print("  cradletrace: nothing recorded")
            return
        print(f"\n  {'-' * 74}")
        print(f"  CRADLE TRACE — {self.target}, {len(self.samples)} control cycles")
        print(f"  {'leg':<14} {'cycles':>6} {'peak m/s':>9} {'floor N':>8} "
              f"{'wall L N':>9} {'wall R N':>9} {'|tool| mm':>12}")
        for leg, r in self.by_leg().items():
            print(f"  {leg:<14} {r['n']:>6} {r['peak_speed']:>9.2f} "
                  f"{r['floor']:>8.1f} {r['left']:>9.1f} {r['right']:>9.1f} "
                  f"{r['drift0'] * 1000:>5.0f}->{r['drift1'] * 1000:<5.0f}")

        i, s = self.escape()
        lo, hi = self.walk_mm()
        if s is None:
            print(f"\n  NO ESCAPE — the fruit never got further than "
                  f"{hi:.0f} mm from the tool while held "
                  f"(gate {ESCAPE_MM:.0f} mm).")
        else:
            print(f"\n  ⚠️ ESCAPE at t={s.t:.2f}s during `{s.leg}` — "
                  f"{float(np.linalg.norm(s.in_tool)) * 1000:.0f} mm from the "
                  f"tool, {s.rel_speed:.2f} m/s relative")
        end = self.samples[-1].pos
        away = float(np.linalg.norm(end[:2] - BIN_POS[:2]))
        print(f"  peak speed while held: {self.peak_rel_held():.2f} m/s relative"
              f"  ({self.peak_held():.2f} m/s in world frame — see ESCAPE_MS)")
        print(f"  fruit in the tool frame while held: {lo:.0f} -> {hi:.0f} mm"
              f"   (the 2F85 ejecting t1 runs 6 -> 784 mm)")
        print(f"  fruit ends at [{end[0]:+.2f} {end[1]:+.2f} {end[2]:+.2f}]"
              f" — {away * 1000:.0f} mm from the crate centre")

    def to_csv(self, path):
        import csv

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "leg", "x", "y", "z", "speed", "rel_speed", "tool_x",
                        "tool_y", "tool_z", "f_floor", "f_left", "f_right",
                        "attached"])
            for s in self.samples:
                w.writerow([f"{s.t:.3f}", s.leg, *[f"{v:.5f}" for v in s.pos],
                            f"{s.speed:.4f}", f"{s.rel_speed:.4f}",
                            *[f"{v:.5f}" for v in s.in_tool],
                            f"{s.f_floor:.3f}", f"{s.f_left:.3f}",
                            f"{s.f_right:.3f}", int(s.attached)])
        print(f"  wrote {path}")


def offset_for(err_mm, direction):
    """A position error of `err_mm`, pointing `direction`.

    Default direction is the one perception actually made on `t1` — the
    estimate came in low and slightly inboard, not along a convenient axis.
    """
    d = np.asarray(direction, dtype=float)
    return d / np.linalg.norm(d) * (err_mm / 1000.0)


# The unit vector of t1's real perception error, from carrytrace's own run:
# estimate [0.5992 0.0772 0.6213] against truth [0.6 0.08 0.6188].
T1_ERR_DIR = np.array([-0.0008, -0.0028, +0.0025])


def run_one(err_mm, fruit="t1", speed=None, windowed=False, csv=None,
            verbose=False, direction=None):
    import mujoco

    import week2_vinea as wv
    from fr5 import reset_home
    from plant_row import Row
    from reach import DEFAULT_SPEED
    from vinea_gripper import Blade

    speed = DEFAULT_SPEED if speed is None else speed
    model = wv.build_scene()
    data = mujoco.MjData(model)
    # ⚠️ `reset_home` before anything else, which `week2_vinea.run_trials` also
    # does and which the first version of this file left out. A fresh `MjData`
    # sits at `qpos0` — for an FR5 that is all zeros, the arm standing straight
    # up, where the Jacobian loses rank. Entry 46. The cycle still ran and
    # still reported plausible numbers; it just flew every leg out of a
    # singularity, and the tip never emptied the cradle.
    reset_home(model, data)
    reacher = wv.make_reacher(model, data, speed=speed)
    blade = Blade(model, data)
    row = Row(model, data)
    row.reset()
    mujoco.mj_forward(model, data)

    truth = row.pos(fruit).copy()
    est = truth + offset_for(err_mm, T1_ERR_DIR if direction is None else direction)
    print(f"\n  estimate {np.round(est, 4)} vs truth {np.round(truth, 4)}"
          f" -> {np.linalg.norm(est - truth) * 1000:.1f} mm")

    trace = CradleTrace(model, data, row, fruit)

    # The stem check and the contact accumulator both have to run every physics
    # substep. `pick_one` installs `row.update` itself, so chain rather than
    # replace it — dropping it would stop stems detaching mid-cycle.
    def install():
        inner = reacher.on_substep

        def both():
            if inner is not None:
                inner()
            trace.substep()
        reacher.on_substep = both

    if windowed:
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

            res = wv.pick_one(reacher, blade, row, fruit, on_tick=tick,
                              verbose=True, est=est, on_leg=trace.set_leg)
            install()
            print("\n  run finished — close the window to quit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
    else:
        # on_substep is set inside pick_one, so chain after the first tick.
        primed = [False]

        def tick(_t=None):
            if not primed[0]:
                install()
                primed[0] = True
            trace.tick(_t)

        res = wv.pick_one(reacher, blade, row, fruit, on_tick=tick,
                          verbose=verbose, est=est, on_leg=trace.set_leg)

    print(f"\n  cut={res['cut']} carried={res['carried']} "
          f"crate={res['in_bin']}  cycle {res['seconds']:.1f} s")
    trace.report()
    if csv:
        trace.to_csv(csv)
    return trace, res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fruit", default="t1")
    ap.add_argument("--err", type=float, default=3.9,
                    help="position error in mm (default 3.9, t1's own)")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--windowed", action="store_true",
                    help="open a viewer and WATCH it, at real speed")
    ap.add_argument("--sweep", action="store_true",
                    help="0 / 3.9 / 8 / 15 / 25 mm of error, the honest test — "
                         "one clean run at one error is not a result")
    args = ap.parse_args()

    if not args.windowed:
        os.environ.setdefault("MUJOCO_GL", "egl")

    if not args.sweep:
        run_one(args.err, args.fruit, args.speed, args.windowed, args.csv,
                args.verbose)
        return

    print(f"\n{'=' * 78}\n  bug 40 — the cradle, against position error")
    out = []
    for err in (0.0, 3.9, 8.0, 15.0, 25.0):
        print(f"\n{'-' * 78}\n  error {err:.1f} mm")
        trace, res = run_one(err, args.fruit, args.speed)
        out.append((err, trace, res))
    print(f"\n{'=' * 78}\n  SUMMARY — {args.fruit}, cradle and blade\n")
    print(f"  {'est err':>8} {'peak m/s held':>14} {'escaped':>8} {'cut':>5} "
          f"{'carried':>8} {'crated':>7} {'end dist mm':>12}")
    from mission import BIN_POS
    for err, trace, res in out:
        end = trace.samples[-1].pos
        away = float(np.linalg.norm(end[:2] - BIN_POS[:2])) * 1000
        _, s = trace.escape()
        print(f"  {err:>6.1f}mm {trace.peak_held():>14.2f} "
              f"{'YES' if s else 'no':>8} {str(res['cut']):>5} "
              f"{str(res['carried']):>8} {str(res['in_bin']):>7} {away:>12.0f}")


if __name__ == "__main__":
    main()
