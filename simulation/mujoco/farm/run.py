#!/usr/bin/env python3
"""A whole shift: scout the house, plan the route, harvest it into the crate.

This is the chain the other modules are pieces of, run end to end with nothing
handed to it. The robot is told the house exists and where the aisle is; it is
not told where a single tomato is.

    1. SCOUT     drive the aisle, survey from the moving camera, fuse a map
    2. PLAN      stop cover over the ripe fruit, pick order at each stop
    3. HARVEST   drive to each stop, work it, drop the fruit in the crate

⚠️ **The one thing the executor still needs that a camera cannot give it is a
name.** `mission.Planner.plan` takes a body name, because a simulator's fruit
are named bodies — so each mapped position is associated to the nearest truss
in the pool, exactly as `deck_cam.DeckSurvey.look` does and for exactly the same
reason. That step reads ground truth and it is **bookkeeping, not perception**:
the position the arm is sent to is the map's, never the simulator's. A real
machine has no such step, because it picks at a position.

    ./.venv/bin/python simulation/mujoco/farm/run.py               # a shift
    ./.venv/bin/python simulation/mujoco/farm/run.py --truth       # skip scouting
    ./.venv/bin/python simulation/mujoco/farm/run.py --stops 2     # short
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import armframe  # noqa: E402
from farm import crop as fcrop  # noqa: E402
from farm import house, route, trolley  # noqa: E402
from farm.scout import HouseMap, Scout, Sighting  # noqa: E402

# How close a mapped position has to be to a truss for the executor to accept
# the name. 120 mm — wider than `deck_cam.DECK_GATE_M`'s 60 because these
# positions come off a moving base and carry its error too, and narrower than
# the ~500 mm gap between fruit so it cannot claim a neighbour's name.
NAME_GATE_M = 0.12


@dataclass
class Shift:
    """What a whole run did."""

    rows: list = field(default_factory=list)
    house_map: object = None
    route: object = None
    scout_s: float = 0.0
    drive_s: float = 0.0
    pick_s: float = 0.0
    drive_m: float = 0.0
    ripe_truth: int = 0
    # The house as it actually was. Carried so a miss can be attributed to a
    # *fruit* rather than only to an attempt: a ripe tomato the scout never
    # mapped never becomes a row in `rows`, so counting rows alone reports a
    # clean shift while fruit are left on the plant. See `farm/misses.py`.
    trusses: list = field(default_factory=list)
    aisle: int = 0

    @property
    def crated(self):
        return sum(1 for r in self.rows if r.get("in_bin"))

    def report(self):
        from outcomes import report as bucket_report

        print(f"\n{'=' * 78}")
        print(f"  SHIFT — {len(self.rows)} attempts on "
              f"{self.ripe_truth} ripe fruit in the house")
        bucket_report(self.rows, "the shift")
        total = self.scout_s + self.drive_s + self.pick_s
        print(f"\n  where the time went")
        for name, s in (("scouting the house", self.scout_s),
                        ("driving between stops", self.drive_s),
                        ("picking", self.pick_s)):
            bar = "#" * int(round(28 * s / max(total, 1e-9)))
            print(f"    {name:<22} {s:>7.1f} s  {bar}")
        print(f"    {'total':<22} {total:>7.1f} s")
        print(f"    trolley odometer      {self.drive_m:>7.1f} m")
        if self.crated:
            print(f"\n  {self.crated} tomatoes in the crate · "
                  f"{total / self.crated:.1f} s each, all in")


def associate(sightings, model, data, names, gate=NAME_GATE_M):
    """Mapped positions -> truss body names. Bookkeeping; see the module note."""
    pairs = sorted(
        (float(np.linalg.norm(s.pos - data.body(n).xpos)), i, n)
        for i, s in enumerate(sightings) for n in names
        if np.linalg.norm(s.pos - data.body(n).xpos) <= gate)
    taken, out = set(), {}
    for _d, i, n in pairs:
        if i in taken or n in out.values():
            continue
        taken.add(i)
        out[i] = n
    return out


def run(seed=None, aisle=0, n_per_row=14, speed=0.4, use_truth=False,
        max_stops=None, on_tick=None, on_event=None, verbose=True,
        scene=None):
    """Scout, plan and harvest. Returns a `Shift`.

    `scene` is an optional `(model, data, trusses)` a caller has already built.
    ⚠️ `farm.watch` needs it: a `mujoco.Renderer` is bound to one `MjModel`, so
    a viewer that wants to show panels for the whole shift has to own the model
    before the shift starts. Building a second one here would leave the window
    rendering a house nothing is stepping.
    """
    import mujoco

    from carrytrace import CarryTrace
    from incident import Blackbox
    from mission import Guard, Planner, park_arm, reset_park
    from outcomes import classify
    from plant_row import Row
    from reach import Gripper
    from week2_pick import Aborted, execute, make_reacher, anchor_posture

    def say(kind, **info):
        if on_event is not None:
            on_event(kind, **info)

    if scene is None:
        seed = fcrop.resolve_seed(seed)
        trusses = fcrop.spawn(n_per_row=n_per_row, seed=seed)
        model = trolley.build(aisle=aisle, arms=("a",), trusses=trusses,
                              wrist_cam=True, deck_cam=True, seed=seed)
        data = mujoco.MjData(model)
    else:
        model, data, trusses = scene
    mujoco.mj_forward(model, data)
    park_q = armframe.park_posture(model, data)
    reset_park(model, data, park_q)

    names = [t.name for t in trusses]
    row = Row(model, data, names=names, homes={t.name: t.pos for t in trusses})
    mujoco.mj_forward(model, data)
    drive = trolley.Drive(model, data)

    left, right = house.serves(aisle)
    shift = Shift(ripe_truth=sum(1 for t in trusses
                                 if t.ripe and t.row == right),
                  trusses=list(trusses), aisle=aisle)

    # --- 1. scout ------------------------------------------------------------
    t0 = _time.perf_counter()
    if use_truth:
        house_map = HouseMap(
            sightings=[Sighting(pos=t.pos, stage=t.stage, hue=float("nan"),
                                row=t.row, truth=t) for t in trusses],
            aisle=aisle)
        if verbose:
            print("\n  ⚠️ SCOUT SKIPPED — routing the operator's own answer")
    else:
        if verbose:
            print(f"\n{'=' * 78}\n  1. SCOUT — driving the aisle, "
                  f"mapping what is there")
        say("phase", phase="SCOUT")
        scout = Scout(model, data, aisle=aisle)
        try:
            house_map = scout.run(drive, on_tick=on_tick, verbose=verbose,
                                  on_frame=lambda *a: say("scout_frame",
                                                          i=a[0], n=a[1], y=a[2],
                                                          seen=a[3]))
        finally:
            scout.close()
        if verbose:
            print(f"\n{house_map.table()}")
    shift.scout_s = drive.travelled / trolley.DRIVE_SPEED if not use_truth else 0.0
    shift.house_map = house_map
    say("map", house_map=house_map)

    # --- 2. plan -------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 78}\n  2. PLAN — where to stop and what to take")
    say("phase", phase="PLAN")
    r = route.plan(house_map, aisle=aisle, arm="a")
    if max_stops:
        r.stops = r.stops[:max_stops]
    shift.route = r
    if verbose:
        print(r.table())
    say("route", route=r)

    # --- 3. harvest ----------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 78}\n  3. HARVEST — {r.n_fruit} ripe fruit in "
              f"{len(r.stops)} stops")
    say("phase", phase="HARVEST")
    odo0 = drive.travelled

    for si, stop in enumerate(r.stops, 1):
        say("stop", i=si, n=len(r.stops), y=stop.y, fruit=list(stop.fruit))
        if verbose:
            print(f"\n  --- stop {si}/{len(r.stops)} at y={stop.y:+.2f}, "
                  f"{stop.n} fruit ---")
        t_drive = drive.drive_to(stop.y, on_tick=on_tick)
        shift.drive_s += t_drive
        park_arm(model, data, park_q)
        mujoco.mj_forward(model, data)

        # Names for the executor. Done per stop, against the fruit still on the
        # plant, so a fruit already picked cannot claim a name a later one wants.
        standing = [t.name for t in trusses if row.attached(t.name)]
        ident = associate(stop.fruit, model, data, standing)

        for fi, fruit in enumerate(stop.fruit):
            name = ident.get(fi)
            rec = {"stop": si, "stop_y": stop.y, "stage": fruit.stage,
                   "row": fruit.row, "seen": name is not None}
            if name is None:
                rec["outcome"] = "not_detected"
                shift.rows.append(rec)
                if verbose:
                    print(f"    {fruit.stage:<8} at y={fruit.pos[1]:+.2f}: "
                          f"no truss within {NAME_GATE_M * 1000:.0f} mm")
                continue
            rec["fruit"] = name

            say("select", fruit=name, stage=fruit.stage, stop=si,
                left=len(stop.fruit) - fi)
            p0 = _time.perf_counter()
            with armframe.at_trolley(model, data):
                planner = Planner(model, data, row, lessons=None,
                                  clearance=0.040, park_q=park_q, speed=speed)
                m = planner.plan(name)
            rec["t_plan"] = _time.perf_counter() - p0

            if not m.ok:
                rec["refused"] = True
                rec["refused_reason"] = str(m.breaches[0] if m.breaches
                                            else "no route")
                rec["outcome"] = classify(rec)
                shift.rows.append(rec)
                say("refused", fruit=name, why=rec["refused_reason"])
                if verbose:
                    print(f"    {name} ({fruit.stage}): REFUSED — "
                          f"{rec['refused_reason']}")
                continue

            reacher = make_reacher(model, data, speed=speed)
            # ⚠️ Before anything else. Without this the IK solver plans base
            # motion the trolley will never make; see `armframe.pin_base`.
            armframe.pin_base(reacher)
            anchor_posture(reacher, model, data, park_q)
            gripper = Gripper(model, data)
            trace = CarryTrace(model, data, row, name,
                               Blackbox(model, data, row, name))
            guard = Guard(model, data, row, name)
            guard.armed = False
            say("fly", fruit=name, mission=m, trace=trace, guard=guard)

            tick = trace.tick if on_tick is None else (
                lambda t=None, _tr=trace: (_tr.tick(t), on_tick(t)))
            f0 = _time.perf_counter()
            try:
                with armframe.at_trolley(model, data):
                    res = execute(m, reacher, gripper, row, box=trace,
                                  guard=guard, on_tick=tick)
                aborted = res.get("aborted")
            except Aborted as stop_why:
                res = {"in_bin": False, "grasped": False, "broke": False,
                       "lost": 0, "disturbed": 0, "seconds": 0.0,
                       "peak_n": 0.0, "clearance": float("nan")}
                aborted = stop_why.why
            shift.pick_s += res["seconds"]
            rec.update({
                "in_bin": bool(res["in_bin"]), "grasped": bool(res["grasped"]),
                "broke": bool(res["broke"]), "peak_n": float(res["peak_n"]),
                "lost": int(res["lost"]), "disturbed": int(res["disturbed"]),
                "aborted": str(aborted) if aborted else None,
                "peak_held_ms": trace.peak_held(),
                "t_fly": float(res["seconds"]),
            })
            rec["clean"] = bool(res["in_bin"] and not res["lost"]
                                and not res["disturbed"] and not aborted)
            rec["outcome"] = classify(rec)
            shift.rows.append(rec)
            say("result", fruit=name, rec=rec)
            if verbose:
                print(f"    {name} ({fruit.stage}): {rec['outcome']:<14} "
                      f"crate={rec['in_bin']} {rec['t_fly']:.1f}s")

    shift.drive_m = drive.travelled
    say("phase", phase="DONE")
    return shift


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", action="store_true",
                    help="skip scouting and route the real crop")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=14, help="fruit per row")
    ap.add_argument("--stops", type=int, default=None, help="cap the stops")
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)
    shift = run(seed=args.seed, aisle=args.aisle, n_per_row=args.n,
                speed=args.speed, use_truth=args.truth, max_stops=args.stops)
    shift.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
