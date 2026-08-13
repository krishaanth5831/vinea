#!/usr/bin/env python3
"""A whole shift on trusses: scout the house, judge each cluster, cut and crate.

`farm/run.py` is this chain for **loose** fruit — one tomato per stem, one
ripeness decision per tomato. This is the tomato-on-the-vine version, and three
things change:

    1. SCOUT     the same camera, the same colour bands, the same fused map —
                 and then the blobs are **grouped into clusters**, because the
                 unit of work is a truss and the map has to be a map of trusses
    2. PLAN      one decision per cluster: is *enough* of this truss red? See
                 `farm.truss.RIPE_FRACTION`, which is a commercial threshold
                 and not a perception one
    3. HARVEST   the arm grips the stem and the blade **cuts**. It does not pull

⚠️ **The routing is `farm.route`, unchanged, and that is the point of the
design.** A `truss.Cluster` answers `.pos`, `.ripe`, `.row` and `.stage`
exactly as a `scout.Sighting` does, so the stop-cover, the pick order and the
whole itinerary are the same code deciding the same problem over a different
unit. Nothing in `route.py` knows a truss from a tomato, and nothing in it had
to change.

⚠️ **The one thing the executor still needs that a camera cannot give it is a
name** — same as `farm.run`, same reason, same bookkeeping. See that module's
note; it applies here word for word, against truss bodies instead of fruit.

    ./.venv/bin/python simulation/mujoco/farm/trussrun.py            # a shift
    ./.venv/bin/python simulation/mujoco/farm/trussrun.py --truth    # skip scouting
    ./.venv/bin/python simulation/mujoco/farm/trussrun.py --stops 2  # short
    ./.venv/bin/python simulation/mujoco/farm/trussrun.py --recall   # score the CV
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import armframe, house, route, trolley  # noqa: E402
from farm import truss as ft  # noqa: E402

# How close a mapped grasp point has to be to a truss body for the executor to
# accept the name. Wider than `farm.run.NAME_GATE_M`'s 120 mm because a cluster
# position is inferred from the topmost fruit up the rachis, so it carries the
# fruit's own deprojection error plus the offset's, and narrower than the
# 350 mm `truss.MIN_TRUSS_SEP` so it cannot claim a neighbour's name.
NAME_GATE_M = 0.18


@dataclass
class ClusterMap:
    """Everything the robot believes about the house, as trusses.

    Quacks like `scout.HouseMap` — `route.plan` reads `.sightings` and asks each
    for `.ripe`, `.row` and `.pos`, and a `Cluster` answers all three.
    """

    sightings: list = field(default_factory=list)     # truss.Cluster
    aisle: int = 0
    drive_m: float = 0.0
    drive_s: float = 0.0
    frames: int = 0
    blobs: int = 0                                    # fruit before grouping

    @property
    def ripe(self):
        return [c for c in self.sightings if c.ripe]

    def table(self, indent="  "):
        out = [f"{indent}{len(self.sightings)} trusses grouped from "
               f"{self.blobs} detected fruit; {len(self.ripe)} ripe at "
               f"a {ft.RIPE_FRACTION:.2f} red fraction"]
        out.append(f"{indent}{'truss':>6} {'row':>4} {'y':>7} {'fruit':>6} "
                   f"{'red':>4} {'frac':>6}  take?")
        for i, c in enumerate(self.sightings):
            out.append(f"{indent}{i:>6} {c.row:>4} {c.pos[1]:>7.2f} "
                       f"{c.n_fruit:>6} {c.n_red:>4} {c.red_fraction:>6.2f}"
                       f"  {'YES' if c.ripe else 'no'}")
        return "\n".join(out)


def survey(model, data, aisle=0, drive=None, on_tick=None, on_frame=None,
           verbose=True, gap=ft.CLUSTER_GAP):
    """Drive the aisle, detect fruit by colour, then group them into trusses.

    ⚠️ **The detector is `farm.scout`'s, untouched.** Same two hue bands, same
    circularity gates, same deprojection, same fusion across frames. Ripeness
    stays colour-only, which is what was asked for — no shape model, no depth
    cue, no learned classifier. What this adds is entirely *after* perception:
    the fruit that came back get grouped, and each group gets one verdict.

    That separation is what makes the cluster rule measurable. A miss here is
    either a fruit the detector did not see or a fruit the grouper put in the
    wrong truss, and `--recall` reports the two apart.

    ⚠️ **The grouping is clean and the counting is not, and the bias has a
    direction.** Measured on seed 7 with `--recall`: 0 phantom clusters and a
    23 mm median error on the grasp point, but only 3 of 14 trusses came back
    with all six fruit — the rest are occluded by their own neighbours. The
    verdict still agreed with the truth on 13 of 14, because the rule reads a
    *fraction* and a fraction survives losing members.

    It survives losing them **evenly**, though, and these are not lost evenly.
    `farm.scout`'s own docstring measures the green band as the weak one — green
    fruit sit 6 hue from a stem and 13 from a leaf, and only circularity
    separates them. So the fruit this pipeline drops are disproportionately the
    unripe ones, which inflates the measured red fraction and makes the rule
    **more eager than the ground truth would justify**. It is the error a truss
    picker can least afford, since the cost of eagerness is a downgraded
    cluster. Do not read the 13/14 as "the counting does not matter".
    """
    from farm.scout import Scout

    scout = Scout(model, data, aisle=aisle)
    try:
        hm = scout.run(drive if drive is not None else trolley.Drive(model, data),
                       on_tick=on_tick, on_frame=on_frame, verbose=verbose)
    finally:
        scout.close()

    clusters = ft.group(hm.sightings, gap=gap)
    for c in clusters:
        c.row = int(round((c.pos[0] - house.ROW_X0) / house.ROW_PITCH))
    keep = [c for c in clusters
            if 0 <= c.row < house.N_ROWS
            and abs(c.pos[0] - house.row_x(c.row)) < 0.25]
    return ClusterMap(sightings=keep, aisle=aisle, drive_m=hm.drive_m,
                      drive_s=hm.drive_s, frames=hm.frames,
                      blobs=len(hm.sightings))


def truth_map(trusses, aisle=0):
    """The operator's own answer, shaped like a scouted map.

    ⚠️ Skips perception entirely, which is useful for exactly one thing: telling
    a routing or gripping failure apart from a detection one.
    """
    from farm.scout import Sighting

    out = []
    for t in trusses:
        members = [Sighting(pos=p, stage=s, hue=float("nan"), row=t.row)
                   for p, s in zip(t.fruit_pos(), t.stages)]
        out.append(ft.Cluster(members=members, pos=t.pos.copy(), row=t.row,
                              truth=t))
    return ClusterMap(sightings=out, aisle=aisle,
                      blobs=sum(t.n_fruit for t in trusses))


@dataclass
class Shift:
    """What a whole truss run did."""

    rows: list = field(default_factory=list)
    cluster_map: object = None
    route: object = None
    scout_s: float = 0.0
    drive_s: float = 0.0
    pick_s: float = 0.0
    drive_m: float = 0.0
    ripe_truth: int = 0
    trusses: list = field(default_factory=list)
    aisle: int = 0

    @property
    def crated(self):
        return sum(1 for r in self.rows if r.get("in_bin"))

    @property
    def fruit_crated(self):
        return sum(r.get("n_fruit", 0) for r in self.rows if r.get("in_bin"))

    @property
    def green_crated(self):
        """Green fruit cut with a truss the rule accepted. The downgrade."""
        return sum(r.get("n_green", 0) for r in self.rows if r.get("in_bin"))

    def report(self):
        from outcomes import report as bucket_report

        print(f"\n{'=' * 78}")
        print(f"  TRUSS SHIFT — {len(self.rows)} attempts on "
              f"{self.ripe_truth} ripe trusses in the house")
        bucket_report(self.rows, "the shift")
        total = self.scout_s + self.drive_s + self.pick_s
        print(f"\n  where the time went")
        for name, s in (("scouting the house", self.scout_s),
                        ("driving between stops", self.drive_s),
                        ("cutting and carrying", self.pick_s)):
            bar = "#" * int(round(28 * s / max(total, 1e-9)))
            print(f"    {name:<22} {s:>7.1f} s  {bar}")
        print(f"    {'total':<22} {total:>7.1f} s")
        print(f"    trolley odometer      {self.drive_m:>7.1f} m")
        if self.crated:
            print(f"\n  {self.crated} trusses in the crate — "
                  f"{self.fruit_crated} tomatoes, "
                  f"{self.crated * ft.TRUSS_MASS:.1f} kg, "
                  f"{total / self.crated:.1f} s each")
            print(f"  ⚠️ {self.green_crated} of those tomatoes were still "
                  f"green. TOV is graded as one item, so\n     those are a "
                  f"downgrade on the truss they were cut with, not fruit that "
                  f"ripens later.\n     That number is what "
                  f"`truss.RIPE_FRACTION` buys and spends — see --sweep.")


def associate(clusters, model, data, names, gate=NAME_GATE_M):
    """Mapped grasp points -> truss body names. Bookkeeping; see the note."""
    pairs = sorted(
        (float(np.linalg.norm(c.pos - data.body(n).xpos)), i, n)
        for i, c in enumerate(clusters) for n in names
        if np.linalg.norm(c.pos - data.body(n).xpos) <= gate)
    taken, out = set(), {}
    for _d, i, n in pairs:
        if i in taken or n in out.values():
            continue
        taken.add(i)
        out[i] = n
    return out


def run(seed=None, aisle=0, n_per_row=6, speed=0.4, use_truth=False,
        max_stops=None, on_tick=None, on_event=None, verbose=True, scene=None,
        threshold=None):
    """Scout, judge, cut and crate. Returns a `Shift`.

    `scene` is an optional `(model, data, trusses)` a caller has already built —
    `farm.watch_truss` needs it for the reason `farm.run` documents.

    `threshold` overrides `truss.RIPE_FRACTION` for this run, which is what
    `--sweep` drives.
    """
    import mujoco

    from carrytrace import CarryTrace
    from incident import Blackbox
    from mission import Guard, Planner, park_arm, reset_park
    from outcomes import classify
    from plant_row import Row
    from reach import Gripper
    from week2_pick import Aborted, execute, make_reacher, anchor_posture

    th = ft.RIPE_FRACTION if threshold is None else float(threshold)

    def say(kind, **info):
        if on_event is not None:
            on_event(kind, **info)

    if scene is None:
        from farm import crop as fcrop

        seed = fcrop.resolve_seed(seed, label="truss crop")
        trusses = ft.spawn(n_per_row=n_per_row, seed=seed)
        model = ft.build(aisle=aisle, arms=("a",), trusses=trusses,
                         wrist_cam=True, deck_cam=True, seed=seed)
        data = mujoco.MjData(model)
    else:
        model, data, trusses = scene
    mujoco.mj_forward(model, data)
    park_q = armframe.park_posture(model, data)
    reset_park(model, data, park_q)

    names = [t.name for t in trusses]
    # ⚠️ `TRUSS_SNAP_N`, not `plant_row.SNAP_N`. A 0.8 kg cluster hangs at
    # 7.85 N and the loose crop's 12 N threshold breaks it under gravity plus
    # any handling transient — see the constant, which is the one number in this
    # crop taken from published measurement rather than from the Week 1 scene.
    row = Row(model, data, names=names, snap_n=ft.TRUSS_SNAP_N,
              homes={t.name: t.pos for t in trusses})
    mujoco.mj_forward(model, data)
    drive = trolley.Drive(model, data)

    left, right = house.serves(aisle)
    shift = Shift(ripe_truth=sum(1 for t in trusses
                                 if t.red_fraction >= th and t.row == right),
                  trusses=list(trusses), aisle=aisle)

    # --- 1. scout ------------------------------------------------------------
    if use_truth:
        cmap = truth_map(trusses, aisle=aisle)
        if verbose:
            print("\n  ⚠️ SCOUT SKIPPED — routing the operator's own answer")
    else:
        if verbose:
            print(f"\n{'=' * 78}\n  1. SCOUT — driving the aisle, mapping "
                  f"fruit, grouping them into trusses")
        say("phase", phase="SCOUT")
        cmap = survey(model, data, aisle=aisle, drive=drive, on_tick=on_tick,
                      on_frame=lambda *a: say("scout_frame", i=a[0], n=a[1],
                                              y=a[2], seen=a[3]),
                      verbose=verbose)
        if verbose:
            print(f"\n{cmap.table()}")
    shift.scout_s = drive.travelled / trolley.DRIVE_SPEED if not use_truth else 0.0
    shift.cluster_map = cmap
    say("map", cluster_map=cmap)

    # --- 2. plan -------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 78}\n  2. PLAN — which trusses are ripe enough, and "
              f"where to stop")
    say("phase", phase="PLAN")
    # ⚠️ The threshold is applied here, by stamping each cluster, because
    # `route.plan` filters on `.ripe` and a Cluster's `.ripe` reads the module
    # default. A run sweeping the threshold has to override it per cluster.
    for c in cmap.sightings:
        c.threshold = th
    r = route.plan(_ThresholdView(cmap, th), aisle=aisle, arm="a")
    if max_stops:
        r.stops = r.stops[:max_stops]
    shift.route = r
    if verbose:
        print(r.table())
    say("route", route=r)

    # --- 3. harvest ----------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 78}\n  3. HARVEST — {r.n_fruit} trusses in "
              f"{len(r.stops)} stops, cut at the stem")
    say("phase", phase="HARVEST")

    for si, stop in enumerate(r.stops, 1):
        say("stop", i=si, n=len(r.stops), y=stop.y, fruit=list(stop.fruit))
        if verbose:
            print(f"\n  --- stop {si}/{len(r.stops)} at y={stop.y:+.2f}, "
                  f"{stop.n} trusses ---")
        t_drive = drive.drive_to(stop.y, on_tick=on_tick)
        shift.drive_s += t_drive
        park_arm(model, data, park_q)
        mujoco.mj_forward(model, data)

        standing = [t.name for t in trusses if row.attached(t.name)]
        ident = associate(stop.fruit, model, data, standing)

        for fi, cluster in enumerate(stop.fruit):
            name = ident.get(fi)
            truth = next((t for t in trusses if t.name == name), None)
            rec = {"stop": si, "stop_y": stop.y,
                   "stage": cluster.stage, "row": cluster.row,
                   "seen": name is not None,
                   "n_red_seen": cluster.n_red, "n_seen": cluster.n_fruit,
                   "red_fraction": cluster.red_fraction}
            if truth is not None:
                rec["n_fruit"] = truth.n_fruit
                rec["n_green"] = sum(1 for s in truth.stages if s == "green")
                rec["n_red_true"] = truth.n_red
            if name is None:
                rec["outcome"] = "not_detected"
                shift.rows.append(rec)
                if verbose:
                    print(f"    truss at y={cluster.pos[1]:+.2f}: no truss "
                          f"within {NAME_GATE_M * 1000:.0f} mm")
                continue
            rec["fruit"] = name

            say("select", fruit=name, stage=cluster.stage, stop=si,
                left=len(stop.fruit) - fi, cluster=cluster)
            p0 = _time.perf_counter()
            with armframe.at_trolley(model, data) as fr:
                # ⚠️ `bin_drop_up`, and it is the difference between crating a
                # truss and dragging it through the crate wall. The default is
                # a loose fruit's 0.28 m, which puts the bottom of a cluster
                # hanging 0.27 m below the tool *under the crate floor*. See
                # `truss.CRATE_DROP_UP`, which derives it from the rachis —
                # and `truss.drop_height`, which raises it as the crate fills,
                # because the second truss of a stop is carried in over the
                # first one and struck it.
                #
                # ⚠️ `carry_axis="into_row"` — the wrist does **not** turn
                # down. A truss hangs from the collar the pads are holding, so
                # turning the tool through 90° swings 0.80 kg on a quarter-metre
                # lever and pries the cluster out of the pads. It hangs straight
                # down as gripped, and the release drops it straight in. See
                # `Planner.carry_axis`, which carries the measurement.
                drop_up = ft.drop_height(model, data, names, fr.bin_pos)
                # ⚠️ **The roll is left free here on purpose.** This is arm
                # a, whose wrist rolls to ~43 deg on its own and lifts the pad
                # clear of the cluster's top fruit — which is why every
                # single-armed seed crates. It is the *mirrored* arm that
                # cannot do that and needs help; see
                # `truss.MIRRORED_GRASP_ROLL` and `two_arm_farm_truss`.
                # Pinning a roll here was measured to cost picks.
                planner = Planner(model, data, row, lessons=None,
                                  clearance=0.040, park_q=park_q, speed=speed,
                                  frame=fr, bin_drop_up=drop_up,
                                  carry_axis="into_row")
                m = planner.plan(name)
            rec["drop_up"] = float(drop_up)
            rec["t_plan"] = _time.perf_counter() - p0

            if not m.ok:
                rec["refused"] = True
                rec["refused_reason"] = str(m.breaches[0] if m.breaches
                                            else "no route")
                rec["outcome"] = classify(rec)
                shift.rows.append(rec)
                say("refused", fruit=name, why=rec["refused_reason"])
                if verbose:
                    print(f"    {name}: REFUSED — {rec['refused_reason']}")
                continue

            reacher = make_reacher(model, data, speed=speed, frame=m.frame)
            armframe.pin_base(reacher)
            anchor_posture(reacher, model, data, park_q)
            gripper = Gripper(model, data)
            trace = CarryTrace(model, data, row, name,
                               Blackbox(model, data, row, name))
            guard = Guard(model, data, row, name)
            guard.armed = False
            # ⚠️ The blade. Without it the arm pulls a compliant weld to 75 N
            # and the release throws the cluster out of the pads — see
            # `truss.Cutter`, which carries the measurement.
            cutter = ft.Cutter(model, data, row, name, gripper)
            say("fly", fruit=name, mission=m, trace=trace, guard=guard,
                cutter=cutter)

            def tick(t=None, _tr=trace, _c=cutter):
                _tr.tick(t)
                _c.tick(t)
                if on_tick is not None:
                    on_tick(t)

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
            # ⚠️ **Re-scored on the fruit.** `week2_pick` scores `in_bin` with
            # `fr5.crate_contains(body.xpos, ...)`, and a truss body's origin is
            # its grasp point — 274 mm above the fruit once the cluster is
            # standing in the crate, so the height test fails on every correctly
            # crated truss. `truss.in_crate` asks the crate about the tomatoes.
            # Overridden here rather than in `week2_pick` so the loose-fruit
            # pipeline keeps scoring exactly as Weeks 1-4 measured it.
            res["in_bin"] = ft.in_crate(model, data, name,
                                        m.arm_frame.bin_pos)
            rec.update({
                "in_bin": bool(res["in_bin"]), "grasped": bool(res["grasped"]),
                "broke": bool(res["broke"]), "peak_n": float(res["peak_n"]),
                "lost": int(res["lost"]), "disturbed": int(res["disturbed"]),
                "aborted": str(aborted) if aborted else None,
                "peak_held_ms": trace.peak_held(),
                "t_fly": float(res["seconds"]),
                "cut_s": cutter.cut_at,
            })
            rec["clean"] = bool(res["in_bin"] and not res["lost"]
                                and not res["disturbed"] and not aborted)
            rec["outcome"] = classify(rec)
            shift.rows.append(rec)
            say("result", fruit=name, rec=rec)
            if verbose:
                print(f"    {name} ({rec.get('n_red_true', '?')}"
                      f"/{rec.get('n_fruit', '?')} red): "
                      f"{rec['outcome']:<14} crate={rec['in_bin']} "
                      f"{'cut' if cutter.cut else 'NOT CUT'} "
                      f"{rec['t_fly']:.1f}s")

    shift.drive_m = drive.travelled
    say("phase", phase="DONE")
    return shift


class _ThresholdView:
    """A map whose clusters answer `.ripe` at a caller-chosen threshold.

    ⚠️ Exists so `--sweep` can re-route the same map at several thresholds
    without mutating the clusters, and so `route.plan` needs no threshold
    parameter. `route.plan` reads `.sightings` and each sighting's `.ripe`; this
    hands it the same clusters wearing a different verdict.
    """

    def __init__(self, cmap, threshold):
        self.sightings = [_At(c, threshold) for c in cmap.sightings]
        self.aisle = cmap.aisle


class _At:
    """One cluster, judged at a given threshold. Delegates everything else."""

    def __init__(self, cluster, threshold):
        self._c = cluster
        self._th = threshold

    @property
    def ripe(self):
        return self._c.ripe_at(self._th)

    def __getattr__(self, k):
        return getattr(self._c, k)


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", action="store_true",
                    help="skip scouting and route the real crop")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=6, help="trusses per row")
    ap.add_argument("--stops", type=int, default=None, help="cap the stops")
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"red fraction to take a truss "
                         f"(default {ft.RIPE_FRACTION})")
    ap.add_argument("--recall", action="store_true",
                    help="score the cluster CV against the real trusses "
                         "and stop before harvesting")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    if args.recall:
        import mujoco

        from farm import crop as fcrop
        from mission import reset_park

        seed = fcrop.resolve_seed(args.seed, label="truss crop")
        trusses = ft.spawn(n_per_row=args.n, seed=seed)
        model = ft.build(aisle=args.aisle, arms=("a",), trusses=trusses,
                         wrist_cam=True, deck_cam=True, seed=seed)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        reset_park(model, data, armframe.park_posture(model, data))
        mujoco.mj_forward(model, data)
        cmap = survey(model, data, aisle=args.aisle, verbose=True)
        print(f"\n{cmap.table()}")

        s = ft.score(cmap.sightings, trusses)
        err = np.asarray(s["err_mm"], float)
        print(f"\n  --- the cluster CV against the truth ---")
        print(f"  trusses found          {s['found']}/{s['truth']}")
        print(f"  phantom clusters       {s['phantom']}")
        print(f"  right fruit count      {s['n_right']}/{s['found']}")
        print(f"  grasp point error      median {np.nanmedian(err):.0f} mm, "
              f"worst {np.nanmax(err):.0f} mm")
        matched = [c for c in cmap.sightings if c.truth is not None]
        agree = sum(1 for c in matched
                    if c.ripe == (c.truth.red_fraction >= ft.RIPE_FRACTION))
        print(f"  same take/leave verdict as the truth   {agree}/{len(matched)}")
        print(f"\n  ⚠️ A wrong verdict here is a *counting* error — the rule is "
              f"the same rule.\n     Colour is read per fruit and the fruit it "
              f"missed are the ones it under-counts.")
        return 0

    shift = run(seed=args.seed, aisle=args.aisle, n_per_row=args.n,
                speed=args.speed, use_truth=args.truth, max_stops=args.stops,
                threshold=args.threshold)
    shift.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
