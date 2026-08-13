#!/usr/bin/env python3
"""Two arms, one trolley, a house of **trusses** — mapped, judged, cut, crated.

`two_arm_farm.py` is this machine on **loose** fruit: two deck heads, two
routes merged into one itinerary, both arms flying inside one physics loop. This
is the same machine on tomatoes-on-the-vine, and everything that makes the
two-arm version hard is unchanged — the interlock on the shared deck centre, the
mandatory arm-vs-arm clearance, the one-clock/one-plant/two-control-laws
structure. `farm.duo` runs both; see its docstring, which is where the
concurrency argument lives.

What changes is the crop and the three things it drags with it:

    the unit      a truss, not a tomato. Six fruit on one rachis, cut once
    the decision  "is *enough* of this cluster red?" — `truss.RIPE_FRACTION`,
                  set from a sweep, not from taste
    the release   a **blade**. The cluster is severed at the stem, because
                  pulling 0.8 kg off a compliant weld throws it out of the pads

⚠️ **This file is a thin front end and that is deliberate.** `farm.duo.run`
grew three optional hooks — `snap_n`, `on_pick` and `truth_map` — rather than
being forked, so the concurrency engine that both crops depend on has exactly
one implementation. If two-arm harvesting has a bug, it has it in both sims and
gets fixed once. The panels are `two_arm_farm`'s own `Windows`, imported rather
than copied, for the same reason.

    ./.venv/bin/python simulation/mujoco/two_arm_farm_truss.py
    ./.venv/bin/python simulation/mujoco/two_arm_farm_truss.py --truth --stops 2
    ./.venv/bin/python simulation/mujoco/two_arm_farm_truss.py --headless
    ./.venv/bin/python simulation/mujoco/two_arm_farm_truss.py --shot

or, from the repo root, `python 2armfarmtruss.py` with the same flags.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from farm import decks, duo, house, trolley  # noqa: E402
from farm import truss as ft  # noqa: E402
from farm import trussrun as ftr  # noqa: E402


class TrussDuoScout(duo.DuoScout):
    """Both deck heads scanning, and the blobs grouped into trusses after.

    ⚠️ **Nothing about the perception changes — only what is done with it.**
    The heads, the aiming, the two-frames-from-one-physics-state discipline and
    the colour bands are all `duo.DuoScout`'s, inherited rather than restated.
    This overrides one method, and only to run `truss.group` over the fused
    result and hand back a map whose unit is a cluster.

    That is the whole reason `farm.route` needs no truss support: a
    `truss.Cluster` answers `.pos`, `.ripe`, `.row` and `.stage`, so the merge,
    the assignment and both arms' itineraries are the existing code working on
    a different object.
    """

    def run(self, drive, state=None, on_tick=None, on_frame=None, verbose=True):
        hm = super().run(drive, state=state, on_tick=on_tick,
                         on_frame=on_frame, verbose=verbose)
        clusters = ft.group(hm.sightings)
        for c in clusters:
            c.row = int(round((c.pos[0] - house.ROW_X0) / house.ROW_PITCH))
        keep = [c for c in clusters
                if 0 <= c.row < house.N_ROWS
                and abs(c.pos[0] - house.row_x(c.row)) < 0.25]
        return ftr.ClusterMap(sightings=keep, aisle=self.aisle,
                              drive_m=hm.drive_m, drive_s=hm.drive_s,
                              frames=hm.frames, blobs=len(hm.sightings))


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=8, help="trusses per row")
    ap.add_argument("--arms", type=int, default=2, choices=(1, 2))
    ap.add_argument("--stops", type=int, default=None, help="cap the stops")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None,
                    help="repeat a house; omitted draws a new one and prints it")
    ap.add_argument("--truth", action="store_true",
                    help="skip the mapping pass and route the real crop")
    ap.add_argument("--stride", type=float, default=None,
                    help="metres between mapping stops")
    ap.add_argument("--fps", type=int, default=10, help="panel rate")
    ap.add_argument("--hsv-hz", type=float, default=3.0,
                    help="how often the ripeness overlay re-detects")
    ap.add_argument("--panel-scale", type=float, default=1.0,
                    help="render live panels at this fraction of tile size")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"red fraction to cut a truss "
                         f"(default {ft.RIPE_FRACTION})")
    ap.add_argument("--shot", action="store_true",
                    help="render every panel once and exit")
    ap.add_argument("--headless", action="store_true",
                    help="run with no windows at all")
    ap.add_argument("--out", default=None,
                    help="record to <out>_sensors.mp4 and <out>_mission.mp4")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import mujoco

    import two_arm_farm as taf
    from farm import crop as fcrop

    arms = ("a", "b")[: args.arms]
    th = ft.RIPE_FRACTION if args.threshold is None else args.threshold
    seed = fcrop.resolve_seed(args.seed, label="truss crop")
    trusses = ft.spawn(n_per_row=args.n, seed=seed)
    ripe = sum(1 for t in trusses if t.red_fraction >= th)
    print(f"  house: {len(trusses)} trusses ({sum(t.n_fruit for t in trusses)} "
          f"fruit, {ft.TRUSS_MASS} kg each), {ripe} ripe at {th:.2f} · arms "
          f"{', '.join(a.upper() for a in arms)} · seed {seed}")

    model = ft.build(aisle=args.aisle, arms=arms, trusses=trusses,
                     wrist_cam=True, arm_decks=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = duo.DuoState(arms=arms, aisle=args.aisle)
    state.trusses = list(trusses)

    if args.shot:
        from mission import park_arm, reset_park

        from farm import armframe

        parks = {t: armframe.park_posture(model, data, t, arms=arms)
                 for t in arms}
        reset_park(model, data, parks[arms[0]],
                   prefix=trolley.ARM_PREFIX[arms[0]])
        for t in arms:
            park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
        hs = decks.heads(model, data, arms=arms, aisle=args.aisle)
        for t in arms:
            state.state[t].head = hs.get(t)
        if "a" in hs:
            hs["a"].aim(data, -decks.SCAN_HALF_DEG)
        if "b" in hs:
            hs["b"].aim(data, +decks.SCAN_HALF_DEG)
        mujoco.mj_forward(model, data)
        taf.where_is_everything(model, data, state)
        taf.shot(model, data, state)
        return 0

    windows = None if args.headless else taf.Windows(
        model, data, state, fps=args.fps, hsv_hz=args.hsv_hz, out=args.out,
        arms=arms, panel_scale=args.panel_scale)
    on_tick = None if windows is None else windows.tick

    # The blade, one per mission. See `truss.Cutter` for why a truss is cut
    # rather than pulled, and `duo.run`'s `on_pick` for why it is registered on
    # the machine's per-cycle watchers rather than inside a leg.
    cutters = {}

    # `duo.run` builds the `Row` itself, so the blade needs a handle on it.
    # `after_reset` is the one hook that is handed it, and it fires long before
    # the first pick — so by the time `on_pick` runs this is populated.
    #
    # ⚠️ The tool site **and the pad geoms** are the arm's own, via `prefix`.
    # `tool0` unprefixed is arm a's, and a blade that measured its proximity
    # against arm a while arm b was doing the cutting would refuse to cut
    # anything arm b reached — the same trap `week2_pick.execute` documents at
    # length for the grasp check. The pads are the same trap one level down:
    # the blade now fires on measured pad load (see `truss.CUT_HOLD_N`), so
    # arm b reading arm a's pads would never see a grip at all. One `prefix`
    # feeds both.
    _row = [None]

    def grab_row(row):
        _row[0] = row

    def on_pick(tag, name, gripper, prefix):
        c = ft.Cutter(model, data, _row[0], name, gripper, prefix=prefix)
        cutters[name] = c
        return c.tick

    # ⚠️ **The carry is the truss's, not the loose crop's, and both halves of
    # it are the same argument.** A cluster hangs `RACHIS_LEN` below the collar
    # the pads are holding, so the release height has to clear the crate *and
    # whatever is already in it* (`truss.drop_height`), and the wrist must not
    # turn the cluster over on the way (`carry_axis`). `farm.trussrun` passes
    # exactly these; this is the two-armed spelling of it, and each arm asks
    # about its own crate — one crate per arm, mirrored with it. See
    # `trolley._crate_local`.
    _names = [t.name for t in trusses]

    def plan_opts(model_, data_, tag, fr):
        return {"bin_drop_up": ft.drop_height(model_, data_, _names,
                                              fr.bin_pos),
                "carry_axis": "into_row"}

    def score_in_bin(model_, data_, name, fr):
        return ft.in_crate(model_, data_, name, fr.bin_pos)

    t0 = time.perf_counter()
    try:
        duo.run(model, data, trusses, state, arms=arms, aisle=args.aisle,
                speed=args.speed, use_truth=args.truth, max_stops=args.stops,
                on_tick=on_tick, stride=args.stride, verbose=True,
                scout_cls=TrussDuoScout, snap_n=ft.TRUSS_SNAP_N,
                on_pick=on_pick, after_reset=grab_row,
                truth_map=lambda ts, aisle: ftr.truth_map(ts, aisle=aisle),
                plan_opts=plan_opts, score_in_bin=score_in_bin)
        if windows is not None:
            windows.flush(args.fps * 3)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        wall = time.perf_counter() - t0
        if windows is not None:
            windows.close()
        print(f"\n  {wall:.0f} s wall")

    duo.report(state)
    cut = sum(1 for c in cutters.values() if c.cut)
    print(f"\n  the blade went through on {cut}/{len(cutters)} attempts")
    print(f"  ⚠️ A truss that was reached but not cut is a truss the pads "
          f"never closed on.\n     `truss.Cutter` requires grip AND proximity, "
          f"so it cannot sever a neighbour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
