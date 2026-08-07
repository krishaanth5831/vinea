#!/usr/bin/env python3
"""How much does kg/hr move when SNAP_N moves?

`SNAP_N = 12 N` is the softest input to the whole throughput figure. It decides
whether a pick succeeds, so it flows straight into the clean rate and therefore
into kg/hr — and it was set by the *simulator* rather than by a plant. Gripping
an attached fruit loads the stem 6.0-8.6 N by itself, so anything under ~9 N
detaches on contact; 12 N is above the 5-10 N the MVP doc suggested, and the
reason is the tool, not the tomato.

That is hard to defend on its own. "The number moves by X% across a plausible
9-20 N range" is enormously more defensible than one figure at 12 N, so this
sweeps it and reports the sensitivity.

⚠️ SNAP_N is only ever read by `Row.update()` — `add_row` takes it but does not
use it — so sweeping it needs no recompile and no scene change. The layout is
held fixed across the sweep so the only thing varying is the threshold.

⚠️ The Week 4 instructions say to turn this one **by hand**, because Week 4 is
failure diagnosis and the intuition matters more than the value. Running this
script produces the number; it does not produce the intuition.

    ./.venv/bin/python simulation/mujoco/week4_snap.py
    ./.venv/bin/python simulation/mujoco/week4_snap.py --n 8 --values 9,12,16,20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="fruit per layout")
    ap.add_argument("--values", default="9,12,16,20",
                    help="SNAP_N values in newtons")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/snap_sweep.jsonl")
    args = ap.parse_args()

    import mujoco

    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from outcomes import instability
    from picklog import PickLog, throughput
    from plant_row import Row
    from week4_place import (Crop, auto_layout, harvest_placed, park_spot,
                             pool_trusses)

    values = [float(v) for v in args.values.split(",")]
    layout = auto_layout(args.n, seed=args.seed)   # held fixed across the sweep

    print(f"\n{'=' * 78}")
    print(f"  SNAP_N SENSITIVITY — {args.n} fruit, one fixed layout, "
          f"seed {args.seed}")
    print(f"  values: {', '.join(f'{v:g} N' for v in values)}")

    log = PickLog(args.out, meta={"layout": f"snap_s{args.seed}"})
    results = []
    for snap in values:
        pool = pool_trusses()
        model = build_scene(trusses=pool)
        data = mujoco.MjData(model)
        names = [nm for nm, _, _ in pool]
        row = Row(model, data, names=names, snap_n=snap,
                  homes={nm: park_spot(i) for i, nm in enumerate(names)})
        park_q = park_posture(model)
        reset_park(model, data, park_q)
        row.reset()
        mujoco.mj_forward(model, data)

        crop = Crop(model, data, row, names)
        crop.apply(layout)
        print(f"\n{'-' * 78}\n  SNAP_N = {snap:g} N")
        log.meta["snap_n"] = snap
        rows = harvest_placed(model, data, row, crop, park_q, log=log,
                              seed=args.seed)
        t = throughput(rows)
        fired, lucky, measured = instability(rows)
        results.append((snap, rows, t, fired, measured))

    print(f"\n{'=' * 78}\n  SENSITIVITY OF kg/hr TO A NUMBER NOBODY MEASURED\n")
    print(f"  {'SNAP_N':>8} {'clean':>8} {'cycle s':>9} {'kg/hr':>8} "
          f"{'unstable':>9}")
    for snap, rows, t, fired, measured in results:
        c = sum(1 for r in rows if r.get("outcome") == "clean")
        print(f"  {snap:>6g} N {c}/{len(rows):<6} "
              f"{t['mean_s'] if t else 0:>9.1f} {t['kg_hr'] if t else 0:>8.1f} "
              f"{fired}/{measured:<8}")

    kgs = [t["kg_hr"] for _s, _r, t, _f, _m in results if t]
    if len(kgs) > 1 and max(kgs) > 0:
        spread = (max(kgs) - min(kgs)) / max(kgs) * 100
        print(f"\n  kg/hr spans {min(kgs):.1f}-{max(kgs):.1f} across "
              f"{min(values):g}-{max(values):g} N — a {spread:.0f}% swing on a "
              f"number set by the simulator.")
        print(f"  Quote the throughput with that range attached, not without it.")
    log.close()


if __name__ == "__main__":
    main()
