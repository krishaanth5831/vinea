#!/usr/bin/env python3
"""Where can this arm ACTUALLY pick, in the scene it is actually working in?

`week4_place.py` inherited its placement envelope from `week1_mousereach.py`,
which measured it cell by cell on a 31x24 grid — and measured it against a
**different scene**:

    Week 1   board at x=0.55, crate at y=-0.80
    Week 2+  row   at x=0.60, crate at y=-0.52   <- 280 mm closer

That matters more than it looks. `week1_mousereach.py` records why the crate is
at -0.80 in the first place: at y=-0.55 it sat directly under the path the arm
sweeps while pulling fruit loose, the gripper hit its wall on the way to the
pre-grasp point, and the pick aborted 179 mm short. Week 2 then moved the crate
back in to -0.52 for its own reasons, and nobody re-measured the envelope.

So the "guaranteed pickable" rectangle is a guarantee about a scene that no
longer exists. This re-measures it against the real one, the same way: put one
fruit at each grid point, fly a whole pick, and record whether it reaches the
crate.

This is bug #5 in the bug log — "the every-click-is-pickable guarantee is
unguarded" — being paid off.

    ./.venv/bin/python simulation/mujoco/week4_envelope.py            # 7x5, ~18 min
    ./.venv/bin/python simulation/mujoco/week4_envelope.py --ny 9 --nz 7
    ./.venv/bin/python simulation/mujoco/week4_envelope.py --out envelope.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# What each cell prints in the map.
MARK = {"clean": "O", "carry_dropped": "d", "carry_ejected": "X",
        "grasp_failed": "g", "unreachable": "u", "refused": "r",
        "guard_abort": "a", "collateral": "c", "unknown": "?"}


def probe(y, z, speed=None):
    """One fruit at (y, z), one full pick. Returns the outcome bucket.

    A fresh model per cell. Rebuilding costs ~1 s against a ~30 s pick and it
    guarantees no state leaks between cells — the failure mode bug 39 was.
    """
    import mujoco

    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from plant_row import Row
    from week4_place import Crop, harvest_placed, park_spot, pool_trusses

    pool = pool_trusses()
    model = build_scene(trusses=pool)
    data = mujoco.MjData(model)
    names = [n for n, _, _ in pool]
    row = Row(model, data, names=names,
              homes={n: park_spot(i) for i, n in enumerate(names)})
    q = park_posture(model)
    reset_park(model, data, q)
    row.reset()
    mujoco.mj_forward(model, data)

    crop = Crop(model, data, row, names)
    # Bypass the spacing/zone check deliberately: this measurement is what
    # *defines* the zone, so it must be free to probe outside the current one.
    name = crop.free()[0]
    import numpy as np

    from plant_row import ROW_X

    pos = np.array([ROW_X, y, z])
    row.place(name, pos)
    data.eq_active[row.eq_id[name]] = 1
    crop._show(name, True)
    row.home[name] = pos.copy()
    crop.placed[name] = pos
    crop.zones[name] = "probe"
    crop.version += 1
    mujoco.mj_forward(model, data)

    rows = harvest_placed(model, data, row, crop, q, speed=speed)
    return rows[0]["outcome"] if rows else "unknown"


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ny", type=int, default=7, help="grid columns in y")
    ap.add_argument("--nz", type=int, default=5, help="grid rows in z")
    ap.add_argument("--y-range", type=float, nargs=2, default=(-0.55, 0.55))
    ap.add_argument("--z-range", type=float, nargs=2, default=(0.15, 0.95))
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--out", default="runs/envelope.json")
    args = ap.parse_args()

    import numpy as np

    ys = np.linspace(*args.y_range, args.ny)
    zs = np.linspace(*args.z_range, args.nz)
    total = args.ny * args.nz
    print(f"\n{'=' * 78}")
    print(f"  ENVELOPE SWEEP — {args.ny} x {args.nz} = {total} cells, one full "
          f"pick each")
    print(f"  y {args.y_range[0]:+.2f}..{args.y_range[1]:+.2f}   "
          f"z {args.z_range[0]:.2f}..{args.z_range[1]:.2f}")
    print(f"  expect roughly {total * 32 / 60:.0f} min\n")

    grid = {}
    t0 = time.perf_counter()
    done = 0
    for z in zs:
        for y in ys:
            out = probe(float(y), float(z), speed=args.speed)
            grid[(round(float(y), 4), round(float(z), 4))] = out
            done += 1
            print(f"  [{done:>3}/{total}] y{y:+.3f} z{z:.3f} -> {out}"
                  f"   ({time.perf_counter() - t0:.0f}s)", flush=True)

    # --- the map -------------------------------------------------------------
    print(f"\n{'=' * 78}\n  PICKABLE MAP   O=clean  d=dropped  X=ejected  "
          f"g=grasp failed\n                 u=unreachable  r=refused  "
          f"a=guard abort\n")
    print("        " + "".join(f"{y:+7.2f}" for y in ys))
    for z in reversed(zs):
        cells = "".join(
            f"{MARK.get(grid[(round(float(y), 4), round(float(z), 4))], '?'):>7}"
            for y in ys)
        print(f"  z{z:5.2f} {cells}")

    clean = [(y, z) for (y, z), o in grid.items() if o == "clean"]
    print(f"\n  {len(clean)}/{total} cells picked cleanly "
          f"({100 * len(clean) / total:.0f}%)")

    if clean:
        cy = [y for y, _ in clean]
        cz = [z for _, z in clean]
        print(f"  clean cells span y {min(cy):+.2f}..{max(cy):+.2f}   "
              f"z {min(cz):.2f}..{max(cz):.2f}")
        # Which columns are entirely clean — the honest basis for a rectangle.
        good_y = [y for y in ys
                  if all(grid[(round(float(y), 4), round(float(z), 4))] == "clean"
                         for z in zs)]
        good_z = [z for z in zs
                  if all(grid[(round(float(y), 4), round(float(z), 4))] == "clean"
                         for y in ys)]
        print(f"  fully clean columns (y): "
              f"{', '.join(f'{y:+.2f}' for y in good_y) or 'none'}")
        print(f"  fully clean rows    (z): "
              f"{', '.join(f'{z:.2f}' for z in good_z) or 'none'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"y_range": list(args.y_range), "z_range": list(args.z_range),
         "ny": args.ny, "nz": args.nz,
         "cells": [{"y": y, "z": z, "outcome": o}
                   for (y, z), o in grid.items()]}, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
