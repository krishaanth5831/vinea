#!/usr/bin/env python3
"""Walk a grid over the mousereach board and check every point actually crates.

    ./.venv/bin/python tests/board_walk.py            # 42 points, ~40 s
    ./.venv/bin/python tests/board_walk.py --grid 5 4 # coarser, for a quick look

Bug log entry 5. `week1_mousereach.py` advertises, in its module docstring and
on screen in the banner, that **every click on the board is pickable**. That is
a real guarantee to make to somebody else driving, and nothing checks it.

`PANEL_HALF_Y`, `PANEL_Z_LO` and `PANEL_Z_HI` were measured on a 31x24 grid
against one specific gripper, one `APPROACH_GAP`, one `RETRACT_GAP` and one
crate position. Change any of those and the board silently contains points that
abort. It has already happened twice:

  * entry 15 — widening the board to y +/-0.55 put the crate inside the arm's
    pull path, and picks aborted 179 mm short;
  * entry 42 — `week4_place` inherited this envelope into a scene where the
    crate had moved, and only 21 of 49 cells picked cleanly. Nobody
    re-measured, and the "arm keeps dropping tomatoes" complaint was mostly
    fruit placed where no pick was ever possible.

⚠️ **If this fails, the answer is not to shrink the board.** The board is a
claim about the arm; a failing point is the claim being wrong, and moving the
boundary in to cover it up is exactly what entry 42 records someone having
effectively done. Print the failures, decide deliberately.

Not an inner-loop test — this is ~40 seconds against `tests/test_sim.py`'s one.
Nightly, or before a commit that touches the gripper, the gaps, the crate or
the board.
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "simulation" / "mujoco"))

import numpy as np  # noqa: E402


def walk(cols, rows, verbose=False):
    import mujoco
    import week1_mousereach as mr
    from fr5 import MAX_REACH_GRIPPER
    from reach import Gripper, Reacher

    model = mr.build_scene()
    data = mujoco.MjData(model)
    reacher = Reacher(model, data, standoff=0.0, max_reach=MAX_REACH_GRIPPER,
                      reached_mm=mr.REACHED_MM_LOADED, mocap=None)
    gripper = Gripper(model, data)
    gripper.open()
    eq_id = model.equality("peduncle").id
    park = mr.park_pose(model)

    # The board's own advertised extent, endpoints included. A guarantee about
    # "anywhere on the board" has to be tested at the edges — that is where
    # entry 15 bit, and an inset grid would have missed it.
    ys = np.linspace(-mr.PANEL_HALF_Y, mr.PANEL_HALF_Y, cols)
    zs = np.linspace(mr.PANEL_Z_LO, mr.PANEL_Z_HI, rows)

    results = {}
    for z in zs:
        for y in ys:
            pos = mr.tomato_on_panel(mr.panel_to_world(float(y), float(z)))
            mr.place_tomato(model, data, eq_id, pos)
            sink = io.StringIO()
            with redirect_stdout(sink):
                r = mr.pick_cycle(reacher, gripper, eq_id, pos, park,
                                  verbose=False)
            results[(round(float(y), 3), round(float(z), 3))] = r
            if verbose:
                print(f"  ({y:+.2f}, {z:.2f})  "
                      f"reached={r['reached']} snapped={r['snapped']} "
                      f"crated={r['in_crate']}  peak {r['peak_n']:5.1f} N")
    return ys, zs, results


def render(ys, zs, results):
    """The board as it actually behaves, drawn the way it is drawn in COMMANDS."""
    print(f"\n  {'':>6}" + "".join(f"{y:+7.2f}" for y in ys))
    for z in reversed(zs):
        cells = []
        for y in ys:
            r = results[(round(float(y), 3), round(float(z), 3))]
            if r["in_crate"]:
                cells.append("O")          # crated
            elif not r["reached"]:
                cells.append("x")          # never got to the pre-grasp point
            elif not r["snapped"]:
                cells.append("s")          # gripped, stem never gave
            else:
                cells.append("d")          # picked and dropped
        print(f"  z {z:.2f}" + "".join(f"{c:>7}" for c in cells))
    print("\n  O crated   x could not reach   s stem held   d dropped en route")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs=2, type=int, default=(7, 6),
                    metavar=("COLS", "ROWS"),
                    help="points across and up the board (default 7 6 = 42)")
    ap.add_argument("--verbose", action="store_true", help="a line per point")
    args = ap.parse_args()

    cols, rows = args.grid
    import week1_mousereach as mr
    print(f"Walking the week1_mousereach board: y ±{mr.PANEL_HALF_Y:.2f}, "
          f"z {mr.PANEL_Z_LO:.2f}–{mr.PANEL_Z_HI:.2f}")
    print(f"  {cols} x {rows} = {cols * rows} points · crate at "
          f"[{mr.CRATE_POS[0]:.2f} {mr.CRATE_POS[1]:.2f}] · "
          f"approach {mr.APPROACH_GAP:.2f} m · retract {mr.RETRACT_GAP:.2f} m")

    ys, zs, results = walk(cols, rows, args.verbose)
    render(ys, zs, results)

    failed = [(k, r) for k, r in results.items() if not r["in_crate"]]
    n = len(results)
    print(f"\n  {n - len(failed)}/{n} crated")
    if not failed:
        print("  the board's guarantee holds at every point tested.")
        return 0

    print(f"\n  ⚠️  {len(failed)} point(s) on the advertised board did NOT crate.")
    print("  Do not move the board to cover these. Watch one and see why:\n")
    for (y, z), r in failed:
        why = ("could not reach the pre-grasp point" if not r["reached"]
               else "stem never gave" if not r["snapped"]
               else f"dropped, fruit ended at {r['tomato'].round(3).tolist()}")
        print(f"    ({y:+.2f}, {z:.2f})  {why}")
    print("\n    ./.venv/bin/python simulation/mujoco/week1_mousereach.py "
          f"--click {failed[0][0][0]:.2f} {failed[0][0][1]:.2f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
