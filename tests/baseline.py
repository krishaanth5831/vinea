#!/usr/bin/env python3
"""Capture Week 1's two pick demos as text, so a refactor can be held to them.

    ./.venv/bin/python tests/baseline.py --write     # record
    ./.venv/bin/python tests/baseline.py             # compare, exit 1 on drift

Bug log entry 7 asks for the duplicated pick cycle to be extracted, and says
Week 1's numbers must come back unchanged. They cannot be: the 42/42 was 42
hand-run mouse clicks and the 6/6 was a six-position sweep, and neither has a
command in this repo any more. That is bug 6's own sentence about itself.

So this is the bar instead, and it is a stricter one than either count. It runs
both cycles over a fixed set of points and records **every line they print** —
each waypoint, each arrival error in millimetres, each snap force, each verdict.
A crated count can stay at 42 while every arm inside it moves differently; this
cannot.

⚠️ **Text, not video.** Two runs of identical code produce different `.mp4`
hashes — the encoder is not deterministic — so a rendered artifact cannot be a
regression bar. It nearly reported a regression that was not there.

The points are chosen to exercise the branches, not to sample the board evenly:
the four corners and the centre for the reachable extremes, two that force the
long +y-to-crate move the TRANSIT waypoint exists for (entry 17), and the low
edge where the crate sits under the pull path (entry 15).
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

HERE = Path(__file__).resolve().parent / "baselines"

# (y, z) on the board. See the module docstring for why these ones.
CLICKS = [
    (0.00, 0.60),    # centre — the default placement
    (-0.50, 0.20),   # low left  \  the corners of the advertised board
    (0.50, 0.20),    # low right  |
    (-0.50, 0.90),   # high left  |
    (0.50, 0.90),    # high right /
    (0.30, 0.85),    # high +y — the long carry TRANSIT exists for (entry 17)
    (0.45, 0.75),    # again, further out
    (-0.45, 0.30),   # low -y, over the crate side (entry 15)
    (0.00, 0.15),    # bottom edge
    (0.00, 0.95),    # top edge
    (-0.20, 0.45),   # off-centre mid
    (0.20, 0.70),    # off-centre high
]


def capture_mousereach():
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

    out = io.StringIO()
    with redirect_stdout(out):
        crated = 0
        for y, z in CLICKS:
            pos = mr.tomato_on_panel(mr.panel_to_world(y, z))
            mr.place_tomato(model, data, eq_id, pos)
            print(f"\n=== click ({y:+.2f}, {z:+.2f}) ===")
            r = mr.pick_cycle(reacher, gripper, eq_id, pos, park)
            crated += bool(r["in_crate"])
            print(f"  result reached={r['reached']} snapped={r['snapped']} "
                  f"in_crate={r['in_crate']} peak_n={r['peak_n']:.4f} "
                  f"tomato={r['tomato'].round(6).tolist()}")
        print(f"\n{crated}/{len(CLICKS)} crated")
    return out.getvalue()


def capture_gripper():
    import mujoco
    import week1_gripper as wg
    from fr5 import MAX_REACH_GRIPPER
    from reach import Gripper, Reacher

    model = wg.build_scene()
    data = mujoco.MjData(model)
    reacher = Reacher(model, data, standoff=0.0, max_reach=MAX_REACH_GRIPPER,
                      reached_mm=wg.REACHED_MM_LOADED)
    gripper = Gripper(model, data)
    gripper.open()

    out = io.StringIO()
    with redirect_stdout(out):
        r = wg.run_pick(reacher, gripper)
        print(f"  result grasped={r['grasped']} in_bin={r['in_bin']} "
              f"tomato={r['tomato'].round(6).tolist()}")
    return out.getvalue()


CAPTURES = {
    "week1_mousereach": capture_mousereach,
    "week1_gripper": capture_gripper,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="record the current behaviour as the baseline")
    args = ap.parse_args()

    HERE.mkdir(exist_ok=True)
    drift = []
    for name, fn in CAPTURES.items():
        got = fn()
        path = HERE / f"{name}.txt"
        if args.write:
            path.write_text(got)
            print(f"  wrote {path.relative_to(REPO)}  ({len(got.splitlines())} lines)")
            continue
        if not path.exists():
            print(f"  [MISS] {name}: no baseline — run with --write")
            drift.append(name)
            continue
        want = path.read_text()
        if got == want:
            print(f"  [SAME] {name}  ({len(got.splitlines())} lines byte-identical)")
        else:
            drift.append(name)
            wl, gl = want.splitlines(), got.splitlines()
            print(f"  [DRIFT] {name}")
            shown = 0
            for i in range(max(len(wl), len(gl))):
                a = wl[i] if i < len(wl) else "<missing>"
                b = gl[i] if i < len(gl) else "<missing>"
                if a != b:
                    print(f"      line {i + 1}\n        was: {a}\n        now: {b}")
                    shown += 1
                    if shown == 20:
                        print("      ... (truncated)")
                        break
    if args.write:
        return 0
    if drift:
        print(f"\n{len(drift)} capture(s) drifted: {', '.join(drift)}")
        return 1
    print(f"\nAll {len(CAPTURES)} captures byte-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
