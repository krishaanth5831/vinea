#!/usr/bin/env python3
"""Both arms' eyes, side by side, with the ripeness call drawn on them live.

    +------------------------+------------------------+
    | arm A wrist cam        | arm B wrist cam        |
    | ripe/unripe boxes      | ripe/unripe boxes      |
    +------------------------+------------------------+
    | deck cam - the mapping pass down the aisle      |
    | + what the head is doing right now              |
    +------------------------+------------------------+

`farm/watch.py` shows a whole shift and gives the wrist camera one tile out of
six. This shows the *perception* instead: two arms, two eyes, one head, at a
size where the boxes are readable. It is the view for answering "what does the
robot think it is looking at", which is a different question from "is the shift
going well".

⚠️ **The classifier is `farm.overlay`, which is the Week 3 colour control.** No
model was trained for this and none is loaded. See that module for why the
overlay must not run a *better* detector than the robot uses.

⚠️ **Arm B's panel is black until arm B is fitted.** It is drawn as an explicit
"not fitted" card rather than left blank or filled with arm A's view, because a
split view that silently shows the same camera twice is the exact failure this
whole file exists to make visible.

The deck panel is the articulated head, and it is *turning* while this runs —
each stop it faces arm A's row and then arm B's, which is what puts fruit in
front of arm B at all. See `farm/scout.py`.

    ./.venv/bin/python simulation/mujoco/farm/eyes.py             # both arms
    ./.venv/bin/python simulation/mujoco/farm/eyes.py --arms 1    # arm A only
    ./.venv/bin/python simulation/mujoco/farm/eyes.py --out eyes.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import armframe, house, overlay, scout, trolley  # noqa: E402

TILE_W, TILE_H = 640, 480
INK = (232, 232, 232)
DIM = (130, 130, 130)


def _label(img, text, sub=None):
    """Caption a panel. Same convention as `farm.watch._label`."""
    import cv2

    h = img.shape[0]
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (18, 18, 18), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, INK, 1,
                cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    DIM, 1, cv2.LINE_AA)
    return img


def _missing(w, h, text):
    """A panel for an arm that is not on the machine."""
    import cv2

    img = np.full((h, w, 3), 20, np.uint8)
    cv2.putText(img, text, (16, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "run with --arms 2 to fit it", (16, h // 2 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (70, 70, 70), 1, cv2.LINE_AA)
    return img


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", type=int, default=2, choices=(1, 2))
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=14, help="fruit per row")
    ap.add_argument("--stride", type=float, default=scout.SCOUT_STRIDE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default=None, help="record instead of showing")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import cv2
    import mujoco

    from mission import reset_park
    from reach import CTRL_DT
    from week3_watch import Sink

    arms = ("a", "b")[: args.arms]
    seed = fcrop.resolve_seed(args.seed)
    trusses = fcrop.spawn(n_per_row=args.n, seed=seed)
    model = trolley.build(aisle=args.aisle, arms=arms, trusses=trusses,
                          wrist_cam=True, deck_cam=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Park every fitted arm, each in its own frame. An unparked arm sits at
    # HOME, which points its wrist camera at the ceiling.
    for tag in arms:
        q = armframe.park_posture(model, data, tag, arms=arms)
        reset_park(model, data, q, prefix=trolley.ARM_PREFIX[tag])
    for tag in arms:
        from mission import park_arm

        park_arm(model, data, armframe.park_posture(model, data, tag,
                                                    arms=arms),
                 prefix=trolley.ARM_PREFIX[tag])
    mujoco.mj_forward(model, data)

    cams = {tag: (trolley.ARM_PREFIX[tag] + "wrist") for tag in arms}
    renderers = {name: mujoco.Renderer(model, height=TILE_H, width=TILE_W,
                                       max_geom=30000)
                 for name in list(cams.values()) + [scout.SCOUT_CAM]}
    detector = scout.StageDetector()

    left, right = house.serves(args.aisle)
    rows = {"a": right, "b": left}
    sink = Sink(live=args.out is None, out=args.out, fps=args.fps,
                title="vinea - both arms' eyes")
    sc = scout.Scout(model, data, aisle=args.aisle, stride=args.stride,
                     arms=arms, detector=detector)
    drive = trolley.Drive(model, data)
    state = {"n": 0, "seen": 0}
    every = max(1, int(round((1.0 / args.fps) / CTRL_DT)))

    def compose():
        tiles = []
        for tag in ("a", "b"):
            if tag not in arms:
                tiles.append(_missing(TILE_W, TILE_H,
                                      f"arm {tag.upper()} not fitted"))
                continue
            name = cams[tag]
            renderers[name].update_scene(data, camera=name)
            img = cv2.cvtColor(renderers[name].render(), cv2.COLOR_RGB2BGR)
            counts = overlay.annotate(img, detector=detector)
            overlay.tally(img, counts)
            _label(img, f"arm {tag.upper()} wrist cam  -  row r{rows[tag]}")
            tiles.append(img)

        renderers[scout.SCOUT_CAM].update_scene(data, camera=scout.SCOUT_CAM)
        deck = cv2.cvtColor(renderers[scout.SCOUT_CAM].render(),
                            cv2.COLOR_RGB2BGR)
        counts = overlay.annotate(deck, detector=detector)
        overlay.tally(deck, counts)
        pan = sc.head.current(data)[0] if sc.head is not None else 0.0
        facing = min(rows, key=lambda t: abs(
            ((scout.ROW_PAN[t] - pan + 180) % 360) - 180)) if arms else "a"
        wide = np.hstack(tiles)
        # ⚠️ Resize *then* caption. Captioning first and stretching the panel to
        # the full width blows the text up with it, which looks like a bug in
        # the viewer and is one in the ordering.
        deck = cv2.resize(deck, (wide.shape[1], TILE_H))
        _label(deck, f"deck cam  -  head at pan {pan:+.0f} deg, "
                     f"facing r{rows.get(facing, right)}  |  "
                     f"{state['seen']} fruit mapped so far")
        return np.vstack([wide, deck])

    def tick(_t=None):
        state["n"] += 1
        if state["n"] % every:
            return
        sink.push(compose())

    def on_frame(i, n, y, seen, _rgb):
        state["seen"] = len(seen)
        tick()

    print(f"  {len(trusses)} fruit, {sum(1 for t in trusses if t.ripe)} ripe · "
          f"arms {', '.join(a.upper() for a in arms)}\n")
    try:
        hm = sc.run(drive, on_tick=tick, on_frame=on_frame)
        state["seen"] = len(hm.sightings)
        for _ in range(args.fps * 2):
            tick()
        print(f"\n  {len(hm.sightings)} fruit mapped, "
              f"{sum(1 for s in hm.sightings if s.ripe)} of them ripe")
        per = {}
        for s in hm.sightings:
            per[s.row] = per.get(s.row, 0) + 1
        for tag in arms:
            print(f"    arm {tag.upper()}'s row r{rows[tag]}: "
                  f"{per.get(rows[tag], 0)} mapped")
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        sc.close()
        sink.close()
        for r in renderers.values():
            r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
