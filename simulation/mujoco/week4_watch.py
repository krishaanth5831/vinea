#!/usr/bin/env python3
"""Click tomatoes onto the green area, then watch four camera panels harvest them.

Week 3's `week4_watch`-shaped sibling: the same OpenCV four-panel window, with
Week 4's free placement in front of it. Put up to 15 fruit anywhere on the green
board by clicking it, press SPACE, and watch the scan, the plan and the picks in
the deck camera, the eye-in-hand camera, a clean scene shot and a live stats
panel — all at once.

    ./.venv/bin/python simulation/mujoco/week4_watch.py

    +----------------------+----------------------+
    | deck cam + the plan  | wrist cam, live      |
    | CLICK HERE TO PLACE  | detections + error   |
    +----------------------+----------------------+
    | clean scene shot     | stats / placements   |
    +----------------------+----------------------+

**Clicking works by ray-casting, not by guessing.** A click in the deck panel is
turned into a ray through that camera's pinhole (`camera.pixel_ray`, the same
maths Week 3's deprojection gate is built on) and intersected with the board
plane. So the tomato lands exactly where the cursor was, from any camera angle,
and moving the deck camera does not break it.

⚠️ **The green board is the arm's actual working area** — y +/-0.55, z 0.15-0.95,
measured cell by cell on a 31x24 grid in Week 1. Anywhere on it is reachable by
construction. Clicks off the board, or closer than 200 mm to an existing fruit,
are refused with the reason printed on the stats panel rather than silently
ignored.

Quit with the QUIT button, q, Esc, or the window's X — all four work, and
whatever picks finished are still reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from week4_place import (Crop, GUARANTEED_HALF_Y, GUARANTEED_Z, MAX_FRUIT,  # noqa: E402
                         MIN_SPACING, PLACE_BOARD, auto_layout, check,
                         harvest_placed, park_spot, pool_trusses, random_seed)

# The four-panel layout from week3_watch.Views.frame():
#     [ deck | wrist ]
#     [ scene | stats ]
# Each tile is TILE_W x TILE_H, so the deck panel — the one you click — is the
# top-left quadrant.
DECK_TILE = (0, 0)

# Watching speed. The measurement tools stay at reach.DEFAULT_SPEED (0.15) so
# their numbers remain comparable with every earlier week; this is for sitting
# and looking at it, where 30 s a pick is a long time to stare at a screen.
WATCH_SPEED = 0.4


def panel_pixel(x, y, tile_w, tile_h, view):
    """Window pixel -> (panel name, pixel within that panel).

    Only the deck panel is clickable for placement; the others are read-only.
    In single-panel views the whole window is that panel.
    """
    if view == "deck":
        return "deck", (x, y)
    if view in ("wrist", "scene"):
        return view, (x, y)
    col, rw = x // tile_w, y // tile_h
    name = {(0, 0): "deck", (1, 0): "wrist",
            (0, 1): "scene", (1, 1): "stats"}.get((col, rw))
    return name, (x - col * tile_w, y - rw * tile_h)


def click_to_board(model, data, camera, u, v, width, height):
    """A pixel in a camera panel -> (y, z) on the board plane, or None.

    Casts the ray through the pinhole and intersects the plane the board sits
    on. Returns None when the ray runs parallel to the board or points away
    from it, which is what a click on the sky or the floor does.
    """
    from camera import Intrinsics, pixel_ray

    intr = Intrinsics.from_model(model, camera, width=width, height=height)
    cid = model.camera(camera).id
    R = data.cam_xmat[cid].reshape(3, 3)
    C = data.cam_xpos[cid]
    ray = pixel_ray(intr, R, u, v)

    plane_x = float(data.xpos[model.body(PLACE_BOARD).id][0])
    if abs(ray[0]) < 1e-6:
        return None
    t = (plane_x - C[0]) / ray[0]
    if t <= 0:
        return None
    p = C + t * ray
    return float(p[1]), float(p[2])


class Placer:
    """Placement state, and the lines it contributes to the stats panel."""

    def __init__(self, model, data, crop, view, tile):
        self.model, self.data, self.crop = model, data, crop
        self.view = view
        self.tile_w, self.tile_h = tile
        self.note = "click the green board to place a tomato"
        self.note_colour = (200, 200, 200)
        self.go = False
        self.quit = False

    def on_click(self, x, y):
        name, (u, v) = panel_pixel(x, y, self.tile_w, self.tile_h, self.view)
        if name not in ("deck", "scene"):
            self.note = f"clicks place fruit in the deck panel, not '{name}'"
            self.note_colour = (120, 230, 255)
            return
        cam = self.deck_camera if name == "deck" else self.scene_camera
        hit = click_to_board(self.model, self.data, cam, u, v,
                             self.tile_w, self.tile_h)
        if hit is None:
            self.note = "that ray never meets the board"
            self.note_colour = (110, 110, 250)
            return
        yy, zz = hit
        ok, why, _zn = check(yy, zz, self.crop.placed)
        if not ok:
            self.note = why
            self.note_colour = (110, 110, 250)
            return
        nm, zn = self.crop.place(yy, zz, quiet=True)
        self.note = (f"placed {nm} at y{yy:+.3f} z{zz:.3f}  [{zn}]"
                     if nm else "pool exhausted")
        self.note_colour = (140, 250, 150)

    def lines(self):
        out = [("CLICK the green board to place a tomato", (240, 210, 140)),
               (f"  placed {len(self.crop.placed)}/{MAX_FRUIT}"
                f"   min spacing {MIN_SPACING * 1000:.0f} mm", (200, 200, 200)),
               (f"  board  y +/-{GUARANTEED_HALF_Y}  "
                f"z {GUARANTEED_Z[0]}-{GUARANTEED_Z[1]}", (200, 200, 200)),
               ("", (0, 0, 0)),
               ("SPACE  harvest what you placed", (240, 210, 140)),
               ("A      auto-fill    C  clear    Q  quit", (200, 200, 200)),
               ("", (0, 0, 0)),
               (self.note[:52], self.note_colour)]
        if self.crop.placed:
            out.append(("", (0, 0, 0)))
            out.append(("placed", (240, 210, 140)))
            for n, p in list(self.crop.placed.items())[-8:]:
                out.append((f"  {n}  y{p[1]:+.3f}  z{p[2]:.3f}",
                            (185, 185, 185)))
        return out

    def key(self, k):
        if k in (32,):                       # space
            if self.crop.placed:
                self.go = True
            else:
                self.note = "place at least one tomato first"
                self.note_colour = (110, 110, 250)
        elif k in (ord("a"), ord("A")):
            # ⚠️ A fresh random seed on every press. Seeding from the number
            # already placed gave the identical arrangement each time, which
            # measures the simulator rather than the robot — the whole point of
            # pressing it repeatedly is to see a layout the planner has not met.
            before = len(self.crop.placed)
            for _n, yy, zz in auto_layout(MAX_FRUIT - before,
                                          seed=random_seed()):
                self.crop.place(yy, zz, quiet=True)
            self.note = (f"auto-filled {before} -> {len(self.crop.placed)}"
                         f" (new arrangement)")
            self.note_colour = (140, 250, 150)
        elif k in (ord("c"), ord("C")):
            self.crop.clear()
            self.note = "cleared"
            self.note_colour = (200, 200, 200)
        elif k in (ord("q"), ord("Q"), 27):
            self.quit = True


def main():
    import argparse
    import os

    # OpenCV draws the window here, not MuJoCo, so rendering is always
    # offscreen — exactly as week3_watch does it, and for the same reason:
    # compositing four panels needs offscreen renders, and mixing a GLFW
    # context with them is how the teardown segfault gets invited back.
    os.environ.setdefault("MUJOCO_GL", "egl")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", default="all",
                    choices=["all", "deck", "wrist", "scene"])
    ap.add_argument("--deck-camera", default="row",
                    choices=["row", "aisle", "wide"])
    ap.add_argument("--scene-camera", default="aisle",
                    choices=["row", "aisle", "wide"])
    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"])
    ap.add_argument("--grid", type=int, default=None,
                    help="pre-place N fruit instead of clicking them")
    ap.add_argument("--seen", action="store_true",
                    help="plan from the camera instead of being told")
    ap.add_argument("--speed", type=float, default=WATCH_SPEED,
                    help=f"fraction of the FR5's rated joint speed "
                         f"(default {WATCH_SPEED}; measurement tools use "
                         f"0.15 so their numbers stay comparable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--wrist-every", type=int, default=4)
    ap.add_argument("--out", default=None, help="record the whole session")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import mujoco

    from camera import SensorCamera
    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from outcomes import report
    from picklog import PickLog, throughput
    from plant_row import Row
    from week3_perceive import build_detector
    from week3_watch import Overlay, Sink, TILE_H, TILE_W, Views, _pump

    pool = pool_trusses()
    model = build_scene(wrist_cam=True, trusses=pool, place_board=True)
    data = mujoco.MjData(model)
    names = [n for n, _, _ in pool]
    row = Row(model, data, names=names,
              homes={n: park_spot(i) for i, n in enumerate(names)})
    park_q = park_posture(model)
    reset_park(model, data, park_q)
    row.reset()
    mujoco.mj_forward(model, data)

    crop = Crop(model, data, row, names)
    sensor = SensorCamera(model, camera="wrist")
    detector = build_detector(args.detector)
    ov = Overlay()
    views = Views(model, data, sensor, detector, ov, view=args.view,
                  deck=args.deck_camera, scene_cam=args.scene_camera,
                  row=row, wrist_every=args.wrist_every)

    placer = Placer(model, data, crop, args.view, (TILE_W, TILE_H))
    placer.deck_camera = args.deck_camera
    placer.scene_camera = args.scene_camera
    # ⚠️ Keys route through Sink, not a waitKey in the loop below. `Sink.push`
    # owns the only cv2.waitKey; a second one competes for the same event queue
    # and drops keypresses at random.
    sink = Sink(live=True, out=args.out, fps=args.fps,
                title="vinea - week 4: click to place, SPACE to harvest",
                on_click=placer.on_click, on_key=placer.key)

    if args.grid:
        crop.apply(auto_layout(min(args.grid, MAX_FRUIT), seed=args.seed))
        placer.note = f"pre-placed {len(crop.placed)} — SPACE to harvest"
        placer.note_colour = (140, 250, 150)

    # --- phase 1: place ------------------------------------------------------
    # The stats panel is swapped for the placement panel by overriding the
    # bound method. Views builds its own lines during the harvest, and this puts
    # them back when placing is done.
    original_stats = views.stats_lines
    views.stats_lines = placer.lines
    print(f"\n  window open — click the GREEN BOARD to place tomatoes.")
    print(f"  SPACE harvest · A auto-fill · C clear · Q quit\n")

    try:
        while not placer.go and not placer.quit and not sink.stopped:
            mujoco.mj_step(model, data)
            sink.push(views.frame())
    except KeyboardInterrupt:
        placer.quit = True

    if placer.quit or sink.stopped or not crop.placed:
        print("  quit before harvesting")
        views.close()
        sink.close()
        return

    # --- phase 2: harvest, same window ---------------------------------------
    views.stats_lines = original_stats
    print(f"  harvesting {len(crop.placed)} fruit\n")
    log = PickLog(args.log, meta={"layout": "clicked",
                                  "detector": args.detector if args.seen
                                  else "told"}) if args.log else None
    rows = []
    # ⚠️ Decimate the render. `on_tick` fires every control cycle — 100 Hz of
    # simulated time — and `_pump` renders three 640x480 camera views and
    # composites four panels each call. Rendering at 100 Hz to show 30 fps is
    # three times the work for no visible difference, and it is what made this
    # run far slower than the same harvest in a plain viewer.
    from reach import CTRL_DT

    every = max(1, int(round((1.0 / args.fps) / CTRL_DT)))
    ticks = [0]

    def pump(_t=None):
        ticks[0] += 1
        if ticks[0] % every == 0:
            _pump(views, sink)

    print(f"  rendering every {every} control cycles (~{args.fps} fps)")
    try:
        rows = harvest_placed(
            model, data, row, crop, park_q, speed=args.speed,
            sensor=sensor if args.seen else None,
            detector=detector if args.seen else None,
            log=log, seed=args.seed, on_tick=pump)
    except KeyboardInterrupt:
        print("\n  stopped early — reporting the picks that finished")

    views.close()
    sink.close()
    if rows:
        report(rows, f"{len(crop.placed)} clicked fruit")
        t = throughput(rows)
        if t:
            print(f"\n  cycle {t['mean_s']:.1f} s mean · "
                  f"{t['kg_hr']:.1f} kg/hr single arm")
    if log:
        log.close()


if __name__ == "__main__":
    main()
