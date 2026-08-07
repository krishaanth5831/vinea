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

**The deck panel is a real sensor now.** It used to be the `row` cinematic
framing — a camera 1.6 m out in the aisle, bolted to nothing, that no machine
could carry — wearing the label "deck cam". It is now `deck_cam.DECK_NAME`: the
RGBD camera on the mast behind the arm, which you can see in the scene panel.
So the top-left tile is genuinely what the robot's survey saw, and the plan
drawn over it was made from that viewpoint.

**Clicking works by ray-casting, not by guessing.** A click in the deck panel is
turned into a ray through that camera's pinhole (`camera.pixel_ray`, the same
maths Week 3's deprojection gate is built on) and intersected with the board
plane. So the tomato lands exactly where the cursor was, from any camera angle,
and moving the deck camera does not break it.

⚠️ **The green board is the arm's actual working area** — the band re-measured
in `week4_envelope.py`, cell by cell, one full pick each. Anywhere on it is
reachable by construction, and clicks off it are refused with the reason on the
stats panel rather than silently ignored.

⚠️ **There is no minimum spacing.** Put them as close together as you like,
down to touching. That used to be refused at 200 mm; see `week4_place.TOUCHING`
for why the rule went and what replaced it — the short version is that a close
pair is a *pick-order* problem, not a placement error, and the deck camera is
what solves it.

Quit with the QUIT button, q, Esc, or the window's X — all four work, and
whatever picks finished are still reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deck_cam import DECK_NAME, DeckSurvey  # noqa: E402
from week4_place import (Crop, GUARANTEED_HALF_Y, GUARANTEED_Z,  # noqa: E402
                         MAX_FRUIT, PLACE_BOARD, auto_layout, check,
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
        if not nm:
            self.note = "pool exhausted"
            self.note_colour = (110, 110, 250)
            return
        # An accepted-but-tight placement now comes back with a reason too.
        # Amber rather than green: it went in, and it is going to be work.
        self.note = f"placed {nm} at y{yy:+.3f} z{zz:.3f}  [{zn}]"
        self.note_colour = (140, 250, 150)
        if why:
            self.note = why
            self.note_colour = (120, 210, 250)

    def lines(self):
        pts = list(self.crop.placed.values())
        closest = min((float(np.hypot(a[1] - b[1], a[2] - b[2]))
                       for i, a in enumerate(pts) for b in pts[i + 1:]),
                      default=float("nan"))
        out = [("CLICK the green board to place a tomato", (240, 210, 140)),
               (f"  placed {len(self.crop.placed)}/{MAX_FRUIT}"
                + ("" if not pts[1:] else
                   f"   closest pair {closest * 1000:.0f} mm"),
                (200, 200, 200)),
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


class Thoughts:
    """What the arm is doing right now, as lines for the stats panel.

    Two sources, and it needs both. `harvest_placed` fires events at the
    decision points — which fruit was selected, what the planner returned,
    whether it refused — and those are latched here. Everything else is read
    **live off the simulator every frame**: which leg is executing, where the
    tool is, how close it is to the nearest fruit it is not picking. A panel
    built only from events freezes between them, which is what "no live data"
    looked like.
    """

    PHASE_COLOUR = {"SURVEY": (250, 170, 90), "SELECT": (240, 210, 140),
                    "SCAN": (250, 200, 90), "PLAN": (120, 230, 255),
                    "PICK": (140, 250, 150), "DONE": (200, 200, 200)}

    # What each leg of a mission is actually for, in words rather than jargon.
    LEG_MEANING = {
        "park": "at the staging plane, outside the crop",
        "clear": "backing out to a lane that clears the row",
        "lane": "traversing outside the canopy",
        "align": "lining up square-on to the fruit",
        "stage": "holding at the staging plane",
        "approach": "moving in, fingers open, touching nothing",
        "insert": "closing the last gap to the fruit",
        "grip": "ramping the fingers shut on it",
        "pull": "pulling until the peduncle lets go",
        "extract": "backing out with the fruit held",
        "turn": "rotating the wrist to carry",
        "carry": "carrying it to the crate",
        "over_bin": "positioned over the crate",
        "release": "opening the fingers to drop it in",
        "withdraw": "retreating to park",
        "home": "returning to the park posture",
        "settle": "holding still to let contact settle",
    }

    def __init__(self, model, data, crop, ov):
        self.model, self.data, self.crop, self.ov = model, data, crop, ov
        self.phase = "SELECT"
        self.target = None
        self.queue = []
        self.mission = None
        self.trace = None
        self.guard = None
        self.t_plan = None
        self.err_mm = None
        self.refused = None
        self.results = []
        self.tool_site = model.site("tool0").id
        # None means "no deck camera in this run", which the panel says out
        # loud rather than showing an empty ordering line.
        self.survey_n = None
        self.risk = None
        self.block = None
        self.scan_poses = None

    # --- latched from harvest_placed -----------------------------------------

    def event(self, kind, **info):
        if kind == "survey":
            self.phase = "SURVEY"
            self.survey_n = len(info.get("survey", {}))
            self.plan = info.get("plan")
            self.scan_poses = info.get("poses")
        elif kind == "select":
            self.phase, self.target = "SELECT", info["fruit"]
            self.queue = info.get("queue", [])
            self.mission = self.trace = self.guard = None
            self.err_mm = self.refused = self.t_plan = None
            self.ov.mission = None
            step = next((s for s in getattr(self, "plan", None).steps
                         if s.fruit == self.target),
                        None) if getattr(self, "plan", None) else None
            self.risk = step.risk if step is not None else None
            self.block = step.block if step is not None else None
        elif kind == "scan":
            self.phase = "SCAN"
        elif kind == "seen":
            self.err_mm = info.get("err_mm")
            est = info.get("est")
            if est is not None:
                self.ov.scan[info["fruit"]] = (est, (self.err_mm or 0) / 1000)
        elif kind == "plan":
            self.phase = "PLAN"
        elif kind == "refused":
            self.phase, self.refused = "PLAN", info.get("why")
        elif kind == "fly":
            self.phase = "PICK"
            self.mission = info.get("mission")
            self.trace = info.get("trace")
            self.guard = info.get("guard")
            self.t_plan = info.get("t_plan")
            # Feeds week3_watch.decorate, which draws the checked route into
            # the deck panel — the plan becomes visible, not just described.
            self.ov.mission = self.mission
        elif kind == "result":
            rec = info["rec"]
            good = rec.get("outcome") == "clean"
            self.results.append(
                (f"  {rec['fruit']}  {rec.get('outcome')}",
                 (140, 250, 150) if good else (110, 110, 250)))
            self.phase = "DONE"
        self.ov.phase = self.phase
        self.ov.target = self.target

    # --- read live, every frame ----------------------------------------------

    def _leg(self):
        return getattr(self.trace, "leg", "") if self.trace else ""

    def lines(self):
        import numpy as np

        col = self.PHASE_COLOUR.get(self.phase, (230, 230, 230))
        out = [(f"PHASE   {self.phase}", col)]

        done = len(self.results)
        out.append((f"  fruit {done}/{len(self.crop.placed)} attempted",
                    (200, 200, 200)))
        if self.target:
            out.append((f"  target  {self.target}", (240, 210, 140)))

        # What it is choosing between, and — since the deck camera arrived —
        # *why* that is the order. The old version of this line said "order as
        # placed - not optimised", which was true and is not any more.
        left = [n for n in self.queue if n != self.target]
        if left:
            out.append((f"  then    {' '.join(left[:6])}", (150, 150, 150)))
        if self.survey_n is not None:
            how = (f"in {len(self.scan_poses)} head poses" if self.scan_poses
                   else "in one frame")
            out.append((f"  deck saw {self.survey_n}/{len(self.crop.placed)} "
                        f"{how}", (120, 230, 255)))
            plan = getattr(self, "plan", None)
            if plan is not None:
                out.append((f"  order minimises expected refusals: "
                            f"{plan.lost:.2f} fruit"
                            + ("  (proved optimal)" if plan.optimal else ""),
                            (120, 120, 120)))
            if self.block is not None:
                # ⚠️ `block` and not `risk`: block is the one that decides
                # whether this pick gets refused, which is what the operator
                # watching wants to know about the fruit the arm is going for.
                out.append((f"  this one: worst neighbour {self.block:.2f}, "
                            f"crowding {self.risk:.2f}", (120, 120, 120)))
        else:
            out.append(("  (placement order - deck camera off)", (120, 120, 120)))

        if self.phase == "SURVEY":
            out.append(("", (0, 0, 0)))
            out.append(("deck camera: one frame from the mast,", (250, 200, 90)))
            out.append(("arm parked. Finds the row and orders it.", (250, 200, 90)))
        if self.phase == "SCAN":
            out.append(("", (0, 0, 0)))
            out.append(("wrist cam: driving to where the DECK", (250, 200, 90)))
            out.append(("camera said it is, then one frame.", (250, 200, 90)))
        if self.err_mm is not None:
            c = ((140, 250, 150) if self.err_mm < 5
                 else (120, 230, 255) if self.err_mm < 15 else (110, 110, 250))
            out.append((f"  camera error  {self.err_mm:.1f} mm", c))

        if self.refused:
            out.append(("", (0, 0, 0)))
            out.append(("REFUSED - no route cleared the crop", (110, 110, 250)))
            out.append((f"  {str(self.refused)[:44]}", (170, 170, 170)))

        # ⚠️ Live numbers BEFORE the leg list. `_panel` stops drawing at about
        # 19 lines, and a full mission is ~12 legs — so putting these last meant
        # the one thing that changes every frame was the thing cut off.
        if self.phase == "PICK" and self.guard is not None:
            out.append(("", (0, 0, 0)))
            try:
                d, _part, obstacle = self.guard.clearance.worst(self.data)
                c = ((110, 110, 250) if d < 0.02
                     else (120, 230, 255) if d < 0.05 else (140, 250, 150))
                out.append((f"  nearest crop {d * 1000:5.0f} mm ({obstacle})", c))
            except Exception:
                pass
            tool = self.data.site_xpos[self.tool_site]
            out.append((f"  tool [{tool[0]:+.2f} {tool[1]:+.2f} {tool[2]:+.2f}]",
                        (170, 170, 170)))
            if self.trace is not None:
                held = self.trace.peak_held()
                if held:
                    c = (110, 110, 250) if held > 1.0 else (170, 170, 170)
                    out.append((f"  fruit peak {held:.2f} m/s while held", c))

        if self.mission is not None:
            m = self.mission
            out.append(("", (0, 0, 0)))
            out.append((f"PLAN lane '{m.lane}' clears "
                        f"{m.clearance * 1000:.0f} mm", (240, 210, 140)))
            if self.t_plan:
                out.append((f"  checked in {self.t_plan:.2f} s", (150, 150, 150)))
            leg = self._leg()
            names = [lg.name for lg in m.legs]
            i = names.index(leg) if leg in names else 0
            out.append((f"  leg {i + 1}/{len(names)}", (150, 150, 150)))
            # A window around the current leg rather than the whole list, so the
            # panel shows where it is without needing 12 lines to do it.
            for j in range(max(0, i - 1), min(len(names), i + 3)):
                here = j == i
                out.append((f" {'>' if here else ' '} {names[j]}",
                            (140, 250, 150) if here else (150, 150, 150)))
                if here:
                    why = self.LEG_MEANING.get(names[j])
                    if why:
                        out.append((f"     {why}", (200, 200, 200)))

        if self.results:
            out.append(("", (0, 0, 0)))
            out.append(("done", (240, 210, 140)))
            out += self.results[-5:]
        return out


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
    # ⚠️ The default is now the *real* deck camera, not the "row" cinematic
    # framing. This panel has been labelled "deck cam" since Week 3 while
    # showing a camera bolted to nothing, 1.6 m out in the aisle, that no part
    # of the robot could ever have. It is now the sensor the machine actually
    # carries — so what you see in this panel is what the survey saw, and the
    # plan drawn over it is a plan made from this viewpoint.
    ap.add_argument("--deck-camera", default=DECK_NAME,
                    choices=[DECK_NAME, "row", "aisle", "wide"])
    ap.add_argument("--scene-camera", default="aisle",
                    choices=["row", "aisle", "wide"])
    ap.add_argument("--no-deck", action="store_true",
                    help="pick in placement order, as this did before the "
                         "chassis camera existed")
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
    model = build_scene(wrist_cam=True, deck_cam=True, trusses=pool,
                        place_board=True)
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
    deck = None if args.no_deck else DeckSurvey(model,
                                                detector=build_detector(
                                                    args.detector))
    ov = Overlay()
    views = Views(model, data, sensor, detector, ov, view=args.view,
                  deck=args.deck_camera, scene_cam=args.scene_camera,
                  row=row, wrist_every=args.wrist_every)
    views.panel_title = "place the fruit"

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
    # ⚠️ NOT week3_watch's own stats_lines. That one reads an Overlay that
    # week3_watch.run() populates as it goes — and this file drives the harvest
    # with harvest_placed, which never touched it. The panel therefore sat
    # empty for the whole run, which is what "no live data" was.
    thoughts = Thoughts(model, data, crop, ov)
    views.stats_lines = thoughts.lines
    views.panel_title = "what the arm is doing"
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
            log=log, seed=args.seed, on_tick=pump,
            on_event=thoughts.event, deck=deck)
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
