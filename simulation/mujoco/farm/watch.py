#!/usr/bin/env python3
"""Watch a whole shift: six panels, including the map the robot is building.

    +----------------+----------------+----------------+
    | scout cam      | wrist cam      | down the aisle |
    | what it maps   | what it picks  |                |
    +----------------+----------------+----------------+
    | THE MAP        | what it is     | the machine    |
    | top-down, live | thinking       |                |
    +----------------+----------------+----------------+

The map panel is the one that does not exist anywhere else in this repo, and it
is the point of this file. Everything else the robot knows is a number in a
terminal; the map is the only place you can *see* that it drove eight metres,
looked at a crop, decided four of forty fruit were worth taking and then went
and took them.

⚠️ **The map is drawn from `HouseMap`, not from the simulator.** Every dot on it
is a place the robot believes a fruit is, coloured by the ripeness it believes
that fruit has. Where a dot is wrong, the panel shows it wrong — which is the
only way to watch a perception failure happen rather than read about it after.
The ground truth is drawn too, faintly, precisely so the gap is visible.

    ./.venv/bin/python simulation/mujoco/farm/watch.py            # the shift
    ./.venv/bin/python simulation/mujoco/farm/watch.py --truth    # skip scouting
    ./.venv/bin/python simulation/mujoco/farm/watch.py --stops 2  # short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import house, overlay, route, run as frun, trolley  # noqa: E402

TILE_W, TILE_H = 560, 420

# Panel colours, BGR for OpenCV. The stage colours match `crop.STAGES` so a dot
# on the map is the same colour as the tomato it stands for in the camera panel
# — the whole point of a map is that you can match it to the thing.
STAGE_BGR = {
    "red": (60, 60, 220), "turning": (60, 160, 240),
    "breaker": (60, 220, 240), "green": (80, 190, 90),
}
INK = (232, 232, 232)
DIM = (120, 120, 120)
ACCENT = (140, 250, 150)
WARN = (110, 110, 250)


class MapPanel:
    """A top-down plan of the house, drawn from what the robot believes.

    ⚠️ Drawn with the **long axis horizontal**. The house is 8 m along the row
    and 5.8 m across it, so putting y across the screen fills a 4:3 panel; the
    other way round wastes half of it on empty aisle. It also means the trolley
    travels left-to-right, which is how anyone would sketch it on paper.
    """

    def __init__(self, w=TILE_W, h=TILE_H, pad=34):
        self.w, self.h, self.pad = w, h, pad
        self.y0, self.y1 = -house.HOUSE_HALF_Y - 0.6, house.HOUSE_HALF_Y + 0.6
        self.x0 = house.ROW_X0 - 0.9
        self.x1 = house.row_x(house.N_ROWS - 1) + 0.9
        sx = (w - 2 * pad) / (self.y1 - self.y0)
        sy = (h - 2 * pad - 16) / (self.x1 - self.x0)
        self.s = min(sx, sy)

    def px(self, x, y):
        """World (x, y) -> panel pixel. Note the axis swap; see the docstring."""
        u = self.pad + (y - self.y0) * self.s
        v = self.pad + 10 + (x - self.x0) * self.s
        return int(round(u)), int(round(v))

    def draw(self, house_map=None, rte=None, trolley_y=None, aisle=0,
             picked=None, truth=None, target=None):
        import cv2

        img = np.full((self.h, self.w, 3), 22, np.uint8)
        picked = picked or set()

        # --- the building ---------------------------------------------------
        for i in range(house.N_ROWS):
            a, b = self.px(house.row_x(i), self.y0), self.px(house.row_x(i),
                                                             self.y1)
            cv2.line(img, a, b, (46, 62, 46), 9)
            cv2.putText(img, f"r{i}", (6, a[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (110, 150, 110), 1, cv2.LINE_AA)
        for i, ax in house.aisles():
            a, b = self.px(ax, self.y0), self.px(ax, self.y1)
            cv2.line(img, a, b, (58, 58, 58), 1, cv2.LINE_AA)

        # --- the truth, faint, so a wrong dot is visible as wrong ------------
        if truth:
            for t in truth:
                u, v = self.px(t.x, t.y)
                cv2.circle(img, (u, v), 4, (70, 70, 70), 1, cv2.LINE_AA)

        # --- the route ------------------------------------------------------
        if rte is not None and rte.stops:
            ax = house.aisle_x(rte.aisle)
            pts = [self.px(ax, s.y) for s in rte.stops]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (90, 90, 140), 1, cv2.LINE_AA)
            for k, (p, s) in enumerate(zip(pts, rte.stops), 1):
                cv2.drawMarker(img, p, (150, 150, 210), cv2.MARKER_TRIANGLE_UP,
                               9, 1)
                cv2.putText(img, str(k), (p[0] - 3, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 210), 1,
                            cv2.LINE_AA)

        # --- what the robot mapped ------------------------------------------
        if house_map is not None:
            for s in house_map.sightings:
                u, v = self.px(s.pos[0], s.pos[1])
                col = STAGE_BGR.get(s.stage, INK)
                if id(s) in picked or (s.truth is not None
                                       and s.truth.name in picked):
                    # Taken. A hollow ring, so the map shows the row emptying.
                    cv2.circle(img, (u, v), 5, col, 1, cv2.LINE_AA)
                    cv2.line(img, (u - 3, v - 3), (u + 3, v + 3), col, 1)
                else:
                    cv2.circle(img, (u, v), 4, col, -1, cv2.LINE_AA)
                if s.ripe:
                    cv2.circle(img, (u, v), 7, col, 1, cv2.LINE_AA)

        # --- the machine ----------------------------------------------------
        if trolley_y is not None:
            ax = house.aisle_x(aisle)
            a = self.px(ax - trolley.DECK_W / 2, trolley_y - trolley.DECK_L / 2)
            b = self.px(ax + trolley.DECK_W / 2, trolley_y + trolley.DECK_L / 2)
            cv2.rectangle(img, a, b, (220, 200, 120), 1, cv2.LINE_AA)
            c = self.px(ax + house.ARM_OFFSET, trolley_y)
            cv2.drawMarker(img, c, (220, 200, 120), cv2.MARKER_SQUARE, 7, 2)
            if target is not None:
                cv2.line(img, c, self.px(target[0], target[1]), ACCENT, 1,
                         cv2.LINE_AA)

        # --- legend ---------------------------------------------------------
        cv2.putText(img, "THE MAP  (top-down, what the robot believes)",
                    (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1,
                    cv2.LINE_AA)
        x = 8
        for name in ("green", "breaker", "turning", "red"):
            cv2.circle(img, (x + 4, self.h - 10), 4, STAGE_BGR[name], -1,
                       cv2.LINE_AA)
            cv2.putText(img, name, (x + 12, self.h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
            x += 12 + 8 * len(name) + 10
        cv2.putText(img, "ring = ripe   x = taken", (x, self.h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
        return img


class Thoughts:
    """What the robot is doing, latched from `run`'s events."""

    def __init__(self):
        self.phase = "START"
        self.map = None
        self.route = None
        self.stop = None
        self.target = None
        self.stage = None
        self.results = []
        self.scout_frame = None
        self.refused = None

    def event(self, kind, **info):
        if kind == "phase":
            self.phase = info["phase"]
        elif kind == "scout_frame":
            self.scout_frame = (info["i"] + 1, info["n"], info["y"],
                                len(info["seen"]))
        elif kind == "map":
            self.map = info["house_map"]
        elif kind == "route":
            self.route = info["route"]
        elif kind == "stop":
            self.stop = (info["i"], info["n"], info["y"], len(info["fruit"]))
            self.target = self.refused = None
        elif kind == "select":
            self.target, self.stage = info["fruit"], info["stage"]
            self.refused = None
        elif kind == "refused":
            self.refused = info["why"]
        elif kind == "result":
            r = info["rec"]
            self.results.append((r.get("fruit"), r.get("outcome")))

    def lines(self):
        out = [(f"PHASE   {self.phase}", ACCENT)]
        if self.phase == "SCOUT" and self.scout_frame:
            i, n, y, k = self.scout_frame
            out += [(f"  frame {i}/{n} at y={y:+.2f}", INK),
                    (f"  {k} fruit banded in this frame", DIM),
                    ("", INK),
                    ("driving the aisle, looking sideways.", (250, 200, 90)),
                    ("nothing is picked in this pass.", (250, 200, 90))]
        if self.map is not None:
            out.append((f"  mapped {len(self.map.sightings)} fruit, "
                        f"{len(self.map.ripe)} ripe", INK))
        if self.route is not None:
            out.append((f"  route: {len(self.route.stops)} stops, "
                        f"{self.route.n_fruit} to take", INK))
            if self.route.skipped:
                out.append((f"  {len(self.route.skipped)} ripe left — other row",
                            DIM))
        if self.stop:
            i, n, y, k = self.stop
            out += [("", INK), (f"STOP {i}/{n} at y={y:+.2f}", ACCENT),
                    (f"  {k} fruit to take here", INK)]
        if self.target:
            out.append((f"  target {self.target} ({self.stage})", (240, 210, 140)))
        if self.refused:
            out += [("", INK), ("REFUSED - no route cleared the crop", WARN),
                    (f"  {str(self.refused)[:44]}", DIM)]
        if self.results:
            out.append(("", INK))
            out.append(("done", (240, 210, 140)))
            for name, outcome in self.results[-6:]:
                good = outcome == "clean"
                out.append((f"  {name}  {outcome}", ACCENT if good else WARN))
        return out


def _panel(w, h, lines, title=None):
    import cv2

    img = np.full((h, w, 3), 22, np.uint8)
    y = 22
    if title:
        cv2.putText(img, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y += 22
    for text, colour in lines:
        if y > h - 10:
            break
        cv2.putText(img, text[:56], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    colour, 1, cv2.LINE_AA)
        y += 19
    return img


def _label(img, text):
    import cv2

    cv2.putText(img, text, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (210, 210, 210), 1, cv2.LINE_AA)
    return img


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--stops", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default=None, help="record instead of showing")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import cv2
    import mujoco

    from reach import CTRL_DT
    from week3_watch import Sink

    thoughts = Thoughts()
    mp = MapPanel()
    state = {"model": None, "data": None, "renderers": {}, "picked": set(),
             "truth": None, "sink": None, "n": 0}

    every = max(1, int(round((1.0 / args.fps) / CTRL_DT)))

    def build_renderers(model):
        return {name: mujoco.Renderer(model, height=TILE_H, width=TILE_W,
                                      max_geom=30000)
                for name in ("scout", "aisle", "house", "wrist")}

    def compose():
        model, data = state["model"], state["data"]
        rs = state["renderers"]
        out = {}
        for name in ("scout", "aisle", "house", "wrist"):
            rs[name].update_scene(data, camera=name)
            out[name] = cv2.cvtColor(rs[name].render(), cv2.COLOR_RGB2BGR)
        # The ripeness call, drawn where it is made. Same classifier the mapping
        # pass uses — see `farm.overlay` for why it must not be a better one.
        overlay.tally(out["wrist"], overlay.annotate(out["wrist"]))
        ty = float(data.body(trolley.TROLLEY).xpos[1])
        tgt = None
        if thoughts.target:
            try:
                tgt = data.body(thoughts.target).xpos
            except KeyError:
                tgt = None
        m = mp.draw(house_map=thoughts.map, rte=thoughts.route, trolley_y=ty,
                    aisle=args.aisle, picked=state["picked"],
                    truth=state["truth"], target=tgt)
        stats = _panel(TILE_W, TILE_H, thoughts.lines(), "the robot's shift")
        top = np.hstack([_label(out["scout"], "scout cam - what it maps"),
                         _label(out["aisle"], "down the aisle"),
                         _label(out["house"], "the house")])
        bot = np.hstack([m, stats,
                         _label(out["wrist"], "wrist cam - what it picks")])
        return np.vstack([top, bot])

    def tick(_t=None):
        state["n"] += 1
        if state["n"] % every:
            return
        if state["sink"] is None or state["model"] is None:
            return
        state["sink"].push(compose())

    def on_event(kind, **info):
        thoughts.event(kind, **info)
        if kind == "result" and info["rec"].get("in_bin"):
            state["picked"].add(info["rec"].get("fruit"))

    # `run` builds the model, so the renderers cannot exist until it has. The
    # first `scout_frame` event is the earliest point everything is up.
    real_run = frun.run

    def patched(*a, **kw):
        return real_run(*a, **kw)

    # Build the scene here rather than inside `run`, so the viewer can hold
    # renderers on it for the whole shift.
    import farm.crop as _c

    seed = _c.resolve_seed(args.seed)
    trusses = _c.spawn(n_per_row=args.n, seed=seed)
    state["truth"] = trusses

    print(f"  building the house — {len(trusses)} fruit, "
          f"{sum(1 for t in trusses if t.ripe)} ripe")

    # A first model just to open the window at the right size.
    model = trolley.build(aisle=args.aisle, arms=("a",), trusses=trusses,
                          wrist_cam=True, deck_cam=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state["model"], state["data"] = model, data
    state["renderers"] = build_renderers(model)

    sink = Sink(live=args.out is None, out=args.out, fps=args.fps,
                title="vinea - a shift in the greenhouse")
    state["sink"] = sink
    sink.push(compose())

    try:
        shift = frun.run(seed=seed, aisle=args.aisle, n_per_row=args.n,
                         speed=args.speed, use_truth=args.truth,
                         max_stops=args.stops, on_tick=tick,
                         on_event=on_event, verbose=True,
                         scene=(model, data, trusses))
        shift.report()
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        for r in state["renderers"].values():
            try:
                r.close()
            except Exception:
                pass
        sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
