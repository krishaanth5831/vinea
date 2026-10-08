#!/usr/bin/env python3
"""Watch a truss shift: six panels, including the map of clusters it is building.

    +----------------+----------------+----------------+
    | scout cam      | wrist cam      | down the aisle |
    | what it maps   | what it cuts   |                |
    +----------------+----------------+----------------+
    | THE MAP        | what it is     | the machine    |
    | trusses, live  | thinking       |                |
    +----------------+----------------+----------------+

`farm/watch.py` is this viewer for **loose** fruit. The difference you can
actually see is the map panel: there, every dot is a tomato and its colour is
that tomato's ripeness. Here, every dot is a *truss*, drawn as the cluster it
is — six fruit in their real positions, each in its own colour — with a ring
round the ones the rule says to take.

⚠️ **That is the whole point of watching this variant.** The cluster decision is
the thing that is hard to believe from a number: "4 of 6 red, take it" reads
fine in a table and looks either obviously right or obviously wrong when you can
see the truss. The panel draws the fraction next to each cluster so the verdict
and its evidence are in the same picture.

⚠️ The map is drawn from what the robot **believes** — `trussrun.ClusterMap`,
grouped from colour blobs — not from the simulator. Ground truth is drawn
faintly underneath, so a mis-grouped truss shows up as a cluster that does not
sit on one.

    ./.venv/bin/python simulation/mujoco/farm/watch_truss.py            # the shift
    ./.venv/bin/python simulation/mujoco/farm/watch_truss.py --truth    # skip scouting
    ./.venv/bin/python simulation/mujoco/farm/watch_truss.py --stops 2  # short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import house, overlay, trolley  # noqa: E402
from farm import truss as ft  # noqa: E402
from farm import trussrun as ftr  # noqa: E402
from farm.watch import (INK, DIM, ACCENT, WARN, STAGE_BGR, TILE_H,  # noqa: E402
                        TILE_W, _label, _panel, seat_scout_head)


class TrussMapPanel:
    """A top-down plan of the house, drawn as trusses rather than as fruit.

    ⚠️ Long axis horizontal, exactly as `watch.MapPanel` — see its docstring for
    why. What changes is what a mark means: one mark here is a *cluster*, and
    the fruit inside it are drawn at their own offsets so a six-fruit truss
    looks like a six-fruit truss and its ripeness gradient is visible.
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
        u = self.pad + (y - self.y0) * self.s
        v = self.pad + 10 + (x - self.x0) * self.s
        return int(round(u)), int(round(v))

    def draw(self, cmap=None, rte=None, trolley_y=None, aisle=0, picked=None,
             truth=None, target=None, threshold=ft.RIPE_FRACTION):
        import cv2

        img = np.full((self.h, self.w, 3), 22, np.uint8)
        picked = picked or set()

        for i in range(house.N_ROWS):
            a, b = self.px(house.row_x(i), self.y0), self.px(house.row_x(i),
                                                             self.y1)
            cv2.line(img, a, b, (46, 62, 46), 9)
            cv2.putText(img, f"r{i}", (6, a[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (110, 150, 110), 1, cv2.LINE_AA)
        for i, ax in house.aisles():
            a, b = self.px(ax, self.y0), self.px(ax, self.y1)
            cv2.line(img, a, b, (58, 58, 58), 1, cv2.LINE_AA)

        # --- the truth, faint: one small ring per real truss -----------------
        if truth:
            for t in truth:
                u, v = self.px(t.x, t.y)
                cv2.circle(img, (u, v), 6, (70, 70, 70), 1, cv2.LINE_AA)

        if rte is not None and rte.stops:
            ax = house.aisle_x(rte.aisle)
            pts = [self.px(ax, s.y) for s in rte.stops]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (90, 90, 140), 1, cv2.LINE_AA)
            for k, p in enumerate(pts, 1):
                cv2.drawMarker(img, p, (150, 150, 210), cv2.MARKER_TRIANGLE_UP,
                               9, 1)
                cv2.putText(img, str(k), (p[0] - 3, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 210), 1,
                            cv2.LINE_AA)

        # --- what the robot grouped ------------------------------------------
        if cmap is not None:
            for c in cmap.sightings:
                u, v = self.px(c.pos[0], c.pos[1])
                take = c.ripe_at(threshold)
                gone = c.truth is not None and c.truth.name in picked
                # Each member fruit, at its own y, in its own stage colour. The
                # cluster's gradient is the thing the decision is made on, so it
                # is what the panel draws.
                for m in c.members:
                    mu, mv = self.px(m.pos[0], m.pos[1])
                    col = STAGE_BGR.get(m.stage, INK)
                    cv2.circle(img, (mu, mv), 2, col, -1, cv2.LINE_AA)
                if gone:
                    cv2.drawMarker(img, (u, v), ACCENT, cv2.MARKER_TILTED_CROSS,
                                   9, 1)
                elif take:
                    cv2.circle(img, (u, v), 8, ACCENT, 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (u, v), 8, (90, 90, 90), 1, cv2.LINE_AA)
                cv2.putText(img, f"{c.n_red}/{c.n_fruit}", (u - 9, v + 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            ACCENT if take else DIM, 1, cv2.LINE_AA)

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

        cv2.putText(img, f"THE MAP  (trusses; ring = take at "
                         f"{threshold:.2f})", (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
        x = 8
        for name in ("green", "breaker", "turning", "red"):
            cv2.circle(img, (x + 4, self.h - 10), 3, STAGE_BGR[name], -1,
                       cv2.LINE_AA)
            cv2.putText(img, name, (x + 12, self.h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
            x += 12 + 8 * len(name) + 10
        cv2.putText(img, "n/6 = red", (x, self.h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
        return img


class Thoughts:
    """What the robot is doing, latched from `trussrun`'s events."""

    def __init__(self, threshold=ft.RIPE_FRACTION):
        self.phase = "START"
        self.map = None
        self.route = None
        self.stop = None
        self.target = None
        self.cluster = None
        self.results = []
        self.scout_frame = None
        self.refused = None
        self.threshold = threshold
        self.cut = 0
        self.fruit = 0
        self.green = 0

    def event(self, kind, **info):
        if kind == "phase":
            self.phase = info["phase"]
        elif kind == "scout_frame":
            self.scout_frame = (info["i"] + 1, info["n"], info["y"],
                                len(info["seen"]))
        elif kind == "map":
            self.map = info["cluster_map"]
        elif kind == "route":
            self.route = info["route"]
        elif kind == "stop":
            self.stop = (info["i"], info["n"], info["y"], len(info["fruit"]))
            self.target = self.refused = self.cluster = None
        elif kind == "select":
            self.target = info["fruit"]
            self.cluster = info.get("cluster")
            self.refused = None
        elif kind == "refused":
            self.refused = info["why"]
        elif kind == "result":
            r = info["rec"]
            self.results.append((r.get("fruit"), r.get("outcome")))
            if r.get("in_bin"):
                self.cut += 1
                self.fruit += r.get("n_fruit", 0)
                self.green += r.get("n_green", 0)

    def lines(self):
        out = [(f"PHASE   {self.phase}", ACCENT),
               (f"  take a truss at {self.threshold:.2f} red", DIM)]
        if self.phase == "SCOUT" and self.scout_frame:
            i, n, y, k = self.scout_frame
            out += [(f"  frame {i}/{n} at y={y:+.2f}", INK),
                    (f"  {k} fruit banded in this frame", DIM),
                    ("", INK),
                    ("driving the aisle, looking sideways.", (250, 200, 90)),
                    ("fruit now, trusses after grouping.", (250, 200, 90))]
        if self.map is not None:
            out.append((f"  {len(self.map.sightings)} trusses from "
                        f"{self.map.blobs} fruit", INK))
            out.append((f"  {len(self.map.ripe)} ripe enough to cut", INK))
        if self.route is not None:
            out.append((f"  route: {len(self.route.stops)} stops, "
                        f"{self.route.n_fruit} to cut", INK))
        if self.stop:
            i, n, y, k = self.stop
            out += [("", INK), (f"STOP {i}/{n} at y={y:+.2f}", ACCENT),
                    (f"  {k} trusses to cut here", INK)]
        if self.target:
            out.append((f"  target {self.target}", (240, 210, 140)))
        if self.cluster is not None:
            c = self.cluster
            out.append((f"  {c.n_red}/{c.n_fruit} red "
                        f"({c.red_fraction:.2f}) -> CUT", ACCENT))
        if self.refused:
            out += [("", INK), ("REFUSED - no route cleared the crop", WARN),
                    (f"  {str(self.refused)[:44]}", DIM)]
        if self.results:
            out.append(("", INK))
            out.append((f"crated {self.cut} trusses, {self.fruit} tomatoes",
                        (240, 210, 140)))
            if self.green:
                out.append((f"  {self.green} green cut with them", WARN))
            for name, outcome in self.results[-4:]:
                good = outcome == "clean"
                out.append((f"  {name}  {outcome}", ACCENT if good else WARN))
        return out


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=8, help="trusses per row")
    ap.add_argument("--stops", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"red fraction to cut a truss "
                         f"(default {ft.RIPE_FRACTION})")
    ap.add_argument("--out", default=None, help="record instead of showing")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import cv2
    import mujoco

    from reach import CTRL_DT
    from week3_watch import Sink

    th = ft.RIPE_FRACTION if args.threshold is None else args.threshold
    thoughts = Thoughts(threshold=th)
    mp = TrussMapPanel()
    state = {"model": None, "data": None, "renderers": {}, "picked": set(),
             "truth": None, "sink": None, "n": 0, "head": None}

    every = max(1, int(round((1.0 / args.fps) / CTRL_DT)))

    def build_renderers(model):
        return {name: mujoco.Renderer(model, height=TILE_H, width=TILE_W,
                                      max_geom=30000)
                for name in ("scout", "aisle", "house", "wrist")}

    def compose():
        model, data = state["model"], state["data"]
        rs = state["renderers"]
        out = {}
        # The scout head is world-parented and does not ride the trolley on its
        # own; see `farm.watch.seat_scout_head`, which is shared rather than
        # reimplemented so the two viewers cannot drift apart on it.
        seat_scout_head(state["head"], model, data)
        for name in ("scout", "aisle", "house", "wrist"):
            rs[name].update_scene(data, camera=name)
            out[name] = cv2.cvtColor(rs[name].render(), cv2.COLOR_RGB2BGR)
        wrist_counts = overlay.annotate(out["wrist"])
        ty = float(data.body(trolley.TROLLEY).xpos[1])
        tgt = None
        if thoughts.target:
            try:
                tgt = data.body(thoughts.target).xpos
            except KeyError:
                tgt = None
        m = mp.draw(cmap=thoughts.map, rte=thoughts.route, trolley_y=ty,
                    aisle=args.aisle, picked=state["picked"],
                    truth=state["truth"], target=tgt, threshold=th)
        stats = _panel(TILE_W, TILE_H, thoughts.lines(), "the truss shift")
        top = np.hstack([_label(out["scout"], "scout cam - what it maps"),
                         _label(out["aisle"], "down the aisle"),
                         _label(out["house"], "the house")])
        bot = np.hstack([m, stats,
                         _label(out["wrist"],
                                f"wrist cam - HSV: ripe {wrist_counts['ripe']}"
                                f", unripe {wrist_counts['unripe']}")])
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

    from farm import crop as _c

    seed = _c.resolve_seed(args.seed, label="truss crop")
    trusses = ft.spawn(n_per_row=args.n, seed=seed)
    state["truth"] = trusses

    print(f"  building the house — {len(trusses)} trusses, "
          f"{sum(t.n_fruit for t in trusses)} fruit, "
          f"{sum(1 for t in trusses if t.red_fraction >= th)} ripe at "
          f"{th:.2f}")

    model = ft.build(aisle=args.aisle, arms=("a",), trusses=trusses,
                     wrist_cam=True, deck_cam=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state["model"], state["data"] = model, data
    state["renderers"] = build_renderers(model)

    from farm.scout import ScoutHead

    try:
        state["head"] = ScoutHead(model, aisle=args.aisle)
    except RuntimeError:
        state["head"] = None

    sink = Sink(live=args.out is None, out=args.out, fps=args.fps,
                title="vinea - a truss shift in the greenhouse")
    state["sink"] = sink
    sink.push(compose())

    try:
        shift = ftr.run(seed=seed, aisle=args.aisle, n_per_row=args.n,
                        speed=args.speed, use_truth=args.truth,
                        max_stops=args.stops, on_tick=tick,
                        on_event=on_event, verbose=True, threshold=th,
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
