#!/usr/bin/env python3
"""Watch the perception pipeline work: scan the row, plan a route, fly it.

`week3_perceive.py` is the instrument — it measures and prints numbers. This is
the window onto the same pipeline, for the times when the question is "what is
it actually doing?" rather than "what is the p95?". Nothing here changes a
single measurement: it calls the same `Perception`, the same `Planner`, the
same executor, and it draws what they were already doing.

Three phases, and the phase name is on every frame:

    SCAN   the arm sweeps the staging plane along the row, taking one wrist
           frame per stop, detecting, deprojecting, and building a map of where
           it thinks every truss is. This is the part that looks like a robot
           looking at a plant.
    PLAN   for the fruit about to be picked, the checked route is drawn into
           the scene — every waypoint as a labelled marker, the previewed tool
           path as a line, the estimate and the ground truth as two markers you
           can see the gap between. Held on screen so it can be read.
    PICK   the route is flown. The waypoints stay drawn, so the arm can be
           watched arriving at each one.

Four views:

    --view deck    the fixed chassis camera, with the plan drawn into it
    --view wrist   the eye-in-hand sensor, with detections, estimates and the
                   ground-truth error printed on each box
    --view scene   a clean wide shot, no annotations — just the robot picking
    --view all     all three at once, plus a live stats panel

Live in a window, or straight to an mp4:

    ./.venv/bin/python simulation/mujoco/week3_watch.py --view all
    ./.venv/bin/python simulation/mujoco/week3_watch.py --view wrist --fruit t0
    ./.venv/bin/python simulation/mujoco/week3_watch.py --view all \\
        --out week3_watch.mp4

⚠️ Rendering is always offscreen (EGL) even for the live window, which is drawn
by OpenCV rather than by MuJoCo's own viewer. That is deliberate: compositing
four panels needs the frames as arrays, and mixing GLFW's context with
offscreen renderers is how the teardown segfault in the bug log gets invited
back. The cost is that the live view is not interactive — no orbiting, no
dragging. Use `week3_perceive.py --pick --windowed` for that.
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TILE_W, TILE_H = 640, 480

# Waypoint colours, by what the leg is for. The pull and the grip are the legs
# that touch the plant, so they are the ones worth picking out of a line of
# otherwise identical markers.
LEG_RGBA = {
    "settle":   (0.55, 0.55, 0.60, 0.9),
    "clear":    (0.35, 0.70, 1.00, 0.9),
    "lane":     (0.35, 0.70, 1.00, 0.9),
    "align":    (0.35, 0.70, 1.00, 0.9),
    "approach": (1.00, 0.80, 0.20, 0.95),
    "insert":   (1.00, 0.55, 0.10, 0.95),
    "grip":     (1.00, 0.30, 0.30, 0.95),
    "close":    (1.00, 0.30, 0.30, 0.95),
    "pull":     (1.00, 0.10, 0.45, 1.0),
    "grasp":    (1.00, 0.10, 0.45, 1.0),
    "extract":  (0.55, 1.00, 0.45, 0.95),
    "turn":     (0.55, 1.00, 0.45, 0.95),
    "carry":    (0.30, 0.95, 0.75, 0.95),
    "drop":     (0.30, 0.95, 0.75, 0.95),
    "release":  (0.30, 0.95, 0.75, 0.95),
}
DEFAULT_RGBA = (0.8, 0.8, 0.8, 0.9)


class Overlay:
    """What the drawn scene should currently contain.

    A mutable bag rather than arguments, because the render callback fires deep
    inside the executor's control loop and cannot be handed new parameters
    mid-leg.
    """

    def __init__(self):
        self.phase = "SCAN"
        self.target = None
        self.mission = None
        self.sightings = {}
        self.scan = {}          # name -> (estimate, error_m)
        self.note = ""
        self.results = []       # finished per-fruit lines for the stats panel


def _add(scene, gtype, size, pos, rgba, mat=None):
    """Append one decoration geom to a rendered scene, if there is room."""
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, gtype,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        (np.eye(3).flatten() if mat is None
         else np.asarray(mat, dtype=np.float64).flatten()),
        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1
    return g


def _line(scene, a, b, rgba, width=0.0025):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))
    scene.ngeom += 1


def decorate(scene, ov, data):
    """Draw the pipeline's internal state into a rendered scene.

    This is the answer to "figuring out the waypoints" — the route the planner
    checked is not otherwise visible anywhere, because `Planner.plan` moves
    nothing. It builds a leg list, previews it on a scratch `MjData`, measures
    clearances, and returns. Everything it decided is in `mission.legs` and
    `mission.path`, and until now those were only ever printed.
    """
    import mujoco

    # What the camera thinks it saw, against what is actually there. Two
    # markers with a line between them, so the error is a length on screen
    # rather than a number in a log.
    for name, (est, err) in ov.scan.items():
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], est,
             (0.20, 1.00, 0.90, 0.55))
        truth = data.body(name).xpos
        if err > 0.002:
            _line(scene, est, truth, (1.0, 0.9, 0.2, 0.9), width=0.0035)

    if ov.mission is None:
        return

    m = ov.mission

    # The previewed tool path — where the planner *believes* the tool will
    # travel. It is the thing the clearance was measured along, so it is the
    # thing worth seeing next to the crop.
    if getattr(m, "path", None) is not None and len(m.path) > 1:
        for a, b in zip(m.path[:-1], m.path[1:]):
            _line(scene, a, b, (0.30, 0.85, 1.00, 0.55), width=0.0016)

    # The waypoints themselves, in order, with a line joining consecutive ones.
    goals = [(leg.name, np.asarray(leg.goal, float))
             for leg in m.legs if leg.goal is not None]
    for i, (name, g) in enumerate(goals):
        rgba = LEG_RGBA.get(name, DEFAULT_RGBA)
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.011, 0, 0], g, rgba)
        if i:
            _line(scene, goals[i - 1][1], g, (*rgba[:3], 0.35), width=0.0012)


def _panel(w, h, lines, title=None):
    """A dark text tile for the composite view."""
    import cv2

    img = np.full((h, w, 3), 18, np.uint8)
    y = 34
    if title:
        cv2.putText(img, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (235, 235, 235), 2, cv2.LINE_AA)
        y += 30
    for text, colour in lines:
        if y > h - 10:
            break
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    colour, 1, cv2.LINE_AA)
        y += 22
    return img


def _banner(img, ov):
    """Phase, target and note, burned across the top of a tile."""
    import cv2

    colour = {"SCAN": (250, 200, 90), "PLAN": (120, 230, 255),
              "PICK": (140, 250, 150)}.get(ov.phase, (230, 230, 230))
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (18, 18, 18), -1)
    label = ov.phase + (f"  ->  {ov.target}" if ov.target else "")
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                colour, 2, cv2.LINE_AA)
    if ov.note:
        cv2.putText(img, ov.note, (250, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (215, 215, 215), 1, cv2.LINE_AA)
    return img


class Views:
    """Renders whichever panels the chosen view needs, and composites them.

    One renderer per camera, held open for the run. Rebuilding them per frame
    costs a GL context each time and is the difference between a watchable
    frame rate and a slideshow.
    """

    def __init__(self, model, data, sensor, detector, ov, view="all",
                 deck="row", scene_cam="aisle", size=(TILE_W, TILE_H),
                 row=None, wrist_every=4):
        import mujoco

        self.model, self.data = model, data
        self.sensor, self.detector, self.ov = sensor, detector, ov
        self.row = row
        self.view = view
        self.w, self.h = size
        self.deck_name, self.scene_name = deck, scene_cam
        # The wrist panel is a **live feed**: the detector re-runs as the arm
        # moves, so the boxes track the fruit instead of freezing on whatever
        # was last measured. Re-running it on every control tick would cost a
        # 1280x960 render plus a detect per 10 ms of sim time and drop the
        # frame rate through the floor, so it refreshes every `wrist_every`
        # pumped frames — fast enough to read as continuous, cheap enough to
        # watch.
        self.wrist_every = max(1, wrist_every)
        self._pumped = 0

        need = set()
        if view in ("deck", "all"):
            need.add(deck)
        if view in ("scene", "all"):
            need.add(scene_cam)
        self.renderers = {
            name: mujoco.Renderer(model, height=self.h, width=self.w,
                                  max_geom=20000)
            for name in need
        }
        # The wrist panel is drawn from the *sensor's* own frames, not a fresh
        # render, so the picture on screen is literally the array the detector
        # was given. Anything else risks showing a view the pipeline never saw.
        self._last_wrist = np.full((self.h, self.w, 3), 24, np.uint8)

    def _render(self, name, decorated=True):
        import cv2

        r = self.renderers[name]
        r.update_scene(self.data, camera=self.model.camera(name).id)
        if decorated:
            decorate(r.scene, self.ov, self.data)
        return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

    def set_wrist(self, bgr):
        import cv2

        self._last_wrist = cv2.resize(bgr, (self.w, self.h))

    def live_wrist(self, force=False):
        """Re-run the detector on a fresh sensor frame and redraw the panel.

        This is what makes the wrist view a feed rather than a slideshow: the
        boxes, the fitted radius, the 3D estimate and the ground-truth error
        are all recomputed from the frame the sensor is looking at *right now*,
        including while the arm is mid-pick with a tomato in the gripper.

        ⚠️ Nothing here feeds the planner. The pick was planned from the frame
        taken at the staging pose and it is not being re-planned; this is a
        monitor. Wiring it into the plan would be closed-loop visual servoing,
        which is a Week 4 question and a much bigger one — the arm occludes the
        fruit it is reaching for, and re-planning against a partially occluded
        target mid-approach is how you drive into a truss.
        """
        from detect import draw, estimate, match, truth_boxes

        self._pumped += 1
        if not force and self._pumped % self.wrist_every:
            return
        if self.view not in ("wrist", "all"):
            return

        rgb, depth = self.sensor.both(self.data)
        R, C = self.sensor.pose(self.data)
        intr = self.sensor.intr
        dets = [estimate(d, depth, intr, R, C) for d in self.detector(rgb)]
        names = self.row.names if self.row is not None else []
        truth = truth_boxes(self.model, self.data, intr, R, C, names)
        match(dets, truth)
        errors = {}
        for d in dets:
            if d.matched and d.est is not None:
                errors[d.matched] = float(
                    np.linalg.norm(d.est - truth[d.matched]["pos"]))
        self.set_wrist(draw(rgb, dets, truth, errors))

    def stats_lines(self):
        ov = self.ov
        out = [(f"detector  {self.detector.name}", (200, 200, 200)),
               (f"sensor    {self.sensor.name}", (200, 200, 200)),
               ("", (0, 0, 0)),
               ("row map (estimate vs truth)", (240, 210, 140))]
        for name in sorted(ov.scan):
            _, err = ov.scan[name]
            colour = ((140, 250, 150) if err < 0.005
                      else (120, 230, 255) if err < 0.015 else (110, 110, 250))
            out.append((f"  {name}   {err * 1000:6.1f} mm", colour))
        if ov.mission is not None:
            m = ov.mission
            out += [("", (0, 0, 0)),
                    (f"plan  {m.lane}  clearance "
                     f"{m.clearance * 1000:.0f} mm", (240, 210, 140))]
            for leg in m.legs:
                if leg.goal is None:
                    continue
                out.append((f"  {leg.name:<9} "
                            f"[{leg.goal[0]:+.3f} {leg.goal[1]:+.3f} "
                            f"{leg.goal[2]:+.3f}]", (185, 185, 185)))
        if ov.results:
            out.append(("", (0, 0, 0)))
            out.append(("done", (240, 210, 140)))
            out += ov.results[-4:]
        return out

    def frame(self):
        import cv2

        if self.view == "wrist":
            return _banner(self._last_wrist.copy(), self.ov)
        if self.view == "deck":
            return _banner(self._render(self.deck_name), self.ov)
        if self.view == "scene":
            # No decoration: this is the shot that has to look like a machine
            # working in a greenhouse, not like a debug view.
            return _banner(self._render(self.scene_name, decorated=False),
                           self.ov)

        deck = _banner(self._render(self.deck_name), self.ov)
        cv2.putText(deck, f"deck cam '{self.deck_name}' + plan",
                    (10, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (210, 210, 210), 1, cv2.LINE_AA)
        wrist = self._last_wrist.copy()
        cv2.putText(wrist, "wrist cam (what the detector sees)",
                    (10, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (210, 210, 210), 1, cv2.LINE_AA)
        scene = self._render(self.scene_name, decorated=False)
        cv2.putText(scene, f"scene '{self.scene_name}'",
                    (10, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (210, 210, 210), 1, cv2.LINE_AA)
        stats = _panel(self.w, self.h, self.stats_lines(),
                       getattr(self, "panel_title", "week 3 pipeline"))
        return np.vstack([np.hstack([deck, wrist]),
                          np.hstack([scene, stats])])

    def close(self):
        for r in self.renderers.values():
            try:
                r.close()
            except Exception:
                pass


class Sink:
    """Where frames go: an OpenCV window, an mp4, or both.

    ⚠️ **Quitting has to be handled here and it has to be forceful.**
    `cv2.imshow` *creates* a window when none exists, so closing the window
    with its X button does nothing at all — the very next frame builds it
    again, and it reads as the window refusing to close. Three ways out are
    wired up, and all three end the run rather than just hiding the window:

        * the **QUIT button** drawn in the top-right corner, clicked;
        * **q** or **Esc**;
        * the window's own **X**, detected via `WND_PROP_VISIBLE`.

    Any of them raises `KeyboardInterrupt` out of the render callback, which
    unwinds the executor from wherever it is mid-leg. That is the same
    mechanism `week2_pick.run_windowed` uses, and `execute` only catches
    `Aborted`, so it propagates cleanly.
    """

    QUIT_W, QUIT_H, QUIT_PAD = 140, 36, 12

    def __init__(self, live=True, out=None, fps=30, title="vinea - week 3",
                 on_click=None, on_key=None):
        # `on_click(x, y)` receives every left-click that is not on the QUIT
        # button, in window pixels. Week 4 uses it to place fruit by clicking
        # the deck panel; leaving it None keeps Week 3's behaviour exactly.
        #
        # `on_key(code)` receives every other keypress. It has to live here
        # because `push` owns the only `cv2.waitKey` — a second one in the
        # caller's loop competes for the same event queue and swallows keys at
        # random, which is exactly how Week 4's SPACE stopped working.
        self.on_key = on_key
        self.on_click = on_click
        self.live, self.out, self.fps, self.title = live, out, fps, title
        self.writer = None
        self.stopped = False
        self._window = False
        self._quit_rect = None

    # -- the window -----------------------------------------------------------
    def _ensure_window(self):
        import cv2

        if self._window:
            return
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self._on_mouse)
        self._window = True

    def _on_mouse(self, event, x, y, flags, param):
        import cv2

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._quit_rect is not None:
            x0, y0, x1, y1 = self._quit_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.stopped = True
                return
        if self.on_click is not None:
            self.on_click(x, y)

    def _draw_quit(self, frame):
        """Draw the button. On the *displayed* copy only — never the recording.

        A QUIT button burned into `week3_watch.mp4` would be in every frame of
        a clip that might end up in front of a grower.
        """
        import cv2

        h, w = frame.shape[:2]
        x1, y1 = w - self.QUIT_PAD, self.QUIT_PAD + self.QUIT_H
        x0, y0 = x1 - self.QUIT_W, self.QUIT_PAD
        self._quit_rect = (x0, y0, x1, y1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (48, 48, 205), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (235, 235, 235), 1)
        cv2.putText(frame, "QUIT  (q)", (x0 + 16, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return frame

    def _closed_by_user(self):
        """True once the window has been dismissed with its own X."""
        import cv2

        try:
            return cv2.getWindowProperty(self.title,
                                         cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            # The window is gone entirely, which is the same answer.
            return True

    # -- frames ---------------------------------------------------------------
    def push(self, frame):
        import cv2

        if self.stopped:
            raise KeyboardInterrupt

        if self.out is not None:
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    self.out, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError(f"could not open writer for {self.out}")
            self.writer.write(frame)

        if self.live:
            self._ensure_window()
            cv2.imshow(self.title, self._draw_quit(frame.copy()))
            # 1 ms is enough to pump the event queue and deliver the mouse
            # callback. Physics is much slower than real time here, so no extra
            # pacing is needed — unlike the passive viewer, which outruns the
            # wall clock and needs a sleep.
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                self.stopped = True
            elif k != 255 and self.on_key is not None:
                self.on_key(k)
            if not self.stopped and self._closed_by_user():
                self.stopped = True
            if self.stopped:
                self.close_window()
                raise KeyboardInterrupt

    def hold(self, frame, seconds):
        for _ in range(max(1, int(seconds * self.fps))):
            self.push(frame)

    def close_window(self):
        import cv2

        if self._window:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            # Windows only actually go away once the event loop runs again.
            for _ in range(4):
                cv2.waitKey(1)
            self._window = False

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"\n  wrote {self.out}")
        self.close_window()


def scan(model, data, row, park_q, sensor, detector, views, sink, ov,
         stops=6, speed=None):
    """Sweep the staging plane along the row, looking. Returns the row map.

    ⚠️ This is a **scan, not a measurement run.** It exists so the map the
    planner uses comes from looking at the row rather than from `fruit_home`,
    and so there is something to watch. The numbers it produces are the same
    numbers `week3_perceive.py --budget` reports; if the two ever disagree,
    trust the budget, because this one is optimised for being watchable.

    The arm is *not* reset between stops — a scan is one continuous traverse,
    which is both what a real machine does and what makes the video read as a
    machine working rather than a sequence of teleports.
    """
    from camera import stage
    from detect import draw, truth_boxes
    from mission import STAGE_X
    from week3_perceive import Perception
    from plant_row import fruit_home

    ov.phase = "SCAN"
    ov.target = None
    per = Perception(model, sensor, detector,
                     {n: fruit_home(n) for n in row.names})

    ys = np.linspace(0.28, -0.26, stops)
    z = 0.64
    print(f"\n  SCAN — {stops} stops along the row at x={STAGE_X:.2f}, "
          f"z={z:.2f}")
    print(f"  {'stop':>5} {'y':>7}   seen")

    first = True
    for i, y in enumerate(ys):
        ov.note = f"stop {i + 1}/{stops}   y={y:+.2f} m"
        stage(model, data, park_q, np.array([STAGE_X, y, z]), row,
              speed=speed, reset=("arm" if first else None),
              on_tick=lambda _t=None: _pump(views, sink))
        first = False
        if sink.stopped:
            break

        sightings, rep = per.look(data)
        R, C = rep["R"], rep["C"]
        truth = truth_boxes(model, data, sensor.intr, R, C, row.names)
        errors = {n: float(np.linalg.norm(s.err))
                  for n, s in sightings.items()}
        views.set_wrist(draw(rep["rgb"], rep["dets"], truth, errors))

        for n, s in sightings.items():
            # Keep the best look at each truss. A fruit seen from four stops
            # should be remembered from the stop that saw it squarest, not from
            # whichever one happened to come last.
            e = float(np.linalg.norm(s.err))
            if n not in ov.scan or e < ov.scan[n][1]:
                ov.scan[n] = (s.est.copy(), e)

        print(f"  {i + 1:5d} {y:+7.2f}   "
              + (", ".join(f"{n} {errors[n] * 1000:.1f}mm"
                           for n in sorted(sightings)) or "-"))
        sink.hold(views.frame(), 0.7)

    ov.note = ""
    found = sorted(ov.scan)
    missing = [n for n in row.names if n not in ov.scan]
    print(f"\n  row map: {len(found)}/{len(row.names)} trusses located"
          + (f"   NOT SEEN: {', '.join(missing)}" if missing else ""))
    for n in found:
        print(f"    {n}  {ov.scan[n][0].round(4)}  "
              f"err {ov.scan[n][1] * 1000:.1f} mm")
    return ov.scan


def _pump(views, sink):
    views.live_wrist()
    sink.push(views.frame())


def run(model, data, row, park_q, sensor, detector, views, sink, ov,
        targets, speed=None, clearance=None, stops=6):
    """Scan the row, then plan and pick each target, drawing as it goes."""
    import mujoco

    from camera import stage
    from detect import draw, truth_boxes
    from incident import Blackbox
    from mission import (CLEARANCE, DEFAULT_SPEED, GUARD_STOP, Guard, Planner,
                         STAGE_X, reset_park)
    from week2_pick import anchor_posture, execute, make_reacher
    from week3_perceive import Perception, _plan_perceived
    from plant_row import fruit_home
    from reach import Gripper

    clearance = CLEARANCE if clearance is None else clearance
    # ⚠️ `None` means "the default", and it has to be resolved here rather than
    # passed on: `Reacher` multiplies the speed by each joint's velocity limit
    # and `None` is not a number. `camera.stage` resolves it the same way.
    speed = DEFAULT_SPEED if speed is None else speed

    reset_park(model, data, park_q)
    row.reset()
    mujoco.mj_forward(model, data)

    scan(model, data, row, park_q, sensor, detector, views, sink, ov,
         stops=stops, speed=speed)

    per = Perception(model, sensor, detector,
                     {n: fruit_home(n) for n in row.names})
    # Published on the sink so a quit mid-run can still report the picks that
    # completed. Same list object, mutated in place, so appends are visible to
    # the handler in `main`.
    log = []
    sink.partial_log = log

    for name in targets:
        if sink.stopped:
            break
        if not row.attached(name):
            continue

        ov.phase = "PLAN"
        ov.target = name
        ov.mission = None

        # Look once more from directly in front of the target. The scan's map
        # is what tells the arm where to go and look; this frame is what the
        # plan is actually built on.
        nominal = ov.scan.get(name, (fruit_home(name), 0.0))[0]
        ov.note = "moving to look"
        # ⚠️ **`reset="arm"` — park the posture, keep the row.** Two separate
        # things have to be true here and they pull in opposite directions.
        #
        # The posture must be reset: the scan is one continuous traverse, and
        # carrying it straight into the pick was worth 1/1 -> 0/1 on t3, with
        # an estimate of 1.7 mm and a route clearing by 62 mm. The grasp missed
        # because the arm arrived from a posture the scan had wound it into.
        # That is Week 2's `anchor_posture` finding — the same point in space
        # reached with the shoulder swung round is a different problem for
        # local differential IK.
        #
        # The *crop* must not be reset: this is a row harvest, so fruit already
        # picked are in the crate and their stems are broken, and they have to
        # stay that way. A full `mj_resetData` does both at once and there is no
        # flag on it to do only the first, which is why `park_arm` exists.
        stage(model, data, park_q,
              np.array([STAGE_X, nominal[1], nominal[2]]), row, speed=speed,
              reset="arm", on_tick=lambda _t=None: _pump(views, sink))

        sightings, rep = per.look(data)
        truth = truth_boxes(model, data, sensor.intr, rep["R"], rep["C"],
                            row.names)
        errors = {n: float(np.linalg.norm(s.err))
                  for n, s in sightings.items()}
        views.set_wrist(draw(rep["rgb"], rep["dets"], truth, errors))
        for n, s in sightings.items():
            ov.scan[n] = (s.est.copy(), float(np.linalg.norm(s.err)))

        if name not in sightings:
            ov.note = "NOT DETECTED - logged as a miss"
            print(f"\n  {name}: MISS — not detected")
            sink.hold(views.frame(), 2.0)
            log.append({"fruit": name, "seen": False, "err_mm": float("nan"),
                        "in_bin": False, "grasped": False, "clean": False,
                        "refused": None, "note": "not detected"})
            ov.results.append((f"  {name}  MISS", (110, 110, 250)))
            continue

        s = sightings[name]
        ov.note = f"estimate err {s.err_mm:.1f} mm"
        print(f"\n  {name}: estimate {s.est.round(4)} vs truth "
              f"{s.truth.round(4)} -> {s.err_mm:.1f} mm")

        reacher = make_reacher(model, data, speed=speed)
        anchor_posture(reacher, model, data, park_q)
        gripper = Gripper(model, data)
        planner = Planner(model, data, row, lessons=None, clearance=clearance,
                          park_q=park_q, speed=speed)
        mission = _plan_perceived(planner, row, name, sightings)
        ov.mission = mission

        print(f"  {name}: {mission.summary().splitlines()[0]}")
        for leg in mission.legs:
            if leg.goal is not None:
                print(f"      {leg.name:<9} -> [{leg.goal[0]:+.3f} "
                      f"{leg.goal[1]:+.3f} {leg.goal[2]:+.3f}]")

        if not mission.ok:
            ov.note = "REFUSED"
            sink.hold(views.frame(), 2.5)
            log.append({"fruit": name, "seen": True, "err_mm": s.err_mm,
                        "in_bin": False, "grasped": False, "clean": False,
                        "refused": "breach", "note": "refused"})
            ov.results.append((f"  {name}  REFUSED", (110, 110, 250)))
            continue

        # Hold the plan on screen before anything moves. The whole point of the
        # PLAN phase is that the route is checked and drawn *before* the arm
        # commits to it, which is exactly the property `Planner.plan` has and
        # nothing until now made visible.
        ov.note = (f"route '{mission.lane}' checked - clearance "
                   f"{mission.clearance * 1000:.0f} mm")
        sink.hold(views.frame(), 2.5)

        ov.phase = "PICK"
        box = Blackbox(model, data, row, name)
        guard = Guard(model, data, row, name,
                      stop=min(GUARD_STOP, 0.4 * clearance))
        guard.armed = False
        r = execute(mission, reacher, gripper, row, box=box, guard=guard,
                    on_tick=lambda _t=None: _pump(views, sink), verbose=False)
        clean = bool(r["in_bin"] and not r["lost"] and not r["disturbed"])
        print(f"  {name}: grasped={r['grasped']} crate={r['in_bin']} "
              f"{'CLEAN' if clean else 'not clean'}")
        log.append({"fruit": name, "seen": True, "err_mm": s.err_mm,
                    "in_bin": r["in_bin"], "grasped": r["grasped"],
                    "clean": clean, "refused": None,
                    "clearance": r["clearance"], "lost": r["lost"],
                    "disturbed": r["disturbed"], "note": ""})
        ov.results.append(
            (f"  {name}  {s.err_mm:4.1f} mm  "
             f"{'CLEAN' if clean else 'not clean'}",
             (140, 250, 150) if clean else (120, 230, 255)))
        ov.mission = None

    ov.phase = "DONE"
    ov.target = None
    ov.note = ""
    sink.hold(views.frame(), 2.0)
    return log


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", default="all",
                    choices=["all", "deck", "wrist", "scene"],
                    help="all = deck + wrist + scene + stats (default)")
    ap.add_argument("--deck-camera", default="row",
                    choices=["row", "aisle", "wide"],
                    help="which fixed camera is the 'deck' view")
    # `aisle` rather than `wide`: compared side by side at a pick pose, `wide`
    # is the establishing shot and puts the arm small in one corner, which is
    # the wrong frame for "just the robot picking". `aisle` looks down the path
    # with the arm, the truss and the greenhouse all legible.
    ap.add_argument("--scene-camera", default="aisle",
                    choices=["row", "aisle", "wide"],
                    help="which fixed camera is the clean 'scene' view")
    ap.add_argument("--fruit", nargs="+", default=None, metavar="tN",
                    help="pick only these (default: the whole row)")
    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"])
    ap.add_argument("--stops", type=int, default=6,
                    help="how many places the scan stops to look")
    ap.add_argument("--wrist-every", type=int, default=4,
                    help="re-run the detector for the live wrist feed every "
                         "Nth frame (1 = every frame, slowest and smoothest)")
    ap.add_argument("--speed", type=float, default=None,
                    help="fraction of rated joint speed; 0.4 is brisk")
    ap.add_argument("--clearance", type=float, default=None, help="mm")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--out", default=None, help="write an mp4")
    ap.add_argument("--no-window", action="store_true",
                    help="render to --out only, no live window")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    # Always offscreen — see the module docstring. The live window is OpenCV's,
    # not MuJoCo's, so this does not conflict with showing it.
    os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    from camera import SensorCamera
    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from plant_row import Row
    from week3_perceive import build_detector, report_pick

    model = build_scene(wrist_cam=True)
    data = mujoco.MjData(model)
    park = park_posture(model)
    reset_park(model, data, park)
    row = Row(model, data)
    mujoco.mj_forward(model, data)

    targets = list(row.names) if args.fruit is None else list(args.fruit)
    unknown = [n for n in targets if n not in row.names]
    if unknown:
        raise SystemExit(f"no such fruit: {', '.join(unknown)} "
                         f"(row has {', '.join(row.names)})")

    sensor = SensorCamera(model, "wrist")
    detector = build_detector(args.detector, args.conf)
    ov = Overlay()
    views = Views(model, data, sensor, detector, ov, view=args.view,
                  deck=args.deck_camera, scene_cam=args.scene_camera,
                  row=row, wrist_every=args.wrist_every)
    sink = Sink(live=not args.no_window, out=args.out, fps=args.fps)

    print(f"\n  view '{args.view}' · detector '{detector.name}' · "
          f"picking {', '.join(targets)}")
    if sink.live:
        print("  to stop early: click QUIT (top right), press q or Esc, or "
              "close the window")

    log = []
    try:
        log = run(model, data, row, park, sensor, detector, views, sink, ov,
                  targets, speed=args.speed,
                  clearance=None if args.clearance is None
                  else args.clearance / 1000.0,
                  stops=args.stops)
    except KeyboardInterrupt:
        # Quit button, q/Esc, the window's X, or Ctrl-C. All four land here,
        # and whatever the run managed before that is still worth printing —
        # three clean picks out of five is a result, and throwing it away
        # because the window was closed on the fourth is not.
        log = getattr(sink, "partial_log", log)
        print("\n  stopped early")
    finally:
        views.close()
        sensor.close()
        sink.close()
    if log:
        report_pick(log)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
