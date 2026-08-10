#!/usr/bin/env python3
"""One full row, two arms, two live windows: map, plan, travel, pick, crate.

This is the whole cycle in one command, watchable. Two OpenCV windows open, both
fed from offscreen EGL renders:

    WINDOW 1 — SENSORS                 WINDOW 2 — MISSION
    +--------------+--------------+    +----------------+----------------+
    | arm1 wrist   | arm2 wrist   |    | THE MAP        | DOWN THE AISLE |
    | HSV ripeness | HSV ripeness |    | what it found, | trolley + both |
    +--------------+--------------+    | and what it    | arms working   |
    | arm1 deck    | arm2 deck    |    | did about it   |                |
    | own row, own | own row, own |    +----------------+----------------+
    | pan/tilt     | pan/tilt     |    | PIPELINE       | PER-ARM STATS  |
    +--------------+--------------+    | what each arm  | pick times,    |
                                       | is doing NOW   | running mean   |
                                       +----------------+----------------+

⚠️ **Both windows are OpenCV windows over offscreen EGL renders, and there is no
GLFW viewer anywhere in this process.** `mujoco.viewer.launch_passive` needs a
GLFW context and `mujoco.Renderer` under `MUJOCO_GL=egl` needs an EGL one; the
two do not coexist in one process on this machine, which `COMMANDS.md` already
records. Everything here goes through `week3_watch.Sink`, one per window — it
already owns the quit handling, the window lifecycle and the mp4 writer.

--- the run ------------------------------------------------------------------

    1. MAP      drive the aisle, both deck heads scanning their own rows,
                fusing one house map. Nothing is picked in this pass.
    2. PLAN     one route per arm, merged into one trolley itinerary. Visible
                in window 2 before either arm commits.
    3. TRAVEL   drive to each stop.
    4. PICK     both arms work, one at a time. See `farm.duo` for why that is
                forced rather than chosen.
    5. CRATE    each arm has its own crate, riding on the deck beside it.

--- three things this window is careful about --------------------------------

⚠️ **The pipeline text is read out of the running mission, not scripted.**
`farm.duo.ArmState` is written where the work happens and the current leg is
read live from `mission.Guard.leg`, which `week2_pick.execute` sets as it flies.
There is no list of captions and no timer driving them.

⚠️ **The map is what the robot believes, not what is there.** Every dot is a
`scout.Sighting` — a position the deck cameras estimated and a stage the HSV
classifier called. Ground truth is drawn faintly underneath so a wrong dot reads
as wrong. `--truth` replaces the map with the operator's own answer and says so.

⚠️ **The stats are `execute`'s returned `seconds`**, the executor's simulated
clock, banked per arm. Not wall time — wall time here is a statement about this
laptop and how many panels it was compositing.

    ./.venv/bin/python simulation/mujoco/two_arm_farm.py            # the row
    ./.venv/bin/python simulation/mujoco/two_arm_farm.py --seed 7   # repeat one
    ./.venv/bin/python simulation/mujoco/two_arm_farm.py --truth    # skip mapping
    ./.venv/bin/python simulation/mujoco/two_arm_farm.py --shot     # stills, no run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from farm import crop as fcrop  # noqa: E402
from farm import decks, duo, house, overlay, trolley  # noqa: E402

# --- panel geometry ----------------------------------------------------------

SENS_W, SENS_H = 480, 360        # one tile of the 2x2 sensors window
MISS_W, MISS_H = 620, 430        # one tile of the 2x2 mission window

INK = (232, 232, 232)
DIM = (128, 128, 128)
ACCENT = (140, 250, 150)
WARN = (110, 110, 250)
AMBER = (90, 200, 245)
ARM_COL = {"a": (235, 190, 110), "b": (150, 210, 255)}

# Stage colours, BGR, matched to `farm.overlay.STAGE_BGR` so a dot on the map is
# the same colour as the box drawn round that fruit in a camera panel.
STAGE_BGR = dict(overlay.STAGE_BGR)


def _title(img, text, colour=INK):
    """A title bar across the top of a panel."""
    import cv2

    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (18, 18, 18), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1,
                cv2.LINE_AA)
    return img


def _missing(w, h, text, sub=""):
    """A panel for a camera that is not on this machine.

    ⚠️ Drawn as an explicit card rather than left black or filled with the other
    arm's view. `farm/eyes.py` makes the same argument and it is the right one: a
    split view that silently shows one camera twice is exactly the failure these
    windows exist to make visible.
    """
    import cv2

    img = np.full((h, w, 3), 20, np.uint8)
    cv2.putText(img, text, (16, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (95, 95, 95), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (16, h // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (72, 72, 72), 1, cv2.LINE_AA)
    return img


def _text_panel(w, h, lines, title=None, colour=INK):
    import cv2

    img = np.full((h, w, 3), 22, np.uint8)
    y = 24
    if title:
        _title(img, title, colour)
        y = 48
    for text, col in lines:
        if y > h - 8:
            break
        cv2.putText(img, text[:64], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    col, 1, cv2.LINE_AA)
        y += 18
    return img


# --- the map -----------------------------------------------------------------

class DuoMapPanel:
    """A top-down plan of the house: what was found, and what happened to it.

    ⚠️ Drawn with the **long axis horizontal**. The house is 8 m along the row
    and 5.8 m across, so putting y across the screen fills the panel; the other
    way round wastes half of it on empty aisle. `farm/watch.py` chose the same
    and for the same reason.
    """

    def __init__(self, w=MISS_W, h=MISS_H, pad=30):
        self.w, self.h, self.pad = w, h, pad
        self.y0, self.y1 = -house.HOUSE_HALF_Y - 0.6, house.HOUSE_HALF_Y + 0.6
        self.x0 = house.ROW_X0 - 0.9
        self.x1 = house.row_x(house.N_ROWS - 1) + 0.9
        sx = (w - 2 * pad) / (self.y1 - self.y0)
        sy = (h - 2 * pad - 34) / (self.x1 - self.x0)
        self.s = min(sx, sy)

    def px(self, x, y):
        u = self.pad + (y - self.y0) * self.s
        v = self.pad + 26 + (x - self.x0) * self.s
        return int(round(u)), int(round(v))

    def draw(self, state, data):
        import cv2

        img = np.full((self.h, self.w, 3), 22, np.uint8)
        aisle = state.aisle

        # --- the building ----------------------------------------------------
        for i in range(house.N_ROWS):
            a = self.px(house.row_x(i), self.y0)
            b = self.px(house.row_x(i), self.y1)
            worked = any(state.state[t].row == i for t in state.arms)
            cv2.line(img, a, b, (52, 74, 52) if worked else (40, 50, 40),
                     9 if worked else 6)
            cv2.putText(img, f"r{i}", (5, a[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (110, 150, 110) if worked else (70, 80, 70), 1,
                        cv2.LINE_AA)
        for _i, ax in house.aisles():
            cv2.line(img, self.px(ax, self.y0), self.px(ax, self.y1),
                     (56, 56, 56), 1, cv2.LINE_AA)

        # --- ground truth, faint, so a wrong dot reads as wrong --------------
        #
        # ⚠️ Drawn *offset upward by 6 px*, not on top of the mapped dot. The
        # first version put both at the same pixel in dark grey, and on the two
        # worked rows — which are drawn as a thick green bar — the truth markers
        # were simply invisible, so the panel showed the two rows the robot cares
        # about as empty and the two it ignores as full. A faint marker that
        # cannot be seen against its own background is not restraint, it is a
        # missing feature.
        for t in state.trusses:
            u, v = self.px(t.pos[0], t.pos[1])
            cv2.circle(img, (u, v - 6), 2, (120, 120, 120), -1, cv2.LINE_AA)

        # --- the routes ------------------------------------------------------
        ax = house.aisle_x(aisle)
        for tag, r in (state.routes or {}).items():
            if not r.stops:
                continue
            col = ARM_COL[tag]
            pts = [self.px(ax, s.y) for s in r.stops]
            for p, q in zip(pts, pts[1:]):
                cv2.line(img, p, q, tuple(int(c * 0.55) for c in col), 1,
                         cv2.LINE_AA)
            for p in pts:
                cv2.drawMarker(img, p, col, cv2.MARKER_TRIANGLE_UP, 8, 1)

        # --- what the robot mapped, and what it did about it -----------------
        targets = {state.state[t].target for t in state.arms
                   if state.state[t].target}
        if state.house_map is not None:
            for s in state.house_map.sightings:
                u, v = self.px(s.pos[0], s.pos[1])
                col = STAGE_BGR.get(s.stage, INK)
                name = s.truth.name if s.truth is not None else None
                # The name a sighting was matched to, if it has been attempted.
                for n in (name,):
                    pass
                picked = name in state.picked if name else False
                refused = name in state.refused if name else False
                missed = name in state.missed if name else False
                skipped = id(s) in state.skipped
                targeted = name in targets if name else False

                if picked:
                    # Taken. A hollow ring with a cross, so the row empties.
                    cv2.circle(img, (u, v), 5, col, 1, cv2.LINE_AA)
                    cv2.line(img, (u - 3, v - 3), (u + 3, v + 3), col, 1)
                    cv2.line(img, (u - 3, v + 3), (u + 3, v - 3), col, 1)
                elif refused:
                    cv2.drawMarker(img, (u, v), WARN, cv2.MARKER_TILTED_CROSS,
                                   9, 2)
                elif missed:
                    cv2.drawMarker(img, (u, v), (80, 80, 200),
                                   cv2.MARKER_DIAMOND, 8, 1)
                else:
                    cv2.circle(img, (u, v), 4, col, -1, cv2.LINE_AA)
                if s.ripe and not (picked or refused or missed):
                    ring = DIM if skipped else col
                    cv2.circle(img, (u, v), 7, ring, 1, cv2.LINE_AA)
                if targeted:
                    cv2.circle(img, (u, v), 10, ACCENT, 1, cv2.LINE_AA)

        # --- the machine -----------------------------------------------------
        ty = float(data.body(trolley.TROLLEY).xpos[1])
        p = self.px(ax - trolley.DECK_W / 2, ty - trolley.DECK_L / 2)
        q = self.px(ax + trolley.DECK_W / 2, ty + trolley.DECK_L / 2)
        cv2.rectangle(img, p, q, (215, 200, 130), 1, cv2.LINE_AA)
        for tag in state.arms:
            col = ARM_COL[tag]
            c = self.px(ax + trolley.ARM_X[tag], ty + trolley.ARM_Y[tag])
            live = state.active == tag
            cv2.drawMarker(img, c, col, cv2.MARKER_SQUARE, 8, 2 if live else 1)
            tgt = state.state[tag].target
            if tgt:
                try:
                    fp = data.body(tgt).xpos
                    cv2.line(img, c, self.px(fp[0], fp[1]), ACCENT, 1,
                             cv2.LINE_AA)
                except KeyError:
                    pass

        # --- legend ----------------------------------------------------------
        _title(img, "THE MAP  —  what the robot believes it found")
        y = self.h - 8
        x = 8
        for nm in ("green", "breaker", "turning", "red"):
            cv2.circle(img, (x + 4, y - 4), 4, STAGE_BGR[nm], -1, cv2.LINE_AA)
            cv2.putText(img, nm, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        DIM, 1, cv2.LINE_AA)
            x += 14 + 7 * len(nm)
        cv2.putText(img, "O ripe  x picked  X refused  <> lost  () target",
                    (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1,
                    cv2.LINE_AA)
        cv2.putText(img, "small grey dot above = ground truth", (8, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 110, 110), 1,
                    cv2.LINE_AA)
        return img


class AisleCam:
    """A slow-tracking camera on the aisle: the 'is the machine working' shot.

    ⚠️ **Tracking, not bolted.** The model's own `aisle` camera stands at the
    end of the bay and is the right shot for a still — but the trolley drives
    8 m down the row, so over a run the machine shrinks to a few pixels at the
    far end and the shot stops being about the machine at all.

    ⚠️ **And it lags rather than locks.** A camera welded to the trolley's y
    makes the trolley stationary and the greenhouse fly past, which reads as a
    scrolling background and hides the one thing this panel exists to show —
    that the chassis is travelling. A first-order lag at `LAG` keeps the machine
    in frame while letting it visibly move within it.

    A free `MjvCamera` rather than a camera in the model: it needs no recompile
    (the model must not be rebuilt mid-run) and nothing else in the scene can be
    steered by it.
    """

    # ⚠️ Azimuth 90 puts the camera square behind the machine looking straight
    # down the aisle, which is the framing that shows the trolley, both arms,
    # both deck masts and both crates at once, with the crop rows walling the
    # shot in. Angles off it (115 deg was tried) swing the camera into the near
    # row and fill the frame with leaves — the aisle is only 1.6 m wide, so
    # there is very little angular room between "down the aisle" and "inside the
    # crop".
    LAG = 0.06            # per panel frame, toward the trolley's y
    DIST = 4.5
    AZIMUTH = 90.0
    ELEVATION = -12.0
    HEIGHT = 0.85

    def __init__(self, model, data, aisle=0):
        import mujoco

        self.cam = mujoco.MjvCamera()
        self.cam.distance = self.DIST
        self.cam.azimuth = self.AZIMUTH
        self.cam.elevation = self.ELEVATION
        self.ax = house.aisle_x(aisle)
        self.y = float(data.body(trolley.TROLLEY).xpos[1])
        self._apply()

    def _apply(self):
        self.cam.lookat[:] = [self.ax, self.y, self.HEIGHT]

    def follow(self, data):
        target = float(data.body(trolley.TROLLEY).xpos[1])
        self.y += (target - self.y) * self.LAG
        self._apply()
        return self.cam


# --- the pipeline and stats panels -------------------------------------------

def pipeline_lines(state, data):
    """What the robot is doing right now, per arm. Read from `state`, not acted.

    Every line here is either a field `farm.duo` wrote where the work happened
    or something derived from one. See that module's note on exposure.
    """
    out = [(f"PHASE   {state.phase.upper()}", ACCENT)]
    if state.detail:
        out.append((f"        {state.detail}", DIM))
    out.append(("", INK))

    for tag in state.arms:
        st = state.state[tag]
        col = ARM_COL[tag]
        live = state.active == tag
        mark = ">" if live else " "
        out.append((f"{mark} {st.line()}", col if live else DIM))
        pan = st.deck_pan(data)
        bits = [f"row r{st.row}", f"deck pan {pan:+.0f}deg"]
        if st.stage:
            bits.append(st.stage)
        g = st.guard
        if g is not None and np.isfinite(getattr(g, "min_seen", np.inf)):
            bits.append(f"min clear {g.min_seen * 1000:.0f}mm")
        out.append((f"    {'  '.join(bits)}", DIM))
        if st.phase == "refused" and st.detail:
            out.append((f"    {st.detail[:56]}", WARN))
        out.append(("", INK))

    if state.active is None and state.phase == "pick":
        out.append(("  both arms idle — serialised, see farm/duo.py", DIM))
    elif state.active:
        other = [t for t in state.arms if t != state.active]
        if other:
            out.append((f"  arm{2 if state.active == 'a' else 1} is STOWED "
                        f"while arm{1 if state.active == 'a' else 2} flies",
                        AMBER))
    return out


def stats_lines(state):
    out = []
    tot = state.totals()
    for tag in state.arms:
        st = state.state[tag]
        s = st.stats
        out.append((f"{st.name.upper()}  row r{st.row}", ARM_COL[tag]))
        out.append((f"   crated {s.crated:<3} refused {s.refused:<3} "
                    f"missed {s.missed:<3}", INK))
        last = f"{s.last_s:.1f}s" if s.pick_s else "-"
        mean = f"{s.mean_s:.1f}s" if s.pick_s else "-"
        out.append((f"   last pick {last:<8} running mean {mean}", INK))
        if s.pick_s:
            times = "  ".join(f"{v:.0f}" for v in s.pick_s[-9:])
            out.append((f"   picks (s): {times}", DIM))
        out.append((f"   total flying {s.total_s:.1f}s", DIM))
        out.append(("", INK))

    out.append((f"BOTH  crated {tot['crated']}  refused {tot['refused']}  "
                f"missed {tot['missed']}", ACCENT))
    if state.house_map is not None:
        hm = state.house_map
        out.append((f"      map {len(hm.sightings)} found, "
                    f"{len(hm.ripe)} called ripe", INK))
    ripe = sum(1 for t in state.trusses if t.ripe)
    out.append((f"      house truly has {ripe} ripe", DIM))
    out.append(("", INK))
    out.append(("seconds are execute()'s sim clock, not wall time", DIM))
    return out


# --- the compositor ----------------------------------------------------------

class Windows:
    """Two `Sink`s, five cameras, three cadences.

    ⚠️ **The cadences are decoupled on purpose and this is the performance
    story.** Five camera renders plus an HSV pass per frame, on top of physics,
    will not run at the control rate and must not be made to by speeding the
    physics up or cutting a measured constant. Instead:

        physics       every control cycle, `reach.CTRL_DT`, untouched
        panels        `--fps` (default 10) — the renders
        HSV overlay   `--hsv-hz` (default 3) — the detector, whose boxes are
                      cached and re-drawn on the intervening frames

    The overlay cadence is the one that matters: `overlay.annotate` runs an HSV
    threshold, a contour pass and a circularity filter over a full frame, twice
    (once per wrist), and it costs more than the renders do. Caching its boxes
    changes how often the labels update, not what they say.
    """

    def __init__(self, model, data, state, fps=10, hsv_hz=3.0, out=None,
                 arms=("a", "b")):
        import mujoco

        from week3_watch import Sink

        self.model, self.data, self.state = model, data, state
        self.arms = tuple(arms)
        self.map = DuoMapPanel()
        self.detector = None
        self._boxes = {}
        self.frames = 0
        self._phase = {}
        import time as _t2

        self._last = _t2.perf_counter()

        from reach import CTRL_DT

        self.every = max(1, int(round((1.0 / max(fps, 1e-6)) / CTRL_DT)))
        self.hsv_every = max(1, int(round(fps / max(hsv_hz, 1e-6))))
        self._panel_n = 0

        self.wrist = {t: (trolley.ARM_PREFIX[t] + "wrist") for t in self.arms}
        self.deck = {t: decks.DECK_CAM[t] for t in self.arms}

        def has(cam):
            try:
                model.camera(cam)
                return True
            except KeyError:
                return False

        self.r = {}
        for t in self.arms:
            for cam, (w, h) in ((self.wrist[t], (SENS_W, SENS_H)),
                                (self.deck[t], (SENS_W, SENS_H))):
                if has(cam):
                    self.r[cam] = mujoco.Renderer(model, height=h, width=w,
                                                  max_geom=30000)
        self.r["aisle"] = mujoco.Renderer(model, height=MISS_H, width=MISS_W,
                                          max_geom=30000)
        self.aisle_cam = AisleCam(model, data, aisle=state.aisle)

        self.sensors = Sink(live=out is None, out=None if out is None
                            else f"{out}_sensors.mp4", fps=fps,
                            title="vinea — SENSORS  (2 wrist + 2 deck)")
        self.mission = Sink(live=out is None, out=None if out is None
                            else f"{out}_mission.mp4", fps=fps,
                            title="vinea — MISSION  (map, aisle, pipeline)")

    def close(self):
        for r in self.r.values():
            try:
                r.close()
            except Exception:
                pass
        for s in (self.sensors, self.mission):
            s.close()

    def _render(self, cam, w, h):
        import cv2

        if cam not in self.r:
            return None
        self.r[cam].update_scene(self.data, camera=cam)
        return cv2.cvtColor(self.r[cam].render(), cv2.COLOR_RGB2BGR)

    def _wrist_panel(self, tag, fresh):
        import cv2

        from farm.scout import StageDetector

        cam = self.wrist[tag]
        img = self._render(cam, SENS_W, SENS_H)
        st = self.state.state[tag]
        if img is None:
            return _missing(SENS_W, SENS_H, f"{st.name} wrist: not fitted",
                            "build with wrist_cam=True")
        # ⚠️ **The detector runs at `--hsv-hz`; the boxes are drawn every frame.**
        # The first cut cached only the *counts* and drew boxes on detector
        # frames alone, which at 3 Hz against 10 fps panels meant boxes appeared
        # on one frame in three and blinked. Caching the calls and re-drawing
        # them is both cheaper to look at and honest, provided the held frames
        # say they are held — which `overlay.draw(stale=True)` does by thinning
        # the box, and the caption repeats in words.
        stale = not (fresh or cam not in self._boxes)
        if not stale:
            if self.detector is None:
                self.detector = StageDetector()
            self._boxes[cam] = overlay.find(img, detector=self.detector)
        counts = overlay.draw(img, self._boxes.get(cam, []), stale=stale)
        overlay.tally(img, counts)
        _title(img, f"{st.name.upper()} WRIST  —  row r{st.row}  —  HSV "
                    f"{'held' if stale else 'live'}", ARM_COL[tag])
        return img

    def _deck_panel(self, tag):
        import cv2

        cam = self.deck[tag]
        img = self._render(cam, SENS_W, SENS_H)
        st = self.state.state[tag]
        if img is None:
            return _missing(SENS_W, SENS_H, f"{st.name} deck: not fitted",
                            "build with arm_decks=True")
        pan, tilt = (st.head.current(self.data) if st.head is not None
                     else (0.0, 0.0))
        _title(img, f"{st.name.upper()} DECK  —  row r{st.row}  —  "
                    f"pan {pan:+.0f}deg", ARM_COL[tag])
        # On its own strip rather than floated over the render — text at
        # `SENS_H - 10` sits on the tile seam and the descenders are clipped by
        # the `vstack` below it.
        cv2.rectangle(img, (0, SENS_H - 22), (img.shape[1], SENS_H),
                      (24, 24, 24), -1)
        cv2.putText(img, f"{st.phase}  {st.detail}"[:52], (8, SENS_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)
        return img

    def compose_sensors(self, fresh):
        tiles_top, tiles_bot = [], []
        for tag in ("a", "b"):
            if tag not in self.arms:
                nm = f"arm{1 if tag == 'a' else 2}"
                tiles_top.append(_missing(SENS_W, SENS_H, f"{nm} not fitted",
                                          "run with --arms 2"))
                tiles_bot.append(_missing(SENS_W, SENS_H, f"{nm} not fitted",
                                          "run with --arms 2"))
                continue
            tiles_top.append(self._wrist_panel(tag, fresh))
            tiles_bot.append(self._deck_panel(tag))
        return np.vstack([np.hstack(tiles_top), np.hstack(tiles_bot)])

    def compose_mission(self):
        st = self.state
        import cv2

        m = self.map.draw(st, self.data)
        self.r["aisle"].update_scene(self.data,
                                     camera=self.aisle_cam.follow(self.data))
        aisle = cv2.cvtColor(self.r["aisle"].render(), cv2.COLOR_RGB2BGR)
        ty = float(self.data.body(trolley.TROLLEY).xpos[1])
        _title(aisle, f"DOWN THE AISLE  —  trolley at y={ty:+.2f} m, "
                      f"both arms")
        pipe = _text_panel(MISS_W, MISS_H, pipeline_lines(st, self.data),
                           "PIPELINE  —  what each arm is doing now")
        stats = _text_panel(MISS_W, MISS_H, stats_lines(st),
                            "PER-ARM STATS  —  from execute()'s seconds")
        return np.vstack([np.hstack([m, aisle]), np.hstack([pipe, stats])])

    def tick(self, _t=None):
        """The `on_tick` handed to the runner. Cheap on the frames it skips."""
        self.frames += 1
        if self.frames % self.every:
            return
        self._panel_n += 1
        fresh = (self._panel_n % self.hsv_every) == 0
        self.sensors.push(self.compose_sensors(fresh))
        self.mission.push(self.compose_mission())
        # ⚠️ Booked per phase, not as one average. The mapping pass renders two
        # 1280x960 depth+RGB sensor pairs per scan pose on top of the panels,
        # and the harvest does not — so a single mean over the whole run is a
        # number that describes neither. See `rates`.
        import time as _t2

        now = _t2.perf_counter()
        ph = self.state.phase
        book = self._phase.setdefault(ph, [0, 0.0])
        book[0] += 1
        book[1] += now - self._last
        self._last = now

    def rates(self):
        """(phase, frames, seconds, fps) per phase, busiest first."""
        out = [(ph, n, s, n / s if s > 0 else float("nan"))
               for ph, (n, s) in self._phase.items()]
        out.sort(key=lambda r: -r[2])
        return out

    def flush(self, n=1):
        for _ in range(n):
            self._panel_n += 1
            self.sensors.push(self.compose_sensors(True))
            self.mission.push(self.compose_mission())


# --- stills ------------------------------------------------------------------

def shot(model, data, state, out_dir=None):
    """Render every panel once and write it out, so the aim can be checked.

    ⚠️ The point is to *look* at these rather than trust a quaternion. A camera
    whose xyaxes are subtly wrong still renders a plausible greenhouse.
    """
    import cv2

    out_dir = Path(__file__).resolve().parents[2] if out_dir is None \
        else Path(out_dir)
    w = Windows(model, data, state, out=None)
    try:
        written = []
        for name, img in (("sensors", w.compose_sensors(True)),
                          ("mission", w.compose_mission())):
            p = out_dir / f"twoarm_{name}.png"
            cv2.imwrite(str(p), img)
            written.append(p.name)
        for tag in state.arms:
            for kind, cam in (("wrist", w.wrist[tag]), ("deck", w.deck[tag])):
                img = w._render(cam, SENS_W, SENS_H)
                if img is None:
                    continue
                p = out_dir / f"twoarm_{kind}_{tag}.png"
                cv2.imwrite(str(p), img)
                written.append(p.name)
        w.r["aisle"].update_scene(data, camera=w.aisle_cam.follow(data))
        p = out_dir / "twoarm_aisle.png"
        cv2.imwrite(str(p), cv2.cvtColor(w.r["aisle"].render(),
                                         cv2.COLOR_RGB2BGR))
        written.append(p.name)
    finally:
        w.close()
    print(f"  wrote {', '.join(written)}")
    return written


def where_is_everything(model, data, state):
    """Print each camera's lens position and forward vector. The arithmetic
    behind the stills — a still shows what it shows, this says where from."""
    cams = ["aisle"]
    for tag in state.arms:
        cams += [trolley.ARM_PREFIX[tag] + "wrist", decks.DECK_CAM[tag]]
    print(f"\n  {'camera':<10} {'lens xyz':<24} {'forward':<20} points at")
    for cam in cams:
        try:
            cid = model.camera(cam).id
        except KeyError:
            print(f"  {cam:<10} NOT IN MODEL")
            continue
        pos = data.cam_xpos[cid]
        fwd = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
        # Which row it is aimed at, if any: walk forward to the row planes.
        aim = "—"
        if abs(fwd[0]) > 1e-3:
            best = None
            for i in range(house.N_ROWS):
                t = (house.row_x(i) - pos[0]) / fwd[0]
                if t > 0.05:
                    hit = pos + t * fwd
                    if abs(hit[1] - pos[1]) < 4.0 and 0.2 < hit[2] < 2.0:
                        if best is None or t < best[0]:
                            best = (t, i, hit)
            if best:
                aim = f"row r{best[1]} at {best[0]:.2f} m, z={best[2][2]:.2f}"
        print(f"  {cam:<10} {str(np.round(pos, 3)):<24} "
              f"{str(np.round(fwd, 2)):<20} {aim}")


def main():
    import argparse
    import os
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=12, help="fruit per row")
    ap.add_argument("--arms", type=int, default=2, choices=(1, 2))
    ap.add_argument("--stops", type=int, default=None, help="cap the stops")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None,
                    help="repeat a house; omitted draws a new one and prints it")
    ap.add_argument("--truth", action="store_true",
                    help="skip the mapping pass and route the real crop")
    ap.add_argument("--stride", type=float, default=None,
                    help="metres between mapping stops")
    ap.add_argument("--fps", type=int, default=10, help="panel rate")
    ap.add_argument("--hsv-hz", type=float, default=3.0,
                    help="how often the ripeness overlay re-detects")
    ap.add_argument("--shot", action="store_true",
                    help="render every panel once and exit")
    ap.add_argument("--headless", action="store_true",
                    help="run with no windows at all")
    ap.add_argument("--out", default=None,
                    help="record to <out>_sensors.mp4 and <out>_mission.mp4")
    args = ap.parse_args()

    # ⚠️ EGL, always. Both windows are OpenCV over offscreen renders and there
    # is no GLFW viewer in this process — see the module docstring.
    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import mujoco

    arms = ("a", "b")[: args.arms]
    seed = fcrop.resolve_seed(args.seed)
    trusses = fcrop.spawn(n_per_row=args.n, seed=seed)
    ripe = sum(1 for t in trusses if t.ripe)
    print(f"  house: {len(trusses)} fruit, {ripe} ripe · arms "
          f"{', '.join(a.upper() for a in arms)} · seed {seed}")

    model = trolley.build(aisle=args.aisle, arms=arms, trusses=trusses,
                          wrist_cam=True, arm_decks=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = duo.DuoState(arms=arms, aisle=args.aisle)
    state.trusses = list(trusses)

    if args.shot:
        from mission import park_arm, reset_park

        from farm import armframe

        parks = {t: armframe.park_posture(model, data, t, arms=arms)
                 for t in arms}
        reset_park(model, data, parks[arms[0]],
                   prefix=trolley.ARM_PREFIX[arms[0]])
        for t in arms:
            park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
        hs = decks.heads(model, data, arms=arms, aisle=args.aisle)
        for t in arms:
            state.state[t].head = hs.get(t)
        # Point them apart, so the still shows two independent heads.
        if "a" in hs:
            hs["a"].aim(data, -decks.SCAN_HALF_DEG)
        if "b" in hs:
            hs["b"].aim(data, +decks.SCAN_HALF_DEG)
        mujoco.mj_forward(model, data)
        where_is_everything(model, data, state)
        shot(model, data, state)
        return 0

    windows = None if args.headless else Windows(
        model, data, state, fps=args.fps, hsv_hz=args.hsv_hz, out=args.out,
        arms=arms)
    on_tick = None if windows is None else windows.tick

    t0 = time.perf_counter()
    try:
        duo.run(model, data, trusses, state, arms=arms, aisle=args.aisle,
                speed=args.speed, use_truth=args.truth, max_stops=args.stops,
                on_tick=on_tick, stride=args.stride, verbose=True)
        if windows is not None:
            windows.flush(args.fps * 3)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        wall = time.perf_counter() - t0
        if windows is not None:
            n = windows._panel_n
            print(f"\n  --- frame rate ---")
            print(f"  {n} composed frames in {wall:.0f} s wall — "
                  f"{n / max(wall, 1e-9):.1f} panel fps overall "
                  f"({2 * n} window pushes across 2 windows)")
            print(f"\n  {'phase':<10} {'frames':>7} {'seconds':>9} {'fps':>7}")
            for ph, fr, secs, fps in windows.rates():
                print(f"  {ph:<10} {fr:>7} {secs:>9.0f} {fps:>7.1f}")
            print(f"\n  ⚠️ The mapping pass renders two 1280x960 RGB+depth "
                  f"sensor pairs per\n     scan pose on top of the four panels; "
                  f"the harvest renders panels only.\n     One mean over both "
                  f"describes neither — read the per-phase rates.")
            windows.close()
        else:
            print(f"\n  {wall:.0f} s wall, no windows")
    duo.report(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
