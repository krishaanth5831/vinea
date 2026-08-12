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
    | HSV ripeness | HSV ripeness |    +----------------+----------------+
    | own row+pan  | own row+pan  |    | PIPELINE       | PER-ARM STATS  |
    +--------------+--------------+    | what each arm  | pick times,    |
                                       | is doing NOW   | running mean   |
                                       +----------------+----------------+

⚠️ **All four camera panels draw the same classifier, and there is only one of
it.** `farm.overlay`, over `farm.scout.StageDetector` and `stage_of` — which is
the code `duo.DuoScout.look` calls to build the map, so a deck box is a *view*
of a decision the robot already makes rather than a second opinion computed for
the display. The deck panels additionally print the two measured ways that
classifier is wrong — per-band recall, green weakest at 74%, and touching fruit
being dropped rather than merged — because an overlay is the most persuasive
thing in the window and the easiest place to lie. Both numbers were re-measured
on *these* cameras rather than carried over from the shared scouting head, which
sits 100 mm further off the row and does worse; see `farm.overlay.RECALL`.

⚠️ **Both windows are OpenCV windows over offscreen EGL renders, and there is no
GLFW viewer anywhere in this process.** `mujoco.viewer.launch_passive` needs a
GLFW context and `mujoco.Renderer` under `MUJOCO_GL=egl` needs an EGL one; the
two do not coexist in one process on this machine, which `COMMANDS.md` already
records. Everything here goes through `week3_watch.Sink`, one per window — it
already owns the quit handling, the window lifecycle and the mp4 writer.

--- the run ------------------------------------------------------------------

    1. MAP      drive the aisle, both deck heads scanning their own rows,
                fusing one house map. Nothing is picked in this pass.
    2. PLAN     one route per arm, covered jointly into one trolley itinerary,
                then every fruit assigned to exactly one arm and logged.
                Visible in window 2 before either arm commits.
    3. TRAVEL   drive to each stop.
    4. PICK     **both arms work at the same time**, stepped inside one physics
                loop. They interlock only on the legs that swing them through
                the middle of the deck. See `farm.duo`.
    5. CRATE    each arm has its own crate, riding on the deck beside it.

--- three things this window is careful about --------------------------------

⚠️ **The pipeline text is read out of the running mission, not scripted.**
`farm.duo.ArmState` is written where the work happens and the current leg is
read live from `week2_pick.MissionRun.leg`, which the executor sets as it flies.
There is no list of captions and no timer driving them. The route verdict, the
clearance breach in millimetres, the guard's live margin, the abort distance and
the map-versus-truth position error are all fields on the mission objects — see
`pipeline_lines`, which asserts this and then only formats.

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

# The clearance budget a refusal is measured against, so the panel can print
# "breach 26 mm against 40 mm" rather than a bare number. Imported rather than
# restated: `farm.duo` plans at this and the two must not drift.
CLEARANCE_MM = 40.0

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
    # ⚠️ 78 characters, not 64. The pipeline panel is 620 px wide and this font
    # at 0.40 is about 7 px a character, so 64 was clipping ~20% of the width
    # off every line — "missed 0" arrived as "miss". A panel that silently
    # truncates the number you are reading it for is worse than a smaller font.
    for text, col in lines:
        if y > h - 8:
            break
        cv2.putText(img, text[:78], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    col, 1, cv2.LINE_AA)
        y += 17
    return img


# --- the map -----------------------------------------------------------------

class DuoMapPanel:
    """A top-down plan of the house: what was found, and what happened to it.

    ⚠️ Drawn with the **long axis horizontal**. The house is 8 m along the row
    and 5.8 m across, so putting y across the screen fills the panel; the other
    way round wastes half of it on empty aisle. `farm/watch.py` chose the same
    and for the same reason.
    """

    def __init__(self, w=MISS_W, h=MISS_H, pad=30, underlay=None, title=None):
        self.w, self.h, self.pad = w, h, pad
        self.underlay = underlay
        self.title = title
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

    def unpx(self, u, v):
        """Panel pixel -> world (x, y). The inverse of `px`, and its neighbour.

        ⚠️ **Written here rather than in whatever wants to click**, because the
        two have to agree exactly and a second copy of `pad + 26` somewhere else
        is a copy that gets left behind the next time this panel is re-laid out.
        `farm.mousereach` places fruit through this, so a drift between the two
        would be a tomato appearing somewhere other than where the cursor was.
        """
        y = self.y0 + (u - self.pad) / self.s
        x = self.x0 + (v - self.pad - 26) / self.s
        return float(x), float(y)

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
        for p in state.truth_points():
            u, v = self.px(p[0], p[1])
            cv2.circle(img, (u, v - 6), 2, (120, 120, 120), -1, cv2.LINE_AA)

        # --- anything the caller wants drawn in world coordinates -------------
        # ⚠️ A hook, not a feature: `farm.mousereach` draws the reachable bands
        # and the cursor here, and it has to be *under* the mapped dots so a
        # band never hides a fruit. Nothing else uses it and the default is
        # None, so an ordinary run composites exactly what it did before.
        if self.underlay is not None:
            self.underlay(img, self)

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
                # ⚠️ Under `--truth` a sighting carries its own `truth`; on a
                # real mapping pass it carries nothing but a position and a
                # stage, and the link to a truss name is made per stop by
                # `duo.associate`. `state.named` keeps that link so the map can
                # show what *happened* to a dot on a scouted run and not only on
                # a cheated one. Truth first, because it is exact where it
                # exists.
                name = (s.truth.name if s.truth is not None
                        else state.named.get(id(s)))
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
        _title(img, self.title or "THE MAP  —  what the robot believes it found")
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
    #
    # ⚠️ **A first-order lag alone does not keep the machine in frame, and that
    # is not a tuning problem, it is what a first-order lag does.** Chasing a
    # target moving at constant speed, it settles at a *steady-state error* of
    # `speed * tau` and stays there for the whole traverse — at the old 0.06
    # per frame and 10 fps that is a 1.7 s time constant against a 0.35 m/s
    # trolley, so the machine sat ~0.6 m off-centre for every drive and drifted
    # further at any higher speed. Smooth, and consistently badly framed.
    #
    # So the follow is smooth *and* bounded: exponential toward the trolley,
    # then clamped so it can never be more than `MAX_LAG_M` behind. The clamp
    # only engages during sustained travel, and while it is engaged the camera
    # moves at exactly the trolley's speed — so it is still not a snap, it has
    # simply stopped falling further behind.
    TAU_S = 0.35          # follow time constant, seconds
    MAX_LAG_M = 0.40      # never further off centre than this
    DIST = 3.2            # close enough that the deck and both arms fill it
    AZIMUTH = 90.0
    ELEVATION = -12.0
    HEIGHT = 0.85

    def __init__(self, model, data, aisle=0, fps=10):
        import mujoco

        self.cam = mujoco.MjvCamera()
        self.cam.distance = self.DIST
        self.cam.azimuth = self.AZIMUTH
        self.cam.elevation = self.ELEVATION
        self.ax = house.aisle_x(aisle)
        self.y = float(data.body(trolley.TROLLEY).xpos[1])
        # ⚠️ Converted from a time constant using the panel period, not left as
        # a per-frame fraction. A per-frame constant means the camera follows
        # twice as fast at 20 fps as at 10 — the framing would change with
        # `--fps`, which is a display setting and has no business altering what
        # the shot looks like.
        dt = 1.0 / max(float(fps), 1e-6)
        self.alpha = 1.0 - np.exp(-dt / self.TAU_S)
        self._apply()

    def _apply(self):
        self.cam.lookat[:] = [self.ax, self.y, self.HEIGHT]

    def follow(self, data):
        target = float(data.body(trolley.TROLLEY).xpos[1])
        self.y += (target - self.y) * self.alpha
        # The bound. `np.clip` on the error rather than on `self.y`, so the sign
        # is handled for a trolley driving in either direction.
        err = target - self.y
        if abs(err) > self.MAX_LAG_M:
            self.y = target - np.copysign(self.MAX_LAG_M, err)
        self._apply()
        return self.cam

    def lag(self, data):
        """How far off centre the machine currently is, in metres. For the log."""
        return abs(float(data.body(trolley.TROLLEY).xpos[1]) - self.y)


# --- the pipeline and stats panels -------------------------------------------

def pipeline_lines(state, data):
    """What the robot is doing right now, per arm. Read from `state`, not acted.

    ⚠️ **Every line here is a field `farm.duo` wrote where the work happened, or
    something derived from one.** There is no scripted sequence and no timer:
    the phase comes from the executor's current leg via `ArmState.follow_leg`,
    the route verdict from the `Mission` the planner returned, the breach from
    `Mission.breaches[0]`, the guard's margin from the live `Guard`, and the
    position error from the map's estimate against `mjData`'s truth. If a line
    is wrong, the robot is wrong.

    Anything the panel wants that the mission objects did not already expose was
    added to `ArmState` rather than reconstructed here — see that class.
    """
    out = [(f"PHASE   {state.phase.upper()}", ACCENT)]
    if state.detail:
        out.append((f"        {state.detail}", DIM))
    flying = getattr(state, "flying", set())
    if len(flying) > 1:
        out.append((f"        both arms flying — "
                    f"{len(flying)} missions in one physics loop", ACCENT))
    out.append(("", INK))

    for tag in state.arms:
        st = state.state[tag]
        col = ARM_COL[tag]
        live = tag in flying
        mark = ">" if live else " "
        out.append((f"{mark} {st.line()}", col if live else DIM))

        # --- the target, what the map thought it was, and how wrong that was
        if st.target:
            bits = [st.target]
            if st.stage:
                bits.append(st.stage)
            if st.est_pos is not None:
                p = st.est_pos
                bits.append(f"est [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]")
            err = st.est_err_mm
            if np.isfinite(err):
                bits.append(f"err {err:.0f}mm vs truth")
            out.append((f"    {'  '.join(bits)}", INK))

        # --- what the planner decided, and if it refused, by how much
        if st.route_ok is True:
            out.append((f"    route ACCEPTED  {st.route_label}"
                        f"  ({st.route_tried} tried)", ACCENT))
        elif st.route_ok is False:
            out.append((f"    route REFUSED   {st.route_tried} tried", WARN))
            if st.refuse_reason:
                out.append((f"      {st.refuse_reason[:56]}", WARN))
            if np.isfinite(st.breach_mm):
                out.append((f"      breach {st.breach_mm:.0f} mm against "
                            f"{CLEARANCE_MM:.0f} mm", WARN))

        # --- the guard: live margin, and where it stopped things if it did
        g = st.guard
        if g is not None and np.isfinite(getattr(g, "min_seen", np.inf)):
            gcol = WARN if g.min_seen * 1000 < 25 else DIM
            out.append((f"    guard ARMED     min clear "
                        f"{g.min_seen * 1000:.0f}mm  on `{g.leg or '-'}`",
                        gcol if g.armed else DIM))
        elif g is not None:
            out.append((f"    guard {'armed' if g.armed else 'disarmed'}", DIM))
        if st.abort_leg:
            out.append((f"    ABORTED on `{st.abort_leg}` at "
                        f"{st.abort_mm:.0f} mm", WARN))

        # --- what it is waiting on, if anything
        if st.waiting_on:
            out.append((f"    WAITING ON  {st.waiting_on[:50]}", AMBER))

        s = st.stats
        mean = f"{s.mean_s:.1f}s" if s.pick_s else "-"
        out.append((f"    row r{st.row}  deck pan {st.deck_pan(data):+.0f}deg"
                    f"  |  picks {s.crated}  mean {mean}  "
                    f"refused {s.refused}  missed {s.missed}", DIM))
        out.append(("", INK))

    centre = getattr(state, "centre", None)
    if centre is not None and centre.holder:
        out.append((f"  deck centre held by "
                    f"arm{1 if centre.holder == 'a' else 2} — the other arm "
                    f"waits before its crossing legs", AMBER))
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
    # ⚠️ The concurrency claim as a live number, booked by `duo.Machine` rather
    # than asserted by this panel. A control cycle counts once both arms have a
    # mission in flight on it.
    mach = getattr(state, "machine", None)
    if mach is not None and mach.cycles:
        pct = 100 * mach.concurrent_cycles / mach.cycles
        out.append((f"      both arms flying on {mach.concurrent_cycles} of "
                    f"{mach.cycles} cycles ({pct:.0f}%)", ACCENT))
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

    ⚠️ **The cadences are decoupled on purpose**, and must not be re-coupled by
    speeding the physics up or cutting a measured constant:

        physics       every control cycle, `reach.CTRL_DT`, untouched
        panels        `--fps` (default 10) — the renders
        HSV overlay   `--hsv-hz` (default 3) — the detector, whose boxes are
                      cached and re-drawn on the intervening frames

    ⚠️ **This docstring used to claim the cadences were "the performance story",
    and a profile says they are not.** Measured on one two-armed pick, headless,
    with no panels rendered at all — 123.5 s of work with the renderer switched
    off entirely:

        plant_row.weld_force        48.3 s   39%   one efc scan per fruit per
                                                   physics substep
        daqp IK solve               44.1 s   36%   a QP over 317 DOF to move
                                                   a 6-DOF arm
        mink task objective          7.4 s    6%
        mj_step                      5.7 s    5%
        guard clearance              2.5 s    2%
        rendering                    0.0 s    0%   there was none

    The panels are ~2 ms of an ~39 ms control cycle during the harvest — about
    5%. Rendering *is* the cost during the mapping pass, which renders two
    1280x960 RGB+depth sensor pairs per scan pose, and that is why the rates are
    booked per phase.

    So the wins were taken where the time actually was (`plant_row.weld_forces`,
    one scan for the whole row instead of one per fruit), and the render-side
    work here is the honest remainder: one `Renderer` per camera reused for the
    whole run, and a live panel resolution that is a display choice rather than
    the detector's.

    ⚠️ **The detector keeps its own resolution.** Dropping the live panels to
    save compositing time must not quietly drop what the HSV classifier sees —
    that would change the measurement, not the display. `--panel-scale` moves
    the view panels only; `farm.scout` renders at `camera.RENDER_W/H` and is
    untouched by it.
    """

    def __init__(self, model, data, state, fps=10, hsv_hz=3.0, out=None,
                 arms=("a", "b"), panel_scale=1.0, on_click=None, on_key=None,
                 on_move=None, record=None, live=None, map_panel=None):
        """`out` records *instead of* showing; `record` records *as well as*.

        ⚠️ **Two parameters and not one, because they are different requests.**
        `--out` is the batch recorder and has always implied headless — it is
        how the repo's videos are made. `--record` is "I am about to demo this
        live and I will forget to screen-capture it", which means the windows
        have to be up and clickable while the mp4 is being written. `Sink`
        already does both independently; it was this constructor that tied
        `live` to `out is None`.

        `on_click(x, y)` and `on_key(code)` are routed to the **MISSION**
        window only. The sensors window has no click target — the panels there
        are camera views, and placing a fruit by clicking a camera view means
        ray-casting through a lens that is panning while you aim at it.
        """
        import mujoco

        from week3_watch import Sink

        self.model, self.data, self.state = model, data, state
        self.arms = tuple(arms)
        # Live view resolution. The tile geometry the panels are composited at
        # never changes — the *render* is done smaller and scaled up, so the
        # window layout, the captions and the mp4 dimensions are all unaffected
        # and only the pixels the GPU has to produce move.
        self.scale = float(np.clip(panel_scale, 0.25, 1.0))
        self.rw = max(64, int(SENS_W * self.scale))
        self.rh = max(64, int(SENS_H * self.scale))
        self.aw = max(64, int(MISS_W * self.scale))
        self.ah = max(64, int(MISS_H * self.scale))
        self.map = DuoMapPanel() if map_panel is None else map_panel
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

        # ⚠️ **One `Renderer` per camera, built once and kept for the whole
        # run.** A `mujoco.Renderer` allocates an offscreen framebuffer and a
        # scene of `max_geom` — 30000 here, because the house is 5433 geoms
        # before the crop — so constructing one per frame would pay that
        # allocation at the panel rate and thrash the GPU. They are closed in
        # `close()`, which is why `shot()` has a `finally`.
        self.r = {}
        for t in self.arms:
            for cam in (self.wrist[t], self.deck[t]):
                if has(cam):
                    self.r[cam] = mujoco.Renderer(model, height=self.rh,
                                                  width=self.rw,
                                                  max_geom=30000)
        self.r["aisle"] = mujoco.Renderer(model, height=self.ah, width=self.aw,
                                          max_geom=30000)
        self.aisle_cam = AisleCam(model, data, aisle=state.aisle, fps=fps)
        self.worst_lag = 0.0

        # ⚠️ **Window titles are ASCII only, and this is a hard constraint of the
        # backend rather than a style choice.** OpenCV 5's Qt highgui keys its
        # window registry by the title string, and a non-ASCII one does not
        # round-trip: `namedWindow` appears to succeed, the window is never
        # registered, and the very next `setMouseCallback` dies with
        #
        #     error: (-27:Null pointer) NULL window handler
        #
        # These two titles used an em dash, to match this repo's prose. Every
        # other `Sink` title in the repo happens to use a plain hyphen, so
        # nothing had ever exercised the failure. Verified directly: the same
        # call with `-` succeeds and with `—` raises.
        #
        # Panel *captions* are unaffected — `cv2.putText` renders an em dash
        # fine — so the dashes inside the frames stay.
        # ⚠️ Three states, not two: show, show-and-record, and neither. The
        # last one is not "do less work" — the panels are still composed, the
        # HSV overlay still runs and the map is still drawn, so a headless
        # regression run exercises the same code the live window does. It only
        # skips `cv2.imshow`.
        stem = out if out is not None else record
        show = (out is None) if live is None else bool(live)
        self.sensors = Sink(live=show, out=None if stem is None
                            else f"{stem}_sensors.mp4", fps=fps,
                            title="vinea - SENSORS (2 wrist + 2 deck)")
        self.mission = Sink(live=show, out=None if stem is None
                            else f"{stem}_mission.mp4", fps=fps,
                            title="vinea - MISSION (map, aisle, pipeline)",
                            on_click=on_click, on_key=on_key, on_move=on_move)

    def close(self):
        for r in self.r.values():
            try:
                r.close()
            except Exception:
                pass
        for s in (self.sensors, self.mission):
            s.close()

    def _render(self, cam, w, h):
        """Render `cam` and return it at the tile size `w`x`h`.

        ⚠️ The *render* happens at `--panel-scale` of the tile and is resized up
        to it. That keeps the panel layout, the captions and the mp4 dimensions
        identical whatever the scale is, so the flag is a cost knob and not a
        format change — a viewer at scale 0.5 and one at 1.0 produce the same
        shaped video.
        """
        import cv2

        if cam not in self.r:
            return None
        self.r[cam].update_scene(self.data, camera=cam)
        img = cv2.cvtColor(self.r[cam].render(), cv2.COLOR_RGB2BGR)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

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

    def _deck_panel(self, tag, fresh=True):
        """One arm's deck camera, with the ripeness call drawn on it.

        ⚠️ **The same classifier the wrist panels use and the same one the
        mapping pass scores itself with** — `farm.overlay`, over
        `farm.scout.StageDetector` and `farm.scout.stage_of`, banded against
        `farm.crop.STAGES`. There is deliberately no second classifier here: the
        deck path already calls exactly this code in `duo.DuoScout.look`, which
        is where every ripeness on the map comes from, so drawing it is a *view*
        of a decision the robot already makes.

        ⚠️ **These boxes are not the map's boxes, and the panel says so.** The
        mapping pass detects on a 1280x960 `SensorCamera` render with depth;
        this is a 480x360 view render with none. Same thresholds, same code,
        smaller frame — so a fruit can be boxed here and missed there, or the
        reverse. `banner`'s recall line is the number that says how often. The
        wrist panels have had this property since they were written; naming it
        is new.

        ⚠️ **Display-only.** Nothing in this method is read by the router, the
        planner, the guard or the executor, and nothing it computes is stored
        anywhere they can see. `self._boxes` is a viewer cache keyed by camera
        name and read only by `draw`.
        """
        import cv2

        from farm.scout import StageDetector

        cam = self.deck[tag]
        img = self._render(cam, SENS_W, SENS_H)
        st = self.state.state[tag]
        if img is None:
            return _missing(SENS_W, SENS_H, f"{st.name} deck: not fitted",
                            "build with arm_decks=True")
        pan, tilt = (st.head.current(self.data) if st.head is not None
                     else (0.0, 0.0))

        # Same cadence and the same held-box handling as `_wrist_panel`: the
        # detector runs at `--hsv-hz`, the calls are cached, and the boxes are
        # redrawn thinner on the frames in between so a held box is not a claim
        # about the frame under it. See that method for why caching the counts
        # alone was wrong.
        stale = not (fresh or cam not in self._boxes)
        if not stale:
            if self.detector is None:
                self.detector = StageDetector()
            # ⚠️ The deck panel is the one place in this window that knows its
            # own range: the head sits over its arm's plate at
            # `house.WORK_STANDOFF` from the row it works and never leaves it.
            # So "how big is one tomato in this frame" is arithmetic here and a
            # guess anywhere else, and the size half of `looks_fused` is only
            # switched on where it is arithmetic.
            self._boxes[cam] = overlay.find(
                img, detector=self.detector,
                fruit_px=overlay.fruit_diameter_px(SENS_H, decks.SCOUT_FOVY,
                                          house.WORK_STANDOFF))
        calls = self._boxes.get(cam, [])
        overlay.draw(img, calls, stale=stale)

        _title(img, f"{st.name.upper()} DECK  -  row r{st.row}  -  "
                    f"pan {pan:+.0f}deg  -  HSV "
                    f"{'held' if stale else 'live'}", ARM_COL[tag])
        # Counts by band, the measured recall, and the cluster limitation — on
        # their own strip rather than floated over the render, because text at
        # `SENS_H - 10` sits on the tile seam and the descenders are clipped by
        # the `vstack` below it. What the arm is doing keeps its line above the
        # strip: it was here before the overlay was and the PIPELINE panel is a
        # whole window away.
        overlay.banner(img, calls)
        cv2.rectangle(img, (0, SENS_H - 60), (img.shape[1], SENS_H - 42),
                      (24, 24, 24), -1)
        cv2.putText(img, f"{st.phase}  {st.detail}"[:52], (6, SENS_H - 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, DIM, 1, cv2.LINE_AA)
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
            tiles_bot.append(self._deck_panel(tag, fresh))
        return np.vstack([np.hstack(tiles_top), np.hstack(tiles_bot)])

    def compose_mission(self):
        st = self.state
        import cv2

        m = self.map.draw(st, self.data)
        self.r["aisle"].update_scene(self.data,
                                     camera=self.aisle_cam.follow(self.data))
        aisle = cv2.cvtColor(self.r["aisle"].render(), cv2.COLOR_RGB2BGR)
        if aisle.shape[1] != MISS_W or aisle.shape[0] != MISS_H:
            aisle = cv2.resize(aisle, (MISS_W, MISS_H),
                               interpolation=cv2.INTER_LINEAR)
        ty = float(self.data.body(trolley.TROLLEY).xpos[1])
        lag = self.aisle_cam.lag(self.data)
        self.worst_lag = max(self.worst_lag, lag)
        _title(aisle, f"DOWN THE AISLE  —  trolley at y={ty:+.2f} m  —  "
                      f"cam {lag * 100:.0f} cm off centre")
        pipe = _text_panel(MISS_W, MISS_H, pipeline_lines(st, self.data),
                           "PIPELINE  —  what each arm is doing now")
        stats = _text_panel(MISS_W, MISS_H, stats_lines(st),
                            "PER-ARM STATS  —  from execute()'s seconds")
        return np.vstack([np.hstack([m, aisle]), np.hstack([pipe, stats])])

    def push_frame(self, fresh=None):
        """Compose and push both windows once. `fresh=None` follows `--hsv-hz`.

        Split out of `tick` so a caller with its own loop — the placement phase
        of `farm.mousereach`, which is not being driven by a run at all — can
        put frames up at the same cadence and with the same detector rate
        instead of re-detecting on every frame it happens to draw.
        """
        self._panel_n += 1
        if fresh is None:
            fresh = (self._panel_n % self.hsv_every) == 0
        self.sensors.push(self.compose_sensors(fresh))
        self.mission.push(self.compose_mission())

    def tick(self, _t=None):
        """The `on_tick` handed to the runner. Cheap on the frames it skips."""
        self.frames += 1
        if self.frames % self.every:
            return
        self.push_frame()
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
            self.push_frame(fresh=True)


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

    # ⚠️ **A subcommand, checked before argparse sees anything.** `mousereach`
    # is the same machine, the same windows and the same `duo.run` with a person
    # holding the mouse instead of `crop.spawn` — so it belongs on this command
    # rather than in a fork of it, and it takes its own flags. Dispatching here
    # rather than through `add_subparsers` keeps every existing invocation
    # byte-identical: `two_arm_farm.py --seed 7` never reaches this branch.
    if len(sys.argv) > 1 and sys.argv[1] == "mousereach":
        from farm import mousereach

        return mousereach.main(sys.argv[2:])

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
    ap.add_argument("--panel-scale", type=float, default=1.0,
                    help="render the live view panels at this fraction of "
                         "their tile size and scale up (0.25-1.0). The panel "
                         "layout and the mp4 dimensions do not change. The "
                         "detector keeps its own resolution — this is a "
                         "display cost knob, not a perception one.")
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
        arms=arms, panel_scale=args.panel_scale)
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
            print(f"\n  ⚠️ Panels are ~5% of a harvest control cycle. The cost "
                  f"is physics-side —\n     see the profile in `Windows`. "
                  f"--panel-scale {args.panel_scale:g} "
                  f"({windows.rw}x{windows.rh} tiles).")
            print(f"\n  aisle cam never further than "
                  f"{windows.worst_lag * 100:.0f} cm off the trolley "
                  f"(bound {AisleCam.MAX_LAG_M * 100:.0f} cm)")
            windows.close()
        else:
            print(f"\n  {wall:.0f} s wall, no windows")
    duo.report(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
