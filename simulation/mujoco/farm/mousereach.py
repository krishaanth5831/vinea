#!/usr/bin/env python3
"""Hand somebody the mouse. They hang tomatoes; the machine drives and picks them.

    python 2armfarm.py mousereach
    ./.venv/bin/python simulation/mujoco/two_arm_farm.py mousereach

    click a band      hang a tomato there      SPACE   go and pick them
    T / 1 2 3 4       red / turning / breaker / green  (only red gets picked)
    Z                 hang height, within the measured band
    A  auto-fill      C  clear      R  reset      Q  quit

This is `week4_place.py` on a machine whose base moves, and that is the whole
difference. Week 4's green board is the envelope of an arm bolted to a floor:
one rectangle, one row, one standing position. On the trolley the reachable
region is that same envelope **swept along the rail**, once per arm, on opposite
sides of the aisle — so it is not a board, it is a band down each row, and which
arm gets a fruit is decided by which side of the aisle it is hanging on.

--- what is shared with week4_place, and what could not be ------------------

⚠️ **The placement state machine is `week4_place.Crop` and there is one of it.**
The pool of compiled trusses, the placed set, the version counter, the welds
left live on parked fruit so the running `Guard` can see a late arrival, the
alpha reveal — all of that is scene-independent and all of it is imported. What
is *not* shared is the rule about where fruit may go, because on a bolted arm
that is a rectangle and here it is a swept band; `week4_place.BoltedBoard` and
`RailBand` below are the two answers, injected into the same `Crop`.

A duplicated state machine is already bug 7 in this repo's log, and this is
exactly the place a second one would have been written.

--- where the band comes from ------------------------------------------------

⚠️ **Derived, not re-measured, and it is derivable for a specific reason.** The
arm mounts sit `house.ARM_OFFSET` = 200 mm off the trolley centreline, and that
number was chosen so that against a 1.60 m row pitch the standoff comes back to
`house.WORK_STANDOFF` = 600 mm — the figure every Week 1-4 clearance, cycle time
and envelope cell was measured at. So the envelope carries over verbatim:

    z, in the arm's own frame   week4_place.MARGINAL_Z, 0.42 .. 0.72 m
                                49 cells swept, one full pick each, 18/21 clean
    z, in the world             lifted by trolley.DECK_Z — see `crop.fruit_z`
    y                           the envelope swept along the rail, which is the
                                trolley's travel widened by how far the arm
                                reaches either side of wherever it stops
    x                           the row that arm works from this aisle

`farm.crop.Z_LOCAL` already states the first of those and `RailBand` asserts the
two agree at import, so a drift between them fails loudly instead of quietly
advertising unpickable cells — which is bug log entry 42, exactly.

⚠️ **The y half of the band is asymmetric per arm and that is not cosmetic.**
`duo.CONCURRENT_TOWARD_M` is the measured limit on how far an arm may reach
*toward* the other one while that one is also working — 300 mm, past which the
elbows close on each other. `duo.merge` routes against that asymmetry, so the
band has to be drawn against it too or the demo would accept a click the router
then refuses. See `RailBand.__init__` for the arithmetic.

⚠️ **Nothing here is cached and nothing needs to be.** The band is four
constants and two additions per arm; it is computed in microseconds at startup.
A cache keyed on `ARM_STAGGER` and the deck width would be a file to invalidate
in exchange for saving nothing measurable.

--- what the demo does not do -----------------------------------------------

⚠️ **Placement does not hand the router anything.** A person puts a tomato in
the world; the deck cameras then have to find it, band its colour and estimate
where it is, exactly as they do on a spawned crop. `duo.run` is called with
`use_truth=False` and this module never touches `route.plan`, `merge` or any
sighting. Every run ends by printing, per fruit, how far the estimate the arm
was sent to was from where the tomato actually hung — see `audit`. A column of
zeros there would mean the demo had become a puppet show.

Nothing below changes a contact parameter, `SNAP_N`, the router, the interlock,
the pick order or the parked cradle gripper. If the demo exposes a limitation,
the demo shows it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import two_arm_farm as taf  # noqa: E402
import week4_place as w4p  # noqa: E402
from farm import armframe, duo, house, route, trolley  # noqa: E402
from farm import crop as fcrop  # noqa: E402

# How many trusses are compiled into the scene. Placement is `Row.place` on a
# member of this pool, because a body cannot be added to a compiled model and
# recompiling mid-run invalidates every id `Row`, `RobotSpheres`,
# `CropObstacles` and the reacher have cached. `week4_place`'s docstring is the
# long version.
#
# Larger than `week4_place.POOL` because the fruit are spread over two rows and
# eight metres rather than one board, and a pool that runs out mid-demo is a
# refusal with no honest reason behind it.
POOL = 40

# Where an unplaced truss waits: off the end of the house, clear of every row
# and 2 m from the nearest thing the arm can reach.
#
# ⚠️ Parked with the weld **live** and hidden by alpha only — see
# `week4_place.Crop.restore`. Detaching them would be tidier and would silently
# disarm the safety net for anything placed mid-run.
PARK_X = house.ROW_X0 - 2.0
PARK_PITCH = 0.14

# The heights a click may hang a fruit at, in the arm's own frame. Three, not a
# slider: the band is 300 mm tall and the point of the key is that somebody can
# say "put that one low" without reading a number off a screen. Every value is
# inside `week4_place.MARGINAL_Z` by construction — see `HANG_Z`.
HANG_NAMES = ("low", "mid", "high")

# What a fruit may be hung as. `crop.STAGES`'s own order, and its own `pick`
# flag decides which of them the harvest takes: only red.
#
# ⚠️ Default red, so the demo works if nobody touches the key. A grower hanging
# a green one and watching the robot correctly ignore it is the product
# explaining itself; a grower who has not found the key yet still gets a
# harvest.
STAGES = ("red", "turning", "breaker", "green")

# Tile geometry of the MISSION window, so a window pixel can be turned into a
# map-panel pixel. Imported rather than restated.
TILE = (taf.MISS_W, taf.MISS_H)

INK = (232, 232, 232)
DIM = (128, 128, 128)
GOOD = (140, 250, 150)
BAD = (110, 110, 250)
AMBER = (90, 200, 245)


class Restart(Exception):
    """R was pressed. Unwinds whatever is running, back to placement."""


def hang_z(i):
    """World z of hang height `i`, inside the measured band by construction."""
    lo, hi = fcrop.Z_LOCAL
    frac = (0.15, 0.5, 0.85)[i % 3]
    return fcrop.fruit_z(lo + (hi - lo) * frac)


# --- where fruit may hang on a machine whose base moves -----------------------

class RailBand:
    """The Week 1-4 envelope, swept along the rail, once per arm.

    The rule half of `week4_place.Crop` for the farm scene. See the module
    docstring for why it is a band rather than a board, and where every number
    in it comes from.
    """

    def __init__(self, aisle=0, arms=("a", "b")):
        # ⚠️ The envelope is `week4_place.MARGINAL_*` and it is imported rather
        # than restated. `farm.crop.Z_LOCAL` says in a comment that it *is*
        # those numbers; this asserts it, so the day somebody edits one of them
        # the demo fails at import instead of advertising cells the arm was
        # measured to fail in. Bug log entry 42 is that failure, silently.
        if tuple(fcrop.Z_LOCAL) != tuple(w4p.MARGINAL_Z):
            raise RuntimeError(
                f"farm.crop.Z_LOCAL {tuple(fcrop.Z_LOCAL)} has drifted from "
                f"week4_place.MARGINAL_Z {tuple(w4p.MARGINAL_Z)} — the band "
                f"this demo draws is the second of those and the crop is hung "
                f"against the first. Re-measure or re-import; do not guess.")

        self.aisle = aisle
        self.arms = tuple(arms)
        self.row = {t: armframe._worked_row(aisle, t) for t in self.arms}
        self.row_x = {t: house.row_x(self.row[t]) for t in self.arms}
        self.limit = w4p.MAX_FRUIT

        # z: the measured band, lifted onto the deck. `crop.fruit_z` is the one
        # place that lift is written down.
        #
        # ⚠️ **Two bands, not one, and the outer one is not a promise.** Week 4
        # measured the envelope cell by cell with one full pick each and found
        # two regions, and flattening them is what bug log entry 42 records:
        # `GUARANTEED_Z` picked 10/10, the wider `MARGINAL_Z` 18/21. A click in
        # the outer band is accepted and is *expected* to fail about one time in
        # seven, so it is accepted and labelled rather than accepted and
        # presented as safe.
        self.z = (fcrop.fruit_z(fcrop.Z_LOCAL[0]),
                  fcrop.fruit_z(fcrop.Z_LOCAL[1]))
        self.z_sure = (fcrop.fruit_z(w4p.GUARANTEED_Z[0]),
                       fcrop.fruit_z(w4p.GUARANTEED_Z[1]))

        # y: the rail's travel, widened by the arm's reach at each end, then
        # clipped to the bay. Two windows have to hold and the tighter wins:
        #
        #   route.plan   covers fruit with stops clipped to `y_limits`, and
        #                accepts a fruit within `route.REACH_Y` of one — so a
        #                fruit is routable for need in [lo - REACH, hi + REACH].
        #   duo.merge    covers both arms jointly against the *asymmetric*
        #                window, because an arm reaching toward the other one
        #                may only come `CONCURRENT_TOWARD_M` that way. For arm a
        #                the other arm lies at -y, so its trolley interval is
        #                [need - REACH_Y, need + CONCURRENT_TOWARD_M]; arm b is
        #                the mirror. Non-empty against [lo, hi] gives
        #                need in [lo - toward, hi + reach] for a, mirrored for b.
        #
        # `need` is the trolley y that puts *this arm* at the fruit, which is
        # the fruit's y minus `ARM_Y[tag]` — the arms are staggered 500 mm along
        # the row and a band drawn at the deck's own y would be half a metre out
        # for both of them, in opposite directions.
        lo, hi = trolley.y_limits()
        near, far = duo.CONCURRENT_TOWARD_M, route.REACH_Y
        self.y = {}
        self.y_binding = {}
        for t in self.arms:
            dy = trolley.ARM_Y[t]
            lo_dy, hi_dy = (-near, +far) if t == "a" else (-far, +near)
            merge_lo, merge_hi = lo + lo_dy + dy, hi + hi_dy + dy
            route_lo, route_hi = lo - far + dy, hi + far + dy
            bay_lo, bay_hi = -house.HOUSE_HALF_Y, house.HOUSE_HALF_Y
            y_lo = max(merge_lo, route_lo, bay_lo)
            y_hi = min(merge_hi, route_hi, bay_hi)
            self.y[t] = (y_lo, y_hi)
            # Which of the three actually binds, so the report can say whether
            # the demo is limited by the arm or by the length of the bay.
            self.y_binding[t] = (
                "bay" if y_lo == bay_lo else
                "interlock" if y_lo == merge_lo else "reach",
                "bay" if y_hi == bay_hi else
                "interlock" if y_hi == merge_hi else "reach")

        # How far off a row's centreline a click may land and still be taken as
        # meaning that row. Half the row pitch, so the two worked rows between
        # them own the whole aisle and there is no dead strip in the middle for
        # somebody to click into and get nothing.
        self.snap_x = house.ROW_PITCH / 2

    # --- the Crop rule interface --------------------------------------------

    def world(self, y, z):
        """(y, z) with no row named. Not meaningful here — see `check`.

        ⚠️ Raises rather than guessing a row. `Crop.place(y, z)` is the bolted
        board's coordinate system and there is no answer to it on a scene with
        two rows; the farm path goes through `Crop.place_world` with a position
        this class has already located. A silent default to row 0 would put
        every keyboard-driven placement on one arm.
        """
        raise RuntimeError(
            "RailBand has no (y, z) plane — a placement here names a row. "
            "Use Crop.place_world with a position from RailBand.locate.")

    def arm_of(self, x):
        """Which arm's row a world x means, or None. Nearest worked row wins."""
        best, best_d = None, float("inf")
        for t in self.arms:
            d = abs(float(x) - self.row_x[t])
            if d < best_d:
                best, best_d = t, d
        return best if best_d <= self.snap_x else None

    def locate(self, x, y):
        """(tag, snapped world position) for a raw click, or (None, reason).

        ⚠️ **This is the refusal that has to happen at click time.** The worst
        thing this demo can do in front of a grower is accept a placement and
        discover eight metres later that nothing can reach it, so every reason a
        fruit could not be picked *for geometric reasons* is checked here,
        before a tomato appears. What is deliberately not checked here is
        whether the planner will find a route — that is a kinematic replay
        costing ~150 ms per candidate, it depends on where the other arm is at
        the time, and pretending to know it now would be the same lie in the
        other direction.
        """
        tag = self.arm_of(x)
        if tag is None:
            near = min(range(house.N_ROWS),
                       key=lambda i: abs(float(x) - house.row_x(i)))
            worked = ", ".join(f"r{self.row[t]}"
                               for t in sorted(self.arms,
                                               key=lambda t: self.row[t]))
            # ⚠️ Two different refusals wear the same shape and saying the
            # wrong one is worse than saying nothing. A click just past a
            # *worked* row is a click that missed the band by a few
            # centimetres; a click over r2 is a click on crop this trolley
            # cannot reach from this aisle at all. The first is "try again
            # slightly to the left", the second is "drive the other aisle", and
            # the first version of this said the second in both cases.
            if near in self.row.values():
                d = abs(float(x) - house.row_x(near))
                return None, (f"out of reach - {d * 1000:.0f} mm off r{near}, "
                              f"and the band is the "
                              f"{self.snap_x * 1000:.0f} mm either side of a "
                              f"row; click nearer the aisle")
            return None, (f"out of reach - r{near} is not worked from aisle "
                          f"a{self.aisle}; the trolley reaches {worked}")
        y_lo, y_hi = self.y[tag]
        if y < y_lo or y > y_hi:
            side = "behind" if y < y_lo else "past"
            what = self.y_binding[tag][0 if y < y_lo else 1]
            why = {"bay": "the crop stops there",
                   "reach": "the trolley's travel ends there",
                   "interlock": "the arms may not reach that far toward each "
                                "other while both work"}[what]
            return None, (f"out of reach - {side} the end of the band "
                          f"(y {y_lo:+.2f} to {y_hi:+.2f}); {why}")
        return tag, np.array([self.row_x[tag], float(y), 0.0])

    def zone(self, z):
        """'guaranteed' or 'marginal', the same two words `week4_place` uses."""
        return ("guaranteed" if self.z_sure[0] - 1e-9 <= z <= self.z_sure[1] + 1e-9
                else "marginal")

    def check(self, pos, placed):
        """(ok, why, zone). The authority — `locate` is the friendly front end.

        Re-tests the geometry rather than trusting the caller, because this is
        the function `Crop` asks and anything that can call `place_world` can
        reach it. `zone` is `week4_place`'s own word for how well the arm was
        measured to work there, kept identical so the two scenes' logs read the
        same way.
        """
        x, y, z = (float(v) for v in pos)
        tag, snapped = self.locate(x, y)
        if tag is None:
            return False, snapped, None
        if abs(x - self.row_x[tag]) > 1e-6:
            return False, (f"off the row - a fruit hangs on r{self.row[tag]} "
                           f"at x={self.row_x[tag]:.2f}, not x={x:.2f}"), None
        if not (self.z[0] - 1e-9 <= z <= self.z[1] + 1e-9):
            where = "above" if z > self.z[1] else "below"
            why = ("the stem anchor would be at or through the support bar"
                   if z > self.z[1] else "the fruit would be under the gutter")
            return False, (f"out of reach - {where} the band "
                           f"(z {self.z[0]:.2f} to {self.z[1]:.2f}); "
                           f"{why}"), None
        if len(placed) >= self.limit:
            return False, f"at the {self.limit}-fruit limit", None
        name, d = w4p.nearest_world(pos, placed)
        if name is not None and d < w4p.TOUCHING:
            return False, (f"{d * 1000:.0f} mm from {name} - two tomatoes "
                           f"cannot occupy the same space "
                           f"({w4p.TOUCHING * 1000:.0f} mm is touching)"), None
        # ⚠️ Accepted, with a warning, exactly as `week4_place.check` does. A
        # close pair is a pick-order problem for the deck camera to solve and
        # not a placement to refuse — and at this range it is the *detector*
        # that struggles first, which is the more interesting failure and the
        # one this demo should let somebody create on purpose.
        if name is not None and d < w4p.CLOSE_M:
            return True, (f"{d * 1000:.0f} mm from {name} - the deck cam may "
                          f"fuse these into one blob"), self.zone(z)
        return True, "", self.zone(z)

    # --- what it looks like --------------------------------------------------

    def describe(self):
        out = []
        for t in self.arms:
            y_lo, y_hi = self.y[t]
            out.append(
                f"arm{1 if t == 'a' else 2}  row r{self.row[t]} at "
                f"x={self.row_x[t]:.2f}  y {y_lo:+.2f}..{y_hi:+.2f} "
                f"({'/'.join(self.y_binding[t])})  z {self.z[0]:.2f}..{self.z[1]:.2f}")
        return out


# --- the pool -----------------------------------------------------------------

def park_spot(i):
    """Where pool member `i` waits. Off the end of the house, out of reach."""
    return np.array([PARK_X, -1.0 + PARK_PITCH * (i % 12),
                     0.5 + 0.30 * (i // 12)])


def pool_trusses(n=POOL):
    """A compile-time pool of trusses, parked, one per `Crop` slot.

    ⚠️ Rows are dealt round-robin over the whole house so that
    `crop.add_trusses` draws a support bar on every row — the bars are what the
    fruit visibly hang from and a house with bars over two rows and nothing over
    the others reads as half-built. The row a pool member is *dealt* has no
    other consequence: it is rewritten the moment the truss is placed.
    """
    return [fcrop.Truss(name=f"m{i:02d}", row=i % house.N_ROWS,
                        x=float(park_spot(i)[0]), y=float(park_spot(i)[1]),
                        z=float(park_spot(i)[2]), stage="red")
            for i in range(n)]


# --- the person with the mouse ------------------------------------------------

class Placer:
    """Clicks, keys, and the one line of text that says what just happened.

    ⚠️ Every refusal this object produces is shown *and* spoken: the note goes
    on the map panel where the click was, and the same string is printed. A
    demo whose refusals are silent teaches the person holding the mouse that the
    machine is unreliable rather than that the placement was out of reach.
    """

    def __init__(self, crop, band, trusses, seed=0):
        self.crop = crop
        self.band = band
        self.by_name = {t.name: t for t in trusses}
        self.seed = seed
        self.stage_i = 0
        self.height_i = 1
        self.hover = None            # last pointer position, window pixels
        self.hover_world = None      # ...and what it means, or None
        self.hover_why = ""
        self.note = "click a band to hang a tomato; SPACE to harvest"
        self.note_colour = INK
        self.go = False
        self.reset = False
        self.stages = {}             # name -> the stage it was hung as
        self.map = None              # the live DuoMapPanel, set by the viewer
        self.running = False

    # --- what the cursor is carrying ----------------------------------------

    @property
    def stage(self):
        return STAGES[self.stage_i % len(STAGES)]

    @property
    def z(self):
        return hang_z(self.height_i)

    def cursor_text(self):
        pick = "PICKED" if fcrop.STAGE_BY_NAME[self.stage][2] else "left"
        return (f"{self.stage} ({pick})  "
                f"{HANG_NAMES[self.height_i % 3]} z={self.z:.2f} "
                f"[{self.band.zone(self.z)}]")

    # --- pointer -------------------------------------------------------------

    def _panel_pixel(self, x, y):
        """Window pixel -> map-panel pixel, or None if the pointer is elsewhere.

        The MISSION window is a 2x2 of `MISS_W` x `MISS_H` tiles and the map is
        the top-left one. Placement is in the map and nowhere else — see
        `_map_panel_is_the_click_target` in the module docstring.
        """
        if 0 <= x < TILE[0] and 0 <= y < TILE[1]:
            return x, y
        return None

    def on_move(self, x, y):
        self.hover = (x, y)
        pix = self._panel_pixel(x, y)
        if pix is None or self.map is None:
            self.hover_world, self.hover_why = None, ""
            return
        wx, wy = self.map.unpx(*pix)
        tag, got = self.band.locate(wx, wy)
        if tag is None:
            self.hover_world, self.hover_why = None, got
            return
        got[2] = self.z
        ok, why, _zone = self.band.check(got, self.crop.placed)
        self.hover_world = got
        self.hover_why = "" if ok else why

    def on_click(self, x, y):
        pix = self._panel_pixel(x, y)
        if pix is None:
            self._say("clicks place fruit in THE MAP, the top-left panel", DIM)
            return
        if self.map is None:
            return
        wx, wy = self.map.unpx(*pix)
        tag, got = self.band.locate(wx, wy)
        if tag is None:
            # ⚠️ Refused at click time, with the reason, and no tomato appears.
            self._say(got, BAD)
            return
        got[2] = self.z
        self.place(got)

    # --- placing -------------------------------------------------------------

    def place(self, pos, quiet=False):
        """Hang one fruit. Returns the truss name, or None with a reason said.

        ⚠️ **`check` is called here as well as inside `Crop`, and that is not a
        duplicate test — it is how the *warning* is recovered.**
        `Crop.place_world` returns `(name, zone)` on success and `(None, why)`
        on refusal, so the accepted-but-tight message — "40 mm from m03, the
        deck cam may fuse these" — has nowhere to come out. `week4_watch.Placer`
        solves it the same way and for the same reason. Reading the second value
        as a warning regardless is what this did first, and it printed the zone
        where the warning goes, so every placement carried a meaningless "(a)".
        """
        ok, why, _zn = self.band.check(pos, self.crop.placed)
        if not ok:
            self._say(why, BAD)
            return None
        name, refusal = self.crop.place_world(pos, quiet=True)
        if name is None:
            # Only the pool can still refuse here; the geometry already passed.
            self._say(refusal, BAD)
            return None
        stage = self.stage
        self.stages[name] = stage
        self.crop.recolour(name, fcrop.STAGE_BY_NAME[stage][0])
        # ⚠️ The `Truss` is rewritten too, not only the simulator. `duo.report`
        # and the stats panel count ripe fruit off these objects, and `Row`'s
        # `home` map is built from them — a truss left describing the parking
        # bay it was compiled in would make the demo report a house of 40 fruit
        # with the wrong colours in it.
        t = self.by_name[name]
        t.x, t.y, t.z = (float(v) for v in pos)
        t.stage = stage
        t.row = self.band.row[self.band.arm_of(pos[0])]
        if not quiet:
            arm = 1 if self.band.arm_of(pos[0]) == "a" else 2
            extra = f"  ({why})" if why else ""
            self._say(f"hung {name} {stage} on r{t.row} at y={pos[1]:+.2f} "
                      f"z={pos[2]:.2f} [{self.crop.zones.get(name, '?')}] - "
                      f"arm{arm}{extra}", AMBER if why else GOOD)
        return name

    def auto_fill(self):
        """Fill the bands with a spread of fruit, colours included.

        ⚠️ A fair test rather than a worst case, exactly as
        `week4_place.auto_layout` argues: it spreads them, because a success
        rate measured on an auto-filled house should not secretly be a
        measurement of one pathological cluster. Make it hard by hand — nothing
        stops you.
        """
        rng = np.random.default_rng(w4p.random_seed())
        want = self.band.limit - len(self.crop.placed)
        added, tries = 0, 0
        while added < want and tries < 400:
            tries += 1
            tag = self.band.arms[added % len(self.band.arms)]
            y_lo, y_hi = self.band.y[tag]
            pos = np.array([self.band.row_x[tag],
                            float(rng.uniform(y_lo + 0.3, y_hi - 0.3)),
                            hang_z(int(rng.integers(0, 3)))])
            keep = self.stage_i
            # A spread of stages, so the "only red is taken" half of the demo
            # has something to show without anyone pressing a key.
            self.stage_i = int(rng.integers(0, len(STAGES)))
            if self.place(pos, quiet=True) is not None:
                added += 1
            self.stage_i = keep
        self._say(f"auto-filled to {len(self.crop.placed)} fruit", GOOD)

    def run_script(self, spec):
        """Hang an exact set: 'arm:y:height:stage' entries, comma separated.

        ⚠️ **Not a shortcut past the demo — the same `place` a click calls.** A
        mouse cannot be replayed, and every claim this file makes about ten
        placements or a 70 mm pair has to be reproducible by somebody else. So
        the script drives the identical path: same `RailBand.check`, same
        refusals, same `Crop`, same recolour. What it does not do is drive the
        *pointer*, so the hover preview is the one thing it cannot exercise.
        """
        placed = []
        for entry in spec.replace(";", ",").split(","):
            entry = entry.strip()
            if not entry:
                continue
            bits = entry.split(":")
            if len(bits) != 4:
                raise SystemExit(f"--script entry {entry!r} is not "
                                 f"arm:y:height:stage")
            tag, y, height, stage = bits
            if tag not in self.band.arms:
                raise SystemExit(f"--script names arm {tag!r}; this machine "
                                 f"has {', '.join(self.band.arms)}")
            if stage not in STAGES:
                raise SystemExit(f"--script names stage {stage!r}; "
                                 f"one of {', '.join(STAGES)}")
            self.stage_i = STAGES.index(stage)
            self.height_i = (HANG_NAMES.index(height)
                             if height in HANG_NAMES else int(height))
            pos = np.array([self.band.row_x[tag], float(y), self.z])
            name = self.place(pos)
            placed.append((entry, name))
        return placed

    def clear(self):
        self.crop.clear()
        self.stages.clear()
        self._say("cleared", DIM)

    # --- keys ----------------------------------------------------------------

    def key(self, code):
        if code == 32:                                   # SPACE
            if self.running:
                return
            if self.crop.placed:
                self.go = True
            else:
                self._say("hang at least one tomato first", BAD)
        elif code in (ord("t"), ord("T")):
            self.stage_i = (self.stage_i + 1) % len(STAGES)
            self._say(f"hanging {self.cursor_text()}", INK)
        elif code in (ord("1"), ord("2"), ord("3"), ord("4")):
            self.stage_i = code - ord("1")
            self._say(f"hanging {self.cursor_text()}", INK)
        elif code in (ord("z"), ord("Z")):
            self.height_i = (self.height_i + 1) % 3
            self._say(f"hanging {self.cursor_text()}", INK)
        elif code in (ord("a"), ord("A")):
            if not self.running:
                self.auto_fill()
        elif code in (ord("c"), ord("C")):
            if not self.running:
                self.clear()
        elif code in (ord("r"), ord("R")):
            self.reset = True
        if self.hover is not None:
            # The cursor tag carries the height and the stage, so a key press
            # has to re-evaluate what the pointer is currently over.
            self.on_move(*self.hover)

    def _say(self, text, colour):
        self.note, self.note_colour = str(text), colour
        print(f"    {text}")

    # --- drawing -------------------------------------------------------------

    def underlay(self, img, panel):
        """The bands, the cursor and the note. Drawn under the mapped dots.

        ⚠️ Under, so a band never hides a fruit. The map's job during a run is
        still to show what the robot believes it found; the placement furniture
        is context for that, not a replacement for it.
        """
        import cv2

        self.map = panel
        wash = img.copy()
        for t in self.band.arms:
            col = taf.ARM_COL[t]
            y_lo, y_hi = self.band.y[t]
            x0 = self.band.row_x[t] - self.band.snap_x
            x1 = self.band.row_x[t] + self.band.snap_x
            # The click catchment: anywhere in here means this arm's row, so
            # nobody has to hit a line. Washed, because it is not where the
            # tomato ends up.
            cv2.rectangle(wash, panel.px(x0, y_lo), panel.px(x1, y_hi), col, -1)
        cv2.addWeighted(wash, 0.16, img, 0.84, 0, img)

        for t in self.band.arms:
            col = taf.ARM_COL[t]
            y_lo, y_hi = self.band.y[t]
            rx = self.band.row_x[t]
            # Where the tomato actually lands: the row itself, drawn solid.
            cv2.rectangle(img, panel.px(rx - 0.045, y_lo),
                          panel.px(rx + 0.045, y_hi), col, 1, cv2.LINE_AA)
            u, v = panel.px(rx, y_lo)
            cv2.putText(img, f"arm{1 if t == 'a' else 2} band  r{self.band.row[t]}",
                        (u - 30, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1,
                        cv2.LINE_AA)

        # --- the cursor, carrying what it would hang ------------------------
        if self.hover is not None and self._panel_pixel(*self.hover):
            u, v = self._panel_pixel(*self.hover)
            if self.hover_world is not None and not self.hover_why:
                col = taf.STAGE_BGR.get(self.stage, INK)
                cu, cv_ = panel.px(self.hover_world[0], self.hover_world[1])
                cv2.circle(img, (cu, cv_), 6, col, 1, cv2.LINE_AA)
                cv2.line(img, (u, v), (cu, cv_), col, 1, cv2.LINE_AA)
                cv2.putText(img, self.cursor_text(), (u + 10, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)
            else:
                # ⚠️ Refused *before* the button goes down — a cross where the
                # pointer is, and the reason on the note bar rather than beside
                # the cursor. Beside the cursor it has only the pixels between
                # the pointer and the panel edge, and the reasons are whole
                # sentences: the first version clipped "r2 is not worked from
                # aisle a0; the t" mid-word, near the edge, which is where a
                # refusal is most likely to happen.
                cv2.drawMarker(img, (u, v), BAD, cv2.MARKER_TILTED_CROSS, 9, 1)

        # --- the height strip, so the Z key is visible ----------------------
        sx = panel.w - 16
        z0, z1 = self.band.z
        y0, y1 = 46, panel.h - 40

        def at(z):
            return int(y1 - (z - z0) / max(1e-9, z1 - z0) * (y1 - y0))

        # ⚠️ Two bands on the strip, because the envelope has two and flattening
        # them is bug log entry 42. The filled part is `GUARANTEED_Z` (10/10
        # clean); the outline either side is the wider `MARGINAL_Z` (18/21). A
        # height in the outline is accepted and is expected to fail about one
        # time in seven, and the strip is where that is visible before the
        # click rather than in a log afterwards.
        cv2.rectangle(img, (sx - 5, y0), (sx + 5, y1), (70, 70, 70), 1)
        cv2.rectangle(img, (sx - 4, at(self.band.z_sure[1])),
                      (sx + 4, at(self.band.z_sure[0])), (60, 90, 60), -1)
        for i in range(3):
            yy = at(hang_z(i))
            live = (i == self.height_i % 3)
            cv2.line(img, (sx - 6, yy), (sx + 6, yy),
                     GOOD if live else (110, 110, 110), 2 if live else 1)
        cv2.putText(img, "z", (sx - 3, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    DIM, 1, cv2.LINE_AA)

        # --- the one line that says what just happened -----------------------
        #
        # ⚠️ Directly under the title, not along the bottom. The bottom of this
        # panel already carries `DuoMapPanel`'s own legend — the stage swatches
        # and "small grey dot above = ground truth" — and a note drawn there
        # lands on top of it. Seen in the first recording: the refusal reason
        # and the legend were printed over each other, which is exactly the
        # illegible moment a refusal must not have.
        # ⚠️ What the pointer is over beats what last happened. The person is
        # about to click; "that one would be refused, and here is why" is worth
        # more to them right now than the record of the tomato they hung a
        # second ago, which is on the map in front of them anyway.
        text, colour = ((self.hover_why, BAD) if self.hover_why
                        else (self.note, self.note_colour))
        cv2.rectangle(img, (0, 26), (panel.w, 44), (16, 16, 16), -1)
        cv2.putText(img, text[:88], (6, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, colour, 1, cv2.LINE_AA)
        return img


# --- the demo -----------------------------------------------------------------

def _park_everything(model, data, arms, parks):
    """Both arms on their park postures, from a clean reset. See `duo.run`."""
    import mujoco

    from mission import park_arm, reset_park

    # ⚠️ Reset **once**, then place each arm. `reset_park` is `mj_resetData`
    # underneath, so calling it per arm throws away the arm parked on the
    # previous pass. `farm/eyes.py` records this one.
    reset_park(model, data, parks[arms[0]], prefix=trolley.ARM_PREFIX[arms[0]])
    for t in arms:
        park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
    mujoco.mj_forward(model, data)


def audit(state, placer, crop_placed_before):
    """Did the router ever read a true position? Printed, every run.

    ⚠️ **The one claim this demo would be worthless without.** If placement fed
    the router the coordinates the person clicked, the trolley would drive
    straight to every tomato and the whole thing would be a puppet show. So for
    every fruit that reached a crate this prints the distance between where the
    **map** said it was — the only position the arm was ever sent to — and where
    it actually hung. A column of zeros would mean ground truth had leaked in.

    The estimate is recovered from `state.named`, which `duo.associate` fills as
    each stop is worked: `id(sighting) -> truss name`. The sighting is the deck
    cameras' own fused answer and nothing else was ever handed to `route.plan`.
    """
    if state.house_map is None:
        return []
    est = {}
    for s in state.house_map.sightings:
        name = state.named.get(id(s))
        if name is not None:
            est[name] = np.asarray(s.pos, float)
    rows = []
    for name in sorted(state.picked | state.missed | state.refused):
        truth = crop_placed_before.get(name)
        e = est.get(name)
        if truth is None or e is None:
            rows.append((name, placer.stages.get(name, "?"), None))
            continue
        rows.append((name, placer.stages.get(name, "?"),
                     float(np.linalg.norm(e - np.asarray(truth, float)) * 1000)))
    return rows


def main(argv=None):
    import argparse
    import os
    import time
    import traceback

    ap = argparse.ArgumentParser(
        prog="two_arm_farm.py mousereach",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("--arms", type=int, default=2, choices=(1, 2))
    ap.add_argument("--seed", type=int, default=None,
                    help="repeat a house; omitted draws a new one and prints it")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="fraction of rated joint speed")
    ap.add_argument("--stride", type=float, default=None,
                    help="metres between survey stops; wider is a faster "
                         "survey and a coarser map")
    ap.add_argument("--stops", type=int, default=None, help="cap the stops")
    ap.add_argument("--fps", type=int, default=10, help="panel rate")
    ap.add_argument("--hsv-hz", type=float, default=3.0,
                    help="how often the ripeness overlay re-detects")
    ap.add_argument("--panel-scale", type=float, default=1.0)
    ap.add_argument("--record", default=None, metavar="STEM",
                    help="write <STEM>_sensors.mp4 and <STEM>_mission.mp4 "
                         "**while** the windows are live. Not --out, which "
                         "records instead of showing.")
    ap.add_argument("--place", type=int, default=None, metavar="N",
                    help="auto-fill N fruit and start immediately — the "
                         "unattended path, for checking the demo still runs")
    ap.add_argument("--script", default=None, metavar="SPEC",
                    help="hang an exact set instead of clicking, as "
                         "'arm:y:height:stage' separated by commas — e.g. "
                         "'a:+1.20:mid:red,b:-0.35:low:green'. This is how the "
                         "demo is regression-tested; a mouse cannot be "
                         "replayed and a claim about ten placements has to be "
                         "reproducible.")
    ap.add_argument("--headless", action="store_true",
                    help="no windows. Panels are still composed and still "
                         "recorded with --record, so the overlay and the map "
                         "are exercised exactly as they are live — this hides "
                         "the window, it does not skip the work.")
    args = ap.parse_args([] if argv is None else argv)

    # ⚠️ A hang, caught at the argument rather than at 3am. With no window there
    # is nothing to click and no key to press, so the placement loop below would
    # spin for ever waiting for a SPACE that cannot arrive.
    if args.headless and not (args.script or args.place):
        raise SystemExit(
            "--headless has no mouse and no keyboard, so nothing can ever be "
            "placed or started. Give it --script or --place N.")

    # ⚠️ EGL, always. Both windows are OpenCV over offscreen renders and there
    # is no GLFW viewer in this process — see `two_arm_farm`'s docstring.
    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import mujoco

    from farm import decks

    arms = ("a", "b")[: args.arms]
    # ⚠️ Printed before anything is built, so a run that breaks on a layout is
    # a run somebody can reproduce. `crop.resolve_seed` does the announcing.
    seed = fcrop.resolve_seed(args.seed, label="mousereach")
    print(f"  seed {seed} — pass --seed {seed} to open this same house again")

    pool = pool_trusses()
    model = trolley.build(aisle=args.aisle, arms=arms, trusses=pool,
                          wrist_cam=True, arm_decks=True, seed=seed)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    from plant_row import Row

    parks = {t: armframe.park_posture(model, data, t, arms=arms) for t in arms}
    _park_everything(model, data, arms, parks)
    row = Row(model, data, names=[t.name for t in pool],
              homes={t.name: t.pos for t in pool})

    band = RailBand(aisle=args.aisle, arms=arms)
    crop = w4p.Crop(model, data, row, [t.name for t in pool], rules=band,
                    park=park_spot)
    placer = Placer(crop, band, pool, seed=seed)

    print("\n  --- the reachable band, derived from the Week 1-4 envelope ---")
    for line in band.describe():
        print(f"  {line}")
    print(f"  standoff {house.WORK_STANDOFF * 1000:.0f} mm — the figure every "
          f"Week 1-4 clearance was measured at, which is why the envelope "
          f"carries over\n")

    state = duo.DuoState(arms=arms, aisle=args.aisle)
    state.trusses = []
    hs = decks.heads(model, data, arms=arms, aisle=args.aisle)
    for t in arms:
        state.state[t].head = hs.get(t)

    map_panel = taf.DuoMapPanel(underlay=placer.underlay,
                                title="THE MAP  -  click a band to hang a "
                                      "tomato")
    # ⚠️ Handed over here rather than left for `underlay` to set on the first
    # frame it draws. `on_move` and `on_click` both need the panel to turn a
    # window pixel into a world position, and a pointer can be over the window
    # before the first frame is composited — so the alternative is a short
    # window at startup where the cursor preview silently does nothing.
    placer.map = map_panel
    windows = taf.Windows(model, data, state, fps=args.fps,
                          hsv_hz=args.hsv_hz, arms=arms,
                          panel_scale=args.panel_scale,
                          on_click=placer.on_click, on_key=placer.key,
                          on_move=placer.on_move, record=args.record,
                          live=not args.headless, map_panel=map_panel)

    def sync():
        """`state.trusses` is the placed crop, live. See `DuoState.truth_points`."""
        state.trusses = [placer.by_name[n] for n in crop.placed]

    def tick(_t=None):
        # ⚠️ R has to be able to unwind a run from inside it — there is no other
        # thread and no other loop. Raising here reaches `duo.run`'s callers
        # through the same path a `Sink` quit uses, and every `finally` on the
        # way out (the deck-centre token, the black boxes) still runs.
        if placer.reset:
            raise Restart
        sync()
        windows.tick(_t)

    def reset_scene():
        _park_everything(model, data, arms, parks)
        crop.row = row
        crop.clear()
        trolley.Drive(model, data).park_at(0.0)
        mujoco.mj_forward(model, data)
        placer.stages.clear()
        for t in pool:
            p = park_spot(int(t.name[1:]))
            t.x, t.y, t.z, t.stage = float(p[0]), float(p[1]), float(p[2]), "red"
        placer.reset = False
        placer.running = False
        placer.go = False
        return duo.DuoState(arms=arms, aisle=args.aisle)

    print("  window open. Click a band in THE MAP to hang a tomato.")
    print("    SPACE harvest   T/1-4 ripeness   Z height   A auto-fill")
    print("    C clear         R reset          Q quit\n")

    runs = 0
    try:
        while True:
            # --- place -------------------------------------------------------
            placer.go = False
            placer.running = False
            if runs == 0 and (args.place or args.script):
                if args.script:
                    placer.run_script(args.script)
                else:
                    placer.auto_fill()
                placer.go = True
            while not placer.go and not placer.reset:
                mujoco.mj_step(model, data)
                sync()
                windows.push_frame()
            if placer.reset:
                state = reset_scene()
                windows.state = state
                for t in arms:
                    state.state[t].head = hs.get(t)
                continue

            # --- harvest -----------------------------------------------------
            runs += 1
            placer.running = True
            before = {n: p.copy() for n, p in crop.placed.items()}
            if runs > 1:
                # ⚠️ A fresh state per run, because a second SPACE is a fresh
                # survey of what is left on the plant and not a continuation.
                # Reusing it would carry `picked_by`, the per-arm `ArmStats` and
                # the previous map across, so the summary would count the first
                # run's tomatoes again and the map would still be drawing dots
                # for fruit that are now in a crate.
                state = duo.DuoState(arms=arms, aisle=args.aisle)
                windows.state = state
                for t in arms:
                    state.state[t].head = hs.get(t)
            sync()
            t0 = time.perf_counter()
            try:
                duo.run(model, data, list(pool), state, arms=arms,
                        aisle=args.aisle, speed=args.speed,
                        use_truth=False, max_stops=args.stops,
                        on_tick=tick, stride=args.stride, verbose=True,
                        crop_version=lambda: crop.version,
                        truth=[placer.by_name[n] for n in crop.placed],
                        after_reset=lambda r: (setattr(crop, "row", r),
                                               crop.restore()))
                windows.flush(args.fps * 2)
            except Restart:
                print("\n  R — resetting the scene\n")
            except KeyboardInterrupt:
                raise
            except Exception:
                # ⚠️ **The demo continues.** Somebody is holding the mouse; a
                # traceback on a terminal they are not looking at, followed by a
                # dead window, is the worst possible outcome. The machine is
                # parked, the reason goes on the map panel in plain words, and
                # placement comes back.
                traceback.print_exc()
                placer._say("run stopped - machine parked. R resets, or hang "
                            "more and press SPACE", BAD)
                _park_everything(model, data, arms, parks)
                crop.restore()
            else:
                duo.report(state)
                print(f"\n  {time.perf_counter() - t0:.0f} s wall")
                rows = audit(state, placer, before)
                print(f"\n  --- what the router was actually given ---")
                print(f"  {'fruit':<7} {'hung as':<9} {'map error vs truth':>19}"
                      f"  outcome")
                for name, stage, err in rows:
                    got = ("crated" if name in state.picked else
                           "refused" if name in state.refused else "lost")
                    e = "not mapped" if err is None else f"{err:.0f} mm"
                    print(f"  {name:<7} {stage:<9} {e:>19}  {got}")
                print(f"\n  ⚠️ Every arm was sent to the deck cameras' estimate "
                      f"and never to the\n     position anyone clicked. A "
                      f"column of zeros above would mean the\n     demo had "
                      f"become a puppet show; see `audit`.")
                if getattr(state, "replans", 0):
                    print(f"\n  the crop changed {state.replans} time(s) "
                          f"mid-run and every checked plan was thrown away")
                for name in state.picked:
                    # Out of the crop, left in the crate. See `Crop.retire`.
                    crop.retire(name)
                placer._say(f"run {runs} done - {state.totals()['crated']} "
                            f"crated. Hang more and press SPACE, or R to reset",
                            GOOD)

            placer.running = False
            if args.headless:
                # Nothing can ask for a second run, so do not sit in the
                # placement loop waiting to be asked. See the argument check.
                break
            if placer.reset:
                state = reset_scene()
                windows.state = state
                for t in arms:
                    state.state[t].head = hs.get(t)
    except KeyboardInterrupt:
        print("\n  window closed")
    finally:
        if state.scout is not None:
            state.scout.close()
        windows.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
