#!/usr/bin/env python3
"""Put tomatoes wherever you like, then watch it plan and harvest them.

Up to 15 fruit, anywhere the arm can work, in an arrangement the robot has never
seen — and more can be added while it is working, which it notices.

This is the generalisation `week1_mousereach.py` was pointing at. Its docstring
says it is "the same chain with a detector in place of a mouse"; this is that
chain with fifteen fruit and a planner behind it.

    # PLACE THEM YOURSELF — a window opens with a green board showing exactly
    # where the arm can reach. Double-click it to hang a tomato there.
    #   double-click  place        A  auto-fill        C  clear
    #   SPACE         harvest      Q  quit
    ./.venv/bin/python simulation/mujoco/week4_place.py --windowed

    # watch it work an auto-placed row, live
    ./.venv/bin/python simulation/mujoco/week4_place.py --grid 6 --windowed

    # 4 fruit, then 3 more appear mid-run and the planner notices
    ./.venv/bin/python simulation/mujoco/week4_place.py --grid 4 --add-at 2 --windowed

    # headless / batch
    ./.venv/bin/python simulation/mujoco/week4_place.py --grid 15
    ./.venv/bin/python simulation/mujoco/week4_place.py --grid 12 --save L.json
    ./.venv/bin/python simulation/mujoco/week4_place.py --layout L.json --seen
    ./.venv/bin/python simulation/mujoco/week4_place.py --grid 12 --out week4_place.mp4

    # the old behaviour: no chassis camera, fruit picked in placement order
    ./.venv/bin/python simulation/mujoco/week4_place.py --grid 10 --no-deck

--- how it works, and the four things that are not obvious -------------------

**The deck camera chooses the order and aims the wrist camera.** Before anything
moves, the chassis mast (`deck_cam.py`) sweeps its pan-tilt head across five
poses and fuses what they saw into one survey of the whole row;
`deck_cam.plan_order` turns that into the sequence to work it in, and the wrist
camera is sent to *measured* positions rather than to `crop.placed`. Both matter
and neither is cosmetic — see `harvest_placed`. `--no-deck` puts the old
behaviour back, which was placement order and a look pose read from the answer.

**There is no minimum spacing** — see `TOUCHING` below for the rule that was
removed and the measurement that says it was wrong. Fruit may be placed
touching, and a close pair is then a pick-order problem for the deck camera
rather than a placement the tool refuses.

**The green board IS the arm's range.** It is sized to the band re-measured in
`week4_envelope.py` — 49 cells, one full pick each — so anywhere you can click is
somewhere the arm can work, and there is nothing to guess. It is `contype=0`
scenery and only exists when a window is open, so it changes no dynamics and
batch scenes stay byte-identical to the campaign's.

**You cannot add a body to a compiled model.** So a POOL of trusses is compiled
once and "placing" is `Row.place()`, which already moves the mocap stem anchor
and the fruit's qpos together and zeroes its velocity. Unplaced fruit are parked
far behind the arm. No recompile: every id cached by `Row`, `RobotSpheres`,
`CropObstacles` and the reacher is an index into the model, and recompiling
mid-run invalidates all of them silently.

⚠️ **Where the unplaced fruit park decides whether the safety net works.**
`CropObstacles.__init__` freezes its membership:

    self.fruit = [... for n in row.names if n != target and row.attached(n)]

Positions are read live every cycle; the *list* is built once. `Guard` builds a
`ClearanceModel`, so a Guard watching a mission in flight has that frozen list
too. Therefore the pool parks with its welds **active**: `row.attached()` is
True, every pool member is in the membership from the start, and a fruit
teleported into the scene mid-flight is seen by the running Guard for free.
Park them detached — the tidy-looking choice — and a fruit placed mid-flight is
invisible: the arm drives through it and neither the plan check nor the Guard
says a word.

⚠️ **Changing the crop voids every checked plan.** The planner's guarantee is
that the route it verified clears every fruit it is not picking by CLEARANCE.
Add a fruit after that check and the guarantee is not weakened, it is void — the
route was never checked against the new fruit at all. So the crop set carries a
version, a mission records the version it was planned against, and a mission
whose version is stale is replanned before the next leg or abandoned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_row import FRUIT_R, ROW_X  # noqa: E402

# --- where fruit may be placed -----------------------------------------------
#
# ⚠️ **Re-measured 2026-08-04 against the scene these numbers are actually used
# in.** The first version imported the envelope from `week1_mousereach.py`,
# which measured it cell by cell on a 31x24 grid — but against a *different*
# scene: board at x=0.55 with the crate out at y=-0.80. Week 2 moved the crate
# in to y=-0.52 and put the row at x=0.60, and nobody re-measured. The result
# was a board advertising "anywhere here is pickable" over a region where only
# **21 of 49 cells actually picked cleanly**.
#
# `week4_envelope.py` re-ran the same experiment on the real scene, 7x7 cells,
# one full pick each:
#
#           -0.55  -0.37  -0.18  +0.00  +0.18  +0.37  +0.55
#   z 0.95      u      O      a      a      a      a      g
#   z 0.82      O      a      a      a      a      O      g
#   z 0.68      O      O      O      O      O      O      g
#   z 0.55      O      O      O      O      O      O      O
#   z 0.42      O      O      d      O      O      g      O
#   z 0.28      X      g      g      u      u      X      g
#   z 0.15      u      u      d      g      u      u      X
#
# Both failing bands have a physical cause in the scene rather than in the arm:
#
#   * **Above z 0.83** a fruit's stem anchor is at or above the support bar the
#     trusses hang from (`plant_row.SUPPORT_Z` 0.88 minus `STEM_LEN` 0.05), so
#     the arm is reaching into the bar. Those cells abort on the guard.
#   * **Below z 0.32** the fruit is under the substrate gutter
#     (`greenhouse.GUTTER_Z`), which no real truss ever is.
#
# So the pickable band is genuinely narrow, and it is narrow for horticultural
# reasons rather than arbitrary ones. The five fixed trusses this repo has
# always scored 5/5 on sit at z 0.54-0.72 — inside it, which is exactly why
# they always worked and arbitrary placements did not.
GUARANTEED_HALF_Y = 0.37     # every cell in this box picked clean: 10/10
GUARANTEED_Z = (0.50, 0.70)
# The wider band placement is still allowed in, at 18/21 = 86% measured. A
# placement here is recorded as `marginal` so a failure out at the edge reads as
# the expected result rather than a surprise.
MARGINAL_HALF_Y = 0.55
MARGINAL_Z = (0.42, 0.72)

# ⚠️ **There is no minimum spacing any more, and that is a deliberate removal.**
#
# This used to refuse any placement within 200 mm of another fruit, and the
# reason it gave was this:
#
#     "plant_row.py records what happens below it: a truss placed 127 mm out
#      hung at 4.76 N instead of 1.18 N and spiked to 9.25 N — over SNAP_N — so
#      it snapped itself before the arm ever moved."
#
# ⚠️ **That justification does not survive being measured, and it never
# applied.** The 127 mm in `plant_row.py` is a fruit's distance from the arm's
# *home tool0* — a tomato spawned inside the open gripper, which physics then
# shoves. It is not a fruit-to-fruit distance, and the rule read its own
# evidence wrong. Two hanging fruit, settled for 3 s, peduncle force at the peak:
#
#   centres     70 mm    85 mm   100 mm   140 mm   200 mm
#   peak N       1.18     1.18     1.18     1.18     1.18
#
# 1.18 N is 0.12 kg x 9.81 — each fruit carrying its own weight and nothing
# else, at every spacing down to touching. Neighbouring stems do not load each
# other at all, because the stems are `contype=0` and the welds are independent.
# The rule was guarding against something that does not happen.
#
# What *does* happen at close spacing is a planning problem, not a physics one,
# and refusing the arrangement was the wrong response to it for two reasons:
#
#   * **A grower does not space the crop for the robot.** Trusses hang where the
#     plant puts them, and clusters are the normal case rather than an abuse of
#     the tool. A demo that only accepts comfortably spaced fruit is a demo that
#     has quietly removed the hard part.
#   * **The hard part turned out to be solvable, and the rule was hiding it.**
#     The pair sweep shows that at 100 mm apart *one* of the two is refused and
#     the other plans fine — so picking the plannable one first turns a pair
#     that looks impossible into two clean picks. That is what `deck_cam.
#     plan_order` does, and it could not have been found while the arrangement
#     was forbidden.
#
# What replaces it is not a constraint but information: `check` still reports
# how close the nearest neighbour is, `Crop.place` prints it, and the deck
# camera's order is what actually deals with it.
#
# The one thing still refused is fruit that would occupy the same space —
# `TOUCHING` below — and that is geometry, not policy: two spheres whose centres
# are less than a diameter apart are interpenetrating, and MuJoCo resolves that
# by firing them apart at speed before the arm has moved.
TOUCHING = 2 * FRUIT_R + 0.004      # 70 mm; 4 mm of skin so they rest, not push

# What a *close* neighbour means, for reporting only. Both come from the sweep
# in `deck_cam`: under 120 mm centre to centre a square-on pick is refused,
# under 170 mm it costs a fallback route. Neither refuses a placement here.
CLOSE_M = 0.170
VERY_CLOSE_M = 0.120

# ⚠️ **Fifteen again.** Ten was not a property of the arm — it was arithmetic on
# the old 200 mm lattice: a 1.10 x 0.30 m band holds 5 x 2 slots at that pitch
# and no more. With the spacing rule gone the limit is the pool and the band,
# and 15 fruit fit the measured band comfortably. `auto_layout` still spreads
# them, because an auto-filled row exists to be a fair test rather than a
# worst case; put them on top of each other by hand if that is what you want to
# see.
MAX_FRUIT = 15
POOL = 24

# Where unplaced fruit hang: behind the arm, clear of everything, welds live.
# See the module docstring — this is not tidiness, it is what keeps the Guard
# able to see a fruit that gets placed later.
PARK_X = -1.5
PARK_PITCH = 0.12

# The clickable board. Named so `board_click_to_yz` and the selection test can
# find it without threading the id through every call.
PLACE_BOARD = "place_board"

# Speed for interactive runs. Faster than the 0.15 the measurements use, because
# watching is the point here — see the --speed help.
WATCH_SPEED = 0.4


def pool_trusses(n=POOL):
    """A compile-time pool. Every member starts parked out of the way."""
    return [(f"p{i:02d}", -0.4 + PARK_PITCH * (i % 8), 0.5 + 0.25 * (i // 8))
            for i in range(n)]


def park_spot(i):
    return np.array([PARK_X, -0.4 + PARK_PITCH * (i % 8), 0.5 + 0.25 * (i // 8)])


# --- the placement rules -----------------------------------------------------

def zone(y, z):
    """'guaranteed', 'marginal' or None — measured, not derived.

    See the map above. None means the arm was measured to fail there, so the
    placement is refused rather than accepted and quietly lost.
    """
    if (abs(y) <= GUARANTEED_HALF_Y
            and GUARANTEED_Z[0] <= z <= GUARANTEED_Z[1]):
        return "guaranteed"
    if abs(y) <= MARGINAL_HALF_Y and MARGINAL_Z[0] <= z <= MARGINAL_Z[1]:
        return "marginal"
    return None


def nearest(y, z, placed):
    """(name, metres) of the closest already-placed fruit, or (None, inf)."""
    best, best_d = None, float("inf")
    for name, p in placed.items():
        d = float(np.hypot(p[1] - y, p[2] - z))
        if d < best_d:
            best, best_d = name, d
    return best, best_d


def nearest_world(pos, placed):
    """(name, metres) of the closest already-placed fruit, in full 3D.

    ⚠️ `nearest` above measures in the board's own (y, z) because every fruit on
    a bolted arm's board shares one x, so the two are the same number. They stop
    being the same number the moment fruit hang on more than one row — which is
    what `farm.mousereach` does — and a 2D distance there would call two fruit
    1.6 m apart on opposite sides of the aisle "touching". Kept as a second
    function rather than a widened `nearest` so nothing Weeks 1-4 measured can
    move: `check` still calls the 2D one and gets the identical answer.
    """
    pos = np.asarray(pos, float)
    best, best_d = None, float("inf")
    for name, p in placed.items():
        d = float(np.linalg.norm(np.asarray(p, float) - pos))
        if d < best_d:
            best, best_d = name, d
    return best, best_d


def check(y, z, placed):
    """(ok, reason, zone). Refusing says why — a silent refusal teaches nothing.

    Two refusals left, and neither of them is a spacing policy:

      * outside the band the arm was *measured* to work in (`zone`)
      * close enough to another fruit that the two spheres overlap (`TOUCHING`)

    A crowded-but-separate placement is accepted. `reason` still comes back
    populated in that case so the caller can say how tight it is — see
    `Crop.place`, which prints it — because "the robot took this and it is going
    to be hard" is useful and "refused" is not.
    """
    zn = zone(y, z)
    if zn is None:
        return False, (f"outside the arm's measured envelope "
                       f"(y {y:+.2f}, z {z:.2f})"), None
    if len(placed) >= MAX_FRUIT:
        return False, f"already at the {MAX_FRUIT}-fruit limit", zn
    name, d = nearest(y, z, placed)
    if name is not None and d < TOUCHING:
        return False, (f"{d * 1000:.0f} mm from {name} — two tomatoes cannot "
                       f"occupy the same space ({TOUCHING * 1000:.0f} mm "
                       f"centre to centre is touching)"), zn
    if name is not None and d < VERY_CLOSE_M:
        return True, (f"{d * 1000:.0f} mm from {name} — square-on this pair "
                      f"cannot both be planned; the order is what solves it"), zn
    if name is not None and d < CLOSE_M:
        return True, (f"{d * 1000:.0f} mm from {name} — tight, expect a "
                      f"fallback route"), zn
    return True, "", zn


class BoltedBoard:
    """Where fruit may hang in front of an arm bolted to the floor.

    ⚠️ **The rule half of `Crop`, split out so the state-machine half can be
    reused on a scene this file knows nothing about.** `Crop` is a pool of
    compiled trusses, a placed set, a version counter and the park/show/place
    bookkeeping that keeps a later-placed fruit inside `CropObstacles`' frozen
    membership. None of that is about *where* fruit may go — that is `zone`,
    `TOUCHING` and `MAX_FRUIT`, and on a trolley whose base moves it is a
    different answer entirely (`farm.mousereach.RailBand`).

    Splitting it is not tidiness. The alternative was a second copy of the
    placement state machine on the farm scene, and a duplicated state machine is
    already bug 7 in this repo's log.

    This class is the Week 1-4 answer and it is the default, so every existing
    caller of `Crop(model, data, row, names)` gets exactly what it got before.
    """

    limit = MAX_FRUIT

    #: A click in this scene names a point on the board, so x is the row's.
    def world(self, y, z):
        return np.array([ROW_X, float(y), float(z)])

    def check(self, pos, placed):
        """(ok, why, zone) for a world position. Delegates to `check`."""
        return check(float(pos[1]), float(pos[2]), placed)

    def describe(self):
        return (f"board  |y| <= {MARGINAL_HALF_Y}  z "
                f"{MARGINAL_Z[0]}-{MARGINAL_Z[1]}  (18/21 clean)")


BOLTED = BoltedBoard()


# Lattice pitches `auto_layout` will try, widest first. The first two are the
# thresholds the pair sweep in `deck_cam` found — 170 mm is where the direct
# route survives a neighbour, 120 mm is where a square-on pick starts being
# refused outright — so an auto-filled row says something about which regime it
# is in rather than just "some tomatoes".
AUTO_PITCHES = (0.24, 0.20, 0.17, 0.15, 0.13, 0.11, 0.09)


def auto_layout(n, seed=0, margin=0.02):
    """`n` fruit on a jittered lattice across the measured band.

    An auto-filled row is a *fair test*, not a worst case: it spreads the fruit
    as far apart as the count allows, so a success rate measured on it is not
    secretly a measurement of one pathological cluster. Hand placement is where
    you make it hard — there is nothing stopping you now.

    ⚠️ **Widest pitch that fits, rather than one fixed pitch and a cap.** The
    old version pitched the lattice at the 200 mm minimum plus the jitter, which
    fixed the slot count at 10; asking for 15 printed a warning and silently
    gave you 10. Choosing the pitch from the count instead means `--grid 15`
    means fifteen, and it says which of the pair-sweep regimes the row landed in
    rather than pretending density is not a variable.
    """
    rng = np.random.default_rng(seed)
    pitch = AUTO_PITCHES[-1]
    slots = []
    for p in AUTO_PITCHES:
        ys = np.arange(-MARGINAL_HALF_Y, MARGINAL_HALF_Y + 1e-9, p)
        zs = np.arange(MARGINAL_Z[0], MARGINAL_Z[1] + 1e-9, p)
        cells = [(float(y), float(z)) for y in ys for z in zs]
        pitch, slots = p, cells
        if len(cells) >= n:
            break
    if n > len(slots):
        print(f"  ⚠️ {n} requested; {len(slots)} slots fit the measured band "
              f"even at the {pitch * 1000:.0f} mm lattice")
        n = len(slots)
    elif pitch < CLOSE_M:
        print(f"  ⚠️ {n} fruit needs a {pitch * 1000:.0f} mm lattice — inside "
              f"the {CLOSE_M * 1000:.0f} mm at which a neighbour starts costing "
              f"a fallback route. Expect the order to matter.")
    rng.shuffle(slots)
    out, placed = [], {}
    for y, z in slots:
        if len(out) >= n:
            break
        # Jitter has to keep the spacing, so it is capped at half the slack.
        jy = float(rng.uniform(-margin, margin))
        jz = float(rng.uniform(-margin, margin))
        y2, z2 = y + jy, z + jz
        ok, _why, _zn = check(y2, z2, placed)
        if not ok:
            y2, z2 = y, z
            ok, _why, _zn = check(y2, z2, placed)
            if not ok:
                continue
        name = f"p{len(out):02d}"
        placed[name] = np.array([ROW_X, y2, z2])
        out.append((name, round(y2, 4), round(z2, 4)))
    return out


def random_seed():
    """A fresh seed from system entropy, for interactive auto-fill.

    Batch runs take an explicit `--seed` and stay reproducible; only the
    interactive "give me a new arrangement" path uses this.
    """
    import secrets

    return secrets.randbelow(2 ** 31)


def save_layout(path, layout, meta=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pts = [(y, z) for _n, y, z in layout]
    closest = min((float(np.hypot(a[0] - b[0], a[1] - b[1]))
                   for i, a in enumerate(pts) for b in pts[i + 1:]),
                  default=float("nan"))
    p.write_text(json.dumps(
        {"fruit": [{"name": n, "y": y, "z": z} for n, y, z in layout],
         # Recorded, not enforced. A layout file is a description of an
         # arrangement, and how tight it is decides how much the pick order
         # matters — which is worth knowing when reading a run back.
         "closest_pair_mm": round(closest * 1000, 1), **(meta or {})}, indent=2))
    print(f"  layout: {len(layout)} fruit -> {p}")


def load_layout(path):
    d = json.loads(Path(path).read_text())
    return [(f["name"], f["y"], f["z"]) for f in d["fruit"]]


# --- the live crop -----------------------------------------------------------

class Crop:
    """The placed fruit, and a version that ticks whenever the set changes.

    The version is the whole point. A `Mission` records the version it was
    planned against; if the crop has changed since, that plan's clearance
    guarantee no longer covers the world it is about to be flown in.
    """

    def __init__(self, model, data, row, names, rules=None, park=None):
        self.model = model
        self.data = data
        self.row = row
        self.pool = list(names)
        # ⚠️ The *rule* is injected; the state machine below is not. See
        # `BoltedBoard`. `None` is the Week 1-4 board, so every existing caller
        # is unchanged.
        self.rules = BOLTED if rules is None else rules
        # Where an unplaced truss waits. Also injectable, because "behind the
        # arm at x = -1.5" is behind the arm of *this* scene; a house has rows
        # there. `park_spot`'s reasoning is unchanged either way: parked, welds
        # live, invisible.
        self.park = park_spot if park is None else park
        self.placed = {}        # name -> np.array([x, y, z])
        self.zones = {}
        # Harvested. Neither placed nor available, and — critically — not to be
        # moved. See `retire`.
        self.retired = set()
        self.version = 0
        # Every geom belonging to each truss — the fruit, its stem, and the
        # hanger drawn from the support bar down to it. Kept so an unplaced
        # truss can be hidden; see `_show`.
        self._geoms = {}
        for n in self.pool:
            gids = []
            for gname in (f"{n}_geom", f"stem_{n}_geom", f"stem_{n}_drop"):
                try:
                    gids.append(model.geom(gname).id)
                except KeyError:
                    pass        # the hanger is skipped when the drop is ~0
            self._geoms[n] = gids
        self._rgba0 = {n: [model.geom_rgba[g].copy() for g in gids]
                       for n, gids in self._geoms.items()}
        self.park_all()

    def _show(self, name, visible):
        """Hide or reveal a whole truss.

        ⚠️ Alpha only — **nothing about the physics changes.** That matters:
        the parked pool has to keep its welds active so every member stays in
        `CropObstacles`' frozen membership and a fruit placed mid-flight is
        still seen by the running Guard. Deleting or detaching them to tidy the
        view would silently disarm the safety net. Making them invisible costs
        nothing and the arm still cannot reach x = -1.5 anyway.
        """
        for gid, rgba in zip(self._geoms[name], self._rgba0[name]):
            self.model.geom_rgba[gid] = rgba if visible else [0, 0, 0, 0]

    def restore(self):
        """Put every truss back where this object says it is. Places, or parks.

        ⚠️ **The answer to `mj_resetData`.** `mission.reset_park` is a reset
        underneath, so it returns every fruit to the pose the model was
        *compiled* with — and for a hand-placed crop that pose is the parking
        bay, not anywhere a person put anything. This file gets away with it by
        resetting before the crop is built (see `main`); a scene whose harvest
        *starts* with a reset cannot, so it calls this straight afterwards. See
        `farm.mousereach`.

        ⚠️ Welds go back ON for everything, placed or parked, and that is not
        housekeeping. `CropObstacles.__init__` freezes its membership from
        `row.attached()`, and a `Guard` built on it inherits that frozen list —
        so a fruit that is parked with its weld off is invisible to the running
        safety net, and the arm will drive straight through it the moment it is
        placed. Positions are read live; the list is not.
        """
        import mujoco

        for i, n in enumerate(self.pool):
            if n in self.retired:
                continue
            self.row.place(n, self.placed.get(n, self.park(i)))
            self.data.eq_active[self.row.eq_id[n]] = 1
            # Parked but not placed: hidden, so the reserve is not two dozen
            # tomatoes and their stems floating in shot behind the machine.
            self._show(n, n in self.placed)
        mujoco.mj_forward(self.model, self.data)

    def park_all(self):
        """Every truss to its parking spot. `restore` with nothing placed.

        Kept as its own name because that is what the two callers mean —
        `__init__` and `clear`, both of which run with `placed` empty, so the
        two functions do the identical thing and always have.
        """
        self.restore()

    def recolour(self, name, rgba):
        """Repaint one truss's fruit. Ripeness, as a thing you can hand someone.

        ⚠️ **Updates the remembered colour as well as the live one.** `_show`
        restores from `_rgba0`, so repainting only `model.geom_rgba` gives a
        fruit that reverts to its compiled colour the next time it is hidden and
        revealed — which on this scene is every reset. Nothing about the physics
        changes either way; mass, friction and the weld are all untouched.

        ⚠️ And the change is **real**, not a label: `geom_rgba` is what the
        renderer draws and therefore what the deck cameras see, so a fruit
        repainted green is one the HSV classifier has to band green from the
        pixels. That is the whole point of letting somebody choose — the robot
        is not told what they picked.
        """
        rgba = np.asarray(rgba, float)
        for i, gid in enumerate(self._geoms[name]):
            if self.model.geom(gid).name == f"{name}_geom":
                self._rgba0[name][i] = rgba.copy()
                self.model.geom_rgba[gid] = rgba

    def retire(self, name):
        """Take a harvested truss out of the crop **without moving it**.

        ⚠️ Two things must not happen to a tomato that is sitting in a crate,
        and `restore` would do both. It would teleport it back onto the plant,
        because `placed` still says that is where it hangs — and it would set
        `eq_active` back to 1, re-welding a detached fruit to a stem several
        metres away, which is `Row.settle`'s bug with the whole scene between
        the two ends of the constraint. So a harvested truss leaves the crop
        entirely and is left exactly where the arm put it.

        Bumps the version for the same reason placing does: a plan checked
        against a crop that contained this fruit was checked against a world
        that no longer exists. Wrong in the cautious direction, which is the
        only direction to be wrong in here.
        """
        self.placed.pop(name, None)
        self.zones.pop(name, None)
        self.retired.add(name)
        self.version += 1

    def free(self):
        return [n for n in self.pool
                if n not in self.placed and n not in self.retired]

    def clear(self):
        """Un-place everything and park it again. Bumps the version.

        Retired trusses come back — a clear is "start over", and the crates
        emptying is part of starting over.
        """
        self.placed.clear()
        self.zones.clear()
        self.retired.clear()
        self.park_all()
        self.version += 1

    def place(self, y, z, quiet=False):
        """(name, reason). name is None if the placement was refused.

        The board's own coordinates. `place_world` is the general one; this is
        kept because every Week 1-4 caller speaks (y, z) and because on the
        bolted board that *is* the whole coordinate system.
        """
        return self.place_world(self.rules.world(y, z), quiet=quiet)

    def place_world(self, pos, quiet=False):
        """(name, reason) for a world position. The rule decides; this books it.

        ⚠️ Everything below the `check` is scene-independent, and that is the
        point of the split: the pool, the version bump, the weld left active and
        the alpha reveal are the same bookkeeping whether the arm is bolted to a
        floor or riding a rail. Only `self.rules.check` knows which scene it is.
        """
        import mujoco

        pos = np.asarray(pos, float)
        ok, why, zn = self.rules.check(pos, self.placed)
        if not ok:
            if not quiet:
                print(f"    refused: {why}")
            return None, why
        free = self.free()
        if not free:
            return None, "pool exhausted"
        name = free[0]
        self.row.place(name, pos)
        self.data.eq_active[self.row.eq_id[name]] = 1
        self._show(name, True)
        self.row.home[name] = pos.copy()
        self.placed[name] = pos
        self.zones[name] = zn
        self.version += 1
        mujoco.mj_forward(self.model, self.data)
        if not quiet:
            # `why` is populated on an *accepted* tight placement too — that is
            # the point of dropping the spacing rule. Saying "this is 90 mm from
            # p03 and the order is what solves it" is worth more than refusing.
            note = f"  ⚠️ {why}" if why else ""
            print(f"    placed {name} at y{pos[1]:+.3f} z{pos[2]:.3f}  [{zn}]  "
                  f"(crop v{self.version}){note}")
        return name, zn

    def apply(self, layout, quiet=True):
        for _n, y, z in layout:
            self.place(y, z, quiet=quiet)
        return list(self.placed)

    def row_map(self):
        """What the robot believes about where fruit hang — its prior, not truth.

        A real machine has this from the plant spacing and the previous pass. It
        is emphatically not the same thing as knowing where the fruit *is*, which
        is what the camera is for.
        """
        return {n: p.copy() for n, p in self.placed.items()}


# --- what changed between two looks ------------------------------------------

def diff_maps(before, after, moved_mm=30.0):
    """appeared / vanished / moved. The robot noticing, rather than being told.

    This is the honest half of "recognise the change": a re-scan produces a new
    map and the difference against the previous one is what the robot can
    actually claim to have observed.
    """
    a, b = set(before), set(after)
    appeared = sorted(b - a)
    vanished = sorted(a - b)
    moved = sorted(n for n in a & b
                   if float(np.linalg.norm(after[n] - before[n])) * 1000 > moved_mm)
    return appeared, vanished, moved


# --- harvesting whatever was placed ------------------------------------------

def harvest_placed(model, data, row, crop, park_q, speed=None, clearance=None,
                   sensor=None, detector=None, log=None, verbose=False,
                   add_at=None, add_layout=None, seed=0, on_tick=None,
                   on_event=None, deck=None, deck_order=True):
    """Plan and pick every placed fruit. Returns the log rows.

    `sensor`/`detector` given -> perception in the loop ("seen"). Absent -> the
    planner is handed the crop directly ("told"). Told is the mechanism; seen is
    the honest demo. Both replan on a stale crop version.

    `deck` is a `deck_cam.DeckSurvey`, and it changes two things that used to be
    the script's job rather than the robot's:

      * **The order.** Without it, fruit are picked in the order the pool
        happens to name them — which is placement order, which is nothing. With
        it, the survey is fed to `deck_cam.plan_order` and the row is worked in
        the order that measured cheapest, re-planned from scratch whenever the
        crop changes. See the pair sweep in `deck_cam`: on a close pair the
        order is the difference between harvesting one fruit and harvesting two.
      * **Where the wrist camera looks.** Without it, the staging pose for each
        look came out of `crop.placed` — the operator's ground truth, handed to
        the robot. That is the one cheat left in the Week 4 loop and it was a
        load-bearing one: take it away with only an eye-in-hand sensor and the
        arm has to sweep the row to find anything. With a deck survey the look
        pose is a *measurement*, the arm drives straight to it, and no sweep
        happens because none is needed.

    ⚠️ **The survey runs with the arm parked, always.** `deck_cam.parked`
    enforces it; the reason is in `DECK_MOUNT` — staged mid-row the arm eclipses
    a fruit and the survey comes back one short with nothing to say so.

    `deck_order=False` keeps the survey and throws away only the *order*,
    working the row in placement order instead. That combination exists for one
    caller — `week4_order.py` — and it is the difference between an A/B and an
    anecdote. Passing `deck=None` for the control arm looks equivalent and is
    not: it also hands that arm `crop.placed`, i.e. ground truth, so the control
    would be running with better position knowledge than the treatment and the
    comparison would be measuring two things at once. It was written that way
    first, and the first six-layout result was collected before anyone noticed.
    """
    import time as _time

    import mujoco

    from camera import stage
    from carrytrace import CarryTrace
    from incident import Blackbox
    from mission import (BIN_POS, CLEARANCE, GUARD_STOP, Guard, Planner,
                         STAGE_X, park_arm)
    from outcomes import classify
    from reach import DEFAULT_SPEED, Gripper
    from week2_pick import Aborted, anchor_posture, execute, make_reacher
    from week3_perceive import Perception, _plan_perceived

    speed = DEFAULT_SPEED if speed is None else speed
    clearance = CLEARANCE if clearance is None else clearance

    # `on_event(kind, **info)` narrates the run for a viewer. Everything the
    # loop decides is otherwise only ever printed, which is no use to a panel
    # that has to show what the arm is doing *right now*.
    def say(kind, **info):
        if on_event is not None:
            on_event(kind, **info)

    rows = []

    # --- what the deck camera knows, and when it re-learns it -----------------
    # `survey` is name -> measured position; `plan` is the order it implies.
    # Both are rebuilt from one render whenever the crop changes, because a plan
    # made against an older crop is ordering fruit it has never seen.
    survey = {}
    plan = None
    surveyed_at = None

    def resurvey(why):
        """Sweep the deck head, arm parked. Refreshes the map and the order."""
        nonlocal survey, plan, surveyed_at
        from deck_cam import DeckHead, parked, plan_order

        park_arm(model, data, park_q)
        mujoco.mj_forward(model, data)
        if not parked(model, data, park_q):
            raise RuntimeError(
                "deck survey attempted with the arm off park — it would "
                "eclipse fruit and silently return a short row")
        # Only fruit still on the plant are candidates. A harvested tomato is
        # sitting in the crate, still red and still in shot, and the camera has
        # no opinion about that — the robot does, because it put it there. Left
        # in, its blob would associate to its own name and the order would
        # include a fruit that has already been picked.
        standing = [n for n in crop.pool if n in crop.placed and row.attached(n)
                    and n not in {r["fruit"] for r in rows}]

        # ⚠️ The head is a mocap body, so it does not persist across a
        # `park_arm`; it is rebuilt per survey and returned home afterwards.
        # Left pointing at the last scan pose, `--deck-camera deck` would show a
        # view nobody asked for and every later still would be off-axis.
        head = DeckHead(model, data)
        # ⚠️ Announced *before* the sweep, not after. Every other event here
        # reports something that has finished; this one has to arrive first or a
        # live panel sits on the previous phase for the whole three seconds the
        # head is actually moving — which is precisely the part worth watching.
        say("survey_start", head=head, why=why, standing=list(standing))
        # ⚠️ `on_tick` is what makes the sweep visible. Without it the head
        # teleports between the five poses inside one call and a viewer renders
        # nothing until the survey is over. With it the head walks, at its rated
        # slew, and the panel follows it round.
        seen, rep = deck.scan(data, standing, head=head, on_pose=on_tick)
        head.home(data)
        mujoco.mj_forward(model, data)
        if on_tick is not None:
            on_tick()

        survey = {n: s.est for n, s in seen.items()}
        plan = plan_order(seen)
        surveyed_at = crop.version
        missed = [n for n in standing if n not in survey]
        errs = [s.err_mm for s in seen.values()]
        poses, slew = len(rep["poses"]), rep["slew_s"]
        print(f"\n  --- deck survey ({why}) — {poses} head poses, "
              f"{slew:.1f} s of head slew, arm parked ---")
        print(f"  {len(survey)}/{len(standing)} found"
              + (f", missed {', '.join(missed)}" if missed else "")
              + (f" · position error mean {np.mean(errs):.1f} mm, "
                 f"max {np.max(errs):.1f} mm" if errs else ""))
        if missed:
            print(f"  ⚠️ a fruit the deck camera cannot separate is a fruit "
                  f"this run will not pick. It is not lost — it is unseen, and "
                  f"the two are different failures.")
        if plan.steps:
            print(plan.table())
        say("survey", survey=dict(survey), plan=plan, missed=missed,
            poses=rep["poses"], slew_s=slew)
        return plan

    if deck is not None:
        resurvey("start")

    per = None
    if sensor is not None and detector is not None:
        # ⚠️ The row map the wrist camera associates against. With a deck
        # survey it is a *measurement*; without one it is `crop.row_map()`,
        # which is the operator's ground truth wearing a prior's clothes.
        per = Perception(model, sensor, detector,
                         dict(survey) if deck is not None else crop.row_map())

    targets = list(crop.placed)
    print(f"\n{'=' * 78}\n  HARVEST — {len(targets)} fruit placed, "
          f"{'perception' if per else 'told'}, "
          f"{('deck-planned order' if deck_order else 'placement order, deck survey') if deck is not None else 'placement order'}, "
          f"clearance {clearance * 1000:.0f} mm, crop v{crop.version}")

    done = 0
    while True:
        attempted = {r["fruit"] for r in rows}
        remaining = [n for n in crop.placed if row.attached(n)
                     and n not in attempted]
        if not remaining:
            break

        # --- whose turn -------------------------------------------------------
        if deck is None:
            # No survey: the pool's own naming, which is placement order and
            # carries no information about the row at all.
            name = remaining[0]
            queue = list(remaining)
        else:
            if crop.version != surveyed_at:
                resurvey(f"crop moved to v{crop.version}")
            if deck_order:
                queue = [n for n in plan.order if n in remaining]
            else:
                # The control arm: same survey, same positions, same fruit
                # visible — only the sequence thrown away. Restricted to what
                # the survey saw so both arms attempt exactly the same set, and
                # the ones it did not see fall through to `not_detected` below
                # for both. Equal denominators are checked in `week4_order`.
                queue = [n for n in crop.placed
                         if n in remaining and n in survey]
            if not queue:
                # Everything left was invisible to the deck camera — a fused
                # cluster, most likely. Do NOT fall back to placement order:
                # the positions would have to come from `crop.placed`, which is
                # the ground truth the mast exists to stop reading, and a silent
                # fallback would make a survey failure read as a pick failure.
                #
                # ⚠️ Book them as `not_detected` rather than just stopping.
                # `outcomes.py`'s first rule is that every attempt lands in
                # exactly one named bucket, and a fruit that vanishes from the
                # log entirely is worse than one in the wrong bucket: the report
                # would show 6 attempts on 8 placed fruit and a clean *rate*
                # computed over the 6, which flatters it.
                print(f"\n  {len(remaining)} fruit the deck camera never "
                      f"separated: {', '.join(remaining)}")
                print(f"  logged as not_detected — unseen, not lost, and the "
                      f"two are different failures")
                for n in remaining:
                    miss = {"fruit": n, "seen": False, "err_mm": None,
                            "zone": crop.zones.get(n),
                            "n_fruit": len(crop.placed),
                            "crop_version": crop.version, "seed": seed,
                            "ordered_by": "deck", "missed_by": "deck",
                            "outcome": "not_detected",
                            "t_scan": 0.0, "t_plan": 0.0, "t_total": 0.0}
                    rows.append(miss)
                    if log:
                        log.add(**miss)
                    say("refused", fruit=n, why="deck camera never saw it")
                break
            name = queue[0]
        say("select", fruit=name, queue=queue,
            placed=len(crop.placed), done=len(rows))

        # --- mid-run addition -------------------------------------------------
        if add_at is not None and done == add_at and add_layout:
            print(f"\n  --- adding {len(add_layout)} fruit mid-run, after "
                  f"{done} picks ---")
            # ⚠️ `before` is what the robot *believed*, not what was true. With
            # a deck camera that is the last survey; without one there is
            # nothing to compare against but the operator's own map, which
            # makes the diff a tautology — it reports exactly the fruit the
            # script just added, because the script wrote both sides of it.
            #
            # ⚠️ Fruit already harvested are dropped from `before`. They are
            # genuinely no longer on the plant, so a raw diff lists them under
            # `vanished` — and `vanished` is the bucket that is supposed to mean
            # "a tomato fell off and nobody noticed". Letting the routine
            # successes pile up in it is how a real one would go unread.
            before = {n: p for n, p in
                      (survey if deck is not None else crop.row_map()).items()
                      if row.attached(n) and n not in attempted}
            for _n, y, z in add_layout:
                crop.place(y, z)
            print(f"  crop is now v{crop.version} — every plan checked against "
                  f"an older version is void")

            if deck is not None:
                # Look again. This is the robot *noticing*: the diff is between
                # two surveys, so a fruit that appeared is one the camera saw
                # appear, and a fruit it cannot separate simply does not show up.
                resurvey(f"crop moved to v{crop.version}")
                after = dict(survey)
            else:
                after = crop.row_map()
            appeared, vanished, moved = diff_maps(before, after)
            print(f"  re-scan diff: appeared {appeared or '-'}, "
                  f"vanished {vanished or '-'}, moved {moved or '-'}")
            if per is not None:
                # The row map the wrist camera associates against has to track
                # the new survey, or the added fruit cannot be detected at all.
                per.row_map = after
            targets = list(crop.placed)
            add_at = None
            continue

        t0 = _time.perf_counter()
        park_arm(model, data, park_q)
        mujoco.mj_forward(model, data)

        reacher = make_reacher(model, data, speed=speed)
        anchor_posture(reacher, model, data, park_q)
        gripper = Gripper(model, data)
        planner = Planner(model, data, row, lessons=None, clearance=clearance,
                          park_q=park_q, speed=speed)

        rec = {"fruit": name, "seen": True, "err_mm": None,
               "zone": crop.zones.get(name), "n_fruit": len(crop.placed),
               "crop_version": crop.version, "seed": seed,
               "ordered_by": ("deck" if deck_order else "placement")
                             if deck is not None else "placement",
               "surveyed_by": "deck" if deck is not None else "ground_truth",
               "order_index": len(rows)}
        if deck is not None:
            step = next((s for s in plan.steps if s.fruit == name), None)
            if step is not None:
                rec.update({"deck_risk": round(step.risk, 3),
                            "deck_unblocks": round(step.unblocks, 3),
                            "deck_err_mm": round(float(np.linalg.norm(
                                survey[name] - crop.placed[name]) * 1000), 2)})

        # --- look -------------------------------------------------------------
        t_scan = 0.0
        sights = None
        if per is not None:
            say("scan", fruit=name)
            s0 = _time.perf_counter()
            # ⚠️ Where the wrist camera is sent. With a deck survey this is the
            # deck camera's own measurement; without one it is `crop.placed`,
            # i.e. the answer. The whole point of the mast is that this line
            # stops reading ground truth.
            nominal = survey[name] if deck is not None else crop.placed[name]
            stage(model, data, park_q,
                  np.array([STAGE_X, nominal[1], nominal[2]]), row=row,
                  speed=speed, reset="arm")
            sights, _rep = per.look(data)
            t_scan = _time.perf_counter() - s0
            s = sights.get(name)
            if s is None:
                rec.update({"seen": False, "t_scan": t_scan,
                            "outcome": "not_detected"})
                rows.append(rec)
                if log:
                    log.add(**rec)
                print(f"  {name}: NOT DETECTED")
                done += 1
                continue
            say("seen", fruit=name, err_mm=s.err_mm, est=s.est)
            rec["err_mm"] = s.err_mm
            e = s.err
            rec.update({"err_x": float(e[0]), "err_y": float(e[1]),
                        "err_z": float(e[2]), "associated_to": name})

        # --- plan -------------------------------------------------------------
        say("plan", fruit=name, version=crop.version)
        p0 = _time.perf_counter()
        planned_at = crop.version
        mission = (_plan_perceived(planner, row, name, sights,
                                   fallback=survey if deck is not None else None)
                   if sights else planner.plan(name))
        t_plan = _time.perf_counter() - p0

        if not mission.ok:
            why = mission.breaches[0] if mission.breaches else "no route"
            rec.update({"refused_reason": str(why), "outcome": "refused",
                        "t_scan": t_scan, "t_plan": t_plan,
                        "t_total": t_scan + t_plan})
            rows.append(rec)
            if log:
                log.add(**rec)
            say("refused", fruit=name, why=str(why))
            print(f"  {name}: REFUSED — {why}")
            done += 1
            continue

        # ⚠️ The version gate. If the crop changed while this was being planned
        # or staged, the clearance guarantee does not cover the world the arm is
        # about to fly through. Replanning is ~0.15 s against a 28 s pick, so it
        # is always cheaper than being wrong.
        if crop.version != planned_at:
            print(f"  {name}: crop moved v{planned_at} -> v{crop.version} "
                  f"during planning — replanning")
            continue

        # --- fly --------------------------------------------------------------
        trace = CarryTrace(model, data, row, name,
                           Blackbox(model, data, row, name))
        guard = Guard(model, data, row, name,
                      stop=min(GUARD_STOP, 0.4 * clearance))
        guard.armed = False
        say("fly", fruit=name, mission=mission, trace=trace, guard=guard,
            t_plan=t_plan)
        # The trace has to tick every cycle (it is the ejection discriminator),
        # and a recorder wants the same hook. Chained rather than either one
        # replacing the other.
        if on_tick is None:
            tick = trace.tick
        else:
            def tick(_t=None, _tr=trace):
                _tr.tick(_t)
                on_tick(_t)

        try:
            r = execute(mission, reacher, gripper, row, box=trace, guard=guard,
                        on_tick=tick, verbose=verbose)
            aborted = r.get("aborted")
        except Aborted as stop:
            r = {"grasped": False, "broke": False, "in_bin": False,
                 "lost": 0, "disturbed": 0, "seconds": 0.0, "peak_n": 0.0,
                 "clearance": float("nan")}
            aborted = stop.why

        end = data.body(name).xpos.copy()
        rec.update({
            "in_bin": bool(r["in_bin"]), "grasped": bool(r["grasped"]),
            "broke": bool(r["broke"]), "aborted": str(aborted) if aborted else None,
            "lost": int(r["lost"]), "disturbed": int(r["disturbed"]),
            "clearance_min_mm": (float(r["clearance"]) * 1000
                                 if np.isfinite(r["clearance"]) else None),
            "peak_n": float(r["peak_n"]),
            # The discriminator between a drop and an ejection. See outcomes.py.
            "peak_held_ms": trace.peak_held(),
            "fruit_end_xyz": [float(v) for v in end],
            "t_scan": t_scan, "t_plan": t_plan, "t_fly": float(r["seconds"]),
            "t_total": t_scan + t_plan + float(r["seconds"]),
        })
        rec["clean"] = bool(r["in_bin"] and not r["lost"] and not r["disturbed"]
                            and not aborted)
        rec["outcome"] = classify(rec)
        rows.append(rec)
        if log:
            log.add(**rec)

        say("result", fruit=name, rec=rec)
        flag = "" if rec["outcome"] == "clean" else f"  <-- {rec['outcome']}"
        print(f"  {name}: {rec['outcome']:<14} "
              f"grasp={rec['grasped']} stem={rec['broke']} "
              f"crate={rec['in_bin']} held={rec['peak_held_ms']:.2f} m/s "
              f"{rec['t_total']:.1f}s{flag}")
        done += 1

    return rows


# --- interactive placement ---------------------------------------------------

def add_place_board(spec):
    """The board you click on, sized to exactly where the arm can work.

    This IS the visual answer to "what is the range of the arm" — the green
    rectangle is the guaranteed zone from the 31x24 grid sweep, so anywhere on
    it is reachable by construction and there is nothing to guess.

    Sized and positioned to `GUARANTEED_HALF_Y` / `GUARANTEED_Z`, sat just
    behind the fruit plane so the tomatoes hang in front of it rather than
    inside it. It is a **body**, not a bare worldbody geom, because MuJoCo's
    viewer reports a click as "you selected body N at this local point" and the
    world body is never reported — the same reason `add_plant_panel` is a body.

    `contype=0`: it is a click target, not an obstacle. Giving the planner a
    wall to route around would change every clearance number in the repo.
    """
    import mujoco

    # Two boards, because the measurement found two regions and pretending
    # otherwise is what caused the original bug. The bright green core is where
    # every probe picked clean; the dim amber surround is where most did.
    z_lo, z_hi = MARGINAL_Z
    body = spec.worldbody.add_body(name=PLACE_BOARD,
                                   pos=[ROW_X + 0.10, 0.0, (z_lo + z_hi) / 2])
    body.add_geom(
        name=f"{PLACE_BOARD}_face", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, MARGINAL_HALF_Y, (z_hi - z_lo) / 2],
        rgba=[0.55, 0.42, 0.16, 0.40], contype=0, conaffinity=0,
    )
    g_lo, g_hi = GUARANTEED_Z
    body.add_geom(
        name=f"{PLACE_BOARD}_core", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.010, GUARANTEED_HALF_Y, (g_hi - g_lo) / 2],
        pos=[0.0, 0.0, (g_lo + g_hi) / 2 - (z_lo + z_hi) / 2],
        rgba=[0.20, 0.50, 0.25, 0.60], contype=0, conaffinity=0,
    )
    return body


def board_click_to_yz(model, data, localpos):
    """A click on the board -> the (y, z) a fruit should hang at.

    `localpos` is in the board body's own frame, so y comes straight across and
    z is measured from the board's centre height.
    """
    bid = model.body(PLACE_BOARD).id
    centre_z = float(data.xpos[bid][2])
    return float(localpos[1]), centre_z + float(localpos[2])


def place_visually(model, data, crop):
    """Click the board to place fruit. Returns (start_run, viewer).

    The click mechanism is `week1_mousereach.py`'s, unchanged: the passive
    viewer reports a double-click as `perturb.select` (a body id) plus
    `perturb.localpos`, so no GLFW callback is needed and it works under the
    same viewer the harvest is watched in.

    ⚠️ Prime `last` from whatever the viewer already reports before the loop.
    Starting it at None makes the viewer's initial state read as a brand-new
    click and drops a tomato the moment the window opens.
    """
    import time
    import warnings

    import mujoco
    import mujoco.viewer

    warnings.filterwarnings("ignore", module="glfw")
    board_id = model.body(PLACE_BOARD).id
    go = [False]
    quit_ = [False]

    def on_key(code):
        # GLFW keycodes: space=32, and letters arrive uppercase.
        if code == 32:                      # space — start harvesting
            go[0] = True
        elif code == ord("A"):              # fill the rest automatically
            for _n, y, z in auto_layout(MAX_FRUIT - len(crop.placed),
                                        seed=len(crop.placed) + 1):
                crop.place(y, z)
        elif code == ord("C"):              # clear and start over
            crop.clear()
            print("    cleared — the board is empty again")
        elif code == ord("Q"):
            quit_[0] = True

    # ⚠️ `key_callback` is a launch_passive argument, not a method on the
    # returned Handle — `Handle` has no `set_key_callback` at all. Setting it
    # after the fact behind a hasattr guard silently does nothing, which is how
    # the first version of this shipped with no working SPACE key and no way to
    # start the run.
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)

    print(f"\n  {'=' * 70}")
    print(f"  PLACE THE FRUIT — the green board is exactly where the arm can work")
    print(f"    double-click the board   put a tomato there")
    print(f"    A                        auto-fill the rest ({MAX_FRUIT} max)")
    print(f"    C                        clear them all")
    print(f"    SPACE                    start harvesting")
    print(f"    Q / close the window     quit")
    print(f"  {'=' * 70}\n")

    last = None
    if viewer.perturb is not None:
        last = (int(viewer.perturb.select),
                tuple(np.round(viewer.perturb.localpos, 5)))

    while viewer.is_running() and not go[0] and not quit_[0]:
        with viewer.lock():
            sel = int(viewer.perturb.select)
            localpos = np.array(viewer.perturb.localpos)
        if sel == board_id:
            click = (sel, tuple(np.round(localpos, 5)))
            if click != last:
                last = click
                y, z = board_click_to_yz(model, data, localpos)
                crop.place(y, z)
        elif sel != board_id:
            last = (sel, tuple(np.round(localpos, 5)))

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1 / 120)

    if quit_[0] or not viewer.is_running():
        return False, viewer
    if not crop.placed:
        print("  nothing placed — nothing to harvest")
        return False, viewer
    print(f"\n  starting harvest of {len(crop.placed)} fruit\n")
    return True, viewer


def place_interactively(crop):
    """Type coordinates to place fruit. The headless sibling of the click UI.

    Only used when there is no window — `--no-window` or a machine with no
    display. Placing blind is the worse experience by a long way, which is why
    the click board exists and is the default.
    """
    # ⚠️ These three lines used to name DOME_HALF_Y, DOME_Z, DOME_APEX_HALF_Y
    # and DOME_APEX_Z, none of which exist in this module — the envelope was
    # re-measured into GUARANTEED_*/MARGINAL_* and this print was left behind.
    # It is the first statement in the function, so the no-window placement path
    # raised NameError before reading a single coordinate. Nothing caught it
    # because every documented command passes --windowed, --grid or --layout.
    print(f"\n  Place up to {MAX_FRUIT} fruit. The arm works:")
    print(f"    guaranteed   |y| <= {GUARANTEED_HALF_Y}, "
          f"z {GUARANTEED_Z[0]}-{GUARANTEED_Z[1]}   (10/10 clean)")
    print(f"    marginal     |y| <= {MARGINAL_HALF_Y}, "
          f"z {MARGINAL_Z[0]}-{MARGINAL_Z[1]}   (18/21 clean)")
    print(f"    no minimum spacing — put them as close as you like, down to "
          f"{TOUCHING * 1000:.0f} mm centre to centre")
    print("\n  'y z' to place · 'auto N' to fill · 'list' · 'done'\n")
    while True:
        try:
            line = input("  place> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("done", "go", "q"):
            break
        if line == "list":
            for n, p in crop.placed.items():
                print(f"    {n}  y{p[1]:+.3f} z{p[2]:.3f}  [{crop.zones[n]}]")
            continue
        if line.startswith("auto"):
            bits = line.split()
            n = int(bits[1]) if len(bits) > 1 else MAX_FRUIT
            # ⚠️ A fresh random seed every time. Deriving it from the number
            # already placed made "auto" deterministic — the same arrangement
            # on every press — which is useless for judging repeatability,
            # because repeating one layout only measures the simulator.
            for _nm, y, z in auto_layout(n - len(crop.placed),
                                         seed=random_seed()):
                crop.place(y, z)
            continue
        try:
            y, z = (float(v) for v in line.replace(",", " ").split())
        except ValueError:
            print("    expected: y z   (two numbers, metres)")
            continue
        crop.place(y, z)
    return list(crop.placed)


def watch(model, data, run, viewer=None):
    """Run a harvest with a live viewer open, at wall-clock speed.

    Same shape as `week3_perceive._watch` and `week2_pick.run_windowed`, and it
    inherits the same two hard-won details:

    ⚠️ **The tick has to pace itself.** Physics runs far faster than real time,
    so syncing the viewer without sleeping produces an unwatchable blur. The
    clock advances one control period per tick and the loop sleeps the balance.

    ⚠️ **Hold the window open at the end.** `exit_without_teardown` is
    `os._exit`, so returning from here would kill the process mid-frame and the
    window would vanish the instant the last pick lands — which reads as a
    crash rather than a finish. Physics is deliberately *not* stepped while it
    is held: the run is over, and stepping on with nothing commanding the
    servos would let the arm sag while you watch.
    """
    import contextlib
    import time
    import warnings

    import mujoco.viewer

    from reach import CTRL_DT

    warnings.filterwarnings("ignore", module="glfw")
    rows = []
    # Reuse the placement viewer when there is one, so the window never blinks
    # between placing and picking. nullcontext keeps the `with` shape either way.
    ctx = (contextlib.nullcontext(viewer) if viewer is not None
           else mujoco.viewer.launch_passive(model, data))
    with ctx as viewer:
        clock = [time.perf_counter()]

        def tick(_t=None):
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            clock[0] += CTRL_DT
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

        try:
            rows = run(tick)
            print("\n  run finished — the window stays open, close it to quit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\n  stopped early — reporting the picks that finished")
    return rows


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("what to place (default: interactive)")
    src.add_argument("--grid", type=int, default=None, metavar="N",
                     help=f"auto-place N fruit (max {MAX_FRUIT})")
    src.add_argument("--layout", default=None, help="place from a layout file")
    src.add_argument("--save", default=None, help="write the layout out")
    ap.add_argument("--add-at", type=int, default=None, metavar="K",
                    help="after K picks, add more fruit mid-run and make the "
                         "planner notice")
    ap.add_argument("--add", type=int, default=3, metavar="N",
                    help="how many to add with --add-at (default 3)")
    ap.add_argument("--seen", action="store_true",
                    help="perception in the loop instead of telling the planner")
    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"])
    ap.add_argument("--no-deck", action="store_true",
                    help="switch the chassis camera off and pick in placement "
                         "order, which is what this did before the mast "
                         "existed. Kept so the ordering can be A/B'd.")
    # Interactive runs default to 0.4 of rated joint speed. Watching a pick at
    # 0.15 is 30 s of staring; the measurement tools (week4_run, week4_snap,
    # week4_envelope) keep reach.DEFAULT_SPEED so their numbers stay comparable
    # with every earlier week.
    ap.add_argument("--speed", type=float, default=WATCH_SPEED,
                    help=f"fraction of rated joint speed (default "
                         f"{WATCH_SPEED}; use 0.15 to match the measurements)")
    ap.add_argument("--clearance", type=float, default=None, metavar="MM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="append JSONL rows here")
    ap.add_argument("--no-harvest", action="store_true",
                    help="place and report only — do not pick")
    ap.add_argument("--out", default=None, metavar="MP4",
                    help="record the harvest. Named after the script per the "
                         "repo convention, so a video always says which code "
                         "made it.")
    ap.add_argument("--camera", default="row",
                    choices=["row", "aisle", "wide"],
                    help="scene camera for --out (default row)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--windowed", action="store_true",
                    help="open a live viewer and watch it pick, at real speed")
    args = ap.parse_args()

    # ⚠️ Only force EGL when nothing is being watched. Setting it
    # unconditionally is exactly what made this file headless-only in its first
    # version — EGL has no window to open, so every documented command ran
    # blind and there was no way to actually see a pick. A live viewer needs
    # GLFW; `mujoco.Renderer` is happy under either, so --out still works.
    if not args.windowed:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    from camera import SensorCamera
    from greenhouse import build_scene
    from mission import CLEARANCE, park_posture, reset_park
    from outcomes import report
    from picklog import PickLog, throughput
    from plant_row import Row

    clearance = CLEARANCE if args.clearance is None else args.clearance / 1000

    pool = pool_trusses()
    interactive = not (args.layout or args.grid)
    # The click board only exists when it is going to be clicked. It is
    # contype=0 scenery, so it changes no dynamics either way, but leaving it
    # out of batch runs keeps those scenes byte-identical to the campaign's.
    model = build_scene(wrist_cam=args.seen, deck_cam=not args.no_deck,
                        trusses=pool,
                        place_board=interactive and args.windowed)
    data = mujoco.MjData(model)
    names = [n for n, _, _ in pool]
    # `homes` supplied, because these names are not in plant_row.TRUSSES and
    # fruit_home() would raise. Overwritten per fruit as it is placed.
    row = Row(model, data, names=names,
              homes={n: park_spot(i) for i, n in enumerate(names)})
    park_q = park_posture(model)
    reset_park(model, data, park_q)      # before the crop is built, not after
    row.reset()
    mujoco.mj_forward(model, data)

    crop = Crop(model, data, row, names)

    viewer = None
    if args.layout:
        layout = load_layout(args.layout)
        crop.apply(layout)
        print(f"  loaded {len(crop.placed)} fruit from {args.layout}")
    elif args.grid:
        crop.apply(auto_layout(min(args.grid, MAX_FRUIT), seed=args.seed))
        print(f"  auto-placed {len(crop.placed)} fruit (seed {args.seed})")
    elif args.windowed:
        # The viewer is opened by `place_visually` (it owns the key callback)
        # and handed back, so placing and picking happen in one continuous
        # window. Opening a second viewer later would make the board vanish and
        # the scene appear to restart.
        started, viewer = place_visually(model, data, crop)
        if not started:
            viewer.close()
            return
    else:
        print("  ⚠️ no window: placing blind. Add --windowed to click a board "
              "that shows exactly where the arm can reach.")
        place_interactively(crop)

    if not crop.placed:
        raise SystemExit("nothing placed")

    zones = {}
    for n in crop.placed:
        zones[crop.zones[n]] = zones.get(crop.zones[n], 0) + 1
    spacing = []
    pts = list(crop.placed.values())
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            spacing.append(float(np.hypot(pts[i][1] - pts[j][1],
                                          pts[i][2] - pts[j][2])))
    print(f"\n  {len(crop.placed)} fruit · zones {zones} · closest pair "
          f"{min(spacing) * 1000:.0f} mm" if spacing else "")

    if args.save:
        save_layout(args.save, [(n, float(p[1]), float(p[2]))
                                for n, p in crop.placed.items()],
                    meta={"seed": args.seed})

    if args.no_harvest:
        return

    from week3_perceive import build_detector

    sensor = detector = deck = None
    if args.seen:
        sensor = SensorCamera(model, camera="wrist")
        detector = build_detector(args.detector)
    if not args.no_deck:
        from deck_cam import DeckSurvey

        # ⚠️ Same detector class as the wrist camera, deliberately. The claim is
        # that the deck camera is the *same pipeline at a different range*; give
        # it its own detector and the comparison between what the two see stops
        # meaning anything.
        deck = DeckSurvey(model, detector=build_detector(args.detector))

    add_layout = None
    if args.add_at is not None:
        # Generated against the already-placed fruit, so the additions land
        # where the arm can work rather than on top of the crop.
        add_layout = []
        rng = np.random.default_rng(args.seed + 99)
        tries = 0
        while len(add_layout) < args.add and tries < 500:
            tries += 1
            y = float(rng.uniform(-GUARANTEED_HALF_Y, GUARANTEED_HALF_Y))
            z = float(rng.uniform(*GUARANTEED_Z))
            probe = dict(crop.placed)
            for _n, ay, az in add_layout:
                probe[_n] = np.array([ROW_X, ay, az])
            ok, _why, _zn = check(y, z, probe)
            if ok:
                add_layout.append((f"add{len(add_layout)}", y, z))

    log = PickLog(args.log, meta={
        "layout": args.layout or (f"grid{args.grid}" if args.grid else "manual"),
        "detector": args.detector if args.seen else "told",
        "speed": args.speed, "clearance_mm": clearance * 1000,
        "seed": args.seed}) if args.log else None

    kw = dict(speed=args.speed, clearance=clearance, sensor=sensor,
              detector=detector, log=log, verbose=args.verbose,
              add_at=args.add_at, add_layout=add_layout, seed=args.seed,
              deck=deck)

    if args.windowed and args.out:
        raise SystemExit("--windowed and --out are mutually exclusive: "
                         "recording needs offscreen EGL, a live viewer needs "
                         "GLFW. Run them separately.")

    if args.windowed:
        rows = watch(model, data,
                     lambda tick: harvest_placed(model, data, row, crop,
                                                 park_q, on_tick=tick, **kw),
                     viewer=viewer)
    elif args.out:
        import cv2

        from reach import CTRL_DT

        fps, width, height = 30, 1280, 960
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {args.out}")
        renderer = mujoco.Renderer(model, height=height, width=width)
        frame_every = max(1, int(round((1 / fps) / CTRL_DT)))
        cycles = [0]

        def rec(_t=None):
            cycles[0] += 1
            if cycles[0] % frame_every == 0:
                renderer.update_scene(data, camera=args.camera)
                writer.write(cv2.cvtColor(renderer.render(),
                                          cv2.COLOR_RGB2BGR))

        try:
            rows = harvest_placed(model, data, row, crop, park_q,
                                  on_tick=rec, **kw)
        finally:
            renderer.close()
            writer.release()
            print(f"\n  wrote {args.out}")
    else:
        rows = harvest_placed(model, data, row, crop, park_q, **kw)

    report(rows, f"{len(crop.placed)} placed fruit")
    t = throughput(rows)
    if t:
        print(f"\n  cycle {t['mean_s']:.1f} s mean "
              f"({t['min_s']:.1f}-{t['max_s']:.1f}) · {t['picks_hr']:.0f} picks/hr")
        print(f"  {t['kg_hr']:.1f} kg/hr single arm  ->  "
              f"{t['kg_week']:.0f} kg/week/robot at 24/7")
    if log:
        log.close()
    # Both sensors own a GL context. `SensorCamera.close` exists because
    # headless depth rendering raises EGLError out of Renderer.__del__ at
    # process exit on this machine; closing explicitly keeps it off the common
    # path. See the note in camera.SensorCamera.close.
    for s in (deck, sensor):
        if s is not None:
            s.close()


if __name__ == "__main__":
    main()
