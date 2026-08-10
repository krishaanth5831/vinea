#!/usr/bin/env python3
"""Given a map of the house, decide where to stop and what to take at each.

`deck_cam.plan_order` answers "which of these fruit first" for a row the arm is
already parked in front of. That is the inner problem. This is the outer one:
the fruit are spread over eight metres, the arm can reach about a metre of that
from any one place, and the trolley has to be somewhere. So a shift is

    a sequence of stops, and at each stop a sequence of picks

and the two interact — moving a stop 200 mm changes which fruit are reachable
from it, which changes what the pick order at that stop even is.

--- how it is decided, and which parts are actually hard --------------------

**Stops: a greedy interval cover, and that is provably right.** Every fruit is
reachable from a contiguous window of trolley positions `[y - REACH, y + REACH]`;
covering a set of points with fewest fixed-width intervals is solved by taking
the lowest uncovered point, placing an interval to just contain it, and
sweeping on. There is no cleverness to add here and adding some would be worse.

**Order of stops: drive along the row and do not come back.** With one aisle and
a 1-DOF base, any route that revisits a stretch pays for it twice. A monotone
sweep is optimal for the same reason a bookshelf is sorted once.

⚠️ **What is *not* obvious is which stop a fruit belongs to when two can reach
it**, and the choice matters: assigning it to the wrong one can leave a stop
with a single fruit and force a stop that would otherwise be unnecessary. This
assigns each fruit to the stop whose centre it is nearest, then drops any stop
left empty — which is a heuristic, and is marked as one.

**Picks within a stop: `deck_cam.plan_order`, unchanged.** That is the whole
point of keeping it — the ordering machinery, its cost model and its exact
solver all work on "a set of fruit and where the arm is", which is exactly what
a stop is. ⚠️ It carries its known limitation with it: `deck_cam._pair_risk` is
symmetric, so it finds no ordering signal on close pairs. See the note there.

    ./.venv/bin/python simulation/mujoco/farm/route.py            # plan one
    ./.venv/bin/python simulation/mujoco/farm/route.py --compare  # vs naive
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import house, trolley  # noqa: E402

# How much of the row the arm can work from a standing trolley, either side of
# the trolley's own y. `crop.Y_REACH` is the placement envelope's half-width;
# this is deliberately a little tighter, because a fruit at the very edge of the
# envelope is one the planner refuses about as often as it takes.
REACH_Y = 0.48


@dataclass
class Stop:
    """One place the trolley parks, and what it takes while it is there."""

    y: float
    fruit: list = field(default_factory=list)      # Sightings, in pick order
    plan: object = None                            # the deck_cam OrderedPlan

    @property
    def n(self):
        return len(self.fruit)


@dataclass
class Route:
    """A whole shift for one aisle and one arm."""

    stops: list = field(default_factory=list)
    aisle: int = 0
    arm: str = "a"
    row: int = 0
    skipped: list = field(default_factory=list)    # ripe but out of reach

    @property
    def n_fruit(self):
        return sum(s.n for s in self.stops)

    def drive_m(self, start=None):
        """Metres the trolley covers, including getting to the first stop."""
        if not self.stops:
            return 0.0
        ys = [s.y for s in self.stops]
        first = ys[0] if start is None else start
        return abs(ys[0] - first) + sum(abs(b - a) for a, b in zip(ys, ys[1:]))

    def table(self, indent="  "):
        out = [f"{indent}{len(self.stops)} stops, {self.n_fruit} ripe fruit, "
               f"{self.drive_m():.1f} m of driving"]
        out.append(f"{indent}{'#':>2} {'y':>7} {'take':>5}  order")
        for i, s in enumerate(self.stops, 1):
            names = " ".join(f"{f.stage[0]}@{f.pos[1]:+.2f}" for f in s.fruit)
            out.append(f"{indent}{i:2d} {s.y:>7.2f} {s.n:>5}  {names}")
        if self.skipped:
            out.append(f"{indent}⚠️ {len(self.skipped)} ripe fruit not "
                       f"reachable from this aisle and left on the plant")
        return "\n".join(out)


def cover(ys, reach=REACH_Y, limits=None):
    """Fewest trolley positions covering every y in `ys`. Greedy, and optimal.

    Take the lowest uncovered point, put a stop `reach` above it — the furthest
    place that still reaches it — and sweep on. Every stop is forced by the
    point that created it, so no cover can use fewer.
    """
    lo, hi = trolley.y_limits() if limits is None else limits
    out = []
    rest = sorted(float(y) for y in ys)
    i = 0
    while i < len(rest):
        y = float(np.clip(rest[i] + reach, lo, hi))
        out.append(y)
        while i < len(rest) and rest[i] <= y + reach + 1e-9:
            i += 1
    return out


def plan(house_map, aisle=0, arm="a", start=None, reach=REACH_Y, order=True):
    """Turn a map into a route. Only ripe fruit on the arm's own row.

    ⚠️ **Stops are trolley positions, and the arm is not at the trolley's y.**
    The two arms are staggered `trolley.ARM_STAGGER` along the row, so arm a
    stands 250 mm ahead of the deck centre and arm b 250 mm behind it. A stop
    computed as though the arm were at the trolley's own y parks arm b half a
    metre past every fruit it was sent to. So the cover is solved in the *arm's*
    frame — fruit y minus the arm's own offset — and the resulting stops are
    trolley y, which is what `Drive.drive_to` takes.
    """
    from deck_cam import plan_order

    left, right = house.serves(aisle)
    row = right if arm == "a" else left
    arm_dy = trolley.ARM_Y[arm]

    ripe = [s for s in house_map.sightings if s.ripe and s.row == row]
    other = [s for s in house_map.sightings if s.ripe and s.row != row]

    ys = cover([s.pos[1] - arm_dy for s in ripe], reach=reach)
    route = Route(aisle=aisle, arm=arm, row=row, skipped=other)

    # ⚠️ Nearest-stop assignment, and it is a heuristic rather than a solve. A
    # fruit reachable from two stops has to go to one of them, and picking the
    # nearer centre is the obvious choice — but it is not always the best one,
    # because emptying a stop entirely is worth more than balancing two.
    # Stops that end up empty are dropped below, which recovers the common case.
    for s in ripe:
        k = int(np.argmin([abs(s.pos[1] - arm_dy - y) for y in ys])) if ys else -1
        if k >= 0 and abs(s.pos[1] - arm_dy - ys[k]) <= reach + 1e-6:
            while len(route.stops) <= k:
                route.stops.append(Stop(y=ys[len(route.stops)]))
            route.stops[k].fruit.append(s)
    route.stops = [st for st in route.stops if st.n]
    route.stops.sort(key=lambda st: st.y)

    if order:
        for st in route.stops:
            # The arm's park position at this stop, in world coordinates —
            # which is what `plan_order` measures travel from.
            #
            # ⚠️ **`ARM_X[arm]`, not `+ARM_OFFSET`.** This was hardcoded to the
            # positive offset, so a route planned for arm b measured its travel
            # costs from arm *a*'s mount — 400 mm across the aisle, on the wrong
            # side of the crop. It only ever affected the pick *order* (the cost
            # model's travel term), never which fruit were taken, which is why it
            # survived: a wrong order still harvests, just not in the best
            # sequence. Now it asks the arm the route is for.
            base = np.array([house.aisle_x(aisle) + trolley.ARM_X[arm],
                             st.y + arm_dy, trolley.DECK_Z])
            survey = {f"s{i}": f.pos for i, f in enumerate(st.fruit)}
            st.plan = plan_order(survey, start=base)
            by_name = {f"s{i}": f for i, f in enumerate(st.fruit)}
            st.fruit = [by_name[n] for n in st.plan.order]
    return route


def naive(house_map, aisle=0, arm="a", reach=REACH_Y):
    """The control: stop at every fruit, in the order they were mapped.

    What a machine with no routing does — drive to a fruit, take it, drive to
    the next. It is the baseline the stop-cover has to beat, and it beats it on
    driving distance by construction, which is why `--compare` reports both that
    *and* the fruit count: a route that drives less and takes less is not better.
    """
    left, right = house.serves(aisle)
    row = right if arm == "a" else left
    ripe = [s for s in house_map.sightings if s.ripe and s.row == row]
    route = Route(aisle=aisle, arm=arm, row=row)
    for s in ripe:
        # Trolley y that puts *this arm* at the fruit. See `plan`.
        route.stops.append(Stop(y=float(np.clip(s.pos[1] - trolley.ARM_Y[arm],
                                                *trolley.y_limits())),
                                fruit=[s]))
    return route


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare", action="store_true",
                    help="stop-cover against one-stop-per-fruit")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--truth", action="store_true",
                    help="route the real crop instead of a scouted map — "
                         "isolates the router from the detector")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import mujoco

    from farm import armframe
    from farm.scout import HouseMap, Scout, Sighting
    from mission import reset_park

    seed = fcrop.resolve_seed(args.seed)
    trusses = fcrop.spawn(n_per_row=args.n, seed=seed)

    if args.truth:
        # ⚠️ A map built from the operator's own answer. Useful for exactly one
        # thing: telling a routing problem apart from a perception problem.
        house_map = HouseMap(
            sightings=[Sighting(pos=t.pos, stage=t.stage, hue=float("nan"),
                                row=t.row, truth=t) for t in trusses],
            aisle=args.aisle)
        print("\n  ⚠️ routing GROUND TRUTH, not a scouted map")
    else:
        model = trolley.build(aisle=args.aisle, arms=("a",), trusses=trusses,
                              deck_cam=True, seed=seed)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        park_q = armframe.park_posture(model, data)
        reset_park(model, data, park_q)
        mujoco.mj_forward(model, data)
        scout = Scout(model, data, aisle=args.aisle)
        try:
            house_map = scout.run(trolley.Drive(model, data), verbose=False)
        finally:
            scout.close()
        print(f"\n{house_map.table()}")

    r = plan(house_map, aisle=args.aisle, arm="a")
    left, right = house.serves(args.aisle)
    print(f"\n  --- the route, aisle a{args.aisle}, arm A on row r{right} ---")
    print(r.table())

    if args.compare:
        nv = naive(house_map, aisle=args.aisle, arm="a")
        print(f"\n  --- against one stop per fruit ---")
        print(f"  {'route':<22} {'stops':>6} {'fruit':>6} {'drive m':>9}")
        for name, x in (("stop cover", r), ("one stop per fruit", nv)):
            print(f"  {name:<22} {len(x.stops):>6} {x.n_fruit:>6} "
                  f"{x.drive_m():>9.1f}")
        if nv.n_fruit:
            print(f"\n  same {r.n_fruit} fruit in {len(r.stops)} stops rather "
                  f"than {len(nv.stops)}, and "
                  f"{nv.drive_m() - r.drive_m():+.1f} m less driving.")
            print(f"  ⚠️ Driving is not the expensive part — at "
                  f"{trolley.DRIVE_SPEED} m/s that is "
                  f"{(nv.drive_m() - r.drive_m()) / trolley.DRIVE_SPEED:.0f} s "
                  f"against a pick cycle of ~20 s.")
            print(f"  The stops themselves are what cost: each one is a park, a "
                  f"settle and a survey.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
