#!/usr/bin/env python3
"""A whole house of tomatoes, at random, some of them ripe.

Weeks 1-4 hung five to fifteen fruit on one row, all of them the same red, all
of them targets. Neither of those survives contact with a house:

  * **A row is longer than a reach.** Fruit are spread over the full 8 m bay, so
    which ones a trolley can even attempt depends on where it is standing, and
    that is the whole routing problem.
  * **Most tomatoes are not ready.** A picker walks past far more fruit than
    they take. A machine that cannot tell the difference either strips the
    plant or has to be told, and both are useless.

--- ripeness, and being straight about what it is here ----------------------

⚠️ `ripeness.py` refuses to run on rendered frames, and it is right to: the
Week 1-4 tomatoes are one `rgba` value, so a "ripeness result" measured on them
would be a measurement of a constant in a scene file. That objection is about
*uniform* fruit, and it is answered by making the fruit not uniform — this
module hangs them across the real horticultural colour stages, green through
breaker and turning to red, using the same hue bands `ripeness.BANDS` defines.

So there is genuine signal here and a classifier can be scored against a known
answer. What that does **not** do is make it a ripeness model for real fruit:
the scene sets the hue directly, with no calyx, no shoulder, no ribbing, no
bloom and no lighting variation, so a classifier that reads it perfectly has
demonstrated the *pipeline*, not the perception. The honest claim is "the robot
can carry a ripeness decision through mapping, routing and picking", and the
number that matters is how many ripe fruit reached the crate — not the
classifier's accuracy, which is a measurement of `STAGES` below.

    ./.venv/bin/python simulation/mujoco/farm/crop.py            # a window
    ./.venv/bin/python simulation/mujoco/farm/crop.py --shot     # stills
    ./.venv/bin/python simulation/mujoco/farm/crop.py --stats    # what spawned
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import house, trolley  # noqa: E402
from plant_row import (FRUIT_MASS, FRUIT_R, PEDUNCLE_SOLREF,  # noqa: E402
                       SNAP_N, STEM_LEN, STEM_R)

# --- the colour stages -------------------------------------------------------
#
# USDA runs green -> breaker -> turning -> pink -> light red -> red. `ripeness.
# BANDS` collapses that to four, which is as much as mean hue can honestly
# separate, so these are those four with an RGB the renderer can show and a
# decision attached.
#
# ⚠️ **`pick` is the ground truth the whole run is scored against.** Only `red`
# is taken. A grower harvesting for the Dutch market picks at light red to red
# and leaves turning fruit for the next pass — a machine that takes turning
# fruit is not "keen", it is downgrading the crop.
STAGES = [
    #  name        rgba                          hue    pick
    ("green",    (0.34, 0.55, 0.22, 1.0),        55,   False),
    ("breaker",  (0.72, 0.70, 0.24, 1.0),        28,   False),
    ("turning",  (0.88, 0.48, 0.14, 1.0),        14,   False),
    ("red",      (0.85, 0.16, 0.12, 1.0),         3,   True),
]
STAGE_BY_NAME = {n: (rgba, hue, pick) for n, rgba, hue, pick in STAGES}

# How a truss's fruit are distributed across those stages. Weighted toward the
# unripe end because that is what a plant looks like between passes: a grower
# harvests the same row every 2-3 days and takes a fraction each time.
STAGE_WEIGHTS = (0.34, 0.22, 0.20, 0.24)

# Where fruit may hang, in the arm's own frame — i.e. relative to the deck the
# arm is bolted to. ⚠️ These are exactly `week4_place.MARGINAL_*`, and they are
# imported-by-value rather than re-derived: they are the band `week4_envelope`
# measured cell by cell, one full pick each, and a house that hangs fruit
# outside it is a house whose failures are placement errors wearing the costume
# of routing errors.
Z_LOCAL = (0.42, 0.72)
Y_REACH = 0.55            # +/- from the arm, along the row


def fruit_z(z_local):
    """Arm-frame height -> world height. See `trolley.DECK_Z`."""
    return trolley.DECK_Z + z_local


@dataclass
class Truss:
    """One tomato as the *operator* knows it. The robot has to work it out."""

    name: str
    row: int
    x: float
    y: float
    z: float
    stage: str

    @property
    def ripe(self):
        return STAGE_BY_NAME[self.stage][2]

    @property
    def rgba(self):
        return STAGE_BY_NAME[self.stage][0]

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


def spawn(n_per_row=14, rows=None, seed=None, min_sep=0.075):
    """Hang `n_per_row` fruit on each row, at random, with random ripeness.

    `seed=None` means a **different house every time it is opened**, which is
    the point: a layout the planner has already seen is a layout it can get
    lucky on. Pass a seed to reproduce one.

    ⚠️ `min_sep` is 75 mm, just past touching (`week4_place.TOUCHING` is 70 mm).
    Closer than that and two spheres interpenetrate, which MuJoCo resolves by
    firing them apart — a crop that explodes on the first step. It is *not* the
    old 200 mm placement rule, which was removed in Week 4 and stays removed:
    a close pair is a pick-order problem, and now also a routing one.
    """
    rng = np.random.default_rng(seed)
    rows = list(range(house.N_ROWS)) if rows is None else list(rows)
    stages = [n for n, _, _, _ in STAGES]

    out, k = [], 0
    for r in rows:
        placed = []
        tries = 0
        while len(placed) < n_per_row and tries < 4000:
            tries += 1
            y = float(rng.uniform(-house.HOUSE_HALF_Y + 0.4,
                                  house.HOUSE_HALF_Y - 0.4))
            z = float(rng.uniform(*Z_LOCAL))
            if any(np.hypot(y - py, z - pz) < min_sep for py, pz in placed):
                continue
            placed.append((y, z))
            out.append(Truss(name=f"t{k:03d}", row=r, x=house.row_x(r),
                             y=y, z=fruit_z(z),
                             stage=str(rng.choice(stages, p=STAGE_WEIGHTS))))
            k += 1
    return out


def add_trusses(spec, trusses, snap_n=SNAP_N):
    """Put every truss into a spec as a welded, pickable fruit.

    ⚠️ **The physics here is `plant_row.add_row`'s, deliberately duplicated
    rather than reused, and every number is copied not re-chosen.** `add_row`
    hardcodes `ROW_X` for the stem, the fruit and the support, so it cannot
    place a fruit on row 2 of 4. What it *can* do is define what a peduncle is,
    and that definition — the weld, its zeroed anchor, `PEDUNCLE_SOLREF`, the
    fruit's mass and friction — is the thing Weeks 2-4 measured against. Change
    any of it here and the snap force, the grip transient and the clean rate all
    move while `plant_row.py` still claims the old numbers.
    """
    import mujoco

    # One support bar per row, at the height the stems hang from.
    rows = sorted({t.row for t in trusses})
    for r in rows:
        top = fruit_z(0.88 - 0.0)      # plant_row.SUPPORT_Z, lifted with the crop
        bar = spec.worldbody.add_body(name=f"fc_support_r{r}",
                                      pos=[house.row_x(r), 0.0, top])
        bar.add_geom(name=f"fc_support_r{r}_bar",
                     type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                     fromto=[0, -house.HOUSE_HALF_Y, 0,
                             0, house.HOUSE_HALF_Y, 0],
                     size=[0.008, 0, 0], rgba=[0.45, 0.34, 0.22, 1.0],
                     contype=0, conaffinity=0, mass=0.0)

    for t in trusses:
        stem = spec.worldbody.add_body(
            name=f"stem_{t.name}", pos=[t.x, t.y, t.z + STEM_LEN], mocap=True)
        stem.add_geom(
            name=f"stem_{t.name}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -STEM_LEN], size=[STEM_R, 0, 0],
            rgba=[0.30, 0.50, 0.28, 1.0], contype=0, conaffinity=0)
        drop = fruit_z(0.88) - (t.z + STEM_LEN)
        if drop > 0.001:
            stem.add_geom(
                name=f"stem_{t.name}_drop", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, drop], size=[STEM_R * 0.6, 0, 0],
                rgba=[0.30, 0.50, 0.28, 1.0], contype=0, conaffinity=0)

        fruit = spec.worldbody.add_body(name=t.name, pos=[t.x, t.y, t.z])
        fruit.add_freejoint()
        fruit.add_geom(
            name=f"{t.name}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[FRUIT_R, 0, 0], rgba=list(t.rgba), mass=FRUIT_MASS,
            friction=[0.6, 0.01, 0.001])

        eq = spec.add_equality(
            name=f"peduncle_{t.name}", type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"stem_{t.name}", name2=t.name)
        # The eleven numbers. The first three MUST be written — left to the
        # compiler the anchor comes out [0, 1, 0], a metre off to the side, and
        # the weld behaves like a rubber band with 160 mm of sag. No warning.
        eq.data = [0, 0, 0, 0, 0, -STEM_LEN, 1, 0, 0, 0, 1.0]
        eq.solref = list(PEDUNCLE_SOLREF)
    return [t.name for t in trusses]


def reachable(t, aisle, y, arm="a"):
    """Can the arm on a trolley at `y` in `aisle` work truss `t`?

    Pure geometry against the measured envelope — not a plan. `mission.Planner`
    decides whether a route exists; this decides whether it is worth asking.
    """
    left, right = house.serves(aisle)
    want = right if arm == "a" else left
    if t.row != want:
        return False
    return abs(t.y - y) <= Y_REACH


def stops_for(trusses, aisle, arm="a", window=None):
    """Trolley stops that between them can reach every truss on that row.

    Greedy interval cover along the row: take the lowest un-covered fruit, put a
    stop a window-half above it, sweep on. That is optimal for covering points
    with fixed-width intervals, which is what this is — and it is worth saying
    so, because the *ordering* problem inside a stop is not that easy and gets
    `deck_cam.plan_order` instead.
    """
    w = Y_REACH if window is None else window
    mine = sorted((t for t in trusses if reachable(t, aisle, t.y, arm)),
                  key=lambda t: t.y)
    out = []
    i = 0
    while i < len(mine):
        y = mine[i].y + w
        out.append(float(np.clip(y, *trolley.y_limits())))
        while i < len(mine) and mine[i].y <= y + w:
            i += 1
    return out


def summarise(trusses):
    """Counts by stage and by row, and what a perfect run would take."""
    rows = sorted({t.row for t in trusses})
    by_stage = {n: 0 for n, _, _, _ in STAGES}
    for t in trusses:
        by_stage[t.stage] += 1
    ripe = [t for t in trusses if t.ripe]
    return {"n": len(trusses), "rows": rows, "by_stage": by_stage,
            "ripe": len(ripe),
            "ripe_by_row": {r: sum(1 for t in ripe if t.row == r) for r in rows}}


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("-n", type=int, default=14, help="fruit per row")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    print(__doc__)
    trusses = spawn(n_per_row=args.n, seed=args.seed)
    s = summarise(trusses)
    print(f"\n  --- the house, this time ---")
    print(f"  {s['n']} fruit on {len(s['rows'])} rows · "
          f"{s['ripe']} ripe ({100 * s['ripe'] / max(1, s['n']):.0f}%)")
    print(f"  {'stage':<10} {'count':>6}   {'pick?':>6}")
    for n, _rgba, _hue, pick in STAGES:
        print(f"  {n:<10} {s['by_stage'][n]:>6}   {'yes' if pick else 'no':>6}")
    print(f"\n  ripe per row: "
          + "  ".join(f"r{r} {c}" for r, c in s["ripe_by_row"].items()))
    for a in house.covering_aisles():
        l, r = house.serves(a)
        stops = stops_for(trusses, a, "a")
        print(f"  aisle a{a} works r{l} (arm B) and r{r} (arm A); "
              f"arm A needs {len(stops)} stops")
    if args.stats:
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl" if args.shot else "glfw")
    import mujoco

    model = trolley.build(aisle=0, arms=("a",), trusses=trusses, seed=0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"\n  compiled: {model.ngeom} geoms, {model.nbody} bodies, "
          f"{model.nq} dof")

    if args.shot:
        import cv2

        out_dir = Path(__file__).resolve().parents[3]
        with mujoco.Renderer(model, height=960, width=1280) as r:
            for cam in ("house", "aisle"):
                r.update_scene(data, camera=cam)
                p = out_dir / f"farm_crop_{cam}.png"
                cv2.imwrite(str(p), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
                print(f"  wrote {p.name}")
        return 0

    import time

    import mujoco.viewer

    print("\n  window open — close it to quit")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
