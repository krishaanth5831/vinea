#!/usr/bin/env python3
"""Does picking in the deck camera's order beat picking in placement order?

`deck_cam.plan_order` produces a number — total risk — and a number a model
produces about itself is not evidence. This flies the same layouts twice, once
in the order the deck camera chose and once in the order the fruit happen to be
named, and reports what actually came out of the crate.

    ./.venv/bin/python simulation/mujoco/week4_order.py
    ./.venv/bin/python simulation/mujoco/week4_order.py --layouts 6 --fruit 10
    ./.venv/bin/python simulation/mujoco/week4_order.py --spread   # easy rows

⚠️ **The layouts are deliberately clustered, and that is the point.** On a row
whose closest pair is 220 mm every order is the same order — the pair sweep in
`deck_cam` shows a neighbour costs nothing past 170 mm, so there is nothing to
optimise and this script would measure noise. `cluster_layout` builds rows with
pairs in the 90-150 mm band where the sweep says a neighbour decides whether a
pick is refused. Those rows were **impossible to build at all** until the 200 mm
placement minimum was removed, which is the other reason this file is new.

⚠️ **Both arms of the comparison get the deck camera's positions.** The only
thing that differs is the *order*. Handing the placement-order arm ground truth
instead would be comparing two changes at once and neither number would mean
anything.

What it produced on 3 layouts x 6 fruit at speed 0.15, 18 attempts per arm:

    order              clean    crated   refused   disturbed
    deck-planned       11/18       11         2           0
    placement order    10/18       10         6           0

**Refusals are the signal, not the clean rate.** One fruit of difference on n=18
is noise; 2 refusals against 6 is the mechanism the pair sweep predicts. And the
per-layout split matters more than the total — at 112 mm the two orders tied,
because that spacing costs a fallback route but never a refusal, so there was
nothing for an ordering to win.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cluster_layout(n, seed=0, tries=4000):
    """`n` fruit with several pairs inside the band where order matters.

    Half the fruit are seeded anywhere in the measured envelope; the rest are
    hung deliberately close to one already placed — 90 to 150 mm, which
    straddles the 120 mm at which a square-on pick starts being refused.
    """
    from week4_place import (MARGINAL_HALF_Y, MARGINAL_Z, check, zone)
    from plant_row import ROW_X

    rng = np.random.default_rng(seed)
    placed = {}
    out = []

    def put(y, z):
        ok, _why, _zn = check(y, z, placed)
        if not ok or zone(y, z) is None:
            return False
        name = f"p{len(out):02d}"
        placed[name] = np.array([ROW_X, y, z])
        out.append((name, round(float(y), 4), round(float(z), 4)))
        return True

    n_seeds = max(1, n // 2)
    for _ in range(tries):
        if len(out) >= n_seeds:
            break
        put(float(rng.uniform(-MARGINAL_HALF_Y, MARGINAL_HALF_Y)),
            float(rng.uniform(*MARGINAL_Z)))
    for _ in range(tries):
        if len(out) >= n:
            break
        anchor = placed[list(placed)[rng.integers(len(placed))]]
        theta = float(rng.uniform(0, 2 * np.pi))
        d = float(rng.uniform(0.090, 0.150))
        put(float(anchor[1] + d * np.cos(theta)),
            float(anchor[2] + d * np.sin(theta)))
    return out


def spread_layout(n, seed=0):
    from week4_place import auto_layout

    return auto_layout(n, seed=seed)


def _build(seen, pool):
    import mujoco

    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from plant_row import Row
    from week4_place import Crop, park_spot

    model = build_scene(wrist_cam=seen, deck_cam=True, trusses=pool)
    data = mujoco.MjData(model)
    names = [n for n, _, _ in pool]
    row = Row(model, data, names=names,
              homes={n: park_spot(i) for i, n in enumerate(names)})
    park_q = park_posture(model)
    reset_park(model, data, park_q)
    row.reset()
    mujoco.mj_forward(model, data)
    return model, data, row, park_q, Crop(model, data, row, names), names


def run_one(layout, ordered, args):
    """One harvest. `ordered` picks the deck order; otherwise placement order.

    ⚠️ A fresh model per arm, not a reset. `reset_park` restores every free body
    and re-welds every stem, which is right for a new trial — but the two arms
    have to be *independent* trials, and reusing one model would carry the
    planner's warm-started IK configuration from the first into the second. That
    is a real difference and it is not the one being measured.
    """
    from camera import SensorCamera
    from deck_cam import DeckSurvey, plan_order, score_order
    from week3_perceive import build_detector
    from week4_place import harvest_placed, pool_trusses

    pool = pool_trusses()
    model, data, row, park_q, crop, names = _build(args.seen, pool)
    crop.apply(layout)

    sensor = detector = None
    if args.seen:
        sensor = SensorCamera(model, camera="wrist")
        detector = build_detector(args.detector)
    deck = DeckSurvey(model, detector=build_detector(args.detector))

    # Both arms are surveyed, so both are working from camera positions. Only
    # the order differs. The survey is also what scores the two orders against
    # the geometric model, so the prediction and the outcome come from the same
    # frame.
    seen_map, _rep = deck.look(data, list(crop.placed))
    predicted = plan_order(seen_map)
    baseline = score_order([n for n in crop.placed if n in seen_map], seen_map)

    try:
        rows = harvest_placed(model, data, row, crop, park_q, speed=args.speed,
                              sensor=sensor, detector=detector, seed=0,
                              deck=deck if ordered else None)
    finally:
        deck.close()
        if sensor is not None:
            sensor.close()
    return rows, predicted, baseline


def summarise(rows):
    from picklog import throughput

    t = throughput(rows) or {}
    clean = sum(1 for r in rows if r.get("outcome") == "clean")
    refused = sum(1 for r in rows if r.get("outcome") == "refused")
    # Fruit the deck camera never separated from a neighbour. Reported in its
    # own column rather than folded into the failures, because it is the one
    # bucket the *sensor* owns — the arm never got a chance at these, and a
    # reader comparing the two orders needs to see that separately.
    unseen = sum(1 for r in rows if r.get("outcome") == "not_detected")
    crated = sum(1 for r in rows if r.get("in_bin"))
    disturbed = sum(int(r.get("disturbed") or 0) for r in rows)
    lost = sum(int(r.get("lost") or 0) for r in rows)
    gaps = [r["clearance_min_mm"] for r in rows
            if r.get("clearance_min_mm") is not None]
    return {"n": len(rows), "clean": clean, "crated": crated,
            "refused": refused, "unseen": unseen,
            "disturbed": disturbed, "lost": lost,
            "cycle_s": t.get("mean_s", float("nan")),
            "min_clear_mm": min(gaps) if gaps else float("nan")}


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", type=int, default=4)
    ap.add_argument("--fruit", type=int, default=8)
    ap.add_argument("--spread", action="store_true",
                    help="use auto_layout's spread rows instead of clustered "
                         "ones — the control, where order should NOT matter")
    ap.add_argument("--seen", action="store_true",
                    help="wrist perception in the loop as well")
    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"])
    # ⚠️ 0.15, the speed every pick rate in this repo was taken at. Raising it
    # past mission.CARRY_SPEED's 0.25 cliff makes fruit fly out of the gripper,
    # which would swamp the ordering effect with a payload-retention effect.
    ap.add_argument("--speed", type=float, default=0.15)
    args = ap.parse_args()

    make = spread_layout if args.spread else cluster_layout
    print(f"\n{'=' * 78}")
    print(f"  ORDERING A/B — {args.layouts} layouts x {args.fruit} fruit, "
          f"{'spread' if args.spread else 'clustered'}, "
          f"{'perception' if args.seen else 'told'}, speed {args.speed}")
    print(f"  every layout flown twice: deck-planned order vs placement order")

    t0 = time.perf_counter()
    agg = {"deck": [], "placed": []}
    predictions = []
    for i in range(args.layouts):
        layout = make(args.fruit, seed=1000 + i)
        pts = [(y, z) for _n, y, z in layout]
        closest = min((float(np.hypot(a[0] - b[0], a[1] - b[1]))
                       for j, a in enumerate(pts) for b in pts[j + 1:]),
                      default=float("nan"))
        print(f"\n{'-' * 78}\n  LAYOUT {i + 1}: {len(layout)} fruit, "
              f"closest pair {closest * 1000:.0f} mm")

        for arm, ordered in (("deck", True), ("placed", False)):
            rows, predicted, baseline = run_one(layout, ordered, args)
            agg[arm].extend(rows)
            if arm == "deck":
                predictions.append((predicted.total_risk,
                                    baseline["total_risk"], closest))
            s = summarise(rows)
            print(f"    {arm:<7} clean {s['clean']}/{s['n']} · crated "
                  f"{s['crated']} · refused {s['refused']} · unseen "
                  f"{s['unseen']} · disturbed {s['disturbed']} · cycle "
                  f"{s['cycle_s']:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"  RESULT after {time.perf_counter() - t0:.0f}s wall\n")

    # ⚠️ **Equal denominators, checked rather than assumed.** Both arms are
    # handed identical layouts, so both must produce identical attempt counts —
    # every placed fruit lands in exactly one bucket (`outcomes.py`'s first
    # rule), including fruit the deck camera never separated, which book as
    # `not_detected`.
    #
    # This is checked because it has already gone wrong once and the failure is
    # invisible in the headline: an early version dropped unseen fruit from the
    # log instead of booking them, so the deck arm scored 11/15 against the
    # placement arm's 10/18 — 73% vs 56%, entirely on a shrunken denominator.
    # A clean rate is a ratio, and a ratio whose denominators differ is not a
    # comparison.
    n_deck, n_placed = len(agg["deck"]), len(agg["placed"])
    if n_deck != n_placed:
        print(f"  ⚠️ ATTEMPT COUNTS DIFFER — {n_deck} vs {n_placed}. The clean "
              f"rates below are NOT comparable:")
        print(f"     one arm is being scored over fewer fruit than it was "
              f"given. Find the missing")
        print(f"     rows before reading anything else here.\n")

    print(f"  {'order':<20} {'attempts':>9} {'clean':>8} {'crated':>8} "
          f"{'refused':>8} {'unseen':>7} {'disturbed':>10} {'cycle s':>9} "
          f"{'min gap mm':>11}")
    for arm, label in (("deck", "deck-planned"), ("placed", "placement order")):
        s = summarise(agg[arm])
        pct = 100 * s["clean"] / s["n"] if s["n"] else 0
        print(f"  {label:<20} {s['n']:>9} {s['clean']:>4}/{s['n']:<3} "
              f"{s['crated']:>8} {s['refused']:>8} {s['unseen']:>7} "
              f"{s['disturbed']:>10} {s['cycle_s']:>9.1f} "
              f"{s['min_clear_mm']:>11.0f}   ({pct:.0f}%)")

    # Did the geometric proxy predict the direction of the measured difference?
    # It is a cheap model standing in for a 150 ms kinematic replay, so it is
    # worth knowing whether it is worth anything at all.
    print(f"\n  the model's own prediction, per layout:")
    print(f"  {'closest pair':>13} {'deck risk':>10} {'placed risk':>12}")
    for deck_r, placed_r, closest in predictions:
        print(f"  {closest * 1000:12.0f}  {deck_r:>10.2f} {placed_r:>12.2f}")

    d, p = summarise(agg["deck"]), summarise(agg["placed"])
    if d["n"] and p["n"]:
        print(f"\n  crated {d['crated']} vs {p['crated']} "
              f"({d['crated'] - p['crated']:+d}) · "
              f"refused {d['refused']} vs {p['refused']} "
              f"({d['refused'] - p['refused']:+d}) · "
              f"unseen {d['unseen']} vs {p['unseen']} · "
              f"neighbours disturbed {d['disturbed']} vs {p['disturbed']} "
              f"({d['disturbed'] - p['disturbed']:+d})")
        print(f"\n  ⚠️ n is {d['n']} per arm. Read the direction, not the "
              f"decimal — and note that")
        print(f"     tour length cannot show up in the cycle time at all, "
              f"because mission.park_arm")
        print(f"     teleports between picks. What the order can move here is "
              f"refusals and")
        print(f"     disturbed neighbours; see deck_cam.W_TRAVEL.")


if __name__ == "__main__":
    main()
