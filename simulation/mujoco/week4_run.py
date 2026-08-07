#!/usr/bin/env python3
"""Fifty-plus picks that are actually fifty-plus samples.

Ten runs of one five-fruit row is a repeatability figure — Week 2 already
produced that. A throughput figure needs the *geometry* to vary, because
geometry is what decides whether a route clears, and `week4_place.py` is the
generator that makes that possible.

So this sweeps **crop density** as well as seed: 5, 8, 12 and 15 fruit, several
layouts each. Density is the second axis on kg/hr and nothing in this repo has
ever measured it — the fixed row is five fruit whose closest pair is 172 mm, and
a real grower can tell you what their spacing actually is.

Every attempt appends to one JSONL as it happens, so a crash at pick 43 costs
one pick rather than 43. Re-read it any time with:

    ./.venv/bin/python simulation/mujoco/picklog.py runs/campaign.jsonl

    ./.venv/bin/python simulation/mujoco/week4_run.py --out runs/campaign.jsonl
    ./.venv/bin/python simulation/mujoco/week4_run.py --plan 5x3,15x1 --seen
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 58 attempts across four densities. Each entry is (n_fruit, how many layouts).
DEFAULT_PLAN = ((5, 3), (8, 2), (12, 1), (15, 1))


def parse_plan(s):
    out = []
    for part in s.split(","):
        n, k = part.lower().split("x")
        out.append((int(n), int(k)))
    return tuple(out)


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/campaign.jsonl")
    ap.add_argument("--plan", default=None,
                    help="densities, e.g. '5x3,8x2,12x1,15x1'")
    ap.add_argument("--seen", action="store_true",
                    help="perception in the loop (slower, and the honest number)")
    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"])
    ap.add_argument("--speed", type=float, default=None)
    # ⚠️ **Off by default, and that is not an oversight.** The campaign's whole
    # value is that its 57 attempts are comparable with each other and with the
    # figure in the README; turning the deck camera on changes the pick order,
    # which is a methodology change, not a flag. Run it on to measure what the
    # ordering does to throughput across densities — but report it as its own
    # campaign rather than as more samples of the old one.
    ap.add_argument("--deck", action="store_true",
                    help="survey with the chassis camera and pick in its order "
                         "(a different campaign, not more of the same one)")
    args = ap.parse_args()

    import mujoco

    from camera import SensorCamera
    from greenhouse import build_scene
    from mission import CLEARANCE, park_posture, reset_park
    from outcomes import report
    from picklog import PickLog, throughput
    from plant_row import Row
    from week3_perceive import build_detector
    from week4_place import (Crop, MAX_FRUIT, auto_layout, harvest_placed,
                             park_spot, pool_trusses)

    plan = parse_plan(args.plan) if args.plan else DEFAULT_PLAN
    total = sum(n * k for n, k in plan)
    print(f"\n{'=' * 78}")
    print(f"  CAMPAIGN — {total} attempts across densities "
          f"{', '.join(f'{n}x{k}' for n, k in plan)}")
    print(f"  {'perception' if args.seen else 'told'} · "
          f"{'deck-planned order' if args.deck else 'placement order'} · "
          f"log -> {args.out}")
    print(f"  expect roughly {total * 30 / 60:.0f} min of simulated cycle time")

    log = PickLog(args.out, meta={
        "detector": args.detector if args.seen else "told",
        "speed": args.speed, "clearance_mm": CLEARANCE * 1000})

    all_rows = []
    t_start = time.perf_counter()
    for n_fruit, k in plan:
        for rep in range(k):
            seed = n_fruit * 100 + rep
            print(f"\n{'-' * 78}\n  LAYOUT: {n_fruit} fruit, seed {seed}")

            # A fresh model per layout. Rebuilding is cheap next to the picks and
            # it guarantees no state leaks between layouts — the failure mode
            # bug 39 was, where a reset quietly restored fruit between trials.
            pool = pool_trusses()
            model = build_scene(wrist_cam=args.seen, deck_cam=args.deck,
                                trusses=pool)
            data = mujoco.MjData(model)
            names = [nm for nm, _, _ in pool]
            row = Row(model, data, names=names,
                      homes={nm: park_spot(i) for i, nm in enumerate(names)})
            park_q = park_posture(model)
            reset_park(model, data, park_q)
            row.reset()
            mujoco.mj_forward(model, data)

            crop = Crop(model, data, row, names)
            crop.apply(auto_layout(min(n_fruit, MAX_FRUIT), seed=seed))

            sensor = detector = deck = None
            if args.seen:
                sensor = SensorCamera(model, camera="wrist")
                detector = build_detector(args.detector)
            if args.deck:
                from deck_cam import DeckSurvey

                deck = DeckSurvey(model, detector=build_detector(args.detector))

            log.meta["layout"] = f"auto{n_fruit}s{seed}"
            try:
                rows = harvest_placed(model, data, row, crop, park_q,
                                      speed=args.speed, sensor=sensor,
                                      detector=detector, log=log, seed=seed,
                                      deck=deck)
            finally:
                # Each layout builds a fresh model, so each also builds fresh
                # GL contexts. Not closing them exhausts EGL a few layouts in.
                for s_ in (deck, sensor):
                    if s_ is not None:
                        s_.close()
            all_rows.extend(rows)
            print(f"  running total: {len(all_rows)} attempts, "
                  f"{time.perf_counter() - t_start:.0f}s wall")

    print(f"\n{'=' * 78}\n  CAMPAIGN COMPLETE — {len(all_rows)} attempts, "
          f"{(time.perf_counter() - t_start) / 60:.1f} min wall")
    report(all_rows, "campaign")

    # Density is the axis this campaign exists to add.
    #
    # ⚠️ Grouped by the density that was actually *placed*, not the one asked
    # for, and grouping by the request silently drops a whole layout out of the
    # table when the two differ. It did exactly that on the first campaign: 14
    # rows vanished from the summary while still counting in the headline, which
    # is the kind of quiet inconsistency that makes a number impossible to
    # defend.
    #
    # The reason they used to differ was the 200 mm spacing minimum, which could
    # turn a request for 15 into 14. That rule is gone (see
    # `week4_place.TOUCHING`), so `auto_layout` now returns what it was asked
    # for — but the two can still part company, because `--deck` books fruit the
    # chassis camera never separated as `not_detected` against a crop that did
    # contain them. Keep reading the placed number.
    print(f"\n  {'-' * 66}\n  by crop density (as placed):")
    print(f"  {'fruit':>6} {'n':>4} {'clean':>7} {'cycle s':>9} {'kg/hr':>7}")
    for n_fruit in sorted({r.get("n_fruit") for r in all_rows if r.get("n_fruit")}):
        sub = [r for r in all_rows if r.get("n_fruit") == n_fruit]
        if not sub:
            continue
        t = throughput(sub)
        c = sum(1 for r in sub if r.get("outcome") == "clean")
        print(f"  {n_fruit:>6} {len(sub):>4} {c}/{len(sub):<5} "
              f"{t['mean_s'] if t else 0:>9.1f} {t['kg_hr'] if t else 0:>7.1f}")

    t = throughput(all_rows)
    if t:
        print(f"\n  {'-' * 66}\n  THROUGHPUT — single arm, sim contact, "
              f"no travel, no ripeness selection")
        print(f"  cycle          {t['mean_s']:.1f} s mean "
              f"({t['min_s']:.1f}-{t['max_s']:.1f})")
        print(f"  picks/hr       {t['picks_hr']:.0f}")
        print(f"  clean rate     {t['clean_rate'] * 100:.0f}%")
        print(f"  kg/hr          {t['kg_hr']:.1f}   (0.12 kg/fruit)")
        print(f"  kg/week 24/7   {t['kg_week']:.0f}")
        print(f"  vs the 24,000 kg/week design target: "
              f"{24000 / t['kg_week']:.1f}x short")
    log.close()


if __name__ == "__main__":
    main()
