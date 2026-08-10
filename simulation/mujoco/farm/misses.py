#!/usr/bin/env python3
"""Why does a ripe tomato get left on the plant? Measured, over a batch.

⚠️ **The point of this file is to stop the question being answered by watching.**
This repo has been wrong that way twice — Week 1's "1-in-6 grasp failure" was a
fixture bug the tick log had already disproved, and bug 40's confident
zero-friction story survived four sweeps before measurement refuted both halves
of it. A miss has at least seven distinguishable causes and they need completely
different fixes, so the first job is a breakdown, not a theory.

--- what `outcomes.classify` could not see ----------------------------------

`run()` logs one row per *attempt*, and `outcomes.classify` already buckets
those: refused, guard_abort, grasp_failed, dropped, ejected, clean. That is a
complete account of fruit the robot tried to pick.

It is silent about the ones it never tried, and those are misses too:

    not_mapped     the scouting pass never saw it. No box, no sighting, no
                   entry anywhere — so a shift can report 5/5 clean while
                   ripe fruit stand untouched two metres away.
    not_routed     mapped, but the route never scheduled it. Either it was out
                   of reach of every stop the planner chose, or --stops cut
                   the shift short.

Counting attempts alone therefore measures the robot's *aim* and calls it the
robot's *yield*. This walks the other way round: start from every ripe fruit
that was really in the house, and give each one a reason it is not in the crate.

    ./.venv/bin/python simulation/mujoco/farm/misses.py               # 5 shifts
    ./.venv/bin/python simulation/mujoco/farm/misses.py --shifts 10
    ./.venv/bin/python simulation/mujoco/farm/misses.py --truth       # perfect map
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import house, run as frun  # noqa: E402

# How close a sighting has to be to a real fruit to count as "this one was
# mapped". Deliberately the same 120 mm `farm.scout.score` uses, and generous
# for the same reason: the question here is "did the map contain it at all",
# not "how accurate was the position". A tight gate would report an imprecise
# sighting as a blind detector.
MAP_GATE_M = 0.12

# The truth-side buckets, ahead of everything `outcomes` already names.
NOT_MAPPED = "not_mapped"
NOT_ROUTED = "not_routed"

WHY = {
    NOT_MAPPED: "the scouting pass never saw it",
    NOT_ROUTED: "mapped, but no stop was scheduled that could reach it",
    "not_detected": "at the stop, no truss matched the sighting",
    "misassociated": "the sighting was matched to the wrong truss",
    "unreachable": "the arm could not get to it",
    "refused": "the planner found no route clear of the crop",
    "guard_abort": "flown, then abandoned mid-flight at GUARD_STOP",
    "grasp_failed": "reached it and never held it, or never detached it",
    "dropped": "held it, then lost it before the crate",
    "ejected": "the pads squeezed it out at speed",
    "clean": "in the crate",
}


def attribute(shift):
    """Every ripe fruit on the worked row -> the reason it is or is not crated.

    Returns a list of `(truss, bucket)`. The order of tests is the order the
    shift can fail in, earliest first, so each fruit lands in exactly one
    bucket: a fruit that was never mapped cannot also be a grasp failure.
    """
    _left, right = house.serves(shift.aisle)
    ripe = [t for t in shift.trusses if t.ripe and t.row == right]

    mapped = []
    if shift.house_map is not None:
        mapped = [np.asarray(s.pos, float) for s in shift.house_map.sightings]

    routed = set()
    if shift.route is not None:
        for stop in shift.route.stops:
            for f in stop.fruit:
                routed.add(_key(np.asarray(f.pos, float)))

    # Attempts, by the truss name the executor resolved them to.
    by_name = {}
    for r in shift.rows:
        if r.get("fruit"):
            by_name[r["fruit"]] = r
    unnamed = [r for r in shift.rows if not r.get("fruit")]

    out = []
    for t in ripe:
        pos = np.asarray(t.pos, float)
        if mapped and min(float(np.linalg.norm(pos - m))
                          for m in mapped) > MAP_GATE_M:
            out.append((t, NOT_MAPPED))
            continue
        if not mapped:
            out.append((t, NOT_MAPPED))
            continue
        rec = by_name.get(t.name)
        if rec is not None:
            out.append((t, rec.get("outcome", "grasp_failed")))
            continue
        if _key(pos) not in routed and not _near_routed(pos, routed):
            out.append((t, NOT_ROUTED))
            continue
        # Scheduled, but the executor never resolved a name to it. That is the
        # association step failing, which `run` already logs as a nameless row.
        out.append((t, "not_detected" if unnamed else NOT_ROUTED))
    return out


def _key(pos):
    return (round(float(pos[0]), 3), round(float(pos[1]), 3),
            round(float(pos[2]), 3))


def _near_routed(pos, routed, gate=MAP_GATE_M):
    """Route stops carry *sightings*, whose positions are estimates.

    ⚠️ So an exact key match is the wrong test and would report every routed
    fruit as unrouted — the estimate is millimetres off the truth by
    construction. Matching by distance is the honest version.
    """
    for k in routed:
        if float(np.linalg.norm(pos - np.array(k))) <= gate:
            return True
    return False


def report(tally, shifts, ripe_total):
    """The breakdown, worst first, with what each bucket would take to fix."""
    print(f"\n{'=' * 78}")
    print(f"  WHY RIPE FRUIT ARE NOT IN THE CRATE — {shifts} shift"
          f"{'s' if shifts != 1 else ''}, {ripe_total} ripe fruit")
    print(f"{'=' * 78}\n")
    crated = tally.get("clean", 0)
    print(f"  {'bucket':<16} {'n':>4} {'share':>7}   what it means")
    print("  " + "-" * 74)
    for bucket, n in tally.most_common():
        share = 100.0 * n / max(ripe_total, 1)
        mark = " " if bucket == "clean" else "*"
        print(f"{mark} {bucket:<16} {n:>4} {share:>6.1f}%   "
              f"{WHY.get(bucket, '')}")
    print("  " + "-" * 74)
    print(f"  crated {crated}/{ripe_total} = "
          f"{100.0 * crated / max(ripe_total, 1):.1f}% of the ripe fruit "
          f"that were really there")

    misses = [(b, n) for b, n in tally.most_common() if b != "clean"]
    if not misses:
        print("\n  nothing missed. Do not tune anything.")
        return None
    top, n = misses[0]
    print(f"\n  dominant bucket: {top} — {n} of "
          f"{sum(m for _b, m in misses)} misses "
          f"({100.0 * n / sum(m for _b, m in misses):.0f}% of them)")
    print(f"  {WHY.get(top, '')}")
    print("\n  ⚠️ Fix this one. Anything spent on the others is spent on at "
          f"most {100 - 100.0 * n / sum(m for _b, m in misses):.0f}% of the "
          "problem.")
    return top


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shifts", type=int, default=5)
    ap.add_argument("--truth", action="store_true",
                    help="perfect map — isolates the buckets after perception")
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=14, help="fruit per row")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--stops", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    base = fcrop.resolve_seed(args.seed, label="batch")
    tally = Counter()
    ripe_total = 0
    per_shift = []

    for k in range(args.shifts):
        seed = base + k
        print(f"\n  --- shift {k + 1}/{args.shifts}, seed {seed} ---")
        shift = frun.run(seed=seed, aisle=args.aisle, n_per_row=args.n,
                         speed=args.speed, use_truth=args.truth,
                         max_stops=args.stops, verbose=False)
        rows = attribute(shift)
        for _t, bucket in rows:
            tally[bucket] += 1
        ripe_total += len(rows)
        got = sum(1 for _t, b in rows if b == "clean")
        per_shift.append((seed, got, len(rows)))
        print(f"    {got}/{len(rows)} ripe fruit crated · "
              + " · ".join(f"{b} {n}" for b, n in
                           Counter(b for _t, b in rows).most_common()
                           if b != "clean"))

    print(f"\n  per shift: " + "  ".join(f"{g}/{n}" for _s, g, n in per_shift))
    report(tally, args.shifts, ripe_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
