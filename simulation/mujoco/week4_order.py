#!/usr/bin/env python3
"""Does picking in the deck camera's order beat picking in placement order?

`deck_cam.plan_order` produces a number — expected refusals — and a number a
model produces about itself is not evidence. This flies the same layouts twice,
once in the order the deck camera chose and once in the order the fruit happen
to be named, and reports what actually came out of the crate.

    ./.venv/bin/python simulation/mujoco/week4_order.py
    ./.venv/bin/python simulation/mujoco/week4_order.py --band blocked
    ./.venv/bin/python simulation/mujoco/week4_order.py --band loose
    ./.venv/bin/python simulation/mujoco/week4_order.py --spread   # easy rows

⚠️ **The answer depends on how tight the row is, and a single "clustered"
number hides that.** On a row whose closest pair is 220 mm every order is the
same order — the pair sweep in `deck_cam` shows a neighbour costs nothing past
170 mm. On a row where everything is inside 95 mm, every order is also the same
order, for the opposite reason: there is no sequence that saves a fruit which is
blocked whatever you do. Ordering can only earn anything in between. `BANDS`
names the three regimes and `--band` selects one; rows in any of them were
**impossible to build at all** until the 200 mm placement minimum was removed,
which is the other reason this file is new.

⚠️ **Both arms get the deck camera's positions, and the code did not always do
that.** The control arm is `deck_order=False`, not `deck=None`. Passing
`deck=None` — which is what this file did first — also switches that arm's
staging poses over to `crop.placed`, the operator's ground truth, so the control
would be running with *better* position knowledge than the arm under test. Two
changes at once, and the better-informed one is the baseline.

⚠️ **Two results were thrown away getting here, and both were thrown away for
the same reason: the row was not what the label said.** The first ran the
control on ground truth (above). The second fixed that and still came back null
— 19 refusals against 19 — because `cluster_layout` aimed pairs at 90-150 mm
without enforcing a floor on the *incidental* pairs eight fruit make in a small
envelope, so rows labelled "clustered" had closest pairs of 76-97 mm. That is
the blocked band, where by construction no order helps, reported as though it
were the contested one. The floor is now checked pair by pair; see
`cluster_layout`.

An earlier 3-layout run at 6 fruit, before either fix, produced 2 refusals
against 6. That number is **not** quoted as a result here because the run that
produced it had the ground-truth control, and a measurement taken against a
baseline that was handed the answer is not a measurement.

**What it produced once both were fixed** — `--band contested`, 4 layouts x 8
fruit, 32 attempts per arm:

    order              clean    crated   refused   disturbed   forecast
    deck-planned       19/32       19        12         0        0.00
    placement order    18/32       18        12         0          —

**A tie — and the corrected cost model predicted the tie exactly.** Its forecast
gain was 0.00 fruit on every one of the four layouts, and the measured refusal
difference was 0. The *old* model, on these same layouts, forecast a 3.33-fruit
gain that never appeared. So the value of the rewrite is not that the robot
harvests more; it is that the planner stopped claiming it would.

The reason it forecasts zero is in `deck_cam.BLOCKED_M`: `_pair_risk` is a
function of the separation vector and therefore symmetric, so at 100 mm it
scores both fruit of a pair as blocked — while the sweep it was fitted to says
*one of them plans fine*. The model cannot express the asymmetry the ordering
exists to exploit, so it finds no ordering worth having, and `deck_cam
--optimal` shows exactly that: every method from placement order to the exact
solver produces an identical expected loss.

⚠️ So this file is currently doing its job by returning a negative. The
machinery is right, the search is near-exact, and the thing it is optimising has
no ordering signal in it. The next measurement is `--pairs` recording *which* of
a close pair gets refused rather than just that one does.

⚠️ **Unexplained, and left in rather than tidied away:** the deck-ordered arm
ran a 16.3 s mean cycle against 21.0 s, holding direction on three layouts of
four. That should be impossible — `mission.park_arm` teleports between picks, so
tour length cannot reach the clock. Route complexity is the plausible mechanism
and the per-layout magnitudes are too big for it. Not traced, not claimed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# The tightness bands, in metres, centre to centre. Named because the answer
# turns out to depend on which one you sample, and a single "clustered" number
# hides that completely — see `main`.
#
#   blocked    everything is inside BLOCKED_M of something. Both orders lose the
#              same fruit because there is no order that saves them.
#   contested  straddles the 120 mm at which a square-on pick starts being
#              refused. This is the band the pair sweep says order decides.
#   loose      outside CROWDED_M. Nothing blocks anything; the control.
BANDS = {
    "blocked": (0.075, 0.095),
    "contested": (0.100, 0.150),
    "loose": (0.175, 0.260),
}


def cluster_layout(n, seed=0, tries=4000, band="contested"):
    """`n` fruit with several pairs inside `band`, and none tighter.

    Half the fruit are seeded anywhere in the measured envelope; the rest are
    hung deliberately close to one already placed.

    ⚠️ **The floor is enforced, not assumed.** `week4_place.check` only refuses
    fruit closer than 70 mm (touching), so a generator that merely *aims* at
    90-150 mm still produces incidental pairs far tighter once eight fruit are
    in the envelope. The first six-layout A/B was run this way and came back
    with closest pairs of 76-97 mm on rows labelled "90-150" — which is the
    blocked band, where by construction no order can help, reported as though it
    were the contested one. Every pair is now checked against the band's floor.
    """
    from plant_row import ROW_X
    from week4_place import MARGINAL_HALF_Y, MARGINAL_Z, check, zone

    lo, hi = BANDS[band] if isinstance(band, str) else band
    rng = np.random.default_rng(seed)
    placed = {}
    out = []

    def put(y, z):
        ok, _why, _zn = check(y, z, placed)
        if not ok or zone(y, z) is None:
            return False
        # The floor. Applies to every pair in the row, not just the one being
        # aimed at, which is the whole point.
        if any(np.hypot(y - p[1], z - p[2]) < lo - 1e-9 for p in placed.values()):
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
        d = float(rng.uniform(lo, hi))
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
    import mujoco

    from camera import SensorCamera
    from deck_cam import DeckHead, DeckSurvey, plan_order, score_order
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
    # ⚠️ `scan`, not `look`, and it has to match what `harvest_placed` does or
    # the prediction is scored against a survey the arm never used. The two
    # surveys do not see the same fruit on a clustered row — that is the whole
    # point of the head — so a one-frame prediction next to a scanned harvest
    # would be comparing two different rows.
    head = DeckHead(model, data)
    seen_map, _rep = deck.scan(data, list(crop.placed), head=head)
    head.home(data)
    mujoco.mj_forward(model, data)
    predicted = plan_order(seen_map)
    baseline = score_order([n for n in crop.placed if n in seen_map], seen_map)

    try:
        # ⚠️ **`deck` for both arms, always.** The control differs by
        # `deck_order` and by nothing else. Passing `deck=None` here — which is
        # what this file did first — also hands the control arm `crop.placed`,
        # the operator's ground truth, so it would be running with *better*
        # position knowledge than the arm under test. The first six-layout
        # result was collected that way and had to be thrown out.
        rows = harvest_placed(model, data, row, crop, park_q, speed=args.speed,
                              sensor=sensor, detector=detector, seed=0,
                              deck=deck, deck_order=ordered)
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
    ap.add_argument("--band", default="contested", choices=sorted(BANDS),
                    help="how tight the clustered rows are. 'contested' is the "
                         "band the pair sweep says order decides; 'blocked' is "
                         "tighter than any order can fix; 'loose' is the "
                         "second control")
    ap.add_argument("--seen", action="store_true",
                    help="wrist perception in the loop as well")
    ap.add_argument("--detector", default="hsv", choices=["hsv", "yolo"])
    # ⚠️ 0.15, the speed every pick rate in this repo was taken at.
    #
    # This used to carry a warning that raising it past `mission.CARRY_SPEED`'s
    # 0.25 would make fruit fly out of the gripper and swamp the ordering effect
    # with a payload-retention one. That was true and is now much less so:
    # `greenhouse.PAD_SOLREF` was the cause and `week4_grip.py` fixed it. The
    # remaining `turn`-leg instability does not scale with speed either — 2.07
    # m/s at 0.40 against 2.06 at 0.25. Raising this to shorten a run is now a
    # defensible trade rather than a way to corrupt the measurement, but the
    # default stays where every other number was taken.
    ap.add_argument("--speed", type=float, default=0.15)
    args = ap.parse_args()

    make = (spread_layout if args.spread
            else lambda n, seed: cluster_layout(n, seed=seed, band=args.band))
    lo, hi = BANDS[args.band]
    print(f"\n{'=' * 78}")
    print(f"  ORDERING A/B — {args.layouts} layouts x {args.fruit} fruit, "
          f"{'spread' if args.spread else f'{args.band} band '
             f'({lo * 1000:.0f}-{hi * 1000:.0f} mm)'}, "
          f"{'perception' if args.seen else 'told'}, speed {args.speed}")
    print(f"  every layout flown twice: deck-planned order vs placement order")
    print(f"  ⚠️ both arms get the deck survey — only the order differs")

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
                # ⚠️ `lost` — expected refusals, in fruit — not `total_risk`.
                # It is what `plan_order` now minimises, and it is the column
                # that can be checked against the `refused` count printed two
                # lines down. A prediction in units nothing else is measured in
                # cannot be right or wrong.
                predictions.append((predicted.lost, baseline["lost"], closest))
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
    print(f"\n  the model's own prediction, per layout — expected refusals, "
          f"in fruit:")
    print(f"  {'closest pair':>13} {'deck order':>11} {'placement':>11} "
          f"{'predicted gain':>15}")
    for deck_r, placed_r, closest in predictions:
        print(f"  {closest * 1000:12.0f}  {deck_r:>11.2f} {placed_r:>11.2f} "
              f"{placed_r - deck_r:>15.2f}")
    if predictions:
        gain = sum(p - d for d, p, _c in predictions)
        print(f"  the proxy expects the deck order to save {gain:.2f} fruit "
              f"across these layouts.")
        print(f"  ⚠️ Compare that against the measured `refused` difference "
              f"below, not against\n     the clean rate — refusals are what "
              f"this model claims to predict, and the only\n     thing it can "
              f"be held to.")

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
