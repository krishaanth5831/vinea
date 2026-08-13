#!/usr/bin/env python3
"""kg/hr for one Vinea robot — ONE ARM, from numbers already measured.

⚠️ **This file runs no physics and measures nothing.** It is arithmetic over
figures taken from named, dated runs, plus constants read out of the code. Every
input carries its class — MEASURED, READ or ASSUMED — and the report prints the
class next to the number, so a reader can attack a step without re-deriving it.

⚠️ **v2 is one-armed on purpose**, and it is a different claim from v1's. v1
measured the two-armed `two_arm_farm.py` machine; this measures the single arm
in the two scenes that actually have a throughput instrument:

    week4_run.py    58 attempts, four crop densities, bolted base, NO TRAVEL
    farm/run.py     a whole shift with scouting and driving (what farm/watch.py
                    shows you — same function, one arm, `arms=("a",)`)

They agree to 7% on completely different scenes, which is the strongest thing
in this file. See `--compare`.

⚠️ **The repo already computed kg/hr and this does not replace it.**
`picklog.throughput()` is the Week 4 deliverable and `picklog.py <log>` prints
it. This adds the travel term that scene cannot have, the assumptions, and the
provenance. Where they overlap, they agree by construction — the Week 4 numbers
below are `picklog.throughput`'s own output.

    ./.venv/bin/python simulation/mujoco/farm/throughput.py
    ./.venv/bin/python simulation/mujoco/farm/throughput.py --sensitivity
    ./.venv/bin/python simulation/mujoco/farm/throughput.py --compare
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COMMIT = "522a3f797f7a0e7d09d3c2843d41636a0b9d2db8"   # dev
MEASURED_ON = "2026-08-12"

MEASURED, READ, ASSUMED = "MEASURED", "READ", "ASSUMED"


@dataclass
class Input:
    value: float
    unit: str
    cls: str
    source: str

    def line(self, name, width=30):
        return (f"  {name:<{width}} {self.value:>9.3f} {self.unit:<8} "
                f"{self.cls:<9} {self.source}")


def read_constants():
    """Pulled live from the modules, so this file cannot drift from them."""
    from plant_row import FRUIT_MASS, SNAP_N
    from farm import house, trolley
    from reach import CTRL_DT, DEFAULT_SPEED

    return {
        "fruit_mass": Input(
            FRUIT_MASS, "kg", READ,
            "plant_row.py:30 FRUIT_MASS. Real-world basis (plant_row.py:27-28 "
            "comment, '~120 g') is NO SOURCE — GUESS"),
        "drive_speed": Input(
            trolley.DRIVE_SPEED, "m/s", READ, "farm/trolley.py:88 DRIVE_SPEED"),
        "ctrl_dt": Input(CTRL_DT, "s", READ, "reach.py:59 CTRL_DT"),
        "arm_speed": Input(
            DEFAULT_SPEED, "-", READ,
            "reach.py:64 DEFAULT_SPEED — what week4_run.py resolves --speed "
            "None to (week4_place.py:566)"),
        "snap_n": Input(
            SNAP_N, "N", READ,
            "plant_row.py:83. Simulator artefact — but MEASURED at 0% effect "
            "on kg/hr across 9-20 N, see week4_snap.py"),
        "row_len": Input(
            2 * house.HOUSE_HALF_Y, "m", READ,
            "farm/house.py:89 HOUSE_HALF_Y x 2. ⚠️ chosen so routing is "
            "non-trivial, NOT a real Venlo bay length"),
        "rows_per_pass_1arm": Input(
            1, "rows", READ,
            "farm/route.py:142-143 — one arm works ONE row per aisle pass, "
            "so a 4-row house needs 4 passes"),
    }


# --- MEASURED ----------------------------------------------------------------

RUNS = {
    # week4_run.py — the repo's own throughput campaign. Bolted base, so this
    # is picking alone: no travel, no scouting pass, no ripeness selection.
    "w4_truth": {
        "cmd": ("MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/week4_run.py --out runs/v2_truth.jsonl"),
        "kind": "week4", "attempts": 58, "clean": 48,
        "cycle_mean_s": 28.73, "cycle_p95_s": 34.98,
        "arm_speed": 0.15, "wall_min": 15.7,
    },
    "w4_seen": {
        "cmd": ("MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/week4_run.py --seen --out runs/v2_seen.jsonl"),
        "kind": "week4", "attempts": 58, "clean": 46,
        "cycle_mean_s": 29.74, "cycle_p95_s": 35.81,
        "arm_speed": 0.15, "wall_min": None,
    },
    # farm/run.py — a whole shift, one arm. This is the function farm/watch.py
    # drives; watch.py only adds panels. Three seeds pooled.
    "farm_truth": {
        "cmd": ("for S in 700 701 702; do MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/farm/run.py --truth --seed $S; done"),
        "kind": "farm", "crated": 6,
        "scout_s": 0.0, "drive_s": 34.9, "pick_s": 111.2, "odometer_m": 11.7,
        "arm_speed": 0.4, "seeds": (700, 701, 702),
    },
    "farm_scouted": {
        "cmd": ("for S in 700 701 702; do MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/farm/run.py --seed $S; done"),
        "kind": "farm", "crated": 5,
        "scout_s": 60.0, "drive_s": 22.8, "pick_s": 91.7, "odometer_m": 28.6,
        "arm_speed": 0.4, "seeds": (700, 701, 702),
    },
}

# week4_snap.py --n 8, values 9,12,16,20 N, one fixed layout.
SNAP_SWEEP = ((9, 13.2), (12, 13.2), (16, 13.2), (20, 13.2))

# week4_run.py's own density breakdown, truth campaign. (n_fruit, kg/hr).
DENSITY = ((5, 11.2), (8, 13.9), (12, 11.1), (15, 13.2))

# farm/misses.py --shifts 5, seeds 700-704. Starts from every ripe fruit that
# was really there, not from the attempts the robot chose to make.
MISSES = {
    "truth": {"cmd": "farm/misses.py --shifts 5 --truth --seed 700",
              "ripe": 11, "clean": 11, "buckets": {}},
    "scouted": {"cmd": "farm/misses.py --shifts 5 --seed 700",
                "ripe": 11, "clean": 9, "buckets": {"not_mapped": 2}},
}


def assumptions(hours_per_day=20.0, days_per_week=5.0):
    return {
        "hours_per_day": Input(
            hours_per_day, "hr/day", ASSUMED,
            "(C) Business — Unit Economics & Pricing.md:62 ('~20-22 hr/day'), "
            "which itself has NO SOURCE"),
        "days_per_week": Input(
            days_per_week, "day/wk", ASSUMED,
            "LEMA (Wouter) 2026-07-17 Mon-Fri — said about HUMAN pickers"),
        "crate_swap_s": Input(
            0.0, "s", ASSUMED, "NOT MODELLED — no crate capacity exists"),
        "charge_s": Input(0.0, "s", ASSUMED, "NOT MODELLED — no battery exists"),
        "headland_s": Input(
            0.0, "s", ASSUMED,
            "NOT MODELLED — farm/trolley.py:183 runs one aisle and stops"),
    }


def derive(run, const):
    m = const["fruit_mass"].value
    if run["kind"] == "week4":
        rate = run["clean"] / run["attempts"]
        picks_hr = 3600.0 / run["cycle_mean_s"]
        return {"kg_hr": picks_hr * rate * m, "picks_hr": picks_hr,
                "rate": rate, "per_fruit_s": run["cycle_mean_s"] / rate,
                "cycle_s": run["cycle_mean_s"], "travel": False}
    total = run["scout_s"] + run["drive_s"] + run["pick_s"]
    per = total / run["crated"]
    return {"kg_hr": 3600.0 / per * m, "picks_hr": 3600.0 / per,
            "rate": None, "per_fruit_s": per, "total_s": total,
            "scout_share": 100 * run["scout_s"] / total,
            "drive_share": 100 * run["drive_s"] / total,
            "pick_share": 100 * run["pick_s"] / total,
            "travel": True}


def report(key="farm_scouted", hours_per_day=20.0, days_per_week=5.0):
    const, asm = read_constants(), assumptions(hours_per_day, days_per_week)
    run = RUNS[key]
    d = derive(run, const)

    print(f"\n{'=' * 78}")
    print(f"  kg/hr — ONE ARM — '{key}'")
    print(f"  commit {COMMIT[:7]} (dev) · measured {MEASURED_ON}")
    print(f"{'=' * 78}")
    print(f"\n  THE RUN\n    {run['cmd']}")
    print(f"    arm speed {run['arm_speed']}"
          f"{'  — NO TRAVEL in this scene' if not d['travel'] else ''}")

    print(f"\n  INPUTS")
    for n in ("fruit_mass", "arm_speed", "drive_speed", "snap_n", "row_len"):
        print(const[n].line(n))
    for n in ("hours_per_day", "days_per_week", "crate_swap_s", "charge_s",
              "headland_s"):
        print(asm[n].line(n))

    print(f"\n  THE ARITHMETIC")
    if not d["travel"]:
        print(f"    attempts                  {run['attempts']:>9}      MEASURED")
        print(f"    clean                     {run['clean']:>9}      MEASURED")
        print(f"    clean rate                {d['rate']:>9.3f}")
        print(f"    cycle t_total mean        {run['cycle_mean_s']:>9.2f} s    "
              f"MEASURED (park-to-park; picklog.py:25)")
        print(f"    cycle t_total p95         {run['cycle_p95_s']:>9.2f} s    "
              f"MEASURED")
        print(f"    3600 / cycle              {d['picks_hr']:>9.1f} picks/hr")
        print(f"    x clean rate x FRUIT_MASS")
        print(f"    = KG PER HOUR             {d['kg_hr']:>9.2f} kg/hr   "
              f"⚠️ picking only, no travel")
    else:
        print(f"    scouting                  {run['scout_s']:>9.1f} s    MEASURED")
        print(f"    driving                   {run['drive_s']:>9.1f} s    MEASURED")
        print(f"    picking                   {run['pick_s']:>9.1f} s    MEASURED")
        print(f"    = total machine-seconds   {d['total_s']:>9.1f} s")
        print(f"    / crated fruit            {run['crated']:>9}      MEASURED")
        print(f"    = s per crated fruit      {d['per_fruit_s']:>9.2f} s")
        print(f"    3600 / that               {d['picks_hr']:>9.1f} fruit/hr")
        print(f"    x FRUIT_MASS              {const['fruit_mass'].value:>9.3f} kg")
        print(f"    = KG PER HOUR             {d['kg_hr']:>9.2f} kg/hr")
        print(f"\n  WHERE THE TIME GOES")
        print(f"    scouting {d['scout_share']:5.1f}%   "
              f"driving {d['drive_share']:5.1f}%   "
              f"picking {d['pick_share']:5.1f}%")
        print(f"    trolley odometer          {run['odometer_m']:>9.1f} m")

    wk = d["kg_hr"] * hours_per_day * days_per_week
    print(f"\n  AT THE ASSUMED UPTIME (every term ASSUMED)")
    print(f"    x {hours_per_day:.0f} hr/day x {days_per_week:.0f} day/wk = "
          f"{wk:>8.0f} kg/week/robot   ({wk / 24000:.2f}x of the 24,000 target)")
    return d


def compare():
    """The point of v2: two unrelated scenes, one number."""
    const = read_constants()
    print(f"\n{'=' * 78}\n  TWO SCENES, ONE ARM — do they agree?\n{'=' * 78}")
    print(f"  {'scene':<34} {'cycle s':>9} {'kg/hr':>8}  what it includes")
    for key, note in (
            ("w4_truth", "picking only, told where fruit are"),
            ("w4_seen", "picking only, perception in the loop"),
            ("farm_truth", "+ driving, told where fruit are"),
            ("farm_scouted", "+ driving + scouting — THE FULL PIPELINE")):
        r, d = RUNS[key], derive(RUNS[key], const)
        cyc = r.get("cycle_mean_s") or (r["pick_s"] / r["crated"])
        print(f"  {key:<34} {cyc:>9.1f} {d['kg_hr']:>8.2f}  {note}")

    a = derive(RUNS["w4_seen"], const)["kg_hr"]
    b = derive(RUNS["farm_scouted"], const)["kg_hr"]
    print(f"\n  Full-pipeline agreement: {a:.2f} vs {b:.2f} kg/hr — "
          f"{100 * abs(a - b) / ((a + b) / 2):.1f}% apart,")
    print(f"  on different scenes, different arm speeds (0.15 vs 0.4) and "
          f"different\n  sample sizes (58 attempts vs 5 crated). "
          f"**Headline: ~12 kg/hr, one arm.**")
    print(f"\n  ⚠️ They agree for a reason, not by luck. The farm arm is 2.7x "
          f"faster at the\n     joints, and gives the whole advantage back to "
          f"scouting (34% of its clock)\n     and driving (13%).")

    print(f"\n{'=' * 78}\n  SNAP_N — the artefact everyone expects to matter"
          f"\n{'=' * 78}")
    print(f"  {'SNAP_N':>8} {'kg/hr':>8}")
    for n, kg in SNAP_SWEEP:
        print(f"  {n:>6} N {kg:>8.1f}")
    lo = min(k for _, k in SNAP_SWEEP)
    hi = max(k for _, k in SNAP_SWEEP)
    print(f"\n  {hi - lo:.1f} kg/hr of swing across 9-20 N = "
          f"{100 * (hi - lo) / lo:.0f}%. MEASURED, week4_snap.py --n 8.")
    print(f"  ⚠️ plant_row.py:57, bug log 8 and v1 of this note all say SNAP_N "
          f"'flows straight\n     into kg/hr'. It does not, inside this range. "
          f"The grip already loads the stem\n     6.0-8.6 N and a deliberate "
          f"pull exceeds 20 N, so the threshold never binds.")

    print(f"\n{'=' * 78}\n  CROP DENSITY — the axis week4_run.py exists to add"
          f"\n{'=' * 78}")
    print(f"  {'fruit placed':>13} {'kg/hr':>8}")
    for n, kg in DENSITY:
        print(f"  {n:>13} {kg:>8.1f}")
    print(f"\n  No trend. MEASURED across 5/8/12/15 fruit on one bolted row.")
    print(f"  ⚠️ But that scene has no travel and no scouting. In the farm "
          f"scene the scout\n     pass is a FIXED 20.0 s per shift whatever "
          f"the crop, so there density is the\n     whole game: 20 s over 2 "
          f"ripe fruit is 10 s/fruit, over 20 it is 1.")


def sensitivity(key="farm_scouted"):
    const = read_constants()
    run = RUNS[key]
    b = derive(run, const)["kg_hr"]
    print(f"\n{'=' * 78}\n  SENSITIVITY — +/-25% on one input at a time"
          f"\n{'=' * 78}\n  base {b:.2f} kg/hr ('{key}')\n")
    print(f"  {'input':<24} {'-25%':>9} {'base':>9} {'+25%':>9} {'swing':>9}  "
          f"class")

    rows = [("fruit mass", b * 0.75, b * 1.25, "READ / basis is GUESS")]

    d = derive(run, const)
    if run["kind"] == "farm":
        out = []
        for k in (1.25, 0.75):
            tot = run["scout_s"] + run["drive_s"] + run["pick_s"] * k
            out.append(3600.0 / (tot / run["crated"]) * const["fruit_mass"].value)
        rows.append(("cycle time", out[0], out[1], "MEASURED"))
        out = []
        for k in (0.75, 1.25):
            out.append(3600.0 / (d["total_s"] / (run["crated"] * k))
                       * const["fruit_mass"].value)
        rows.append(("success rate", out[0], out[1], "MEASURED"))
        out = []
        for k in (1.25, 0.75):
            tot = run["scout_s"] * k + run["drive_s"] + run["pick_s"]
            out.append(3600.0 / (tot / run["crated"]) * const["fruit_mass"].value)
        rows.append(("scouting time", out[0], out[1], "MEASURED"))
    else:
        rows.append(("cycle time", b / 1.25, b / 0.75, "MEASURED"))
        rows.append(("success rate", b * 0.75, b * 1.25, "MEASURED"))

    rows.append(("SNAP_N (9-20 N)", b, b, "READ — measured 0% effect"))

    for name, lo, hi, cls in rows:
        print(f"  {name:<24} {lo:>9.2f} {b:>9.2f} {hi:>9.2f} "
              f"{abs(hi - lo):>9.2f}  {cls}")

    worst = max(rows, key=lambda r: abs(r[2] - r[1]))
    print(f"\n  MOST SENSITIVE: {worst[0].upper()} "
          f"(swing {abs(worst[2] - worst[1]):.2f} kg/hr)")
    print(f"  ⚠️ fruit mass and success rate are pure multipliers and tie at "
          f"+/-25% by\n     construction. The tiebreak is which is worse known: "
          f"success rate is MEASURED\n     over 58 attempts; fruit mass is a "
          f"guess in a code comment. Ask a grower.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="farm_scouted", choices=tuple(RUNS))
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--hours", type=float, default=20.0)
    ap.add_argument("--days", type=float, default=5.0)
    args = ap.parse_args()

    report(args.config, args.hours, args.days)
    if args.compare:
        compare()
    if args.sensitivity:
        sensitivity(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
