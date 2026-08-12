#!/usr/bin/env python3
"""kg/hr for one Vinea robot, from numbers that were already measured.

⚠️ **This file runs no physics and measures nothing.** It is arithmetic over
figures taken from runs that are named, dated and committed, plus constants read
out of the code. Every input carries its class — MEASURED, READ or ASSUMED — and
the report prints the class next to the number, so a reader can see which step
to attack without re-deriving anything.

The point of keeping it separate from the runs: a throughput claim that is
computed inside the run that produced it cannot be checked against a different
run. This takes the numbers as data.

    ./.venv/bin/python simulation/mujoco/farm/throughput.py
    ./.venv/bin/python simulation/mujoco/farm/throughput.py --sensitivity
    ./.venv/bin/python simulation/mujoco/farm/throughput.py --config scouted
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The commit every MEASURED figure below was taken on, and the date.
COMMIT = "522a3f797f7a0e7d09d3c2843d41636a0b9d2db8"   # dev
MEASURED_ON = "2026-08-12"

MEASURED, READ, ASSUMED = "MEASURED", "READ", "ASSUMED"


@dataclass
class Input:
    """One number, and everything needed to attack it."""

    value: float
    unit: str
    cls: str
    source: str

    def line(self, name, width=34):
        return (f"  {name:<{width}} {self.value:>10.3f} {self.unit:<9} "
                f"{self.cls:<9} {self.source}")


# --- READ: constants that live in the code -----------------------------------

def read_constants():
    """Pulled live from the modules, so this file cannot drift from them."""
    from plant_row import FRUIT_MASS
    from farm import house, trolley
    from reach import CTRL_DT

    return {
        "fruit_mass": Input(
            FRUIT_MASS, "kg", READ,
            "simulation/mujoco/plant_row.py:30 FRUIT_MASS. Its real-world "
            "basis (plant_row.py:27-28 comment, '~120 g') is NO SOURCE — GUESS"),
        "drive_speed": Input(
            trolley.DRIVE_SPEED, "m/s", READ,
            "simulation/mujoco/farm/trolley.py:88 DRIVE_SPEED"),
        "ctrl_dt": Input(
            CTRL_DT, "s", READ,
            "simulation/mujoco/reach.py:59 CTRL_DT = DT * TICKS_PER_CTRL"),
        "row_len": Input(
            2 * house.HOUSE_HALF_Y, "m", READ,
            "simulation/mujoco/farm/house.py:89 HOUSE_HALF_Y x 2. ⚠️ chosen so "
            "routing is non-trivial, NOT a real Venlo bay length"),
        "rows_per_aisle": Input(
            2, "rows", READ,
            "simulation/mujoco/farm/house.py:131 serves() -> (i, i+1)"),
        "row_pitch": Input(
            house.ROW_PITCH, "m", READ,
            "simulation/mujoco/farm/house.py:81 ROW_PITCH"),
    }


# --- MEASURED: what came out of the runs -------------------------------------
#
# Each of these is one command, run once, on the commit above. The command is
# written out in full because "we measured 18.7 s" is not a measurement.

RUNS = {
    # ⚠️ The headline config. Three seeds pooled, because one shift is one
    # sample: the per-seed kg/hr spread is 13.5 to 16.5, and quoting whichever
    # end suits the argument is exactly the failure this note exists to stop.
    "duo_truth": {
        "cmd": ("for S in 7 11 23; do MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/two_arm_farm.py --headless --truth "
                "--seed $S -n 14 --arms 2; done"),
        "arms": 2,
        "crated": 6 + 5 + 4,
        "attempts": 8 + 7 + 5,
        "harvest_cycles": 13106 + 11231 + 10627,
        "odometer_m": 9.1 + 9.2 + 7.5,
        "scouted": False,
        "pick_s": [16.0, 17.7, 16.2, 20.3, 19.0, 22.9,
                   17.1, 17.0, 19.3, 18.0, 22.9,
                   23.5, 22.9, 23.0, 17.8],
        "speed": 0.5,
        "seeds": (7, 11, 23),
    },
    "duo_truth_s7": {
        "cmd": ("MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/two_arm_farm.py --headless --truth "
                "--seed 7 -n 14 --arms 2"),
        "arms": 2,
        "crated": 6,
        "attempts": 8,
        "harvest_cycles": 13106,
        "odometer_m": 9.1,
        "scouted": False,
        "pick_s": [16.0, 17.7, 16.2, 20.3, 19.0, 22.9],
        "speed": 0.5,
        "seeds": (7,),
    },
    "duo_scouted": {
        "cmd": ("MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/two_arm_farm.py --headless "
                "--seed 7 -n 14 --arms 2"),
        "arms": 2,
        "crated": 6,
        "attempts": 12,
        "harvest_cycles": 13121,
        "odometer_m": 19.5,
        "scouted": True,
        "pick_s": [16.1, 17.8, 16.2, 20.2, 19.0, 23.1],
        "speed": 0.5,
        "seeds": (7,),
    },
    "one_arm_truth": {
        "cmd": ("MUJOCO_GL=egl ./.venv/bin/python -u "
                "simulation/mujoco/two_arm_farm.py --headless --truth "
                "--seed 7 -n 14 --arms 1"),
        "arms": 1,
        "crated": 3,
        "attempts": 3,
        "harvest_cycles": 5248,
        "odometer_m": 7.5,
        "scouted": False,
        "pick_s": [16.8, 17.7, 18.0],
        "speed": 0.5,
        "seeds": (7,),
    },
}

# --- ASSUMED: nothing in the simulator supplies these ------------------------

def assumptions(hours_per_day=20.0, days_per_week=5.0):
    return {
        "hours_per_day": Input(
            hours_per_day, "hr/day", ASSUMED,
            "04 Business Case/(C) Business — Unit Economics & Pricing.md:62 "
            "('~20-22 hr/day'), which itself has NO SOURCE"),
        "days_per_week": Input(
            days_per_week, "day/wk", ASSUMED,
            "LEMA (Wouter) interview 2026-07-17, harvest cadence Mon-Fri — "
            "said about HUMAN pickers, not a robot"),
        "crate_swap_s": Input(
            0.0, "s/fruit", ASSUMED,
            "NOT MODELLED. No crate capacity constant exists in the sim; the "
            "crate never fills. NO SOURCE — GUESS"),
        "charge_s": Input(
            0.0, "s/fruit", ASSUMED,
            "NOT MODELLED. No battery exists in the sim. NO SOURCE — GUESS"),
        "headland_s": Input(
            0.0, "s/fruit", ASSUMED,
            "NOT MODELLED. trolley.y_limits() runs one aisle and stops; there "
            "is no turn and no aisle changeover. NO SOURCE — GUESS"),
    }


def derive(run, const):
    """Machine-seconds per crated fruit, and the kg/hr that follows.

    ⚠️ **Machine-seconds, not arm-seconds.** With the deck-centre interlock the
    two arms are serialised — the runs report 0% of harvest control cycles with
    both arms mid-mission — so summing two arms' pick times understates what the
    machine spent. The control-cycle count is what the machine actually took.
    """
    dt = const["ctrl_dt"].value
    harvest_s = run["harvest_cycles"] * dt
    drive_s = run["odometer_m"] / const["drive_speed"].value
    total_s = harvest_s + drive_s
    per_fruit = total_s / run["crated"]
    fruit_hr = 3600.0 / per_fruit
    kg_hr = fruit_hr * const["fruit_mass"].value
    return {
        "harvest_s": harvest_s, "drive_s": drive_s, "total_s": total_s,
        "per_fruit_s": per_fruit, "fruit_hr": fruit_hr, "kg_hr": kg_hr,
        "harvest_share": 100 * harvest_s / total_s,
        "drive_share": 100 * drive_s / total_s,
        "drive_s_per_fruit": drive_s / run["crated"],
        "success": 100 * run["crated"] / run["attempts"],
    }


def report(config="truth", hours_per_day=20.0, days_per_week=5.0):
    const = read_constants()
    asm = assumptions(hours_per_day, days_per_week)
    key = {"truth": "duo_truth", "scouted": "duo_scouted",
           "one_arm": "one_arm_truth", "seed7": "duo_truth_s7"}[config]
    run = RUNS[key]
    d = derive(run, const)

    print(f"\n{'=' * 78}")
    print(f"  kg/hr FOR ONE VINEA ROBOT — config '{config}'")
    print(f"  commit {COMMIT[:7]} (dev) · measured {MEASURED_ON}")
    print(f"{'=' * 78}")

    print(f"\n  THE RUN THIS IS BUILT ON")
    print(f"    {run['cmd']}")
    print(f"    {run['crated']} crated from {run['attempts']} attempts, "
          f"{run['arms']} arm(s), arm speed {run['speed']}")

    print(f"\n  INPUTS")
    print(f"  {'name':<34} {'value':>10} {'unit':<9} {'class':<9} source")
    for n in ("fruit_mass", "drive_speed", "ctrl_dt", "row_len"):
        print(const[n].line(n))
    for n in ("hours_per_day", "days_per_week", "crate_swap_s", "charge_s",
              "headland_s"):
        print(asm[n].line(n))

    print(f"\n  THE ARITHMETIC, LINE BY LINE")
    print(f"    harvest control cycles      {run['harvest_cycles']:>10,}      "
          f"MEASURED — the run's own counter")
    print(f"    x CTRL_DT                   {const['ctrl_dt'].value:>10.3f} s    "
          f"READ    reach.py:59")
    print(f"    = harvest machine-seconds   {d['harvest_s']:>10.2f} s")
    print(f"    trolley odometer            {run['odometer_m']:>10.2f} m    "
          f"MEASURED — state.drive_m")
    print(f"    / DRIVE_SPEED               {const['drive_speed'].value:>10.2f} m/s  "
          f"READ    trolley.py:88")
    print(f"    = drive seconds             {d['drive_s']:>10.2f} s")
    print(f"    total machine-seconds       {d['total_s']:>10.2f} s")
    print(f"    / crated fruit              {run['crated']:>10}")
    print(f"    = seconds per crated fruit  {d['per_fruit_s']:>10.2f} s")
    print(f"    3600 / that                 {d['fruit_hr']:>10.2f} fruit/hr")
    print(f"    x FRUIT_MASS                {const['fruit_mass'].value:>10.3f} kg   "
          f"READ    plant_row.py:30")
    print(f"    = {'KG PER HOUR':<25} {d['kg_hr']:>10.2f} kg/hr")

    wk = d["kg_hr"] * hours_per_day * days_per_week
    print(f"\n  AT THE ASSUMED UPTIME (all ASSUMED, none measured)")
    print(f"    x {hours_per_day:.0f} hr/day x {days_per_week:.0f} day/wk"
          f"        = {wk:>10.0f} kg/week/robot")
    print(f"    design target (04 Business Case) 24,000 kg/week  ->  "
          f"{wk / 24000:.2f}x of target")

    print(f"\n  WHERE THE TIME GOES")
    print(f"    picking   {d['harvest_share']:>5.1f}%   "
          f"driving {d['drive_share']:>5.1f}%")
    print(f"    drive seconds per crated fruit  {d['drive_s_per_fruit']:.2f} s")
    print(f"    attempt success                 {d['success']:.0f}%")
    return d


def sensitivity(config="truth"):
    """+/-25% on each of the three inputs a claim usually rests on."""
    const = read_constants()
    run = RUNS[{"truth": "duo_truth", "scouted": "duo_scouted",
                "one_arm": "one_arm_truth", "seed7": "duo_truth_s7"}[config]]
    base = derive(run, const)
    b = base["kg_hr"]

    print(f"\n{'=' * 78}\n  SENSITIVITY — +/-25% on one input at a time\n"
          f"{'=' * 78}")
    print(f"  base {b:.2f} kg/hr\n")
    print(f"  {'input':<22} {'-25%':>10} {'base':>10} {'+25%':>10} "
          f"{'swing':>10}")

    rows = []

    # Fruit mass is a pure multiplier.
    lo, hi = b * 0.75, b * 1.25
    rows.append(("fruit mass", lo, hi))

    # Cycle time: scales the harvest half only, driving is untouched.
    for label, f in (("cycle time", None),):
        out = []
        for k in (1.25, 0.75):          # +25% slower cycle -> less kg/hr
            h = base["harvest_s"] * k
            per = (h + base["drive_s"]) / run["crated"]
            out.append(3600.0 / per * const["fruit_mass"].value)
        rows.append((label, out[0], out[1]))

    # Success rate: fewer crated fruit for the same machine-seconds.
    out = []
    for k in (0.75, 1.25):
        crated = run["crated"] * k
        per = base["total_s"] / crated
        out.append(3600.0 / per * const["fruit_mass"].value)
    rows.append(("success rate", out[0], out[1]))

    for name, lo, hi in rows:
        print(f"  {name:<22} {lo:>10.2f} {b:>10.2f} {hi:>10.2f} "
              f"{hi - lo:>10.2f}")

    worst = max(rows, key=lambda r: abs(r[2] - r[1]))
    print(f"\n  most sensitive to: {worst[0].upper()} "
          f"(swing {abs(worst[2] - worst[1]):.2f} kg/hr)")
    print(f"  ⚠️ fruit mass and success rate are both pure multipliers, so they "
          f"tie by\n     construction at +/-25%. Cycle time moves less because "
          f"driving does not\n     scale with it. The tie is real and the "
          f"tiebreak is which input is worse\n     known — fruit mass is a "
          f"GUESS, success rate is MEASURED.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="truth",
                    choices=("truth", "scouted", "one_arm", "seed7"))
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--hours", type=float, default=20.0)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--all", action="store_true",
                    help="every config, side by side")
    args = ap.parse_args()

    if args.all:
        for c in ("one_arm", "truth", "scouted"):
            report(c, args.hours, args.days)
        return 0

    report(args.config, args.hours, args.days)
    if args.sensitivity:
        sensitivity(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
