#!/usr/bin/env python3
"""One row per pick attempt, appended to a file, never overwritten.

Fifty picks is seventy-odd minutes of unattended running, and a crash at pick 43
must not cost 43 picks. So rows are flushed as they happen rather than written at
the end, and the file is JSONL — one JSON object per line — so a partial file is
still a valid file and can be re-read on a later day for a different cut of the
same data without re-running the physics.

⚠️ **Cycle time was the thing missing.** `execute()` computes and returns
`seconds`, and `week3_perceive.harvest()` built its log row without it. That one
dropped field is the input to seconds-per-pick, which is the input to kg/hr.

⚠️ **And the cycle time it returns is not the whole cycle.** Three things:

    1. It is *simulated* time (`clock += CTRL_DT`), not wall clock. That is the
       right choice for kg/hr — it is what a real arm commanded at that speed
       would take, not how fast this laptop runs physics — but it has to be said
       out loud, because "28 seconds" invites the assumption of a stopwatch.
    2. It excludes the look. The stage move, the render and the detector all
       happen before `execute` starts its clock, and all of them are part of a
       real cycle.
    3. It excludes planning (~0.15 s against 28 s — nearly noise, but not zero).

`t_total` here is park-to-park and covers all three. That is the number that
divides into 3600.

    ./.venv/bin/python simulation/mujoco/picklog.py runs/dense15.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from outcomes import classify  # noqa: E402

FIELDS = (
    # provenance — enough to reproduce the exact attempt
    "run_id", "timestamp", "seed", "layout", "n_fruit", "detector", "speed",
    "clearance_mm",
    # what was aimed at
    "fruit", "seen", "err_mm", "err_x", "err_y", "err_z", "associated_to",
    # what happened
    "outcome", "in_bin", "grasped", "broke", "aborted", "refused_reason",
    "lost", "disturbed", "clearance_min_mm", "peak_n", "peak_held_ms",
    "fruit_end_xyz",
    # what it cost
    "t_scan", "t_plan", "t_fly", "t_total",
)


class PickLog:
    """Append-only JSONL. Open it, add rows, and they are on disk immediately."""

    def __init__(self, path=None, run_id=None, meta=None):
        self.path = Path(path) if path else None
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%S")
        self.meta = dict(meta or {})
        self.rows = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, **kw):
        """Record one attempt. Missing fields are written as None, not omitted.

        Writing the absent ones explicitly keeps every row the same shape, so a
        reader never has to guess whether a missing key means "no" or "not
        measured" — a distinction that matters for `peak_held_ms`, where absent
        means the ejection split could not be made.
        """
        row = {"run_id": self.run_id,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **self.meta, **kw}
        row["outcome"] = row.get("outcome") or classify(row)
        full = {k: row.get(k) for k in FIELDS}
        # Anything the caller passed that is not a known field is kept rather
        # than dropped, so an experiment can add a column without editing FIELDS.
        for k, v in row.items():
            if k not in full:
                full[k] = v
        self.rows.append(full)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(full, default=_plain) + "\n")
        return full

    def close(self):
        if self.path:
            print(f"  log: {len(self.rows)} rows -> {self.path}")


def _plain(o):
    """numpy scalars and arrays are not JSON, and silently is not an option."""
    import numpy as np

    if isinstance(o, np.ndarray):
        return [float(x) for x in o.ravel()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bool,)):
        return bool(o)
    return str(o)


def load(path):
    """Read a JSONL log back. Tolerates a truncated final line from a crash."""
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ⚠️ skipping a truncated final line — the run was cut "
                      f"short, the rows before it are intact")
    return rows


def throughput(rows, fruit_kg=0.12, hours_per_week=168):
    """The Week 4 deliverable: picks/hr, kg/hr, kg/week, from measured rows.

    Cycle time is taken over attempts that actually flew — an attempt refused in
    0.15 s of planning is not a 0.15 s pick, and averaging it in would flatter
    the number. Refusals cost throughput by not producing fruit, which is already
    counted through the clean rate.
    """
    flown = [r for r in rows if r.get("t_total")]
    if not flown:
        return None
    n = len(rows)
    clean = sum(1 for r in rows if r.get("outcome") == "clean")
    secs = [r["t_total"] for r in flown]
    mean_s = sum(secs) / len(secs)
    picks_hr = 3600.0 / mean_s
    rate = clean / n if n else 0.0
    kg_hr = picks_hr * rate * fruit_kg
    return {
        "attempts": n, "flown": len(flown), "clean": clean, "clean_rate": rate,
        "mean_s": mean_s, "min_s": min(secs), "max_s": max(secs),
        "picks_hr": picks_hr, "kg_hr": kg_hr,
        "kg_week": kg_hr * hours_per_week,
    }


def main():
    from outcomes import report

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit("usage: picklog.py <log.jsonl>")
    rows = load(sys.argv[1])
    report(rows, Path(sys.argv[1]).name)
    t = throughput(rows)
    if t:
        print(f"\n  cycle {t['mean_s']:.1f} s mean "
              f"({t['min_s']:.1f}-{t['max_s']:.1f})")
        print(f"  {t['picks_hr']:.0f} picks/hr x {t['clean_rate'] * 100:.0f}% "
              f"clean x 0.12 kg = {t['kg_hr']:.1f} kg/hr")
        print(f"  -> {t['kg_week']:.0f} kg/week/robot at 24/7, single arm")


if __name__ == "__main__":
    main()
