#!/usr/bin/env python3
"""Every pick attempt lands in exactly one named bucket, decided by the code.

Week 4's deliverable is a defensible kg/hr, and a success rate with no diagnosis
behind it is not defensible — the first person who asks "why 78%?" ends the
conversation. Fifty picks is over an hour of unattended running, so the log has
to carry the diagnosis rather than a human watching.

Two rules, both about honesty:

  * **Exactly one bucket per attempt.** Where an attempt could be two, the
    earliest break wins — an attempt that was refused never got the chance to
    fail at the grasp. `classify` walks them in that order deliberately.

  * **UNKNOWN must stay empty.** If it fires, the taxonomy is incomplete and
    something has been found. Do not fold a surprise into the nearest bucket.

⚠️ CARRY_LOST splits into a drop and an **ejection**, and the split is not
cosmetic. Bug 40: a 3.9 mm position error puts `t1` on the floor two metres away
at 2.3 m/s. A dropped fruit is lost revenue; a fruit leaving the gripper at
speed is a projectile in a building with people and glass in it. `in_bin: False`
prints identically for both, which is why it went unseen from Week 2 to Week 3.

⚠️ The discriminator is **peak speed while held**, not where the fruit ended up.
End position was tried first and is a trap: an ejected fruit keeps sliding on the
floor for the rest of the run, so the same failure reads as 2 m on a single pick
and 10 m on a five-fruit row. Peak held speed is time-invariant. Measured on this
row: 0.42-0.65 m/s on picks that crate, 2.3-2.5 m/s on the one that ejects.

    ./.venv/bin/python simulation/mujoco/outcomes.py     # the buckets, explained
"""

from __future__ import annotations

# --- the buckets -------------------------------------------------------------

NOT_DETECTED = "not_detected"    # the detector never put a box on it
MISASSOCIATED = "misassociated"  # a box was matched to the wrong truss
UNREACHABLE = "unreachable"      # outside the envelope, or IK could not get there
REFUSED = "refused"              # no route cleared the crop; the planner declined
GUARD_ABORT = "guard_abort"      # flown, then abandoned mid-flight at GUARD_STOP
GRASP_FAILED = "grasp_failed"    # reached it, and never held it or never detached
CARRY_DROPPED = "carry_dropped"  # held, detached, released short of the crate
CARRY_EJECTED = "carry_ejected"  # held, detached, left the gripper at speed
COLLATERAL = "collateral"        # crated, but something else came off the plant
CLEAN = "clean"                  # in the crate, nothing else touched
UNKNOWN = "unknown"              # the taxonomy is incomplete. Must never fire.

ORDER = (CLEAN, NOT_DETECTED, MISASSOCIATED, UNREACHABLE, REFUSED, GUARD_ABORT,
         GRASP_FAILED, CARRY_DROPPED, CARRY_EJECTED, COLLATERAL, UNKNOWN)

FAILURES = tuple(b for b in ORDER if b != CLEAN)

WHY = {
    CLEAN: "in the crate, nothing else touched",
    NOT_DETECTED: "the detector never saw it — lost revenue, nothing worse",
    MISASSOCIATED: "a box was matched to the wrong truss",
    UNREACHABLE: "outside the arm's envelope, or IK could not get there",
    REFUSED: "no route cleared the crop — the designed-in safe failure",
    GUARD_ABORT: "abandoned mid-flight at the guard threshold, on purpose",
    GRASP_FAILED: "reached it and did not come away with it",
    CARRY_DROPPED: "held and detached, then dropped short of the crate",
    CARRY_EJECTED: "LEFT THE GRIPPER AT SPEED — a projectile, not a drop",
    COLLATERAL: "crated, but knocked another fruit off the plant",
    UNKNOWN: "the taxonomy is incomplete — read the row and add a bucket",
}

# Above this, a fruit that is supposed to be gripped is not being gripped any
# more. See the module docstring for why this and not end position.
EJECT_MS = 1.0


def classify(r):
    """One attempt -> one bucket. `r` is a log row; missing keys are tolerated.

    Order is the order the loop can break in, earliest first. That is what makes
    the bucket unique: a refusal is decided before a grasp is ever attempted, so
    a refused attempt cannot also be a grasp failure.
    """
    if not r.get("seen", True):
        return NOT_DETECTED
    if r.get("misassociated"):
        return MISASSOCIATED
    if r.get("unreachable"):
        return UNREACHABLE
    if r.get("refused"):
        return REFUSED
    if r.get("aborted"):
        return GUARD_ABORT

    grasped = r.get("grasped", False)
    broke = r.get("broke", r.get("stem", False))
    if not (grasped and broke):
        return GRASP_FAILED

    if not r.get("in_bin", False):
        # The one place the split matters. Absent a trace, fall back to dropped
        # rather than inventing an ejection — under-claiming a safety finding is
        # the safe direction to be wrong in.
        peak = r.get("peak_held_ms")
        if peak is not None and peak > EJECT_MS:
            return CARRY_EJECTED
        return CARRY_DROPPED

    if r.get("lost", 0) or r.get("disturbed", 0):
        return COLLATERAL
    if r.get("clean", True):
        return CLEAN
    return UNKNOWN


def tally(rows):
    """{bucket: count}, every bucket present even at zero.

    Zeros are kept on purpose. A bucket that never fires reports 0 rather than
    being absent, because "we never saw this" and "we cannot see this" are
    different claims and a missing key hides the difference.
    """
    out = {b: 0 for b in ORDER}
    for r in rows:
        out[r.get("outcome") or classify(r)] += 1
    return out


def report(rows, title="outcomes"):
    """Print the breakdown. Returns (clean, total)."""
    n = len(rows)
    counts = tally(rows)
    clean = counts[CLEAN]
    print(f"\n  {'-' * 66}")
    print(f"  {title}: {n} attempts")
    for b in ORDER:
        c = counts[b]
        if c == 0 and b not in (CLEAN, UNKNOWN):
            continue
        flag = ""
        if b == UNKNOWN and c:
            flag = "   <-- TAXONOMY INCOMPLETE"
        if b == CARRY_EJECTED and c:
            flag = "   <-- safety, not yield"
        print(f"    {b:<15} {c:>4}   {WHY[b]}{flag}")
    if n:
        print(f"\n  CLEAN {clean}/{n}  ({100 * clean / n:.0f}%)")
    return clean, n


def main():
    print(__doc__)
    print("  the buckets, in the order an attempt can break:\n")
    for i, b in enumerate(ORDER, 1):
        print(f"    {i:>2}. {b:<15} {WHY[b]}")
    print(f"\n  ejection threshold: {EJECT_MS} m/s peak while held")
    print("\n  worked example — the Week 3 row, as classified:")
    rows = [
        {"fruit": "t0", "seen": True, "grasped": True, "broke": True,
         "in_bin": True, "clean": True},
        {"fruit": "t1", "seen": True, "grasped": True, "broke": True,
         "in_bin": False, "clean": False, "peak_held_ms": 2.32},
        {"fruit": "t2", "seen": True, "grasped": True, "broke": True,
         "in_bin": True, "clean": True},
        {"fruit": "t3", "seen": True, "grasped": True, "broke": True,
         "in_bin": True, "clean": True},
        {"fruit": "t4", "seen": True, "grasped": True, "broke": True,
         "in_bin": True, "clean": True},
    ]
    for r in rows:
        print(f"    {r['fruit']}  ->  {classify(r)}")
    report(rows, "week3 row")


if __name__ == "__main__":
    main()
