# Parked: the Vinea cradle-and-blade gripper

Not abandoned — **early**. Parked deliberately on 2026-08-03.

## What is in here

| | |
|---|---|
| `vinea_gripper.py` | The MVP's own end effector: a passive cradle that supports a truss from below, plus a blade that severs the peduncle. One moving part against the Robotiq 2F85's eight. |
| `week2_vinea.py` | The same plant row picked with it — `approach → insert → CUT → lift → carry → tip out`, with no grip and no pull. |

## Why it is parked

The design is good and the argument for it still holds. `SNAP_N` — the force
at which a peduncle lets go — cannot be chosen freely with the 2F85, because
closing the gripper on an attached fruit loads the stem by 6.0-8.6 N all by
itself, which puts a floor under the threshold. A blade removes that coupling
entirely: it cuts rather than tears, so the detach force stops being a property
of the gripper.

None of which matters yet. Week 2's actual problem turned out to be that the
arm was knocking neighbouring tomatoes off the plant on 8 of 10 picks while
reporting 10/10, and that is a *motion* problem — it is fixed by planning the
route and checking it clears the crop, not by changing the tool. Building a
second end effector before the first one could complete a cycle without
stripping the row would have been solving the wrong problem carefully.

So: get picking clean with the stand-in gripper, close the perception loop,
produce a defensible kg/hr. Then design the tool, with real numbers to design
against.

## Reviving it

These files were written against the pre-planner Week 2 and will not import as
they stand. `week2_vinea.py` pulls its cycle constants from `week2_pick`, and
those moved to `mission.py` when the planner landed. The work to revive it is:

1. Point the imports at `mission.py`.
2. Re-measure `MAX_REACH_VINEA` — reach is measured, not derived from the mount
   offset, and the tool has changed.
3. Express the cycle as a `mission.Leg` list. The cradle's shape genuinely
   differs from the 2F85's (there is no close and no pull, and the tip-out
   replaces the release), so it needs its own leg sequence rather than a
   parameter on the existing one.
4. Keep `ROLL_COST_CRADLE` high. A cradle cares completely which way up it is —
   left free, the solver rotated the tool 90°, the channel floor acted as a side
   wall, and the fruit was clipped on the way in at 17.4 N.
