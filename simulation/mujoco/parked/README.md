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

**Step 1 is done as of 2026-08-12** — `week2_vinea.py` imports and runs again,
because bug 40 needed it to. Eleven of its thirteen borrowed constants were in
`mission.py`; the two that were not, `SETTLE_S` and `TURN_MAX_S`, are in
`legacy_cycle.py`, which is where this file's own flat state machine belongs
too. It also needed the parent directory on `sys.path`, which it had never
needed while `parked/` *was* `simulation/mujoco/`.

It reproduces its recorded numbers exactly: **5/5 cut, 4/5 carried, 2/5 crated.**

Still to do:

2. Re-measure `MAX_REACH_VINEA` — reach is measured, not derived from the mount
   offset, and the tool has changed. **Not done**; every number below was taken
   with the recorded 1.050 m and none of them depends on it.
3. Express the cycle as a `mission.Leg` list. The cradle's shape genuinely
   differs from the 2F85's (there is no close and no pull, and the tip-out
   replaces the release), so it needs its own leg sequence rather than a
   parameter on the existing one. **Not done** — and it is why `cradletrace.py`
   exists instead of `carrytrace.py` being pointed at this tool.
4. Keep `ROLL_COST_CRADLE` high. A cradle cares completely which way up it is —
   left free, the solver rotated the tool 90°, the channel floor acted as a side
   wall, and the fruit was clipped on the way in at 17.4 N.

## `cradletrace.py` — bug 40's experiment

```bash
# 🪟 WATCH one, at real speed
./.venv/bin/python simulation/mujoco/parked/cradletrace.py --windowed

# 📝 t1 at its own 3.9 mm of perception error — the bug 40 case
./.venv/bin/python simulation/mujoco/parked/cradletrace.py --err 3.9

# 📝 0 / 3.9 / 8 / 15 / 25 mm — one clean run at one error is not a result
./.venv/bin/python simulation/mujoco/parked/cradletrace.py --sweep
```

**The cradle does not eject.** Twenty runs, five fruit, error 0 to 40 mm: on
every fruit the tool actually cradles, the fruit stays within **20 mm** of the
tool point through every carrying leg. The 2F85 on the same fruit at 3.9 mm
runs **6 → 784 mm** inside one carry and launches. The cradle's number is flat
against error, which is what "not an instability" looks like.

⚠️ It is **not** a better tool on the numbers that matter commercially: crate
rate is 8/20 here, the same 40% the row run reports. It does not throw fruit;
it drops it. Those are different problems and only the first one is the safety
one.

Nothing in here is imported by the shipped cycle, and the Robotiq milestone is
untouched. That is still the point of this folder.
