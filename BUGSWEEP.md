# Bugsweep — morning report

Branch `bugsweep`, off `dev` at `522a3f7`. One commit per bug. Nothing merged,
nothing pushed, PR #7 untouched.

> **The sprint deliverable is a validated kg/hr and it is not produced. None of
> the work below is that deliverable.** You said you were choosing this
> knowingly; this line is here because you asked for it to be.

---

## 1. Contradictions — things that do not hold up against what the log claims

*This section is written first and updated as I go, because you asked for it
first. If a bug's stated cause turned out to be wrong, it is here rather than
in the per-bug detail below.*

### C1. Two of the four numbers I was told to hold constant cannot be reproduced

The protocol says Week 1's **42/42** and **6/6** must come back byte-identical
after bug 7. Neither has a command in this repo.

- **42/42** is from the build log — "42/42 mouse picks" — a hand-run count of 42
  double-clicks in an interactive viewer session. `week1_mousereach.py` has no
  sweep mode; `--click` takes points one at a time and the only counter is the
  `N/N tomatoes picked and crated` line at the end of whatever you passed it.
- **6/6** is the six-position table sweep from entry 23's diagnosis
  ("Move it clear and Week 1's own code scores 6/6"). `week1_gripper.py` picks
  exactly one hardcoded tomato at `TOMATO_POS`; there is no six-position sweep
  anywhere in the repo. `grep -rn "sweep" simulation/mujoco/` finds one hit, in
  an unrelated module.

Both were real measurements. Both were run in a terminal that is now closed,
which is bug 6's own sentence about itself. **This is not a reason to skip the
bar** — it is a reason the bar needed building first, and it is the strongest
single argument for having done bug 6 before anything else.

What I am holding bug 7 to instead: a byte-for-byte diff of the full stdout of
both demos over a fixed set of points, captured on the commit before the
refactor and again after. That is a stricter test than a crated count — it
compares every waypoint, every arrival error in millimetres and every snap
force, not just the final tally. Baselines are committed under
`tests/baselines/` so the comparison is re-runnable rather than being another
number in a closed terminal.

### C2. Bug 4 does not fail "on every successful move" — it fails on half

The entry says mounting a gripper on `week1_targetreach`'s scene makes it
"report failure on every successful move". Measured, with a 2F85 on, over ten
targets spread across the workspace: the steady-state droop is **3.70 to
6.05 mm**, and the constant it is compared against is 5.0. So it *straddles*.
The interactive loop never reports arrival at **5 of the 10**, and reports it
late at the other five (0.86 s vs 0.70 s at the default target).

The diagnosis is right and the fix is right; the severity is overstated by 2x.
Worth correcting because "always fails" and "fails half the time, depending
where you point it" are different bugs to hunt — the second one gets blamed on
the arm.

Entry 14 already contains the reason, in its own words: the gripper's droop
"straddl[es] the hardcoded 5 mm". That was written down and then not applied
to the one place that had copied the arrival test out of `drive_to`.

### C3. Bug 58's repro grep misses half of what bug 58 lists, and the list is short by one

The entry's repro is:

```
grep -n 'site("tool0")\|body("wrist3_link")' simulation/mujoco/*.py
```

It matches **double quotes only**. Three of the five modules the entry names —
`greenhouse.py:651`, `camera.py`'s print helpers, and (see below) — write
`data.site('tool0')` with single quotes and do not appear in its output. Anyone
running the entry's own repro would conclude the entry overstated itself.

Worse: there is a **sixth** module, `week3_perceive.py:726`, which the entry
does not name. It is the pose caption on the Week 3 deprojection gate — the
0.39 mm number. Found by re-running the repro with both quote styles:

```
grep -rn "site('tool0')\|site(\"tool0\")" simulation/mujoco/
```

### C4. Two demos print a scene size that is not reproducible

`camera.py` and `greenhouse.py` both print a geom count, and it changes run to
run on **unchanged code**: camera.py gave 803 / 800 / 794 on three consecutive
runs, greenhouse.py 786 / 792 / 786. The decorative foliage is placed unseeded.

Nothing depends on it — `collidable geoms: 34` is identical every time, and
that is the number physics cares about — but it does mean any before/after diff
of those two files' stdout shows a spurious difference forever, and a geom
count quoted from either one is not a fact anyone can check. Noted rather than
fixed: seeding it is a change to a scene, and no bug in the log asks for it.

### C6. Bug 40's "six metres outside the greenhouse" is mostly a fruit rolling

This is the biggest one, and it changes how the entry should be argued.

The entry's headline is that `t1` "ends the run at `[6.04, -8.99, 0.03]` — six
metres outside the greenhouse", and the severity case is built on it: "a fruit
leaving the gripper at speed is a projectile in a building with people and
glass in it."

**`z = 0.03` is the floor.** The fruit is not in flight at that position; it is
lying on the ground. And it is still *moving*: at the last sample of the trace
it is doing **0.58 m/s with zero deceleration**, having travelled in a straight
line at exactly that speed for the previous 2.5 seconds. The run ends because
the *cycle* ends, not because the fruit stopped.

The reason it never stops is bug **41**, in a second place nobody connected it
to. Entry 41 established that `geom_condim = 3` makes the pads' rolling
friction coefficient unreachable. The **fruit and the floor are also condim 3**
— `t1_geom` `friction=[0.6, 0.01, 0.001]`, `floor` `friction=[1.0, 0.005,
1e-4]` — so a sphere that reaches the floor has no rolling resistance available
to it at any coefficient, and rolls at constant speed until something stops it.
Confirmed independently on the cradle's `t3`, which rolls **3.79 m at a
constant 0.20 m/s**.

So the distance is a function of how long the run happened to continue. **What
this does not change:** the peak speed while held, 1.54–2.32 m/s against 0.42
for a clean pick, is real, is measured while the fruit is still in the gripper,
and is the number the safety argument should rest on. The mechanism is real.
Only the six metres is not evidence for it.

### C7. Bug 40's own numbers have moved, and only the unstable one

Re-run today, three consecutive times, identical each time:

| run | log, 2026-08-04 | today |
|---|---|---|
| `t1` ground truth — crates | 0.42 m/s, 25 mm | **0.42 m/s**, 44 mm |
| `t3` perception — crates | 0.65 m/s, 93 mm | **0.64 m/s**, 53 mm |
| `t1` perception — **ejects** | 2.32 m/s, 2046 mm | **1.54 m/s, 1255 mm** |

The two runs that do not eject reproduce to 0.01 m/s. The one that does has
moved by a third. That is not a regression — it is what an unstable event looks
like next to stable ones, and it is *additional evidence* for the entry's own
"geometric instability" conclusion rather than against it. Something between
2026-08-04 and now (the two-arm work, entries 63–68) perturbed the run enough
to change where the instability lands, and did not touch the stable ones at
all.

⚠️ It does mean **the 2.32 m/s and the 2046 mm should not be quoted again**
without re-running.

### C8. The one-sided pad unload is real, and the per-leg table cannot show it

Confirmed in the CSV, and it is the entry's diagnosis holding up: the left pad
decays **63 → 0 N over 12 control cycles** while the right holds ~112 N, the
fruit walks **1.3 → 7.8 mm** along tool x, and then it goes.

But `carrytrace`'s own summary table reports **145.4 / 144.5 N** for the carry
leg — near-symmetric, because it prints the per-leg *peak* and each pad peaks
at a different moment. The instrument that found this bug cannot display its
signature in its default output; you have to take the `--csv` and look. Worth a
line in the entry so the next person does not read the table and conclude the
pads were balanced.

### C5. The mp4 encoder is not deterministic, so video is not a regression bar

Two runs of *identical* code produce different `.mp4` hashes. I nearly used a
video hash as the byte-identical check for bug 3 and it would have reported a
regression that was not there. Every comparison in this report is against
printed numbers or dumped state, never a rendered artifact.

### C2b. Bug 3's line numbers have drifted

The entry names `week1_reach.py:152` and `fr5.py:503`. The calls are at
`week1_reach.py:158` and `fr5.py:543`. Both are the right call in the right
function (`main()` in each), so the diagnosis is intact — recording it only
because the entry reads as though it were checked recently and it was not.

---

## 2. What I would do next, ranked

**0. Produce kg/hr.** Not on this list properly, because it is not bug work and
you already know. Everything below is worth less than it.

**1. Re-quote bug 40 without the six metres, and keep the peak speed.** C6 is a
correction to the most quotable number in the repo, and the number is in
CLAUDE.md, the README and the bug log. The mechanism survives intact and the
safety argument survives on peak-speed-while-held; what does not survive is
"six metres outside the greenhouse", which is a rolling artefact of condim 3.
Fix it before it reaches a grower or an investor deck, because it is exactly
the kind of detail someone technical will pull on.

**2. Decide what the cradle result buys.** It does not eject, flat across a
0–40 mm error range. That closes the product argument the entry wanted closed —
with the caveats in §3 attached, especially that its crate rate is no better.
The honest framing is "the cradle removes the projectile failure mode and does
not yet remove the drop failure mode", and that is a stronger, more defensible
claim than the one the log was reaching for.

**3. Give bug 41 a second entry, or widen the one it has.** Its finding is not
about pads. `condim = 3` on the fruit and the floor is why the ejection metric
was meaningless, and nobody had connected those. Cheap to write down, and it
retires a class of "why did the fruit end up over there" questions.

**4. Decide the transit waypoint for `week1_gripper`** (entry 17). Measured: 2
of 23 start postures stall, worst 201 mm. It is a real latent bug and a
one-line fix, but the fix needs a transit point chosen for that scene, which is
a tuned constant and therefore yours. My recommendation is to add it — 2/23 is
not rare — but not to reuse the row scene's point.

**5. Sweep the `j1` version of bug 58.** `farm/trolley.py:147` records that
`week2_pick` and `carrytrace` also look up `j1` by bare name. `tool_pos` closed
the tool-site instance of this shape; the shape has now appeared three times
(55, 68, 58) and the comment says where the fourth is.

**6. Seed the decorative foliage** (C4), or stop printing the geom count. Low
value, five minutes, and it makes two demos' output diffable.

**Not recommended:** touching the contact model, `SNAP_N`, or anything on your
forbidden list. Nothing this run found argues for it, and bug 40's measurement
still says no constant is the answer.

---

## 3. Per-bug detail

### Bug 6 — there are no tests — **DONE**, commit `242bec5`

**What it was.** `tests/` contained `.gitkeep` and nothing else. Every check
through five build weeks was ad-hoc and unrepeatable.

**Did it reproduce.** Yes, trivially — `ls tests/` is the repro.

**What I built.** `tests/test_sim.py`, in `scripts/phase0_smoketest.py`'s style
(a `@check` decorator, a PASS/FAIL table, non-zero exit). No test framework;
the repo has none and pytest is not even installed in `.venv`. Five checks, the
four the entry names plus one that bug 3 needs:

| check | what it asserts | result |
|---|---|---|
| a pick completes end to end | `week1_gripper.run_pick` grasps *and* crates | fruit at `[0.328, -0.619, 0.049]` |
| weld holds and releases | held < 1 mm after 1 s of gravity, reads its own 1.18 N, falls once `eq_active = 0` | held **0.367 mm** at **1.18 N**, fell 567 mm |
| tool0 on the pinch site | `tool0` and `gr_pinch` coincide, at three postures not one | worst gap **0.000 mm** |
| reset_home restores a free body | spawn pose restored, **and the keyframe does not restore it** | 0.0 µm vs the keyframe's **747 mm** |
| reset_home on the keyframe scenes | `week1_reach`'s and `fr5.main`'s scenes come back at HOME | both |

**Runtime 1.06 s.** That was the binding constraint — a suite that is not run
before every commit is not a suite — and it is affordable only because a
headless pick in this repo takes under a second of wall clock.

Three decisions worth arguing with:

- **The tool-frame check runs at three postures, not one.** Two frames on the
  same rigid body agree everywhere or nowhere, so one check at home cannot
  distinguish "the offset is right" from "the offset is wrong along an axis
  home happens to hide".
- **The reset check asserts the broken behaviour too.** It requires
  `mj_resetDataKeyframe` to still strand the fruit at the origin. If MuJoCo
  ever changes its short-keyframe padding, that assertion is the only thing in
  the repo that will notice, and entries 3 and 11 both become wrong quietly.
- **Nothing asserts a headline number.** 42/42, 10/10, 0.39 mm and 20/20 stay
  in the build log where their assumptions are written next to them. Pinning
  them into an assertion makes every honest change to the physics a failing
  build, and teaches people to edit the expected value.

**Numbers re-run.** None yet — this commit adds files and changes no code path.
`tests/test_sim.py` 5/5, `scripts/phase0_smoketest.py` unaffected.

**Same defect alive elsewhere.** No. But note that this closes bug 6 only in
the sense the entry asks for — Weeks 2 through 5 remain untested, and the four
mechanisms here are Week 1's. That is the entry's own scope, not a shortcut.

---

### Bug 3 — two scenes reset via the keyframe — **DONE**, commit `7795edc`

**What it was.** `week1_reach.py` and `fr5.main()` reset through the `home`
keyframe. `spec.add_key` stored a six-number vector — the bare arm, before
there was a gripper or a tomato — and MuJoCo pads a short keyframe with zeros
rather than each body's spawn pose.

**Did it reproduce.** Yes. Added a free body at `[0.55, 0.15, 0.50]` to both
named scenes; both reset it to `[0, 0, 0]`, **758.3 mm** from spawn, silently.
`reset_home` puts it back to 0.0 mm in both.

**What I changed.** The two calls, plus the comment explaining why. Entry 11
fixed once and left alive in two places.

**Numbers re-run.**

| check | result |
|---|---|
| `week1_reach` full trajectory, 1500 IK steps at float precision | **bit-identical** — post-reset qpos, ctrl, mocap, and every logged step |
| `fr5.py` and `fr5.py --gripper` stdout | identical: tool pos, reach, and the 0.00 mm pinch gap |
| `tests/test_sim.py` | 5/5 |

**Same defect alive elsewhere.** No — `grep -rn mj_resetDataKeyframe` over
`simulation/`, `scripts/` and `tests/` now returns only comments and the test
that asserts the broken behaviour on purpose.

---

### Bug 4 — targetreach reads the module constant — **DONE**, commit `3f6362a`

**What it was.** `week1_targetreach.py`'s interactive loop re-implements
`drive_to`'s arrival test by hand and compares against `reach.REACHED_MM`
instead of `reacher.reached_mm`.

**Did it reproduce.** Yes, but not as stated — see **C2** above. 5/10 targets,
not 10/10, because the droop straddles the constant.

**What I changed.** One line to `reacher.reached_mm`, plus removing the
now-unused import.

**Numbers re-run.** Scripted mode does not go through this path and is
**byte-identical** on both documented invocations (`--random 5 --seed 1`, 4/5
targets reached; `--target 0.6 -0.2 0.7`). `tests/test_sim.py` 5/5.

**Same defect alive elsewhere.** No. This was entry 12's shape and entry 12's
last live site; the log says so and it is right.

---

### Bug 58 — six modules address arm A by name — **DONE**, commit `ebc985b`

**What it was.** Bare `tool0` / `wrist3_link` lookups that mean "arm A" without
saying so. Correct today; wrong under arm B's label the moment any of them is
called on a two-armed scene, and wrong by returning a plausible number.

**Did it reproduce.** Yes — and the entry undercounts. See **C3**: its own grep
finds three of the five it names, and there is a sixth module it does not name
(`week3_perceive.py`, the Week 3 gate's pose captions).

**What I changed.** Named the expression — `fr5.tool_pos(data, prefix="")`,
with `tool_site_name`, `tool_id` and a `WRIST_LINK` constant beside it — and
routed all seven sites through it, threading `prefix` into the nearest
enclosing signature:

| module | site | where the prefix now lives |
|---|---|---|
| `week4_watch.py` | `Thoughts.tool_site` | `Thoughts(..., prefix="")` |
| `plant_row.py` | main's clearance check | extracted to `print_clearances(..., prefix="")` |
| `deck_cam.py` | `_assert_inert` | `fr5.WRIST_LINK`, named explicitly |
| `deck_cam.py` | `vs_sweep`'s arm-distance total | `prefix` local, from `args.arm` if present |
| `camera.py` | three gate pose captions | `prefix` local |
| `greenhouse.py` | parked-tool print | explicit `tool_pos(data)` |
| `week3_perceive.py` | gate pose captions (**unlisted**) | `prefix` local |

I did **not** add an `--arm` CLI flag to any of them. Five of the six build
single-arm scenes by construction, so the flag would be surface with nothing
behind it; the sites read `getattr(args, "arm", "")` so one can be added later
without touching them again. Say if you would rather have the flags.

**The test the entry asks for.** `tests/test_sim.py` builds
`trolley.build(arms=("a","b"))` and asserts `week4_watch.Thoughts(...,
prefix="b_")` resolves to arm B's tool — 1768 mm from arm A's — while the
default still resolves to arm A.

**Numbers re-run.**

| check | result |
|---|---|
| Week 3 deprojection gate, `camera.py` | **PASS**, byte-identical apart from the unseeded geom count (**C4**) |
| Week 3 gate, `week3_perceive.py --calib` | **PASS at 0.39 mm** worst error, byte-identical |
| `plant_row.py` | byte-identical |
| `deck_cam.py --seed 3` | byte-identical apart from one wall-clock timing line |
| `greenhouse.py` | parked-tool line identical, collidable geoms 34 identical |
| every module in the repo imports | 19 modules + `farm.*` |
| `tests/test_sim.py` | 6/6 |

**Same defect alive elsewhere.** ⚠️ **Yes, and it should be noted against
entries 55, 58 and 68.** `fr5.tool_pos` closes the *tool site* instance of this
shape, but the shape itself is "a module constant that names one instance". I
have not swept for the others — gripper actuator names, pad geom sets, joint
lists — beyond what entry 55 already fixed. `farm/trolley.py:147` carries a
comment saying `week2_pick` and `carrytrace` "all look up `j1`, `tool0`,
`wrist3_link`", which suggests there is a `j1` version of this waiting.

---

### Bug 7 — the pick state machine exists twice — **DONE**, commits `40d2bab` + `72cb554`

Split into two commits on purpose, so you can drop the behaviour change and
keep the extraction, or the reverse.

**What it was.** `week1_gripper.run_pick` and `week1_mousereach.pick_cycle`
were the same seven-state sequence written twice — the `say` and `go` reporting
helpers were character-for-character identical in both — and had already
diverged on the two things that matter.

**Did it reproduce.** Yes for the duplication, and yes for entry **16**. **No
for entry 17**, and that is the finding — see below.

**What I changed.** `simulation/mujoco/pickcycle.py`: one `run_cycle`, a `Plan`
describing a pick, and two `Detach` strategies — `Lift` (a table; the grasp is
the whole test) and `Pull` (a truss; the stem carries the load until it does
not). Four parameters cover every real difference. 126 lines deleted from the
two demos, 77 added.

**The bar, and why it is not the one the entry names.** See **C1**. Held to
`tests/baselines/`, recorded in commit `03df1a4` *before* the refactor: both
cycles over 12 points, 158 and 9 lines of every waypoint, arrival error and
snap force.

| check | result |
|---|---|
| `tests/baselines/week1_mousereach.txt` | **158 lines byte-identical** |
| `tests/baselines/week1_gripper.txt` | **9 lines byte-identical** |
| all three documented invocations + `--wave` | byte-identical |
| the abort branch (no board point reaches it) | byte-identical, including dict key order |
| `tests/test_sim.py` | 6/6 |

The baseline files do not appear in `40d2bab`'s diff, which is the point of
having recorded them two commits earlier.

**Entry 16 — fixed, and it was free.** `week1_gripper` read "home" from
wherever the tool was standing at the top of the cycle. Fixed to `park_pose`,
and the demo is byte-identical across the change, because it runs one cycle
from a fresh reset where the tool is *already* at the home posture. The drift
only ever appeared on a second cycle, which this file has never run — entry
62's shape again: the untested path is the one nobody types.

**⚠️ Entry 17 — does NOT reproduce, and the entry's account of it is wrong.**

The entry says `week1_gripper` "would hit bugs 16 and 17 again from the right
starting posture". Bug 16, yes. Bug 17, not from the posture entry 17 names:

| carry, direct to crate | `week1_gripper` scene | `week1_mousereach` scene |
|---|---|---|
| from `[0.42, +0.25, 1.00]` — entry 17's own case | **arrives, 3.2 mm** | 378 mm short (entry 17) |
| 23 reachable start postures swept | **2 stall, worst 201 mm** | **11 stall, worst 447 mm** |

So it is live but weak, and **the variable is the crate, not the arm**. The row
scene's crate sits at `[0.15, -0.80]`, 300 mm further out in -y than the
table scene's `[0.30, -0.50]`, and that is what turns a long move into one
`mink` cannot walk to.

I did **not** add a transit waypoint. Doing so means choosing a transit point
for this scene — the row scene's does not belong to it — and that is a tuned
constant. It is in the ranked list below as a decision for you.

---

### Bug 5 — the board's guarantee is unguarded — **DONE**, commit `1cf79c6`

**What it was.** `week1_mousereach` promises, in its banner, that every click
on the board is pickable. Nothing checked it, and the numbers behind it were
measured against one gripper, one `APPROACH_GAP`, one `RETRACT_GAP` and one
crate position.

**Did it reproduce.** The *absence* of the check, yes. The guarantee itself
turns out to hold.

**What I built.** `tests/board_walk.py` — a grid over the advertised extent
with endpoints included, one full pick per point, drawn as the board's own map.

**Result: 42/42 on 7×6, and 99/99 on 11×9.** The guarantee holds everywhere I
tested. That 42 is also the first time the build log's "42/42 mouse picks" has
come out of a command rather than out of counting by hand in a window — see
**C1**.

**I checked that it bites rather than merely passes.** Moving the crate to
Week 2's `[0.30, -0.52]` — the move entry 42 blames — takes the same walk to
**18/20**, one unreachable and one dropped. So the guard catches the change it
exists to catch.

⚠️ One caveat worth recording: entry **15**'s own stated crate position,
`[0.15, -0.55]`, does **not** fail today — 20/20. The gaps and the gripper have
both moved since that entry was written, so its geometry no longer describes
this scene. It is a fixed entry so this is not a contradiction, but anyone
reaching for it as a regression case should know it is no longer one.

Runtime ~31 s for 42 points, ~72 s for 99. Nightly or pre-commit, as the entry
asks — not in `test_sim.py`.

---

### Bug 40 — the cradle experiment — **DONE**, commit `6fd9f4f`

**Nothing in the repo's contact model, constants or shipped cycle was changed.
The cradle is not merged into anything.**

**It needed the tool revived first**, which was real work and is step 1 of
`parked/README.md`: eleven borrowed constants had moved to `mission.py`, two
(`SETTLE_S`, `TURN_MAX_S`) to `legacy_cycle.py`, and the file needed its parent
directory on `sys.path` — something it never needed while `parked/` *was*
`simulation/mujoco/`. Revived, it reproduces its recorded numbers exactly:
**5/5 cut, 4/5 carried, 2/5 crated**, with `t1` carried-but-not-crated, which
is the same verdict my trace gives it.

**Why this is not `carrytrace` re-pointed, and could not be.** The prompt asks
for "the same legs", and that is the one part that is not available:

- `carrytrace` rides `week2_pick.execute`, a `mission.Leg` list. This cycle is a
  flat pre-planner state machine with **no `grip` and no `pull`** — the two legs
  the comparison would most want. Making them the same is revival step 3, a new
  leg sequence, which the README explicitly says is not a parameter on the
  existing one.
- `carrytrace` reads **per-pad** force, split left/right, because a one-sided
  unload is its signature. A cradle has no pads; it has a floor and two walls,
  and the floor is load-bearing where the 2F85 had no load-bearing surface at
  all. Those columns are not comparable and the report does not line them up.

**The measure that IS comparable, and two that are not.** This took three
attempts and the failures are part of the result:

1. `carrytrace.ESCAPE_MS` — 1.0 m/s **absolute** — is invalid here. The Robotiq
   cycle carries throttled at `CARRY_SPEED = 0.25`; this one drives at
   `DEFAULT_SPEED` unthrottled and the *tool* crosses 1.0 m/s. Applied as-is it
   flags `t0` and `t2` as ejections at 1.00 and 1.06 m/s — both of which crate,
   79–122 mm from centre, fruit sitting still in the channel throughout. This is
   `carrytrace`'s own footnote happening again in a new file.
2. A **relative** speed threshold fails too: a fruit rattling in an open channel
   hits 0.54 m/s relative to the tool without ever leaving it. A cradle does not
   hold the fruit still, it stops it going anywhere, and those differ.
3. **Displacement in the tool frame** is the one that works. It is what
   "ejected" means, and it needs no calibration against either tool's carry
   speed.

**The result. Twenty runs — five fruit × error 0 / 3.9 / 15 / 40 mm:**

| fruit | fruit in the tool frame, while held | escaped | crated |
|---|---|---|---|
| `t0` | 9 → 20 mm across all four errors | **no** | 4/4 |
| `t1` | 4 → 19 mm | **no** | 0/4 |
| `t2` | 2 → 20 mm | **no** | 4/4 |
| `t4` | 6 → 20 mm | **no** | 0/4 |
| `t3` | 641 → 2322 mm | n/a — see below | 0/4 |

against the 2F85 on the same row:

| run | est err | peak m/s while held | fruit in the tool frame | outcome |
|---|---|---|---|---|
| `t1` ground truth | 0 | 0.42 | 0.2 → **4.7 mm** | crated |
| `t1` perception | 3.9 mm | 1.54 | 5.3 → **783.6 mm** | **ejected** |

**The cradle does not eject.** On every fruit it actually cradles, the fruit
stays within 20 mm of the tool point through every carrying leg, and that
number is **flat against position error from 0 to 40 mm** — ten times the error
that ejects the 2F85. Flat-against-the-input is exactly what "not an
instability" looks like, and it is the same shape of evidence the entry used to
conclude the 2F85's *is* one.

**`t3` is not a counter-example, and I checked rather than assuming.** Its
fruit is 1.1 m from the tool before the carrying legs even begin, channel force
**0.0 N throughout**, `carried=False`: it is knocked off during the approach and
never cradled at all. It then falls to `z = 0.033` and **rolls 3.79 m at a
constant 0.20 m/s** — which is where C6 came from. `week2_vinea`'s own run also
scores `t3` as not carried, so this is the tool's known failure, not something
my harness introduced.

**What it does NOT prove.**

- **Not a like-for-like cycle.** Different legs, different planner (none),
  different carry speed, different arrival tolerance. The ejection question is
  answered; a cycle-time or a success-rate comparison is not, and would be
  dishonest from this data.
- **No perception.** The 3.9 mm is injected as an offset on the *belief*, from
  `t1`'s real error vector. There is no camera in this scene, so it does not
  test whether a cradle is easier or harder to aim from vision.
- **The cut is weak evidence at large error.** `Blade.severed` uses a 55 mm
  tolerance, so `cut=True` at 40 mm of error may be the tolerance being generous
  rather than the blade finding the stem. Irrelevant to the carry, which is the
  question, but it means the cut column should not be quoted.
- **n = 5 fruit, one row, one seed per cell.** And entry 53 says a shift is not
  bit-reproducible.
- **It is not the better tool.** Crate rate **8/20 = 40%**, the same 40% the row
  run reports. It does not throw fruit; it drops it. Only the first of those is
  the safety problem, and trading one for the other is a decision, not a win.
- **MuJoCo contact, not fruit.** Everything entry 9 says still applies, and a
  cradle's whole behaviour depends on friction against a smooth sphere.
