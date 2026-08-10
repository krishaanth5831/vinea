# Bug Log

Bugs found in this repo, what reproduced them, and what was learned. Two states:

- **Open** — reproducible, not fixed. Every Open entry carries a repro you can run.
- **Fixed** — closed, with the one-line lesson.

⚠️ **This file was referenced before it existed.** `COMMANDS.md` and
`mission.py` both pointed at "the Bug Log" for the arm-vs-arm gap, and there was
no such file — the gap was real and honestly described, but the pointer went
nowhere. Created 2026-08-10, starting from the entries those two references
implied plus everything found in the two-arm session.

The numbered bugs referred to in older prose (`bug 40`, `bug #31`, `Bug 5`) predate
this file and live in the commit messages and in `README.md`'s Week 2–4 sections.
They are not re-litigated here.

---

## Open

### O1 — A shift is not bit-reproducible, even with `--seed`

**Repro.** Run the same shift twice with the same seed and compare the crated count:

```bash
./.venv/bin/python simulation/mujoco/farm/run.py --seed 7 --stops 3
./.venv/bin/python simulation/mujoco/farm/run.py --seed 7 --stops 3
```

Observed 2/4 and then 3/4 on the same seed with unchanged harvest code.

**What is and is not reproducible.** `--seed` reproduces the *crop layout* exactly
— verified directly, the spawn is a seeded `np.random.Generator`. What does not
reproduce is contact ordering: MuJoCo running multi-threaded does not guarantee a
stable order for equal-depth contacts, and a grasp is decided by which pad contact
the solver lists first.

**Consequence for any number quoted from one run.** A single shift's count is a
sample, not a result. Anything reported as a headline has to be a mean over
several shifts — which is what `farm/misses.py` is for.

### O2 — Two arms cannot fly at once, and it is architectural

**Repro.** There is no command that does it; that is the bug. Attempting it means
nesting `farm.armframe.at_trolley` for two different arms:

```python
with armframe.at_trolley(model, data, "a"):
    with armframe.at_trolley(model, data, "b"):   # arm a's frame is now gone
        ...
```

The inner context overwrites `mission.PARK`, `STAGE_X`, `BIN_POS`, `ROW_X` and
`INTO_ROW` — which are *module globals* — with arm b's values, and restores arm
a's on exit. Arm a's in-flight mission is then being executed against arm b's
frame.

**Why it is not simply fixed.** `armframe`'s own docstring makes the argument: the
alternative is threading a base transform through `Planner`, `Route`, `_legs`,
`Guard`, `ClearanceModel` and `execute` — six classes, ~30 call sites, in the code
whose numbers are the repo's headline results — and then re-taking every clearance
and cycle-time figure to prove nothing moved.

**What ships instead.** `farm/duo.py` serialises the arms and says so on screen
and in its report. This is a real limitation of the build, not a presentational
choice, and it is stated rather than hidden.

### O3 — `deck_cam`, `week4_watch` and `plant_row` still read the bare `tool0`

**Repro.**

```bash
grep -n 'site("tool0")\|body("wrist3_link")' simulation/mujoco/*.py
```

`deck_cam.py:1479,1724`, `week4_watch.py:295`, `plant_row.py:428`,
`greenhouse.py:651` and `camera.py`'s print helpers all use the unprefixed name.

**Why it is Open rather than Fixed.** Arm a is deliberately the unprefixed arm
(`trolley.ARM_PREFIX`), so every one of these resolves and is correct for every
command that uses them — they are all single-arm Week 1–4 demos. They would be
wrong for arm b, and none of them is reachable from a two-armed run: `farm/duo.py`
uses the three that are (`week2_pick.execute`, `carrytrace`, `incident`), and
those were parameterised (see F5).

**The trap.** The failure mode is silent and plausible, not a `KeyError`. If one
of these is ever called on a two-armed scene it will return arm a's numbers under
arm b's label.

---

## Fixed

### F1 — The two arms were parked inside each other

**Repro (before the fix).** Build a two-armed trolley, park both arms, ask MuJoCo
what is touching:

```python
model = trolley.build(aisle=0, arms=("a", "b"), trusses=trusses, seed=7)
# ... park both arms ...
for i in range(data.ncon):
    c = data.contact[i]   # forearm_link vs b_forearm_link, dist -0.0827
```

83 mm of interpenetration between the two forearms, at rest, before anything had
moved. Visible in a render: the arms cross in an X over the middle of the deck.

**Cause.** Not the posture — the mounting. `ARM_X` puts the bases 400 mm apart,
which is what gives each arm its 600 mm standoff, but an FR5's upper arm and
forearm are ~400 mm each and the park posture folds the elbow ~330 mm back behind
the shoulder, into the aisle centreline, from both sides at once.

**Fix.** Stagger the mounts 500 mm along the row (`trolley.ARM_STAGGER`). Measured:

| stagger | sphere gap | arm-vs-arm contacts |
|---|---|---|
| 0.00 m | −0.028 m | 2 |
| 0.40 m | +0.038 m | 4 |
| **0.50 m** | **+0.111 m** | **0** |
| 0.60 m | +0.196 m | 0 |

Rotating the idle arm's j1 was swept (−90°..+120°) and never opens the gap — both
links sweep the centreline whatever the base angle. Moving `PARK` works but pushes
the tool from 0.32 m out to 0.47–0.57 m against a row at 0.60 m, re-creating the
"pads inside the panel" problem `park_posture` exists to avoid.

**Lesson.** A geometry check that has never been run is not a geometry check that
passes — the clash had been shipping since the second arm was fitted, and the only
reason nobody saw it was that nothing was looking.

### F2 — Nothing checked arm against arm

**Repro (before the fix).**

```python
cm = ClearanceModel(model, row, target, prefix="")
list(vars(cm))          # ['prefix', 'spheres', 'crop'] — no other arm
```

Two mounts 400 mm apart, 922 mm of reach each, working volumes overlapping by
1.44 m, and each arm planning as though the other were not there.

**Fix.** `mission.ArmObstacles` covers the other arm with the same conservative
sphere cover `RobotSpheres` already uses; `ClearanceModel`, `Guard` and `Planner`
take an `others=` tuple of prefixes. Empty by default, so Weeks 1–4 are untouched.
`ARM_CLEARANCE` is 40 mm — the crop's budget, not structure's 15 mm — and
`contact_ok` never exempts the other arm.

**Lesson.** An obstacle set is a claim about what the world contains, and anything
left out of it is not "unmodelled", it is *asserted absent*.

### F3 — The idle arm at PARK aborted the working arm's picks

**Repro (after F1, before F3).** Fly an arm b pick with arm a parked:

```
ABORT  arm a within 14 mm during `approach`
```

Correct behaviour from `Guard` — and a machine that cannot harvest.

**Cause.** F1 fixed the *resting* clash. An arm reaching into its row still sweeps
within 14 mm of the other arm parked out in front of it, because `PARK` is 0.32 m
ahead of the shoulder and that is exactly where the other arm's approach passes.

**Fix.** `duo.STOW` — the idle arm folds up over its own base instead of resting
out front. Chosen by sweeping candidate stow postures and, for each, driving the
other arm through 18 postures spanning its working envelope:

| arm a stowed as | worst gap to arm b working |
|---|---|
| park (what shipped) | +0.024 m — below `ARM_CLEARANCE`, aborts |
| j1 −90° | +0.219 m |
| folded, j1 −90° | +0.062 m |
| folded up, j1 0 | +0.222 m |
| j1 +90° | +0.249 m |
| **folded up, j1 +90°** | **+0.318 m** |

**Lesson.** "Parked" is a posture, not a safety property — where an idle arm rests
is part of the other arm's obstacle problem.

### F4 — The planner previewed with 13 DOF and the executor flew with 6

**Repro (before the fix).** `Planner._check` called
`mink.solve_ik(cfg, tasks, PREVIEW_DT, "daqp", 1e-3)` with no `limits`, while the
executor runs behind `armframe.pin_base`. The preview was therefore free to reach
by driving the trolley and folding the other arm.

**Consequence.** This is `pin_base`'s bug on the planning side, and worse there:
`pin_base` produces legs that arrive short, which shows up in the leg report. An
unpinned preview produces a route *checked in a posture the arm will never adopt*,
so the clearance it reports is a measurement of a different machine.

**Fix.** `Planner(..., pin=(...))`, the same explicit contract `park_posture`
already used; `farm/duo.py` passes the drive joint and every other arm's six.

**Lesson.** If the previewer and the executor do not have the same degrees of
freedom, the preview is not a preview.

### F5 — `execute()` scored arm B's picks against arm A's gripper

**Repro (before the fix).** Fly an arm b pick and read the grasp check. In
`week2_pick.execute`:

```python
held = np.linalg.norm(fruit.xpos - data.site(TOOL_SITE).xpos)
```

`TOOL_SITE` is `"tool0"`, which is arm a's. For an arm b pick this measured the
fruit against a gripper half a machine away, returned ~1 m every time, and
`grasped` came back `False` on picks that had worked.

**Fix.** `execute` takes its tool site from `reacher.tool_site`, which already
knows its arm. Same for `_run_leg`'s hold-position goal. `carrytrace.CarryTrace`
and `incident.Blackbox` gained a `prefix` for the same reason — the pad-force
trace that found bug 40 would otherwise have been a recording of an idle gripper.

**Lesson.** A module constant that names one instance of a thing becomes a bug the
moment there are two, and it fails by returning a plausible number rather than by
raising.

### F6 — `route.plan` costed arm B's pick order from arm A's mount

**Repro (before the fix).** `farm/route.py` built the travel-cost origin as
`house.aisle_x(aisle) + house.ARM_OFFSET` regardless of which arm it was planning
— the positive offset, hardcoded — so arm b's ordering was measured from a point
400 mm away on the far side of the aisle.

**Why it survived.** It only ever changed the *sequence*, never which fruit were
taken. A wrong order still harvests, just not in the best order.

**Fix.** `trolley.ARM_X[arm]`, plus the `ARM_Y[arm]` stagger term that F1 made
necessary.

**Lesson.** A default that happens to be right for the first case is
indistinguishable from a parameter until the second case arrives.

### F7 — The map drew ground truth invisibly on exactly the rows that mattered

**Repro (before the fix).** `two_arm_farm.py --shot`, then look at
`twoarm_mission.png`: rows r2 and r3 show ground-truth fruit markers, r0 and r1
show none — and r0/r1 are the two rows the arms actually work.

**Cause.** Truth markers were drawn as dark grey (66,66,66) outlines at the same
pixel as the mapped dot. The two worked rows are drawn as a thick bright-green
bar, and a dark grey outline on it is invisible.

**Fix.** Offset the truth marker 6 px above the mapped dot and lighten it.

**Lesson.** "Faint, so it does not distract" and "invisible against its own
background" are one CSS value apart, and only looking at the render tells them
apart.

### F8 — The stale claim that the chassis never moves

**Repro.** `README.md`, "What this does not prove": *"The chassis never moves.
Every number here is measured with the arm bolted in one place."*

That was true through Week 4. Week 5 built a pipe-rail trolley with a real
prismatic drive joint and a position servo:

```bash
./.venv/bin/python simulation/mujoco/farm/trolley.py --drive
```

drives the length of the house and prints the odometer.

**Fix.** The claim is scoped to Weeks 1–4 and the Week 5 position stated
separately.

**Lesson.** A limitations section decays faster than the code it describes,
because fixing the limitation does not touch the file that documents it.
