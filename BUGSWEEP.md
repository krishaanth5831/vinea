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

### C2. Bug 3's line numbers have drifted

The entry names `week1_reach.py:152` and `fr5.py:503`. The calls are at
`week1_reach.py:158` and `fr5.py:543`. Both are the right call in the right
function (`main()` in each), so the diagnosis is intact — recording it only
because the entry reads as though it were checked recently and it was not.

---

## 2. What I would do next, ranked

*Filled in at the end of the run.*

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
