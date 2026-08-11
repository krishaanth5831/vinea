# Bug-log updates, ready to paste

Written in the vault bug log's own format. Each entry below moves from **Open**
to **Fixed** with a date and a one-line lesson. Paste them into
`(C) Bug Log — vinea_sim.md` yourself — I did not touch the vault.

Fixed entries in that file are grouped by date under a heading; these all
belong under a new one:

    All 2026-08-12, the bugsweep branch.

---

- [x] **6. There were no tests; `tests/` held `.gitkeep` and nothing else.** Fixed 2026-08-12. `tests/test_sim.py`, in `scripts/phase0_smoketest.py`'s style rather than a second convention — a `@check` decorator, a PASS/FAIL table, non-zero exit, no framework (there is none in the repo and pytest is not installed). Five checks in **1.06 s**, which is the binding constraint: a suite slower than that does not get run before every commit whatever the README says. A pick completes end to end (grasped *and* crated, not one or the other); the weld holds **0.367 mm** under 1 s of gravity while reading its own **1.18 N**, and the fruit falls 567 mm once `eq_active = 0`; `tool0` sits **0.000 mm** off `gr_pinch` at three postures, not one, because two frames on the same rigid body agree everywhere or nowhere; and `reset_home` restores a free body to **0.0 µm** of its spawn pose while `mj_resetDataKeyframe` strands it **747 mm** away. That last assertion is deliberately two-sided — it requires the keyframe to *still* be broken, so if MuJoCo ever changes its short-keyframe padding, this is the only thing in the repo that will notice and entries **3** and **11** do not quietly become wrong. **No headline number is asserted anywhere**: 42/42, 10/10, 0.39 mm and 20/20 stay in the build log with their assumptions written next to them, because a test that pins a measurement turns an honest change to the physics into a failing build and teaches people to edit the expected value. ⚠️ Scope is the four mechanisms the entry names, which are Week 1's. Weeks 2–5 remain untested. *Lesson: the cost of having no tests was not the bugs they would have caught — it was that **58** and **7** both sat open with "not worth doing against no test" written on them, so one missing thing was holding three entries shut.*

---

## Note against the Open section, not a fix

**Entries 42/42 and 6/6 have no command that produces them.** Both are real measurements from the build log — 42 hand-run mouse clicks, and entry 23's six-position table sweep — and neither can be re-run, because `week1_mousereach.py` has no sweep mode and `week1_gripper.py` picks one hardcoded fruit. This is bug **6**'s own sentence about itself, found while trying to use those numbers as the regression bar for **7**. Bug **7** is therefore held to a byte-for-byte stdout diff over a fixed point set, captured before and after and committed under `tests/baselines/`, which compares every waypoint and arrival error rather than a final tally.

**Entry 3's line numbers have drifted.** The calls are at `week1_reach.py:158` and `fr5.py:543`, not `:152` and `:503`. Right function in both cases, so the diagnosis stands.
