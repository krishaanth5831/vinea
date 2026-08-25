# vinea — agent guide

Modular autonomous robot for Dutch commercial greenhouses (truss-tomato harvesting first, scouting second, same base). **Simulation-first: MuJoCo 3.10 + `mink` differential IK on one laptop. No hardware, and ROS is not in the current demo loop** — the `vinea_*` directories are empty ROS 2 skeletons.

This repo is measurement-driven: nearly every constant in it was read off a sweep, and `COMMANDS.md` records which run produced which number. Changing a constant without re-running its sweep invalidates a claim.

## Environment

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/phase0_smoketest.py     # expect 6/6 — MuJoCo, EGL, URDF, IK, CUDA, ROS 2
```

Always call `./.venv/bin/python` explicitly and run from the repo root — the system Python has none of this. If the smoke test isn't 6/6, no number below is trustworthy; report that instead of working around it.

There is **no lint, no typecheck, no test runner and no CI**. The gate is: the smoke test, plus the specific measurement script that covers what you changed (each module prints the numbers it stands on).

**Known inconsistency, do not silently "fix" it:** `requirements.txt` targets Python 3.14 / Ubuntu 26.04 / ROS 2 Lyrical, while `scripts/setup.sh`, `scripts/run_sim.sh` and `docs/decisions/001-why-ros2.md` say ROS 2 Humble. `docs/architecture.md` is a marked placeholder still describing Gazebo Fortress + MoveIt 2 and a `simulation/gazebo/worlds/greenhouse_row.world` that doesn't exist. Ask before reconciling any of this.

## Map

| Path | What it is | Safe to parallelize? |
|---|---|---|
| `simulation/mujoco/` | the actual system: scene, arm, gripper, planner, perception, week1–5 demos | **No** — heavily shared modules and global bindings |
| `simulation/mujoco/farm/` | multi-arm / whole-house work (`armframe.py` rebinds globals) | No — global frame rebinding |
| `scripts/` | env setup, `phase0_smoketest.py` | Yes |
| `docs/` | concept, architecture placeholder, ADRs | Yes |
| `layouts/`, `models/` | layout JSON fixtures, detector weights | Yes (consumers depend on paths) |
| `tests/`, `vinea_*/` | empty placeholders | Yes — isolated |

Read before editing: `README.md` (current status), `COMMANDS.md` (what every script does and every warning), the target module's top-of-file command docs, and for planner/scene work `mission.py`, `reach.py`, `fr5.py`, `greenhouse.py`, plus the relevant `farm/*`.

Do not hand-edit / do not read into context: `.venv/`, `__pycache__/`, `build/`, `install/`, `log/`, `logs/`, `third_party/`, `models/weights/*.pt`, `runs/` and JSONL pick logs, `*.mp4`, generated stills (`greenhouse_*.png`, `deck_cam_*.png`, `farm_*.png`, `twoarm_*.png`), `_fr5_mujoco.urdf`, `MUJOCO_LOG.TXT`.

## Rules that will bite you

- **`--out` and a live window are mutually exclusive everywhere.** Recording forces EGL offscreen; a window needs GLFW. You cannot watch and record in one run.
- **Two default speeds, on purpose.** Interactive Week 4 tools default to `0.4`; every *measurement* tool stays at `0.15` because all recorded numbers were taken there. Pass `--speed 0.15` when comparing against a recorded number.
- **Pass `--no-lessons` to any run that isn't deliberately about learning.** Otherwise it writes shared `simulation/lessons.json` and quietly moves the baseline it's being compared against.
- Use `--seed` where supported so a randomized house is reproducible.
- Never edit a measured constant (e.g. `greenhouse.PAD_SOLREF`, the pick-envelope bands, cost-model thresholds) without re-running the sweep that produced it and quoting the new numbers in the commit body.
- **Commits:** subsystem-prefixed imperative subjects — `farm/duo: …`, `COMMANDS: …`, `README: …` — and measurement/bug-log wording for findings.
- No secrets in this repo; `.env` is ignored and nothing documents keys.

---

# Working with agents in this repo

## Token discipline

1. `rg` for the symbol or constant, then read only the matching range (`sed -n 'A,Bp'`). Don't open a module to find out what's in it.
2. `COMMANDS.md` is long — grep it for the script you care about; never read it end to end.
3. Never read the do-not-read list above (weights, mp4s, stills, logs, `.venv`).
4. Never re-read a file you already read this session.
5. Verify narrowest-first: the single measurement script for what you touched, headless, smallest `-n`/`--trials`; save the long multi-trial run for the end, once.
6. Sim runs are slow and chatty — always run headless with `--no-lessons` and pipe through `| tail -30`. Paste the verdict lines, never the whole trace.
7. Close out a unit of work with a <=15-line summary of the constants changed, the runs used and their numbers, then drop the exploration transcript.

## Parallel execution: orchestrator + workers

`simulation/mujoco/` is a genuinely shared codebase — this repo parallelizes worse than a web app, and pretending otherwise produces two agents editing the same scene. Fan out only when the units are truly disjoint, e.g.:

- one worker per **independent measurement sweep** (each runs read-only scripts and reports numbers) — this is the highest-value fan-out here, because the sweeps are slow and independent;
- perception vs planning vs docs, when the interfaces between them are frozen;
- docs/`scripts/`/layout work alongside anything else.

Do **not** fan out two workers into `simulation/mujoco/` core modules (`reach.py`, `greenhouse.py`, `mission.py`) or into `farm/` at the same time.

The orchestrator writes no feature code. It:

1. Splits along the boundaries above and decides explicitly which module each worker owns.
2. **Freezes contracts first, itself:** module-level constants, function signatures in `reach.py`/`greenhouse.py`/`mission.py`, scene body/geom names. A worker that needs one changed stops and reports.
3. Owns the venv/`requirements.txt` and any change to a measured constant — serially, before fan-out.
4. Hands each worker its own worktree and its own venv, plus a brief. 2-4 concurrent (GPU/CPU-bound: more concurrent MuJoCo runs than cores makes every timing number noisy — if a worker reports timings, run it alone).
5. Reviews each diff and the numbers behind it, merges in dependency order, re-runs the smoke test plus the affected sweeps once, opens the PR.

Worker brief (under ~20 lines — the worker reads this file; don't re-explain the repo):

```
Goal:            <one sentence, observable outcome or a number to produce>
Worktree:        ../vinea-<slug>   Branch: agent/<slug>
Files you own:   <explicit module paths — edit ONLY these>
Do not touch:    requirements.txt, simulation/lessons.json, measured constants, other workers' modules
Contracts:       <signatures/constants, read-only>
Done when:       <exact ./.venv/bin/python ... --headless --no-lessons commands and expected verdicts>
Report:          diff summary, commands run verbatim, the numbers they printed, assumptions
Stop and ask if: a measured constant would have to change, or the smoke test isn't 6/6
```

Anti-patterns: two workers in one simulation module; a worker running without `--no-lessons`; parallel timing runs on a loaded machine; a worker reconciling the ROS Humble/Lyrical inconsistency on its own.

## One worktree per agent

Agents must never share a working tree — concurrent edits, `git switch` and generated artifacts (stills, mp4s, `lessons.json`) clobber each other.

```bash
# orchestrator, once
git fetch origin && git switch -c integration/<task> origin/main && git push -u origin integration/<task>

# per worker
git worktree add ../vinea-<slug> -b agent/<slug> integration/<task>
cd ../vinea-<slug>
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # per-worktree; .venv is NOT shared
./.venv/bin/python scripts/phase0_smoketest.py                        # 6/6 before doing anything
```

- One worktree per worker, `../vinea-<slug>`, branch `agent/<slug>`; the worker stays inside it.
- Each worktree gets its own `.venv`, its own `simulation/lessons.json` and its own generated stills/videos — that isolation is the point: a shared `lessons.json` silently couples two agents' results.
- Only one worker at a time may run a **windowed** (GLFW) session; headless EGL runs are fine in parallel, but keep concurrent heavy runs at or below core count if any number is a timing.
- Integrate by merging `agent/*` into `integration/<task>`, then one PR into `main` (the repo's habit is `dev -> main` via PR). Workers never push to `main` and never merge each other.
- Clean up: `git worktree remove ../vinea-<slug>`, then `git worktree prune`.

## Done means

`scripts/phase0_smoketest.py` is 6/6; the measurement script covering your change prints its expected verdict and you quoted the numbers; no measured constant moved without its sweep re-run; `README.md`/`COMMANDS.md` updated when behaviour, a flag or a number changed; the diff contains no generated artifacts.
