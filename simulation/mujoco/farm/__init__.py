#!/usr/bin/env python3
"""A whole greenhouse, worked end to end: scout it, plan it, harvest it.

Weeks 1-4 answered "can the arm pick a tomato without wrecking the plant", with
the chassis bolted in one place and the fruit hung inside its own envelope. That
is a manipulation question and it is answered. This package asks the next one,
which is a *logistics* question: given four rows of crop and a machine that has
to drive to reach any of it, what does a shift actually look like?

    scout    drive the house, survey from the moving deck camera, and come back
             with one map: every fruit, where it is, and whether it is ripe
    plan     from that map, choose which trolley stops to make, in what order,
             and which fruit to take at each — the whole house, not one row
    harvest  execute it, dropping fruit into a crate that rides on the trolley

⚠️ **Deliberately separate from `week1_*` .. `week4_*`.** Those files carry
measured numbers — a 46% clean rate, a 31.3 s cycle, a 40 mm clearance — and
every one of them was taken in a one-row scene with a fixed base. Rebuilding
that scene underneath them would invalidate the lot silently. So this package
imports freely from them and changes none of them: `farm.house` is a new scene
next to `greenhouse.py`, not a replacement for it.

Run any of these; each opens a window unless it says otherwise:

    ./.venv/bin/python simulation/mujoco/farm/house.py       # the house, rendered
    ./.venv/bin/python simulation/mujoco/farm/trolley.py     # the base, driving
    ./.venv/bin/python simulation/mujoco/farm/crop.py        # a random crop
    ./.venv/bin/python simulation/mujoco/farm/scout.py       # the mapping pass
    ./.venv/bin/python simulation/mujoco/farm/route.py       # the plan over a map
    ./.venv/bin/python simulation/mujoco/farm/run.py         # the whole shift
    ./.venv/bin/python simulation/mujoco/farm/watch.py       # all of it, watched
"""

import sys
from pathlib import Path

# The week1-4 modules live one directory up and import each other by bare name
# (`from greenhouse import ...`). Putting that directory on the path here means
# every module in this package can do the same, and none of them needs its own
# copy of this incantation.
_MUJOCO = Path(__file__).resolve().parent.parent
if str(_MUJOCO) not in sys.path:
    sys.path.insert(0, str(_MUJOCO))
