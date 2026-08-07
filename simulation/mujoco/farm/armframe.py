#!/usr/bin/env python3
"""Make Weeks 1-4's world-frame planner work for an arm that moves.

⚠️ **This module exists because of one design fact in `mission.py`, and it is
worth stating plainly rather than working around quietly: the Week 1-4 planner
is written in absolute world coordinates.**

    PARK     = [0.32, -0.10, 0.50]     where the arm rests
    STAGE_X  = 0.32                    the plane it approaches from
    BIN_POS  = [0.30, -0.52, 0.00]     where the crate is
    ROW_X    = 0.60                    where the crop is

Every one of those is a module constant, and every one of them is correct for
exactly one machine: an arm bolted to the origin with a crate beside it. That
was a reasonable thing to write when the base could not move. It stops being
reasonable the moment the arm is on a trolley, because all four are then
functions of where the trolley is standing — a mission planned against `PARK`
while the trolley is at y = +3 m parks the arm three metres behind itself, and
`BIN_POS` aims the release at a patch of floor the machine drove away from.

**What this does.** The arm's geometry relative to its own base has not changed
at all — that is the entire point of `house.ARM_OFFSET` and `trolley.DECK_Z`.
So the fix is a pure translation: rebind those constants to
`week1_4_value + arm_base_world` for as long as the trolley is parked, and put
them back afterwards.

⚠️ **Rebinding another module's globals is a real cost and it is taken with
open eyes.** The alternative is threading a frame through `Planner`, `Route`,
`_legs`, `Guard`, `ClearanceModel` and `execute` — six classes and about
thirty call sites in code whose numbers are the repo's headline results, for a
refactor with no measurement attached. This is the smaller, more reversible
change, and it is confined to one context manager that is impossible to leave
open by accident:

    with at_trolley(model, data):
        mission = planner.plan(name)
        execute(mission, ...)

**The right fix eventually** is for `mission` to plan in the arm's frame and be
handed a base transform, at which point this file deletes itself. It is not the
right fix *today*, because doing it means re-taking every clearance and cycle
number in the README to prove nothing moved.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mission  # noqa: E402
import plant_row  # noqa: E402
from farm import house, trolley  # noqa: E402

# The Week 1-4 values, captured at import so a nested or aborted `at_trolley`
# can always restore the truth rather than whatever the last caller left.
_BASE = {
    "PARK": np.array(mission.PARK, dtype=float),
    "STAGE_X": float(mission.STAGE_X),
    "BIN_POS": np.array(mission.BIN_POS, dtype=float),
    "ROW_X": float(mission.ROW_X),
}


def arm_base(model, data, tag="a"):
    """World position of the arm's shoulder, right now.

    Read out of `mjData`, not computed from constants: the trolley's `qpos` is
    the only thing that knows where the machine actually is, and a helper that
    recomputed it from `aisle_x` plus a remembered `y` would be right until the
    first time somebody drove without telling it.
    """
    body = "base_link" if tag == "a" else f"{tag}_base_link"
    return data.body(body).xpos.copy()


def offsets(model, data, tag="a"):
    """The translation from the Week 1-4 world to this one, as a dict.

    Returned rather than applied so a caller can print it, which is the only way
    anyone is going to trust a module that rewrites globals.
    """
    base = arm_base(model, data, tag)
    aisle = _aisle_of(model, data)
    return {
        "arm_base": base,
        "PARK": _BASE["PARK"] + base,
        "STAGE_X": _BASE["STAGE_X"] + base[0],
        "BIN_POS": trolley.crate_pos(model, data),
        "ROW_X": house.row_x(_worked_row(aisle, tag)),
    }


def _aisle_of(model, data):
    """Which aisle the trolley is standing in, from its world x."""
    x = data.body(trolley.TROLLEY).xpos[0]
    return int(round((x - house.ROW_PITCH / 2 - house.ROW_X0)
                     / house.ROW_PITCH))


def _worked_row(aisle, tag):
    left, right = house.serves(aisle)
    return right if tag == "a" else left


def _patch_default(cls, field, value):
    """Rebind a dataclass field's default. Returns the old value.

    ⚠️ **`mission.Route.stage_x` defaults to `STAGE_X`, and a dataclass captures
    that at class-creation time — into `__init__.__defaults__`, not into a class
    attribute.** So rebinding `mission.STAGE_X` moves the constant and leaves
    every `Route()` still staging in the Week 1-4 frame.

    This was not a theoretical problem. The first trolley reach gate passed
    10/10 and every single route came back labelled `+100cm`, i.e. "staged a
    metre further back than normal" — which is not a fallback the planner chose,
    it is `back = STAGE_X - self.stage_x` computing 1.32 - 0.32 because the
    default never moved. The plans were being built against a staging plane
    0.68 m *behind* the arm's own shoulder and reported as successes.
    """
    import inspect

    names = [p for p in inspect.signature(cls.__init__).parameters
             if p != "self"]
    defaults = list(cls.__init__.__defaults__)
    # __defaults__ covers the *trailing* parameters, so index from the end.
    i = names.index(field) - (len(names) - len(defaults))
    old = defaults[i]
    defaults[i] = value
    cls.__init__.__defaults__ = tuple(defaults)
    return old


@contextmanager
def at_trolley(model, data, tag="a", verbose=False):
    """Plan and fly in the Week 1-4 planner as if the arm were at the origin.

    ⚠️ The crate is read live and the rest are read once, on entry. That is not
    an oversight: `BIN_POS` has to be where the crate *is*, and the crate rides
    the trolley — but `PARK` and `STAGE_X` are the frame a **single mission** is
    planned and flown in, and a mission whose park pose moved halfway through
    would be a mission that never returns to where it started. So the trolley
    must not drive inside this block, and `Drive.drive_to` is not called from
    within one anywhere in this package.
    """
    off = offsets(model, data, tag)
    if verbose:
        print(f"  arm frame: base {off['arm_base'].round(3)} · "
              f"PARK {off['PARK'].round(3)} · "
              f"row x {off['ROW_X']:.2f} · crate {off['BIN_POS'].round(3)}")
    saved = (mission.PARK, mission.STAGE_X, mission.BIN_POS, mission.ROW_X,
             plant_row.ROW_X)
    old_defaults = []
    try:
        mission.PARK = off["PARK"]
        mission.STAGE_X = off["STAGE_X"]
        mission.BIN_POS = off["BIN_POS"]
        mission.ROW_X = off["ROW_X"]
        # `plant_row.ROW_X` too: `mission` imported `ROW_X` *from* it, so the
        # two names are separate bindings to the same original float and
        # rebinding one leaves the other stale. Anything reading it through
        # `plant_row` would keep seeing 0.60.
        plant_row.ROW_X = off["ROW_X"]
        # And the dataclass defaults, which are a third copy again. See
        # `_patch_default` for what happens when this is missed.
        for cls in (mission.Route, mission.Mission):
            old_defaults.append(
                (cls, _patch_default(cls, "stage_x", off["STAGE_X"])))
        yield off
    finally:
        for cls, old in old_defaults:
            _patch_default(cls, "stage_x", old)
        (mission.PARK, mission.STAGE_X, mission.BIN_POS, mission.ROW_X,
         plant_row.ROW_X) = saved


def park_posture(model, data, tag="a", **kw):
    """`mission.park_posture`, solved in the arm's actual frame."""
    with at_trolley(model, data, tag):
        return mission.park_posture(model, **kw)


def check(model, data, tag="a"):
    """Print what the rebinding does, and assert it restores itself."""
    before = (np.array(mission.PARK), mission.STAGE_X,
              np.array(mission.BIN_POS), mission.ROW_X)
    print(f"\n  --- the arm's frame, on a trolley at "
          f"y={data.body(trolley.TROLLEY).xpos[1]:+.2f} ---")
    print(f"  {'constant':<10} {'week 1-4':<24} {'here':<24}")
    with at_trolley(model, data, tag) as off:
        for k in ("PARK", "BIN_POS"):
            print(f"  {k:<10} {str(_BASE[k].round(3)):<24} "
                  f"{str(np.asarray(off[k]).round(3)):<24}")
        for k in ("STAGE_X", "ROW_X"):
            print(f"  {k:<10} {_BASE[k]:<24.3f} {off[k]:<24.3f}")
    after = (np.array(mission.PARK), mission.STAGE_X,
             np.array(mission.BIN_POS), mission.ROW_X)
    ok = (np.allclose(before[0], after[0]) and before[1] == after[1]
          and np.allclose(before[2], after[2]) and before[3] == after[3])
    print(f"\n  restored on exit: {'yes' if ok else 'NO — globals leaked'}")
    return ok
