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
import week2_pick  # noqa: E402
import week4_place  # noqa: E402
from farm import house, trolley  # noqa: E402

# ⚠️ **Every module that did `from mission import BIN_POS` holds its own
# binding, and rebinding `mission.BIN_POS` does not touch it.** `from X import
# y` copies the reference at import time; there is no live link back.
#
# This is not hypothetical bookkeeping. `week2_pick.execute` scores a pick with
#
#     "in_bin": crate_contains(fruit.xpos, BIN_POS, BIN_HALF, BIN_WALL)
#
# reading the `BIN_POS` it imported at line 68 — the Week 1-4 crate, at the
# world origin. A pick that carried the fruit to the real crate and dropped it
# neatly inside was scored against a crate two metres away and came back
# `in_bin: False`. Calling `crate_contains` by hand on the same numbers returned
# True. That is the whole bug, and it looks exactly like a carry failure.
#
# So the rebinding walks a list of (module, attribute) rather than one module.
# Anything added to Weeks 1-4 that imports one of these by name has to be added
# here too, which is a real maintenance cost and the reason the module docstring
# says this file should eventually delete itself.
_HOLDERS = {
    "PARK": [(mission, "PARK")],
    "STAGE_X": [(mission, "STAGE_X")],
    "BIN_POS": [(mission, "BIN_POS"), (week2_pick, "BIN_POS")],
    "ROW_X": [(mission, "ROW_X"), (plant_row, "ROW_X"),
              (week2_pick, "ROW_X"), (week4_place, "ROW_X")],
}

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
    saved = [(mod, attr, getattr(mod, attr))
             for key in _HOLDERS for mod, attr in _HOLDERS[key]]
    old_defaults = []
    try:
        for key, holders in _HOLDERS.items():
            for mod, attr in holders:
                setattr(mod, attr, off[key])
        # And the dataclass defaults, which are a copy again — one the `from X
        # import y` list above cannot reach. See `_patch_default`.
        for cls in (mission.Route, mission.Mission):
            old_defaults.append(
                (cls, _patch_default(cls, "stage_x", off["STAGE_X"])))
        yield off
    finally:
        for cls, old in old_defaults:
            _patch_default(cls, "stage_x", old)
        for mod, attr, value in saved:
            setattr(mod, attr, value)


def pin_base(reacher):
    """Stop the IK solver from planning motion the base will never make.

    ⚠️ **This is the bug the mocap deck head was designed to avoid, arriving by
    the front door.** `reach.Reacher` builds `mink.Configuration(model)` over
    the *whole* model, and its `VelocityLimit` names only the six arm joints —
    so every other DOF in the model is, as far as the solver is concerned,
    free and unlimited. The trolley's slide joint is one of those.

    The solver therefore solves for "arm *and* base", happily translating the
    chassis along the row to help the tool reach. Then `Reacher.step` writes
    `data.ctrl[arm_ctrl]` — six numbers, the arm only — so the base never
    actually moves and the arm lands wherever the six joints alone can get to.

    What that looks like is not an error. It looks like this, on a pick the
    planner had just verified to 43 mm of clearance:

        clear     arm 3.6 mm          <- fine, the base was where it wanted it
        lane      DID NOT ARRIVE — 72 mm short
        approach  DID NOT ARRIVE — 82 mm short
        insert    DID NOT ARRIVE — 89 mm short
        check     fruit 318 mm from the tool — DROPPED

    Every leg short by about the same amount, in the direction the base would
    have moved. It reads as an arm that cannot reach, and it is an arm being
    asked to make up for a chassis that was never told to move.

    Pinning the joint to zero velocity is the whole fix: the solver then has to
    find a six-joint answer, which is the only kind that can be executed.

    ⚠️ **And it has to survive `set_speed`.** `mission._legs` gives the carrying
    legs their own speed cap, so `_run_leg` calls `reacher.set_speed(...)` part
    way through every mission — and `set_speed` *rebuilds* `self.limits` from
    `JOINTS` alone, throwing the pin away. The symptom is beautifully confusing:
    the approach, insert and pull legs arrive to 4 mm, and then `extract` — the
    first carrying leg, and the first one to change speed — is 84 mm short. So
    the instance's `set_speed` is wrapped rather than the limit set patched once.
    """
    import mink

    from fr5 import JOINTS
    from reach import JOINT_VELOCITY

    def apply(speed):
        caps = {j: JOINT_VELOCITY[j] * speed for j in JOINTS}
        caps[trolley.DRIVE_JOINT] = 0.0
        reacher.limits = [mink.VelocityLimit(reacher.model, caps)]

    inner = reacher.set_speed

    def set_speed(speed):
        inner(speed)
        apply(reacher.speed)

    reacher.set_speed = set_speed
    apply(reacher.speed)
    return reacher


def park_posture(model, data, tag="a", **kw):
    """`mission.park_posture`, solved in the arm's actual frame.

    ⚠️ Also has to pin the base, and cannot use `pin_base` to do it — it builds
    its own `mink.Configuration` internally rather than taking a `Reacher`. It
    happens to get away with it: it starts from `reset_home`, where the trolley
    is at zero, and the posture task holds the whole configuration near its seed
    — so the drift is small. `check_park` below measures it rather than assuming.
    """
    with at_trolley(model, data, tag):
        return mission.park_posture(model, **kw)


def check(model, data, tag="a"):
    """Print what the rebinding does, and assert it restores itself."""
    before = [(mod, attr, np.array(getattr(mod, attr), dtype=float))
              for key in _HOLDERS for mod, attr in _HOLDERS[key]]
    print(f"\n  --- the arm's frame, on a trolley at "
          f"y={data.body(trolley.TROLLEY).xpos[1]:+.2f} ---")
    print(f"  {'constant':<10} {'week 1-4':<24} {'here':<24}")
    with at_trolley(model, data, tag) as off:
        for k in ("PARK", "BIN_POS"):
            print(f"  {k:<10} {str(_BASE[k].round(3)):<24} "
                  f"{str(np.asarray(off[k]).round(3)):<24}")
        for k in ("STAGE_X", "ROW_X"):
            print(f"  {k:<10} {_BASE[k]:<24.3f} {off[k]:<24.3f}")
    leaked = [f"{mod.__name__}.{attr}"
              for mod, attr, was in before
              if not np.allclose(np.array(getattr(mod, attr), dtype=float), was)]
    print(f"\n  rebound in {len(before)} places across "
          f"{len({m.__name__ for m, _a, _v in before})} modules")
    print(f"  restored on exit: {'yes' if not leaked else 'NO — ' + ', '.join(leaked)}")
    return not leaked
