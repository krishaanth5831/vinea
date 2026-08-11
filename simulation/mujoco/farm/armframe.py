#!/usr/bin/env python3
"""Where one arm's world is, when the arm is bolted to a moving trolley.

⚠️ **This module used to rebind `mission`'s globals and it no longer does.**
The Week 1-4 planner was written in absolute world coordinates:

    PARK     = [0.32, -0.10, 0.50]     where the arm rests
    STAGE_X  = 0.32                    the plane it approaches from
    BIN_POS  = [0.30, -0.52, 0.00]     where the crate is
    ROW_X    = 0.60                    where the crop is
    INTO_ROW = [+1, 0, 0]              which way the canopy is

Every one is correct for exactly one machine: an arm bolted to the origin with
a crate beside it. On a trolley all five are functions of where the machine is
standing, so this module computed the arm's own values and **wrote them into
`mission`, `week2_pick`, `plant_row` and `week4_place` for the duration of a
mission**, restoring them on the way out.

That worked, and it had one cost that turned out to be the binding constraint
on the whole machine: **two arms mid-mission at once need two conflicting sets
of those globals in one interpreter**, so the second arm could not fly. It was
not unimplemented, it was inexpressible. `farm/duo.py` serialised the arms and
said so on screen; Bug Log 57 recorded it as architectural.

`mission.ArmFrame` is the fix the old docstring here promised: the five values
as a *value*, carried on the `Planner` and stamped onto the `Mission` it
produces, so the executor reads the frame off the plan it was handed. Two arms
hold two frames and nothing global moves. This module's job is now only to
**compute** an arm's frame from where its trolley is standing:

    frame = armframe.frame(model, data, "b")
    planner = Planner(model, data, row, prefix="b_", frame=frame)
    mission = planner.plan(name)     # carries the frame with it
    execute(mission, ...)            # reads it back off the mission

`at_trolley` is kept as a context manager over the same numbers, because
`trolley.reach_gate`, `carrytrace` and the Week 3-4 demos still use it and
their measurements were taken through it. It now delegates to `frame()` and
rebinds nothing that the two-armed path reads.
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
from fr5 import JOINTS as _JOINTS  # noqa: E402

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
    "INTO_ROW": [(mission, "INTO_ROW"), (week2_pick, "INTO_ROW")],
}

# ⚠️ **Arm b is not a translation of arm a, it is a reflection, and translating
# it is the bug this constant exists to stop.** The two arms work rows on
# opposite sides of the aisle, so arm b is bolted down turned 180° about z (see
# `trolley.ARM_YAW`). Weeks 1-4 are written in world coordinates that assume the
# row is at *larger* x than the arm — `INTO_ROW` is `[+1, 0, 0]`, `ROW_X` is
# `STAGE_X + 0.28`, `PARK` is on the -y side. Every one of those has the wrong
# sign for arm b.
#
# Applied naively, with translation only, `park_posture` for arm b aims at a
# point 1.20 m through the crop on the far side of its row and reports "park
# posture missed by 1841 mm". That is the *good* failure. The quiet one is
# `INTO_ROW`, which would have arm b approach every fruit from behind.
#
# diag(-1, -1, 1): a half turn about z. x and y flip, height does not.
MIRROR = {"a": np.array([1.0, 1.0, 1.0]),
          "b": np.array([-1.0, -1.0, 1.0])}
_MIRROR = MIRROR      # the old private name, kept for external readers

# The Week 1-4 values, captured at import so a nested or aborted `at_trolley`
# can always restore the truth rather than whatever the last caller left.
_BASE = {
    "PARK": np.array(mission.PARK, dtype=float),
    "STAGE_X": float(mission.STAGE_X),
    "BIN_POS": np.array(mission.BIN_POS, dtype=float),
    "ROW_X": float(mission.ROW_X),
    "INTO_ROW": np.array(mission.INTO_ROW, dtype=float),
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

    Returned rather than applied so a caller can print it. `frame()` is the same
    numbers in the shape `mission` actually consumes; this stays because
    `check()` and the Week 3-4 demos print it by key.
    """
    base = arm_base(model, data, tag)
    aisle = _aisle_of(model, data)
    m = MIRROR[tag]
    return {
        "arm_base": base,
        # Reflect about the arm's own base, then translate. See `MIRROR`.
        "PARK": m * _BASE["PARK"] + base,
        "STAGE_X": m[0] * _BASE["STAGE_X"] + base[0],
        "BIN_POS": trolley.crate_pos(model, data, tag),
        "ROW_X": house.row_x(_worked_row(aisle, tag)),
        # A direction, so it reflects but does not translate.
        "INTO_ROW": m * _BASE["INTO_ROW"],
    }


def frame(model, data, tag="a"):
    """This arm's world, as a `mission.ArmFrame`. The thing to pass to `Planner`.

    ⚠️ **Read fresh, and therefore read where the trolley is standing *now*.**
    Every value except `INTO_ROW` depends on the drive joint, so a frame is only
    valid while the machine is parked. Build one per stop, after the drive and
    before the plan — which is what `farm.duo` does, and why `Drive.drive_to` is
    never called between building a frame and finishing the mission planned in
    it. A mission whose park pose moved halfway through is a mission that never
    returns to where it started.

    ⚠️ The crate is read out of `mjData` rather than derived, because it rides
    the trolley and `trolley.crate_pos` is the only thing that knows where it
    ended up. The rest are arithmetic on the arm's base.
    """
    from mission import ArmFrame

    off = offsets(model, data, tag)
    return ArmFrame(park=np.asarray(off["PARK"], float),
                    stage_x=float(off["STAGE_X"]),
                    bin_pos=np.asarray(off["BIN_POS"], float),
                    row_x=float(off["ROW_X"]),
                    into_row=np.asarray(off["INTO_ROW"], float))


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
    """Rebind the Week 1-4 globals to this arm's frame, and yield that frame.

    ⚠️ **Prefer `frame()` and `Planner(frame=...)`.** This is kept for the
    callers whose numbers were measured through it and for the handful of
    modules that still read `mission.BIN_POS` and friends by import
    (`carrytrace`, the Week 3-4 demos). It rebinds globals, which is safe only
    while one arm is planning — `farm.duo` no longer uses it, and that is what
    lets both arms fly at once.

    ⚠️ **The yielded `ArmFrame` is the part that reaches `Planner`.** Since
    `mission.WORLD` captures the Week 1-4 values at import, rebinding
    `mission.PARK` does *not* change what a `Planner` built with no `frame=`
    plans against. That is deliberate — a stable `WORLD` is what guarantees the
    single-armed numbers cannot move — but it means a caller inside this block
    must pass the yielded frame on:

        with at_trolley(model, data, tag) as fr:
            planner = Planner(..., frame=fr)

    ⚠️ The crate is read live and the rest are read once, on entry. `BIN_POS`
    has to be where the crate *is*, and the crate rides the trolley — but `PARK`
    and `STAGE_X` are the frame a **single mission** is planned and flown in,
    and a mission whose park pose moved halfway through would be a mission that
    never returns to where it started. So the trolley must not drive inside this
    block, and `Drive.drive_to` is not called from within one anywhere.
    """
    off = offsets(model, data, tag)
    fr = frame(model, data, tag)
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
        yield fr
    finally:
        for cls, old in old_defaults:
            _patch_default(cls, "stage_x", old)
        for mod, attr, value in saved:
            setattr(mod, attr, value)


def pin_base(reacher, others=()):
    """Stop the IK solver from planning motion the base will never make.

    `others` names the *other* arms on the machine, by prefix — `("b_",)` when
    this Reacher drives arm a of a two-armed trolley. Their six joints are
    pinned alongside the drive joint, for exactly the same reason and against
    exactly the same failure: see the note on `Reacher.prefix`. A Reacher built
    for arm a has arm b inside its `mink.Configuration`, so the QP is free to
    "reach" by folding the other arm, and `step` then writes only arm a's six
    actuators. The tool lands short and the pick reads as an arm that cannot
    reach.

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

    def apply(speed):
        caps = {j: reacher.joint_velocity[j] * speed for j in reacher.joints}
        caps[trolley.DRIVE_JOINT] = 0.0
        for p in others:
            for j in JOINTS:
                caps[p + j] = 0.0
        reacher.limits = [mink.VelocityLimit(reacher.model, caps)]

    inner = reacher.set_speed

    def set_speed(speed):
        inner(speed)
        apply(reacher.speed)

    reacher.set_speed = set_speed
    apply(reacher.speed)
    return reacher


def park_posture(model, data, tag="a", arms=None, **kw):
    """`mission.park_posture`, solved in the arm's actual frame.

    ⚠️ Has to pin everything the solve must not use, and cannot use `pin_base`
    to do it — it builds its own `mink.Configuration` internally rather than
    taking a `Reacher`. With one arm it got away with not pinning at all: the
    solve starts from `reset_home` with the trolley at zero and the posture task
    holds the configuration near its seed, so the drift was small enough that
    `check_park` measured it as noise.

    ⚠️ **With two arms it stops being noise.** The QP gets 13 free DOF for one
    tool point and only six are read back, which came out as "park posture
    missed by 1950 mm" for arm b — the correct frame, solved against the wrong
    set of joints. `arms` names every arm fitted so the others can be pinned;
    it defaults to just this one, which is the single-armed behaviour.
    """
    fitted = (tag,) if arms is None else tuple(arms)
    pin = [trolley.DRIVE_JOINT]
    for p in trolley.other_arms(tag, fitted):
        pin += [p + j for j in _JOINTS]
    # ⚠️ And the scratch has to stand where the machine stands. `PARK` below is
    # a world point derived from this trolley's current y; solving it against a
    # fresh MjData, where the drive joint is zero, asks the arm to park wherever
    # the trolley happened to start. See `mission.park_posture`'s `hold`.
    hold = {trolley.DRIVE_JOINT:
            float(data.qpos[model.joint(trolley.DRIVE_JOINT).qposadr[0]])}
    # ⚠️ Handed the frame rather than solved inside a rebinding block. Same
    # numbers, and it no longer mutates module state that a concurrently
    # planning second arm is reading. See the module docstring.
    return mission.park_posture(model, prefix=trolley.ARM_PREFIX[tag],
                                pin=tuple(pin), hold=hold,
                                frame=frame(model, data, tag), **kw)


def check(model, data, tag="a"):
    """Print what the rebinding does, and assert it restores itself."""
    before = [(mod, attr, np.array(getattr(mod, attr), dtype=float))
              for key in _HOLDERS for mod, attr in _HOLDERS[key]]
    print(f"\n  --- the arm's frame, on a trolley at "
          f"y={data.body(trolley.TROLLEY).xpos[1]:+.2f} ---")
    print(f"  {'constant':<10} {'week 1-4':<24} {'here':<24}")
    with at_trolley(model, data, tag) as fr:
        here = {"PARK": fr.park, "BIN_POS": fr.bin_pos,
                "STAGE_X": fr.stage_x, "ROW_X": fr.row_x}
        for k in ("PARK", "BIN_POS"):
            print(f"  {k:<10} {str(_BASE[k].round(3)):<24} "
                  f"{str(np.asarray(here[k]).round(3)):<24}")
        for k in ("STAGE_X", "ROW_X"):
            print(f"  {k:<10} {_BASE[k]:<24.3f} {here[k]:<24.3f}")
    leaked = [f"{mod.__name__}.{attr}"
              for mod, attr, was in before
              if not np.allclose(np.array(getattr(mod, attr), dtype=float), was)]
    print(f"\n  rebound in {len(before)} places across "
          f"{len({m.__name__ for m, _a, _v in before})} modules")
    print(f"  restored on exit: {'yes' if not leaked else 'NO — ' + ', '.join(leaked)}")
    return not leaked
