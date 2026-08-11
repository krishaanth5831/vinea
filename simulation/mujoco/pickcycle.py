#!/usr/bin/env python3
"""The pick cycle, once.

    open the fingers -> approach -> engage -> close -> detach -> carry ->
    release -> park

Bug log entry 7: this sequence existed twice, in `week1_gripper.run_pick` and
`week1_mousereach.pick_cycle`, near-identical and drifting. `week1_mousereach`
had the transit waypoint (entry 17) and the fixed park pose (entry 16);
`week1_gripper` had neither, so the same two bugs were one starting posture
away from happening again in a file where they had already been fixed.

**What actually differs between a table pick and a row pick** is smaller than
the duplication made it look, and it is all here as parameters:

  * where the pre-grasp point is — above the fruit for a table, out along the
    row normal for a truss;
  * how the fruit comes free — a *lift*, which is free once the fingers hold,
    against a *pull*, which has to load a peduncle until it gives. That is the
    one real difference between picking something up and harvesting it;
  * whether an unreachable pre-grasp point aborts the cycle or is simply
    reported;
  * whether there is a transit waypoint on the way to the crate.

Everything else — the state names, the arrival reporting, the order, the holds
— is shared, and was already identical in both copies.

⚠️ **The two flags that are entry 7's actual finding** are `transit` and
`park`. `week1_gripper` passes `transit=None` and a `park` read from wherever
the tool happens to be, because that is what it did before this file existed
and a refactor that changes a measured demo is two changes wearing one commit.
Both are one argument away from correct and `week1_gripper` says so at its call
site. See `tests/baseline.py` for what turning them on moves.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Plan:
    """One pick, described. Every field is a difference between the two demos.

    Positions are world points for `tool0` — with a gripper mounted that site
    sits between the fingertips, so a waypoint is where the fingers should
    actually be rather than where some offset behind them should be.
    """

    approach: np.ndarray            # pre-grasp: fingers open, touching nothing
    grasp: np.ndarray               # the fruit centre, fingers around it
    release: np.ndarray             # hold it here and open
    park: np.ndarray                # where the cycle ends

    detach: "Detach"                # lift, or pull until the stem gives

    transit: Optional[np.ndarray] = None   # waypoint on the way to the crate
    engage_label: str = "engage"           # "descend" on a table
    park_label: str = "park"               # "home" in week1_gripper
    approach_note: str = ""
    close_note: str = "fingers closed on the fruit"
    grip_s: float = 1.0                    # time for the fingers to travel
    park_hold_s: float = 0.4
    abort_note: Optional[str] = None       # None = never abort, only report
    done_note: Optional[Callable[[bool], str]] = None
    crated: Optional[Callable[[], bool]] = None


class Detach:
    """How the fruit comes off. The one genuinely different step.

    Two implementations below. Both are handed the cycle's `go` and `say` so
    they print in the same voice as every other state, and both return a dict
    that the caller folds into its own result.
    """

    label = "detach"

    def run(self, go, say, on_tick):        # pragma: no cover - interface
        raise NotImplementedError

    def aborted(self):
        """What this step reports when the cycle never got near the fruit."""
        return {}


@dataclass
class Lift(Detach):
    """Pick a loose object up. Nothing holds it down, so the grasp is the test.

    The honest check is the fruit's own height: if the fingers did not hold, the
    arm goes up and the tomato stays on the table.
    """

    to: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body: object = None             # the thing being lifted
    start_height: float = 0.0
    clearing: float = 0.0           # count it held if it rose by this much

    label = "lift"

    def run(self, go, say, on_tick):
        go(self.label, self.to)
        held = float(self.body.xpos[2]) - self.start_height > self.clearing
        say("check", "holding the tomato" if held
            else "GRASP FAILED — fruit left behind")
        return {"grasped": held}

    def aborted(self):
        return {"grasped": False}


@dataclass
class Pull(Detach):
    """Harvest it. The stem carries the load until it does not.

    Nothing commands the break — the arm backs off along the row normal and the
    weld gives out. The threshold is watched on every *physics* substep where
    the caller allows it, because a stiff arm loads a peduncle far faster than
    a 100 Hz control loop can notice: see entry 24.
    """

    to: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force: Callable[[], float] = None      # newtons through the stem, now
    release_weld: Callable[[], None] = None
    snap_n: float = 0.0

    label = "pull"

    def run(self, go, say, on_tick):
        state = {"broken": False, "peak": 0.0, "at": 0.0}

        def tick(t):
            f = self.force()
            state["peak"] = max(state["peak"], f)
            if not state["broken"] and f > self.snap_n:
                self.release_weld()
                state["broken"] = True
                state["at"] = f
            if on_tick is not None:
                on_tick(t)

        go(self.label, self.to, tick=tick)
        if state["broken"]:
            # The load is checked once per control cycle and a stiff arm
            # retracting at speed loads the stem far faster than that, so the
            # force it actually breaks at overshoots the threshold, often by a
            # lot. The threshold is a floor, not the number the stem sees.
            say("snap", f"stem gave at {state['at']:.1f} N "
                        f"(threshold {self.snap_n:.1f} N — one cycle's overshoot)")
        else:
            say("snap", f"stem HELD — peak only {state['peak']:.1f} N "
                        f"of {self.snap_n:.1f} N")
        return {"snapped": state["broken"], "peak_n": state["peak"]}

    def aborted(self):
        return {"snapped": False, "peak_n": 0.0}


def run_cycle(reacher, gripper, plan, on_tick=None, verbose=True):
    """Drive one whole pick. Returns what happened.

    Written as a flat sequence on purpose, exactly as both copies were. Every
    step is "put the tool here" or "move the fingers", and reading it top to
    bottom should tell you what the arm is about to do.
    """
    from reach import hold

    data = reacher.data

    def say(state, note=""):
        if verbose:
            p = data.site(reacher.tool_site).xpos
            print(f"  {state:<9} tool [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}]   {note}")

    def go(state, target, tick=None, note=""):
        r = reacher.drive_to(target, on_tick=tick or on_tick)
        arrived = (f"arm {r['arm_mm']:.1f} mm" if r["reached"]
                   else f"DID NOT ARRIVE — {r['arm_mm']:.0f} mm short")
        say(state, f"{arrived}   {note}".rstrip())
        return r

    # 1. Approach — fingers open, touching nothing. Square-on along whichever
    #    line the caller chose; coming at fruit diagonally is how you knock it
    #    off the plant.
    gripper.open()
    reached = go("approach", plan.approach, note=plan.approach_note)
    if plan.abort_note is not None and not reached["reached"]:
        say("abort", plan.abort_note)
        go(plan.park_label, plan.park)
        return {"reached": False, "in_crate": False,
                **plan.detach.aborted()}

    # 2. Close the last stretch with the fingers open around the fruit.
    go(plan.engage_label, plan.grasp)

    # 3. Grip. The arm holds position while the fingers travel.
    gripper.close()
    hold(reacher, plan.grasp, plan.grip_s, on_tick)
    say("close", plan.close_note)

    # 4. Get it free. Lift, or pull until the stem gives.
    outcome = plan.detach.run(go, say, on_tick)

    # 5. Carry and drop. The transit waypoint exists because `mink` is
    #    *differential* IK: it walks downhill from the posture it is in, with no
    #    map of the workspace, so one large move from high on the row at +y down
    #    to the crate at -y can walk into a corner and stall a long way short
    #    (measured: 378 mm). Splitting it in two fixes it, and is what real
    #    pick-and-place does anyway.
    if plan.transit is not None:
        go("transit", plan.transit)
    go("carry", plan.release)
    gripper.open()
    hold(reacher, plan.release, plan.grip_s, on_tick)
    say("release", "fingers opened")

    # 6. Out of the way. A *fixed* pose, or the next cycle starts somewhere new
    #    and the one after that somewhere newer — entry 16.
    go(plan.park_label, plan.park)
    hold(reacher, plan.park, plan.park_hold_s, on_tick)

    crated = bool(plan.crated()) if plan.crated is not None else False
    if plan.done_note is not None:
        say("done", plan.done_note(crated))
    return {"reached": True, "in_crate": crated, **outcome}
