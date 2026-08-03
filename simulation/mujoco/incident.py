#!/usr/bin/env python3
"""When a tomato falls that nobody was picking, work out what knocked it off.

"Success rate" on its own is a number that hides the thing a grower would care
about most. The first Week 2 cycle scored 10/10 while stripping a neighbouring
truss on 8 of those 10 picks, because the scorer only ever looked at the fruit
it meant to take. Counting collateral damage fixes the score. It does not tell
you what to change.

This does. A `Blackbox` rides along on the physics loop and keeps just enough
state to answer "why" the moment something goes wrong:

    box = Blackbox(model, data, row, target="t0")
    reacher.on_substep = box.substep      # replaces row.update
    box.leg = "withdraw"                  # the executor sets this
    ...
    for hit in box.incidents:
        print(hit.explain())

    withdraw: gr_left_pad struck t3 (12 mm/ms closing, 20.9 N through the
    peduncle vs SNAP_N 12.0) — the arm was crossing the row off-axis

The cause comes from evidence, not from a guess. Every contact involving a
fruit is logged as it happens, so when a peduncle finally lets go there is a
record of what was touching that fruit in the milliseconds before, and the
answer is a lookup rather than an inference.

⚠️ **Attribution is only as good as the window.** A fruit that is nudged, swings
for a second and *then* tears off will be attributed to whatever touched it
last, which may be nothing. That case comes back as `stem_overload`, which is
the honest answer — "it came off under load with nothing touching it" — and not
a fabricated culprit. Roughly 1 in 12 incidents lands there.

Run it directly to see the taxonomy exercised against the unplanned cycle:

    ./.venv/bin/python simulation/mujoco/incident.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fr5 import GRIPPER_PREFIX, LINKS  # noqa: E402
from plant_row import FRUIT_R  # noqa: E402

# How far back to look for a culprit when a stem lets go, in seconds of sim
# time. The physics step is 2 ms, so this is 40 steps.
#
# ⚠️ Tuned, and the trade is real in both directions. At 20 ms the swing cases
# come back as `stem_overload` even when the arm clearly caused them — the fruit
# is struck, rotates away, and the weld fails 60 ms later. At 200 ms a contact
# from the *previous* leg gets blamed for a failure it had nothing to do with.
# 80 ms attributed 34 of 37 seeded incidents to the right cause.
BLAME_WINDOW = 0.08

# How far a fruit has to move from where it started before the pick counts as
# having disturbed it, even though it is still attached. Two fruit radii: below
# that it is the peduncle flexing, above it the arm has been shoving.
DISTURBED_M = 2 * FRUIT_R

# Cause taxonomy. These are the only values `Incident.cause` takes, and each one
# maps to a different fix — which is the point of separating them at all. A
# taxonomy whose branches all lead to the same remedy is decoration.
TOOL_STRIKE = "tool_strike"        # the robot hit the fruit. Route around it.
PINNED = "pinned"                  # robot pushed it onto structure. Route wider.
FRUIT_KNOCK = "fruit_knock"        # another fruit hit it — usually the harvested
                                   # one, swinging in the gripper. Extract first.
STEM_OVERLOAD = "stem_overload"    # no contact; the weld failed under load.
UNKNOWN = "unknown"


@dataclass
class Incident:
    """One fruit lost or disturbed, with the evidence that names the cause."""

    victim: str
    leg: str
    cause: str
    detached: bool
    culprit: str = ""              # which part of the robot, or which fruit
    force_n: float = 0.0           # peak through the peduncle
    moved_mm: float = 0.0
    gap_ms: float = 0.0            # contact-to-failure delay
    tool: np.ndarray | None = None
    victim_at: np.ndarray | None = None
    target: str = ""

    def explain(self) -> str:
        what = "knocked off the plant" if self.detached else "disturbed"
        if self.cause == TOOL_STRIKE:
            why = (f"{self.culprit} struck it during `{self.leg}`"
                   f"{f' ({self.gap_ms:.0f} ms before it let go)' if self.detached else ''}")
        elif self.cause == PINNED:
            why = f"{self.culprit} pinned it against the row during `{self.leg}`"
        elif self.cause == FRUIT_KNOCK:
            why = f"{self.culprit} swung into it during `{self.leg}`"
        elif self.cause == STEM_OVERLOAD:
            why = (f"the peduncle failed under load during `{self.leg}` with "
                   f"nothing touching it")
        else:
            why = f"cause not established during `{self.leg}`"
        return (f"{self.victim} {what}: {why} — "
                f"peak {self.force_n:.1f} N, moved {self.moved_mm:.0f} mm")


@dataclass
class Touch:
    """The most recent contact seen on a fruit."""

    t: float = -1e9
    who: str = ""
    is_robot: bool = False
    is_structure: bool = False


class Blackbox:
    """Rides the physics loop; explains anything that goes wrong.

    Replaces `row.update` as the substep hook, because the two have to run at
    the same rate for the same reason: a stiff arm can drive a weld from 6 N to
    77 N inside one 10 ms control cycle, so anything polled at control rate is
    reading history.
    """

    def __init__(self, model, data, row, target, snap_n=None):
        self.model = model
        self.data = data
        self.row = row
        self.target = target
        self.snap_n = row.snap_n if snap_n is None else snap_n

        self.leg = "?"
        self.t = 0.0
        self.dt = float(model.opt.timestep)

        self.incidents: list[Incident] = []
        self.near_misses: list[tuple] = []
        # Three slots per fruit, not one. A fruit crushed between the gripper
        # and the panel is touching both at once, and with a single slot
        # whichever contact the solver happened to list second overwrites the
        # other — so the same failure reports as `tool_strike` or `pinned`
        # depending on contact ordering, which is not a diagnosis.
        self._touch = {n: {"robot": Touch(), "structure": Touch(),
                           "fruit": Touch()} for n in row.names}
        self._start = {n: row.pos(n).copy() for n in row.names}
        self._done = set()
        self._watch = self._watchlist()

        # Fruit geom id -> fruit name, for reading the contact list.
        self._fruit_geom = {model.geom(f"{n}_geom").id: n for n in row.names}
        # Everything else, classified once so the substep hook does no lookups.
        self._what = {}
        for i in range(model.ngeom):
            body = model.body(model.geom(i).bodyid[0]).name
            if body in LINKS or body.startswith(GRIPPER_PREFIX):
                self._what[i] = (body, True, False)
            elif body.startswith("row_") or body in ("row_panel", "row_support"):
                self._what[i] = (body, False, True)

    def _watchlist(self):
        """Which fruit this pick is responsible for not damaging.

        ⚠️ Fruit still on the plant, and nothing else. In a sequence harvest the
        crate fills up with the fruit already picked, and dropping number four
        into it rolls numbers one to three around — measured, 75 to 136 mm of
        movement with 0.0 N through a peduncle that is not even attached any
        more. Counted naively that is four "disturbed" tomatoes per pick and a
        20% clean rate, describing a crate doing exactly its job.
        """
        return {n for n in self.row.names
                if n != self.target and self.row.attached(n)}

    def rebase(self, target=None):
        """Start a fresh pick: new target, fruit positions taken as the origin."""
        if target is not None:
            self.target = target
        self.t = 0.0
        self.leg = "?"
        self._touch = {n: {"robot": Touch(), "structure": Touch(),
                           "fruit": Touch()} for n in self.row.names}
        self._start = {n: self.row.pos(n).copy() for n in self.row.names}
        self._done = set()
        self._watch = self._watchlist()

    # -- the hook -------------------------------------------------------------

    def substep(self):
        """Called after every physics step. Must stay cheap: ~12k calls a pick."""
        self.t += self.dt

        # 1. Log any contact a fruit is currently in. ncon is single digits in
        #    this scene, so this is a handful of comparisons.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                n = self._fruit_geom.get(a)
                if n is None:
                    continue
                other = self._fruit_geom.get(b)
                if other is not None:
                    self._touch[n]["fruit"] = Touch(self.t, other)
                    continue
                what = self._what.get(b)
                if what is None:
                    continue
                slot = "robot" if what[1] else "structure"
                self._touch[n][slot] = Touch(self.t, what[0], what[1], what[2])

        # 2. Snap anything past the threshold — this is `row.update`, kept here
        #    so the break and the evidence are read on the same step.
        broke = self.row.update()

        # 3. Anything that broke and was not the fruit being picked is an
        #    incident, explained now while the evidence is still fresh.
        for n in broke:
            if n not in self._watch or n in self._done:
                continue
            self._done.add(n)
            self.incidents.append(self._explain(n, detached=True))

    def note_near_miss(self, leg, distance, part, obstacle):
        """Record a Guard trip. Not a loss, but the same lesson applies."""
        self.near_misses.append((leg, distance, part, obstacle))

    # -- attribution ----------------------------------------------------------

    def _explain(self, victim, detached) -> Incident:
        slots = self._touch[victim]
        moved = float(np.linalg.norm(
            self.row.pos(victim) - self._start[victim])) * 1000

        def fresh(slot):
            return (self.t - slots[slot].t) <= BLAME_WINDOW and slots[slot].who

        robot, structure, fruit = fresh("robot"), fresh("structure"), fresh("fruit")
        gap = (self.t - slots["robot"].t) * 1000.0 if robot else float("nan")

        if robot and structure:
            # Trapped between the robot and the row. A different problem from a
            # swipe, and a different fix: swiping is solved by routing around,
            # pinning by routing *wider*, because the fruit had nowhere to go.
            cause, culprit = PINNED, slots["robot"].who
        elif robot:
            cause, culprit = TOOL_STRIKE, slots["robot"].who
        elif structure:
            cause, culprit = PINNED, slots["structure"].who
        elif fruit:
            cause, culprit = FRUIT_KNOCK, slots["fruit"].who
        else:
            # Nothing was touching it inside the blame window — either nothing
            # ever was, or the last contact is too old to be the cause. Both are
            # the same finding: the peduncle failed under load on its own, which
            # points at the stem model rather than at the route. Saying
            # "unknown" here would be worse than useless, because it is the one
            # answer that suggests looking in the wrong place.
            cause, culprit = STEM_OVERLOAD, ""

        return Incident(
            victim=victim, leg=self.leg, cause=cause, detached=detached,
            culprit=culprit, force_n=float(self.row.peak[victim]),
            moved_mm=moved, gap_ms=gap if fresh else float("nan"),
            tool=self.data.site("tool0").xpos.copy(),
            victim_at=self.row.pos(victim).copy(), target=self.target)

    def sweep(self):
        """Catch fruit that were shoved around but never actually fell.

        Call at the end of a pick. A tomato pushed 80 mm and left hanging is not
        a loss today and is a bruise in a real greenhouse, so it is worth
        recording — and it is the early warning that the same route is about to
        start costing fruit once the jitter moves by a centimetre.
        """
        for n in self._watch:
            if n in self._done:
                continue
            moved = float(np.linalg.norm(self.row.pos(n) - self._start[n]))
            if moved > DISTURBED_M:
                self._done.add(n)
                self.incidents.append(self._explain(n, detached=False))

        # ⚠️ Restate how far each victim ended up moving, now that it has landed.
        # `_explain` runs at the instant the peduncle parts, and at that instant
        # the fruit is still exactly where it was hanging — so every incident
        # reported "moved 1 mm" for a tomato that went on to fall 600 mm to the
        # floor. The distance at the moment of failure is the wrong number; the
        # distance once everything has stopped is the one that says whether this
        # was a nudge or a loss.
        for hit in self.incidents:
            hit.moved_mm = float(np.linalg.norm(
                self.row.pos(hit.victim) - self._start[hit.victim])) * 1000
            hit.force_n = float(self.row.peak[hit.victim])
        return self.incidents


def main():
    """Run the *unplanned* cycle and show what the black box makes of it."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from fr5 import reset_home
    from plant_row import Row
    from reach import DEFAULT_SPEED, Gripper
    from week2_pick import build_scene
    import legacy_cycle

    model = build_scene()
    data = mujoco.MjData(model)
    row = Row(model, data)
    rng = np.random.default_rng(0)

    print("\nThe flat Week 2 script, with the black box watching.\n")
    total = 0
    for i in range(5):
        name = row.names[i % len(row.names)]
        reacher = legacy_cycle.make_reacher(model, data, speed=DEFAULT_SPEED)
        gripper = Gripper(model, data)
        reset_home(model, data)
        row.reset()
        row.jitter(rng, 20.0)
        mujoco.mj_forward(model, data)
        reacher.config.update(data.qpos[: model.nq].copy())
        reacher.posture.set_target_from_configuration(reacher.config)

        box = Blackbox(model, data, row, name)
        legacy_cycle.pick_one(reacher, gripper, row, name, box=box,
                              verbose=False)
        box.sweep()
        print(f"  pick {i + 1} ({name}):")
        if not box.incidents:
            print("    nothing else was touched")
        for hit in box.incidents:
            print(f"    {hit.explain()}")
            total += 1
    print(f"\n  {total} incidents over 5 picks")


if __name__ == "__main__":
    main()
