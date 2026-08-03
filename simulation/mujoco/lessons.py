#!/usr/bin/env python3
"""Do not make the same mistake twice.

`incident.py` works out *why* a tomato fell. This is what turns that answer
into behaviour: an attributed incident becomes a constraint the planner reads
before it plans the next pick, so a route that has cost a fruit once is not
planned again.

    lessons = Lessons.load()
    planner = Planner(model, data, row, lessons=lessons)

    mission = planner.plan("t0")          # lessons already applied
    ...
    lessons.learn(box.incidents, mission) # anything that went wrong
    lessons.confirm(mission, box.incidents)
    lessons.save()

**What this is.** Case-based constraint learning. Each lesson is a rule with a
cause, a victim, a remedy and a count of how often it has held. The planner
already searches a space of routes and rejects the ones that come too close to
the crop; a lesson changes the *cost of being close to one particular fruit*,
which pushes the search onto a different route without any of the machinery
knowing why.

**What this is not.** It is not reinforcement learning, and calling it that
would be a lie a technical cofounder would catch in ten minutes. There is no
policy, no reward, no gradient, and it cannot discover a manoeuvre nobody
programmed. What it can do is notice that a described failure recurs and widen
a specific constraint until it stops.

That trade is deliberate and it is the right one for this stage:

  - **It works on one example.** An RL agent needs thousands of episodes to
    learn "do not hit t3". This needs one, which matters when a pick is 24 s.
  - **It is inspectable.** `lessons.json` is a readable file of sentences. When
    a grower asks why the robot avoids one truss, there is an answer. A policy
    network has no answer.
  - **It cannot silently regress.** A learned weight can quietly get worse. A
    constraint can only ever refuse more routes, so the failure mode is a
    refused pick — visible, countable, and safe — never a knocked truss.

The honest limitation is the obvious one: it generalises only as far as the
feature it keys on. A lesson about t3 applies to t3. Making it a lesson about
*geometry* — "fruit within 120 mm below and 40 mm across the pull line get
struck during a straight-down pull" — is the Week 4 version, and the record
already carries the relative offsets needed to do it.

Run it directly to see the store built from scratch against a live row:

    ./.venv/bin/python simulation/mujoco/lessons.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from incident import (  # noqa: E402
    FRUIT_KNOCK,
    PINNED,
    STEM_OVERLOAD,
    TOOL_STRIKE,
    UNKNOWN,
)

STORE = Path(__file__).resolve().parents[2] / "simulation" / "lessons.json"

# How much extra room a first offence buys, and how much each repeat adds.
#
# ⚠️ The escalation matters more than the starting value. A single fixed bump
# either overshoots — refusing picks that were always fine — or undershoots and
# the same fruit goes on the floor every run with a lesson on file saying it
# should not. Starting small and widening only when it demonstrably did not
# work converges on the smallest constraint that actually holds, which is also
# the one that costs the least cycle time.
FIRST_BUMP = 0.030
REPEAT_BUMP = 0.025
MAX_KEEPOUT = 0.150     # past this the row is not pickable and refusing is honest

# Legs that are just the arm getting from A to B. They bucket together because
# the fix is the same for all of them — route further from the fruit — whereas
# a strike during `insert` or `pull` means the grasp itself needs to change.
# The legacy cycle's names are in here too, so lessons learned from the old
# unplanned script transfer to the planner rather than being filed separately.
TRANSIT_LEGS = {"clear", "lane", "align", "extract", "carry", "withdraw",
                "ready", "turn",
                "home", "retreat", "traverse", "approach"}


@dataclass
class Lesson:
    """One learned constraint, and the evidence that earned it."""

    key: str
    cause: str
    victim: str
    leg: str
    remedy: str                  # keepout | pull_out | none
    amount: float = 0.0
    seen: int = 0                # times this incident has happened
    repeats: int = 0             # times it happened *after* the lesson existed
    held: int = 0                # missions planned with it that stayed clean
    targets: list = field(default_factory=list)
    culprit: str = ""
    first: str = ""
    note: str = ""

    def line(self) -> str:
        rate = f"{self.held}/{self.held + self.repeats}" if (
            self.held + self.repeats) else "untested"
        amount = (f" +{self.amount * 1000:.0f} mm"
                  if self.remedy == "keepout" else "")
        return (f"[{self.key}] {self.note}\n"
                f"      remedy: {self.remedy}{amount} · seen {self.seen}x · "
                f"held {rate}")


def _key(cause, victim, leg):
    """A lesson's identity: what happened, to what, during which leg.

    Deliberately not keyed on the *target* fruit. "The gripper hits t3 on the
    way out" is the same problem whichever tomato it was carrying, and keying on
    the target would make the robot relearn it five times — once per fruit in
    the row — which is exactly the behaviour that makes a learning system look
    like it is not learning.
    """
    bucket = "transit" if leg in TRANSIT_LEGS else leg
    return f"{cause}:{victim}:{bucket}"


class Lessons:
    """The store. Reads at plan time, writes after a failure."""

    def __init__(self, path=STORE, lessons=None, enabled=True):
        self.path = Path(path)
        self.enabled = enabled
        self.lessons: dict[str, Lesson] = lessons or {}

    # -- persistence ----------------------------------------------------------

    @classmethod
    def load(cls, path=STORE, enabled=True):
        path = Path(path)
        if not path.exists():
            return cls(path, enabled=enabled)
        raw = json.loads(path.read_text())
        return cls(path, {k: Lesson(**v) for k, v in raw.items()},
                   enabled=enabled)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {k: asdict(v) for k, v in self.lessons.items()}, indent=2) + "\n")
        return self.path

    def forget(self):
        self.lessons = {}

    # -- reading them ---------------------------------------------------------

    def advise(self, target, row, route):
        """Apply everything learned so far to a candidate route.

        Returns (route, [labels]) — the labels go into `Mission.applied` so the
        run log says which lesson shaped which pick. A learning system that
        cannot show its working is indistinguishable from a random one.
        """
        if not self.enabled or not self.lessons:
            return route, []

        keepout = dict(route.keepout)
        pull = route.pull
        applied = []
        for lesson in self.lessons.values():
            if lesson.victim not in row.names or lesson.victim == target:
                continue
            if lesson.remedy == "keepout":
                keepout[lesson.victim] = max(
                    keepout.get(lesson.victim, 0.0), lesson.amount)
                applied.append(lesson.key)
            elif lesson.remedy == "pull_out":
                pull = "out"
                applied.append(lesson.key)

        route.keepout = keepout
        route.pull = pull
        return route, applied

    # -- writing them ---------------------------------------------------------

    def learn(self, incidents, mission=None):
        """Turn incidents into constraints. Returns the lessons touched."""
        touched = []
        for hit in incidents:
            lesson = self._for(hit)
            lesson.seen += 1
            if hit.target and hit.target not in lesson.targets:
                lesson.targets.append(hit.target)

            known = mission is not None and lesson.key in (mission.applied or [])
            if known:
                # It happened again with the lesson already in force, so the
                # remedy was not enough. Widen it.
                lesson.repeats += 1
                if lesson.remedy == "keepout":
                    lesson.amount = min(MAX_KEEPOUT,
                                        lesson.amount + REPEAT_BUMP)
            self.lessons[lesson.key] = lesson
            touched.append(lesson)
        return touched

    def confirm(self, mission, incidents):
        """Credit the lessons that were in force and were not violated."""
        if mission is None:
            return
        broken = {_key(h.cause, h.victim, h.leg) for h in incidents}
        for key in (mission.applied or []):
            lesson = self.lessons.get(key)
            if lesson is not None and key not in broken:
                lesson.held += 1

    def _for(self, hit) -> Lesson:
        key = _key(hit.cause, hit.victim, hit.leg)
        if key in self.lessons:
            return self.lessons[key]

        remedy, amount, note = self._remedy(hit)
        return Lesson(key=key, cause=hit.cause, victim=hit.victim, leg=hit.leg,
                      remedy=remedy, amount=amount, culprit=hit.culprit,
                      first=date.today().isoformat(), note=note)

    @staticmethod
    def _remedy(hit):
        """Which knob to turn, given what went wrong.

        The mapping is the whole design. A taxonomy whose branches all lead to
        the same fix is decoration; each of these changes the plan differently.
        """
        where = f"during `{hit.leg}`"
        if hit.cause == TOOL_STRIKE and hit.leg == "pull":
            return ("pull_out", 0.0,
                    f"{hit.culprit} descends into {hit.victim} when the pull "
                    f"goes straight down — pull back and down instead")
        if hit.cause in (TOOL_STRIKE, PINNED):
            verb = "pinned against the row" if hit.cause == PINNED else "struck"
            return ("keepout", FIRST_BUMP,
                    f"{hit.victim} gets {verb} by {hit.culprit} {where} — "
                    f"keep the arm further off it")
        if hit.cause == FRUIT_KNOCK:
            return ("keepout", FIRST_BUMP,
                    f"the fruit in the gripper swings into {hit.victim} "
                    f"{where} — carry it further from the row")
        if hit.cause == STEM_OVERLOAD:
            # ⚠️ No geometric remedy, and inventing one would be worse than
            # none: a keepout on a fruit nothing touched buys nothing and costs
            # cycle time on every future pick. Recording it as a known-unfixable
            # is the honest outcome, and it is also the signal that SNAP_N or
            # the peduncle model — not the route — is what needs the attention.
            return ("none", 0.0,
                    f"{hit.victim} came off {where} with nothing touching it — "
                    f"the peduncle model, not the route")
        return ("none", 0.0, f"{hit.victim} was lost {where}, cause unknown")

    # -- reporting ------------------------------------------------------------

    def report(self) -> str:
        if not self.lessons:
            return "  no lessons on file"
        out = []
        for lesson in sorted(self.lessons.values(),
                             key=lambda x: (-x.seen, x.key)):
            out.append("  " + lesson.line())
        active = sum(1 for x in self.lessons.values() if x.remedy != "none")
        out.append(f"\n  {len(self.lessons)} lessons, {active} with a remedy "
                   f"the planner can act on")
        return "\n".join(out)


def main():
    """Learn from what the *unplanned* cycle actually did, then plan with it.

    Everything here is measured, not staged: the incidents come from running
    `legacy_cycle.py` with the black box attached, which is the same run the
    README quotes.
    """
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    import legacy_cycle
    from fr5 import reset_home
    from greenhouse import build_scene
    from incident import Blackbox
    from mission import Planner, park_posture, reset_park
    from plant_row import Row
    from reach import DEFAULT_SPEED, Gripper

    model = build_scene(greenhouse=False)
    data = mujoco.MjData(model)
    row = Row(model, data)
    park = park_posture(model)
    store = Lessons(path=Path("/tmp/lessons_demo.json"))

    print("\n1. Run the unplanned cycle and watch what it costs.\n")
    rng = np.random.default_rng(0)
    hits = []
    for i in range(5):
        name = row.names[i % len(row.names)]
        reacher = legacy_cycle.make_reacher(model, data, speed=DEFAULT_SPEED)
        gripper = Gripper(model, data)
        reset_home(model, data)
        row.reset()
        row.jitter(rng, 20.0)
        mujoco.mj_forward(model, data)
        reacher.sync()

        box = Blackbox(model, data, row, name)
        legacy_cycle.pick_one(reacher, gripper, row, name, box=box,
                              verbose=False)
        box.sweep()
        for hit in box.incidents:
            print(f"  pick {i + 1} ({name}): {hit.explain()}")
            hits.append(hit)
    if not hits:
        print("  nothing was disturbed — nothing to learn")
        return

    print(f"\n2. {len(hits)} incidents become constraints.\n")
    store.learn(hits)
    print(store.report())

    print("\n3. What the planner does with them.\n")
    reset_park(model, data, park)
    row.reset()
    mujoco.mj_forward(model, data)

    plain = Planner(model, data, row)
    taught = Planner(model, data, row, lessons=store)
    for name in row.names:
        before = plain.plan(name)
        after = taught.plan(name)
        keep = ", ".join(sorted(after.applied)) or "-"
        print(f"  {name}: {before.clearance * 1000:5.0f} mm -> "
              f"{after.clearance * 1000:5.0f} mm   "
              f"route {before.lane}->{after.lane}   lessons: {keep}")

    print("\n  ⚠️ On this row the planner already routes clear of t3 without\n"
          "  being told — the numbers above barely move, and that is the\n"
          "  honest result. The lesson is not what fixes this failure; the\n"
          "  plan is. What the lesson buys is that the margin around t3 is now\n"
          "  a hard constraint, so no future route — a different lane, a\n"
          "  camera-estimated position, a fruit that has moved — can quietly\n"
          "  give it back.")

    print("\n4. If it happens again anyway, the constraint widens.\n")
    m = taught.plan("t0")
    store.learn([h for h in hits if h.victim == "t3"][:1], m)
    print(store.report())
    print(f"\n  (store would be written to {store.path})")


if __name__ == "__main__":
    main()
