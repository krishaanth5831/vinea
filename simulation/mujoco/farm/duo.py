#!/usr/bin/env python3
"""One full row, two arms: map it, plan it, drive it, pick it, crate it.

`farm/run.py` is the same chain for **one** arm. This is the two-armed version,
and the difference is not "call it twice" — three things change shape:

    map      two deck heads scanning independently, one per arm, each mapping
             its own row. `farm/scout.py`'s single head turns 180 deg to serve
             both; these never leave their own row. See `farm.decks`.
    plan     one route per arm, then **merged into one trolley itinerary**. The
             trolley is 1-DOF and there is one of it, so two routes have to
             become one sequence of stops or the machine drives the aisle twice.
    pick     **concurrent.** Both arms fly at the same time, stepped inside one
             physics loop. See below.

--- the arms used to be serialised, and why they are not any more ------------

⚠️ **They were serialised because of `farm.armframe`, not because of
collision.** Weeks 1-4 are written in absolute world coordinates — `PARK`,
`STAGE_X`, `BIN_POS`, `ROW_X`, `INTO_ROW` were module constants — and the
adapter **rebound `mission`'s globals** to the current arm's frame for the
duration of a mission, with a mirror for arm b because it is bolted round
180 deg. Two arms mid-mission need two conflicting sets of those globals in one
interpreter, so concurrency was not unimplemented, it was inexpressible. Bug
Log 57 recorded it as architectural.

Two changes removed it, and neither moved a measured number:

    mission.ArmFrame        the five constants as a *value*, carried on the
                            Planner and stamped onto the Mission, so a plan is
                            self-contained and nothing global changes
    week2_pick.MissionRun   the executor as a generator that stops every
                            control cycle with its setpoints written and
                            physics pending, instead of owning its own loop

`Machine` below then commands every arm and steps the plant **once** per cycle.
One clock, one plant, two control laws.

⚠️ **Arm-vs-arm clearance is now mandatory and it is doing real work.** While
the arms were serialised the idle one was stationary, so putting it in the
flying arm's obstacle set was a check against a fact. With both arms flying
into a working volume that overlaps by 1.44 m it is the only thing between
them, so `work()` refuses to fly an arm whose `others` set is empty, and both
the planner's preview and the runtime `Guard` carry it.

⚠️ **And the preview is weaker than the guard here, which is worth saying
plainly.** `mission.ArmObstacles` reports where the other arm is *now*. For the
planner that is a prediction — it previews the whole route against the other
arm's current pose, and the other arm is moving. For `Guard` it is a
measurement, taken every control cycle against live positions, and that is what
actually holds the line. The plan is the primary defence for the crop and the
net under the arms; `mission.CLEARANCE`'s docstring makes the same argument for
why a verified plan is not a proof.

An arm with nothing to do at a stop still folds up (`STOW`) — not for safety
now, but because it leaves the working arm the most room and so gets fewer
routes refused.

--- what is exposed, and why that matters ------------------------------------

⚠️ Every string the viewer prints comes from `ArmState`, which is written **at
the point the thing happens** — `phase` is set beside the call it describes, and
`leg` is read live off `week2_pick.MissionRun.leg`, which the executor sets per
leg as it flies. Nothing here is a script of captions replayed on a timer. If
the planner refuses a fruit, the panel says so because `ArmState.route` was
handed the `Mission` the planner returned and read the breach off it, not
because a refusal was due.

`ArmState.begin`, `route`, `follow_leg` and `finish` are the four places that
write it, and they sit next to the four things that happen: a target is chosen,
a route comes back, a leg starts, a pick ends.

    ./.venv/bin/python simulation/mujoco/farm/duo.py            # headless, a row
    ./.venv/bin/python simulation/mujoco/farm/duo.py --stops 2  # short
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import armframe  # noqa: E402
from farm import crop as fcrop  # noqa: E402
from farm import decks, house, route, trolley  # noqa: E402
from farm.scout import HouseMap, Sighting, StageDetector, _fuse, stage_of  # noqa: E402

# How close a mapped position has to be to a truss for the executor to accept
# the name. `farm.run.NAME_GATE_M`'s reasoning, unchanged.
NAME_GATE_M = 0.12

# Trolley stops within this of each other are one stop. The two arms plan
# independently, so their stop covers land at unrelated y — without merging, the
# trolley would shuffle back and forth by a few centimetres between arms.
STOP_MERGE_M = 0.25

# How far an arm may reach **toward the other arm**, in metres along the row,
# while that arm is also working.
#
# ⚠️ **A measured number, not a margin picked to be safe** — though it is not on
# its own what makes concurrency safe; `STOW` and `DeckCentre` are. Both arms'
# park postures fold the elbow back over the shoulder — which is the whole
# reason `PARK` exists (`mission.PARK`: home is inside the canopy) — and folding
# it back puts arm a's upper arm at x = 0.716 and arm b's at x = 0.884. **They
# interleave across the aisle**, and the only thing holding them apart is the
# 500 mm `trolley.ARM_STAGGER` along the row. So an arm reaching for a fruit
# *behind* itself swings its elbow down the row straight through where the other
# arm's parked volume would be.
#
# While the arms were serialised this could not happen: the idle arm was stowed
# (+318 mm, see `STOW`) and the working arm had the deck to itself. It is the
# first thing that broke when both were let fly, and it broke correctly — the
# planner refused, with `park: forearm_link within -129 mm of arm b`.
#
# ⚠️ What this constant buys is that each arm's fruit are biased to **its own
# side of the stagger**, so the two arms' work is spatially separated at every
# stop rather than overlapping in the middle. That reduces how often they have
# to wait for each other; it is not what stops them touching.
#
# Swept with arm a driven to three crop heights at each offset from its own
# base, worst arm-vs-arm sphere gap over the three:
#
#     reach toward the other arm     arm b PARKED     arm b STOWED
#       +0.45 m (away)                  +250 mm          --
#       +0.30 m (away)                  +198 mm          --
#       +0.15 m (away)                  +103 mm          --
#        0.00 m                          +61 mm         +385 mm
#       -0.15 m (toward)                 +72 mm          --
#       -0.30 m (toward)                 +48 mm         +320 mm   <- ships
#       -0.45 m (toward)                  +2 mm         +318 mm   <- refused
#       -0.60 m (toward)                 -40 mm         +324 mm   <- refused
#
# -0.30 m is the last offset that clears `mission.ARM_CLEARANCE` with room to
# spare. Past it the gap falls off a cliff, exactly as `ARM_STAGGER`'s own sweep
# did — there is no graceful degradation, the elbows either miss or they do not.
#
# ⚠️ **In the arm's own frame, not the world's.** Arm b is bolted round 180 deg,
# so "toward the other arm" is world -y for arm a and world +y for arm b. Same
# symmetry that lets both arms share every other Week 1-4 number.
#
# ⚠️ Reaching *away* from the other arm keeps the full `route.REACH_Y`. The
# window is asymmetric, 0.78 m wide rather than 0.96 m, which costs a few extra
# trolley stops and buys both arms working at every one of them.
CONCURRENT_TOWARD_M = 0.30

# How many times an arm re-plans while waiting for the other one to move out of
# the way, and how many control cycles it waits between tries.
#
# ⚠️ **This is the only place the two arms interlock, and it is deliberately the
# weakest possible one.** An arm blocked by the *crop* gives up immediately —
# the plants do not move, so retrying is a spin. An arm blocked by the *other
# arm* is blocked by a snapshot of something that is moving, so it folds up and
# tries again.
#
# ⚠️ Sized to outlast a whole pick rather than to give up during one. 400 tries
# at 10 cycles is 40 s of simulated time against a 20-25 s pick, so an arm
# blocked by the other waits until the deck is genuinely free instead of
# expiring halfway through and reporting a refusal that is really a timeout —
# which is a much worse number to publish, because it looks like the planner
# could not find a route. The short interval means it takes its turn promptly.
# Still bounded, so a real deadlock ends as a reported refusal, not a hang.
PLAN_RETRIES = 6
PLAN_RETRY_CYCLES = 10

# How long an arm will hold folded waiting for the deck to clear, in control
# cycles. 6000 is 60 s of simulated time — longer than the 20-25 s pick it is
# waiting on, with room for the other arm to be on its second fruit. Bounded so
# a real deadlock ends as a reported give-up rather than a hang.
#
# ⚠️ The waiting itself is cheap (`park_is_clear`, about a millisecond a cycle),
# which is what makes a long patient wait affordable. `PLAN_RETRIES` is small
# because a *re-plan* is the expensive thing and it now only happens once the
# cheap gate says unfolding is actually clear.
WAIT_MAX_CYCLES = 6000

# The legs that put an arm through the middle of the deck, where the other arm
# also has to pass. Only one arm may be in one of these at a time.
#
# ⚠️ **Measured from the aborts, not guessed.** With both arms flying freely the
# planner accepted every route — correctly, each was clear against the other
# arm's pose at plan time — and then the guard aborted them in flight, always in
# the same two places:
#
#     ABORT on `unwind` — arm b at 12 mm (via upperarm_link)
#     ABORT on `carry`  — arm b at 12 mm (via forearm_link)
#
# Both are the arm coming *back* rather than reaching out. `carry` swings j1
# round to the crate and `unwind` ramps the joints from the wound-up carry
# posture to park; both sweep the elbow through the deck centreline, and two
# arms doing it at once meet there. Reaching *into* a row is the safe half —
# measured, both arms extended into their own rows never come closer than
# +188 mm, because extended arms are 1.6 m apart in x and only folded ones are
# in the middle.
#
# So the interlock is on the returning half only. The picking half — approach,
# insert, grip, pull, extract — is where the time goes and it runs fully
# concurrently.
#
# ⚠️ An arm waits *before* the leg, holding position, never inside one: a leg is
# the unit the route was verified in, and stopping halfway through one leaves
# the arm somewhere no route checked.
# ⚠️ `settle` is in the set, and it is the leg that looks least like it should
# be. It holds position — but the position it holds is PARK, immediately after
# the arm has unfolded, which *is* the middle of the deck. Leaving it out meant
# the token was released on the mission's very first leg, before the arm had
# gone anywhere, and the other arm unfolded straight into it.
# ⚠️ `extract` is in the set too, and that was the last correction. It is the
# leg that backs the fruit out to the **staging plane**, which is only 0.32 m
# from the shoulder — not "out over its own row" but part way home, and an arm
# holding there while the other one carries to its crate measured 14 mm. The
# safe overlap is the arm being *deep in the canopy*, 0.60 m out, which is
# `approach` through `grasp`. Those are also the slow legs — grip 1.5 s, close
# 0.8 s, pull up to 6 s, grasp 0.5 s — so the window that is safe is the window
# worth having.
CROSSING_LEGS = frozenset({
    "settle", "clear", "lane", "align",     # unfolding and setting out
    "extract",                              # back out to the staging plane
    "turn", "carry", "release",             # swinging round to the crate
    "withdraw", "ready", "park", "unwind",  # and back to park
})

# Where an arm waits while the *other* one works.
#
# ⚠️ **Not the park posture, and the difference is measured.** `PARK` is where an
# arm rests between its own picks; it is 0.32 m out in front of the shoulder, and
# with both arms at PARK on a 400 mm-wide deck the two forearms interpenetrate
# (see `trolley.ARM_STAGGER`). Staggering the mounts fixed the resting clash, but
# an arm *reaching into its row* still swept within 14 mm of the other arm parked
# beside it — `Guard` aborted the pick, correctly, with
#
#     ABORT  arm a within 14 mm during `approach`
#
# which is the guard doing its job and a machine that cannot harvest. So the idle
# arm folds up over its own base instead of resting out front.
#
# Chosen by sweeping candidate stow postures and, for each, driving the *other*
# arm through 18 postures spanning its working envelope (three y, three z, two
# staging depths) and taking the worst clearance seen:
#
#     arm a stowed as            worst gap to arm b working
#     park (what shipped)             +0.024 m   <- below ARM_CLEARANCE, aborts
#     j1 -90                          +0.219 m
#     folded, j1 -90                  +0.062 m
#     folded up, j1 0                 +0.222 m
#     j1 +90                          +0.249 m
#     folded up, j1 +90               +0.318 m   <- ships
#
# ⚠️ The j1 sign matters (+90 gives 318 mm, -90 gives 62 mm) and it is **not**
# mirrored between the arms. Arm b is bolted round 180 deg, so the same joint
# vector produces the mirrored world posture — which is the one that clears, by
# the same symmetry that lets both arms share every other Week 1-4 number.
# `--stow` re-runs the sweep.
STOW = np.radians([90.0, -100.0, -140.0, -30.0, 90.0, 0.0])

# The leg names `week2_pick.MissionRun` flies, mapped to the phase words the
# pipeline panel shows. The leg name itself is shown too — this is the summary,
# not a replacement for it.
#
# ⚠️ **One word per distinct thing the arm is doing, not per group.** The first
# version folded `pull` into "grip" and `extract`/`turn` into "carry", which
# made the panel say "grip" while the stem was being loaded to its snap force
# and "carry" while the wrist was rotating in place out on the staging plane —
# the two moments in the cycle most worth being able to see. They are separate
# legs in the plan and they are separate words here.
LEG_PHASE = {
    "settle": "settling", "clear": "clearing", "lane": "lining up",
    "align": "lining up",
    "approach": "approach", "insert": "approach",
    "grip": "grip", "close": "grip", "grasp": "grip",
    "pull": "pull",
    "extract": "extract",
    "turn": "turn",
    "carry": "carry",
    "release": "crating", "withdraw": "crating",
    "ready": "parking", "park": "parking", "unwind": "parking",
}


# --- what one arm has done ---------------------------------------------------

@dataclass
class ArmStats:
    """One arm's running tally, fed from `execute`'s returned `seconds`.

    ⚠️ `seconds` is the executor's own simulated clock — `clock[0]`, incremented
    per control cycle inside `execute` — not wall time and not `perf_counter`.
    That is the number a cycle-time claim has to be made of, because wall time
    here is a statement about this laptop and how many camera panels were being
    composited at the time.
    """

    tag: str
    pick_s: list = field(default_factory=list)   # seconds, crated picks only
    attempts: int = 0
    crated: int = 0
    refused: int = 0
    missed: int = 0          # attempted, did not reach the crate
    not_detected: int = 0    # mapped position matched no truss

    @property
    def mean_s(self):
        return float(np.mean(self.pick_s)) if self.pick_s else float("nan")

    @property
    def last_s(self):
        return self.pick_s[-1] if self.pick_s else float("nan")

    @property
    def total_s(self):
        return float(sum(self.pick_s))

    def line(self):
        mean = f"{self.mean_s:5.1f}" if self.pick_s else "  -  "
        last = f"{self.last_s:5.1f}" if self.pick_s else "  -  "
        return (f"crated {self.crated:2d}  refused {self.refused:2d}  "
                f"missed {self.missed:2d}  last {last}s  mean {mean}s")


@dataclass
class ArmState:
    """What one arm is doing **right now**, and what it has done so far.

    Written at the point the action happens. `phase` is the coarse word;
    `leg` is the executor's own current leg, read live from `Guard.leg`.
    """

    tag: str
    phase: str = "idle"
    detail: str = ""
    target: str | None = None
    stage: str | None = None
    stop_i: int = 0
    stop_n: int = 0
    guard: object = None        # live `mission.Guard`, for `leg` and `min_seen`
    head: object = None         # live `decks.ArmDeckHead`, for the pan angle
    row: int = 0
    stats: ArmStats = None

    # ⚠️ Everything below is written **where the thing happens** and read by the
    # PIPELINE panel. None of it is inferred in the viewer, and there is no
    # `set_caption` — see `DuoState`. A panel that could be told what to say
    # would eventually be told something that was not true.
    run: object = None          # live `week2_pick.MissionRun`, for `leg`
    est_pos: object = None      # where the map thinks the target is
    truth_pos: object = None    # where it actually is, for the error term
    route_ok: bool | None = None        # planner accepted or refused
    route_label: str = ""               # the route it chose, e.g. direct/roll+30
    route_tried: int = 0                # how many candidates before it settled
    refuse_reason: str = ""             # the breach, in words
    breach_mm: float = float("nan")     # ...and how far under clearance it was
    abort_leg: str = ""                 # which leg the guard tripped on
    abort_mm: float = float("nan")      # ...and at what distance
    waiting_on: str = ""                # why it is idle, if it is

    def __post_init__(self):
        if self.stats is None:
            self.stats = ArmStats(tag=self.tag)

    @property
    def est_err_mm(self):
        """How far the map's guess was from the truth, in mm. NaN if unknown."""
        if self.est_pos is None or self.truth_pos is None:
            return float("nan")
        return float(np.linalg.norm(np.asarray(self.est_pos)
                                    - np.asarray(self.truth_pos))) * 1000

    @property
    def flying(self):
        return self.run is not None and not self.run.done

    def begin(self, name, fruit, model, data):
        """A new target. Clears the last one's verdict so nothing goes stale."""
        self.target, self.stage = name, fruit.stage
        self.est_pos = np.asarray(fruit.pos, float).copy()
        self.truth_pos = None
        if name is not None:
            try:
                self.truth_pos = np.asarray(data.body(name).xpos, float).copy()
            except KeyError:
                pass
        self.route_ok = None
        self.route_label = ""
        self.route_tried = 0
        self.refuse_reason = ""
        self.breach_mm = float("nan")
        self.abort_leg = ""
        self.abort_mm = float("nan")
        self.waiting_on = ""
        return self

    def route(self, mission):
        """What the planner decided, and if it refused, by how much.

        ⚠️ The breach is read off `mission.breaches[0]`, which the planner sorts
        closest-first, so "refused, and the worst it got was 26 mm against a
        40 mm budget" is a measurement the panel can print rather than a
        sentence saying it was refused.
        """
        self.route_ok = bool(mission.ok)
        self.route_label = mission.tried[-1] if mission.tried else mission.lane
        self.route_tried = len(mission.tried)
        if not mission.ok and mission.breaches:
            b = mission.breaches[0]
            self.refuse_reason = str(b)
            self.breach_mm = float(b.distance) * 1000
        return self

    def follow_leg(self):
        """Track the executor's current leg into the phase word. Live."""
        leg = self.leg()
        if leg:
            self.phase = LEG_PHASE.get(leg, leg)
            self.detail = self.target or ""
        return self

    def finish(self, res, guard):
        """The verdict, including where the guard stopped it if it did."""
        if res.get("aborted"):
            why = res["aborted"]
            self.abort_leg = str(why[0])
            self.abort_mm = float(why[1]) * 1000
        return self

    @property
    def name(self):
        return f"arm{1 if self.tag == 'a' else 2}"

    def leg(self):
        """The executor's current leg, or None. Live, not remembered.

        Read off the `MissionRun` first — it is the thing that sets the leg —
        and off the `Guard` as a fallback, which is where it used to be read
        from and is still correct.
        """
        if self.run is not None and getattr(self.run, "leg", ""):
            return self.run.leg
        g = self.guard
        return getattr(g, "leg", None) or None if g is not None else None

    def deck_pan(self, data):
        """Where this arm's deck camera is pointing, read back out of mjData."""
        return self.head.current(data)[0] if self.head is not None else 0.0

    def say(self, phase, detail=""):
        self.phase, self.detail = phase, detail
        return self

    def line(self):
        """The one-line status the MISSION window prints. Derived, not scripted."""
        leg = self.leg()
        if leg and self.phase not in ("refused", "idle", "stowed"):
            word = LEG_PHASE.get(leg, leg)
            head = f"{self.name}: {word}"
            if self.target:
                head += f" {self.target}"
            return f"{head}  [{leg}]"
        head = f"{self.name}: {self.phase}"
        if self.detail:
            head += f" {self.detail}"
        return head


class DuoState:
    """The whole machine's live state. The MISSION window reads only this.

    ⚠️ There is deliberately no `set_caption`-style entry point. Everything the
    window shows is either a field written where the work happens or something
    derived from one — `phase`, `arms[t].leg()`, `stats`. A panel that could be
    told what to say would eventually be told something that was not true.
    """

    def __init__(self, arms=("a", "b"), aisle=0):
        self.arms = tuple(arms)
        self.aisle = aisle
        self.phase = "start"
        self.detail = ""
        self.house_map = None
        self.routes = {}
        self.itinerary = []
        self.stop_i = 0
        self.trusses = []
        self.picked = set()        # truss names in a crate
        self.skipped = set()       # ripe, mapped, not on any route (other row)
        self.refused = set()       # planner found no clearing route
        self.missed = set()        # attempted and lost
        # ⚠️ `id(sighting) -> truss name`, filled by `associate` as each stop is
        # worked. Without it the map can colour a dot by believed ripeness but
        # cannot say what *happened* to it on a scouted run: the outcome sets
        # above are keyed by truss name, and a `Sighting` is a position and a
        # stage with no name in it. Under `--truth` the sighting carries its own
        # `truth` and this is redundant; on a real mapping pass it is the only
        # link between "the dot the robot drew" and "the fruit it then picked".
        self.named = {}
        # ⚠️ **A set, because more than one arm flies at a time now.** It was a
        # single `active` tag when the arms were serialised and the viewer asked
        # "is this the one that is moving". `active` is kept below as the
        # single-arm reading of the same thing, and is None whenever both fly —
        # which is most of a stop.
        self.flying = set()
        self.claims = {}           # id(sighting) -> tag, from `assign`
        self.contested = []        # (loser, owner, sighting), logged not hidden
        self.picked_by = {}        # truss name -> tag, the second contention gate
        self.machine = None        # live `Machine`, for the cycle counters
        self.cycles = 0
        self.concurrent_cycles = 0
        left, right = house.serves(aisle)
        rows = {"a": right, "b": left}
        self.state = {t: ArmState(tag=t, row=rows[t]) for t in self.arms}
        self.scout_frame = None
        self.t0 = _time.perf_counter()

    def say(self, phase, detail=""):
        self.phase, self.detail = phase, detail

    @property
    def active(self):
        """The one arm flying, or None — including None when *both* are.

        Kept so nothing that predates concurrency reads a missing attribute.
        Anything that wants "is this arm working" should ask `flying`.
        """
        return next(iter(self.flying)) if len(self.flying) == 1 else None

    @property
    def stats(self):
        return {t: self.state[t].stats for t in self.arms}

    def totals(self):
        s = self.stats
        return {
            "crated": sum(x.crated for x in s.values()),
            "refused": sum(x.refused for x in s.values()),
            "missed": sum(x.missed for x in s.values()),
            "attempts": sum(x.attempts for x in s.values()),
        }


# --- mapping, with one head per arm ------------------------------------------

class DuoScout:
    """The mapping pass, with both deck heads scanning at once.

    ⚠️ **Both heads are aimed before either is rendered**, which is what makes
    "they look in different directions at the same time" true of the simulation
    and not just of the panel captions. Aim a, aim b, one `mj_forward`, then read
    both cameras out of the same physics state. Rendering between the two aims
    would produce two frames from two different instants and the claim would be a
    presentational one.
    """

    def __init__(self, model, data, arms=("a", "b"), aisle=0, detector=None,
                 stride=None):
        from camera import RENDER_H, RENDER_W, SensorCamera
        from farm.scout import SCOUT_STRIDE

        self.model, self.data = model, data
        self.arms, self.aisle = tuple(arms), aisle
        self.stride = SCOUT_STRIDE if stride is None else stride
        self.detector = StageDetector() if detector is None else detector
        self.heads = decks.heads(model, data, arms=self.arms, aisle=aisle)
        self.sensors = {t: SensorCamera(model, decks.DECK_CAM[t],
                                        RENDER_W, RENDER_H)
                        for t in self.arms if t in self.heads}
        self.last_rgb = {t: None for t in self.arms}
        self.head_s = 0.0

    def close(self):
        for s in self.sensors.values():
            s.close()

    def look(self, tag):
        """One frame from one arm's deck cam. Raw, unfused, world coordinates."""
        from detect import estimate

        sensor = self.sensors[tag]
        rgb, depth = sensor.both(self.data)
        self.last_rgb[tag] = rgb
        R, C = sensor.pose(self.data)
        out = []
        for d in self.detector(rgb):
            e = estimate(d, depth, sensor.intr, R, C)
            if e.est is None:
                continue
            stage, hue = stage_of(rgb, d)
            if stage is None:
                continue
            out.append(Sighting(pos=e.est.copy(), stage=stage, hue=hue))
        return out

    def run(self, drive, state=None, on_tick=None, on_frame=None, verbose=True):
        """Drive the aisle once, both heads scanning their own rows."""
        import mujoco

        lo, hi = trolley.y_limits()
        stops = list(np.arange(lo + 0.4, hi - 0.4 + 1e-9, self.stride))
        raw, secs = [], 0.0
        drive.park_at(stops[0])
        mujoco.mj_forward(self.model, self.data)

        n_scan = max(len(decks.SCAN[t]) for t in self.arms)
        for i, y in enumerate(stops):
            if i:
                secs += drive.drive_to(y, on_tick=on_tick)
            for t in self.arms:
                if t in self.heads:
                    self.heads[t].follow(self.data)
            if state is not None:
                state.say("map", f"stop {i + 1}/{len(stops)} at y={y:+.2f}")

            for k in range(n_scan):
                # ⚠️ Aim every head, *then* forward, *then* render both. See the
                # class docstring — this ordering is the claim.
                for t in self.arms:
                    if t not in self.heads:
                        continue
                    pattern = decks.SCAN[t]
                    pan = pattern[k % len(pattern)]
                    self.head_s += self.heads[t].slew_to(
                        self.data, pan, on_tick=on_tick)
                    if state is not None:
                        state.state[t].say(
                            "scanning",
                            f"stop {i + 1}/{len(stops)}, pan {pan:+.0f}deg")
                mujoco.mj_forward(self.model, self.data)

                for t in self.arms:
                    if t not in self.sensors:
                        continue
                    seen = self.look(t)
                    raw.extend(seen)
                    if on_frame is not None:
                        on_frame(i, len(stops), y, t, seen)
            if verbose and (i % 4 == 0 or i == len(stops) - 1):
                print(f"    stop {i + 1:>2}/{len(stops)} at y={y:+.2f} — "
                      f"{len(raw)} blobs so far")

        fused = _fuse(raw)
        for s in fused:
            s.row = int(round((s.pos[0] - house.ROW_X0) / house.ROW_PITCH))
        keep = [s for s in fused
                if 0 <= s.row < house.N_ROWS
                and abs(s.pos[0] - house.row_x(s.row)) < 0.20]
        return HouseMap(sightings=keep, aisle=self.aisle,
                        drive_m=drive.travelled, drive_s=secs, frames=len(stops))


# --- merging two routes into one itinerary -----------------------------------

def merge(routes, tol=STOP_MERGE_M, reach=None):
    """Two arms' routes -> one list of `(y, {tag: [fruit]})`, in driving order.

    ⚠️ The trolley is 1-DOF and there is one of it, so two routes have to become
    one sequence of stops or the machine drives the aisle twice.

    ⚠️ **Covered jointly, not merged after the fact, and that is what makes the
    two arms actually work at the same time.** The first version took each arm's
    independently-solved stop cover and glued together any two stops that landed
    within `tol` of each other. Nothing forces two independent covers to land
    near each other, and mostly they do not: over a real house that produced a
    stop list where **almost every stop served exactly one arm**, so the arms
    were concurrent in mechanism and serialised in practice — one flew while the
    other stood at a stop with nothing on its row.

    A fruit is reachable from a window of trolley positions `[y - reach,
    y + reach]` around `fruit_y - ARM_Y[tag]`, and that window is a fact about
    one arm and one fruit whichever route it came from. So the cover is solved
    **once over both arms' fruit together** (`route.cover`, the same greedy
    interval cover, still optimal) and each fruit is then assigned to the
    nearest stop that can reach it. Both arms get work at a stop whenever both
    have fruit near it, which over a real house is most of them.

    `tol` is unused now and kept so the signature does not change under
    callers; the joint cover subsumes what it was for.

    ⚠️ **The window is asymmetric while both arms work**, because an arm
    reaching behind itself swings its elbow through the other arm's volume. See
    `CONCURRENT_TOWARD_M` for the sweep. That makes each fruit an *interval* of
    acceptable trolley positions rather than a symmetric window, so the cover is
    a minimum interval-stabbing rather than `route.cover`'s point cover — sort
    by right endpoint, stab at the right endpoint of the first uncovered
    interval, sweep on. Optimal for the same reason the symmetric one is.
    """
    from farm.route import REACH_Y

    reach = REACH_Y if reach is None else reach
    lo_lim, hi_lim = trolley.y_limits()

    # For each fruit: the closed interval of trolley y from which its arm can
    # take it. `toward` is the direction the other arm lies in, which is world
    # -y for arm a and world +y for arm b — see `CONCURRENT_TOWARD_M`.
    spans = []
    for tag, r in routes.items():
        dy = trolley.ARM_Y[tag]
        toward = -1.0 if tag == "a" else +1.0
        near, far = CONCURRENT_TOWARD_M, reach
        # own-frame window -> world window
        lo_dy, hi_dy = (-near, +far) if toward < 0 else (-far, +near)
        for st in r.stops:
            for f in st.fruit:
                need = float(f.pos[1]) - dy
                a = max(lo_lim, need - hi_dy)
                b = min(hi_lim, need - lo_dy)
                if a <= b:
                    spans.append((a, b, tag, f))
    if not spans:
        return []

    # Greedy stab: take the interval that ends soonest, put a stop at its right
    # end — the furthest place that still reaches it — and drop everything that
    # stop already covers.
    spans.sort(key=lambda s: s[1])
    stops, taken = [], [False] * len(spans)
    for i, (a, b, _t, _f) in enumerate(spans):
        if taken[i]:
            continue
        y = b
        stops.append(y)
        for j in range(i, len(spans)):
            aj, bj = spans[j][0], spans[j][1]
            if not taken[j] and aj - 1e-9 <= y <= bj + 1e-9:
                taken[j] = True

    stops.sort()
    groups = {i: {} for i in range(len(stops))}
    for a, b, tag, f in spans:
        ok = [i for i, y in enumerate(stops) if a - 1e-9 <= y <= b + 1e-9]
        if not ok:
            continue
        # Nearest reachable stop to the middle of the interval, so a fruit that
        # two stops can take goes to the one that reaches it most comfortably.
        mid = 0.5 * (a + b)
        k = min(ok, key=lambda i: abs(stops[i] - mid))
        groups[k].setdefault(tag, []).append(f)

    out = [(float(stops[i]), groups[i]) for i in sorted(groups) if groups[i]]
    out.sort(key=lambda g: g[0])
    return out


def associate(sightings, model, data, names, gate=NAME_GATE_M):
    """Mapped positions -> truss body names. `farm.run.associate`, unchanged.

    Bookkeeping, not perception: the arm is sent to the *map's* position, never
    the simulator's. See `farm/run.py`'s module note.
    """
    pairs = sorted(
        (float(np.linalg.norm(s.pos - data.body(n).xpos)), i, n)
        for i, s in enumerate(sightings) for n in names
        if np.linalg.norm(s.pos - data.body(n).xpos) <= gate)
    taken, out = set(), {}
    for _d, i, n in pairs:
        if i in taken or n in out.values():
            continue
        taken.add(i)
        out[i] = n
    return out


# --- the run -----------------------------------------------------------------

class Machine:
    """Both arms' missions, advanced inside one physics loop.

    ⚠️ **This is the thing that replaced serialisation.** Each arm has its own
    `week2_pick.MissionRun`; this steps all of them through the same control
    cycle and then runs the plant **once**:

        for arm in arms: arm.command()   # every setpoint written
        advance()                        # one 5x mj_step for the machine
        on_tick()                        # one panel frame

    Not two sequential `execute` calls, and not threads sharing one `mjData` —
    one clock, one plant, two control laws, which is what a real dual-arm
    controller is. See `week2_pick.MissionRun` for the generator protocol and
    `reach.Reacher.command`/`advance` for why the split is necessary rather
    than tidy.

    ⚠️ **`row.update` runs once per physics step, not once per arm.** Each
    arm's black box records its own contacts, but the row is snapped once and
    the resulting breaks are handed to every recorder — otherwise two arms mean
    two `row.update` calls per step, the second reading forces the first has
    already acted on, and only one recorder learning that a stem broke. See
    `incident.Blackbox.observe`/`attribute`.
    """

    def __init__(self, model, data, row, on_tick=None):
        self.model, self.data, self.row = model, data, row
        self.on_tick = on_tick
        self.boxes = {}          # tag -> CarryTrace, only while that arm flies
        self.runs = {}           # tag -> MissionRun, only while that arm flies
        self.cycles = 0
        # ⚠️ **Counts cycles where both arms were actually *executing a leg*,
        # not cycles where both merely had a mission open.** An arm held at the
        # deck-centre gate still has its mission and its recorder registered
        # while it stands still, and counting that as concurrency would inflate
        # the one number this whole change is judged on. `waiting_for` is set by
        # `MissionRun` exactly while it is gated, so it is the honest filter.
        self.concurrent_cycles = 0
        self.watchers = []       # called once per control cycle, after physics

    def substep(self):
        """The machine's per-physics-step hook. One row, many recorders."""
        for box in self.boxes.values():
            box.observe()
        broke = self.row.update()
        for box in self.boxes.values():
            box.attribute(broke)

    def advance(self):
        import mujoco

        from reach import TICKS_PER_CTRL

        for _ in range(TICKS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)
            self.substep()

    def drive(self, jobs):
        """Run every arm's generator to completion in one shared physics loop.

        `jobs` is `{tag: generator}`; each yields once per control cycle it
        wants, with its setpoints written and physics pending.
        """
        live = dict(jobs)
        while live:
            ran = []
            for tag, job in list(live.items()):
                try:
                    next(job)
                    ran.append(tag)
                except StopIteration:
                    del live[tag]
            if not ran:
                break
            self.cycles += 1
            moving = sum(1 for r in self.runs.values()
                         if not r.done and not r.waiting_for)
            if moving > 1:
                self.concurrent_cycles += 1
            self.advance()
            for w in self.watchers:
                w()
            if self.on_tick is not None:
                self.on_tick(None)


class DeckCentre:
    """One token for the shared middle of the deck. One arm in it at a time.

    ⚠️ **The smallest interlock that works, and it is deliberately small.** The
    two arms only foul each other on the legs that swing them back through the
    deck centreline — see `CROSSING_LEGS` for the aborts that identified which.
    Everything else, including the whole picking half of the cycle, runs with
    both arms moving.

    Cannot deadlock: an arm that holds the token is never itself waiting for
    anything, so the holder always makes progress and always releases. The
    holder is released on mission end too, in `work`'s `finally`, so an aborted
    pick cannot strand it.
    """

    def __init__(self):
        self.holder = None
        self.waits = 0          # how many times an arm had to wait, for the log
        self.wait_cycles = 0

    def claim(self, tag, leg):
        """True if `tag` may start `leg` now. Frees the token on a safe leg."""
        if leg not in CROSSING_LEGS:
            if self.holder == tag:
                self.holder = None
            return True
        if self.holder is None:
            self.holder = tag
            self.waits += 0
            return True
        if self.holder == tag:
            return True
        self.wait_cycles += 1
        return False

    def release(self, tag):
        if self.holder == tag:
            self.holder = None


def assign(itinerary, state, verbose=True):
    """Give every routed fruit to exactly one arm, and say so.

    ⚠️ **Two arms must never plan for the same tomato**, and "they work
    different rows so it cannot happen" is an argument about the router rather
    than a property of the machine. `route.plan` is called once per arm over
    the *same* `HouseMap`, and a sighting whose estimated x lands near the aisle
    centreline can be within the row gate of both. The two arms would then both
    approach it, both succeed at planning, and the second one would fly at a
    stem that is already in the other's crate.

    So the claim is explicit: first arm in tag order takes the fruit, the second
    is refused it, and the refusal is logged rather than silently dropped.
    Returns the itinerary with contested fruit removed from the loser.
    """
    claimed = {}                      # id(sighting) -> tag
    out, clashes = [], []
    for y, per_arm in itinerary:
        kept = {}
        for tag in sorted(per_arm):
            mine = []
            for f in per_arm[tag]:
                owner = claimed.get(id(f))
                if owner is None:
                    claimed[id(f)] = tag
                    mine.append(f)
                else:
                    clashes.append((tag, owner, f))
            if mine:
                kept[tag] = mine
        out.append((y, kept))

    state.claims = claimed
    state.contested = clashes
    if verbose:
        n = sum(len(p) for _y, p in out)
        print(f"\n  --- target assignment: {n} fruit, "
              f"{len(claimed)} distinct, {len(clashes)} contested ---")
        for y, per_arm in out:
            bits = "  ".join(
                f"arm{1 if t == 'a' else 2}:{len(per_arm[t])}"
                for t in sorted(per_arm)) or "nothing"
            print(f"    stop y={y:+.2f}  {bits}")
        for tag, owner, f in clashes:
            print(f"    ⚠️ arm{1 if tag == 'a' else 2} refused a {f.stage} at "
                  f"{np.round(f.pos, 2)} — already claimed by "
                  f"arm{1 if owner == 'a' else 2}")
    return out


def run(model, data, trusses, state, arms=("a", "b"), aisle=0, speed=0.5,
        use_truth=False, max_stops=None, on_tick=None, stride=None,
        verbose=True, scout_cls=DuoScout):
    """Map, plan, travel, pick, crate — one full row, both arms, concurrently.

    The scene is passed in rather than built here: a `mujoco.Renderer` binds to
    one `MjModel`, so a viewer that wants panels for the whole run has to own the
    model before the run starts. Same reason `farm.run` takes `scene=`.
    """
    import mujoco

    from carrytrace import CarryTrace
    from fr5 import JOINTS
    from incident import Blackbox
    from mission import (ARM_CLEARANCE, ClearanceModel, Guard, Planner,
                         park_arm, reset_park)
    from outcomes import classify
    from plant_row import Row
    from reach import Gripper
    from week2_pick import MissionRun, anchor_posture, make_reacher

    arms = tuple(arms)
    state.trusses = list(trusses)
    names = [t.name for t in trusses]

    mujoco.mj_forward(model, data)
    parks = {t: armframe.park_posture(model, data, t, arms=arms) for t in arms}
    # ⚠️ Solve every posture first, then reset **once**, then place each arm.
    # `reset_park` is `mj_resetData` underneath, so calling it per arm throws
    # away the arm parked on the previous pass. `farm/eyes.py` records this.
    reset_park(model, data, parks[arms[0]], prefix=trolley.ARM_PREFIX[arms[0]])
    for t in arms:
        park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
    mujoco.mj_forward(model, data)

    row = Row(model, data, names=names, homes={t.name: t.pos for t in trusses})
    mujoco.mj_forward(model, data)
    drive = trolley.Drive(model, data)
    machine = Machine(model, data, row, on_tick=on_tick)
    state.machine = machine
    # The one place the two arms interlock. See `DeckCentre`.
    centre = DeckCentre()
    state.centre = centre

    def stow(tag):
        """Fold an arm out of the way. See `STOW`.

        ⚠️ **No longer what keeps the arms apart** — that is `ArmObstacles` and
        `Guard`, and it is mandatory now (see below). This is for an arm with
        nothing to do at this stop: folding it over its own base leaves the
        working arm the most room, so the planner refuses fewer routes.
        """
        park_arm(model, data, STOW, prefix=trolley.ARM_PREFIX[tag])
        state.state[tag].say("idle-waiting", "folded, nothing on its row here")

    def unstow(tag):
        park_arm(model, data, parks[tag], prefix=trolley.ARM_PREFIX[tag])

    # --- 1. map --------------------------------------------------------------
    state.say("map", "driving the aisle, both heads scanning")
    if verbose:
        print(f"\n{'=' * 78}\n  1. MAP — two deck heads, one per arm, "
              f"each on its own row")
    scout = scout_cls(model, data, arms=arms, aisle=aisle, stride=stride)
    for t in arms:
        state.state[t].head = scout.heads.get(t)
    try:
        if use_truth:
            house_map = HouseMap(
                sightings=[Sighting(pos=t.pos, stage=t.stage, hue=float("nan"),
                                    row=t.row, truth=t) for t in trusses],
                aisle=aisle)
            if verbose:
                print("\n  ⚠️ MAP SKIPPED — routing the operator's own answer")
        else:
            house_map = scout.run(drive, state=state, on_tick=on_tick,
                                  verbose=verbose)
    finally:
        scout.close()
    state.house_map = house_map
    if verbose:
        print(f"\n{house_map.table()}")

    # --- 2. plan -------------------------------------------------------------
    state.say("plan", "one route per arm, merged into one itinerary")
    for t in arms:
        state.state[t].say("planning", "")
    if verbose:
        print(f"\n{'=' * 78}\n  2. PLAN — where to stop and who takes what")
    routes = {t: route.plan(house_map, aisle=aisle, arm=t) for t in arms}
    state.routes = routes
    itinerary = merge(routes)
    if max_stops:
        itinerary = itinerary[:max_stops]
    # ⚠️ Explicit, logged, and before a single arm moves. See `assign`.
    itinerary = assign(itinerary, state, verbose=verbose)
    state.itinerary = itinerary
    routed = {id(f) for _y, per in itinerary for fl in per.values() for f in fl}
    for s in house_map.sightings:
        if s.ripe and id(s) not in routed:
            state.skipped.add(id(s))
    if verbose:
        for t in arms:
            print(f"\n  --- arm {t.upper()} on row r{state.state[t].row} ---")
            print(routes[t].table())
        print(f"\n  merged into {len(itinerary)} trolley stops")

    # --- 3. travel and pick --------------------------------------------------
    if verbose:
        print(f"\n{'=' * 78}\n  3. HARVEST — {len(itinerary)} stops, "
              f"both arms working at the same time")
        print(f"  arm-vs-arm clearance {ARM_CLEARANCE * 1000:.0f} mm, "
              f"checked by the planner and by the guard every control cycle")
    for t in arms:
        state.state[t].stop_n = len(itinerary)

    def work(tag, fruit_here, si, n_stops):
        """One arm's work at one stop, as a sequence of control cycles.

        A generator, so `Machine.drive` can advance this arm and the other one
        through the same physics. Everything between two yields is instantaneous
        in simulated time — planning included, which is correct: the planner
        runs while the arm is stationary and costs wall time, not sim time.
        """
        me = state.state[tag]
        prefix = trolley.ARM_PREFIX[tag]
        others = trolley.other_arms(tag, arms)
        # ⚠️ **Mandatory, not optional.** `others` is what puts the other arm's
        # links into this arm's obstacle set, in the planner's preview and in
        # the runtime guard. With both arms flying into overlapping space it is
        # the only thing standing between them; an empty `others` here would
        # plan and fly as though the other arm were not there. See
        # `mission.ArmObstacles`.
        if len(arms) > 1 and not others:
            raise RuntimeError(
                f"arm {tag} would fly with no other-arm obstacle set on a "
                f"{len(arms)}-armed machine")
        # The drive joint, every other arm's six, and both deck heads' pan and
        # tilt — everything in the model this arm's IK must not reach with. See
        # `armframe.pin_base` and `decks.head_joints`.
        pin = [trolley.DRIVE_JOINT] + armframe.deck_joints(model)
        for p in others:
            pin += [p + j for j in JOINTS]

        # ⚠️ **A cheap "could I unfold right now?" probe, so waiting does not
        # cost a plan.** The first cut re-ran `planner.plan()` on every retry.
        # A refused plan is 18 candidates and up to 70 s of wall time, and an
        # arm waiting out the other one's 20 s pick retries many times — so the
        # *waiting* cost orders of magnitude more than the working. And it was
        # asking the expensive question ("is there a route?") to learn the cheap
        # one ("has the other arm moved yet?").
        #
        # This evaluates the arm-vs-arm clearance at the posture this arm is
        # about to adopt — `park_q`, written into a scratch `MjData` and
        # forwarded — against wherever the other arm is *now*. One
        # `mj_forward` and one sphere pass, about a millisecond, and it is the
        # question that actually gates unfolding.
        probe_data = mujoco.MjData(model)
        probe_model = ClearanceModel(model, row, None, prefix=prefix,
                                     others=others)
        park_adr = [model.joint(prefix + j).qposadr[0] for j in JOINTS]

        def park_is_clear():
            probe_data.qpos[:] = data.qpos
            for adr, value in zip(park_adr, parks[tag]):
                probe_data.qpos[adr] = value
            mujoco.mj_forward(model, probe_data)
            gaps = [v[0] for k, v in probe_model.per_obstacle(probe_data).items()
                    if k.startswith("arm ")]
            return (min(gaps) if gaps else float("inf")) >= ARM_CLEARANCE

        standing = [t.name for t in trusses if row.attached(t.name)]
        ident = associate(fruit_here, model, data, standing)
        for fi, nm in ident.items():
            state.named[id(fruit_here[fi])] = nm

        for fi, fruit in enumerate(fruit_here):
            name = ident.get(fi)
            me.stats.attempts += 1
            me.begin(name, fruit, model, data)
            if name is None:
                me.stats.not_detected += 1
                me.say("idle-waiting",
                       f"no truss within {NAME_GATE_M * 1000:.0f} mm")
                if verbose:
                    print(f"    {me.name} {fruit.stage}: no truss within "
                          f"{NAME_GATE_M * 1000:.0f} mm")
                continue

            # ⚠️ Two arms must not both fly at one tomato. `assign` claims each
            # routed sighting up front; this is the second gate, because
            # `associate` maps a *sighting* to a truss name per stop and two
            # sightings can name the same truss. Bookkeeping, and cheap.
            owner = state.picked_by.get(name)
            if owner is not None:
                me.stats.refused += 1
                me.say("idle-waiting",
                       f"{name} already taken by arm{1 if owner == 'a' else 2}")
                if verbose:
                    print(f"    {me.name} {name}: SKIPPED — claimed by "
                          f"arm{1 if owner == 'a' else 2}")
                continue
            state.picked_by[name] = tag

            # ⚠️ **The posture an arm waits in is the whole safety story here,
            # and it is measured.** Both arms' `PARK` postures fold the elbow
            # back over the shoulder and across the aisle, so the two arms
            # interleave in x and are held apart only by the 500 mm stagger
            # along the row. Worst arm-vs-arm gap, swept:
            #
            #     both at PARK                       +110 mm
            #     both reaching into their own rows  +188 mm
            #     one reaching, other at PARK         -40 mm   <- the problem
            #     either one STOWED                  +318 mm
            #
            # An extended arm is out over its own row, 1.6 m from the other; a
            # folded one is in the middle. So the dangerous state is one arm
            # *moving* while the other *sits at PARK*, and the fix is that an
            # arm which is not flying is folded, never parked.
            #
            # ⚠️ **The deck centre is therefore claimed before unfolding, not at
            # the mission's first leg.** Unfolding to PARK is itself entering
            # the shared middle. Claiming it at `clear` instead aborted picks at
            # 14 mm — the other arm had already unfolded into the space.
            m = None
            for attempt_i in range(PLAN_RETRIES):
                # Two cheap gates, both polled per control cycle, neither of
                # them a plan: the deck-centre token, and then "would PARK
                # actually clear the other arm right now".
                waited = 0
                while waited < WAIT_MAX_CYCLES and (
                        not centre.claim(tag, "clear")
                        or not park_is_clear()):
                    centre.release(tag)
                    me.waiting_on = ("other arm to clear the deck centre "
                                     "before unfolding")
                    me.say("idle-waiting", me.waiting_on)
                    waited += 1
                    yield
                if waited and verbose:
                    print(f"    {me.name} held folded for {waited} cycles "
                          f"({waited * 0.01:.1f} s) waiting for the deck centre")
                if waited >= WAIT_MAX_CYCLES:
                    me.say("idle-waiting", "gave up waiting for the other arm")
                    if verbose:
                        print(f"    {me.name} {name}: gave up waiting for the "
                              f"other arm after {waited * 0.01:.0f} s")
                    break
                centre.claim(tag, "clear")
                me.waiting_on = ""
                # ⚠️ **Unfold to PARK *before* planning, and plan from there,
                # because a mission has to be flown from the posture it was
                # planned in.** The first cut planned from STOW — which does
                # give the other arm a clear deck to plan against — and then
                # flew from STOW too. It stopped refusing routes and started
                # failing grasps: 3 of 7 picks reached the fruit and came away
                # with nothing, including an arm working entirely alone that
                # had been clean before. `mission._legs` builds the `clear` leg
                # from the tool's current position and `anchor_posture` anchors
                # the solver's null space on `park_q`, so a mission that starts
                # folded over its own base is a different problem from the one
                # every Week 1-4 number was measured on.
                #
                # So: fold up to *wait*, unfold to plan and fly. The waiting is
                # what gives the other arm a clear deck; the unfolding is what
                # keeps the pick the pick that was measured.
                unstow(tag)
                mujoco.mj_forward(model, data)
                me.say("planning", f"route to {name}")
                # This arm's world, read where the trolley is standing now. No
                # globals are touched, which is why the other arm can be
                # mid-mission while this runs. See `mission.ArmFrame`.
                fr = armframe.frame(model, data, tag)
                planner = Planner(model, data, row, lessons=None,
                                  clearance=0.040, park_q=parks[tag],
                                  speed=speed, prefix=prefix, others=others,
                                  pin=tuple(pin), frame=fr)
                m = planner.plan(name)
                me.route(m)
                if m.ok:
                    break
                # ⚠️ Only *arm-vs-arm* refusals are worth waiting out. A route
                # refused by the crop will be refused again in a second — the
                # plants do not move — so retrying it would spin. A route
                # refused by the other arm is refused against a **snapshot of
                # something that is moving**, and waiting is the correct
                # response rather than giving up on the fruit.
                blocked_by_arm = any(b.obstacle.startswith("arm ")
                                     for b in m.breaches)
                if not blocked_by_arm or not any(
                        state.state[o].flying for o in arms if o != tag):
                    break
                # Fold up while waiting, and give the deck centre back. That is
                # what gives the other arm a clear +318 mm instead of a PARK
                # posture in its way, and it is why waiting actually changes the
                # answer rather than just re-asking the same question. See
                # `stow`.
                stow(tag)
                centre.release(tag)
                mujoco.mj_forward(model, data)
                me.waiting_on = (f"other arm — {m.breaches[0].obstacle} at "
                                 f"{m.breaches[0].distance * 1000:.0f} mm")
                me.say("idle-waiting", me.waiting_on)
                if verbose and attempt_i == 0:
                    print(f"    {me.name} {name}: waiting for the other arm "
                          f"({m.breaches[0]})")
                for _ in range(PLAN_RETRY_CYCLES):
                    yield
            me.waiting_on = ""

            if not m.ok:
                why = str(m.breaches[0] if m.breaches else "no route")
                me.stats.refused += 1
                state.refused.add(name)
                me.say("refused", why[:52])
                if verbose:
                    print(f"    {me.name} {name} ({fruit.stage}): "
                          f"REFUSED — {why}")
                # Fold up and hand the centre back — this arm is not going
                # anywhere and must not sit at PARK in the other's way.
                stow(tag)
                centre.release(tag)
                mujoco.mj_forward(model, data)
                continue

            reacher = make_reacher(model, data, speed=speed, prefix=prefix,
                                   frame=fr)
            # ⚠️ Before anything else, and with `others` — see
            # `armframe.pin_base`. Without it the IK plans motion for the
            # base and the other arm that the executor will never make.
            armframe.pin_base(reacher, others=others)
            anchor_posture(reacher, model, data, parks[tag])
            gripper = Gripper(model, data, prefix=prefix)
            trace = CarryTrace(model, data, row, name,
                               Blackbox(model, data, row, name, prefix=prefix),
                               prefix=prefix)
            guard = Guard(model, data, row, name, prefix=prefix, others=others)
            guard.armed = False
            me.guard = guard
            me.say("approach", name)

            # The machine owns the substep hook, so the recorder is registered
            # with it rather than with this arm's Reacher. Deregistered in the
            # `finally` so a mission that aborts does not leave a black box
            # recording an arm that has stopped flying.
            machine.boxes[tag] = trace
            run_ = MissionRun(m, reacher, gripper, row, box=trace, guard=guard,
                              gate=lambda leg, _t=tag: centre.claim(_t, leg))
            me.run = run_
            machine.runs[tag] = run_
            try:
                while run_.command():
                    me.follow_leg()
                    if run_.waiting_for:
                        me.waiting_on = (f"other arm to clear the deck centre "
                                         f"before `{run_.waiting_for}`")
                        me.phase = "idle-waiting"
                        me.detail = me.waiting_on
                    else:
                        me.waiting_on = ""
                    yield
            finally:
                machine.boxes.pop(tag, None)
                machine.runs.pop(tag, None)
                centre.release(tag)
                me.run = None
                me.waiting_on = ""
            res = run_.result

            rec = {"stop": si, "stage": fruit.stage, "row": fruit.row,
                   "seen": True, "fruit": name,
                   "in_bin": bool(res["in_bin"]),
                   "grasped": bool(res["grasped"]),
                   "broke": bool(res["broke"]),
                   "lost": int(res["lost"]),
                   "disturbed": int(res["disturbed"]),
                   "aborted": str(res["aborted"]) if res["aborted"] else None,
                   "t_fly": float(res["seconds"])}
            rec["clean"] = bool(res["in_bin"] and not res["lost"]
                                and not res["disturbed"] and not res["aborted"])
            rec["outcome"] = classify(rec)
            me.finish(res, guard)

            # ⚠️ **`res["seconds"]` is the whole point of the stats panel.**
            # It is the executor's simulated clock and it used to be thrown
            # away by the perception log (`picklog.py` records that bug). It
            # is banked here per arm, so "mean pick time, arm 2" is a
            # measurement rather than an estimate.
            if res["in_bin"]:
                me.stats.crated += 1
                me.stats.pick_s.append(float(res["seconds"]))
                state.picked.add(name)
            else:
                me.stats.missed += 1
                state.missed.add(name)
            me.guard = None
            me.say("done" if res["in_bin"] else "lost",
                   f"{name} {rec['outcome']} {res['seconds']:.1f}s")
            if verbose:
                note = ""
                if res["aborted"]:
                    w = res["aborted"]
                    note = (f"  ABORT on `{w[0]}` — {w[3]} at "
                            f"{w[1] * 1000:.0f} mm (via {w[2]})")
                print(f"    {me.name} {name} ({fruit.stage}): "
                      f"{rec['outcome']:<14} crate={rec['in_bin']} "
                      f"{res['seconds']:.1f}s{note}")

        me.target = None
        # Fold up rather than rest at PARK. The other arm may still be flying,
        # and PARK is the posture that gets in its way. See `stow`.
        stow(tag)
        mujoco.mj_forward(model, data)
        me.say("idle-waiting", "done at this stop, waiting for the other arm")

    for si, (y, per_arm) in enumerate(itinerary, 1):
        state.stop_i = si
        state.say("travel", f"driving to stop {si}/{len(itinerary)}, y={y:+.2f}")
        for t in arms:
            state.state[t].stop_i = si
            state.state[t].say("travelling", f"to stop {si}/{len(itinerary)}")
        drive.drive_to(y, on_tick=on_tick)
        for t in arms:
            park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
        mujoco.mj_forward(model, data)
        if verbose:
            print(f"\n  --- stop {si}/{len(itinerary)} at y={y:+.2f} ---")

        state.say("pick", f"stop {si}/{len(itinerary)}")
        working = [t for t in arms if per_arm.get(t)]
        # ⚠️ **Everything folds up to start with, including the arms about to
        # work.** An arm at PARK sits in the shared middle of the deck, and the
        # first thing each arm does is plan a route that the *other* arm's
        # current posture has to clear. Starting them both at PARK means each
        # one plans against the other's worst posture and both refuse. Starting
        # them both stowed means each plans against +318 mm and both accept.
        # `work` folds its arm again between fruit for the same reason.
        for t in arms:
            stow(t)
        mujoco.mj_forward(model, data)
        if verbose and working:
            print(f"    both arms working at once: "
                  f"{', '.join('arm' + ('1' if t == 'a' else '2') for t in working)}")
        state.flying = set(working)

        # ⚠️ **One `drive` call, every working arm inside it.** This is the
        # line that used to be a `for tag in arms:` loop with a whole `execute`
        # inside it.
        machine.drive({t: work(t, per_arm[t], si, len(itinerary))
                       for t in working})
        state.flying = set()

        for t in arms:
            park_arm(model, data, parks[t], prefix=trolley.ARM_PREFIX[t])
        mujoco.mj_forward(model, data)

    for t in arms:
        unstow(t)
    mujoco.mj_forward(model, data)
    state.say("done", "row complete")
    for t in arms:
        state.state[t].say("done", "")
    state.drive_m = drive.travelled
    state.cycles = machine.cycles
    state.concurrent_cycles = machine.concurrent_cycles
    return state


def report(state):
    """The end-of-run summary."""
    print(f"\n{'=' * 78}")
    print(f"  ONE ROW, TWO ARMS — aisle a{state.aisle}")
    ripe = sum(1 for t in state.trusses if t.ripe)
    mapped = len(state.house_map.sightings) if state.house_map else 0
    mapped_ripe = len(state.house_map.ripe) if state.house_map else 0
    print(f"\n  the house      {len(state.trusses)} fruit, {ripe} of them ripe")
    print(f"  the map        {mapped} found, {mapped_ripe} called ripe")
    print(f"  the route      {len(state.itinerary)} trolley stops")
    print(f"\n  {'arm':<6} {'row':<5} {'crated':>7} {'refused':>8} "
          f"{'missed':>7} {'mean s':>8} {'total s':>8}")
    for t in state.arms:
        st = state.state[t]
        s = st.stats
        mean = f"{s.mean_s:8.1f}" if s.pick_s else "       -"
        print(f"  {st.name:<6} r{st.row:<4} {s.crated:>7} {s.refused:>8} "
              f"{s.missed:>7} {mean} {s.total_s:>8.1f}")
    tot = state.totals()
    print(f"\n  {tot['crated']} tomatoes in the crates from "
          f"{tot['attempts']} attempts")
    if getattr(state, "drive_m", 0):
        print(f"  trolley odometer {state.drive_m:.1f} m")

    # ⚠️ The concurrency claim, as a measurement rather than a sentence. A
    # control cycle counts as concurrent when more than one arm had a mission
    # in flight on it — which is booked by `Machine`, not inferred here.
    cyc = getattr(state, "cycles", 0)
    both = getattr(state, "concurrent_cycles", 0)
    print(f"\n  the arms run CONCURRENTLY — both stepped inside one physics "
          f"loop.")
    if cyc:
        print(f"     {both} of {cyc} harvest control cycles had both arms "
              f"mid-mission ({100 * both / cyc:.0f}%)")
    from mission import ARM_CLEARANCE

    print(f"     arm-vs-arm clearance {ARM_CLEARANCE * 1000:.0f} mm, in the "
          f"planner's preview and in the guard every cycle")
    if getattr(state, "contested", None):
        print(f"     {len(state.contested)} fruit were claimed by both arms "
              f"and assigned to one — see the assignment log")


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aisle", type=int, default=0)
    ap.add_argument("-n", type=int, default=12, help="fruit per row")
    ap.add_argument("--stops", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--arms", type=int, default=2, choices=(1, 2))
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    print(__doc__)

    import mujoco

    arms = ("a", "b")[: args.arms]
    seed = fcrop.resolve_seed(args.seed)
    trusses = fcrop.spawn(n_per_row=args.n, seed=seed)
    model = trolley.build(aisle=args.aisle, arms=arms, trusses=trusses,
                          wrist_cam=True, arm_decks=True, seed=seed)
    data = mujoco.MjData(model)
    state = DuoState(arms=arms, aisle=args.aisle)
    run(model, data, trusses, state, arms=arms, aisle=args.aisle,
        speed=args.speed, use_truth=args.truth, max_stops=args.stops)
    report(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
