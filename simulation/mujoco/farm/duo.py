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
    pick     serialised. One arm flies at a time. See below — this is forced,
             not chosen.

--- why the arms are serialised, which is the honest answer -----------------

⚠️ **`farm.armframe.at_trolley` rebinds `mission`'s module globals.** Weeks 1-4
are written in absolute world coordinates — `PARK`, `STAGE_X`, `BIN_POS`,
`ROW_X`, `INTO_ROW` are module constants — and the adapter rebinds all of them
to the current arm's frame for the duration of a mission, with a *mirror* for
arm b because it is bolted round 180 deg. Two arms mid-mission at once would
need two conflicting sets of those globals in one interpreter. Concurrent arms
are not merely uncollided here, they are **structurally impossible** until
`mission` is refactored to plan in the arm's own frame — which `armframe`'s own
docstring says is the right eventual fix and why that file should delete itself.

So: one arm flies, the other is stowed, and the viewer says so on screen. That
is a real limitation of this build and it is stated rather than hidden.

⚠️ **Serialising is not on its own enough, and this is the part that bit.** A
stationary arm is still 22 kg of steel inside the other's working volume — the
mounts are 400 mm apart and each arm reaches 922 mm. So the idle arm is *also*
put in the flying arm's obstacle set (`mission.ArmObstacles`) and *also* moved
out of the way (`STOW`). Belt and braces, because the failure mode is two arms
occupying the same cubic metre and the cost of being wrong is both of them.

--- what is exposed, and why that matters ------------------------------------

⚠️ Every string the viewer prints comes from `ArmState`, which is written **at
the point the thing happens** — `phase` is set beside the call it describes, and
`leg` is read live out of `mission.Guard.leg`, which `week2_pick.execute` sets
per leg as it flies. Nothing here is a script of captions replayed on a timer. If
the planner refuses a fruit, the panel says so because `refuse()` was called with
the breach the planner returned, not because a refusal was due.

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

# The leg names `week2_pick.execute` flies, grouped into words a person watching
# can follow. The leg name itself is shown too — this is the summary, not a
# replacement for it.
LEG_PHASE = {
    "settle": "settling", "clear": "clearing", "lane": "lining up",
    "approach": "approach", "insert": "approach",
    "grip": "grip", "close": "grip", "pull": "grip", "grasp": "grip",
    "extract": "carry", "turn": "carry", "carry": "carry",
    "release": "crating", "withdraw": "crating",
    "ready": "returning", "park": "returning", "unwind": "returning",
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

    def __post_init__(self):
        if self.stats is None:
            self.stats = ArmStats(tag=self.tag)

    @property
    def name(self):
        return f"arm{1 if self.tag == 'a' else 2}"

    def leg(self):
        """The executor's current leg, or None. Live, not remembered."""
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
        self.active = None         # which arm is flying, or None
        left, right = house.serves(aisle)
        rows = {"a": right, "b": left}
        self.state = {t: ArmState(tag=t, row=rows[t]) for t in self.arms}
        self.scout_frame = None
        self.t0 = _time.perf_counter()

    def say(self, phase, detail=""):
        self.phase, self.detail = phase, detail

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

def merge(routes, tol=STOP_MERGE_M):
    """Two arms' routes -> one list of `(y, {tag: [fruit]})`, in driving order.

    ⚠️ The trolley is 1-DOF and there is one of it. Two independently planned
    stop covers land at unrelated y, so driven naively the machine shuffles back
    and forth by a few centimetres between arms. Stops within `tol` become one
    stop at their mean, which both arms can reach — `route.REACH_Y` is 0.48 m,
    so moving a stop by up to half of `tol` costs nothing anybody can measure.
    """
    items = []
    for tag, r in routes.items():
        for st in r.stops:
            items.append((float(st.y), tag, list(st.fruit)))
    items.sort(key=lambda z: z[0])

    out = []
    for y, tag, fruit in items:
        if out and abs(y - out[-1][0]) <= tol:
            group = out[-1]
            group[1].setdefault(tag, []).extend(fruit)
            group[2].append(y)
            group[0] = float(np.mean(group[2]))
        else:
            out.append([y, {tag: list(fruit)}, [y]])
    return [(g[0], g[1]) for g in out]


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

def run(model, data, trusses, state, arms=("a", "b"), aisle=0, speed=0.5,
        use_truth=False, max_stops=None, on_tick=None, stride=None,
        verbose=True, scout_cls=DuoScout):
    """Map, plan, travel, pick, crate — one full row, both arms. Fills `state`.

    The scene is passed in rather than built here: a `mujoco.Renderer` binds to
    one `MjModel`, so a viewer that wants panels for the whole run has to own the
    model before the run starts. Same reason `farm.run` takes `scene=`.
    """
    import mujoco

    from carrytrace import CarryTrace
    from fr5 import JOINTS
    from incident import Blackbox
    from mission import Guard, Planner, park_arm, reset_park
    from outcomes import classify
    from plant_row import Row
    from reach import Gripper
    from week2_pick import Aborted, anchor_posture, execute, make_reacher

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

    def stow(tag):
        """Fold an arm out of the other's way. See `STOW`."""
        park_arm(model, data, STOW, prefix=trolley.ARM_PREFIX[tag])
        state.state[tag].say("stowed", "folded, waiting its turn")

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
              f"one arm flying at a time")
    for t in arms:
        state.state[t].stop_n = len(itinerary)

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
        for tag in arms:
            fruit_here = per_arm.get(tag, [])
            me = state.state[tag]
            if not fruit_here:
                me.say("idle", "nothing on its row here")
                stow(tag)
                mujoco.mj_forward(model, data)
                continue

            # ⚠️ Serialised: this arm works, every other arm folds away. See the
            # module docstring for why concurrency is not merely unimplemented.
            for other in arms:
                if other != tag:
                    stow(other)
            unstow(tag)
            mujoco.mj_forward(model, data)
            state.active = tag

            prefix = trolley.ARM_PREFIX[tag]
            others = trolley.other_arms(tag, arms)
            pin = [trolley.DRIVE_JOINT]
            for p in others:
                pin += [p + j for j in JOINTS]

            standing = [t.name for t in trusses if row.attached(t.name)]
            ident = associate(fruit_here, model, data, standing)

            for fi, fruit in enumerate(fruit_here):
                name = ident.get(fi)
                me.stats.attempts += 1
                me.target, me.stage = name, fruit.stage
                if name is None:
                    me.stats.not_detected += 1
                    me.say("no match",
                           f"nothing within {NAME_GATE_M * 1000:.0f} mm")
                    if verbose:
                        print(f"    {me.name} {fruit.stage}: no truss within "
                              f"{NAME_GATE_M * 1000:.0f} mm")
                    continue

                me.say("planning", f"route to {name}")
                with armframe.at_trolley(model, data, tag):
                    planner = Planner(model, data, row, lessons=None,
                                      clearance=0.040, park_q=parks[tag],
                                      speed=speed, prefix=prefix,
                                      others=others, pin=tuple(pin))
                    m = planner.plan(name)

                if not m.ok:
                    why = str(m.breaches[0] if m.breaches else "no route")
                    me.stats.refused += 1
                    state.refused.add(name)
                    me.say("refused", why[:52])
                    if verbose:
                        print(f"    {me.name} {name} ({fruit.stage}): "
                              f"REFUSED — {why}")
                    continue

                reacher = make_reacher(model, data, speed=speed, prefix=prefix)
                # ⚠️ Before anything else, and with `others` — see
                # `armframe.pin_base`. Without it the IK plans motion for the
                # base and the other arm that the executor will never make.
                armframe.pin_base(reacher, others=others)
                anchor_posture(reacher, model, data, parks[tag])
                gripper = Gripper(model, data, prefix=prefix)
                trace = CarryTrace(model, data, row, name,
                                   Blackbox(model, data, row, name,
                                            prefix=prefix),
                                   prefix=prefix)
                guard = Guard(model, data, row, name, prefix=prefix,
                              others=others)
                guard.armed = False
                me.guard = guard
                me.say("flying", name)

                tick = trace.tick if on_tick is None else (
                    lambda t=None, _tr=trace: (_tr.tick(t), on_tick(t)))
                try:
                    with armframe.at_trolley(model, data, tag):
                        res = execute(m, reacher, gripper, row, box=trace,
                                      guard=guard, on_tick=tick)
                    aborted = res.get("aborted")
                except Aborted as stop_why:
                    res = {"in_bin": False, "grasped": False, "broke": False,
                           "lost": 0, "disturbed": 0, "seconds": 0.0,
                           "peak_n": 0.0, "clearance": float("nan")}
                    aborted = stop_why.why

                rec = {"stop": si, "stage": fruit.stage, "row": fruit.row,
                       "seen": True, "fruit": name,
                       "in_bin": bool(res["in_bin"]),
                       "grasped": bool(res["grasped"]),
                       "broke": bool(res["broke"]),
                       "lost": int(res["lost"]),
                       "disturbed": int(res["disturbed"]),
                       "aborted": str(aborted) if aborted else None,
                       "t_fly": float(res["seconds"])}
                rec["clean"] = bool(res["in_bin"] and not res["lost"]
                                    and not res["disturbed"] and not aborted)
                rec["outcome"] = classify(rec)

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
                    print(f"    {me.name} {name} ({fruit.stage}): "
                          f"{rec['outcome']:<14} crate={rec['in_bin']} "
                          f"{res['seconds']:.1f}s")

            me.target = None
            state.active = None
            stow(tag)
            mujoco.mj_forward(model, data)

    for t in arms:
        unstow(t)
    mujoco.mj_forward(model, data)
    state.say("done", "row complete")
    for t in arms:
        state.state[t].say("done", "")
    state.drive_m = drive.travelled
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
    print(f"\n  ⚠️ the arms are SERIALISED — one flies while the other stows.")
    print(f"     See farm/duo.py: `armframe.at_trolley` rebinds mission's "
          f"globals per arm,")
    print(f"     so two arms mid-mission at once is not currently expressible.")


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
