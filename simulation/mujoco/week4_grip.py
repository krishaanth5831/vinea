#!/usr/bin/env python3
"""Why the tomato fell out of the gripper, and the number that stopped it.

The symptom was the one you would report from watching: the arm reaches in, the
fingers close, the stem parts, and then the tomato simply falls out and lands on
the floor. Twenty-six of fifty-seven attempts in the Week 4 campaign came back
clean and the rest were mostly filed as `grasp_failed`, which is a bucket that
says "the tool failed once the arm arrived" and stops there.

⚠️ **The bucket was hiding two different failures and the log already knew.**
Fourteen of the twenty `grasp_failed` rows have `broke: true` and a resting
position **3 to 39 metres from the row** — those are not failed grasps, they are
fruit leaving the gripper at speed. Thirty-four of the fifty-seven attempts
exceeded 1 m/s while held. `outcomes.instability` was already reporting that and
it was already being read past.

**What is actually happening.** Traced substep by substep (`--trace`), the fruit
is *not* flung at the moment the peduncle parts — at that instant it is still
centred, 2.3 mm off the tool site, with the pads loaded at ~99 N. It is lost a
second and a half later, during `extract`:

    t-98 ms   pads 86 N   fruit 24 mm off centre     <- already sliding
    t-28 ms   pads 76 N   fruit 30 mm off centre
    t-18 ms   pads  0 N   fruit 33 mm off centre     <- contact gone
    t+52 ms   pads  0 N   fruit 80 mm off centre, falling at 9.8 m/s²

The fruit **creeps out of a closed gripper under its own weight**. A soft
MuJoCo contact does not hold a tangential load indefinitely; it drifts. Once the
centre has walked ~30 mm the sphere is off the edge of a 37 mm pad and there is
nothing left to hold.

**What it is not.** Two plausible causes were tested and neither is it, which is
why neither was changed:

    pad friction / condim   rolling and torsional friction added on top of
                            sliding (condim 4 and 6, roll out to 0.05):
                            1/3 crated at every setting. The sphere is not
                            rolling out, it is sliding out.
    the close command       100 through 255 on the Robotiq scale: identical
                            results and identical peak pad force. The fingers
                            are not failing to squeeze — they squeeze at 99 N.

**The cause** is `greenhouse.PAD_SOLREF`, and it is a two-sided trade with both
sides measured here. Run the sweeps:

    ./.venv/bin/python simulation/mujoco/week4_grip.py            # the gate
    ./.venv/bin/python simulation/mujoco/week4_grip.py --solref   # the trade
    ./.venv/bin/python simulation/mujoco/week4_grip.py --close    # ramp vs step
    ./.venv/bin/python simulation/mujoco/week4_grip.py --trace    # the creep
    ./.venv/bin/python simulation/mujoco/week4_grip.py --windowed # watch one

⚠️ **The old value's justification had expired.** The pads were softened to 0.02
to keep the *closing* transient off the peduncle, when the grip was a step
input. `--close` shows that softening never actually fixed that — 0.02 still
snapped the stem, at 16.97 N against a SNAP_N of 12.0 — and that
`reach.Gripper.ramp`, which arrived later, fixes it at every stiffness tested.
So the softening had been buying nothing and costing three picks in eight. That
is the interesting part of this file: not that a number was wrong, but that it
was right when it was chosen and nobody re-took the measurement when the thing
it was compensating for was fixed properly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greenhouse import PAD_SOLREF, build_scene  # noqa: E402
from plant_row import SNAP_N, Row  # noqa: E402

# The layouts the sweeps run over. Eight single-fruit picks, each at a different
# spot in the placement band, because a grasp that holds at one height and one
# reach is not evidence about the workspace. Single fruit on purpose: this file
# is about the tool, and a neighbour would fold refusals into the count.
SEEDS = (7, 11, 23, 31, 47, 53, 61, 71)

# Above this the fruit has left the pads. A 2F85 pad is 37 mm long and the fruit
# is a 33 mm sphere, so a centre that has walked past ~40 mm is over the edge
# and the millimetre figures beyond it are only recording where it rolled to.
ESCAPED_MM = 40.0


def _scene(solref=None):
    """A one-fruit scene with the pads set to `solref` (None = what ships)."""
    from week4_place import pool_trusses

    pool = pool_trusses()
    model = build_scene(wrist_cam=True, trusses=pool)
    if solref is not None:
        for i in range(model.ngeom):
            if "pad" in (model.geom(i).name or ""):
                model.geom_solref[i] = list(solref)
    return model, [n for n, _, _ in pool]


def _staged(model, names, seed):
    """Fresh data, arm parked, exactly one fruit placed. Returns its name."""
    import mujoco

    from mission import park_arm, park_posture, reset_park
    from week4_place import Crop, auto_layout, park_spot

    data = mujoco.MjData(model)
    q = park_posture(model)
    reset_park(model, data, q)
    row = Row(model, data, names=names,
              homes={n: park_spot(i) for i, n in enumerate(names)})
    row.reset()
    mujoco.mj_forward(model, data)

    crop = Crop(model, data, row, names)
    crop.park_all()
    for _n, y, z in auto_layout(1, seed=seed):
        crop.place(y, z, quiet=True)
    target = list(crop.placed)[0]

    park_arm(model, data, q)
    mujoco.mj_forward(model, data)
    return data, q, row, target


def one_pick(solref=None, seed=7, speed=0.4, on_tick=None, trace=False,
             model_data=None):
    """Fly one full pick. Returns what happened to the fruit, and why.

    The measurement that matters is `creep`: how far the fruit's centre gets
    from the tool site while it is supposed to be held. `in_bin` is the outcome,
    but creep is the mechanism, and a config can crate the fruit by luck — the
    `--solref` sweep has three settings that crate 8/8, and only one of them is
    still holding the fruit when it does.

    `model_data` lets a caller that already owns a scene — `windowed`, which has
    a viewer attached to one — pass it in rather than have a second compiled
    here that nothing is looking at.
    """
    import mujoco

    from carrytrace import HELD as CARRY_HELD
    from carrytrace import CarryTrace
    from fr5 import TOOL_SITE
    from mission import Planner
    from reach import Gripper
    from week2_pick import anchor_posture, execute, make_reacher

    if model_data is None:
        model, names = _scene(solref)
        data, q, row, target = _staged(model, names, seed)
    else:
        from mission import park_posture
        from week4_place import park_spot

        model, data, names, target = model_data
        q = park_posture(model)
        row = Row(model, data, names=names,
                  homes={n: park_spot(i) for i, n in enumerate(names)})

    reacher = make_reacher(model, data, speed=speed)
    anchor_posture(reacher, model, data, q)
    gripper = Gripper(model, data)
    planner = Planner(model, data, row, lessons=None, clearance=0.040,
                      park_q=q, speed=speed)
    mission = planner.plan(target)
    if not mission.ok:
        return None

    pads = [i for i in range(model.ngeom) if "pad" in (model.geom(i).name or "")]
    fruit_geoms = {model.body(target).geomadr[0] + k
                   for k in range(model.body(target).geomnum[0])}
    stat = {"grip_n": 0.0, "released": None}
    steps = []
    ft = np.zeros(6)
    clock = [0.0]

    def pad_n():
        tot = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            if ((con.geom1 in pads and con.geom2 in fruit_geoms)
                    or (con.geom2 in pads and con.geom1 in fruit_geoms)):
                mujoco.mj_contactForce(model, data, c, ft)
                tot += abs(ft[0])
        return tot

    class Inner:
        """The `Blackbox` slot. This file is about the tool, not about incidents,
        and a single fruit cannot have collateral — so the incident recorder
        would have nothing to record. The peduncle watch goes here."""

        leg = ""

        def substep(self):
            clock[0] += model.opt.timestep
            was = row.attached(target)
            broke = row.update()
            if was and not row.attached(target) and stat["released"] is None:
                stat["released"] = clock[0]
            # What the *close* puts into the peduncle: the number the softening
            # was originally chosen to hold down.
            if self.leg in ("insert", "grip", "close") and row.attached(target):
                stat["grip_n"] = max(stat["grip_n"], row.force(target))
            if trace:
                steps.append({"t": clock[0], "leg": self.leg,
                              "pad_n": pad_n(), "attached": row.attached(target),
                              "gap_mm": float(np.linalg.norm(
                                  data.body(target).xpos
                                  - data.site(TOOL_SITE).xpos)) * 1000})
            return broke

        def rebase(self, *_a):
            pass

        def sweep(self):
            return []

    # ⚠️ **The speed and the drift are read off `CarryTrace`, not measured here
    # again.** The campaign's `peak_held_ms` column is `CarryTrace.peak_held`,
    # and a second implementation would produce a number that looked like it and
    # was not comparable to it. A first pass at this file did exactly that —
    # `body.cvel[3:]` instead of the free joint's `qvel` — and reported 2.09 m/s
    # on a pick that was holding the fruit perfectly well, because `cvel` is
    # com-based and picks up a spin term the linear velocity does not have.
    trace_box = CarryTrace(model, data, row, target, Inner())
    r = execute(mission, reacher, gripper, row, box=trace_box, guard=None,
                on_tick=(trace_box.tick if on_tick is None
                         else lambda t=None: (trace_box.tick(t), on_tick(t))))

    # Drift is the fruit's distance from the pinch centre *in the tool's own
    # frame* — `CarryTrace.in_tool`. Measured in world coordinates instead it
    # would be dominated by the arm carrying the fruit across the room.
    held = [s for s in trace_box.samples if s.leg in CARRY_HELD]
    creep = max((float(np.linalg.norm(s.in_tool)) * 1000 for s in held),
                default=0.0)
    return {"fruit": target, "in_bin": bool(r["in_bin"]),
            "grasped": bool(r["grasped"]), "broke": bool(r["broke"]),
            "pull_n": float(r["peak_n"]), "seconds": float(r["seconds"]),
            "creep": creep, "vmax": trace_box.peak_held(),
            "trace": trace_box, "steps": steps, **stat}


def close_on_fruit(solref=None, ramp_s=1.5, seed=7):
    """Reach a hanging fruit and shut the fingers. Returns the peak the stem saw.

    The original experiment, isolated: no pull, no carry, just the close. This
    is the measurement `PAD_SOLREF` was first chosen by.

    ⚠️ Read `row.peak`, never a maximum taken after `row.update`. `update` cuts
    the weld on the same call that sees the over-threshold sample, so a peak
    read afterwards misses exactly the spike that mattered and reports the
    hanging weight instead — a snapped stem comes back as a placid 1.18 N.
    """
    import mujoco

    from fr5 import TOOL_SITE
    from reach import Gripper, hold
    from week2_pick import anchor_posture, make_reacher

    model, names = _scene(solref)
    data, q, row, target = _staged(model, names, seed)

    reacher = make_reacher(model, data, speed=0.4)
    anchor_posture(reacher, model, data, q)
    reacher.on_substep = row.update
    gripper = Gripper(model, data)
    gripper.open()

    goal = data.body(target).xpos.copy()
    # Let the weld's startup overshoot ring down first — the same reason
    # `mission._legs` opens with a settle leg. Without it this measures the
    # spawn transient rather than the grip.
    hold(reacher, data.site(TOOL_SITE).xpos.copy(), 0.3)
    for gap in (0.28, 0.09, 0.0):
        reacher.drive_to(np.array([goal[0] - gap, goal[1], goal[2]]))
    hanging = row.force(target)
    row.peak[target] = 0.0          # the books open at the close, not before

    if ramp_s <= 0:
        gripper.close()             # the Week 2 way: a step input
        hold(reacher, goal, 1.0)
    else:
        gripper.ramp(reacher, goal, ramp_s)
        hold(reacher, goal, 0.8)
    return {"hanging_n": float(hanging), "peak_n": float(row.peak[target]),
            "survived": bool(row.attached(target))}


# --- the sub-commands --------------------------------------------------------

def sweep_solref(args):
    """The trade: soft enough to keep the stem on, stiff enough to keep the fruit."""
    print(f"\n  --- pad solref against the grasp, {len(SEEDS)} picks each ---")
    print(f"  SNAP_N is {SNAP_N:.1f} N. A creep over {ESCAPED_MM:.0f} mm is the "
          f"fruit off the edge of the pad;")
    print("  the metre-scale figures are only where it came to rest afterwards.")
    print(f"\n  {'pad solref':<16} {'crated':>8} {'grasped':>8} {'grip N':>8} "
          f"{'pull N':>8} {'creep mm':>10}  note")
    rows = []
    for solref in ([0.002, 1.0], [0.004, 1.0], [0.006, 1.0], [0.008, 1.0],
                   [0.012, 1.0], [0.020, 1.0]):
        got = [one_pick(solref, s, speed=args.speed) for s in SEEDS]
        got = [g for g in got if g]
        if not got:
            print(f"  {str(solref):<16}   no route planned")
            continue
        creep = max(g["creep"] for g in got)
        note = "holds" if creep < ESCAPED_MM else "FRUIT LEAVES THE PADS"
        if tuple(solref) == tuple(PAD_SOLREF):
            note += "   <-- ships"
        print(f"  {str(solref):<16} {sum(g['in_bin'] for g in got):>4}/"
              f"{len(got):<3} {sum(g['grasped'] for g in got):>4}/{len(got):<3} "
              f"{max(g['grip_n'] for g in got):>8.1f} "
              f"{np.mean([g['pull_n'] for g in got]):>8.1f} {creep:>10.0f}  {note}")
        rows.append((solref, got))
    print("\n  Soft holds the peduncle together at the close and lets the fruit")
    print("  slide out during the carry. Stiff does the opposite. The shipped")
    print(f"  value is the stiffest one whose grip transient still clears SNAP_N.")
    return rows


def sweep_close(args):
    """Is the softening still doing the job it was chosen for? No."""
    print(f"\n  --- closing on a hanging fruit: what the stem sees ---")
    print(f"  SNAP_N is {SNAP_N:.1f} N. Over it, the stem parts on contact —")
    print("  which looks like a successful harvest on video and is not one.")
    print(f"\n  {'pad solref':<16} {'close':>12} {'hanging N':>10} {'peak N':>9} "
          f"{'stem':>10}")
    for solref in ([0.004, 1.0], list(PAD_SOLREF), [0.020, 1.0]):
        for label, ramp_s in (("step", 0.0), ("ramped 1.5 s", 1.5)):
            r = close_on_fruit(solref, ramp_s)
            tag = "survives" if r["survived"] else "SNAPS"
            mark = "   <-- ships" if (tuple(solref) == tuple(PAD_SOLREF)
                                      and ramp_s > 0) else ""
            print(f"  {str(solref):<16} {label:>12} {r['hanging_n']:>10.2f} "
                  f"{r['peak_n']:>9.2f} {tag:>10}{mark}")
    print("\n  Read the step column: softening the pads never fixed this —")
    print("  every stiffness snaps the stem on a step input, including the 0.02")
    print("  that was chosen to prevent exactly that. `reach.Gripper.ramp` is")
    print("  what fixed it, and it fixes it at every stiffness — which is what")
    print("  freed the pads to be stiff enough to hold on to the fruit.")


def trace(args):
    """The substep trace that found it. One pick, the fruit's grip on the tool."""
    print(f"\n  --- one pick at pad solref {list(PAD_SOLREF)}, substep by substep ---")
    r = one_pick(None, args.seed, speed=args.speed, trace=True)
    if r is None:
        print("  no route planned for that seed")
        return
    print(f"  fruit {r['fruit']} · crated {r['in_bin']} · stem broke {r['broke']}"
          f" · peak creep {r['creep']:.0f} mm")
    if r["released"]:
        print(f"  the peduncle parted at t = {r['released']:.3f} s")

    speeds = {}
    for s in r["trace"].samples:
        speeds[s.leg] = max(speeds.get(s.leg, 0.0), s.speed)

    print(f"\n  {'leg':<9} {'ms':>7} {'gap mm':>9} {'max m/s':>9} "
          f"{'pad N min':>10} {'pad N max':>10}")
    legs = []
    for e in r["steps"]:
        if not legs or legs[-1][0] != e["leg"]:
            legs.append((e["leg"], []))
        legs[-1][1].append(e)
    for leg, es in legs:
        print(f"  {leg:<9} {(es[-1]['t'] - es[0]['t']) * 1000:>7.0f} "
              f"{es[-1]['gap_mm']:>9.1f} {speeds.get(leg, 0.0):>9.3f} "
              f"{min(x['pad_n'] for x in es):>10.1f} "
              f"{max(x['pad_n'] for x in es):>10.1f}")
    print("\n  `gap mm` is the fruit's centre from the tool site — 0 is dead")
    print("  centre between the pads. Watch it during `extract`: that column is")
    print("  the whole bug, and every other column looks fine while it happens.")


def gate(args):
    """Does the shipped value hold the fruit and spare the stem? Both, or fail."""
    from carrytrace import ESCAPE_MS

    print(f"\n  --- the gate: pad solref {list(PAD_SOLREF)} ---")
    got = [one_pick(None, s, speed=args.speed) for s in SEEDS]
    got = [g for g in got if g]
    crated = sum(g["in_bin"] for g in got)
    creep = max(g["creep"] for g in got)
    grip = max(g["grip_n"] for g in got)
    thrown = sum(1 for g in got if g["vmax"] > ESCAPE_MS)

    print(f"\n  {'seed':<6} {'fruit':<6} {'crated':>7} {'grip N':>8} "
          f"{'pull N':>8} {'creep mm':>10} {'held m/s':>9}")
    for s, g in zip(SEEDS, got):
        print(f"  {s:<6} {g['fruit']:<6} {str(g['in_bin']):>7} "
              f"{g['grip_n']:>8.2f} {g['pull_n']:>8.2f} {g['creep']:>10.1f} "
              f"{g['vmax']:>9.2f}")

    # ⚠️ **Two verdicts, deliberately not one.** This file owns whether the pads
    # hold the fruit; it does not own what the wrist does with it afterwards.
    # Folding both into a single PASS/FAIL would either hide a real residual or
    # make the gate fail forever on something changing `PAD_SOLREF` cannot fix —
    # and a gate that always fails is a gate everybody learns to ignore.
    ok = crated == len(got) and creep < ESCAPED_MM and grip < SNAP_N
    print(f"\n  --- the grasp, which is what PAD_SOLREF decides ---")
    print(f"  crated              {crated}/{len(got)}")
    print(f"  worst creep         {creep:.1f} mm   (escapes past "
          f"{ESCAPED_MM:.0f} mm)")
    print(f"  worst grip force    {grip:.2f} N    (SNAP_N {SNAP_N:.1f} N)")
    print(f"  {'PASS' if ok else 'FAIL'}")

    print(f"\n  --- the carry, which it does not ---")
    print(f"  thrown at >{ESCAPE_MS:.0f} m/s     {thrown}/{len(got)}   "
          f"(carrytrace.ESCAPE_MS)")
    if thrown:
        worst = max(got, key=lambda g: g["vmax"])
        print(f"  worst is seed {SEEDS[got.index(worst)]} at "
              f"{worst['vmax']:.2f} m/s, and it still crates.")
        print(f"\n  ⚠️ RESIDUAL, and it is **not** this file's bug — it is older")
        print(f"  and it is in the `turn` leg. `--trace --seed 71` shows the pad")
        print(f"  force spiking to 761 N and collapsing to 15 N while the wrist")
        print(f"  rotates: the fruit is being rattled, not carried. Three things")
        print(f"  say it is separate from pad compliance:")
        print(f"    · the same seed at the old solref 0.02 was *worse* — 1.21 m/s")
        print(f"      and the fruit never reached the crate at all (943 mm creep)")
        print(f"    · it does not scale with cycle speed: 2.07 m/s at 0.40,")
        print(f"      2.06 at 0.25, and at 0.15 the pick stops crating entirely")
        print(f"    · it starts at `turn`, long after the grasp is established")
        print(f"  `mission.TURN_DONE_DEG` carries a note about a 2F85 rolling a")
        print(f"  sphere out of its pads past ~22°, measured on the soft pads.")
        print(f"  That measurement is now due to be re-taken, which is a Week 5")
        print(f"  job with its own sweep and not a constant to nudge here.")
    return ok


def windowed(args):
    """Watch one pick at wall-clock speed. The fruit should stay in the pads."""
    import time
    import warnings

    import mujoco.viewer

    from reach import CTRL_DT

    warnings.filterwarnings("ignore", module="glfw")
    # ⚠️ `one_pick` compiles its own scene, so the viewer has to be opened on
    # *that* model — not on a second one built here, which would be a window
    # onto a world nothing is stepping. The pick is therefore built first and
    # the viewer attached to it, rather than the other way round.
    model, names = _scene(None)
    data, _q, _row, target = _staged(model, names, args.seed)
    print(f"\n  one pick at pad solref {list(PAD_SOLREF)}, fruit {target}")
    print(f"  watch the gap between the fruit and the pads during the carry")
    print(f"  close the window to quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        clock = [time.perf_counter()]

        def tick(_t=None):
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            clock[0] += CTRL_DT
            # Physics runs far faster than real time; without the sleep this is
            # an unwatchable blur. Same pacing as `week4_place.watch`.
            time.sleep(max(0.0, clock[0] - time.perf_counter()))
            clock[0] = max(clock[0], time.perf_counter() - CTRL_DT)

        try:
            r = one_pick(None, args.seed, speed=args.speed, on_tick=tick,
                         model_data=(model, data, names, target))
            if r:
                print(f"\n  crated {r['in_bin']} · creep {r['creep']:.1f} mm "
                      f"· peak held {r['vmax']:.2f} m/s")
            print("  finished — the window stays open, close it to quit")
            # ⚠️ Physics is deliberately not stepped while the window is held.
            # The run is over; stepping on with nothing commanding the servos
            # would let the arm sag while you watch.
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\n  stopped early")


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--solref", action="store_true",
                   help="sweep pad compliance against the grasp")
    p.add_argument("--close", action="store_true",
                   help="closing transient, ramped against a step input")
    p.add_argument("--trace", action="store_true",
                   help="one pick, substep by substep — where the fruit goes")
    p.add_argument("--windowed", action="store_true",
                   help="watch one pick in a live viewer")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=0.4)
    args = p.parse_args()

    print(__doc__)
    if args.solref:
        sweep_solref(args)
    elif args.close:
        sweep_close(args)
    elif args.trace:
        trace(args)
    elif args.windowed:
        windowed(args)
    else:
        raise SystemExit(0 if gate(args) else 1)


if __name__ == "__main__":
    main()
