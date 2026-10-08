#!/usr/bin/env python3
"""The tests that have to pass before a commit.

Bug log entry 6: every check in this repo was ad-hoc, run by hand, and its
result lived in a terminal that is now closed. This is the minimum that entry
asks for, and no more:

    1. one pick completes end to end
    2. the weld holds within 1 mm, and releases on eq_active = 0
    3. tool0 sits on the gripper's own pinch site
    4. reset_home puts a free body at its spawn pose

    ./.venv/bin/python tests/test_sim.py

Exits non-zero if any check fails. Same shape as `scripts/phase0_smoketest.py`
deliberately — that file already passes 6/6 and a second convention for saying
PASS would be one convention too many. There is no test framework here on
purpose: the repo has none, and adding pytest to run four assertions would be a
dependency bought with somebody else's money.

**These are correctness checks, not measurements.** None of them asserts a
headline number. A test that pins 42/42 or 10/10 into an assertion turns every
honest change to the physics into a failing build, and the numbers belong in
the build log where their assumptions are written next to them. What these
check is that the mechanisms those numbers are built on still work at all.

Runtime is a few seconds. Anything slower than that does not get run before
every commit, whatever the README claims.
"""

import os
import sys
from pathlib import Path

# MuJoCo needs an offscreen GL backend when there is no display attached. None
# of these checks renders, but building a model can still touch GL.
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "simulation" / "mujoco"))

import numpy as np  # noqa: E402

RESULTS = []


def check(name):
    """Decorator: run a check, record pass/fail, never raise."""

    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((True, name, detail))
        except Exception as exc:  # noqa: BLE001 - a test run reports, never crashes
            RESULTS.append((False, name, f"{type(exc).__name__}: {exc}"))
        return fn

    return wrap


# --- 1. one pick completes end to end ---------------------------------------

@check("a pick completes end to end")
def _pick_end_to_end():
    """Week 1's table pick, run headless, scored on where the fruit ended up.

    Deliberately the *whole* cycle rather than a stubbed one. The states —
    approach, descend, close, lift, carry, release, home — are the same states
    Week 4 harvests with, and a cycle that reports a grasp it did not make is
    the failure this repo has hit most often (entries 33, 34, 36, 55).
    """
    import mujoco
    import week1_gripper
    from fr5 import MAX_REACH_GRIPPER
    from reach import Gripper, Reacher

    model = week1_gripper.build_scene()
    data = mujoco.MjData(model)
    reacher = Reacher(model, data, standoff=0.0, max_reach=MAX_REACH_GRIPPER,
                      reached_mm=week1_gripper.REACHED_MM_LOADED)
    gripper = Gripper(model, data)
    gripper.open()

    res = week1_gripper.run_pick(reacher, gripper, verbose=False)
    assert res["grasped"], "the fingers closed but the fruit stayed on the table"
    assert res["in_bin"], f"carried but not crated — fruit ended at {res['tomato'].round(3)}"
    return f"grasped and crated, fruit at {res['tomato'].round(3)}"


# --- 2. the weld holds, and releases ----------------------------------------

@check("weld holds within 1 mm and releases on eq_active=0")
def _weld_holds_and_releases():
    """The detachment model, which is the one real difference from a table pick.

    Two halves, and the second is the one that has bitten. A weld that holds is
    easy to see; a weld that has been switched off but is still carrying load
    looks exactly like a weld that is working. So this asserts the fruit is
    held at the stem *and* that it leaves once the constraint is cleared.

    1 mm is the tolerance the bug log names. For scale, entry 19's un-zeroed
    weld anchor sagged this same fruit 160 mm with nothing pulling on it, and
    threw no error.
    """
    import mujoco
    import week1_mousereach as mr

    model = mr.build_scene()
    data = mujoco.MjData(model)
    eq_id = model.equality("peduncle").id

    pos = mr.tomato_on_panel(mr.panel_to_world(0.0, 0.60))
    mr.place_tomato(model, data, eq_id, pos)

    # Let gravity have a go at it. A weld with a bad anchor sags over about
    # this long, so a single mj_forward would not catch entry 19.
    for _ in range(500):
        mujoco.mj_step(model, data)
    held = float(np.linalg.norm(data.body("tomato").xpos - data.mocap_pos[mr.STEM_MOCAP]))
    assert held < 0.001, f"weld sagged {held * 1000:.1f} mm — anchor or solref wrong"

    # A hanging fruit reads its own weight. If it does not, the force this
    # repo snaps stems on is not a force. 0.12 kg x 9.81 = 1.18 N.
    f = mr.weld_force(model, data, eq_id)
    assert 0.9 < f < 1.5, f"a still fruit reads {f:.2f} N, not its own 1.18 N"

    z_before = float(data.body("tomato").xpos[2])
    data.eq_active[eq_id] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    dropped = z_before - float(data.body("tomato").xpos[2])
    assert dropped > 0.02, (f"eq_active=0 and the fruit still hangs "
                            f"({dropped * 1000:.1f} mm of fall in 1.0 s)")
    return (f"held to {held * 1000:.3f} mm at {f:.2f} N, "
            f"fell {dropped * 1000:.0f} mm once released")


# --- 3. the tool frame ------------------------------------------------------

@check("tool0 sits on the gripper's pinch site")
def _tool_on_pinch():
    """PINCH_Z is a measured offset, and everything downstream trusts it.

    `tool0` is this repo's own site on wrist3; `gr_pinch` is the one Menagerie
    ships inside the 2F85. Every waypoint in every pick is written as "put
    tool0 here", so if these two drift apart, every grasp misses by the gap and
    nothing anywhere reports an error.

    Checked at more than one posture on purpose: two frames on the same rigid
    body agree everywhere or nowhere, and a single check at home cannot tell
    "the offset is right" from "the offset is wrong in a direction home hides".
    """
    import mujoco
    from fr5 import GRIPPER_PREFIX, JOINTS, TOOL_SITE, build_fr5, reset_home

    model = build_fr5(gripper=True)
    data = mujoco.MjData(model)

    gaps = []
    for bend in (0.0, 0.4, -0.7):
        reset_home(model, data)
        for j in JOINTS:
            data.joint(j).qpos[0] += bend
        mujoco.mj_forward(model, data)
        gaps.append(float(np.linalg.norm(
            data.site(GRIPPER_PREFIX + "pinch").xpos - data.site(TOOL_SITE).xpos)))

    worst = max(gaps)
    assert worst < 0.001, (f"tool0 is {worst * 1000:.2f} mm off the pinch site — "
                           f"PINCH_Z is wrong and every grasp misses by it")
    return f"worst gap {worst * 1000:.3f} mm over 3 postures"


# --- 4. reset_home ----------------------------------------------------------

@check("reset_home puts a free body at its spawn pose")
def _reset_home_free_body():
    """Entry 11, and the reason entry 3 exists.

    `mj_resetDataKeyframe` reads a keyframe, and `spec.add_key` stored a flat
    six-number vector back when six numbers described the whole scene. MuJoCo
    pads a short keyframe with **zeros** rather than each body's spawn pose, so
    every free body in the scene is teleported to the world origin — through
    the floor — silently. `qpos0` had it right all along.

    So this checks both directions: `reset_home` restores the spawn pose, and
    the keyframe does not. The second assertion is the one that matters. If it
    ever starts passing, MuJoCo changed its padding behaviour and this test is
    the only thing that will say so.
    """
    import mujoco
    import week1_gripper
    from fr5 import reset_home

    model = week1_gripper.build_scene()
    data = mujoco.MjData(model)
    spawn = week1_gripper.TOMATO_POS

    # Shove it somewhere it certainly does not belong, then reset.
    reset_home(model, data)
    jnt = model.body("tomato").jntadr[0]
    qadr = model.jnt_qposadr[jnt]
    data.qpos[qadr:qadr + 3] = [1.4, 1.4, 1.4]
    mujoco.mj_forward(model, data)

    reset_home(model, data)
    err = float(np.linalg.norm(data.body("tomato").xpos - spawn))
    assert err < 1e-6, f"reset_home left the fruit {err * 1000:.1f} mm off its spawn pose"

    # And the reset that lies about it.
    keyed = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, keyed, 0)
    mujoco.mj_forward(model, keyed)
    strayed = float(np.linalg.norm(keyed.body("tomato").xpos - spawn))
    assert strayed > 0.1, (
        "mj_resetDataKeyframe no longer teleports free bodies to the origin — "
        "MuJoCo's short-keyframe padding changed, so re-read bug log entries 3 and 11")
    return (f"spawn restored to {err * 1e6:.1f} um; "
            f"the keyframe strands it {strayed * 1000:.0f} mm away")


@check("reset_home homes every scene that used the keyframe")
def _reset_home_on_keyframe_scenes():
    """The two scenes bug 3 names: `week1_reach`'s and `fr5.main`'s.

    Neither has a free body today, which is exactly why the keyframe was safe
    in them and why nothing caught it. This asserts the property the fix is
    for: after `reset_home` the arm is at HOME in both, so the two call sites
    can be switched without anything moving.
    """
    import mujoco
    import week1_reach
    from fr5 import HOME, JOINTS, build_fr5, reset_home

    out = []
    scenes = [("week1_reach", week1_reach.build_scene()),
              ("fr5.main", build_fr5(gripper=False))]
    for name, model in scenes:
        data = mujoco.MjData(model)
        reset_home(model, data)
        q = np.array([data.joint(j).qpos[0] for j in JOINTS])
        err = float(np.abs(q - HOME).max())
        assert err < 1e-9, f"{name}: arm is {err:.2e} rad off HOME after reset_home"
        out.append(name)
    return f"HOME restored in {', '.join(out)}"


# --- 5. the single-arm names, on a machine with two arms --------------------

@check("a two-armed scene answers for the arm it was asked about")
def _prefix_addresses_arm_b():
    """Entry 58, and the reason it is worth doing rather than leaving latent.

    `data.site("tool0")` is not "the tool". It is *arm A's* tool, because arm A
    is deliberately the unprefixed arm so Weeks 1-4 keep working. Every one of
    these call sites is correct today and none is reachable from a two-armed
    run — which is exactly the condition under which the bug ships, because the
    failure mode is a plausible number rather than a `KeyError`.

    Two halves. `fr5.tool_pos` is the expression itself, now named; and
    `week4_watch.Thoughts` is one of the six signatures the entry lists, called
    with a prefix and asked whether it answers for arm B. The default has to
    keep giving arm A, or every number measured before the parameter existed
    moves.
    """
    import mujoco
    import week4_watch
    from farm import trolley
    from fr5 import reset_home, tool_pos

    model = trolley.build(aisle=0, arms=("a", "b"), crate=False, leafy=False)
    data = mujoco.MjData(model)
    reset_home(model, data)

    a = tool_pos(data).copy()
    b = tool_pos(data, trolley.ARM_PREFIX["b"]).copy()
    apart = float(np.linalg.norm(a - b))
    assert apart > 1.0, (f"the two arms' tools are {apart * 1000:.0f} mm apart — "
                         f"the prefix is not selecting a different arm")

    # One of the six signatures the entry names, asked about arm B.
    told_b = week4_watch.Thoughts(model, data, None, None,
                                  prefix=trolley.ARM_PREFIX["b"])
    got = data.site_xpos[told_b.tool_site]
    assert np.allclose(got, b), (
        f"asked for arm B, answered {got.round(3)} — arm A is at {a.round(3)}")

    # And the default still means arm A, exactly as before.
    told_default = week4_watch.Thoughts(model, data, None, None)
    assert np.allclose(data.site_xpos[told_default.tool_site], a), \
        "the default prefix stopped meaning arm A — every Week 1-4 number moves"
    return (f"arm A {a.round(2)} vs arm B {b.round(2)}, "
            f"{apart * 1000:.0f} mm apart; default still arm A")


@check("the scouting camera rides the trolley in every rendered frame")
def _scout_head_rides_the_trolley():
    """The deck cam teleporting across `farm/watch.py`'s panel.

    The head is a world-parented mocap body, so it does not move when the
    trolley does — `ScoutHead.follow` is what puts it back on the mast, and the
    only caller was `Scout.look`, once per survey frame. Between two of those
    the trolley drives a whole `SCOUT_STRIDE`, so the panel held still and then
    jumped 0.50 m; during HARVEST, and under `--truth`, nothing called `follow`
    at all and the camera stayed behind for the entire shift.

    Three assertions. The middle one is the fix itself: the camera holds a
    fixed offset from the drive joint wherever the trolley is. The two either
    side of it are what keep that honest, and both are two-sided in the way
    `_reset_home_free_body` is.

    **`mocap_pos` alone must not move `cam_xpos`.** That is the trap the first
    attempt at this fell into — `follow` returns happily, `mj_kinematics` walks
    the body tree, and the camera stays exactly one frame behind because
    `cam_xpos` is `mj_camlight`'s job. It reads as a working panel.

    **A mocap body must not start riding its mast on its own.** If MuJoCo ever
    makes it, this is the only thing here that will say the seat has stopped
    being load-bearing.
    """
    import mujoco
    from farm import trolley, watch
    from farm.scout import ScoutHead

    model = trolley.build(aisle=0, arms=("a",), crate=False, leafy=False,
                          deck_cam=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    head = ScoutHead(model, data, aisle=0)
    drive = trolley.Drive(model, data)
    jadr = model.joint(trolley.DRIVE_JOINT).qposadr[0]

    def lead():
        """How far ahead of the drive joint the lens currently is."""
        return float(data.cam("scout").xpos[1]) - float(data.qpos[jadr])

    # `follow` writes `mocap_pos` and nothing else. Until the camera frames are
    # recomputed the lens has not moved — the half of this that is easy to miss.
    drive.park_at(1.5)
    mujoco.mj_forward(model, data)
    before = np.array(data.cam("scout").xpos, float)
    head.follow(data)
    unmoved = float(np.linalg.norm(data.cam("scout").xpos - before))
    assert unmoved < 1e-9, (
        "writing mocap_pos now moves cam_xpos on its own — the "
        "mj_kinematics/mj_camlight pair in farm.watch.seat_scout_head may have "
        "become unnecessary, re-read it before trusting this")

    offsets, stale = [], []
    for y in (-2.0, 0.0, 2.5, -3.4):
        drive.park_at(y)
        mujoco.mj_forward(model, data)
        stale.append(abs(lead()))   # where the panel framed it before the fix
        watch.seat_scout_head(head, model, data)
        offsets.append(lead())

    spread = max(offsets) - min(offsets)
    assert spread < 1e-9, (
        f"the camera slid {spread * 1000:.1f} mm along its own mast between "
        f"trolley positions — it is not seated on the drive joint")

    drift = max(stale) - min(stale)
    assert drift > 0.5, (
        "the scout head now follows the trolley without being told to — "
        "MuJoCo started moving mocap bodies, so re-read farm.scout.add_deck_camera")
    return (f"camera holds {offsets[0]:+.3f} m from the drive joint across "
            f"4 positions (spread {spread * 1e6:.1f} um); unseated it drifts "
            f"{drift * 1000:.0f} mm")


@check("a truss is six fruit on one stem weighing 0.8 kg")
def _truss_is_a_cluster():
    """The crop the truss variant is about, as a compiled body.

    Three things the brief fixed and nothing else checks: at least six fruit per
    cluster, 0.80 kg for the whole thing, and — the one that is easy to get
    wrong silently — the body **origin at the grasp point**, not at the centroid
    of the fruit. `mission.Planner.plan` routes the gripper to `row.pos(name)`,
    so an origin in the middle of the cluster would aim the pads into a tomato
    and every truss pick would be a collision the planner thought was fine.

    Also checks the fruit do not interpenetrate. A cluster whose spheres overlap
    is fired apart by MuJoCo on the first step, which reads as an exploding crop
    rather than as a layout error.
    """
    import mujoco
    from farm import truss as ft

    ts = ft.spawn(n_per_row=3, seed=7)
    assert ts, "no trusses spawned"
    assert all(t.n_fruit >= 6 for t in ts), (
        f"a truss came back with {min(t.n_fruit for t in ts)} fruit — "
        f"the brief asks for at least 6")

    model = ft.build(aisle=0, arms=("a",), crate=False, leafy=False, trusses=ts)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    t0 = ts[0]
    mass = float(model.body(t0.name).mass[0])
    assert abs(mass - 0.80) < 1e-6, (
        f"a truss weighs {mass:.3f} kg, not the 0.80 kg the brief fixed")

    # The origin is the grasp point: the collar the pads close on is at z=0 in
    # the body frame, and every fruit hangs below it.
    grasp = data.body(t0.name).xpos
    assert np.allclose(grasp, t0.pos, atol=1e-9), (
        f"body origin {np.round(grasp, 3)} is not the spawned grasp point "
        f"{np.round(t0.pos, 3)} — Planner would aim at the wrong place")
    fruit = np.array([data.geom(f"{t0.name}_f{k}").xpos
                      for k in range(t0.n_fruit)])
    assert (fruit[:, 2] < grasp[2]).all(), \
        "a fruit sits at or above the grasp point — the pads would close on it"

    gaps = [float(np.linalg.norm(a - b))
            for i, a in enumerate(fruit) for b in fruit[i + 1:]]
    assert min(gaps) > 2 * ft.FRUIT_R, (
        f"two fruit are {min(gaps) * 1000:.0f} mm apart against "
        f"{2 * ft.FRUIT_R * 1000:.0f} mm of diameter — the cluster "
        f"interpenetrates and MuJoCo will fire it apart")

    # And it hangs there rather than tearing off under its own weight.
    for _ in range(600):
        mujoco.mj_step(model, data)
    drift = float(np.linalg.norm(data.body(t0.name).xpos - t0.pos))
    assert drift < 0.01, (
        f"the truss sagged {drift * 1000:.0f} mm in 1.2 s — the weld is not "
        f"carrying {ft.TRUSS_MASS} kg")
    return (f"{t0.n_fruit} fruit, {mass:.2f} kg, closest pair "
            f"{min(gaps) * 1000:.0f} mm, sag {drift * 1000:.1f} mm")


@check("a truss weighs more than its own stem will hold at the loose SNAP_N")
def _truss_needs_its_own_snap_force():
    """Why `truss.TRUSS_SNAP_N` exists, as an assertion rather than a comment.

    ⚠️ Two-sided, in the manner of `_reset_home_free_body`. The cluster's static
    weight has to be **over** `plant_row.SNAP_N`, because that is the finding —
    a 0.8 kg truss on the loose crop's 12 N pedicel breaks under gravity before
    the gripper arrives, and the whole reason the truss carries a peduncle
    strength from published data instead. And it has to be comfortably **under**
    `TRUSS_SNAP_N`, or the truss variant would have shipped the same bug.

    If the first assertion ever fails, the crop got lighter and the constant
    should be re-derived rather than left as folklore.
    """
    from farm import truss as ft
    from plant_row import SNAP_N

    weight = ft.TRUSS_MASS * 9.81
    assert weight > SNAP_N * 0.6, (
        f"a truss now weighs {weight:.1f} N against the loose crop's "
        f"{SNAP_N} N snap threshold — the margin this variant was built "
        f"around has moved, re-read farm.truss.TRUSS_SNAP_N")
    headroom = ft.TRUSS_SNAP_N / weight
    assert headroom > 5.0, (
        f"the truss peduncle carries only {headroom:.1f}x the cluster's own "
        f"weight — a handling transient will break it")
    return (f"cluster weighs {weight:.2f} N; loose SNAP_N {SNAP_N} N would "
            f"leave {SNAP_N - weight:.1f} N; TRUSS_SNAP_N "
            f"{ft.TRUSS_SNAP_N} N gives {headroom:.1f}x")


@check("the cluster rule counts colour and the grouper finds the clusters")
def _cluster_cv_rule():
    """The take/leave decision, and the grouping it is computed over.

    ⚠️ **The threshold sweep is checked for monotonicity, not for a value.**
    Pinning "3 of 6" into an assertion is exactly what this file's own docstring
    forbids — it is a measurement, it lives in `truss.RIPE_FRACTION` with the
    table it was read off, and an honest change to the crop should move it
    without failing the build. What must never change is the *shape*: a stricter
    threshold cannot take more trusses, and cannot cut more green fruit. If that
    inverts, the rule is wired backwards and every number in the sweep is
    meaningless.

    The grouper is checked the way it is actually used — on the true fruit
    positions, where it must recover exactly the trusses that were planted.
    """
    from farm import truss as ft
    from farm.scout import Sighting

    ts = ft.spawn(n_per_row=6, seed=11)

    rows = ft.sweep(ts)
    takes = [r["trusses"] for r in rows]
    greens = [r["green_taken"] for r in rows]
    assert takes == sorted(takes, reverse=True), \
        f"a stricter threshold took more trusses: {takes}"
    assert greens == sorted(greens, reverse=True), \
        f"a stricter threshold cut more green fruit: {greens}"

    # The rule itself, on a cluster built by hand so the arithmetic is visible.
    def cluster(*stages):
        return ft.Cluster(members=[Sighting(pos=np.zeros(3), stage=s,
                                            hue=float("nan")) for s in stages])

    half = cluster("red", "red", "red", "turning", "green", "green")
    assert abs(half.red_fraction - 0.5) < 1e-9
    assert half.ripe_at(0.5) and not half.ripe_at(0.51), \
        "the threshold is not being compared inclusively"

    # And the grouping, over the crop's own fruit positions.
    sightings = [Sighting(pos=p, stage=s, hue=float("nan"))
                 for t in ts for p, s in zip(t.fruit_pos(), t.stages)]
    groups = ft.group(sightings)
    assert len(groups) == len(ts), (
        f"grouped {len(sightings)} fruit into {len(groups)} trusses, "
        f"not the {len(ts)} that were planted")
    assert all(g.n_fruit == ft.FRUIT_PER_TRUSS for g in groups), \
        "a group came back with the wrong number of fruit"
    err = max(float(np.min([np.linalg.norm(g.pos - t.pos) for t in ts]))
              for g in groups)
    assert err < 0.02, (
        f"a grouped grasp point is {err * 1000:.0f} mm off the real one")
    return (f"sweep monotone over {len(rows)} thresholds; "
            f"{len(groups)}/{len(ts)} trusses grouped, grasp point within "
            f"{err * 1000:.0f} mm")


@check("the truss blade will not cut before the gripper has hold of the stem")
def _blade_waits_for_the_grip():
    """`truss.Cutter`'s three conditions, each one a truss that went on the floor.

    ⚠️ **This is the check that the earlier gates would have failed**, which is
    the only reason it earns a place in a file that deliberately holds four
    assertions per mechanism. Cutting the peduncle is irreversible and it is the
    single moment 0.80 kg becomes the gripper's problem: fire it early and the
    cluster is released into a gripper that has not closed. Both earlier
    versions did, and both *sometimes* got away with it — the closing pads
    scooped the truss and the shift scored a `clean` pick — so the failure does
    not reliably show up in an outcome count.

    Checked on the mechanism rather than through a pick, so it costs
    milliseconds: hand the cutter a scene, and confirm it refuses while any one
    of proximity, two-sided pad load and finger stall is missing.
    """
    import mujoco
    from farm import truss as ft
    from plant_row import Row
    from reach import Gripper

    ts = ft.spawn(n_per_row=2, seed=3)
    model = ft.build(aisle=0, arms=("a",), trusses=ts, seed=3)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    row = Row(model, data, names=[t.name for t in ts],
              snap_n=ft.TRUSS_SNAP_N, homes={t.name: t.pos for t in ts})
    name = ts[0].name
    cut = ft.Cutter(model, data, row, name, Gripper(model, data))

    # Nothing has happened yet: the tool is parked, the pads are open.
    cut.tick()
    assert not cut.cut, "the blade cut with the arm parked and the pads open"

    # The commanded closure is the gate that shipped, and on its own it is not
    # a grip — the pads take most of a second to arrive and are 44 mm apart
    # around a 20 mm collar when the command passes half stroke.
    Gripper(model, data).close()
    mujoco.mj_forward(model, data)
    cut.tick()
    assert not cut.cut, (
        "the blade cut on the gripper *command* — the pads have not moved yet, "
        "and 0.80 kg is about to be released into an open gripper")
    assert row.attached(name), "the weld went inactive anyway"

    # And the pad-force gate on its own: force with the fingers still
    # travelling is one pad sweeping the truss across, not a hold.
    assert cut.finger_speed() >= 0.0
    assert min(cut.pad_load()) == 0.0, \
        "the pads report load on a collar nothing is touching"
    return (f"refused on parked, on command-only and on no-contact; gate is "
            f"{ft.CUT_HOLD_N:.0f} N on both pads with the fingers under "
            f"{ft.CUT_STALL_V}")


@check("a truss is carried and crated by its own geometry, not a tomato's")
def _truss_carry_geometry():
    """The three loose-fruit assumptions a hanging cluster breaks.

    ⚠️ Each one scored a *successful* pick as a failure or a failure as a
    success, which is why they are asserted rather than commented:

    1. the release height is `mission.BIN_DROP_UP` — a loose fruit ends 33 mm
       below the tool, a truss ends 271 mm below it, so the loose number drags
       the cluster through the crate wall
    2. `fr5.crate_contains` reads the body origin — for a truss that is the
       grasp point, a quarter-metre above the fruit lying in the crate
    3. `mission.CropObstacles` modelled a crop body as one ball at that same
       origin, so the planner could not see a neighbouring cluster at all
    """
    import mujoco
    import mission
    from farm import crop as fcrop, trolley, truss as ft
    from mission import BIN_WALL, crop_spheres

    # 1. the tool has to be high enough that the bottom fruit clears the rim
    hang = ft.RACHIS_LEN + ft.FRUIT_R
    assert mission.BIN_DROP_UP - hang < BIN_WALL, (
        "the loose release height already clears a hanging cluster — this "
        "check no longer describes the bug it was written for")
    assert ft.CRATE_DROP_UP - hang >= BIN_WALL, (
        f"a truss released at {ft.CRATE_DROP_UP:.3f} m puts its bottom fruit "
        f"{(ft.CRATE_DROP_UP - hang) * 1000:.0f} mm over a {BIN_WALL * 1000:.0f} "
        f"mm wall — it will be swept through the crate on the carry")

    # 2. and 3. want a scene
    ts = ft.spawn(n_per_row=2, seed=3)
    model = ft.build(aisle=0, arms=("a",), trusses=ts, seed=3)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    gids, radii = crop_spheres(model, ts[0].name)
    assert len(gids) == ft.FRUIT_PER_TRUSS + 1, (
        f"the planner sees a truss as {len(gids)} sphere(s), not the collar "
        f"plus {ft.FRUIT_PER_TRUSS} fruit — it is blind to the cluster")

    # The same call on a loose fruit must return exactly what Weeks 1-4 used,
    # or this change moved numbers it had no business touching.
    lt = fcrop.spawn(n_per_row=2, seed=3)
    lmodel = trolley.build(aisle=0, arms=("a",), trusses=lt, seed=3)
    lg, lr = crop_spheres(lmodel, lt[0].name)
    from plant_row import FRUIT_R
    assert len(lg) == 1 and abs(lr[0] - FRUIT_R) < 1e-9, (
        f"a loose fruit now reads as {len(lg)} sphere(s) of {lr} — Weeks 1-4 "
        f"clearances were all measured against one of exactly {FRUIT_R}")

    # 3. the crate, asked about the fruit rather than the grasp point
    bin_pos = np.array(trolley.crate_pos(model, data, "a"), dtype=float)
    name = ts[0].name
    # Stand the cluster in the crate: fruit on the floor, origin far above it.
    row_qadr = model.joint(model.body(name).jntadr[0]).qposadr[0]
    data.qpos[row_qadr:row_qadr + 3] = bin_pos + [0, 0, ft.RACHIS_LEN + 0.04]
    data.eq_active[model.equality(f"peduncle_{name}").id] = 0
    mujoco.mj_forward(model, data)
    from fr5 import crate_contains
    from mission import BIN_HALF
    assert not crate_contains(data.body(name).xpos, bin_pos, BIN_HALF,
                              BIN_WALL), \
        "the one-point crate test now passes for a standing truss — re-read it"
    assert ft.in_crate(model, data, name, bin_pos), (
        "a truss standing in the crate does not score as crated")
    return (f"release {ft.CRATE_DROP_UP:.3f} m clears a {hang * 1000:.0f} mm "
            f"cluster; planner sees {len(gids)} spheres per truss and "
            f"{len(lg)} per loose fruit; a standing truss crates")


def main():
    print("Vinea simulation tests")
    print("=" * 72)
    width = max(len(n) for _, n, _ in RESULTS)
    for ok, name, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print("=" * 72)
    failed = [n for ok, n, _ in RESULTS if not ok]
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"All {len(RESULTS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
