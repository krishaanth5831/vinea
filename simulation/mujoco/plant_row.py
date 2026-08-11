#!/usr/bin/env python3
"""A row of tomatoes hanging on stems that break when you pull hard enough.

This is Week 2's one genuinely new mechanic. Week 1 picked a sphere off a
table; a harvest pulls fruit off a plant, and the plant has to let go.

MuJoCo has no breakable joint, so detachment is built from two pieces that do
exist: a **weld equality** holding fruit to stem, and a line of Python that
switches the weld off once the force through it passes a threshold. That is the
model the roadmap settled on — not fracture, not FEM, both of which are a
different sprint and neither of which MuJoCo does.

    from plant_row import Row, add_row
    add_row(spec)                  # before compile
    row = Row(model, data)         # after
    row.update()                   # every control tick — this is what snaps

Run it directly to check the two numbers that have to be right before anything
built on top of this means anything:

    ./.venv/bin/python simulation/mujoco/plant_row.py
"""

import numpy as np

# --- the plant ---------------------------------------------------------------
# A vine tomato is roughly 65 mm across and 120 g. Both are carried over from
# week1_gripper unchanged so the contact behaviour stays comparable.
FRUIT_R = 0.033
FRUIT_MASS = 0.12

# How far the fruit hangs below the point the stem holds it. Short, because a
# long peduncle turns the fruit into a pendulum and the swing costs settling
# time on every one of the hundreds of picks this week runs.
STEM_LEN = 0.05
STEM_R = 0.004

# Where the row stands and how tall the support is. The FR5 reaches ~1.1 m to
# the fingertips from a shoulder at 0.152 m, and the MVP spec's Dutch rows run
# to three metres — which is the argument for the lift, not something to fake
# here. These trusses sit in the band this arm can actually serve; the rest of
# the row height is out of scope and said so, rather than pretended.
ROW_X = 0.60

# ⚠️ The support bar must clear the *gripper*, not just the fruit. At 0.78 it
# sat 40 mm above the highest truss, and once jitter pushed that truss up to
# z=0.740 the open pads fouled the bar on the way in — the grip load on the
# peduncle went 3.8 N -> 5.7 N and the fruit came out of the gripper during the
# carry. It looked like a grasp failure and was a scene-clearance failure.
# The pads are ~37 mm tall, so leave room for them plus the jitter.
SUPPORT_Z = 0.88

# ⚠️ SNAP_N IS MADE UP. There is no measurement behind it.
#
# No force gauge has been near a real peduncle for this project and there is no
# hardware budget to change that. It is an assumption, it directly decides
# whether a pick succeeds, and it flows straight into the kg/hr figure in
# Week 4. It is tuned here only so that a deliberate pull detaches the fruit and
# an accidental brush does not.
#
# The fruit's own weight is 1.18 N, so anything at or below that detaches the
# crop on contact with nothing but gravity. That is the floor from physics.
#
# ⚠️ There is a second floor, and it is not physics — it is this simulation.
# Closing the gripper on a fruit that is still attached loads the peduncle all
# by itself, and measured over ten jittered grasps that load peaked between
# **6.0 and 8.6 N** even after the close was ramped and the pads softened.
# Anything below ~9 N therefore detaches the fruit *on grip*, before the arm
# pulls at all — which still looks like a successful harvest on video and is
# not one. So SNAP_N cannot be chosen freely: the gripper sets its lower bound.
#
# 12 N sits about 3.4 N above that, and far below the >100 N the pull can
# reach, so a deliberate pull detaches and a settled grip (3.6-4.2 N) does not.
#
# The Week 2 instructions suggest starting at 5-10 N. That range is not
# reachable here for the reason above, and the gap is worth more than the
# number: it says the threshold is entangled with the gripper model, so a
# different gripper moves it. The Vinea cradle-and-blade design would remove
# the coupling entirely — it severs rather than pulls.
#
# A grower will ask where this came from. The answer is "assumed, not
# measured", and saying that first is what makes the rest credible.
SNAP_N = 12.0

# How stiff the peduncle is, as MuJoCo solref [timeconst, dampratio].
#
# ⚠️ This is not cosmetic and the default is wrong for this job. Left at
# MuJoCo's [0.02, 1] the weld is effectively rigid, and a rigid weld turns the
# gripper's own closing motion into force: the 2F85's four-bar linkage sweeps
# the pads slightly along the approach axis as they shut, and against a rigid
# constraint that sweep measured **15.8 N** — well over SNAP_N, so every fruit
# detached the instant it was gripped and never survived to be pulled. It looks
# like a working harvest on video and it is not one.
#
# A real peduncle stretches. Making this one compliant lets the fruit give a
# millimetre instead of generating force, and the grip load drops to 4.6 N —
# under the threshold, so the stem survives the grip and breaks on the pull,
# which is the behaviour being modelled. Measured across the range:
#
#   solref        sag     N after close    peak N on pull
#   [0.02, 1]    0.37 mm     15.77  (snaps on grip)  25.26
#   [0.05, 1]    1.23 mm      4.64  (survives)       16.07
#   [0.08, 1]    3.14 mm      2.03                   14.05
#   [0.20, 1]   19.61 mm      0.45                   12.91
#
# The cost is sag: 1.23 mm rather than 0.37. The Week 2 checklist asks for
# under 1 mm, but that number exists to catch the anchor trap below, which
# sags 160 mm — a millimetre still says the anchor is right.
PEDUNCLE_SOLREF = [0.05, 1.0]

# Where the fruit hang, as (name, y, z). Kept to five: every free body costs
# contact solver time, and this loop runs hundreds of times this week. Spread in
# y and z so the success rate means something across the workspace rather than
# at one spot.
#
# ⚠️ Every one of these has to clear the arm's *home posture*, which puts tool0
# at [0.671, -0.102, 0.478] with the fingers open around it. This is not a
# detail: a fruit spawned inside the open gripper is pushed the moment physics
# starts, and it never recovers. It is exactly what cost Week 1 its 1-in-6 —
# the [0.60, -0.10] sweep position sits 72 mm from home tool0, and the "failed
# grasp" there was the arm knocking the tomato off the table before the pick
# began. It reads as a grasp bug and is a scene-layout bug.
#
# A first pass at this row put t4 at (0.00, 0.50), 127 mm out. It hung at
# 4.76 N instead of 1.18 N and spiked to 9.25 N — over SNAP_N, so it snapped
# itself before the arm moved. Keep everything 200 mm clear and check it:
# `plant_row.py` prints the distances.
TRUSSES = [
    ("t0", 0.22, 0.72),
    ("t1", 0.08, 0.62),
    ("t2", -0.08, 0.70),
    ("t3", -0.24, 0.62),
    ("t4", 0.26, 0.54),
]


def fruit_home(name, trusses=TRUSSES):
    """Where a named fruit hangs when nothing has touched it.

    `trusses` defaults to the module's own five-fruit row, which is what every
    Week 1-3 caller wants. Week 4 builds rows from a pool whose names are not in
    that list, and looking them up here would raise — so the list is a parameter
    rather than a global read.
    """
    for n, y, z in trusses:
        if n == name:
            return np.array([ROW_X, y, z])
    raise KeyError(name)


def add_row(spec, trusses=TRUSSES, snap_n=SNAP_N):
    """Put the stand, the support bar and the hanging fruit into a spec.

    Call before `spec.compile()`. Returns the list of truss names, in order.
    """
    import mujoco

    # The back panel. Week 1 built this with collision off because it was
    # scenery you clicked on; here it is a thing the arm can hit, which is the
    # point — an approach that would have gone through the plant now fails
    # visibly instead of silently.
    panel = spec.worldbody.add_body(
        name="row_panel", pos=[ROW_X + 0.09, 0.0, (0.25 + 0.95) / 2])
    panel.add_geom(
        name="row_panel_face", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.38, (0.95 - 0.25) / 2], rgba=[0.25, 0.42, 0.28, 1.0],
    )

    # The horizontal support the stems hang from, at truss height.
    bar = spec.worldbody.add_body(
        name="row_support", pos=[ROW_X, 0.0, SUPPORT_Z])
    bar.add_geom(
        name="row_support_bar", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, -0.34, 0, 0, 0.34, 0], size=[0.008, 0, 0],
        rgba=[0.45, 0.34, 0.22, 1.0],
    )

    names = []
    for name, y, z in trusses:
        # The stem is a fixed body hanging off the support down to the fruit.
        # Collision is off: it is a 4 mm capsule that exists to be seen and to
        # anchor the weld, and giving the solver a thin geom to resolve against
        # the gripper buys nothing but instability.
        # Mocap, so the whole truss can be moved between attempts without
        # recompiling the model. A mocap body is drawn and can be teleported
        # via `data.mocap_pos`, but the solver never pushes it — which is what
        # an anchor point should be. Step 5's jitter moves the stem and the
        # fruit together; rebuilding the model 200 times instead would cost
        # more than the picks do.
        stem = spec.worldbody.add_body(
            name=f"stem_{name}", pos=[ROW_X, y, z + STEM_LEN], mocap=True)
        stem.add_geom(
            name=f"stem_{name}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -STEM_LEN], size=[STEM_R, 0, 0],
            rgba=[0.30, 0.50, 0.28, 1.0], contype=0, conaffinity=0,
        )
        # A hanger from the support down to the stem, so the row reads as
        # connected on video. Purely visual.
        drop = SUPPORT_Z - (z + STEM_LEN)
        if drop > 0.001:
            stem.add_geom(
                name=f"stem_{name}_drop", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, drop], size=[STEM_R * 0.6, 0, 0],
                rgba=[0.30, 0.50, 0.28, 1.0], contype=0, conaffinity=0,
            )

        fruit = spec.worldbody.add_body(name=name, pos=[ROW_X, y, z])
        fruit.add_freejoint()
        fruit.add_geom(
            name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[FRUIT_R, 0, 0], rgba=[0.85, 0.16, 0.12, 1.0],
            mass=FRUIT_MASS,
            # Menagerie's finger pads carry priority=1, so on any pad contact
            # *their* friction wins and this value governs only fruit-against-
            # crate and fruit-against-fruit. See Step 4 in the instructions.
            friction=[0.6, 0.01, 0.001],
        )

        eq = spec.add_equality(
            name=f"peduncle_{name}", type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"stem_{name}", name2=name,
        )
        # ⚠️ Eleven numbers: anchor(3), relpose pos(3), relpose quat(4),
        # torquescale(1). The first three MUST be written.
        #
        # Leave `data` alone and the compiler fills the anchor in as [0, 1, 0]
        # — an anchor a full metre off to the side. Every gram of fruit then
        # acts on a one-metre lever arm and the weld behaves like a rubber band.
        # Measured on this scene: 160.1 mm of sag with the default, 0.4 mm with
        # the anchor zeroed. It throws no error and prints no warning.
        eq.data = [0, 0, 0,   0, 0, -STEM_LEN,   1, 0, 0, 0,   1.0]
        eq.solref = list(PEDUNCLE_SOLREF)     # compliant; see the note above
        names.append(name)

    return names


def weld_force(model, data, eq_id):
    """Newtons the peduncle is currently carrying.

    Reads the constraint force straight out of the solver's efc arrays, picking
    the rows that belong to this one equality.

    Sanity-check it before trusting anything built on it: a 0.12 kg fruit
    hanging still must read 0.12 * 9.81 = 1.18 N. If it does not, the reading is
    wrong and so is every threshold set against it.

    ⚠️ **One equality at a time, and that is what makes it expensive in a row.**
    Each call builds two boolean masks the full length of `efc_*` — so asking
    for every fruit costs `n_fruit * n_efc`, per physics substep. Use
    `weld_forces` when more than one is wanted; this stays for the single-fruit
    callers and returns exactly the same number.
    """
    import mujoco

    rows = np.flatnonzero(
        (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        & (data.efc_id == eq_id))
    return float(np.linalg.norm(data.efc_force[rows][:3])) if rows.size else 0.0


def weld_forces(model, data, eq_ids):
    """`weld_force` for many equalities at once: `{eq_id: newtons}`.

    ⚠️ **One pass over `efc_*` for the whole row, instead of one pass per
    fruit, and it was the single largest cost in the simulation.** `Row.update`
    runs every *physics substep* — 500 Hz, not 100 — and asked `weld_force` for
    each of 48 fruit, each of which scanned all ~400 constraint rows twice. That
    is 38,400 masked comparisons per substep. Profiled over one two-armed pick
    it was **48.3 s of 123.5 s, 39% of the entire run**, and it never appeared
    in a rendering budget because it is not rendering.

    The equality rows for one constraint are contiguous and in `efc_id` order,
    so a single `argsort`-free pass that groups by id gives every fruit's force
    for the cost of one fruit's. The arithmetic per fruit is unchanged — same
    rows, same `norm` of the same first three components — so every force,
    every `peak`, and every snap decision is bit-identical to before.
    """
    import mujoco

    out = {int(e): 0.0 for e in eq_ids}
    eq_rows = np.flatnonzero(
        data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    if not eq_rows.size:
        return out

    ids = data.efc_id[eq_rows]
    force = data.efc_force[eq_rows]
    # `efc` rows are emitted in constraint order, so each equality's rows are
    # contiguous and `return_index` gives the first of each block.
    uniq, start, counts = np.unique(ids, return_index=True, return_counts=True)

    # The first three rows of a weld are the translational ones — the same
    # `[:3]` slice `weld_force` takes. A weld that produced fewer (it cannot
    # today, but a disabled one produces none) is padded with zeros rather than
    # indexed out of range.
    off = np.arange(3)
    idx = start[:, None] + off[None, :]
    valid = off[None, :] < counts[:, None]
    rows = force[np.where(valid, idx, 0)] * valid
    norms = np.linalg.norm(rows, axis=1)

    for eq_id, n in zip(uniq, norms):
        e = int(eq_id)
        if e in out:
            out[e] = float(n)
    return out


class Row:
    """The live row: which stems are still attached, and when they let go.

    `data.eq_active` is a per-equality on/off switch that lives in mjData, not
    mjModel. That is exactly right for this: breaking a stem is a *state*
    change, so resetting the scene puts every stem back without rebuilding the
    model.
    """

    def __init__(self, model, data, names=None, snap_n=SNAP_N, homes=None):
        """`homes` is where each fruit hangs when untouched.

        Left None it is looked up in the module's own TRUSSES, which is right
        for Weeks 1-3. Week 4 places fruit wherever it likes, so there is no
        list to look them up in and the caller supplies the map instead.
        """
        self.model = model
        self.data = data
        self.snap_n = snap_n
        self.names = [n for n, _, _ in TRUSSES] if names is None else list(names)
        self.eq_id = {n: model.equality(f"peduncle_{n}").id for n in self.names}
        self.body_id = {n: model.body(n).id for n in self.names}
        self.mocap_id = {n: model.body(f"stem_{n}").mocapid[0] for n in self.names}
        self.qadr = {n: model.joint(model.body(n).jntadr[0]).qposadr[0]
                     for n in self.names}
        self.home = ({n: fruit_home(n) for n in self.names} if homes is None
                     else {n: np.asarray(homes[n], dtype=float)
                           for n in self.names})
        self.peak = {n: 0.0 for n in self.names}

    def reset(self):
        """Re-attach every stem and forget the force history."""
        for n in self.names:
            self.data.eq_active[self.eq_id[n]] = 1
        self.peak = {n: 0.0 for n in self.names}

    def attachment(self):
        """Which stems are attached right now, as a snapshot to restore later."""
        return {n: bool(self.data.eq_active[self.eq_id[n]]) for n in self.names}

    def settle(self, snapshot=None):
        """Forget the force history, and undo any snap during a settling period.

        ⚠️ Not `reset`, and the difference is the whole of sequence harvesting.
        Every pick begins by holding still while the weld finds its resting
        force, because the startup transient overshoots SNAP_N and would snap
        stems nothing has touched. `reset` clears that by re-attaching *every*
        stem — which is right when the row is being rebuilt for a fresh trial,
        and catastrophic when it is not: on the second pick of a sequence it
        re-welds the tomato already sitting in the crate to its stem back on the
        plant, and the constraint hauls it across the scene. Measured, that
        registered as 305 N through a peduncle and read as a mystery loss with
        no contact behind it.

        Passing the snapshot taken before the settle restores exactly what was
        attached then, and leaves harvested fruit harvested.
        """
        if snapshot is not None:
            for n, attached in snapshot.items():
                self.data.eq_active[self.eq_id[n]] = 1 if attached else 0
        self.peak = {n: 0.0 for n in self.names}

    def place(self, name, pos):
        """Move a whole truss — stem anchor and fruit together.

        Both have to move as one. Move only the fruit and the weld drags it
        back to the stem; move only the stem and the weld yanks the fruit
        across the scene. Either way the next pick starts with a swinging
        tomato and the attempt measures the swing, not the grasp.
        """
        pos = np.asarray(pos, dtype=float)
        self.data.mocap_pos[self.mocap_id[name]] = pos + [0, 0, STEM_LEN]
        a = self.qadr[name]
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.data.qvel[self.model.body(name).dofadr[0]:
                       self.model.body(name).dofadr[0] + 6] = 0.0

    def jitter(self, rng, mm):
        """Nudge every fruit by up to `mm` in each axis. Returns the offsets.

        Picking the same coordinate ten times mostly proves the simulator is
        deterministic. Week 3 feeds positions from a camera that has real error
        in it and Week 4 quotes a success rate over varied fruit, so the number
        only means something if the target moved.
        """
        offsets = {}
        for n in self.names:
            d = rng.uniform(-mm / 1000, mm / 1000, size=3)
            offsets[n] = d
            self.place(n, self.home[n] + d)
        return offsets

    def force(self, name):
        return weld_force(self.model, self.data, self.eq_id[name])

    def attached(self, name):
        return bool(self.data.eq_active[self.eq_id[name]])

    def update(self):
        """Snap any stem carrying more than snap_n. Call every control tick.

        Returns the names that broke on this call, which is almost always
        empty and occasionally one.

        ⚠️ **One `efc` pass for the whole row, not one per fruit.** This runs on
        every *physics substep*, so the per-fruit version cost `n_fruit * n_efc`
        masked comparisons 500 times a second — 39% of a two-armed run's total
        time, measured. See `weld_forces`. The forces, the peaks and the snap
        decisions are unchanged; only the number of array scans is.
        """
        live = [n for n in self.names if self.attached(n)]
        if not live:
            return []
        forces = weld_forces(self.model, self.data,
                             [self.eq_id[n] for n in live])
        broke = []
        peak, eq = self.peak, self.eq_id
        for n in live:
            f = forces[eq[n]]
            if f > peak[n]:
                peak[n] = f
            if f > self.snap_n:
                self.data.eq_active[eq[n]] = 0
                broke.append(n)
        return broke

    def pos(self, name):
        return self.data.body(name).xpos.copy()

    def stem_pos(self, name):
        """Where the peduncle is — the point a blade would have to reach."""
        return self.data.body(f"stem_{name}").xpos.copy()


def print_clearances(model, data, row, prefix=""):
    """How far each fruit sits from the named arm's tool, at the current pose.

    This is entry 23's check, kept as code: the "1-in-6 grasp failure" was a
    fixture bug, a truss spawned 72 mm from the home posture and *inside* the
    open gripper, so the arm knocked it off the plant before the pick began.
    A row that fails this is a scene bug and reads exactly like a control bug.

    ⚠️ `prefix` names which arm. It defaults to `""` — the unprefixed Week 1-4
    arm — so this row scene is unaffected. Without it, running the check on a
    two-armed scene measures every clearance against arm A while reporting it
    under whichever arm the caller had in mind. See `fr5.tool_pos`.
    """
    from fr5 import tool_pos

    tool = tool_pos(data, prefix)
    print(f"\n  clearance from home {prefix}tool0 {tool.round(3)}:")
    for n in row.names:
        d = float(np.linalg.norm(data.body(n).xpos - tool)) * 1000
        print(f"    {n}: {d:5.0f} mm   {'ok' if d > 200 else 'TOO CLOSE'}")


def main():
    """Verify the two numbers Step 2 stands on, before anything uses them."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from fr5 import build_fr5_spec, reset_home

    print("Step 2 check — the weld that holds fruit on a stem\n")

    # --- the anchor trap, measured both ways --------------------------------
    for label, zero_anchor in [("compiler default", False), ("anchor zeroed", True)]:
        spec = build_fr5_spec(gripper=True)
        add_row(spec)
        if not zero_anchor:
            # Put the compiler's own default back, to measure what it costs.
            for eq in spec.equalities:
                if eq.name.startswith("peduncle_"):
                    eq.data = [0, 1, 0,  0, 0, -STEM_LEN,  1, 0, 0, 0,  1.0]
        model = spec.compile()
        data = mujoco.MjData(model)
        reset_home(model, data)
        start = {n: data.body(n).xpos.copy() for n, _, _ in TRUSSES}
        for _ in range(1000):        # 2 s at a 2 ms timestep
            mujoco.mj_step(model, data)
        sag = max(float(np.linalg.norm(data.body(n).xpos - start[n])) * 1000
                  for n, _, _ in TRUSSES)
        print(f"  sag after 2 s, {label:<18} {sag:7.1f} mm")

    # --- does weld_force read the fruit's weight? ---------------------------
    spec = build_fr5_spec(gripper=True)
    add_row(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset_home(model, data)
    row = Row(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)

    expected = FRUIT_MASS * 9.81
    print(f"\n  expected resting force  {expected:.2f} N  "
          f"({FRUIT_MASS} kg x 9.81)")
    for n in row.names:
        f = row.force(n)
        ok = "ok" if abs(f - expected) < 0.05 else "WRONG"
        print(f"    {n}: {f:.2f} N   {ok}")

    # --- how far is each fruit from the arm's home posture? -----------------
    print_clearances(model, data, row)

    # --- does it snap, and only when pulled? --------------------------------
    #
    # Settle first. The weld takes ~50 ms to find its resting force and the
    # transient overshoots; snapping during it would break stems that nothing
    # ever pulled on. The pick loop settles for the same reason.
    print(f"\n  SNAP_N = {SNAP_N} N  (assumed, not measured)")
    name = row.names[0]
    for pull in [0.0, 2.0, 4.5, 5.5, 20.0]:
        data2 = mujoco.MjData(model)
        reset_home(model, data2)
        r2 = Row(model, data2)
        bid = r2.body_id[name]
        for _ in range(100):                      # 0.2 s of settling
            mujoco.mj_step(model, data2)
        r2.reset()                                # forget the transient
        broke_at = None
        for i in range(1500):
            data2.xfrc_applied[bid] = [0, 0, -pull, 0, 0, 0]
            mujoco.mj_step(model, data2)
            if broke_at is None and name in r2.update():
                broke_at = i * model.opt.timestep
        # `is not None`, not truthiness — a snap on the very first tick gives
        # broke_at == 0.0, which is falsy, and reads as "held" when it is the
        # opposite. That is how this test lied the first time it was run.
        verdict = ("SNAPPED at %.2f s" % broke_at if broke_at is not None
                   else "held")
        print(f"    pull {pull:5.1f} N down -> peak {r2.peak[name]:6.2f} N  "
              f"{verdict}")


if __name__ == "__main__":
    main()
