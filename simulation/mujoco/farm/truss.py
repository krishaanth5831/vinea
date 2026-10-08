#!/usr/bin/env python3
"""Tomatoes on the vine: a whole cluster on one rachis, picked by the stem.

`farm/crop.py` hangs **loose** fruit — one tomato per stem, each picked off the
plant on its own. This module hangs **trusses**: six tomatoes on a shared
rachis, taken as one unit by cutting the stem above the top fruit. They are the
two products a Dutch glasshouse actually sells, and they are different jobs for
the robot, not a cosmetic change:

    loose   the arm grasps a 66 mm sphere, one decision per fruit, 120 g
    truss   the arm grasps a 20 mm stem, one decision per *cluster*, 800 g

⚠️ **The ripeness decision is the part that changes shape.** A loose picker asks
"is this tomato red?" and the answer is about the thing it is holding. A truss
picker cannot ask that, because it takes six fruit with one cut and the truss is
graded and sold as one item — a single green fruit in the cluster downgrades
the whole thing, and a truss left one pass too long has soft fruit at the base.
So the question becomes "is *enough* of this cluster red?", and the threshold on
that is a commercial decision the machine has to be told. See `RIPE_FRACTION`,
which was chosen by sweeping it rather than by picking a number that sounded
right — `--sweep` reproduces the measurement.

--- where the numbers come from ---------------------------------------------

Commercial TOV ("tomato on the vine") is harvested as a complete truss of 4-6
fruit at 100-120 g each, and truss varieties are bred to carry 5-6 fruit at up
to 140 g. Six fruit at 125 g plus a 50 g rachis is **0.80 kg**, which is where
`TRUSS_MASS` comes from and why six is the fruit count rather than a round
number someone liked. Sources are in the repo's README notes; the figures are
Rijk Zwaan's truss varieties and Dutch Greenhouses' TOV agronomy.

⚠️ **Fruit do not ripen at the same time within a truss, and modelling them as
independent draws would make the cluster decision meaningless.** A truss ripens
*basipetally* — the fruit nearest the main vine colours first and the distal tip
last — so a real cluster is a gradient, not a random assortment. `spawn` draws
one maturity per truss and steps it down the rachis (`RIPEN_GRADIENT`). That is
what produces the partly-ripe trusses the threshold has to adjudicate; six
independent draws would produce almost none, and the threshold would then be
measuring the random number generator.

    ./.venv/bin/python simulation/mujoco/farm/truss.py --stats   # what spawned
    ./.venv/bin/python simulation/mujoco/farm/truss.py --sweep   # the threshold
    ./.venv/bin/python simulation/mujoco/farm/truss.py --shot    # stills
    ./.venv/bin/python simulation/mujoco/farm/truss.py           # a window
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402
from farm import house, trolley  # noqa: E402
# BIN_HALF and BIN_WALL are the crate's own geometry and are the two names in
# `mission` that `farm.armframe` does *not* rebind per arm — see its `_HOLDERS`
# note. Importing them by value is therefore safe here, where importing
# `BIN_POS` by value would not be.
from mission import BIN_HALF, BIN_WALL  # noqa: E402
from plant_row import PEDUNCLE_SOLREF, STEM_LEN, STEM_R  # noqa: E402

# --- the cluster -------------------------------------------------------------

# ⚠️ Six, and it is a floor rather than a preference. The brief asks for at
# least six; commercial TOV runs 4-6 and truss varieties are bred to 5-6, so six
# is the top of the real range and the hardest version of the reach problem —
# a longer rachis puts the distal fruit further below the grasp point.
FRUIT_PER_TRUSS = 6

# The whole cluster, in kg. Six fruit at 125 g and a 50 g rachis.
#
# ⚠️ **This is 6.7x the loose fruit's 120 g and that is the point of the
# variant.** `plant_row.FRUIT_MASS` is what every Week 1-4 grip number was
# measured against; a truss is most of a kilogram hanging off a 20 mm stem held
# in a 2F85, which is a genuinely harder hold and is why `--grip` exists below.
TRUSS_MASS = 0.80
RACHIS_MASS = 0.05
FRUIT_MASS = (TRUSS_MASS - RACHIS_MASS) / FRUIT_PER_TRUSS      # 0.125 kg

FRUIT_R = 0.033           # same fruit as `plant_row`, so the same gripper fits

# What the truss's own stem will carry before it lets go, in newtons.
#
# ⚠️ **`plant_row.SNAP_N` is 12 N and a truss breaks its own stem under gravity
# at that figure.** The cluster weighs 0.80 kg — 7.85 N hanging on the weld
# before anything touches it — so 12 N leaves 4.2 N of headroom against a grasp
# transient that `plant_row` itself measures at 6.0-8.6 N. Measured here before
# this constant existed: every truss snapped during the approach and was on the
# floor before the gripper closed, scored `grasp_failed` with `broke=True`.
#
# ⚠️ **So this is the one number in the crop that is *not* copied from
# `plant_row`, and unlike that one it is not assumed.** `plant_row.SNAP_N`'s own
# comment says, at length, that 12 N is made up and that no force gauge has been
# near a peduncle for this project. There is published data:
#
#   tomato abscission-zone pull-off force, by stem diameter
#     5-6 mm                40.3 N        tensile strength 1.69 MPa
#     6-7 mm (mean 6.27)    44.8 N        tensile strength 1.44 MPa
#     7-8 mm (mean 7.75)    72.0 N        tensile strength 1.53 MPa
#
#   — Wang et al., *Tomato Pedicel Physical Characterization for Fruit-Pedicel
#     Separation Tomato Harvesting Robot*, Agronomy 14(10):2274, 2024.
#
# Note the vocabulary, because the repo has been using one word for two things:
# a **pedicel** carries one fruit, a **peduncle** is the branch carrying the
# whole cluster. `plant_row`'s `peduncle_*` welds are pedicels. This variant has
# an actual peduncle, and it is thicker — strength is roughly constant at
# ~1.5 MPa and force scales with cross-section, so the 8-10 mm stem a six-fruit
# truss hangs on lands at 75-118 N. 75 N is the conservative end of that and
# also within a newton of the 7-8 mm group's measured 72 N.
#
# That leaves 9.5x headroom over the cluster's static weight, which is the same
# ratio `plant_row`'s 12 N gives its 1.18 N fruit — so the truss pick is not
# being made easy, it is being held to the same margin.
TRUSS_SNAP_N = 75.0

# The rachis, in the body's own frame. The body origin is the **grasp point**,
# which is what lets `mission.Planner.plan(name)` route to a truss with no
# change at all: it plans to `row.pos(name)`, and for a truss that has to be the
# stem above the top fruit rather than the centroid of the cluster.
GRIP_R = 0.010            # the collar the pads close on
GRIP_LEN = 0.045          # how much clean stem there is to grab
RACHIS_R = 0.005
FRUIT_DZ = 0.038          # step down the rachis between fruit
FRUIT_DY = 0.040          # alternating either side, the herringbone a truss has
FRUIT_Z0 = 0.048          # the first fruit, below the grip collar

# How far the bottom of the cluster hangs below the grasp point.
RACHIS_LEN = FRUIT_Z0 + (FRUIT_PER_TRUSS - 1) * FRUIT_DZ       # 0.238 m


def fruit_offsets(n=FRUIT_PER_TRUSS):
    """Where each fruit sits in the truss body's frame, proximal first.

    ⚠️ Alternating either side of the rachis, which is what a truss looks like
    and also what keeps the fruit from interpenetrating. Consecutive fruit are
    `sqrt((2*FRUIT_DY)^2 + FRUIT_DZ^2)` apart — 88 mm against the 66 mm that
    would touch — so the cluster is stable on the first step. Stacked straight
    down the rachis at `FRUIT_DZ` they would overlap by half a fruit and MuJoCo
    would fire the truss apart before the arm moved.
    """
    out = []
    for k in range(n):
        side = 1.0 if k % 2 == 0 else -1.0
        out.append(np.array([0.0, side * FRUIT_DY, -(FRUIT_Z0 + k * FRUIT_DZ)]))
    return out


# --- how ripe a truss is -----------------------------------------------------

# ⚠️ **The gradient along the rachis, and the reason it is not zero.** Tomatoes
# on one truss ripen basipetally: proximal fruit first, distal last. This is how
# much of the ripening scale one fruit position costs, so a six-fruit truss
# spans 5 * RIPEN_GRADIENT of it and a cluster is nearly always a mix. Set it to
# 0 and every truss becomes uniform, every threshold gives the same answer, and
# `--sweep` measures nothing.
RIPEN_GRADIENT = 0.13

# Ripening score above which each stage begins. The four stages are
# `crop.STAGES`, unchanged, so a truss fruit and a loose fruit of the same name
# are the same colour and the same detector reads both.
STAGE_EDGES = (("green", 0.30), ("breaker", 0.55), ("turning", 0.78),
               ("red", 9.99))

# ⚠️ **The threshold this whole variant exists to exercise, and it is set from
# `--sweep` rather than from taste.** A truss is taken when at least this
# fraction of its fruit read `red` — 0.50 is "at least 3 of 6".
#
# There is no published number to copy. The literature on robotic tomato
# harvesting classifies *fruit* ripeness and leaves the cluster decision to the
# grower's schedule; several systems skip it entirely, on the grounds that in a
# structured glasshouse the bottom-most truss is ready when the weekly round
# reaches it. So this was measured instead. `--sweep` over 60-odd trusses, and
# the same shape on all four seeds tried (5, 11, 23, 41):
#
#   take at    trusses   red taken   of all red   green   downgraded   red left
#    1 of 6         39         145         100%       6            6          0
#    2 of 6         36         142          98%       3            3          3
#    3 of 6         28         126          87%       0            0         19   <-- ships
#    4 of 6         21         105          72%       0            0         40
#    6 of 6          7          42          29%       0            0        103
#
# **3 of 6 is the knee, and the column that decides it is `downgraded`** — the
# number of harvested trusses carrying at least one green fruit. TOV is graded
# and sold one truss at a time, so that is the unit the mistake is paid in. 3 of
# 6 is the *loosest* threshold that is zero-downgrade on every seed tried (2 of
# 6 downgrades trusses on three seeds out of four) and it still brings in 84-91%
# of all the red fruit in the house.
#
# Tightening further is pure loss: 4 of 6 gives up another 15 points of red
# capture and cannot improve on zero. Loosening to 1 of 6 takes every red fruit
# there is and downgrades six trusses to do it.
#
# ⚠️ The `unripe taken` column is deliberately *not* what this is optimised
# against. Commercial practice harvests just after breaker and the fruit colours
# in transit, so breaker and turning fruit cut with the truss are the normal
# product. Green is the mistake.
RIPE_FRACTION = 0.50

# How far apart two detected fruit can be and still be called the same truss.
#
# ⚠️ Sits in a genuinely wide gap, which is the only reason clustering from
# colour blobs alone is honest here. Consecutive fruit *within* a truss are
# 88 mm apart; two trusses are at least `MIN_TRUSS_SEP` = 350 mm apart. Anything
# from about 0.10 to 0.25 m gives the same grouping, so this is not a tuned
# number and a small change to the crop layout cannot silently break it.
CLUSTER_GAP = 0.15

# Trusses on one row, centre to centre. See `CLUSTER_GAP`.
MIN_TRUSS_SEP = 0.35

# Where a truss's *grasp point* may hang, in the arm's frame. Tighter than
# `crop.Z_LOCAL` at the bottom because the cluster hangs `RACHIS_LEN` below the
# grasp and the whole thing has to stay off the gutter.
GRIP_Z_LOCAL = (0.62, 0.72)


@dataclass
class Truss:
    """One cluster as the *operator* knows it. The robot has to work it out."""

    name: str
    row: int
    x: float
    y: float
    z: float                       # the grasp point, on the stem
    stages: list = field(default_factory=list)     # proximal -> distal

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    @property
    def n_fruit(self):
        return len(self.stages)

    @property
    def n_red(self):
        return sum(1 for s in self.stages if s == "red")

    @property
    def red_fraction(self):
        return self.n_red / max(1, self.n_fruit)

    @property
    def ripe(self):
        """The ground-truth cluster decision, at the shipped threshold."""
        return self.red_fraction >= RIPE_FRACTION

    def fruit_pos(self):
        """World positions of each fruit, proximal first."""
        return [self.pos + o for o in fruit_offsets(self.n_fruit)]

    def rgba(self, k):
        return fcrop.STAGE_BY_NAME[self.stages[k]][0]


def stage_for(score):
    """Ripening score -> colour stage. See `STAGE_EDGES`."""
    for name, edge in STAGE_EDGES:
        if score < edge:
            return name
    return "red"


def spawn(n_per_row=5, rows=None, seed=None, n_fruit=FRUIT_PER_TRUSS,
          min_sep=MIN_TRUSS_SEP):
    """Hang `n_per_row` trusses on each row, each a gradient of ripeness.

    ⚠️ One maturity is drawn per *truss*, not per fruit, and then stepped down
    the rachis by `RIPEN_GRADIENT`. That is the whole reason a cluster decision
    is a decision — see the module docstring.

    The maturity range runs past both ends of the stage scale on purpose, so the
    house contains fully-green clusters nobody should touch and fully-red ones
    everybody should, as well as the partial ones the threshold adjudicates. A
    range that only produced partial trusses would make every threshold look
    equally defensible.
    """
    rng = np.random.default_rng(seed)
    rows = list(range(house.N_ROWS)) if rows is None else list(rows)

    out, k = [], 0
    for r in rows:
        placed, tries = [], 0
        while len(placed) < n_per_row and tries < 4000:
            tries += 1
            y = float(rng.uniform(-house.HOUSE_HALF_Y + 0.5,
                                  house.HOUSE_HALF_Y - 0.5))
            if any(abs(y - py) < min_sep for py in placed):
                continue
            placed.append(y)
            z = float(rng.uniform(*GRIP_Z_LOCAL))

            # One maturity for the cluster, stepped down the rachis.
            # ⚠️ The range is set so the house is roughly two-fifths clearly
            # unready, two-fifths partial and one-fifth fully red. Partial
            # clusters are the only ones the threshold has an opinion about, so
            # a distribution that produced few of them would make `--sweep` a
            # flat table and the choice of threshold unfalsifiable.
            m = float(rng.uniform(0.15, 1.60))
            stages = [stage_for(m - i * RIPEN_GRADIENT
                                + float(rng.normal(0.0, 0.03)))
                      for i in range(n_fruit)]
            out.append(Truss(name=f"c{k:03d}", row=r, x=house.row_x(r), y=y,
                             z=fcrop.fruit_z(z), stages=stages))
            k += 1
    return out


# --- putting them in a scene -------------------------------------------------

def add_trusses(spec, trusses):
    """Put every truss into a spec as one welded, pickable cluster.

    ⚠️ **One rigid body per truss, and that is what "picked by the stem" means
    physically.** The rachis and all six fruit are geoms on a single body with
    one free joint, so cutting the stem hands the arm the whole cluster and the
    fruit cannot be lost individually. The alternative — six welded fruit on a
    carrier — models fruit being shaken off during the carry, which is a real
    failure mode, but it is not the one this variant is about and it would make
    the 0.8 kg an emergent number rather than a stated one.

    ⚠️ **The body origin is the grasp point.** Everything downstream depends on
    it: `mission.Planner.plan` routes to `row.pos(name)`, `plant_row.Row` reads
    the body pose, and `farm.run.associate` matches a mapped position to a body
    name by distance. Putting the origin at the cluster's centroid instead would
    aim the gripper into the middle of six tomatoes.

    The weld, its anchor and `PEDUNCLE_SOLREF` are `crop.add_trusses`'s, copied
    rather than re-chosen for the reason that function documents: they are what
    Weeks 2-4 measured against.
    """
    import mujoco

    rows = sorted({t.row for t in trusses})
    for r in rows:
        top = fcrop.fruit_z(0.88)
        bar = spec.worldbody.add_body(name=f"tr_support_r{r}",
                                      pos=[house.row_x(r), 0.0, top])
        bar.add_geom(name=f"tr_support_r{r}_bar",
                     type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                     fromto=[0, -house.HOUSE_HALF_Y, 0,
                             0, house.HOUSE_HALF_Y, 0],
                     size=[0.008, 0, 0], rgba=[0.45, 0.34, 0.22, 1.0],
                     contype=0, conaffinity=0, mass=0.0)

    offsets = fruit_offsets()
    for t in trusses:
        # The stem the truss hangs from: a mocap body, exactly as the loose crop
        # does it, so `plant_row.Row` can place and reset a truss unchanged.
        stem = spec.worldbody.add_body(
            name=f"stem_{t.name}", pos=[t.x, t.y, t.z + STEM_LEN], mocap=True)
        stem.add_geom(
            name=f"stem_{t.name}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -STEM_LEN], size=[STEM_R, 0, 0],
            rgba=[0.30, 0.50, 0.28, 1.0], contype=0, conaffinity=0)
        drop = fcrop.fruit_z(0.88) - (t.z + STEM_LEN)
        if drop > 0.001:
            stem.add_geom(
                name=f"stem_{t.name}_drop", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, drop], size=[STEM_R * 0.6, 0, 0],
                rgba=[0.30, 0.50, 0.28, 1.0], contype=0, conaffinity=0)

        body = spec.worldbody.add_body(name=t.name, pos=[t.x, t.y, t.z])
        body.add_freejoint()

        # The collar the pads close on. Thicker than the rachis below it: a
        # truss stem carries a knuckle where it meets the vine, and a 2F85
        # closing on 10 mm of radius has something to hold. Friction is the
        # loose fruit's, so the hold is not bought with a number nothing else
        # uses.
        #
        # ⚠️ **Named `{name}_geom`, which is the loose crop's name for the fruit
        # sphere, and that is deliberate.** `carrytrace.CarryTrace` and
        # `incident.Blackbox` both look up exactly that name to find "the geom
        # the pads are holding" — the thing whose pad forces are measured and
        # whose contacts are attributed to this pick. For a truss that geom is
        # the collar, not any one tomato, so the name goes here and both
        # recorders work unchanged and measure the right surface.
        body.add_geom(
            name=f"{t.name}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, GRIP_LEN / 2, 0, 0, -GRIP_LEN / 2],
            size=[GRIP_R, 0, 0], rgba=[0.32, 0.52, 0.30, 1.0],
            mass=RACHIS_MASS, friction=[0.6, 0.01, 0.001])
        body.add_geom(
            name=f"{t.name}_rachis", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, -GRIP_LEN / 2, 0, 0, -RACHIS_LEN],
            size=[RACHIS_R, 0, 0], rgba=[0.30, 0.50, 0.28, 1.0],
            mass=0.0, contype=0, conaffinity=0)

        for k, off in enumerate(offsets[:t.n_fruit]):
            # A short pedicel out to each fruit, so the cluster reads as a truss
            # rather than as beads floating beside a stick.
            body.add_geom(
                name=f"{t.name}_ped{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, off[2] + 0.012] + list(off),
                size=[RACHIS_R * 0.8, 0, 0], rgba=[0.30, 0.50, 0.28, 1.0],
                mass=0.0, contype=0, conaffinity=0)
            body.add_geom(
                name=f"{t.name}_f{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=list(off), size=[FRUIT_R, 0, 0], rgba=list(t.rgba(k)),
                mass=FRUIT_MASS, friction=[0.6, 0.01, 0.001])

        eq = spec.add_equality(
            name=f"peduncle_{t.name}", type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"stem_{t.name}", name2=t.name)
        # The eleven numbers, and the first three MUST be written — see
        # `crop.add_trusses`, which learned that the hard way.
        eq.data = [0, 0, 0, 0, 0, -STEM_LEN, 1, 0, 0, 0, 1.0]
        eq.solref = list(PEDUNCLE_SOLREF)
    return [t.name for t in trusses]


def build(aisle=0, arms=("a",), crate=True, wrist_cam=False, deck_cam=False,
          leafy=True, pitch=0.32, seed=0, trusses=None, arm_decks=False):
    """`trolley.build`, with trusses hung instead of loose fruit.

    Same signature and the same scene in every other respect, so a viewer can
    swap one call and get the truss house. `trolley.build` takes its crop as a
    list of `crop.Truss` and calls `crop.add_trusses`; there is no hook to pass
    a different crop model, so this rebuilds the scene the same way rather than
    threading a callback through a function five other files call.
    """
    import mujoco

    import greenhouse as gh
    from farm import trolley as _t

    spec = mujoco.MjSpec()
    # Solver options are global and do not survive `attach` — see
    # `trolley.build`, which documents why every one of these is load-bearing.
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    worked = house.serves(aisle)
    house.add_house(spec, leafy=leafy, pitch=pitch, worked_rows=worked,
                    seed=seed)
    if trusses is not None:
        add_trusses(spec, trusses)
    _t.add_trolley(spec, aisle=aisle, arms=arms, crate=crate)

    if wrist_cam:
        from camera import add_wrist_camera

        for tag in arms:
            add_wrist_camera(spec, prefix=_t.ARM_PREFIX[tag])
    if deck_cam:
        from farm.scout import add_deck_camera

        add_deck_camera(spec, aisle=aisle)
    if arm_decks:
        from farm.decks import add_arm_deck_cameras

        add_arm_deck_cameras(spec, aisle=aisle, arms=arms)

    for geom in spec.geoms:
        if "pad" in geom.name:
            geom.solref = list(gh.PAD_SOLREF)

    return spec.compile()


# --- the blade ---------------------------------------------------------------

# What the pads have to be pushing on the collar, per pad, before the blade will
# cut. **Both** pads, measured off the contacts, in newtons.
#
# ⚠️ **This replaces a commanded-closure gate, and that gate was the reason the
# truss went on the floor.** The first version cut at `ctrl >= 0.55 * 255` on
# the stated grounds that the pads had met a 20 mm stem by then. They have not.
# Closing the 2F85 on nothing and reading the gap between the pad centres:
#
#     ctrl      0     80    120    140    180    200    220    255
#     gap mm   93.2   68.5  54.9   47.9   33.5   26.2   18.8    8.8
#
# The pads touch each other at 8.8 mm of centre separation, so a 20 mm collar is
# held at ~29 mm of it — ctrl ≈ 200. At the 140 the blade was firing at, the
# pads are 44 mm apart around a 20 mm stem: 12 mm of air on each side. Measured
# through the real mission on seed 7, the pad *forces* on the collar during the
# grip ramp are
#
#     ctrl      121    138     155     172     206     240
#     pad L N   0.00   9.56   35.48   48.71   50.20   53.61
#     pad R N   0.00   0.00   24.85   32.83   35.33   39.40
#
# — so at ctrl 140 exactly one pad is touching, and it is *pushing the truss
# sideways*. The weld was released there, into a gripper that had not got hold
# of anything, and 0.80 kg fell. Sometimes the closing pads scooped it on the
# way past and the run scored `clean` anyway, which is worse than failing: the
# shift report's own "the fruit was flung and happened to land in the crate"
# line was that.
#
# 5 N is one-eighth of the ~43 N the weaker pad settles at, and two orders of
# magnitude above the 0.05 N of a graze, so it is a hold and not a touch.
CUT_HOLD_N = 5.0

# How slowly the fingers have to be moving before the blade will cut — the
# gripper's own tendon velocity, in the actuator's units.
#
# ⚠️ **Force on both pads is not on its own a grip, and seed 23 is the proof.**
# The pads sweep in on an arc, and the first one to arrive *pushes the truss
# across into the second*, so both pads read load while they are still 55 mm
# apart and closing at full ramp speed. Cutting there dropped the cluster: the
# weld let go, and the only thing left holding 0.80 kg was a pair of pads still
# travelling, which squeezed it out sideways. Measured, per control cycle:
#
#     seed 23   pads first both loaded   19 / 29 N   gap 55.0 mm   |v| 0.52
#               fingers stalled          32 / 47 N   gap 47.3 mm   |v| 0.02
#     seed  7   pads first both loaded   33 / 23 N   gap 51.5 mm   |v| 0.28
#               fingers stalled          50 / 35 N   gap 45.6 mm   |v| 0.05
#
# The velocity separates them by more than an order of magnitude, because it is
# asking the one question that matters: have the pads *arrived*. A 2F85 that has
# taken up on a stem has stopped; one that is still on its way has not, whatever
# it is bouncing off on the way past. Both trusses stall at a 46 mm pad gap,
# which is what a 20 mm collar between these pads measures.
#
# 0.05 is a tenth of the ~0.5 the fingers travel at under the 1.5 s grip ramp,
# and three times the ~0.015 they settle to. A much slower ramp would narrow
# that margin — this is a stall test, and it assumes the fingers were moving.
CUT_STALL_V = 0.05

# How near the tool has to be to the collar for the cut to be *this* truss's.
CUT_REACH = 0.06

# How the *mirrored* arm has to grasp a truss. See `mission.ArmFrame.roll_sign`.
#
# ⚠️ **The 2F85 travels in OPEN, at a 93 mm span**, so each pad rides ~46 mm off
# the tool axis on the way in — and `fruit_offsets` always puts fruit 0 at
# +40 mm in y, 48 mm below the grasp point, with a 33 mm radius. Whether the pad
# on that side rides over the fruit or through it is the wrist roll.
#
# ⚠️ **And the two arms cannot reach the same rolls.** Measured as the height of
# the +y pad above the -y pad, driving each arm to a truss collar:
#
#     commanded roll   -90   -45     0   +45   +90
#     arm a  +y pad     -0    +1    +3   +65   +93     can lift it clear
#     arm b  +y pad     -7    -7    -7   -64   -92     never lifts at all
#
# Arm a, left free, rolls to ~43 deg and lifts its pad over fruit 0 — which is
# why every single-armed run crates and why this module asks nothing of it. Arm
# b is bolted round 180 deg (`trolley.ARM_YAW`) and its +y pad never goes
# positive at any roll, so it cannot copy arm a. Measured, that is the whole
# two-arm failure:
#
#     arm a   gr_right_pad1  hit COLLAR   tool  8 mm out   -> gripped
#     arm b   b_gr_left_pad1 hit fruit0   tool 34 mm out   -> shoved 850 mm
#
# So arm b grasps **level** — the best roll it can reach — and takes its
# clearance from aiming a little higher up the collar instead. The collar runs
# +/- 22.5 mm about the grasp point and the fruit's crown is 15 mm below its
# lower end, so 15 mm of lift keeps both pads on clean stem with 7 mm to spare.
#
# ⚠️ **Neither half works alone and both were measured.** Level roll on its own
# still put arm b's pad into fruit 0; the offset on its own cost a working seed
# a pick; the same roll mirrored onto both arms made the two-arm run worse than
# doing nothing. Together, on the mirrored arm only, the blade goes from 2/4 to
# 4/4 with no grasp failures.
#
# `roll_cost` is 0.5 against `mission.ROLL_COST_PINNED`'s 0.08, because 0.08
# does not hold a roll: at that cost a commanded roll came back as the free-roll
# answer. Position cost is 1.0 and still dominates, so the tool goes where sent.
MIRRORED_GRASP_ROLL = 0.0
MIRRORED_GRASP_OFFSET = np.array([0.0, 0.0, 0.015])
ROLL_COST = 0.5

# How high above the crate the tool holds the truss before opening the pads.
#
# ⚠️ **Derived, not chosen, and `mission.BIN_DROP_UP`'s 0.28 m is a loose
# fruit's number.** A tomato in the pads ends 33 mm below the tool; a truss
# hangs its whole rachis below the collar, so the bottom of the cluster is
# `RACHIS_LEN + FRUIT_R` = 271 mm down. At 0.28 m the bottom fruit sits 11 mm
# *below the crate floor* and the `carry` leg sweeps the cluster straight
# through the crate wall — measured, that knocked the truss out of the pads
# every time and `in_bin` came back False on picks the grip had not failed.
#
# So: clear the wall by the cluster's own length plus a margin. The margin is
# `mission.BIN_DROP_UP`'s own clearance over a loose fruit, kept the same rather
# than re-chosen.
CRATE_CLEAR = 0.04
CRATE_DROP_UP = RACHIS_LEN + FRUIT_R + BIN_WALL + CRATE_CLEAR      # 0.431 m

# How far a crate may be heaped above its rim and still be "the crate".
#
# ⚠️ Two trusses fill this one level — 320 mm across, 120 mm deep, against a
# cluster about 100 mm thick lying down — so a rim-height test calls every
# truss from the third on a miss, and reads as a machine that cannot crate
# rather than a crate that is full. Two clusters' worth of heap is the working
# allowance; past that the crate wants swapping, which this sim does not model.
# See `in_crate`, which spends it, and `drop_height`, which carries the tool
# over it.
CRATE_HEAP = 2 * (2 * FRUIT_R + FRUIT_DY)                          # 0.212 m


class Cutter:
    """The blade. **A truss is cut, not pulled**, and that is not a detail.

    ⚠️ **The pull-to-snap cycle does not survive a heavier crop, and this is
    the measurement that says so.** Weeks 1-4 detach a fruit by gripping it and
    driving the arm away until the stem gives: `plant_row.Row.snap` watches the
    weld and releases it above `snap_n`. That works on a 0.12 kg fruit at 12 N.
    Run it on a 0.80 kg truss and the arm loads a compliant weld to 75 N — the
    literature figure in `TRUSS_SNAP_N` — and when it lets go, all of that
    stored energy goes into the cluster at once. Measured through the real
    `week2_pick.execute`: the stem snapped at 75.2 N and 81.4 N on two trusses,
    both left the pads on release, and both were on the floor by `extract`
    (`grasp_failed`, 733 mm and 825 mm from the tool). The grip was never the
    problem — holding to 75 N of pull *is* the grip working.

    So this variant severs instead. Once the pads have closed on the collar the
    blade parts the peduncle above them, the weld goes inactive with no stored
    load, and the arm lifts a cluster that is simply resting in the gripper.
    That is what a TOV harvester does — `plant_row.SNAP_N`'s own note says the
    Vinea cradle-and-blade design "severs rather than pulls" and would remove
    the coupling between the detach threshold and the gripper model. This is
    that, for the crop that needs it.

    ⚠️ It cuts on **grip and proximity together**, never on grip alone. A pass
    that only checked the pads would sever whichever truss the arm happened to
    be near when it closed on another one.

    ⚠️ **And "grip" means the pads have arrived and taken up on the stem, which
    is three measurements and not one.** The blade fires when

        the tool is within `CUT_REACH` of the collar     — it is this truss
        both pads carry `CUT_HOLD_N` on the collar       — it is between them
        the fingers have stopped, `CUT_STALL_V`          — they have arrived

    and every clause has a dropped truss behind it. Cutting on the *commanded*
    closure fired with the pads 44 mm apart around a 20 mm collar. Cutting on
    pad force alone fired while the pads were still travelling and shoving the
    cluster from one side to the other. Both put 0.80 kg on the floor; both
    sometimes got away with it, which is worse, because a scooped truss lands in
    the crate and the shift scores it `clean`.
    """

    def __init__(self, model, data, row, name, gripper, reach=CUT_REACH,
                 hold_n=CUT_HOLD_N, stall_v=CUT_STALL_V, tool_site="tool0",
                 prefix=""):
        import mujoco

        from carrytrace import LEFT_PADS, RIGHT_PADS

        self.mj = mujoco
        self.model, self.data, self.row = model, data, row
        self.name = name
        self.gripper = gripper
        self.reach, self.hold_n, self.stall_v = reach, hold_n, stall_v
        self.site = model.site(prefix + tool_site).id
        self.eq = row.eq_id[name]
        self.bid = model.body(name).id
        # The collar is the truss's `{name}_geom` — see `add_trusses`, which
        # gives it that name precisely so the pad-force recorders find it.
        self.collar = model.geom(f"{name}_geom").id
        self.left = [model.geom(prefix + n).id for n in LEFT_PADS]
        self.right = [model.geom(prefix + n).id for n in RIGHT_PADS]
        self._buf = np.zeros(6)
        self.cut_at = None       # seconds, when the blade went through
        self.hold_at = None      # (left N, right N) the blade cut on

    def finger_speed(self):
        """How fast the tendon is still travelling. Zero means arrived."""
        return float(abs(self.data.actuator_velocity[self.gripper.index]))

    @property
    def cut(self):
        return self.cut_at is not None

    def pad_load(self):
        """Normal force each pad is putting on the collar, in newtons.

        Left and right separately, because the failure being guarded against is
        exactly one pad touching: a single loaded pad is not a grip, it is the
        gripper shoving the truss sideways on its way past.
        """
        f = [0.0, 0.0]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.collar not in (g1, g2):
                continue
            other = g2 if g1 == self.collar else g1
            if other in self.left:
                side = 0
            elif other in self.right:
                side = 1
            else:
                continue
            self.mj.mj_contactForce(self.model, self.data, i, self._buf)
            f[side] += abs(float(self._buf[0]))
        return f

    def tick(self, t=None):
        """Call every control cycle. Cheap, and idempotent once it has fired."""
        if self.cut:
            return
        tool = self.data.site_xpos[self.site]
        if float(np.linalg.norm(self.data.xpos[self.bid] - tool)) > self.reach:
            return
        load = self.pad_load()
        if min(load) < self.hold_n:
            return
        if self.finger_speed() > self.stall_v:
            return
        self.data.eq_active[self.eq] = 0
        self.cut_at = float(self.data.time)
        self.hold_at = tuple(load)


# --- did it land in the crate ------------------------------------------------

def in_crate(model, data, name, bin_pos, half=BIN_HALF, wall=BIN_WALL,
             need=0.5):
    """Is this truss in the crate? Scored on the **fruit**, not the body origin.

    ⚠️ **`fr5.crate_contains` reads one point, and for a truss that point is in
    the wrong place.** It is handed `data.body(name).xpos`, which for a loose
    tomato is the middle of the tomato and for a truss is the *grasp point* —
    the top of the collar, by construction (see `add_trusses`). A truss standing
    in the crate has its fruit on the crate floor and that origin 274 mm above
    them, so `pos[2] < crate_z + BIN_WALL` is false and a perfectly crated truss
    scored `in_bin: False`. Measured: it fails for every upright landing, and
    for a cluster lying flat it depends on which way the rachis happens to point
    — the origin is 238 mm from the fruit and the crate is 320 mm across.

    So ask the question of the thing that is actually in the crate. A truss
    counts when at least `need` of its fruit are inside, which is the same
    commercial unit the rest of this module works in: a cluster half out of the
    crate is not crated, and one fruit poking over the rim is not a miss.

    ⚠️ **"Inside" is the footprint and the floor, not the rim, because two
    trusses fill this crate to the rim and the third is not a failure.** The
    crate is 320 mm across and 120 mm deep; a cluster lying in it is about
    100 mm thick, and measured on seed 13 the *first* truss settled with its
    top fruit at z 0.391 against a rim at 0.386. Scored against the rim, every
    truss from the third on is a miss however neatly it lands — which says the
    machine cannot crate, when what is true is that the crate is full. A real
    TOV crate is heaped and swapped, and this sim has no crate swap.

    So the height test is "resting above the crate floor" rather than "below
    the rim", bounded by `CRATE_HEAP` so a cluster still in the gripper or
    perched on the crate wall cannot pass. The footprint test is what keeps a
    truss on the floor beside the crate out: its fruit sit at z 0.033, well
    below a crate floor at 0.266, and off to one side of it.
    """
    bin_pos = np.asarray(bin_pos, dtype=float)
    inside = 0
    total = 0
    for p in fruit_geom_pos(model, data, name):
        total += 1
        if (abs(p[0] - bin_pos[0]) < half and abs(p[1] - bin_pos[1]) < half
                and bin_pos[2] < p[2] < bin_pos[2] + wall + CRATE_HEAP):
            inside += 1
    return total > 0 and inside >= need * total


def fruit_geom_pos(model, data, name):
    """Where this truss's fruit are right now, in world coordinates."""
    out = []
    for k in range(FRUIT_PER_TRUSS):
        try:
            gid = model.geom(f"{name}_f{k}").id
        except KeyError:
            break
        out.append(data.geom_xpos[gid])
    return out


def crate_top(model, data, names, bin_pos, half=BIN_HALF):
    """How high the crate is filled, in world z. The rim if it is empty.

    Only fruit standing over the crate footprint count — a truss that ended up
    on the floor beside the crate is not in the way of the next one.
    """
    bin_pos = np.asarray(bin_pos, dtype=float)
    top = float(bin_pos[2] + BIN_WALL)
    for n in names:
        for p in fruit_geom_pos(model, data, n):
            if (abs(p[0] - bin_pos[0]) < half
                    and abs(p[1] - bin_pos[1]) < half):
                top = max(top, float(p[2]) + FRUIT_R)
    return top


def drop_height(model, data, names, bin_pos):
    """How far above the crate to hold the *tool* for the next release.

    ⚠️ **`CRATE_DROP_UP` clears an empty crate, and by the second truss of a
    stop the crate is not empty.** Measured on seed 11: the first cluster
    settled with its top fruit at z 0.391 against a rim at 0.386 — a couple of
    millimetres proud of it — and the second truss was carried in at a height
    chosen for the rim, struck it at 0.389, and was knocked out of the pads.
    Every pick after the first at a stop had the same 40 mm of clearance
    against a crate that was filling up.

    So the height is read off the crate rather than assumed: clear whatever is
    actually in it by a whole cluster plus `CRATE_CLEAR`. It grows as the crate
    fills, which is correct and is also the honest failure mode — a crate
    filled past the arm's reach makes the planner refuse the pick rather than
    quietly throwing trusses at it.
    """
    bin_pos = np.asarray(bin_pos, dtype=float)
    fill = crate_top(model, data, names, bin_pos) - float(bin_pos[2])
    return max(CRATE_DROP_UP, fill + RACHIS_LEN + FRUIT_R + CRATE_CLEAR)


# --- the cluster decision, from what the camera saw --------------------------

@dataclass
class Cluster:
    """One truss as the *robot* has it: a group of blobs and a verdict."""

    members: list = field(default_factory=list)     # scout.Sighting
    pos: np.ndarray = None                          # the estimated grasp point
    row: int = -1
    truth: object = None

    @property
    def n_fruit(self):
        return len(self.members)

    @property
    def n_red(self):
        return sum(1 for s in self.members if s.stage == "red")

    @property
    def red_fraction(self):
        return self.n_red / max(1, self.n_fruit)

    def ripe_at(self, threshold=RIPE_FRACTION):
        return self.red_fraction >= threshold

    @property
    def ripe(self):
        return self.ripe_at(RIPE_FRACTION)

    @property
    def stage(self):
        """A single label for the cluster, so it prints like a loose fruit."""
        return "red" if self.ripe else "turning"


def group(sightings, gap=CLUSTER_GAP):
    """Single-linkage grouping of detected fruit into trusses.

    ⚠️ **Position only — no truss is ever read out of the scene.** The whole
    claim of the cluster decision is that it is made from what the camera saw,
    so this groups the deprojected blob positions and nothing else. It works
    because the crop layout leaves a wide gap between "two fruit on one rachis"
    and "two fruit on different rachises"; see `CLUSTER_GAP`.

    Single-linkage rather than k-means because the number of trusses is not
    known — that is the thing being worked out — and because a chain of fruit
    down a rachis is exactly the shape single-linkage handles and k-means does
    not.
    """
    pts = [np.asarray(s.pos, float) for s in sightings]
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) <= gap:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(sightings[i])

    out = []
    for members in buckets.values():
        # Proximal fruit first — highest z is nearest the vine.
        members = sorted(members, key=lambda s: -s.pos[2])
        top = np.asarray(members[0].pos, float)
        # The grasp point is up the stem from the topmost fruit, by exactly the
        # offset the crop was built with. This is geometry the robot knows about
        # its own crop, not a position read out of the simulator.
        grasp = top + np.array([0.0, -np.sign(top[1] - np.mean(
            [m.pos[1] for m in members])) * FRUIT_DY, FRUIT_Z0])
        grasp[1] = float(np.mean([m.pos[1] for m in members]))
        out.append(Cluster(members=members, pos=grasp))
    out.sort(key=lambda c: c.pos[1])
    return out


def score(clusters, trusses, gate=0.20):
    """Match grouped clusters against the real trusses. Recall and errors."""
    pairs = sorted((float(np.linalg.norm(c.pos - t.pos)), i, j)
                   for i, c in enumerate(clusters)
                   for j, t in enumerate(trusses)
                   if np.linalg.norm(c.pos - t.pos) <= gate)
    took_c, took_t = set(), set()
    for _d, i, j in pairs:
        if i in took_c or j in took_t:
            continue
        took_c.add(i)
        took_t.add(j)
        clusters[i].truth = trusses[j]
    matched = [c for c in clusters if c.truth is not None]
    return {
        "found": len(matched), "truth": len(trusses),
        "phantom": len(clusters) - len(matched),
        "err_mm": [float(np.linalg.norm(c.pos - c.truth.pos) * 1000)
                   for c in matched] or [float("nan")],
        "n_right": sum(1 for c in matched if c.n_fruit == c.truth.n_fruit),
    }


def sweep(trusses, n_fruit=FRUIT_PER_TRUSS):
    """What each threshold would take, against the crop as it really is.

    ⚠️ **Scored on the ground truth on purpose.** This is asking what the
    *rule* is worth, not what the detector is worth; mixing the two would let a
    perception error be read as a badly-chosen threshold. The detector is scored
    separately by `score` above and by `trussrun.py --recall`.

    ⚠️ **The thresholds are exact `k/n` fractions, and the first version of this
    was wrong for want of that.** Sweeping round decimals puts the comparison on
    a float boundary: `5/6 = 0.8333` fails `>= 0.84`, so a row labelled 0.84 was
    silently reporting "all six red" and two rows of the table were duplicates
    of their neighbours. Every threshold is now the exact fraction that admits
    "at least k of n red", and `at_least` is what the table is keyed on.

    ⚠️ **`downgraded` is the column that decides this, not `green_taken`.** TOV
    is graded and sold as one item, so the cost of cutting a green fruit is
    borne by the *truss it was cut with* — one truss with three green fruit is
    one downgraded item, not three. Counting fruit overstates a concentrated
    mistake and understates a spread-out one.

    `red_left` is the other side: fruit already red on a truss the rule refused,
    which by the next pass is going soft on the plant.
    """
    rows = []
    for k in range(0, n_fruit + 1):
        th = k / n_fruit
        taken = [t for t in trusses if t.n_red >= k]
        left = [t for t in trusses if t.n_red < k]
        rows.append({
            "at_least": k,
            "threshold": th,
            "trusses": len(taken),
            "red_taken": sum(t.n_red for t in taken),
            "green_taken": sum(sum(1 for s in t.stages if s == "green")
                               for t in taken),
            "unripe_taken": sum(sum(1 for s in t.stages if s != "red")
                                for t in taken),
            "downgraded": sum(1 for t in taken
                              if any(s == "green" for s in t.stages)),
            "red_left": sum(t.n_red for t in left),
        })
    return rows


def summarise(trusses):
    """Counts, per stage and per cluster, and what each threshold would take."""
    by_stage = {n: 0 for n, _, _, _ in fcrop.STAGES}
    for t in trusses:
        for s in t.stages:
            by_stage[s] += 1
    dist = {k: 0 for k in range(FRUIT_PER_TRUSS + 1)}
    for t in trusses:
        dist[t.n_red] += 1
    return {"n": len(trusses), "fruit": sum(t.n_fruit for t in trusses),
            "by_stage": by_stage, "red_per_truss": dist,
            "ripe": sum(1 for t in trusses if t.ripe),
            "rows": sorted({t.row for t in trusses})}


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", action="store_true", help="what spawned")
    ap.add_argument("--sweep", action="store_true",
                    help="what each ripe-fraction threshold would take")
    ap.add_argument("--shot", action="store_true", help="write stills")
    ap.add_argument("--grip", action="store_true",
                    help="can the gripper hold 0.8 kg by the stem?")
    ap.add_argument("-n", type=int, default=5, help="trusses per row")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    print(__doc__)
    seed = fcrop.resolve_seed(args.seed, label="truss crop")
    trusses = spawn(n_per_row=args.n, seed=seed)
    s = summarise(trusses)

    print(f"\n  --- the house, this time ---")
    print(f"  {s['n']} trusses on {len(s['rows'])} rows, "
          f"{s['fruit']} fruit, {TRUSS_MASS} kg each")
    print(f"  {s['ripe']} trusses ripe at the shipped "
          f"threshold of {RIPE_FRACTION:.2f}")
    print(f"\n  {'stage':<10} {'fruit':>6}")
    for n, _rgba, _hue, _pick in fcrop.STAGES:
        print(f"  {n:<10} {s['by_stage'][n]:>6}")
    print(f"\n  red fruit per truss (this is what the threshold adjudicates)")
    for k, c in s["red_per_truss"].items():
        bar = "#" * c
        print(f"    {k}/{FRUIT_PER_TRUSS} red  {c:>3}  {bar}")

    if args.sweep:
        print(f"\n  --- what each threshold takes, over {s['n']} trusses ---")
        print(f"  ⚠️ Scored against the real crop, so this measures the RULE "
              f"and not the detector.\n")
        total_red = sum(t.n_red for t in trusses)
        print(f"  {'take at':>9} {'frac':>6} {'trusses':>8} {'red taken':>10} "
              f"{'of all red':>11} {'unripe':>7} {'green':>6} "
              f"{'downgraded':>11} {'red left':>9}")
        print("  " + "-" * 88)
        for r in sweep(trusses):
            mark = "  <-- ships" if abs(r["threshold"] - RIPE_FRACTION) < 1e-6 \
                else ""
            pct = 100.0 * r["red_taken"] / max(1, total_red)
            print(f"  {r['at_least']:>4} of {FRUIT_PER_TRUSS} "
                  f"{r['threshold']:>6.2f} {r['trusses']:>8} "
                  f"{r['red_taken']:>10} {pct:>10.0f}% {r['unripe_taken']:>7} "
                  f"{r['green_taken']:>6} {r['downgraded']:>11} "
                  f"{r['red_left']:>9}{mark}")
        print(f"\n  Read it as a trade, and read the RIGHT two columns.")
        print(f"\n  ⚠️ `unripe` is mostly not a cost. Commercial practice "
              f"harvests just after breaker\n     and the fruit colours in "
              f"transit, so a breaker or turning tomato cut with the\n     "
              f"truss is the normal product, not a mistake. **Green** is the "
              f"mistake — and its\n     unit is `downgraded`, the number of "
              f"trusses carrying at least one green fruit,\n     because TOV "
              f"is graded and sold one truss at a time.")
        print(f"\n  ⚠️ The other cost is `red left`: fruit already red on a "
              f"truss the rule refused.\n     By the next pass those are soft. "
              f"Waiting for the tip of the truss is paid for\n     by the base "
              f"of it.")
        return 0

    if args.stats:
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl" if (args.shot or args.grip)
                          else "glfw")
    import mujoco

    model = build(aisle=0, arms=("a",), trusses=trusses, seed=0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"\n  compiled: {model.ngeom} geoms, {model.nbody} bodies, "
          f"{model.nq} dof")
    m0 = float(model.body(trusses[0].name).mass[0])
    print(f"  one truss weighs {m0:.3f} kg "
          f"({FRUIT_PER_TRUSS} fruit at {FRUIT_MASS * 1000:.0f} g "
          f"+ {RACHIS_MASS * 1000:.0f} g of rachis)")

    if args.shot:
        import cv2

        out_dir = Path(__file__).resolve().parents[3]
        with mujoco.Renderer(model, height=960, width=1280) as r:
            for cam in ("house", "aisle"):
                r.update_scene(data, camera=cam)
                p = out_dir / f"farm_truss_{cam}.png"
                cv2.imwrite(str(p), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
                print(f"  wrote {p.name}")
        return 0

    import time

    import mujoco.viewer

    print("\n  window open — close it to quit")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
