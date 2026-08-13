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

# How closed the pads have to be, on Robotiq's 0-255 scale, before the blade
# will cut. Well past the point where the pads have met a 20 mm stem, so the cut
# cannot happen while the gripper is still travelling.
CUT_CLOSED = 0.55 * 255.0

# How near the tool has to be to the collar for the cut to be *this* truss's.
CUT_REACH = 0.06


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
    """

    def __init__(self, model, data, row, name, gripper, reach=CUT_REACH,
                 closed=CUT_CLOSED, tool_site="tool0"):
        self.model, self.data, self.row = model, data, row
        self.name = name
        self.gripper = gripper
        self.reach, self.closed = reach, closed
        self.site = model.site(tool_site).id
        self.eq = row.eq_id[name]
        self.bid = model.body(name).id
        self.cut_at = None       # seconds, when the blade went through

    @property
    def cut(self):
        return self.cut_at is not None

    def tick(self, t=None):
        """Call every control cycle. Cheap, and idempotent once it has fired."""
        if self.cut:
            return
        if float(self.data.ctrl[self.gripper.index]) < self.closed:
            return
        tool = self.data.site_xpos[self.site]
        if float(np.linalg.norm(self.data.xpos[self.bid] - tool)) > self.reach:
            return
        self.data.eq_active[self.eq] = 0
        self.cut_at = float(self.data.time)


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
