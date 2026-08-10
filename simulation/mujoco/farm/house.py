#!/usr/bin/env python3
"""Four rows of a Venlo tomato house, at the dimensions a real one is built to.

`greenhouse.py` builds one working row and two more for the backdrop, around an
arm bolted to the floor. That was right for Weeks 1-4, where the question was
whether the arm could pick at all. It is the wrong room for a machine that has
to *drive*, because everything that decides a route — how far apart the rows
are, where the rails run, how long a bay is, which row you can reach from which
aisle — was scenery there and is the problem here.

⚠️ **This is a second scene, not a replacement.** Every number in the Week 1-4
README was measured in `greenhouse.build_scene`, and rebuilding that scene under
those numbers would invalidate them without a single test failing. `house.py`
sits beside it and shares its plant-drawing code; `greenhouse.py` is untouched.

--- what is real, and where it came from ------------------------------------

Dutch glasshouse tomato is grown on **high-wire**, over a **pipe-rail** system,
and the pipe rail is not a metaphor: the heating pipes *are* the rails every
trolley in the house runs on. That single fact settles most of the layout.

    heating pipe        51 mm outside diameter, 2.2 mm wall — the industry
                        standard, and the reason a trolley wheel is a grooved
                        roller rather than a tyre
    rail gauge          550 mm centre to centre. Trolleys are built to order
                        per house because the c.t.c. varies (415-550 mm is the
                        range suppliers quote); 550 is the common Venlo tomato
                        figure and what Berg Hortimotive's Benomic ships as.
    row pitch           1.60 m, path centre to path centre
    high wire           2.60 m
    gutter top          0.32 m

⚠️ **The 1.60 m row pitch is what forces a two-armed machine, and it is worth
following the arithmetic rather than taking it on trust.** A trolley sits in the
aisle, so its centreline is 800 mm from the crop on either side. With the 2F85
on the flange the FR5 reaches 1.100 m (`fr5.MAX_REACH_GRIPPER`), so 800 mm is
73% of reach — geometrically fine, and still the wrong place to work, because
every Week 1-4 number was measured at a **600 mm** standoff. Reach fraction is
exactly what `reach.why_not` blames when a pick fails, so moving the standoff
would put every one of those numbers back in question.

Mounting the arm 200 mm off the trolley centreline, toward the row it is
working, puts it back at 600 mm — every measured clearance, cycle time and
success rate carries over unchanged. And the 200 mm on the *other* side of the
centreline is then the mount for the second arm, working the opposite row. So
"leave room for a second arm" is not a nicety here: the pitch of a real Dutch
greenhouse is what makes a two-armed trolley the natural machine, and a
one-armed one a machine that drives every aisle twice.

    row 0        aisle        row 1        aisle        row 2 ...
    x=0.0     x=0.8         x=1.6       x=2.4        x=3.2
              |  |                      |  |
           rails at                  rails at
        0.8 +/- 0.275              2.4 +/- 0.275
              ^                       ^
       arm A at x=0.6           and so on: each aisle can work
       arm B at x=1.0           the row on either side of it

Run it:

    ./.venv/bin/python simulation/mujoco/farm/house.py            # a window
    ./.venv/bin/python simulation/mujoco/farm/house.py --shot     # stills
    ./.venv/bin/python simulation/mujoco/farm/house.py --plan     # the layout
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import greenhouse as gh  # noqa: E402
from greenhouse import _decor, _plant  # noqa: E402

# --- the house ---------------------------------------------------------------

N_ROWS = 4
ROW_PITCH = 1.60          # path centre to path centre, Venlo standard
ROW_X0 = 0.0              # the first crop row; the rest step by ROW_PITCH

# Half-length of the bay, in y. A real Venlo bay runs far longer than this — the
# houses are hectares — but the point of the number here is that it has to be
# long enough that **one trolley stop cannot work the whole row**, which is what
# makes routing a problem at all. At 4 m half-length the arm's 1.1 m working
# window in y needs roughly ten stops per row.
HOUSE_HALF_Y = 4.0

# --- the pipe rail -----------------------------------------------------------
#
# ⚠️ These two are the interface between the house and the machine, and getting
# either wrong means a trolley that does not fit the building it is sold into.
# Suppliers build each trolley to a measured c.t.c. for exactly this reason.
PIPE_D = 0.051            # 51 mm OD heating pipe — the industry standard
RAIL_CTC = 0.550          # centre to centre; the common Venlo tomato gauge
RAIL_Z = 0.10             # pipe centreline above the concrete

# Roof pane width along the row. A Venlo bay is glazed in panes about 1.0-1.2 m
# wide between aluminium glazing bars; 1.10 m is the common figure and it is what
# sets the visible rhythm of the roof from underneath. Drawing only, like every
# structural dimension in this file.
PANE_Y = 1.10

# Height of the concrete upstand the side glazing sits on. Real houses have one
# so the glass does not meet the floor, and the outermost gutter drains onto it.
UPSTAND_Z = 0.55

# Where the arm sits relative to the trolley's centreline. See the note above:
# this is what preserves the 600 mm standoff every Week 1-4 number was taken at.
ARM_OFFSET = 0.20
WORK_STANDOFF = ROW_PITCH / 2 - ARM_OFFSET      # 0.60 m, by construction


def row_x(i):
    """World x of crop row `i`."""
    return ROW_X0 + i * ROW_PITCH


def aisle_x(i):
    """World x of the aisle between rows `i` and `i+1` — where a trolley runs."""
    return row_x(i) + ROW_PITCH / 2


def aisles():
    """Every aisle a trolley can work from, as (index, x)."""
    return [(i, aisle_x(i)) for i in range(N_ROWS - 1)]


def serves(aisle_i):
    """Which crop rows a trolley in aisle `aisle_i` can reach: (left, right)."""
    return aisle_i, aisle_i + 1


def covering_aisles():
    """A minimal set of aisles whose two-armed trolleys cover every row.

    ⚠️ Every *other* aisle, not every aisle. A two-armed trolley in aisle 0
    works rows 0 and 1; one in aisle 2 works rows 2 and 3. Driving aisle 1 as
    well would re-work both middle rows, which is the mistake that turns a
    two-armed machine back into a one-armed one.
    """
    return [i for i in range(0, N_ROWS - 1, 2)]


# --- drawing -----------------------------------------------------------------

def _rails(spec, x, tag):
    """One aisle's pair of heating pipes, which are also its rails."""
    import mujoco

    body = spec.worldbody.add_body(name=f"fh_rails_{tag}", pos=[0, 0, 0])
    for side, sx in (("l", -1), ("r", 1)):
        _decor(body, f"fh_rail_{tag}_{side}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[x + sx * RAIL_CTC / 2, -HOUSE_HALF_Y - 0.6, RAIL_Z,
                       x + sx * RAIL_CTC / 2, HOUSE_HALF_Y + 0.6, RAIL_Z],
               size=[PIPE_D / 2, 0, 0], rgba=gh.PIPE_WHITE)
    # The bend that joins the pair at the headland. Real pipe rail is one
    # continuous run bent back on itself, which is why a trolley can be pushed
    # on at the end of a row rather than lifted on.
    for sy in (-1, 1):
        _decor(body, f"fh_railbend_{tag}_{'n' if sy < 0 else 'p'}",
               mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[x - RAIL_CTC / 2, sy * (HOUSE_HALF_Y + 0.6), RAIL_Z,
                       x + RAIL_CTC / 2, sy * (HOUSE_HALF_Y + 0.6), RAIL_Z],
               size=[PIPE_D / 2, 0, 0], rgba=gh.PIPE_WHITE)
    # Pipe-rail supports: the pressed-steel saddles the pipes sit in, every
    # 1.5 m. Without them the pipes read as floating tubes.
    for i, y in enumerate(np.arange(-HOUSE_HALF_Y, HOUSE_HALF_Y + 0.01, 1.5)):
        _decor(body, f"fh_railsup_{tag}_{i}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[x, y, RAIL_Z / 2], size=[RAIL_CTC / 2 + 0.03, 0.02,
                                             RAIL_Z / 2],
               rgba=gh.STEEL)
    return body


def _crop_row(spec, tag, x, leafy=True, pitch=0.32, seed=0):
    """Gutter, slabs, plants and wires for one row.

    `pitch` is the plant spacing. ⚠️ It is looser than `greenhouse.py`'s 0.16 m
    and that is a rendering decision, not a horticultural one: four rows over an
    8 m bay at true density is ~200 plants and ~3000 geoms of pure decor, which
    costs frame rate in a window whose whole purpose is being watched. The
    worked rows can be drawn at true density by passing pitch=0.16.
    """
    import mujoco

    body = spec.worldbody.add_body(name=f"fh_row_{tag}", pos=[x, 0, 0])

    for i, y in enumerate(np.arange(-HOUSE_HALF_Y, HOUSE_HALF_Y + 0.01, 1.2)):
        _decor(body, f"fh_gutterleg_{tag}_{i}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[0, y, 0, 0, y, gh.GUTTER_Z - 0.02], size=[0.012, 0, 0],
               rgba=gh.STEEL)

    _decor(body, f"fh_gutter_{tag}", mujoco.mjtGeom.mjGEOM_BOX,
           pos=[0, 0, gh.GUTTER_Z - 0.02], size=[0.11, HOUSE_HALF_Y, 0.02],
           rgba=gh.GUTTER_WHITE)
    for side, sy in (("l", -1), ("r", 1)):
        _decor(body, f"fh_gutterlip_{tag}_{side}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[sy * 0.10, 0, gh.GUTTER_Z + 0.01],
               size=[0.012, HOUSE_HALF_Y, 0.035], rgba=gh.GUTTER_WHITE)

    for i, y in enumerate(np.arange(-HOUSE_HALF_Y + 0.3, HOUSE_HALF_Y, 0.66)):
        _decor(body, f"fh_slab_{tag}_{i}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[0, y, gh.GUTTER_Z + 0.03], size=[0.075, 0.30, 0.035],
               rgba=gh.SLAB)

    _decor(body, f"fh_highwire_{tag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[0.04, -HOUSE_HALF_Y, gh.HIGH_WIRE_Z,
                   0.04, HOUSE_HALF_Y, gh.HIGH_WIRE_Z],
           size=[0.004, 0, 0], rgba=gh.STEEL)

    for i, y in enumerate(np.arange(-HOUSE_HALF_Y + 0.1, HOUSE_HALF_Y, pitch)):
        _plant(spec, f"{tag}{i}", x - gh.PLANT_X_OFFSET, float(y),
               leafy=leafy, lean=0.04 * (1 if i % 2 else -1), seed=seed + i)
    return body


def _structure(spec):
    """Posts, gutters, glazing bars and glass, sized to four rows."""
    import mujoco

    body = spec.worldbody.add_body(name="fh_structure", pos=[0, 0, 0])
    eaves, ridge = 3.10, 3.85
    x_lo = ROW_X0 - 1.60
    x_hi = row_x(N_ROWS - 1) + 1.60

    for i, y in enumerate(np.arange(-HOUSE_HALF_Y - 1.2, HOUSE_HALF_Y + 0.01,
                                    gh.POST_PITCH)):
        for j, x in enumerate(np.arange(x_lo, x_hi + 0.01, ROW_PITCH * 2)):
            _decor(body, f"fh_post_{i}_{j}", mujoco.mjtGeom.mjGEOM_BOX,
                   fromto=[x, y, 0, x, y, eaves], size=[0.035, 0.035, 0],
                   rgba=gh.STEEL)

    # Gutters along the eaves, one per bay boundary.
    for j, x in enumerate(np.arange(x_lo, x_hi + 0.01, ROW_PITCH * 2)):
        _decor(body, f"fh_eaves_{j}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[x, 0, eaves], size=[0.06, HOUSE_HALF_Y + 1.2, 0.05],
               rgba=gh.STEEL)

    # A Venlo roof is a run of small spans, ridge to gutter, per bay.
    for j, x in enumerate(np.arange(x_lo, x_hi - 0.01, ROW_PITCH * 2)):
        mid = x + ROW_PITCH
        _decor(body, f"fh_ridge_{j}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[mid, -HOUSE_HALF_Y - 1.2, ridge,
                       mid, HOUSE_HALF_Y + 1.2, ridge],
               size=[0.03, 0, 0], rgba=gh.STEEL)
        for side, sx in (("l", 0), ("r", 1)):
            x0 = x + sx * ROW_PITCH * 2
            _decor(body, f"fh_glass_{j}_{side}", mujoco.mjtGeom.mjGEOM_BOX,
                   fromto=[x0, 0, eaves, mid, 0, ridge],
                   size=[0.004, HOUSE_HALF_Y + 1.2, 0], rgba=gh.GLASS)
            # Glazing bars — the aluminium astragals the panes sit between,
            # running ridge-to-gutter every PANE_Y. They are the single most
            # recognisable thing about a Venlo roof from underneath, which is
            # the only angle any camera in this scene ever sees it from: a
            # continuous sheet of glass reads as a warehouse skylight, and a
            # barcode of thin bars reads as a glasshouse.
            for k, yy in enumerate(np.arange(-HOUSE_HALF_Y - 1.2,
                                             HOUSE_HALF_Y + 1.21, PANE_Y)):
                _decor(body, f"fh_bar_{j}_{side}_{k}",
                       mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[x0, float(yy), eaves, mid, float(yy), ridge],
                       size=[0.013, 0, 0], rgba=gh.STEEL)

    # Side walls. The gables close the ends of the bay (`_gables`); without
    # these the *sides* run out into open sky, which is visible in any wide
    # shot and in the deck camera whenever it faces the outermost row. Glass to
    # the eaves, on a post grid, with a solid concrete upstand at the bottom —
    # a real house has one, and it is what the outermost gutter drains onto.
    for side, wx in (("l", x_lo - 0.8), ("r", x_hi + 0.8)):
        _decor(body, f"fh_wall_{side}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[wx, 0, (UPSTAND_Z + eaves) / 2],
               size=[0.004, HOUSE_HALF_Y + 1.2, (eaves - UPSTAND_Z) / 2],
               rgba=gh.GLASS)
        _decor(body, f"fh_upstand_{side}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[wx, 0, UPSTAND_Z / 2],
               size=[0.06, HOUSE_HALF_Y + 1.2, UPSTAND_Z / 2],
               rgba=[0.72, 0.72, 0.70, 1.0])
        for k, yy in enumerate(np.arange(-HOUSE_HALF_Y - 1.2,
                                         HOUSE_HALF_Y + 1.21, PANE_Y)):
            _decor(body, f"fh_wallbar_{side}_{k}",
                   mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[wx, float(yy), UPSTAND_Z, wx, float(yy), eaves],
                   size=[0.016, 0, 0], rgba=gh.STEEL)

    # The energy-screen rail: a wire run just under the gutters that a thermal
    # screen is drawn along at night. Every Dutch house has one and it is the
    # thing that reads as "this is a climate-controlled building" rather than a
    # shed with plants in it.
    for j, x in enumerate(np.arange(x_lo, x_hi + 0.01, ROW_PITCH)):
        _decor(body, f"fh_screenwire_{j}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[float(x), -HOUSE_HALF_Y - 1.2, eaves - 0.12,
                       float(x), HOUSE_HALF_Y + 1.2, eaves - 0.12],
               size=[0.005, 0, 0], rgba=gh.STEEL)

    # ⚠️ No floor here. `_lighting` lays a textured plane sized to the whole
    # house; a box floor on top of it z-fights with the plane and reads as a
    # shimmering grey haze that moves with the camera.
    return body


def _lighting(spec):
    """Daylight through glass over a bay eight metres long.

    ⚠️ Not `greenhouse._lighting`. That one puts two lamps over a 3 m scene and
    a 9 x 9 m floor plane, which is right for one row and leaves this house a
    silhouette with its far end in the dark — the first render of it came back
    with the crop unreadable and the aisle fading to black at 4 m. Lights have
    to be spread down the bay, and the floor has to reach the end of it.

    Shadows stay off for the same reason they are off next door: a greenhouse
    roof is a barcode of rafters, and filming a crop through moving rafter
    shadows makes the fruit unreadable, which defeats the point of the capture.
    """
    import mujoco

    spec.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.62, 0.73, 0.85], rgb2=[0.88, 0.92, 0.96], width=512, height=512)
    spec.add_texture(
        name="concrete", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.74, 0.74, 0.72], rgb2=[0.79, 0.79, 0.77], width=512, height=512)
    # ⚠️ reflectance 0. At 0.05 the concrete mirrors the crop and the aisle
    # reads as a wet floor or a canal, which is distracting in every shot and
    # actively misleading in the deck camera, where a reflected tomato is a
    # red blob the detector has to be trusted to ignore.
    mat = spec.add_material(name="concrete", texrepeat=[26, 26],
                            reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "concrete"

    mid_x = (ROW_X0 + row_x(N_ROWS - 1)) / 2
    span = max(row_x(N_ROWS - 1) - ROW_X0, 2 * HOUSE_HALF_Y) / 2 + 3.0
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[mid_x, 0, 0], size=[span, span, 0.05], material="concrete")

    # Overhead daylight on a **grid**, two across the house by three along it,
    # so each pair of rows has a lamp roughly above it rather than all four rows
    # sharing a line of lamps down the middle. The old layout put every lamp at
    # `mid_x`, which lit the two inner rows well and the outer two obliquely —
    # visible as a darker far side of the aisle in the deck camera at pan 180.
    #
    # ⚠️ Six, not more: MuJoCo's default renderer takes a limited number of
    # lights and quietly ignores the rest, so "add one every metre" produces a
    # scene that is dark at one end for no visible reason.
    #
    # ⚠️ **And the obvious accompanying change — raising the ambient term to
    # model diffuse light through glass — makes detection strictly worse.** That
    # was measured, not assumed, because the reasoning for it was good: a Venlo
    # roof really does scatter light, and the ripeness bands have a minimum
    # *value* of 55-60, so shadowed fruit should drop out of them. What that
    # argument misses is that the same bands have a minimum **saturation** of
    # 95-100, and ambient light washes saturation out everywhere at once.
    # Three houses, both rows, 168 fruit:
    #
    #     lighting                          found      phantoms
    #     line of 4 + 2 side fills (old)    101/168        7
    #     grid of 6 + ambient 0.32           91/168       14     <- the "fix"
    #     grid of 6, ambient untouched      104/168        5     <- shipped
    #     grid of 6 + 2 side fills          102/168        6
    #
    # So the grid is kept and the ambient is not. Lifting the shadows cost ten
    # fruit and doubled the phantom rate.
    xs = np.linspace(ROW_X0 + ROW_PITCH / 2,
                     row_x(N_ROWS - 1) - ROW_PITCH / 2, 2)
    for i, y in enumerate(np.linspace(-HOUSE_HALF_Y + 1.0,
                                      HOUSE_HALF_Y - 1.0, 3)):
        for j, x in enumerate(xs):
            spec.worldbody.add_light(
                pos=[float(x), float(y), 3.7], dir=[0, 0, -1],
                diffuse=[0.50, 0.50, 0.48], specular=[0.04, 0.04, 0.04],
                castshadow=False)
    return spec


def _gables(spec):
    """The end walls. Without them the aisle runs out into open sky.

    Glass to the eaves with a solid kickboard at the bottom, which is how a
    Venlo gable is actually built, and — more to the point here — what stops
    the far end of every aisle camera being a bright rectangle of nothing.
    """
    import mujoco

    body = spec.worldbody.add_body(name="fh_gables", pos=[0, 0, 0])
    x_lo = ROW_X0 - 1.60
    x_hi = row_x(N_ROWS - 1) + 1.60
    mid_x = (x_lo + x_hi) / 2
    half_x = (x_hi - x_lo) / 2
    for tag, sy in (("n", -1), ("p", 1)):
        y = sy * (HOUSE_HALF_Y + 1.2)
        _decor(body, f"fh_gable_kick_{tag}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[mid_x, y, 0.35], size=[half_x, 0.02, 0.35],
               rgba=[0.82, 0.83, 0.84, 1.0])
        _decor(body, f"fh_gable_glass_{tag}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[mid_x, y, (0.70 + 3.10) / 2],
               size=[half_x, 0.008, (3.10 - 0.70) / 2], rgba=gh.GLASS)
        # Glazing bars, so the pane reads as a wall rather than as haze.
        for j, x in enumerate(np.arange(x_lo, x_hi + 0.01, 0.8)):
            _decor(body, f"fh_gablebar_{tag}_{j}", mujoco.mjtGeom.mjGEOM_BOX,
                   fromto=[x, y, 0.70, x, y, 3.10], size=[0.02, 0.03, 0],
                   rgba=gh.STEEL)
    return body


def _cameras(spec):
    """Fixed viewpoints. The trolley carries its own; these are for watching.

    ⚠️ **Every one of these lives inside the gables.** The first version put the
    aisle camera 1.8 m beyond the end wall, so the shot was taken *through* the
    gable and a glazing bar stood dead centre in every frame. The house is a
    closed building now, and a camera outside it sees the outside of it.
    """
    mid_x = (ROW_X0 + row_x(N_ROWS - 1)) / 2
    inside = HOUSE_HALF_Y + 0.9        # just inside the gable at +/- HALF_Y+1.2

    def look(pos, at):
        return gh._look_at(pos, at)

    # A corner establishing shot, high, looking down the length of the house.
    p = [ROW_X0 - 1.25, -inside, 3.05]
    spec.worldbody.add_camera(name="house", pos=p,
                              xyaxes=look(p, [mid_x, 0.6, 0.9]))
    # Down the first working aisle, at about a picker's eye height.
    p = [aisle_x(0), -inside, 1.55]
    spec.worldbody.add_camera(name="aisle", pos=p,
                              xyaxes=look(p, [aisle_x(0), 0.0, 0.85]))
    # Straight down. This is the frame the route map is drawn against, so it is
    # square to the house on purpose: +x right, +y up, no perspective tricks.
    spec.worldbody.add_camera(
        name="overhead", pos=[mid_x, 0.0, 9.0], xyaxes=[1, 0, 0, 0, 1, 0])
    return spec


def add_house(spec, leafy=True, pitch=0.32, worked_rows=(0, 1), seed=0):
    """Put the four-row house into a spec. Call before `spec.compile()`.

    `worked_rows` are drawn at true plant density; the rest are drawn loose,
    because they are backdrop and the geom budget is better spent where the
    cameras are pointed.
    """
    _lighting(spec)
    _structure(spec)
    _gables(spec)
    for i in range(N_ROWS):
        _crop_row(spec, f"r{i}", row_x(i), leafy=leafy,
                  pitch=0.16 if i in worked_rows else pitch,
                  seed=seed + 100 * i)
    for i, x in aisles():
        _rails(spec, x, f"a{i}")
    _cameras(spec)
    return spec


def build_house(leafy=True, pitch=0.32, worked_rows=(0, 1), seed=0):
    """A compiled house with nothing in it — no arm, no trolley, no fruit."""
    import mujoco

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    # ⚠️ The offscreen buffer defaults to 640x480 and every camera panel in this
    # package renders at 1280x960. `fr5.build_fr5_spec` sets this for the Week
    # 1-4 scenes; a house built without the arm has to set it itself or the
    # first `Renderer` call raises about the framebuffer rather than about the
    # thing you were doing.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    add_house(spec, leafy=leafy, pitch=pitch, worked_rows=worked_rows,
              seed=seed)
    return spec.compile()


# --- what it is, in numbers --------------------------------------------------

def describe():
    """The layout, as the arithmetic that produced it."""
    from fr5 import MAX_REACH_GRIPPER

    out = [f"\n  --- four-row Venlo house ---",
           f"  bay length        {2 * HOUSE_HALF_Y:.1f} m",
           f"  row pitch         {ROW_PITCH:.2f} m   (path centre to centre)",
           f"  heating pipe      {PIPE_D * 1000:.0f} mm OD, the rail itself",
           f"  rail gauge        {RAIL_CTC * 1000:.0f} mm c.t.c.",
           f"", f"  {'row':<6} {'x':>7}      {'aisle':<7} {'x':>7} {'serves':>10}"]
    for i in range(N_ROWS):
        line = f"  {'r' + str(i):<6} {row_x(i):>7.2f}"
        if i < N_ROWS - 1:
            l, r = serves(i)
            line += f"      {'a' + str(i):<7} {aisle_x(i):>7.2f} " \
                    f"{'r' + str(l) + ' r' + str(r):>10}"
        out.append(line)

    centre = ROW_PITCH / 2
    out += [
        f"",
        f"  from an aisle centreline the crop is {centre * 1000:.0f} mm away,",
        f"  which is {100 * centre / MAX_REACH_GRIPPER:.0f}% of the FR5's "
        f"{MAX_REACH_GRIPPER:.3f} m reach — workable, and not where",
        f"  any Week 1-4 number was measured.",
        f"  mounting the arm {ARM_OFFSET * 1000:.0f} mm off centre puts it at "
        f"{WORK_STANDOFF * 1000:.0f} mm, which is exactly where they were.",
        f"  the other {ARM_OFFSET * 1000:.0f} mm is the second arm's mount, "
        f"working the opposite row.",
        f"",
        f"  covering the house takes aisles "
        f"{', '.join('a' + str(i) for i in covering_aisles())} "
        f"— every other one, two arms each.",
    ]
    return "\n".join(out)


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot", action="store_true", help="write stills, no window")
    ap.add_argument("--plan", action="store_true", help="print the layout only")
    ap.add_argument("--pitch", type=float, default=0.32)
    ap.add_argument("--no-leaves", action="store_true")
    args = ap.parse_args()

    print(__doc__)
    print(describe())
    if args.plan:
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl" if args.shot else "glfw")
    import mujoco

    model = build_house(leafy=not args.no_leaves, pitch=args.pitch)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"\n  compiled: {model.ngeom} geoms, {model.nbody} bodies, "
          f"{model.nq} dof")

    if args.shot:
        import cv2

        out_dir = Path(__file__).resolve().parents[3]
        with mujoco.Renderer(model, height=960, width=1280) as r:
            for cam in ("house", "aisle", "overhead"):
                r.update_scene(data, camera=cam)
                p = out_dir / f"farm_house_{cam}.png"
                cv2.imwrite(str(p), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
                print(f"  wrote {p.name}")
        return 0

    import mujoco.viewer

    print("\n  window open — orbit it. close it to quit")
    with mujoco.viewer.launch_passive(model, data) as v:
        import time

        while v.is_running():
            v.sync()
            time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
