#!/usr/bin/env python3
"""Put the arm in a Dutch greenhouse instead of on a grey floor next to a board.

Week 1 and the first half of Week 2 ran against a green rectangle standing in
for a plant row. That was the right call while the question was "can the arm
reach and grip"; it is the wrong backdrop for a recorded demo, because the
thing the demo has to make believable is that this machine works **in the rows
that already exist** — which is the entire product claim.

So this builds the room around the row: a substrate gutter, plants going up
strings to a high wire, leaves, the heating pipes that double as the harvesting
trolley's rails, a concrete path, glazed roof and gable. The layout follows a
Venlo-type high-wire tomato house:

    row spacing        1.60 m   (path centre to path centre)
    gutter top         0.32 m
    high wire          2.60 m
    heating rail gauge 0.52 m   (the pipes are the rail — no separate track)

The fruit sit at z = 0.54-0.72, which is low on the plant, and that is correct
rather than convenient: in high-wire tomato the ripe trusses are the *lowest*
ones and the picker works the bottom of the canopy. The rest of the plant is
out of this arm's band and the README says so instead of pretending otherwise.

⚠️ **All of this is scenery, and deliberately so.** Every geom added here
carries `contype=0, conaffinity=0` — it is drawn and it cannot be hit. The
collision-relevant geometry is exactly what it was before: the fruit, the
support bar and the row panel from `plant_row.py`, plus the floor and the
crate. That is not laziness. Week 2's contact behaviour is tuned against those
surfaces and those numbers are in the README; quietly adding a dozen collidable
leaves would change every one of them and the change would be invisible in the
diff. If a leaf ever needs to be a real obstacle it should be added to
`plant_row` with the measurement to justify it, not smuggled in with the decor.

    from greenhouse import build_scene
    model = build_scene()                    # greenhouse
    model = build_scene(greenhouse=False)    # the old bare rig

Run it directly to render a still from each camera:

    ./.venv/bin/python simulation/mujoco/greenhouse.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fr5 import add_crate, build_fr5_spec  # noqa: E402
from plant_row import ROW_X, SUPPORT_Z, add_row  # noqa: E402

# --- the house ---------------------------------------------------------------
ROW_PITCH = 1.60        # centre to centre; the arm works the row at ROW_X
GUTTER_Z = 0.32         # top of the substrate gutter
HIGH_WIRE_Z = 2.60      # where the strings are clipped
BAY_Y = 3.2             # half-length of the visible run of row, in y
RAIL_GAUGE = 0.52       # heating-pipe centres — these are the trolley's rails
RAIL_Z = 0.10
POST_PITCH = 2.4        # roof post spacing along y

# No crop and no structure is drawn behind this line in y. Every camera lives
# between y = -1.85 and -2.60 looking back along +y, so anything further out
# than this is not scenery — it is an object in front of the lens.
CAMERA_CLEAR = -0.55

# Where the plants stand. Spaced 160 mm along the row, which is about a real
# stem density (2.5 stems/m²) at this row pitch. The trusses from `plant_row`
# hang at y = 0.26 .. -0.24, so the working plants are the middle four.
PLANT_Y = [round(-0.40 + 0.16 * i, 3) for i in range(6)]

# How far *behind* the fruit the stems stand.
#
# ⚠️ Not cosmetic drift — without it the capture is unusable. The trusses hang
# at x = ROW_X and the plants were drawn at x = ROW_X too, so the stems rendered
# in the same plane as the fruit and the row came out as five red spheres behind
# a picket fence. Setting the fruit in front of the stems is also what a real
# truss does: it hangs out of the canopy toward the aisle, which is the entire
# reason a machine can reach one.
PLANT_X_OFFSET = 0.055

LEAF_GREEN = [0.19, 0.42, 0.20, 1.0]
LEAF_DARK = [0.13, 0.31, 0.15, 1.0]
STEM_GREEN = [0.28, 0.46, 0.24, 1.0]
STRING_WHITE = [0.88, 0.88, 0.84, 1.0]
STEEL = [0.62, 0.65, 0.70, 1.0]
PIPE_WHITE = [0.86, 0.87, 0.88, 1.0]
GUTTER_WHITE = [0.90, 0.91, 0.92, 1.0]
SLAB = [0.79, 0.80, 0.76, 1.0]
GLASS = [0.72, 0.82, 0.88, 0.13]


def _decor(body, name, gtype, **kw):
    """Add a geom that is drawn and cannot be collided with.

    Every geom in this module goes through here, so the invariant in the module
    docstring — greenhouse geometry never changes the physics — is enforced in
    one place rather than by remembering to type two keywords twenty times.
    """
    kw.setdefault("contype", 0)
    kw.setdefault("conaffinity", 0)
    return body.add_geom(name=name, type=gtype, **kw)


# Leaf geometry. ⚠️ Both of these started much larger and it was the single
# worst thing about the first render: a 340 mm leaf held flat reads as a lily
# pad, and six plants of them stack into a pagoda that hides the entire crop.
# A real tomato leaf is that long, but it is *compound and drooping* — it hangs
# at 30-45° and its outline is broken up by leaflets. Modelling one leaf as
# three small drooping ellipsoids costs two extra geoms and looks like a plant.
LEAF_LEN = 0.105
LEAF_DROOP = 0.62       # radians below horizontal
LEAF_SPACING = 0.17
LEAF_FROM_Z = 0.86      # nothing below this: the bottom of the plant is deleafed
GOLDEN_ANGLE = 2.39996  # tomato phyllotaxis is close enough to a golden spiral


def _leaflet(body, name, base, azimuth, droop, length, rgba, width=0.42):
    """One drooping leaflet, pointing away from the stem along `azimuth`.

    Placed with an explicit quaternion rather than `euler`. MuJoCo's euler
    sequence is a compiler option, so a triple that looks right today silently
    means something else if `eulerseq` ever changes — and a leaf that rotates
    90° about the wrong axis reads as a bug in the plant, not in the convention.
    """
    import mujoco

    d = np.array([np.cos(azimuth) * np.cos(droop),
                  np.sin(azimuth) * np.cos(droop),
                  -np.sin(droop)])
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(up, d)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        quat, np.column_stack([d, side, np.cross(d, side)]).flatten())

    return _decor(body, name, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  pos=(np.asarray(base) + d * length * 0.55).tolist(),
                  size=[length, length * width, 0.0055],
                  quat=quat.tolist(), rgba=rgba)


def _plant(spec, tag, x, y, top=HIGH_WIRE_Z, leafy=True, lean=0.04, seed=0):
    """One tomato plant: string, stem up to the high wire, leaves, a head.

    The stem leans slightly along the row, which is what a high-wire crop
    actually does — the plants are lowered and leaned as they grow — and it is
    also what stops a row of identical vertical sticks reading as a fence.
    """
    import mujoco

    rng = np.random.default_rng(abs(hash((tag, seed))) % (2 ** 32))
    body = spec.worldbody.add_body(name=f"gh_plant_{tag}", pos=[x, y, 0])

    # The support string, clipped to the high wire and running down to the slab.
    _decor(body, f"gh_string_{tag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[lean, 0.02, top, 0, 0.02, GUTTER_Z], size=[0.0018, 0, 0],
           rgba=STRING_WHITE)

    # The stem, in two leaning segments so it is not a perfect line.
    mid = GUTTER_Z + (top - GUTTER_Z) * 0.55
    _decor(body, f"gh_stem_{tag}_lo", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[0, 0, GUTTER_Z, lean * 0.6, 0, mid], size=[0.0085, 0, 0],
           rgba=STEM_GREEN)
    _decor(body, f"gh_stem_{tag}_hi", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[lean * 0.6, 0, mid, lean, 0, top], size=[0.0072, 0, 0],
           rgba=STEM_GREEN)

    if not leafy:
        return body

    # Leaves on a golden spiral up the stem, none below LEAF_FROM_Z: the bottom
    # of a high-wire plant is deleafed to the ripening truss, which is exactly
    # why an arm can reach in there at all.
    # Stagger the first leaf per plant. Starting every plant at the same height
    # puts every leaf on a shared tier and the row renders as horizontal bands,
    # which is the tell that a crop was generated rather than grown.
    z, i = LEAF_FROM_Z + rng.uniform(0.0, LEAF_SPACING), 0
    while z < top - 0.12:
        frac = (z - GUTTER_Z) / (top - GUTTER_Z)
        base = np.array([lean * frac, 0.0, z])
        az = i * GOLDEN_ANGLE + rng.uniform(-0.15, 0.15)
        droop = LEAF_DROOP + rng.uniform(-0.12, 0.12)
        rgba = LEAF_GREEN if i % 2 else LEAF_DARK
        # Petiole plus two leaflets fanned off it — the broken outline is what
        # stops it reading as a disc.
        _leaflet(body, f"gh_leaf_{tag}_{i}a", base, az, droop, LEAF_LEN, rgba)
        _leaflet(body, f"gh_leaf_{tag}_{i}b",
                 base + np.array([0, 0, -0.012]), az + 0.42, droop + 0.18,
                 LEAF_LEN * 0.72, rgba)
        _leaflet(body, f"gh_leaf_{tag}_{i}c",
                 base + np.array([0, 0, -0.012]), az - 0.42, droop + 0.18,
                 LEAF_LEN * 0.72, rgba)
        z += LEAF_SPACING
        i += 1

    # The growing head, above the last leaf.
    _decor(body, f"gh_head_{tag}", mujoco.mjtGeom.mjGEOM_ELLIPSOID,
           pos=[lean, 0, top - 0.05], size=[0.055, 0.04, 0.06],
           rgba=LEAF_GREEN)
    return body


def _decor_truss(body, tag, y, z, count=5, ripe=False, spread=0.055):
    """A truss of fruit that is scenery — nothing here is pickable.

    Green ones on the neighbouring rows and either side of the working band,
    because a row with five red tomatoes and nothing else on it does not look
    like a crop. They are also the honest thing to draw: at any moment most of
    a tomato plant's fruit is unripe, and Week 3's detector has to tell the
    difference. Rendering only ripe fruit would quietly make that problem look
    easier than it is.
    """
    import mujoco

    rgba = [0.80, 0.15, 0.11, 1.0] if ripe else [0.42, 0.60, 0.24, 1.0]
    _decor(body, f"gh_truss_{tag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[0.02, y, z + 0.10, -0.01, y, z + 0.02], size=[0.0035, 0, 0],
           rgba=STEM_GREEN)
    for i in range(count):
        a = i * GOLDEN_ANGLE
        _decor(body, f"gh_fruit_{tag}_{i}", mujoco.mjtGeom.mjGEOM_SPHERE,
               pos=[-0.01 + spread * 0.45 * np.cos(a),
                    y + spread * np.sin(a),
                    z - 0.012 * i],
               size=[0.027, 0, 0], rgba=rgba)


def _crop_row(spec, tag, x, leafy=True):
    """A whole row: gutter, slabs, plants, and the truss-support wire."""
    import mujoco

    body = spec.worldbody.add_body(name=f"gh_row_{tag}", pos=[x, 0, 0])

    # Legs holding the gutter up off the floor.
    for i, y in enumerate(np.arange(-BAY_Y, BAY_Y + 0.01, 1.2)):
        _decor(body, f"gh_gutterleg_{tag}_{i}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[0, y, 0, 0, y, GUTTER_Z - 0.02], size=[0.012, 0, 0],
               rgba=STEEL)

    # The gutter itself — a white trough running the length of the bay.
    _decor(body, f"gh_gutter_{tag}", mujoco.mjtGeom.mjGEOM_BOX,
           pos=[0, 0, GUTTER_Z - 0.02], size=[0.11, BAY_Y, 0.02],
           rgba=GUTTER_WHITE)
    for side, sy in (("l", -1), ("r", 1)):
        _decor(body, f"gh_gutterlip_{tag}_{side}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[sy * 0.10, 0, GUTTER_Z + 0.01], size=[0.012, BAY_Y, 0.035],
               rgba=GUTTER_WHITE)

    # Rockwool slabs sitting in the gutter, one per pair of plants.
    for i, y in enumerate(np.arange(-BAY_Y + 0.3, BAY_Y, 0.66)):
        _decor(body, f"gh_slab_{tag}_{i}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[0, y, GUTTER_Z + 0.03], size=[0.075, 0.30, 0.035],
               rgba=SLAB)

    # The high wire, and the truss-support wire the fruit actually hang from.
    _decor(body, f"gh_highwire_{tag}", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[0.04, -BAY_Y, HIGH_WIRE_Z, 0.04, BAY_Y, HIGH_WIRE_Z],
           size=[0.004, 0, 0], rgba=STEEL)

    # Plants. The bay is long; only the stretch in front of the arm needs the
    # full leaf treatment, and every leaf costs render time on 30 fps captures.
    #
    # ⚠️ Nothing is planted at y < CAMERA_CLEAR. Every camera in this scene sits
    # at y = -1.85 to -2.60 and looks back along +y, so a plant at y = -3.0 is
    # not background, it is 600 mm in front of the lens. The first render put
    # nineteen of them there and the arm was filmed through a hedge.
    # ⚠️ `x`, not 0. `_plant` hangs its body off `spec.worldbody`, not off the
    # row body created above, so the coordinate it takes is a world one. Passing
    # 0 put all three rows' plants on top of each other at the origin — which is
    # where the arm is — and the first render was the robot filmed from inside a
    # bush. Bodies that look nested here are not: only `_decor` geoms are.
    px = x + PLANT_X_OFFSET
    for i, y in enumerate(PLANT_Y):
        _plant(spec, f"{tag}{i}", px, y, leafy=leafy, seed=i)
    for i, y in enumerate(np.arange(CAMERA_CLEAR, BAY_Y, 0.42)):
        if -0.5 < y < 0.5:
            continue                      # already planted, in detail, above
        _plant(spec, f"{tag}b{i}", px, float(y), leafy=False, seed=100 + i)

    # Unripe trusses down the row, and a leafy backdrop for the working band.
    for i, y in enumerate(np.arange(CAMERA_CLEAR + 0.2, BAY_Y, 0.42)):
        if -0.42 < y < 0.42:
            continue                      # the pickable fruit live here
        _decor_truss(body, f"{tag}{i}", float(y), 0.55 + 0.16 * (i % 3))

    return body


def _backdrop(spec):
    """Foliage between the fruit and the row panel, so the panel stops reading
    as a painted board.

    The panel has to stay: it is the collision backstop that makes an approach
    which would have gone through the plant fail visibly instead of silently,
    and it is 80 mm behind the fruit where the gripper needs the room. What it
    does not have to be is *visible*. Leaves at x = ROW_X + 0.06, fanned along
    the row rather than out toward the camera, cover it without covering the
    fruit the capture exists to show.
    """
    body = spec.worldbody.add_body(name="gh_backdrop", pos=[0, 0, 0])
    rng = np.random.default_rng(7)
    i = 0
    for y in np.arange(-0.42, 0.43, 0.105):
        for z in (0.46, 0.60, 0.74, 0.86):
            az = (np.pi / 2 if i % 2 else -np.pi / 2) + rng.uniform(-0.5, 0.5)
            _leaflet(body, f"gh_bd_{i}", [ROW_X + 0.062, float(y), float(z)],
                     az, 0.30 + rng.uniform(-0.1, 0.2), 0.085,
                     LEAF_GREEN if i % 3 else LEAF_DARK)
            i += 1
    return body


def _rails(spec):
    """Heating pipes. In a Dutch house these *are* the harvesting trolley's rail.

    Worth drawing rather than assuming: the pipe gauge is the reason a harvester
    can run in an existing greenhouse without a rebuild, and it is the single
    strongest argument in the whole product story. A render that shows the robot
    standing on the rails makes it in one frame.
    """
    import mujoco

    body = spec.worldbody.add_body(name="gh_rails", pos=[0, 0, 0])
    for side, sy in (("l", -1), ("r", 1)):
        y = sy * RAIL_GAUGE / 2
        _decor(body, f"gh_rail_{side}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[-0.9, y, RAIL_Z, ROW_X + 0.9, y, RAIL_Z],
               size=[0.026, 0, 0], rgba=PIPE_WHITE)
    # The bend that joins them at the end of the row.
    _decor(body, "gh_rail_bend", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[ROW_X + 0.9, -RAIL_GAUGE / 2, RAIL_Z,
                   ROW_X + 0.9, RAIL_GAUGE / 2, RAIL_Z],
           size=[0.026, 0, 0], rgba=PIPE_WHITE)
    for i, x in enumerate(np.arange(-0.7, ROW_X + 0.9, 0.8)):
        _decor(body, f"gh_railtie_{i}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[float(x), 0, RAIL_Z - 0.03], size=[0.02, RAIL_GAUGE / 2, 0.008],
               rgba=STEEL)
    return body


def _structure(spec):
    """Posts, gutters, glazing bars and glass. The room, not the crop."""
    import mujoco

    body = spec.worldbody.add_body(name="gh_structure", pos=[0, 0, 0])
    eaves, ridge = 3.10, 3.85
    # ⚠️ The bay has to be wider than the crop it holds. At x_lo = -1.30 a roof
    # post stood 350 mm from the wide camera and filled a third of the frame
    # with a dark bar. A Venlo bay is 8-9.6 m anyway, so widening it is also the
    # more accurate number.
    x_lo, x_hi = -2.20, ROW_X + 2 * ROW_PITCH + 0.60

    for i, y in enumerate(np.arange(CAMERA_CLEAR - POST_PITCH, BAY_Y + 0.01,
                                    POST_PITCH)):
        for j, x in enumerate((x_lo, x_hi)):
            _decor(body, f"gh_post_{i}_{j}", mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[float(x), float(y), eaves / 2],
                   size=[0.045, 0.045, eaves / 2], rgba=STEEL)
        # Truss across the bay, and the two rafters up to the ridge.
        _decor(body, f"gh_truss_{i}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[x_lo, float(y), eaves, x_hi, float(y), eaves],
               size=[0.03, 0, 0], rgba=STEEL)
        mid = (x_lo + x_hi) / 2
        _decor(body, f"gh_rafter_{i}_a", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[x_lo, float(y), eaves, mid, float(y), ridge],
               size=[0.022, 0, 0], rgba=STEEL)
        _decor(body, f"gh_rafter_{i}_b", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[mid, float(y), ridge, x_hi, float(y), eaves],
               size=[0.022, 0, 0], rgba=STEEL)

    # Eaves gutters and the ridge, running the length of the bay.
    for j, x in enumerate((x_lo, x_hi)):
        _decor(body, f"gh_eave_{j}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[float(x), 0, eaves + 0.03], size=[0.07, BAY_Y, 0.035],
               rgba=STEEL)
    mid = (x_lo + x_hi) / 2
    _decor(body, "gh_ridge", mujoco.mjtGeom.mjGEOM_CAPSULE,
           fromto=[mid, -BAY_Y, ridge, mid, BAY_Y, ridge],
           size=[0.03, 0, 0], rgba=STEEL)

    # Glazing. Two sloped panes and the two side walls. Alpha is low — glass
    # you can see the sky through, that does not fog the crop behind it.
    span = mid - x_lo
    pitch = np.arctan2(ridge - eaves, span)
    for j, (x0, sign) in enumerate(((x_lo, 1), (x_hi, -1))):
        _decor(body, f"gh_glass_{j}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[x0 + sign * span / 2, 0, (eaves + ridge) / 2],
               size=[np.hypot(span, ridge - eaves) / 2, BAY_Y, 0.006],
               euler=[0, float(-sign * pitch), 0], rgba=GLASS)
    for j, x in enumerate((x_lo, x_hi)):
        _decor(body, f"gh_wall_{j}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[float(x), 0, eaves / 2], size=[0.006, BAY_Y, eaves / 2],
               rgba=GLASS)

    # SON-T fixtures over the path, on the truss line.
    for i, y in enumerate(np.arange(-BAY_Y + 1.0, BAY_Y, 2.2)):
        _decor(body, f"gh_lamp_{i}", mujoco.mjtGeom.mjGEOM_BOX,
               pos=[ROW_X - 0.35, float(y), eaves - 0.42],
               size=[0.10, 0.06, 0.05], rgba=[0.85, 0.86, 0.88, 1.0])
        _decor(body, f"gh_lampstem_{i}", mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[ROW_X - 0.35, float(y), eaves - 0.37,
                       ROW_X - 0.35, float(y), eaves],
               size=[0.006, 0, 0], rgba=STEEL)
    return body


def _lighting(spec):
    """Daylight through glass, plus enough fill that the row is not a silhouette.

    ⚠️ Shadows off. A greenhouse is thirty-odd thin capsules and a pane of glass
    above the working area, and with shadows on the crop is filmed through a
    moving barcode of rafter shadows — the fruit stop being readable, which
    matters because the whole point of the capture is watching a red sphere get
    picked. The floor keeps a little reflectance, so depth still reads.
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
    mat = spec.add_material(name="concrete", texrepeat=[14, 14], reflectance=0.08)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "concrete"

    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[9, 9, 0.05],
        material="concrete")

    spec.worldbody.add_light(pos=[0.4, 0.0, 3.6], dir=[0, 0, -1],
                             diffuse=[0.55, 0.55, 0.52], castshadow=False)
    spec.worldbody.add_light(pos=[-1.6, -2.2, 2.8], dir=[0.5, 0.6, -1],
                             diffuse=[0.38, 0.40, 0.42], castshadow=False)
    spec.worldbody.add_light(pos=[2.4, 1.8, 2.6], dir=[-0.6, -0.5, -1],
                             diffuse=[0.26, 0.28, 0.30], castshadow=False)


def _look_at(pos, target):
    """`xyaxes` for a camera at `pos` aimed at `target`.

    MuJoCo cameras look along their own -z and are declared with the *other* two
    axes, which makes hand-written xyaxes a small trigonometry exercise every
    time the framing moves 100 mm. Naming the point being looked at is both
    easier to get right and easier to read six weeks later.
    """
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return [*right, *up]


def _cameras(spec):
    """Three framings: the working shot, down the aisle, and the wide.

    ⚠️ Every one of them has to sit on the *arm's* side of the row. The crop
    wall is opaque and at x = ROW_X + 0.09, so a camera out at +x films the back
    of it with the entire demo hidden behind. The first attempt at this in Week
    2 did exactly that and the fruit was not visible in a single frame.
    """
    # The working shot: row, arm and crate in one frame, which is what the
    # weekly capture has to show.
    spec.worldbody.add_camera(
        name="row", pos=[0.26, -1.60, 1.12],
        xyaxes=_look_at([0.26, -1.60, 1.12], [0.50, -0.10, 0.58]))
    # Down the path — the shot that says "this is standing in a greenhouse".
    spec.worldbody.add_camera(
        name="aisle", pos=[-0.10, -2.35, 1.50],
        xyaxes=_look_at([-0.10, -2.35, 1.50], [0.60, 0.35, 0.80]))
    # Wide, from over the neighbouring path. Aimed high on purpose: this is the
    # frame that has to contain the roof, because "it works in the greenhouse
    # you already have" is the claim and the glazing is what shows it. Aimed at
    # the arm instead, it is a picture of a robot in a field.
    spec.worldbody.add_camera(
        name="wide", pos=[-1.85, -2.95, 1.70],
        xyaxes=_look_at([-1.85, -2.95, 1.70], [0.95, 0.45, 1.42]))


def add_greenhouse(spec, leafy=True):
    """Add the house around a scene that already has the arm and the row in it.

    Call after `add_row`, before `spec.compile()`.
    """
    _lighting(spec)
    _rails(spec)
    _structure(spec)
    _crop_row(spec, "work", ROW_X, leafy=leafy)
    _crop_row(spec, "next", ROW_X + ROW_PITCH, leafy=leafy)
    _crop_row(spec, "back", ROW_X + 2 * ROW_PITCH, leafy=False)
    _backdrop(spec)
    _cameras(spec)
    return spec


def build_scene(greenhouse=True, leafy=True, bin_pos=None, bin_half=None,
                bin_wall=None, wrist_cam=False, trusses=None,
                place_board=False, deck_cam=False):
    """The whole Week 2 scene: FR5 + 2F85 + plant row + crate, in a greenhouse.

    This is the one place the scene is assembled, so the planner, the executor,
    the incident recorder and the demos all measure against the same world.

    `wrist_cam` adds Week 3's eye-in-hand sensor camera. It defaults **off** and
    that is on purpose: a camera changes no dynamics, but Week 1's and Week 2's
    demos are asserted to produce byte-identical numbers and the cheapest way to
    keep that true is for their scene to be untouched. Week 3 opts in.

    `deck_cam` adds the chassis survey camera and its post — the sensor that
    finds the row in one frame and decides the pick order. Off by default for
    exactly the same reason, and Week 4 opts in. Both are `contype=0`, so
    turning either on cannot move a clearance number; see
    `camera.add_camera_housing`.
    """
    from mission import BIN_HALF, BIN_POS, BIN_WALL

    bin_pos = BIN_POS if bin_pos is None else bin_pos
    bin_half = BIN_HALF if bin_half is None else bin_half
    bin_wall = BIN_WALL if bin_wall is None else bin_wall

    spec = build_fr5_spec(with_scene=not greenhouse, gripper=True)
    # `trusses` lets Week 4 compile a pool of fruit instead of the fixed row.
    # None keeps the five-truss default, so every earlier caller is untouched.
    add_row(spec) if trusses is None else add_row(spec, trusses=trusses)

    # Soften the finger pads. Menagerie ships them at solref [0.004, 1] — five
    # times stiffer than MuJoCo's own default — which suits the rigid objects
    # their benchmarks grip and does not suit a tomato.
    #
    # ⚠️ The pads carry `priority=1`, and priority does not only decide
    # friction: on any pad contact *all* of the pads' contact parameters win and
    # the fruit's are ignored outright. Softening solref on the fruit geom
    # therefore changes nothing — measured, the close transient stayed at
    # 26.01 N across an 8x sweep of fruit solref. The pads are the geom to
    # soften, and doing it takes that transient to 11.05 N.
    for geom in spec.geoms:
        if "pad" in geom.name:
            geom.solref = [0.02, 1.0]

    add_crate(spec, name="bin", pos=bin_pos, half=bin_half, wall=bin_wall)

    if wrist_cam:
        from camera import add_wrist_camera
        add_wrist_camera(spec)

    if deck_cam:
        from deck_cam import add_deck_camera
        add_deck_camera(spec)

    if place_board:
        # Week 4's click target. contype=0 scenery, so it changes no dynamics
        # and no tuned contact number; it exists to be selected by the viewer.
        from week4_place import add_place_board
        add_place_board(spec)

    if greenhouse:
        add_greenhouse(spec, leafy=leafy)
    else:
        spec.worldbody.add_camera(
            name="row", pos=[0.28, -1.85, 1.20],
            xyaxes=[0.9949, -0.1013, 0.0, 0.0399, 0.3922, 0.9190])
    return spec.compile()


def main():
    """Render a still from each camera, and report what the scenery costs."""
    import argparse
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import time

    import cv2
    import mujoco

    from mission import PARK, park_posture, reset_park

    ap = argparse.ArgumentParser(description="Render the greenhouse.")
    ap.add_argument("--bare", action="store_true", help="the old rig, for comparison")
    ap.add_argument("--no-leaves", action="store_true")
    args = ap.parse_args()

    model = build_scene(greenhouse=not args.bare, leafy=not args.no_leaves)
    data = mujoco.MjData(model)
    q = park_posture(model)
    reset_park(model, data, q)

    print(f"\n{'bare rig' if args.bare else 'greenhouse'}: {model.ngeom} geoms, "
          f"{model.nbody} bodies, {model.nlight} lights")
    collidable = sum(1 for i in range(model.ngeom)
                     if model.geom(i).contype[0] or model.geom(i).conaffinity[0])
    print(f"  collidable geoms: {collidable}   "
          f"(scenery adds {model.ngeom - collidable} that cannot be hit)")
    print(f"  arm parked at {data.site('tool0').xpos.round(3)}, "
          f"PARK is {PARK.round(3)}")

    # What the decor costs per physics step — it must be ~nothing, because it is
    # all contype=0 and the broadphase never considers it.
    t0 = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_step(model, data)
    print(f"  2000 steps in {time.perf_counter() - t0:.2f} s")

    out_dir = Path(__file__).resolve().parents[2]
    with mujoco.Renderer(model, height=960, width=1280) as r:
        for cam in (["row"] if args.bare else ["row", "aisle", "wide"]):
            r.update_scene(data, camera=cam)
            path = out_dir / f"greenhouse_{cam}.png"
            cv2.imwrite(str(path), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
            print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
