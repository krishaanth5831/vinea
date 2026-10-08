#!/usr/bin/env python3
"""Draw what the colour classifier thinks, onto the frame it thought it about.

⚠️ **This is the Week 3 "option A" control detector and nothing else.** No model
was built, trained or fine-tuned for it. `farm.scout.StageDetector` runs two
OpenCV `inRange` bands over the frame's hue, keeps the blobs round enough to be
fruit, and `farm.scout.stage_of` reads the mean hue inside each box back against
`crop.STAGES`. That is the whole classifier, it is the same code the mapping
pass scores itself with, and drawing it changes none of it — the boxes here are
a *view* of a decision the robot already makes, not a second opinion computed
for the display.

That distinction matters because an overlay is the most persuasive thing in the
window and the easiest place to lie. A panel that ran a better detector than the
robot uses would show a machine that sees clearly and picks badly, and the gap
would look like a manipulation problem.

--- what the labels mean ----------------------------------------------------

`ripe` and `unripe` are `crop.STAGES`'s own `pick` flag, so they are exactly the
decision the harvest acts on: only **red** is taken. A `turning` fruit is
labelled unripe and boxed in amber, which is correct and is what a grower does —
it is next pass's fruit, not this one's.

The box colour is the *stage*, the word is the *decision*. Both are drawn
because they disagree usefully: four stages collapsing into two answers is where
a classifier that is nearly right stops mattering.

⚠️ Green fruit are the weak band and the overlay does not hide it — see the
measurement in `farm/scout.py`'s docstring. Green and foliage are 6 hue apart
and only circularity separates them, so a green tomato is often simply not
boxed. It costs the harvest nothing (nobody was picking it) and it would cost a
yield forecast a great deal.

--- what the deck cameras have to say out loud ------------------------------

⚠️ **An overlay on a deck camera is read as a claim about the crop, so the two
measured ways it is wrong are drawn on the frame rather than left in a README.**
`RECALL` and `CLUSTER_NOTE` below are printed by `banner`, small, on every deck
panel — so when a green fruit goes unboxed live in front of somebody, the 33%
that predicted it is already on the screen next to the miss.

Neither number is tuned and neither may be. The thresholds that produce them are
`farm.scout`'s and they are the same thresholds the mapping pass scores itself
with; moving one to make a demo look better would move the map with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farm import crop as fcrop  # noqa: E402

# BGR, because everything downstream of a render in this repo is an OpenCV
# image. One colour per stage, matched to the fruit's own `rgba` so a boxed
# tomato and its box are recognisably the same thing.
STAGE_BGR = {
    "green":   (80, 200, 80),
    "breaker": (60, 220, 240),
    "turning": (60, 160, 240),
    "red":     (60, 60, 220),
}
UNKNOWN_BGR = (200, 200, 200)

# Every band, in the order a person reads them: ripe first, because that is the
# one the harvest acts on. `crop.STAGES` is the other way round (green first,
# because that is the order fruit ripen in) and both orders are right for what
# they are — this one is a legend, that one is a lifecycle.
BAND_ORDER = ("red", "turning", "breaker", "green")

# --- the two measured failures, drawn on the frame ---------------------------
#
# ⚠️ **Measured on the cameras these numbers are drawn on, which is not the
# camera README.md quotes.** Per-stage recall of the colour classifier, and the
# fraction of what it found that it then banded correctly:
#
#   (recall %, correctly-banded-when-found %)
#
# ⚠️ **The famous 33% green is the *shared* scouting head and it does not apply
# here.** `farm/scout.py --recall` drives `scout.Scout`: one pan-tilt unit on
# the aisle centreline, 0.70 m from both rows, turning 180 degrees to serve
# them. `farm.decks` is a different camera — one head per arm, over its own
# arm's plate, 0.60 m from the row it works and never leaving it, sweeping three
# pan poses instead of one. Closer and with more looks per fruit, so it does
# better, and quoting the shared head's number on this panel would be printing a
# measurement of a camera that is not in the frame.
#
# Re-measured over four houses (seeds 7, 11, 23, 41), both worked rows, 96 fruit,
# `duo.DuoScout` scored with `scout.score`:
#
#     stage      in house   mapped   recall   banded right
#     green            42       31      74%           100%
#     breaker          16       16     100%            94%
#     turning          14       13      93%            85%
#     red              24       24     100%            92%   <- the picked one
#
# Green is still the weak band and still for the reason `farm/scout.py`
# measured before the detector was designed — hue 49 against a stem at 55 and a
# leaf at 62, with only circularity between them — it is just less weak at
# 600 mm than at 700 mm. It costs the harvest nothing, because only red is
# picked; it would cost a scouting yield forecast a great deal, which is Vinea's
# second module. So it goes on the window either way.
RECALL = {
    "red":     (100.0, 92.0),
    "turning": (93.0, 85.0),
    "breaker": (100.0, 94.0),
    "green":   (74.0, 100.0),
}

# ⚠️ **The other measured failure, and it is not the one this repo had written
# down.** README.md records four fruit in a 70 mm cluster fusing into a single
# blob and coming back as ONE detection positioned 48 mm off. That was measured
# on the Week 4 chassis mast at 1.3 m. Re-measured on `farm.decks` at its own
# 0.60 m standoff, head-on at pan 0, on the panel these words are printed on:
#
#     one fruit            -> 1 detection, box 37x36 px, elongation 1.03
#     a pair at 100 mm     -> 2 detections
#     a pair at  70 mm     -> **0 detections**
#     four at  71 mm pitch -> **0 detections**
#
# Closer up, touching fruit are not merged into one confident box — the fused
# contour is not round enough to pass `scout.RIPE_CIRCULARITY` and the whole
# cluster is **dropped**. Which is worse than a fat box and much easier to miss:
# a wrong box is on the screen and a missing one is not. So the caption says
# what actually happens rather than repeating the 1.3 m finding at a range where
# it does not hold.
#
# (Across a whole survey the pair does come back, as *three* sightings for two
# fruit — the head sees it from angles that resolve it and angles that do not,
# and `scout._fuse` cannot merge what arrives more than `FUSE_M` apart. That is
# a map problem rather than a panel one; `farm/misses.py` is where it lands.)
CLUSTER_NOTE = ("measured: touching fruit are DROPPED, not merged - "
                "a 70 mm pair returned 0 of 2 at this range")

# When a blob is drawn as suspected-fused rather than as one confident fruit.
#
# ⚠️ **Elongation only, and the circularity test that used to sit beside it was
# removed because it was measured firing on the wrong thing.** The idea was that
# a fused pair has a waist and a waist costs perimeter — true, but a *single*
# fruit partly occluded by its neighbour has a cut contour and costs the same
# perimeter, and that is what it actually caught: on the 100 mm pair above it
# flagged the far fruit, which is one tomato and not a cluster. A flag that
# fires on occlusion while claiming to mean size is worse than no flag.
#
# What is left is range-free and says only what it can see: one sphere projects
# round from any distance, so a box half again as long as it is wide is not one
# sphere. On the measurements above it fires on the real fused cluster
# (elongation 1.64) and not on the occluded single (1.06).
#
# ⚠️ A square 2x2 cluster defeats it, and at this range a tight cluster is not
# returned at all — which is why `CLUSTER_NOTE` is printed on **every** deck
# frame and not only on frames where this fires. The flag marks what it can
# catch; the caption states the limitation whether or not it caught anything.
CLUSTER_ELONGATION = 1.45

# ⚠️ Optional, and off unless a caller passes the geometry. A deck panel knows
# its own standoff and its own intrinsics, so it can say "one fruit is 37 px
# across here" and call anything much wider than that more than one fruit. It is
# the test the shape tests are a proxy for, and it is only available where the
# range is known — which is why it is a parameter and not a constant.
CLUSTER_SIZE_RATIO = 1.6


def is_ripe(stage):
    """Whether the harvest would take this stage. `crop.STAGES`'s own flag."""
    entry = fcrop.STAGE_BY_NAME.get(stage)
    return bool(entry[2]) if entry else False


def looks_fused(u0, v0, u1, v1, fruit_px=None):
    """Is this blob too big to be one tomato? A suspicion, never a count.

    ⚠️ Every caller has to render it as a suspicion. It is a shape test on a
    colour blob, not a measurement of how many fruit are behind it — the deck
    camera cannot separate them, which is precisely the failure being flagged,
    so nothing here can count them either.

    `fruit_px` is one fruit's apparent diameter in this frame, if the caller
    knows the range. See `CLUSTER_SIZE_RATIO`.
    """
    w, h = max(1, u1 - u0), max(1, v1 - v0)
    if max(w, h) / min(w, h) >= CLUSTER_ELONGATION:
        return True
    if fruit_px:
        return bool(min(w, h) > CLUSTER_SIZE_RATIO * fruit_px)
    return False


def fruit_diameter_px(height, fovy, standoff, radius=None):
    """One fruit's apparent diameter, in pixels, at a known range.

    ⚠️ `fovy` is the **vertical** field of view, so the frame's height is what
    sets the scale — `camera.Intrinsics.from_model`'s derivation, restated here
    because a panel renders at its own size and does not have an `Intrinsics`
    to hand. Getting it off the width instead is right at 4:3 and quietly wrong
    at every other aspect ratio.
    """
    from plant_row import FRUIT_R

    r = FRUIT_R if radius is None else radius
    f = 0.5 * height / np.tan(np.radians(fovy) / 2)
    return 2.0 * r * f / max(standoff, 1e-6)


def find(bgr, detector=None, min_side=6, fruit_px=None):
    """Run the classifier and return its calls, without drawing anything.

    Returns a list of `(u0, v0, u1, v1, stage, ripe, fused)`. Split out from
    `annotate` so a viewer can run the detector at a lower rate than it
    composites panels and still draw a box on every frame — see
    `two_arm_farm.Windows`. Running it per panel frame is the expensive thing
    here; it costs more than the renders.

    ⚠️ `fused` is `looks_fused`'s suspicion, computed here rather than in `draw`
    so that a viewer holding cached calls across frames holds the flag with the
    box it belongs to. A box redrawn without it would be a confident single
    fruit on a frame the detector had doubts about.

    ⚠️ Runs the detector on the **RGB** of what it is given. Every detector in
    this repo takes RGB (they were written against `mujoco.Renderer.render()`),
    and the panels are BGR by the time they reach a viewer. Feeding BGR to an
    HSV threshold does not fail, it silently swaps red and blue: the ripe band
    then matches nothing and the frame reads as a house with no tomatoes in it.
    """
    import cv2

    from farm.scout import StageDetector, stage_of

    if detector is None:
        detector = StageDetector()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = []
    for d in detector(rgb):
        if min(d.width, d.height) < min_side:
            continue
        stage, _hue = stage_of(rgb, d)
        if stage is None:
            continue
        u0, v0 = int(d.u0), int(d.v0)
        u1, v1 = int(d.u1), int(d.v1)
        out.append((u0, v0, u1, v1, stage, is_ripe(stage),
                    looks_fused(u0, v0, u1, v1, fruit_px=fruit_px)))
    return out


def draw(bgr, calls, stale=False):
    """Draw `find`'s calls onto a frame in place. Returns the counts.

    `stale=True` marks boxes that were computed on an earlier frame — a viewer
    running the detector below its panel rate has to say so, because a box drawn
    over a moving frame is a claim about *this* frame unless it is labelled
    otherwise. Drawn thinner, which is enough to read as "held" without turning
    the panel into a legend.
    """
    import cv2

    counts = {"ripe": 0, "unripe": 0, "fused": 0}
    for call in calls:
        u0, v0, u1, v1, stage, ripe = call[:6]
        fused = bool(call[6]) if len(call) > 6 else False
        counts["ripe" if ripe else "unripe"] += 1
        colour = STAGE_BGR.get(stage, UNKNOWN_BGR)

        if fused:
            # ⚠️ **Not a confident single box.** A blob the shape tests call
            # fused is drawn open-cornered rather than closed, so it does not
            # read as "here is one tomato and here is where it is" — which is
            # the claim the deck camera has been measured *not* to be able to
            # make at 70 mm spacing. The corners are the honest picture: the
            # detector found something in here and cannot say how many.
            counts["fused"] += 1
            _corners(bgr, u0, v0, u1, v1, colour, 1 if stale else 2)
        else:
            cv2.rectangle(bgr, (u0, v0), (u1, v1), colour, 1 if stale else 2)

        # The decision, then the stage it came from. Ripe fruit get a filled
        # tag so they read at a glance in a panel a third of a screen wide —
        # which is the only size these are ever actually looked at.
        text = f"{'RIPE' if ripe else 'unripe'} {stage}"
        if fused:
            # ⚠️ "1 detection", not "4 fruit". The count the camera returned is
            # the only number it is entitled to print; how many tomatoes are
            # actually in there is the thing it could not work out.
            text = f"1 detection, cluster? {stage}"
        scale, thick = 0.42, 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      scale, thick)
        ty = max(th + 2, v0 - 3)
        if ripe and not stale and not fused:
            cv2.rectangle(bgr, (u0, ty - th - 3), (u0 + tw + 4, ty + 2),
                          colour, -1)
            cv2.putText(bgr, text, (u0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (20, 20, 20), thick, cv2.LINE_AA)
        else:
            cv2.putText(bgr, text, (u0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, colour, thick, cv2.LINE_AA)
    return counts


def _corners(bgr, u0, v0, u1, v1, colour, thick):
    """An open-cornered box. Deliberately not a rectangle — see `draw`."""
    import cv2

    d = max(4, min(u1 - u0, v1 - v0) // 4)
    for x, sx in ((u0, 1), (u1, -1)):
        for y, sy in ((v0, 1), (v1, -1)):
            cv2.line(bgr, (x, y), (x + sx * d, y), colour, thick)
            cv2.line(bgr, (x, y), (x, y + sy * d), colour, thick)


def by_band(calls):
    """Counts per colour stage, in `BAND_ORDER`. For a caption, not a decision.

    ⚠️ Kept separate from `draw`'s ripe/unripe tally rather than replacing it.
    They answer different questions and both are worth a panel: the bands are
    what the classifier said, ripe/unripe is what the harvest will do about it.
    Four stages collapsing into two answers is exactly where a classifier that
    is nearly right stops mattering, and a caption that only showed one side of
    that could not show it happening.
    """
    counts = {name: 0 for name in BAND_ORDER}
    for call in calls:
        stage = call[4]
        if stage in counts:
            counts[stage] += 1
    return counts


def band_line(calls):
    """`red 3  turning 1  breaker 0  green 2`, plus any fused blobs. ASCII."""
    counts = by_band(calls)
    line = "  ".join(f"{n} {counts[n]}" for n in BAND_ORDER)
    fused = sum(1 for c in calls if len(c) > 6 and c[6])
    if fused:
        line += f"   |  {fused} cluster?"
    return line


def recall_line():
    """The measured per-band recall, small enough to sit under a caption.

    ⚠️ Per band, never averaged. See `RECALL`.
    """
    return "measured recall  " + "  ".join(
        f"{n[:5]} {RECALL[n][0]:.0f}%" for n in BAND_ORDER)


def banner(bgr, calls, note="", lines=3):
    """The deck panel's footer: what was called, and how it is known to fail.

    ⚠️ **Three lines, and the last two are not optional decoration.** Somebody
    watching coloured boxes appear over tomatoes will believe the boxes. The
    recall line is there so that when a green fruit goes unboxed in front of
    them, the 33% that predicted it is already on the screen; the cluster line
    is there whether or not `looks_fused` caught anything, because the tight
    clusters it cannot catch are the ones that matter.

    ASCII only — this string reaches `cv2.putText`, and the panel it lands on is
    one somebody is being shown.
    """
    import cv2

    h, w = bgr.shape[:2]
    top = h - (12 * lines + 6)
    cv2.rectangle(bgr, (0, top), (w, h), (24, 24, 24), -1)
    rows = [(band_line(calls), (235, 235, 235)),
            (recall_line(), (150, 150, 150)),
            (note or CLUSTER_NOTE, (150, 150, 150))]
    y = top + 13
    for text, colour in rows[:lines]:
        cv2.putText(bgr, text[:74], (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    colour, 1, cv2.LINE_AA)
        y += 12
    return bgr


def annotate(bgr, detector=None, min_side=6):
    """Box and label every fruit the colour classifier finds. Returns counts.

    `bgr` is modified in place — it is a render nobody else owns. `find` then
    `draw`; kept as one call because every existing caller wants both.
    """
    return draw(bgr, find(bgr, detector=detector, min_side=min_side))


def tally(bgr, counts, note="HSV colour threshold - the Week 3 control"):
    """Footer saying what the classifier just did, and what it is.

    The provenance line is not decoration. Somebody watching a window with
    coloured boxes on it will assume a model produced them unless told
    otherwise, and this is the cheapest place to tell them.
    """
    import cv2

    h, w = bgr.shape[:2]
    cv2.rectangle(bgr, (0, h - 30), (w, h), (24, 24, 24), -1)
    cv2.putText(bgr, f"ripe {counts['ripe']}   unripe {counts['unripe']}",
                (6, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bgr, note, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (150, 150, 150), 1, cv2.LINE_AA)
    return bgr
