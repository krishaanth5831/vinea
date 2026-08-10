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


def is_ripe(stage):
    """Whether the harvest would take this stage. `crop.STAGES`'s own flag."""
    entry = fcrop.STAGE_BY_NAME.get(stage)
    return bool(entry[2]) if entry else False


def annotate(bgr, detector=None, min_side=6):
    """Box and label every fruit the colour classifier finds. Returns counts.

    `bgr` is modified in place — it is a render nobody else owns.

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
    counts = {"ripe": 0, "unripe": 0}
    for d in detector(rgb):
        if min(d.width, d.height) < min_side:
            continue
        stage, _hue = stage_of(rgb, d)
        if stage is None:
            continue
        ripe = is_ripe(stage)
        counts["ripe" if ripe else "unripe"] += 1
        colour = STAGE_BGR.get(stage, UNKNOWN_BGR)

        u0, v0 = int(d.u0), int(d.v0)
        u1, v1 = int(d.u1), int(d.v1)
        cv2.rectangle(bgr, (u0, v0), (u1, v1), colour, 2)

        # The decision, then the stage it came from. Ripe fruit get a filled
        # tag so they read at a glance in a panel a third of a screen wide —
        # which is the only size these are ever actually looked at.
        text = f"{'RIPE' if ripe else 'unripe'} {stage}"
        scale, thick = 0.42, 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      scale, thick)
        ty = max(th + 2, v0 - 3)
        if ripe:
            cv2.rectangle(bgr, (u0, ty - th - 3), (u0 + tw + 4, ty + 2),
                          colour, -1)
            cv2.putText(bgr, text, (u0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (20, 20, 20), thick, cv2.LINE_AA)
        else:
            cv2.putText(bgr, text, (u0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, colour, thick, cv2.LINE_AA)
    return counts


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
