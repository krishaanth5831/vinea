#!/usr/bin/env python3
"""Ripeness, on real photographs, deliberately outside the sim pick loop.

⚠️ **This does not run on rendered frames, and that is the entire point.**

Ripeness is a colour and texture classification problem. The tomatoes in
`plant_row.py` are uniform red spheres — one RGBA value, one specular
highlight, no calyx, no ribbing, no shoulder, no ripeness spread at all. There
is no signal to classify. A ripeness "result" measured in this renderer would
be a measurement of `rgba=[0.85, 0.16, 0.12, 1.0]`, which is a constant the
scene file sets, and quoting it to a grower would be dishonest in exactly the
way a detector fine-tuned on these spheres is dishonest.

So this is a **separate demo on real imagery**: run the same detection stack
over real greenhouse photographs, band the fruit by colour, and show it beside
the sim loop with the caption *"same detection stack, run on real imagery."* It
is a slide, not a build, and it makes no claim about the harvesting pipeline.

    ./.venv/bin/python simulation/mujoco/ripeness.py --images DIR --out OUT.png

**No images ship with this repo.** The intended source is the Laboro Tomato
dataset, which is licensed (CC BY-NC-SA 4.0 — non-commercial) and needs its
terms accepted by a person, not fetched by a script. That acceptance is a
decision with a commercial consequence for Vinea and it is not one to make in
passing; the settled decisions already list ripeness as deferred, with
"license Laboro Tomato" as the open item. Point `--images` at any directory of
real tomato photographs to run it.

⚠️ The bands below are **not a ripeness model.** They are the standard
horticultural colour stages reduced to mean hue, which is a demonstration of
the pipeline shape, not a classifier anyone should trust. A real one needs the
dataset, its labels, and a training run — Week 5 and beyond.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Mean hue on OpenCV's 0-179 scale, over the detected fruit's pixels. The USDA
# colour stages for tomato run green -> breaker -> turning -> pink -> light red
# -> red; these collapse that to four bands, which is as much as mean hue can
# honestly separate.
BANDS = [
    ("red",      (0, 8),   (60, 60, 220)),
    ("turning",  (8, 22),  (60, 160, 240)),
    ("breaker",  (22, 35), (60, 220, 240)),
    ("green",    (35, 90), (80, 200, 80)),
]

MIN_SAT = 60      # below this the pixel has no usable hue at all


def band_of(hue):
    for name, (lo, hi), colour in BANDS:
        if lo <= hue < hi:
            return name, colour
    return "unknown", (160, 160, 160)


def classify(rgb, det):
    """Mean hue over the saturated pixels inside a detection's box."""
    import cv2

    u0, v0 = max(0, int(det.u0)), max(0, int(det.v0))
    u1 = min(rgb.shape[1], int(det.u1))
    v1 = min(rgb.shape[0], int(det.v1))
    if u1 <= u0 or v1 <= v0:
        return None, 0.0

    patch = cv2.cvtColor(rgb[v0:v1, u0:u1], cv2.COLOR_RGB2HSV)
    sat = patch[:, :, 1] >= MIN_SAT
    if sat.sum() < 20:
        return None, 0.0

    # ⚠️ Circular mean, not the arithmetic one. Red straddles the hue wrap at
    # 0/179, so averaging a ripe tomato's pixels arithmetically lands it in the
    # greens — the single most common way this measurement is got wrong.
    h = patch[:, :, 0][sat].astype(float) * (2 * np.pi / 180.0)
    mean = np.degrees(np.arctan2(np.sin(h).mean(), np.cos(h).mean())) / 2.0
    if mean < 0:
        mean += 180.0
    return float(mean), float(sat.mean())


def main():
    import cv2

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="directory of REAL tomato photographs")
    ap.add_argument("--out", default="ripeness.png",
                    help="contact sheet to write")
    ap.add_argument("--detector", default="yolo", choices=["hsv", "yolo"])
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max", type=int, default=6, help="images to show")
    args = ap.parse_args()

    from detect import HSVDetector, YoloDetector

    folder = Path(args.images)
    if not folder.is_dir():
        print(f"  no such directory: {folder}")
        print("  This script needs REAL photographs. See the module docstring:")
        print("  the Laboro Tomato dataset is licensed CC BY-NC-SA 4.0 and its")
        print("  terms have to be accepted by a person before it is downloaded.")
        return 2

    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        print(f"  no images in {folder}")
        return 2
    paths = paths[: args.max]

    detector = (HSVDetector() if args.detector == "hsv"
                else YoloDetector(conf=args.conf))
    print(f"\n  ripeness on REAL imagery · detector '{detector.name}' · "
          f"{len(paths)} images")
    print("  same detection stack as the sim loop. No sim frames involved.\n")
    print(f"  {'image':>28} {'fruit':>6} " +
          " ".join(f"{n:>9}" for n, _, _ in BANDS))

    tiles, totals = [], {n: 0 for n, _, _ in BANDS}
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = detector(rgb)

        counts = {n: 0 for n, _, _ in BANDS}
        img = bgr.copy()
        for d in dets:
            hue, _ = classify(rgb, d)
            if hue is None:
                continue
            name, colour = band_of(hue)
            if name in counts:
                counts[name] += 1
                totals[name] += 1
            cv2.rectangle(img, (int(d.u0), int(d.v0)), (int(d.u1), int(d.v1)),
                          colour, 2)
            cv2.putText(img, f"{name} h{hue:.0f}",
                        (int(d.u0), max(14, int(d.v0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

        print(f"  {p.name[:28]:>28} {len(dets):6d} " +
              " ".join(f"{counts[n]:9d}" for n, _, _ in BANDS))

        h = 360
        tiles.append(cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)))

    if tiles:
        width = max(t.shape[1] for t in tiles)
        rows = [cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1],
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
                for t in tiles]
        sheet = np.vstack(rows)
        banner = np.full((44, width, 3), 20, np.uint8)
        cv2.putText(banner, "same detection stack, run on real imagery",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imwrite(args.out, np.vstack([banner, sheet]))
        print(f"\n  wrote {args.out}")

    print(f"\n  totals: " + "  ".join(f"{n} {totals[n]}"
                                      for n, _, _ in BANDS))
    print("  ⚠️ Mean hue is not a ripeness model. It shows the pipeline shape.")
    print("     A real classifier needs the labelled dataset and a training "
          "run.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
